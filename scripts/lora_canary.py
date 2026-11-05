"""LoRA canary (stage 0 of research/notes/9-lora-plan.md): is an adapter actually applied by the scorer's serving stack?

    make   build a PEFT adapter for Gemma 4 31B (runs where torch + safetensors are, e.g. the vast instance):
           lora_canary.py make --config /models/<model>/config.json --out /adapters/loud --rank 16 --scale 0.05
           --scale 0 gives an exact no-op (lora_B = 0). Targets default to q_proj,o_proj,gate_proj,up_proj,down_proj
           (no k_proj/v_proj: HF and vLLM treat V differently on the global layers, note 9 §2).
    cmd    print the vLLM command adk_submission's VllmServer builds for a submission's adapters/ dir (the scorer path):
           lora_canary.py cmd --model /models/<model> --adapters <submission>/adapters [--tp 1]
    check  compare base vs adapters through the OpenAI API (identical? different? deterministic? speed):
           lora_canary.py check --server http://127.0.0.1:8002/v1 --names zero,loud
"""

import argparse
import json
import sys
import time
import urllib.request

MODEL = 'gemma-4-31b-it-qat-w4a16-ct'
PREFIX = 'base_model.model.model.language_model.layers.{}.{}.{}.lora_{}.weight'
PARENT = {'q_proj': 'self_attn', 'k_proj': 'self_attn', 'v_proj': 'self_attn', 'o_proj': 'self_attn',
          'gate_proj': 'mlp', 'up_proj': 'mlp', 'down_proj': 'mlp'}


def shapes(cfg, layer, module):
    t = cfg.get('text_config', cfg)
    hidden, inter, heads = t['hidden_size'], t['intermediate_size'], t['num_attention_heads']
    full = (t.get('layer_types') or ['sliding_attention'] * t['num_hidden_layers'])[layer] == 'full_attention'
    hd = t.get('global_head_dim') if full and t.get('global_head_dim') else t['head_dim']
    kvh = (t.get('num_global_key_value_heads') or t['num_key_value_heads']) if full else t['num_key_value_heads']
    return {'q_proj': (hidden, heads * hd), 'k_proj': (hidden, kvh * hd), 'v_proj': (hidden, kvh * hd),
            'o_proj': (heads * hd, hidden), 'gate_proj': (hidden, inter), 'up_proj': (hidden, inter),
            'down_proj': (inter, hidden)}[module]  # (in_features, out_features)


def make(a):
    import torch
    from safetensors.torch import save_file
    cfg = json.load(open(a.config))
    t = cfg.get('text_config', cfg)
    layers = range(t['num_hidden_layers']) if a.layers == 'all' else [int(x) for x in a.layers.split(',')]
    targets = a.targets.split(',')
    g = torch.Generator().manual_seed(a.seed)
    tensors = {}
    for n in layers:
        for m in targets:
            fin, fout = shapes(cfg, n, m)
            A = torch.randn(a.rank, fin, generator=g) / fin ** 0.5
            B = torch.randn(fout, a.rank, generator=g) * a.scale if a.scale else torch.zeros(fout, a.rank)
            tensors[PREFIX.format(n, PARENT[m], m, 'A')] = A.to(torch.bfloat16).contiguous()
            tensors[PREFIX.format(n, PARENT[m], m, 'B')] = B.to(torch.bfloat16).contiguous()
    import os
    os.makedirs(a.out, exist_ok=True)
    save_file(tensors, f'{a.out}/adapter_model.safetensors')
    json.dump({'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': a.rank, 'lora_alpha': a.rank, 'lora_dropout': 0.0,
               'target_modules': targets, 'bias': 'none', 'use_dora': False, 'use_rslora': False,
               'modules_to_save': None, 'fan_in_fan_out': False, 'inference_mode': True,
               'base_model_name_or_path': 'google/gemma-4-31b-it-qat-w4a16-ct', 'rank_pattern': {}, 'alpha_pattern': {},
               'layers_to_transform': None if a.layers == 'all' else list(layers)},
              open(f'{a.out}/adapter_config.json', 'w'), indent=1)
    size = os.path.getsize(f'{a.out}/adapter_model.safetensors')
    print(f'{a.out}: rank {a.rank}, scale {a.scale}, {len(tensors)} tensors, {size / 1e6:.1f} MB')


def cmd(a):
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    adapters = discover_adapters(a.adapters)
    cfg = VllmConfig(model=a.model, port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
                     max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=a.gpu_util,
                     enable_auto_tool_choice=True, enable_lora=True, max_loras=8, max_lora_rank=128,
                     tensor_parallel_size=a.tp)  # the hosts' Getting Started notebook settings (2026-10-01)
    c = VllmServer(cfg, adapter_manifest=adapters).build_cmd()
    print(' '.join(json.dumps(x) if (' ' in x or '{' in x) else x for x in c))


def post(server, body):
    req = urllib.request.Request(server.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    return d, time.time() - t0


def check(a):
    msgs = [{'role': 'user', 'content': 'Write a Python function that returns the n-th Fibonacci number, then explain '
                                        'its complexity in two sentences.'}]
    out = {}
    for name in [MODEL] + a.names.split(','):
        runs = []
        for s in range(2):
            d, dt = post(a.server, {'model': name, 'messages': msgs, 'max_tokens': a.tokens, 'temperature': 0,
                                    'seed': 1, 'logprobs': True, 'top_logprobs': 1,
                                    'chat_template_kwargs': {'enable_thinking': False}})
            c = d['choices'][0]
            lp = [x['logprob'] for x in (c.get('logprobs') or {}).get('content') or []]
            runs.append((c['message'].get('content') or '', lp, d['usage']['completion_tokens'] / dt))
        out[name] = runs
        text, lp, tps = runs[0]
        print(f'== {name}: deterministic={runs[0][0] == runs[1][0]}  {tps:.1f} tok/s  mean logprob '
              f'{sum(lp) / max(len(lp), 1):.3f}\n   {text[:160]!r}')
    base = out[MODEL][0][0]
    for name in a.names.split(','):
        print(f'{name}: identical to base = {out[name][0][0] == base}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('make')
    m.add_argument('--config', required=True)
    m.add_argument('--out', required=True)
    m.add_argument('--rank', type=int, default=16)
    m.add_argument('--scale', type=float, default=0.05)
    m.add_argument('--targets', default='q_proj,o_proj,gate_proj,up_proj,down_proj')
    m.add_argument('--layers', default='all')
    m.add_argument('--seed', type=int, default=0)
    c = sub.add_parser('cmd')
    c.add_argument('--model', required=True)
    c.add_argument('--adapters', required=True)
    c.add_argument('--tp', type=int, default=1)
    c.add_argument('--gpu-util', type=float, default=0.90)
    k = sub.add_parser('check')
    k.add_argument('--server', default='http://127.0.0.1:8002/v1')
    k.add_argument('--names', default='zero,loud')
    k.add_argument('--tokens', type=int, default=120)
    a = ap.parse_args()
    {'make': make, 'cmd': cmd, 'check': check}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
