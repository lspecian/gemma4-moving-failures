"""Training/serving consistency check for a LoRA adapter on Gemma 4 31B (stage 0, research/notes/9-lora-plan.md).

Runs ON the GPU instance next to a vLLM server that has the adapter mounted (scripts/lora_canary.py cmd). For one fixed
chat transcript it gets per-token log-probs of the transcript tokens from
  - vLLM (prompt_logprobs) for the base model and the adapter,
  - HF transformers on CPU (dequantized w4a16 -> bf16) without and with the same adapter via PEFT,
and compares the adapter's effect (adapter minus base) in both. If HF and vLLM apply the adapter the same way, the
deltas correlate strongly; a weak correlation means a trained adapter would behave differently when served.

    python3 lora_hf_vs_vllm.py --model /models/gemma-4-31b-it-qat-w4a16-ct --adapter /adapters/mild --name mild
"""

import argparse
import json
import time
import urllib.request

import torch

SERVED = 'gemma-4-31b-it-qat-w4a16-ct'
CHAT = [
    {'role': 'user', 'content': 'In rich/console.py the export_html method ignores the inline_styles flag for the '
                                'code block. Which line would you change, and how?'},
    {'role': 'assistant', 'content': 'The template wraps the output in <pre><code>{code}</code></pre> without a '
                                     'font-family on the code element, so browsers fall back to their default '
                                     'monospace font. I would add style="font-family:inherit" to the code tag in '
                                     'CONSOLE_HTML_FORMAT so it inherits the font stack set on the pre element.'},
]


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=600))


def vllm_logprobs(server, ids, model):
    d = post(server + '/v1/completions', {'model': model, 'prompt': ids, 'max_tokens': 1, 'temperature': 0,
                                          'prompt_logprobs': 0})
    out = []
    for i, entry in enumerate(d['choices'][0]['prompt_logprobs'][1:], start=1):
        tok = ids[i]
        lp = entry.get(str(tok)) if isinstance(entry, dict) else None
        out.append(lp['logprob'] if isinstance(lp, dict) else lp)
    return out


def hf_logprobs(model, ids):
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(input_ids=x).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    return [lp[i - 1, ids[i]].item() for i in range(1, len(ids))]


def corr(a, b):
    a, b = torch.tensor(a), torch.tensor(b)
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--server', default='http://127.0.0.1:8000')
    ap.add_argument('--threads', type=int, default=96)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    ids = post(a.server + '/tokenize', {'model': SERVED, 'messages': CHAT, 'add_generation_prompt': False})['tokens']
    print('tokens', len(ids), flush=True)
    v_base = vllm_logprobs(a.server, ids, SERVED)
    v_ad = vllm_logprobs(a.server, ids, a.name)

    from transformers import AutoModelForCausalLM, CompressedTensorsConfig
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map='cpu',
                                                 quantization_config=CompressedTensorsConfig(run_compressed=False))
    model.eval()
    print(f'HF model loaded in {time.time() - t0:.0f}s: {type(model).__name__}', flush=True)
    t0 = time.time()
    h_base = hf_logprobs(model, ids)
    print(f'HF base forward {time.time() - t0:.0f}s', flush=True)
    from peft import PeftModel
    # PEFT matches plain names ("q_proj") in the vision tower too (Gemma4ClippableLinear, unsupported): restrict the
    # targets to the language model with a regex for HF. Training configs need the same regex.
    import os
    import shutil
    hf_dir = f'/tmp/{a.name}_hf'
    shutil.rmtree(hf_dir, ignore_errors=True)
    shutil.copytree(a.adapter, hf_dir)
    cfg = json.load(open(f'{hf_dir}/adapter_config.json'))
    cfg['target_modules'] = r'.*language_model.*\.(' + '|'.join(cfg['target_modules']) + ')'
    json.dump(cfg, open(f'{hf_dir}/adapter_config.json', 'w'))
    pm = PeftModel.from_pretrained(model, hf_dir)
    loaded = sum(1 for n, _ in pm.named_modules() if n.endswith('lora_A'))
    print('PEFT lora_A modules:', loaded, flush=True)
    h_ad = hf_logprobs(pm, ids)

    n = min(len(v_base), len(h_base))
    pairs = [(vb, va, hb, ha) for vb, va, hb, ha in zip(v_base[:n], v_ad[:n], h_base[:n], h_ad[:n])
             if None not in (vb, va)]
    vb, va, hb, ha = map(list, zip(*pairs))
    dv = [y - x for x, y in zip(vb, va)]
    dh = [y - x for x, y in zip(hb, ha)]
    res = {'tokens': len(ids), 'compared': len(pairs),
           'base_mean_lp_vllm': sum(vb) / len(vb), 'base_mean_lp_hf': sum(hb) / len(hb),
           'base_corr_vllm_hf': corr(vb, hb), 'base_mean_abs_diff': sum(abs(x - y) for x, y in zip(vb, hb)) / len(vb),
           'adapter_mean_lp_vllm': sum(va) / len(va), 'adapter_mean_lp_hf': sum(ha) / len(ha),
           'delta_mean_abs_vllm': sum(map(abs, dv)) / len(dv), 'delta_mean_abs_hf': sum(map(abs, dh)) / len(dh),
           'delta_corr_vllm_hf': corr(dv, dh)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(f'/root/hf_vs_vllm_{a.name}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
