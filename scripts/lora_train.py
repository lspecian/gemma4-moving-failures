"""Stage-1 LoRA self-RFT training for Gemma 4 31B (research/notes/9-lora-plan.md, option A) + export for the scorer.

Runs on 1x H200 (141 GB) via scripts/lora_train_h200.sh; `--tiny` runs every code path on CPU with a tiny random
Gemma 4 (same architecture: k_eq_v global layers, sliding layers, vision tower, 262k vocab, compressed-tensors
w4a16 checkpoint) instead of the 31B weights.

    python3 scripts/lora_train.py --model /models/gemma-4-31b-it-qat-w4a16-ct --data data/rft/train_tokens.jsonl \
        --out /root/lora_out --name rft1
    python3 scripts/lora_train.py --tiny --data data/rft/train_tokens.jsonl --out /tmp/lora_tiny   # CPU self-test

Model: the w4a16-ct checkpoint loaded with transformers 5.x and dequantized to bf16 on load
(CompressedTensorsConfig(run_compressed=False), as scripts/lora_hf_vs_vllm.py): exactly the grid vLLM serves. The
fake-quant forward wrappers compressed-tensors leaves on the Linear layers are switched off after checking they do
not change the output (the weights are already on the grid). SDPA attention (FlashAttention-2 fails on the global
layers' head_dim 512), gradient checkpointing (non-reentrant), no KV cache.

LoRA: r=16, alpha=32, dropout 0.05 on q_proj|o_proj|gate_proj|up_proj|down_proj of the language model only (a regex:
plain names also match the vision tower's Gemma4ClippableLinear). No k_proj/v_proj: on the 10 global layers HF uses
V = k_proj(x) (attention_k_eq_v) while vLLM fuses K into the V slot without the k_proj LoRA (note 9 §2); with
--allow-kv the exporter copies each global layer's k_proj LoRA to v_proj so vLLM reproduces V = K.

Loss: token-level cross-entropy on the labelled tokens only (rft_build.py: the model's own turns), normalised by the
number of labelled tokens per optimizer step (grad accumulation 8, batch 1). Logits are computed only at labelled
positions, in chunks (262k vocab), with Gemma 4's final_logit_softcapping (30.0, text_config) applied as in HF/vLLM.

Export: adapters/<name>/{adapter_config.json, adapter_model.safetensors} in the layout the competition expects and
vLLM loads: keys base_model.model.model.language_model.layers.N.{self_attn,mlp}.X_proj.lora_{A,B}.weight, bf16, one
rank, plain target_modules list, no modules_to_save/rank_pattern/alpha_pattern/DoRA/bias. The raw PEFT save (regex
target_modules, for HF reloading) goes to peft_raw/.

Speed options (all opt-in; the defaults are the rft1/rft2 path, plus per-step timing/memory fields in the log).
rft2 ran at ~470 tok/s. The main cost is attention: transformers' SDPA path gives the 50 sliding-window layers
(window 1024, head_dim 256) a dense [n, n] bool mask, so the kernel (mem-efficient; flash takes no mask) computes the
whole n x n square, ~n/1024 times the work the window needs (11x at 11k tokens) and 2x a causal kernel. The 10 global
layers (head_dim 512, causal, no mask) cannot use flash (max head_dim 256) and fall back to mem-efficient as well.
  --attn-sliding chunked   sliding layers: query blocks of --sliding-block rows attend to the keys in their window only
                           (exact; SDPA with a small [block, block+window] mask; ~2x the window's work instead of n/1024x)
  --attn-sliding flex      sliding layers: FlexAttention (torch.compile'd) with a sliding-window block mask (block-sparse)
  --attn-sliding fa2       sliding layers: flash_attn.flash_attn_func(window_size=(1023, 0)), if flash-attn is installed
  --attn-global flex       global layers: FlexAttention causal (head_dim 512: may not fit Triton's shared memory; bench it)
  --sdpa-backend X         force one SDPA backend (efficient|cudnn|flash|math) for every SDPA call, forward and backward
  --ckpt full|none|skip:N|attn|mlp
                           gradient checkpointing per layer (full = current), none, N evenly spaced layers without it,
                           or in every layer only the attention / only the MLP sub-block (selective)
  --lora-bf16              LoRA matmuls in bf16 under autocast (fp32 master weights, grads and Adam states); saves the
                           fp32 copies PEFT makes of every LoRA input (memory and bandwidth)
  --tf32                   TF32 tensor cores for the (fp32) LoRA matmuls
  --compile                torch.compile each decoder layer (dynamic shapes; attention stays outside the graph)
  --fused-adam             fused AdamW kernel
  --pack-tokens N          pack sequences of one optimizer step into forwards of <= N tokens (position_ids restart per
                           sequence, attention computed per sequence: exact). Little gain at >= 4k tokens per sequence.
Memory (saved activations measured per token with the 31B dims; fits rft1 98-99 GB at 24.9k and rft2 84.5 GB at
15.3k within ~1.5 GB): weights + LoRA/Adam ~62.7 GiB; checkpointed layer inputs 630 KiB/token (60 x 10.5); the layer
being recomputed ~565 KiB/token (sliding) / ~706 (global) with fp32 LoRA, ~500 / ~630 with --lora-bf16; CE chunk ~3
GiB; the hf path's dense sliding mask n^2 x 3 B. Peak ~ 62.7 + n x 1.34 MB + 3 GiB: 8k ~76, 16k ~87, 32k ~110 GiB.
Without checkpointing ~35 MB/token (only <= 2k tokens fit); every layer left out of checkpointing (skip:N) adds
~0.55 MB/token: about 12 layers fit at 8k, 4 at 16k, 1 at 32k, i.e. at most ~5% less recomputation.
Every option gives the same loss and gradients as the default path within bf16 rounding: `--selftest` checks that on
the tiny model (CPU) and `--bench` reports the loss and gradient error of each configuration against the first one on
the real model. `--bench N` times N forward+backward passes per configuration on the longest and the median training
sequence (and lengths given in --bench-lens) and prints tok/s and peak memory; `--bench-attn` times the attention
kernels alone (no model needed, seconds) and shows which SDPA kernel runs.
"""

import argparse
import contextlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
TARGETS = 'q_proj,o_proj,gate_proj,up_proj,down_proj'
BASE_NAME = 'google/gemma-4-31b-it-qat-w4a16-ct'
KEY_RE = re.compile(r'^base_model\.model\.model\.language_model\.layers\.(\d+)\.(self_attn|mlp)\.'
                    r'(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_([AB])\.weight$')


def log(*x):
    print(time.strftime('%H:%M:%S'), *x, flush=True)


# ------------------------------------------------------------------ data

def load_rows(path, max_len, val_tasks, limit=0, crop=0):
    rows = []
    for line in open(path):
        r = json.loads(line)
        if crop and len(r['input_ids']) > crop:  # --tiny: a window that contains labelled tokens
            first = next(i for i, x in enumerate(r['labels']) if x != -100)
            s = max(0, min(first - crop // 2, len(r['input_ids']) - crop))
            r['input_ids'], r['labels'] = r['input_ids'][s:s + crop], r['labels'][s:s + crop]
        n_lab = sum(x != -100 for x in r['labels'][1:])
        if n_lab == 0:
            continue
        r['n_lab'] = n_lab
        rows.append(r)
        if limit and len(rows) >= limit:
            break
    longest = max(len(r['input_ids']) for r in rows)
    max_len = min(max_len or longest, 32768)
    dropped = [r['id'] for r in rows if len(r['input_ids']) > max_len]
    rows = [r for r in rows if len(r['input_ids']) <= max_len]
    val = [r for r in rows if r['task'] in val_tasks]
    train = [r for r in rows if r['task'] not in val_tasks]
    return train, val, max_len, dropped


# ------------------------------------------------------------------ tiny model (CPU self-test)

def tiny_config(real_config):
    from transformers import Gemma4Config
    c = json.load(open(real_config))
    t = c['text_config']
    t.update(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
             num_global_key_value_heads=1, head_dim=16, global_head_dim=32, num_hidden_layers=6, sliding_window=16,
             layer_types=['sliding_attention', 'sliding_attention', 'full_attention'] * 2)
    v = c['vision_config']
    v.update(hidden_size=32, intermediate_size=64, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
             global_head_dim=16, num_hidden_layers=1, position_embedding_size=64)
    c.pop('quantization_config', None)
    return Gemma4Config(**{k: v for k, v in c.items() if k not in ('architectures', 'transformers_version')})


def build_tiny_checkpoint(real_config, out_dir, quantize=True, seed=0):
    """A random tiny Gemma 4 saved like the real checkpoint: text Linear layers int4 pack-quantized, group 32 (16 here
    for the small dims), symmetric, via compressed-tensors; embeddings, lm_head and the vision tower unquantized."""
    import torch
    from transformers import Gemma4ForConditionalGeneration
    torch.manual_seed(seed)
    cfg = tiny_config(real_config)
    model = Gemma4ForConditionalGeneration(cfg).to(torch.bfloat16)
    with torch.no_grad():  # random init is ~N(0, 0.02): make the tiny model's output depend on its weights
        for n, p in model.named_parameters():
            if p.dim() == 2 and 'language_model.layers' in n:
                p.normal_(0, 0.08)
    if quantize:
        from compressed_tensors.compressors import ModelCompressor
        from compressed_tensors.quantization import (QuantizationArgs, QuantizationConfig, QuantizationScheme,
                                                     QuantizationStatus, apply_quantization_config)
        from compressed_tensors.quantization.utils import calculate_qparams
        scheme = QuantizationScheme(targets=['re:.*language_model.*_proj$'], weights=QuantizationArgs(
            num_bits=4, type='int', symmetric=True, strategy='group', group_size=16))
        qcfg = QuantizationConfig(config_groups={'group_0': scheme}, format='pack-quantized',
                                  quantization_status=QuantizationStatus.FROZEN)
        apply_quantization_config(model, qcfg)
        for _, m in model.named_modules():
            if getattr(m, 'quantization_scheme', None) is not None and hasattr(m, 'weight_scale'):
                w = m.weight.data.float()
                g = w.reshape(w.shape[0], -1, 16)
                mn, mx = g.amin(-1), g.amax(-1)
                scale, zp = calculate_qparams(mn, mx, m.quantization_scheme.weights)
                m.weight_scale.data.copy_(scale.to(m.weight_scale.dtype))
                if hasattr(m, 'weight_zero_point'):
                    m.weight_zero_point.data.copy_(zp.to(m.weight_zero_point.dtype))
        compressor = ModelCompressor.from_pretrained_model(model, quantization_format='pack-quantized')
        compressor.compress_model(model)
        model.save_pretrained(out_dir)
        compressor.update_config(out_dir)
    else:
        model.save_pretrained(out_dir)
    return out_dir


# ------------------------------------------------------------------ model

def disable_qdq(model, device):
    """compressed-tensors keeps a fake-quant forward on dequantized Linear layers; the weights are already on the grid,
    so QDQ is an identity that costs time and memory. Check on one layer, then switch it off everywhere."""
    import torch
    mods = [m for _, m in model.named_modules() if getattr(m, 'quantization_scheme', None) is not None]
    if not mods:
        return 0, None
    m = mods[0]
    x = torch.randn(3, m.weight.shape[1], dtype=m.weight.dtype, device=device)
    with torch.no_grad():
        a = m(x)
        m.quantization_enabled = False
        b = m(x)
    diff = (a.float() - b.float()).abs().max().item()
    for m in mods:
        m.quantization_enabled = False
    return len(mods), diff


def load_model(a, device):
    import torch
    from transformers import AutoModelForCausalLM, CompressedTensorsConfig
    cfg = json.load(open(Path(a.model) / 'config.json'))
    kw = dict(dtype=torch.bfloat16, attn_implementation=a.attn, device_map={'': device})
    if 'quantization_config' in cfg:
        kw['quantization_config'] = CompressedTensorsConfig(run_compressed=False)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    log(f'loaded {type(model).__name__} in {time.time() - t0:.0f}s')
    n, diff = disable_qdq(model, device)
    log(f'fake-quant wrappers switched off on {n} modules (output change on one layer: {diff})')
    if diff is not None and diff > 0:
        raise SystemExit('QDQ changed the output: weights are not on the quantization grid')
    bad = [n for n, p in model.named_parameters() if p.is_floating_point() and p.dtype != torch.bfloat16]
    if bad:
        log(f'warning: {len(bad)} non-bf16 parameters, e.g. {bad[:3]}')
    model.config.use_cache = False
    return model


def text_parts(model):
    """(decoder returning last hidden states, output embedding weight, softcap)."""
    tc = model.config.get_text_config()
    inner = model.model  # Gemma4Model (multimodal wrapper) or Gemma4TextModel
    return inner, model.get_output_embeddings().weight, tc.final_logit_softcapping


def add_lora(model, a):
    from peft import LoraConfig, get_peft_model
    targets = a.targets.split(',')
    if not a.allow_kv and {'k_proj', 'v_proj'} & set(targets):
        raise SystemExit('k_proj/v_proj need --allow-kv (global-layer V = K mismatch, see module doc)')
    regex = r'.*language_model.*\.(' + '|'.join(targets) + ')'
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules=regex, bias='none',
                     task_type='CAUSAL_LM', use_rslora=False, use_dora=False)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    pm = get_peft_model(model, cfg)  # LoRA weights fp32 (autocast_adapter_dtype), base bf16 frozen
    n_mod = sum(1 for n, _ in pm.named_modules() if n.endswith('.lora_A'))
    trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    stray = [n for n, p in pm.named_parameters() if p.requires_grad and 'language_model' not in n]
    if stray:
        raise SystemExit(f'LoRA outside the language model: {stray[:3]}')
    log(f'LoRA r={a.rank} alpha={a.alpha} dropout={a.dropout} on {n_mod} modules, {trainable / 1e6:.1f}M params')
    return pm, regex


# ------------------------------------------------------------------ loss

def _ce_chunk(h, w, y, softcap):
    import torch.nn.functional as F
    logits = (h @ w.t()).float()
    if softcap:
        logits = (logits / softcap).tanh() * softcap
    return F.cross_entropy(logits, y, reduction='sum')


def seq_loss(inner, w, softcap, ids, labels, chunk, device):
    """Sum of CE over labelled next-token positions and their count. Logits only where labels != -100."""
    import torch
    from torch.utils.checkpoint import checkpoint
    x = torch.tensor([ids], device=device)
    y = torch.tensor(labels[1:], device=device)
    h = inner(input_ids=x, use_cache=False).last_hidden_state[0, :-1]
    sel = (y != -100).nonzero().squeeze(-1)
    h, y = h.index_select(0, sel), y.index_select(0, sel)
    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, len(y), chunk):
        if torch.is_grad_enabled():
            total = total + checkpoint(_ce_chunk, h[s:s + chunk], w, y[s:s + chunk], softcap, use_reentrant=False)
        else:
            total = total + _ce_chunk(h[s:s + chunk], w, y[s:s + chunk], softcap)
    return total, len(y)


def evaluate(inner, w, softcap, rows, chunk, device, rt=None):
    import torch
    tot = n = 0
    with torch.no_grad():
        for r in rows:
            with fwd_ctx(rt, device):
                s, k = seq_loss(inner, w, softcap, r['input_ids'], r['labels'], chunk, device)
            tot += s.item()
            n += k
    return tot / max(n, 1)


def _ce_chunk_each(h, w, y, softcap):
    import torch.nn.functional as F
    logits = (h @ w.t()).float()
    if softcap:
        logits = (logits / softcap).tanh() * softcap
    return F.cross_entropy(logits, y, reduction='none')


def packed_loss(inner, w, softcap, rows, chunk, device):
    """Several sequences in one forward (--pack-tokens): position_ids restart per sequence and the attention
    dispatcher (ATTN.docs) attends within each sequence only, so every sequence sees exactly what it sees alone.
    Returns (sum of CE over all labelled positions, count, [(sum, count) per sequence])."""
    import torch
    from torch.utils.checkpoint import checkpoint
    ids, pos, ys, doc = [], [], [], []
    for d, r in enumerate(rows):
        n = len(r['input_ids'])
        ids += r['input_ids']
        pos += range(n)
        ys += r['labels'][1:] + [-100]  # a sequence's last position predicts nothing (not the next sequence)
        doc += [d] * n
    ATTN.docs = [len(r['input_ids']) for r in rows]
    x = torch.tensor([ids], device=device)
    y = torch.tensor(ys, device=device)
    h = inner(input_ids=x, position_ids=torch.tensor([pos], device=device), use_cache=False).last_hidden_state[0]
    sel = (y != -100).nonzero().squeeze(-1)
    h, y = h.index_select(0, sel), y.index_select(0, sel)
    parts = []
    for s in range(0, len(y), chunk):
        if torch.is_grad_enabled():
            parts.append(checkpoint(_ce_chunk_each, h[s:s + chunk], w, y[s:s + chunk], softcap, use_reentrant=False))
        else:
            parts.append(_ce_chunk_each(h[s:s + chunk], w, y[s:s + chunk], softcap))
    each = torch.cat(parts)
    d = torch.tensor(doc, device=device).index_select(0, sel)
    per_sum = torch.zeros(len(rows), dtype=torch.float32, device=device).index_add(0, d, each)
    per_n = torch.bincount(d, minlength=len(rows)).tolist()
    return each.sum(), len(y), [(per_sum[i], per_n[i]) for i in range(len(rows))]


def pack_bins(group, pack_tokens):
    """First-fit decreasing: the sequences of one optimizer step in bins of <= pack_tokens tokens (a longer sequence
    gets a bin of its own). The loss of a step is a sum over its sequences, so binning does not change it."""
    if not pack_tokens:
        return [[r] for r in group]
    bins = []
    for r in sorted(group, key=lambda r: -len(r['input_ids'])):
        for b in bins:
            if sum(len(x['input_ids']) for x in b) + len(r['input_ids']) <= pack_tokens:
                b.append(r)
                break
        else:
            bins.append([r])
    return bins


def bin_loss(inner, w, cap, b, chunk, device):
    """(loss sum, labelled count, [(sum, count) per sequence]) for one bin; one sequence = the default seq_loss."""
    ATTN.docs = None
    if len(b) == 1:
        s, k = seq_loss(inner, w, cap, b[0]['input_ids'], b[0]['labels'], chunk, device)
        return s, k, [(s, k)]
    return packed_loss(inner, w, cap, b, chunk, device)


# ------------------------------------------------------------------ speed options (opt-in, see module doc)

# Attention dispatcher state (module-level: it must stay valid between a forward and its backward, where checkpointed
# layers recompute their attention).
ATTN = SimpleNamespace(sliding='hf', glob='hf', block=1024, docs=None, flex=None, flex_compile=True, flex_kw=None,
                       masks={}, blk_masks={})
SDPA_BACKENDS = ('auto', 'efficient', 'cudnn', 'flash', 'math')


def _sdpa_backend_ctx(name):
    if name in (None, 'auto'):
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([{'efficient': SDPBackend.EFFICIENT_ATTENTION, 'cudnn': SDPBackend.CUDNN_ATTENTION,
                         'flash': SDPBackend.FLASH_ATTENTION, 'math': SDPBackend.MATH}[name]])


def fwd_ctx(rt, device):
    """Forward context: forced SDPA backend + bf16 autocast for --lora-bf16. Nothing when rt is None/default."""
    import torch
    st = contextlib.ExitStack()
    if rt is not None:
        st.enter_context(_sdpa_backend_ctx(rt.sdpa_backend))
        if rt.lora_bf16:
            st.enter_context(torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16))
    return st


def bwd_ctx(rt):
    """Backward context: the same SDPA backend (checkpointed layers recompute their attention during backward).
    Autocast is not needed there: checkpoint restores the forward's autocast state for the recomputation."""
    return _sdpa_backend_ctx(rt.sdpa_backend if rt is not None else None)


def _repeat_kv(x, n):
    return x if n == 1 else x.repeat_interleave(n, dim=1)  # == transformers' repeat_kv (kv head j -> q heads j*n..)


def attn_hf(q, k, v, window, scale):
    """Exactly the SDPA call transformers makes for Gemma 4: sliding layers a dense [n, n] bool mask (kv repeated),
    global layers is_causal (GQA inside the kernel when head_dim <= 256, else kv repeated)."""
    import torch
    import torch.nn.functional as F
    n, rep = q.shape[2], q.shape[1] // k.shape[1]
    if window:
        key = ('dense', window, n, q.device)
        mask = ATTN.masks.get(key)
        if mask is None:
            i = torch.arange(n, device=q.device)
            mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - window)
            _cache_mask(key, mask)
        return F.scaled_dot_product_attention(q, _repeat_kv(k, rep), _repeat_kv(v, rep), attn_mask=mask, scale=scale)
    if rep > 1 and k.shape[-1] <= 256:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
    return F.scaled_dot_product_attention(q, _repeat_kv(k, rep), _repeat_kv(v, rep), is_causal=True, scale=scale)


def attn_chunked(q, k, v, window, scale, block=None):
    """Sliding-window attention by query blocks: rows [s, s+block) attend to keys [s-window, s+block) under the
    window mask (kv > q - window, kv <= q), so each block costs block x (block + window) instead of block x n.
    Exact: a query's softmax only ever covers its own window."""
    import torch
    import torch.nn.functional as F
    block = block or ATTN.block
    n, rep = q.shape[2], q.shape[1] // k.shape[1]
    k, v = _repeat_kv(k, rep), _repeat_kv(v, rep)
    outs = []
    for s in range(0, n, block):
        e = min(n, s + block)
        k0 = max(0, s - window)  # one extra (masked) key keeps the key length a multiple of 16 for the kernels
        key = ('blk', window, e - s, s - k0, q.device)  # the mask depends on the block's shape only
        mask = ATTN.blk_masks.get(key)
        if mask is None:
            qi = torch.arange(s - k0, e - k0, device=q.device)[:, None]
            ki = torch.arange(0, e - k0, device=q.device)[None, :]
            mask = ATTN.blk_masks[key] = (ki <= qi) & (ki > qi - window)
        outs.append(F.scaled_dot_product_attention(q[:, :, s:e], k[:, :, k0:e], v[:, :, k0:e], attn_mask=mask,
                                                   scale=scale))
    return outs[0] if len(outs) == 1 else torch.cat(outs, 2)


def _cache_mask(key, m):
    if len(ATTN.masks) >= 4:  # dense [n, n] masks: 1 GB at 32k tokens
        ATTN.masks.clear()
    ATTN.masks[key] = m


def attn_flex(q, k, v, window, scale):
    """FlexAttention with a block mask (sliding window or causal): only the blocks the mask touches are computed."""
    import torch
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    if ATTN.flex is None:
        if ATTN.flex_compile:
            try:
                import torch._dynamo
                torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            except Exception:
                pass
            ATTN.flex = torch.compile(flex_attention, dynamic=True)
        else:
            ATTN.flex = flex_attention
    n = q.shape[2]
    key = ('flex', window, n, q.device)
    bm = ATTN.masks.get(key)
    if bm is None:
        if window:
            def mask_mod(b, h, qi, ki):
                return (ki <= qi) & (qi - ki < window)
        else:
            def mask_mod(b, h, qi, ki):
                return ki <= qi
        bm = create_block_mask(mask_mod, None, None, n, n, device=q.device)
        _cache_mask(key, bm)
    kw = {'kernel_options': ATTN.flex_kw} if ATTN.flex_kw and not window else {}
    return ATTN.flex(q, k, v, block_mask=bm, scale=scale, enable_gqa=q.shape[1] != k.shape[1], **kw)


def attn_fa2(q, k, v, window, scale):
    """flash-attn 2 (pip flash-attn; head_dim <= 256, so sliding layers only), native window and GQA."""
    from flash_attn import flash_attn_func
    o = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), softmax_scale=scale, causal=True,
                        window_size=(window - 1, 0) if window else (-1, -1))
    return o.transpose(1, 2)


ATTN_IMPLS = {'hf': attn_hf, 'chunked': attn_chunked, 'flex': attn_flex, 'fa2': attn_fa2}


def _lt_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None,
                  **kwargs):
    """transformers AttentionInterface function 'lt_attn': q [B, H, n, d], k/v [B, Hkv, n, d] -> ([B, n, H, d], None).
    No mask is built for it (masking_utils returns None for unknown implementations); the pattern comes from
    sliding_window (HF semantics: key j visible to query i iff i - window < j <= i) and ATTN.docs (packing)."""
    import torch
    if dropout:
        raise ValueError('attention dropout is not supported by the lt_attn dispatcher')
    fn = ATTN_IMPLS[ATTN.sliding if sliding_window else ATTN.glob]
    if ATTN.docs is None:
        out = fn(query, key, value, sliding_window, scaling)
    else:
        outs, s = [], 0
        for n in ATTN.docs:
            outs.append(fn(query[:, :, s:s + n], key[:, :, s:s + n], value[:, :, s:s + n], sliding_window, scaling))
            s += n
        out = torch.cat(outs, 2)
    return out.transpose(1, 2).contiguous(), None


def _disable_dynamo(fn):
    import torch
    try:
        return torch.compiler.disable(fn)
    except Exception:
        return fn


def text_layers(model):
    """The decoder layers of the language model (PEFT model, HF model or inner model)."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
    return [m for m in model.modules() if isinstance(m, Gemma4TextDecoderLayer)]


def default_runtime(a):
    """The speed configuration from the command line (SimpleNamespace); bench configs override fields of it."""
    return SimpleNamespace(attn_sliding=a.attn_sliding, attn_global=a.attn_global, sdpa_backend=a.sdpa_backend,
                           sliding_block=a.sliding_block, ckpt=a.ckpt, lora_bf16=a.lora_bf16, tf32=a.tf32,
                           compile=a.compile, pack=a.pack_tokens, custom=False)


def plain_runtime():
    """The default (rft1/rft2) configuration."""
    return SimpleNamespace(attn_sliding='hf', attn_global='hf', sdpa_backend='auto', sliding_block=1024, ckpt='full',
                           lora_bf16=False, tf32=False, compile=False, pack=0, custom=False)


def runtime_is_default(rt):
    return (rt.attn_sliding == 'hf' and rt.attn_global == 'hf' and rt.sdpa_backend == 'auto' and rt.ckpt == 'full'
            and not rt.lora_bf16 and not rt.tf32 and not rt.compile and not rt.pack and not rt.custom)


def _ckpt_wrap(mod, on):
    import torch
    from torch.utils.checkpoint import checkpoint
    if on and not hasattr(mod, '_lt_forward'):
        orig = mod.forward

        def fwd(*args, **kw):
            if mod.training and torch.is_grad_enabled():
                return checkpoint(orig, *args, use_reentrant=False, **kw)
            return orig(*args, **kw)
        mod._lt_forward = orig
        mod.forward = fwd
    elif not on and hasattr(mod, '_lt_forward'):
        mod.forward = mod._lt_forward
        del mod._lt_forward
        mod.__dict__.pop('forward', None)


def apply_runtime(pm, rt, a):
    """Switch the model to speed configuration rt (all switches are reversible, for --bench)."""
    import torch
    layers = text_layers(pm)
    tc = pm.get_base_model().config.get_text_config()
    # attention
    custom = rt.attn_sliding != 'hf' or rt.attn_global != 'hf' or bool(rt.pack) or rt.custom
    if rt.attn_global not in ('hf', 'flex'):
        raise SystemExit(f'--attn-global {rt.attn_global}: global layers support hf or flex')
    if custom:
        from transformers import AttentionInterface
        AttentionInterface.register('lt_attn', _disable_dynamo(_lt_attention))
        for m in pm.modules():
            if type(m).__name__ == 'Gemma4TextAttention' and m.config is not tc:
                raise SystemExit('attention modules do not share the text config: cannot switch attention')
    tc._attn_implementation_internal = 'lt_attn' if custom else a.attn
    ATTN.sliding, ATTN.glob, ATTN.block = rt.attn_sliding, rt.attn_global, rt.sliding_block
    ATTN.masks.clear()
    ATTN.blk_masks.clear()
    # gradient checkpointing
    mode = rt.ckpt
    if mode != 'none' and not a.grad_ckpt:
        raise SystemExit('--ckpt needs gradient checkpointing (drop --no-grad-ckpt)')
    skip = set()
    if mode.startswith('skip:'):
        k = int(mode.split(':')[1])
        skip = {int((i + 0.5) * len(layers) / k) for i in range(k)} if k else set()
    elif mode not in ('full', 'none', 'attn', 'mlp'):
        raise SystemExit(f'bad --ckpt {mode}')
    for i, l in enumerate(layers):
        l.gradient_checkpointing = (mode == 'full' or (mode.startswith('skip:') and i not in skip))
        _ckpt_wrap(l.self_attn, mode == 'attn')
        _ckpt_wrap(l.mlp, mode == 'mlp')
    # LoRA input casting (PEFT casts every LoRA input to fp32 = the LoRA weight dtype; --lora-bf16 uses autocast)
    for m in pm.modules():
        if hasattr(m, 'cast_input_dtype_enabled'):
            m.cast_input_dtype_enabled = not rt.lora_bf16
    # TF32
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(rt.tf32)
    # torch.compile per decoder layer
    for l in layers:
        if rt.compile and getattr(l, '_compiled_call_impl', None) is None:
            l.compile(dynamic=True, backend=a.compile_backend)
        elif not rt.compile:
            l._compiled_call_impl = None
    if rt.compile:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    return layers


def describe_runtime(rt):
    """The fields that differ from the default path, e.g. 'attn_sliding=flex,lora_bf16=True' ('base' if none)."""
    ref = vars(plain_runtime())
    d = {k: v for k, v in vars(rt).items() if v != ref.get(k)}
    if rt.attn_sliding != 'chunked':
        d.pop('sliding_block', None)
    return ','.join(f'{k}={v}' for k, v in d.items()) or 'base'


def parse_bench_configs(spec, base):
    """'a=b,c=d;e=f' -> list of runtime namespaces (fields of default_runtime), each starting from base."""
    out = []
    for part in [p.strip() for p in spec.split(';') if p.strip()]:
        rt = SimpleNamespace(**vars(base))
        if part != 'base':
            for kv in part.split(','):
                k, v = kv.split('=', 1)
                k = k.strip().replace('-', '_')
                if not hasattr(rt, k):
                    raise SystemExit(f'unknown bench config key {k} (known: {", ".join(vars(rt))})')
                cur = getattr(rt, k)
                if isinstance(cur, bool):
                    v = v.lower() in ('1', 'true', 'yes', 'on')
                elif isinstance(cur, int):
                    v = int(v)
                setattr(rt, k, v)
        out.append(rt)
    return out


# tf32=1 only changes the rounding of the fp32 LoRA matmuls: its "grad err" vs base is the bf16 noise floor to judge
# the other configs' errors by.
DEFAULT_BENCH_CONFIGS = ('base; tf32=1; attn_sliding=chunked; attn_sliding=flex; attn_sliding=flex,attn_global=flex; '
                         'attn_sliding=chunked,lora_bf16=1; attn_sliding=flex,lora_bf16=1; '
                         'attn_sliding=flex,lora_bf16=1,compile=1; attn_sliding=flex,lora_bf16=1,ckpt=skip:4')


def _sync(device):
    import torch
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def _peak_gb(device):
    import torch
    return round(torch.cuda.max_memory_allocated() / 2 ** 30, 1) if str(device).startswith('cuda') else None


def bench_rows(train, spec, seed=0):
    """Bench inputs: 'longest', 'median', 'group' (one optimizer step's sequences, packed when the config packs) or a
    token count (training sequences concatenated and cut to that length)."""
    by_len = sorted(train, key=lambda r: len(r['input_ids']))
    out = []
    for s in [x.strip() for x in spec.split(',') if x.strip()]:
        if s == 'longest':
            out.append((f'longest {len(by_len[-1]["input_ids"])}', [by_len[-1]]))
        elif s == 'median':
            r = by_len[len(by_len) // 2]
            out.append((f'median {len(r["input_ids"])}', [r]))
        elif s.startswith('group'):
            k = int(s[5:] or 8)
            g = random.Random(seed).sample(train, min(k, len(train)))
            out.append((f'group {len(g)}x {sum(len(r["input_ids"]) for r in g)}', g))
        else:
            n = int(s)
            ids, lab = [], []
            for r in by_len[::-1]:
                ids += r['input_ids']
                lab += r['labels']
                if len(ids) >= n:
                    break
            if len(ids) < n:
                continue
            row = {'id': f'synthetic_{n}', 'task': '-', 'input_ids': ids[:n], 'labels': lab[:n]}
            row['n_lab'] = sum(x != -100 for x in row['labels'][1:])
            out.append((f'tokens {n}', [row]))
    return out


def run_bench(a, pm, inner, w, cap, train, device, out):
    """--bench N: per configuration and bench input, 1 warm-up + N timed forward+backward passes; tok/s, peak memory,
    loss and gradient error vs the first configuration. LoRA dropout is off and lora_B is randomised (so lora_A gets
    gradients too): the adapter is not trained, the run exits after the table."""
    import torch
    params = [p for p in pm.parameters() if p.requires_grad]
    g = torch.Generator(device='cpu').manual_seed(a.seed)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if '.lora_B.' in n:
                p.copy_(torch.randn(p.shape, generator=g) * 1e-3)
    for m in pm.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    base = default_runtime(a)
    configs = parse_bench_configs(a.bench_configs, base)
    inputs = bench_rows(train, a.bench_lens, a.seed)
    log(f'bench: {len(configs)} configs x {len(inputs)} inputs ({", ".join(x[0] for x in inputs)}), '
        f'{a.bench} timed passes each')
    refs, results = {}, []
    for rt in configs:
        apply_runtime(pm, rt, a)
        for label, rows in inputs:
            rec = {'config': describe_runtime(rt), 'input': label}
            try:
                bins = pack_bins(rows, rt.pack)
                denom = sum(r['n_lab'] for r in rows)
                tok = sum(len(r['input_ids']) for r in rows)
                times, peak, first = [], 0.0, None
                for it in range(1 + a.bench):
                    for p in params:
                        p.grad = None
                    if str(device).startswith('cuda'):
                        torch.cuda.reset_peak_memory_stats()
                    tf = tb = 0.0
                    loss = 0.0
                    for b in bins:
                        _sync(device)
                        t0 = time.time()
                        with fwd_ctx(rt, device):
                            s, k, _ = bin_loss(inner, w, cap, b, a.ce_chunk, device)
                        _sync(device)
                        t1 = time.time()
                        with bwd_ctx(rt):
                            (s / denom).backward()
                        _sync(device)
                        tf, tb = tf + t1 - t0, tb + time.time() - t1
                        loss += s.item()
                    if it == 0:
                        first = tf + tb
                    else:
                        times.append((tf, tb))
                    peak = max(peak, _peak_gb(device) or 0.0)
                fwd = sum(t[0] for t in times) / len(times)
                bwd = sum(t[1] for t in times) / len(times)
                rec.update(tokens=tok, labelled=denom, fwd_s=round(fwd, 3), bwd_s=round(bwd, 3),
                           tok_per_s=round(tok / (fwd + bwd)), first_pass_s=round(first, 1), peak_gb=peak,
                           loss=loss / denom)
                grads = [p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)
                         for p in params]
                if label not in refs:
                    refs[label] = (loss / denom, grads, rec['config'])
                else:
                    rl, rg, rc = refs[label]
                    rec['vs'] = rc
                    num = sum(((x - y) ** 2).sum() for x, y in zip(grads, rg)).sqrt().item()
                    den = sum((y ** 2).sum() for y in rg).sqrt().item()
                    rec.update(loss_diff=abs(loss / denom - rl), grad_rel_err=num / max(den, 1e-30))
                del grads
            except torch.cuda.OutOfMemoryError:
                rec['error'] = 'OOM'
            except Exception as e:  # report and go on: one broken option must not cost the rented hour
                rec['error'] = f'{type(e).__name__}: {str(e)[:300]}'
            for p in params:
                p.grad = None
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
            results.append(rec)
            log('bench ' + json.dumps(rec))
    # optimizer step: foreach (default) vs fused
    for p in params:
        p.grad = torch.zeros_like(p)
    for fused in ((False, True) if str(device).startswith('cuda') else (False,)):
        opt = torch.optim.AdamW(params, lr=0.0, fused=fused or None)
        ts = []
        for _ in range(4):
            _sync(device)
            t0 = time.time()
            torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
            opt.step()
            _sync(device)
            ts.append(time.time() - t0)
        results.append({'config': f'optimizer fused={fused}', 'opt_step_ms': round(1000 * min(ts[1:]), 1)})
        log('bench ' + json.dumps(results[-1]))
        del opt
    json.dump(results, open(out / 'bench.json', 'w'), indent=1)
    print_bench_table(results)
    return results


def print_bench_table(results):
    w = max([len(r['config']) for r in results] + [6])
    print(f'\n{"config":{w}s} {"input":16s} {"tok/s":>7s} {"fwd s":>7s} {"bwd s":>7s} {"peak GB":>7s} '
          f'{"1st s":>6s} {"dloss":>8s} {"grad err":>8s}')
    for r in results:
        if 'opt_step_ms' in r:
            print(f'{r["config"]:{w}s} optimizer step {r["opt_step_ms"]} ms')
            continue
        if 'error' in r:
            print(f'{r["config"]:{w}s} {r["input"]:16s} {r["error"][:100]}')
            continue
        print(f'{r["config"]:{w}s} {r["input"]:16s} {r["tok_per_s"]:7d} {r["fwd_s"]:7.2f} {r["bwd_s"]:7.2f} '
              f'{r["peak_gb"] or 0:7.1f} {r["first_pass_s"]:6.1f} {r.get("loss_diff", 0):8.1e} '
              f'{r.get("grad_rel_err", 0):8.1e}')
    print(flush=True)


# ------------------------------------------------------------------ attention microbench / check (no model needed)

def attn_dims(a):
    """(heads, [(kind, kv_heads, head_dim, window)]) from the model config (or Gemma 4 31B's when not downloaded)."""
    t = {}
    p = Path(a.model) / 'config.json'
    if p.exists():
        t = json.load(open(p)).get('text_config', {})
    h = t.get('num_attention_heads', 32)
    return h, [('sliding', t.get('num_key_value_heads', 16), t.get('head_dim', 256), t.get('sliding_window', 1024)),
               ('global', t.get('num_global_key_value_heads', 4) if t.get('attention_k_eq_v', True)
                else t.get('num_key_value_heads', 16), t.get('global_head_dim', 512), None)]


def attn_check(fn, h, hkv, d, window, n, device, backend='auto', seed=0):
    """Relative error of fn's output and q/k/v gradients (bf16) vs an fp32 math-SDPA reference with the HF mask."""
    import torch
    import torch.nn.functional as F
    g = torch.Generator(device='cpu').manual_seed(seed)
    q0, k0, v0 = (torch.randn(1, x, n, d, generator=g).to(device) for x in (h, hkv, hkv))
    go = torch.randn(1, h, n, d, generator=g).to(device)

    def run(f, dt, be):
        q, k, v = (t.detach().to(dt).clone().requires_grad_() for t in (q0, k0, v0))  # leaves in every dtype
        with _sdpa_backend_ctx(be):
            o = f(q, k, v)
            o.backward(go.to(dt))
        return [t.float() for t in (o, q.grad, k.grad, v.grad)]

    def ref(q, k, v):
        i = torch.arange(n, device=device)
        m = i[None, :] <= i[:, None]
        if window:
            m = m & (i[None, :] > i[:, None] - window)
        rep = h // hkv
        return F.scaled_dot_product_attention(q, _repeat_kv(k, rep), _repeat_kv(v, rep), attn_mask=m, scale=1.0)
    r = run(ref, torch.float32, 'math')
    got = run(lambda q, k, v: fn(q, k, v, window, 1.0), torch.bfloat16 if str(device).startswith('cuda')
              else torch.float32, backend)
    return max(((x - y).norm() / y.norm()).item() for x, y in zip(got, r))


def bench_attention(a, device):
    """--bench-attn: forward+backward time and peak memory of each attention implementation for both layer kinds at
    the --bench-attn-lens lengths, an accuracy check vs fp32 math, and the CUDA kernels SDPA picks on its own."""
    import torch
    h, kinds = attn_dims(a)
    cuda = str(device).startswith('cuda')
    lens = [int(x) for x in a.bench_attn_lens.split(',')] if cuda else [64, 160]
    dt = torch.bfloat16 if cuda else torch.float32
    try:
        import flash_attn  # noqa: F401
        has_fa = True
    except Exception:
        has_fa = False
    log(f'attention bench on {torch.cuda.get_device_name() if cuda else "cpu"}, torch {torch.__version__}, '
        f'cudnn {torch.backends.cudnn.version() if cuda else "-"}, flash-attn {"yes" if has_fa else "no"}; '
        f'heads {h}, kinds {kinds}')
    results = []
    for kind, hkv, d, window in kinds:
        cands = [('hf', 'auto'), ('hf', 'efficient'), ('hf', 'cudnn'), ('hf', 'flash'), ('hf', 'math')]
        if window:
            cands += [('chunked', 'auto'), ('chunked:512', 'auto'), ('chunked:2048', 'auto'), ('chunked', 'cudnn'),
                      ('flex', 'auto')] + ([('fa2', 'auto')] if has_fa else [])
        else:
            cands += [('flex', 'auto')]
        for impl, be in cands:
            name, _, blk = impl.partition(':')
            base_fn = ATTN_IMPLS[name]
            fn = (lambda q, k, v, w_, s_, f=base_fn, b=int(blk): f(q, k, v, w_, s_, block=b)) if blk else base_fn
            rec = {'kind': kind, 'impl': impl, 'backend': be}
            try:
                n_chk = 3 * window + 77 if window else 2077
                if not cuda:
                    n_chk = 3 * window + 7 if window else 77
                rec['rel_err'] = float(f'{attn_check(fn, h, hkv, d, window, n_chk, device, be):.2e}')
            except Exception as e:
                rec['error'] = f'{type(e).__name__}: {str(e)[:200]}'
                results.append(rec)
                log('attn ' + json.dumps(rec))
                continue
            for n in lens:
                if be == 'math' and n > 8192:
                    continue
                try:
                    q = torch.randn(1, h, n, d, device=device, dtype=dt, requires_grad=True)
                    k = torch.randn(1, hkv, n, d, device=device, dtype=dt, requires_grad=True)
                    v = torch.randn(1, hkv, n, d, device=device, dtype=dt, requires_grad=True)
                    go = torch.randn(1, h, n, d, device=device, dtype=dt)
                    ts = []
                    for it in range(4 if cuda else 2):
                        if cuda:
                            torch.cuda.reset_peak_memory_stats()
                        base_mem = torch.cuda.memory_allocated() if cuda else 0
                        _sync(device)
                        t0 = time.time()
                        with _sdpa_backend_ctx(be):
                            o = fn(q, k, v, window, 1.0)
                            _sync(device)
                            t1 = time.time()
                            o.backward(go)
                        _sync(device)
                        if it:
                            ts.append((t1 - t0, time.time() - t1))
                        q.grad = k.grad = v.grad = None
                        del o
                    f_ = sum(t[0] for t in ts) / len(ts)
                    b_ = sum(t[1] for t in ts) / len(ts)
                    keys = min(n, window) if window else n / 2  # useful (unmasked) keys per query, on average
                    flops = 4 * h * d * n * keys * 3.5  # fwd (2 matmuls) + bwd (5 matmuls incl. recompute) ~ 3.5x fwd
                    rec[f'n{n}'] = {'fwd_ms': round(1000 * f_, 2), 'bwd_ms': round(1000 * b_, 2),
                                    'useful_tflops': round(flops / (f_ + b_) / 1e12, 1),
                                    'peak_gb': round((torch.cuda.max_memory_allocated() - base_mem) / 2 ** 30, 2)
                                    if cuda else None}
                    del q, k, v, go
                except torch.cuda.OutOfMemoryError:
                    rec[f'n{n}'] = 'OOM'
                except Exception as e:
                    rec[f'n{n}'] = f'{type(e).__name__}: {str(e)[:120]}'
                if cuda:
                    torch.cuda.empty_cache()
            if cuda and impl == 'hf' and be == 'auto':  # which kernels SDPA picks by itself
                try:
                    from torch.profiler import ProfilerActivity, profile
                    n = lens[0]
                    q = torch.randn(1, h, n, d, device=device, dtype=dt, requires_grad=True)
                    k = torch.randn(1, hkv, n, d, device=device, dtype=dt, requires_grad=True)
                    v = torch.randn(1, hkv, n, d, device=device, dtype=dt, requires_grad=True)
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        fn(q, k, v, window, 1.0).sum().backward()
                        _sync(device)
                    ev = sorted(prof.key_averages(), key=lambda e: -getattr(e, 'device_time_total',
                                                                              getattr(e, 'cuda_time_total', 0)))
                    rec['top_kernels'] = [e.key[:90] for e in ev[:4]]
                except Exception as e:
                    rec['top_kernels'] = f'profiler failed: {e}'
            results.append(rec)
            log('attn ' + json.dumps(rec))
    out = Path(a.out)
    json.dump(results, open(out / 'bench_attn.json', 'w'), indent=1)
    print(f'\n{"kind":8s} {"impl":13s} {"backend":10s} {"rel err":>8s}  ' +
          '  '.join(f'{"n=" + str(n) + " ms (TF/s, GB)":>26s}' for n in lens))
    for r in results:
        if 'error' in r:
            print(f'{r["kind"]:8s} {r["impl"]:13s} {r["backend"]:10s} {r["error"][:100]}')
            continue
        cells = []
        for n in lens:
            c = r.get(f'n{n}')
            if c is None:
                cells.append(f'{"-":>26s}')
            elif isinstance(c, str):
                cells.append(f'{c[:26]:>26s}')
            else:
                cells.append(f'{c["fwd_ms"] + c["bwd_ms"]:9.1f} ({c["useful_tflops"]:5.1f}, {c["peak_gb"] or 0:5.1f})'
                             .rjust(26))
        print(f'{r["kind"]:8s} {r["impl"]:13s} {r["backend"]:10s} {r["rel_err"]:8.1e}  ' + '  '.join(cells))
        if r.get('top_kernels'):
            print(f'{"":33s} SDPA picks: {r["top_kernels"]}')
    print(flush=True)
    return results


def check_attention_runtime(rt, a, device):
    """Before training with a non-default attention: compare it with the fp32 reference on random inputs with the real
    head dims (catches a broken kernel/backend on the GPU at hand). Aborts when the error exceeds 3x the default
    path's own bf16 error (+1e-3)."""
    h, kinds = attn_dims(a)
    for kind, hkv, d, window in kinds:
        impl = rt.attn_sliding if window else rt.attn_global
        if impl == 'hf' and rt.sdpa_backend == 'auto':
            continue
        n = (3 * window + 77) if window else 2077
        if not str(device).startswith('cuda'):
            return
        fn = ATTN_IMPLS[impl]
        err = attn_check(fn, h, hkv, d, window, n, device, rt.sdpa_backend)
        ref = attn_check(attn_hf, h, hkv, d, window, n, device, 'auto')
        log(f'attention check {kind} ({impl}, sdpa {rt.sdpa_backend}): rel err {err:.2e} vs default path {ref:.2e}')
        if err > 3 * ref + 1e-3:
            raise SystemExit(f'{kind} attention {impl} disagrees with the reference')


# ------------------------------------------------------------------ selftest (tiny model, CPU)

SELFTEST_CHECKS = [  # (name, runtime overrides, dtypes, forward only)
    ('dispatcher, HF-equivalent attention', dict(custom=True), ('fp32', 'bf16'), False),
    ('sliding chunked, block 16', dict(attn_sliding='chunked', sliding_block=16), ('fp32', 'bf16'), False),
    ('sliding chunked, block 24', dict(attn_sliding='chunked', sliding_block=24), ('fp32', 'bf16'), False),
    ('sliding chunked, block 1024 (one block)', dict(attn_sliding='chunked'), ('fp32', 'bf16'), False),
    ('sdpa backend math', dict(sdpa_backend='math'), ('fp32', 'bf16'), False),
    ('flex sliding + global (CPU: forward only)', dict(attn_sliding='flex', attn_global='flex'), ('fp32', 'bf16'),
     True),
    ('ckpt none', dict(ckpt='none'), ('fp32', 'bf16'), False),
    ('ckpt skip:2', dict(ckpt='skip:2'), ('fp32', 'bf16'), False),
    ('ckpt attn (selective)', dict(ckpt='attn'), ('fp32', 'bf16'), False),
    ('ckpt mlp (selective)', dict(ckpt='mlp'), ('fp32', 'bf16'), False),
    ('pack all 3 sequences', dict(pack=100000), ('fp32', 'bf16'), False),
    ('pack 2+1 + chunked', dict(pack=300, attn_sliding='chunked', sliding_block=16), ('fp32', 'bf16'), False),
    ('lora bf16 (autocast)', dict(lora_bf16=True), ('bf16',), False),
    ('tf32 (no-op on CPU)', dict(tf32=True), ('fp32', 'bf16'), False),
    ('compile', dict(compile=True), ('fp32', 'bf16'), False),
    ('compile + chunked + lora bf16 + skip:2 + pack', dict(compile=True, attn_sliding='chunked', sliding_block=16,
                                                           lora_bf16=True, ckpt='skip:2', pack=300), ('bf16',), False),
]


def selftest(a, device):
    """--selftest: on the tiny model (3 sequences longer than its sliding window of 16, different lengths), loss and
    LoRA gradients of every speed option vs the default path, in fp32 (tight) and bf16 (the training dtype).
    Then dropout consistency: with dropout on, checkpointed (recomputed) and non-checkpointed runs from the same seed
    must give the same gradients, eager and compiled."""
    import torch
    a.dropout = 0.3  # PEFT builds nn.Dropout modules; set to p=0 for the comparisons, back to 0.3 for the dropout test
    model = load_model(a, device)
    with torch.no_grad():  # temper the random attention: with unit q/k norms (and Gemma 4's scaling 1.0) the tiny
        for n, p in model.named_parameters():  # model's softmax is near-argmax, bf16 rounding flips it and its
            if 'language_model' in n and ('q_norm' in n or 'k_norm' in n):  # gradients turn chaotic (58% bf16 vs
                p.mul_(0.3)  # fp32 error on the default path); x0.3 brings that to ~1.5%
    pm, _ = add_lora(model, a)
    torch.manual_seed(1)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if '.lora_' in n:  # lora_B = 0 at init would leave lora_A without gradients
                p.normal_(0, 0.02)
    drops = [m for m in pm.modules() if isinstance(m, torch.nn.Dropout)]
    inner, w, cap = text_parts(pm.get_base_model())
    rows, _, _, _ = load_rows(a.data, 0, set(), limit=3, crop=160)
    for r, n in zip(rows, (160, 137, 101)):  # different lengths: blocks, packing offsets
        r['input_ids'], r['labels'] = r['input_ids'][:n], r['labels'][:n]
        r['n_lab'] = sum(x != -100 for x in r['labels'][1:])
    rows = [r for r in rows if r['n_lab']]
    log(f'selftest: {len(rows)} sequences, lengths {[len(r["input_ids"]) for r in rows]}, sliding window '
        f'{pm.get_base_model().config.get_text_config().sliding_window}')
    params = [p for p in pm.parameters() if p.requires_grad]
    base = plain_runtime()

    def run(rt, grad=True, seed=None):
        apply_runtime(pm, rt, a)
        pm.train()
        for p in params:
            p.grad = None
        if seed is not None:
            torch.manual_seed(seed)
        denom = sum(r['n_lab'] for r in rows)
        total = 0.0
        with torch.set_grad_enabled(grad):
            for b in pack_bins(rows, rt.pack):
                with fwd_ctx(rt, device):
                    s, _, _ = bin_loss(inner, w, cap, b, a.ce_chunk, device)
                if grad:
                    with bwd_ctx(rt):
                        (s / denom).backward()
                total += s.item()
        return total / denom, [p.grad.detach().float().clone() for p in params] if grad else None

    def err(g, ref):
        return (sum(((x - y) ** 2).sum() for x, y in zip(g, ref)).sqrt() / sum((y ** 2).sum() for y in ref).sqrt()
                ).item()

    failures, lines, bf16_runs = [], [], {}
    for dt in ('bf16', 'fp32'):  # bf16 first (the loaded weights), then the same weights in fp32
        if dt == 'fp32':
            pm.float()
        for m in drops:
            m.p = 0.0
        ref_loss, ref_g = run(base)
        again_loss, again_g = run(base)
        lines.append(f'[{dt}] default path: loss {ref_loss:.6f}; repeat: dloss {abs(again_loss - ref_loss):.1e}, '
                     f'grad err {err(again_g, ref_g):.1e}')
        if dt == 'bf16':
            bf16_runs['default path'] = (ref_loss, ref_g)
        for name, over, dts, fwd_only in SELFTEST_CHECKS:
            if dt not in dts:
                continue
            rt = SimpleNamespace(**{**vars(base), **over})
            try:
                res = run(rt, grad=not fwd_only)
            except Exception as e:
                lines.append(f'[{dt}] {name}: ERROR {type(e).__name__}: {str(e)[:200]}')
                failures.append(f'{dt} {name}')
                continue
            if dt == 'bf16':  # judged below against the fp32 reference
                bf16_runs[name] = res
                continue
            loss, g = res
            dl = abs(loss - ref_loss) / abs(ref_loss)
            ge = err(g, ref_g) if g is not None else float('nan')
            ok = dl <= 1e-5 and (g is None or ge <= 1e-4)
            lines.append(f'[fp32] {name}: rel dloss {dl:.1e}, grad rel err {"-" if g is None else f"{ge:.1e}"}'
                         f' -> {"ok" if ok else "FAIL"}')
            if not ok:
                failures.append(f'fp32 {name}')
        if dt == 'fp32':
            # bf16 runs vs the fp32 reference: each option within 1.5x the default bf16 path's own rounding error
            l0 = abs(bf16_runs['default path'][0] - ref_loss) / abs(ref_loss)
            g0 = err(bf16_runs['default path'][1], ref_g)
            lines.append(f'[bf16 vs fp32] default path: rel dloss {l0:.1e}, grad rel err {g0:.1e} (= the bf16 noise '
                         f'floor of this tiny random model)')
            for name, (loss, g) in bf16_runs.items():
                if name == 'default path':
                    continue
                dl = abs(loss - ref_loss) / abs(ref_loss)
                ge = err(g, ref_g) if g is not None else float('nan')
                ok = dl <= 1.5 * l0 + 3e-4 and (g is None or ge <= 1.5 * g0 + 1e-2)
                lines.append(f'[bf16 vs fp32] {name}: rel dloss {dl:.1e}, grad rel err '
                             f'{"-" if g is None else f"{ge:.1e}"} -> {"ok" if ok else "FAIL"}')
                if not ok:
                    failures.append(f'bf16 {name}')
            # dropout: recomputation (checkpointing) must redraw the forward's masks
            for m in drops:
                m.p = 0.3
            for comp in (False, True):
                on = SimpleNamespace(**{**vars(base), 'compile': comp})
                off = SimpleNamespace(**{**vars(base), 'compile': comp, 'ckpt': 'none'})
                l1, g1 = run(on, seed=7)
                l2, g2 = run(off, seed=7)
                ge = err(g1, g2)
                ok = abs(l1 - l2) <= 1e-5 * abs(l2) and ge <= 1e-4
                lines.append(f'[fp32] dropout 0.3, ckpt full vs none{" (compiled)" if comp else ""}: dloss '
                             f'{abs(l1 - l2):.1e}, grad rel err {ge:.1e} -> {"ok" if ok else "FAIL"}')
                if not ok:
                    failures.append(f'dropout ckpt consistency compile={comp}')
    apply_runtime(pm, base, a)
    for line in lines:
        log(line)
    log('attention microbench (CPU smoke):')
    bench_attention(a, device)
    if failures:
        raise SystemExit(f'selftest FAILED: {failures}')
    log('selftest passed')


# ------------------------------------------------------------------ export

def export_adapter(raw_dir, out_dir, rank, alpha, text_config, allow_kv):
    """PEFT save -> the scorer/vLLM layout (see module doc). Returns the tensor dict written."""
    import torch
    from safetensors.torch import load_file, save_file
    raw = load_file(str(Path(raw_dir) / 'adapter_model.safetensors'))
    out = {}
    for k, v in raw.items():
        k2 = k.replace('base_model.model.model.layers.', 'base_model.model.model.language_model.layers.')  # text-only
        if not KEY_RE.match(k2):
            raise SystemExit(f'unexpected adapter key {k}')
        out[k2] = v.to(torch.bfloat16).contiguous()
    for k, v in out.items():
        r = v.shape[0] if k.endswith('lora_A.weight') else v.shape[1]
        if r != rank:
            raise SystemExit(f'non-uniform rank: {k} {tuple(v.shape)}')
    if allow_kv and text_config.get('attention_k_eq_v'):
        for i, lt in enumerate(text_config['layer_types']):
            if lt != 'full_attention':
                continue
            for ab in 'AB':
                kk = f'base_model.model.model.language_model.layers.{i}.self_attn.k_proj.lora_{ab}.weight'
                vk = kk.replace('k_proj', 'v_proj')
                if kk in out and vk not in out:
                    out[vk] = out[kk].clone()
    mods = sorted({KEY_RE.match(k).group(3) for k in out})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_file(out, str(Path(out_dir) / 'adapter_model.safetensors'), metadata={'format': 'pt'})
    json.dump({'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': rank, 'lora_alpha': alpha, 'lora_dropout': 0.0,
               'target_modules': mods, 'bias': 'none', 'use_dora': False, 'use_rslora': False,
               'modules_to_save': None, 'fan_in_fan_out': False, 'inference_mode': True,
               'base_model_name_or_path': BASE_NAME, 'rank_pattern': {}, 'alpha_pattern': {},
               'layers_to_transform': None},
              open(Path(out_dir) / 'adapter_config.json', 'w'), indent=1)
    return out


def check_export(pm, exported):
    """Exported tensors == the trained LoRA weights (bf16-rounded)."""
    import torch
    live = {}
    for n, p in pm.named_parameters():
        if '.lora_' in n:
            k = n.replace('.default.weight', '.weight')
            k = k.replace('base_model.model.model.layers.', 'base_model.model.model.language_model.layers.')
            live[k] = p.detach()
    missing = [k for k in live if k not in exported]
    worst = max(((live[k].to(torch.bfloat16).float() - exported[k].float().to(live[k].device)).abs().max().item()
                 for k in live if k in exported), default=0.0)
    return len(live), missing, worst


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='/models/gemma-4-31b-it-qat-w4a16-ct')
    ap.add_argument('--data', default=str(ROOT / 'data/rft/train_tokens.jsonl'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', default='rft1', help='adapter name (adapters/<name>/), ^[a-zA-Z0-9_.-]+$')
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--dropout', type=float, default=0.05)
    ap.add_argument('--targets', default=TARGETS)
    ap.add_argument('--allow-kv', action='store_true', help='allow k_proj/v_proj (exports global k_proj as v_proj too)')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=float, default=0.03, help='warmup fraction of optimizer steps')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--max-grad-norm', type=float, default=1.0)
    ap.add_argument('--max-len', type=int, default=0, help='0 = longest sequence in the data (cap 32768)')
    ap.add_argument('--val-tasks', default='', help='comma-separated tasks held out for validation loss')
    ap.add_argument('--ce-chunk', type=int, default=1024, help='labelled positions per logits chunk')
    ap.add_argument('--no-preflight', dest='preflight', action='store_false',
                    help='skip the forward+backward on the longest sequence before training (OOM/speed check)')
    ap.add_argument('--attn', default='sdpa')
    ap.add_argument('--no-grad-ckpt', dest='grad_ckpt', action='store_false')
    ap.add_argument('--eval-before', action='store_true', help='validation loss before training too')
    ap.add_argument('--max-steps', type=int, default=0, help='stop after N optimizer steps (smoke tests)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tiny', action='store_true', help='CPU self-test on a tiny random Gemma 4 (see module doc)')
    ap.add_argument('--tiny-config', default=str(ROOT / 'data/rft/tokenizer/config.json'))
    sp = ap.add_argument_group('speed options (opt-in; defaults = the rft1/rft2 path, see module doc)')
    sp.add_argument('--attn-sliding', choices=['hf', 'chunked', 'flex', 'fa2'], default='hf',
                    help='sliding-window layers: hf = transformers SDPA with a dense n x n mask')
    sp.add_argument('--attn-global', choices=['hf', 'flex'], default='hf', help='global (head_dim 512) layers')
    sp.add_argument('--sdpa-backend', choices=SDPA_BACKENDS, default='auto', help='force one SDPA kernel')
    sp.add_argument('--sliding-block', type=int, default=1024, help='query block of --attn-sliding chunked')
    sp.add_argument('--flex-kernel-options', default='', help='JSON kernel_options for FlexAttention (global layers)')
    sp.add_argument('--no-flex-compile', dest='flex_compile', action='store_false', help='eager FlexAttention (slow)')
    sp.add_argument('--ckpt', default='full', help='gradient checkpointing: full | none | skip:N | attn | mlp')
    sp.add_argument('--lora-bf16', action='store_true', help='LoRA matmuls in bf16 (autocast), fp32 master weights')
    sp.add_argument('--tf32', action='store_true', help='TF32 for fp32 matmuls (the fp32 LoRA path)')
    sp.add_argument('--compile', action='store_true', help='torch.compile each decoder layer')
    sp.add_argument('--compile-backend', default='inductor')
    sp.add_argument('--fused-adam', action='store_true')
    sp.add_argument('--pack-tokens', type=int, default=0, help='pack sequences of a step into forwards of <= N tokens')
    sp.add_argument('--bench', type=int, default=0, metavar='N',
                    help='time N forward+backward passes per --bench-configs config and --bench-lens input, then exit')
    sp.add_argument('--bench-configs', default=DEFAULT_BENCH_CONFIGS,
                    help='";"-separated configs, each "base" or comma-separated key=value overrides of the flags above '
                         '(attn_sliding, attn_global, sdpa_backend, sliding_block, ckpt, lora_bf16, tf32, compile, pack)')
    sp.add_argument('--bench-lens', default='median,longest',
                    help='bench inputs: longest, median, groupK (K sequences = one step), or token counts')
    sp.add_argument('--bench-attn', action='store_true',
                    help='attention kernel microbench (no model weights needed; --model config.json if present), exit')
    sp.add_argument('--bench-attn-lens', default='4096,8192,16384,32768')
    sp.add_argument('--selftest', action='store_true', help='CPU equivalence test of every speed option (implies --tiny)')
    a = ap.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+', a.name):
        raise SystemExit('bad adapter name')
    if a.selftest:
        a.tiny = True
    if not a.grad_ckpt and a.ckpt == 'full':
        a.ckpt = 'none'
    ATTN.flex_compile = a.flex_compile
    ATTN.flex_kw = json.loads(a.flex_kernel_options) if a.flex_kernel_options else None

    import torch
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ATTN.flex_compile = ATTN.flex_compile and device.startswith('cuda')  # inductor has no CPU flex kernel here
    if a.tiny:
        a.model = str(out / 'tiny_model')
        shutil.rmtree(a.model, ignore_errors=True)
        build_tiny_checkpoint(a.tiny_config, a.model)
        log(f'tiny compressed-tensors checkpoint at {a.model}')
    if a.bench_attn:
        bench_attention(a, device)
        return
    if a.selftest:
        selftest(a, device)
        return

    val_tasks = {x for x in a.val_tasks.split(',') if x}
    train, val, max_len, dropped = load_rows(a.data, a.max_len, val_tasks, limit=16 if a.tiny else 0,
                                             crop=96 if a.tiny else 0)
    n_lab = sum(r['n_lab'] for r in train)
    log(f'data: {len(train)} train / {len(val)} val sequences, max_len {max_len} ({len(dropped)} longer dropped), '
        f'{sum(len(r["input_ids"]) for r in train)} tokens, {n_lab} labelled')

    model = load_model(a, device)
    tcfg = model.config.get_text_config()
    log(f'final_logit_softcapping {tcfg.final_logit_softcapping}, attention_k_eq_v {tcfg.attention_k_eq_v}, '
        f'attn {model.config._attn_implementation}')
    if a.tiny:  # our loss == HF's own (logits, softcap, shift) on one sequence
        inner, w, cap = text_parts(model)
        r = train[0]
        with torch.no_grad():
            s, k = seq_loss(inner, w, cap, r['input_ids'], r['labels'], 7, device)
            hf = model(input_ids=torch.tensor([r['input_ids']]), labels=torch.tensor([r['labels']])).loss.item()
        log(f'loss check: chunked CE {s.item() / k:.5f} vs HF forward {hf:.5f}')
        if abs(s.item() / k - hf) > 2e-2 * max(1.0, hf):
            raise SystemExit('chunked loss differs from the HF loss')

    pm, regex = add_lora(model, a)
    hf_model = pm.get_base_model()
    inner, w, cap = text_parts(hf_model)
    params = [p for p in pm.parameters() if p.requires_grad]
    rt = default_runtime(a)
    if a.bench:
        out_b = out / 'bench'
        out_b.mkdir(exist_ok=True)
        run_bench(a, pm, inner, w, cap, train, device, out_b)
        return
    if not runtime_is_default(rt):
        apply_runtime(pm, rt, a)
        check_attention_runtime(rt, a, device)
        log(f'speed options: {describe_runtime(rt)}')
    else:
        rt = None  # the unchanged rft1/rft2 path
    fused = True if a.fused_adam and device.startswith('cuda') else None
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=fused)
    steps_per_epoch = math.ceil(len(train) / a.grad_accum)
    total = steps_per_epoch * a.epochs if not a.max_steps else min(a.max_steps, steps_per_epoch * a.epochs)
    warm = max(1, round(a.warmup * total)) if a.warmup > 0 else 0

    def lr_at(step):  # step = 0-based optimizer step
        if step < warm:
            return a.lr * (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    meta = {'args': vars(a), 'max_len': max_len, 'train_sequences': len(train), 'val_sequences': len(val),
            'labelled_tokens': n_lab, 'optimizer_steps': total, 'warmup_steps': warm, 'target_regex': regex,
            'softcap': tcfg.final_logit_softcapping, 'device': device}
    logf = open(out / 'train_log.jsonl', 'w')
    if val and a.eval_before:
        pm.eval()
        meta['val_loss_before'] = evaluate(inner, w, cap, val, a.ce_chunk, device, rt)
        log(f'val loss before: {meta["val_loss_before"]:.4f}')
    pm.train()
    cuda = device.startswith('cuda')
    if a.preflight:  # fail fast on the longest sequence, and estimate the run time
        r = max(train, key=lambda r: len(r['input_ids']))
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        ATTN.docs = None
        with fwd_ctx(rt, device):
            s, k = seq_loss(inner, w, cap, r['input_ids'], r['labels'], a.ce_chunk, device)
        with bwd_ctx(rt):
            (s / k).backward()
        if cuda:
            torch.cuda.synchronize()
        dt = time.time() - t0
        opt.zero_grad(set_to_none=True)
        n_tok = sum(len(x['input_ids']) for x in train) * a.epochs
        meta['preflight'] = {'tokens': len(r['input_ids']), 'sec': round(dt, 1), 'loss': s.item() / k,
                             'est_train_minutes': round(n_tok / (len(r['input_ids']) / dt) / 60, 1)}
        if cuda:
            meta['preflight']['peak_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
            torch.cuda.empty_cache()
        log(f'preflight (longest sequence): {meta["preflight"]} (estimate is an upper bound: shorter sequences are '
            f'faster per token{"; includes compilation" if rt is not None and rt.compile else ""})')
    t_start, step, tokens_seen = time.time(), 0, 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    first_pass, peak_all = [], 0.0
    pack = rt.pack if rt is not None else 0
    for epoch in range(a.epochs):
        order = list(range(len(train)))
        random.Random(a.seed + epoch).shuffle(order)
        for g in range(0, len(order), a.grad_accum):
            if a.max_steps and step >= a.max_steps:
                break
            group = [train[i] for i in order[g:g + a.grad_accum]]
            denom = sum(r['n_lab'] for r in group)
            for pg in opt.param_groups:
                pg['lr'] = lr_at(step)
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t0, loss_sum, t_fwd, t_bwd = time.time(), 0.0, 0.0, 0.0
            bins = pack_bins(group, pack)
            for b in bins:
                ta = time.time()
                with fwd_ctx(rt, device):
                    s, k, per = bin_loss(inner, w, cap, b, a.ce_chunk, device)
                _sync(device)
                tb = time.time()
                with bwd_ctx(rt):
                    (s / denom).backward()
                loss_sum += s.item()  # waits for the backward
                t_fwd, t_bwd = t_fwd + tb - ta, t_bwd + time.time() - tb
                tokens_seen += sum(len(r['input_ids']) for r in b)
                if epoch == 0:
                    first_pass += [ps.item() / pk for ps, pk in per]
            tc = time.time()
            gn = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            _sync(device)
            step += 1
            sec = time.time() - t0
            n_tok = sum(len(r['input_ids']) for r in group)
            rec = {'step': step, 'epoch': epoch + 1, 'loss': loss_sum / denom, 'grad_norm': gn, 'lr': lr_at(step - 1),
                   'seqs': len(group), 'labelled': denom, 'sec': round(sec, 1), 'tok_per_s': round(n_tok / sec),
                   'tokens': n_tok, 'max_seq': max(len(r['input_ids']) for r in group), 'fwd_s': round(t_fwd, 2),
                   'bwd_s': round(t_bwd, 2), 'opt_s': round(time.time() - tc, 3)}
            if pack:
                rec['forwards'] = len(bins)
            if cuda:
                rec['peak_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)  # this step's peak
                rec['reserved_gb'] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 1)
                peak_all = max(peak_all, rec['peak_gb'])
            logf.write(json.dumps(rec) + '\n')
            logf.flush()
            log(' '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in rec.items()))
        if val:
            pm.eval()
            vl = evaluate(inner, w, cap, val, a.ce_chunk, device, rt)
            meta.setdefault('val_loss', []).append(vl)
            log(f'epoch {epoch + 1} val loss {vl:.4f}')
            pm.train()
    meta['train_seconds'] = round(time.time() - t_start)
    meta['tokens_seen'] = tokens_seen
    meta['train_tok_per_s'] = round(tokens_seen / max(meta['train_seconds'], 1))
    meta['first_epoch_mean_loss'] = sum(first_pass) / max(len(first_pass), 1)
    if cuda:
        meta['peak_gb'] = max(peak_all, round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))

    # ---- save + export
    pm.eval()
    raw = out / 'peft_raw'
    pm.save_pretrained(str(raw))
    adir = out / 'adapters' / a.name
    exported = export_adapter(raw, adir, a.rank, a.alpha, json.load(open(Path(a.model) / 'config.json'))['text_config']
                              if 'text_config' in json.load(open(Path(a.model) / 'config.json')) else {}, a.allow_kv)
    n_live, missing, worst = check_export(pm, exported)
    size = (adir / 'adapter_model.safetensors').stat().st_size
    meta.update(adapter_dir=str(adir), adapter_tensors=len(exported), adapter_mb=round(size / 1e6, 1),
                export_check={'live_tensors': n_live, 'missing': missing[:5], 'max_abs_diff_bf16': worst})
    log(f'exported {len(exported)} tensors ({size / 1e6:.1f} MB) to {adir}; live {n_live}, missing {len(missing)}, '
        f'max |diff| vs trained (bf16) {worst:.2e}')
    if missing or size > 3 * 2 ** 30:
        raise SystemExit('export check failed')

    if a.tiny:  # reload the exported adapter on a fresh base with PEFT, compare logits with the trained model
        from peft import PeftModel
        if rt is not None:  # compare like with like: the trained model back on the default (HF SDPA) path
            apply_runtime(pm, plain_runtime(), a)
        ids = torch.tensor([train[0]['input_ids']])
        with torch.no_grad():
            for n, p in pm.named_parameters():  # the exported weights are bf16: compare like with like
                if '.lora_' in n:
                    p.copy_(p.to(torch.bfloat16).to(p.dtype))
            ref = hf_model(input_ids=ids).logits.float()
        hf_dir = out / 'reload_hf'
        shutil.rmtree(hf_dir, ignore_errors=True)
        shutil.copytree(adir, hf_dir)
        c = json.load(open(hf_dir / 'adapter_config.json'))
        c['target_modules'] = r'.*language_model.*\.(' + '|'.join(c['target_modules']) + ')'
        json.dump(c, open(hf_dir / 'adapter_config.json', 'w'))
        fresh = load_model(a, device)
        with torch.no_grad():
            base = fresh(input_ids=ids).logits.float()
        re_pm = PeftModel.from_pretrained(fresh, str(hf_dir))
        re_pm.eval()
        with torch.no_grad():
            got = re_pm(input_ids=ids).logits.float()
        d_reload = (got - ref).abs().max().item()
        d_effect = (ref - base).abs().max().item()
        meta['reload_check'] = {'max_abs_logit_diff_reload_vs_trained': d_reload,
                                'max_abs_logit_effect_of_adapter': d_effect}
        log(f'reload check: |reloaded - trained| max {d_reload:.4f}; adapter effect |trained - base| max {d_effect:.4f}')
        if d_reload > 1e-3 * max(d_effect, 1.0):
            raise SystemExit('reloaded adapter does not reproduce the trained model')
    json.dump(meta, open(out / 'train_meta.json', 'w'), indent=1)
    log(f'done: {step} steps in {meta["train_seconds"]}s; meta in {out / "train_meta.json"}')


if __name__ == '__main__':
    sys.exit(main())
