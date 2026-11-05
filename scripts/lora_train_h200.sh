#!/usr/bin/env bash
# Stage-1 LoRA training on a rented GPU (1x H200 141 GB), driven from this Mac in one command:
# upload data + scripts, install pinned deps, download the model, train (scripts/lora_train.py), export, fetch.
#
#   scripts/lora_train_h200.sh <instance_id> [name] [-- extra lora_train.py args]
#   e.g. scripts/lora_train_h200.sh 53500000 rft1 -- --val-tasks rich_3518,fastapi_14873 --eval-before
#   EXTRA_ARGS='--attn-sliding flex --lora-bf16' scripts/lora_train_h200.sh 53500000 rft3   # same, via env
#
# Speed options (lora_train.py module doc; all opt-in, defaults = the rft2 path):
#   BENCH=1 scripts/lora_train_h200.sh <id> [name]      # benchmark only, then exit (no training, no adapter):
#     1. attention microbench (lora_train.py --bench-attn, no model needed, ~2 min, runs while the model downloads):
#        ms / TFLOP/s / GB per kernel and length for sliding (head_dim 256, window 1024) and global (head_dim 512)
#        layers, accuracy vs fp32, and which kernel SDPA picks by itself;
#     2. model bench (lora_train.py --bench $BENCH_N): forward+backward tok/s, peak memory, loss and gradient error
#        vs the default path, per config in BENCH_CONFIGS (";"-separated, e.g. "base; attn_sliding=flex,lora_bf16=1")
#        on BENCH_LENS inputs (default "median,longest"; add e.g. 16384,32768 or group8).
#     Results: results/lora/<name>/bench/{bench_attn.json,bench.json,bench.log}. Then pick flags for EXTRA_ARGS.
#   BENCH_ATTN_ONLY=1 with BENCH=1 stops after step 1 (no model download).
#
# It does NOT rent anything: create the instance yourself (ask first; H200 ~$4.3/h), e.g.
#   vastai search offers 'num_gpus=1 gpu_name=H200 reliability>0.97 inet_down>500 disk_space>=150' -o dph_total
#   vastai create instance <offer> --image vllm/vllm-openai:v0.19.1 --disk 150 --ssh --direct --label gemma4-lora
# The vLLM image has CUDA torch and lets the same box serve the adapter afterwards (vast_up.sh flow,
# scripts/lora_hf_vs_vllm.py). Training needs ~95-100 GB at 25k tokens: an 80 GB H100 does NOT fit the longest
# sequences (try --max-len 12000 there).
#
# Re-running is safe: the model download and pip install are skipped when done; training restarts from scratch.
# The remote job runs under nohup, so a dropped ssh connection does not kill it; re-run this script to re-attach
# (it detects a running job and only waits + fetches).
# Results land in results/lora/<name>/ (adapters/<name>/, train_meta.json, train_log.jsonl, train.log).
# ALWAYS destroy the instance afterwards: vastai destroy instance <instance_id>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE=${1:?usage: lora_train_h200.sh <instance_id> [name] [-- extra args]}
shift
NAME=rft1
if [[ $# -gt 0 && "$1" != "--" ]]; then NAME=$1; shift; fi
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA="${EXTRA_ARGS:-}"                       # passed through verbatim (shell words, already quoted as needed)
if [[ $# -gt 0 ]]; then EXTRA="$EXTRA $(printf '%q ' "$@")"; fi
BENCH=${BENCH:-0}
[[ "$NAME" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo "bad adapter name $NAME"; exit 1; }
DATA="${DATA:-$ROOT/data/rft/train_tokens.jsonl}"   # DATA=... overrides (stage 2: data/rft2/train_tokens_all.jsonl)
OUT="$ROOT/results/lora/$NAME"
[[ -f "$DATA" ]] || { echo "missing $DATA: run scripts/rft_build.py first"; exit 1; }
MODEL=/models/gemma-4-31b-it-qat-w4a16-ct

URL=$(vastai ssh-url "$INSTANCE")            # ssh://root@HOST:PORT
HOST=${URL#ssh://}; PORT=${HOST##*:}; HOST=${HOST%:*}
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -p "$PORT" "$HOST")
SCP=(scp -q -o StrictHostKeyChecking=accept-new -P "$PORT")
until "${SSH[@]}" true 2>/dev/null; do echo "waiting for ssh..."; sleep 10; done

wait_job() {  # poll the nohup'd lora_train.py until it exits; $1 = remote log
  echo "== waiting (progress every minute)"
  while "${SSH[@]}" "pgrep -f '^python3 /root/lora/lora_train.py' >/dev/null"; do
    "${SSH[@]}" "grep -E '^[0-9]{2}:[0-9]{2}:[0-9]{2} ' $1 | tail -1" || true
    sleep 60
  done
}

if [[ "$BENCH" == 1 ]]; then
  if "${SSH[@]}" "pgrep -f '^python3 /root/lora/lora_train.py' >/dev/null"; then
    echo "a lora_train.py job is running on the instance: wait for it (or kill it) before benchmarking"; exit 1
  fi
  echo "== upload"
  "${SSH[@]}" "mkdir -p /root/lora /models"
  "${SCP[@]}" "$ROOT/scripts/lora_train.py" "$ROOT/scripts/lora_requirements.txt" "$HOST:/root/lora/"
  "${SCP[@]}" "$DATA" "$HOST:/root/lora/train_tokens.jsonl"
  echo "== install; model download in the background"
  "${SSH[@]}" "set -e; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    python3 -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"cudnn\", torch.backends.cudnn.version())'
    pip install -q -r /root/lora/lora_requirements.txt
    if [ '${BENCH_ATTN_ONLY:-0}' != 1 ] && [ ! -f /root/lora/download.ok ] && [ ! -f /root/lora/download.running ]; then
      touch /root/lora/download.running; rm -f /root/lora/download.rc
      nohup sh -c 'HF_HUB_ENABLE_HF_TRANSFER=1 hf download google/gemma-4-31B-it-qat-w4a16-ct --local-dir $MODEL \
        >/root/lora/download.log 2>&1; rc=\$?; echo \$rc > /root/lora/download.rc; [ \$rc = 0 ] && touch /root/lora/download.ok; \
        rm -f /root/lora/download.running' >/dev/null 2>&1 &
    fi"
  echo "== attention microbench"
  "${SSH[@]}" "cd /root/lora && mkdir -p bench && python3 /root/lora/lora_train.py --bench-attn --model $MODEL \
    --out /root/lora/bench ${BENCH_ATTN_LENS:+--bench-attn-lens $BENCH_ATTN_LENS} 2>&1 | tee /root/lora/bench/bench_attn.log \
    | grep -v -E '^[0-9]{2}:[0-9]{2}:[0-9]{2} attn '"
  mkdir -p "$OUT/bench"
  "${SCP[@]}" "$HOST:/root/lora/bench/bench_attn.json" "$HOST:/root/lora/bench/bench_attn.log" "$OUT/bench/" || true
  if [[ "${BENCH_ATTN_ONLY:-0}" == 1 ]]; then
    echo "attention bench in $OUT/bench. REMEMBER: vastai destroy instance $INSTANCE"; exit 0
  fi
  until "${SSH[@]}" "test -f /root/lora/download.ok"; do
    if "${SSH[@]}" "test ! -f /root/lora/download.running"; then
      echo "model download failed:"; "${SSH[@]}" "tail -5 /root/lora/download.log"; exit 1
    fi
    echo "waiting for the model download... $("${SSH[@]}" "du -sh $MODEL 2>/dev/null | cut -f1")"; sleep 30
  done
  BC=$(printf '%q' "${BENCH_CONFIGS:-}")
  echo "== model bench (nohup; log /root/lora/bench.log)"
  "${SSH[@]}" "cd /root/lora && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup python3 /root/lora/lora_train.py --model $MODEL --data /root/lora/train_tokens.jsonl --out /root/lora/bout \
      --name $NAME --bench ${BENCH_N:-2} ${BENCH_CONFIGS:+--bench-configs $BC} \
      ${BENCH_LENS:+--bench-lens $BENCH_LENS} $EXTRA > /root/lora/bench.log 2>&1 &"
  wait_job /root/lora/bench.log
  "${SCP[@]}" "$HOST:/root/lora/bench.log" "$OUT/bench/" || true
  "${SCP[@]}" "$HOST:/root/lora/bout/bench/bench.json" "$OUT/bench/" || { tail -20 "$OUT/bench/bench.log"; exit 1; }
  sed -n '/^config /,$p' "$OUT/bench/bench.log"
  echo "bench results: $OUT/bench/ (bench_attn.json, bench.json, logs)"
  echo "REMEMBER: vastai destroy instance $INSTANCE   (or train now: EXTRA_ARGS='<flags>' $0 $INSTANCE $NAME)"
  exit 0
fi

if ! "${SSH[@]}" "pgrep -f '^python3 /root/lora/lora_train.py' >/dev/null"; then
  echo "== upload"
  "${SSH[@]}" "mkdir -p /root/lora /models"
  "${SCP[@]}" "$ROOT/scripts/lora_train.py" "$ROOT/scripts/lora_requirements.txt" "$HOST:/root/lora/"
  "${SCP[@]}" "$DATA" "$HOST:/root/lora/train_tokens.jsonl"

  echo "== install (torch from the image) + model download (23 GB)"
  "${SSH[@]}" 'set -e; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    python3 -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)"
    pip install -q -r /root/lora/lora_requirements.txt
    python3 -c "import transformers, peft, compressed_tensors; print(\"transformers\", transformers.__version__, \"peft\", peft.__version__, \"compressed-tensors\", compressed_tensors.__version__)"
    while [ -f /root/lora/download.running ]; do sleep 10; done  # a BENCH=1 background download: let it finish
    [ -f /root/lora/download.ok ] || { HF_HUB_ENABLE_HF_TRANSFER=1 hf download google/gemma-4-31B-it-qat-w4a16-ct \
      --local-dir /models/gemma-4-31b-it-qat-w4a16-ct >/dev/null && touch /root/lora/download.ok; }
    du -sh /models/gemma-4-31b-it-qat-w4a16-ct'

  echo "== train (nohup; log /root/lora/train.log)${EXTRA:+ extra args:$EXTRA}"
  "${SSH[@]}" "cd /root/lora && rm -rf out && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup python3 /root/lora/lora_train.py --model $MODEL \
      --data /root/lora/train_tokens.jsonl --out /root/lora/out --name $NAME $EXTRA > /root/lora/train.log 2>&1 &"
else
  echo "== training already running: attaching"
fi

wait_job /root/lora/train.log
"${SSH[@]}" "tail -5 /root/lora/train.log"
"${SSH[@]}" "test -f /root/lora/out/train_meta.json" || { echo "training failed: see the log above"; exit 1; }

echo "== fetch -> $OUT"
mkdir -p "$OUT"
"${SCP[@]}" -r "$HOST:/root/lora/out/adapters" "$HOST:/root/lora/out/train_meta.json" \
  "$HOST:/root/lora/out/train_log.jsonl" "$HOST:/root/lora/train.log" "$OUT/"
python3 - "$OUT" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1] + '/train_meta.json'))
keys = ('train_sequences', 'labelled_tokens', 'optimizer_steps', 'preflight', 'first_epoch_mean_loss', 'val_loss_before',
        'val_loss', 'train_seconds', 'train_tok_per_s', 'peak_gb', 'adapter_mb', 'adapter_tensors', 'export_check')
for k in keys:
    if k in m:
        print(f'{k:22s} {m[k]}')
log = [json.loads(l) for l in open(sys.argv[1] + '/train_log.jsonl')]
print('loss by step:', ' '.join(f"{r['loss']:.3f}" for r in log))
if log and 'fwd_s' in log[0]:
    t = {k: sum(r[k] for r in log) for k in ('fwd_s', 'bwd_s', 'opt_s', 'sec', 'tokens')}
    print(f"time split: fwd {t['fwd_s']:.0f}s bwd {t['bwd_s']:.0f}s opt {t['opt_s']:.0f}s of {t['sec']:.0f}s; "
          f"{t['tokens'] / max(t['sec'], 1):.0f} tok/s in steps")
EOF
echo "adapter: $OUT/adapters/$NAME  (copy into the submission as adapters/$NAME and add 'adapter: $NAME' to agent.yaml)"
echo "REMEMBER: vastai destroy instance $INSTANCE   (or keep it for the vLLM consistency check: lora_hf_vs_vllm.py)"
