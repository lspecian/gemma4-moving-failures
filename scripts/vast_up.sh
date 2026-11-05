#!/usr/bin/env bash
# Bring up Kaggle's serving stack on a rented vast.ai GPU and tunnel it to this Mac.
# Server flags mirror adk_submission 0.2.12 VllmServer (reasoning-config enables thinking_token_budget).
#
#   scripts/vast_up.sh <offer_id>          rent, install, serve, then open the tunnel (blocks; Ctrl-C closes tunnel)
#   scripts/vast_up.sh --attach <instance> skip renting; (re)install/serve on an existing instance and tunnel
#
# Afterwards point the harness at it: OPENAI_BASE_URL=http://127.0.0.1:8002/v1
# Destroy when done (billing continues otherwise):  vastai destroy instance <instance>
#
# What it mirrors from the Kaggle scorer (HARNESS_README 3.1): the w4a16-ct model, Kaggle's patched vLLM 0.19.1 and
# its wheelhouse deps, 32k context, gemma4 tool + reasoning parsers, thinking on by default.
# Differences: one GPU (TP=1) instead of 4x L4; LoRA only with LORA=name=dir (gpu util 0.88: its buffers cost KV).
# Find offers: vastai search offers 'num_gpus=1 gpu_name=L40S reliability>0.97 inet_down>500 disk_space>=120' -o dph_total
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_PORT=${LOCAL_PORT:-8002}

if [[ "$1" == --attach ]]; then
  INSTANCE=$2
else
  INSTANCE=$(vastai create instance "$1" --image vllm/vllm-openai:v0.19.1 --disk 120 --ssh --direct \
    --label gemma4-vllm-eval --raw | python3 -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])')
  echo "instance $INSTANCE created; waiting for it to run..."
  until vastai show instance "$INSTANCE" --raw | python3 -c 'import json,sys; sys.exit(json.load(sys.stdin).get("actual_status")!="running")'; do sleep 20; done
fi
URL=$(vastai ssh-url "$INSTANCE")            # ssh://root@HOST:PORT
HOST=${URL#ssh://}; PORT=${HOST##*:}; HOST=${HOST%:*}
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o ConnectTimeout=20 -p "$PORT" "$HOST")
until "${SSH[@]}" true 2>/dev/null; do sleep 10; done

echo "== model download (23 GB) + wheel upload"
# The download runs detached on the instance and is polled with short ssh calls, and the wheels go up in one tar
# stream: on 11-04 the vast ssh proxy closed the download's long ssh session ("closed by remote host") twice, while
# 32 scp connections (one per wheel) were opening, and the setup hung.
"${SSH[@]}" 'pip install -q "huggingface-hub[hf_transfer]>=1.5,<2"; mkdir -p /models /wheels;
  [ -f /models/DL_DONE ] || [ -f /root/dl.started ] || (touch /root/dl.started; setsid nohup sh -c "HF_HUB_ENABLE_HF_TRANSFER=1 \
  hf download google/gemma-4-31B-it-qat-w4a16-ct --local-dir /models/gemma-4-31b-it-qat-w4a16-ct && touch /models/DL_DONE" \
  > /root/dl.log 2>&1 < /dev/null & disown)'
(cd "$ROOT"/data/wheelhouse && ls *.whl | grep -vE '^(adk_|swegemma|google_adk|google_genai|anthropic|loguru|opentelemetry)' \
  | tar cf - -T -) | "${SSH[@]}" 'tar xf - -C /wheels'
until "${SSH[@]}" 'test -f /models/DL_DONE' 2>/dev/null; do sleep 20; done
# Kaggle's model files are the 2026-06-05 release; HF replaced chat_template.jinja (and tokenizer_config.json) on
# 2026-07-15. Serve Kaggle's versions (data/probe/*_kaggle_june.*, HF commit ca9d5250) so prompts render as on the scorer.
scp -q -P "$PORT" "$ROOT/data/probe/chat_template_kaggle_june.jinja" \
  "$HOST":/models/gemma-4-31b-it-qat-w4a16-ct/chat_template.jinja
scp -q -P "$PORT" "$ROOT/data/probe/tokenizer_config_kaggle_june.json" \
  "$HOST":/models/gemma-4-31b-it-qat-w4a16-ct/tokenizer_config.json

echo "== install Kaggle's vLLM + deps over the image"
"${SSH[@]}" 'pip install -q --no-deps --force-reinstall /wheels/*.whl && pip install -q "huggingface-hub>=1.5,<2" \
  && python3 -c "import vllm.entrypoints.openai.api_server; print(\"vllm OK\")"'

# No --default-chat-template-kwargs on purpose: the scorer VllmServer does not set it (L4 probe, 2026-10-05). The agent
# requests carry enable_thinking themselves; ADK's compaction summary call does not, so on Kaggle it runs without thinking.
# LORA=name=/local/adapter/dir serves one adapter like the scorer's VllmServer does for a submission with adapters/
# (--enable-lora --max-loras 1 --max-lora-rank <bucket of r> --lora-modules name=dir; gpu util 0.88: LoRA buffers).
# Several adapters (e.g. to compare them on one server): LORA="rft2=dir2,rft3=dir3" (--max-loras = their count).
if [[ -n "${LORA:-}" ]]; then
  MODS=""; RANK=8; NL=0
  IFS=',' read -r -a LSPECS <<< "$LORA"
  for spec in "${LSPECS[@]}"; do
    LNAME=${spec%%=*}; LDIR=${spec#*=}
    R=$(python3 -c "import json,sys; r=json.load(open(sys.argv[1]))['r']; print(next(b for b in (8,16,32,64,128,256) if b>=r))" "$LDIR/adapter_config.json")
    [ "$R" -gt "$RANK" ] && RANK=$R
    "${SSH[@]}" "mkdir -p /adapters/$LNAME"
    scp -q -P "$PORT" "$LDIR"/adapter_config.json "$LDIR"/adapter_model.safetensors "$HOST:/adapters/$LNAME/"
    MODS="$MODS $LNAME=/adapters/$LNAME"; NL=$((NL + 1))
    echo "== adapter $LNAME (rank bucket $R) uploaded"
  done
  "${SSH[@]}" "echo '--enable-lora --max-loras $NL --max-lora-rank $RANK --lora-modules$MODS --gpu-memory-utilization 0.88' > /root/serve_extra"
else
  "${SSH[@]}" "rm -f /root/serve_extra"
fi
echo "== serve"
"${SSH[@]}" 'cat > /root/serve.sh <<'"'"'EOF'"'"'
#!/usr/bin/env bash
exec vllm serve /models/gemma-4-31b-it-qat-w4a16-ct \
  --served-model-name gemma-4-31b-it-qat-w4a16-ct \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --reasoning-config "{\"reasoning_start_str\": \"<|channel>\", \"reasoning_end_str\": \"<channel|>\"}" \
  --no-scheduler-reserve-full-isl $(cat /root/serve_extra 2>/dev/null)
EOF
chmod +x /root/serve.sh; pgrep -f "^vllm serve" >/dev/null || (setsid nohup /root/serve.sh > /root/vllm.log 2>&1 < /dev/null & disown)'
# poll with short ssh calls (a long one can be closed by the proxy, see the download above)
until "${SSH[@]}" 'curl -sf 127.0.0.1:8000/v1/models >/dev/null' 2>/dev/null; do
  sleep 15
  "${SSH[@]}" 'grep -qE "ValueError|RuntimeError" /root/vllm.log && tail -5 /root/vllm.log' 2>/dev/null && exit 1
done
"${SSH[@]}" 'grep -E "GPU KV cache size|Maximum concurrency" /root/vllm.log | tail -2'

echo "== tunnel: 127.0.0.1:$LOCAL_PORT -> vLLM on instance $INSTANCE (Ctrl-C to close; destroy the instance when done)"
# Reconnect forever (E051/E052 lost 3.5 h when a one-shot tunnel died): ssh exits when the link drops
# (ServerAliveCountMax), the loop reopens it. Ctrl-C / kill of this script ends it.
trap 'exit 0' INT TERM
while true; do
  ssh -N -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -o ConnectTimeout=20 \
    -L "$LOCAL_PORT:127.0.0.1:8000" -p "$PORT" "$HOST"
  echo "$(date +%H:%M:%S) tunnel dropped (exit $?); reconnecting in 5 s" >&2
  sleep 5
done
