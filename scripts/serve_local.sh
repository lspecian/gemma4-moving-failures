#!/usr/bin/env bash
# Serve Gemma 4 31B QAT locally as a stand-in for Kaggle's vLLM server.
#
# The harness (swegemma) talks to $OPENAI_BASE_URL (default http://localhost:8000/v1) and requests the model
# by the alias declared in agent.yaml, so we expose the GGUF under the competition name.
# Mirrors the Kaggle setup where it matters: 32k context, thinking enabled, tool calls parsed.
# Differences: llama.cpp instead of vLLM, Q4_0 GGUF instead of W4A16 compressed-tensors,
# and no LoRA adapters.
set -euo pipefail

MODEL="${MODEL:-$HOME/.lmstudio/models/lmstudio-community/gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-QAT-Q4_0.gguf}"
PORT="${PORT:-8001}"  # 8000 is taken by an OrbStack container on this machine
ALIAS="${ALIAS:-gemma-4-31b-it-qat-w4a16-ct}"  # E4B: MODEL=data/models/gemma-4-E4B-it-qat-gguf/gemma-4-E4B_q4_0-it.gguf ALIAS=gemma-4-e4b-it-qat-w4a16-ct
NP="${NP:-1}"  # parallel slots, 32k context each

exec llama-server \
  -m "$MODEL" \
  --alias "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" \
  -c $((32768 * NP)) \
  -np "$NP" \
  -ngl 999 \
  -fa on \
  --jinja \
  --reasoning on \
  --cache-reuse 256 \
  "$@"
