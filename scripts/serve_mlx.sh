#!/usr/bin/env bash
# Serve Gemma 4 31B with mlx-lm (alternative to scripts/serve_local.sh).
# Note: mlx-community/gemma-4-31b-it-4bit is a plain 4-bit quant, not Google's QAT weights.
set -euo pipefail

MODEL="${MODEL:-$HOME/.lmstudio/models/mlx-community/gemma-4-31b-it-4bit}"
PORT="${PORT:-8002}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# mlx_lm.server loads whatever path the request's "model" field names, so expose the weights
# under the competition alias (a symlink in a scratch dir) and pass that same name as --model.
ALIAS=gemma-4-31b-it-qat-w4a16-ct
LINKDIR="$ROOT/.mlx-alias"
mkdir -p "$LINKDIR" && ln -sfn "$MODEL" "$LINKDIR/$ALIAS"
cd "$LINKDIR"

exec "$ROOT/.venv-mlx/bin/mlx_lm.server" \
  --model "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" \
  --max-tokens 16384 \
  --prompt-cache-size 4 \
  "$@"
