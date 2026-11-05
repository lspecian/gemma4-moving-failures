#!/usr/bin/env bash
# Run the skill unit tests inside the sandbox image (x86, Python 3.13, offline), like on Kaggle.
#   scripts/test_skills.sh [pytest args]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec docker run --rm --platform linux/amd64 --network none \
  -v "$ROOT:/repo:ro" -w /tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  swebench-sandbox:latest \
  python -m pytest /repo/tests -q -p no:cacheprovider "$@"
