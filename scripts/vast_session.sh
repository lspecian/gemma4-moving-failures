#!/usr/bin/env bash
# One vast.ai eval session end to end: rent -> serve (vast_up.sh) -> arms (nebius_session.sh ATTACH=1) -> destroy.
# The instance is destroyed on every exit path (trap), setup has a wall-clock deadline, and the Mac is kept awake
# (2026-11-04: a hung setup ssh + Mac sleep left an idle H100 billing 6 h).
#
#   scripts/vast_session.sh '<offer query>' <max $/h> <setup_deadline_min> <session args for nebius_session.sh ...>
#   e.g. scripts/vast_session.sh 'gpu_name in [H100_SXM] num_gpus=1 reliability>0.97 inet_down>200 disk_space>120 rentable=true' \
#          2.5 35 h100 100 experiments/eval113_every3.txt 8 28 240 80 3 E112_v37 E113_v38b
# Env: VENV (passed to the session). Add the instance to experiments/spend_map.tsv (printed below) and run spend.py after.
[ -z "${CAFFEINATED:-}" ] && exec env CAFFEINATED=1 caffeinate -i "$0" "$@"
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
QUERY=$1; MAXP=$2; SETUP_MIN=$3; shift 3
LOG=logs/vast_session_$(date -u +%m%d_%H%M).log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }

INST=$(scripts/vast_rent.sh "$QUERY" "$MAXP" | tail -1 | awk '{print $1}')
[[ "$INST" =~ ^[0-9]+$ ]] || { say "no instance rented"; exit 1; }
say "rented instance $INST: $(vastai show instance "$INST" --raw | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('gpu_name'), d.get('dph_total'), d.get('geolocation'))")"
UP_PID=""
cleanup() {
  [ -n "$UP_PID" ] && kill "$UP_PID" 2>/dev/null
  scripts/vast_down.sh "$INST" 8002 2>&1 | tail -1 | tee -a "$LOG"
  say "session end"
}
trap cleanup EXIT
trap 'exit 1' INT TERM HUP

scripts/vast_up.sh --attach "$INST" > "logs/vast_up_$INST.log" 2>&1 &
UP_PID=$!
DEADLINE=$(( $(date +%s) + SETUP_MIN * 60 ))
until curl -s -m 10 127.0.0.1:8002/v1/models | grep -q gemma; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then say "SETUP DEADLINE ($SETUP_MIN min)"; tail -5 "logs/vast_up_$INST.log" | tee -a "$LOG"; exit 1; fi
  sleep 15
done
say "server ready; $(grep -E 'GPU KV cache size' "logs/vast_up_$INST.log" | tail -1 | sed 's/.*\] //')"
ATTACH=1 scripts/nebius_session.sh "$@" 2>&1 | tee -a "$LOG"
