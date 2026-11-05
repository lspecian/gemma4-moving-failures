#!/usr/bin/env bash
# One Nebius eval session: bring up a GPU VM (scripts/nebius_up.sh), run arms through pacing proxies, tear down.
# Hard deadline: the VM is deleted after DEADLINE_MIN minutes no matter what (the VM also self-stops after max_hours).
#
#   scripts/nebius_session.sh <gpu h200|h100> <deadline_min> <list> <minutes> <calls> <timeout_s> <turns> <conc> <run> [<run> ...]
#   e.g. scripts/nebius_session.sh h200 100 experiments/nebius_smoke16.txt 8 28 240 80 8 E074_neb_v24_smoke
# Each <run> needs results/<run>/_agent. Proxies: --decode 35 --prefill ${PREFILL:-1400}; ports 8020+.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
GPU=$1; DEADLINE_MIN=$2; LIST=$3; MIN=$4; CALLS=$5; TOUT=$6; TURNS=$7; CONC=$8; shift 8; RUNS=("$@")
MAX_H=$(( (DEADLINE_MIN + 59) / 60 + 1 ))
START=$(date +%s); DEADLINE=$(( START + DEADLINE_MIN * 60 ))
LOG=logs/nebius_session_${RUNS[0]}.log
echo "$(date -u +%H:%M) session start: $GPU, deadline ${DEADLINE_MIN} min, runs ${RUNS[*]}" | tee -a "$LOG"

# ATTACH=1: a server is already on :8002 (e.g. scripts/vast_up.sh); run the arms only, no VM to create or delete
if [ -n "${ATTACH:-}" ]; then sleep 1 & else nohup scripts/nebius_up.sh "$GPU" "$MAX_H" > logs/nebius_up_${RUNS[0]}.log 2>&1 & fi
UP_PID=$!
NAME=""
[ -n "${ATTACH:-}" ] || for i in $(seq 1 720); do
  NAME=$(grep -oE "gemma4-$GPU-[0-9]{4}-[0-9]{4}" logs/nebius_up_${RUNS[0]}.log | head -1)
  grep -q "== tunnel" logs/nebius_up_${RUNS[0]}.log && break
  kill -0 $UP_PID 2>/dev/null || break
  [ "$(date +%s)" -ge $(( START + 45 * 60 )) ] && break      # setup deadline 45 min
  sleep 5
done
teardown() {
  ps -axo pid,command | grep -v grep | grep -E "slow_proxy.py --port 180[0-9][0-9]" | awk '{print $1}' | xargs -r kill 2>/dev/null
  ps -axo pid,command | grep -v grep | grep -E "replay_server.py .*--port 181[2-9][0-9]" | awk '{print $1}' | xargs -r kill 2>/dev/null
  for R in "${RUNS[@]}"; do pkill -f "run_chunked.sh .*${R}" 2>/dev/null; done
  [ -n "$NAME" ] && scripts/nebius_down.sh "$NAME" 8002 | tee -a "$LOG"
  kill $UP_PID 2>/dev/null
  echo "$(date -u +%H:%M) session end ($(( ($(date +%s) - START) / 60 )) min)" | tee -a "$LOG"
}
ok=""; for i in $(seq 1 30); do   # vLLM's API can start after the KV-cache log line and the tunnel may reconnect (10-29)
  curl -s -m 20 127.0.0.1:8002/v1/models | grep -q gemma && { ok=1; break; }; sleep 10; done
if [ -z "$ok" ]; then
  echo "$(date -u +%H:%M) SETUP FAILED" | tee -a "$LOG"; tail -15 logs/nebius_up_${RUNS[0]}.log | sed -E 's/[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/<ip>/g' | tee -a "$LOG"
  teardown; exit 1
fi
echo "$(date -u +%H:%M) SERVER OK on $NAME ($(( ($(date +%s) - START) / 60 )) min setup)" | tee -a "$LOG"
grep -E "GPU KV cache|Maximum concurrency|docker GPU OK|H200|H100" logs/nebius_up_${RUNS[0]}.log | tee -a "$LOG"

PORT=18020   # high ports: 8021 was taken twice right after the first arm started (E077, E079; cause unknown)
for R in "${RUNS[@]}"; do
  OK=""
  for try in 1 2 3 4 5; do
    while lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; do PORT=$((PORT + 1)); done
    nohup python3 scripts/slow_proxy.py --port $PORT --upstream http://127.0.0.1:8002 --decode 35 --prefill "${PREFILL:-1400}" \
      --log logs/${R}_proxy.jsonl --save data/raw/$R --logprobs 3 > logs/${R}_proxy.out 2>&1 &
    for i in $(seq 1 15); do curl -s -m 5 127.0.0.1:$PORT/v1/models | grep -q gemma && { OK=1; break; }; sleep 1; done
    [ -n "$OK" ] && break
    echo "$(date -u +%H:%M) proxy for $R did not answer on :$PORT, trying the next port" | tee -a "$LOG"
    PORT=$((PORT + 1))
  done
  [ -z "$OK" ] && { echo "$(date -u +%H:%M) NO PROXY for $R: arm not started" | tee -a "$LOG"; continue; }
  # per-run task list and environment (results/<run>/_list: a list file; results/<run>/_env: VAR=value lines, e.g.
  # TASKS_FILE / SNAPSHOTS_DIR / TASK_ENV_MANIFEST for unseen100), so one session can run arms on different task sets
  RLIST="$LIST"; [ -f "results/$R/_list" ] && RLIST=$(head -1 "results/$R/_list")
  # optional front server between the harness and the paced proxy (results/<run>/_front: one command line with
  # {UPSTREAM} and {PORT}, e.g. scripts/replay_server.py in handoff mode); ports 18120+
  BASE=http://127.0.0.1:$PORT/v1
  if [ -f "results/$R/_front" ]; then
    FPORT=${FPORT:-18120}
    while lsof -nP -iTCP:$FPORT -sTCP:LISTEN >/dev/null 2>&1; do FPORT=$((FPORT + 1)); done
    FCMD=$(head -1 "results/$R/_front" | sed -e "s#{UPSTREAM}#http://127.0.0.1:$PORT#g" -e "s#{PORT}#$FPORT#g")
    nohup python3 $FCMD > logs/${R}_front.out 2>&1 &
    for i in $(seq 1 30); do curl -s -m 5 127.0.0.1:$FPORT/v1/models | grep -q . && break; sleep 2; done
    echo "$(date -u +%H:%M) front for $R on :$FPORT -> :$PORT" | tee -a "$LOG"
    BASE=http://127.0.0.1:$FPORT/v1; FPORT=$((FPORT + 1))
  fi
  ( if [ -f "results/$R/_env" ]; then set -a; . "results/$R/_env"; set +a; fi
    OPENAI_BASE_URL=$BASE CONCURRENCY=$CONC TASK_ENV=1 SWEGEMMA_KAGGLE_APP=1 TIMEOUT_S=$TOUT MAX_TURNS=$TURNS \
      exec nohup scripts/run_chunked.sh "$RLIST" 1 100 "$MIN" "$CALLS" "$R" > logs/${R}_driver.log 2>&1 ) &
  PORT=$((PORT + 1))
done
sleep 20
RUNPAT=$(IFS='|'; echo "${RUNS[*]}")
while pgrep -f "run_chunked.sh .*(${RUNPAT})" >/dev/null; do   # watch ALL arms (BSD pgrep: extended regex, no -E flag; E078 died of -E)
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "$(date -u +%H:%M) DEADLINE reached: stopping runs" | tee -a "$LOG"; break; fi
  sleep 30
done
for R in "${RUNS[@]}"; do tail -1 logs/${R}_driver.log | tee -a "$LOG"; done
teardown
