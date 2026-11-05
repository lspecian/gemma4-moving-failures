#!/usr/bin/env bash
# Watch a running eval: every 60 s check that the model server answers through the tunnel and that Docker responds.
# After LIMIT seconds of either failing, stop the runs (run_chunked.sh + swegemma) and exit 1 with the reason.
# E051/E052 lost 3.5 h to a dead tunnel and a hung Docker daemon that the per-chunk health check noticed too late.
#
#   scripts/run_watchdog.sh <run-name-pattern> [server_url] [limit_s]      e.g.  scripts/run_watchdog.sh 'E05[34]'
set -u
PAT=$1; SERVER=${2:-http://127.0.0.1:8002}; LIMIT=${3:-300}
with_timeout() { perl -e 'alarm shift; exec @ARGV' "$@"; }   # macOS has no `timeout`
down_server=0; down_docker=0
running() { ps -axo command | grep -v grep | grep -q "run_chunked.sh .*${PAT}"; }
while running; do
  if with_timeout 20 curl -sf -m 15 "$SERVER/v1/models" >/dev/null 2>&1; then down_server=0; else down_server=$((down_server + 60)); fi
  if with_timeout 20 docker ps -q >/dev/null 2>&1; then down_docker=0; else down_docker=$((down_docker + 60)); fi
  if [ $down_server -ge "$LIMIT" ] || [ $down_docker -ge "$LIMIT" ]; then
    echo "$(date +%H:%M) STOPPING runs matching $PAT: server down ${down_server}s, docker down ${down_docker}s"
    pkill -f "run_chunked.sh .*${PAT}"; pkill -f "swegemma_local.py eval.*${PAT}"
    exit 1
  fi
  sleep 60
done
echo "$(date +%H:%M) runs matching $PAT finished; server and docker stayed up"
