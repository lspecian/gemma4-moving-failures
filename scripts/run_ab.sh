#!/usr/bin/env bash
# A/B two agent versions on a task list, alternating per task so machine-speed drift hits both equally.
#
#   scripts/run_ab.sh <versionA> <runA> <versionB> <runB> [task_list] [repeats] [minutes] [tool_calls]
#
# Versions are anything scripts/materialize.sh accepts (agent-v5, WORKTREE, baselines/starter, ...).
# Each run dir holds the exact agent it evaluated (_agent/ + _agent.version) and one sub-dir per repeat:
#   results/<run>/r<k>/<task>/...
# TASK_ENV=1 gives every sandbox its task's dependency versions (scripts/task_env.py; gold passes 115/129
# instead of 52). Default: the stock published environment, as in all runs before 2026-10-03.
set -uo pipefail
A=$1; RA=$2; B=$3; RB=$4; LIST=${5:-experiments/devset_12.txt}; REPS=${6:-1}; MIN=${7:-20}; CALLS=${8:-40}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:8001/v1} OPENAI_API_KEY=local
curl -sf "${OPENAI_BASE_URL%/v1}/health" >/dev/null || { echo "model server not reachable at $OPENAI_BASE_URL"; exit 1; }

scripts/materialize.sh "$A" "results/$RA/_agent"
scripts/materialize.sh "$B" "results/$RB/_agent"

SWEGEMMA=.venv/bin/swegemma
[ "${TASK_ENV:-0}" = 1 ] && SWEGEMMA=".venv/bin/python scripts/swegemma_local.py"
run() {  # <run_name> <repeat> <task>
  $SWEGEMMA eval --tasks data/competition/tasks.jsonl --snapshots-dir data/competition/snapshots \
    --submission-dir "results/$1/_agent" --results-dir "results/$1/r$2/$3" --sandbox docker --task-id "$3" \
    --max-time-minutes "$MIN" --max-tool-calls "$CALLS" --display quiet >> "logs/$1.log" 2>&1
}
for r in $(seq 1 "$REPS"); do
  for t in $(cat "$LIST"); do
    run "$RA" "$r" "$t"
    run "$RB" "$r" "$t"
    echo "$(date +%H:%M) rep $r done $t"
  done
done
.venv/bin/python scripts/summarize.py "results/$RA" "results/$RB"
