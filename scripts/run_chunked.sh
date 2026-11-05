#!/usr/bin/env bash
# Compare agent versions on a long task list at concurrency N, interleaved in chunks so machine/server drift hits
# every arm equally (run_ab.sh alternates per task but runs one task at a time).
#
#   scripts/run_chunked.sh <list> <rounds> <chunk_size> <minutes> <tool_calls> <run1> [<run2> ...]
#
# Each <run> must already hold its agent in results/<run>/_agent (scripts/materialize.sh). Results go to
# results/<run>/r<k>/c<j>/ (one swegemma results dir per chunk). Per-task environments by default (TASK_ENV=1,
# scripts/swegemma_local.py); TASK_ENV=0 for the stock environment. Stops if the model server stops answering.
# TIMEOUT_S / MAX_TURNS: the eval_config timeout_seconds / max_turns (swegemma's local default timeout is 300 s, and
# it also bounds the hidden-test run in grading; Kaggle reads eval_config: v20b 180 s / 100 turns).
# Rerunning the same command resumes: finished chunks are skipped, a partial chunk is rerun.
# Env: OPENAI_BASE_URL (default the vast tunnel :8002), CONCURRENCY (default 4).
# Other task sets (unseen100, research/notes/21-unseen100.md): TASKS_FILE (default data/competition/tasks.jsonl),
# SNAPSHOTS_DIR (default data/competition/snapshots) and TASK_ENV_MANIFEST (read by scripts/task_env.py; default the
# public tasks' manifest data/competition/wheels_local/task_env.json).
set -uo pipefail
LIST=$1; ROUNDS=$2; CHUNK=$3; MIN=$4; CALLS=${CALLS_OVERRIDE:-$5}; shift 5; RUNS=("$@")   # CALLS_OVERRIDE: per-arm budget via results/<run>/_env (local runs ignore eval_config)
MAX_TURNS=${TURNS_OVERRIDE:-${MAX_TURNS:-}}   # TURNS_OVERRIDE: per-arm turn budget via results/<run>/_env
TASKS_FILE=${TASKS_FILE:-data/competition/tasks.jsonl}
SNAPSHOTS_DIR=${SNAPSHOTS_DIR:-data/competition/snapshots}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:8002/v1} OPENAI_API_KEY=local
CONC=${CONCURRENCY:-4}
VENV=${VENV:-.venv}   # VENV=.venv-0211: Kaggle's current harness (swegemma 0.2.11 / adk_submission 0.2.13, since 10-08)
SWEGEMMA=("$VENV/bin/python" scripts/swegemma_local.py)
[ "${TASK_ENV:-1}" = 0 ] && SWEGEMMA=("$VENV/bin/swegemma")

ok() {
  curl -s -m 120 "$OPENAI_BASE_URL/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"gemma-4-31b-it-qat-w4a16-ct","messages":[{"role":"user","content":"say hi"}],"max_tokens":16}' \
    | grep -q '"choices"'
}
cnt() {  # passes / tasks over every chunk of one round
  python3 - "$1" <<'EOF'
import glob, json, sys
L = [json.loads(l) for f in glob.glob(sys.argv[1] + '/c*/task_results.jsonl') for l in open(f)]
print(sum(1 for d in L if d.get('passed') or d.get('resolved')), '/', len(L))
EOF
}

TASKS=($(grep -v -E '^[[:space:]]*(#|$)' "$LIST"))   # skip blank and comment lines
NCH=$(( (${#TASKS[@]} + CHUNK - 1) / CHUNK ))
for r in $(seq 1 "$ROUNDS"); do
  for j in $(seq 0 $((NCH - 1))); do
    part=("${TASKS[@]:$((j * CHUNK)):$CHUNK}")
    # alternate which arm goes first in each chunk
    order=("${RUNS[@]}"); (( (j + r) % 2 )) && order=($(printf '%s\n' "${RUNS[@]}" | tail -r))
    for E in "${order[@]}"; do
      done_f="results/$E/r$r/c$j/task_results.jsonl"
      # resume: skip a chunk whose every task has a result; drop a partial one and rerun it
      if [ -f "$done_f" ] && [ "$(wc -l < "$done_f")" -ge "${#part[@]}" ]; then continue; fi
      rm -rf "results/$E/r$r/c$j"
      ok || { echo "$(date +%H:%M) SERVER NOT RESPONDING before $E round $r chunk $j"; exit 1; }
      "${SWEGEMMA[@]}" eval --tasks "$TASKS_FILE" --snapshots-dir "$SNAPSHOTS_DIR" \
        --submission-dir "results/$E/_agent" --results-dir "results/$E/r$r/c$j" --sandbox docker \
        --task-ids "${part[@]}" --max-time-minutes "$MIN" --max-tool-calls "$CALLS" --concurrency "$CONC" \
        ${TIMEOUT_S:+--timeout-seconds "$TIMEOUT_S"} ${MAX_TURNS:+--max-turns "$MAX_TURNS"} \
        --display quiet >> "logs/$E.log" 2>&1
      echo "$(date +%H:%M) round $r chunk $j/$((NCH - 1)) $E done"
    done
  done
  for E in "${RUNS[@]}"; do echo "$(date +%H:%M) $E round $r: $(cnt "results/$E/r$r")"; done
done
ok && echo "server still OK at end" || echo "SERVER HUNG at end"
