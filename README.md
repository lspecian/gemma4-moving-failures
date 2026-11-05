# Moving Failures Around: code and data

Companion repository for the paper *Moving Failures Around: a small coding agent acts on information it is shown,
not on instructions to find it* ([paper/draft.md](paper/draft.md)), written for the paper track of Kaggle's
[Gemma 4 Developer Agent competition](https://www.kaggle.com/competitions/gemma-4-developer-agent).

The agent drives Gemma 4 31B-it (QAT w4a16, served by vLLM) to fix GitHub issues within 8 minutes and 28 tool calls
per task. The repository holds the agent the paper studies, the evaluation tools that reproduce the scorer's
conditions on rented GPUs, and the per-task outcome of every local run we report.

**Not included:** the competition's data (tasks, harness wheels, sample submission). Get them from the competition's
data page and place them under `data/competition/` (the scripts' default paths). Results' raw traces are not included
because they embed the tasks' problem statements.

## Layout

| Path | What it is |
|---|---|
| `agent/` | The paper's agent (agent-v24): a declarative Google ADK agent with one skill of stdlib scripts (`agent/skills/repo-tools/scripts`): `locate.py`, `show.py`, `search.py`, `edit.py`, `run.py`, `check_patch.py`, and `flow.py`, the flow engine that writes the footer under every result |
| `tests/` | Unit tests for the skill scripts (`scripts/test_skills.sh` runs them inside the sandbox image) |
| `scripts/` | Evaluation and training tools (below) |
| `outcomes/` | One CSV per local run: `repeat, instance_id, outcome (pass/wrong/empty), patch_size, duration_s, tool_calls, llm_calls, error`; `runs.tsv` maps each run id (`E075_...`, as cited in the paper) to its agent version and pass count |
| `experiments/` | Task lists: `eval113.txt` (the 113 local tasks), `unseen100.txt` / `unseen100_ids.txt` (99 tasks from 99 repositories, built by `scripts/unseen_data.py`) |
| `results/_analysis/` | The scripts behind the paper's numbers and figures (grouped by topic; `paper/` holds the leaderboard and baseline analyses) |
| `paper/` | The paper, its outline and figures (`paper/figures/make_figures.py`) |
| `docs/` | Architecture of the agent and notes on the LoRA adapters |

## Evaluation tools

- `scripts/slow_proxy.py`: the pacing proxy. Sits between the harness and a vLLM server and paces responses to a
  target decode speed (35 tok/s) and uncached prefill speed (1,400 tok/s), charging only the prompt tokens the server
  reports as uncached.
- `scripts/task_env.py`: per-task Python environments. The harness's shared environment lets 52 of 129 gold patches
  pass; per-task environments raise that to 115.
- `scripts/swegemma_local.py`, `scripts/run_ab.sh`, `scripts/run_chunked.sh`: run the competition harness locally
  on a task list with given budgets, one or more arms at a time.
- `scripts/replay_server.py`, `scripts/replay_tools.py`: replay a recorded run's model responses through the real
  harness with no GPU; in handoff mode, replay a strong model's run up to its first edit, then forward to Gemma (§4,
  F4 in the paper).
- `scripts/summarize.py` (pass counts, failure modes, paired test), `scripts/trace.py` (per-call view of a run),
  `scripts/verify_gold.py` (grade gold patches), `scripts/check_submission.py` (validate a submission directory).
- `scripts/vast_up.sh`, `vast_rent.sh`, `vast_session.sh`, `vast_down.sh`: rent a GPU on vast.ai, install the scorer's
  vLLM build and model, tunnel it, run the arms and destroy the instance.
- `scripts/lora_train.py`, `rft*_build.py`, `ds_sft_build.py`: the adapter training data and training (F2).

The harness is the competition's `swegemma` package (0.2.11). Docker is needed for the sandboxes (linux/amd64 image).

## The leaderboard agents

Our later leaderboard entries (v30 onwards in the paper) are the community's public "Budget-Fit" stock-tools agent
with small configuration changes: thinking budget 4,096 → 1,024 or 512 tokens and 28 → 36 or 44 tool calls, with the
prompt's budget lines changed to match. That agent is not ours and is not redistributed here.

## License

MIT, see [LICENSE](LICENSE).
