"""Grade the reference (gold) patches through the real harness, with no model involved.

Replaces the agent phase with "return task.patch", then runs the normal Phase 2
verification. Every task should resolve; any that don't reveal a broken local sandbox
(or an unusable task) rather than an agent failure.

By default each sandbox gets its task's dependency versions (scripts/task_env.py, manifest in
data/competition/wheels_local/task_env.json). Pass --stock-env (or SWEGEMMA_TASK_ENV=0) to grade in
the published environment exactly as swegemma builds it. Other task sets: --manifest PATH (or the environment
variable TASK_ENV_MANIFEST) selects their manifest. The empty-patch check is swegemma's own --skip-agent-patch.

    .venv/bin/python scripts/verify_gold.py --tasks data/competition/tasks.jsonl \
        --snapshots-dir data/competition/snapshots --results-dir results/gold_dev12 \
        --submission-dir agent --task-ids $(cat experiments/devset_12.txt)
    .venv/bin/python scripts/verify_gold.py --manifest data/unseen/task_env.json --tasks data/unseen/tasks.jsonl \
        --snapshots-dir data/unseen/snapshots --results-dir results/unseen_gold --submission-dir agent \
        --concurrency 4 [--skip-agent-patch]
"""

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swegemma import cli  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402

import task_env  # noqa: E402


async def _gold_agent(self, task, snapshot_path, task_index, total_tasks, **kwargs):
    from adk_eval_core import SessionTrace  # noqa: PLC0415

    return task.patch, None, SessionTrace()


Evaluator._run_agent_sandbox = _gold_agent

if __name__ == '__main__':
    args = sys.argv[1:]
    if '--stock-env' in args:
        args.remove('--stock-env')
        os.environ['SWEGEMMA_TASK_ENV'] = '0'
    args = task_env.pop_manifest_arg(args)
    task_env.install_hook()
    sys.exit(cli.main(['eval', *args, '--display', 'quiet']))
