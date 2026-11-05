"""`swegemma` CLI with Kaggle's scorer settings for local runs.

Same arguments as `.venv/bin/swegemma`. Two differences from the stock CLI:

- Per-task test environments (scripts/task_env.py): both the agent sandbox (Container A) and the verification
  sandbox (Container B) get the task's dependency versions. SWEGEMMA_TASK_ENV=0 gives the stock published environment.
- The scorer's App context management (HARNESS_README 7.2, Kaggle's scripts/inference.py), which the stock CLI does not
  set: token-threshold events compaction (before every model call, once the last prompt reached 14,336 tokens, all but
  the last 5 events are replaced by an LLM summary of their *text* parts: tool calls and results are dropped) and
  context caching. Local runs before 2026-10-04 ran without it. SWEGEMMA_KAGGLE_APP=0 turns it off.

    .venv/bin/python scripts/swegemma_local.py eval --tasks data/competition/tasks.jsonl ...
"""

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swegemma import cli  # noqa: E402

import task_env  # noqa: E402

# HARNESS_README 7.2 (scripts/inference.py)
COMPACTION = dict(compaction_interval=5, overlap_size=2, token_threshold=14336, event_retention_size=5)
CACHE = dict(min_tokens=2048, ttl_seconds=1800, cache_intervals=10)


def install_kaggle_app():
    """Make the CLI's EvalConfig carry the scorer's compaction and cache configs (unless given explicitly)."""
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig

    base = cli.EvalConfig

    def eval_config(**kw):
        kw.setdefault('events_compaction_config', EventsCompactionConfig(**COMPACTION))
        kw.setdefault('context_cache_config', ContextCacheConfig(**CACHE))
        return base(**kw)

    cli.EvalConfig = eval_config


if __name__ == '__main__':
    task_env.install_hook()
    if os.environ.get('SWEGEMMA_KAGGLE_APP', '1') != '0':
        install_kaggle_app()
    sys.exit(cli.main(sys.argv[1:]))
