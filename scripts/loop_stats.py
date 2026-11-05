"""Per-run behaviour stats from traces: repetition, time to first edit, submission.

    python3 scripts/loop_stats.py results/L001_B results/L001_C ...

For each task run: resolved?, tool calls, distinct calls, the largest number of identical calls,
the call index of the first edit (edit.py, edit_file or write_file), and whether submit_patch was called.
"""

import collections
import glob
import json
import pathlib
import sys

EDIT_SCRIPTS = ('edit.py',)


def is_edit(tc):
    fn, a = tc.get('function_name'), tc.get('arguments') or {}
    return fn in ('edit_file', 'write_file') or (fn == 'run_skill_script' and a.get('file_path') in EDIT_SCRIPTS)


for run in sys.argv[1:]:
    print(f'== {pathlib.Path(run).name}')
    for tf in sorted(glob.glob(f'{run}/**/traces/*.json', recursive=True)):
        task_dir = pathlib.Path(tf).parents[1]
        task_id = pathlib.Path(tf).stem.removeprefix('trace_')
        rows = [json.loads(l) for l in open(task_dir / 'task_results.jsonl')] if (task_dir / 'task_results.jsonl').exists() else []
        res = [r for r in rows if r.get('instance_id') == task_id] or [{}]
        calls = [tc for s in json.load(open(tf))['steps'] for tc in s.get('tool_calls') or []]
        keys = [json.dumps([tc.get('function_name'), tc.get('arguments')], sort_keys=True) for tc in calls]
        first_edit = next((i + 1 for i, tc in enumerate(calls) if is_edit(tc)), None)
        submitted = any(tc.get('function_name') == 'submit_patch' for tc in calls)
        worst = max(collections.Counter(keys).values()) if keys else 0
        r = res[-1]
        print(f"  {task_id:16} {'PASS' if r.get('resolved') else 'fail':4}  calls={len(calls):2} distinct={len(set(keys)):2} "
              f"max_repeat={worst:2} first_edit={first_edit or '-':>2}  submitted={'yes' if submitted else 'no':3} "
              f"{r.get('duration_seconds', 0):.0f}s")
