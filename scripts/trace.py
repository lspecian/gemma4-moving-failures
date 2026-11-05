"""Summarize an agent trajectory: one line per tool call, with thinking size and elapsed time.

    python3 scripts/trace.py results/E004_v1_20m/rich_3043            # a task results dir
    python3 scripts/trace.py results/E004_v1_20m/rich_3043 -v         # also show tool outputs
"""

import glob
import json
import sys

path = sys.argv[1]
verbose = '-v' in sys.argv
f = path if path.endswith('.json') else glob.glob(f'{path}/traces/trace_*.json')[0]
t = json.load(open(f))
n = 0
thought = ''
for s in t.get('steps', []):
    msg = s.get('message') or ''
    if (s.get('extra') or {}).get('event_type') == 'thinking':
        thought = str(msg)
    elif not s.get('tool_calls'):
        thought = ''
    if isinstance(msg, list):
        msg = ' '.join(str(m) for m in msg)
    calls = s.get('tool_calls') or []
    if (s.get('extra') or {}).get('event_type') != 'thinking' and (s.get('source') == 'user' or (not calls and msg)):
        who = s.get('source', '?')
        print(f'   [{who}] {str(msg)[:160]!r}')
    for i, tc in enumerate(calls):
        n += 1
        fn = tc.get('function_name') or tc.get('name')
        args = json.dumps(tc.get('arguments') or {})
        think = f'think={len(thought):>5}ch ' if i == 0 else ' ' * 15
        failed = '"status": "error"' in str((s.get('observation') or {}).get('content'))
        author = str((s.get('extra') or {}).get('author') or '')
        sub = f'[{author}] ' if author in ('writer', 'scout', 'planner', 'executor') else ''  # sub-agent steps
        print(f'{n:3}. {think}{sub}{fn:20} {"FAIL " if failed else ""}{args[:145]}')
        if verbose:
            obs = s.get('observation') or {}
            print(f'       -> {str(obs.get("content"))[:300]!r}')
m = t.get('final_metrics') or {}
print(f"prompt_tokens={m.get('total_prompt_tokens')} completion_tokens={m.get('total_completion_tokens')} steps={m.get('total_steps')}")
