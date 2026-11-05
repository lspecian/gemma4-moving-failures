"""Per failed run: patch size, calls, first edit call, edit attempts/failures, submit, wall; edit failure messages.

    python3 results/_analysis/fails/fails.py results/E103_v32
"""
import ast
import collections
import glob
import json
import sys

run = sys.argv[1]
res = {}
for f in glob.glob(f'{run}/r1/*/task_results.jsonl'):
    for line in open(f):
        r = json.loads(line)
        res[r['instance_id']] = r
errs = collections.Counter()
rows = []
for f in sorted(glob.glob(f'{run}/r1/*/traces/*.json')):
    t = json.load(open(f))
    iid = f.split('trace_')[-1][:-5]
    n, first_edit, ed, ed_fail, sub, tests = 0, None, 0, 0, False, 0
    for s in t['steps']:
        tc = s.get('tool_calls')
        if not tc:
            continue
        tc = tc if isinstance(tc, list) else ast.literal_eval(tc)
        obs = str((s.get('observation') or {}).get('content'))
        for c in tc:
            n += 1
            fn = c.get('function_name')
            a = json.dumps(c.get('arguments') or {})
            if fn in ('edit_file', 'write_file') or (fn == 'run_command' and ('.replace(' in a or 'sed -i' in a)):
                ed += 1
                if '"status": "error"' in obs or 'not found' in obs.lower() or 'missing' in obs.lower():
                    ed_fail += 1
                    errs[obs[:110]] += 1
                elif first_edit is None:
                    first_edit = n
            if fn == 'submit_patch':
                sub = True
            if fn == 'run_command' and 'pytest' in a:
                tests += 1
    r = res.get(iid, {})
    rows.append((r.get('resolved'), iid, r.get('agent_patch_size'), n, first_edit, ed, ed_fail, tests, sub,
                 round(r.get('duration_seconds', 0))))
print('ok  task              patch calls 1st_edit edits fail tests submit wall')
for row in sorted(rows, key=lambda x: (bool(x[0]), x[1])):
    print('%-3s %-17s %5s %5s %8s %5s %4s %5s %6s %4s' % ('Y' if row[0] else '-', *row[1:]))
print('\nedit failure messages:')
for k, v in errs.most_common(8):
    print(f'{v:3} {k!r}')
