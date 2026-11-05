"""Compare run directories: pass rate per task, edit failures, sub-agent use, and a significance test.

    python3 scripts/summarize.py results/E012_v4 results/E013_v5

Accepts both layouts: <run>/<task>/ (old) and <run>/r<k>/<task>/ (run_ab.sh with repeats).
"""

import collections
import glob
import json
import pathlib
import sys
from math import comb


def fisher(a, b, c, d):
    """Two-sided Fisher exact test for [[a, b], [c, d]]."""
    n, r1, c1 = a + b + c + d, a + b, a + c

    def p(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    p0 = p(a)
    return sum(p(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if p(x) <= p0 + 1e-12)


def load(run):
    passed, total = collections.Counter(), collections.Counter()
    stats = collections.Counter()
    for tr in glob.glob(f'{run}/**/task_results.jsonl', recursive=True):
        for line in open(tr):
            r = json.loads(line)
            total[r['instance_id']] += 1
            passed[r['instance_id']] += bool(r['resolved'])
            stats['tool_calls'] += r.get('tool_calls') or 0
    for tf in glob.glob(f'{run}/**/traces/*.json', recursive=True):
        for s in json.load(open(tf))['steps']:
            for tc in s.get('tool_calls') or []:
                fn = tc.get('function_name')
                stats[f'call:{fn}'] += 1
                if '"status": "error"' in str((s.get('observation') or {}).get('content')):
                    stats[f'fail:{fn}'] += 1
    version = pathlib.Path(f'{run}/_agent.version')
    return passed, total, stats, version.read_text().strip() if version.exists() else '?'


runs = {pathlib.Path(r).name: load(r) for r in sys.argv[1:]}
tasks = sorted({t for p, tot, _, _ in runs.values() for t in tot})
print(f"{'task':18}" + ''.join(f'{name[:18]:>20}' for name in runs))
for t in tasks:
    print(f'{t:18}' + ''.join(f'{f"{p[t]}/{tot[t]}" if tot[t] else "-":>20}' for p, tot, _, _ in runs.values()))
print('-' * (18 + 20 * len(runs)))
for label, fn in [
    ('passed', lambda p, tot, s: f'{sum(p.values())}/{sum(tot.values())}'),
    ('edit failures', lambda p, tot, s: f"{s['fail:edit_file']}/{s['call:edit_file']}"),
    ('localizer calls', lambda p, tot, s: str(s['call:localizer'])),
    ('avg tool calls', lambda p, tot, s: f"{s['tool_calls'] / max(sum(tot.values()), 1):.1f}"),
]:
    print(f'{label:18}' + ''.join(f'{fn(p, tot, s):>20}' for p, tot, s, _ in runs.values()))
for name, (_, _, _, v) in runs.items():
    print(f'  {name}: {v}')
if len(runs) == 2:
    (pa, ta, _, _), (pb, tb, _, _) = runs.values()
    a, na, b, nb = sum(pa.values()), sum(ta.values()), sum(pb.values()), sum(tb.values())
    print(f'Fisher exact (pass rate): p = {fisher(a, na - a, b, nb - b):.2f}')
