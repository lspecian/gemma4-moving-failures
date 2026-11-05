"""Paper draft 4 fact-check follow-ups: (1) Gemma's per-run solve rate on the 23 handoff tasks over ALL its real-speed
eval113 runs; (2) pass rate by issue date over the same runs (per task-run), overall and per repository."""
import json, glob, os, collections, math
tasks = {json.loads(l)['instance_id']: json.loads(l) for l in open('data/competition/tasks.jsonl')}
ids113 = {l.strip() for l in open('experiments/eval113.txt') if l.strip() and not l.startswith('#')}
the23 = [x['task'] for x in json.load(open('results/_analysis/ceiling/solvable.json'))]
NOT_GEMMA = ('E067',) + tuple(f'E0{n}' for n in range(82, 92))   # bundle (stock tools), DeepSeek, handoff
GENEROUS = ('E092',)
runs = {}
for d in sorted(glob.glob('results/E0*')):
    name = os.path.basename(d)
    eid = name[:4]
    if not ('E059' <= eid <= 'E094') or eid.startswith(NOT_GEMMA) or eid.startswith(GENEROUS):
        continue
    rs = {}
    for f in glob.glob(f'{d}/r*/*/task_results.jsonl'):
        for l in open(f):
            r = json.loads(l)
            if r['instance_id'] in ids113 or r['instance_id'] in the23:
                rs.setdefault(r['instance_id'], []).append(r['resolved'])
    if rs and any(t in ids113 for t in rs):
        runs[name] = rs
# (1) the 23
n_runs = 0; solves = []
for name, rs in runs.items():
    cov = [t for t in the23 if t in rs]
    if len(cov) < 10: continue
    s = sum(any(rs[t]) for t in cov)
    n_runs += 1; solves.append((name, s, len(cov)))
tot_s = sum(s for _, s, _ in solves); tot_tr = sum(c for _, _, c in solves)
print(f'(1) Gemma real-speed eval113 runs covering the 23: {n_runs}; solves per run on the 23: '
      f'{tot_s}/{n_runs} = {tot_s/n_runs:.2f}; per task-run {tot_s}/{tot_tr} = {tot_s/tot_tr:.1%}')
print('    runs with a solve:', [(n, s) for n, s, _ in solves if s])
# (2) recency: pass rate per task-run by created_at bucket
def bucket(t):
    y = tasks[t]['created_at'][:7]
    if y <= '2024-12': return '<=2024'
    if y <= '2025-06': return '2025H1'
    if y <= '2025-12': return '2025H2'
    return '2026'
agg = collections.defaultdict(lambda: [0, 0]); per_repo = collections.defaultdict(lambda: [0, 0])
for name, rs in runs.items():
    for t, vals in rs.items():
        if t not in ids113: continue
        for v in vals:
            b = bucket(t); agg[b][0] += v; agg[b][1] += 1
            per_repo[(t.split('_')[0], b)][0] += v; per_repo[(t.split('_')[0], b)][1] += 1
def ci(k, n):
    p = k / n; z = 1.96; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return f'{p:.0%} [{c-h:.0%}-{c+h:.0%}]'
ntask = collections.Counter(bucket(t) for t in ids113)
print(f'(2) {len(runs)} runs, {sum(v[1] for v in agg.values())} task-runs')
for b in ('<=2024', '2025H1', '2025H2', '2026'):
    print(f'    {b:7} tasks {ntask[b]:3} pass {ci(*agg[b])}')
for (repo, b), (k, n) in sorted(per_repo.items()):
    if repo in ('rich', 'fastapi'): print(f'    {repo:8} {b:7} {ci(k, n)} (n={n})')
