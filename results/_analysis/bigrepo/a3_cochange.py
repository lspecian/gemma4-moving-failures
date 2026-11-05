"""Big-repo dig, step 3 (LLM-free): runs that edited some but not all gold files (E076 v24 + E077 v25 on unseen100).
For each, rank the repository's files by co-change with the edited gold file(s) over the snapshot's git history (commits
touching <= 20 .py files) and check whether a missed gold file is in the top 3 / top 5 (what a footer could name)."""
import json, glob, re, subprocess, tarfile, tempfile, collections, os
meta = {json.loads(l)['instance_id']: json.loads(l) for l in open('data/unseen/meta.jsonl')}
def res(d):
    a = {}
    for f in glob.glob(f'results/{d}/r*/c*/task_results.jsonl'):
        for l in open(f): x = json.loads(l); a[x['instance_id']] = x
    return a
def edited(run, t):
    fs = glob.glob(f'results/{run}/r1/c*/traces/trace_{t}.json'); out = set()
    for s in json.load(open(fs[0]))['steps'] if fs else []:
        obs = str((s.get('observation') or {}).get('content') or '')
        for tc in s.get('tool_calls') or []:
            ar = tc.get('arguments') or {}
            if str(ar.get('file_path')) == 'edit.py' and 'EDITED' in obs:
                a = ar.get('args') or {}; out.add(str(a.get('target') or a.get('file') or ''))
    return out
cases = []
for run in ('E076_neb_v24_unseen', 'E077_neb_v25_unseen'):
    r = res(run)
    for t, x in r.items():
        gold = [g for g in meta[t]['gold_files'] if g.endswith('.py')]
        e = edited(run, t)
        hit = e & set(gold)
        if hit and not set(gold) <= e and not x['resolved']:
            cases.append((run, t, sorted(hit), sorted(set(gold) - e)))
print('incomplete runs:', len(cases))
stats = collections.Counter()
for run, t, hit, missed in cases:
    snap = f'data/unseen/snapshots/{t}.tgz'
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(snap) as tf:
            gitm = [m for m in tf.getmembers() if m.name.startswith('./.git') or m.name.startswith('.git')]
            tf.extractall(d, members=gitm)
        log = subprocess.run(['git', '-C', d, 'log', '--name-only', '--pretty=format:@@', '-n', '3000'], capture_output=True, text=True).stdout
    co = collections.Counter()
    for commit in log.split('@@'):
        files = [f for f in commit.split() if f.endswith('.py')]
        if not files or len(files) > 20 or not (set(files) & set(hit)): continue
        for f in files:
            if f not in hit and not re.search(r'(^|/)tests?/|test_', f): co[f] += 1
    ranked = [f for f, _ in co.most_common(10)]
    pos = min((ranked.index(m) + 1 for m in missed if m in ranked), default=None)
    stats['top3' if pos and pos <= 3 else 'top5' if pos and pos <= 5 else 'top10' if pos else 'absent'] += 1
    print(f'{run[:4]} {t[:34]:34} edited {[h.split("/")[-1] for h in hit]} missed {[m.split("/")[-1] for m in missed]} -> co-change rank {pos} (of {len(co)} candidates)')
print(dict(stats))
