"""Big-repo dig, step 1: does repository size predict failure? unseen100 (E076 v24, E077 v25) + eval113 (E075 v24)."""
import json, glob, collections, statistics as S
def L(d):
    a = {}
    for f in glob.glob(f'results/{d}/r*/c*/task_results.jsonl'):
        for l in open(f): x = json.loads(l); a[x['instance_id']] = x
    return a
meta = {json.loads(l)['instance_id']: json.loads(l) for l in open('data/unseen/meta.jsonl')}
ids = [l.strip() for l in open('experiments/unseen100_ids.txt')]
a, b = L('E076_neb_v24_unseen'), L('E077_neb_v25_unseen')
def q(vals): vals = sorted(vals); return [vals[int(p * (len(vals) - 1))] for p in (0.25, 0.5, 0.75)]
print('py_files quartiles', q([meta[t]['py_files'] for t in ids]), '| py_loc', q([meta[t]['py_loc'] for t in ids]))
def bucket(n):
    return '<150' if n < 150 else '150-400' if n < 400 else '400-1000' if n < 1000 else '>=1000'
g = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
for t in ids:
    k = bucket(meta[t]['py_files']); r = g[k]
    r[0] += 1; r[1] += a[t]['resolved'] if t in a else 0; r[2] += b[t]['resolved'] if t in b else 0
    r[3] += meta[t]['gold_lines']; r[4] += a[t].get('agent_patch_size', 0) == 0 if t in a else 0
for k in ('<150', '150-400', '400-1000', '>=1000'):
    n, p24, p25, gl, emp = g[k]
    print(f'py files {k:9} n {n:3}  v24 {p24:2} ({100*p24/max(n,1):3.0f}%)  v25 {p25:2} ({100*p25/max(n,1):3.0f}%)  mean gold lines {gl/max(n,1):5.1f}  v24 empty {emp}')
# eval113 sizes for reference: count .py files in public snapshots via task repo (4 repos)
# joint: size x gold lines
print('\nsize x gold size (v24 pass/n):')
for small_gold in (True, False):
    row = []
    for k in ('<150', '150-400', '400-1000', '>=1000'):
        ts = [t for t in ids if bucket(meta[t]['py_files']) == k and (meta[t]['gold_lines'] <= 30) == small_gold]
        row.append(f"{k}: {sum(a[t]['resolved'] for t in ts if t in a)}/{len(ts)}")
    print(' gold<=30 ' if small_gold else ' gold>30  ', ' | '.join(row))
