"""Fit prompt_tokens against what flow.py can measure (E037 v16 eval113 traces)."""
import glob
import json
import sys

import numpy as np

rows = []  # (prompt_tokens, chars, calls, elapsed, chars_raw_stdout_only, task)
for f in sorted(glob.glob('results/E037_v16_eval113/r*/c*/traces/*.json')):
    d = json.load(open(f))
    chars = calls = 0
    stdout_chars = 0
    t0 = None
    for x in d['steps']:
        m = (x.get('metrics') or {}).get('prompt_tokens')
        tcs = x.get('tool_calls') or []
        ts = None
        for t in tcs:
            ts = (t.get('extra') or {}).get('timestamp') or ts
        if m and tcs:
            el = (ts - t0) if (t0 and ts) else 0.0
            rows.append((m, chars, calls, el, stdout_chars, f))
        o = (x.get('observation') or {}).get('content') or ''
        ok = False
        try:
            j = json.loads(o)
            ok = isinstance(j, dict) and 'stdout' in j
        except Exception:
            j = None
        if ok and tcs:
            a = tcs[0].get('arguments') or {}
            args = a.get('args')
            if isinstance(args, dict):
                ac = sum(len('--' + str(k)) + len(str(v)) for k, v in args.items())
            elif isinstance(args, list):
                ac = sum(len(str(v)) for v in args)
            else:
                ac = len(str(args or ''))
            chars += len(j['stdout']) + ac
            stdout_chars += len(j['stdout'])
            calls += 1
            if t0 is None and ts:
                t0 = ts

y = np.array([r[0] for r in rows], float)
C = np.array([r[1] for r in rows], float)
N = np.array([r[2] for r in rows], float)
E = np.array([r[3] for r in rows], float)
print('n', len(rows), 'tasks', len({r[5] for r in rows}))


def fit(X, name):
    X = np.column_stack([np.ones(len(y))] + X)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    print(name, 'coef', np.round(b, 4), 'sd', round(r.std(), 1), 'p95', round(np.percentile(r, 95), 1),
          'p99', round(np.percentile(r, 99), 1), 'max', round(r.max(), 1))
    hi = y > 20000
    print('   >20k: n', hi.sum(), 'mean resid', round(r[hi].mean(), 1), 'sd', round(r[hi].std(), 1),
          'p95', round(np.percentile(r[hi], 95), 1))
    return b, r


fit([C, N], 'chars+calls')
fit([C / 1, N, E], 'chars+calls+elapsed')
b, r = fit([C], 'chars only')

b, r = fit([C, N], 'final')
hi = y > 20000
P95 = float(np.percentile(r[hi], 95))
est = b[0] + b[1] * C + b[2] * N + P95
print('P95 used', P95)
for thr in (15000, 22000, 25500):
    print(thr, 'steps est>=thr', round((est >= thr).mean(), 3), 'runs', len({rows[i][5] for i in np.where(est >= thr)[0]}),
          'actual>=thr steps', round((y >= thr).mean(), 3))
# runs whose actual max >= 28000: first step where est >= 22000/25500 vs actual
runs = {}
for i, row in enumerate(rows):
    runs.setdefault(row[5], []).append(i)
for f, idx in runs.items():
    if max(y[i] for i in idx) >= 27000:
        def first(arr, thr):
            return next((k for k, i in enumerate(idx) if arr[i] >= thr), None)
        print(f.split('/')[-1], f.split('/')[2], 'steps', len(idx), 'max', int(max(y[i] for i in idx)),
              'est22', first(est, 22000), 'est25.5', first(est, 25500), 'act22', first(y, 22000), 'act28', first(y, 28000))
# under-estimates near the limit: actual >= 29000 and est < 25500
bad = [(int(y[i]), int(est[i])) for i in range(len(y)) if y[i] >= 29000 and est[i] < 25500]
print('actual>=29k est<25.5k', len(bad), bad[:10])
