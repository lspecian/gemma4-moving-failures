"""Calibrate flow.py's sawtooth context model (agent-v18, A1) on per-step prompt_tokens.

Kaggle's scorer compacts before a model call once the previous prompt had >= 14,336 tokens: every event but the last
5 becomes an LLM summary of their text parts. flow.py estimates the next prompt from what it records (characters of
results + arguments, number of calls) and resets the estimate at a presumed compaction. This script

  1. refits the growth (prompt ~ base + per_char * chars + per_call * calls) on the uncompacted segments of
     E041 (local, never compacted) + E042 (Kaggle-like, compacting),
  2. fits the post-compaction size (prompt right after a drop) against the characters of the last K flow calls,
  3. replays flow's sawtooth on the E042 runs with the fitted constants and reports how its presumed compactions line
     up with the real ones (same call, early, late, missed, false).

    python3 scripts/fit_ctx_sawtooth.py [glob ...]
"""
import glob
import json
import sys

import numpy as np

COMPACT_AT = 14336
GLOBS = sys.argv[1:] or ['results/E041_v17b_eval113/r*/c*/traces/*.json',
                         'results/E042_v17b_kagglelike/r*/c*/traces/*.json']


def runs(globs):
    """{trace: [(prompt_tokens, flow_chars or None, completion_tokens)]} per model step that made a tool call."""
    out = {}
    for g in globs:
        for f in sorted(glob.glob(g)):
            try:
                d = json.load(open(f))
            except (OSError, ValueError):
                continue  # a trace still being written
            seq = []
            for x in d.get('steps', []):
                m = x.get('metrics') or {}
                p = m.get('prompt_tokens')
                tcs = x.get('tool_calls') or []
                if not p:
                    continue
                fc = None
                o = (x.get('observation') or {}).get('content') or ''
                try:
                    j = json.loads(o)
                except (TypeError, ValueError):
                    j = None
                if tcs and isinstance(j, dict) and 'stdout' in j and tcs[0].get('function_name') == 'run_skill_script':
                    a = tcs[0].get('arguments') or {}
                    args = a.get('args')
                    if isinstance(args, dict):
                        ac = sum(len('--' + str(k)) + len(str(v)) for k, v in args.items())
                    elif isinstance(args, list):
                        ac = sum(len(str(v)) for v in args)
                    else:
                        ac = len(str(args or ''))
                    fc = len(j['stdout']) + ac
                seq.append((p, fc, m.get('completion_tokens') or 0))
            if seq:
                out[f] = seq
    return out


def segments(seq):
    """Split a run at its compactions (a drop of the prompt after a prompt >= COMPACT_AT): [(start, end)]."""
    cuts = [i for i in range(1, len(seq)) if seq[i - 1][0] >= COMPACT_AT and seq[i][0] < seq[i - 1][0] - 1000]
    bounds = [0] + cuts + [len(seq)]
    return list(zip(bounds, bounds[1:])), cuts


def lstsq(X, y, name):
    X = np.column_stack([np.ones(len(y))] + X)
    with np.errstate(all='ignore'):  # spurious matmul warnings of some numpy/Accelerate builds
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
    print(f'{name}: n={len(y)} coef={np.round(b, 4).tolist()} sd={r.std():.0f} p75={np.percentile(r, 75):.0f} '
          f'p90={np.percentile(r, 90):.0f} p95={np.percentile(r, 95):.0f}')
    return b, r


def main():
    data = runs(GLOBS)
    e42 = {f: s for f, s in data.items() if 'E042' in f}
    print('runs', len(data), 'of which E042', len(e42))
    # 1. growth on the first (uncompacted) segment of every run
    ys, C, N, src = [], [], [], []
    for f, seq in data.items():
        (a, b), *_ = segments(seq)[0]
        chars = calls = 0
        for p, fc, _ in seq[a:b]:
            ys.append(p)
            C.append(chars)
            N.append(calls)
            src.append('E042' in f)
            if fc is not None:
                chars += fc
                calls += 1
    y, C, N, src = map(np.array, (ys, C, N, src))
    lstsq([C, N], y, 'growth E041+E042')
    lstsq([C[src], N[src]], y[src], 'growth E042 only')
    lstsq([C[~src], N[~src]], y[~src], 'growth E041 only')
    b, r = lstsq([C, N], y, 'growth (used)')
    near = (y >= 11000) & (y <= 16000)
    print(f'   residual where 11k <= actual <= 16k: n={near.sum()} mean={r[near].mean():.0f} sd={r[near].std():.0f} '
          f'p90={np.percentile(r[near], 90):.0f}')
    base, per_char, per_call = b
    # 2. post-compaction size: the prompt right after a drop vs the last K flow calls' characters
    rows = []
    for f, seq in e42.items():
        for i in segments(seq)[1]:
            before = [fc for _, fc, _ in seq[:i]]
            rows.append((seq[i][0], before, seq[i - 1][0], f))
    print('compactions', len(rows))
    for k in (1, 2, 3):
        yy = np.array([r[0] for r in rows], float)
        last = np.array([sum(x or 0 for x in r[1][-k:]) for r in rows], float)
        nk = np.array([sum(1 for x in r[1][-k:] if x is not None) for r in rows], float)
        lstsq([last], yy, f'post K={k} free slope')
        rest = yy - per_char * last - per_call * nk
        print(f'   K={k} with the growth slopes: intercept mean={rest.mean():.0f} sd={rest.std():.0f} '
              f'p75={np.percentile(rest, 75):.0f} p90={np.percentile(rest, 90):.0f} min={rest.min():.0f}')
    # 3. replay the sawtooth on E042 (K results kept in the estimate, VISIBLE results presumed still in context):
    #    for every (step, earlier result) pair, is the result gone (a real compaction since that dropped it) and does
    #    the replay say so? recall = gone results flagged gone; precision = flagged results really gone.
    K, VISIBLE = 3, 2
    yy = np.array([r[0] for r in rows], float)
    last = np.array([sum(x or 0 for x in r[1][-K:]) for r in rows], float)
    nk = np.array([sum(1 for x in r[1][-K:] if x is not None) for r in rows], float)
    post_base = float(np.mean(yy - per_char * last - per_call * nk))
    print(f'post-compaction base (K={K}, mean) = {post_base:.0f}')
    for margin in (0, 300, 600, 900):
        tp = fp = fn = tn = 0
        for f, seq in e42.items():
            real = segments(seq)[1]
            chars = calls = 0
            mode_base, prev, hist, pres = base, None, [], []
            for i, (p, fc, _) in enumerate(seq):
                est = mode_base + per_char * chars + per_call * calls
                if prev is not None and prev + margin >= COMPACT_AT:
                    pres.append(i)  # presumed compaction before this step
                    keep = hist[-K:]
                    mode_base = post_base
                    chars = sum(x for x in keep if x is not None)
                    calls = sum(1 for x in keep if x is not None)
                    est = mode_base + per_char * chars + per_call * calls
                prev = est
                hist.append(fc)
                if fc is not None:
                    chars += fc
                    calls += 1
            for i in range(len(seq)):
                for m in range(i):
                    if seq[m][1] is None:
                        continue
                    gone = any(m <= j - 4 and j <= i for j in real)        # the last 5 events keep 3 results
                    flagged = any(m <= j - 1 - VISIBLE and j <= i for j in pres)
                    tp += gone and flagged
                    fp += flagged and not gone
                    fn += gone and not flagged
                    tn += not gone and not flagged
        print(f'margin {margin}: gone results flagged (recall) {tp / max(1, tp + fn):.2f}, flagged really gone '
              f'(precision) {tp / max(1, tp + fp):.2f}  (tp {tp} fp {fp} fn {fn} tn {tn})')

if __name__ == '__main__':
    main()
