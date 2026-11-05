"""Combine tests_for (reciprocal rank) with issue-BM25 over test files; report hidden-test-file @1/@3 per weight.
target = 'gold' (tests_for from the gold files: upper bound) or 'top' (from locate's top source file: realistic)."""
import json, sys
def combo(r, target, wb):
    lst = r['ranked'] if target == 'gold' else r['from_top']
    sc = {}
    for i, p in enumerate(lst):
        sc[p] = sc.get(p, 0) + 1.0 / (i + 1)
    bm = r['bm25_tests']; mx = bm[0][0] if bm else 1
    for s, p in (bm if wb else []):  # wB = 0: tests_for alone (the current tool)
        sc[p] = sc.get(p, 0) + wb * s / mx
    return [p for p, _ in sorted(sc.items(), key=lambda kv: -kv[1])]
def evaluate(rows, target, wb):
    f = [r for r in rows if r['existing']]
    at = {1: 0, 3: 0}
    for r in f:
        ranked = combo(r, target, wb)
        rk = next((i + 1 for i, p in enumerate(ranked) if p in r['existing']), None)
        for k in at:
            at[k] += bool(rk and rk <= k)
    return len(f), at[1], at[3]
sets = {n: json.load(open(f'results/_analysis/testfile/{n}_sig.json')) for n in ('eval113', 'unseen100')}
for target in ('gold', 'top'):
    print(f'target = {target}')
    for wb in (0, 0.25, 0.5, 1, 2, 4, 100):
        cells = []
        for n, rows in sets.items():
            m, a1, a3 = evaluate(rows, target, wb)
            cells.append(f'{n} @1 {a1}/{m} @3 {a3}/{m}')
        print(f'  wB={wb:<5}', ' | '.join(cells))
