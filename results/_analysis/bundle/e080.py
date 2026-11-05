"""E080 (public bundle) vs E076 (v24) / E077 (v25) on unseen100: outcomes, discordant tasks, by gold size."""
import json, glob, collections
def load(run):
    out = {}
    for f in glob.glob(f'results/{run}/r1/*/task_results.jsonl'):
        for l in open(f):
            r = json.loads(l); out[r['instance_id']] = r
    return out
meta = {json.loads(l)['instance_id']: json.loads(l) for l in open('data/unseen/meta.jsonl')}
runs = {'v24': load('E076_neb_v24_unseen'), 'v25': load('E077_neb_v25_unseen'), 'bundle': load('E080_bundle_unseen')}
def kind(r):
    if r['resolved']: return 'pass'
    if r.get('error') and 'timeout' in r['error']: t = 'T'
    else: t = ''
    return ('wrong' if r['agent_patch_size'] > 0 else 'empty') + t
for n, R in runs.items():
    c = collections.Counter(kind(r) for r in R.values())
    dur = sum(r['duration_seconds'] for r in R.values()) / len(R)
    calls = sum(r['tool_calls'] for r in R.values()) / len(R)
    to = sum(1 for r in R.values() if r.get('error') and 'timeout' in r['error'])
    print(f"{n:7} n={len(R)} pass={c['pass']} wrong={c['wrong']+c['wrongT']} empty={c['empty']+c['emptyT']} timeouts={to} "
          f"mean dur {dur:.0f}s calls {calls:.1f}")
B = runs['bundle']
for other in ('v24', 'v25'):
    O = runs[other]; common = set(B) & set(O)
    b_only = sorted(t for t in common if B[t]['resolved'] and not O[t]['resolved'])
    o_only = sorted(t for t in common if O[t]['resolved'] and not B[t]['resolved'])
    print(f"bundle vs {other} on {len(common)}: bundle-only {len(b_only)}, {other}-only {len(o_only)}")
ours = {t for t in runs['v24'] if runs['v24'][t]['resolved']} | {t for t in runs['v25'] if runs['v25'][t]['resolved']}
bund = {t for t in B if B[t]['resolved']}
print(f"union ours (v24|v25) {len(ours)}, bundle {len(bund)}, bundle not in ours {len(bund - ours)}: {sorted(bund - ours)}")
print(f"ours not in bundle {len(ours - bund)}; union all {len(ours | bund)}")
print('\nby gold lines:      v24   v25   bundle')
for lo, hi, lab in ((0, 10, '<=10'), (11, 30, '11-30'), (31, 80, '31-80'), (81, 10**9, '>80')):
    ts = [t for t in meta if lo <= meta[t]['gold_lines'] <= hi]
    row = [f"{sum(1 for t in ts if t in R and R[t]['resolved'])}/{sum(1 for t in ts if t in R)}" for R in runs.values()]
    print(f"  {lab:6}", '  '.join(f'{x:>6}' for x in row))
