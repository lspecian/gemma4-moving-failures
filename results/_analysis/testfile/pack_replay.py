"""v27 test excerpt in a replay (scripts/replay_tools.py runs.jsonl): how often it is printed, its size, whether its
test file is one the hidden tests modify, and the edit/patch fidelity vs the reference agent."""
import json, re, sys, statistics as st, collections
out_dir, tasks_path = sys.argv[1], sys.argv[2]
tasks = {json.loads(l)['instance_id']: json.loads(l) for l in open(tasks_path)}
def files(p):
    return set(re.findall(r'^(?:diff --git a/|\+\+\+ b/)(\S+)', p or '', re.M))
HEAD = re.compile(r'^tests of this code in the repository \(the hidden tests usually follow them\): (\S+):', re.M)
per = collections.defaultdict(dict)
for l in open(f'{out_dir}/runs.jsonl'):
    r = json.loads(l); per[r['task']][r['agent']] = r
n = shown = hit = 0; sizes = []; first_call = []; diff = collections.Counter()
for t, ag in per.items():
    if 'v27' not in ag: continue
    n += 1; r = ag['v27']
    hidden = files(tasks[t].get('test_patch'))
    packs = []
    for c in r['calls']:
        if c.get('script') != 'show.py': continue
        out = c.get('out') or ''
        for m in HEAD.finditer(out):
            seg = out[m.start():]
            nxt = re.search(r'^(NEXT:|BUDGET:|MEMO:)', seg, re.M)
            sizes.append(len(seg[:nxt.start()] if nxt else seg)); packs.append(m.group(1))
            first_call.append(int(c['i']))
    shown += bool(packs); hit += bool(set(packs) & hidden)
    diff[r.get('diff')] += 1
print(f'{n} runs: excerpt printed in {shown} ({sum(1 for _ in sizes)} excerpts), its file is a hidden-test file in {hit}; '
      f'chars median {st.median(sizes) if sizes else 0:.0f} max {max(sizes) if sizes else 0}; first at call median '
      f'{st.median(first_call) if first_call else 0}; final patch vs recording: {dict(diff)}')
