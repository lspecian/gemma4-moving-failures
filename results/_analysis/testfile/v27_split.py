"""v27 vs v24 split by whether v27's excerpt showed a file the hidden tests modify (from the traces)."""
import json, glob, re, sys
HEAD = re.compile(r'tests of this code in the repository \(the hidden tests usually follow them\): (\S+):')
def load(run):
    out = {}
    for f in glob.glob(f'results/{run}/r1/*/task_results.jsonl'):
        for l in open(f):
            r = json.loads(l); out[r['instance_id']] = r
    return out
def shown(run, t):
    fs = glob.glob(f'results/{run}/r1/*/traces/trace_{t}.json')
    if not fs: return []
    return HEAD.findall(open(fs[0]).read().replace('\\\\n', '\n'))
for new, old, tasks_path in (('E086_v27_eval113', 'E075_neb_v24_base', 'data/competition/tasks.jsonl'),
                             ('E087_v27_unseen', 'E081_v24_unseen_rerun', 'data/unseen/tasks.jsonl')):
    tasks = {json.loads(l)['instance_id']: json.loads(l) for l in open(tasks_path)}
    N, O = load(new), load(old)
    groups = {'hit': [], 'other file': [], 'none': []}
    for t in N:
        hidden = set(re.findall(r'^(?:diff --git a/|\+\+\+ b/)(\S+)', tasks[t].get('test_patch') or '', re.M))
        s = set(shown(new, t))
        g = 'hit' if s & hidden else ('other file' if s else 'none')
        groups[g].append(t)
    print(new)
    for g, ts in groups.items():
        c = [t for t in ts if t in O]
        print(f"  excerpt {g:10}: {len(ts):3} tasks  v27 pass {sum(N[t]['resolved'] for t in c):3}  v24 pass {sum(O[t]['resolved'] for t in c):3}  "
              f"v27-only {sum(1 for t in c if N[t]['resolved'] and not O[t]['resolved'])}  v24-only {sum(1 for t in c if O[t]['resolved'] and not N[t]['resolved'])}")
