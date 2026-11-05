"""Big-repo dig, step 2: per task and run, the localization/edit funnel from the traces: rank of a gold file in the
first locate.py output, gold file read (show.py target / search hit shown), gold file edited, all gold files edited,
outcome. unseen100 E076 (v24) + E077 (v25), grouped by repo size."""
import json, glob, re, collections
meta = {json.loads(l)['instance_id']: json.loads(l) for l in open('data/unseen/meta.jsonl')}
ids = [l.strip() for l in open('experiments/unseen100_ids.txt')]
def res(d):
    a = {}
    for f in glob.glob(f'results/{d}/r*/c*/task_results.jsonl'):
        for l in open(f): x = json.loads(l); a[x['instance_id']] = x
    return a
def calls(run, t):
    fs = glob.glob(f'results/{run}/r1/c*/traces/trace_{t}.json')
    if not fs: return None
    out = []
    for s in json.load(open(fs[0])).get('steps', []):
        obs = str((s.get('observation') or {}).get('content') or '')
        for tc in s.get('tool_calls') or []:
            ar = tc.get('arguments') or {}
            out.append((str(ar.get('file_path') or tc.get('function_name')), ar.get('args') or {}, obs))
    return out
def funnel(run):
    r = res(run); rows = []
    for t in ids:
        if t not in r: continue
        cs = calls(run, t)
        if cs is None: continue
        gold = [g for g in meta[t]['gold_files'] if g.endswith('.py')] or meta[t]['gold_files']
        rank = None
        loc = next((c for c in cs if c[0] == 'locate.py'), None)
        if loc:
            try:
                out = json.loads(loc[2]).get('stdout', '')
            except Exception:
                out = loc[2].replace('\\n', '\n')
            files = re.findall(r'^\s*\d+\.\d+\s+(\S+)', out, re.M)
            rank = next((i + 1 for i, f in enumerate(files) if f in gold), 'miss')
        def tgt(c): return str(c[1].get('target') or c[1].get('file') or c[1].get('path') or '')
        read = any(c[0] == 'show.py' and tgt(c) in gold for c in cs)
        edited = {tgt(c) for c in cs if c[0] == 'edit.py' and 'EDITED' in c[2]}
        rows.append(dict(t=t, size=meta[t]['py_files'], gl=meta[t]['gold_lines'], rank=rank, read=read,
                         egold=bool(edited & set(gold)), eall=set(gold) <= edited, edited=bool(edited),
                         ok=r[t]['resolved'], n=len(cs)))
    return rows
for run in ('E076_neb_v24_unseen', 'E077_neb_v25_unseen'):
    rows = funnel(run)
    print(f'== {run}: {len(rows)} traced tasks')
    for name, sel in (('<400 py files', lambda x: x['size'] < 400), ('>=400 py files', lambda x: x['size'] >= 400)):
        g = [x for x in rows if sel(x)]
        n = len(g)
        r1 = sum(x['rank'] == 1 for x in g); r5 = sum(isinstance(x['rank'], int) and x['rank'] <= 5 for x in g)
        print(f'  {name:15} n {n:3} | locate gold@1 {r1:2} @5 {r5:2} | read gold {sum(x["read"] for x in g):2} | '
              f'edited any {sum(x["edited"] for x in g):2} | edited a gold file {sum(x["egold"] for x in g):2} | '
              f'all gold files {sum(x["eall"] for x in g):2} | pass {sum(x["ok"] for x in g):2}')
    fails = [x for x in rows if not x['ok']]
    stage = collections.Counter('locate missed & never read gold' if not x['read'] and x['rank'] != 1 else
                                'read gold, no edit at all' if not x['edited'] else
                                'edited, but never a gold file' if not x['egold'] else
                                'edited some, not all gold files' if not x['eall'] else
                                'edited all gold files, still wrong' for x in fails)
    print('  failure stage:', dict(stage))
json.dump(funnel('E076_neb_v24_unseen'), open('results/_analysis/bigrepo/funnel_v24.json', 'w'))
