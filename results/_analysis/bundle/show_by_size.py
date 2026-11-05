"""Median show.py / locate.py time per task vs repository size (unseen100, v24 E076): per-call cost from repo size or from env?"""
import re, glob, json, ast, os, statistics as st
ANSI = re.compile(r'\x1b\[[0-9;]*m'); CALL = re.compile(r'\[\s*([\d.]+)s\]\s+🤖 (\w+)'); RES = re.compile(r'\[\s*([\d.]+)s\]\s+→ (\w+)')
meta = {json.loads(l)['instance_id']: json.loads(l) for l in open('data/unseen/meta.jsonl')}
env = json.load(open('data/unseen/task_env.json'))
rows = []
for logf in glob.glob('results/E076_neb_v24_unseen/r1/*/logs/*.log'):
    tid = os.path.basename(logf)[:-4]; tr = glob.glob(f'results/E076_neb_v24_unseen/r1/*/traces/trace_{tid}.json')
    names = []
    for s in json.load(open(tr[0]))['steps']:
        tc = s.get('tool_calls')
        if tc:
            for c in (ast.literal_eval(tc) if isinstance(tc, str) else tc):
                names.append((c.get('arguments') or {}).get('file_path', c['function_name']))
    durs = []; pend = None
    for line in open(logf, errors='replace'):
        line = ANSI.sub('', line); m = CALL.search(line)
        if m: pend = float(m.group(1)); continue
        m = RES.search(line)
        if m and pend is not None: durs.append(float(m.group(1)) - pend); pend = None
    show = [d for n, d in zip(names, durs) if n == 'show.py']
    loc = [d for n, d in zip(names, durs) if n == 'locate.py'][:1]
    e = env.get(tid) or env.get('tasks', {}).get(tid) or {}
    npk = len(e.get('packages', e.get('wheels', []))) if isinstance(e, dict) else 0
    if show: rows.append((meta[tid]['repo_size_kb'], st.median(show), loc[0] if loc else None, npk, tid))
rows.sort()
q = len(rows) // 4
for i in range(4):
    part = rows[i*q:(i+1)*q] if i < 3 else rows[3*q:]
    print(f"repo size quartile {i+1} ({part[0][0]//1024}-{part[-1][0]//1024} MB): show.py median {st.median(r[1] for r in part):.1f}s, "
          f"first locate median {st.median(r[2] for r in part if r[2] is not None):.1f}s")
print('fastest show medians:', [(r[4], round(r[1],1)) for r in sorted(rows, key=lambda r: r[1])[:5]])
print('slowest show medians:', [(r[4], round(r[1],1), r[0]//1024) for r in sorted(rows, key=lambda r: -r[1])[:6]])
