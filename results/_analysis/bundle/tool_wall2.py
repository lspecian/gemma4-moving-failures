"""Join harness-log tool durations (in order) with the trace's tool names (in order): exact wall time per script."""
import re, glob, json, ast, collections, sys, os, statistics as st
ANSI = re.compile(r'\x1b\[[0-9;]*m')
CALL = re.compile(r'\[\s*([\d.]+)s\]\s+🤖 (\w+)'); RES = re.compile(r'\[\s*([\d.]+)s\]\s+→ (\w+)')
name = sys.argv[1]; agg = collections.defaultdict(list); bypos = collections.defaultdict(list); mism = 0; tasks = 0
for logf in glob.glob(f'results/{name}/r1/*/logs/*.log'):
    tid = os.path.basename(logf)[:-4]
    tr = glob.glob(f'results/{name}/r1/*/traces/trace_{tid}.json')
    if not tr: continue
    names = []
    for s in json.load(open(tr[0]))['steps']:
        tc = s.get('tool_calls')
        if not tc: continue
        for c in (ast.literal_eval(tc) if isinstance(tc, str) else tc):
            k = c['function_name']
            if k == 'run_skill_script': k = (c.get('arguments') or {}).get('file_path', '?')
            names.append(k)
    durs = []; pend = None
    for line in open(logf, errors='replace'):
        line = ANSI.sub('', line); m = CALL.search(line)
        if m: pend = float(m.group(1)); continue
        m = RES.search(line)
        if m and pend is not None: durs.append(float(m.group(1)) - pend); pend = None
    if len(durs) != len([n for n in names if n not in ('submit_patch',)]) and len(durs) != len(names): mism += 1
    tasks += 1
    for i, (k, d) in enumerate(zip(names, durs)):
        agg[k].append(d); bypos[min(i, 5)].append(d)
print(f"{name}: {tasks} tasks ({mism} with call-count mismatch)")
for k, v in sorted(agg.items(), key=lambda kv: -sum(kv[1]))[:10]:
    print(f"   {k:16} calls {len(v):5}  mean {sum(v)/len(v):5.1f}s  median {st.median(v):5.1f}s  p90 {sorted(v)[int(.9*len(v))]:5.1f}s  per task {sum(v)/tasks:5.1f}s")
print('   by call position (0..4, 5+): ' + ', '.join(f"{p}: med {st.median(v):.1f}s mean {st.mean(v):.1f}s" for p, v in sorted(bypos.items())))
