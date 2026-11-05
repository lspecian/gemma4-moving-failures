"""Exact tool wall time from the harness logs: time of the result line (→) minus the call line (🤖), per tool/script."""
import re, glob, json, collections, sys, statistics as st
ANSI = re.compile(r'\x1b\[[0-9;]*m')
CALL = re.compile(r'\[\s*([\d.]+)s\]\s+🤖 (\w+)\s*(.*)')
RES = re.compile(r'\[\s*([\d.]+)s\]\s+→ (\w+)')
def run(name):
    agg = collections.defaultdict(list); per_task = {}; first_locate = []
    for f in glob.glob(f'results/{name}/r1/*/logs/*.log'):
        pend = None; tot = 0.0; seen_loc = False
        for line in open(f, errors='replace'):
            line = ANSI.sub('', line)
            m = CALL.search(line)
            if m:
                key = m.group(2)
                if key == 'run_skill_script':
                    fp = re.search(r'"file_path": "([^"]+)"', m.group(3)); key = fp.group(1) if fp else '?'
                pend = (float(m.group(1)), key); continue
            m = RES.search(line)
            if m and pend:
                dt = float(m.group(1)) - pend[0]; agg[pend[1]].append(dt); tot += dt
                if pend[1] == 'locate.py' and not seen_loc: first_locate.append(dt); seen_loc = True
                pend = None
        per_task[f] = tot
    n = len(per_task)
    print(f"{name}: {n} tasks, tool wall time per task mean {sum(per_task.values())/n:.0f}s median {st.median(per_task.values()):.0f}s")
    for k, v in sorted(agg.items(), key=lambda kv: -sum(kv[1]))[:8]:
        print(f"   {k:16} calls {len(v):5}  mean {sum(v)/len(v):5.1f}s  median {st.median(v):5.1f}s  per task {sum(v)/n:5.1f}s")
    if first_locate:
        print(f"   first locate.py per task: mean {st.mean(first_locate):.1f}s median {st.median(first_locate):.1f}s max {max(first_locate):.0f}s")
for r in sys.argv[1:]: run(r)
