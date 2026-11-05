"""What the successful DeepSeek runs read before their first edit on the solvable-but-missed tasks, vs Gemma's runs:
distinct files read, split into gold / test / other source; for stock tools, read_file paths and cat/sed/grep targets."""
import json, glob, ast, re, statistics as st
tasks = json.load(open('results/_analysis/ceiling/solvable.json'))
GEMMA = ['E069_v22_real', 'E070_v22_real', 'E073_v24_real', 'E075_neb_v24_base']
PATH = re.compile(r'([\w./-]+\.py)')
def calls(run, t):
    fs = glob.glob(f'results/{run}/r1/*/traces/trace_{t}.json')
    if not fs: return None
    out = []
    for s in json.load(open(fs[0]))['steps']:
        tc = s.get('tool_calls')
        if not tc: continue
        for c in (ast.literal_eval(tc) if isinstance(tc, str) else tc):
            a = c.get('arguments') or {}
            name = a.get('file_path') if c['function_name'] == 'run_skill_script' else c['function_name']
            args = a.get('args') if c['function_name'] == 'run_skill_script' else a
            out.append((name, args if isinstance(args, dict) else {}))
    return out
def files_read_before_edit(cs):
    seen = []
    for n, a in cs:
        if n in ('edit.py', 'create.py', 'edit_file', 'write_file'): break
        txt = ' '.join(str(v) for v in a.values())
        if n in ('show.py', 'read_file', 'run_command', 'search.py'):
            for p in PATH.findall(txt):
                p = p.replace('/workspace/', '').lstrip('./')
                if p not in seen: seen.append(p)
    return seen
def kind(p, gold):
    if p in gold: return 'gold'
    if re.search(r'(^|/)(tests?|testing)/|test_', p): return 'test'
    return 'other'
summ = {'ds': [], 'gemma': []}
for x in tasks:
    gold = set(x['gold_files'])
    for run in x['ds_wins']:
        cs = calls(run, x['task'])
        if cs is None: continue
        fr = files_read_before_edit(cs); k = [kind(p, gold) for p in fr]
        summ['ds'].append((x['task'], run[:4], len(fr), k.count('gold'), k.count('test'), k.count('other')))
    for run in GEMMA:
        cs = calls(run, x['task'])
        if cs is None: continue
        fr = files_read_before_edit(cs); k = [kind(p, gold) for p in fr]
        summ['gemma'].append((x['task'], run[:4], len(fr), k.count('gold'), k.count('test'), k.count('other')))
for who in ('ds', 'gemma'):
    rows = summ[who]
    print(f"{who:6} runs {len(rows):3}: distinct files read before first edit median {st.median(r[2] for r in rows):.0f} "
          f"(gold {st.mean(r[3] for r in rows):.1f}, test {st.mean(r[4] for r in rows):.1f}, other {st.mean(r[5] for r in rows):.1f}); "
          f"runs that read a test file before editing: {sum(1 for r in rows if r[4])}/{len(rows)}")

# do the pre-edit test reads hit the files the hidden tests live in (test_patch)?
tasks_all = {json.loads(l)['instance_id']: json.loads(l) for l in open('data/competition/tasks.jsonl')}
def tp_files(t):
    return set(re.findall(r'^(?:diff --git a/|\+\+\+ b/)(\S+)', tasks_all[t]['test_patch'], re.M))
hit = {'ds': [0, 0], 'gemma': [0, 0], 'v24': [0, 0]}
for x in tasks:
    tp = tp_files(x['task'])
    for run in x['ds_wins'] + GEMMA:
        cs = calls(run, x['task'])
        if cs is None: continue
        fr = set(files_read_before_edit(cs))
        who = 'ds' if run not in GEMMA else 'gemma'
        hit[who][0] += 1; hit[who][1] += bool(fr & tp)
        if run in ('E073_v24_real', 'E075_neb_v24_base'):
            hit['v24'][0] += 1; hit['v24'][1] += bool(fr & tp)
for who, (n, h) in hit.items():
    print(f"{who:6}: pre-edit reads include a file the hidden tests modify in {h}/{n} runs")
