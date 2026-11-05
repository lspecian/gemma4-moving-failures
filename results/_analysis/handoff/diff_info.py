"""For the tasks Gemma solved only after the information-only handoff (E090/E091): what DeepSeek's exploration (the
replayed prefix, up to its first edit) touched that Gemma's own four runs never did."""
import ast, glob, json, re, collections
GEMMA = ['E069_v22_real', 'E070_v22_real', 'E073_v24_real', 'E075_neb_v24_base']
def load(run):
    out = {}
    for f in glob.glob(f'results/{run}/r1/*/task_results.jsonl'):
        for l in open(f):
            r = json.loads(l); out[r['instance_id']] = r
    return out
def calls(run, t):
    fs = glob.glob(f'results/{run}/r1/*/traces/trace_{t}.json')
    if not fs: return []
    out = []
    for s in json.load(open(fs[0]))['steps']:
        tc = s.get('tool_calls')
        if not tc: continue
        for c in (ast.literal_eval(tc) if isinstance(tc, str) else tc):
            a = c.get('arguments') or {}
            if c['function_name'] == 'run_skill_script':
                args = a.get('args') if isinstance(a.get('args'), dict) else {}
                out.append((a.get('file_path'), args))
            else:
                out.append((c['function_name'], a))
    return out
EDIT = {'edit.py', 'create.py', 'edit_file', 'write_file'}
def prefix(cs):
    k = next((i for i, (n, _) in enumerate(cs) if n in EDIT), len(cs))
    return cs[:k]
PYF = re.compile(r'([\w./-]+\.py)')
def items(cs):
    """normalized 'information units': files read (path), symbols shown, searches, runs (snippet/test/command)."""
    files, syms, searches, runs = set(), set(), set(), []
    for n, a in cs:
        txt = ' '.join(str(v) for v in a.values())
        if n in ('show.py', 'read_file', 'outline.py'):
            for p in PYF.findall(txt): files.add(p.replace('/workspace/', '').lstrip('./'))
            if a.get('symbol'): syms.add(str(a['symbol']))
        elif n in ('search.py', 'callers.py'):
            searches.add(str(a.get('text') or a.get('query') or a.get('name') or a.get('symbol') or txt)[:60])
        elif n in ('run.py', 'repro.py', 'tests_for.py'):
            runs.append(n + ':' + txt[:90].replace('\n', ' '))
        elif n == 'run_command':
            cmd = str(a.get('command') or txt)
            if re.search(r'\b(cat|sed|head|tail|nl|less)\b', cmd):
                for p in PYF.findall(cmd): files.add(p.replace('/workspace/', '').lstrip('./'))
            elif re.search(r'\b(grep|rg|find|ls)\b', cmd):
                searches.add(cmd[:70])
            else:
                runs.append('cmd:' + cmd[:90].replace('\n', ' '))
        elif n == 'locate.py':
            searches.add('locate:' + str(a.get('query'))[:50])
    return files, syms, searches, runs
tasks = {json.loads(l)['instance_id']: json.loads(l) for l in open('data/competition/tasks.jsonl')}
src = {'E090_handoff_ours_info': 'E084_flash_v24_generous', 'E091_handoff_stock_info': 'E085_flash_bundle_generous'}
solved = collections.defaultdict(list)
for run in src:
    for t, r in load(run).items():
        if r['resolved']: solved[t].append(run)
for t in sorted(solved):
    gold = set(re.findall(r'^(?:diff --git a/|\+\+\+ b/)(\S+)', tasks[t]['patch'], re.M))
    hidden = set(re.findall(r'^(?:diff --git a/|\+\+\+ b/)(\S+)', tasks[t].get('test_patch') or '', re.M))
    g_files, g_syms, g_runs = set(), set(), 0
    for gr in GEMMA:
        f, s, se, ru = items(prefix(calls(gr, t)))
        g_files |= f; g_syms |= s; g_runs = max(g_runs, len(ru))
    print(f'=== {t}  (solved by {", ".join(solved[t])}); gold {sorted(gold)}; hidden tests {sorted(hidden)}')
    print(f'    Gemma (4 runs, before 1st edit): files {sorted(g_files)} | max runs {g_runs}')
    for run in solved[t]:
        cs = prefix(calls(src[run], t))
        f, s, se, ru = items(cs)
        print(f'    DeepSeek {src[run][:4]} prefix {len(cs)} calls: NEW files {sorted(f - g_files)}')
        print(f'       symbols {sorted(s - g_syms)[:8]}')
        print(f'       searches {sorted(se)[:6]}')
        print(f'       runs ({len(ru)}): {ru[:5]}')
