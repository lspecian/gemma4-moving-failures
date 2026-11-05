"""Solvable-but-missed tasks: solved by some DeepSeek run, never by a Gemma run. For each, what the successful DeepSeek
trajectory read / ran / edited vs what Gemma's runs did on the same task (gold files from tasks.jsonl).

    python3 results/_analysis/ceiling/solvable.py [deepseek runs...]
"""
import ast, glob, json, re, sys, collections
GEMMA = ['E069_v22_real', 'E070_v22_real', 'E073_v24_real', 'E075_neb_v24_base']
DS = sys.argv[1:] or ['E082_ds_v4pro_eval113', 'E083_flash_v24_paced', 'E084_flash_v24_generous', 'E085_flash_bundle_generous']
tasks = {json.loads(l)['instance_id']: json.loads(l) for l in open('data/competition/tasks.jsonl')}
def gold_files(t):
    return sorted(set(re.findall(r'^(?:diff --git a/|\+\+\+ b/)(\S+)', tasks[t]['patch'], re.M)))
def results(run):
    out = {}
    for f in glob.glob(f'results/{run}/r1/*/task_results.jsonl'):
        for l in open(f):
            r = json.loads(l); out[r['instance_id']] = r
    return out
def trace(run, t):
    fs = glob.glob(f'results/{run}/r1/*/traces/trace_{t}.json')
    if not fs: return []
    calls = []
    for s in json.load(open(fs[0]))['steps']:
        tc = s.get('tool_calls')
        if not tc: continue
        for c in (ast.literal_eval(tc) if isinstance(tc, str) else tc):
            a = c.get('arguments') or {}
            name = a.get('file_path') if c['function_name'] == 'run_skill_script' else c['function_name']
            args = a.get('args') if c['function_name'] == 'run_skill_script' else a
            calls.append((name, args if isinstance(args, dict) else {}))
    return calls
def touched(calls, kinds):
    out = set()
    for name, a in calls:
        if name in kinds:
            for k in ('target', 'path', 'file_path', 'filepath'):
                if isinstance(a.get(k), str): out.add(a[k].lstrip('/').replace('workspace/', ''))
    return out
READ = {'show.py', 'read_file', 'search.py'}; EDIT = {'edit.py', 'create.py', 'edit_file', 'write_file'}
R = {r: results(r) for r in GEMMA + DS}
gem_solved = {t for r in GEMMA for t, x in R[r].items() if x['resolved']}
rows = []
for t in sorted(tasks):
    ds_wins = [r for r in DS if R[r].get(t, {}).get('resolved')]
    if not ds_wins or t in gem_solved: continue
    g = gold_files(t); best = ds_wins[0]; dcalls = trace(best, t)
    gem = []
    for r in GEMMA:
        if t not in R[r]: continue
        c = trace(r, t); x = R[r][t]
        gem.append({'run': r, 'outcome': 'wrong' if x['agent_patch_size'] else 'empty', 'calls': len(c),
                    'read_gold': bool(set(g) & touched(c, READ)), 'edited_gold': bool(set(g) & touched(c, EDIT)),
                    'timeout': bool(x.get('error') and 'timeout' in x['error'])})
    rows.append({'task': t, 'ds_wins': ds_wins, 'gold_files': g, 'ds_calls': len(dcalls),
                 'ds_tools': collections.Counter(n for n, _ in dcalls).most_common(5),
                 'ds_ran_tests': sum(1 for n, a in dcalls if n in ('run.py', 'tests_for.py', 'run_command')), 'gemma': gem})
print(f'solved by DeepSeek ({", ".join(DS)}) but never by Gemma ({len(gem_solved)} solved by Gemma): {len(rows)}')
for x in rows:
    gs = '; '.join(f"{g['run'][:4]} {g['outcome']}{' TO' if g['timeout'] else ''} read={int(g['read_gold'])} edit={int(g['edited_gold'])}" for g in x['gemma'])
    print(f"- {x['task']}: DS {','.join(r[:4] for r in x['ds_wins'])} ({x['ds_calls']} calls, tests {x['ds_ran_tests']}); gold {len(x['gold_files'])} files | Gemma: {gs}")
json.dump(rows, open('results/_analysis/ceiling/solvable.json', 'w'), indent=1)
