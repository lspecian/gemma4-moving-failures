"""Option 2, step 1 (LLM-free, recorded data only): do the reproductions the agent ran before its first edit already
point at the gold file? From each pre-edit run.py output, take the repository frames of tracebacks (File
"/workspace/<path>", ...; tests, site-packages and the snippet itself excluded) and ask: is a gold file among them, in
particular when locate.py's first file was not a gold file? Runs: E076/E077 (unseen100), E075 (eval113)."""
import json, glob, re, collections
FRAME = re.compile(r'File "(?:/workspace/)?([^"]+\.py)", line \d+, in (\S+)')
TEST = re.compile(r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$|site-packages|snippet\.py|^/|^<')
def gold_of(run, t):
    if 'unseen' in run:
        m = META[t]; return [g for g in m['gold_files'] if g.endswith('.py')]
    return [g for g in re.findall(r'^\+\+\+ b/(\S+\.py)$', PUB[t]['patch'], re.M) if not TEST.search(g)]
META = {json.loads(l)['instance_id']: json.loads(l) for l in open('data/unseen/meta.jsonl')}
PUB = {json.loads(l)['instance_id']: json.loads(l) for l in open('data/competition/tasks.jsonl')}
def stdout(obs):
    try: return json.loads(obs).get('stdout', '') or ''
    except Exception: return obs.replace('\\n', '\n')
for run in ('E076_neb_v24_unseen', 'E077_neb_v25_unseen', 'E075_neb_v24_base'):
    st = collections.Counter()
    for tf in glob.glob(f'results/{run}/r1/c*/traces/trace_*.json'):
        t = tf.split('trace_')[1][:-5]; gold = gold_of(run, t)
        if not gold: continue
        loc1 = None; frames = []; pre = True; nrun = 0
        for s in json.load(open(tf))['steps']:
            obs = str((s.get('observation') or {}).get('content') or '')
            for tc in s.get('tool_calls') or []:
                sc = str((tc.get('arguments') or {}).get('file_path') or tc.get('function_name'))
                if sc == 'locate.py' and loc1 is None:
                    m = re.search(r'^\s*\d+\.\d+\s+(\S+)', stdout(obs), re.M); loc1 = m.group(1) if m else ''
                if sc == 'edit.py' and 'EDITED' in obs: pre = False
                if sc == 'run.py' and pre:
                    nrun += 1
                    frames += [(f, fn) for f, fn in FRAME.findall(stdout(obs)) if not TEST.search(f)]
        files = list(dict.fromkeys(f for f, _ in frames))
        miss = loc1 not in gold
        st['tasks'] += 1; st['locate #1 missed'] += miss
        st['pre-edit repro run'] += nrun > 0
        st['repro traceback with repo frames'] += bool(files)
        st['gold file in a repro traceback'] += bool(set(files) & set(gold))
        if miss:
            st['  of locate misses: repro run'] += nrun > 0
            st['  of locate misses: gold in traceback'] += bool(set(files) & set(gold))
    print(run, dict(st))
