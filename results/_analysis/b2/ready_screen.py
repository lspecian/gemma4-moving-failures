"""B2 screen: check_patch READY rate on gold edits in external workspaces, v24 vs v25 (working tree). Applies each
task's gold patch with `git apply`, runs both check_patch versions (fresh flow state), then reverts with
`git apply -R`. Bare interpreter: third-party imports fail, which is the point (v25 reports them as environment notes)."""
import json, os, re, subprocess, sys, tempfile, collections
ROOT = os.getcwd()
VERS = {'v24': 'results/_analysis/b2/v24/agent/skills/repo-tools/scripts', 'v25': 'results/_analysis/b2/sub_v25/skills/repo-tools/scripts'}
rows = [json.loads(l) for l in open('data/toolbench/external.jsonl')]
mis = set(l.split()[0] for l in open('results/_analysis/b2/mismatch_tasks.txt')) if os.path.exists('results/_analysis/b2/mismatch_tasks.txt') else set()
pick = [r for r in rows if r['instance_id'] in mis][:25] + [r for r in rows if r['instance_id'] not in mis][:15]
out = collections.Counter(); notes = []
for r in pick:
    ws = os.path.join('data/toolbench/ws', r['instance_id'])
    if not os.path.isdir(ws): continue
    patch = re.sub(r'(?ms)^diff --git a/(\S+) .*?(?=^diff --git |\Z)', lambda m: '' if re.search(r'(^|/)(tests?|testing)/|test_[^/]*\.py$|conftest\.py$', m.group(1)) else m.group(0), r['patch'])
    pf = tempfile.NamedTemporaryFile('w', suffix='.patch', delete=False); pf.write(patch); pf.close()
    if subprocess.run(['git', '-C', ws, 'apply', pf.name], capture_output=True).returncode:
        out['apply failed'] += 1; continue
    try:
        res = {}
        for v, d in VERS.items():
            with tempfile.TemporaryDirectory() as st:
                p = subprocess.run([sys.executable, os.path.join(ROOT, d, 'check_patch.py')], cwd=tempfile.gettempdir(),
                                   capture_output=True, text=True, timeout=300,
                                   env=dict(os.environ, WORKSPACE=os.path.abspath(ws), REPO_TOOLS_DIR=st, PYTHONDONTWRITEBYTECODE='1'))
            res[v] = 'READY' if re.search(r'^READY', p.stdout, re.M) else ('PROBLEMS' if 'PROBLEMS' in p.stdout else 'other')
            if v == 'v25' and 'NOTE:' in p.stdout and len(notes) < 5:
                notes.append((r['instance_id'], [l for l in p.stdout.splitlines() if l.startswith('NOTE:')][:1]))
        grp = 'monorepo' if r['instance_id'] in mis else 'other'
        out[(grp, res['v24'], res['v25'])] += 1
    finally:
        subprocess.run(['git', '-C', ws, 'apply', '-R', pf.name], capture_output=True); os.unlink(pf.name)
for k, v in sorted(out.items(), key=str): print(k, v)
for n in notes: print(' ', n)
