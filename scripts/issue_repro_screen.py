"""LLM-free screen: does running the issue's own code show the bug? (research/notes/23; LOG 2026-10-15)

The handoff experiments (E088-E092) showed that what Gemma lacks is the outcome of experiments that expose the
failure, and that it rarely runs such experiments itself. If the harness ran the issue's own snippet at the start of a
task and showed the result, that evidence would cost no model call and no judgement. This script measures whether it
would exist:
  1. extract Python snippets from the problem statement (fenced blocks tagged python/py/pycon or untagged, and >>>
     sessions; shell commands, tracebacks and plain output are skipped; the snippet must parse);
  2. run each in the task's sandbox (snapshot + per-task environment, offline, 20 s) at the base commit, then again
     after applying the gold patch;
  3. a snippet is EVIDENCE when its output (stdout + the exception line) differs between base and gold.

    .venv/bin/python scripts/issue_repro_screen.py --tasks data/competition/tasks.jsonl --list experiments/eval113.txt \
        --out results/_analysis/issue_repro/eval113.jsonl
    .venv/bin/python scripts/issue_repro_screen.py --tasks data/unseen/tasks.jsonl --list experiments/unseen100.txt \
        --snapshots data/unseen/snapshots --manifest data/unseen/task_env.json --out results/_analysis/issue_repro/unseen100.jsonl
"""

import argparse
import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

FENCE = re.compile(r'```[ \t]*([\w+-]*)[^\n]*\n(.*?)```', re.S)
PY_TAGS = {'', 'python', 'py', 'python3', 'pycon', 'pytb-no'}
SHELL = re.compile(r'^\s*(\$ |pip |git |python -m |python3 -m |uv |poetry |conda |cd |export |curl |>>> import os; os\.system)')
TIMEOUT = 20


def snippets(text):
    """Python snippets of an issue: [(kind, code)]."""
    out = []
    for tag, body in FENCE.findall(text or ''):
        tag = tag.lower()
        if tag not in PY_TAGS:
            continue
        body = body.replace('\r', '')
        if 'Traceback (most recent call last)' in body.splitlines()[0:1] or body.lstrip().startswith('Traceback'):
            continue
        if '>>>' in body:  # a session: keep the prompt lines only
            lines = [l[4:] if l.startswith(('>>> ', '... ')) else (l[3:] if l in ('>>>', '...') else None)
                     for l in body.splitlines()]
            code = '\n'.join(l for l in lines if l is not None)
            kind = 'session'
        else:
            if any(SHELL.match(l) for l in body.splitlines()[:3]):
                continue
            code, kind = textwrap.dedent(body), 'block'
        code = code.strip('\n')
        if not code or 'uvicorn.run' in code or 'input(' in code:
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        if not any(isinstance(n, ast.Call) for n in ast.walk(tree)):
            continue  # a definition or a data blob: nothing runs
        out.append((kind, code))
    if not out:  # a bare >>> session outside fences
        lines = [l.strip() for l in (text or '').splitlines()]
        sess = [l[4:] for l in lines if l.startswith('>>> ')]
        if sess:
            code = '\n'.join(sess)
            try:
                ast.parse(code)
                out.append(('session', code))
            except SyntaxError:
                pass
    return out


ADDR = re.compile(r' at 0x[0-9a-f]+|\d+\.\d+s\b|/tmp/[^\s\'"]+')


def norm(s):
    return ADDR.sub('', s or '').strip()


def run_snippet(code, ws, env_path):
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(code)
        path = f.name
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join(p for p in (env_path, ws, os.path.join(ws, 'src')) if p)
    try:
        p = subprocess.run([sys.executable, path], cwd=ws, capture_output=True, text=True, timeout=TIMEOUT, env=env)
        err_lines = [l for l in p.stderr.strip().splitlines() if l and not l.startswith(' ')]
        last_err = err_lines[-1] if err_lines else ''
        status = 'ok' if p.returncode == 0 else 'error'
        return {'status': status, 'out': p.stdout[-1500:], 'err': last_err[-300:]}
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'out': '', 'err': ''}
    finally:
        os.unlink(path)


ENV_FAIL = re.compile(r"^(ModuleNotFoundError|ImportError): .*(No module named|cannot import name)")


def inside(a):
    import replay_tools as rt
    tasks = {json.loads(l)['instance_id']: json.loads(l) for l in open(a.tasks)}
    ids = [x.strip() for x in open(a.list) if x.strip() and not x.startswith('#')]
    with open(a.out, 'w') as out:
        for tid in ids:
            t = tasks[tid]
            sn = snippets(t['problem_statement'])
            rec = {'task': tid, 'n_snippets': len(sn), 'snippets': []}
            if sn:
                work = tempfile.mkdtemp(prefix='irs_')
                ws = os.path.join(work, 'ws')
                try:
                    rt.prepare(os.path.join(a.snapshots, f'{tid}.tgz'), ws)
                    env_path = rt.task_env_path(tid, work, a.manifest, a.spcache)
                    base = [run_snippet(code, ws, env_path) for _, code in sn]
                    with tempfile.NamedTemporaryFile('w', suffix='.patch', delete=False) as pf:
                        pf.write(t['patch'].rstrip('\n') + '\n')  # the task patches lack the final newline
                    ap = subprocess.run(['git', '-C', ws, 'apply', '--whitespace=nowarn', '--recount', pf.name],
                                        capture_output=True)
                    os.unlink(pf.name)
                    gold = [run_snippet(code, ws, env_path) for _, code in sn] if ap.returncode == 0 else [None] * len(sn)
                    for (kind, code), b, g in zip(sn, base, gold):
                        envfail = bool(ENV_FAIL.match(b['err'])) and (g is None or bool(ENV_FAIL.match(g['err'] or '')))
                        differs = g is not None and (b['status'], norm(b['out']), norm(b['err'])) != \
                            (g['status'], norm(g['out']), norm(g['err']))
                        rec['snippets'].append({'kind': kind, 'lines': code.count('\n') + 1, 'base': b, 'gold': g,
                                                'env_fail': envfail, 'evidence': differs})
                finally:
                    shutil.rmtree(work, ignore_errors=True)
            out.write(json.dumps(rec) + '\n')
            out.flush()
            ev = sum(s['evidence'] for s in rec['snippets'])
            print(f"{tid:50} snippets {len(sn)} evidence {ev}", flush=True)


def summarize(path):
    recs = [json.loads(l) for l in open(path)]
    with_sn = [r for r in recs if r['n_snippets']]
    ev = [r for r in recs if any(s['evidence'] for s in r['snippets'])]
    base_err = [r for r in with_sn if any(s['base']['status'] == 'error' and not s['env_fail'] for s in r['snippets'])]
    envf = [r for r in with_sn if all(s['env_fail'] for s in r['snippets'])]
    print(f'{len(recs)} tasks; with a runnable snippet {len(with_sn)}; evidence (output changes with the gold patch) '
          f'{len(ev)}; a snippet raising at base (not an import failure) {len(base_err)}; only import failures '
          f'{len(envf)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--list', required=True)
    ap.add_argument('--snapshots', default='data/competition/snapshots')
    ap.add_argument('--manifest', default='data/competition/wheels_local/task_env.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--spcache', default=None)
    a = ap.parse_args()
    if a.inside:
        return inside(a)
    import replay_tools as rt
    spcache = a.spcache or rt.SP_CACHE
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rel = lambda p: '/repo/' + os.path.relpath(os.path.abspath(p), ROOT)  # noqa: E731
    inner = ['python', '/repo/scripts/issue_repro_screen.py', '--inside', '--tasks', rel(a.tasks), '--list', rel(a.list),
             '--snapshots', rel(a.snapshots), '--manifest', rel(a.manifest), '--out', '/out/' + os.path.basename(a.out),
             '--spcache', '/spcache']
    setup = f'tar --no-same-owner -xf /spcache/sp_base.tar -C {rt.SP_DIR} && ' \
        if os.path.exists(os.path.join(spcache, 'sp_base.tar')) else ''
    cmd = ['docker', 'run', '--rm', '--platform', 'linux/amd64', '--network', 'none', '-v', f'{ROOT}:/repo:ro',
           '-v', f'{os.path.dirname(os.path.abspath(a.out))}:/out', '-v', f'{spcache}:/spcache:ro', '-w', '/tmp',
           '-e', 'PYTHONDONTWRITEBYTECODE=1', rt.IMAGE, 'bash', '-c', setup + ' '.join(shlex.quote(x) for x in inner)]
    subprocess.run(cmd, check=True)
    summarize(a.out)


if __name__ == '__main__':
    main()
