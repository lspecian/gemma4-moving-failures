"""Measure locate.py's file/function accuracy against the gold patches of the public tasks.

    python3 scripts/eval_locate.py [--tasks experiments/gold_passing_52.txt] [--no-history] [--rebuild]
                                   [--locate path/to/other/locate.py]

Competition data is read from $COMPETITION_DATA (default: <repo>/data/competition; use the main checkout's data
dir from a worktree). Each task's .py files are extracted once into $TMPDIR/kb_trees/<id> together with a minimal
.git (HEAD at the snapshot's commit, objects shared with the newest snapshot of the same repo via alternates), so
the knowledge base sees the same history the sandbox has. KBs are cached in $TMPDIR/kb_eval/<id> (--rebuild
forces fresh builds; the build time is reported). Reports how often a gold file is ranked 1st / in the top
3/5/10, how often a gold function is in the top-5 file x top-3 function list, and how often locate's best match
(previewed only when confident) is a gold function.
`--locate` evaluates another locate.py (e.g. an old version from git) that has rank(root, text).
Beware of overfitting: the hidden tasks come from other (private) repositories.
"""

import argparse
import ast
import collections
import importlib.util
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(os.environ.get('COMPETITION_DATA', ROOT / 'data/competition'))
sys.dont_write_bytecode = True  # never leave .pyc files in agent/ (the harness rejects them)
sys.path.insert(0, str(ROOT / 'agent/skills/repo-tools/scripts'))
import kb  # noqa: E402
import locate  # noqa: E402

TMP = pathlib.Path(tempfile.gettempdir())
TREES = TMP / 'kb_trees'
GITS = TMP / 'kb_git'
KBS = TMP / 'kb_eval'


def git(cwd, *args):
    return subprocess.run(['git', '-C', str(cwd), *args], capture_output=True, text=True)


def shared_git(repo, tasks):
    """The .git of the newest snapshot of repo (its history contains every older snapshot's HEAD)."""
    dst = GITS / repo.replace('/', '__')
    if not (dst / 'objects').exists():
        latest = max((t for t in tasks if t['repo'] == repo), key=lambda t: t['created_at'])
        tmp = dst.with_suffix('.partial')
        shutil.rmtree(tmp, ignore_errors=True)
        with tarfile.open(DATA / f"snapshots/{latest['instance_id']}.tgz") as tf:
            members = [m for m in tf.getmembers() if m.name.startswith('./.git/') and '..' not in m.name]
            tf.extractall(tmp, members=members, filter='data')
        GITS.mkdir(exist_ok=True)
        (tmp / '.git').rename(dst)
        shutil.rmtree(tmp, ignore_errors=True)
    return dst


def _head_sha(gitdir):
    head = (gitdir / 'HEAD').read_text().strip()
    if not head.startswith('ref: '):
        return head
    ref = head[5:]
    if (gitdir / ref).exists():
        return (gitdir / ref).read_text().strip()
    for line in (gitdir / 'packed-refs').read_text().splitlines():
        if line.endswith(' ' + ref):
            return line.split()[0]
    raise RuntimeError(f'cannot resolve {ref} in {gitdir}')


def tree_for(task, tasks):
    iid = task['instance_id']
    dst = TREES / iid
    if (dst / '.git/HEAD').exists() and any(dst.rglob('*.py')):
        return dst
    shutil.rmtree(dst, ignore_errors=True)
    tmp = dst.with_suffix('.partial')
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(DATA / f'snapshots/{iid}.tgz') as tf:
        members = [m for m in tf.getmembers() if m.isfile() and not m.name.startswith('/') and '..' not in m.name
                   and ((m.name.endswith('.py') and '/.git/' not in m.name) or m.name in (
                       './.git/HEAD', './.git/packed-refs') or m.name.startswith('./.git/refs/'))]
        tf.extractall(tmp, members=members, filter='data')
    sha = _head_sha(tmp / '.git')
    shutil.rmtree(tmp / '.git')
    shared = shared_git(task['repo'], tasks)
    g = tmp / '.git'
    for d in ('objects/info', 'refs/heads', 'refs/tags'):
        (g / d).mkdir(parents=True)
    (g / 'HEAD').write_text(sha + '\n')
    (g / 'config').write_text('[core]\n\trepositoryformatversion = 0\n\tbare = false\n')
    (g / 'objects/info/alternates').write_text(str(shared / 'objects') + '\n')
    if git(tmp, 'cat-file', '-e', sha + '^{commit}').returncode != 0:  # not in the shared history: own objects
        shutil.rmtree(g)
        with tarfile.open(DATA / f'snapshots/{iid}.tgz') as tf:
            tf.extractall(tmp, members=[m for m in tf.getmembers() if m.name.startswith('./.git/')], filter='data')
    git(tmp, 'read-tree', sha)
    TREES.mkdir(exist_ok=True)
    tmp.rename(dst)
    return dst


def gold(task, root):
    """Gold file -> set of names of functions whose body the patch touches."""
    hunks, cur = {}, None
    for line in task['patch'].splitlines():
        m = re.match(r'^--- a/(\S+)', line)
        if m:
            cur = m.group(1)
            continue
        if line.startswith('--- /dev/null'):
            cur = None
            continue
        m = re.match(r'^@@ -(\d+),?(\d*)', line)
        if m and cur and cur.endswith('.py'):
            a = int(m.group(1))
            hunks.setdefault(cur, []).append((a, a + max(int(m.group(2) or 1), 1)))
    out = {}
    for f, hs in hunks.items():
        p = root / f
        if not p.exists():
            continue
        try:
            tree = ast.parse(p.read_text(errors='replace'))
        except SyntaxError:
            out[f] = set()
            continue
        out[f] = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  for a, b in hs if n.lineno <= b - 3 and n.end_lineno >= a + 3}
    return out


def load_module(path):
    spec = importlib.util.spec_from_file_location('locate_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', help='file with instance ids (default: all public tasks)')
    ap.add_argument('--no-history', action='store_true', help='disable the similar-commit signal')
    ap.add_argument('--rebuild', action='store_true', help='rebuild every KB (to measure build time)')
    ap.add_argument('--locate', help='evaluate this locate.py instead (old versions: rank(root, text))')
    ap.add_argument('-v', action='store_true', help='print one line per task')
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                    help='override a locate.py weight for an ablation, e.g. --set HISTORY_WEIGHT=0.5')
    args = ap.parse_args()
    for kv in args.set:
        name, value = kv.split('=', 1)
        setattr(locate, name, type(getattr(locate, name))(value))
    tasks = [json.loads(line) for line in open(DATA / 'tasks.jsonl')]
    run = tasks
    if args.tasks:
        keep = set(pathlib.Path(args.tasks).read_text().split())
        run = [t for t in tasks if t['instance_id'] in keep]
    other = load_module(args.locate) if args.locate else None
    rows, builds, ranks, t0 = [], [], [], time.time()
    for t in run:
        root = tree_for(t, tasks)
        g = gold(t, root)
        if not g:
            continue
        t1 = time.time()
        if other:
            ranked, best_fn, *_ = other.rank(str(root), t['problem_statement'])
        else:
            os.environ['REPO_TOOLS_DIR'] = str(KBS / t['instance_id'])
            db = KBS / t['instance_id'] / 'kb.sqlite'
            if args.rebuild and db.exists():
                db.unlink()
            fresh = not db.exists()
            conn = kb.open(str(root))
            if fresh:
                builds.append(time.time() - t1)
            t1 = time.time()
            ranked, best_fn, _, q, score = locate.rank(str(root), t['problem_statement'], conn=conn,
                                                       history=not args.no_history)
            best, confident = locate.pick_best(ranked, best_fn, q, locate.low_confidence(ranked, score))
            conn.close()
            kb._OPEN.clear()
        ranks.append(time.time() - t1)
        fns = {(p, f[1].split('.')[-1]) for p in ranked[:5] for _, f in best_fn[p][:3]}
        gfn = {(f, n) for f, ns in g.items() for n in ns}
        row = {k: any(f in ranked[:k] for f in g) for k in (1, 3, 5, 10)}
        row |= {'hasfn': bool(gfn), 'fn': bool(gfn & fns), 'repo': t['repo']}
        if not other:  # v18: the best match (previewed only when confident) is a gold function
            row['best'] = bool(best and (best[0], best[1].split('.')[-1]) in gfn)
            row['shown'] = bool(best and confident)
        rows.append(row)
        if args.v:
            pos = min((ranked.index(f) + 1 for f in g if f in ranked), default=0)
            print(f"{t['instance_id']:20s} gold@{pos or '-':<3} fn={'Y' if row['fn'] else 'N' if gfn else '-'} "
                  f"{sorted(g)[0]}  top={ranked[0]}")
    n = len(rows)
    print(f'{n} tasks ({len(run) - n} skipped: new-file-only gold), {time.time() - t0:.0f}s')
    print('file  ' + '  '.join(f'@{k}: {sum(r[k] for r in rows) / n:.1%}' for k in (1, 3, 5, 10)))
    fv = [r for r in rows if r['hasfn']]
    print(f'function in printed list: {sum(r["fn"] for r in fv) / len(fv):.1%} (n={len(fv)})')
    if not other:
        sh = [r for r in fv if r['shown']]
        print(f'best match is a gold function: {sum(r["best"] for r in fv) / len(fv):.1%}; previewed (confident) '
              f'{len(sh) / len(fv):.1%}, of which gold {sum(r["best"] for r in sh) / max(len(sh), 1):.1%}')
    per = collections.defaultdict(list)
    for r in rows:
        per[r['repo']].append(r)
    print('per repo @1/@5: ' + ', '.join(f'{repo.split("/")[1]} {sum(r[1] for r in rs) / len(rs):.0%}/'
                                        f'{sum(r[5] for r in rs) / len(rs):.0%} (n={len(rs)})'
                                        for repo, rs in sorted(per.items())))
    if builds:
        print(f'KB build: median {statistics.median(builds):.2f}s, max {max(builds):.2f}s (n={len(builds)})')
    print(f'rank: median {statistics.median(ranks):.2f}s, max {max(ranks):.2f}s')


if __name__ == '__main__':
    main()
