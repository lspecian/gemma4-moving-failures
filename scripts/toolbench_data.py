"""toolbench external tasks: select a SWE-bench "train" sample and build one workspace per instance (no model).

    .venv/bin/python scripts/toolbench_data.py select      # download the parquet once, write the selection
    .venv/bin/python scripts/toolbench_data.py setup [--workers 4] [--only ID,ID]   # clones + workspaces (cached)

select: princeton-nlp/SWE-bench, split train (public data, competition rules section 6), downloaded once into
data/toolbench/swebench/. Eligible instances: repository not in EXCLUDE_REPOS (fastapi = our local tasks; typeshed:
stubs; mxnet / spark-nlp: almost no Python), a problem statement of >= 40 characters, a gold patch whose non-test
files include 1-3 modified .py files and no added/deleted .py file (non-.py files such as news fragments are allowed and
ignored), at most MAX_CHANGED changed lines in those .py files. Selection: with random.Random(SEED), each repository's
eligible instances are shuffled and at most PER_REPO taken, round-robin over the repositories until N instances.
The instance ids go to experiments/toolbench_external.txt (with the criteria as # comments); the records (repo,
base_commit, patch, problem_statement, created_at) to data/toolbench/external.jsonl.

setup: one blob-less clone per repository (git clone --filter=blob:none --no-checkout, gc.auto=0) in
data/toolbench/repos/<owner>__<name>; per instance, the clone checks out base_commit (fetching only that tree's blobs),
the tree is copied (copy-on-write, `cp -c`) to data/toolbench/ws/<instance_id>, which gets its own .git sharing the
clone's objects (alternates) with HEAD = base_commit, so `git log` shows the instance's own history (locate's history
signal) and nothing after it. Then the harness's setup: .git/info/exclude, a `baseline` commit tagged
_swegemma_baseline (scripts/replay_tools.py prepare()). A ready workspace has .git/toolbench_ok. Instances of one
repository are set up sequentially (the clone's checkout is shared), repositories in parallel.
Skipped instances (clone failure, base commit missing, gold file missing at base_commit) are written with the reason to
experiments/toolbench_external_skipped.txt.
"""

import argparse
import collections
import concurrent.futures
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TB = os.path.join(ROOT, 'data', 'toolbench')
PARQUET = os.path.join(TB, 'swebench', 'data', 'train-00000-of-00001.parquet')
RECORDS = os.path.join(TB, 'external.jsonl')
REPOS = os.path.join(TB, 'repos')
WS = os.path.join(TB, 'ws')
SELECTION = os.path.join(ROOT, 'experiments', 'toolbench_external.txt')
SKIPPED = os.path.join(ROOT, 'experiments', 'toolbench_external_skipped.txt')
SEED, N, PER_REPO, MAX_CHANGED = 0, 300, 10, 200
EXCLUDE_REPOS = {'tiangolo/fastapi', 'python/typeshed', 'apache/mxnet', 'JohnSnowLabs/spark-nlp'}
TEST_PATH = re.compile(r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$')
BASELINE = '_swegemma_baseline'
GIT_EXCLUDE = '__pycache__/\n*.pyc\n.pytest_cache/\n*.egg-info/\nbuild/\ndist/\n.coverage\n.adk_exec_*.py\n'
GIT_ENV = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0', GIT_AUTHOR_NAME='toolbench',
               GIT_AUTHOR_EMAIL='toolbench@local', GIT_COMMITTER_NAME='toolbench',
               GIT_COMMITTER_EMAIL='toolbench@local')


def patch_files(patch):
    """[(path, kind)] of a unified git diff: kind added | deleted | modified."""
    out = []
    for m in re.finditer(r'^diff --git a/(\S+) b/(\S+)\n((?:(?!diff --git).*\n){0,6})', patch, re.M):
        head = m.group(3)
        kind = 'added' if 'new file mode' in head or '--- /dev/null' in head else \
            'deleted' if 'deleted file mode' in head or '+++ /dev/null' in head else 'modified'
        out.append((m.group(2), kind))
    return out


def changed_lines(patch, files):
    n, cur = 0, None
    for line in patch.splitlines():
        if line.startswith('diff --git'):
            cur = line.split(' b/', 1)[-1]
        elif cur in files and line[:1] in '+-' and not line.startswith(('+++', '---')):
            n += 1
    return n


def eligible(r):
    """None when the instance qualifies, else the reason."""
    if r['repo'] in EXCLUDE_REPOS:
        return 'repo excluded'
    if len((r['problem_statement'] or '').strip()) < 40:
        return 'short statement'
    files = patch_files(r['patch'])
    src = [(p, k) for p, k in files if not TEST_PATH.search(p)]
    py = [(p, k) for p, k in src if p.endswith('.py')]
    if any(k != 'modified' for _, k in py):
        return 'adds or deletes a .py file'
    if not 1 <= len(py) <= 3:
        return f'{len(py)} non-test .py files'
    if changed_lines(r['patch'], {p for p, _ in py}) > MAX_CHANGED:
        return 'too many changed lines'
    return None


def select():
    if not os.path.exists(PARQUET):
        from huggingface_hub import hf_hub_download
        hf_hub_download('princeton-nlp/SWE-bench', 'data/train-00000-of-00001.parquet', repo_type='dataset',
                        local_dir=os.path.join(TB, 'swebench'))
    import pyarrow.parquet as pq
    rows = pq.read_table(PARQUET, columns=['repo', 'instance_id', 'base_commit', 'patch', 'problem_statement',
                                           'created_at']).to_pylist()
    by_repo = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r['instance_id']):
        if eligible(r) is None:
            by_repo[r['repo']].append(r)
    rng = random.Random(SEED)
    queues = {}
    for repo in sorted(by_repo):
        rs = by_repo[repo][:]
        rng.shuffle(rs)
        queues[repo] = rs[:PER_REPO]
    chosen = []
    while len(chosen) < N and any(queues.values()):
        for repo in sorted(queues):
            if queues[repo] and len(chosen) < N:
                chosen.append(queues[repo].pop(0))
    chosen.sort(key=lambda r: r['instance_id'])
    os.makedirs(TB, exist_ok=True)
    with open(RECORDS, 'w') as f:
        for r in chosen:
            f.write(json.dumps({k: r[k] for k in ('instance_id', 'repo', 'base_commit', 'patch', 'problem_statement',
                                                  'created_at')}) + '\n')
    counts = collections.Counter(r['repo'] for r in chosen)
    with open(SELECTION, 'w') as f:
        f.write(f'# toolbench external tasks: SWE-bench train (princeton-nlp/SWE-bench), seed {SEED}, {len(chosen)} '
                f'instances from {len(counts)} repositories (<= {PER_REPO} each)\n'
                f'# eligible: not {sorted(EXCLUDE_REPOS)}; statement >= 40 chars; gold changes 1-3 existing non-test '
                f'.py files, <= {MAX_CHANGED} changed lines, no added/deleted .py (scripts/toolbench_data.py)\n')
        f.writelines(r['instance_id'] + '\n' for r in chosen)
    print(f'{len(chosen)} instances, {len(counts)} repos: ' + ', '.join(f'{k} {v}' for k, v in sorted(counts.items())))


def load_records(ids=None):
    recs = [json.loads(l) for l in open(RECORDS) if l.strip()]
    keep = [l.strip() for l in open(SELECTION) if l.strip() and not l.startswith('#')]
    recs = {r['instance_id']: r for r in recs}
    out = [recs[i] for i in keep if i in recs]
    return [r for r in out if not ids or r['instance_id'] in ids]


def git(cwd, *a, check=True, timeout=600):
    r = subprocess.run(['git', '-c', 'core.quotepath=false', '-c', 'commit.gpgsign=false', '-c', 'filter.lfs.smudge=',
                        '-c', 'filter.lfs.process=', '-c', 'filter.lfs.required=false', '-c', 'advice.detachedHead=false',
                        '-C', cwd, *a], capture_output=True, text=True, errors='replace', env=GIT_ENV, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f'git {" ".join(a[:3])}: {r.stderr.strip()[-300:]}')
    return r.stdout


def mirror_dir(repo):
    return os.path.join(REPOS, repo.replace('/', '__'))


def ensure_mirror(repo):
    d = mirror_dir(repo)
    if os.path.isdir(os.path.join(d, '.git')):
        return d
    tmp = d + '.partial'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(REPOS, exist_ok=True)
    r = subprocess.run(['git', 'clone', '-q', '--filter=blob:none', '--no-checkout', f'https://github.com/{repo}.git',
                        tmp], capture_output=True, text=True, env=GIT_ENV, timeout=3600)
    if r.returncode:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f'clone failed: {r.stderr.strip()[-300:]}')
    git(tmp, 'config', 'gc.auto', '0')
    git(tmp, 'config', 'maintenance.auto', 'false')
    os.rename(tmp, d)
    return d


def ws_ready(iid):
    return os.path.exists(os.path.join(WS, iid, '.git', 'toolbench_ok'))


def make_ws(rec, mirror):
    """data/toolbench/ws/<id> at base_commit (see the module docstring). Returns None or the skip reason."""
    iid, sha = rec['instance_id'], rec['base_commit']
    ws = os.path.join(WS, iid)
    if ws_ready(iid):
        return None
    if git(mirror, 'cat-file', '-t', sha, check=False).strip() != 'commit':
        git(mirror, 'fetch', '-q', 'origin', sha, check=False)
        if git(mirror, 'cat-file', '-t', sha, check=False).strip() != 'commit':
            return 'base_commit not in the repository'
    try:
        git(mirror, 'checkout', '-q', '-f', '--detach', sha)
    except RuntimeError as e:
        return f'checkout failed: {e}'
    for p, kind in patch_files(rec['patch']):
        if kind == 'modified' and p.endswith('.py') and not TEST_PATH.search(p) and \
                not os.path.isfile(os.path.join(mirror, p)):
            return f'gold file missing at base_commit: {p}'
    shutil.rmtree(ws, ignore_errors=True)
    os.makedirs(ws)
    for name in os.listdir(mirror):
        if name == '.git':
            continue
        subprocess.run(['cp', '-cRP', os.path.join(mirror, name), ws], check=True, capture_output=True)
    git(ws, 'init', '-q')
    with open(os.path.join(ws, '.git', 'objects', 'info', 'alternates'), 'w') as f:
        f.write(os.path.join(mirror, '.git', 'objects') + '\n')
    git(ws, 'update-ref', 'refs/heads/main', sha)
    git(ws, 'symbolic-ref', 'HEAD', 'refs/heads/main')
    git(ws, 'read-tree', 'HEAD')
    git(ws, 'update-index', '-q', '--refresh', check=False)
    os.makedirs(os.path.join(ws, '.git', 'info'), exist_ok=True)
    with open(os.path.join(ws, '.git', 'info', 'exclude'), 'a') as f:
        f.write(GIT_EXCLUDE)
    git(ws, 'add', '-A')
    git(ws, 'commit', '-q', '-m', 'baseline', '--allow-empty', '--no-verify')
    git(ws, 'tag', '-f', BASELINE)
    open(os.path.join(ws, '.git', 'toolbench_ok'), 'w').write(f'{iid} {sha}\n')
    return None


def setup_repo(job):
    repo, recs = job
    t0, skipped = time.time(), []
    try:
        mirror = ensure_mirror(repo)
    except Exception as e:  # noqa: BLE001
        return repo, [(r['instance_id'], str(e)[:200]) for r in recs if not ws_ready(r['instance_id'])], time.time() - t0
    for r in recs:
        try:
            why = make_ws(r, mirror)
        except Exception as e:  # noqa: BLE001
            why = f'setup error: {e}'[:300]
            shutil.rmtree(os.path.join(WS, r['instance_id']), ignore_errors=True)
        if why:
            skipped.append((r['instance_id'], why))
    return repo, skipped, time.time() - t0


def setup(workers, only=None):
    recs = load_records(only)
    by_repo = collections.defaultdict(list)
    for r in recs:
        by_repo[r['repo']].append(r)
    jobs = sorted(by_repo.items(), key=lambda kv: -len(kv[1]))
    old = {}
    if os.path.exists(SKIPPED) and only:
        old = {l.split()[0]: l.strip() for l in open(SKIPPED) if l.strip() and not l.startswith('#')}
    skipped, t0 = [], time.time()
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        for repo, sk, secs in ex.map(setup_repo, jobs):
            skipped += sk
            print(f'{repo}: {len(by_repo[repo]) - len(sk)}/{len(by_repo[repo])} ready, {secs:.0f}s'
                  + (f'; skipped {sk}' if sk else ''), flush=True)
    for iid, _ in skipped:
        old.pop(iid, None)
    for r in recs:  # set up now: no longer skipped
        if ws_ready(r['instance_id']):
            old.pop(r['instance_id'], None)
    lines = sorted(list(old.values()) + [f'{iid}  {why}' for iid, why in skipped])
    with open(SKIPPED, 'w') as f:
        f.write('# toolbench external instances that could not be set up (scripts/toolbench_data.py setup)\n')
        f.writelines(l + '\n' for l in lines)
    print(f'setup done in {time.time() - t0:.0f}s: {sum(ws_ready(r["instance_id"]) for r in recs)}/{len(recs)} ready, '
          f'{len(lines)} skipped')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=('select', 'setup'))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--only', default=None, help='comma-separated instance ids (setup)')
    a = ap.parse_args()
    if a.cmd == 'select':
        select()
    else:
        setup(a.workers, set(a.only.split(',')) if a.only else None)


if __name__ == '__main__':
    main()
