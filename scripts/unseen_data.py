"""unseen100: SWE-style tasks from ~100 GitHub repositories we never tuned on, built into OUR harness (like eval113).

    .venv/bin/python scripts/unseen_data.py select                 # pool -> 200 candidates (one per repository)
    .venv/bin/python scripts/unseen_data.py snapshot [--workers 4] [--only ID,ID] [--first N]
    .venv/bin/python scripts/unseen_data.py tasks                  # data/unseen/tasks.jsonl + meta.jsonl
    .venv/bin/python scripts/task_env.py build --tasks data/unseen/tasks.jsonl --snapshots data/unseen/snapshots \
        --manifest data/unseen/task_env.json --hints data/unseen/meta.jsonl --generic
    .venv/bin/python scripts/verify_gold.py --manifest data/unseen/task_env.json --tasks data/unseen/tasks.jsonl \
        --snapshots-dir data/unseen/snapshots --submission-dir agent --results-dir results/unseen_gold/final ...
    (the same with --skip-agent-patch on the gold passes -> results/unseen_empty/final)
    .venv/bin/python scripts/unseen_data.py report                 # per-task table + failure reasons
    .venv/bin/python scripts/unseen_data.py finalize               # experiments/unseen100.txt + funnel.json
Details and results: research/notes/21-unseen100.md.

Sources (downloaded by W4 into results/_analysis/W4/, research/notes/20-parts/W4-data-sources.md): SWE-rebench-
leaderboard (lb.pkl), SWE-bench-Live full (live_full.parquet), SWE-rebench V2 Python (rebench_v2.parquet).

select: Python tasks created >= 2025-02-01 (Gemma 4 cutoff January 2025), repository not one of our 4 eval or 31
tuning repos (data/toolbench/external.jsonl), gold patch touching 1-3 non-test .py files with <= 200 changed lines in
them, problem statement >= 150 characters with >= 50 characters outside URLs, a test_patch touching >= 1 .py file,
FAIL_TO_PASS / PASS_TO_PASS restricted to the files of the test_patch (the harness runs only those files) with >= 1
FAIL_TO_PASS left (PASS_TO_PASS is then emptied in tasks.jsonl, see write_tasks). Same instance in several sources: leaderboard > Live > V2. Then one task per repository (GitHub
canonical name, so renamed repositories count once): random.Random(SEED) among the repository's tasks, preferring
tasks whose dataset environment used Python >= 3.11. Repositories are ranked in tiers (random order within a tier):
A = pure Python (compiled-language bytes < 3% per the GitHub languages API) and repository <= 500 MB, and the task's
Python config >= 3.11 or unknown; B = pure Python with an older config; C = the rest. The first N_CANDIDATES go to
results/_analysis/unseen/candidates.jsonl. Layout is NOT a criterion (it is recorded after the snapshot is built).

snapshot: per candidate, a shallow fetch of base_commit (depth 2,500, then 500 or 100 while the tgz is > 250 MB; kb.py reads <= 2,000 commits) into a
scratch bare repo, `git fast-export | git fast-import` into a fresh repository (no remote, no tags, nothing after
base_commit), branches main + export_ref at the imported base commit, user Agent <agent@eval>, checked out, then
tar.gz of `./` incl. .git -> data/unseen/snapshots/<instance_id>.tgz, the format of data/competition/snapshots
(the harness extracts with tar --no-same-owner). Scratch under results/_analysis/unseen/git/ is deleted per task.
The snapshot's tree is classified on the way (layout, project root, runner hints, size, Python versions, compiled
code) into results/_analysis/unseen/snapmeta/<id>.json.

tasks: data/unseen/tasks.jsonl (harness schema) and data/unseen/meta.jsonl (source, install hints for task_env.py,
layout class, runner hints, repo size) for every candidate with a snapshot; symlinks data/unseen/wheels and
data/unseen/sandbox -> ../competition/ (the harness looks for the published wheels and sandbox/setup.py next to the
tasks file).

finalize: reads the gold and empty-patch verification runs, keeps tasks that pass with the gold patch AND fail with
an empty patch, writes experiments/unseen100.txt (random order, seed SEED) and results/_analysis/unseen/funnel.json.
"""

import argparse
import collections
import concurrent.futures
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from toolbench_data import TEST_PATH, changed_lines, patch_files  # noqa: E402

W4 = os.path.join(ROOT, 'results', '_analysis', 'W4')
SCRATCH = os.path.join(ROOT, 'results', '_analysis', 'unseen')
OUT = os.path.join(ROOT, 'data', 'unseen')
SNAPS = os.path.join(OUT, 'snapshots')
CANDIDATES = os.path.join(SCRATCH, 'candidates.jsonl')
POOL_STATS = os.path.join(SCRATCH, 'select_stats.json')
GH_META = os.path.join(SCRATCH, 'gh_meta.json')
HEAVY = os.path.join(SCRATCH, 'heavy_deps.json')
SNAPMETA = os.path.join(SCRATCH, 'snapmeta')
SEED, N_CANDIDATES, CUTOFF = 0, 200, '2025-02-01'  # 150 planned; 200 after the first pass yielded 51%
DEPTHS, MAX_SNAPSHOT_MB = (2500, 500, 100), 250  # history depth; kb.py reads <= 2,000 commits
MAX_PY_FILES, MAX_CHANGED, MIN_STATEMENT, MIN_TEXT = 3, 200, 150, 50
EVAL_REPOS = {'fastapi/fastapi', 'tiangolo/fastapi', 'textualize/rich', 'psf/requests', 'encode/httpx'}
COMPILED = {'C', 'C++', 'Cython', 'Rust', 'Fortran', 'Cuda', 'Go', 'Zig', 'Objective-C', 'Assembly', 'Meson'}
URL = re.compile(r'https?://\S+')
GIT_ENV = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0', GIT_AUTHOR_NAME='Agent',
               GIT_AUTHOR_EMAIL='agent@eval', GIT_COMMITTER_NAME='Agent', GIT_COMMITTER_EMAIL='agent@eval',
               GIT_CONFIG_NOSYSTEM='1')


def iso(ts):
    """'2025-06-13 10:18:30' / Timestamp / '2025-06-13T10:18:30Z' -> '2025-06-13T10:18:30Z'."""
    s = str(ts).strip().replace(' ', 'T')
    s = re.sub(r'(\.\d+)?(\+00:00|Z)?$', '', s)
    return s + 'Z'


def as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        v = v.strip()
        if v.startswith('['):
            try:
                return [str(x) for x in json.loads(v)]
            except Exception:
                return []
        return [v] if v else []
    try:
        return [str(x) for x in list(v)]
    except TypeError:
        return []


def test_files_of(test_patch):
    from swegemma.models.task import extract_test_files_from_patch
    return extract_test_files_from_patch(test_patch)


# ---------------------------------------------------------------- select
def load_sources():
    import pandas as pd
    import pyarrow.parquet as pq
    rows = []
    lb = pd.read_pickle(os.path.join(W4, 'lb.pkl'))
    for r in lb.to_dict('records'):
        ic = r.get('install_config') or {}
        rows.append(dict(source='swe-rebench-leaderboard', instance_id=r['instance_id'], repo=r['repo'],
                         base_commit=r['base_commit'], problem_statement=r['problem_statement'] or '',
                         patch=r['patch'], test_patch=r['test_patch'], hints_text=r.get('hints_text') or '',
                         created_at=iso(r['created_at']), FAIL_TO_PASS=as_list(r['FAIL_TO_PASS']),
                         PASS_TO_PASS=as_list(r['PASS_TO_PASS']),
                         environment_setup_commit=r.get('environment_setup_commit') or '',
                         hint=dict(install=[ic.get('install') or ''] + as_list(ic.get('pre_install')),
                                   pip_packages=as_list(ic.get('pip_packages')), reqs_path=as_list(ic.get('reqs_path')),
                                   python=str(ic.get('python') or ''), test_cmd=str(ic.get('test_cmd') or ''))))
    lv = pq.read_table(os.path.join(W4, 'live_full.parquet')).to_pandas()
    for r in lv.to_dict('records'):
        rows.append(dict(source='swe-bench-live', instance_id=r['instance_id'], repo=r['repo'],
                         base_commit=r['base_commit'], problem_statement=r['problem_statement'] or '',
                         patch=r['patch'], test_patch=r['test_patch'], hints_text=r.get('hints_text') or '',
                         created_at=iso(r['created_at']), FAIL_TO_PASS=as_list(r['FAIL_TO_PASS']),
                         PASS_TO_PASS=as_list(r['PASS_TO_PASS']), environment_setup_commit='',
                         hint=dict(install=[], pip_packages=[], reqs_path=[], python='',
                                   test_cmd=' && '.join(as_list(r.get('test_cmds'))))))
    cols = ['instance_id', 'repo', 'language', 'created_at', 'base_commit', 'patch', 'test_patch',
            'problem_statement', 'FAIL_TO_PASS', 'PASS_TO_PASS', 'install_config']
    v2 = pq.read_table(os.path.join(W4, 'rebench_v2.parquet'), columns=cols,
                       filters=[('language', '=', 'python')]).to_pandas()
    for r in v2.to_dict('records'):
        ic = r.get('install_config') or {}
        ds = ic.get('docker_specs') or {}
        rows.append(dict(source='swe-rebench-v2', instance_id=r['instance_id'], repo=r['repo'],
                         base_commit=r['base_commit'], problem_statement=r['problem_statement'] or '',
                         patch=r['patch'], test_patch=r['test_patch'], hints_text='',
                         created_at=iso(r['created_at']), FAIL_TO_PASS=as_list(r['FAIL_TO_PASS']),
                         PASS_TO_PASS=as_list(r['PASS_TO_PASS']), environment_setup_commit='',
                         hint=dict(install=as_list(ic.get('install')), pip_packages=[], reqs_path=[],
                                   python=str(ds.get('python_version') or ''), test_cmd=str(ic.get('test_cmd') or ''))))
    return rows


def tuned_repos():
    repos = set(EVAL_REPOS)
    for line in open(os.path.join(ROOT, 'data', 'toolbench', 'external.jsonl')):
        if line.strip():
            repos.add(json.loads(line)['repo'].lower())
    return repos


def check(r, excluded):
    """None when the task qualifies, else the reason. Fills r['gold_files'], r['gold_lines'], restricts F2P/P2P."""
    if r['created_at'] < CUTOFF:
        return 'created before 2025-02-01'
    if r['repo'].lower() in excluded:
        return 'tuning/eval repository'
    src = [(p, k) for p, k in patch_files(r['patch']) if not TEST_PATH.search(p)]
    py = [p for p, k in src if p.endswith('.py')]
    if not 1 <= len(py) <= MAX_PY_FILES:
        return f'gold touches {len(py)} non-test .py files'
    n = changed_lines(r['patch'], set(py))
    if n > MAX_CHANGED:
        return 'gold > 200 changed lines'
    st = (r['problem_statement'] or '').strip()
    if len(st) < MIN_STATEMENT or len(URL.sub('', st).strip()) < MIN_TEXT:
        return 'short or URL-only statement'
    tfiles = [f for f in test_files_of(r['test_patch']) if f.endswith('.py')]
    if not tfiles:
        return 'test_patch touches no .py file'
    tset = set(tfiles)
    if any('::' not in x for x in r['FAIL_TO_PASS']):
        return 'non-pytest FAIL_TO_PASS ids'
    f2p = [x for x in r['FAIL_TO_PASS'] if x.split('::')[0] in tset]
    p2p = [x for x in r['PASS_TO_PASS'] if x.split('::')[0] in tset]
    if not f2p:
        return 'no FAIL_TO_PASS in the test_patch files'
    r.update(gold_files=py, gold_lines=n, f2p_dropped=len(r['FAIL_TO_PASS']) - len(f2p),
             p2p_dropped=len(r['PASS_TO_PASS']) - len(p2p), FAIL_TO_PASS=f2p, PASS_TO_PASS=p2p, test_files=tfiles)
    return None


def gh_meta(repos):
    """GitHub canonical name, size and languages per repository (cached)."""
    meta = json.load(open(GH_META)) if os.path.exists(GH_META) else {}
    todo = [r for r in sorted(repos) if r not in meta]

    def one(repo):
        r = subprocess.run(['gh', 'api', f'repos/{repo}'], capture_output=True, text=True)
        if r.returncode:
            return repo, {'error': (r.stderr or r.stdout).strip()[-200:]}
        d = json.loads(r.stdout)
        lang = subprocess.run(['gh', 'api', f'repos/{repo}/languages'], capture_output=True, text=True)
        langs = json.loads(lang.stdout) if lang.returncode == 0 else {}
        return repo, {'full_name': d['full_name'], 'size_kb': d.get('size', 0), 'archived': d.get('archived'),
                      'fork': d.get('fork'), 'languages': langs, 'license': (d.get('license') or {}).get('spdx_id')}
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        for i, (repo, m) in enumerate(ex.map(one, todo), 1):
            meta[repo] = m
            if i % 50 == 0:
                print(f'  gh meta {i}/{len(todo)}', flush=True)
                json.dump(meta, open(GH_META, 'w'), indent=1, sort_keys=True)
    json.dump(meta, open(GH_META, 'w'), indent=1, sort_keys=True)
    return meta


HEAVY_RE = re.compile(r'(?<![\w.-])(torch|torchvision|torchaudio|torchmetrics|pytorch-lightning|lightning|torch-geometric|keras|tensorflow(?:-cpu|-gpu)?|tf-nightly|paddlepaddle(?:-gpu)?|'
                      r'mxnet|cupy(?:-cuda\w+)?|triton|vllm|deepspeed|jax|jaxlib|nvidia-(?!ml-py)[\w-]+)(?:\[[^\]]*\])?\s*'
                      r'(?:[<>=!~;,"\'\]\n]|$)', re.I | re.M)


def heavy_deps(repo, sha):
    """Heavy packages (torch, tensorflow, jax, CUDA stacks) named in the root dependency files at base_commit."""
    import base64
    r = subprocess.run(['gh', 'api', f'repos/{repo}/contents?ref={sha}'], capture_output=True, text=True)
    if r.returncode:
        return []
    names = [e['path'] for e in json.loads(r.stdout) if e['type'] == 'file' and re.fullmatch(
        r'pyproject\.toml|setup\.py|setup\.cfg|(dev-|test-)?requirements[^/]*\.(txt|in)|environment\.ya?ml', e['name'])]
    names += [e['path'] for e in json.loads(r.stdout) if e['type'] == 'dir' and e['name'] == 'requirements']
    found = set()
    for n in names:
        r = subprocess.run(['gh', 'api', f'repos/{repo}/contents/{n}?ref={sha}'], capture_output=True, text=True)
        if r.returncode:
            continue
        d = json.loads(r.stdout)
        entries = d if isinstance(d, list) else [d]
        for e in entries:
            if e.get('type') != 'file':
                continue
            if 'content' not in e:
                r2 = subprocess.run(['gh', 'api', f'repos/{repo}/contents/{e["path"]}?ref={sha}'], capture_output=True,
                                    text=True)
                if r2.returncode:
                    continue
                e = json.loads(r2.stdout)
            text = base64.b64decode(e.get('content', '')).decode('utf-8', 'replace')
            text = '\n'.join(l for l in text.splitlines() if not l.lstrip().startswith('#'))
            found |= {m.lower() for m in HEAVY_RE.findall(text)}
    return sorted(found)


def compiled_share(langs):
    total = sum(langs.values()) or 1
    return sum(v for k, v in langs.items() if k in COMPILED) / total


def select():
    os.makedirs(SCRATCH, exist_ok=True)
    rows = load_sources()
    excluded = tuned_repos()
    funnel = collections.OrderedDict()
    funnel['rows (Python, 3 sources)'] = len(rows)
    by_src = collections.Counter(r['source'] for r in rows)
    reasons = collections.Counter()
    seen, ok = {}, []
    prio = {'swe-rebench-leaderboard': 0, 'swe-bench-live': 1, 'swe-rebench-v2': 2}
    for r in sorted(rows, key=lambda r: (prio[r['source']], r['instance_id'])):
        if r['instance_id'] in seen:
            reasons['duplicate instance (other source)'] += 1
            continue
        seen[r['instance_id']] = r['source']
        why = check(r, excluded)
        if why:
            reasons[why] += 1
        else:
            ok.append(r)
    funnel['eligible tasks'] = len(ok)
    meta = gh_meta({r['repo'] for r in ok})
    by_repo = collections.defaultdict(list)
    dropped_meta = collections.Counter()
    for r in ok:
        m = meta.get(r['repo'], {})
        if 'error' in m or not m:
            dropped_meta['GitHub repository unavailable'] += 1
            continue
        canon = m['full_name']
        if canon.lower() in excluded:
            dropped_meta['tuning/eval repository (renamed)'] += 1
            continue
        r['repo_canonical'] = canon
        by_repo[canon.lower()].append(r)
    funnel['eligible tasks with a live GitHub repo'] = sum(map(len, by_repo.values()))
    funnel['eligible repositories'] = len(by_repo)
    rng = random.Random(SEED)
    picks = []
    for key in sorted(by_repo):
        tasks = sorted(by_repo[key], key=lambda r: r['instance_id'])
        rng.shuffle(tasks)
        modern = [t for t in tasks if py_minor(t['hint']['python']) >= 11]
        pick = (modern or tasks)[0]
        m = meta[pick['repo']]
        cs = compiled_share(m['languages'])
        pv = py_minor(pick['hint']['python'])
        pure = cs < 0.03
        small = m['size_kb'] <= 500_000
        tier = 'A' if pure and small and (pv >= 11 or pv == 0) else ('B' if pure and small else 'C')
        pick.update(tier=tier, compiled_share=round(cs, 4), repo_size_kb=m['size_kb'],
                    repo_tasks_in_pool=len(tasks))
        picks.append(pick)
    rng.shuffle(picks)
    picks.sort(key=lambda r: r['tier'])  # stable: random order within a tier
    # heavy binary dependencies (GB-sized wheels, GPU stacks): checked in rank order until N_CANDIDATES are clean
    heavy_cache = json.load(open(HEAVY)) if os.path.exists(HEAVY) else {}
    chosen, heavy = [], []
    for p in picks:
        if len(chosen) >= N_CANDIDATES:
            break
        if p['tier'] == 'C':
            chosen.append(p)
            continue
        key = f"{p['repo_canonical']}@{p['base_commit']}"
        if key not in heavy_cache:
            heavy_cache[key] = heavy_deps(p['repo_canonical'], p['base_commit'])
            json.dump(heavy_cache, open(HEAVY, 'w'), indent=1, sort_keys=True)
        if heavy_cache[key]:
            p['heavy'] = heavy_cache[key]
            heavy.append(p)
        else:
            chosen.append(p)
    funnel['repositories screened for heavy dependencies'] = len(chosen) + len(heavy)
    funnel['dropped: heavy binary dependencies'] = len(heavy)
    for i, r in enumerate(chosen):
        r['order'] = i
    with open(CANDIDATES, 'w') as f:
        for r in chosen:
            f.write(json.dumps(r) + '\n')
    stats = dict(funnel=funnel, rows_by_source=by_src, rejected=reasons.most_common(), dropped_meta=dropped_meta,
                 tiers_all=collections.Counter(p['tier'] for p in picks),
                 tiers_chosen=collections.Counter(r['tier'] for r in chosen),
                 eligible_by_source=collections.Counter(r['source'] for r in ok),
                 chosen_by_source=collections.Counter(r['source'] for r in chosen),
                 repos_by_source_pool=collections.Counter(
                     src for src, n in {(p['source'], p['repo_canonical'].lower()): 1 for v in by_repo.values()
                                        for p in v}.keys()),
                 heavy=[(p['repo_canonical'], p['heavy']) for p in heavy],
                 n_candidates=len(chosen))
    json.dump(stats, open(POOL_STATS, 'w'), indent=1)
    print(json.dumps(stats, indent=1))


def py_minor(v):
    m = re.match(r'3\.(\d+)', str(v or ''))
    return int(m.group(1)) if m else 0


def load_candidates(only=None, first=None):
    rows = [json.loads(l) for l in open(CANDIDATES) if l.strip()]
    if first:
        rows = rows[:first]
    return [r for r in rows if not only or r['instance_id'] in only]


# ---------------------------------------------------------------- snapshot
def git(cwd, *a, check=True, timeout=1800, inp=None):
    r = subprocess.run(['git', '-c', 'core.quotepath=false', '-c', 'commit.gpgsign=false', '-c', 'filter.lfs.smudge=',
                        '-c', 'filter.lfs.process=', '-c', 'filter.lfs.required=false', '-c', 'advice.detachedHead=false',
                        '-c', 'core.autocrlf=false', '-C', cwd, *a], capture_output=True, text=True, errors='replace',
                       env=GIT_ENV, timeout=timeout, input=inp)
    if check and r.returncode:
        raise RuntimeError(f'git {" ".join(a[:3])}: {r.stderr.strip()[-300:]}')
    return r.stdout


def snap_path(iid):
    return os.path.join(SNAPS, f'{iid}.tgz')


def make_snapshot(r):
    """Build data/unseen/snapshots/<id>.tgz (see the module docstring). Returns (status, info).
    History depth DEPTHS[0], reduced while the snapshot is larger than MAX_SNAPSHOT_MB."""
    iid = r['instance_id']
    if os.path.exists(snap_path(iid)) and os.path.exists(os.path.join(SNAPMETA, f'{iid}.json')):
        return 'cached', json.load(open(os.path.join(SNAPMETA, f'{iid}.json')))
    for depth in DEPTHS:
        status, info = _build_snapshot(r, depth)
        if status == 'tree differs from upstream base_commit':  # fast-export lost a path (shallow boundary)
            status, info = _build_snapshot(r, depth, shallow_copy=True)
        if status != 'ok' or info['snapshot_mb'] <= MAX_SNAPSHOT_MB or depth == DEPTHS[-1]:
            break
    if status == 'ok':
        info['history_depth'] = depth
        os.makedirs(SNAPMETA, exist_ok=True)
        json.dump(info, open(os.path.join(SNAPMETA, f'{iid}.json'), 'w'), indent=1, sort_keys=True)
    return status, info


def _build_snapshot(r, depth, shallow_copy=False):
    iid, sha, repo = r['instance_id'], r['base_commit'], r['repo_canonical']
    work = os.path.join(SCRATCH, 'git', iid)
    shutil.rmtree(work, ignore_errors=True)
    bare, fresh = os.path.join(work, 'src.git'), os.path.join(work, 'repo')
    os.makedirs(bare)
    t0 = time.time()
    try:
        git(bare, 'init', '-q', '--bare')
        url = f'https://github.com/{repo}.git'
        out = subprocess.run(['git', '-C', bare, 'fetch', '-q', '--no-tags', f'--depth={depth}', url, sha],
                             capture_output=True, text=True, env=GIT_ENV, timeout=3600)
        if out.returncode:
            return 'fetch failed', {'error': out.stderr.strip()[-300:]}
        git(bare, 'update-ref', 'refs/heads/export_ref', sha)
        os.makedirs(fresh)
        git(fresh, 'init', '-q', '-b', 'main')
        git(fresh, 'config', 'core.ignorecase', 'false')  # macOS default true mangles case-only renames in history
        exp = subprocess.Popen(['git', '-C', bare, 'fast-export', '--signed-tags=strip', '--tag-of-filtered-object=drop',
                                '--reencode=yes', 'refs/heads/export_ref'], stdout=subprocess.PIPE, env=GIT_ENV)
        imp = subprocess.run(['git', '-C', fresh, 'fast-import', '--quiet', '--date-format=raw-permissive'],
                             stdin=exp.stdout, capture_output=True, env=GIT_ENV, timeout=3600)
        exp.stdout.close()
        history = 'fast-export'
        if exp.wait() or imp.returncode or shallow_copy:
            # fast-import rejects some old history (malformed idents): copy the shallow history as it is instead
            # (still nothing after base_commit; the repository keeps a .git/shallow boundary)
            err = imp.stderr.decode(errors='replace')[-300:]
            shutil.rmtree(fresh, ignore_errors=True)
            os.makedirs(fresh)
            git(fresh, 'init', '-q', '-b', 'main')
            git(fresh, 'config', 'core.ignorecase', 'false')
            f2 = subprocess.run(['git', '-C', fresh, 'fetch', '-q', '--no-tags', '--update-shallow', bare,
                                 'refs/heads/export_ref:refs/heads/export_ref'], capture_output=True, text=True,
                                env=GIT_ENV, timeout=3600)
            if f2.returncode:
                return 'fast-export failed', {'error': err}
            history = ('shallow copy (fast-export lost a path)' if shallow_copy else
                       'shallow copy (fast-import rejected the history)')
        git(fresh, 'update-ref', 'refs/heads/main', 'refs/heads/export_ref')
        git(fresh, 'config', 'user.email', 'agent@eval')
        git(fresh, 'config', 'user.name', 'Agent')
        git(fresh, 'checkout', '-q', '-f', 'main')  # scratch repository under results/_analysis/unseen/git
        dirty = git(fresh, 'status', '--porcelain', '--untracked-files=no').strip()
        if dirty:
            return 'checkout not clean (case-insensitive paths?)', {'error': dirty[:300]}
        git(fresh, 'repack', '-adq')
        git(fresh, 'gc', '-q', '--prune=now')
        shutil.rmtree(os.path.join(fresh, '.git', 'logs', 'refs', 'remotes'), ignore_errors=True)
        for key in ('core.ignorecase', 'core.precomposeunicode'):  # macOS-only settings of `git init`
            git(fresh, 'config', '--unset', key, check=False)
        info = classify(fresh, r)
        info['history'] = history
        info['commits'] = int(git(fresh, 'rev-list', '--count', 'HEAD').strip())
        ct = int(git(fresh, 'log', '-1', '--format=%ct', 'HEAD').strip())
        info['base_commit_date'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(ct))
        info['imported_head'] = git(fresh, 'rev-parse', 'HEAD').strip()
        info['upstream_tree_matches'] = git(fresh, 'rev-parse', 'HEAD^{tree}').strip() == \
            git(bare, 'rev-parse', f'{sha}^{{tree}}').strip()
        if not info['upstream_tree_matches']:
            return 'tree differs from upstream base_commit', {'error': 'HEAD^{tree} != upstream'}
        os.makedirs(SNAPS, exist_ok=True)
        tmp = snap_path(iid) + '.part'
        env = dict(os.environ, COPYFILE_DISABLE='1')
        tr = subprocess.run(['tar', '--no-xattrs', '--no-acls', '--no-fflags', '--no-mac-metadata', '--uid', '0',
                             '--gid', '0', '--uname', 'root', '--gname', 'root', '-czf', tmp, '-C', fresh, '.'],
                            capture_output=True, text=True, env=env)
        if tr.returncode:
            return 'tar failed', {'error': tr.stderr[-300:]}
        os.replace(tmp, snap_path(iid))
        info['snapshot_mb'] = round(os.path.getsize(snap_path(iid)) / 1e6, 1)
        info['build_s'] = round(time.time() - t0, 1)
        return 'ok', info
    except Exception as e:  # noqa: BLE001
        return 'error', {'error': str(e)[-300:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------- classification
PROJECT_FILES = ('pyproject.toml', 'setup.py', 'setup.cfg')


def module_name_v24(rel):
    """How agent-v24 derives the import name of a file: the path minus a leading src/ (B2, LOG 2026-10-10)."""
    parts = rel[:-3].split('/')
    if parts and parts[0] == 'src':
        parts = parts[1:]
    if parts and parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts) if parts and all(p.isidentifier() for p in parts) else None


def import_root(ws, rel):
    """(true module name or None, import root dir): the parent of the top-most __init__.py chain holding the file."""
    parts = rel[:-3].split('/')
    d = parts[:-1]
    i = len(d)
    while i > 0 and os.path.exists(os.path.join(ws, *d[:i], '__init__.py')):
        i -= 1
    pkg = d[i:]
    if not pkg and parts[-1] != '__init__':
        return None, '/'.join(d[:i]) or '.'
    return '.'.join(pkg + ([] if parts[-1] == '__init__' else [parts[-1]])), '/'.join(d[:i]) or '.'


def project_root(ws, rel):
    """Nearest directory (from the file upward) with pyproject.toml / setup.py / setup.cfg; '' if none."""
    d = os.path.dirname(rel)
    while True:
        if any(os.path.isfile(os.path.join(ws, d, f)) for f in PROJECT_FILES):
            return d or '.'
        if not d:
            return ''
        d = os.path.dirname(d)


def read_text(p):
    try:
        return open(p, encoding='utf-8', errors='replace').read()
    except OSError:
        return ''


def python_versions(ws, proj):
    """requires-python and the highest 'Programming Language :: Python :: 3.x' classifier of the project."""
    import tomllib
    base = os.path.join(ws, '' if proj in ('.', '') else proj)
    req, minors = '', set()
    text = read_text(os.path.join(base, 'pyproject.toml'))
    try:
        proj_t = tomllib.loads(text).get('project', {}) if text else {}
        req = proj_t.get('requires-python', '') or ''
        poetry = tomllib.loads(text).get('tool', {}).get('poetry', {}) if text else {}
        req = req or str(poetry.get('dependencies', {}).get('python', ''))
    except Exception:
        pass
    for f in ('pyproject.toml', 'setup.cfg', 'setup.py'):
        t = read_text(os.path.join(base, f))
        minors |= {int(m) for m in re.findall(r'Programming Language :: Python :: 3\.(\d+)', t)}
        if not req:
            m = re.search(r'python_requires\s*[=:]\s*["\']?([^"\'\n]+)', t)
            req = m.group(1).strip() if m else ''
    return req, (f'3.{max(minors)}' if minors else '')


def classify(ws, r):
    gold = [(p, k) for p, k in patch_files(r['patch']) if not TEST_PATH.search(p) and p.endswith('.py')
            and k != 'deleted']
    files = []
    for rel, k in gold:
        if not os.path.exists(os.path.join(ws, rel) if k == 'modified' else os.path.join(ws, os.path.dirname(rel))):
            files.append({'file': rel, 'missing': True})
            continue
        true, root = import_root(ws, rel)
        files.append({'file': rel, 'true': true, 'root': root, 'v24': module_name_v24(rel),
                      'project': project_root(ws, rel), 'added': k == 'added'})
    roots = {f['root'] for f in files if 'root' in f}
    def kind(f):
        if f.get('missing'):
            return 'missing'
        if f['true'] is None:
            return 'namespace'
        return 'flat' if f['root'] == '.' else ('src' if f['root'] == 'src' else 'nested')
    kinds = collections.Counter(kind(f) for f in files)
    layout = ('nested' if kinds['nested'] else 'namespace' if kinds['namespace'] else 'src' if kinds['src']
              else 'flat' if kinds['flat'] else 'unknown')
    mismatch = any(f.get('true') and f['true'] != f['v24'] for f in files)
    projects = sorted({f['project'] for f in files if f.get('project')})
    proj = projects[0] if projects else '.'
    py_req, py_max = python_versions(ws, proj)
    # repo size and compiled code
    n_files = n_py = loc = 0
    compiled = []
    for dp, dn, fn in os.walk(ws):
        if '.git' in dn:
            dn.remove('.git')
        for f in fn:
            n_files += 1
            if f.endswith('.py'):
                n_py += 1
                try:
                    with open(os.path.join(dp, f), 'rb') as fh:
                        loc += fh.read().count(b'\n')
                except OSError:
                    pass
            elif f.endswith(('.pyx', '.pxd', '.c', '.cpp', '.cc', '.rs')) and len(compiled) < 5:
                compiled.append(os.path.relpath(os.path.join(dp, f), ws))
    pyproject = read_text(os.path.join(ws, '' if proj == '.' else proj, 'pyproject.toml'))
    backend = (re.search(r'build-backend\s*=\s*["\']([^"\']+)', pyproject) or [None, ''])[1]
    hints = {
        'pytest_ini': os.path.exists(os.path.join(ws, 'pytest.ini')),
        'pyproject_pytest': '[tool.pytest' in read_text(os.path.join(ws, 'pyproject.toml')),
        'setup_cfg_pytest': '[tool:pytest]' in read_text(os.path.join(ws, 'setup.cfg')),
        'tox_ini': os.path.exists(os.path.join(ws, 'tox.ini')),
        'noxfile': os.path.exists(os.path.join(ws, 'noxfile.py')),
        'asyncio_mode_auto': bool(re.search(r'asyncio_mode\s*=\s*["\']?auto', read_text(os.path.join(ws, 'pyproject.toml'))
                                            + read_text(os.path.join(ws, 'pytest.ini'))
                                            + read_text(os.path.join(ws, 'setup.cfg')))),
        'conftests': len(glob.glob(os.path.join(ws, '**', 'conftest.py'), recursive=True)),
        'dataset_test_cmd': r['hint']['test_cmd'][:200],
    }
    return dict(layout=layout, layout_kinds=dict(kinds), v24_name_mismatch=mismatch, import_roots=sorted(roots),
                gold=files, project_roots=projects, nested_project=proj not in ('.', ''), build_backend=backend,
                requires_python=py_req, max_classifier=py_max, dataset_python=r['hint']['python'],
                files=n_files, py_files=n_py, py_loc=loc, compiled_files=compiled, runner=hints)


def snapshot(workers, only=None, first=None):
    rows = load_candidates(only, first)
    log = os.path.join(SCRATCH, 'snapshot_log.jsonl')
    t0 = time.time()
    res = collections.Counter()
    with concurrent.futures.ThreadPoolExecutor(workers) as ex, open(log, 'a') as lf:
        for r, (status, info) in zip(rows, ex.map(make_snapshot, rows)):
            res[status] += 1
            lf.write(json.dumps({'instance_id': r['instance_id'], 'status': status,
                                 'error': info.get('error', '')}) + '\n')
            print(f"{r['instance_id']}: {status} " + (f"{info.get('snapshot_mb')} MB {info.get('layout')} "
                  f"{info.get('build_s')}s" if status in ('ok', 'cached') else info.get('error', '')[:200]), flush=True)
    print(f'snapshot done in {time.time() - t0:.0f}s: {dict(res)}')


# ---------------------------------------------------------------- tasks + meta
TASK_KEYS = ('instance_id', 'repo', 'base_commit', 'problem_statement', 'patch', 'test_patch', 'hints_text',
             'created_at', 'FAIL_TO_PASS', 'PASS_TO_PASS', 'environment_setup_commit')


def layout_of(gold):
    """Layout class of a task from its gold files' import roots (files under dot-directories ignored):
    flat (root = repository root), src (root = src/), nested (root deeper than one directory, or a top-level directory
    with a project file of its own: a monorepo package root), namespace (a top-level directory without __init__.py
    and without its own project file, or a module outside any package below the root). Priority
    nested > namespace > src > flat."""
    kinds = set()
    for f in gold:
        if f.get('missing') or f['file'].startswith('.') or '/.' in f['file']:
            continue
        root = f['root']
        if f['true'] is None:
            kinds.add('flat' if root == '.' else 'namespace')
        elif root == '.':
            kinds.add('flat')
        elif root == 'src':
            kinds.add('src')
        elif '/' in root or f.get('project') not in ('.', '', None):
            kinds.add('nested')
        else:
            kinds.add('namespace')
    for k in ('nested', 'namespace', 'src', 'flat'):
        if k in kinds:
            return k
    return 'unknown'


def write_tasks():
    rows = load_candidates()
    os.makedirs(OUT, exist_ok=True)
    for name in ('wheels', 'sandbox'):  # the harness finds wheels/ and sandbox/setup.py next to tasks.jsonl
        link = os.path.join(OUT, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join('..', 'competition', name), link)
    n = 0
    with open(os.path.join(OUT, 'tasks.jsonl'), 'w') as tf, open(os.path.join(OUT, 'meta.jsonl'), 'w') as mf:
        for r in rows:
            mp = os.path.join(SNAPMETA, f"{r['instance_id']}.json")
            if not (os.path.exists(snap_path(r['instance_id'])) and os.path.exists(mp)):
                continue
            sm = json.load(open(mp))
            task = {k: r.get(k, '') for k in TASK_KEYS}
            task['repo'] = r['repo_canonical']
            # PASS_TO_PASS left empty: the harness already requires every test in the test_patch files to pass
            # (exit code 0), and its loose name matching fails a required node when any parametrization or
            # same-prefix sibling is skipped/xfailed (first pass: 3 of 39 gold patches failed only on that)
            task['PASS_TO_PASS'] = []
            if not task['environment_setup_commit']:
                del task['environment_setup_commit']
            tf.write(json.dumps(task) + '\n')
            meta = dict(instance_id=r['instance_id'], repo=r['repo_canonical'], source=r['source'], tier=r['tier'],
                        order=r['order'], upstream_base_commit=r['base_commit'], gold_files=r['gold_files'],
                        gold_lines=r['gold_lines'], f2p=len(r['FAIL_TO_PASS']), p2p=len(r['PASS_TO_PASS']),
                        f2p_dropped=r['f2p_dropped'], p2p_dropped=r['p2p_dropped'], test_files=r['test_files'],
                        repo_size_kb=r['repo_size_kb'], compiled_share=r['compiled_share'],
                        install_hint=r['hint'], env_cutoff=max(r['created_at'], sm.get('base_commit_date') or ''),
                        layout=layout_of(sm['gold']), layout_raw=sm['layout'], history=sm.get('history', 'fast-export'),
                        **{k: sm[k] for k in ('layout_kinds', 'v24_name_mismatch', 'import_roots',
                                              'project_roots', 'nested_project', 'build_backend', 'requires_python',
                                              'max_classifier', 'dataset_python', 'files', 'py_files', 'py_loc',
                                              'compiled_files', 'runner', 'commits', 'snapshot_mb', 'gold',
                                              'history_depth', 'base_commit_date')})
            mf.write(json.dumps(meta) + '\n')
            n += 1
    print(f'wrote {n} tasks to data/unseen/tasks.jsonl (+ meta.jsonl)')


# ---------------------------------------------------------------- finalize
def results_of(d):
    """{instance_id: TaskResult row (+ test_output from test_outputs/<id>.log)} over every run under d."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '**', 'task_results.jsonl'), recursive=True)):
        for line in open(f):
            if line.strip():
                x = json.loads(line)
                iid = x.get('instance_id') or x.get('task_id')
                log = os.path.join(os.path.dirname(f), 'test_outputs', f'{iid}.log')
                if os.path.exists(log):
                    x['test_output'] = open(log, errors='replace').read()
                out[iid] = x
    return out


FAILURE_CLASSES = [  # (class, regex over the pytest output), first match wins; order matters
    ('network access (sandbox is offline)', r'Temporary failure in name resolution|NameResolutionError|Max retries '
     r'exceeded|Failed to download|urlopen error|ConnectionRefusedError|Connection refused|getaddrinfo failed'),
    ('system library missing', r'cannot open shared object file|cannot load library|libpq library not found|'
     r'fontconfig not found|cannot load MPI|RPM Python bindings|No module named .(gi|osgeo|_tkinter|tkinter)\b'),
    ('external program missing', r"No such file or directory: '(?!/)[\w.-]+'|requires \w+ binary|"
     r'command not found|executable not found'),
    ('pytest 9 incompatibility', r'hookimpl definition|has no attribute .io.|call_runtest_hook|PytestRemovedIn'),
    ('harness pytest config (-p no:anyio / root pytest.ini)', r"fixture 'anyio_backend"),
    ('fixture of a missing plugin', r"fixture '\w+' not found"),
]


def failure_reason(x):
    """Short reason for a task that did not resolve, from the harness's TaskResult (error / pytest output)."""
    err = x.get('error_message') or x.get('error') or ''
    out = x.get('test_output') or ''
    if 'Failed to apply test_patch' in err:
        return 'test_patch does not apply'
    if 'Failed to apply agent patch' in err:
        return 'gold patch does not apply'
    if x.get('test_exit_code') in (124, 137, -9) or 'Timed out' in out or 'timed out' in err.lower():
        return 'test run timed out'
    if x.get('test_exit_code') in (139, -11):
        return 'crash (segfault)'
    for name, rx in FAILURE_CLASSES:
        if re.search(rx, out):
            return name
    m = re.search(r"(?:ModuleNotFoundError|ImportError): No module named '([^']+)'", out)
    if m:
        return f'missing module {m.group(1).split(".")[0]}'
    if re.search(r'ImportError: cannot import name|AttributeError: module .* has no attribute|'
                 r'InvalidVersion|removed in the NumPy', out):
        return 'version mismatch'
    if re.search(r'errors? during collection|ERROR collecting|INTERNALERROR|ImportError while loading conftest', out):
        return 'collection error (other)'
    if 'Required test node did not pass' in err:
        return 'required test missing or not passing'
    if re.search(r'(\d+) (failed|errors?)', out):
        return 'tests fail in the test files (other)'
    if 'No passing tests' in err or 'no tests ran' in out:
        return 'no tests ran'
    return (err or 'unknown')[:80]


def pytest_seconds(x):
    m = re.findall(r' in ([\d.]+)s', x.get('test_output') or '')
    return float(m[-1]) if m else None


def report(gold_dir, empty_dir):
    """Per-task table + failure reasons of the gold and empty-patch runs (results/_analysis/unseen/)."""
    meta = {json.loads(l)['instance_id']: json.loads(l) for l in open(os.path.join(OUT, 'meta.jsonl')) if l.strip()}
    gold, empty = results_of(gold_dir), results_of(empty_dir)
    rows = []
    for i, m in meta.items():
        g, e = gold.get(i), empty.get(i)
        rows.append(dict(instance_id=i, repo=m['repo'], source=m['source'], layout=m['layout'],
                         mismatch=m['v24_name_mismatch'], gold=None if g is None else bool(g.get('resolved')),
                         empty=None if e is None else bool(e.get('resolved')),
                         reason='' if g is None or g.get('resolved') else failure_reason(g),
                         pytest_s=pytest_seconds(g) if g else None,
                         duration_s=round(g.get('duration_seconds') or 0) if g else None))
    with open(os.path.join(SCRATCH, 'tasks_report.tsv'), 'w') as f:
        keys = list(rows[0])
        f.write('\t'.join(keys) + '\n')
        for r in rows:
            f.write('\t'.join(str(r[k]) for k in keys) + '\n')
    reasons = collections.Counter(r['reason'] for r in rows if r['gold'] is False)
    print(f"gold: {sum(r['gold'] is True for r in rows)} pass / {sum(r['gold'] is False for r in rows)} fail / "
          f"{sum(r['gold'] is None for r in rows)} not run; empty-patch pass (dropped): "
          f"{[r['instance_id'] for r in rows if r['gold'] and r['empty']]}")
    for k, v in reasons.most_common():
        print(f'  {v:3d}  {k}')
    return rows


# gold passes excluded after re-verification (reason recorded in research/notes/21-unseen100.md)
FLAKY = {'bluesky__bluesky-1932': 'self-SIGINT timing test: gold failed 1 of 7 runs (under load, conc 4)'}


def finalize(gold_dir, empty_dir, ids_file=None):
    meta = {json.loads(l)['instance_id']: json.loads(l) for l in open(os.path.join(OUT, 'meta.jsonl')) if l.strip()}
    gold, empty = results_of(gold_dir), results_of(empty_dir)
    kept = sorted(i for i in meta if gold.get(i, {}).get('resolved') and i in empty and not empty[i].get('resolved')
                  and i not in FLAKY)
    rng = random.Random(SEED)
    rng.shuffle(kept)
    path = ids_file or os.path.join(ROOT, 'experiments', 'unseen100.txt')
    with open(path, 'w') as f:
        f.write(f'# unseen100: {len(kept)} gold-verified tasks, one per repository, never tuned on '
                f'(scripts/unseen_data.py, research/notes/21-unseen100.md); random order, seed {SEED}\n'
                '# run with TASKS_FILE=data/unseen/tasks.jsonl SNAPSHOTS_DIR=data/unseen/snapshots '
                'TASK_ENV_MANIFEST=data/unseen/task_env.json\n')
        f.writelines(i + '\n' for i in kept)
    print(f'kept {len(kept)} -> {path}')
    stats = funnel_stats(meta, gold, empty, set(kept))
    json.dump(stats, open(os.path.join(SCRATCH, 'funnel.json'), 'w'), indent=1)
    print(json.dumps(stats, indent=1))
    return kept


def funnel_stats(meta, gold, empty, kept):
    """Counts per stage, per source, layout / Python mix of the kept set, failure reasons (for note 21)."""
    sel = json.load(open(POOL_STATS))
    cands = load_candidates()
    manifest = json.load(open(os.path.join(OUT, 'task_env.json')))['tasks']
    resolved = {i for i, e in manifest.items() if not e['pin_source'].startswith('none')}
    gp = {i for i in meta if gold.get(i, {}).get('resolved')}
    stage = lambda ids: collections.Counter(meta[i]['source'] for i in ids if i in meta)
    reasons = collections.Counter(failure_reason(gold[i]) for i in meta if i in gold and not gold[i].get('resolved'))
    empty_pass = sorted(i for i in gp if empty.get(i, {}).get('resolved'))
    lay = lambda ids: dict(collections.Counter(meta[i]['layout'] for i in ids))
    mism = lambda ids: sum(meta[i]['v24_name_mismatch'] for i in ids)
    py = lambda ids, k: dict(sorted(collections.Counter(meta[i][k] or '?' for i in ids).items()))
    lower = lambda r: (re.search(r'>=?\s*3\.(\d+)', r or '') or [None, '?'])[1]
    return {
        'select': sel['funnel'], 'rows_by_source': sel['rows_by_source'], 'rejected': sel['rejected'][:12],
        'candidates': len(cands), 'candidates_by_source': dict(collections.Counter(r['source'] for r in cands)),
        'snapshots_built': len(meta), 'env_resolved': len(resolved & set(meta)),
        'env_heavy_skipped': sorted(i for i, e in manifest.items() if e.get('heavy_skipped')),
        'gold_run': sum(1 for i in meta if i in gold), 'gold_pass': len(gp), 'empty_pass_dropped': empty_pass,
        'flaky_dropped': sorted(FLAKY), 'kept': len(kept),
        'by_source': {src: {'candidates': sum(r['source'] == src for r in cands), 'built': stage(meta)[src],
                            'gold_pass': stage(gp)[src], 'kept': stage(kept)[src]}
                      for src in sorted({r['source'] for r in cands})},
        'gold_fail_reasons': reasons.most_common(),
        'layout': {'candidates_built': lay(meta), 'gold_pass': lay(gp), 'kept': lay(kept)},
        'v24_name_mismatch': {'built': [mism(meta), len(meta)], 'kept': [mism(kept), len(kept)]},
        'nested_project_kept': sum(meta[i]['nested_project'] for i in kept),
        'python_kept': {'dataset_python': py(kept, 'dataset_python'), 'max_classifier': py(kept, 'max_classifier'),
                        'requires_python_min': dict(sorted(collections.Counter(
                            '3.' + lower(meta[i]['requires_python']) if lower(meta[i]['requires_python']) != '?'
                            else '?' for i in kept).items()))},
        'size_kept': {'py_files_median': sorted(meta[i]['py_files'] for i in kept)[len(kept) // 2] if kept else 0,
                      'py_loc_median': sorted(meta[i]['py_loc'] for i in kept)[len(kept) // 2] if kept else 0,
                      'snapshot_mb_total': round(sum(meta[i]['snapshot_mb'] for i in kept)),
                      'gold_lines_median': sorted(meta[i]['gold_lines'] for i in kept)[len(kept) // 2] if kept else 0},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=('select', 'snapshot', 'tasks', 'report', 'finalize'))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--only', default=None)
    ap.add_argument('--first', type=int, default=None)
    ap.add_argument('--gold', default=os.path.join(ROOT, 'results', 'unseen_gold', 'final'))
    ap.add_argument('--empty', default=os.path.join(ROOT, 'results', 'unseen_empty', 'final'))
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    if a.cmd == 'select':
        select()
    elif a.cmd == 'snapshot':
        snapshot(a.workers, only, a.first)
    elif a.cmd == 'tasks':
        write_tasks()
    elif a.cmd == 'report':
        report(a.gold, a.empty)
    else:
        finalize(a.gold, a.empty)


if __name__ == '__main__':
    main()
