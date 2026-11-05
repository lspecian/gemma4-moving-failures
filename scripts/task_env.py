"""Per-task test environments for local swegemma runs (gold-pass the public tasks).

Why: swegemma 0.2.7 injects ONE environment into every sandbox: the highest version of every
wheel in data/competition/wheels (starlette 1.6.0, pygments 2.21, the released fastapi /
requests / httpx / rich wheels, ...), and sandbox/setup.py --fast-path installs nothing per task.
The published cache clearly holds per-task versions (60 starlette releases, sqlmodel 0.0.24/25/27/31
= exactly the `sqlmodel==` pins of the fastapi tasks), but nothing selects them, so most public
tasks fail even with the gold patch (results/gold_all: 52/129). Kaggle staff: the hidden test set
"validates at 100% with gold patches" and its Docker sandbox "uses the same base image but installs
a different set of dependencies" (threads 742882, 744259). This module recreates that intent for
the public repos: each task gets the dependency versions its repo pins at its base commit.

Rules (per task, decided offline by `build`, applied inside the sandbox at run time):
  1. The repo's own released wheel (fastapi/requests/httpx/rich from the cache) is removed from
     site-packages, so the editable /workspace checkout is what gets imported (src/ layouts were
     shadowed: requests/httpx tests ran against the released package).
  2. Target versions come from the repo at base commit: its lockfile (uv.lock / poetry.lock) when it
     has one, else `uv pip compile` of its declared test requirements with --exclude-newer = the
     task's created_at (what `pip install -r requirements-*.txt` gave CI at that time).
  3. Every package already in the published environment is switched to its target version.
  4. Test dependencies missing from the published cache (inline-snapshot, pytest-httpbin, attrs,
     trustme, ...) are added when the repo declares them for tests AND the tests import them (or they
     are pytest plugins), plus their runtime dependencies at target versions.
  Packages owned by the image (pytest, pluggy, pip, setuptools, wheel, pytest-timeout, iniconfig)
  are never touched. Wheels not in the published cache are downloaded once into
  data/competition/wheels_local/ (linux x86_64, CPython 3.13); data/competition/wheels/ is unchanged,
  so stock runs are unaffected. Pins with no usable wheel fall back to the published version.

Usage:
    .venv/bin/python scripts/task_env.py build            # resolve + download + write manifest
    .venv/bin/python scripts/task_env.py show fastapi_13920
In code (verify_gold.py, swegemma_local.py):  import task_env; task_env.install_hook()
Disable at run time with SWEGEMMA_TASK_ENV=0.

Other task sets (unseen100, scripts/unseen_data.py): `build` takes --tasks, --snapshots and --manifest (defaults: the
public tasks, data/competition/snapshots, the manifest above), --hints (a jsonl with per-task install hints) and
--generic (rules for arbitrary repositories, below). At run time the hook reads the manifest named by the environment
variable TASK_ENV_MANIFEST (unset: the manifest above, exactly as before). Wheels always live in wheels_local/.

    .venv/bin/python scripts/task_env.py build --tasks data/unseen/tasks.jsonl --snapshots data/unseen/snapshots \
        --manifest data/unseen/task_env.json --hints data/unseen/meta.jsonl --generic [ids...]
    TASK_ENV_MANIFEST=data/unseen/task_env.json .venv/bin/python scripts/task_env.py show <id>

--generic (never used for the public tasks, whose manifest is unchanged): the project is the gold files' nearest
directory with pyproject.toml / setup.py / setup.cfg (a nested project in a monorepo counts); its declared
requirements also come from setup.cfg / setup.py (install_requires, extras_require), tox.ini [testenv] deps, the
conventional test extras (test, tests, testing, dev) and the dataset's install hints (pip install commands, -r files,
pip_packages); its build-system requires (setuptools-scm, hatch-vcs, ...) are installed so the harness's editable
install succeeds offline; test files anywhere in the tree (not only tests/) count for imports; the resolution cutoff
is the hint's env_cutoff (later of created_at and the base commit date). A nested project is pip-installed editable
in the sandbox at run time (entry key `editable`), falling back to a .pth entry for its import root; nested import
roots of the gold files without a project of their own get a .pth entry (`pth`), as a provisioned environment would
have them importable.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'competition'
WHEELS = DATA / 'wheels'
LOCAL = DATA / 'wheels_local'
MANIFEST = LOCAL / 'task_env.json'
SP = '/usr/local/lib/python3.13/site-packages'

IMAGE_OWNED = {'pytest', 'pytest-timeout', 'pluggy', 'pip', 'setuptools', 'wheel', 'iniconfig'}
# Declared test requirements that are tooling, never imported by tests (not installed).
TOOLING = {'pytest-cov', 'coverage', 'mypy', 'ruff', 'black', 'isort', 'flake8', 'pre-commit',
           'pytest-sugar', 'pytest-codspeed', 'codespell', 'asv', 'tox', 'nox', 'build', 'twine'}
# import name -> distribution name, where they differ
IMPORT_ALIASES = {'jwt': 'pyjwt', 'yaml': 'pyyaml', 'attr': 'attrs', 'multipart': 'python-multipart',
                  'python_multipart': 'python-multipart', 'socks': 'pysocks', 'bs4': 'beautifulsoup4',
                  'dateutil': 'python-dateutil', 'dotenv': 'python-dotenv', 'pil': 'pillow',
                  'email_validator': 'email-validator', 'git': 'gitpython', 'brotli': 'brotli',
                  'zmq': 'pyzmq', 'markdown_it': 'markdown-it-py', 'importlib_metadata': 'importlib-metadata'}
GENERIC_ALIASES = {'sklearn': 'scikit-learn', 'skimage': 'scikit-image', 'cv2': 'opencv-python',
                   'openssl': 'pyopenssl', 'crypto': 'pycryptodome', 'serial': 'pyserial', 'magic': 'python-magic',
                   'docx': 'python-docx', 'jose': 'python-jose', 'ruamel': 'ruamel-yaml', 'google': 'protobuf',
                   'dns': 'dnspython', 'gi': 'pygobject', 'pkg_resources': 'setuptools', 'mpl_toolkits': 'matplotlib',
                   'win32api': 'pywin32', 'fitz': 'pymupdf', 'slugify': 'python-slugify', 'jwt': 'pyjwt',
                   'kafka': 'kafka-python', 'snappy': 'python-snappy', 'telegram': 'python-telegram-bot',
                   'freezegun': 'freezegun', 'xdist': 'pytest-xdist', 'pytest_asyncio': 'pytest-asyncio'}
TEST_EXTRAS = {'test', 'tests', 'testing', 'dev', 'develop', 'development'}
ONLY_BINARY = ('--only-binary', ':all:')
# GPU / ML stacks (torch + CUDA wheels are ~4 GB per sandbox): never installed by generic builds, nor any wheel > 150 MB
HEAVY_DIST = re.compile(r'torch|torch(vision|audio|metrics)|triton|nvidia-(?!ml-py$).+|tensorflow.*|tf-.+|jax|jaxlib|'
                        r'paddle.*|mxnet.*|cupy.*|deepspeed|vllm|xformers|bitsandbytes|onnxruntime-gpu')
MAX_WHEEL_MB = 150
STDLIB = set(sys.stdlib_module_names) | {'imp', 'telnetlib', 'distutils', 'conftest', 'pytest', '_pytest', 'py'}
# non-test requirement files / dependency groups (docs, translations, CI helpers)
NON_TEST = re.compile(r'docs(?!-tests)|translation|github-actions|insiders|release', re.I)
MARKER_ENV = {'python_version': '3.13', 'python_full_version': '3.13.15', 'sys_platform': 'linux',
              'platform_system': 'Linux', 'platform_machine': 'x86_64', 'os_name': 'posix',
              'implementation_name': 'cpython', 'platform_python_implementation': 'CPython',
              'platform_release': '', 'platform_version': '', 'implementation_version': '3.13.15',
              'extra': ''}
PIP_PLATFORM = ['--python-version', '3.13', '--implementation', 'cp', '--abi', 'cp313', '--abi', 'abi3',
                '--abi', 'none'] + [x for p in ('manylinux_2_28_x86_64', 'manylinux_2_17_x86_64',
                                                'manylinux2014_x86_64', 'manylinux_2_5_x86_64',
                                                'manylinux1_x86_64') for x in ('--platform', p)]


def _req_name(r: str) -> str | None:
    from packaging.requirements import Requirement
    try:
        return norm(Requirement(r).name)
    except Exception:
        return None


def manifest_path() -> Path:
    """The manifest the run-time hook reads: $TASK_ENV_MANIFEST (relative to the cwd, else to the repo), else MANIFEST."""
    p = os.environ.get('TASK_ENV_MANIFEST', '').strip()
    if not p:
        return MANIFEST
    q = Path(p)
    if not q.is_absolute():
        q = (Path.cwd() / q) if (Path.cwd() / q).exists() else ROOT / q
    return q.resolve()


def norm(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def wheel_name_version(fn: str) -> tuple[str, str]:
    parts = Path(fn).name[:-4].split('-')
    return norm(parts[0]), parts[1]


# ---------------------------------------------------------------- snapshot reading
def read_snapshot(tgz: Path, repo: str) -> dict:
    """Dependency files, first-party top-level names and test-file imports at base commit."""
    files: dict[str, str] = {}
    py_sources: dict[str, str] = {}
    with tarfile.open(tgz) as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            n = m.name[2:] if m.name.startswith('./') else m.name
            if n.startswith('.git/'):
                continue
            if re.fullmatch(r'(pyproject\.toml|uv\.lock|poetry\.lock|setup\.py|setup\.cfg|'
                            r'(tests?/)?requirements[^/]*\.txt)', n):
                files[n] = t.extractfile(m).read().decode('utf-8', 'replace')
            elif n.endswith('.py'):
                py_sources[n] = t.extractfile(m).read().decode('utf-8', 'replace')
    own = own_dist(files, repo)
    imp_re = re.compile(r'^\s*(?:import|from)\s+([A-Za-z0-9_]+)', re.M)
    pkg = own.replace('-', '_')
    first_party = {n.split('/')[0] for n in py_sources if '/' in n} - {'tests', 'test', 'src', pkg}
    ref_re = re.compile(r'\b(' + '|'.join(map(re.escape, sorted(first_party))) + r')(?:\.([A-Za-z0-9_]+))?\b') \
        if first_party else None
    is_test = lambda n: n.split('/')[0] in ('tests', 'test', 'testing') or n.endswith('conftest.py')
    scanned, todo, imports = set(), {n for n in py_sources if is_test(n)}, set()
    while todo:  # follow references from tests into first-party helper code (fastapi: docs_src.<topic>)
        scanned |= todo
        prefixes = set()
        for n in todo:
            text = py_sources[n]
            imports |= {m.lower() for m in imp_re.findall(text)}
            for d, sub in (ref_re.findall(text) if ref_re else []):
                prefixes.add(f'{d}/{sub}' if sub else f'{d}/')
        todo = {n for n in py_sources if any(n == p + '.py' or n.startswith(p if p.endswith('/') else p + '/')
                                            for p in prefixes)} - scanned
    return {'files': files, 'imports': imports, 'own': own}


def own_dist(files: dict, repo: str) -> str:
    try:
        py = tomllib.loads(files.get('pyproject.toml', ''))
        name = py.get('project', {}).get('name') or py.get('tool', {}).get('poetry', {}).get('name')
        if name:
            return norm(name)
    except Exception:
        pass
    return norm(repo.rsplit('/', 1)[-1])


def lock_pins(files: dict) -> dict[str, str] | None:
    """Versions pinned by uv.lock / poetry.lock, for entries whose markers match CPython 3.13 linux."""
    from packaging.markers import Marker
    from packaging.version import Version

    def applies(markers) -> bool:
        markers = [markers] if isinstance(markers, str) else (markers or [])
        try:
            return not markers or any(Marker(m).evaluate(MARKER_ENV) for m in markers)
        except Exception:
            return True
    for lf in ('uv.lock', 'poetry.lock'):
        if lf in files:
            pins: dict[str, list[tuple[bool, str]]] = {}
            for p in tomllib.loads(files[lf]).get('package', []):
                if p.get('version'):
                    ok = applies(p.get('resolution-markers') if lf == 'uv.lock' else p.get('markers'))
                    pins.setdefault(norm(p['name']), []).append((ok, p['version']))
            return {k: max([v for ok, v in vs if ok] or [v for _, v in vs], key=Version)
                    for k, vs in pins.items()}
    return None


def declared_requirements(files: dict, own: str, generic: dict | None = None) -> tuple[list[str], list[str]]:
    """(runtime, test) requirement strings the repo declares for running its tests.

    runtime = [project] dependencies + the extras its test setup installs (`-e .[x]` lines, `own[x]`
    self-references, and the conventional `all` extra) + non-optional poetry dependencies.
    test    = test dependency groups, test requirement files (docs/CI-only sources skipped), poetry
              dev dependencies.
    generic (build --generic only, from generic_inputs()): extras/runtime declared in setup.cfg / setup.py, extra
    test requirements (tox.ini, install hints), wanted extras (test extras, hinted), hinted groups, -r files.
    """
    from packaging.requirements import Requirement
    py = {}
    try:
        py = tomllib.loads(files.get('pyproject.toml', ''))
    except Exception:
        pass
    proj = py.get('project', {})
    extras = proj.get('optional-dependencies', {})
    runtime: list[str] = list(proj.get('dependencies', []))
    test: list[str] = []
    if generic:
        extras = {**generic['extras'], **extras}
        runtime += [r for r in generic['runtime'] if r not in runtime]
        test += generic['test']
    wanted_extras = {'all'} & set(extras)
    if generic:
        want = {norm(x) for x in generic['wanted']}
        wanted_extras |= {e for e in extras if norm(e) in want or generic['all_extras']}
    groups = py.get('dependency-groups', {})

    def group(name, seen=()):
        out = []
        for x in groups.get(name, []):
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict) and 'include-group' in x and x['include-group'] not in seen:
                if not NON_TEST.search(x['include-group']):
                    out += group(x['include-group'], (*seen, name))
        return out
    for g in groups:
        if 'test' in g and not NON_TEST.search(g) or g == 'dev':
            test += group(g)
        elif generic and (g in generic['groups'] or generic['all_groups']) and not NON_TEST.search(g):
            test += group(g)
    poetry = py.get('tool', {}).get('poetry', {})
    for sect, dest in [(poetry.get('dependencies', {}), runtime), (poetry.get('dev-dependencies', {}), test)] + [
            (g.get('dependencies', {}), test) for n, g in poetry.get('group', {}).items() if not NON_TEST.search(n)]:
        for k, v in sect.items():
            if k.lower() == 'python' or (isinstance(v, dict) and v.get('optional')):
                continue  # poetry extras are not installed by a plain `poetry install`
            dest.append(k + poetry_spec(v))

    def req_file(path, seen):
        if path in seen or path not in files:
            return []
        seen.add(path)
        out = []
        for line in files[path].splitlines():
            line = line.split(' #')[0].strip()
            if not line or line.startswith('#') or 'git+' in line or '${' in line:
                continue
            if line.startswith(('-r ', '--requirement ')):
                inc = line.split(None, 1)[1].strip()
                inc = str(Path(path).parent / inc) if '/' in path else inc
                if not NON_TEST.search(inc):
                    out += req_file(inc, seen)
            elif line.startswith('-e ') or line.startswith('.'):
                m = re.search(r'\[([^\]]+)\]', line)
                wanted_extras.update(x.strip() for x in (m.group(1).split(',') if m else []))
            elif not line.startswith('-'):
                out.append(line)
        return out
    seen: set[str] = set()
    for path in (generic or {}).get('runtime_files', []):
        runtime += req_file(path, seen)
    for path in sorted(files):
        if re.search(r'requirements[^/]*\.txt$', path) and not NON_TEST.search(path):
            test += req_file(path, seen)
    for path in (generic or {}).get('req_files', []):
        test += req_file(path, seen)

    def clean(reqs):
        out = []
        for r in reqs:
            try:
                req = Requirement(r)
            except Exception:
                continue
            if norm(req.name) == own:  # self-reference: pull in its extras
                wanted_extras.update(req.extras)
                continue
            out.append(str(req))
        return out
    test = clean(test)
    done: set[str] = set()
    while wanted_extras - done:
        ex = sorted(wanted_extras - done)[0]
        done.add(ex)
        runtime += extras.get(ex, [])
    return clean(runtime), test


# ---------------------------------------------------------------- generic rules (build --generic)
GENERIC_DEP_FILE = re.compile(r'(pyproject\.toml|uv\.lock|poetry\.lock|setup\.py|setup\.cfg|tox\.ini|'
                              r'(tests?/)?requirements[^/]*\.(txt|in)|requirements/[^/]*\.(txt|in))')
GENERIC_TEST_FILE = re.compile(r'(^|/)(tests?|testing)/|(^|/)(test_[^/]*|[^/]*_test|conftest)\.py$')


def parse_install_hint(hint: dict | None) -> dict:
    """Extras, groups, requirements, -r files and editable dirs from a dataset's install commands / pip_packages.

    Understands `pip install` (also `python -m pip`, `uv pip`), `uv sync` and `poetry install` options."""
    out = {'extras': set(), 'all_extras': False, 'groups': set(), 'all_groups': False, 'reqs': [], 'req_files': [],
           'editable': []}
    hint = hint or {}
    cmds = []
    for c in hint.get('install') or []:
        cmds += [x.strip() for x in re.split(r'&&|;|\n', str(c or '')) if x.strip()]

    def local(target):  # '.', './pkg', '.[a,b]', 'pkg/[x]', '/testbed'
        m = re.fullmatch(r'["\']?([^\[\]"\']*)(?:\[([^\]]*)\])?["\']?', target)
        path, ex = (m.group(1), m.group(2)) if m else (target, None)
        path = re.sub(r'^(/testbed|\$PWD|\.)/?', '', path).strip('/') or '.'
        out['extras'] |= {x.strip() for x in (ex or '').split(',') if x.strip()}
        if path != '.':
            out['editable'].append(path)
    for c in cmds:
        try:
            tok = shlex.split(c)
        except ValueError:
            tok = c.split()
        if 'install' in tok and any(t in ('pip', 'pip3') or t.endswith('/pip') for t in tok[:tok.index('install')]):
            rest, i = tok[tok.index('install') + 1:], 0
            while i < len(rest):
                t = rest[i]
                if t in ('-e', '--editable'):
                    if i + 1 < len(rest):
                        local(rest[i + 1])
                    i += 2
                    continue
                if t in ('-r', '--requirement'):
                    if i + 1 < len(rest):
                        out['req_files'].append(re.sub(r'^(/testbed/|\./)', '', rest[i + 1]))
                    i += 2
                    continue
                if t in ('-c', '--constraint', '-i', '--index-url', '--extra-index-url', '-f', '--find-links',
                         '--target', '-t', '--prefix', '--root', '--python'):
                    i += 2
                    continue
                if t.startswith('-'):
                    i += 1
                    continue
                if t.startswith(('.', '/')) or (('/' in t) and '://' not in t and not re.match(r'^[A-Za-z0-9_.-]+\[', t)):
                    local(t)
                elif '://' not in t and not t.startswith('git+'):
                    out['reqs'].append(t)
                i += 1
        elif tok[:2] == ['uv', 'sync'] or tok[:2] == ['poetry', 'install']:
            for i, t in enumerate(tok):
                nxt = tok[i + 1] if i + 1 < len(tok) else ''
                if t in ('--all-extras',):
                    out['all_extras'] = True
                elif t in ('--extra', '-E', '--extras'):
                    out['extras'] |= {x for x in re.split(r'[ ,]', nxt) if x}
                elif t in ('--group', '--with', '--only'):
                    out['groups'] |= {x for x in nxt.split(',') if x}
                elif t in ('--all-groups', '--dev'):
                    out['all_groups'] = out['all_groups'] or t == '--all-groups'
                    out['groups'].add('dev')
                elif t.startswith('--extra=') or t.startswith('--group='):
                    k, v = t[2:].split('=', 1)
                    (out['extras'] if k == 'extra' else out['groups']).add(v)
    for p in hint.get('pip_packages') or []:
        out['reqs'] += [x for x in str(p).split() if x and not x.startswith('-')]
    out['req_files'] += [str(x) for x in hint.get('reqs_path') or [] if x]
    return out


def _literal(node, names: dict):
    import ast
    if isinstance(node, ast.Name) and node.id in names:
        return _literal(names[node.id], names)
    if isinstance(node, (ast.List, ast.Tuple)):
        out = []
        for e in node.elts:
            v = _literal(e, names)
            if isinstance(v, list):
                out += v
            elif isinstance(v, str):
                out.append(v)
        return out
    if isinstance(node, ast.Dict):
        return {_literal(k, names): _literal(v, names) or [] for k, v in zip(node.keys, node.values)
                if isinstance(_literal(k, names), str)}
    if isinstance(node, ast.BinOp):  # a + b of lists
        a, b = _literal(node.left, names), _literal(node.right, names)
        return (a or []) + (b or []) if isinstance(a, list) and isinstance(b, list) else None
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def setup_py_requirements(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """install_requires (+ tests_require as extra 'tests_require') and extras_require of a setup.py, when literal."""
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [], {}
    names = {t.id: n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
             for t in n.targets if isinstance(t, ast.Name)}
    runtime, extras = [], {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, 'id', getattr(n.func, 'attr', '')) == 'setup':
            for kw in n.keywords:
                v = _literal(kw.value, names)
                if kw.arg == 'install_requires' and isinstance(v, list):
                    runtime += [x for x in v if isinstance(x, str)]
                elif kw.arg == 'extras_require' and isinstance(v, dict):
                    extras.update({k: [x for x in (vv if isinstance(vv, list) else [vv]) if isinstance(x, str)]
                                   for k, vv in v.items()})
                elif kw.arg == 'tests_require' and isinstance(v, list):
                    extras['tests_require'] = [x for x in v if isinstance(x, str)]
    return runtime, extras


def setup_cfg_requirements(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """[options] install_requires and [options.extras_require] of a setup.cfg."""
    import configparser
    cp = configparser.ConfigParser(interpolation=None)
    try:
        cp.read_string(text)
    except Exception:
        return [], {}
    lines = lambda v: [x.split('#')[0].strip() for x in v.splitlines() if x.split('#')[0].strip()]
    runtime = lines(cp.get('options', 'install_requires', fallback=''))
    extras = {k: lines(v) for k, v in cp.items('options.extras_require')} if cp.has_section('options.extras_require') else {}
    return runtime, extras


def tox_deps(text: str) -> list[str]:
    """`deps =` of tox.ini's [testenv] (not -r lines, factor-conditional lines kept without the factor)."""
    import configparser
    cp = configparser.ConfigParser(interpolation=None)
    try:
        cp.read_string(text)
    except Exception:
        return []
    out = []
    for line in cp.get('testenv', 'deps', fallback='').splitlines():
        line = line.split('#')[0].strip()
        line = re.sub(r'^[\w,{}-]+:\s*', '', line) if re.match(r'^[\w,{}-]+:\s', line) else line
        if line and not line.startswith(('-', '{', '.')):
            out.append(line)
    return out


def build_requires(files: dict) -> list[str]:
    """[build-system] requires of the project, without what the sandbox image already has."""
    from packaging.requirements import Requirement
    try:
        reqs = tomllib.loads(files.get('pyproject.toml', '')).get('build-system', {}).get('requires', [])
    except Exception:
        return []
    out = []
    for r in reqs:
        try:
            name = norm(Requirement(r).name)
        except Exception:
            continue
        if name not in IMAGE_OWNED | {'hatchling', 'poetry-core', 'pdm-backend', 'flit-core', 'editables',
                                      'setuptools', 'wheel'}:
            out.append(r)
    return out


def optional_requirements(pfiles: dict) -> list[str]:
    """Every requirement of the project's extras and dependency groups (pyproject, poetry, setup.cfg, setup.py)."""
    out: list[str] = []
    try:
        py = tomllib.loads(pfiles.get('pyproject.toml', ''))
    except Exception:
        py = {}
    for reqs in py.get('project', {}).get('optional-dependencies', {}).values():
        out += [r for r in reqs if isinstance(r, str)]
    for reqs in py.get('dependency-groups', {}).values():
        out += [r for r in reqs if isinstance(r, str)]
    poetry = py.get('tool', {}).get('poetry', {})
    sects = [poetry.get('dependencies', {}), poetry.get('dev-dependencies', {})] + \
        [g.get('dependencies', {}) for g in poetry.get('group', {}).values()]
    for sect in sects:
        for k, v in sect.items():
            if k.lower() != 'python':
                out.append(k + poetry_spec(v))
    for reqs in list(setup_cfg_requirements(pfiles.get('setup.cfg', ''))[1].values()) + \
            list(setup_py_requirements(pfiles.get('setup.py', ''))[1].values()):
        out += [r for r in reqs if isinstance(r, str)]
    return out


def tool_dev_requirements(pfiles: dict) -> list[str]:
    """Test/dev environments of project managers: hatch envs (default, *test*, dev), [tool.pdm.dev-dependencies],
    [tool.uv] dev-dependencies."""
    try:
        tool = tomllib.loads(pfiles.get('pyproject.toml', '')).get('tool', {})
    except Exception:
        return []
    out = []
    for name, env in tool.get('hatch', {}).get('envs', {}).items():
        if isinstance(env, dict) and (name in ('default', 'dev') or 'test' in name) and not NON_TEST.search(name):
            out += [r for r in env.get('dependencies', []) + env.get('extra-dependencies', []) if isinstance(r, str)]
    for name, reqs in tool.get('pdm', {}).get('dev-dependencies', {}).items():
        if not NON_TEST.search(name):
            out += [r for r in reqs if isinstance(r, str) and not r.startswith('-e')]
    out += [r for r in tool.get('uv', {}).get('dev-dependencies', []) if isinstance(r, str)]
    return out


def generic_inputs(pfiles: dict, files: dict, hint: dict) -> dict:
    """The `generic` argument of declared_requirements() for a project (pfiles) in a repository (files)."""
    rt, ex = setup_cfg_requirements(pfiles.get('setup.cfg', ''))
    rt2, ex2 = setup_py_requirements(pfiles.get('setup.py', ''))
    extras = {**ex2, **ex}
    test = tox_deps(files.get('tox.ini', '')) + list(hint['reqs']) + tool_dev_requirements(pfiles)
    req_files = [p for p in hint['req_files'] + list(hint.get('root_reqs', [])) if p in files]
    req_files += sorted(p for p in files if re.fullmatch(r'requirements/[^/]*(test|dev)[^/]*\.(txt|in)', p)
                        and not NON_TEST.search(p) and p not in req_files)
    # an application's own requirements.txt / requirements.in (project root, gold import roots) are runtime
    runtime_files = [p for p in ['requirements.txt', 'requirements.in'] + list(hint.get('root_reqs', []))
                     if p in files and re.search(r'(^|/)requirements\.(txt|in)$', p)]
    return {'extras': extras, 'runtime': rt + [r for r in rt2 if r not in rt], 'test': test,
            'wanted': TEST_EXTRAS | {'tests_require'} | hint['extras'], 'all_extras': hint['all_extras'],
            'groups': hint['groups'], 'all_groups': hint['all_groups'], 'req_files': req_files,
            'runtime_files': runtime_files}


def _flatten_requirements(text: str) -> str:
    """Join backslash continuations and drop --hash options (pip-compile --generate-hashes output)."""
    text = re.sub(r'\\\s*\n', ' ', text)
    return '\n'.join(re.sub(r'\s--hash[= ]\S+', '', line).rstrip() for line in text.splitlines())


def hard_imports(text: str) -> set[str]:
    """Top-level modules a file imports unconditionally: import statements at module level or in function bodies that
    are not inside try/except, if/else or with blocks (optional imports: try/except ImportError, `if HAS_X:`,
    `if TYPE_CHECKING:`). Relative imports are ignored."""
    import ast
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set()
    out: set[str] = set()

    def visit(body):
        for n in body:
            if isinstance(n, ast.Import):
                out.update(a.name.split('.')[0].lower() for a in n.names)
            elif isinstance(n, ast.ImportFrom) and not n.level and n.module:
                out.add(n.module.split('.')[0].lower())
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(n.body)
    visit(tree.body)
    return out


def read_snapshot_generic(tgz: Path, repo: str, project: str = '.', extra: tuple = (), roots: tuple = ()) -> dict:
    """read_snapshot() for arbitrary repositories: dependency files of the repository root (`files`) and of the
    project directory (`pfiles`, keys relative to it); requirement files in the gold files' import roots (`roots`,
    listed in `root_reqs`); imports of test files anywhere in the tree and of the first-party code they reference;
    every pyproject.toml up to 4 levels deep (`pyprojects`, by directory); directory and module names (`local`)."""
    files: dict[str, str] = {}
    pfiles: dict[str, str] = {}
    py_sources: dict[str, str] = {}
    pyprojects: dict[str, str] = {}
    dirs: set[str] = set()
    root_reqs: list[str] = []
    pre = '' if project in ('.', '') else project.strip('/') + '/'
    root_pre = tuple(r.strip('/') + '/' for r in roots if r not in ('.', ''))
    with tarfile.open(tgz) as t:
        for m in t.getmembers():
            n = m.name[2:] if m.name.startswith('./') else m.name
            if n.startswith('.git/') or n == '.git':
                continue
            if m.isdir():
                dirs.add(n.rstrip('/'))
                continue
            if not m.isfile():
                continue
            parent = n.rsplit('/', 1)[0] if '/' in n else ''
            while parent and parent not in dirs:
                dirs.add(parent)
                parent = parent.rsplit('/', 1)[0] if '/' in parent else ''
            dep = GENERIC_DEP_FILE.fullmatch(n) or n in extra
            pdep = pre and n.startswith(pre) and GENERIC_DEP_FILE.fullmatch(n[len(pre):])
            rdep = any(n.startswith(r) and re.fullmatch(r'requirements[^/]*\.(txt|in)', n[len(r):]) for r in root_pre)
            ppj = n.endswith('pyproject.toml') and n.count('/') <= 4
            if n.endswith('.py') or dep or pdep or rdep or ppj:
                text = t.extractfile(m).read().decode('utf-8', 'replace')
                if re.search(r'requirements[^/]*\.(txt|in)$', n):
                    text = _flatten_requirements(text)
                if dep or rdep:
                    files[n] = text
                if rdep:
                    root_reqs.append(n)
                if pdep:
                    pfiles[n[len(pre):]] = text
                if ppj:
                    pyprojects[n.rsplit('/', 1)[0] if '/' in n else '.'] = text
                if n.endswith('.py'):
                    py_sources[n] = text
    if not pre:
        pfiles = files
    own = own_dist(pfiles, repo if not pre else project.rstrip('/').rsplit('/', 1)[-1])
    imp_re = re.compile(r'^\s*(?:import|from)\s+([A-Za-z0-9_]+)', re.M)
    local = {d.rsplit('/', 1)[-1].lower() for d in dirs} | {n.rsplit('/', 1)[-1][:-3].lower() for n in py_sources}
    # as read_snapshot(): follow references from tests into first-party helper code (dummyserver/, docs_src/, ...)
    first_party = {n.split('/')[0] for n in py_sources if '/' in n} - {'tests', 'test', 'src'}
    ref_re = re.compile(r'\b(' + '|'.join(map(re.escape, sorted(first_party))) + r')(?:\.([A-Za-z0-9_]+))?\b') \
        if first_party else None
    scanned, todo, imports, hard = set(), {n for n in py_sources if GENERIC_TEST_FILE.search(n)}, set(), set()
    while todo:
        scanned |= todo
        prefixes = set()
        for n in todo:
            text = py_sources[n]
            imports |= {m.lower() for m in imp_re.findall(text)}
            hard |= hard_imports(text)
            for d, sub in (ref_re.findall(text) if ref_re else []):
                prefixes.add(f'{d}/{sub}' if sub else f'{d}/')
        todo = {n for n in py_sources if any(n == p + '.py' or n.startswith(p if p.endswith('/') else p + '/')
                                            for p in prefixes)} - scanned
    src_hard: set[str] = set()
    for n, text in py_sources.items():
        if not GENERIC_TEST_FILE.search(n) and (not pre or n.startswith(pre)) and not n.startswith(('docs/', 'doc/')):
            src_hard |= hard_imports(text)
    return {'files': files, 'pfiles': pfiles, 'imports': imports, 'hard_imports': hard, 'own': own, 'dirs': dirs,
            'local': local, 'pyprojects': pyprojects, 'root_reqs': root_reqs, 'source_hard_imports': src_hard}


def local_packages(pyprojects: dict[str, str], project: str) -> dict[str, str]:
    """{distribution name: directory} of the sibling packages a monorepo installs from its own tree: uv workspace
    members (root [tool.uv.workspace] members globs), [tool.uv.sources] path/workspace entries and poetry path
    dependencies of the root and the project."""
    import fnmatch
    names = {}
    for d, text in pyprojects.items():
        try:
            n = tomllib.loads(text).get('project', {}).get('name') or \
                tomllib.loads(text).get('tool', {}).get('poetry', {}).get('name')
        except Exception:
            continue
        if n:
            names[d] = norm(n)
    out: dict[str, str] = {}
    for src_dir in {'.', project or '.'}:
        try:
            py = tomllib.loads(pyprojects.get(src_dir, ''))
        except Exception:
            continue
        tool = py.get('tool', {})
        for pat in tool.get('uv', {}).get('workspace', {}).get('members', []):
            for d, n in names.items():
                if d != '.' and fnmatch.fnmatch(d, pat.strip('/')):
                    out[n] = d
        for k, v in tool.get('uv', {}).get('sources', {}).items():
            if isinstance(v, dict) and v.get('path'):
                d = os.path.normpath(os.path.join('' if src_dir == '.' else src_dir, v['path']))
                if d in names:
                    out[names[d]] = d
            elif isinstance(v, dict) and v.get('workspace') and norm(k) in names.values():
                out[norm(k)] = next(d for d, n in names.items() if n == norm(k))
        poetry = tool.get('poetry', {})
        for sect in [poetry.get('dependencies', {}), poetry.get('dev-dependencies', {})] + \
                [g.get('dependencies', {}) for g in poetry.get('group', {}).values()]:
            for k, v in sect.items():
                if isinstance(v, dict) and v.get('path'):
                    d = os.path.normpath(os.path.join('' if src_dir == '.' else src_dir, v['path']))
                    if d in names:
                        out[names[d]] = d
    return out


def poetry_spec(v) -> str:
    if isinstance(v, dict):
        v = v.get('version', '')
    v = str(v).strip()
    if not v or v == '*':
        return ''
    out = []
    for part in v.split(','):
        part = part.strip()
        m = re.fullmatch(r'([\^~])\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?', part)
        if not m:
            out.append(part if re.match(r'[<>=!~]', part) else '==' + part)
            continue
        op, a, b, c = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        lo = '.'.join(x for x in (m.group(2), b, c) if x is not None)
        if op == '^':
            hi = f'{a + 1}.0.0' if a > 0 or b is None else (f'0.{int(b) + 1}.0' if int(b) > 0 or c is None else f'0.0.{int(c) + 1}')
        else:
            hi = f'{a}.{int(b) + 1}.0' if b is not None else f'{a + 1}.0.0'
        out.append(f'>={lo},<{hi}')
    return ','.join(out)


def greedy_pins(runtime: list[str], test: list[str], cutoff: str) -> tuple[dict | None, list[str], list[str]]:
    """Resolution when the full set conflicts (generic builds): the runtime requirements (each dropped if it alone
    breaks the set), then each test requirement in order, kept when the set still resolves."""
    def ok(reqs):
        p, _ = compile_pins(reqs, cutoff, ONLY_BINARY)
        return p if p is not None else compile_pins(reqs, cutoff)[0]
    kept, dropped = [], []
    base = ok(runtime)
    if base is None:
        for r in runtime:
            if ok(kept + [r]) is not None:
                kept.append(r)
            else:
                dropped.append(r)
    else:
        kept = list(runtime)
    for r in test:
        if ok(kept + [r]) is not None:
            kept.append(r)
        else:
            dropped.append(r)
    pins = ok(kept)
    return pins, kept, dropped


def compile_pins(reqs: list[str], cutoff: str, extra_args: tuple = ()) -> tuple[dict[str, str] | None, str]:
    with tempfile.NamedTemporaryFile('w', suffix='.in', delete=False) as f:
        f.write('\n'.join(reqs) + '\n')
    cmd = ['uv', 'pip', 'compile', f.name, '--python-version', '3.13', '--python-platform',
           'x86_64-manylinux_2_28', '--exclude-newer', cutoff, '--no-header', '--no-annotate', '-q', *extra_args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(f.name)
    if r.returncode != 0:
        return None, r.stderr.strip()[-400:]
    pins = {}
    for line in r.stdout.splitlines():
        m = re.match(r'^([A-Za-z0-9_.\-]+)==(\S+)', line.strip())
        if m:
            pins[norm(m.group(1))] = m.group(2)
    return pins, ''


# ---------------------------------------------------------------- wheels
class WheelStore:
    """Finds a CPython 3.13 linux x86_64 wheel for (dist, version): published cache, then wheels_local,
    then pip download into wheels_local, then a pure-python wheel built from the sdist."""

    def __init__(self) -> None:
        from swegemma.harness.container_setup import _is_wheel_compatible_py313
        self.ok = _is_wheel_compatible_py313
        LOCAL.mkdir(parents=True, exist_ok=True)
        self.failed: dict[tuple[str, str], str] = {}
        self.cache: dict[tuple[str, str], Path] = {}
        for d in (WHEELS, LOCAL):
            for f in sorted(d.glob('*.whl')):
                if self.ok(f.name):
                    self.cache.setdefault(wheel_name_version(f.name), f)

    def latest(self, dist: str) -> Path | None:
        """Newest linux/cp313 wheel of dist (for a new dependency whose pinned version has none)."""
        if (dist, '*') in self.cache:
            return self.cache[(dist, '*')]
        before = set(LOCAL.glob('*.whl'))
        subprocess.run(['uvx', '--python', '3.13', '--from', 'pip', 'pip', 'download', '--no-deps',
                        '--only-binary=:all:', *PIP_PLATFORM, '-d', str(LOCAL), '-q', dist],
                       capture_output=True, text=True)
        new = [f for f in set(LOCAL.glob('*.whl')) - before if wheel_name_version(f.name)[0] == dist]
        if not new:
            from packaging.version import Version
            have = [f for (d, v), f in self.cache.items() if d == dist and v != '*']
            new = sorted(have, key=lambda f: Version(wheel_name_version(f.name)[1]))[-1:]
        if new and self.ok(new[0].name):
            self.cache[(dist, '*')] = new[0]
            self.cache.setdefault(wheel_name_version(new[0].name), new[0])
            return new[0]
        return None

    def get(self, dist: str, version: str) -> Path | None:
        key = (dist, version)
        if key in self.cache:
            return self.cache[key]
        if key in self.failed:
            return None
        before = set(LOCAL.glob('*.whl'))
        pip = ['uvx', '--python', '3.13', '--from', 'pip', 'pip']
        r = subprocess.run(pip + ['download', '--no-deps', '--only-binary=:all:', *PIP_PLATFORM,
                                  '-d', str(LOCAL), '-q', f'{dist}=={version}'], capture_output=True, text=True)
        if r.returncode != 0:
            with tempfile.TemporaryDirectory() as td:
                r2 = subprocess.run(pip + ['wheel', '--no-deps', '--no-binary=:all:', '-w', td, '-q',
                                           f'{dist}=={version}'], capture_output=True, text=True)
                for f in Path(td).glob('*.whl'):
                    if f.name.endswith('-none-any.whl'):
                        f.rename(LOCAL / f.name)
                if r2.returncode != 0:
                    r = r2
        for f in set(LOCAL.glob('*.whl')) - before:
            if self.ok(f.name):
                self.cache.setdefault(wheel_name_version(f.name), f)
        if key not in self.cache:
            # filename version spelling can differ (e.g. 2.0 vs 2.0.0): accept any new wheel of this dist
            new = [f for f in set(LOCAL.glob('*.whl')) - before if wheel_name_version(f.name)[0] == dist]
            if new and self.ok(new[0].name):
                self.cache[key] = new[0]
            else:
                self.failed[key] = (r.stderr or r.stdout).strip().splitlines()[-1:][0] if (r.stderr or r.stdout).strip() else 'no wheel'
                return None
        return self.cache[key]


def wheel_meta(path: Path) -> dict:
    """Top-level import names, pytest11 plugin flag and unconditional requirements of a wheel."""
    from packaging.requirements import Requirement
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        tops, reqs, plugin = set(), [], False
        for n in names:
            top = n.split('/', 1)[0]
            if top.endswith('.dist-info'):
                if n.endswith('/METADATA'):
                    for line in z.read(n).decode('utf-8', 'replace').splitlines():
                        if line.startswith('Requires-Dist:'):
                            try:
                                req = Requirement(line.split(':', 1)[1].strip())
                            except Exception:
                                continue
                            if req.marker is None or req.marker.evaluate(MARKER_ENV):
                                reqs.append(norm(req.name))
                elif n.endswith('/entry_points.txt'):
                    plugin = '[pytest11]' in z.read(n).decode('utf-8', 'replace')
                elif n.endswith('/top_level.txt'):
                    tops |= {x.strip().split('/')[0].lower() for x in z.read(n).decode().splitlines() if x.strip()}
            elif not top.endswith('.data'):
                tops.add((top[:-3] if top.endswith('.py') else top.split('.')[0]).lower())
        files = [n for n in names if not n.endswith(('.pyc', '/RECORD'))]
    return {'tops': tops, 'reqs': reqs, 'plugin': plugin, 'files': files}


@functools.lru_cache(maxsize=None)
def accepts_pytest(path: Path, version: str) -> bool:
    """False when the wheel's own requirement on pytest excludes `version` (the image's pytest)."""
    from packaging.requirements import Requirement
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith('.dist-info/METADATA'):
                for line in z.read(n).decode('utf-8', 'replace').splitlines():
                    if line.startswith('Requires-Dist:'):
                        try:
                            req = Requirement(line.split(':', 1)[1].strip())
                        except Exception:
                            continue
                        if norm(req.name) == 'pytest' and (req.marker is None or req.marker.evaluate(MARKER_ENV)):
                            return not version or req.specifier.contains(version, prereleases=True)
    return True


REMOVED_PYTEST_API = re.compile(
    r'def pytest_(?:collect_file|ignore_collect|pycollect_makemodule|report_header|report_collectionfinish)'
    r'\([^)]*\b(?:path|startdir)\b|\bpy\.io\b|\bcall_runtest_hook\b')


@functools.lru_cache(maxsize=None)
def uses_removed_pytest_api(path: Path) -> bool:
    """A pytest plugin wheel using hook arguments / helpers that pytest 9 removed (py.path `path`/`startdir` hook
    arguments, `py.io`, `_pytest.runner.call_runtest_hook`)."""
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith('.py') and '.dist-info/' not in n:
                if REMOVED_PYTEST_API.search(z.read(n).decode('utf-8', 'replace')):
                    return True
    return False


@functools.lru_cache(maxsize=None)
def wheel_requested_extras(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """(dependency, extras) for the unconditional requirements of a wheel that request extras."""
    from packaging.requirements import Requirement
    out = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith('.dist-info/METADATA'):
                for line in z.read(n).decode('utf-8', 'replace').splitlines():
                    if line.startswith('Requires-Dist:'):
                        try:
                            req = Requirement(line.split(':', 1)[1].strip())
                        except Exception:
                            continue
                        if req.extras and (req.marker is None or req.marker.evaluate(MARKER_ENV)):
                            out.append((norm(req.name), tuple(sorted(norm(e) for e in req.extras))))
    return out


@functools.lru_cache(maxsize=None)
def wheel_extra_reqs(path: Path) -> dict[str, list[str]]:
    """{extra: [dependency names]} declared by a wheel (Requires-Dist with `extra == "x"`)."""
    from packaging.requirements import Requirement
    out: dict[str, list[str]] = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith('.dist-info/METADATA'):
                for line in z.read(n).decode('utf-8', 'replace').splitlines():
                    m = re.match(r'Requires-Dist:\s*(.+)', line)
                    if not m or 'extra' not in m.group(1):
                        continue
                    try:
                        req = Requirement(m.group(1).strip())
                    except Exception:
                        continue
                    for ex in re.findall(r'extra\s*==\s*["\']([^"\']+)', str(req.marker)):
                        env = dict(MARKER_ENV, extra=ex)
                        if req.marker.evaluate(env):
                            out.setdefault(norm(ex), []).append(norm(req.name))
    return out


# ---------------------------------------------------------------- build
def image_packages(image: str = 'swebench-sandbox:latest') -> dict[str, str]:
    """Packages pip-installed in the sandbox image itself (typer, rich, pygments, shellingham, ...)."""
    r = subprocess.run(['docker', 'run', '--rm', '--platform', 'linux/amd64', '--network', 'none', image,
                        'pip', 'list', '--format', 'json'], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f'cannot list packages of {image}: {r.stderr[-300:]}')
    return {norm(p['name']): p['version'] for p in json.loads(r.stdout)}


def build(only: list[str] | None = None, tasks_path: Path | None = None, snapshots: Path | None = None,
          manifest: Path | None = None, hints_path: Path | None = None, generic: bool = False) -> None:
    """Resolve every task's environment and write the manifest. Defaults: the public tasks, their snapshots and
    MANIFEST (rules 1-4 only). generic=True adds the rules for arbitrary repositories (module docstring)."""
    from packaging.requirements import Requirement
    from swegemma.harness.container_setup import _inspect_wheels_dir
    from swegemma.models import load_tasks

    tasks_path = tasks_path or DATA / 'tasks.jsonl'
    snapshots = snapshots or DATA / 'snapshots'
    manifest = manifest or MANIFEST
    hints = {}
    if hints_path:
        hints = {h['instance_id']: h for h in map(json.loads, Path(hints_path).read_text().splitlines()) if h}
    aliases = {**GENERIC_ALIASES, **IMPORT_ALIASES} if generic else IMPORT_ALIASES
    base_wheels, large, _, _ = _inspect_wheels_dir(WHEELS)
    base_env = {wheel_name_version(p.name)[0]: p for p in list(base_wheels) + list(large.values())}
    # what a stock sandbox has: image packages, overwritten by the published cache
    present = image_packages() | {d: wheel_name_version(p.name)[1] for d, p in base_env.items()}
    store = WheelStore()
    meta_cache: dict[Path, dict] = {}
    meta = lambda p: meta_cache.setdefault(p, wheel_meta(p))
    old = json.loads(manifest.read_text()) if manifest.exists() else {'tasks': {}}
    out = {'tasks': dict(old.get('tasks', {})), 'base_files': {}}
    tasks = [t for t in load_tasks(tasks_path) if not only or t.instance_id in only]
    for i, task in enumerate(tasks, 1):
        tgz = snapshots / f'{task.instance_id}.tgz'
        extra_entry: dict = {}
        build_names: set[str] = set()
        local_names: set[str] = set()
        source_generic = ''
        if not generic:
            snap = read_snapshot(tgz, task.repo)
            own = snap['own']
            runtime_reqs, test_reqs = declared_requirements(snap['files'], own)
            pins = lock_pins(snap['files'])
            cutoff = task.created_at
        else:
            h = hints.get(task.instance_id, {})
            project = (h.get('project_roots') or ['.'])[0] or '.'
            hint = parse_install_hint(h.get('install_hint'))
            gold_roots = tuple(sorted({g['root'] for g in h.get('gold') or [] if g.get('root') not in (None, '.', '')}))
            snap = read_snapshot_generic(tgz, task.repo, project, tuple(hint['req_files']), gold_roots)
            hint['root_reqs'] = snap['root_reqs']
            own = snap['own']
            runtime_reqs, test_reqs = declared_requirements(
                snap['pfiles'], own, generic_inputs(snap['pfiles'], snap['files'], hint))
            # sibling packages of a monorepo come from the tree (editable), with their own dependencies
            loc = local_packages(snap['pyprojects'], project)
            local_names, todo = {own}, [r for r in runtime_reqs + test_reqs]
            editable_dirs: list[str] = []
            while todo:
                r = todo.pop(0)
                try:
                    name = norm(Requirement(r).name)
                except Exception:
                    continue
                if name in loc and name not in local_names:
                    local_names.add(name)
                    editable_dirs.append(loc[name])
                    try:
                        deps = tomllib.loads(snap['pyprojects'][loc[name]]).get('project', {}).get('dependencies', [])
                    except Exception:
                        deps = []
                    runtime_reqs += [d for d in deps if d not in runtime_reqs]
                    todo += deps
            strip_local = lambda reqs: [r for r in reqs if _req_name(r) not in local_names]
            runtime_reqs, test_reqs = strip_local(runtime_reqs), strip_local(test_reqs)
            # optional dependencies the tests import (`import networkx` inside a test of an optional feature)
            declared = {_req_name(r) for r in runtime_reqs + test_reqs} - {None}
            imported_g = {norm(aliases.get(m, m)) for m in snap['imports']}
            src_hard = {norm(aliases.get(m, m)) for m in snap['source_hard_imports']}
            for r in optional_requirements(snap['pfiles']):
                name = _req_name(r)
                if name and (name in imported_g or name in src_hard) and name not in declared \
                        and name not in local_names:
                    test_reqs.append(r)  # imported by tests, or unconditionally by the package's own modules
                    declared.add(name)
            breqs = build_requires(snap['pfiles'])
            build_names = {_req_name(r) for r in breqs} - {None}
            cutoff = h.get('env_cutoff') or task.created_at
            locks = [f for f in (snap['pfiles'], snap['files']) if 'uv.lock' in f or 'poetry.lock' in f]
            pins = lock_pins(locks[0]) if locks else None
            if pins is not None and breqs:  # lockfiles do not pin the build backend's plugins
                bp, _ = compile_pins(breqs, cutoff, ONLY_BINARY)
                pins = {**(bp or {}), **pins}
            if pins is None:
                runtime_reqs = runtime_reqs + [r for r in breqs if r not in runtime_reqs]
                # wheels-only resolution first: versions without a cp313 wheel would fall back to the newest wheel
                # and break the set (numba + numpy 2.5)
                pins, err = compile_pins(runtime_reqs + test_reqs, cutoff, ONLY_BINARY)
                if pins is not None:
                    source_generic = f'uv pip compile --only-binary :all: --exclude-newer {cutoff}'
            extra_entry = _roots_entry(project, h.get('gold') or [], snap['dirs'])
            for d in editable_dirs:
                if d != (project or '.').strip('/') and d != '.':
                    extra_entry.setdefault('editable', []).append(
                        {'dir': d, 'root': f'{d}/src' if f'{d}/src' in snap['dirs'] else d})
            extra_entry.update(generic=True, cutoff=cutoff, project=project)
            if local_names - {own}:
                extra_entry['local_packages'] = sorted(local_names - {own})
        source = 'lockfile' if not generic or not source_generic else source_generic
        if pins is None:
            source = f'uv pip compile --exclude-newer {cutoff}'
            pins, err = compile_pins(runtime_reqs + test_reqs, cutoff)
            if pins is None and generic:  # conflicting pins across files, one unresolvable requirement: greedy
                pins, kept_reqs, dropped = greedy_pins(runtime_reqs, test_reqs, cutoff)
                if pins is not None:
                    source = f'uv pip compile --exclude-newer {cutoff} (greedy: dropped {dropped})'
            if pins is None:  # unresolvable on py3.13: published versions stay
                source, pins = f'none (uv pip compile failed: {err})', {}
        notes: list[str] = []
        install: dict[str, Path] = {}

        heavy: set[str] = set()

        def want(dist: str, why: str) -> bool:
            ver = pins.get(dist)
            if ver is None:
                return False
            if generic and HEAVY_DIST.fullmatch(dist):  # GPU/ML stacks: GBs per sandbox, never installed
                heavy.add(dist)
                return False
            p = store.get(dist, ver)
            if p is None and dist not in present:
                p = store.latest(dist)
                if p is not None:
                    notes.append(f'{dist}=={ver} ({why}): no linux/cp313 wheel, using {p.name}')
            if p is None:
                notes.append(f'{dist}=={ver} ({why}): no linux/cp313 wheel, '
                             + ('kept published version' if dist in present else 'not installed'))
                return False
            if generic and p.stat().st_size > MAX_WHEEL_MB * 1e6:
                heavy.add(dist)
                return False
            install[dist] = p
            return True
        # rule 3: switch packages of the stock sandbox to the repo's versions
        for dist, ver in sorted(present.items()):
            if dist not in IMAGE_OWNED and dist != own and dist in pins and ver != pins[dist]:
                want(dist, 'pinned')
        # rule 4: missing runtime/extra and test dependencies, then their dependency closure
        runtime_names = {norm(Requirement(r).name) for r in runtime_reqs}
        test_names = {norm(Requirement(r).name) for r in test_reqs}
        imported = {norm(aliases.get(m, m)) for m in snap['imports']}
        skip = set(present) | IMAGE_OWNED | (TOOLING - ({'pytest-cov'} if generic else set())) | {own} | local_names
        queue = [(d, 'runtime/extra dependency') for d in sorted(runtime_names - skip)]
        queue += [(d, 'test dependency') for d in sorted(test_names - skip - runtime_names)
                  if d in imported or d.startswith('pytest-')]
        queue += [(d, 'imported by tests') for d in sorted(imported - skip - runtime_names - test_names)
                  if d in pins]  # locked/resolved and imported by tests, declared only transitively
        queue += [(d, 'build backend') for d in sorted(build_names - skip - runtime_names - test_names)]
        added = [d for d, why in queue if d not in install and want(d, why)]
        # closure also runs through stock packages the tests use (published flask lacks blinker)
        used = set(added) | ((set(present) & (runtime_names | test_names | imported)) - IMAGE_OWNED - {own})
        seen: set[str] = set()

        def close(stack):
            while stack:
                dist = stack.pop()
                if dist in seen:
                    continue
                seen.add(dist)
                path = install.get(dist) or base_env.get(dist)
                if path is None:
                    continue  # image-installed: pip already satisfied its dependencies
                deps = list(meta(path)['reqs'])
                if generic:  # extras requested by installed packages (zarr -> numcodecs[crc32c] -> crc32c)
                    for dep_name, extras in wheel_requested_extras(path):
                        dep_path = install.get(dep_name) or base_env.get(dep_name)
                        if dep_path is None and dep_name not in present and want(dep_name, f'required by {dist}'):
                            added.append(dep_name)
                            dep_path = install[dep_name]
                        if dep_path is not None:
                            deps += [d for e in extras for d in wheel_extra_reqs(dep_path).get(e, [])]
                for dep in deps:
                    if dep in IMAGE_OWNED or dep == own or dep in seen:
                        continue
                    if dep not in present and dep not in install:
                        if not want(dep, f'required by {dist}'):
                            continue
                        added.append(dep)
                    stack.append(dep)
        close(sorted(used))
        if generic:  # extras requested by the project's own requirements (pydantic[email] -> email-validator)
            for r in runtime_reqs + test_reqs:
                try:
                    req = Requirement(r)
                except Exception:
                    continue
                d = norm(req.name)
                path = install.get(d) or base_env.get(d)
                if not req.extras or path is None:
                    continue
                for dep in [x for e in req.extras for x in wheel_extra_reqs(path).get(norm(e), [])]:
                    if dep not in present and dep not in install and dep not in IMAGE_OWNED and \
                            want(dep, f'extra of {d}'):
                        added.append(dep)
                        close([dep])
        if generic:  # third-party modules the tests import that nothing declares or provides: resolve them too
            provided = {t.replace('-', '_') for t in present} | {t for p in list(install.values()) +
                                                                 list(base_env.values()) for t in meta(p)['tops']}
            for imp in sorted(snap['hard_imports']):
                dist = norm(aliases.get(imp, imp))
                if (imp in STDLIB or imp in snap['local'] or imp in provided or imp.startswith('_') or dist in skip
                        or dist in install):
                    continue
                p2, _ = compile_pins(runtime_reqs + test_reqs + [dist], cutoff, ONLY_BINARY)
                if not p2 or dist not in p2:
                    notes.append(f'{imp}: imported by tests, undeclared, unresolvable')
                    continue
                pins.update({k: v for k, v in p2.items() if k not in pins})
                if want(dist, 'imported by tests, undeclared'):
                    added.append(dist)
                    close([dist])
        if generic:  # a pytest plugin whose pinned release excludes the image's pytest: its newest release
            for dist, path in sorted(install.items()):
                if meta(path)['plugin'] and (not accepts_pytest(path, present.get('pytest', ''))
                                             or uses_removed_pytest_api(path)):
                    newest = store.latest(dist)
                    if newest is not None and newest != path and accepts_pytest(newest, present.get('pytest', '')):
                        notes.append(f'{dist}: pytest plugin {wheel_name_version(path.name)[1]} incompatible with '
                                     f'pytest {present.get("pytest")}, using {wheel_name_version(newest.name)[1]}')
                        install[dist] = newest
        # cleaned out before extraction: the repo's own released package + every package replaced
        remove = sorted(({own} & set(present)) | (set(install) & set(present)))
        for dist in remove:
            if dist in base_env:
                out['base_files'][dist] = meta(base_env[dist])['files']
        out['tasks'][task.instance_id] = {
            'repo': task.repo, 'own': own, 'pin_source': source, 'remove': remove,
            'install': sorted(str(p.relative_to(DATA)) for p in install.values()),
            'added': sorted(added), 'notes': notes, **extra_entry,
        }
        if heavy:
            out['tasks'][task.instance_id]['heavy_skipped'] = sorted(heavy)
        switched = sorted(set(install) - set(added))
        print(f'[{i}/{len(tasks)}] {task.instance_id}: pins from {source.split(" (")[0]}; switch '
              f'{len(switched)}, add {sorted(added)}; own removed={own in remove}'
              + (f'; {len(notes)} notes' if notes else '')
              + (f'; heavy skipped {sorted(heavy)}' if heavy else '')
              + (f'; editable {[e["dir"] for e in extra_entry["editable"]]}' if extra_entry.get('editable') else '')
              + (f'; pth {extra_entry["pth"]}' if extra_entry.get('pth') else ''), flush=True)
    for t in out['tasks'].values():  # entries kept from earlier partial builds
        for dist in t['remove']:
            if dist not in out['base_files'] and dist in base_env:
                out['base_files'][dist] = meta(base_env[dist])['files']
    manifest.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f'wrote {manifest} ({len(out["tasks"])} tasks); pins without a usable wheel: {len(store.failed)}')
    for (d, v), e in sorted(store.failed.items()):
        print(f'  no wheel: {d}=={v}: {e[:160]}')


def _roots_entry(project: str, gold: list[dict], dirs: set[str]) -> dict:
    """Manifest keys for import roots a provisioned environment would make importable (generic builds only).

    editable: the nested project (not the repository root: the harness installs that one), with the .pth fallback root
    (its src/ if present). pth: import roots of gold files deeper than one directory and outside that project (the
    harness's fast path already puts /workspace and every top-level directory on sys.path)."""
    entry: dict = {}
    proj = '' if project in ('.', '') else project.strip('/')
    if proj:
        entry['editable'] = [{'dir': proj, 'root': f'{proj}/src' if f'{proj}/src' in dirs else proj}]
    pth = set()
    for g in gold:
        root = g.get('root')
        if not root or root == '.' or '/' not in root or g.get('true') is None:
            continue
        if proj and (root == proj or root.startswith(proj + '/')):
            continue
        pth.add(root)
    if pth:
        entry['pth'] = sorted(pth)
    return entry


# ---------------------------------------------------------------- runtime hook
_TASK = contextvars.ContextVar('swegemma_task_env_task', default=None)
_REMOVE_SCRIPT = r'''
import json, os, re, shutil, sys
sp = sys.argv[1]
spec = json.load(open(sys.argv[2]))
norm = lambda s: re.sub(r"[-_.]+", "-", s).lower()
dirs = set()
for e in os.listdir(sp):  # pip-installed copies (sandbox image) list their files in RECORD
    rec = os.path.join(sp, e, "RECORD")
    if e.endswith(".dist-info") and norm(e[:-10].rsplit("-", 1)[0]) in spec and os.path.exists(rec):
        if not os.path.exists(os.path.join(sp, e, "direct_url.json")):
            spec[norm(e[:-10].rsplit("-", 1)[0])] += [l.split(",")[0] for l in open(rec) if not l.startswith((".", "/"))]
for dist, files in spec.items():
    for f in files:
        p = os.path.join(sp, f)
        if f.split("/", 1)[0].endswith(".dist-info"):
            continue
        if os.path.isfile(p) or os.path.islink(p):
            os.remove(p)
        d = os.path.dirname(f)
        while d:
            dirs.add(d); d = os.path.dirname(d)
    for e in os.listdir(sp):  # every dist-info of this dist (cache wheel, image install) except editables
        if e.endswith(".dist-info") and norm(e[:-10].rsplit("-", 1)[0]) == dist:
            if not os.path.exists(os.path.join(sp, e, "direct_url.json")):
                shutil.rmtree(os.path.join(sp, e), ignore_errors=True)
for d in sorted(dirs, key=lambda x: -x.count("/")):
    p = os.path.join(sp, d)
    if os.path.isdir(p):
        left = [x for x in os.listdir(p) if x != "__pycache__"]
        if not left:
            shutil.rmtree(p, ignore_errors=True)
'''
_manifest_cache: dict | None = None
ROOTS_PTH = '_swegemma_task_env_roots.pth'


def _manifest() -> dict:
    global _manifest_cache
    if _manifest_cache is None:
        path = manifest_path()
        _manifest_cache = json.loads(path.read_text()) if path.exists() else {'tasks': {}}
    return _manifest_cache


def roots_command(entry: dict) -> str:
    """Shell command for a generic entry's `editable` / `pth` keys ('' when it has none): pip-install each nested
    project editable (offline, no deps: they come from the wheels), else put its import root on sys.path via a .pth
    file; add every `pth` root."""
    pth = f'{SP}/{ROOTS_PTH}'
    cmds = []
    for e in entry.get('editable') or []:
        d, root = shlex.quote(f'/workspace/{e["dir"]}'), shlex.quote(f'/workspace/{e["root"]}')
        cmds.append(f'(python3 -m pip install -q --no-index --no-deps --no-build-isolation -e {d} >/dev/null 2>&1 '
                    f'|| echo {root} >> {pth})')
    for p in entry.get('pth') or []:
        cmds.append(f'echo {shlex.quote(f"/workspace/{p}")} >> {pth}')
    return ('cd /workspace && ' + '; '.join(cmds)) if cmds else ''


def apply_task_env(docker, container_id: str, task_id: str) -> None:
    entry = _manifest()['tasks'].get(task_id)
    if not entry:
        return
    parts = [entry['remove'], entry['install']]
    roots = roots_command(entry)
    if roots:  # generic entries only: the public tasks' keys (and markers) are unchanged
        parts += [entry.get('editable') or [], entry.get('pth') or []]
    key = hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:16]
    marker = f'.swegemma_task_env_{key}'
    if docker.exec(container_id, f'test -f {SP}/{marker}').exit_code == 0:
        return
    if entry['remove']:
        spec = {d: list(_manifest()['base_files'].get(d, [])) for d in entry['remove']}
        with tempfile.TemporaryDirectory() as td:
            sf, pf = Path(td) / 'task_env_remove.json', Path(td) / 'task_env_remove.py'
            sf.write_text(json.dumps(spec))
            pf.write_text(_REMOVE_SCRIPT)
            docker.copy_to(container_id, sf, '/tmp/')
            docker.copy_to(container_id, pf, '/tmp/')
        r = docker.exec(container_id, f'python3 -S /tmp/task_env_remove.py {SP} /tmp/task_env_remove.json; '
                                      'rm -f /tmp/task_env_remove.py /tmp/task_env_remove.json')
        if r.exit_code != 0:
            raise RuntimeError(f'task_env removal failed for {task_id}: {r.stderr or r.stdout}')
    wheels = [DATA / w for w in entry['install']]
    if wheels:
        from swegemma.harness.container_setup import _build_unpacked_wheels_tar, _extract_tar_into_site_packages
        tar = _build_unpacked_wheels_tar(wheels, f'task_env_{key}', marker)
        _extract_tar_into_site_packages(docker, container_id, tar)
    else:
        docker.exec(container_id, f'touch {SP}/{marker}')
    if roots:
        docker.exec(container_id, roots)


def install_hook() -> bool:
    """Patch swegemma (in this process) so every sandbox gets its task's environment."""
    if os.environ.get('SWEGEMMA_TASK_ENV', '1') == '0':
        print('task_env: disabled (SWEGEMMA_TASK_ENV=0), stock environment', file=sys.stderr)
        return False
    mpath = manifest_path()
    if not mpath.exists():
        raise SystemExit(f'task_env: {mpath} missing; run scripts/task_env.py build')
    from swegemma import evaluate
    from swegemma.harness import container_setup as cs

    orig_ensure = cs._ensure_container_site_packages
    if getattr(orig_ensure, '_task_env', False):
        return True

    def ensure(docker, container_id, repo, config=None):
        orig_ensure(docker, container_id, repo, config)
        task_id = _TASK.get()
        if task_id and hasattr(docker, 'extract_archive_to_container'):
            apply_task_env(docker, container_id, task_id)
    ensure._task_env = True
    cs._ensure_container_site_packages = ensure

    orig_eval = evaluate.Evaluator.evaluate_task

    async def evaluate_task(self, task, *args, **kwargs):
        token = _TASK.set(task.instance_id)
        try:
            return await orig_eval(self, task, *args, **kwargs)
        finally:
            _TASK.reset(token)
    evaluate.Evaluator.evaluate_task = evaluate_task
    shown = mpath.relative_to(ROOT) if mpath.is_relative_to(ROOT) else mpath
    print(f'task_env: per-task environments from {shown}', file=sys.stderr)
    return True


def pop_manifest_arg(args: list[str]) -> list[str]:
    """Remove `--manifest PATH` / `--manifest=PATH` from a CLI argument list and export it as TASK_ENV_MANIFEST."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a == '--manifest' and i + 1 < len(args):
            os.environ['TASK_ENV_MANIFEST'] = args[i + 1]
            i += 2
            continue
        if a.startswith('--manifest='):
            os.environ['TASK_ENV_MANIFEST'] = a.split('=', 1)[1]
        else:
            out.append(a)
        i += 1
    return out


def main(argv: list[str]) -> None:
    import argparse
    cmd = argv[0] if argv else ''
    if cmd == 'build':
        ap = argparse.ArgumentParser(prog='task_env.py build')
        ap.add_argument('ids', nargs='*')
        ap.add_argument('--tasks', type=Path, default=None, help='tasks.jsonl (default: the public tasks)')
        ap.add_argument('--snapshots', type=Path, default=None, help='snapshot dir (default: data/competition/snapshots)')
        ap.add_argument('--manifest', type=Path, default=None, help=f'output (default: {MANIFEST.relative_to(ROOT)})')
        ap.add_argument('--hints', type=Path, default=None, help='jsonl with per-task install hints (--generic)')
        ap.add_argument('--generic', action='store_true', help='rules for arbitrary repositories')
        a = ap.parse_args(argv[1:])
        build(a.ids or None, a.tasks, a.snapshots, a.manifest, a.hints, a.generic)
    elif cmd == 'show':
        ids = pop_manifest_arg(argv[1:])
        m = _manifest()
        for t in ids:
            print(t, json.dumps(m['tasks'].get(t), indent=1))
    else:
        print(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:])
