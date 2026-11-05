"""scripts/task_env.py options for other task sets (unseen100) and scripts/unseen_data.py helpers.

The public tasks' behaviour must not change: default manifest, unchanged marker keys and commands for entries without
the generic keys, declared_requirements() without `generic` identical."""

import hashlib
import io
import json
import os
import pathlib
import sys
import tarfile
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.dont_write_bytecode = True
import task_env as te  # noqa: E402
import unseen_data as ud  # noqa: E402


# ---------------------------------------------------------------- manifest selection
def test_manifest_path_default_is_the_public_manifest(monkeypatch):
    monkeypatch.delenv('TASK_ENV_MANIFEST', raising=False)
    assert te.manifest_path() == te.MANIFEST
    assert te.MANIFEST == ROOT / 'data' / 'competition' / 'wheels_local' / 'task_env.json'


def test_manifest_path_from_env(monkeypatch, tmp_path):
    m = tmp_path / 'm.json'
    m.write_text('{"tasks": {}}')
    monkeypatch.setenv('TASK_ENV_MANIFEST', str(m))
    assert te.manifest_path() == m.resolve()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('TASK_ENV_MANIFEST', 'm.json')  # relative to the cwd when it exists there
    assert te.manifest_path() == m.resolve()
    monkeypatch.setenv('TASK_ENV_MANIFEST', 'data/unseen/task_env.json')  # else relative to the repository
    assert te.manifest_path() == (ROOT / 'data' / 'unseen' / 'task_env.json').resolve()
    monkeypatch.setenv('TASK_ENV_MANIFEST', '  ')
    assert te.manifest_path() == te.MANIFEST


def test_pop_manifest_arg(monkeypatch):
    monkeypatch.delenv('TASK_ENV_MANIFEST', raising=False)
    assert te.pop_manifest_arg(['--tasks', 't.jsonl', '--concurrency', '4']) == ['--tasks', 't.jsonl',
                                                                                 '--concurrency', '4']
    assert 'TASK_ENV_MANIFEST' not in os.environ
    out = te.pop_manifest_arg(['--manifest', 'data/unseen/task_env.json', '--tasks', 'x'])
    assert out == ['--tasks', 'x'] and os.environ['TASK_ENV_MANIFEST'] == 'data/unseen/task_env.json'
    assert te.pop_manifest_arg(['a', '--manifest=b.json']) == ['a'] and os.environ['TASK_ENV_MANIFEST'] == 'b.json'


def test_manifest_cache_reads_selected_file(monkeypatch, tmp_path):
    m = tmp_path / 'm.json'
    m.write_text(json.dumps({'tasks': {'x_1': {'remove': [], 'install': []}}, 'base_files': {}}))
    monkeypatch.setenv('TASK_ENV_MANIFEST', str(m))
    monkeypatch.setattr(te, '_manifest_cache', None)
    assert list(te._manifest()['tasks']) == ['x_1']


# ---------------------------------------------------------------- run-time application
class FakeDocker:
    def __init__(self, present=()):
        self.cmds, self.present = [], set(present)

    def exec(self, cid, cmd):
        self.cmds.append(cmd)
        code = 0 if not cmd.startswith('test -f') or cmd.split()[-1] in self.present else 1
        return type('R', (), {'exit_code': code, 'stdout': '', 'stderr': ''})()

    def copy_to(self, cid, src, dst):
        self.cmds.append(f'copy {pathlib.Path(src).name} {dst}')


def legacy_key(entry):
    return hashlib.sha1(json.dumps([entry['remove'], entry['install']]).encode()).hexdigest()[:16]


def test_apply_legacy_entry_unchanged(monkeypatch):
    """An entry without editable/pth (every public task) gets the same marker and commands as before."""
    entry = {'remove': [], 'install': []}
    monkeypatch.setattr(te, '_manifest_cache', {'tasks': {'fastapi_1': entry}, 'base_files': {}})
    d = FakeDocker()
    te.apply_task_env(d, 'c', 'fastapi_1')
    marker = f'.swegemma_task_env_{legacy_key(entry)}'
    assert d.cmds == [f'test -f {te.SP}/{marker}', f'touch {te.SP}/{marker}']
    d2 = FakeDocker(present={f'{te.SP}/{marker}'})
    te.apply_task_env(d2, 'c', 'fastapi_1')  # already applied: nothing else runs
    assert d2.cmds == [f'test -f {te.SP}/{marker}']
    d3 = FakeDocker()
    te.apply_task_env(d3, 'c', 'not_in_manifest')
    assert d3.cmds == []


def test_apply_generic_entry_runs_roots(monkeypatch):
    entry = {'remove': [], 'install': [], 'editable': [{'dir': 'libs/core', 'root': 'libs/core/src'}],
             'pth': ['lib/spack']}
    monkeypatch.setattr(te, '_manifest_cache', {'tasks': {'u_1': entry}, 'base_files': {}})
    d = FakeDocker()
    te.apply_task_env(d, 'c', 'u_1')
    assert d.cmds[0].startswith('test -f') and legacy_key(entry) not in d.cmds[0]
    assert d.cmds[1].startswith('touch ')
    roots = d.cmds[2]
    assert roots == te.roots_command(entry)
    assert 'pip install -q --no-index --no-deps --no-build-isolation -e /workspace/libs/core' in roots
    assert f'|| echo /workspace/libs/core/src >> {te.SP}/{te.ROOTS_PTH}' in roots
    assert f'echo /workspace/lib/spack >> {te.SP}/{te.ROOTS_PTH}' in roots
    assert te.roots_command({'remove': [], 'install': []}) == ''


def test_roots_entry():
    gold = [{'file': 'libs/core/src/pkg/a.py', 'root': 'libs/core/src', 'true': 'pkg.a'},
            {'file': 'lib/spack/spack/x.py', 'root': 'lib/spack', 'true': 'spack.x'},
            {'file': 'keep/wm/w.py', 'root': 'keep', 'true': 'wm.w'},  # top level: harness fast path covers it
            {'file': 'src/pkg/b.py', 'root': 'src', 'true': 'pkg.b'},
            {'file': 'tools/x/s.py', 'root': 'tools/x', 'true': None}]  # not in a package
    e = te._roots_entry('libs/core', gold, {'libs', 'libs/core', 'libs/core/src'})
    assert e == {'editable': [{'dir': 'libs/core', 'root': 'libs/core/src'}], 'pth': ['lib/spack']}
    assert te._roots_entry('.', gold[2:4], set()) == {}
    assert te._roots_entry('pkgdir', [], {'pkgdir'}) == {'editable': [{'dir': 'pkgdir', 'root': 'pkgdir'}]}


# ---------------------------------------------------------------- generic requirement sources
def test_parse_install_hint_pip():
    h = te.parse_install_hint({'install': ["pip install -e . --quiet && pip install 'autoray>=0.6.11,<0.8.0' "
                                           "--quiet && pip install pytest pytest-cov flaky"],
                               'pip_packages': ['pytest-mock'], 'reqs_path': ['requirements/ci.txt']})
    assert h['reqs'] == ['autoray>=0.6.11,<0.8.0', 'pytest', 'pytest-cov', 'flaky', 'pytest-mock']
    assert h['extras'] == set() and h['editable'] == [] and h['req_files'] == ['requirements/ci.txt']


def test_parse_install_hint_editable_extras_reqfiles_uv_poetry():
    h = te.parse_install_hint({'install': ['pip install -q -r requirements/test.in', 'pip install -q -e hypothesis-python/',
                                           'pip install -q -e .[email] pytest babel', 'uv sync --extra dev --group test',
                                           'python -m pip install -e "./sub[a,b]"', 'poetry install --with docs,ci',
                                           'uv pip install --system -e /testbed[full]']})
    assert h['req_files'] == ['requirements/test.in']
    assert h['editable'] == ['hypothesis-python', 'sub']
    assert h['extras'] == {'email', 'dev', 'a', 'b', 'full'}
    assert h['groups'] == {'test', 'docs', 'ci'}
    assert h['reqs'] == ['pytest', 'babel']
    assert te.parse_install_hint(None)['reqs'] == []
    assert te.parse_install_hint({'install': ['uv sync --all-extras --all-groups']})['all_extras']


def test_setup_py_and_cfg_requirements():
    py = textwrap.dedent('''\
        from setuptools import setup
        BASE = ["requests>=2", "click"]
        setup(name="x", install_requires=BASE + ["attrs"],
              extras_require={"test": ["pytest-mock", "hypothesis"], "docs": ["sphinx"]},
              tests_require=["mock"])
    ''')
    rt, ex = te.setup_py_requirements(py)
    assert rt == ['requests>=2', 'click', 'attrs']
    assert ex == {'test': ['pytest-mock', 'hypothesis'], 'docs': ['sphinx'], 'tests_require': ['mock']}
    assert te.setup_py_requirements('def (') == ([], {})
    cfg = textwrap.dedent('''\
        [options]
        install_requires =
            numpy>=1.20  # comment
            pandas
        [options.extras_require]
        testing =
            pytest-cov
            responses
    ''')
    assert te.setup_cfg_requirements(cfg) == (['numpy>=1.20', 'pandas'], {'testing': ['pytest-cov', 'responses']})
    tox = '[testenv]\ndeps =\n    pytest\n    py311: freezegun\n    -rrequirements.txt\n    {[base]deps}\n'
    assert te.tox_deps(tox) == ['pytest', 'freezegun']


def test_build_requires_skips_image_backends():
    files = {'pyproject.toml': '[build-system]\nrequires = ["setuptools>=64", "setuptools_scm[toml]>=8", "wheel"]\n'}
    assert te.build_requires(files) == ['setuptools_scm[toml]>=8']
    files = {'pyproject.toml': '[build-system]\nrequires = ["hatchling", "hatch-vcs"]\n'}
    assert te.build_requires(files) == ['hatch-vcs']
    assert te.build_requires({}) == []


def test_declared_requirements_generic_is_additive():
    pyproject = textwrap.dedent('''\
        [project]
        name = "demo"
        dependencies = ["attrs"]
        [project.optional-dependencies]
        test = ["pytest-mock"]
        all = ["rich"]
        [dependency-groups]
        lint = ["ruff"]
        integ = ["respx"]
    ''')
    files = {'pyproject.toml': pyproject, 'requirements-dev.txt': 'hypothesis\n'}
    legacy = te.declared_requirements(files, 'demo')
    assert legacy == te.declared_requirements(files, 'demo', None)
    assert legacy == (['attrs', 'rich'], ['hypothesis'])
    hint = te.parse_install_hint({'install': ['uv sync --group integ'], 'pip_packages': ['freezegun']})
    gen = te.generic_inputs(files, files, hint)
    rt, test = te.declared_requirements(files, 'demo', gen)
    assert rt == ['attrs', 'rich', 'pytest-mock']  # + the conventional test extra
    assert set(test) == {'hypothesis', 'freezegun', 'respx'}  # + hinted group and pip package


def _tgz(tmp_path, files):
    p = tmp_path / 'snap.tgz'
    with tarfile.open(p, 'w:gz') as t:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo('./' + name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return p


def test_read_snapshot_generic_nested_project(tmp_path):
    tgz = _tgz(tmp_path, {
        'pyproject.toml': '[tool.uv.workspace]\nmembers = ["libs/*"]\n',
        'uv.lock': 'version = 1\n',
        'libs/core/pyproject.toml': '[project]\nname = "demo-core"\ndependencies = ["pydantic"]\n',
        'libs/core/requirements/test.txt': 'pytest-asyncio\n',
        'libs/core/src/demo_core/__init__.py': '',
        'libs/core/tests/unit/test_a.py': 'import respx\nfrom demo_core import x\n',
        'tools/conftest.py': 'import freezegun\n',
        '.git/HEAD': 'ref: refs/heads/main\n',
    })
    s = te.read_snapshot_generic(tgz, 'org/demo', 'libs/core')
    assert s['own'] == 'demo-core'
    assert set(s['pfiles']) == {'pyproject.toml', 'requirements/test.txt'}
    assert set(s['files']) == {'pyproject.toml', 'uv.lock'}
    assert {'respx', 'demo_core', 'freezegun'} <= s['imports']
    assert {'libs', 'libs/core', 'libs/core/src', 'libs/core/src/demo_core'} <= s['dirs']  # from file paths too
    root = te.read_snapshot_generic(tgz, 'org/demo')
    assert root['pfiles'] is root['files'] and root['own'] == 'demo'


# ---------------------------------------------------------------- unseen_data helpers
def test_iso():
    assert ud.iso('2025-06-13 10:18:30') == '2025-06-13T10:18:30Z'
    assert ud.iso('2025-06-13T10:18:30Z') == '2025-06-13T10:18:30Z'
    assert ud.iso('2025-06-13 10:18:30.123+00:00') == '2025-06-13T10:18:30Z'


def test_import_root_and_v24_name(tmp_path):
    for d in ('libs/core/src/pkg/sub', 'flat/x', 'nsroot/pkg'):
        (tmp_path / d).mkdir(parents=True)
    for f in ('libs/core/src/pkg/__init__.py', 'libs/core/src/pkg/sub/__init__.py', 'flat/__init__.py',
              'flat/x/__init__.py', 'nsroot/pkg/__init__.py'):
        (tmp_path / f).write_text('')
    assert ud.import_root(str(tmp_path), 'libs/core/src/pkg/sub/m.py') == ('pkg.sub.m', 'libs/core/src')
    assert ud.module_name_v24('libs/core/src/pkg/sub/m.py') is not None
    assert ud.module_name_v24('src/pkg/m.py') == 'pkg.m'
    assert ud.import_root(str(tmp_path), 'flat/x/m.py') == ('flat.x.m', '.')
    assert ud.import_root(str(tmp_path), 'nsroot/pkg/m.py') == ('pkg.m', 'nsroot')
    assert ud.import_root(str(tmp_path), 'scripts/run.py') == (None, 'scripts')


def test_layout_of():
    g = lambda f, root, true='m', project='.': {'file': f, 'root': root, 'true': true, 'project': project}
    assert ud.layout_of([g('pkg/a.py', '.')]) == 'flat'
    assert ud.layout_of([g('pkg/a.py', '.'), g('src/p/b.py', 'src')]) == 'src'
    assert ud.layout_of([g('libs/core/p/a.py', 'libs/core')]) == 'nested'
    assert ud.layout_of([g('sub/p/a.py', 'sub', project='sub')]) == 'nested'  # top-level dir with its own project
    assert ud.layout_of([g('keep/p/a.py', 'keep')]) == 'namespace'
    assert ud.layout_of([g('noxfile.py', '.', None), g('.github/h.py', '.github', None)]) == 'flat'
    assert ud.layout_of([{'file': 'x.py', 'missing': True}]) == 'unknown'


def test_heavy_regex():
    found = lambda t: sorted({m.lower() for m in ud.HEAVY_RE.findall(t)})
    assert found('dependencies = ["torch>=2", "numpy"]') == ['torch']
    assert found('jax[cpu]>=0.4\nnvidia-cudnn-cu12==9\n') == ['jax', 'nvidia-cudnn-cu12']
    assert found('torchmetrics>=1\npytorch-lightning\n') == ['pytorch-lightning', 'torchmetrics']
    assert found('jaxtyping\nnvidia-ml-py>=12\npyjax\n') == []


def test_optional_requirements_all_sources():
    pyproject = textwrap.dedent('''\
        [project]
        name = "mesa"
        [project.optional-dependencies]
        network = ["networkx>=3"]
        [dependency-groups]
        viz = ["matplotlib"]
        [tool.poetry.group.extra.dependencies]
        scipy = "^1.10"
    ''')
    cfg = '[options.extras_require]\nio =\n    h5py\n'
    py = 'from setuptools import setup\nsetup(extras_require={"fast": ["orjson"]})\n'
    reqs = te.optional_requirements({'pyproject.toml': pyproject, 'setup.cfg': cfg, 'setup.py': py})
    assert reqs == ['networkx>=3', 'matplotlib', 'scipy>=1.10,<2.0.0', 'h5py', 'orjson']
    assert te.optional_requirements({}) == []


def test_flatten_hashed_requirements():
    text = 'blinker==1.9.0 \\\n    --hash=sha256:aa \\\n    --hash=sha256:bb\n    # via flask\nflask==3.1.3 \\\n    --hash=sha256:cc\n'
    flat = te._flatten_requirements(text)
    assert [l.strip() for l in flat.splitlines() if l.strip() and not l.strip().startswith('#')] == \
        ['blinker==1.9.0', 'flask==3.1.3']
    rt, test = te.declared_requirements({'requirements.txt': flat}, 'demo')
    assert test == ['blinker==1.9.0', 'flask==3.1.3']


def test_tool_dev_requirements():
    pyproject = textwrap.dedent('''\
        [tool.hatch.envs.default]
        dependencies = ["python-dotenv", "django"]
        [tool.hatch.envs.hatch-test]
        extra-dependencies = ["pytest-django"]
        [tool.hatch.envs.docs]
        dependencies = ["mkdocs"]
        [tool.pdm.dev-dependencies]
        test = ["respx", "-e file:///${PROJECT_ROOT}/sub"]
        [tool.uv]
        dev-dependencies = ["freezegun"]
    ''')
    assert te.tool_dev_requirements({'pyproject.toml': pyproject}) == \
        ['python-dotenv', 'django', 'pytest-django', 'respx', 'freezegun']


def test_local_packages_workspace_and_sources():
    pyprojects = {
        '.': '[tool.uv.workspace]\nmembers = ["opentelemetry-*", "exporter/*"]\n'
             '[tool.uv.sources]\nshared = { path = "libs/shared" }\n',
        'opentelemetry-api': '[project]\nname = "opentelemetry-api"\n',
        'opentelemetry-sdk': '[project]\nname = "opentelemetry-sdk"\ndependencies = ["opentelemetry-api"]\n',
        'exporter/otlp': '[project]\nname = "opentelemetry-exporter-otlp"\n',
        'libs/shared': '[project]\nname = "shared"\n',
        'docs/examples/x': '[project]\nname = "example-x"\n',
    }
    loc = te.local_packages(pyprojects, 'opentelemetry-sdk')
    assert loc == {'opentelemetry-api': 'opentelemetry-api', 'opentelemetry-sdk': 'opentelemetry-sdk',
                   'opentelemetry-exporter-otlp': 'exporter/otlp', 'shared': 'libs/shared'}
    assert te.local_packages({'.': '[project]\nname = "x"\n'}, '.') == {}


def test_read_snapshot_generic_follows_helpers_and_root_requirements(tmp_path):
    tgz = _tgz(tmp_path, {
        'pyproject.toml': '[project]\nname = "urllib3"\n',
        'test/test_x.py': 'from dummyserver.app import thing\nimport hypothesis\n',
        'dummyserver/app.py': 'import quart\n',
        'unrelated/mod.py': 'import secretlib\n',
        'app/backend/requirements.txt': 'aiohttp==3.9 \\\n    --hash=sha256:00\n',
        'app/backend/prepdocslib/__init__.py': '',
        'pkgs/a/pyproject.toml': '[project]\nname = "a"\n',
    })
    s = te.read_snapshot_generic(tgz, 'urllib3/urllib3', roots=('app/backend',))
    assert {'dummyserver', 'hypothesis', 'quart'} <= s['imports'] and 'secretlib' not in s['imports']
    assert s['root_reqs'] == ['app/backend/requirements.txt']
    assert s['files']['app/backend/requirements.txt'].split() == ['aiohttp==3.9']
    assert set(s['pyprojects']) == {'.', 'pkgs/a'}
    assert {'dummyserver', 'app', 'prepdocslib', 'test_x', 'backend'} <= s['local']


def test_hard_imports_skip_guarded():
    text = textwrap.dedent('''\
        import os, numpy as np
        from pkg.sub import x
        from . import local
        try:
            import aesara
        except ImportError:
            aesara = None
        if HAS_JAX:
            import jax
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            import pandas
        def test_graph():
            import networkx as nx
            with ctx():
                import inside_with
        class TestX:
            def test_y(self):
                from scipy import sparse
    ''')
    assert te.hard_imports(text) == {'os', 'numpy', 'pkg', 'typing', 'networkx', 'scipy'}
    assert te.hard_imports('def (') == set()


def _wheel(tmp_path, name, version, requires=(), plugin=False):
    import zipfile
    p = tmp_path / f'{name}-{version}-py3-none-any.whl'
    meta = f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n' + ''.join(f'Requires-Dist: {r}\n'
                                                                             for r in requires)
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr(f'{name}/__init__.py', '')
        z.writestr(f'{name}-{version}.dist-info/METADATA', meta)
        if plugin:
            z.writestr(f'{name}-{version}.dist-info/entry_points.txt', f'[pytest11]\n{name} = {name}\n')
    return p


def test_wheel_extras_and_pytest_acceptance(tmp_path):
    zarr = _wheel(tmp_path, 'zarr', '3.0', ['numcodecs[crc32c,Msgpack]>=0.14', 'numpy', 'fsspec; extra == "remote"'])
    numcodecs = _wheel(tmp_path, 'numcodecs', '0.15', ['numpy', 'crc32c>=2.7; extra == "crc32c"',
                                                       'msgpack; extra == "msgpack"', 'zfpy; extra == "zfpy"'])
    assert te.wheel_requested_extras(zarr) == [('numcodecs', ('crc32c', 'msgpack'))]
    assert te.wheel_extra_reqs(numcodecs) == {'crc32c': ['crc32c'], 'msgpack': ['msgpack'], 'zfpy': ['zfpy']}
    assert te.wheel_extra_reqs(zarr) == {'remote': ['fsspec']}
    old = _wheel(tmp_path, 'pytest_asyncio', '0.21.0', ['pytest<9,>=7.0.0'], plugin=True)
    new = _wheel(tmp_path, 'pytest_asyncio', '1.2.0', ['pytest<10,>=8.2'], plugin=True)
    free = _wheel(tmp_path, 'flaky', '3.8.1', [], plugin=True)
    assert not te.accepts_pytest(old, '9.1.1') and te.accepts_pytest(new, '9.1.1')
    assert te.accepts_pytest(free, '9.1.1') and te.accepts_pytest(old, '')


def test_generic_runtime_files():
    files = {'requirements.txt': 'flask==3.1\n', 'requirements-dev.txt': 'pytest-mock\n',
             'app/backend/requirements.txt': 'azure-identity==1.2\n'}
    hint = te.parse_install_hint(None)
    hint['root_reqs'] = ['app/backend/requirements.txt']
    gen = te.generic_inputs(files, files, hint)
    assert gen['runtime_files'] == ['requirements.txt', 'app/backend/requirements.txt']
    rt, test = te.declared_requirements(files, 'demo', gen)
    assert rt == ['flask==3.1', 'azure-identity==1.2'] and test == ['pytest-mock']
    assert te.declared_requirements(files, 'demo') == ([], ['azure-identity==1.2', 'pytest-mock', 'flask==3.1'])


def test_heavy_dists():
    heavy = lambda d: bool(te.HEAVY_DIST.fullmatch(d))
    assert all(map(heavy, ['torch', 'torchvision', 'triton', 'nvidia-cudnn-cu12', 'tensorflow-cpu', 'jax', 'jaxlib',
                           'cupy-cuda12x', 'tf-keras']))
    assert not any(map(heavy, ['nvidia-ml-py', 'jaxtyping', 'xgboost', 'numpy', 'torchdiffeq']))


def test_removed_pytest_api_detection(tmp_path):
    import zipfile
    def plugin(name, code):
        p = tmp_path / f'{name}-1.0-py3-none-any.whl'
        with zipfile.ZipFile(p, 'w') as z:
            z.writestr(f'{name}/plugin.py', code)
            z.writestr(f'{name}-1.0.dist-info/METADATA', f'Name: {name}\nVersion: 1.0\n')
        return p
    assert te.uses_removed_pytest_api(plugin('old_a', 'def pytest_collect_file(path, parent):\n    pass\n'))
    assert te.uses_removed_pytest_api(plugin('old_b', 'import py\ntw = py.io.TerminalWriter()\n'))
    assert te.uses_removed_pytest_api(plugin('old_c', 'from _pytest.runner import call_runtest_hook\n'))
    assert not te.uses_removed_pytest_api(plugin('new_a', 'def pytest_collect_file(file_path, parent):\n    pass\n'))


def test_read_snapshot_generic_source_hard_imports(tmp_path):
    tgz = _tgz(tmp_path, {
        'pyproject.toml': '[project]\nname = "dmcg"\n[project.optional-dependencies]\nblack = ["black>=24"]\n',
        'dmcg/format.py': 'import black\ntry:\n    import isort\nexcept ImportError:\n    isort = None\n',
        'tests/test_f.py': 'import dmcg\n',
        'docs/conf.py': 'import sphinx_rtd_theme\n',
    })
    s = te.read_snapshot_generic(tgz, 'org/dmcg')
    assert 'black' in s['source_hard_imports'] and 'isort' not in s['source_hard_imports']
    assert 'sphinx_rtd_theme' not in s['source_hard_imports']
