"""agent-v19 locate.py ranking (scripts/locate_bench.py: 299 SWE-bench-train + 112 local tasks).

- File and package names: BM25 of the query over each file's last two path components (path_scores).
- A mentioned path names files at a path boundary (main.py is not domain.py); a bare file name mentioned (main.py,
  __init__.py, common.py from a traceback) is worth MENTION_WEIGHT divided by the number of files it can name.
- Vendored code (_vendor/, third_party/, ...) and Django-style tests.py modules are auxiliary like tests and docs; a
  tools/ or scripts/ directory below the top level is source (conan/tools/). Not demoted: a non-test file the issue
  mentions by path (setup.py), docs/examples when the issue is about docs or examples.
- Named functions: a plain word the reporter's code calls (np.array(...), display(...)) names nothing; several named
  functions in one file are listed dotted ones first, then the most similar to the issue first (not mention order).
- Error and quoted text of the issue found as a phrase among a file's string literals: 1 / the number of such files.
- File names with an extension (meson.build, setup.py) are not dotted module names; a module counts once per file;
  dunders (__init__ from __init__.py) never count as "defines"; "defines X" evidence only for rare definitions.
- A def nested in a def is ranked, listed and previewed as its outermost enclosing def (request_response, not
  request_response.app.inner): the unit the model reads and edits.
- Each file's signals are kept in q['features'] (for the benchmark).
"""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'agent/skills/repo-tools/scripts'
sys.dont_write_bytecode = True  # never leave .pyc files in agent/ (the harness rejects them)
sys.path.insert(0, str(SCRIPTS))

TASKINSTANCE = textwrap.dedent('''\
    class TaskInstance:
        def handle_failure(self, error):
            self.state = "failed"
            self.email_alert(error)

        def email_alert(self, error):
            subject, body = self.get_email_subject_content(error)
            send(subject, body)

        def get_email_subject_content(self, error):
            subject_template = load("subject_template")
            html_content = load("html_content_template")
            return render(subject_template, error), render(html_content, error)
    ''')

FILES = {
    'app/__init__.py': '',
    'app/models/__init__.py': '',
    'app/models/taskinstance.py': TASKINSTANCE,
    'app/file_proxy.py': 'class FileProxy:\n    def fileno(self):\n        return self.file.fileno()\n',
    'app/live.py': ('def refresh(console):\n    if console.isatty():\n        console.isatty()\n'
                    '        return console.isatty()\n'),
    'app/cli/main.py': 'def main():\n    run()\n',
    'app/domain.py': 'def lookup(name):\n    return name\n',
    'app/numeric.py': ('def array(values):\n    return list(values)\n\n\n'
                       'def array_to_list(values):\n    return list(values)\n'),
    'app/paths.py': ('def absolute(path):\n    if not path.startswith("/"):\n'
                     '        raise ValueError("Tried to form an absolute path to a source dir")\n    return path\n'),
    'meson.py': 'def build():\n    return 1\n',
    'app/_vendor/six.py': 'def array(values):\n    return values\n',
    'app/blog/tests.py': 'def test_lookup():\n    assert True\n',
    'setup.py': 'from setuptools import setup\n\nsetup(name="app", install_requires=["click>=7"])\n',
    'docs_src/license/tutorial001.py': 'def license_info():\n    return {"identifier": "MIT", "name": "license"}\n',
    'app/tools/apple.py': 'def fix_apple_shared_install_name(conanfile):\n    return conanfile\n',
    'app/routing.py': ('def request_response(func):\n    async def app(scope):\n        async def inner(request):\n'
                       '            return await func(request)\n        return inner\n    return app\n\n\n'
                       'def websocket_session(func):\n    return func\n'),
}


def git(ws, *args):
    subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    ws = tmp_path / 'ws'
    for rel, text in FILES.items():
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text(text)
    git(ws, 'init', '-q')
    git(ws, 'add', '-A')
    git(ws, 'commit', '-qm', 'baseline')
    monkeypatch.setenv('REPO_TOOLS_DIR', str(tmp_path / 'state'))
    import kb
    import locate
    kb._OPEN.clear()
    conn = kb.open(str(ws))
    yield locate, conn, ws
    conn.close()
    kb._OPEN.clear()


def rank(loc, conn, ws, text):
    ranked, best_fn, reasons, q, score = loc.rank(str(ws), text, conn=conn)
    return ranked, best_fn, reasons, q


def test_is_aux_vendored_and_django_tests():
    import locate as loc
    assert loc.is_aux('pip/_vendor/requests/models.py') and loc.is_aux('src/third_party/x.py')
    assert loc.is_aux('wagtail/wagtaildocs/tests.py') and loc.is_aux('tests/test_a.py')
    assert not loc.is_aux('app/vendors.py') and not loc.is_aux('app/testing_utils.py')
    assert loc.is_aux('tools/train.py') and loc.is_aux('scripts/release.py') and loc.is_aux('setup.py')
    assert not loc.is_aux('conan/tools/apple/apple.py') and not loc.is_aux('mesonbuild/scripts/depfixer.py')
    assert loc.is_aux('tests/tools/helpers.py') and loc.is_aux('docs/scripts/build.py')


def test_path_matches_unit():
    import locate as loc
    assert loc.path_matches('main.py', 'compose/cli/main.py')
    assert not loc.path_matches('main.py', 'compose/cli/domain.py')            # v18: endswith matched domain.py
    assert loc.path_matches('/home/u/src/proj/conda/core/solve.py', 'conda/core/solve.py')
    assert loc.path_matches('./conda/core/solve.py', 'conda/core/solve.py')
    assert loc.path_matches('C:\\proj\\conda\\core\\solve.py', 'conda/core/solve.py')
    assert not loc.path_matches('core/solve.py', 'conda/core/resolve.py')


def test_bare_file_name_mention_is_divided(repo):
    loc, conn, ws = repo
    ranked, _, reasons, q = rank(loc, conn, ws, 'crash in __init__.py and main.py')
    f = q['features']
    inits = [p for p in ranked if p.endswith('__init__.py')]
    assert len(inits) == 2 and all(abs(f[p]['mention'] - loc.MENTION_WEIGHT / 2) < 1e-9 for p in inits)
    assert f['app/cli/main.py']['mention'] == loc.MENTION_WEIGHT
    assert 'mention' not in f.get('app/domain.py', {})
    assert not any('defines __init__' in r for rs in reasons.values() for r in rs)  # dunders never "define"


def test_path_scores_file_and_package_names():
    import locate as loc
    paths = ['rich/file_proxy.py', 'rich/live.py', 'rich/console.py', 'pkg/proxy/__init__.py', 'a/b/c/deep.py']
    s = loc.path_scores(paths, loc.tokenize('proxy isatty'))
    assert set(s) == {'rich/file_proxy.py', 'pkg/proxy/__init__.py'} and max(s.values()) == 1.0
    assert loc.path_scores(paths, loc.tokenize('deep c'))['a/b/c/deep.py'] == 1.0  # the last two components
    assert 'a/b/c/deep.py' not in loc.path_scores(paths, loc.tokenize('a b'))


def test_file_named_like_the_issue_ranks_first(repo):
    loc, conn, ws = repo
    ranked, _, _, q = rank(loc, conn, ws, 'proxy isatty')
    assert ranked[0] == 'app/file_proxy.py', (ranked[:3], q['features'])
    assert q['features']['app/file_proxy.py']['path'] == loc.PATH_WEIGHT


def test_named_functions_most_similar_first(repo):
    loc, conn, ws = repo
    text = ('handle_failure email_alert get_email_subject_content subject_template html_content_template '
            'render load')
    ranked, best_fn, _, q = rank(loc, conn, ws, text)
    assert ranked[0] == 'app/models/taskinstance.py'
    assert [x[1][1] for x in best_fn[ranked[0]][:3]][0] == 'TaskInstance.get_email_subject_content'
    assert set(q['named'][ranked[0]] and [f[1] for f in q['named'][ranked[0]]]) == {
        'TaskInstance.handle_failure', 'TaskInstance.email_alert', 'TaskInstance.get_email_subject_content'}
    # a dotted name still comes first, however similar the others are
    ranked, best_fn, _, q = rank(loc, conn, ws, text + ' TaskInstance.handle_failure')
    assert best_fn[ranked[0]][0][1][1] == 'TaskInstance.handle_failure'


def test_plain_called_word_names_nothing(repo):
    loc, conn, ws = repo
    paths = {p for (p,) in conn.execute('SELECT path FROM files')}

    def named(text):
        return {p: [fn[1] for fn in fns] for p, fns in
                loc.named_functions(conn, text, loc.parse_issue(text), paths).items()}

    assert named('x = np.array(values) fails') == {}                      # the reporter's own call
    assert named('`array` fails') == {'app/numeric.py': ['array']}       # backticked: named
    assert named('array_to_list(values) fails') == {'app/numeric.py': ['array_to_list']}
    assert named('app.numeric.array is wrong') == {'app/numeric.py': ['array']}  # dotted: named
    assert named('numeric.array is wrong') == {'app/numeric.py': ['array']}      # ... with the module's tail


def test_quoted_and_error_text_in_string_literals(repo):
    loc, conn, ws = repo
    ranked, _, reasons, q = rank(loc, conn, ws, 'meson fails with "Tried to form an absolute path to a source dir"')
    assert ranked[0] == 'app/paths.py' and 'contains error text' in reasons['app/paths.py']
    assert q['features']['app/paths.py']['error_text'] == 1.0
    ranked, _, reasons, q = rank(loc, conn, ws, 'ValueError: Tried to form an absolute path to a source dir')
    assert 'contains error text' in reasons['app/paths.py']


def test_file_names_are_not_modules(repo):
    loc, conn, ws = repo
    _, _, reasons, q = rank(loc, conn, ws, 'meson.build is ignored, see setup.py')
    assert not any(r.startswith('module') for r in reasons.get('meson.py', []))
    _, _, reasons, q = rank(loc, conn, ws, 'app.numeric.array and app.numeric.array_to_list')
    assert q['features']['app/numeric.py']['module'] == 1.0                # once per file


def test_vendored_copy_is_auxiliary(repo):
    loc, conn, ws = repo
    ranked, _, _, q = rank(loc, conn, ws, 'array of values')
    assert ranked.index('app/numeric.py') < ranked.index('app/_vendor/six.py')
    assert q['features']['app/_vendor/six.py']['aux'] == loc.AUX_FACTOR


def test_locate_output_shape_unchanged(repo):
    loc, conn, ws = repo
    env = dict(os.environ, WORKSPACE=str(ws), PYTHONDONTWRITEBYTECODE='1', REPO_TOOLS_FLOW='off')
    r = subprocess.run([sys.executable, str(SCRIPTS / 'locate.py'), '--query', 'proxy isatty fileno'],
                       capture_output=True, text=True, env=env, timeout=120)
    lines = r.stdout.splitlines()
    assert lines[0].split()[1] == 'app/file_proxy.py' and lines[0].split()[0].replace('.', '').isdigit(), r.stdout
    assert lines[1].startswith('      FileProxy.fileno  L2-3'), r.stdout


def test_nested_defs_fold_into_the_outer_def(repo):
    loc, conn, ws = repo
    text = 'inner request await func request_response.app.inner'
    ranked, best_fn, _, q = rank(loc, conn, ws, text)
    assert ranked[0] == 'app/routing.py'
    quals = [x[1][1] for x in best_fn['app/routing.py']]
    assert quals[0] == 'request_response' and sorted(quals) == ['request_response', 'websocket_session']
    assert best_fn['app/routing.py'][0][1] == ('app/routing.py', 'request_response', 1, 6)
    assert q['fold'][('app/routing.py', 'request_response.app.inner')][1] == 'request_response'
    assert [f[1] for f in q['named']['app/routing.py']] == ['request_response']  # the dotted name, folded


def test_mentioned_and_docs_files_are_not_demoted(repo):
    loc, conn, ws = repo
    ranked, _, _, q = rank(loc, conn, ws, 'setup.py: relax the click version bound install_requires')
    assert ranked[0] == 'setup.py' and 'aux' not in q['features']['setup.py'], ranked[:3]
    _, _, _, q = rank(loc, conn, ws, 'license identifier MIT')
    assert q['features']['docs_src/license/tutorial001.py']['aux'] == loc.AUX_FACTOR
    ranked, _, _, q = rank(loc, conn, ws, 'Fix example of license identifier in documentation MIT')
    assert ranked[0] == 'docs_src/license/tutorial001.py' and 'aux' not in q['features'][ranked[0]]
    ranked, _, _, q = rank(loc, conn, ws, 'fix_apple_shared_install_name ignores the tool')
    assert ranked[0] == 'app/tools/apple.py' and 'aux' not in q['features'][ranked[0]]
