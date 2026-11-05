"""agent-v18 "lean" locate.py (research/notes/13-tool-efficiency.md B and D; code-review P1, audit #4 and section 3).

- P1: functions the issue names (bare identifiers defined in at most NAMED_MAX_FILES source files, or a dotted
  qualname, dunders included) come first within their file, in the order the issue mentions them; all of them are
  listed with their line ranges; the best match is the first of them.
- Preview: at most PREVIEW_LINES lines, only when confident (named, or a clear file and function margin); the def
  line plus the window of body lines that mention the issue's terms; long docstrings and signatures folded to one
  marker line each.
- No names block, no `identifiers:`/`signal:` lines; low confidence, a past commit and the tests on one meta line.
- Facts for flow.py through flow.fact: candidates, identifier_count, low_confidence, thin, shown.
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from notation import calls

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'agent/skills/repo-tools/scripts'
sys.dont_write_bytecode = True  # never leave .pyc files in agent/ (the harness rejects them)
sys.path.insert(0, str(SCRIPTS))

MODELS = textwrap.dedent('''\
    """Dependency models."""
    import inspect


    class Dependant:
        def __init__(self, call=None, scope=None):
            self.call = call
            self.scope = scope

        @property
        def is_gen_callable(self) -> bool:
            if inspect.isgeneratorfunction(self.call):
                return True
            return False

        @property
        def is_async_gen_callable(self) -> bool:
            if inspect.isasyncgenfunction(self.call):
                return True
            return False

        @property
        def is_coroutine_callable(self) -> bool:
            if inspect.iscoroutinefunction(self.call):
                return True
            return False

        @property
        def computed_scope(self):
            if self.scope:
                return self.scope
            if self.is_gen_callable or self.is_async_gen_callable or self.is_coroutine_callable:
                return "request"
            if self.is_gen_callable and self.is_async_gen_callable and self.is_coroutine_callable:
                return "function"
            return None
    ''')

SIGNATURE = ''.join(f'    option_{i}=None,\n' for i in range(8))
BODY = ''.join(f'    step_{i} = prepare(step_{i - 1 if i else 0})\n' for i in range(40))
CLIENT = ('"""HTTP client."""\n\n\n'
          'def prepare(x):\n    return x\n\n\n'
          'def send_request(\n    url,\n' + SIGNATURE + '):\n'
          '    """Send a request.\n\n    Retries on failure.\n    More text.\n    """\n'
          + BODY.replace('    step_30 = prepare(step_29)\n',
                         '    step_30 = prepare(step_29)\n    attempts = retry_budget - 1\n')
          + '    return step_39\n')

LEFT = 'def draw_box(width):\n    return "-" * width\n'
RIGHT = 'def draw_bar(width):\n    return "=" * width\n'
COMMON = 'def get_value(x):\n    return x\n'


def git(ws, *args):
    subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True,
                   capture_output=True)


@pytest.fixture
def ws(tmp_path):
    ws = tmp_path / 'ws'
    files = {'pkg/__init__.py': '', 'pkg/models.py': MODELS, 'pkg/client.py': CLIENT, 'draw/left.py': LEFT,
             'draw/right.py': RIGHT, 'tests/test_models.py': 'from pkg.models import Dependant\n\n\n'
                                                            'def test_scope():\n    assert Dependant().scope is None\n'}
    for i in range(4):
        files[f'common/c{i}.py'] = COMMON
    for rel, text in files.items():
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text(text)
    git(ws, 'init', '-q')
    git(ws, 'add', '-A')
    git(ws, 'commit', '-qm', 'baseline')
    return ws


FACTS_RUNNER = '''
import json, sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import flow
got = {}
flow.fact = lambda **kw: got.update(kw)
import locate
code = locate.main(sys.argv[2:])
print('FACTS=' + json.dumps(got))
sys.exit(code)
'''


def locate(ws, **args):
    """locate.py in the object form, flow off; returns (exit code, output without the facts line, facts)."""
    argv = [x for k, v in args.items() for x in ('--' + k, str(v))]
    env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'), PYTHONDONTWRITEBYTECODE='1',
               REPO_TOOLS_FLOW='off')
    r = subprocess.run([sys.executable, '-c', FACTS_RUNNER, str(SCRIPTS), *argv], capture_output=True, text=True,
                       env=env, timeout=120)
    assert not r.stderr.strip(), r.stderr
    out, _, facts = r.stdout.rpartition('FACTS=')
    return r.returncode, out, json.loads(facts)


def file_lines(out):
    return [ln for ln in out.splitlines() if ln[:1].isdigit()]


def preview(out):
    lines = out.splitlines()
    i = next(i for i, ln in enumerate(lines) if ln.startswith('--- best match: '))
    return lines[i + 1:]


# ---------------------------------------------------------------- P1: named functions first

def test_p1_named_functions_beat_a_similar_body(ws):
    """fastapi_14458: computed_scope uses all three names (BM25 best), the issue names the three functions."""
    code, out, facts = locate(ws, query='is_coroutine_callable is_async_gen_callable is_gen_callable')
    assert code == 0
    assert file_lines(out)[0].split()[1] == 'pkg/models.py'
    assert '[defines 3 issue names]' in file_lines(out)[0]
    named = {'Dependant.is_coroutine_callable', 'Dependant.is_async_gen_callable', 'Dependant.is_gen_callable'}
    fns = out.split('\n', 1)[1].splitlines()[:3]
    assert {f.split()[0] for f in fns} == named, out  # the named ones first (v19: the most similar first), with ranges
    assert all(f.startswith('      ') and f.split()[1].startswith('L') for f in fns)
    best = fns[0].split()
    assert f'--- best match: pkg/models.py {best[0]} {best[1]} (named in the issue)' in out, out
    assert [c[1] for c in facts['candidates'][:3]] == [f.split()[0] for f in fns]
    assert facts['shown'] == ['pkg/models.py'] + [int(x) for x in best[1][1:].split('-')]
    assert facts['low_confidence'] is False


def test_p1_without_the_named_preference_bm25_picks_the_user(ws, monkeypatch):
    """The test above is meaningful: on similarity alone, the function using the names wins."""
    monkeypatch.setenv('REPO_TOOLS_DIR', str(ws.parent / 'state'))
    import kb
    import locate as loc
    kb._OPEN.clear()
    conn = kb.open(str(ws))
    text = 'is_coroutine_callable is_async_gen_callable is_gen_callable'
    monkeypatch.setattr(loc, 'named_functions', lambda *a: {})
    ranked, best_fn, *_ = loc.rank(str(ws), text, conn=conn)
    assert best_fn[ranked[0]][0][1][1] == 'Dependant.computed_scope'
    monkeypatch.undo()
    monkeypatch.setenv('REPO_TOOLS_DIR', str(ws.parent / 'state'))
    ranked, best_fn, _, q, _ = loc.rank(str(ws), text, conn=conn)
    assert best_fn[ranked[0]][0][1][1] in {'Dependant.is_coroutine_callable', 'Dependant.is_async_gen_callable',
                                           'Dependant.is_gen_callable'}
    conn.close()
    kb._OPEN.clear()


def test_p1_named_functions_unit(ws, monkeypatch):
    monkeypatch.setenv('REPO_TOOLS_DIR', str(ws.parent / 'state'))
    import kb
    import locate as loc
    kb._OPEN.clear()
    conn = kb.open(str(ws))
    paths = {p for (p,) in conn.execute('SELECT path FROM files')}

    def named(text):
        return {p: [fn[1] for fn in fns] for p, fns in loc.named_functions(conn, text, loc.parse_issue(text),
                                                                             paths).items()}

    assert named('`get_value` returns None') == {}               # defined in 4 files: says nothing about where
    assert named('`__init__` fails') == {}                        # a bare dunder never counts
    assert named('Dependant.__init__ fails') == {'pkg/models.py': ['Dependant.__init__']}  # ... a dotted one does
    assert named('pkg.models.Dependant.computed_scope is wrong') == {'pkg/models.py': ['Dependant.computed_scope']}
    assert named('`send_request` and `prepare`') == {'pkg/client.py': ['send_request', 'prepare']}
    assert named('`test_scope` breaks') == {}                     # tests are never "the place"
    conn.close()
    kb._OPEN.clear()


# ---------------------------------------------------------------- preview: short, focused, only when confident

def test_preview_window_at_the_issue_terms_with_folds(ws):
    code, out, facts = locate(ws, query='retry_budget is ignored by `send_request` pkg/client.py')
    assert '--- best match: pkg/client.py send_request L8-' in out, out
    rows = preview(out)
    assert len(rows) <= 25
    assert rows[0] == '    8  def send_request('
    assert rows[1].strip() == '... (L9-18: signature)'                     # 11-line signature: one marker
    assert rows[2].strip().startswith('... (L19-23: docstring "Send a request.")')  # docstring: one marker
    assert any('attempts = retry_budget - 1' in r for r in rows)            # the line the issue is about
    assert rows[3].strip().startswith('... (L24-') and rows[-1].strip().startswith('... (L')  # where the rest is
    assert not any('Retries on failure' in r for r in rows)
    shown = facts['shown']
    window = [int(r.split()[0]) for r in rows[4:-1]]
    assert shown == ['pkg/client.py', window[0], window[-1]] and window == list(range(window[0], window[-1] + 1))


def test_preview_whole_short_function_and_full_shown_range(ws):
    code, out, facts = locate(ws, query='Dependant.computed_scope returns request')
    rows = preview(out)
    assert rows[0].endswith('def computed_scope(self):') and rows[-1].endswith('return None')
    assert not any(r.strip().startswith('...') for r in rows)
    assert facts['shown'] == ['pkg/models.py', 29, 36]


def test_no_preview_when_not_confident(ws):
    code, out, facts = locate(ws, query='the width of drawn things is wrong')
    assert code == 0 and facts['low_confidence'] is True, out
    assert '--- best match' not in out and 'shown' not in facts
    assert 'low confidence: the top files score close' in out
    assert facts['candidates']  # the list still goes to flow


def test_pick_best_unit():
    import locate as loc
    fn_a, fn_b = ('a.py', 'f', 1, 5), ('a.py', 'g', 7, 9)
    q = {'named': {}}
    assert loc.pick_best(['a.py', 'b.py'], {'a.py': [(1.0, fn_a), (0.5, fn_b)]}, q, low=False) == (fn_a, True)
    assert loc.pick_best(['a.py', 'b.py'], {'a.py': [(1.0, fn_a), (0.95, fn_b)]}, q, low=False) == (fn_a, False)
    assert loc.pick_best(['a.py', 'b.py'], {'a.py': [(1.0, fn_a), (0.5, fn_b)]}, q, low=True) == (fn_a, False)
    named = {'named': {'a.py': [fn_a]}}
    assert loc.pick_best(['a.py', 'b.py'], {'a.py': [(1.0, fn_a), (0.99, fn_b)]}, named, low=False) == (fn_a, True)
    # v19: a named function in a top file that does not clearly lead is not previewed either (external 17% right)
    assert loc.pick_best(['a.py', 'b.py'], {'a.py': [(1.0, fn_a), (0.99, fn_b)]}, named, low=True) == (fn_a, False)
    assert loc.pick_best(['a.py'], {'a.py': []}, q, low=False) == (None, False)


def test_preview_lines_unit():
    import ast
    import locate as loc
    src = 'def f(a):\n    """One line."""\n' + ''.join(f'    x{i} = {i}\n' for i in range(40)) + '    return special\n'
    lines = src.split('\n')
    tree = ast.parse(src)
    q = {'idents': {'special'}, 'tokens': ['special']}
    rows = loc.preview_lines(lines, tree, 1, 43, q)
    assert len(rows) <= loc.PREVIEW_LINES and rows[0] == (1, 'def f(a):') and rows[1] == (2, '    """One line."""')
    assert rows[-1] == (43, '    return special')            # the window ends at the term (no marker after it)
    assert rows[2] == (None, '... (L3-21)') and len(rows) == loc.PREVIEW_LINES  # no marker after: one more line
    short = loc.preview_lines(lines[:5] + [''], ast.parse('\n'.join(lines[:5])), 1, 5, q)
    assert [r[0] for r in short] == [1, 2, 3, 4, 5]           # fits: printed whole, no markers


# ---------------------------------------------------------------- output diet

def test_no_names_block_no_signal_lines_one_meta_line(ws):
    code, out, facts = locate(ws, query='Dependant.computed_scope returns request')
    for gone in ('Names used here', 'identifiers:', 'signal:', 'LOW CONFIDENCE', 'history:'):
        assert gone not in out
    meta = [ln for ln in out.splitlines() if 'tests: ' in ln]
    assert len(meta) == 1 and meta[0].endswith('tests: run.py  test=tests/test_models.py')
    assert calls(meta[0]) == [('run.py', {'test': 'tests/test_models.py'})]
    assert len(file_lines(out)) <= 4   # k defaults to 4 files


def test_facts_for_flow(ws):
    code, out, facts = locate(ws, query='is_coroutine_callable is_async_gen_callable is_gen_callable')
    assert facts['identifier_count'] == 3 and facts['thin'] is False
    c = facts['candidates'][0]
    assert c[0] == 'pkg/models.py' and c[1] in {'Dependant.is_coroutine_callable', 'Dependant.is_async_gen_callable',
                                                 'Dependant.is_gen_callable'} and c[5] == 'defines 3 issue names'
    code, out, facts = locate(ws, query='fix padding')  # nothing code-like, few words: thin
    assert facts['identifier_count'] == 0 and facts['thin'] is True
    code, out, facts = locate(ws, query='fix padding', show='no')
    assert '--- best match' not in out and 'shown' not in facts


def test_k_object_form(ws):
    code, out, facts = locate(ws, query='width draw', k='2')
    assert len(file_lines(out)) == 2
    assert {c[0] for c in facts['candidates']} == {ln.split()[1] for ln in file_lines(out)}
