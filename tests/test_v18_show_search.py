"""agent-v18 "lean": show.py, search.py, names.py, outline.py, callers.py (research/notes/13-tool-efficiency.md 4B;
13-appendix/code-review.md show/search/outline/callers/names, P8; audit.md #3, #6, #9).

Kaggle compacts the conversation once a prompt reaches 14,336 tokens, so outputs must average <= ~480 tokens; show.py
was 60% of all tool output (814 tokens per read; docstrings 18% of its line characters, padded line numbers 15%, the
names block 12% with 18% of its entries ever used).
1. show.py code lines are `123  code` (no left padding; two spaces, which edit.py's SHOW_PREFIX strips when the model
   copies them into OLD/NEW).
2. Every docstring is folded to its first line plus `... (docstring lines a-b)`, except in an explicit start/end range
   that holds it whole and it has at most 8 lines (E039: explicit ranges are otherwise printed whole).
3. Signatures longer than 8 lines fold to the def line plus one line of parameter names with a ready call.
4. A symbol read of a function longer than 60 lines: its head and a 40-line window around the lines with the issue's
   terms (the locate query), then `rest: show.py ...`; "start" alone: the def line and 20 lines on each side.
5. One `names:` line of at most 3 task-related names, never one printed earlier in the task (sqlite meta 'show:names',
   until a compaction removes it) or defined in lines already read, none for files outside locate's candidates.
6. Output capped at 3,500 characters (an explicit range 4,200) with a ready call for the rest.
7. search.py: zero hits run the fallbacks in the same call (ignoring case, the identifier split into words, non-.py
   files), labelled; 10 hits; a cap. outline.py: parameter names only, names on one line, methods wrapped when long.
   callers.py: attributes' assignments as definitions (P8), tests/docs references capped.
Run with scripts/test_skills.sh (sandbox image, py3.13).
"""

import os
import pathlib
import re
import sqlite3
import subprocess
import sys
import textwrap

import pytest

from notation import assert_notation, calls

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'agent/skills/repo-tools/scripts'
sys.path.insert(0, str(SCRIPTS))
import edit  # noqa: E402
import flow  # noqa: E402
import names  # noqa: E402
import show  # noqa: E402

CODE = re.compile(r'^(\d+)(?:  |$)')

CONSOLE = textwrap.dedent('''\
    """Console module.

    It renders things.
    More text here.
    """
    import os


    class Console:
        """A console.

        Long description line 1.
        Long description line 2.
        """

        def __init__(
            self,
            width=80,
            height=25,
            color=None,
            force_terminal=None,
            force_interactive=None,
            environ=None,
            *,
            record=False,
            **extra,
        ):
            self.width = width
            self._force_terminal = force_terminal
            self._environ = environ or os.environ
            self.is_interactive = (
                self.is_terminal if force_interactive is None else force_interactive
            )

        @property
        def is_terminal(self):
            """Short doc."""
            if self._force_terminal is not None:
                return self._force_terminal
            return self._environ.get("TERM") != "dumb"

        def render(self, text):
    ''') + ''.join(f'        step_{i} = {i}\n' for i in range(40)) + (
    '        if self.is_interactive:\n'
    '            tty_interactive = self._environ.get("TTY_INTERACTIVE")\n'
    '            if tty_interactive == "0":\n'
    '                return None\n') + ''.join(f'        tail_{i} = {i}\n' for i in range(40)) + '        return text\n'
LIVE = textwrap.dedent('''\
    from pay.console import Console


    class Live:
        def __init__(self, console=None):
            self.console = console or Console()

        def start(self):
            if self.console.is_interactive:
                return self.console.width
            return 0
    ''')
OTHER = textwrap.dedent('''\
    from pay.console import Console


    def other(console):
        return console.is_terminal and Console.render
    ''')


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / 'repo'
    for rel, text in {'pay/__init__.py': '', 'pay/console.py': CONSOLE, 'pay/live.py': LIVE,
                      'pay/other.py': OTHER}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def env_of(ws, flow_on=False):
    env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'), PYTHONDONTWRITEBYTECODE='1')
    env.pop('REPO_TOOLS_FLOW', None)
    if not flow_on:
        env['REPO_TOOLS_FLOW'] = 'off'
    return env


def run(script, ws, flow_on=False, **args):
    argv = [x for k, v in args.items() for x in ('--' + k, str(v))]
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *argv], capture_output=True, text=True,
                       env=env_of(ws, flow_on), timeout=120)
    assert 'Traceback' not in r.stderr, r.stderr
    return r.returncode, r.stdout


def seed(ws, query='', candidates=('pay/console.py',)):
    """The flow state locate.py leaves: the query and candidate files."""
    old = {k: os.environ.get(k) for k in ('WORKSPACE', 'REPO_TOOLS_DIR')}
    os.environ.update(WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'))
    try:
        conn = flow.connect()
        flow.set_meta(conn, 'query', query)
        for rank, path in enumerate(candidates):
            conn.execute('INSERT INTO candidates(rank, path, symbol, start, "end", score, evidence) '
                         'VALUES(?,?,?,?,?,?,?)', (rank, path, '', None, None, 1.0, ''))
        conn.close()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build_kb(ws):
    """The KB locate.py builds on its first call (show.py resolves obj.attr through it, never builds it)."""
    r = subprocess.run([sys.executable, str(SCRIPTS / 'kb.py')], capture_output=True, text=True, env=env_of(ws),
                       timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def payload(out):
    """The script's own output (without flow's footer lines)."""
    return '\n'.join(l for l in out.splitlines() if not l.startswith(('NEXT', 'BUDGET', '[', 'MEMO', 'note:')))


def code_lines(out):
    return [int(m.group(1)) for m in map(CODE.match, out.splitlines()) if m]


def line_of(text, needle):
    return next(i for i, l in enumerate(text.splitlines(), 1) if needle in l)


# ---------------------------------------------------------------- 1. the line format edit.py strips

def test_code_lines_are_unpadded_and_edit_strips_them(ws):
    code, out = run('show.py', ws, target='pay/console.py', start='35', end='40')
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == 'pay/console.py L35-40 of 127'
    assert lines[1] == '35      @property' and lines[2] == '36      def is_terminal(self):'
    copied = '\n'.join(lines[1:7])                                 # what the model pastes into OLD
    assert edit.SHOW_PREFIX.match(lines[1]) and edit.strip_show_prefixes(copied) == '\n'.join(
        CONSOLE.splitlines()[34:40])
    blank = run('show.py', ws, target='pay/console.py', start='6', end='8')[1].splitlines()
    assert blank[2] == '7' and edit.SHOW_PREFIX.match(blank[2])    # a blank line: the number alone
    assert flow.parse_facts('show.py', ['--target', 'pay/console.py', '--start', '35', '--end', '40'], out) == \
        dict(path='pay/console.py', label='', start=35, end=40)   # the header stays parseable


# ---------------------------------------------------------------- 2. docstrings

def test_every_docstring_folds_to_its_first_line_in_definition_reads(ws):
    code, out = run('show.py', ws, target='pay/console.py', symbol='Console')
    assert '\n10      """A console.\n... (docstring lines 11-14)\n' in out, out
    code, out = run('show.py', ws, target='pay/console.py', start='1', end='8')   # module docstring of 5 lines
    assert code_lines(out) == list(range(1, 9))                    # explicit, inside, short: whole
    code, out = run('show.py', ws, target='pay/console.py', start='9')            # start alone in the class: folded
    assert '... (docstring lines 11-14)' in out


def test_explicit_range_folds_only_long_docstrings_entirely_inside(ws):
    src = 'def f():\n    """Doc.\n' + ''.join(f'    line {i}\n' for i in range(10)) + '    """\n    return 1\n'
    (ws / 'pay/doc.py').write_text(src)
    code, out = run('show.py', ws, target='pay/doc.py', start='1', end='14')
    assert code_lines(out) == [1, 2, 14] and '... (docstring lines 3-13)' in out
    code, out = run('show.py', ws, target='pay/doc.py', start='5', end='14')       # starts inside it: asked for
    assert code_lines(out) == list(range(5, 15))
    code, out = run('show.py', ws, target='pay/doc.py', start='6')                # start alone inside it: kept
    assert 6 in code_lines(out) and 'docstring lines' not in out


# ---------------------------------------------------------------- 3. long signatures

def test_long_signature_folds_to_parameter_names_with_a_ready_call(ws):
    code, out = run('show.py', ws, target='pay/console.py', symbol='Console.__init__')
    lines = out.splitlines()
    assert lines[1] == '16      def __init__('
    assert lines[2] == ('... (signature lines 17-27 folded; parameters: self, width, height, color, force_terminal, '
                        'force_interactive, environ, *, record, **extra) full: show.py  target=pay/console.py  '
                        'start=16  end=27'), lines[2]
    assert lines[3].startswith('28          self.width = width')
    assert calls(lines[2]) == [('show.py', {'target': 'pay/console.py', 'start': '16', 'end': '27'})]
    assert_notation(lines[2])
    code, out = run('show.py', ws, target='pay/console.py', start='16', end='27')  # the ready call: all of it
    assert code_lines(out) == list(range(16, 28))


# ---------------------------------------------------------------- 4. windows

def test_long_function_symbol_read_is_head_and_window_at_the_issue_terms(ws):
    seed(ws, query='TTY_INTERACTIVE env var disables is_interactive animations')
    code, out = run('show.py', ws, target='pay/console.py', symbol='Console.render')
    hit = line_of(CONSOLE, 'tty_interactive = self._environ')
    start, end = line_of(CONSOLE, 'def render('), len(CONSOLE.splitlines())
    shown = code_lines(out)
    assert out.startswith(f'pay/console.py Console.render L{start}-{end} of {end}')
    assert shown[0] == start and hit in shown and len(shown) <= 41, shown
    assert f'... (lines {start + 1}-{shown[1] - 1} not shown)' in out
    rest = [l for l in out.splitlines() if l.startswith('... Console.render continues to line')][0]
    assert calls(rest) == [('show.py', {'target': 'pay/console.py', 'start': str(shown[-1] + 1), 'end': str(end)})]
    assert flow.parse_facts('show.py', [], out)['path'] == 'pay/console.py'


def test_long_function_without_issue_terms_starts_at_the_body(ws):
    code, out = run('show.py', ws, target='pay/console.py', symbol='Console.render')   # no locate query
    start = line_of(CONSOLE, 'def render(')
    assert code_lines(out)[:3] == [start, start + 1, start + 2] and len(code_lines(out)) == 41


def test_best_window_unit():
    lines = [f'x{i}' for i in range(100)]
    lines[69] = 'if self.is_interactive:'
    assert show.best_window(lines, 1, 100, 40, {'is_interactive': 3}) == 60          # 10 lines above the hit
    assert show.best_window(lines, 1, 100, 40, {}) == 1
    lines[95] = '# is_interactive comment'
    lines[97] = 'is_interactive = 1'
    assert show.best_window(lines, 1, 100, 40, {'is_interactive': 3}) == 60          # hits 70, 96, 98 together
    assert show.best_window(lines, 50, 60, 40, {'is_interactive': 3}) == 50


def test_issue_terms_unit(monkeypatch):
    import types
    monkeypatch.setattr(flow, 'peek', lambda: types.SimpleNamespace(conn=None))
    monkeypatch.setattr(flow, 'get_meta', lambda c, k, d=None: 'Fix TTY_INTERACTIVE in Console.isTerminal when the '
                                                               '__call__ of a class dependency streams')
    t = show.issue_terms()
    assert t['tty_interactive'] == 3 and t['isterminal'] == 3 and t['__call__'] == 3
    assert t['console'] == 1 and t['dependency'] == 1 and 'the' not in t and 'class' not in t and 'fix' not in t


def test_start_alone_is_the_def_line_and_20_lines_each_side(ws):
    hit = line_of(CONSOLE, 'tty_interactive = self._environ')
    start = line_of(CONSOLE, 'def render(')
    code, out = run('show.py', ws, target='pay/console.py', start=str(hit))
    assert code_lines(out) == [start] + list(range(hit - 20, hit + 21)), out
    assert f'... (lines {start + 1}-{hit - 21} not shown; Console.render spans L{start}-' in out
    code, out = run('show.py', ws, target='pay/console.py', start='38')        # a short definition: whole
    assert code_lines(out) == list(range(35, 41)) and out.startswith('pay/console.py Console.is_terminal L35-40')


# ---------------------------------------------------------------- 5. names

def test_names_line_at_most_three_related_never_twice(ws):
    build_kb(ws)
    seed(ws, query='TTY_INTERACTIVE is_interactive', candidates=('pay/live.py', 'pay/console.py'))
    code, out = run('show.py', ws, flow_on=True, target='pay/live.py', symbol='Live.start')
    line = [l for l in out.splitlines() if l.startswith('names: ')]
    n = line_of(CONSOLE, 'self.is_interactive = (')
    assert line == [f'names: self.console.is_interactive pay/console.py:{n} = ( [in Console.__init__] | '
                    f'self.console L6 = console or Console() [in Live.__init__] | '
                    f'self.console.width pay/console.py:28 = width [in Console.__init__]'], out
    code, out = run('show.py', ws, flow_on=True, target='pay/live.py', start='8', end='11')   # same names again
    assert 'names:' not in out, out


def test_names_skip_lines_already_read_and_files_outside_the_candidates(ws):
    seed(ws, query='is_terminal force_terminal', candidates=('pay/console.py',))
    code, out = run('show.py', ws, target='pay/console.py', symbol='Console.is_terminal')      # no task memory
    assert 'names: self._force_terminal L29 = force_terminal [in Console.__init__] | self._environ L30' in out, out
    run('show.py', ws, flow_on=True, target='pay/console.py', start='26', end='34')   # reads the assignments
    code, out = run('show.py', ws, flow_on=True, target='pay/console.py', symbol='Console.is_terminal')
    assert not [l for l in out.splitlines() if l.startswith('names:') and ('_environ' in l or '_force_terminal' in l)]
    code, out = run('show.py', ws, flow_on=True, target='pay/other.py', symbol='other')       # not a candidate
    assert 'names:' not in out


def test_names_printed_before_a_compaction_are_printed_again(ws, monkeypatch, capsys):
    """A name printed at a call whose result a compaction removed (flow State.visible_from) is printed again."""
    import types
    build_kb(ws)
    monkeypatch.setenv('WORKSPACE', str(ws))
    monkeypatch.setenv('REPO_TOOLS_DIR', str(ws.parent / 'state'))
    monkeypatch.setenv('REPO_TOOLS_FLOW', 'on')
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)')
    st = types.SimpleNamespace(conn=conn, candidates=[], viewed=[], edited_paths=[], visible_from=0, n=3)
    monkeypatch.setattr(flow, 'peek', lambda: st)
    names._MODS.clear()
    names._STATE.update(files=0, bytes=0, kb=None, kb_tried=False)
    show.print_names('pay/live.py', [(8, 11)])
    first = capsys.readouterr().out
    assert first.startswith('names: self.console.is_interactive pay/console.py:31'), first
    assert flow.get_meta(conn, 'show:names') == '{"self.console.is_interactive": 4, "self.console": 4, ' \
                                                '"self.console.width": 4}'
    show.print_names('pay/live.py', [(8, 11)])
    assert capsys.readouterr().out == ''                  # printed at call 4, still in the context
    st.visible_from, st.n = 6, 9                          # a compaction removed the results before call 6
    show.print_names('pay/live.py', [(8, 11)])
    assert capsys.readouterr().out == first


def test_names_line_unit(ws, monkeypatch):
    monkeypatch.setenv('WORKSPACE', str(ws))
    names._MODS.clear()
    line, shown = names.names_line('pay/console.py', [(36, 40)], query='force_terminal environ')
    assert shown == ['self._force_terminal', 'self._environ'], line
    assert line == ('names: self._force_terminal L29 = force_terminal [in Console.__init__] | '
                    'self._environ L30 = environ or os.environ [in Console.__init__]')
    names._MODS.clear()
    assert names.names_line('pay/console.py', [(36, 40)], skip={'self._force_terminal', 'self._environ'},
                            query='x') == ('', [])
    assert names._short_text('self.size', 'def size(self, width: int) -> int:') == '(self, width: int) -> int'
    assert names._short_text('X', 'class X(Base):') == 'class X(Base)'


def test_v19_names_line_is_deterministic_count_caps_not_a_clock(ws, monkeypatch):
    """v19: the names line depended on a 0.2 s clock (machine load changed the output of the same call); now count
    caps: MAX_USES distinct names, MAX_FILES other files of MAX_PARSE_BYTES together."""
    import time
    monkeypatch.setenv('WORKSPACE', str(ws))
    assert not hasattr(names, 'TIME_LIMIT')
    names._MODS.clear()
    want = names.names_line('pay/console.py', [(36, 40)], query='force_terminal environ')
    clock = iter(range(0, 10 ** 6, 5))   # a clock that jumps 5 s per reading: a v18 run would print nothing
    monkeypatch.setattr(time, 'perf_counter', lambda: float(next(clock)))
    names._MODS.clear()
    assert names.names_line('pay/console.py', [(36, 40)], query='force_terminal environ') == want
    monkeypatch.setattr(names, 'MAX_USES', 1)  # only the first distinct name (in source order) is resolved
    names._MODS.clear()
    assert names.names_line('pay/console.py', [(36, 40)], query='force_terminal environ')[1] == \
        ['self._force_terminal']
    monkeypatch.setattr(names, 'MAX_PARSE_BYTES', 0)  # no other file fits: imported names stay unresolved
    names._MODS.clear()
    names._STATE.update(files=0, bytes=0)
    assert names._budget_ok(1) is False and names._budget_ok(0) is True


def test_v19_show_symbol_inside_try_except(ws):
    """locate lists defs inside try/if blocks (the KB); show.py found them 'not defined' (fastapi EmailStr)."""
    (ws / 'pay/compat.py').write_text(textwrap.dedent('''\
        try:
            from email_validator import EmailStr
        except ImportError:
            class EmailStr(str):
                @classmethod
                def validate(cls, v):
                    return str(v)
        if True:
            LIMIT = 3
        '''))
    code, out = run('show.py', ws, target='pay/compat.py', symbol='EmailStr.validate')
    assert code == 0 and out.startswith('pay/compat.py EmailStr.validate L5-7'), out
    code, out = run('show.py', ws, target='pay/compat.py', symbol='LIMIT')
    assert code == 0 and '9      LIMIT = 3' in out, out


# ---------------------------------------------------------------- 6. the cap

def test_output_is_capped_with_a_ready_call_for_the_rest(ws):
    wide = 'def wide():\n' + ''.join(f'    v{i} = "{"x" * 80}"\n' for i in range(60)) + '    return 1\n'
    (ws / 'pay/wide.py').write_text(wide)
    code, out = run('show.py', ws, target='pay/wide.py', start='30')
    assert len(out) <= show.CAP and 30 in code_lines(out)
    code, out = run('show.py', ws, target='pay/wide.py', symbol='wide')
    last = out.rstrip().splitlines()[-1]
    assert len(out) <= show.CAP and last.startswith('... cut at 3500 characters; rest: show.py  target=pay/wide.py')
    shown = code_lines(out)
    assert calls(last)[0][1]['start'] == str(shown[-1] + 1)
    code, out = run('show.py', ws, target='pay/wide.py', start='1', end='50')       # explicit: up to 4,200
    assert show.CAP < len(out) <= show.EXPLICIT_CAP and code_lines(out)[-1] > shown[-1]
    out = subprocess.run([sys.executable, '-c', textwrap.dedent(f'''\
        import sys; sys.dont_write_bytecode = True; sys.path.insert(0, {str(SCRIPTS)!r})
        import flow, show
        show.main(['--target', 'pay/wide.py', '--symbol', 'wide'])
        print('FACTS', flow._FACTS)''')], capture_output=True, text=True, env=env_of(ws)).stdout
    assert f"'end': {shown[-1]}" in out.splitlines()[-1]                           # flow follows the cut


def test_capped_unit():
    show.LISTING.clear()
    show.LISTING.update(rel='a.py', end=500)
    out = 'a.py L1-300 of 500\n' + ''.join(f'{i}  {"y" * 30}\n' for i in range(1, 301)) + 'names: n L1: x\n'
    cut = show.capped(out, 1000)
    assert len(cut) <= 1000 and 'names:' not in cut and cut.rstrip().splitlines()[-1].startswith(
        '... cut at 1000 characters; rest: show.py  target=a.py  start=')
    assert show.capped('short\n', 1000) == 'short\n'


# ---------------------------------------------------------------- 7. search, outline, callers

def test_search_zero_hits_runs_the_fallbacks_in_the_same_call(ws):
    code, out = run('search.py', ws, text='isInteractive')
    assert out.startswith("NOTE: no matches for 'isInteractive'; showing matches for the words is interactive in "
                          "this order (any case and separator)"), out
    assert 'pay/console.py:31 Console.__init__:\n' in out and '31> ' in out          # 3 hits: with context
    code, out = run('search.py', ws, text='interactiveTty')                       # all words, any order
    assert "showing matches for lines with all of interactive, tty (any case)" in out and 'pay/console.py:' in out
    code, out = run('search.py', ws, text='Tty_Interactive')
    assert "showing matches ignoring case" in out
    (ws / 'docs').mkdir()
    (ws / 'docs/index.md').write_text('Set NO_SPINNER to hide spinners.\n')
    code, out = run('search.py', ws, text='NO_SPINNER')
    assert 'no .py file matches; 1 match in other files (.md):' in out
    code, out = run('search.py', ws, text='zz_nowhere')
    assert code == 1 and out.startswith("no matches for 'zz_nowhere' (none either ignoring case, as separate "
                                        "words, in non-.py text files)")


def test_search_ident_words_unit():
    import search
    assert search.ident_words('isCoroutineCallable') == ['is', 'coroutine', 'callable']
    assert search.ident_words('TTY_INTERACTIVE') == ['tty', 'interactive']
    assert search.ident_words('HTTPServer') == ['http', 'server']
    assert search.ident_words('two words') == [] and search.ident_words('x') == ['x']


def test_search_ten_hits_and_cap(ws):
    (ws / 'pay/many.py').write_text(''.join(f'N{i} = "needle {"z" * 150}"\n' for i in range(40)))
    code, out = run('search.py', ws, text='needle')
    assert len([l for l in out.splitlines() if l.startswith('pay/many.py:')]) == 10
    assert all(len(l) <= 140 for l in out.splitlines()[:10]) and len(out) <= 1500
    import search
    assert search.CAP == 3500 and search.MAX_HITS == 10


def test_outline_is_lean(ws):
    code, out = run('outline.py', ws, files='pay/console.py')
    lines = out.splitlines()
    assert lines[0] == 'pay/console.py (127 lines)'
    assert '    L16-33  def __init__(self, width, height, color, force_terminal, ...)' in lines, lines
    assert '    L36-40  def is_terminal(self)' in lines
    many = ''.join(f'class C{c}:\n' + ''.join(f'    def m{i}(self, a, b):\n        return a\n\n' for i in range(12))
                   for c in range(5)) + 'X = 1\nY = 2\n'
    (ws / 'pay/many.py').write_text(many)
    code, out = run('outline.py', ws, files='pay/many.py')
    n = len(many.splitlines())
    assert f'  names: X L{n - 1}, Y L{n}' in out and '  L1-36  class C0' in out
    assert '    methods: m0 L2-3, m1 L5-6, m2 L8-9' in out and 'def m0' not in out     # > 40 definitions
    code, out = run('outline.py', ws, files='pay/many.py', grep='m11')
    assert 'def m11(self, a, b)' in out and 'm10' not in out


def test_callers_lists_attribute_assignments_and_caps_tests(ws):
    (ws / 'tests').mkdir()
    (ws / 'tests/test_live.py').write_text('from pay.live import Live\n\n\n' + ''.join(
        f'def test_{i}():\n    assert Live().console.is_interactive is not {i}\n\n\n' for i in range(9)))
    code, out = run('callers.py', ws, name='is_interactive')
    lines = out.splitlines()
    n = line_of(CONSOLE, 'self.is_interactive = (')
    assert lines[0].startswith('is_interactive: 1 definition(s), ')
    assert lines[1] == f'  attribute pay/console.py:{n} Console.is_interactive in Console.__init__  | ' \
                       'self.is_interactive = ('
    assert len([l for l in lines if l.startswith('  tests/')]) == 5
    assert lines[-1] == '  ... 4 more references: tests/test_live.py (4)'
