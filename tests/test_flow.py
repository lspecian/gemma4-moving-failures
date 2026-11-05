"""Tests for the flow engine (agent/skills/repo-tools/scripts/flow.py, status.py, hypothesis.py).

The rules are exercised by replaying call sequences through flow.before/flow.after the way the scripts will
(letters as in the E020 trace shapes: L locate, S show, Q search, O outline, e edit, R run, C check_patch), with
REPO_TOOLS_DIR and WORKSPACE pointing at temp dirs. A few tests run the real scripts under flow.main to check
that facts are parsed from their actual output. Run with scripts/test_skills.sh (sandbox image, py3.13).

Argument schema v2/v3: the model passes args as an OBJECT ({"k": "v"} reaches a script as argv ["--k", "v"]) and
the scripts keep their positional form. The `sim` fixture replays every sequence three times: positional args, object
args, and alternating ("mixed"), so each replay also checks that both forms are the same call for the rules.
Every footer suggests calls in the quote-free notation of schema v3 (tests/notation.py), e.g.
`edit.py  target=a.py  old=<exact current lines>  new=<new lines>`: never JSON, never array args. The model-facing
key for the repository file is "target" and search.py's directory filter is "within" (never "file"/"path", which
make the model skip ADK's file_path); flow.canon() still keys them as "file" / "path" internally, and every alias
("path", "file", "file_path", "dir", "in", ...) is the same call.
"""

import os
import pathlib
import re
import subprocess
import sys
import textwrap
import time

import pytest

sys.dont_write_bytecode = True
SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'agent/skills/repo-tools/scripts'
sys.path.insert(0, str(SCRIPTS))
import flow  # noqa: E402
from notation import assert_notation, calls, render  # noqa: E402

MODULE = textwrap.dedent('''\
    """Payment helpers."""


    class Invoice:
        def __init__(self, amount):
            self.amount = amount

        def total(self, tax=0.1):
            if self.amount == None:
                return 0
            return self.amount * (1 + tax)


    def parse_header(value):
        return value.split(",")
    ''')

LOCATE_OUT = textwrap.dedent('''\
    identifiers: Invoice, total
    1.50 pay/invoice.py  [defines total]
          Invoice.total  L8-11
          Invoice.__init__  L5-6
    0.80 pay/http.py
          send  L5-6
    0.30 tests/test_invoice.py
          test_total  L4-5
    --- best match: pay/invoice.py Invoice.total L8-11
        8      def total(self, tax=0.1):
        9          if self.amount == None:
       10              return 0
       11          return self.amount * (1 + tax)
    next: if the shown function is the place, edit it with edit.py; otherwise show.py another one.
    ''')
THIN_LOCATE_OUT = LOCATE_OUT.replace('identifiers: Invoice, total', 'identifiers: (none)')


def git(ws, *args):
    subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True,
                   capture_output=True)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    (repo / 'pay').mkdir(parents=True)
    (repo / 'pay/__init__.py').write_text('')
    (repo / 'pay/invoice.py').write_text(MODULE)
    (repo / 'pay/http.py').write_text('def parse_header(v):\n    return v\n\n\ndef send(req):\n    return req\n')
    (repo / 'tests').mkdir()
    (repo / 'tests/test_invoice.py').write_text('from pay.invoice import Invoice\n\n\ndef test_total():\n'
                                                '    assert round(Invoice(10).total()) == 11\n')
    git(repo, 'init', '-q')
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'baseline')
    monkeypatch.setenv('WORKSPACE', str(repo))
    monkeypatch.setenv('REPO_TOOLS_DIR', str(tmp_path / 'state'))
    return repo


call = render  # the exact rendering of a suggested call: call('show.py', target='a.py') == 'show.py  target=a.py'


def obj(**kw):
    """argv of the object form, as run_skill_script builds it from args={...}."""
    return [x for k, v in kw.items() for x in ('--' + k, str(v))]


def suggested_args(line):
    """The args of every call suggested in a footer line, parsed with the notation's rule."""
    return [a for _, a in calls(line)]


def check_footer(text):
    """Rule 8: at most 3 lines: an optional BUDGET line first, then 'NEXT:' lines whose pre-filled calls are in
    the notation of schema v3 (script, then key=value pairs separated by two spaces; no JSON, no array args).
    v18c (rule 13): an optional MEMO line above them, not counted."""
    lines = text.splitlines()
    if lines and lines[0].startswith('MEMO: '):
        assert len(lines[0]) <= flow.MEMO_CHARS, text
        lines = lines[1:]
    assert len(lines) <= flow.FOOTER_LINES, text
    if lines and lines[0].startswith('BUDGET:'):
        lines = lines[1:]
    for line in lines:
        assert line.startswith('NEXT:'), line
        assert_notation(line)
    return text


def refusal_footer(msg):
    return '\n'.join(l for l in msg.splitlines() if l.startswith(('MEMO: ', 'NEXT:', 'BUDGET:')))


def nxt(footer):
    """The NEXT lines of a footer (without the BUDGET line)."""
    return '\n'.join(l for l in footer.splitlines() if not l.startswith(('BUDGET:', 'MEMO: ')))


class Sim:
    """Drives flow.before/after like the scripts do. Each method returns ('refused', msg) or ('ran', footer).
    form: 'positional', 'object' or 'mixed' (alternates per call) - the argv shape the letter methods use."""

    def __init__(self, form='positional'):
        self.log = []
        self.form = form
        self.i = 0

    def pick(self, positional, object_form):
        self.i += 1
        form = self.form if self.form != 'mixed' else ('object' if self.i % 2 else 'positional')
        return object_form if form == 'object' else positional

    def call(self, script, args, status=0, out='', **facts):
        msg = flow.before(script, args)
        if msg:
            self.log.append('x')
            check_footer(refusal_footer(msg))
            return 'refused', msg
        footer = flow.after(script, args, status, out, **facts)
        check_footer(footer)
        self.log.append('.')
        return 'ran', footer

    def L(self, text='Invoice total wrong when amount is None', out=LOCATE_OUT, **kw):
        return self.call('locate.py', self.pick([text], obj(query=text)), 0, out, **kw)

    def S(self, path='pay/invoice.py', start=1, end=20, status=0):
        out = f'{path} L{start}-{end} of 40\n' + ''.join(f'{i:5d}  x\n' for i in range(start, end + 1))
        args = self.pick([path, str(start), str(end)], obj(target=path, start=start, end=end))
        return self.call('show.py', args, status, out if status == 0 else 'usage: show.py')

    def Q(self, text='amount', status=0):
        return self.call('search.py', self.pick([text], obj(text=text)), status,
                         'pay/invoice.py:9:         if self.amount == None:\n')

    def O(self, path='pay/invoice.py'):
        return self.call('outline.py', self.pick([path], obj(files=path)), 0, f'{path}\n  class Invoice L4-11\n')

    def e(self, ok=False, old='if self.amount == None:', new='if self.amount is None:', closest=(9, 10), status=None):
        if status == 2:
            return self.call('edit.py', self.pick(['pay/invoice.py'], obj(target='pay/invoice.py')), 2,
                             'usage: edit.py FILE OLD NEW')
        args = self.pick(['pay/invoice.py', old, new], obj(target='pay/invoice.py', old=old, new=new))
        if ok:
            return self.call('edit.py', args, 0, 'EDITED pay/invoice.py (exact)\n')
        out = 'NOT APPLIED: OLD not found (also tried decoded escapes and ignoring indentation).\n'
        if closest:
            out += f'CLOSEST: pay/invoice.py {closest[0]} {closest[1]}\n'
        return self.call('edit.py', args, 1, out)

    def R(self, code=0, snippet='from pay.invoice import Invoice; print(Invoice(None).total())', status=0, out=None):
        if status == 2:  # malformed: a packed string / an unknown key
            return self.call('run.py', self.pick(['snippet\n' + snippet], obj(code=snippet)), 2,
                             'Run a quick check: ...')
        return self.call('run.py', self.pick(['snippet', snippet], obj(snippet=snippet)), 0,
                         f'{out or "..."}\nexit code: {code}\n')

    def Rx(self, i):
        """A snippet that reproduced something (a traceback, distinct output): not a probe for rule 3c."""
        return self.R(code=1, snippet=f'print({i})', out=f'Traceback (most recent call last):\nValueError: {i}')

    def T(self, code=1, target='tests/test_invoice.py'):
        return self.call('run.py', self.pick(['test', target], obj(test=target)), 0, f'1 failed\nexit code: {code}\n')

    def C(self, ready=True):
        out = 'changed files:\n  pay/invoice.py | 2 +-\nREADY: call submit_patch now.\n' if ready else \
            'NO CHANGES: the patch is empty. Make the source edit before submitting.\n'
        return self.call('check_patch.py', self.pick([], []), 0 if ready else 1, out)

    def H(self, cmd, *vals):
        """hypothesis.py (a real subprocess) in this Sim's argument form."""
        if cmd == 'add':
            o = obj(**{('add' if i == 0 else f'add{i + 1}'): v for i, v in enumerate(vals)})
        elif cmd == 'list':
            o = obj(list='yes')
        else:
            o = obj(**{cmd: vals[0]}, evidence=' '.join(vals[1:]))
        return run_script('hypothesis.py', *self.pick([cmd, *vals], o))


@pytest.fixture(params=['positional', 'object', 'mixed'])
def sim(ws, request):
    return Sim(request.param)


# ---------------------------------------------------------------- rule 1: locate first

def test_rule1_lookaround_before_locate_runs_and_asks_for_locate(sim):
    """v13: reads before locate run (v12 refused every other one); the footer's first NEXT is locate.py."""
    for step in (lambda: sim.Q('TTY_COMPATIBLE'), lambda: sim.O(), lambda: sim.S()):
        kind, f = step()
        assert kind == 'ran'
        assert nxt(f).startswith('NEXT: locate.py  query=<issue title copied verbatim>')
    kind, f = sim.call('callers.py', ['Console.print'], 0, 'pay/invoice.py:9 in Invoice.total\n')
    assert kind == 'ran' and 'locate.py  query=' in f and 'Console.print' in f
    assert set(sim.log) == {'.'}


def test_rule1_prefills_search_terms_into_locate(sim):
    _, msg = sim.Q('_SVG_FONT_FAMILY')
    assert 'NEXT: ' + call('locate.py', query='<issue title copied verbatim> <every identifier, path and error '
                                        'message in the issue> _SVG_FONT_FAMILY') in msg


def test_rule1_edit_run_and_locate_are_not_blocked(sim):
    assert sim.R()[0] == 'ran'
    assert sim.e(ok=True)[0] == 'ran'
    assert sim.L()[0] == 'ran'


def test_replay_rich_3777_QSSSQQQQ_points_to_locate(sim):
    """rich_3777 r1 (E020) opened with search/show loops and never called locate. v13: they all run, every footer asks
    for locate first, reads before locate don't eat the exploration budget, and afterwards reading works."""
    seq = [sim.Q('TTY_COMPATIBLE'), sim.S('rich/console.py', 940, 970), sim.S('rich/console.py', 1, 50),
           sim.S('rich/console.py', 60, 90), sim.Q('self.is_terminal()'), sim.Q('Progress'), sim.Q('rich/'),
           sim.Q('is_terminal')]
    assert [k for k, _ in seq] == ['ran'] * 8
    assert all(calls(nxt(f))[0][0] == 'locate.py' for _, f in seq)
    assert sim.L()[0] == 'ran'
    assert sim.S()[0] == 'ran'
    assert sim.Q('Live')[0] == 'ran'
    assert flow.State(flow.connect()).look_since_edit == 2


# ---------------------------------------------------------------- rule 2: repeats

def test_rule2_identical_search_replays_stored_output(sim):
    """fastapi_15661 ran search "__version__" seven times in a row. Every other repeat is replayed (rule 3b: the
    call after a refusal runs)."""
    sim.L()
    hit = 'fastapi/__init__.py:3: __version__ = "0.1"\n'
    kind, _ = sim.call('search.py', ['"__version__"'], 0, hit)
    assert kind == 'ran'
    kinds = []
    for _ in range(6):
        kind, msg = sim.call('search.py', ['"__version__"'], 0, hit)
        kinds.append(kind)
        if kind == 'refused':
            assert msg.startswith('ALREADY DONE at call ')
            assert 'fastapi/__init__.py:3: __version__ = "0.1"' in msg
    assert kinds == ['refused', 'ran'] * 3


def test_rule2_normalises_packed_and_quoted_args(sim):
    sim.L()
    assert sim.call('show.py', ['pay/invoice.py', '1', '20'], 0, 'pay/invoice.py L1-20 of 40\n')[0] == 'ran'
    for i, args in enumerate((['pay/invoice.py 1 20'], ['["pay/invoice.py", "1", "20"]'],
                              ['/workspace/pay/invoice.py', '1', '20'])):
        assert sim.call('show.py', args)[0] == 'refused'
        sim.e(ok=False, old=f'guess {i}')  # a failed edit changes nothing: the next repeat is replayed again
    assert sim.call('search.py', ['_SVG_FONT_FAMILY'], 0, 'x.py:1: a\n')[0] == 'ran'
    assert sim.call('search.py', ['"_SVG_FONT_FAMILY"'])[0] == 'refused'
    assert sim.call('search.py', ['_SVG_FONT_FAMILY', 'rich/'])[0] == 'ran'  # different scope is a new call


def test_rule2_identical_failed_edit_refused(sim):
    sim.L()
    sim.e(ok=False, old='x = 1')
    kind, msg = sim.e(ok=False, old='x = 1')
    assert kind == 'refused' and 'ALREADY DONE at call 2' in msg and 'NOT APPLIED' in msg


def test_rule2_rerun_allowed_after_successful_edit(sim):
    sim.L()
    assert sim.T(code=1)[0] == 'ran'
    assert sim.T(code=1)[0] == 'refused'   # same failing test, nothing changed
    sim.e(ok=True)
    assert sim.T(code=0)[0] == 'ran'       # the edit changed the code: rerun is legit
    assert sim.C()[0] == 'ran'
    assert sim.C()[0] == 'refused'         # check_patch twice without an edit replays READY
    kind, f = sim.C()                      # rule 3b: the call after a refusal runs
    assert kind == 'ran' and nxt(f) == 'NEXT: submit_patch (the patch is ready)'
    kind, msg = sim.C()
    assert kind == 'refused' and 'READY' in msg and 'NEXT: submit_patch' in msg


def test_rule2_usage_errors_are_rerun_and_not_counted(sim):
    sim.L()
    for _ in range(3):
        sim.S(status=2)
    st = flow.State(flow.connect())
    assert st.look_since_edit == 0


# ---------------------------------------------------------------- rule 3: exploration budget

def test_rule3_budget_pushes_the_edit_after_the_budget(sim):
    """v13: the budget is soft: reads run, and from the budget on the first NEXT is the line-range edit."""
    sim.L()
    footers = [sim.S(start=1 + 10 * i, end=10 + 10 * i)[1] for i in range(flow.N_EXPLORE_FIRST)]
    assert '(2 reads left before you should edit)' in footers[0] and '(1 read left before you should edit)' in footers[1]
    edit_8_11 = call('edit.py', target='pay/invoice.py', start='8', end='11', new='<new text for lines 8-11>')
    assert nxt(footers[2]).splitlines()[0] == (f'NEXT: {flow.N_EXPLORE_FIRST} reads since locate: edit now, even if not '
                                               'fully sure (an imperfect change in the right place beats no change): '
                                               + edit_8_11)
    kind, f = sim.Q('something')
    assert kind == 'ran'
    lines = nxt(f).splitlines()
    assert lines[0].startswith(f'NEXT: {flow.N_EXPLORE_FIRST + 1} reads since locate: edit now') and edit_8_11 in lines[0]
    assert lines[1] == ('NEXT: the code most likely to change: pay/invoice.py Invoice.total lines 8-11 (shown at call 1); '
                        'or edit by text: ' + call('edit.py', target='pay/invoice.py', old='<exact current lines>',
                                                   new='<new lines>'))
    assert '    9          if self.amount' not in f  # a pointer, not a code dump


def test_rule3_budget_resets_after_an_edit_attempt(sim):
    sim.L()
    for i in range(flow.N_EXPLORE_FIRST):
        sim.S(start=1 + 10 * i, end=10 + 10 * i)
    kind, f = sim.O()
    assert kind == 'ran' and 'reads since locate: edit now' in f
    sim.e(ok=False, old='guess')  # a failed edit prints the closest region: that is progress
    kind, f = sim.O('pay/http.py')
    assert kind == 'ran' and 'reads since' not in f


FASTAPI_LOCATE_OUT = textwrap.dedent('''\
    identifiers: include_router
    1.83 fastapi/routing.py  [defines include_router; similar past commits]
          APIRouter.include_router  L1261-1503
          APIRouter.websocket  L1185-1250
    1.32 fastapi/applications.py  [defines include_router]
          FastAPI.include_router  L1339-1542
    0.43 tests/test_empty_router.py
          test_include_empty  L33-36
    ''')


def test_replay_fastapi_14258_LLSSSSSSRRRRSSSS(sim):
    """fastapi_14258 (an earlier run): two locates, then shows/runs until the budget ran out, never edited. v13: every
    call runs; the footers of the reads past the budget and of the runs past the cap push the edit of locate's top
    file."""
    assert sim.call('locate.py', ['include_router'], 0, FASTAPI_LOCATE_OUT)[0] == 'ran'
    assert sim.L('Show a clear error on attempt to include router into itself include_router',
                 out=FASTAPI_LOCATE_OUT)[0] == 'ran'
    shows = [sim.S('fastapi/routing.py', 1261 + 100 * i, 1360 + 100 * i) for i in range(6)]
    assert [k for k, _ in shows] == ['ran'] * 6
    assert all('reads since locate: edit now' in f for _, f in shows[2:])
    runs = [sim.R(snippet=f'print({i})') for i in range(4)]
    # v21b: from call CALLS_EDIT_AT (10) the budget says edit now and the snippet gate starts at 3 snippets
    assert [k for k, _ in runs] == ['ran'] * 3 + ['refused']
    assert runs[3][1].startswith('NOT RUN: 3 snippets and no edit yet; snippets are refused until you edit.')
    # v16 (rule 3c): the same output, no error: probes that do not reproduce; sweep or edit
    assert 'snippets did not reproduce it: sweep the invariant' in runs[1][1]
    assert all('edit.py  target=fastapi/routing.py' in f for _, f in runs[1:])
    later = [sim.S('fastapi/applications.py', 1 + 50 * i, 40 + 50 * i)[0] for i in range(4)]
    assert later == ['ran'] * 4
    _, f = sim.S('fastapi/applications.py', 900, 950)
    assert first_edit(f)['target'] == 'fastapi/routing.py'  # locate's top file, which it read; not the last one
    assert ''.join(sim.log) == '.' * 11 + 'x' + '.' * 5


def test_rule3_third_locate_counts_as_lookaround(sim):
    sim.L('a')
    sim.L('b')
    for i in range(flow.N_EXPLORE_FIRST - 1):
        sim.S(start=1 + 10 * i, end=10 + 10 * i)
    kind, f = sim.L('c')  # 3rd locate: the 3rd look-around
    assert kind == 'ran' and nxt(f).startswith(f'NEXT: {flow.N_EXPLORE_FIRST} reads since locate: edit now')
    assert sim.L('d')[0] == 'ran'
    assert flow.State(flow.connect()).look_since_edit == flow.N_EXPLORE_FIRST + 1


def test_rule3_runs_before_first_edit_push_the_edit(sim):
    """requests_7315 r1: LRRRRRRRRRR - reproduction attempts instead of an edit. v13: they run (v12 refused every
    other one); from MAX_RUNS_BEFORE_EDIT runs on, the footer's first NEXT is the line-range edit."""
    sim.L()
    kinds, footers = [], []
    for i in range(10):
        k, text = sim.Rx(i)  # each reproduces something new: rule 3c's sweep nudge does not apply
        kinds.append(k)
        footers.append(text)
    # v16 (rule 3c): the snippet after MAX_SNIPPETS_BEFORE_EDIT is refused; v17 (L5): until an edit, every second one
    assert kinds == ['ran'] * 5 + ['refused', 'ran'] * 2 + ['refused']
    assert footers[5].startswith('NOT RUN: 5 snippets and no edit yet; snippets are refused until you edit.')
    assert 'NEXT: 5 snippets and no edit yet: edit the most suspicious line now' in footers[5]
    assert 'NEXT: make the fix now: edit.py  target=pay/invoice.py  ' in footers[0]
    edit_8_11 = call('edit.py', target='pay/invoice.py', start='8', end='11', new='<new text for lines 8-11>')
    for i in range(flow.MAX_RUNS_BEFORE_EDIT - 1, 4):
        first = nxt(footers[i]).splitlines()[0]
        assert first.startswith(f'NEXT: {i + 1} runs and no edit yet: edit now, even if not fully sure') and \
            first.endswith(edit_8_11), first
    for i, n in ((4, 5), (6, 6), (8, 7)):
        first = nxt(footers[i]).splitlines()[0]
        assert first.startswith(f'NEXT: {n} snippets and no edit yet: the next snippet is not run until you edit') \
            and first.endswith(edit_8_11), first
    assert sim.e(ok=True)[0] == 'ran'
    assert sim.R(snippet='print("after")')[0] == 'ran'


def test_rule3_usage_error_runs_do_not_count(sim):
    sim.L()
    for i in range(5):
        kind, f = sim.R(snippet=f'print({i})', status=2)
        assert kind == 'ran' and nxt(f) == ''  # usage errors: the script's own message is the guidance
    assert sim.R()[0] == 'ran'


# ---------------------------------------------------------------- rule 4: edit cascade

def test_rule4_lines_command_after_two_failed_edits(sim):
    """fastapi_14258 r2: Leeeeeeee."""
    sim.L()
    _, f1 = sim.e(ok=False, old='a1', closest=(120, 128))
    assert 'retry edit.py' in f1 and 'edit.py  target=pay/invoice.py  start=120  end=128  new=' in f1
    _, f2 = sim.e(ok=False, old='a2', closest=(120, 128))
    assert nxt(f2) == ('NEXT: 2 edits failed; stop matching text and replace the lines by number (write the new text '
                       'with real indentation): edit.py  target=pay/invoice.py  start=120  end=128  '
                       'new=<new text for lines 120-128>')
    for i in range(3, 8):
        kind, f = sim.e(ok=False, old=f'a{i}', closest=(121, 129))
        assert kind == 'ran' and f'{i} edits failed' in f and 'target=pay/invoice.py  start=121  end=129  ' in f
    assert sim.e(ok=False, old='a7')[0] == 'refused'  # identical to the previous failed edit
    _, ok = sim.e(ok=True)
    assert 'verify once' in ok


def test_rule4_without_closest_uses_small_viewed_region(sim):
    sim.L()
    sim.S(start=8, end=11)
    sim.e(ok=False, old='x', closest=None)
    _, f = sim.e(ok=False, old='y', closest=None)
    assert '2 edits failed' in f and 'edit.py  target=pay/invoice.py  start=8  end=11  ' in f


def test_rule4_edit_usage_errors_are_not_failures(sim):
    sim.L()
    sim.e(ok=False, old='x')
    sim.e(status=2)
    assert flow.State(flow.connect()).failed_streak == 1


# ---------------------------------------------------------------- rules 5 and 6: verify cap, passing check

def test_rule6_passing_check_after_edit(sim):
    sim.L()
    sim.e(ok=True)
    _, f = sim.R(code=0)
    # v17 (L6): the existing tests of the edited function (Invoice.total) come before check_patch
    assert nxt(f) == ('NEXT: the check passes; now the existing tests that use total: run.py  test=tests/test_invoice.py'
                      '  k=test_total\nNEXT: then check_patch.py and submit_patch')
    _, f = sim.T(code=0)
    assert nxt(f) == 'NEXT: the check passes: check_patch.py, then submit_patch'
    _, f = sim.C(ready=True)
    assert nxt(f) == 'NEXT: submit_patch (the patch is ready)'


def test_rule6_not_before_an_edit(sim):
    sim.L()
    _, f = sim.R(code=0)
    assert 'make the fix now' in f and 'check_patch' not in f


def test_rule5_verify_cap(sim):
    sim.L()
    sim.e(ok=True)
    _, f1 = sim.T(code=1)
    assert nxt(f1).startswith('NEXT: if the failure comes from your change')
    _, f2 = sim.R(code=1)
    assert nxt(f2) == 'NEXT: 2 checks since the last edit: stop verifying: ' + call('check_patch.py') + ', then submit_patch'
    _, f3 = sim.R(code=1, snippet='print(3)')
    assert 'stop verifying' in f3
    kind, msg = sim.R(code=1, snippet='print(4)')
    assert kind == 'refused' and msg.startswith('NOT RUN: 3 checks since the last edit')
    sim.e(ok=True, old='b', new='c')  # a new edit opens a new verification window
    assert sim.R(code=0, snippet='print(5)')[0] == 'ran'


def test_check_patch_empty_points_to_edit(sim):
    sim.L()
    _, f = sim.C(ready=False)
    assert 'the patch is empty' in f and 'edit.py  target=pay/invoice.py  ' in f


# ---------------------------------------------------------------- rule 7: thin issues

@pytest.fixture
def hyp_on(monkeypatch):
    """Rule 7 is off since v19 (flow.HYPOTHESIS_MODE); these tests keep its routine working when switched on."""
    monkeypatch.setattr(flow, 'HYPOTHESIS_MODE', True)


def test_rule7_off_thin_issue_stays_normal(sim):
    """v19: a thin issue keeps the normal footers (hypothesis mode cost edits in E053-E056)."""
    _, f = sim.L('Fix #3104', out=THIN_LOCATE_OUT)
    assert flow.State(flow.connect()).mode == 'normal'
    assert 'hypothesis.py' not in f and ('show.py  target=' in f or 'edit.py  target=' in f), f


@pytest.mark.parametrize('idents,low,thin', [('(none)', False, True), ('a', False, False), ('a', True, True),
                                             ('a, b', True, False), ('a, b, c', False, False)])
def test_rule7_thin_detection(hyp_on, sim, idents, low, thin):
    out = LOCATE_OUT.replace('Invoice, total', idents) + ('LOW CONFIDENCE: top files score close\n' if low else '')
    _, f = sim.L(out=out)
    st = flow.State(flow.connect())
    assert (st.mode == 'hypothesis') == thin
    assert ('hypothesis.py  add=' in f) == thin


def test_rule7_explicit_thin_fact_wins(hyp_on, sim):
    sim.L(thin=True)
    assert flow.State(flow.connect()).mode == 'hypothesis'


def test_rule7_hypothesis_routine(hyp_on, sim, ws):
    """rich_3105 ("Fix #3104"): nothing to locate on. Guess, test, confirm, edit."""
    _, f = sim.L('Fix #3104', out=THIN_LOCATE_OUT)
    lines = f.splitlines()
    assert len(lines) == 3
    assert call('hypothesis.py', add='<file.py:function does X wrong because Y>', add2='<second guess>') in lines[0]
    assert 'run.py  snippet=' in lines[1] and 'hypothesis.py  confirm=1  evidence=' in lines[1]
    assert 'pay/invoice.py' in lines[2]
    code, out = sim.H('add', 'pay/http.py:parse_header ignores quotes', 'pay/invoice.py:Invoice.total mishandles None')
    assert code == 0
    assert 'H1 [untested] pay/http.py:parse_header ignores quotes' in out
    assert 'H2 [untested] pay/invoice.py:Invoice.total mishandles None' in out
    assert 'NEXT: test H1' in out and 'run.py  snippet=' in out
    check_footer(refusal_footer(out))
    _, f = sim.R(code=0, snippet='from pay.http import parse_header; print(parse_header("a"))')
    assert 'record what the run showed for H1' in f
    assert call('hypothesis.py', confirm='1', evidence='<what the output showed>') in f
    assert '  or  ' + call('hypothesis.py', reject='1', evidence='<what the output showed>') in f
    code, out = sim.H('reject', 'H1', 'quotes are fine')
    assert 'H1 [rejected] pay/http.py:parse_header ignores quotes  -- quotes are fine' in out and 'NEXT: test H2' in out
    sim.R(code=0, snippet='from pay.invoice import Invoice; print(Invoice(None).total())')
    code, out = sim.H('confirm', '2', 'printed 0 instead of None')
    assert 'H2 [confirmed]' in out
    assert 'NEXT: edit where H2 is confirmed: edit.py  target=pay/invoice.py  ' in out


def test_rule7_hypothesis_mode_has_larger_budgets(hyp_on, sim):
    sim.L(out=THIN_LOCATE_OUT)
    runs = [sim.Rx(i) for i in range(5)]
    assert [k for k, _ in runs] == ['ran'] * 5
    assert ['runs and no edit yet' in f for _, f in runs] == [False] * 3 + [True, False]  # MAX_RUNS_BEFORE_EDIT_THIN
    assert 'NEXT: 5 snippets and no edit yet: the next snippet is not run until you edit' in runs[4][1]  # v17 L5
    sim.e(ok=False, old='guess')  # an edit attempt opens a new exploration window
    footers = [sim.S(start=1 + 10 * i, end=5 + 10 * i) for i in range(7)]
    assert [k for k, _ in footers] == ['ran'] * 7
    assert not any('reads since' in f for _, f in footers[:5])
    lines = nxt(footers[5][1]).splitlines()
    assert lines[0].startswith(f'NEXT: {flow.N_EXPLORE_THIN} reads since the last edit: edit now')
    assert lines[1] == 'NEXT: or, if you cannot tell where yet, write guesses: ' + call(
        'hypothesis.py', add='<guess 1>', add2='<guess 2>')


def test_rule7_better_locate_leaves_hypothesis_mode(hyp_on, sim):
    sim.L('Fix #3104', out=THIN_LOCATE_OUT)
    assert sim.L('Invoice.total returns 0 for None amount')[0] == 'ran'
    assert flow.State(flow.connect()).mode == 'normal'


# ---------------------------------------------------------------- passing paths are not disturbed

def test_passing_path_requests_7309(sim):
    """requests_7309 r1: LRReRC! - the shape we want; every footer points to the next step of it."""
    _, f = sim.L()
    assert 'replace its lines: ' + call('edit.py', target='pay/invoice.py', start='8', end='11',
                                        new='<new text for lines 8-11>') in f
    assert 'or by text: ' + call('edit.py', target='pay/invoice.py', old='<exact current lines>', new='<new lines>') in f
    assert call('show.py', target='pay/http.py', symbol='send') in f  # the alternative is a source file, not a test
    sim.R(code=0)
    sim.R(code=0, snippet='print(2)')
    _, f = sim.e(ok=True)
    assert call('run.py', test='tests/test_invoice.py') in f  # test file guessed from the edited module
    _, f = sim.R(code=0, snippet='print(3)')
    assert call('check_patch.py') in f
    _, f = sim.C()
    assert nxt(f) == 'NEXT: submit_patch (the patch is ready)'
    assert set(sim.log) == {'.'}


def test_passing_path_fastapi_14372_with_usage_errors(sim):
    """fastapi_14372 r2 (passed): usage errors and a post-failure show must not trigger refusals."""
    sim.L()
    sim.L('Make the result of Depends() and Security() hashable')
    sim.S(status=2)
    sim.S('fastapi/param_functions.py', 2221, 2298)
    sim.call('show.py', ['fastapi/params.py'], 2, 'usage')
    sim.S('fastapi/params.py', 765, 769)
    sim.S('fastapi/param_functions.py', 2301, 2381)
    sim.S('fastapi/params.py', 772, 774)
    sim.R(status=2)
    sim.R(code=0)
    sim.e(ok=False, old='@dataclass\\n  class Depends:', closest=(765, 766))
    sim.S('fastapi/params.py', 760, 780)
    sim.call('edit.py', ['--lines', 'fastapi/params.py', '765', '766', 'a', 'b'], 2, 'usage')
    sim.e(ok=False, old='x1', closest=(765, 766))
    sim.e(ok=False, old='x2', closest=(765, 766))
    sim.e(ok=True, old='  @dataclass\\n  class Depends:')
    sim.R(code=1, snippet='print("after")')
    sim.e(ok=True, old='  @dataclass\\n  class Security(Depends):')
    sim.R(code=0, snippet='print("after2")')
    _, f = sim.C()
    assert ''.join(sim.log) == '.' * 20, ''.join(sim.log)  # v13: the 4th read runs too (v12 refused it)
    assert nxt(f) == 'NEXT: submit_patch (the patch is ready)'


def test_replay_rich_3944_search_loop(sim):
    """rich_3944 r1: QLSSSQQQQQQQSQQQQQQSQ - identical repeats are replayed (never two refusals in a row); v13: the
    other reads run past the budget, with footers that push the edit."""
    assert sim.call('search.py', ['-E', 'font'])[0] == 'ran'  # before locate: runs, the footer asks for locate
    sim.L('fix fonts rich/_export_format.py')
    kinds = [sim.S('rich/_export_format.py', 1, 100)[0], sim.S('rich/_export_format.py', 1, 100)[0],
             sim.Q('_SVG_FONT_FAMILY')[0], sim.Q('"_SVG_FONT_FAMILY"')[0], sim.Q('_SVG_FONT_FAMILY')[0],
             sim.call('search.py', ['rich/console.py', '_SVG_FONT_FAMILY'])[0], sim.Q('font-family')[0]]
    assert kinds == ['ran', 'refused', 'ran', 'refused', 'ran', 'ran', 'ran']
    kind, f = sim.Q('_export_format')
    assert kind == 'ran' and re.match(r'NEXT: \d+ reads since locate: edit now', nxt(f))
    assert sim.call('search.py', ['rich/console.py', '_SVG_FONT_FAMILY'])[0] == 'refused'  # repeats still replayed
    assert sim.Q('_SVG_FONT_FAMILY')[0] == 'ran'  # rule 3b


# ---------------------------------------------------------------- footers, state, robustness

def test_every_footer_is_bounded(sim):
    outs = [sim.L(), sim.S(), sim.Q(), sim.O(), sim.e(), sim.e(old='z'), sim.R(), sim.e(ok=True), sim.T(), sim.C()]
    for kind, text in outs:
        if kind == 'ran':
            check_footer(text)
        else:
            assert len(text.splitlines()) <= 40


def test_search_footer_points_to_source_hit(sim):
    sim.L()
    _, f = sim.call('search.py', ['amount'], 0, 'pay/invoice.py:9:     if self.amount == None:\ntests/t.py:3: x\n')
    assert call('show.py', target='pay/invoice.py', start='1', end='34') in f


def test_state_resets_when_workspace_changes(sim, tmp_path, monkeypatch):
    sim.L()
    sim.e(ok=True)
    other = tmp_path / 'other'
    other.mkdir()
    monkeypatch.setenv('WORKSPACE', str(other))
    st = flow.State(flow.connect())
    assert st.n == 0 and not st.located


def test_never_crashes_on_broken_state_dir(tmp_path, monkeypatch):
    blocker = tmp_path / 'file'
    blocker.write_text('not a dir')
    monkeypatch.setenv('REPO_TOOLS_DIR', str(blocker))
    assert flow.before('show.py', ['a.py', 'f']) is None
    assert flow.after('show.py', ['a.py', 'f'], 0, 'x') == ''


def test_never_crashes_on_corrupt_db(tmp_path, monkeypatch):
    d = tmp_path / 'state'
    d.mkdir()
    (d / 'state.sqlite').write_bytes(b'garbage' * 100)
    monkeypatch.setenv('REPO_TOOLS_DIR', str(d))
    assert flow.before('search.py', ['x']) is None
    assert flow.after('search.py', ['x'], 0, 'x') == ''


def test_fast(sim):
    """before+after per script call, median of 20 (robust to hiccups; the sandbox image may run emulated)."""
    times = []
    for i in range(20):
        t = time.perf_counter()
        args = ['pay/invoice.py', str(i + 1), str(i + 5)]
        if not flow.before('show.py', args):
            flow.after('show.py', args, 0, f'pay/invoice.py L{i + 1}-{i + 5} of 40\n')
        times.append(time.perf_counter() - t)
        if i == 0:
            sim.L()
    # Budget: negligible next to 10-30 s model turns. 0.25 s tolerates x86 emulation on a loaded machine
    # (0.10 s measured under emulation with other test suites running; ~1 ms natively).
    assert sorted(times)[len(times) // 2] < 0.25, times


def test_status_script(sim, ws):
    code, out = run_script('status.py')
    assert code == 0 and 'locate: not run yet' in out and 'locate.py' in out.splitlines()[-1]
    assert out.startswith('MEMO: no edit yet; locate.py not run yet\n')   # v18c: the MEMO line first
    sim.L()
    sim.S(start=8, end=11)
    sim.e(ok=False, old='x', closest=(9, 10))
    code, out = run_script('status.py')
    lines = out.strip().splitlines()
    assert code == 0 and len(lines) <= 16
    assert lines[0] == 'MEMO: no edit yet (1 failed). Code to change: pay/invoice.py Invoice.total L8-11 (read at call 3)'
    assert lines[1].startswith('calls so far: 4 (0 refused) | mode: normal | edits: 0 ok, 1 failed')
    assert 'locate top: pay/invoice.py Invoice.total L8-11; pay/http.py send L5-6' in out
    assert 'edit call 4: pay/invoice.py FAILED' in out
    assert lines[-1].startswith('NEXT:') and len([l for l in lines if l.startswith('NEXT:')]) == 1
    assert flow.State(flow.connect()).n == 5  # status.py itself counts as a call


def test_hypothesis_script_usage_and_errors(ws):
    assert run_script('hypothesis.py')[0] == 2
    assert run_script('hypothesis.py', 'guess', 'x')[0] == 2
    code, out = run_script('hypothesis.py', 'confirm', '9', 'e')
    assert code == 1 and 'no hypothesis H9' in out
    code, out = run_script('hypothesis.py', 'add H one packed')  # ["add ..."] packed into one string
    assert code == 0 and 'H1 [untested] H one packed' in out
    code, out = run_script('hypothesis.py', '["reject", "#1", "nope"]')
    assert 'H1 [rejected] H one packed  -- nope' in out
    code, out = run_script('hypothesis.py', 'list')
    assert code == 0 and 'H1 [rejected]' in out


def test_hypothesis_script_object_form(ws):
    code, out = run_script('hypothesis.py', *obj(add='pkg/a.py:f drops x', add2='pkg/b.py:g', add3='pkg/c.py:h'))
    assert code == 0 and 'H3 [untested] pkg/c.py:h' in out
    code, out = run_script('hypothesis.py', *obj(confirm='H2', evidence='--flag printed [0]'))  # value starts with --
    assert code == 0 and 'H2 [confirmed] pkg/b.py:g  -- --flag printed [0]' in out
    code, out = run_script('hypothesis.py', '{"reject": "1", "evidence": "fine"}')  # object packed into one string
    assert code == 0 and 'H1 [rejected] pkg/a.py:f drops x  -- fine' in out
    code, out = run_script('hypothesis.py', *obj(list='yes'))
    assert code == 0 and out.count('\nH') + out.startswith('H') == 3
    assert run_script('hypothesis.py', *obj(add=' '))[0] == 2
    code, out = run_script('hypothesis.py', *obj(confirm='1', reject='2'))
    assert code == 2 and 'not both' in out
    code, out = run_script('hypothesis.py', *obj(guess='x'))
    assert code == 2 and out.startswith('USAGE ERROR: unknown key guess; valid keys: add (add2, add3), list')
    assert 'Example: hypothesis.py  add=' in out
    assert_notation(out.split('\nusage:')[0])
    code, out = run_script('hypothesis.py')  # {} -> no args -> usage
    assert code == 2 and 'hypothesis.py  list=yes' in out
    assert 'args=' not in out and '{"' not in out


def test_status_script_accepts_object_form(ws):
    code, out = run_script('status.py', *obj(verbose='yes'))
    assert code == 0 and out.startswith('MEMO: no edit yet; locate.py not run yet\ncalls so far:')
    assert 'locate: not run yet' in out
    assert call('locate.py', query='<issue title copied verbatim> <every identifier, path and error message in the '
                                   'issue>') in out.splitlines()[-1]


# ---------------------------------------------------------------- argument schema v2: object args == positional

SAME_CALL = [  # (script, positional argv, object argv): one call, one canonical key
    ('locate.py', ['Invoice total wrong'], obj(query='Invoice total wrong')),
    ('locate.py', ['--k', '5', '--no-show', 'Invoice  "total"'], obj(query='Invoice total', k='5', show='no')),
    ('locate.py', ['Invoice total'], obj(query='Invoice total', show='yes')),  # default flag value
    ('show.py', ['pay/invoice.py', 'Invoice.total'], obj(target='pay/invoice.py', symbol='Invoice.total')),
    ('show.py', ['pay/invoice.py', '140', '200'], obj(target='pay/invoice.py', start='140', end='200')),
    ('show.py', ['pay/invoice.py 140 200'], obj(file='/workspace/pay/invoice.py', start='140', end='200')),
    ('show.py', ['["pay/invoice.py", "140", "200"]'], ['{"file": "pay/invoice.py", "start": 140, "end": 200}']),
    ('outline.py', ['a/b.py', 'c/d.py'], obj(files='a/b.py c/d.py')),
    ('outline.py', ['--grep', 'export', 'a/b.py'], obj(files='a/b.py', grep='export')),
    ('search.py', ['"literal"'], obj(text='literal')),
    ('search.py', ['-E', 'def (a|b)'], obj(regex='def (a|b)')),
    ('search.py', ['BrokenPipeError', 'rich/'], obj(text='BrokenPipeError', within='rich/')),
    ('search.py', ['BrokenPipeError', 'rich/'], obj(text='BrokenPipeError', path='rich/')),
    ('search.py', ['rich/console.py', '_SVG_FONT_FAMILY'], obj(text='_SVG_FONT_FAMILY', within='rich/console.py')),
    ('callers.py', ['apply_discount'], obj(name='apply_discount')),
    ('tests_for.py', ['a/b.py'], obj(target='a/b.py')),
    ('tests_for.py', ['Class.method'], obj(target='Class.method')),
    ('history.py', ['a/b.py'], obj(target='a/b.py')),
    ('history.py', ['--similar', 'fix the export'], obj(similar='fix the export')),
    ('edit.py', ['a/b.py', 'x = 1', 'x = 2'], obj(target='a/b.py', old='x = 1', new='x = 2')),
    ('edit.py', ['--lines', 'a/b.py', '151', '153', '    pass'], obj(file='a/b.py', start='151', end='153', new='    pass')),
    ('edit.py', ['a/b.py', '--old', '--new'], obj(target='a/b.py', old='--old', new='--new')),  # values look like keys
    ('create.py', ['a/new.py', 'x = 1\n'], obj(target='a/new.py', content='x = 1\n')),
    ('create.py', ['a/new.py', 'x = 1\n'], obj(filename='a/new.py', content='x = 1\n')),
    # schema v3: target (canonical) and its aliases are one key; ADK parameters misplaced inside args are dropped
    ('show.py', ['a.py', 'f'], obj(path='a.py', symbol='f')),
    ('show.py', ['a.py', 'f'], obj(dir='a.py', symbol='f')),
    ('show.py', ['a.py', 'f'], obj(within='a.py', symbol='f')),
    ('edit.py', ['a/b.py', 'x', 'y'], obj(**{'in': 'a/b.py'}, old='x', new='y')),
    ('show.py', ['a.py', 'f'], obj(file_path='a.py', symbol='f')),
    ('show.py', ['a.py', 'f'], obj(filepath='a.py', symbol='f')),
    ('show.py', ['a.py', 'f'], obj(skill_name='repo-tools', file_path='show.py', path='a.py', symbol='f')),
    ('edit.py', ['a/b.py', 'x', 'y'], obj(skill_name='repo-tools', path='a/b.py', old='x', new='y')),
    ('edit.py', ['a/b.py', 'x', 'y'], obj(file_path='edit.py', path='a/b.py', old='x', new='y')),
    ('edit.py', ['a/b.py', 'x', 'y'], obj(target='a/b.py', old='x', new='y', file_path='scripts/edit.py')),
    ('history.py', ['a/b.py'], obj(file_name='a/b.py')),
    ('search.py', ['x', 'rich/'], obj(text='x', file_path='rich/')),  # search's directory filter
    ('search.py', ['x', 'rich/'], obj(text='x', dir='rich/')),
    ('search.py', ['x', 'rich/'], obj(text='x', **{'in': 'rich/'})),
    ('search.py', ['x', 'rich/'], obj(text='x', target='rich/')),
    ('run.py', ['test', 'tests/test_x.py'], obj(test='tests/test_x.py')),
    ('run.py', ['test', 'tests/test_x.py', '-k', 'html'], obj(test='tests/test_x.py', k='html')),
    ('run.py', ['snippet', 'print(x[0])'], obj(snippet='print(x[0])')),  # the payload that hung vLLM as an array
    ('check_patch.py', [], []),
    ('check_patch.py', ['--clean'], obj(clean='yes')),
    ('check_patch.py', [], obj(clean='no')),
    ('hypothesis.py', ['add', 'g1', 'g2'], obj(add='g1', add2='g2')),
    ('hypothesis.py', ['confirm', 'H1', 'it printed None'], obj(confirm='1', evidence='it printed None')),
    ('hypothesis.py', ['reject', '#2', 'fine'], obj(reject='2', evidence='fine')),
    ('hypothesis.py', ['list'], obj(list='yes')),
    ('status.py', [], []),
]


@pytest.mark.parametrize('script,positional,object_form', SAME_CALL)
def test_same_call_in_both_forms(script, positional, object_form):
    assert flow.canon(script, positional) == flow.canon(script, object_form)
    assert flow.action_key(script, positional) == flow.action_key(script, object_form)


@pytest.mark.parametrize('script,a,b', [
    ('show.py', obj(target='a.py', symbol='f'), obj(file='a.py', symbol='g')),
    ('search.py', obj(text='x'), obj(regex='x')),
    ('search.py', obj(text='x'), obj(text='x', within='rich/')),
    ('tests_for.py', obj(target='a.py'), obj(target='b.py')),
    ('edit.py', obj(file='a.py', old='x', new='y'), obj(file='a.py', old='x', new='y ')),  # whitespace matters
    ('edit.py', obj(file='a.py', old='x', new='y'), ['a.py', 'x']),                     # usage error != edit
    ('run.py', obj(snippet='print(1)'), obj(test='print(1)')),
    ('run.py', obj(snippet='print(1)'), obj(code='print(1)')),                          # unknown key != valid call
    ('check_patch.py', [], obj(clean='yes')),
])
def test_different_calls_stay_different(script, a, b):
    assert flow.action_key(script, a) != flow.action_key(script, b)


@pytest.mark.parametrize('form', [obj(target='a.py', symbol='f'), obj(path='a.py', symbol='f'),
                                  obj(file='a.py', symbol='f'), ['a.py', 'f'], obj(file_path='a.py', symbol='f'),
                                  obj(**{'in': 'a.py'}, symbol='f'), ['--symbol', 'f', '--path', '/workspace/a.py']])
def test_target_path_file_and_positional_are_one_call(form):
    """canon keys the repository file as "file" internally, whatever the model called it."""
    assert flow.canon('show.py', form) == {'file': 'a.py', 'symbol': 'f'}
    assert flow.action_key('show.py', form) == flow.action_key('show.py', ['a.py', 'f'])


def test_alias_keys():
    assert flow.alias_keys('edit.py', {'target': 'a.py', 'old': 'x'}) == {'file': 'a.py', 'old': 'x'}
    assert flow.alias_keys('edit.py', {'skill_name': 'repo-tools', 'file_path': 'edit.py', 'path': 'a.py'}) == \
        {'file': 'a.py'}
    assert flow.alias_keys('edit.py', {'file_path': 'a.py'}) == {'file': 'a.py'}  # a repository file, not the script
    assert flow.alias_keys('edit.py', {'path': 'a.py'}) == {'file': 'a.py'}
    assert flow.alias_keys('search.py', {'text': 'x', 'path': 'rich/'}) == {'text': 'x', 'path': 'rich/'}
    for key in ('within', 'dir', 'directory', 'in', 'file', 'file_path', 'target'):
        assert flow.alias_keys('search.py', {'text': 'x', key: 'rich/'}) == {'text': 'x', 'path': 'rich/'}, key
    assert flow.alias_keys('tests_for.py', {'target': 'a.py'}) == {'target': 'a.py'}  # its own key
    assert flow.alias_keys('run.py', {'test': 't.py', 'skill_name': 'repo-tools'}) == {'test': 't.py'}
    assert flow.alias_keys('status.py', {}) == {}


def test_cmd_renders_the_notation():
    assert flow.cmd('show.py', {'file': 'a.py', 'symbol': 'f'}) == 'show.py  target=a.py  symbol=f'
    assert flow.cmd('search.py', {'text': 'x', 'path': 'rich/'}) == 'search.py  text=x  within=rich/'
    assert flow.cmd('tests_for.py', {'target': 'a.py'}) == 'tests_for.py  target=a.py'
    assert flow.SHOWN_KEY == {'file': 'target', 'path': 'within'}
    assert flow.cmd('check_patch.py', {}) == flow.cmd('check_patch.py') == flow.CHECK == 'check_patch.py'
    assert flow.cmd('run.py', {'snippet': 'a  =  1\nprint(a)'}) == 'run.py  snippet=a = 1 print(a)'  # one line
    assert not hasattr(flow, 'args_obj')


def test_footer_keeps_two_space_separators(sim):
    sim.L()
    _, f = sim.Q('amount')
    assert 'show.py  target=' in f and 'edit.py  target=' in f, f
    assert flow.format_footer(['NEXT: a  b\n  c', 'd']) == 'NEXT: a  b c\nNEXT: d'


def test_canon_object_values_are_read_by_position():
    assert flow.canon('edit.py', ['--file', 'a.py', '--old', '--new', '--new', '-- x']) == \
        {'file': 'a.py', 'old': '--new', 'new': '-- x'}
    assert flow.canon('run.py', obj(snippet='print(x[0])\n--y')) == {'snippet': 'print(x[0])\n--y'}
    assert flow.norm_args(['{"file": "a.py", "start": 140}']) == ['--file', 'a.py', '--start', '140']
    assert flow.object_args(['--file', 'a.py', 'x']) is None and flow.object_args([]) is None


def test_repeat_detection_across_forms(ws):
    sim = Sim()
    sim.L()
    assert sim.call('show.py', ['pay/invoice.py', 'Invoice.total'], 0, 'pay/invoice.py Invoice.total L8-11\n')[0] == 'ran'
    kind, msg = sim.call('show.py', obj(file='/workspace/pay/invoice.py', symbol='Invoice.total'))
    assert kind == 'refused' and msg.startswith('ALREADY DONE at call 2')
    assert sim.call('search.py', obj(regex='amount|total'), 0, 'pay/invoice.py:9: x\n')[0] == 'ran'
    assert sim.call('search.py', ['-E', 'amount|total'])[0] == 'refused'
    fail = 'NOT APPLIED: OLD not found\nCLOSEST: pay/invoice.py 9 10\n'
    assert sim.call('edit.py', obj(target='pay/invoice.py', old='x', new='y'), 1, fail)[0] == 'ran'
    kind, msg = sim.call('edit.py', ['pay/invoice.py', 'x', 'y'])
    assert kind == 'refused' and 'ALREADY DONE at call 6' in msg and 'NOT APPLIED' in msg
    assert sim.call('run.py', ['test', 'tests/test_invoice.py'], 1, 'exit code: 1\n')[0] == 'ran'
    assert sim.call('run.py', obj(test='tests/test_invoice.py'))[0] == 'refused'
    assert sim.call('run.py', obj(test='tests/test_invoice.py', k='total'), 1, 'exit code: 1\n')[0] == 'ran'


def test_budgets_and_counts_across_forms(ws):
    sim = Sim()
    sim.L()
    calls_ = [('show.py', ['pay/invoice.py', '1', '10']), ('search.py', obj(text='amount')),
              ('show.py', obj(target='pay/http.py', symbol='send'))]
    for script, args in calls_:
        assert sim.call(script, args, 0, 'x\n')[0] == 'ran'
    kind, f = sim.call('callers.py', obj(name='total'), 0, 'x\n')
    assert kind == 'ran' and nxt(f).startswith(f'NEXT: {flow.N_EXPLORE_FIRST + 1} reads since locate: edit now')
    assert flow.State(flow.connect()).look_since_edit == flow.N_EXPLORE_FIRST + 1  # both forms counted
    # edit cascade: one failed edit in each form, then the line-number edit in both forms is one call
    fail = 'NOT APPLIED: OLD not found\nCLOSEST: pay/invoice.py 9 10\n'
    sim.call('edit.py', ['pay/invoice.py', 'a', 'b'], 1, fail)
    _, f = sim.call('edit.py', obj(target='pay/invoice.py', old='c', new='d'), 1, fail)
    assert '2 edits failed' in f and call('edit.py', target='pay/invoice.py', start='9', end='10',
                                          new='<new text for lines 9-10>') in f
    st = flow.State(flow.connect())
    assert st.failed_streak == 2 and [e[1] for e in st.edits] == ['pay/invoice.py'] * 2
    ok = 'EDITED pay/invoice.py (lines 9-10)\n'
    assert sim.call('edit.py', obj(file='pay/invoice.py', start='9', end='10', new='x'), 0, ok)[0] == 'ran'
    assert flow.State(flow.connect()).edited_paths == ['pay/invoice.py']
    # checks since the edit: run.py test/snippet in both forms are counted with their kind and target
    sim.call('run.py', obj(test='tests/test_invoice.py'), 0, 'exit code: 1\n')
    sim.call('run.py', ['snippet', 'print(1)'], 0, 'exit code: 1\n')
    checks = flow.State(flow.connect()).checks
    assert [c[1:] for c in checks] == [('test', 'tests/test_invoice.py', '1'), ('snippet', None, '1')]
    kind, _ = sim.call('run.py', obj(snippet='print(2)'), 0, 'exit code: 1\n')
    assert kind == 'ran'
    assert sim.call('run.py', obj(snippet='print(3)'))[0] == 'refused'  # MAX_CHECKS_HARD counts both forms


def test_outline_facts_from_object_form(ws):
    sim = Sim()
    sim.L()
    _, f = sim.call('outline.py', obj(files='pay/http.py pay/invoice.py'), 0, 'pay/http.py\n  def send L5-6\n')
    assert call('show.py', target='pay/http.py', symbol='<Class.method>') in f


def test_locate_prefill_from_object_form(ws):
    kind, f = Sim().call('search.py', obj(regex='_SVG_FONT_FAMILY|font-family', within='rich/'), 1, 'no matches\n')
    assert kind == 'ran'
    assert [a['query'] for s, a in calls(f) if s == 'locate.py' and a] == [
        '<issue title copied verbatim> <every identifier, path and error message in the issue> '
        '_SVG_FONT_FAMILY|font-family rich/']


def test_long_values_are_elided_in_suggestions(ws):
    sim = Sim()
    long_path = 'pkg/' + 'x' * 300 + '.py'
    sim.L(out=LOCATE_OUT.replace('pay/invoice.py', long_path))
    _, f = sim.call('show.py', obj(target=long_path, start='1', end='5'), 0, f'{long_path} L1-5 of 9\n    1  x\n')
    check_footer(f)
    assert all(len(line) <= flow.LINE_CHARS for line in f.splitlines())
    assert '…' in f and all(len(v) <= flow.VALUE_CHARS for line in f.splitlines() for a in suggested_args(line)
                            for v in a.values())


def test_no_footer_suggests_array_args(sim):
    """Every footer and refusal of a long mixed run renders only object-form calls (check_footer asserts it)."""
    texts = [sim.Q('x')[1], sim.L(out=THIN_LOCATE_OUT)[1], sim.S()[1], sim.R()[1], sim.e()[1], sim.e(old='q')[1],
             sim.e(old='r', closest=None)[1], sim.e(ok=True)[1], sim.T()[1], sim.R(code=1, snippet='p')[1],
             sim.R(snippet='p2')[1], sim.C(ready=False)[1], sim.C()[1]]
    code, out = run_script('status.py')
    for t in texts + [out]:
        check_footer(refusal_footer(t))
        assert 'repo-tools' not in t and 'skill_name' not in t


# ---------------------------------------------------------------- flow.main with the real scripts

def run_script(name, *args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, str(SCRIPTS / name), *args], capture_output=True, text=True, env=env,
                       timeout=60)
    return r.returncode, r.stdout


def run_hooked(name, *args, cwd):
    """Run a real script through flow.main, as the integrated scripts will."""
    mod = name[:-3]
    code = (f'import sys; sys.dont_write_bytecode = True; sys.path.insert(0, {str(SCRIPTS)!r})\n'
            f'import flow, {mod}\nsys.exit(flow.main({name!r}, {mod}.main))\n')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-c', code, *args], capture_output=True, text=True, env=env, cwd=cwd,
                       timeout=60)
    return r.returncode, r.stdout


def test_real_scripts_under_flow_main(ws, tmp_path):
    cwd = tmp_path
    code, out = run_hooked('search.py', 'amount', cwd=cwd)
    assert code == 0 and out.startswith('pay/invoice.py:') and 'NEXT: locate.py  query=' in out  # runs (v13)
    code, out = run_hooked('locate.py', 'Invoice.total() returns wrong value when self.amount is None', cwd=cwd)
    assert code == 0 and out.rstrip().splitlines()[-1].startswith('NEXT:')
    st = flow.State(flow.connect())
    assert st.candidates and st.candidates[0][0] == 'pay/invoice.py'
    assert st.viewed and st.viewed[0][0] == 'pay/invoice.py'  # the function locate printed counts as viewed
    code, out = run_hooked('show.py', 'pay/invoice.py', 'Invoice.total', cwd=cwd)
    assert 'NEXT: if this is the code to change' in out
    assert flow.State(flow.connect()).viewed[0][:3] == ('pay/invoice.py', 8, 11)
    code, out = run_hooked('show.py', 'pay/invoice.py', 'Invoice.total', cwd=cwd)
    assert out.startswith('ALREADY DONE at call 3')
    code, out = run_hooked('edit.py', 'pay/invoice.py', 'if self.amount == Nothing:', 'x', cwd=cwd)
    assert code == 1 and 'NEXT: copy the old text exactly' in out
    code, out = run_hooked('edit.py', 'pay/invoice.py', 'if self.amount == None:', 'if self.amount is None:',
                           cwd=cwd)
    assert code == 0 and 'NEXT: verify once' in out and 'tests/test_invoice.py' in out
    code, out = run_hooked('run.py', 'test', 'tests/test_invoice.py', cwd=cwd)
    assert 'exit code: 0' in out and 'NEXT: the check passes' in out
    code, out = run_hooked('check_patch.py', cwd=cwd)
    assert 'READY' in out and out.rstrip().endswith('NEXT: submit_patch (the patch is ready)')
    assert not list(SCRIPTS.glob('__pycache__')), 'bytecode written into agent/'


def test_flow_main_records_exceptions_and_reraises(ws):
    def boom(argv):
        print('partial')
        raise ValueError('x')

    with pytest.raises(ValueError):
        flow.main('edit.py', boom, ['a.py', 'f', 'g'])
    st = flow.State(flow.connect())
    assert st.actions[-1][1] == 'edit.py' and st.actions[-1][3] == 'fail' and st.actions[-1][4] == 'partial\n'


def test_flow_main_facts_override_parsing(ws, capsys):
    sim = Sim()
    sim.L()

    def fake_show(argv):
        flow.fact(path='pay/http.py', start=5, end=6)
        print('unparseable header')
        return 0

    assert flow.main('show.py', fake_show, ['pay/http.py', 'send']) == 0
    assert flow.State(flow.connect()).viewed[0][:3] == ('pay/http.py', 5, 6)
    assert 'edit.py  target=pay/http.py  start=5  end=6  new=' in capsys.readouterr().out


# ---------------------------------------------------------------- rule 9: budget awareness

def age(seconds):
    """Pretend the first recorded call happened `seconds` earlier."""
    conn = flow.connect()
    conn.execute('UPDATE actions SET ts = ts - ?', (seconds,))
    conn.close()


def test_budget_line_from_third_call(sim):
    _, f1 = sim.L()
    _, f2 = sim.S(start=8, end=11)
    assert 'BUDGET' not in f1 + f2
    _, f3 = sim.Q()
    first = f3.splitlines()[0]
    assert re.fullmatch(r'BUDGET: call 3 of 40 \| 0m0\ds of 3m40s used', first), first
    assert len(f3.splitlines()) <= flow.FOOTER_LINES


def test_budget_elapsed_is_measured_from_first_call(sim):
    sim.L()
    age(80)
    sim.S(start=8, end=11)
    _, f = sim.Q()
    assert f.startswith('BUDGET: call 3 of 40 | 1m20s of 3m40s used')
    assert 'edit your best candidate now' not in f  # 36% < BUDGET_EDIT_FRAC: no escalation yet


def test_budget_half_time_without_edit_escalates_to_edit(sim):
    sim.L()
    sim.S(start=8, end=11)
    age(flow.TIME_BUDGET_S * flow.BUDGET_EDIT_FRAC + 5)
    _, f = sim.Q()
    lines = f.splitlines()
    assert lines[0].endswith('| edit your best candidate now')
    assert lines[1] == 'NEXT: edit your best candidate now: ' + call('edit.py', target='pay/invoice.py', start='8',
                                                                     end='11', new='<new text for lines 8-11>')
    assert lines[2] == ('NEXT: or by text: edit.py  target=pay/invoice.py  old=<exact current lines>  '
                        'new=<new lines>')


def test_budget_half_time_escalation_shows_even_before_call_3(sim):
    sim.L()
    age(flow.TIME_BUDGET_S * flow.BUDGET_EDIT_FRAC + 5)
    _, f = sim.S(start=8, end=11)
    assert f.startswith('BUDGET: call 2 of 40') and 'edit your best candidate now' in f


def test_budget_half_time_with_edit_does_not_escalate(sim):
    sim.L()
    sim.e(ok=True)
    age(flow.TIME_BUDGET_S * flow.BUDGET_EDIT_FRAC + 5)
    _, f = sim.T(code=1)
    assert 'edit your best candidate' not in f and 'if the failure comes from your change' in f


def test_budget_keeps_edit_repair_instructions(sim):
    sim.L()
    sim.e(ok=False, old='a', closest=(9, 10))
    age(flow.TIME_BUDGET_S * flow.BUDGET_EDIT_FRAC + 5)
    _, f = sim.e(ok=False, old='b', closest=(9, 10))
    assert '2 edits failed' in f and 'edit your best candidate now' in f.splitlines()[0]


def test_budget_finish_with_edit(sim):
    sim.L()
    sim.e(ok=True)
    age(flow.TIME_BUDGET_S * flow.BUDGET_FINISH_FRAC + 5)
    _, f = sim.S(start=1, end=5)
    lines = f.splitlines()
    assert lines[0].endswith('| run check_patch.py and submit_patch now')
    assert lines[1] == 'NEXT: time is nearly up: ' + call('check_patch.py') + ', then submit_patch'
    assert len(lines) == 2
    _, f = sim.C()
    assert nxt(f) == 'NEXT: submit_patch (the patch is ready)'


def test_budget_finish_without_edit(sim):
    sim.L()
    age(flow.TIME_BUDGET_S * flow.BUDGET_FINISH_FRAC + 5)
    _, f = sim.S(start=8, end=11)
    assert f.splitlines()[0].endswith('| edit your best candidate now, then check_patch.py and submit_patch')
    assert 'NEXT: edit your best candidate now' in f


def test_budget_by_call_count(sim):
    """v21 V2: finish escalates only near the harness's hard cap (CALLS_FINISH_AT of TASK_MAX_CALLS), not from
    calls / TARGET_CALLS: 20 calls at real speed are ~150 s of 220. v21b: edit now again from CALLS_EDIT_AT calls."""
    sim.L()
    while flow.State(flow.connect()).n < flow.CALLS_EDIT_AT - 2:
        sim.call('search.py', [f'term{flow.State(flow.connect()).n}'], 0, 'pay/invoice.py:9: x\n')
    _, f = sim.S(start=8, end=11)
    budget = next(l for l in f.splitlines() if l.startswith('BUDGET: '))
    assert re.match(rf'BUDGET: call {flow.CALLS_EDIT_AT - 1}(?: of |/){flow.TASK_MAX_CALLS} \| ', budget), budget
    assert 'edit' not in budget, budget
    _, f = sim.S(start=8, end=10)
    budget = next(l for l in f.splitlines() if l.startswith('BUDGET: '))
    assert re.match(rf'BUDGET: call {flow.CALLS_EDIT_AT}(?: of |/){flow.TASK_MAX_CALLS} \| ', budget), budget
    assert budget.endswith(' | edit now') or budget.endswith(' | edit your best candidate now'), budget
    sim.e(ok=True)
    for i in range(flow.TARGET_CALLS):
        sim.call('search.py', [f'after{i}'])
    _, f = sim.R(code=1)
    assert 'submit_patch now' not in f, f  # 20+ calls with an edit: no call-count finish before CALLS_FINISH_AT
    while flow.State(flow.connect()).n < flow.CALLS_FINISH_AT - 1:
        sim.call('search.py', [f'last{flow.State(flow.connect()).n}'])
    _, f = sim.R(code=1)
    assert 'check_patch.py and submit_patch now' in f
    assert f'NEXT: {flow.TASK_MAX_CALLS - flow.CALLS_FINISH_AT} calls left: ' in f and 'time is nearly up' not in f


def test_budget_on_refusals(sim):
    sim.L()
    sim.S(start=8, end=11)
    age(flow.TIME_BUDGET_S * flow.BUDGET_EDIT_FRAC + 5)
    kind, msg = sim.call('search.py', ['x'], 0, 'pay/invoice.py:1: x\n')
    assert kind == 'ran'
    kind, msg = sim.call('search.py', ['x'])
    assert kind == 'refused' and 'BUDGET: call 4 of 40' in msg and 'edit your best candidate now' in msg


def test_status_shows_budget(sim):
    sim.L()
    sim.S(start=8, end=11)
    age(flow.TIME_BUDGET_S * flow.BUDGET_EDIT_FRAC + 5)
    code, out = run_script('status.py')
    lines = out.strip().splitlines()
    assert any(l.startswith('BUDGET: call 2 of 40') and 'edit your best candidate now' in l for l in lines)
    assert lines[-1].startswith('NEXT: edit your best candidate now')


# ================================================================ agent-v12: payload, no streaks, edit early, repeats
# E024 (v11): in the 17 failed runs that never edited, 48% of the calls were wasted: 47 budget refusals ("NOT RUN:
# 4 look-around calls ...") followed by more refused reads, 30 ALREADY DONE repeats, 24 usage errors. Refusals were
# followed by an applied edit 1 time out of 52. v12: the budget refusal carries the code to change and a pre-filled
# line-range edit (B), a refusal is never followed by another (C), footers push the line-range edit early (E), and
# usage errors / refusals are not "already done" (R). v13 (E025: refusals with the code were followed by an edit 1 time
# in 47): reads are never refused; the over-budget footer pushes the line-range edit and points at the code.

BIG = ('def handle(request, a=1):\n    """Handle a request.\n' + ''.join(f'    doc line {i}\n' for i in range(40))
       + '    """\n' + ''.join(f'    x{i} = request + {i}\n' for i in range(80)) + '    return x0\n')
# lines: 1 def, 2-42 docstring, 43 closing quotes, 44-123 body, 124 return
BIG_LOCATE_OUT = textwrap.dedent('''\
    identifiers: handle
    2.00 pay/big.py  [defines handle]
          handle  L1-124
    0.50 pay/invoice.py
          Invoice.total  L8-11
    --- best match: pay/big.py handle L1-124 (truncated)
    ''') + ''.join(f'{i:5d}  x\n' for i in range(1, 51))


def first_edit(text):
    """The args of the first edit.py call suggested in a footer or refusal."""
    return next(a for s, a in calls('\n'.join(l for l in text.splitlines() if l.startswith('NEXT:'))) if s == 'edit.py')


def test_C_never_two_refusals_in_a_row(sim):
    """After any refusal (repeat, check cap) the next call runs; its footer still pushes the edit."""
    assert [sim.Q('a')[0], sim.Q('b')[0], sim.Q('c')[0]] == ['ran'] * 3  # v13: reads before locate run
    sim.L()
    for i in range(10):
        sim.S('pay/invoice.py', 1 + i, 2 + i)
    for i in range(6):
        sim.R(snippet=f'print({i})')  # v21b: past call CALLS_EDIT_AT the gate starts at 3 snippets: 4th and 6th refused
    for _ in range(4):
        sim.Q('same')
    log = ''.join(sim.log)
    assert 'xx' not in log and log.endswith('x.x.x'), log  # identical repeats still refused, every other call
    assert log.count('x') == 4 and log.startswith('.' * 17 + 'x.x'), log


def test_C_call_after_a_refusal_still_pushes_the_edit(sim):
    sim.L()
    for t in ('a', 'b', 'c'):
        sim.Q(t)
    assert sim.Q('c')[0] == 'refused'  # an identical repeat
    kind, f = sim.Q('e')
    assert kind == 'ran'
    assert nxt(f).splitlines()[0] == (f'NEXT: {flow.N_EXPLORE_FIRST + 1} reads since locate: edit now, even if not '
                                      'fully sure (an imperfect change in the right place beats no change): ' + call(
                                          'edit.py', target='pay/invoice.py', start='8', end='11',
                                          new='<new text for lines 8-11>'))


def test_C_lookaround_before_locate_does_not_use_the_budget(sim):
    sim.Q('a')
    assert sim.Q('b')[0] == 'ran'  # after the rule-1 refusal
    sim.L()
    assert flow.State(flow.connect()).look_since_edit == 0


def test_B_over_budget_footer_edits_the_best_region(sim):
    """v13: no payload; the over-budget footer edits the best region (locate's top file first) and points at it."""
    sim.L()
    sim.S('pay/http.py', 1, 2)
    sim.S('pay/http.py', 5, 6)
    sim.O()
    kind, f = sim.Q('amount')
    assert kind == 'ran'
    assert first_edit(f) == {'target': 'pay/invoice.py', 'start': '8', 'end': '11', 'new': '<new text for lines 8-11>'}
    assert 'the code most likely to change: pay/invoice.py Invoice.total lines 8-11 (shown at call 1)' in f
    assert 'NEXT: or by text' not in f and 'or edit by text: ' + call(
        'edit.py', target='pay/invoice.py', old='<exact current lines>', new='<new lines>') in f
    check_footer(f)


def test_B_region_follows_locate_rank_not_the_latest_view(sim):
    """fastapi_14258 r1: the model read the right function (locate's top file), wandered to the second file, and the
    budget NEXT pushed an edit of the last-viewed wrong file."""
    sim.L()
    sim.S('pay/http.py', 5, 6)
    assert flow.best_region(flow.State(flow.connect())) == ('pay/invoice.py', 8, 11, 'Invoice.total')


def test_B_close_scores_pick_both_files(sim):
    sim.L(out=LOCATE_OUT.replace('0.80 pay/http.py', '1.40 pay/http.py'))
    regs = flow.pick_regions(flow.State(flow.connect()))
    assert [(r['path'], r['label']) for r in regs] == [('pay/invoice.py', 'Invoice.total'), ('pay/http.py', 'send')]


def test_B_distant_scores_pick_one_region(sim):
    sim.L()
    regs = flow.pick_regions(flow.State(flow.connect()))
    assert [(r['path'], r['label']) for r in regs] == [('pay/invoice.py', 'Invoice.total')]


def test_B_never_tests_or_docs(sim):
    sim.L(out=LOCATE_OUT.replace('1.50 pay/invoice.py', '0.10 pay/invoice.py'))
    sim.S('tests/test_invoice.py', 1, 5)
    sim.S('tests/test_invoice.py', 4, 5)
    sim.S('tests/test_invoice.py', 2, 3)
    _, msg = sim.Q('x')
    assert 'tests/' not in msg.split('BUDGET:')[0] and first_edit(msg)['target'] != 'tests/test_invoice.py'


def test_B_every_read_runs_past_the_budget(sim):
    sim.L()
    for t in ('a', 'b', 'c'):
        sim.Q(t)
    assert sim.S('pay/invoice.py', 8, 11)[0] == 'ran'
    assert sim.S('pay/http.py', 5, 6)[0] == 'ran'
    assert sim.O()[0] == 'ran' and set(sim.log) == {'.'}


def test_B_pointer_says_where_the_region_was_shown(sim, ws):
    (ws / 'pay/big.py').write_text(BIG)
    sim.L(out=BIG_LOCATE_OUT)
    for t in ('a', 'b'):
        sim.Q(t)
    _, f = sim.Q('c')
    e = first_edit(f)
    show = call('show.py', target='pay/big.py', start=e['start'], end=e['end'])
    assert f'the code most likely to change: pay/big.py handle lines {e["start"]}-{e["end"]}, not shown yet: {show}' in f
    sim.call('show.py', obj(target='pay/big.py', start=e['start'], end=e['end']), 0,
             f'pay/big.py L{e["start"]}-{e["end"]} of 124\n')
    _, f = sim.Q('d')
    assert f'pay/big.py handle lines {e["start"]}-{e["end"]} (shown at call 5)' in f


def test_B_long_function_body_window(sim, ws):
    (ws / 'pay/big.py').write_text(BIG)
    sim.L(out=BIG_LOCATE_OUT)
    for t in ('a', 'b'):
        sim.Q(t)
    _, f = sim.Q('c')
    e = first_edit(f)
    assert e['target'] == 'pay/big.py' and e['start'] == '44' and int(e['end']) == 44 + flow.PAYLOAD_LINES - 7
    assert 'narrow start/end to the lines you change' in f


def test_B_long_function_window_at_the_viewed_lines(sim, ws):
    (ws / 'pay/big.py').write_text(BIG)
    sim.L(out=BIG_LOCATE_OUT)
    sim.S('pay/big.py', 100, 110)
    sim.Q('a')
    _, f = sim.Q('b')
    assert (first_edit(f)['start'], first_edit(f)['end']) == ('100', '124')


def test_B_no_region_known_falls_back_to_a_text_edit(sim):
    sim.L(out='identifiers: x\n')
    for t in ('a', 'b', 'c'):
        sim.Q(t)
    kind, f = sim.Q('d')
    assert kind == 'ran' and 'edit.py  target=<file.py>' in f and 'prints the closest region' in f


def test_E_budget_is_three_before_the_first_edit_and_four_after(sim):
    sim.L()
    over = [('reads since' in sim.S('pay/invoice.py', i, i)[1]) for i in (1, 2, 3, 4)]
    assert over == [False, False, True, True]
    sim.e(ok=False, old='guess')
    over = [('reads since' in sim.S('pay/invoice.py', i, i)[1]) for i in (5, 6, 7, 12, 13)]
    assert over == [False, False, False, True, True]
    assert flow.N_EXPLORE_FIRST == 3 and flow.N_EXPLORE == 4


def test_E_confident_locate_offers_the_line_edit_first(sim):
    _, f = sim.L()
    lines = nxt(f).splitlines()
    assert lines[0] == 'NEXT: if the function shown above is the place, replace its lines: ' + call(
        'edit.py', target='pay/invoice.py', start='8', end='11', new='<new text for lines 8-11>')
    assert lines[1] == 'NEXT: or by text: ' + call('edit.py', target='pay/invoice.py', old='<exact current lines>',
                                                   new='<new lines>')
    assert lines[2].startswith('NEXT: otherwise read ONE other candidate: show.py  target=pay/http.py')


def test_E_unsure_locate_keeps_the_text_edit(sim):
    _, f = sim.L(out=LOCATE_OUT + 'LOW CONFIDENCE: top files score close\n')
    assert nxt(f).startswith('NEXT: if the function shown above is the place, edit it: ' + call(
        'edit.py', target='pay/invoice.py', old='<exact current lines>', new='<new lines>'))


# v19: after locate, the edit line only when its code was printed; else the read of the best match / top candidate
NO_PREVIEW_OUT = LOCATE_OUT.split('--- best match')[0]


def test_v19_locate_without_preview_reads_the_top_candidate_first(sim):
    _, f = sim.L(out=NO_PREVIEW_OUT)
    lines = nxt(f).splitlines()
    assert lines == ['NEXT: read the top candidate (its code is not shown above): '
                     + call('show.py', target='pay/invoice.py', symbol='Invoice.total'),
                     'NEXT: otherwise read ONE other candidate: '  # confident: the next one in locate's order
                     + call('show.py', target='pay/invoice.py', symbol='Invoice.__init__')]
    assert 'edit.py' not in f


def test_v19_locate_without_preview_low_confidence_offers_another_file(sim):
    _, f = sim.L(out=NO_PREVIEW_OUT + 'low confidence: the top files score close\n')
    lines = nxt(f).splitlines()
    assert lines[0].endswith(call('show.py', target='pay/invoice.py', symbol='Invoice.total'))
    assert lines[1] == 'NEXT: otherwise read ONE other candidate: ' + call('show.py', target='pay/http.py',
                                                                          symbol='send')


def test_v19_read_of_the_top_candidate_then_its_line_edit(sim, ws):
    """locate (nothing shown) -> show.py of the top candidate -> the footer's first NEXT is its line-range edit."""
    sim.L(out=NO_PREVIEW_OUT)
    _, f = sim.S('pay/invoice.py', 8, 11)
    assert nxt(f).splitlines()[0].startswith('NEXT: if this is the code to change (2 reads left before you should '
                                             'edit), replace its lines: ' + call(
                                                 'edit.py', target='pay/invoice.py', start='8', end='11',
                                                 new='<new text for lines 8-11>'))


def test_v19_locate_window_of_a_long_function_reads_it(sim, ws):
    """Only a window of a long function was printed: read the function (show.py symbol), not 'edit it'."""
    (ws / 'pay/big.py').write_text(BIG)
    _, f = sim.L(out=BIG_LOCATE_OUT, shown=('pay/big.py', 60, 75))
    lines = nxt(f).splitlines()
    assert lines[0] == 'NEXT: read the best match (only part of it is shown above): ' + call(
        'show.py', target='pay/big.py', symbol='handle')
    assert lines[1] == 'NEXT: otherwise read ONE other candidate: ' + call('show.py', target='pay/invoice.py',
                                                                          symbol='Invoice.total')
    truncated = LOCATE_OUT.replace('Invoice.total  L8-11', 'Invoice.total  L8-30')  # printed 8-11 of 8-30
    _, f = sim.L(text='again', out=truncated)
    assert nxt(f).startswith('NEXT: read the best match (only part of it is shown above): ' + call(
        'show.py', target='pay/invoice.py', symbol='Invoice.total'))


def test_v19_locate_body_shown_with_folded_docstring_offers_its_line_edit(sim, ws):
    """Only the docstring was folded (the body printed to its end): the function counts as shown, edit early."""
    (ws / 'pay/big.py').write_text(BIG)
    _, f = sim.L(out=BIG_LOCATE_OUT, shown=('pay/big.py', 44, 124))
    lines = nxt(f).splitlines()
    assert lines[0] == 'NEXT: if the function shown above is the place, replace its lines (narrow start/end to the ' \
        'lines you change): ' + call('edit.py', target='pay/big.py', start='44', end='124',
                                     new='<new text for lines 44-124>')
    assert lines[1].startswith('NEXT: or by text: ')


def test_v19_locate_folded_signature_is_not_shown_reads_it(sim, ws):
    """A folded long signature (rich_3052's PromptBase.ask overload: 11 signature lines, body `...`): the change is
    likely in the parameters the model has not seen, so read it rather than edit the body line."""
    (ws / 'pay/ask.py').write_text('def ask(\n    cls,\n    prompt,\n    default=None,\n    stream=None,\n):\n    ...\n')
    out = ('identifiers: ask\n2.00 pay/ask.py  [defines ask]\n      ask  L1-7\n0.50 pay/invoice.py\n'
           '      Invoice.total  L8-11\n--- best match: pay/ask.py ask L1-7 (named in the issue)\n')
    _, f = sim.L(out=out, shown=('pay/ask.py', 7, 7))
    assert nxt(f).startswith('NEXT: read the best match (only part of it is shown above): ' + call(
        'show.py', target='pay/ask.py', symbol='ask'))


def test_v19_second_locate_after_reading_the_top_candidate_keeps_the_edit(sim):
    sim.L(out=NO_PREVIEW_OUT)
    sim.S('pay/invoice.py', 8, 11)
    _, f = sim.L(text='Invoice total None amount', out=NO_PREVIEW_OUT)
    assert nxt(f).startswith('NEXT: if the function shown above is the place, edit it: ')
    assert 'symbol=Invoice.total' not in f


def test_E_show_of_a_candidate_function_offers_its_line_edit(sim):
    sim.L()
    _, f = sim.S('pay/http.py', 5, 6)
    assert nxt(f).splitlines()[0].startswith('NEXT: if this is the code to change (2 reads left before you should '
                                             'edit), replace its lines: ' + call(
                                                 'edit.py', target='pay/http.py', start='5', end='6',
                                                 new='<new text for lines 5-6>'))
    _, f = sim.S('pay/__init__.py', 1, 1)  # not one of locate's files: the text edit first, as before
    assert nxt(f).startswith('NEXT: if this is the code to change (1 read left before you should edit): '
                             'edit.py  target=pay/__init__.py  old=')


def test_R_usage_errors_and_refusals_are_not_already_done(sim):
    sim.L()
    args = obj(symbol='Invoice.total')
    assert sim.call('show.py', args, 2, 'USAGE ERROR: x\n')[0] == 'ran'
    kind, f = sim.call('show.py', args, 0, 'pay/invoice.py Invoice.total L8-11\n')
    assert kind == 'ran'
    sim.Q('a')
    kind, msg = sim.Q('a')
    assert kind == 'refused' and msg.startswith('ALREADY DONE')
    kind, out = sim.Q('a')  # the refused call again: it runs now (and a refusal is not a result to replay)
    assert kind == 'ran'


# ---------------------------------------------------------------- H: phase hook

class DummyPhase:
    """A phase module: the 'planner' phase may not edit before locate ran twice; footers get a phase line."""
    seen = []

    @staticmethod
    def before(st, script, argv):
        DummyPhase.seen.append((script, st.n, st.last_refused, st.located, st.look_since_edit))
        if script == 'edit.py' and sum(1 for a in st.real if a[1] == 'locate.py') < 2:
            return 'PHASE: the planner does not edit; finish the plan first.'
        return None

    @staticmethod
    def footer(st, script, status, lines):
        return [f'NEXT: phase planner ({status}) after {script}'] + list(lines)[:1]


def test_H_phase_hook_before_and_footer(sim, monkeypatch):
    DummyPhase.seen = []
    monkeypatch.setattr(flow, 'phase', DummyPhase)
    _, f = sim.L()
    assert nxt(f).splitlines()[0] == 'NEXT: phase planner (ok) after locate.py'
    assert len(nxt(f).splitlines()) == 2
    kind, msg = sim.e(ok=True)
    assert kind == 'refused' and msg == 'PHASE: the planner does not edit; finish the plan first.'
    st = flow.State(flow.connect())
    assert st.actions[-1][3:] == ('refused', 'phase') and not st.edits
    kind, msg = sim.e(ok=True, old='x')  # a phase refusal is not overridden by rule 3b
    assert kind == 'refused'
    kind, f = sim.Q('a')                  # flow's own rules: 3b after the phase refusal (and no refusal here anyway)
    assert kind == 'ran' and 'phase planner (ok) after search.py' in f
    assert DummyPhase.seen[0] == ('locate.py', 1, False, False, 0) and DummyPhase.seen[2][2] is True
    sim.L('again')
    kind, _ = sim.e(ok=True, old='y')
    assert kind == 'ran'


def test_H_phase_refusal_footer_status(sim, monkeypatch):
    monkeypatch.setattr(flow, 'phase', DummyPhase)
    sim.L()
    sim.Q('x')
    kind, msg = sim.Q('x')  # flow's repeat refusal: the phase footer sees status 'refused'
    assert kind == 'refused' and 'NEXT: phase planner (refused) after search.py' in msg


def test_H_broken_phase_module_is_ignored(sim, monkeypatch):
    class Broken:
        @staticmethod
        def before(st, script, argv):
            raise RuntimeError('boom')

        @staticmethod
        def footer(st, script, status, lines):
            return 1 / 0

    monkeypatch.setattr(flow, 'phase', Broken)
    kind, f = sim.L()
    assert kind == 'ran' and 'replace its lines' in f
    assert sim.e(ok=True)[0] == 'ran'


def test_H_phase_module_found_next_to_flow(ws, tmp_path):
    """phase.py is imported like a sibling module (sys.path); absent, flow.phase is None."""
    assert flow.phase is None
    d = tmp_path / 'phasedir'
    d.mkdir()
    (d / 'phase.py').write_text('def before(st, script, argv):\n'
                                '    return "PHASE STOP " + script if script == "search.py" else None\n'
                                'def footer(st, script, status, lines):\n'
                                '    return ["NEXT: from phase.py"]\n')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(d))
    r = subprocess.run([sys.executable, str(SCRIPTS / 'search.py'), '--text', 'amount'], capture_output=True,
                       text=True, env=env, timeout=60)
    assert r.stdout.strip() == 'PHASE STOP search.py'
    r = subprocess.run([sys.executable, str(SCRIPTS / 'locate.py'), '--query', 'Invoice total amount'],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.stdout.rstrip().endswith('NEXT: from phase.py')
    assert not list(d.glob('__pycache__')) and not list(SCRIPTS.glob('__pycache__'))
