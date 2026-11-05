"""agent-v21 flow rule 14 and V2 (research/notes/16-wrong-patches-real-speed.md, levers V1-V3, V6):

V1 verify gate: after a successful edit check_patch.py says "NOT VERIFIED: <reason>" instead of READY while at least
   VERIFY_MIN_LEFT_S seconds are left, until a run since the LAST successful edit verified it: a related test file ran
   (pass or fail), or a snippet whose output changed vs before the edit (or, with no pre-edit run, exited 0). Not: no
   run, a snippet broken in its own imports, a snippet with the same output as before the edit. The footer gives the
   check (the related tests first). At most VERIFY_GATE_MAX check_patch calls per edit; off when the context is
   nearly full or near the call cap. A failing related test: "may encode the old behaviour" (V6, rich_3130).
V2 time-only budget: "time is nearly up / submit now" from the time; the call count only near the hard cap.
V3 symptom unchanged: run.py prints "SAME OUTPUT as before your edit at call E (call M): ..." or "output changed vs
   before the edit (call M)"; the MEMO says "repro unchanged".
"""

import os
import pathlib
import re
import subprocess
import sys
import textwrap

import pytest

from notation import assert_notation

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'agent/skills/repo-tools/scripts'
sys.path.insert(0, str(SCRIPTS))
import flow  # noqa: E402

TEXT = textwrap.dedent('''\
    class Text:
        def __init__(self, plain=""):
            self.plain = plain
            self._spans = []

        def append(self, text):
            offset = len(self.plain)
            self.plain += text.plain
            for start, end, style in text._spans:
                self._spans.append((start + offset, end + offset, style))
            return self

        def copy(self):
            return Text(self.plain)
    ''')
TEST_TEXT = textwrap.dedent('''\
    from pay.text import Text


    def test_append():
        t = Text("a")
        t.append(Text("b"))
        assert t.plain == "ab"


    def test_append_spans():
        t, other = Text("a"), Text("b")
        other._spans.append((0, 1, "bold"))
        t.append(other)
        assert t._spans == [(1, 2, "bold")]
    ''')
LOCATE_OUT = textwrap.dedent('''\
    identifiers: Text, append
    1.50 pay/text.py  [defines append]
          Text.append  L6-11
    --- best match: pay/text.py Text.append L6-11
        6      def append(self, text):
       11          return self
    ''')
EDIT_ARGS = {'target': 'pay/text.py', 'start': '9', 'end': '9',
             'new': '        for start, end, style in list(text._spans):'}
EDIT2_ARGS = {'target': 'pay/text.py', 'start': '8', 'end': '8', 'new': '        self.plain = self.plain + text.plain'}
TESTS_LINE = 'run.py  test=tests/test_text.py  k=test_append or test_append_spans'
REPRO = 'from pay.text import Text\nt = Text("a")\nt.append(t)\nprint(t.plain)'
SELF_SPANS = 'from pay.text import Text\nt = Text("a")\nt._spans.append((0, 1, "b"))\nt.append(t)\nprint(t._spans)'


def git(ws, *args):
    return subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def ws(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    files = {'pay/__init__.py': '', 'pay/text.py': TEXT, 'tests/__init__.py': '', 'tests/test_text.py': TEST_TEXT,
             'tests/test_misc.py': 'def test_list():\n    assert [1]\n'}
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, 'init', '-q')
    git(root, 'add', '-A')
    git(root, 'commit', '-qm', 'baseline')
    monkeypatch.setenv('WORKSPACE', str(root))
    monkeypatch.setenv('REPO_TOOLS_DIR', str(tmp_path / 'state'))
    monkeypatch.delenv('REPO_TOOLS_VERIFY_MIN_LEFT_S', raising=False)
    monkeypatch.setattr(flow, 'COMPACT_AT', 0)
    return root


def run(script, ws, env=None, **args):
    """A real script under the flow hook (ADK's argv: --key value)."""
    argv = [x for k, v in args.items() for x in ('--' + k, str(v))]
    e = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'), PYTHONDONTWRITEBYTECODE='1',
             REPO_TOOLS_COMPACT_AT='0')
    e.pop('REPO_TOOLS_FLOW', None)
    e.update(env or {})
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *argv], capture_output=True, text=True, env=e,
                       timeout=120)
    assert 'Traceback' not in r.stderr, r.stderr
    return r.returncode, r.stdout


def obj(**kw):
    return [x for k, v in kw.items() for x in ('--' + k, str(v))]


def step(script, args, out='', status=0, **facts):
    msg = flow.before(script, args)
    if msg:
        return 'refused', msg
    return 'ran', flow.after(script, args, status, out, **facts)


def nexts(out):
    return [l for l in out.splitlines() if l.startswith('NEXT:')]


def memo(out):
    return next((l for l in out.splitlines() if l.startswith('MEMO: ')), '')


def age(seconds):
    """Pretend the task started `seconds` earlier."""
    conn = flow.connect()
    conn.execute('UPDATE actions SET ts = ts - ?', (seconds,))
    conn.close()


def state():
    return flow.State(flow.connect())


def locate():
    return step('locate.py', obj(query='Text append loops'), LOCATE_OUT)


def edit(ws, args=EDIT_ARGS):
    code, out = run('edit.py', ws, **args)
    assert code == 0 and out.startswith('EDITED'), out
    return out


# ---------------------------------------------------------------- V1: the gate in check_patch.py

def test_nothing_ran_since_the_edit_is_not_verified(ws):
    """fastapi_14616 / 5624 / 14361 E063: edit -> check_patch -> submit with minutes left and nothing run."""
    locate()
    edit(ws)                                           # call 2
    code, out = run('check_patch.py', ws)              # call 3
    assert code == 1 and 'READY' not in out, out
    line = next(l for l in out.splitlines() if l.startswith('NOT VERIFIED'))
    assert line.startswith('NOT VERIFIED: nothing ran since your edit at call 2. 3m3') and line.endswith(
        ' left: run the check below before you submit, then check_patch.py again.'), line
    assert nexts(out) == [f'NEXT: run the existing tests that use append: {TESTS_LINE}',
                          'NEXT: then check_patch.py and submit_patch'], out
    for l in nexts(out):
        assert_notation(l)
    st = state()
    assert not st.patch_ready and flow.get_meta(st.conn, 'unverified:3') == '1'


def test_a_related_test_run_verifies_pass_or_fail(ws):
    locate()
    edit(ws)
    code, out = run('run.py', ws, test='tests/test_text.py', k='test_append or test_append_spans')
    assert '2 passed' in out
    code, out = run('check_patch.py', ws)
    assert code == 0 and 'READY: call submit_patch now.' in out and 'NOT VERIFIED' not in out, out
    # a failing related test is information too (it may also pin the old behaviour)
    (ws / 'tests/test_text.py').write_text(TEST_TEXT.replace('== "ab"', '== "ba"'))
    git(ws, 'commit', '-qam', 'red test')   # HEAD changes: a new task state
    locate()
    edit(ws, EDIT2_ARGS)
    code, out = run('run.py', ws, test='tests/test_text.py')
    assert 'exit code: 1' in out
    assert nexts(out) == ['NEXT: if the failure comes from your change, fix it with edit.py; a failing test may encode '
                          'the old behaviour that the issue asks to change: if your repro confirms the fix, keep it',
                          'NEXT: then check_patch.py and submit_patch'], out
    code, out = run('check_patch.py', ws)
    assert 'READY: call submit_patch now.' in out, out


def test_an_unrelated_test_file_does_not_verify(ws):
    """fastapi_14986 E060/E061 ran test_local_docs.py, which does not touch the edited code."""
    locate()
    edit(ws)
    run('run.py', ws, test='tests/test_misc.py')
    code, out = run('check_patch.py', ws)
    assert 'NOT VERIFIED: the tests at call 3 do not use the code you changed.' in out, out
    assert nexts(out)[0] == f'NEXT: run the existing tests that use append: {TESTS_LINE}', out


def test_a_snippet_broken_in_its_own_imports_does_not_verify(ws):
    """fastapi_12942 E063: the issue's example with a wrong import -> ImportError -> 'if unrelated: submit'."""
    locate()
    edit(ws)
    code, out = run('run.py', ws, snippet='from pay.text import Tag\nprint(Tag)')
    assert 'ImportError' in out and 'SAME OUTPUT' not in out and 'unrelated' not in out, out
    assert nexts(out)[0] == ('NEXT: fix the snippet of call 3 (ImportError at snippet.py line 1) and rerun it: run.py  '
                             'snippet=<the code of call 3 with its imports fixed>'), out
    assert nexts(out)[1] == f'NEXT: or run the existing tests that use append: {TESTS_LINE}', out
    assert 'broken: ImportError at snippet.py line 1' in flow.memo(state())
    code, out = run('check_patch.py', ws)
    assert ('NOT VERIFIED: the snippet at call 3 failed in its own code (ImportError at snippet.py line 1) before '
            'reaching the repository: the snippet itself is broken, fix its imports and rerun it.') in out, out
    code, out = run('run.py', ws, snippet='from pay.text import Text\nprint(undefined_name)')
    assert 'NameError' in out and nexts(out)[0].startswith('NEXT: fix the snippet of call 5 (NameError at snippet.py '
                                                           'line 2)'), out


def test_an_error_inside_the_repository_is_not_a_broken_snippet(ws):
    """An ImportError raised by the edited package (its last frame is a repository file) is about the edit."""
    (ws / 'pay/broken.py').write_text('from pay.text import Missing\n')
    git(ws, 'add', '-A')
    git(ws, 'commit', '-qm', 'x')
    locate()
    edit(ws)
    code, out = run('run.py', ws, snippet='import pay.broken')
    assert 'ImportError' in out and 'fix the snippet' not in out, out
    assert flow.setup_error(out) == ''
    assert nexts(out)[0].startswith('NEXT: if the failure comes from your change, fix it with edit.py; if it is the '
                                    'behaviour the issue asks for, run the existing tests that use append: '), out


def test_setup_error_unit():
    tb = 'Traceback (most recent call last):\n  File "snippet.py", line {n}, in <module>\n    x\n{exc}\nexit code: 1'
    assert flow.setup_error(tb.format(n=1, exc="ImportError: cannot import name 'Tag' from 'fastapi.params'")) == \
        'ImportError at snippet.py line 1'
    assert flow.setup_error(tb.format(n=3, exc="ModuleNotFoundError: No module named 'numpy'")) == \
        'ModuleNotFoundError at snippet.py line 3'
    assert flow.setup_error(tb.format(n=2, exc="NameError: name 'Foo' is not defined")) == \
        'NameError at snippet.py line 2'
    assert flow.setup_error('  File "snippet.py", line 4\n    print(\n         ^\nSyntaxError: \'(\' was never closed\n'
                            'exit code: 1') == 'SyntaxError at snippet.py line 4'
    # the snippet's own assertion, a TypeError at the call site, a repository frame: information, not broken
    assert flow.setup_error(tb.format(n=5, exc='AssertionError')) == ''
    assert flow.setup_error(tb.format(n=5, exc="TypeError: Depends() got an unexpected keyword argument 'scope'")) == ''
    assert flow.setup_error('Traceback (most recent call last):\n  File "snippet.py", line 1, in <module>\n'
                            '  File "/workspace/pay/a.py", line 3, in <module>\nImportError: x\nexit code: 1') == ''
    assert flow.setup_error('42\nexit code: 0') == ''


def test_snippet_with_no_pre_edit_run_verifies_when_it_exits_0(ws):
    locate()
    edit(ws)
    code, out = run('run.py', ws, snippet='raise SystemExit(3)')
    code, out = run('check_patch.py', ws)
    assert 'NOT VERIFIED: the snippet at call 3 failed (exit 3) and there is no run of it before the edit.' in out, out
    run('run.py', ws, snippet=REPRO)
    code, out = run('check_patch.py', ws)
    assert 'READY: call submit_patch now.' in out, out


def test_gate_is_per_last_edit(ws):
    locate()
    edit(ws)
    run('run.py', ws, test='tests/test_text.py')
    code, out = run('check_patch.py', ws)
    assert 'READY' in out
    edit(ws, EDIT2_ARGS)                                       # a new edit: verify again
    code, out = run('check_patch.py', ws)
    assert 'NOT VERIFIED: nothing ran since your edit at call 5.' in out, out


# ---------------------------------------------------------------- V1: time threshold edges, caps, context

def test_time_threshold_edges(ws):
    locate()
    edit(ws)
    st = state()
    st.now = st.first_ts + flow.TIME_BUDGET_S - flow.VERIFY_MIN_LEFT_S          # exactly VERIFY_MIN_LEFT_S left
    assert flow.verify_gate(st) == 'nothing ran since your edit at call 2'
    st.now += 0.5                                                              # just under it: READY as before
    assert flow.verify_gate(st) is None
    assert flow.VERIFY_MIN_LEFT_S == 60


def test_time_threshold_with_real_scripts(ws):
    locate()
    edit(ws)
    age(flow.TIME_BUDGET_S - flow.VERIFY_MIN_LEFT_S - 5)       # ~65 s left: gated
    code, out = run('check_patch.py', ws)
    assert 'NOT VERIFIED: nothing ran since your edit at call 2. 1m0' in out, out
    age(10)                                                    # ~55 s left: READY, as v20b
    code, out = run('check_patch.py', ws)
    assert 'READY: call submit_patch now.' in out and 'NOT VERIFIED' not in out, out


def test_at_most_two_gated_check_patch_calls_per_edit(ws):
    locate()
    edit(ws)
    code, out = run('check_patch.py', ws)                      # call 3: NOT VERIFIED
    assert 'NOT VERIFIED' in out
    code, out = run('check_patch.py', ws)                      # call 4: identical, nothing ran: replayed (rule 2)
    assert out.startswith('ALREADY DONE at call 3') and 'NOT VERIFIED' in out, out
    assert nexts(out)[0] == f'NEXT: run the existing tests that use append: {TESTS_LINE}', out
    code, out = run('check_patch.py', ws)                      # call 5: VERIFY_GATE_MAX reached: READY
    assert 'READY: call submit_patch now.' in out and 'NOT VERIFIED' not in out, out


def test_check_patch_is_not_replayed_after_a_run(ws):
    locate()
    edit(ws)
    run('check_patch.py', ws)
    run('run.py', ws, snippet=REPRO)
    code, out = run('check_patch.py', ws)
    assert 'ALREADY DONE' not in out and 'READY: call submit_patch now.' in out, out


def test_gate_off_when_the_context_is_nearly_full(ws):
    """Rule 10 wins: a context overflow is graded with an EMPTY patch."""
    locate()
    edit(ws)
    conn = flow.connect()
    flow.set_meta(conn, 'ctx_chars', str(int((flow.CTX_WARN_TOKENS - flow.CTX_BASE_TOKENS) / flow.CTX_TOKENS_PER_CHAR)))
    conn.close()
    code, out = run('check_patch.py', ws)
    assert 'READY: call submit_patch now.' in out, out


def test_gate_off_near_the_call_cap(ws):
    locate()
    edit(ws)
    for i in range(flow.CALLS_FINISH_AT - 2):
        step('status.py', [], '')
    code, out = run('check_patch.py', ws)
    assert 'READY: call submit_patch now.' in out, out


def test_check_caps_step_aside_while_unverified(ws):
    """Rule 5: 'N checks since the last edit: stop verifying' and the hard cap do not stop the check the gate asks
    for."""
    locate()
    edit(ws)
    for i in range(flow.MAX_CHECKS_HARD):
        code, out = run('run.py', ws, snippet=f'from pay.text import Missing{i}')
        assert 'stop verifying' not in out and 'NOT RUN' not in out, out
    code, out = run('run.py', ws, test='tests/test_text.py')
    assert 'NOT RUN' not in out and '2 passed' in out, out
    code, out = run('run.py', ws, snippet=REPRO)                 # verified now: the hard cap applies again
    assert out.startswith('NOT RUN: 4 checks since the last edit'), out


def test_no_tests_collected_does_not_verify(ws):
    locate()
    edit(ws)
    code, out = run('run.py', ws, test='tests/test_text.py', k='nomatch')
    assert 'exit code: 5' in out
    assert nexts(out) == ['NEXT: no test matched; run the whole file: run.py  test=tests/test_text.py'], out
    code, out = run('check_patch.py', ws)
    assert 'NOT VERIFIED: the test run at call 3 ran no tests (exit 5).' in out, out


def test_post_edit_footer_points_to_the_tests_first(ws):
    locate()
    run('run.py', ws, snippet=REPRO)                       # call 2: the repro
    out = edit(ws)                                         # call 3
    assert nexts(out) == [f'NEXT: verify once: {TESTS_LINE}  (the existing tests that use append)',
                          'NEXT: then re-run your repro of call 2, unchanged: run.py  snippet=<the same code as call 2>'
                          '  then check_patch.py and submit_patch'], out
    code, out = run('run.py', ws, test='tests/test_text.py', k='test_append or test_append_spans')
    assert nexts(out) == ['NEXT: the tests pass; now re-run your repro of call 2, unchanged: run.py  snippet=<the same '
                          'code as call 2>', 'NEXT: then check_patch.py and submit_patch'], out


# ---------------------------------------------------------------- V3: symptom unchanged

def test_same_output_after_the_edit(ws):
    """rich_3180: the post-edit repro printed exactly the pre-edit buggy output, and the run submitted."""
    locate()
    run('run.py', ws, snippet=REPRO)                       # call 2
    edit(ws)                                               # call 3
    code, out = run('run.py', ws, snippet=REPRO)           # call 4: same code, same output
    assert ('SAME OUTPUT as before your edit at call 3 (call 2): the change did not affect this; if it shows the bug, '
            'the cause is elsewhere') in out, out
    assert nexts(out)[0].startswith('NEXT: your edit did not change this output (call 2 printed the same before it): '
                                    'if it shows the bug, the cause is elsewhere: edit the code that produces it: '
                                    'edit.py  target=pay/text.py'), out
    assert 'repro unchanged (call 4 = call 2 before the edit)' in memo_of_state()
    code, out = run('check_patch.py', ws)
    assert ('NOT VERIFIED: the snippet at call 4 printed the SAME OUTPUT as before your edit (call 2): the edit did not '
            'change what it shows.') in out, out


def memo_of_state():
    return flow.memo(state())


def test_changed_output_after_the_edit(ws):
    locate()
    run('run.py', ws, snippet=SELF_SPANS)                  # call 2: [(0, 1, 'b'), (1, 2, 'b')] or a loop
    edit(ws)
    code, out = run('run.py', ws, snippet=SELF_SPANS + '\n')   # whitespace differs: the same code
    assert 'output changed vs before the edit (call 2)' in out and 'SAME OUTPUT' not in out, out
    code, out = run('check_patch.py', ws)
    assert 'READY: call submit_patch now.' in out, out


def test_similar_code_is_compared_short_code_only_when_identical(ws):
    locate()
    step('run.py', obj(snippet='print(1)'), 'x\nexit code: 0\n')                                   # call 2
    long = 'from pay.text import Text\nt = Text("abc")\nt.append(Text("d"))\nprint(t.plain, len(t._spans))'
    step('run.py', obj(snippet=long), 'abcd 0\nexit code: 0\n')                                    # call 3
    step('edit.py', obj(**EDIT_ARGS), 'EDITED pay/text.py (lines)\n')                              # call 4
    st = state()
    assert flow.pre_edit_run(st, 'print(2)', 'x\nexit code: 0') is None          # short: not the same code
    assert flow.pre_edit_run(st, 'print(1)', 'x\nexit code: 0') == (2, True)
    assert flow.pre_edit_run(st, long.replace('"d"', '"e"'), 'abce 0\nexit code: 0') == (3, False)
    assert flow.pre_edit_run(st, long + '\nprint("done")', 'abcd 0\nexit code: 0') == (3, True)
    # other code, identical output: compared only when the output is long enough to mean something
    body = 'Lines([<text \' \' []>])\nexit code: 0'
    assert flow.pre_edit_run(st, 'import os\nprint(os.sep)', 'abcd 0\nexit code: 0') is None
    step('run.py', obj(snippet='from x import y'), body + '\n')                                   # call 5 (post-edit)
    step('edit.py', obj(**EDIT2_ARGS), 'EDITED pay/text.py (lines)\n')                             # call 6
    st = state()
    assert flow.pre_edit_run(st, 'print("something else entirely")', body) == (5, True)
    # rich_3180: the pre-edit run printed the buggy line twice, the post-edit one once: nothing new
    twice = body.replace('exit code', "Lines([<text ' ' []>])\nexit code")
    step('run.py', obj(snippet='print("x" * 99)'), twice + '\n')                                  # call 7
    step('edit.py', obj(**EDIT_ARGS), 'EDITED pay/text.py (lines)\n')                             # call 8
    assert flow.pre_edit_run(state(), 'print("x" * 99)\nprint(1)', body) == (7, True)
    assert flow.pre_edit_run(state(), 'print("x" * 99)', body.replace('exit code: 0', 'exit code: 1')) == (7, False)
    assert flow.pre_edit_run(state(), 'print("x" * 99)', 'new line\n' + body) == (7, False)


def test_v3_masks_addresses_and_follows_same_output_references(ws):
    locate()
    out1 = '<pay.text.Text object at 0x10a2b3c40>\n' + 'z' * 200 + '\nexit code: 0\n'
    step('run.py', obj(snippet='print(1)'), out1)                                                    # call 2
    step('run.py', obj(snippet='print(2)'), 'same output as call 2; its last line: zzz\nexit code: 0\n')  # call 3
    step('edit.py', obj(**EDIT_ARGS), 'EDITED pay/text.py (lines)\n')                                # call 4
    st = state()
    out2 = out1.replace('0x10a2b3c40', '0x7f00aa11bb22')
    assert flow.pre_edit_run(st, 'print(2)', out2) == (3, True)
    assert flow.pre_edit_run(st, 'print(1)', out2.replace('zzz', 'zzy', 1)) == (2, False)


def test_v3_facts_computed_without_run_py(ws):
    """Facts parsed from the output when a caller (a replay, a test) does not supply them."""
    locate()
    step('run.py', obj(snippet=REPRO), 'aa\nexit code: 0\n')
    step('edit.py', obj(**EDIT_ARGS), 'EDITED pay/text.py (lines)\n')
    _, f = step('run.py', obj(snippet=REPRO), 'aa\nexit code: 0\n')
    assert flow.run_info(state(), 4) == {'setup': '', 'pre': 2, 'same': True}
    assert nexts(f)[0].startswith('NEXT: your edit did not change this output'), f
    assert flow.verify_state(state())[0] is False


# ---------------------------------------------------------------- V2: time-only budget

def test_many_fast_calls_do_not_say_submit_now(ws):
    """rich_3180/3063/4070: at real speed 20 calls fit in ~150 s and the call count said 'time is nearly up'."""
    locate()
    for i in range(22):
        step('status.py', [], '')
    age(100)                                               # ~120 s left
    _, f = step('edit.py', obj(**EDIT_ARGS), 'EDITED pay/text.py (lines)\n')
    budget = next(l for l in f.splitlines() if l.startswith('BUDGET: '))
    assert re.fullmatch(r'BUDGET: call 24(?: of |/)40 \| 1m4\ds(?: of |/)3m40s(?: used)?', budget), budget
    assert 'nearly up' not in f and 'submit_patch now' not in f, f


def test_time_phase_edges():
    class St:
        n, has_edit, first_ts, compact = 5, True, 1000.0, False
    st = St()
    st.now = st.first_ts + flow.TIME_BUDGET_S * flow.BUDGET_FINISH_FRAC - 0.5
    assert flow.budget(st)[0] == 'ok'
    st.now += 1
    assert flow.budget(st)[0] == 'finish'
    st.has_edit = False
    st.now = st.first_ts + flow.TIME_BUDGET_S * flow.BUDGET_EDIT_FRAC + 0.5
    assert flow.budget(st)[0] == 'edit'
    st.now -= 1
    assert flow.budget(st)[0] == 'ok'
    st.n = flow.CALLS_EDIT_AT
    assert flow.budget(st)[0] == 'edit'
    st.n, st.has_edit = flow.CALLS_FINISH_AT, True
    assert flow.budget(st)[0] == 'finish'
    st.n = flow.CALLS_FINISH_AT - 1
    assert flow.budget(st)[0] == 'ok'
    # 'finish' starts with (1 - BUDGET_FINISH_FRAC) * TIME_BUDGET_S = 44 s left: below the gate's 60 s
    assert flow.TIME_BUDGET_S * (1 - flow.BUDGET_FINISH_FRAC) < flow.VERIFY_MIN_LEFT_S


def test_run_py_test_timeout_shrinks_with_the_time_left(ws):
    import run as run_mod
    locate()
    assert run_mod.test_timeout() == run_mod.TIMEOUT
    age(flow.TIME_BUDGET_S - 70)                           # 70 s left: 40 s for the test
    assert 39 <= run_mod.test_timeout() <= 40
    age(60)                                                # 10 s left: the minimum
    assert run_mod.test_timeout() == run_mod.TEST_TIMEOUT_MIN


def test_a_broken_pre_edit_snippet_is_not_suggested_unchanged(ws):
    """fastapi_12942 E063: the only pre-edit snippet died on its own import; 'the same code as call N' would die
    again."""
    (ws / 'tests/test_text.py').unlink()
    git(ws, 'commit', '-qam', 'no tests')
    locate()
    run('run.py', ws, snippet='from pay.text import Tag\nprint(Tag)')            # call 2: broken
    out = edit(ws)                                                               # call 3
    assert nexts(out)[0] == ('NEXT: verify once: run.py  snippet=<the code of call 2 with its imports fixed>  (your '
                             'snippet of call 2 with its imports fixed)'), out
    code, out = run('check_patch.py', ws)
    assert nexts(out)[0] == ('NEXT: re-run your snippet of call 2 with its imports fixed: run.py  snippet=<the code '
                             'of call 2 with its imports fixed>'), out
    run('run.py', ws, snippet='from pay.text import Text\nprint(Text)')          # call 5: a good one
    run('run.py', ws, snippet='from pay.text import Text\nprint(Text("a").plain)')
    out = edit(ws, EDIT2_ARGS)                                                   # the latest good snippet
    assert 'snippet=<the same code as call 6>' in nexts(out)[0], out


def test_generic_function_names_do_not_pick_the_related_tests(ws):
    """fastapi_15661 E063: an edit in a script's main() suggested tests/test_dependency_overrides.py k=main."""
    (ws / 'pay/cli.py').write_text('def main():\n    return 1\n')
    (ws / 'tests/test_cli_other.py').write_text('def test_main():\n    main = 1\n    assert main\n')
    git(ws, 'add', '-A')
    git(ws, 'commit', '-qm', 'cli')
    locate()
    out = edit(ws, {'target': 'pay/cli.py', 'start': '2', 'end': '2', 'new': '    return 2'})
    assert 'k=main' not in out and 'test_cli_other' not in out, out


def test_v21b_edit_phase_from_call_10_and_late_snippet_cap():
    """v21b (E065): edit now again from CALLS_EDIT_AT calls or BUDGET_EDIT_FRAC of the time; finish stays time-only
    below CALLS_FINISH_AT; once the edit phase is reached the snippet gate starts at SNIPPETS_BEFORE_EDIT_LATE."""
    class St:
        n, has_edit, first_ts, compact = 5, False, 1000.0, False
    st = St()
    st.now = st.first_ts + 30
    assert flow.budget(st)[0] == 'ok' and flow.snippet_cap(st) == flow.MAX_SNIPPETS_BEFORE_EDIT
    st.n = flow.CALLS_EDIT_AT
    assert flow.budget(st)[0] == 'edit' and flow.snippet_cap(st) == flow.SNIPPETS_BEFORE_EDIT_LATE
    st.n, st.now = 5, st.first_ts + flow.TIME_BUDGET_S * 0.4 + 0.5
    assert flow.budget(st)[0] == 'edit' and flow.snippet_cap(st) == flow.SNIPPETS_BEFORE_EDIT_LATE
    st.has_edit, st.n, st.now = True, flow.CALLS_FINISH_AT - 1, st.first_ts + 30
    assert flow.budget(st)[0] == 'ok' and flow.snippet_cap(st) == flow.MAX_SNIPPETS_BEFORE_EDIT
    assert flow.CALLS_EDIT_AT == 10 and flow.SNIPPETS_BEFORE_EDIT_LATE == 3
