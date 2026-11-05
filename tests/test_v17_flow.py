"""agent-v17 flow.py: levers L1, L5, L6 of research/notes/12-failures-eval113.md.

- L1 context: a calibrated, conservative estimate (fit over 4028 E037 calls + p95 residual) replaces v16's chars/3.5;
  flow.context_tokens() for other scripts; from CTX_WARN_TOKENS "context nearly full" NEXT lines, from CTX_STOP_TOKENS
  only the edit (no edit yet) or refused reads/snippets (edit in place); from COMPACT_FROM_TOKENS compact footers
  (2 NEXT lines, no secondary lines, <= 260 chars, short BUDGET line).
- L5 snippet gate: from MAX_SNIPPETS_BEFORE_EDIT pre-edit snippets on, snippets are refused until an edit (rule 3b:
  every second one runs), naming the region and its line-range edit.
- L6 tests after an edit: repro re-run, then the existing tests that mention the edited function (run.py test= k=),
  then check_patch and submit; bounded scan; not repeated; only with >= 45 s left.
v18c (tests/test_v18c_flow.py): the sawtooth compaction model resets the estimate at 14,336, so the warn/stop tests
here run with flow.COMPACT_AT = 0 (rule 10 as the safety net without compaction); the compact footers keep v17b's own
estimate and threshold (flow.ctx_estimate_v17b, FOOTER_COMPACT_FROM_TOKENS); a MEMO line may precede a footer.
Run with scripts/test_skills.sh (sandbox image, py3.13).
"""

import os
import pathlib
import subprocess
import sys
import textwrap
import time

import pytest

from notation import assert_notation, calls

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


    def test_copy():
        assert Text("x").copy().plain == "x"
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
EDIT_OUT = textwrap.dedent('''\
    EDITED pay/text.py (lines)
    --- a/pay/text.py
    +++ b/pay/text.py
    @@ -8,3 +8,3 @@
             self.plain += text.plain
    -        for start, end, style in text._spans:
    +        for start, end, style in list(text._spans):
                 self._spans.append((start + offset, end + offset, style))
    ''')
TESTS_LINE = 'run.py  test=tests/test_text.py  k=test_append or test_append_spans'


def git(ws, *args):
    return subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def ws(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    files = {'pay/__init__.py': '', 'pay/text.py': TEXT, 'tests/__init__.py': '', 'tests/test_text.py': TEST_TEXT,
             'tests/test_misc.py': 'def test_list():\n    x = []\n    x.append(1)\n    assert x\n',
             # big or irrelevant directories are not scanned, however often they mention the name
             'node_modules/pkg/test_x.py': 'def test_a():\n' + '    append()\n' * 50,
             'docs/test_doc.py': 'def test_b():\n' + '    append()\n' * 50}
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, 'init', '-q')
    git(root, 'add', '-A')
    git(root, 'commit', '-qm', 'baseline')
    monkeypatch.setenv('WORKSPACE', str(root))
    monkeypatch.setenv('REPO_TOOLS_DIR', str(tmp_path / 'state'))
    monkeypatch.setattr(flow, 'COMPACT_AT', 0)   # v18c: rule 10 without the compaction model (the safety net)
    return root


def run(script, ws, **args):
    """A real script under the flow hook."""
    argv = [x for k, v in args.items() for x in ('--' + k, str(v))]
    env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'), PYTHONDONTWRITEBYTECODE='1',
               REPO_TOOLS_COMPACT_AT='0')
    env.pop('REPO_TOOLS_FLOW', None)
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *argv], capture_output=True, text=True, env=env,
                       timeout=120)
    assert 'Traceback' not in r.stderr, r.stderr
    return r.returncode, r.stdout


def obj(**kw):
    return [x for k, v in kw.items() for x in ('--' + k, str(v))]


def step(script, args, out='', status=0, **facts):
    """One call through flow.before/after, as flow.main does it: ('refused', text) or ('ran', footer)."""
    msg = flow.before(script, args)
    if msg:
        return 'refused', msg
    return 'ran', flow.after(script, args, status, out, **facts)


def nexts(out):
    return [l for l in out.splitlines() if l.startswith('NEXT:')]


def snippet(i, body='42', code=0):
    return obj(snippet=f'print({i})'), f'{body}\nexit code: {code}\n'


def set_ctx(tokens):
    """Make the stored context estimate (before the next call) about `tokens`."""
    conn = flow.connect()
    n = flow._ctx_parts(conn)[1]
    chars = (tokens - flow.ctx_estimate(0, n)) / flow.CTX_TOKENS_PER_CHAR
    flow.set_meta(conn, 'ctx_chars', str(max(0, int(chars) + 1)))
    # v18c: rule 8b's compact footers keep v17b's own (cumulative) estimate: set it to the same value
    n = conn.execute('SELECT COUNT(*) FROM actions').fetchone()[0]
    chars = (tokens - flow.ctx_estimate_v17b(0, n)) / flow.V17B_CTX_TOKENS_PER_CHAR
    flow.set_meta(conn, 'ctx_chars_all', str(max(0, int(chars) + 1)))
    conn.close()


def age(seconds):
    """Pretend the task started `seconds` earlier."""
    conn = flow.connect()
    conn.execute('UPDATE actions SET ts = ts - ?', (seconds,))
    conn.close()


def locate():
    return step('locate.py', obj(query='Text append loops'), LOCATE_OUT)


def edit():
    return step('edit.py', obj(**EDIT_ARGS), EDIT_OUT)


def show(i=0, size=200):
    return step('show.py', obj(target='pay/text.py', start=str(1 + i), end='14'),
                f'pay/text.py L{1 + i}-14 of 14\n' + 'x' * size)


# ---------------------------------------------------------------- L1: the calibrated estimate

def test_l1_estimate_is_the_fit_plus_its_margin():
    """v18c: the sawtooth refit on E041 + E042 (tests/test_v18c_flow.py has the compaction model)."""
    assert (flow.CTX_BASE_TOKENS, flow.CTX_TOKENS_PER_CHAR, flow.CTX_TOKENS_PER_CALL, flow.CTX_MARGIN_TOKENS) == \
        (5240, 0.2953, 336, 2000)
    assert flow.ctx_estimate(0, 0) == 5240 + 2000
    # a typical overflowing run (E037 fastapi_14262 r2 at its 14th call): ~ 45k chars over 14 calls
    assert flow.ctx_estimate(45000, 14) == int(5240 + 0.2953 * 45000 + 336 * 14 + 2000)
    assert flow.COMPACT_FROM_TOKENS == 14000 < flow.CTX_WARN_TOKENS == 24000 < flow.CTX_STOP_TOKENS == 27500 \
        < flow.CTX_LIMIT_TOKENS == 31232
    assert not hasattr(flow, 'CHARS_PER_TOKEN')  # v16's heuristic is gone


def test_l1_context_tokens_counts_outputs_arguments_and_calls(ws):
    assert flow.context_tokens() == flow.ctx_estimate(0, 0)
    locate()
    show(size=3000)
    st = flow.State(flow.connect())
    assert st.ctx_chars >= len(LOCATE_OUT) + 3000 and len(st.actions) == 2
    assert flow.context_tokens() == st.ctx_tokens == flow.ctx_estimate(st.ctx_chars, 2)
    assert isinstance(flow.context_tokens(), int)


def test_l1_context_tokens_is_zero_on_errors(ws, tmp_path, monkeypatch):
    (tmp_path / 'afile').write_text('x')
    monkeypatch.setenv('REPO_TOOLS_DIR', str(tmp_path / 'afile' / 'state'))  # cannot be created
    assert flow.context_tokens() == 0


def test_l1_context_tokens_from_inside_a_script(ws):
    """show.py/search.py call flow.context_tokens() inside main(): the calls before this one."""
    locate()
    set_ctx(16000)
    code = ('import sys; sys.dont_write_bytecode = True; sys.path.insert(0, %r)\n'
            'import flow; t = flow.context_tokens(); print(t, t >= flow.COMPACT_FROM_TOKENS)' % str(SCRIPTS))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env).stdout.split()
    assert 16000 <= int(out[0]) < 16300 and out[1] == 'True', out


# ---------------------------------------------------------------- L1: warn and stop

def test_l1_warn_without_an_edit_pushes_the_line_range_edit(ws):
    locate()
    set_ctx(flow.CTX_WARN_TOKENS + 500)
    kind, f = show()
    assert kind == 'ran'
    first = nexts(f)[0]
    assert first.startswith('NEXT: context nearly full: make your edit now: edit.py  target=pay/text.py  start=6'), f
    for line in nexts(f):
        assert_notation(line)


def test_l1_warn_with_an_edit_says_check_patch_then_submit(ws):
    locate()
    edit()
    set_ctx(flow.CTX_WARN_TOKENS + 500)
    kind, f = show()
    assert kind == 'ran'                                  # below CTX_STOP_TOKENS reads still run
    assert nexts(f) == ['NEXT: context nearly full: check_patch.py, then submit_patch now'], f
    _, f = step('check_patch.py', [], 'changed files:\n  pay/text.py | 2 +-\nREADY: call submit_patch now.\n')
    assert nexts(f) == ['NEXT: submit_patch (the patch is ready)'], f


def test_l1_warn_keeps_the_repair_line_of_a_failed_edit(ws):
    locate()
    set_ctx(flow.CTX_WARN_TOKENS + 500)
    step('edit.py', obj(target='pay/text.py', old='a', new='b'), 'NOT APPLIED: OLD not found\nCLOSEST: pay/text.py 9 10\n',
         status=1)
    _, f = step('edit.py', obj(target='pay/text.py', old='c', new='d'),
                'NOT APPLIED: OLD not found\nCLOSEST: pay/text.py 9 10\n', status=1)
    first = nexts(f)[0]
    assert first.startswith('NEXT: context nearly full: 2 edits failed; stop matching text and replace the lines by '
                            'number'), f
    assert ('edit.py', {'target': 'pay/text.py', 'start': '9', 'end': '10', 'new': '<new text for lines 9-10>'}) \
        in calls(first)


def test_l1_stop_without_an_edit_refuses_reads_and_leaves_only_the_edit(ws):
    """v17b (E039 fastapi_14262: 20 reads after "make your edit now", then an overflow with an empty patch)."""
    locate()
    set_ctx(flow.CTX_STOP_TOKENS + 500)
    for kind, msg in (show(), step('run.py', *snippet(1)), step('search.py', obj(text='plain'))):
        assert kind == 'refused' and msg.startswith('NOT RUN: context nearly full (~'), msg   # no 3b either
        assert 'there is no edit yet: edit now' in msg
        lines = nexts(msg)
        assert len(lines) == 1 and 'edit.py  target=' in lines[0], msg
    assert step('run.py', obj(test='tests/test_text.py'), '2 passed\nexit code: 0\n')[0] == 'ran'
    assert step('edit.py', obj(**EDIT_ARGS), EDIT_OUT)[0] == 'ran'


def test_l1_stop_with_an_edit_refuses_reads_and_snippets_but_not_finishing(ws):
    locate()
    edit()
    set_ctx(flow.CTX_STOP_TOKENS + 500)
    refused = [step('show.py', obj(target='pay/text.py', symbol='Text.copy')),
               step('search.py', obj(text='plain')),                  # right after a refusal: still refused
               step('locate.py', obj(query='something else')),
               step('run.py', *snippet(7)),
               step('outline.py', obj(files='pay/text.py'))]
    for kind, msg in refused:
        assert kind == 'refused' and msg.startswith('NOT RUN: context nearly full (~'), msg
        assert 'EMPTY patch' in msg and nexts(msg) == ['NEXT: context nearly full: check_patch.py, then submit_patch now']
    assert step('run.py', obj(test='tests/test_text.py'), '2 passed\nexit code: 0\n')[0] == 'ran'
    assert step('edit.py', obj(**dict(EDIT_ARGS, new='        for start, end, style in tuple(text._spans):')),
                EDIT_OUT)[0] == 'ran'
    kind, f = step('check_patch.py', [], 'changed files:\n  pay/text.py | 2 +-\nREADY: call submit_patch now.\n')
    assert kind == 'ran' and nexts(f) == ['NEXT: submit_patch (the patch is ready)']


def test_l1_replay_real_scripts_stop(ws):
    """Real scripts under the hook: after the edit, with the context at the stop line, show.py is not run."""
    code, out = run('locate.py', ws, query='Text.append pay/text.py')
    code, out = run('edit.py', ws, **EDIT_ARGS)
    assert code == 0 and out.startswith('EDITED pay/text.py'), out
    set_ctx(flow.CTX_STOP_TOKENS + 100)
    code, out = run('show.py', ws, target='pay/text.py', symbol='Text.copy')
    assert out.startswith('NOT RUN: context nearly full') and 'def copy' not in out, out
    code, out = run('check_patch.py', ws)
    assert 'READY' in out, out


# ---------------------------------------------------------------- L1: compact footers

def test_l1_compact_footer_two_short_lines_no_text_alternative(ws):
    _, normal = locate()
    assert any(l.startswith('NEXT: or by text: ') for l in nexts(normal)), normal   # below the line: unchanged
    set_ctx(flow.FOOTER_COMPACT_FROM_TOKENS + 200)
    _, f = step('locate.py', obj(query='Text append spans offset'), LOCATE_OUT)
    nx = nexts(f)
    assert 1 <= len(nx) <= flow.COMPACT_NEXT_LINES, f
    assert not any(l.startswith(('NEXT: or by text', 'NEXT: if the old text')) for l in nx), f
    assert ('edit.py', {'target': 'pay/text.py', 'start': '6', 'end': '11', 'new': '<new text for lines 6-11>'}) \
        in calls(nx[0]), f
    for i in range(3):  # the 3rd read is over the exploration budget: the pointer line loses its old-text tail
        _, f = show(i)
    lines = [l for l in f.splitlines() if not l.startswith('MEMO: ')]   # v18c: a MEMO line may come first
    assert lines[0].startswith('BUDGET: call 5/40 | 0m0') and lines[0].endswith('s/3m40s'), f
    nx = nexts(f)
    assert len(nx) == 2 and nx[0].startswith('NEXT: 3 reads since locate: edit now'), f
    assert nx[1].startswith('NEXT: the code most likely to change: pay/text.py Text.append lines 6-11 (shown at call '), f
    assert 'by text' not in f and all(len(l) <= flow.COMPACT_LINE_CHARS for l in lines), f
    for line in nx:
        assert_notation(line)


def test_l1_compact_budget_line_keeps_the_phase(ws):
    locate()
    set_ctx(flow.FOOTER_COMPACT_FROM_TOKENS + 200)
    age(230)
    _, f = show()
    first = [l for l in f.splitlines() if not l.startswith('MEMO: ')][0]   # v18c: a MEMO line may come first
    assert first.startswith('BUDGET: call 2/40 | 3m5') and first.endswith(' | edit now, then submit_patch'), f


def test_l1_fit_line_never_cuts_the_call():
    edit_call = 'edit.py  target=pay/text.py  start=6  end=11  new=<new text for lines 6-11>'
    lead = ('NEXT: 12 reads since locate: edit now, even if not fully sure (an imperfect change in the right place '
            'beats no change) (narrow start/end to the lines you change), the code is long and the issue is longer: ')
    line = lead + edit_call
    out = flow.fit_line(line, 260)
    assert len(out) <= 260 and out.endswith(': ' + edit_call) and out.startswith('NEXT: 12 reads since locate: edit now')
    assert '(an imperfect' not in out
    longer = lead * 2 + edit_call + '  then check_patch.py and submit_patch'
    out = flow.fit_line(longer, 260)
    assert len(out) <= 260 and edit_call in out and calls(out)[0] == ('edit.py', {
        'target': 'pay/text.py', 'start': '6', 'end': '11', 'new': '<new text for lines 6-11>'})
    assert flow.fit_line('NEXT: short', 260) == 'NEXT: short'
    huge = 'NEXT: x: show.py  target=' + 'a/' * 200 + 'b.py'
    assert len(flow.fit_line(huge, 260)) <= 260
    assert flow.fit_line('NEXT: ' + 'word ' * 80, 100).endswith('…') and len(flow.fit_line('NEXT: ' + 'w ' * 80, 100)) \
        <= 100


# ---------------------------------------------------------------- L5: the snippet gate persists until an edit

def test_l5_gate_persists_until_an_edit_every_second_snippet_runs(ws):
    locate()
    assert [step('run.py', *snippet(i, str(i)))[0] for i in range(5)] == ['ran'] * 5
    kinds = []
    for i in range(5, 15):
        kind, text = step('run.py', *snippet(i, str(i)))
        kinds.append((kind, text))
    assert [k for k, _ in kinds] == ['refused', 'ran'] * 5
    n = 5
    for kind, text in kinds:
        if kind == 'refused':
            assert text.startswith(f'NOT RUN: {n} snippets and no edit yet; snippets are refused until you edit. '
                                   'The code most likely to change: pay/text.py Text.append lines 6-11.\n'), text
            assert nexts(text)[0].startswith(f'NEXT: {n} snippets and no edit yet: edit the most suspicious line now: ')
        else:
            n += 1
            first = nexts(text)[0]
            assert first.startswith(f'NEXT: {n} snippets and no edit yet: the next snippet is not run until you edit; '
                                    'edit the most suspicious line now: '), text
        for line in nexts(text):
            assert_notation(line)
        assert ('edit.py', {'target': 'pay/text.py', 'start': '6', 'end': '11',
                            'new': '<new text for lines 6-11>'}) in calls(nexts(text)[0])
    edit()
    assert [step('run.py', *snippet(i, str(i)))[0] for i in range(20, 23)] == ['ran'] * 3   # rule 5's cap after it


def test_l5_gate_fires_from_the_cap_footer_on(ws):
    locate()
    footers = [step('run.py', *snippet(i, f'Traceback (most recent call last):\nValueError: {i}', 1))[1]
               for i in range(5)]
    assert 'snippets and no edit yet' not in footers[3]
    assert nexts(footers[4])[0].startswith('NEXT: 5 snippets and no edit yet: the next snippet is not run until you '
                                           'edit'), footers[4]


def test_l5_tests_reads_and_failed_edits(ws):
    locate()
    for i in range(5):
        step('run.py', *snippet(i, str(i)))
    assert step('run.py', obj(test='tests/test_text.py'), '3 passed\nexit code: 0\n')[0] == 'ran'      # not a snippet
    read = obj(snippet='print(open("pay/text.py").read())')
    assert step('run.py', read, TEXT + 'exit code: 0\n')[0] == 'ran'                                 # a read
    assert step('run.py', *snippet(9))[0] == 'refused'
    step('edit.py', obj(target='pay/text.py', old='nope', new='x'), 'NOT APPLIED: OLD not found\n', status=1)
    assert step('run.py', *snippet(10))[0] == 'refused'            # a failed edit is not an edit


# ---------------------------------------------------------------- L6: existing tests after an edit

def test_l6_names_and_tests_of_the_edited_function(ws):
    locate()
    edit()
    st = flow.State(flow.connect())
    assert flow.edited_names(st) == ['append', 'Text']
    assert flow.related_tests(['append']) == ('tests/test_text.py', 'test_append or test_append_spans', 'append')
    assert flow.related_tests(['copy']) == ('tests/test_text.py', 'test_copy', 'copy')
    assert flow.related_tests(['nowhere_used']) is None


def test_l6_k_is_the_name_when_many_tests_use_it(ws):
    many = ''.join(f'\n\ndef test_append_{i}():\n    Text("a").append(Text("b"))\n' for i in range(6))
    (ws / 'tests/test_text.py').write_text(TEST_TEXT + many)
    assert flow.related_tests(['append']) == ('tests/test_text.py', 'append', 'append')


def test_l6_scan_is_bounded(ws, monkeypatch):
    for i in range(30):
        (ws / f'tests/test_gen{i}.py').write_text('def test_x():\n    append()\n' * 3)
    monkeypatch.setattr(flow, 'TEST_SCAN_FILES', 3)
    t0 = time.time()
    found = flow.related_tests(['append'])
    assert time.time() - t0 < 2 * flow.TEST_SCAN_S + 1
    assert found is not None and found[0] != 'node_modules/pkg/test_x.py'


def test_l6_edit_footer_is_the_tests_then_check_patch(ws):
    locate()
    _, f = edit()
    lines = nexts(f)
    assert lines[0] == f'NEXT: verify once: {TESTS_LINE}  (the existing tests that use append)', f
    assert lines[1] == 'NEXT: then check_patch.py and submit_patch'
    for line in lines:
        assert_notation(line)
    _, f = step('run.py', obj(test='tests/test_text.py', k='test_append or test_append_spans'),
                '2 passed\nexit code: 0\n')
    assert nexts(f) == ['NEXT: the check passes: check_patch.py, then submit_patch'], f


def test_l6_tests_first_then_repro_rerun(ws):
    """v21 V1: after an edit the footer points to the existing tests first, then the repro of before the edit."""
    locate()
    step('run.py', *snippet(1, 'Traceback (most recent call last):\nRuntimeError: list changed size', 1))  # call 2
    step('run.py', *snippet(2, 'fine'))                                                                   # call 3
    _, f = edit()
    lines = nexts(f)
    assert lines[0] == f'NEXT: verify once: {TESTS_LINE}  (the existing tests that use append)', f
    assert lines[1] == ('NEXT: then re-run your repro of call 2, unchanged: run.py  snippet=<the same code as call 2>  '
                        'then check_patch.py and submit_patch'), f
    for line in lines:
        assert_notation(line)
    _, f = step('run.py', *snippet(1, 'ok'))                     # the repro passes now
    assert nexts(f) == [f'NEXT: the check passes; now the existing tests that use append: {TESTS_LINE}',
                        'NEXT: then check_patch.py and submit_patch'], f
    _, f = step('run.py', obj(test='tests/test_text.py', k='test_append'), '1 passed\nexit code: 0\n')
    assert nexts(f) == ['NEXT: the check passes: check_patch.py, then submit_patch'], f   # not repeated


def test_l6_a_failing_test_is_not_cut_short_by_the_verify_cap(ws):
    locate()
    edit()
    step('run.py', *snippet(1, 'ok'))
    _, f = step('run.py', obj(test='tests/test_text.py', k='test_append'),
                'FAILED tests/test_text.py::test_append_spans\nexit code: 1\n')
    first = nexts(f)[0]
    assert first.startswith('NEXT: if the failure comes from your change, fix it with edit.py'), f
    assert 'stop verifying' not in f


def test_l6_not_with_under_45_seconds_left(ws):
    locate()
    age(flow.TIME_BUDGET_S - flow.POST_EDIT_TEST_LEFT_S + 5)
    _, f = edit()
    assert 'existing tests' not in f
    assert nexts(f) == ['NEXT: time is nearly up: check_patch.py, then submit_patch'], f


def test_l6_finish_phase_by_calls_keeps_the_tests_while_time_is_left(ws):
    """v21 V2: the finish phase from the call count starts only at CALLS_FINISH_AT (near the hard cap)."""
    locate()
    for i in range(17):
        step('status.py', [], '')
    _, f = edit()
    assert [l for l in f.splitlines() if not l.startswith('MEMO: ')][0].startswith('BUDGET: call 19 of 40 | ')
    assert 'nearly up' not in f and 'existing tests that use append' in nexts(f)[0], f
    for i in range(flow.CALLS_FINISH_AT - 20):
        step('status.py', [], '')
    _, f = edit()  # the same edit again (applied again): call CALLS_FINISH_AT
    assert nexts(f) == [f'NEXT: {flow.TASK_MAX_CALLS - flow.CALLS_FINISH_AT} calls left: run the existing tests that '
                        f'use append once: {TESTS_LINE}', 'NEXT: then check_patch.py and submit_patch'], f


def test_l6_replay_real_scripts(ws):
    """locate, a repro, the edit, the repro again, the suggested tests, check_patch: real scripts under the hook."""
    code, out = run('locate.py', ws, query='Text.append pay/text.py spans')
    code, out = run('run.py', ws, snippet='from pay.text import Text\nt = Text("a")\nt.append(t)\nprint(t.plain)')
    assert 'exit code: 0' in out
    code, out = run('edit.py', ws, **EDIT_ARGS)
    assert code == 0, out
    lines = nexts(out)
    assert lines[0] == f'NEXT: verify once: {TESTS_LINE}  (the existing tests that use append)', out   # v21 V1
    assert lines[1].startswith('NEXT: then re-run your repro of call 2, unchanged: run.py  snippet=<the same code'), out
    code, out = run('run.py', ws, snippet='from pay.text import Text\nt = Text("a")\nt.append(t)\nprint(t.plain)')
    # v21 V3: the repro prints what it printed before the edit
    assert 'SAME OUTPUT as before your edit at call 3 (call 2): the change did not affect this' in out, out
    assert nexts(out)[1] == f'NEXT: if this output is the correct behaviour, run the existing tests that use append: ' \
        f'{TESTS_LINE}', out
    code, out = run('run.py', ws, test='tests/test_text.py', k='test_append or test_append_spans')
    assert '2 passed' in out and 'NEXT: the check passes: check_patch.py, then submit_patch' in out, out


# ---------------------------------------------------------------- L2 guard: an edit that deletes a used definition

def test_definition_guard_footer_keeps_the_definition(ws):
    locate()
    _, f = step('edit.py', obj(target='pay/text.py', start='6', end='14', new='    def append(self, text):\n        pass'),
                'NOT APPLIED: the edit (lines 6-14) deletes `def copy` (lines 13-14), and NEW does not define it, but '
                'the code still uses it; nothing was changed.\nCLOSEST: pay/text.py 6 11\n', status=1,
                ok=False, path='pay/text.py', guard='definition', closest=('pay/text.py', 6, 11))
    assert nexts(f) == ['NEXT: redo the edit and keep the definition it deleted (change only the lines around it, or '
                        'copy it into NEW unchanged): edit.py  target=pay/text.py  start=6  end=11  '
                        'new=<new text for lines 6-11>'], f
    assert 'still imports' not in f
    assert_notation(f)


def test_definition_guard_real_edit(ws):
    (ws / 'pay/util.py').write_text('def helper(x):\n    return x + 1\n\n\ndef main(x):\n    return helper(x)\n')
    code, out = run('edit.py', ws, target='pay/util.py', start='1', end='2', new='OFFSET = 2')  # no new def: not a rename
    assert code == 1 and out.startswith('NOT APPLIED'), out
    assert 'NEXT: redo the edit and keep the definition it deleted' in out and \
        'edit.py  target=pay/util.py  start=1  end=2' in out, out
