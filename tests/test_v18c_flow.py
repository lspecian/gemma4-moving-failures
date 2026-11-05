"""agent-v18c flow.py: agent-v17b's footers (BUDGET line, NEXT lines, pointer line, old-text alternative, region
choice, edit windows, rule numbering: tests/test_flow.py, test_v16.py, test_v17_flow.py) plus four ports from v18b:

1. strip_separator(): ADK puts a bare '--' before positional_args; flow.main drops it.
2. The sawtooth context model (rule 10/13): CTX_BASE_TOKENS 5240 + 0.2953 * chars + 336 * calls, a presumed compaction
   when the prompt that produced a call was >= COMPACT_AT (REPO_TOOLS_COMPACT_AT), then CTX_POST_BASE_TOKENS + the last
   CTX_KEEP_CALLS calls; results before compacted_at - CTX_VISIBLE_CALLS + 1 are gone (State.visible);
   context_tokens() and rule 10 read it.
3. After a presumed compaction: an identical call whose result is gone runs (no ALREADY DONE replay) and does not count
   toward the exploration budget; check_patch.py prints the full diff (<= 30 lines).
4. MEMO (<= 250 chars, current line numbers) on its own line above the v17b footer, when the estimate first reaches
   MEMO_AT_TOKENS (once per cycle) and on the result right before a presumed compaction; status.py prints it.
Also P5 (a closest region below CLOSEST_MIN_SIMILARITY is not suggested; a failed edit without one clears it) and
the v18 run.py "same output as call N" counting as a snippet that did not reproduce.
Run with scripts/test_skills.sh (sandbox image, py3.13).
"""

import os
import pathlib
import re
import subprocess
import sys
import textwrap

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
LONG = ('import functools\n\n\n'                                     # 1-3
        '@functools.lru_cache\n'                                     # 4
        'def render(value):\n'                                       # 5
        + ''.join(f'    x{i} = value + {i}\n' for i in range(68))     # 6-73
        + '    return x0\n'                                          # 74
        '\n\n'
        'def helper(v):\n'                                           # 77
        '    return v\n')                                            # 78
LOCATE_OUT = textwrap.dedent('''\
    identifiers: Text, append
    1.50 pay/text.py  [defines append]
          Text.append  L6-11
    1.40 pay/render.py
          render  L5-74
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
READY_OUT = 'changed files:\n  pay/text.py | 2 +-\nREADY: call submit_patch now.\n'


def git(ws, *args):
    return subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def ws(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    files = {'pay/__init__.py': '', 'pay/text.py': TEXT, 'pay/render.py': LONG,
             'pay/other.py': 'def other():\n    return 1\n', 'tests/__init__.py': '',
             'tests/test_text.py': 'from pay.text import Text\n\n\ndef test_append():\n    t = Text("a")\n'
                                   '    t.append(Text("b"))\n    assert t.plain == "ab"\n'}
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, 'init', '-q')
    git(root, 'add', '-A')
    git(root, 'commit', '-qm', 'baseline')
    monkeypatch.setenv('WORKSPACE', str(root))
    monkeypatch.setenv('REPO_TOOLS_DIR', str(tmp_path / 'state'))
    monkeypatch.setattr(flow, 'COMPACT_AT', 14336)
    return root


def obj(**kw):
    return [x for k, v in kw.items() for x in ('--' + k, str(v))]


def step(script, args, out='', status=0, **facts):
    """One call through flow.before/after, as flow.main does it: ('refused', text) or ('ran', footer)."""
    msg = flow.before(script, args)
    if msg:
        return 'refused', msg
    return 'ran', flow.after(script, args, status, out, **facts)


def run(script, ws, *argv, env=None, **args):
    """A real script under the flow hook (argv as given, then the object args)."""
    argv = list(argv) + [x for k, v in args.items() for x in ('--' + k, str(v))]
    e = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'), PYTHONDONTWRITEBYTECODE='1')
    e.pop('REPO_TOOLS_FLOW', None)
    e.update(env or {})
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *argv], capture_output=True, text=True, env=e,
                       timeout=120)
    assert 'Traceback' not in r.stderr, r.stderr
    return r.returncode, r.stdout


def nexts(out):
    return [l for l in out.splitlines() if l.startswith('NEXT:')]


def memo_line(out):
    return next((l for l in out.splitlines() if l.startswith('MEMO: ')), None)


def state():
    return flow.State(flow.connect())


def locate():
    return step('locate.py', obj(query='Text append loops'), LOCATE_OUT)


def show(start, end, path='pay/text.py', label='Text.append', size=0):
    return step('show.py', obj(target=path, start=str(start), end=str(end)), f'{path} {label} L{start}-{end} of 80\n'
                + 'x' * size)


def filler(i, size=2500):
    """A look-around call that adds `size` characters; never a repeat."""
    return step('search.py', obj(text=f'nothing{i}', within='pay/other.py'), 'y' * size)


def compact_now():
    """Look-around calls until a compaction is presumed; returns the number of the call before it."""
    for i in range(30):
        filler(i)
        st = state()
        if st.compacted_at:
            return st.compacted_at
    raise AssertionError('no compaction')


def edit():
    (pathlib.Path(os.environ['WORKSPACE']) / 'pay/text.py').write_text(
        TEXT.replace('in text._spans:', 'in list(text._spans):'))
    return step('edit.py', obj(**EDIT_ARGS), EDIT_OUT)


# ---------------------------------------------------------------- 1. the ADK separator

def test_strip_separator_unit():
    """ADK puts '--' before positional_args (E044 fastapi_14356: create.py made a file named '--')."""
    assert flow.strip_separator(['--', 'repro.py']) == ['repro.py']
    assert flow.strip_separator(['--target', 'a.py', '--', 'x']) == ['--target', 'a.py', 'x']
    assert flow.strip_separator(['--new', '--', '--old', 'y']) == ['--new', '--', '--old', 'y']   # a value '--' stays
    assert flow.strip_separator(['--query', 'q']) == ['--query', 'q']
    assert flow.strip_separator([]) == []


def test_flow_main_drops_the_separator(ws):
    """positional_args ["pay/text.py", "Text.append"] arrive as ['--', 'pay/text.py', 'Text.append']: the read runs
    and is recorded as the same call as the object form (no file named '--', no usage error)."""
    code, out = run('show.py', ws, '--', 'pay/text.py', 'Text.append')
    assert code == 0 and out.startswith('pay/text.py Text.append L6-11'), out
    st = state()
    assert st.actions[-1][2] == flow.action_key('show.py', obj(target='pay/text.py', symbol='Text.append'))
    assert st.viewed[0][:3] == ('pay/text.py', 6, 11)
    code, out = run('create.py', ws, '--', 'pay/new_mod.py', 'X = 1\n')
    assert code == 0 and (ws / 'pay/new_mod.py').exists() and not (ws / '--').exists(), out


# ---------------------------------------------------------------- 2. the sawtooth estimate

def test_constants_are_the_v18_refit():
    assert (flow.CTX_BASE_TOKENS, flow.CTX_TOKENS_PER_CHAR, flow.CTX_TOKENS_PER_CALL) == (5240, 0.2953, 336)
    assert (flow.CTX_POST_BASE_TOKENS, flow.CTX_KEEP_CALLS, flow.CTX_VISIBLE_CALLS) == (3420, 3, 2)
    assert flow.COMPACT_MARGIN_TOKENS == 0 and flow.CTX_MARGIN_TOKENS == 2000 and flow.MEMO_AT_TOKENS == 12500
    assert flow.MEMO_CHARS == 250 and flow.COMPACT_FROM_TOKENS == 14000
    assert flow.ctx_estimate(0, 0) == 5240 + 2000
    assert abs(flow.ctx_central(1000, 2) - (5240 + 295.3 + 672)) < 1e-6
    assert flow.ctx_estimate(1000, 3, flow.CTX_POST_BASE_TOKENS) == int(3420 + 295.3 + 1008 + 2000)
    assert flow.MEMO_AT_TOKENS < flow.COMPACT_AT - flow.COMPACT_MARGIN_TOKENS < flow.CTX_WARN_TOKENS


def test_compact_at_env_override():
    code = ('import sys; sys.dont_write_bytecode = True; sys.path.insert(0, %r)\nimport flow; print(flow.COMPACT_AT)'
            % str(SCRIPTS))
    for val, want in (('', '14336'), ('0', '0'), ('20000', '20000')):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        env.pop('REPO_TOOLS_COMPACT_AT', None)
        if val:
            env['REPO_TOOLS_COMPACT_AT'] = val
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env).stdout
        assert out.strip() == want, (val, out)


def test_sawtooth_resets_at_the_threshold_and_sets_visibility(ws):
    locate()
    seen = []
    for i in range(12):
        filler(i)
        st = state()
        seen.append((st.n, st.ctx_central, st.compacted_at))
        if st.compacted_at:
            break
    n, central, at = seen[-1]
    assert at == n, seen
    before = seen[-2][1]   # the estimate of the prompt that produced call n
    assert before + flow.COMPACT_MARGIN_TOKENS >= flow.COMPACT_AT > seen[-3][1] + flow.COMPACT_MARGIN_TOKENS
    # after: the post-compaction base + the last 3 calls (each ~2.5k chars and its footer)
    assert flow.CTX_POST_BASE_TOKENS + 3 * 336 + 0.2953 * 7500 <= central < \
        flow.CTX_POST_BASE_TOKENS + 3 * 336 + 0.2953 * 9500, seen
    assert flow.context_tokens() == st.ctx_tokens == int(central + flow.CTX_MARGIN_TOKENS)
    assert st.visible_from == n - 1 and st.visible(n) and st.visible(n - 1) and not st.visible(n - 2)
    # rule 8b keeps v17b's own cumulative estimate: the footers stay compact as v17b's did (no sawtooth there)
    assert st.compact and st.ctx_v17b >= flow.FOOTER_COMPACT_FROM_TOKENS > st.ctx_tokens
    for i in range(12, 40):   # it grows again from there, and compacts again later
        filler(i)
        if state().compactions == 2:
            break
    assert state().compactions == 2 and state().compacted_at > n


def test_no_compaction_everything_visible(ws):
    locate()
    show(6, 11)
    st = state()
    assert st.compacted_at == 0 and st.visible_from == 0 and st.visible(1) and st.visible(2) and not st.visible(0)


def test_compaction_model_off(ws, monkeypatch):
    monkeypatch.setattr(flow, 'COMPACT_AT', 0)
    locate()
    vals = []
    for i in range(10):
        filler(i)
        vals.append(state().ctx_central)
    assert vals == sorted(vals) and vals[-1] > 15000 and state().compacted_at == 0


def test_context_tokens_inside_a_script_after_a_compaction(ws):
    locate()
    compact_now()
    code = ('import sys; sys.dont_write_bytecode = True; sys.path.insert(0, %r)\n'
            'import flow; print(flow.context_tokens())' % str(SCRIPTS))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1')).stdout
    assert int(out) == state().ctx_tokens < flow.COMPACT_FROM_TOKENS


def test_rule10_reads_the_sawtooth(ws, monkeypatch):
    """Without the compaction model the same calls reach rule 10's warning; with it the estimate drops first."""
    big = 'x' * 14500
    for compact_at in (0, 14336):
        monkeypatch.setattr(flow, 'COMPACT_AT', compact_at)
        conn = flow.connect()
        for t in flow.TABLES:
            conn.execute(f'DELETE FROM {t}')
        conn.close()
        locate()
        for i in range(4):
            _, f = show(1 + i, 20, size=len(big))
        warned = 'context nearly full' in f
        assert warned == (compact_at == 0), (compact_at, f)


def test_compact_footers_keep_v17bs_trigger(ws):
    """Rule 8b (compact footers) reads v17b's cumulative estimate and threshold, unchanged; the sawtooth does not
    move it (show.py/search.py read context_tokens() against COMPACT_FROM_TOKENS)."""
    assert (flow.V17B_CTX_BASE_TOKENS, flow.V17B_CTX_TOKENS_PER_CHAR, flow.V17B_CTX_TOKENS_PER_CALL,
            flow.V17B_CTX_MARGIN_TOKENS, flow.FOOTER_COMPACT_FROM_TOKENS) == (5094, 0.3147, 284, 2800, 15000)
    locate()
    rows = []
    for i in range(20):
        _, f = filler(i, size=1500)
        st = state()
        rows.append((st.ctx_v17b, st.compact, f.splitlines()))
    conn = flow.connect()
    total = int(flow.get_meta(conn, 'ctx_chars_all'))
    n = conn.execute('SELECT COUNT(*) FROM actions').fetchone()[0]
    conn.close()
    assert flow.ctx_estimate_v17b(total, n) == rows[-1][0]
    for est, compact, lines in rows:
        assert compact == (est >= 15000)
        budget = next((l for l in lines if l.startswith('BUDGET: ')), None)
        if budget is None:   # call 2: the BUDGET line starts at call 3
            continue
        assert budget.startswith('BUDGET: call ') and (('/40 |' in budget) == compact), lines
    assert any(c for _, c, _ in rows) and not all(c for _, c, _ in rows) and state().compactions >= 1


# ---------------------------------------------------------------- 3. after a compaction

def test_replay_while_visible_is_v17bs(ws):
    locate()
    show(6, 11, size=90)
    kind, text = show(6, 11, size=90)
    assert kind == 'refused' and text.startswith('ALREADY DONE at call 2: identical show.py call and nothing was '
                                                 'edited since, so it was not rerun. Its output was:\n'), text


def test_lost_result_runs_again_and_is_not_counted(ws):
    locate()
    show(6, 11)
    compact_now()
    st = state()
    assert not st.visible(2)
    before = st.look_since_edit
    kind, f = show(6, 11)                    # identical to call 2, whose result is gone
    assert kind == 'ran' and not f.startswith('ALREADY DONE')
    st = state()
    assert st.n in st.recovery and st.look_since_edit == before
    kind, text = show(6, 11)                 # now it is visible again: replayed
    assert kind == 'refused' and text.startswith('ALREADY DONE at call ')
    kind, f = show(1, 5, label='Text.__init__')   # a new read still counts
    assert state().look_since_edit == before + 1


def test_reread_of_a_range_shown_only_before_the_compaction_is_a_recovery(ws):
    locate()
    show(6, 11)
    compact_now()
    before = state().look_since_edit
    step('show.py', obj(target='pay/text.py', symbol='Text.append'), 'pay/text.py Text.append L6-11 of 80\n')
    st = state()
    assert st.n in st.recovery and st.look_since_edit == before


def test_check_patch_prints_the_diff_after_a_compaction(ws):
    run('locate.py', ws, query='Text.append pay/text.py')
    code, out = run('edit.py', ws, **EDIT_ARGS)
    assert code == 0, out
    run('run.py', ws, test='tests/test_text.py')   # v21 V1: verified by the related tests
    code, out = run('check_patch.py', ws)
    assert 'READY' in out and 'git diff' not in out, out
    compact_now()
    code, out = run('check_patch.py', ws)
    assert 'your edits (git diff' in out and '+        for start, end, style in list(text._spans):' in out, out
    assert 'READY' in out and out.rstrip().endswith('NEXT: submit_patch (the patch is ready)'), out


# ---------------------------------------------------------------- 4. MEMO

def test_memo_once_per_cycle_and_before_a_compaction(ws):
    locate()
    memos = []
    for i in range(12):
        _, f = filler(i)
        st = state()
        memos.append((st.n, memo_line(f), st.ctx_central, st.compacted_at))
        if st.compacted_at:
            break
    with_memo = [m for m in memos if m[1]]
    assert len(with_memo) == 2, memos
    first, second = with_memo
    assert first[2] >= flow.MEMO_AT_TOKENS and all(m[2] < flow.MEMO_AT_TOKENS for m in memos if m[0] < first[0])
    assert second[3] == second[0]                            # the call right before the compaction
    assert all(len(m[1]) <= flow.MEMO_CHARS for m in with_memo)
    _, f = filler(99)
    assert memo_line(f) is None                              # not again right after it
    for i in range(100, 130):                                # the next cycle: its own threshold MEMO
        _, f = filler(i)
        if memo_line(f):
            break
    st = state()
    assert memo_line(f) and st.compactions == 1 and st.ctx_central >= flow.MEMO_AT_TOKENS, f


def test_memo_is_its_own_line_above_the_v17b_footer(ws):
    """The MEMO line is added above the footer; the BUDGET/NEXT lines are those v17b prints for the same state."""
    locate()
    show(6, 11)
    for i in range(12):
        _, f = filler(i)
        if memo_line(f):
            break
    lines = f.splitlines()
    assert lines[0].startswith('MEMO: no edit yet. Code to change: pay/text.py Text.append L6-11'), f
    rest = lines[1:]
    assert 1 <= len(rest) <= flow.FOOTER_LINES and rest[0].startswith('BUDGET: call ') and \
        all(l.startswith('NEXT: ') for l in rest[1:]), f
    st = state()
    assert st.memo is None
    same = flow.format_footer(flow.suggest(st, 'search.py', {'hits': []}), st, 'search.py')
    clock = re.compile(r'\d+m\d\ds')
    assert clock.sub('T', same) == clock.sub('T', '\n'.join(rest)), (same, f)   # the footer without its MEMO
    for line in rest[1:]:
        assert_notation(line)


def test_refusal_at_a_compaction_carries_the_memo(ws):
    locate()
    show(6, 11)
    conn = flow.connect()                                    # the prompt that produced the next call is past it
    chars, n, base = flow._ctx_parts(conn)
    flow.set_meta(conn, 'ctx_chars', str(int((flow.COMPACT_AT - flow.ctx_central(0, n)) / flow.CTX_TOKENS_PER_CHAR)
                                         + 10))
    conn.close()
    kind, msg = show(6, 11)                                  # an identical repeat: replayed (v17b's form)
    assert kind == 'refused' and msg.startswith('ALREADY DONE at call 2: identical show.py call'), msg
    assert memo_line(msg) == 'MEMO: no edit yet. Code to change: pay/text.py Text.append L6-11 (read at call 2); ' \
        'other: pay/render.py render L4-74', msg
    tail = msg.splitlines()[msg.splitlines().index(memo_line(msg)) + 1:]
    assert tail and all(l.startswith(('BUDGET: ', 'NEXT: ')) for l in tail), msg
    assert state().compacted_at == 3


def test_memo_without_an_edit_names_the_v17b_region(ws):
    locate()
    show(6, 11)
    text = flow.memo(state())
    assert text == 'MEMO: no edit yet. Code to change: pay/text.py Text.append L6-11 (read at call 2); ' \
        'other: pay/render.py render L4-74', text
    st = state()
    assert flow.best_region(st)[:3] == ('pay/text.py', 6, 11)   # the same region the edit pushes name


def test_memo_before_locate_and_without_code(ws):
    assert flow.memo(state()) == 'MEMO: no edit yet; locate.py not run yet'
    step('locate.py', obj(query='x'), 'nothing found\n')
    assert flow.memo(state()) == 'MEMO: no edit yet; no code read yet'
    assert flow.memo(None) == ''


def test_memo_with_an_edit_verdict_and_ready(ws):
    locate()
    step('run.py', obj(snippet='print(1)'), 'Traceback (most recent call last):\nRuntimeError: x\nexit code: 1\n')
    edit()
    assert flow.memo(state()) == 'MEMO: edited pay/text.py L9 at call 3; not run since the edit; patch not checked'
    step('run.py', obj(snippet='print(1)'), 'ok\nexit code: 0\n')
    step('check_patch.py', [], READY_OUT)
    assert flow.memo(state()) == ('MEMO: edited pay/text.py L9 at call 3; repro (call 4) exit 0; '
                                  'patch READY (call 5): submit_patch')


def test_memo_uses_current_line_numbers(ws):
    locate()
    edit()
    text = (ws / 'pay/text.py').read_text()
    (ws / 'pay/text.py').write_text('# header 1\n# header 2\n' + text)   # a later edit above it: 2 lines
    step('edit.py', obj(target='pay/text.py', start='1', end='1', new='# header 1\n# header 2\nclass Text:'),
         'EDITED pay/text.py (lines)\n@@ -1,1 +1,3 @@\n+# header 1\n+# header 2\n class Text:\n')
    st = state()
    assert flow.edited_ranges(st) == [('pay/text.py', 1, 2), ('pay/text.py', 11, 11)]
    assert flow.memo(st).startswith('MEMO: edited pay/text.py L1-2, L11 at call 3')


def test_memo_is_capped(ws):
    long = 'pkg/' + 'very_long_directory_name/' * 8 + 'module.py'
    (ws / long).parent.mkdir(parents=True)
    (ws / long).write_text('def f():\n    return 1\n')
    step('locate.py', obj(query='f'), '', candidates=[(long, 'f', 1, 2, 1.0, '')])
    step('show.py', obj(target=long, start='1', end='2'), f'{long} f L1-2 of 2\n')
    text = flow.memo(state())
    assert text.startswith('MEMO: no edit yet. Code to change: pkg/') and len(text) <= flow.MEMO_CHARS


def test_status_prints_the_memo_first(ws):
    locate()
    edit()
    code, out = run('status.py', ws)
    lines = out.strip().splitlines()
    assert code == 0 and lines[0] == 'MEMO: edited pay/text.py L9 at call 2; not run since the edit; patch not checked', out
    assert lines[1].startswith('calls so far: 2 (0 refused)') and lines[-1].startswith('NEXT: '), out
    assert 'MEMO' not in '\n'.join(lines[1:])


# ---------------------------------------------------------------- P5 and v18 run.py output

def test_p5_low_similarity_closest_is_ignored(ws):
    locate()
    show(6, 11)
    bad = 'NOT APPLIED: OLD not found; closest region (similarity 0.55), lines 13-14\nCLOSEST: pay/text.py 13 14\n'
    _, f = step('edit.py', obj(target='pay/text.py', old='zzz', new='y'), bad, status=1)
    assert state().closest is None and 'start=13' not in f, f
    assert nexts(f) == ['NEXT: copy the old text exactly from the region printed above and retry edit.py'], f
    good = bad.replace('0.55', '0.97')
    _, f = step('edit.py', obj(target='pay/text.py', old='zz', new='y'), good, status=1)
    assert state().closest == ('pay/text.py', 13, 14)
    assert '2 edits failed' in nexts(f)[0] and 'start=13  end=14' in nexts(f)[0], f
    _, f = step('edit.py', obj(target='pay/text.py', old='z', new='y'), 'NOT APPLIED: x\n', status=1,
                closest=('pay/text.py', 2, 4))                   # similarity unknown: used
    assert 'start=2  end=4' in nexts(f)[0], f


def test_p5_failed_edit_without_closest_forgets_the_earlier_region(ws):
    locate()
    show(6, 11)
    step('edit.py', obj(target='pay/text.py', old='a', new='b'), 'NOT APPLIED: x\nCLOSEST: pay/text.py 13 14\n', status=1)
    assert state().closest == ('pay/text.py', 13, 14)
    _, f = step('edit.py', obj(target='pay/text.py', old='c', new='d'), 'NOT APPLIED: no similar lines\n', status=1,
                similarity=0.3)
    assert state().closest is None and 'start=13' not in f, f
    assert '2 edits failed; replace the lines by number: edit.py  target=pay/text.py  start=6  end=11' in nexts(f)[0], f


def test_same_output_snippet_counts_as_not_reproducing(ws):
    locate()
    step('run.py', obj(snippet='print(1)'), 'Traceback (most recent call last):\nValueError: x\nexit code: 1\n')
    step('run.py', obj(snippet='print(2)'), 'same output as call 2; its last line: ValueError: x\nexit code: 1\n')
    st = state()
    assert len(st.snippets) == 2 and st.norepro == 1


def test_parse_facts_reads_similarity_and_lowercase_low_confidence():
    out = 'NOT APPLIED: OLD not found; closest region (similarity 0.72), lines 3-5\n'
    assert flow.parse_facts('edit.py', obj(target='a.py', old='x', new='y'), out)['similarity'] == 0.72
    out = '1.50 pay/text.py  [defines append]\n      Text.append  L6-11\nlow confidence: the top files score close\n'
    facts = flow.parse_facts('locate.py', obj(query='x'), out)
    assert facts['low_confidence'] and facts['candidates'][0][:4] == ('pay/text.py', 'Text.append', 6, 11)


# ---------------------------------------------------------------- v17b footers are unchanged below the threshold

def test_footers_identical_with_and_without_the_compaction_model(ws, monkeypatch):
    """Below the compaction threshold the ports change nothing a footer says (same calls, both settings)."""
    def seq():
        conn = flow.connect()
        for t in flow.TABLES:
            conn.execute(f'DELETE FROM {t}')
        conn.close()
        out = [locate()[1], show(6, 11)[1], step('search.py', obj(text='spans'), 'pay/text.py:9: x\n')[1],
               show(13, 14, label='Text.copy')[1], show(1, 5, label='Text.__init__')[1],
               step('run.py', obj(snippet='print(1)'), 'ok\nexit code: 0\n')[1], show(6, 11)[1], edit()[1],
               step('run.py', obj(test='tests/test_text.py'), '1 passed\nexit code: 0\n')[1],
               step('check_patch.py', [], READY_OUT)[1]]
        return [re.sub(r'\d+m\d\ds of', 'XmYYs of', o) for o in out]
    monkeypatch.setattr(flow, 'COMPACT_AT', 0)
    off = seq()
    monkeypatch.setattr(flow, 'COMPACT_AT', 14336)
    on = seq()
    assert on == off
    assert 'MEMO' not in '\n'.join(on)
    assert nexts(on[4])[0].startswith('NEXT: 4 reads since locate: edit now, even if not fully sure'), on[4]
    assert nexts(on[4])[1].startswith('NEXT: the code most likely to change: pay/text.py Text.append lines 6-11 '
                                      '(shown at call 2)'), on[4]


# ---------------------------------------------------------------- v18b edit.py's stale line numbers with v18c's flow

def test_stale_line_numbers_are_shifted_with_v17b_viewed_rows(ws):
    """v18b edit.py (P4) shifts line numbers the model read before its own edit moved them; it reads flow's viewed
    rows (v18c keeps them as read, like v17b) and edit.py's own memory of the hunks."""
    run('locate.py', ws, query='Text.copy returns a Text pay/text.py')
    code, out = run('show.py', ws, target='pay/text.py', start='13', end='14')
    assert code == 0 and '13      def copy(self):' in out, out
    code, out = run('edit.py', ws, target='pay/text.py', start='1', end='1',
                    new='# a header line\n# another one\nclass Text:')
    assert code == 0 and 'EDITED' in out, out
    code, out = run('edit.py', ws, target='pay/text.py', start='13', end='14',
                    new='    def copy(self):\n        return Text(str(self.plain))')
    assert code == 0, out
    assert 'numbered as read at call 2' in out and 'used the current numbers 15-16' in out, out
    text = (ws / 'pay/text.py').read_text()
    assert 'return Text(str(self.plain))' in text and text.count('def copy') == 1
    assert '    def append(self, text):' in text and 'return self\n' in text
