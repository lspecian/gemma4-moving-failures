"""agent-v16: levers L1, L2, L4, L6-L10 of research/notes/11-failures-all-tasks.md (agent/skills/repo-tools/scripts:
flow.py, run.py, edit.py, guards.py, search.py, locate.py, check_patch.py; agent/prompts/system.md, SKILL.md).

- L1 probing: pre-edit snippets that did not reproduce anything (exit 0, no traceback/...Error, or the same output as
  an earlier one) -> from the 2nd the footer says "sweep the invariant over many inputs in ONE snippet, or edit the
  most suspicious line now" with the line-range edit; the 6th pre-edit snippet is refused once.
- L2 backslashes: a JSON-escaping note under show/run/search/locate output with a backslash (twice per task); edit.py
  warns when a changed line has twice the backslashes in a row of the line it replaced.
- L4 several sites: an ambiguous OLD lists every site with its function and a ready line-range edit; after an edit,
  "the same code also appears at" names other places that still have a changed line.
- L6 prompt rules; L7 search overflow per file, locate title words joined by _; L8 15 s snippets and a traceback of
  where a timed-out run was; L9 context guard; L10 edit.py start without end (tests/test_edit_create.py);
  check_patch warns about added `while True:` loops without a way out.
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
import guards  # noqa: E402

# rich_3480: the same loop in Text.append and Text.append_text (gold fixes both)
TEXT = textwrap.dedent('''\
    """Text with spans."""


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

        def append_text(self, text):
            offset = len(self.plain)
            self.plain += text.plain
            for start, end, style in text._spans:
                self._spans.append((start + offset, end + offset, style))
            return self
    ''')
# rich_3063 / rich_3278: a raw regex with backslashes
MARKUP = r'''import re

RE_ESCAPE = re.compile(r"(\\*)(\[[a-z#/@][^[]*?])")


def escape(markup):
    return RE_ESCAPE.sub(r"\1\1\\\2", markup)
'''
# rich_3935: "fix for padding width" means Table._get_padding_width, not the Padding renderable
TABLE = textwrap.dedent('''\
    class Table:
        def __init__(self, padding=1, pad_edge=True):
            self.padding = padding
            self.pad_edge = pad_edge

        def _get_padding_width(self, column_index):
            return self.padding * 2

        def render(self, rows):
            return [str(r) for r in rows]
    ''')
PADDING = textwrap.dedent('''\
    """Padding: add padding around a renderable; padding width and padding height."""


    class Padding:
        """Draw padding (width, height) around a renderable."""

        def __init__(self, renderable, pad=(0, 0, 0, 0), width=None):
            self.renderable = renderable
            self.pad = pad
            self.width = width

        def measure(self, max_width):
            """Measure the padding width: the width of the renderable plus left and right padding."""
            width = self.width or max_width
            return width + self.pad[1] + self.pad[3]

        def padding_size(self):
            return self.pad
    ''')
LOOP = 'def spin(n):\n    while n:\n        n += 1\n'


def git(ws, *args):
    return subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def ws(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    files = {'pay/__init__.py': '', 'pay/text.py': TEXT, 'pay/markup.py': MARKUP, 'pay/table.py': TABLE,
             'pay/padding.py': PADDING, 'pay/loop.py': LOOP,
             'tests/test_loop.py': 'from pay.loop import spin\n\n\ndef test_spin():\n    spin(1)\n'}
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, 'init', '-q')
    git(root, 'add', '-A')
    git(root, 'commit', '-qm', 'baseline')
    monkeypatch.setenv('WORKSPACE', str(root))
    monkeypatch.setenv('REPO_TOOLS_DIR', str(tmp_path / 'state'))
    return root


def run(script, ws, flow_on=False, **args):
    argv = [x for k, v in args.items() for x in ('--' + k, str(v))]
    env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'), PYTHONDONTWRITEBYTECODE='1')
    env.pop('REPO_TOOLS_FLOW', None)
    if not flow_on:
        env['REPO_TOOLS_FLOW'] = 'off'
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *argv], capture_output=True, text=True, env=env,
                       timeout=120)
    assert 'Traceback' not in r.stderr, r.stderr
    return r.returncode, r.stdout


def py(ws, code, *argv):
    """Run code with the scripts importable (to patch a module constant before calling its main)."""
    env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'), PYTHONDONTWRITEBYTECODE='1',
               REPO_TOOLS_FLOW='off')
    pre = f'import sys; sys.dont_write_bytecode = True; sys.path.insert(0, {str(SCRIPTS)!r})\n'
    r = subprocess.run([sys.executable, '-c', pre + textwrap.dedent(code), *argv], capture_output=True, text=True,
                       env=env, timeout=120)
    return r.stdout + r.stderr


def nexts(out):
    return [l for l in out.splitlines() if l.startswith('NEXT:')]


def obj(**kw):
    return [x for k, v in kw.items() for x in ('--' + k, str(v))]


def step(script, args, out='', status=0, **facts):
    """One call through flow.before/after, as flow.main does it: ('refused', text) or ('ran', footer)."""
    msg = flow.before(script, args)
    if msg:
        return 'refused', msg
    return 'ran', flow.after(script, args, status, out, **facts)


LOCATE_OUT = textwrap.dedent('''\
    identifiers: Text, append
    1.50 pay/text.py  [defines append]
          Text.append  L9-14
    0.80 pay/markup.py
          escape  L6-7
    --- best match: pay/text.py Text.append L9-14
        9      def append(self, text):
       14          return self
    ''')


def snippet(i, body='42', code=0):
    """A run.py snippet call (args, out) whose output is BODY."""
    return obj(snippet=f'print({i})'), f'{body}\nexit code: {code}\n'


# ---------------------------------------------------------------- L1: probing before the first edit

def test_l1_state_counts_snippets_that_did_not_reproduce(ws):
    step('locate.py', obj(query='Text append loops'), LOCATE_OUT)
    for i, (body, code) in enumerate([('42', 0),                                                # no error: probe
                                      ('Traceback (most recent call last):\nValueError: x', 1),  # reproduced
                                      ('TIMEOUT', 'TIMEOUT'),                                    # a hang reproduces
                                      ("KeyError: 'rate' printed by the snippet", 0),             # an ...Error word
                                      ('42', 0)]):                                               # same output again
        step('run.py', *snippet(i, body, code))
    step('run.py', obj(test='tests/test_loop.py'), '1 passed\nexit code: 0\n')                   # tests: not snippets
    st = flow.State(flow.connect())
    assert len(st.snippets) == 5 and st.norepro == 2


def test_l1_footer_after_two_snippets_that_did_not_reproduce(ws):
    step('locate.py', obj(query='Text append loops'), LOCATE_OUT)
    _, f1 = step('run.py', *snippet(1, 'hellohello'))
    assert 'did not reproduce' not in f1
    _, f2 = step('run.py', *snippet(2, "['', 'b', 'a']"))
    first = nexts(f2)[0]
    assert first.startswith('NEXT: 2 snippets did not reproduce it: sweep the invariant over many inputs in ONE '
                            'snippet, or edit the most suspicious line now: '), f2
    assert ('edit.py', {'target': 'pay/text.py', 'start': '9', 'end': '14', 'new': '<new text for lines 9-14>'}) \
        in calls(first)
    for line in nexts(f2):
        assert_notation(line)


def test_l1_reproducing_snippets_keep_the_run_cap_footer(ws):
    step('locate.py', obj(query='Text append loops'), LOCATE_OUT)
    for i in range(3):
        _, f = step('run.py', *snippet(i, f'Traceback (most recent call last):\nValueError: {i}', 1))
    assert nexts(f)[0].startswith('NEXT: 3 runs and no edit yet: edit now'), f


def test_l1_sixth_snippet_refused_once_and_never_twice_in_a_row(ws):
    step('locate.py', obj(query='Text append loops'), LOCATE_OUT)
    kinds = [step('run.py', *snippet(i, str(i)))[0] for i in range(5)]
    assert kinds == ['ran'] * 5
    kind, msg = step('run.py', *snippet(5, '5'))
    assert kind == 'refused' and msg.startswith('NOT RUN: 5 snippets and no edit yet;'), msg
    assert 'NEXT: 5 snippets and no edit yet: edit the most suspicious line now: edit.py  target=pay/text.py' in msg
    for line in nexts(msg):
        assert_notation(line)
    # v17 (L5): no longer once per task; persistent until an edit, every second one runs (rule 3b)
    assert [step('run.py', *snippet(i, str(i)))[0] for i in range(6, 12)] == ['ran', 'refused'] * 3


def test_l1_refusal_skipped_right_after_another_refusal(ws):
    step('locate.py', obj(query='Text append loops'), LOCATE_OUT)
    for i in range(5):
        step('run.py', *snippet(i, str(i)))
    assert step('run.py', *snippet(4, '4'))[0] == 'refused'          # an identical repeat (rule 2)
    assert step('run.py', *snippet(9, '9'))[0] == 'ran'              # rule 3b: not refused twice in a row
    assert step('run.py', *snippet(10, '10'))[0] == 'refused'        # the snippet cap


def test_l1_not_after_an_edit(ws):
    step('locate.py', obj(query='Text append loops'), LOCATE_OUT)
    for i in range(3):
        step('run.py', *snippet(i, str(i)))
    step('edit.py', obj(target='pay/text.py', old='a', new='b'), 'EDITED pay/text.py (exact)\n')
    st = flow.State(flow.connect())
    assert st.snippets == [] and st.norepro == 0
    kinds = [step('run.py', *snippet(i, str(i)))[0] for i in range(10, 13)]
    assert kinds == ['ran'] * 3  # post-edit checks: rule 5's cap, not the snippet cap


def test_l1_replay_real_scripts_rich_3506_shape(ws):
    """rich_3506: hand-picked inputs that never reproduce. Real locate/run.py under flow: the 2nd snippet's footer
    asks for one sweeping snippet or the edit; the 6th is not run."""
    code, out = run('locate.py', ws, flow_on=True, query='Text.append infinite loop pay/text.py')
    assert code == 0, out
    outs = [run('run.py', ws, flow_on=True, snippet=f'from pay.text import Text\nprint(len(Text("{"a" * i}").plain))')
            for i in range(6)]
    assert 'did not reproduce' not in outs[0][1]
    assert nexts(outs[1][1])[0].startswith('NEXT: 2 snippets did not reproduce it: sweep'), outs[1][1]
    assert outs[5][1].startswith('NOT RUN: 5 snippets and no edit yet'), outs[5][1]


# ---------------------------------------------------------------- L2: backslashes in JSON-encoded results

def test_l2_note_under_code_with_backslashes_twice_per_task(ws):
    outs = [run('show.py', ws, flow_on=True, target='pay/markup.py', start=str(1 + i), end='7')[1] for i in range(3)]
    note = [l for l in outs[0].splitlines() if l.startswith('note:')]
    assert note == [flow.BACKSLASH_NOTE], outs[0]
    assert '\\' not in flow.BACKSLASH_NOTE  # the note itself would be doubled too
    assert flow.BACKSLASH_NOTE in outs[1] and flow.BACKSLASH_NOTE not in outs[2]
    assert outs[0].index('note:') < outs[0].index('NEXT:')


def test_l2_no_note_without_backslashes_or_for_edits(ws):
    _, out = run('show.py', ws, flow_on=True, target='pay/text.py', symbol='Text.append')
    assert 'note:' not in out
    _, out = run('search.py', ws, flow_on=True, text='RE_ESCAPE')
    assert flow.BACKSLASH_NOTE in out
    _, out = run('run.py', ws, flow_on=True, snippet='print("a\\\\b")')  # prints a, a backslash, b
    assert flow.BACKSLASH_NOTE in out
    _, out = run('show.py', ws, flow_on=True, target='pay/markup.py', symbol='escape')
    assert flow.BACKSLASH_NOTE not in out  # twice per task
    st = flow.State(flow.connect())
    assert flow.get_meta(st.conn, 'backslash_notes') == '2'


def test_l2_code_backslash_ignores_the_scripts_own_lines():
    assert not flow._code_backslash('NOTE: read a\\b as x\nexit code: 0\n')
    assert flow._code_backslash('    3  RE = re.compile(r"\\d+")\n')


def test_l2_doubled_backslashes_warning(ws):
    code, out = run('edit.py', ws, target='pay/markup.py', start='3', end='3',
                    new=r'RE_ESCAPE = re.compile(r"(\\\\*)(\[[a-z#/@][^[]*?])")')
    assert code == 0
    warn = [l for l in out.splitlines() if l.startswith('WARNING: line 3 has 4 backslashes in a row where the old line '
                                                       'had 2.')]
    assert warn and 'halve them' in warn[0], out


def test_l2_doubled_backslashes_unit():
    before = 'X = r"(\\\\*)"\nY = 1\n'
    assert guards.doubled_backslashes(before, 'X = r"(\\\\\\\\*)"\nY = 1\n')
    assert guards.doubled_backslashes(before, 'X = r"(\\\\*)$"\nY = 1\n') == []        # same run length
    assert guards.doubled_backslashes('Y = 1\n', 'Y = 1\nZ = "\\\\\\\\"\n') == []        # an added line
    assert guards.doubled_backslashes(before, 'X = 2\nY = 1\n') == []


# ---------------------------------------------------------------- L4: several sites

def test_l4_ambiguous_old_lists_every_site_with_a_ready_edit(ws):
    # v18 edit.py: each site once, one line (the footer carries the ready call of the first, see the next test)
    old = '        for start, end, style in text._spans:'
    code, out = run('edit.py', ws, target='pay/text.py', old=old, new='        for start, end, style in list(text._spans):')
    assert code == 1 and out == ('NOT APPLIED: OLD occurs 2 times in pay/text.py; edit each one that has the bug by its '
                                 'lines (they move by the lines each edit adds):\n'
                                 '  12-12 Text.append\n  19-19 Text.append_text\n'
                                 # v20: one-line OLD and NEW at <= 4 sites: a ready multi-edit call for all of them
                                 'to change all of them in one call: edit.py  target=pay/text.py  start=12  end=12  '
                                 'new=for start, end, style in list(text._spans):  start2=19  end2=19  '
                                 'new2=for start, end, style in list(text._spans):\n'), out
    assert (ws / 'pay/text.py').read_text() == TEXT
    assert_notation(out)


def test_l4_ambiguous_footer_names_the_sites(ws):
    step('locate.py', obj(query='Text append loops'), LOCATE_OUT)
    code, out = run('edit.py', ws, flow_on=True, target='pay/text.py', old='self.plain += text.plain',
                    new='self.plain = self.plain + text.plain')
    first = nexts(out)[0]
    assert first.startswith('NEXT: OLD is in 2 places (pay/text.py:11 Text.append, pay/text.py:18 Text.append_text): '
                            'fix each one that has the bug by line numbers, starting with: '), out
    assert ('edit.py', {'target': 'pay/text.py', 'start': '11', 'end': '11', 'new': '<new text for lines 11-11>'}) \
        in calls(first)


def test_l4_after_an_edit_the_same_code_elsewhere_is_named(ws):
    step('locate.py', obj(query='Text append loops'), LOCATE_OUT)
    code, out = run('edit.py', ws, flow_on=True, target='pay/text.py', start='12', end='12',
                    new='        for start, end, style in list(text._spans):')
    assert code == 0, out
    assert 'the same code also appears at: pay/text.py:19 Text.append_text; if it has the same bug, fix it too' in out
    first = nexts(out)[0]
    assert first == ('NEXT: if pay/text.py:19 Text.append_text has the same bug, fix it too: '
                     'edit.py  target=pay/text.py  start=19  end=19  new=<new text for lines 19-19>'), out
    assert 'verify once' in nexts(out)[1]
    for line in nexts(out):
        assert_notation(line)


def test_l4_no_also_for_trivial_lines_or_rewrites(ws):
    before = TEXT
    after = TEXT.replace('            return self\n', '            return None\n', 1)
    import edit
    assert edit.also_at('pay/text.py', before, after) == []                  # `return self` is not distinctive
    body = '\n'.join(f'        x{i} = self.compute_value({i})' for i in range(10))
    big = before.replace('    def append(self, text):\n', '    def append(self, text):\n' + body + '\n')
    rewrite = big.replace('self.compute_value', 'self.other_value')
    copy = rewrite + '\n' + body + '\n'
    assert edit.also_at('pay/text.py', big + '\n' + body + '\n', copy) == []  # a 10-line rewrite: no siblings


# ---------------------------------------------------------------- L6: prompt and skill text

def test_l6_prompt_rules_and_names():
    prompt = (ROOT / 'agent/prompts/system.md').read_text()
    for rule in ('through the public entry point it uses, with native types',
                 'try other input forms',
                 'After the edit, re-run the same repro; do not submit while it still shows the bug',
                 'Results are JSON-encoded: every backslash of a file is shown doubled',
                 '(start alone: that one line'):
        assert rule in prompt, rule
    assert '\\' not in prompt
    assert 'name: swe_agent_v' in (ROOT / 'agent/agent.yaml').read_text()
    skill = (ROOT / 'agent/skills/repo-tools/SKILL.md').read_text()
    assert 'snippets\n15 s' in skill and 'start and no end replaces that one line' in skill  # v18 SKILL.md


# ---------------------------------------------------------------- L7: search overflow, locate title words

def test_l7_search_overflow_names_the_files(ws):
    (ws / 'pay/a.py').write_text(''.join(f'A{i} = "http://a/{i}"\n' for i in range(12)))
    (ws / 'pay/b.py').write_text(''.join(f'B{i} = "http://b/{i}"\n' for i in range(5)))
    (ws / 'pay/highlight.py').write_text('URL = r"(file|http|https)://[-0-9a-zA-Z$_+!`(),.?/;:&=%#]*"\n')
    (ws / 'tests/test_urls.py').write_text(''.join(f'T{i} = "http://t/{i}"\n' for i in range(3)))
    code, out = run('search.py', ws, text='http')
    assert code == 0
    last = out.rstrip().splitlines()[-1]
    # v17 L9: the regex literal ranks first (rich_3067), so it is no longer the hit that is cut off
    assert out.startswith('pay/highlight.py:1: URL = r"(file|http|https)'), out
    # v18 search.py: 10 hits (was 15)
    assert last == ('... 11 more matches in 3 files: pay/b.py (5), pay/a.py (3), tests/test_urls.py (3); '
                    'narrow with within=<file or dir>'), out
    assert flow.parse_facts('search.py', obj(text='http'), out)['hits'][-1] == ('pay/a.py', 9)


def test_l7_search_overflow_caps_the_file_list():
    import search
    rest = [(False, f'pay/m{i}.py', 1, 'x') for i in range(12)]
    line = search.overflow_line(rest)
    assert line.startswith('... 12 more matches in 12 files: pay/m0.py (1), ') and 'and 7 more files;' in line  # v18: 5


def test_l7_locate_title_words_name_a_definition(ws):
    code, out = run('locate.py', ws, query='fix for padding width')
    assert code == 0
    lines = out.splitlines()
    assert lines[0].split()[1] == 'pay/table.py' and 'title names _get_padding_width' in lines[0], out
    assert lines[1].split() == ['Table._get_padding_width', 'L6-7'], out
    # v19: pay/padding.py is named like the issue (path match) and scores close: no preview, low confidence


def test_l7_locate_title_words_unit(ws):
    import kb
    import locate
    conn = kb.open(str(ws))
    paths = {p for (p,) in conn.execute('SELECT path FROM files')}
    assert locate.title_defs(conn, 'fix for padding width\n\nFixes #3871', paths) == {
        'pay/table.py': ('_get_padding_width', ('pay/table.py', 'Table._get_padding_width', 6, 7))}
    assert locate.title_defs(conn, 'Fix _get_padding_width', paths) == {}   # identifiers are locate's other signals
    assert locate.title_defs(conn, 'padding size is wrong', paths) == {
        'pay/padding.py': ('padding_size', ('pay/padding.py', 'Padding.padding_size', 17, 18))}
    assert locate.title_defs(conn, 'fix the width', paths) == {}             # one word is not a name
    ranked, *_ = locate.rank(str(ws), 'fix for padding width', conn=conn)
    assert ranked[0] == 'pay/table.py'
    w = locate.TITLE_WEIGHT
    try:
        locate.TITLE_WEIGHT = 0.0
        assert locate.rank(str(ws), 'fix for padding width', conn=conn)[0][0] == 'pay/padding.py'
    finally:
        locate.TITLE_WEIGHT = w


# ---------------------------------------------------------------- L8: snippet timeout and where it hung

def test_l8_snippet_timeout_is_15_and_shows_where_it_loops(ws):
    out = py(ws, '''
        import run
        assert run.SNIPPET_TIMEOUT == 15 and run.TIMEOUT == 90
        run.SNIPPET_TIMEOUT = 2
        sys.exit(run.main(["--snippet", "from pay.loop import spin\\nprint('start')\\nspin(1)"]))
    ''')
    lines = out.splitlines()
    assert lines[:2] == ['start', 'exit code: TIMEOUT'], out
    assert lines[2] == 'still running after 2s at (most recent call first):'
    assert re.fullmatch(r'  pay/loop\.py:[23] spin: (?:while n:|n \+= 1)', lines[3]), out   # where it loops
    assert lines[4] == '  snippet.py:3 <module>: spin(1)', out
    assert lines[-1] == 'timed out after 2s: the code does not return; the lines above show where it loops'
    assert 'runpy' not in out and '<string>' not in out


def test_l8_test_timeout_shows_the_test_frame(ws):
    out = py(ws, '''
        import run
        run.TIMEOUT = 3
        sys.exit(run.main(["--test", "tests/test_loop.py"]))
    ''')
    assert 'exit code: TIMEOUT' in out and re.search(r'^  pay/loop\.py:[23] spin: ', out, re.M), out
    assert '  tests/test_loop.py:5 test_spin: spin(1)' in out
    assert 'pluggy' not in out and '_pytest' not in out  # the callers below the repository's frames are left out
    assert 'timed out after 3s: pick a smaller test (-k) or skip verification' in out


def test_l8_errors_keep_their_traceback_without_wrapper_frames(ws):
    code, out = run('run.py', ws, snippet='import json\nx = 1\njson.loads("{")')
    assert code == 0 and 'exit code: 1' in out
    assert 'File "snippet.py", line 3, in <module>' in out and 'JSONDecodeError' in out
    assert 'runpy' not in out and '<string>' not in out and 'run_path' not in out
    code, out = run('run.py', ws, snippet='import sys\nprint(sys.argv[0].endswith(".py"), __name__)')
    assert out.startswith('True __main__\nexit code: 0'), out
    code, out = run('run.py', ws, test='tests/test_loop.py', k='nothing')
    assert 'exit code: 5' in out  # pytest's own exit status passes through the wrapper (no tests selected)
    assert git(ws, 'status', '--porcelain', '--untracked-files=all') == ''


# ---------------------------------------------------------------- L9: context guard

def test_l9_context_nearly_full_pushes_the_edit_then_submit(ws, monkeypatch):
    monkeypatch.setattr(flow, 'COMPACT_AT', 0)   # v18c: the safety net without compaction (tests/test_v18c_flow.py)
    step('locate.py', obj(query='Text append loops'), LOCATE_OUT)
    big = 'x' * 14500  # v18c: the sawtooth fit (5240 + 0.2953 * chars + 336 * calls + 2000) crosses 24000 at
    for i in range(3):   # the 4th read
        _, f = step('show.py', obj(target='pay/text.py', start=str(1 + i), end='20'), 'pay/text.py L1-20 of 21\n' + big)
    assert 'context nearly full' not in f
    _, f = step('show.py', obj(target='pay/text.py', start='5', end='20'), 'pay/text.py L5-20 of 21\n' + big)
    st = flow.State(flow.connect())
    assert st.ctx_tokens >= flow.CTX_WARN_TOKENS
    first = nexts(f)[0]
    assert first.startswith('NEXT: context nearly full: make your edit now: edit.py  target=pay/text.py'), f
    step('edit.py', obj(target='pay/text.py', old='a', new='b'), 'EDITED pay/text.py (exact)\n')
    _, f = step('run.py', *snippet(1, 'ok'))
    assert nexts(f) == ['NEXT: context nearly full: check_patch.py, then submit_patch now'], f
    _, f = step('check_patch.py', [], 'changed files:\n  pay/text.py | 2 +-\nREADY: call submit_patch now.\n')
    assert nexts(f) == ['NEXT: submit_patch (the patch is ready)'], f


def test_l9_counts_arguments_and_refusals(ws):
    step('locate.py', obj(query='q' * 1000), LOCATE_OUT)
    st = flow.State(flow.connect())
    assert st.ctx_chars >= 1000 + len(LOCATE_OUT)


# ---------------------------------------------------------------- infinite loops added by the patch

@pytest.mark.parametrize('body,flagged', [
    ('    while True:\n        x = 1\n', True),
    ('    while 1:\n        x = 1\n', True),
    ('    while True:\n        if x:\n            break\n', False),
    ('    while True:\n        for i in y:\n            break\n', True),       # breaks the inner loop only
    ('    while True:\n        def g():\n            return 1\n', True),     # returns from a nested def
    ('    while True:\n        return 1\n', False),
    ('    while True:\n        raise StopIteration\n', False),
    ('    while True:\n        yield 1\n', False),                          # a generator
    ('    while True:\n        sys.exit(0)\n', False),
    ('    while x:\n        x = 1\n', False),                             # not a constant condition
    ('    while False:\n        x = 1\n', False),
])
def test_infinite_loops_unit(body, flagged):
    text = 'def f(x, y):\n' + body
    assert guards.infinite_loops(text) == ([2] if flagged else [])
    assert guards.infinite_loops(text, lines={5}) == []


def test_check_patch_warns_about_an_added_endless_loop(ws):
    (ws / 'pay/loop.py').write_text(LOOP + '\n\ndef wait():\n    while True:\n        pass\n')
    git(ws, 'commit', '-qam', 'existing endless loop')   # pre-existing: not this patch's
    code, out = run('check_patch.py', ws)
    assert code == 1 and 'NO CHANGES' in out
    (ws / 'pay/text.py').write_text(TEXT.replace('        for start, end, style in text._spans:\n',
                                                 '        while True:\n            offset += 0\n'
                                                 '        for start, end, style in text._spans:\n', 1))
    code, out = run('check_patch.py', ws, flow_on=True)
    assert code == 0 and 'READY: call submit_patch now.' in out
    assert 'WARNING: pay/text.py:12 `while True:` has no break, return or raise inside: if it never ends, the hidden ' \
           'tests hang and fail' in out
    assert 'pay/loop.py' not in out.split('WARNING')[1]
    (ws / 'pay/new_mod.py').write_text('def g():\n    while 1:\n        pass\n')
    code, out = run('check_patch.py', ws)
    assert 'WARNING: pay/new_mod.py:2 `while True:`' in out


def test_no_bytecode_in_scripts_dir():
    assert not list(SCRIPTS.rglob('__pycache__')) and not list(SCRIPTS.rglob('*.pyc'))


def test_l1_hypothesis_mode_keeps_the_hypothesis_step(ws, monkeypatch):
    monkeypatch.setattr(flow, 'HYPOTHESIS_MODE', True)  # rule 7 is off since v19
    thin = LOCATE_OUT.replace('identifiers: Text, append', 'identifiers: (none)') + 'signal: thin=yes\n'
    step('locate.py', obj(query='fix it'), thin)
    assert flow.State(flow.connect()).mode == 'hypothesis'
    step('run.py', *snippet(1, 'a'))
    _, f = step('run.py', *snippet(2, 'b'))
    lines = nexts(f)
    assert lines[0].startswith('NEXT: 2 snippets did not reproduce it: sweep the invariant'), f
    assert lines[1].startswith('NEXT: the issue names few identifiers; write down 2-3 concrete guesses first: '), f
