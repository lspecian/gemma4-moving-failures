"""agent-v17: show.py / search.py levers L1 (context diet), L8 (show window), L9 (search beyond .py, ranking) of
research/notes/12-failures-eval113.md (agent/skills/repo-tools/scripts: show.py, search.py, names.py).

- L8 show.py start without end inside a definition longer than the window: the definition's first lines and a
  window that contains the line (rich_3777: start=724 in the 125-line Console.__init__ printed L623-722); from the
  line itself when the previous read of the file ended just before it.
- L1 compact mode: from flow.COMPACT_FROM_TOKENS (flow.context_tokens()) show.py prints narrower windows (60, then 40
  lines), shorter names blocks and outlines, search.py fewer hits; one line says so. Without flow.context_tokens
  (or when it fails) nothing changes. Long docstrings are folded in definition reads and long ranges; the names
  block lists 8 names (was 12).
- L9 search.py: no .py match -> other text files (.md, .txt, ...) labelled as such (fastapi_14492: within=docs);
  hits ranked regex literals > code (whole word) > code (part of a word) > comments/docstrings (rich_3067).
v18 (tests/test_v18_show_search.py) changed what some of these assert: unpadded line numbers, every docstring folded
(an explicit range keeps those of at most 8 lines), start alone = the def line and 20 lines on each side, a window
for long functions, one task-related `names:` line, 10 search hits.
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


def long_init(n_body):
    """A class whose __init__ has a 4-line signature and n_body numbered assignments (rich Console.__init__)."""
    body = ''.join(f'        self.attr{i} = {i}\n' for i in range(n_body))
    return ('class Console:\n'
            '    def __init__(\n'
            '        self,\n'
            '        width=80,\n'
            '    ):\n' + body +
            '\n'
            '    def size(self):\n'
            '        return self.attr0\n')


DOCSTRING = textwrap.dedent('''\
    def render(text, width):
        """Render text.

        Args:
            text: the text.
            width: the width.
            more: line 1.
            more: line 2.
            more: line 3.
            more: line 4.
            more: line 5.
            more: line 6.

        Returns:
            the lines.
        """
        lines = text.split()
        return [l[:width] for l in lines]
    ''')
MANY_NAMES = ''.join(f'def f{i}():\n    return {i}\n\n\n' for i in range(12)) + \
    'def use():\n' + ''.join(f'    x{i} = f{i}()\n' for i in range(12))


@pytest.fixture
def ws(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    files = {'pay/__init__.py': '', 'pay/console.py': long_init(130), 'pay/huge.py': long_init(200),
             'pay/render.py': DOCSTRING, 'pay/names.py': MANY_NAMES,
             'pay/plain.py': ''.join(f'V{i} = {i}\n' for i in range(150))}
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    monkeypatch.setenv('WORKSPACE', str(root))
    monkeypatch.setenv('REPO_TOOLS_DIR', str(tmp_path / 'state'))
    return root


def run(script, ws, **args):
    argv = [x for k, v in args.items() for x in ('--' + k, str(v))]
    env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'), PYTHONDONTWRITEBYTECODE='1',
               REPO_TOOLS_FLOW='off')
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *argv], capture_output=True, text=True, env=env,
                       timeout=120)
    assert 'Traceback' not in r.stderr, r.stderr
    return r.returncode, r.stdout


def py(ws, code, tokens=None, viewed=None, **args):
    """Run a script's main in-process with flow.context_tokens() patched to `tokens` (None: removed from flow, as
    before v17; 'fail': raises) and flow.peek().viewed to `viewed`. Returns stdout."""
    argv = [x for k, v in args.items() for x in ('--' + k, str(v))]
    pre = textwrap.dedent(f'''\
        import sys, types; sys.dont_write_bytecode = True; sys.path.insert(0, {str(SCRIPTS)!r})
        import flow
        tokens, viewed = {tokens!r}, {viewed!r}
        if tokens is None:
            for k in ('context_tokens', 'COMPACT_FROM_TOKENS', 'COMPACT_STRONG_TOKENS'):
                if hasattr(flow, k):
                    delattr(flow, k)
        else:
            def ctx():
                if tokens == 'fail':
                    raise RuntimeError('no state')
                return tokens
            flow.context_tokens, flow.COMPACT_FROM_TOKENS = ctx, 15000
        if viewed is not None:
            flow.peek = lambda: types.SimpleNamespace(viewed=viewed, candidates=[], conn=None)
        ''')
    env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'), PYTHONDONTWRITEBYTECODE='1',
               REPO_TOOLS_FLOW='off')
    r = subprocess.run([sys.executable, '-c', pre + textwrap.dedent(code), *argv], capture_output=True, text=True,
                       env=env, timeout=120)
    assert 'Traceback' not in r.stderr, r.stderr
    return r.stdout


SHOW = 'import show; show.main(sys.argv[1:])'
SEARCH = 'import search; search.main(sys.argv[1:])'


def numbered(out):
    """The line numbers of the code lines (v18: `123  code`, a blank line the number alone)."""
    return [int(m.group(1)) for m in map(re.compile(r'^(\d+)(?:  |$)').match, out.splitlines()) if m]


# ---------------------------------------------------------------- L8: the window contains the line asked for

def test_l8_start_deep_in_a_long_definition_is_shown(ws):
    """rich_3777: __init__ spans L2-135 (134 lines > 100); start=120 used to print L2-101."""
    code, out = run('show.py', ws, target='pay/console.py', start='120')
    assert code == 0
    first = out.splitlines()[0]
    assert first.startswith('pay/console.py Console.__init__ L') and first.endswith(' of 138'), out
    shown = numbered(out)
    # v18: the def line, then 20 lines on each side of the line (clipped to the definition's end)
    assert 120 in shown and shown[0] == 2 and shown[1:] == list(range(95, 136)), shown
    assert '... (lines 3-94 not shown; Console.__init__ spans L2-135)' in out
    assert flow.parse_facts('show.py', ['--target', 'pay/console.py', '--start', '120'], out)['end'] == shown[-1]


def test_l8_window_is_centred_and_says_where_the_definition_continues(ws):
    code, out = run('show.py', ws, target='pay/huge.py', start='60')   # a 205-line __init__: a 41-line window
    shown = numbered(out)
    assert shown[0] == 2 and shown[1:] == list(range(40, 81)), shown  # v18: the def line, then 60 +- 20
    last = [l for l in out.splitlines() if l.startswith('... Console.__init__ continues to line 205; rest: ')]
    assert last, out
    assert calls(last[0]) == [('show.py', {'target': 'pay/huge.py', 'start': str(shown[-1] + 1),
                                           'end': str(shown[-1] + 100)})]
    assert_notation(last[0])


def test_l8_start_near_the_head_prints_one_block(ws):
    code, out = run('show.py', ws, target='pay/console.py', start='8')
    shown = numbered(out)
    assert shown == list(range(2, 43)) and 'not shown' not in out, shown  # v18: one block of 41 lines


def test_l8_short_definition_is_shown_whole(ws):
    code, out = run('show.py', ws, target='pay/console.py', start='137')
    assert numbered(out) == [137, 138] and out.startswith('pay/console.py Console.size L137-138 of 138'), out


def test_l8_start_that_continues_the_previous_read_starts_there(ws):
    """fastapi_14099/14962: start=347 right after a window that ended at 346 printed the same window again."""
    out = py(ws, SHOW, viewed=[('pay/huge.py', 20, 79, 3, '')], target='pay/huge.py', start='80')
    shown = numbered(out)
    assert shown[:2] == [2, 80] and shown[-1] == 120, shown
    out = py(ws, SHOW, viewed=[('pay/other.py', 20, 79, 3, '')], target='pay/huge.py', start='80')
    assert numbered(out)[1] == 60                                       # another file: centred


def test_l8_line_past_the_end_of_the_file(ws):
    code, out = run('show.py', ws, target='pay/console.py', start='500')
    assert code == 1 and out.startswith('pay/console.py has only 138 lines; e.g. show.py  target=pay/console.py')


# ---------------------------------------------------------------- L1: docstrings, names block

def test_l1_symbol_read_folds_a_long_docstring(ws):
    code, out = run('show.py', ws, target='pay/render.py', symbol='render')
    # v18: every docstring folded to its first line
    assert numbered(out) == [1, 2, 17, 18] and '\n... (docstring lines 3-16)\n' in out, out
    assert '\n17      lines = text.split()' in out and 'more: line 3' not in out


def test_l1_explicit_ranges_fold_only_long_ranges_and_never_the_asked_lines(ws):
    code, out = run('show.py', ws, target='pay/render.py', start='1', end='18')
    assert numbered(out) == [1, 2, 17, 18]  # v18: a docstring of more than 8 lines inside the range is folded
    code, out = run('show.py', ws, target='pay/render.py', start='3', end='18')    # the range starts inside it
    assert numbered(out) == list(range(3, 19))
    code, out = run('show.py', ws, target='pay/render.py', start='10')             # start inside the docstring
    assert 10 in numbered(out)


def test_l1_names_block_lists_eight(ws):
    code, out = run('show.py', ws, target='pay/names.py', symbol='use')
    assert 'names:' not in out and 'Names used here' not in out  # v18: module functions unrelated to the task: none


# ---------------------------------------------------------------- L1: compact mode

def test_l1_no_compact_without_flow_estimate_or_when_it_fails(ws):
    for tokens in (None, 'fail', 14999):
        out = py(ws, SHOW, tokens=tokens, target='pay/plain.py', start='1', end='100')
        assert numbered(out) == list(range(1, 101)) and 'compact' not in out, (tokens, out)


def test_l1_compact_keeps_explicit_ranges_whole(ws):
    """v17b (E039: compact mode doubled the share of explicit ranges returned short, 13% -> 26%, each followed by a
    re-read): an explicit start/end range keeps the normal 100-line limit at every compact level."""
    for tokens in (15000, 21000):
        out = py(ws, SHOW, tokens=tokens, target='pay/plain.py', start='1', end='100')
        assert numbered(out) == list(range(1, 101)) and 'compact' not in out, (tokens, out)
    out = py(ws, SHOW, tokens=21000, target='pay/plain.py', start='1', end='30')
    assert numbered(out) == list(range(1, 31)) and 'compact' not in out


def test_l1_compact_line_window_still_contains_the_line(ws):
    out = py(ws, SHOW, tokens=21000, target='pay/console.py', start='120')
    shown = numbered(out)
    assert 120 in shown and shown[0] == 2 and len(shown) <= 22, shown  # v18 compact level 2: 120 +- 10
    assert 'compact:' in out


def test_l1_compact_names_block_top_entries_only(ws):
    out = py(ws, SHOW, tokens=16000, target='pay/names.py', symbol='use')
    assert 'names:' not in out  # v18: see test_v18_show_search for the compact names line


def test_l1_compact_long_symbol_is_head_and_body(ws):
    out = py(ws, SHOW, tokens=16000, target='pay/console.py', symbol='Console.__init__')
    shown = numbered(out)
    assert out.startswith('pay/console.py Console.__init__ L2-135') and len(shown) <= 40, out  # v18: a window
    assert '... Console.__init__ continues to line 135; rest: show.py  target=pay/console.py' in out
    assert 'compact:' in out


def test_l1_compact_search_fewer_hits(ws):
    (ws / 'pay/urls.py').write_text(''.join(f'U{i} = "http://x/{i}"\n' for i in range(30)))
    out = py(ws, SEARCH, tokens=None, text='http')
    assert len([l for l in out.splitlines() if l.startswith('pay/urls.py:')]) == 10 and 'compact' not in out  # v18
    out = py(ws, SEARCH, tokens=16000, text='http')
    assert len([l for l in out.splitlines() if l.startswith('pay/urls.py:')]) == 8
    assert '... 22 more matches in 1 file: pay/urls.py (22)' in out and out.rstrip().endswith(
        'compact: context is large; fewer hits shown, narrow with within= if needed')
    out = py(ws, SEARCH, tokens=21000, text='http')
    assert len([l for l in out.splitlines() if l.startswith('pay/urls.py:')]) == 6


# ---------------------------------------------------------------- L9: search beyond .py files, ranking

def test_l9_search_falls_back_to_text_files_and_labels_them(ws):
    """fastapi_14492: search MIT within=docs said "no matches" because docs/ has no .py files."""
    (ws / 'docs/en').mkdir(parents=True)
    (ws / 'docs/en/metadata.md').write_text('# Metadata\n\nlicense_info: MIT\n')
    (ws / 'docs/de').mkdir(parents=True)
    (ws / 'docs/de/metadata.md').write_text('# Metadaten\n\nLizenz: MIT\n')
    code, out = run('search.py', ws, text='MIT', within='docs')
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == 'no .py file matches in docs; 2 matches in other files (.md):', out
    assert lines[1] == 'docs/en/metadata.md:3:' and lines[4] == '3> license_info: MIT', out   # 2 hits: context
    assert lines[6] == 'docs/de/metadata.md:3:', out                                             # translation last
    out = py(ws, SEARCH + '; print(flow._FACTS["hits"])', text='MIT', within='docs')
    assert out.rstrip().splitlines()[-1] == "[('docs/en/metadata.md', 3), ('docs/de/metadata.md', 3)]", out


def test_l9_text_files_only_when_no_py_file_matches(ws):
    (ws / 'README.md').write_text('render text\n')
    code, out = run('search.py', ws, text='render')
    assert 'README.md' not in out and 'pay/render.py' in out
    code, out = run('search.py', ws, text='zz_nowhere_zz')
    assert code == 1 and out.startswith("no matches for 'zz_nowhere_zz'")


def test_l9_ranking_puts_regex_literals_and_code_before_docs(ws):
    """rich_3067: the url regex in rich/highlighter.py was cut off behind emoji-table and docstring hits."""
    (ws / 'pay/a_emoji.py').write_text(''.join(f'E{i} = "curly_{i}"\n' for i in range(5)))   # 'url' inside a word
    (ws / 'pay/b_doc.py').write_text('def f():\n    """The url of it.\n\n    url again\n    """\n    # url comment\n'
                                     '    return 1\n')
    (ws / 'pay/c_styles.py').write_text('STYLES = {"repr.url": 1}\n')
    (ws / 'pay/z_highlight.py').write_text('import re\nURL = re.compile(r"(?P<url>https?://\\S*)")\n')
    code, out = run('search.py', ws, text='url')
    hits = [m.group(1) for m in map(re.compile(r'^(pay/\S+?:\d+)').match, out.splitlines()) if m]
    assert hits[:2] == ['pay/z_highlight.py:2', 'pay/c_styles.py:1'], out
    assert hits.index('pay/a_emoji.py:1') < hits.index('pay/b_doc.py:2'), hits   # code before docstrings/comments
    nxt = flow.parse_facts('search.py', ['--text', 'url'], out)['hits'][0]
    assert nxt == ('pay/z_highlight.py', 2)                                       # NEXT reads the best hit


def test_l9_rank_unit():
    import search
    src = {'m.py': ['x = "url"', '# url', 'y = curl', 'Z = re.compile("url")']}
    hits = [(False, 'm.py', i + 1, l) for i, l in enumerate(src['m.py'])] + [(True, 'tests/t.py', 1, 'url')]
    out = search.rank(hits, re.compile('url'), lambda rel: src.get(rel, ['url']))
    assert [h[2] for h in out] == [4, 1, 3, 2, 1] and out[-1][1] == 'tests/t.py'
