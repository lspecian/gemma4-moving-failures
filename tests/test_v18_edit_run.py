"""agent-v18 "lean": edit.py, create.py, run.py, argfix.py, guards.py and the alias scripts
(research/notes/13-tool-efficiency.md; 13-appendix/code-review.md P4-P6; audit.md #7, #8, #10).

1. Alias scripts repro.py / reproduce_issue.py / reproduce_bug.py run their code like run.py snippet, recorded as
   run.py by the flow engine; create.py with a scratch .py target (or no target) and Python content does the same.
2. edit.py without target: the file the model was reading, when unambiguous.
3. "start" without "end" and a multi-line NEW that rewrites the lines at START: applied to them (v17b refused 47
   such calls; v19: infer_end, see test_v19_edit.py), refused with the likely range when unsure.
4. P4: line numbers read before an edit that moved them are shifted (NOTE) or refused with the current numbers.
5. P6: an indentation repair never leaves code after a return/raise (fastapi_14458 r1 call 10).
6. P5: no closest region below similarity 0.6.
7. Message diet: one-line sites, single-example usage errors, syntax errors with NEW numbered and marked.
8. run.py: pytest output without separators/library frames, "same output as call N", the last expression's repr.
Run with scripts/test_skills.sh (sandbox image, py3.13).
"""

import os
import pathlib
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
import guards  # noqa: E402
import run as run_py  # noqa: E402

# fastapi_14458: the same two lines in three properties
MODELS = textwrap.dedent('''\
    import inspect


    class Dependant:
        def __init__(self, call=None):
            self.call = call

        @property
        def is_gen_callable(self):
            if inspect.isgeneratorfunction(self.call):
                return True
            dunder_call = getattr(self.call, "__call__", None)
            return inspect.isgeneratorfunction(dunder_call)

        @property
        def is_async_gen_callable(self):
            if inspect.isasyncgenfunction(
                self.call
            ):
                return True
            dunder_call = getattr(self.call, "__call__", None)
            return inspect.isasyncgenfunction(dunder_call)

        @property
        def is_coroutine_callable(self):
            if inspect.iscoroutinefunction(self.call):
                return True
            dunder_call = getattr(self.call, "__call__", None)
            return inspect.iscoroutinefunction(dunder_call)
    ''')
# rich_4079: an if/elif chain in a loop, and an if/else
MARKDOWN = textwrap.dedent('''\
    def render(tokens, context, syntax=None):
        for token in tokens:
            node_type = token.type
            if node_type == "text":
                context.on_text(token.content, node_type)
            elif node_type == "hardbreak":
                context.on_text("\\n", node_type)
            elif node_type == "softbreak":
                context.on_text(" ", node_type)
            else:
                context.on_other(token)
        return context


    def on_text(stack, text, node_type, syntax=None):
        if node_type in {"fence", "code_inline"} and syntax is not None:
            highlight_text = syntax.highlight(text)
            highlight_text.rstrip()
            stack.top.on_text(highlight_text)
        else:
            stack.top.on_text(text)
    ''')
# fastapi_14583 / fastapi_13207: an import block, a multi-line annotated assignment
COMPAT = textwrap.dedent('''\
    from collections.abc import Sequence
    from typing import Any
    from typing import Union

    override_mode: Union[str, None] = (
        "validation"
    )


    def is_sequence(value: Any) -> bool:
        return isinstance(value, Sequence)
    ''')
CALC = textwrap.dedent('''\
    def total(values):
        result = 0
        for v in values:
            result += v
        return result


    def mean(values):
        return total(values) / len(values)
    ''')
NOTES = 'just some notes\nabout nothing\n'


def git(ws, *args):
    return subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def ws(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    files = {'pkg/__init__.py': '', 'pkg/models.py': MODELS, 'pkg/markdown.py': MARKDOWN, 'pkg/compat.py': COMPAT,
             'pkg/calc.py': CALC, 'docs/notes.txt': NOTES,
             'tests/test_calc.py': 'import json\nfrom pkg.calc import mean\n\n\ndef test_mean():\n'
                                   '    assert mean([1, 2]) == 2\n\n\ndef test_json():\n    json.loads("{x")\n'}
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, 'init', '-q')
    git(root, 'add', '-A')
    git(root, 'commit', '-qm', 'baseline')
    monkeypatch.setenv('WORKSPACE', str(root))
    monkeypatch.setenv('REPO_TOOLS_DIR', str(tmp_path / 'state'))
    return root


def run(script, ws, *positional, flow_on=True, **args):
    argv = list(positional) + [x for k, v in args.items() for x in ('--' + k, str(v))]
    env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'), PYTHONDONTWRITEBYTECODE='1')
    env['REPO_TOOLS_FLOW'] = 'on' if flow_on else 'off'
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *argv], capture_output=True, text=True, env=env,
                       timeout=180)
    assert 'Traceback' not in r.stderr, r.stderr
    return r.returncode, r.stdout


def body(out):
    """The script's own output, without the flow footer (BUDGET/NEXT lines and the backslash note)."""
    return '\n'.join(l for l in out.splitlines() if not l.startswith(('NEXT:', 'BUDGET:', 'note: results', '[')))


def read(script, args, path, start, end, out='x\n'):
    """Record a read in the flow state the way flow.main does after show.py (facts path/start/end)."""
    flow.after(script, [x for k, v in args.items() for x in ('--' + k, str(v))], 0, out, path=path, start=start,
               end=end)


def src(ws, rel):
    return (ws / rel).read_text()


def line_of(text, line, after=0):
    return text.split('\n').index(line, after) + 1


def actions():
    st = flow.State(flow.connect())
    return [(a[1], a[3]) for a in st.actions]


# ---------------------------------------------------------------- 1. aliases

@pytest.mark.parametrize('script,key', [('repro.py', 'content'), ('reproduce_issue.py', 'code'),
                                        ('reproduce_bug.py', 'snippet')])
def test_alias_scripts_run_the_code_as_a_snippet(ws, script, key):
    code, out = run(script, ws, **{key: 'from pkg.calc import mean\nprint(mean([1, 2, 3]))'})
    assert code == 0, out
    assert out.startswith(f'NOTE: {script} is not one of the scripts; ran the code as run.py snippet=<the code> '
                          '(call run.py next time)\n2.0\nexit code: 0'), out
    assert actions() == [('run.py', 'ok')]  # recorded and budgeted as run.py
    assert not (ws / script).exists()


def test_alias_positional_code_and_missing_code(ws):
    code, out = run('repro.py', ws, 'print(40 + 2)')
    assert code == 0 and '42\nexit code: 0' in out, out
    code, out = run('repro.py', ws, target='repro.py')
    assert code == 2 and body(out).startswith('USAGE ERROR: repro.py is not one of the scripts; run code with '
                                              'run.py  snippet='), out
    assert_notation(body(out))


def test_alias_code_unit():
    assert run_py.alias_code(['--content', 'x = 1']) == 'x = 1'
    assert run_py.alias_code(['--target', 'repro.py', '--code', 'y']) == 'y'
    assert run_py.alias_code(['--weird', 'print(1)']) == 'print(1)'
    assert run_py.alias_code(['snippet', 'print(2)']) == 'print(2)'
    assert run_py.alias_code(['--target', 'repro.py']) is None
    assert run_py.alias_code([]) is None
    assert set(run_py.ALIASES) == {p.name for p in SCRIPTS.glob('repro*.py')}


def test_create_scratch_python_runs_as_a_snippet(ws):
    """29 'REFUSED: repro.py looks like a scratch file' and 25 create.py calls without target in E040/E041."""
    code, out = run('create.py', ws, target='repro.py', content='print("hi")')
    assert code == 0 and body(out) == ('NOTE: repro.py looks like a scratch file and every new file ends up in the '
                                       'patch, so nothing was created; its content ran as run.py snippet instead\n'
                                       'hi\nexit code: 0'), out
    assert not (ws / 'repro.py').exists() and actions() == [('run.py', 'ok')]
    code, out = run('create.py', ws, content='import pkg\nprint(pkg.__name__)')
    assert code == 0 and body(out).startswith('NOTE: no target given and the content is Python code, so nothing was '
                                              'created; it ran as run.py snippet instead (to add a repository file, '
                                              'give target=<path>)\npkg\nexit code: 0'), out


def test_create_without_target_and_non_python_content_is_still_a_usage_error(ws):
    code, out = run('create.py', ws, content='name: Release\non:\n  push: [main\n')
    assert code == 2 and body(out).startswith('USAGE ERROR: create.py needs the key target'), out
    assert '\nusage:' not in out


def test_create_real_files_unchanged(ws):
    code, out = run('create.py', ws, flow_on=False, target='pkg/new_mod.py', content='X = 1\n')
    assert code == 0 and out.startswith('CREATED pkg/new_mod.py (1 lines, python syntax OK)'), out
    assert 'part of the patch' not in out  # v18: dropped


# ---------------------------------------------------------------- 2. no target

def test_edit_without_target_uses_the_file_that_has_old(ws):
    read('show.py', {'target': 'pkg/calc.py'}, 'pkg/calc.py', 1, 9)
    read('show.py', {'target': 'pkg/compat.py'}, 'pkg/compat.py', 1, 11)
    code, out = run('edit.py', ws, old='        result += v', new='        result += v or 0')
    assert code == 0, out
    assert out.startswith('NOTE: no target given; OLD is in pkg/calc.py, a file you read, so edited that\n'
                          'EDITED pkg/calc.py (exact)'), out
    assert 'result += v or 0' in src(ws, 'pkg/calc.py')
    st = flow.State(flow.connect())
    assert st.edits[-1][1] == 'pkg/calc.py' and st.edits[-1][2] == 1  # recorded under the inferred file


def test_edit_without_target_and_old_in_two_read_files_is_a_usage_error(ws):
    read('show.py', {'target': 'pkg/calc.py'}, 'pkg/calc.py', 1, 9)
    read('show.py', {'target': 'pkg/models.py'}, 'pkg/models.py', 1, 30)
    code, out = run('edit.py', ws, old='return', new='return None')
    assert code == 2 and body(out).startswith('USAGE ERROR: edit.py needs the key target'), out


def test_edit_without_target_by_lines_uses_the_file_whose_read_covers_them(ws):
    read('show.py', {'target': 'pkg/calc.py'}, 'pkg/calc.py', 1, 5)
    read('show.py', {'target': 'pkg/compat.py'}, 'pkg/compat.py', 9, 11)
    n = line_of(CALC, '    result = 0')
    code, out = run('edit.py', ws, start=n, end=n, new='    result = 0.0')
    assert code == 0 and out.startswith(f'NOTE: no target given; lines {n}-{n} are in pkg/calc.py, the file you read '
                                        'them in\nEDITED pkg/calc.py'), out


def test_edit_without_target_and_nothing_read_is_a_usage_error(ws):
    code, out = run('edit.py', ws, old='x', new='y')
    assert code == 2 and 'needs the key target' in out and out.count('edit.py  target=') == 1, out


# ---------------------------------------------------------------- 3. start without end

def test_start_only_if_header_rewrite_replaces_only_the_if_clause(ws):
    """rich_4079 r1 (refused at 275 s): NEW = a new condition + the first body line; the elif chain stays."""
    n = line_of(MARKDOWN, '        if node_type == "text":')
    new = '        if node_type in {"text", "inline"}:\n            context.on_text(token.content, node_type)'
    code, out = run('edit.py', ws, target='pkg/markdown.py', start=n, new=new)
    assert code == 0, out
    assert out.startswith(f'NOTE: no end given: NEW ends with line {n + 1}, so it replaced lines {n}-{n + 1}\n'
                          f'EDITED pkg/markdown.py (lines {n}-{n + 1})'), out
    assert src(ws, 'pkg/markdown.py') == MARKDOWN.replace('if node_type == "text":', 'if node_type in {"text", "inline"}:')


def test_start_only_if_clause_without_else_keeps_the_else(ws):
    """rich_4079 r2: NEW rewrote the if clause (6 lines) and left out the else."""
    n = line_of(MARKDOWN, '    if node_type in {"fence", "code_inline"} and syntax is not None:')
    new = ('    if node_type in {"fence", "code_inline"} and syntax is not None:\n'
           '        highlight_text = syntax.highlight(text)\n        highlight_text = highlight_text.rstrip()\n'
           '        stack.top.on_text(highlight_text)')
    code, out = run('edit.py', ws, target='pkg/markdown.py', start=n, new=new)
    assert code == 0 and f'NEW ends with line {n + 3}, so it replaced lines {n}-{n + 3}\n' in out, out
    assert src(ws, 'pkg/markdown.py') == MARKDOWN.replace('        highlight_text.rstrip()\n',
                                                          '        highlight_text = highlight_text.rstrip()\n')


def test_start_only_with_else_in_new_replaces_the_whole_statement(ws):
    n = line_of(MARKDOWN, '    if node_type in {"fence", "code_inline"} and syntax is not None:')
    new = ('    if node_type in {"fence", "code_inline"} and syntax:\n        stack.top.on_text(syntax.highlight(text))\n'
           '    else:\n        stack.top.on_text(text)')
    code, out = run('edit.py', ws, target='pkg/markdown.py', start=n, new=new)
    assert code == 0 and f'so it replaced lines {n}-{n + 5}\n' in out, out
    assert src(ws, 'pkg/markdown.py').endswith(new + '\n')


def test_start_only_import_block_extends_over_the_lines_new_repeats(ws):
    """fastapi_14583: NEW = the import lines 1-3 plus a new import; replacing line 1 alone duplicated 2-3."""
    new = 'from collections.abc import Sequence\nfrom typing import Any\nfrom typing import Union\nimport copy'
    code, out = run('edit.py', ws, target='pkg/compat.py', start=1, new=new)
    assert code == 0 and 'NOTE: no end given: NEW repeats lines 1-3 and adds 1 new line, so it replaced lines 1-3 only' \
        in out, out
    assert src(ws, 'pkg/compat.py') == COMPAT.replace('from typing import Union\n', 'from typing import Union\nimport copy\n')


def test_start_only_multiline_assignment(ws):
    """fastapi_13207: the annotated assignment spans 3 lines; NEW rewrites it."""
    n = line_of(COMPAT, 'override_mode: Union[str, None] = (')
    new = 'override_mode: Union[str, None] = (\n    "serialization"\n)'
    code, out = run('edit.py', ws, target='pkg/compat.py', start=n, new=new)
    assert code == 0 and f'NEW rewrites lines {n}-{n + 2} line for line, so it replaced them' in out, out
    assert '    "serialization"\n)\n' in src(ws, 'pkg/compat.py') and '"validation"' not in src(ws, 'pkg/compat.py')


def test_start_only_other_kind_is_still_refused(ws):
    """fastapi_14360: START is an assignment, NEW starts with an if: the v17b refusal with its ready range."""
    n = line_of(CALC, '    result = 0')
    code, out = run('edit.py', ws, target='pkg/calc.py', start=n, new='    if not values:\n        return 0')
    assert code == 2 and out.startswith('NOT APPLIED: "start" without "end" and a NEW of 2 lines'), out
    assert src(ws, 'pkg/calc.py') == CALC


def test_start_only_that_would_drop_a_line_is_not_applied(ws):
    """NEW repeats the if header and its first line only: it ends with line n + 1, which changes nothing (v18 refused
    extending over the whole if, which would have deleted the rest)."""
    n = line_of(MARKDOWN, '    if node_type in {"fence", "code_inline"} and syntax is not None:')
    new = ('    if node_type in {"fence", "code_inline"} and syntax is not None:\n'
           '        highlight_text = syntax.highlight(text)')
    code, out = run('edit.py', ws, target='pkg/markdown.py', start=n, new=new)
    assert code == 1 and 'NOT APPLIED: the edit changes nothing' in out and src(ws, 'pkg/markdown.py') == MARKDOWN, out


def test_start_only_extension_that_would_not_parse_falls_back_to_the_refusal(ws):
    n = line_of(CALC, '    for v in values:')
    code, out = run('edit.py', ws, target='pkg/calc.py', start=n, new='    for v in values:\n        result += (v')
    assert code == 2 and out.startswith('NOT APPLIED: "start" without "end"'), out
    assert 'no end given' not in out and src(ws, 'pkg/calc.py') == CALC


def test_infer_end_unit():
    lines = MARKDOWN.split('\n')
    n = line_of(MARKDOWN, '        if node_type == "text":')
    # two clauses reproduced, the first rewritten: NEW ends with the elif's body line
    assert edit.infer_end(MARKDOWN, n, ['        if node_type == "x":', lines[n], '        elif node_type == "hardbreak":',
                                        lines[n + 2]], True)[:2] == (n + 3, True)
    # NEW repeats the header and adds a line: an insertion after it, not a rewrite of the if
    end, sure, why, _ = edit.infer_end(MARKDOWN, n, [lines[n - 1], '            count += 1'], True)
    assert (end, sure) == (n, True) and why.startswith(f'NEW repeats line {n} and adds 1 new line'), why
    d = line_of(MARKDOWN, 'def on_text(stack, text, node_type, syntax=None):')
    assert edit.infer_end(MARKDOWN, d, ['def other(x):', '    return x'], True)[1] is False  # another def: unsure


# ---------------------------------------------------------------- 4. stale line numbers (P4)

def test_stale_site_numbers_after_an_insert_are_shifted(ws):
    """fastapi_14458 r1 calls 9-11: the ambiguous-OLD list, an edit at the first site adding 2 lines, then an edit
    of the second site with its old numbers (a SyntaxError and a re-read in v17b)."""
    old = '            return True\n        dunder_call = getattr(self.call, "__call__", None)'
    new = ('            return True\n        if inspect.isclass(self.call):\n            return False\n'
           '        dunder_call = getattr(self.call, "__call__", None)')
    code, out = run('edit.py', ws, target='pkg/models.py', old=old, new=new)
    assert code == 1 and 'OLD occurs 3 times in pkg/models.py' in out, out
    sites = [(int(l.split()[0].split('-')[0]), int(l.split()[0].split('-')[1])) for l in out.splitlines()
             if l.startswith('  ') and '-' in l.split()[0]]
    assert len(sites) == 3, out
    (a1, b1), (a2, b2), (a3, b3) = sites
    code, out = run('edit.py', ws, target='pkg/models.py', start=a1, end=b1, new=new)
    assert code == 0, out
    assert (f'remaining sites of that OLD, current lines: {a2 + 2}-{b2 + 2} Dependant.is_async_gen_callable, '
            f'{a3 + 2}-{b3 + 2} Dependant.is_coroutine_callable') in out.splitlines(), out
    code, out = run('edit.py', ws, target='pkg/models.py', start=a2, end=b2, new=new)
    assert code == 0, out
    assert out.startswith(f'NOTE: lines {a2}-{b2} are numbered as read at call 1, before your edit at call 2 added 2 '
                          f'lines above them; used the current numbers {a2 + 2}-{b2 + 2}\n'
                          f'EDITED pkg/models.py (lines {a2 + 2}-{b2 + 2})'), out
    text = src(ws, 'pkg/models.py')
    assert text.count('if inspect.isclass(self.call):') == 2 and guards.unreachable(MODELS, text) == []
    compile(text, 'models.py', 'exec')


def test_numbers_printed_after_the_edit_are_used_as_given(ws):
    """The "also appears at" lines of a successful edit carry current numbers: no shift."""
    old = '            return True\n        dunder_call = getattr(self.call, "__call__", None)'
    code, out = run('edit.py', ws, target='pkg/models.py', old=old, new='x')
    a1 = int(out.splitlines()[1].split()[0].split('-')[0])
    new = ('            return True\n        if inspect.isclass(self.call):\n            return False\n'
           '        dunder_call = getattr(self.call, "__call__", None) or self.call')
    code, out = run('edit.py', ws, target='pkg/models.py', start=a1, end=a1 + 1, new=new)
    assert code == 0, out
    also = [l for l in out.splitlines() if l.startswith('the same code also appears at: ')]
    assert also and 'remaining sites' not in out, out
    n = int(also[0].split('pkg/models.py:')[1].split()[0])
    code, out = run('edit.py', ws, target='pkg/models.py', start=n, end=n,
                    new='        dunder_call = getattr(self.call, "__call__", None) or self.call')
    assert code == 0 and out.startswith(f'EDITED pkg/models.py (lines {n}-{n})'), out


def test_numbers_from_a_read_after_the_edit_are_used_as_given(ws):
    n = line_of(CALC, '    result = 0')
    code, out = run('edit.py', ws, target='pkg/calc.py', start=n, end=n, new='    """Sum."""\n    result = 0')
    assert code == 0, out
    m = line_of(src(ws, 'pkg/calc.py'), '        result += v')
    read('show.py', {'target': 'pkg/calc.py', 'start': 1, 'end': 9}, 'pkg/calc.py', 1, 9)
    code, out = run('edit.py', ws, target='pkg/calc.py', start=m, end=m, new='        result += v or 0')
    assert code == 0 and out.startswith(f'EDITED pkg/calc.py (lines {m}-{m})'), out


def test_stale_read_numbers_shift_when_new_matches_the_moved_lines(ws):
    read('show.py', {'target': 'pkg/calc.py'}, 'pkg/calc.py', 1, 9)
    n = line_of(CALC, '    result = 0')
    code, out = run('edit.py', ws, target='pkg/calc.py', start=n, end=n, new='    """Sum."""\n    result = 0')
    assert code == 0, out
    m = line_of(CALC, '    return total(values) / len(values)')  # numbered as read before the edit
    code, out = run('edit.py', ws, target='pkg/calc.py', start=m, end=m,
                    new='    return total(values) / max(1, len(values))')
    assert code == 0 and out.startswith(f'NOTE: line {m} is numbered as read at call 1, before your edit at call 2 '
                                        f'added 1 line above it; used the current number {m + 1}\n'), out
    assert src(ws, 'pkg/calc.py').endswith('    return total(values) / max(1, len(values))\n')


def test_stale_numbers_of_lines_the_edit_changed_are_refused(ws):
    read('show.py', {'target': 'pkg/calc.py'}, 'pkg/calc.py', 1, 9)
    code, out = run('edit.py', ws, target='pkg/calc.py', start=3, end=4,
                    new='    for v in values:\n        result += v\n        result += 0')
    assert code == 0, out
    code, out = run('edit.py', ws, target='pkg/calc.py', start=4, end=5, new='        result -= v\n    return result')
    assert code == 1 and body(out).startswith('NOT APPLIED: lines 4-5 are numbered as read at call 1, and your edit at '
                                              'call 2 changed those lines since'), out
    assert 'show.py  target=pkg/calc.py  start=1  end=15' in out
    assert_notation(body(out))


def test_stale_numbers_with_an_unclear_new_are_refused_with_the_current_lines(ws):
    read('show.py', {'target': 'pkg/calc.py'}, 'pkg/calc.py', 1, 9)
    code, out = run('edit.py', ws, target='pkg/calc.py', start=1, end=1, new='import math\n\n\ndef total(values):')
    assert code == 0, out
    m = line_of(CALC, 'def mean(values):')
    code, out = run('edit.py', ws, target='pkg/calc.py', start=m, end=m, new='zzz = 1')
    assert code == 1, out
    lines = body(out).splitlines()
    assert lines[0] == (f'NOT APPLIED: line {m} is numbered as read at call 1, before your edit at call 2 added 3 lines '
                        f'above it; it is now {m + 3}, and NEW does not tell which you meant, so nothing was changed. '
                        'Lines now:'), out
    assert lines[1] == f'{m + 3:5d}  def mean(values):'
    assert lines[-1] == f'CLOSEST: pkg/calc.py {m + 3} {m + 3}'
    assert_notation(body(out))


def test_map_range_unit():
    assert edit.map_range(10, 12, [[2, 3, 2, 5]]) == (12, 14)    # 1 line replaced by 3 above
    assert edit.map_range(10, 12, [[4, 4, 4, 6]]) == (12, 14)    # 2 lines inserted after line 4
    assert edit.map_range(10, 12, [[10, 10, 10, 11]]) is None    # inserted inside the range
    assert edit.map_range(10, 12, [[11, 12, 11, 12]]) is None    # line 12 replaced
    assert edit.map_range(10, 12, [[20, 22, 20, 21]]) == (10, 12)  # below: unchanged
    assert edit.map_range(10, 12, [[9, 9, 9, 10]]) == (11, 13)   # inserted right after line 9


# ---------------------------------------------------------------- 5. indentation repair (P6)

def test_repair_anchored_on_the_kept_line_not_the_first_line(ws):
    """fastapi_14458 r1 call 10: NEW's first line copied at 12 spaces, the others at 10; v17b shifted them to 12 and
    put the new check after `return True` (dead code, the run failed)."""
    a = line_of(MODELS, '            return True')
    new = ('            return True\n          if inspect.isclass(self.call):\n              return False\n'
           '          dunder_call = getattr(self.call, "__call__", None)')
    code, out = run('edit.py', ws, target='pkg/models.py', start=a, end=a + 1, new=new)
    assert code == 0, out
    assert out.splitlines()[0] == (f'EDITED pkg/models.py (lines {a}-{a + 1}) (indentation repaired: -2 to match the '
                                   'lines NEW keeps)'), out
    text = src(ws, 'pkg/models.py')
    assert '            return True\n        if inspect.isclass(self.call):\n            return False\n' \
           '        dunder_call' in text
    assert guards.unreachable(MODELS, text) == []


def test_repair_into_dead_code_is_refused(ws):
    a = line_of(CALC, '    return result')
    code, out = run('edit.py', ws, target='pkg/calc.py', start=a, end=a, new='    return result\n  print(result)')
    assert code == 1 and src(ws, 'pkg/calc.py') == CALC, out
    lines = body(out).splitlines()
    assert lines[0].startswith('NOT APPLIED: NEW is not indented like the lines it replaces, and the only '
                               're-indentation that parses (') and f'puts lines {a + 1}-{a + 1} after the `return` at ' \
        f'line {a}, where they never run; nothing was changed.' in lines[0], out
    assert f'>>{a:5d}      return result' in lines
    assert 'Write NEW with the file\'s indentation: line 5 starts at 4 spaces.' in lines
    assert lines[-1] == f'CLOSEST: pkg/calc.py {a} {a}'


def test_new_as_written_with_dead_code_gets_a_warning(ws):
    a = line_of(CALC, '    return result')
    code, out = run('edit.py', ws, target='pkg/calc.py', start=a, end=a, new='    return result\n    print(result)')
    assert code == 0 and f'WARNING: lines {a + 1}-{a + 1} come after the `return` at line {a} in the same block, so ' \
        'they never run' in out, out


def test_unreachable_unit():
    before = 'def f(x):\n    if x:\n        return 1\n    return 2\n'
    after = 'def f(x):\n    if x:\n        return 1\n        y = 2\n    return 2\n'
    assert guards.unreachable(before, after) == [(4, 4, 3, 'return')]
    old_dead = 'def g():\n    return 1\n    x = 2\n'
    assert guards.unreachable(old_dead, old_dead.replace('return 1', 'return 1  # same')) == [(3, 3, 2, 'return')]
    assert guards.unreachable(old_dead, old_dead + '\n\ndef h():\n    return 3\n') == []  # pre-existing dead code
    assert guards.unreachable('x = 1\n', 'def (:\n') == []


# ---------------------------------------------------------------- 6. closest region (P5)

def test_old_not_found_with_nothing_similar_prints_no_region(ws):
    code, out = run('edit.py', ws, target='pkg/calc.py', old='completely unrelated text that is nowhere',
                    new='x')
    assert code == 1 and 'CLOSEST' not in out and 'closest region' not in out, out
    assert body(out).splitlines()[1].startswith('no similar lines in pkg/calc.py either (best similarity 0.')


def test_old_not_found_with_a_near_match_prints_the_region(ws):
    code, out = run('edit.py', ws, target='pkg/calc.py', old='    for value in values:\n        result += value',
                    new='x')
    assert code == 1 and 'closest region (similarity 0.' in out and 'CLOSEST: pkg/calc.py 3 4' in out, out


# ---------------------------------------------------------------- 7. message diet

def test_ambiguous_sites_once(ws):
    code, out = run('edit.py', ws, flow_on=False, target='pkg/models.py',
                    old='        dunder_call = getattr(self.call, "__call__", None)', new='        dunder_call = None')
    assert code == 1 and len(out) <= 600, out  # v20: + the ready multi-edit line for all sites (one-line OLD/NEW)
    assert out.splitlines()[-1].startswith('to change all of them in one call: edit.py  target=pkg/models.py  start=')
    assert out.splitlines()[1:-1] == [f'  {n}-{n} Dependant.{f}' for n, f in (
        (line_of(MODELS, '        dunder_call = getattr(self.call, "__call__", None)'), 'is_gen_callable'),
        (line_of(MODELS, '        dunder_call = getattr(self.call, "__call__", None)', 14), 'is_async_gen_callable'),
        (line_of(MODELS, '        dunder_call = getattr(self.call, "__call__", None)', 22), 'is_coroutine_callable'))]


def test_usage_error_has_one_example(ws):
    code, out = run('edit.py', ws, flow_on=False, target='pkg/calc.py', old='x')
    assert code == 2 and out == ('USAGE ERROR: edit.py needs the key new (valid keys: target, old, new, start, end). E.g. '
                                 'edit.py  target=pkg/calc.py  old=x  new=<replacement text>\n'), out
    code, out = run('run.py', ws, flow_on=False, bogus='1')
    assert code == 2 and len([c for c in calls(out) if c[1]]) == 1 and '\nusage:' not in out, out


def test_syntax_error_prints_new_numbered_with_the_error_marked(ws):
    a = line_of(CALC, '        result += v')
    code, out = run('edit.py', ws, flow_on=False, target='pkg/calc.py', start=a, end=a,
                    new='        result += (v\n        result *= 2')
    assert code == 1 and src(ws, 'pkg/calc.py') == CALC, out
    lines = out.splitlines()
    assert lines[0].startswith(f'NOT APPLIED: the result would not parse: SyntaxError line {a}'), out
    assert f'>>{a:5d}          result += (v' in lines and f'  {a + 1:5d}          result *= 2' in lines, out
    assert f'  {a - 1:5d}      for v in values:' in lines  # one line of context
    assert 'indentation: NEW starts at 8 spaces like the replaced lines.' in lines
    assert lines[-1] == f'CLOSEST: pkg/calc.py {a} {a}'


def test_syntax_error_names_the_expected_indentation(ws):
    a = line_of(CALC, '    result = 0')
    code, out = run('edit.py', ws, flow_on=False, target='pkg/calc.py', start=a, end=a, new='      result = 0\n  x = (')
    assert code == 1 and 'expected indentation: the replaced lines start at 4 spaces; NEW starts at 6.' in out, out


# ---------------------------------------------------------------- 8. run.py

def test_test_output_is_short(ws):
    code, out = run('run.py', ws, flow_on=False, test='tests/test_calc.py')
    assert code == 0 and 'exit code: 1' in out, out
    assert not any(set(l.strip()) <= set('=_!-') and len(l.strip()) > 3 for l in out.splitlines()), out
    assert out == ('--- test_mean\ntests/test_calc.py:6: in test_mean\n    assert mean([1, 2]) == 2\n'
                   'E   assert 1.5 == 2\nE    +  where 1.5 = mean([1, 2])\n'
                   'FAILED tests/test_calc.py::test_mean - assert 1.5 == 2\n') + out[out.index('1 failed in '):], out
    assert 'FAILED tests/test_calc.py::test_mean' in out
    assert len(out) < 700, (len(out), out)


def test_test_output_collapses_library_frames(ws):
    code, out = run('run.py', ws, flow_on=False, test='tests/test_calc.py', k='json')
    assert 'library frame' in out and 'json/decoder.py' not in out and 'tests/test_calc.py:10: in test_json' in out, out
    assert 'E   json.decoder.JSONDecodeError' in out, out


def test_clean_pytest_unit():
    lines = ['F                                                                        [100%]',
             '=================================== FAILURES ===================================',
             '_________________________________ test_tty __________________________________',
             'tests/test_console.py:12: in test_tty',
             '    assert console.is_terminal',
             '/usr/lib/python3.13/site-packages/_pytest/x.py:5: in wrap',
             '    return f()',
             '/usr/lib/python3.13/inspect.py:7: in g',
             '    return h()',
             'E   AssertionError',
             '=============================== warnings summary ===============================',
             'rich/console.py:1: DeprecationWarning: x',
             '  warnings.warn(x)',
             '=========================== short test summary info ============================',
             'FAILED tests/test_console.py::test_tty - AssertionError',
             '!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!',
             '========================= 1 failed, 3 passed in 0.12s =========================']
    assert run_py.clean_pytest(lines) == [
        '--- test_tty', 'tests/test_console.py:12: in test_tty', '    assert console.is_terminal',
        '  ... (2 library frames)', 'E   AssertionError', 'FAILED tests/test_console.py::test_tty - AssertionError',
        '1 failed, 3 passed in 0.12s']


def test_snippet_that_prints_nothing_shows_its_last_expression(ws):
    code, out = run('run.py', ws, flow_on=False, snippet='from pkg.calc import mean\nmean([1, 2])')
    assert out == 'nothing printed; the last expression is: 1.5\nexit code: 0\n', out
    code, out = run('run.py', ws, flow_on=False, snippet='x = [1, 2]\nprint(len(x))\nx')
    assert out == '2\nexit code: 0\n', out  # it printed something: no repr
    code, out = run('run.py', ws, flow_on=False, snippet='x = 1\nx / 0')
    assert 'line 2, in <module>' in out and '    x / 0' in out and '__snippet' not in out, out
    assert 'ZeroDivisionError' in out
    carets = [l for l in out.splitlines() if set(l.strip()) <= set('~^') and l.strip()]
    assert all(l.index('~') <= len('    x / 0') for l in carets), out  # the caret line follows the source line


def test_with_last_expr_unit():
    assert run_py.with_last_expr('print(1)') == 'print(1)'
    assert run_py.with_last_expr('x = 1') == 'x = 1'
    assert run_py.with_last_expr('"doc"') == '"doc"'
    assert run_py.with_last_expr('def (') == 'def ('
    out = run_py.with_last_expr('a = 1; a')
    assert out.splitlines()[0] == 'a = 1; __snippet_last__ = a' and len(out.splitlines()) == 2


def test_snippet_with_the_same_output_as_an_earlier_one(ws):
    loop = 'for i in range(30):\n    print("line", i)\nraise ValueError("boom")'
    code, out1 = run('run.py', ws, snippet=loop)
    assert 'ValueError: boom' in out1
    code, out2 = run('run.py', ws, snippet=loop.replace('range(30)', 'range(0, 30)'))
    first = body(out2).splitlines()
    assert first[0] == 'same output as call 1; its last line: ValueError: boom', out2
    assert first[1] == 'exit code: 1'
    code, out3 = run('run.py', ws, snippet='print("short")')
    code, out4 = run('run.py', ws, snippet='print("sho" + "rt")')
    assert body(out4).startswith('short\nexit code: 0'), out4  # short outputs are printed again


def test_same_output_after_an_edit_names_the_edit(ws):
    snippet = 'for i in range(30):\n    print("value", i)'
    run('run.py', ws, snippet=snippet)
    code, out = run('edit.py', ws, target='pkg/calc.py', old='    result = 0', new='    result = 0.0')
    assert code == 0, out
    code, out = run('run.py', ws, snippet=snippet + '\n')
    assert body(out).startswith('same output as call 1 (before your edit at call 2); its last line: value 29'), out
