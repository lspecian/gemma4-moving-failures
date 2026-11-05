"""Tests for workstream C: create.py, edit.py's cascade breakers, check_patch.py with created files, and the
argument-misuse repairs shared by show/search/outline/run/edit/create (agent/skills/repo-tools/scripts).

Every script runs as a subprocess, the way run_skill_script runs it, with WORKSPACE pointing at a small temp git
repository and REPO_TOOLS_DIR at a temp state dir. Many cases replay argument shapes seen in real traces (E020).
Run with `scripts/test_skills.sh` (sandbox image, Python 3.13).
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap
import time

import pytest

from notation import assert_notation, calls

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'agent/skills/repo-tools/scripts'
MAX_OUTPUT_LINES = 40

INVOICE = textwrap.dedent('''\
    """Payment helpers."""

    TEMPLATE = """\\
    <html>
    <body>{body}</body>
    """


    class Invoice:
        def __init__(self, amount):
            self.amount = amount

        def total(self, tax=0.1):
            if self.amount == None:
                return 0
            return self.amount * (1 + tax)

        def describe(self):
            label = "invoice"
            return f"{label}: {self.amount}"


    def parse_header(value):
        return value.split(",")
    ''')

ROUTING = textwrap.dedent('''\
    def include_router(router, prefix=""):
        """Include a router."""
        if prefix:
            assert prefix.startswith("/"), "A path prefix must start with '/'"
            assert not prefix.endswith("/"), (
                "A path prefix must not end with '/'"
            )
        else:
            for r in router.routes:
                check(r)
        return router


    def check(r):
        return r
    ''')

LONG = ''.join(f'def f{i}(x):\n    return x + {i}\n\n\n' for i in range(60))


def run(script, *args, ws, state=None, env_extra=None):
    env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(state or ws.parent / 'state'), REPO_TOOLS_FLOW='off',
               PYTHONDONTWRITEBYTECODE='1')
    env.update(env_extra or {})
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True, env=env,
                       timeout=120)
    assert not r.stderr.strip() or 'Traceback' not in r.stderr, r.stderr
    return r.returncode, r.stdout


def git(ws, *args):
    return subprocess.run(['git', '-C', str(ws), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'pay').mkdir()
    (root / 'pay/__init__.py').write_text('')
    (root / 'pay/invoice.py').write_text(INVOICE)
    (root / 'pay/routing.py').write_text(ROUTING)
    (root / 'pay/long.py').write_text(LONG)
    (root / 'docs_src').mkdir()
    (root / 'docs_src/first').mkdir()
    (root / 'docs_src/first/__init__.py').write_text('')
    (root / 'docs_src/first/tutorial001.py').write_text('x = 1\n')
    (root / 'tests').mkdir()
    (root / 'tests/test_invoice.py').write_text('from pay.invoice import Invoice\n\n\ndef test_total():\n'
                                                '    assert Invoice(10).total() == 11\n')
    (root / 'CHANGELOG.md').write_text(''.join(f'- change {i}\n' for i in range(150)))
    (root / 'notes.md').write_text('# notes\nshort file\n')
    (root / '.gitignore').write_text('*.generated.py\n')
    git(root, 'init', '-q')
    git(root, '-c', 'user.email=t@t', '-c', 'user.name=t', 'add', '-A')
    git(root, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'baseline')
    return root


def src(ws, rel='pay/invoice.py'):
    return (ws / rel).read_text()


def created_txt(ws):
    p = ws.parent / 'state/created.txt'
    return p.read_text().split() if p.exists() else []


def assert_budget(out):
    assert len(out.splitlines()) <= MAX_OUTPUT_LINES, out


def correction(out):
    """A USAGE ERROR without the script's usage text that follows it."""
    return out.split('\nusage:')[0]


# ================================ create.py ================================

def test_create_python_file_records_it_and_summarises(ws):
    content = 'from fastapi import FastAPI\n\napp = FastAPI()\n\n\ndef main():\n    return app\n'
    code, out = run('create.py', 'docs_src/frontend/tutorial001_py310.py', content, ws=ws)
    assert code == 0, out
    assert out.startswith('CREATED docs_src/frontend/tutorial001_py310.py (7 lines, python syntax OK)')
    assert 'defines: ' in out and 'def main' in out and 'app L3' in out
    assert src(ws, 'docs_src/frontend/tutorial001_py310.py') == content
    assert created_txt(ws) == ['docs_src/frontend/tutorial001_py310.py']
    assert 'new directories: docs_src/frontend' in out
    assert '__init__.py' in out  # sibling packages have one: hint
    assert_budget(out)


def test_create_decodes_literal_escapes_when_raw_does_not_parse(ws):
    code, out = run('create.py', 'pay/tax.py', 'RATE = 0.2\\n\\n\\ndef tax(x):\\n    return x * RATE\\n', ws=ws)
    assert code == 0 and 'decoded escapes' in out, out
    assert src(ws, 'pay/tax.py') == 'RATE = 0.2\n\n\ndef tax(x):\n    return x * RATE\n'


def test_create_keeps_real_escapes_inside_strings(ws):
    content = 'SEP = "\\n"\n\n\ndef join(xs):\n    return SEP.join(xs)\n'
    code, out = run('create.py', 'pay/sep.py', content, ws=ws)
    assert code == 0 and 'decoded' not in out
    assert src(ws, 'pay/sep.py') == content  # the \n escape inside the string literal is preserved


def test_create_decodes_escaped_quotes_with_real_newlines(ws):
    content = 'NAME = \\"pay\\"\n\n\ndef name():\n    return NAME\n'
    code, out = run('create.py', 'pay/name.py', content, ws=ws)
    assert code == 0, out
    assert src(ws, 'pay/name.py').startswith('NAME = "pay"\n')


def test_create_strips_code_fence_and_dedents(ws):
    code, out = run('create.py', 'pay/fenced.py', '```python\nX = 1\n```', ws=ws)
    assert code == 0 and 'code fence removed' in out, out
    assert src(ws, 'pay/fenced.py') == 'X = 1\n'
    code, out = run('create.py', 'pay/indented.py', '    def f():\n        return 1\n', ws=ws)
    assert code == 0 and 'dedented' in out
    assert src(ws, 'pay/indented.py') == 'def f():\n    return 1\n'


def test_create_refuses_syntax_error_and_writes_nothing(ws):
    code, out = run('create.py', 'pay/bad.py', 'def f(:\n    return 1\n', ws=ws)
    assert code == 1 and out.startswith('NOT CREATED') and 'SyntaxError line 1' in out
    assert not (ws / 'pay/bad.py').exists() and created_txt(ws) == []


def test_create_refuses_existing_tracked_file(ws):
    code, out = run('create.py', 'pay/invoice.py', 'X = 1\n', ws=ws)
    assert code == 1 and out.startswith('REFUSED') and 'edit.py' in out
    assert src(ws) == INVOICE


def test_create_replaces_its_own_new_file(ws):
    run('create.py', 'pay/new.py', 'X = 1\n', ws=ws)
    code, out = run('create.py', 'pay/new.py', 'X = 2\n', ws=ws)
    assert code == 0 and out.startswith('REPLACED pay/new.py')
    assert src(ws, 'pay/new.py') == 'X = 2\n' and created_txt(ws) == ['pay/new.py']


@pytest.mark.parametrize('rel', ['tests/test_new.py', 'pkg/tests/helpers.py', 'pay/test_x.py', 'pay/x_test.py',
                                 'conftest.py', 'pyproject.toml', 'setup.cfg'])
def test_create_refuses_tests_and_config(ws, rel):
    code, out = run('create.py', rel, 'X = 1\n', ws=ws)
    assert code == 1 and out.startswith('REFUSED') and 'test' in out
    assert not (ws / rel).exists()


@pytest.mark.parametrize('rel', ['out.log', 'fix.patch'])
def test_create_refuses_scratch_files(ws, rel):
    code, out = run('create.py', rel, 'X = 1\n', ws=ws)
    assert code == 1 and 'scratch' in out and 'run.py snippet' in out
    assert not (ws / rel).exists()


@pytest.mark.parametrize('rel', ['repro.py', 'reproduce_issue.py', 'scratch.py', 'tmp_check.py', 'pay/temp1.py'])
def test_create_runs_scratch_python_as_a_snippet(ws, rel):
    """v18 (tests/test_v18_edit_run.py): a scratch .py file is a reproduction script: its content runs as run.py
    snippet and nothing is created."""
    code, out = run('create.py', rel, 'X = 1\nprint(X + 1)\n', ws=ws)
    assert code == 0 and 'scratch' in out and 'run.py snippet' in out and '2\nexit code: 0' in out, out
    assert not (ws / rel).exists()


def test_create_allows_names_that_only_start_like_scratch(ws):
    for rel in ('pay/templates.py', 'pay/debug.py', 'pay/temperature.py'):
        code, out = run('create.py', rel, 'X = 1\n', ws=ws)
        assert code == 0, (rel, out)


@pytest.mark.parametrize('rel', ['/etc/evil.py', '../evil.py', 'pay/../../evil.py', '.git/hooks/x.py', 'pay/'])
def test_create_refuses_paths_outside_the_repo(ws, rel):
    code, out = run('create.py', rel, 'X = 1\n', ws=ws)
    assert code == 1 and out.startswith('REFUSED')
    assert not (ws.parent / 'evil.py').exists()


def test_create_accepts_workspace_prefix(ws):
    code, out = run('create.py', '/workspace/pay/extra.py', 'X = 1\n', ws=ws)
    assert code == 0 and (ws / 'pay/extra.py').exists() and created_txt(ws) == ['pay/extra.py']


def test_create_large_file_in_one_call(ws):
    content = '"""Generated."""\n\n' + ''.join(f'def handler_{i}(request):\n    """Handle {i}."""\n'
                                             f'    return {{"id": {i}, "text": "line\\n{i}"}}\n\n\n'
                                             for i in range(150))
    assert content.count('\n') > 700
    t = time.time()
    code, out = run('create.py', 'pay/handlers.py', content, ws=ws)
    assert code == 0 and time.time() - t < 10
    assert src(ws, 'pay/handlers.py') == content
    assert '(+' in out and 'more)' in out  # the defines list is capped
    assert_budget(out)


def test_create_large_file_with_escaped_newlines(ws):
    lines = [f'VALUE_{i} = {i}' for i in range(400)]
    code, out = run('create.py', 'pay/values.py', '\\n'.join(lines), ws=ws)
    assert code == 0 and 'decoded escapes' in out
    assert src(ws, 'pay/values.py') == '\n'.join(lines) + '\n'


def test_create_non_python_files(ws):
    code, out = run('create.py', 'docs/en/docs/new.md', '# Title\\n\\nSome text with a \\\\n literal\\n', ws=ws)
    assert code == 0 and src(ws, 'docs/en/docs/new.md').startswith('# Title\n\nSome text')
    code, out = run('create.py', 'docs/raw.md', '# Title\nkeep \\n as typed\n', ws=ws)
    assert code == 0 and src(ws, 'docs/raw.md') == '# Title\nkeep \\n as typed\n'
    code, out = run('create.py', 'pay/data.json', '{\\"a\\": 1}', ws=ws)
    assert code == 0 and json.loads(src(ws, 'pay/data.json')) == {'a': 1}


def test_create_warns_when_path_is_gitignored(ws):
    code, out = run('create.py', 'pay/schema.generated.py', 'X = 1\n', ws=ws)
    assert code == 0 and 'WARNING' in out and 'gitignore' in out


def test_create_refuses_when_a_parent_is_a_file(ws):
    code, out = run('create.py', 'pay/invoice.py/sub.py', 'X = 1\n', ws=ws)
    assert code == 1 and 'is a file' in out


def test_create_notes_module_shadowing(ws):
    code, out = run('create.py', 'pay/invoice/extra.py', 'X = 1\n', ws=ws)
    assert code == 0 and 'pay/invoice.py also exists' in out


def test_create_usage_errors(ws):
    code, out = run('create.py', ws=ws)  # v18: one example, no usage block after it
    assert code == 2 and out.startswith('USAGE ERROR: create.py needs') and '\nusage:' not in out
    code, out = run('create.py', 'pay/x.py', ws=ws)
    assert code == 2 and 'CONTENT' in out and 'create.py  target=pay/x.py  content=' in out
    assert_notation(correction(out))
    code, out = run('create.py', 'pay/x.py', 'X = 1', 'Y = 2', ws=ws)
    assert code == 2 and 'ONE string' in out
    code, out = run('create.py', 'FILE', 'X = 1\n', ws=ws)
    assert code == 2 and 'placeholder' in out
    code, out = run('create.py', 'pay/x.py', '<file content>', ws=ws)
    assert code == 2 and 'placeholder' in out
    code, out = run('create.py', 'pay/x.py', '...', ws=ws)
    assert code == 2 and not (ws / 'pay/x.py').exists()


def test_create_repairs_packed_arguments(ws):
    code, out = run('create.py', json.dumps(['pay/a.py', 'A = 1\n']), ws=ws)
    assert code == 0 and src(ws, 'pay/a.py') == 'A = 1\n' and 'JSON' in out
    code, out = run('create.py', 'pay/b.py\nB = 2\n', ws=ws)
    assert code == 0 and src(ws, 'pay/b.py') == 'B = 2\n'
    code, out = run('create.py', 'pay/c.py\\nC = 3\\n', ws=ws)
    assert code == 0 and src(ws, 'pay/c.py') == 'C = 3\n'
    assert created_txt(ws) == ['pay/a.py', 'pay/b.py', 'pay/c.py']


def test_create_empty_init_file(ws):
    code, out = run('create.py', 'pay/sub/__init__.py', '', ws=ws)
    assert code == 0 and (ws / 'pay/sub/__init__.py').read_text() == ''


def test_create_then_check_patch_and_import(ws):
    run('create.py', 'pay/sse.py', 'from pay.invoice import Invoice\n\n\ndef stream():\n    return Invoice(1)\n', ws=ws)
    code, out = run('check_patch.py', ws=ws)
    assert code == 0 and 'READY' in out
    assert 'pay/sse.py (new file, 5 lines)' in out


# ================================ edit.py ================================

def test_edit_not_found_prints_lines_command_and_closest_line(ws):
    code, out = run('edit.py', 'pay/invoice.py', 'if self.amount == Nothing:\n    return 0', 'x', ws=ws)
    lines = out.strip().splitlines()
    assert code == 1 and 'NOT APPLIED' in out
    n = INVOICE.split('\n').index('        if self.amount == None:') + 1
    assert lines[-1] == f'CLOSEST: pay/invoice.py {n} {n + 1}'
    assert f'edit.py  target=pay/invoice.py  start={n}  end={n + 1}  new=' in lines[-2]
    assert_notation(lines[-2])
    assert src(ws) == INVOICE
    assert_budget(out)


def test_edit_syntax_error_reports_closest_region(ws):
    code, out = run('edit.py', 'pay/invoice.py', 'if self.amount == None:', 'if self.amount == None', ws=ws)
    n = INVOICE.split('\n').index('        if self.amount == None:') + 1
    assert code == 1 and 'would not parse' in out
    assert out.strip().splitlines()[-1] == f'CLOSEST: pay/invoice.py {n} {n}'


def test_edit_ambiguous_has_no_closest_line(ws):
    code, out = run('edit.py', 'pay/invoice.py', 'return', 'yield', ws=ws)
    assert code == 1 and 'CLOSEST' not in out


def test_edit_lines_mode_repairs_consistent_indent_offset(ws):
    # trace fastapi_14258: NEW copied from show.py output carries 2 extra spaces on every line
    n = ROUTING.split('\n').index('    if prefix:') + 1
    new = ('      if router is None:\\n          raise ValueError(\\"no router\\")\\n\\n      if prefix:')
    code, out = run('edit.py', '--lines', 'pay/routing.py', str(n), str(n), new, ws=ws)
    assert code == 0 and 'indentation repaired' in out, out
    assert '    if router is None:\n        raise ValueError("no router")\n\n    if prefix:\n' in src(ws, 'pay/routing.py')


def test_edit_lines_mode_repairs_unindented_first_line(ws):
    n = ROUTING.split('\n').index('    return router') + 1
    code, out = run('edit.py', '--lines', 'pay/routing.py', str(n), str(n),
                    'if router is None:\n        return None\n    return router', ws=ws)
    assert code == 0, out
    assert '    if router is None:\n        return None\n    return router\n' in src(ws, 'pay/routing.py')


def test_edit_flexible_repairs_body_offset(ws):
    # trace fastapi_14372: first line pasted at column 0, the body lines with 2 extra spaces
    code, out = run('edit.py', 'pay/invoice.py', 'class Invoice:\\n      def __init__(self, amount):',
                    'class Invoice:\\n      def __init__(self, amount=0):', ws=ws)
    assert code == 0 and 'indentation repaired' in out, out
    assert 'class Invoice:\n    def __init__(self, amount=0):\n' in src(ws)


def test_edit_indent_repair_never_used_for_non_python(ws):
    code, out = run('edit.py', '--lines', 'notes.md', '2', '2', '  indented text', ws=ws)
    assert code == 0 and src(ws, 'notes.md') == '# notes\n  indented text\n'


def test_edit_strips_line_number_prefixes_copied_from_show(ws):
    n = INVOICE.split('\n').index('        if self.amount == None:') + 1
    old = f'{n:5d}          if self.amount == None:\n{n + 1:5d}              return 0'
    new = f'{n:5d}          if self.amount is None:\n{n + 1:5d}              return 0'
    code, out = run('edit.py', 'pay/invoice.py', old, new, ws=ws)
    assert code == 0 and 'line numbers removed' in out, out
    assert '        if self.amount is None:\n            return 0\n' in src(ws)


def test_edit_near_identical_block_applies(ws):
    # trace fastapi_14258: the model dropped a closing quote while copying a multi-line block
    old = ('    if prefix:\n        assert prefix.startswith("/"), "A path prefix must start with \'/\'\n'
           '        assert not prefix.endswith("/"), (\n            "A path prefix must not end with \'/\'"\n'
           '        )')
    new = ('    if prefix:\n        assert prefix.startswith("/"), "A path prefix must start with \'/\'"\n'
           '        assert not prefix.endswith("/"), (\n            "A path prefix must not end with \'/\'"\n'
           '        )\n        prefix = prefix.rstrip()')
    code, out = run('edit.py', 'pay/routing.py', old, new, ws=ws)
    assert code == 0 and 'near-identical block' in out, out
    assert '        prefix = prefix.rstrip()\n    else:' in src(ws, 'pay/routing.py')


def test_edit_near_identical_needs_a_unique_block(ws):
    block = 'def a(x):\n    y = x + 1\n    z = y * 2\n    return z\n\n\n'
    (ws / 'pay/twice.py').write_text(block + block.replace('def a', 'def b'))
    code, out = run('edit.py', 'pay/twice.py', 'def c(x):\n    y = x + 1\n    z = y * 2\n    return z',
                    'def c(x):\n    return 0', ws=ws)
    assert code == 1 and 'NOT APPLIED' in out and 'CLOSEST' in out


def test_edit_near_identical_rejects_missing_lines(ws):
    old = '    if prefix:\n        assert not prefix.endswith("/"), (\n            "A path prefix must not end with \'/\'"\n        )'
    code, out = run('edit.py', 'pay/routing.py', old, '    pass', ws=ws)
    assert code == 1 and 'NOT APPLIED' in out and src(ws, 'pay/routing.py') == ROUTING


def test_edit_identical_old_and_new(ws):
    code, out = run('edit.py', 'pay/invoice.py', 'return 0', 'return 0', ws=ws)
    assert code == 1 and 'identical' in out


@pytest.mark.parametrize('marker', ['        # ... existing code ...', '        # ...', '        # (rest of the method)',
                                    '        # ... rest unchanged', '        // ... other code'])
def test_edit_refuses_elided_new(ws, marker):
    old = '        if self.amount == None:\n            return 0\n        return self.amount * (1 + tax)'
    new = f'        if self.amount is None:\n{marker}'
    code, out = run('edit.py', 'pay/invoice.py', old, new, ws=ws)
    assert code == 2 and 'elided' in out
    assert src(ws) == INVOICE


def test_edit_allows_ordinary_comments(ws):
    code, out = run('edit.py', 'pay/invoice.py', '            return 0',
                    '            # keep the existing order; the rest of the checks run later\n            return 0', ws=ws)
    assert code == 0, out


def test_edit_placeholders(ws):
    code, out = run('edit.py', 'pay/invoice.py', 'if self.amount == None:', '<new text>', ws=ws)
    assert code == 2 and 'placeholder' in out and src(ws) == INVOICE
    code, out = run('edit.py', 'pay/invoice.py', '<old text>', 'x = 1', ws=ws)
    assert code == 2 and 'placeholder' in out
    code, out = run('edit.py', 'FILE', 'a', 'b', ws=ws)
    assert code == 2 and 'placeholder' in out
    code, out = run('edit.py', '--lines', 'pay/invoice.py', 'START', 'END', 'x = 1', ws=ws)
    assert code == 2 and 'line numbers' in out and src(ws) == INVOICE


def test_edit_line_placeholders_without_flag(ws):
    code, out = run('edit.py', 'pay/invoice.py', 'START', 'END', 'x = 1', ws=ws)
    assert code == 2 and 'placeholders' in out and 'edit.py  target=pay/invoice.py  start=120  end=122' in out
    assert_notation(correction(out))


def test_edit_lines_mode_repairs_body_offset_under_correct_first_line(ws):
    n = ROUTING.split('\n').index('    return router') + 1
    code, out = run('edit.py', '--lines', 'pay/routing.py', str(n), str(n),
                    '    if router is None:\n          return None\n      return router', ws=ws)
    assert code == 0 and 'indentation repaired' in out, out
    assert '    if router is None:\n        return None\n    return router\n' in src(ws, 'pay/routing.py')


def test_script_name_is_only_dropped_when_it_is_not_a_repo_file(ws):
    (ws / 'scripts').mkdir()
    (ws / 'scripts/show.py').write_text('def main():\n    return 1\n')
    code, out = run('show.py', 'scripts/show.py', 'main', ws=ws)
    assert code == 0 and 'def main' in out and 'dropped' not in out
    code, out = run('show.py', 'show.py', 'pay/invoice.py', 'Invoice', ws=ws)
    assert code == 0 and 'dropped the script name' in out and 'class Invoice' in out


def test_edit_missing_file_points_to_create(ws):
    code, out = run('edit.py', 'pay/nope.py', 'a', 'b', ws=ws)
    assert code == 1 and 'not found' in out and 'create.py  target=pay/nope.py  content=' in out
    assert_notation(out)


def test_edit_resolves_unique_path_suffix(ws):
    code, out = run('edit.py', 'invoice.py', 'if self.amount == None:', 'if self.amount is None:', ws=ws)
    assert code == 0 and 'using pay/invoice.py' in out


def test_edit_lines_without_flag(ws):
    # trace fastapi_14986 / rich_3043: ["FILE", "START", "END", "NEW"]
    n = INVOICE.split('\n').index('        if self.amount == None:') + 1
    code, out = run('edit.py', 'pay/invoice.py', str(n), str(n), '        if self.amount is None:', ws=ws)
    assert code == 0 and 'read FILE START END NEW as --lines' in out
    assert '        if self.amount is None:\n' in src(ws)


def test_edit_lines_packed_in_one_string(ws):
    # trace fastapi_14372: ["--lines fastapi/params.py 765 774\n<new text>"]
    n = INVOICE.split('\n').index('        if self.amount == None:') + 1
    code, out = run('edit.py', f'--lines pay/invoice.py {n} {n + 1}\n        if self.amount is None:\n'
                               '            return 0.0', ws=ws)
    assert code == 0, out
    assert '        if self.amount is None:\n            return 0.0\n' in src(ws)


def test_edit_lines_new_split_over_args(ws):
    # trace fastapi_14372: ["--lines", FILE, "765", "766", "  @dataclass(frozen=True)", "  class Depends:"]
    code, out = run('edit.py', '--lines', 'pay/invoice.py', '9', '10', 'class Invoice:', '    def __init__(self):',
                    ws=ws)
    assert code == 2 and 'ONE string (join its lines with \\n)' in out
    # the example joins the given lines into one value (whitespace is collapsed in the notation)
    example = [a for s, a in assert_notation(correction(out)) if s == 'edit.py' and a.get('start') == '9']
    assert example == [{'target': 'pay/invoice.py', 'start': '9', 'end': '10',
                        'new': 'class Invoice: def __init__(self):'}], out
    assert src(ws) == INVOICE


def test_edit_lines_accepts_decorated_numbers(ws):
    n = INVOICE.split('\n').index('        if self.amount == None:') + 1
    code, out = run('edit.py', '--lines', 'pay/invoice.py', f'L{n}', f'`{n}`', '        if self.amount is None:',
                    ws=ws)
    assert code == 0, out


def test_edit_lines_trailing_newline_adds_no_blank_line(ws):
    n = INVOICE.split('\n').index('        if self.amount == None:') + 1
    code, out = run('edit.py', '--lines', 'pay/invoice.py', str(n), str(n), '        if self.amount is None:\\n',
                    ws=ws)
    assert code == 0
    assert '        if self.amount is None:\n            return 0\n' in src(ws)


def test_edit_lines_warns_on_duplicated_boundary_line(ws):
    n = ROUTING.split('\n').index('    if prefix:') + 1
    code, out = run('edit.py', '--lines', 'pay/routing.py', str(n), str(n),
                    '    if not router:\n        return None\n    if prefix:\n'
                    '        assert prefix.startswith("/"), "A path prefix must start with \'/\'"', ws=ws)
    assert code == 0 and 'WARNING' in out and f'line {n + 1}' in out


def test_edit_json_packed_arguments(ws):
    # trace requests_7315: 'FILE", "OLD", "NEW' in one string
    packed = 'pay/invoice.py", "if self.amount == None:", "if self.amount is None:'
    code, out = run('edit.py', packed, ws=ws)
    assert code == 0 and 'JSON' in out and 'if self.amount is None:' in src(ws)


def test_edit_arg_count_errors(ws):
    code, out = run('edit.py', 'pay/invoice.py', 'if self.amount == None:', ws=ws)
    assert code == 2 and 'USAGE ERROR' in out and 'FILE OLD NEW' in out and '\nusage:' not in out
    code, out = run('edit.py', 'pay/invoice.py', 'a', 'b', 'c', 'd', ws=ws)
    assert code == 2 and 'exactly FILE OLD NEW' in out
    code, out = run('edit.py', '--lines', 'pay/invoice.py', '12', '10', 'x', ws=ws)
    assert code == 2 and 'before START' in out


def test_edit_directory_path(ws):
    code, out = run('edit.py', 'pay', 'a', 'b', ws=ws)
    assert code == 1 and 'directory' in out


def test_edit_cascade_then_lines_command_succeeds(ws):
    """A failed edit's --lines suggestion, used as printed, applies."""
    code, out = run('edit.py', 'pay/invoice.py', 'if self.amount == Nothing:\n    return 0',
                    'if self.amount is None:\n    return 0', ws=ws)
    _, rel, start, end = out.strip().splitlines()[-1].split()
    code, out = run('edit.py', '--lines', rel, start, end,
                    '        if self.amount is None:\n            return 0', ws=ws)
    assert code == 0 and '        if self.amount is None:\n            return 0\n' in src(ws)


# ================================ check_patch.py ================================

def test_check_patch_keeps_created_files_and_cleans_scratch(ws):
    run('edit.py', 'pay/invoice.py', 'if self.amount == None:', 'if self.amount is None:', ws=ws)
    run('create.py', 'pay/sse.py', 'def stream():\n    return 1\n', ws=ws)
    (ws / 'repro_issue.py').write_text('print(1)\n')
    (ws / 'debug_output.log').write_text('x\n')
    (ws / 'pay/debug_x.py').write_text('x = 1\n')
    code, out = run('check_patch.py', ws=ws)
    assert code == 1 and 'pay/sse.py (new file' in out
    for rel in ('repro_issue.py', 'debug_output.log', 'pay/debug_x.py'):
        assert f'{rel}: ' in out and 'scratch' in out
    code, out = run('check_patch.py', '--clean', ws=ws)
    assert code == 0 and 'READY' in out
    assert (ws / 'pay/sse.py').exists()
    assert not (ws / 'repro_issue.py').exists() and not (ws / 'debug_output.log').exists()
    assert not (ws / 'pay/debug_x.py').exists()


def test_check_patch_clean_never_deletes_created_files(ws):
    for rel in ('docs_src/frontend/tutorial001_py310.py', 'scripts/prepare_release.py', 'pay/debug.py'):
        code, out = run('create.py', rel, 'X = 1\n', ws=ws)
        assert code == 0, out
    (ws / 'tests/test_new.py').write_text('def test_x():\n    pass\n')
    code, out = run('check_patch.py', '--clean', ws=ws)
    assert code == 0, out
    for rel in ('docs_src/frontend/tutorial001_py310.py', 'scripts/prepare_release.py', 'pay/debug.py'):
        assert (ws / rel).exists(), rel
    assert not (ws / 'tests/test_new.py').exists()


def test_check_patch_flags_new_test_files_and_root_scripts(ws):
    run('edit.py', 'pay/invoice.py', 'if self.amount == None:', 'if self.amount is None:', ws=ws)
    (ws / 'tests/test_new.py').write_text('def test_x():\n    pass\n')
    (ws / 'check_fix.py').write_text('print(1)\n')
    (ws / 'pay/notes.txt').write_text('x\n')
    code, out = run('check_patch.py', ws=ws)
    assert code == 1
    assert 'tests/test_new.py: new test/config file' in out
    assert 'check_fix.py: new .py file at the repository root' in out
    assert 'pay/notes.txt: new non-Python file' in out


def test_check_patch_accepts_package_module_written_without_create(ws):
    (ws / 'pay/helpers.py').write_text('def h():\n    return 1\n')
    code, out = run('check_patch.py', ws=ws)
    assert code == 0 and 'pay/helpers.py (new file, 2 lines; not made with create.py)' in out


def test_check_patch_compiles_and_imports_created_files(ws):
    run('create.py', 'pay/sse.py', 'import does_not_exist_anywhere\n', ws=ws)
    code, out = run('check_patch.py', ws=ws)
    assert code == 1 and 'import pay.sse fails' in out
    (ws / 'pay/sse.py').write_text('def broken(:\n')
    code, out = run('check_patch.py', ws=ws)
    assert code == 1 and 'pay/sse.py: does not compile' in out


def test_check_patch_scripts_dir_is_compile_only(ws):
    run('create.py', 'scripts/prepare_release.py', 'import typer_not_installed\n\n\ndef main():\n    pass\n', ws=ws)
    code, out = run('check_patch.py', ws=ws)
    assert code == 0 and 'READY' in out


def test_check_patch_test_edit_still_flagged(ws):
    run('create.py', 'pay/sse.py', 'X = 1\n', ws=ws)
    (ws / 'tests/test_invoice.py').write_text('# changed\n')
    code, out = run('check_patch.py', ws=ws)
    assert code == 1 and 'tests/test_invoice.py: test/config file' in out
    code, out = run('check_patch.py', '--clean', ws=ws)
    assert code == 0 and (ws / 'pay/sse.py').exists()
    assert src(ws, 'tests/test_invoice.py').startswith('from pay.invoice')


# ================================ show.py ================================

def test_show_start_end_placeholders_on_long_file_give_outline(ws):
    code, out = run('show.py', 'pay/long.py', 'START', 'END', ws=ws)
    assert code == 2 and 'placeholders' in out and 'show.py  target=pay/long.py  start=1  end=100' in out
    assert 'L1-2  def f0' in out and 'more; outline.py pay/long.py' in out
    assert_budget(out)


def test_show_start_end_placeholders_on_short_file_show_it(ws):
    code, out = run('show.py', 'pay/invoice.py', 'START', 'END', ws=ws)
    assert code == 0 and 'NOTE: the whole file (it is short)' in out and 'class Invoice:' in out  # v18 note


def test_show_one_placeholder_with_a_number(ws):
    code, out = run('show.py', 'pay/long.py', 'START', '20', ws=ws)
    assert code == 0 and 'pay/long.py L1-20 of' in out
    code, out = run('show.py', 'pay/long.py', '200', 'END', ws=ws)
    assert code == 0 and 'pay/long.py L200-240 of 240' in out


def test_show_file_only(ws):
    code, out = run('show.py', 'pay/long.py', ws=ws)  # v12: its outline and a ready call, not a usage error
    assert code == 0 and 'NOTE: pay/long.py has 240 lines; its outline:' in out and 'def f1' in out  # v18 note
    assert 'show.py  target=pay/long.py  symbol=f0' in out
    code, out = run('show.py', 'notes.md', ws=ws)
    assert code == 0 and 'short file' in out
    code, out = run('show.py', 'CHANGELOG.md', ws=ws)
    assert code == 0 and 'CHANGELOG.md L1-100 of 150' in out


def test_show_json_packed_and_script_name(ws):
    code, out = run('show.py', '["pay/invoice.py", "Invoice.total"]', ws=ws)
    assert code == 0 and 'Invoice.total' in out and 'JSON' in out
    code, out = run('show.py', 'show.py "pay/invoice.py" "1" "3"', ws=ws)
    assert code == 0 and 'pay/invoice.py L1-3' in out


def test_show_packed_single_string_forms(ws):
    for arg in ('pay/invoice.py Invoice.total', 'pay/invoice.py:Invoice.total', 'pay/invoice.py::Invoice.total'):
        code, out = run('show.py', arg, ws=ws)
        assert code == 0 and 'if self.amount == None:' in out, arg
    code, out = run('show.py', 'pay/invoice.py 1 3', ws=ws)
    assert code == 0 and 'L1-3' in out


def test_show_number_forms(ws):
    code, out = run('show.py', 'pay/invoice.py', 'L1', '`3`', ws=ws)
    assert code == 0 and 'L1-3' in out
    code, out = run('show.py', 'pay/invoice.py', '1', '`', '`3`', ws=ws)  # trace rich_3777
    assert code == 0 and 'L1-3' in out
    code, out = run('show.py', 'pay/invoice.py', '1-3', ws=ws)
    assert code == 0 and 'L1-3' in out
    code, out = run('show.py', 'pay/invoice.py', '3', '1', ws=ws)
    assert code == 0 and 'L1-3' in out and 'swapped' in out


def test_show_single_line_shows_enclosing_definition(ws):
    n = INVOICE.split('\n').index('            return 0') + 1
    code, out = run('show.py', 'pay/invoice.py', str(n), ws=ws)
    assert code == 0 and 'def total' in out and 'def describe' not in out


def test_show_symbol_forms(ws):
    for sym in ('class Invoice', 'Invoice.total()', 'def total', 'pay.invoice.Invoice.total', 'Invoice::total'):
        code, out = run('show.py', 'pay/invoice.py', sym, ws=ws)
        assert code == 0 and 'Invoice' in out.splitlines()[0], (sym, out)
    code, out = run('show.py', 'pay/invoice.py', 'TEMPLATE', ws=ws)
    assert code == 0 and '<body>{body}</body>' in out


def test_show_missing_symbol_suggests_close_names(ws):
    code, out = run('show.py', 'pay/invoice.py', 'Invoice.totl', ws=ws)
    assert code == 1 and 'not found' in out and 'Invoice.total' in out


def test_show_edit_style_args(ws):
    # trace fastapi_14258: show.py ["--lines", FILE, "1411", "1503", "# (I will provide the new code ...)"]
    code, out = run('show.py', '--lines', 'pay/invoice.py', '1', '3', '# new code later', ws=ws)
    assert code == 0 and 'L1-3' in out and 'ignored extra' in out


def test_show_errors(ws):
    code, out = run('show.py', ws=ws)
    assert code == 2 and 'USAGE ERROR' in out and 'E.g. show.py  target=' in out  # v18: no usage block after an example
    code, out = run('show.py', 'pay', 'x', ws=ws)
    assert code == 2 and 'directory' in out
    code, out = run('show.py', 'pay/nope.py', 'x', ws=ws)
    assert code == 2 and 'not found' in out
    code, out = run('show.py', 'FILE', 'SYMBOL', ws=ws)
    assert code == 2 and 'placeholder' in out
    code, out = run('show.py', 'pay/invoice.py', 'total', 'describe', ws=ws)
    assert code == 2 and 'one SYMBOL or two line numbers' in out


def test_show_resolves_unique_suffix(ws):
    code, out = run('show.py', 'invoice.py', 'Invoice', ws=ws)
    assert code == 0 and 'using pay/invoice.py' in out


# ================================ search.py ================================

def test_search_directory_as_text_is_a_usage_error(ws):
    code, out = run('search.py', 'pay/', ws=ws)  # trace rich_3777 ran search.py ["rich/"] three times
    assert code == 2 and 'is a directory' in out and 'outline.py  files=pay/' in out
    assert calls(correction(out)) == [('search.py', {'text': 'def export_html', 'within': 'pay/'}),
                                      ('outline.py', {'files': 'pay/'}), ('search.py', {'text': 'pay/', 'within': '.'})]
    assert_notation(correction(out))
    code, out = run('search.py', 'pay/invoice.py', ws=ws)
    assert code == 2 and 'is a file' in out


def test_search_path_text_escape_hatch(ws):
    code, out = run('search.py', 'pay/', '.', ws=ws)
    assert code in (0, 1) and 'USAGE' not in out


def test_search_packed_regex_flag(ws):
    code, out = run('search.py', '-E "def (total|describe)"', ws=ws)
    assert code == 0 and 'def total' in out and 'def describe' in out and 'split the flag' in out
    code, out = run('search.py', '-E root_path', 'pay/', ws=ws)
    assert 'USAGE' not in out


def test_search_swapped_arguments(ws):
    code, out = run('search.py', 'pay/invoice.py', 'parse_header', ws=ws)  # trace rich_3043
    assert code == 0 and 'swapped' in out and 'pay/invoice.py:' in out


def test_search_two_words_suggests_regex(ws):
    code, out = run('search.py', 'Progress', 'Status', ws=ws)  # trace rich_3777
    assert code == 2 and 'search.py  regex=Progress|Status' in out
    assert_notation(correction(out))


def test_search_retries_without_quotes_and_as_regex(ws):
    code, out = run('search.py', '"describe"', 'pay/', ws=ws)
    assert code == 0 and 'without the surrounding quotes' in out
    code, out = run('search.py', r'def (total|describe)\(', ws=ws)
    assert code == 0 and 'as a regex' in out and 'def describe' in out
    code, out = run('search.py', 'INVOICE', ws=ws)
    assert code == 0 and 'ignoring case' in out


def test_search_invalid_regex_falls_back_to_literal(ws):
    code, out = run('search.py', '-E', 'total(', ws=ws)
    assert code == 0 and 'invalid regex' in out and 'def total' in out


def test_search_json_packed(ws):
    code, out = run('search.py', '["parse_header", "pay/"]', ws=ws)
    assert code == 0 and 'pay/invoice.py:' in out
    code, out = run('search.py', '-E", "total|describe"', ws=ws)  # trace rich_3105
    assert code == 0 and 'def describe' in out


def test_search_skips_hidden_harness_files(ws):
    (ws / '.adk_exec_1234.py').write_text("sys.argv = ['scripts/search.py', 'needle_xyz']\n")
    code, out = run('search.py', 'needle_xyz', ws=ws)
    assert code == 1 and '.adk_exec' not in out


def test_search_text_and_path_packed(ws):
    code, out = run('search.py', 'parse_header pay/invoice.py', ws=ws)
    assert code == 0 and 'pay/invoice.py:' in out and 'two keys' in out
    code, out = run('search.py', 'pay/invoice.py describe', ws=ws)
    assert code == 0 and 'def describe' in out


def test_search_placeholder_and_path_like_text(ws):
    code, out = run('search.py', 'TEXT', ws=ws)
    assert code == 2 and 'placeholder' in out
    code, out = run('search.py', 'rich/.*\\.py', ws=ws)
    assert code == 2 and 'file contents, not file names' in out


def test_search_flags_and_glob(ws):
    code, out = run('search.py', '-i', 'INVOICE', ws=ws)
    assert code == 0 and 'class Invoice' in out and 'ignoring case' not in out
    code, out = run('search.py', '-rn', 'parse_header', ws=ws)
    assert code == 0 and 'ignored the grep flag -rn' in out
    code, out = run('search.py', 'def ', 'pay/*.py', ws=ws)
    assert code == 0 and 'pay/invoice.py:' in out and 'tests/' not in out


def test_search_missing_text(ws):
    code, out = run('search.py', ws=ws)
    assert code == 2 and 'usage' in out


# ================================ outline.py ================================

def test_outline_packed_and_json(ws):
    code, out = run('outline.py', '--grep total pay/invoice.py', ws=ws)  # trace rich_3468
    assert code == 0 and 'def total' in out and 'describe' not in out
    code, out = run('outline.py', '["pay/invoice.py"]', ws=ws)  # trace fastapi_14258
    assert code == 0 and 'class Invoice' in out


def test_outline_directory_lists_files(ws):
    code, out = run('outline.py', 'pay/', ws=ws)
    assert code == 0 and 'pay/invoice.py (' in out and 'Python files' in out
    (ws / '.github/workflows').mkdir(parents=True)
    (ws / '.github/workflows/publish.yml').write_text('name: x\n')
    code, out = run('outline.py', '.github/workflows', ws=ws)
    assert code == 0 and 'publish.yml (1 lines)' in out


def test_outline_grep_after_files_and_non_python(ws):
    code, out = run('outline.py', 'pay/invoice.py', '--grep', 'describe', ws=ws)
    assert code == 0 and 'def describe' in out and 'def total' not in out
    code, out = run('outline.py', 'CHANGELOG.md', ws=ws)
    assert 'not a Python file' in out and 'show.py  target=CHANGELOG.md  start=1  end=100' in out
    assert_notation(out)


def test_outline_errors_and_suffix(ws):
    code, out = run('outline.py', ws=ws)
    assert code == 2 and 'usage' in out
    code, out = run('outline.py', '--grep', ws=ws)
    assert code == 2
    code, out = run('outline.py', 'FILE', ws=ws)
    assert code == 2 and 'placeholder' in out
    code, out = run('outline.py', 'invoice.py', ws=ws)
    assert code == 0 and 'class Invoice' in out and 'using pay/invoice.py' in out


# ================================ run.py ================================

def test_run_snippet_packed_forms(ws):
    for arg in ('snippet\nprint(40 + 2)', 'snippet,\nprint(40 + 2)', 'snippet`, `print(40 + 2)',
                '["snippet", "print(40 + 2)"]'):
        code, out = run('run.py', arg, ws=ws)
        assert code == 0 and '42' in out and 'exit code: 0' in out, (arg, out)


def test_run_python_c_and_bare_code(ws):
    code, out = run('run.py', 'python', '-c', 'print(6 * 7)', ws=ws)
    assert '42' in out and 'python -c' in out
    code, out = run('run.py', 'print(6 * 7)', ws=ws)
    assert '42' in out and 'snippet' in out


def test_run_test_forms(ws):
    for args in (['test tests/test_invoice.py -k total'], ['test', 'tests/test_invoice.py -k total'],
                 ['test', 'tests/test_invoice.py', '-k total'], ['test', 'tests/test_invoice.py', 'total'],
                 ['tests/test_invoice.py'], ['pytest', 'tests/test_invoice.py'],
                 ['python -m pytest tests/test_invoice.py -k total'], ['test', 'tests/test_invoice.py::test_total']):
        code, out = run('run.py', *args, ws=ws)
        assert 'exit code:' in out and 'USAGE' not in out, (args, out)
        if 'No module named pytest' not in out:
            assert '1 passed' in out, (args, out)


def test_run_test_missing_file(ws):
    code, out = run('run.py', 'test', 'tests/test_nope.py', ws=ws)
    assert code == 2 and 'not found' in out
    code, out = run('run.py', 'test', 'test_invoice.py', ws=ws)
    assert 'using tests/test_invoice.py' in out
    code, out = run('run.py', 'test', 'tests/test_x.py', ws=ws)  # the usage example itself
    assert code == 2


def test_run_usage_errors(ws):
    code, out = run('run.py', ws=ws)
    assert code == 2 and out.startswith('USAGE ERROR: run.py needs the key test') and '\nusage:' not in out
    code, out = run('run.py', 'snippet', ws=ws)
    assert code == 2 and 'needs the code' in out
    code, out = run('run.py', 'snippet', '<python code>', ws=ws)
    assert code == 2 and 'placeholder' in out
    code, out = run('run.py', 'snippet', 'x = 1', 'print(x)', ws=ws)
    assert code == 2 and 'ONE string' in out
    code, out = run('run.py', 'test', ws=ws)
    assert code == 2
    code, out = run('run.py', 'lint', 'pay/', ws=ws)
    assert code == 2 and '"test" or "snippet"' in out


# ================================ all scripts ================================

def test_scripts_leave_no_bytecode_and_no_files(ws):
    for script, args in (('show.py', ['pay/invoice.py', 'Invoice']), ('search.py', ['Invoice']),
                         ('outline.py', ['pay/invoice.py']), ('edit.py', ['pay/nope.py', 'a', 'b']),
                         ('run.py', ['snippet', 'print(1)'])):
        run(script, *args, ws=ws, env_extra={'PYTHONDONTWRITEBYTECODE': ''})
    assert not (SCRIPTS / '__pycache__').exists()
    assert git(ws, 'status', '--porcelain') == ''


def test_argfix_is_not_a_command(ws):
    code, out = run('argfix.py', ws=ws)
    assert code == 0 and 'not a command' in out


# ============================ agent-v12: usage errors with a clear intent are run ============================
# E024 had 24 USAGE ERRORs in the failed runs; most had an unambiguous intent (a symbol without its file, a range
# without its file, a file alone, outline's "target", search's "query", edit's "old" together with "start").
# They now run, with one NOTE line saying how the call was read. Genuinely ambiguous calls stay errors.

def obj(**kw):
    """argv of the object form, as run_skill_script builds it from args={...}."""
    return [x for k, v in kw.items() for x in ('--' + k, str(v))]


FLOW_ON = {'REPO_TOOLS_FLOW': 'on'}
BIG_CLASS = ('class Big:\n    """A class with many methods."""\n\n'
             + ''.join(f'    def method_{i}(self):\n        x = {i}\n        y = x + 1\n        z = y * 2\n        return z\n\n'
                       for i in range(30))
             + '    def apply_discount(self, rate):\n        return rate\n')
HUGE_FN = ('def huge(a, b=1):\n    """Do everything.\n' + ''.join(f'    detail {i}\n' for i in range(170))
           + '    """\n    total = a + b\n    return total\n')


def commit(ws, files):
    for rel, text in files.items():
        (ws / rel).write_text(text)
    git(ws, '-c', 'user.email=t@t', '-c', 'user.name=t', 'add', '-A')
    git(ws, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'more')


def test_show_symbol_without_target_is_looked_up(ws):
    code, out = run('show.py', *obj(symbol='Invoice.total'), ws=ws)
    assert code == 0, out
    assert out.startswith('NOTE: no target given; read as show.py  target=pay/invoice.py  symbol=Invoice.total')
    assert 'def total(self, tax=0.1):' in out
    code, out = run('show.py', *obj(symbol='include_router'), ws=ws)  # a unique plain name
    assert code == 0 and 'target=pay/routing.py  symbol=include_router' in out


def test_show_symbol_without_target_ambiguous_lists_ready_calls(ws):
    commit(ws, {'pay/other.py': 'class Other:\n    def describe(self):\n        return 1\n'})
    code, out = run('show.py', *obj(symbol='describe'), ws=ws)
    assert code == 2 and 'describe is defined in 2 places; add the target' in out
    found = assert_notation(correction(out))
    assert ('show.py', {'target': 'pay/invoice.py', 'symbol': 'Invoice.describe'}) in found
    assert ('show.py', {'target': 'pay/other.py', 'symbol': 'Other.describe'}) in found
    code, out = run('show.py', *obj(symbol='Nope'), ws=ws)
    assert code == 2 and 'Nope is not defined in the repository' in out and 'search.py  text=Nope' in out


def test_show_range_without_target_uses_the_file_in_view(ws):
    code, out = run('show.py', *obj(start='3', end='6'), ws=ws)  # nothing viewed yet: still an error
    assert code == 2 and 'needs the key target' in out
    run('locate.py', *obj(query='include_router prefix must start with /'), ws=ws, env_extra=FLOW_ON)
    code, out = run('show.py', *obj(start='3', end='6'), ws=ws)
    assert code == 0, out
    assert out.startswith('NOTE: no target given; read as show.py  target=pay/routing.py  start=3  end=6')
    assert 'pay/routing.py L3-6 of' in out


def test_show_symbol_wins_over_start_end(ws):
    code, out = run('show.py', *obj(target='pay/invoice.py', symbol='Invoice.total', start='1', end='5'), ws=ws)
    assert code == 0 and 'NOTE: symbol and start/end given; showing the symbol' in out and 'def total' in out


def test_show_target_only_prints_outline_or_head(ws):
    code, out = run('show.py', *obj(target='pay/long.py'), ws=ws)
    assert code == 0 and out.startswith('NOTE: pay/long.py has 240 lines; its outline:')  # v18: short note
    assert '  L1-2 def f0' in out and 'e.g. show.py  target=pay/long.py  symbol=f0' in out
    assert_notation(out.split('\n', 1)[1])
    code, out = run('show.py', *obj(target='CHANGELOG.md'), ws=ws)
    assert code == 0 and 'CHANGELOG.md L1-100 of 150' in out and '- change 99' in out


def test_show_missing_path_with_a_unique_file_name(ws):
    code, out = run('show.py', *obj(target='billing/invoice.py', symbol='Invoice.total'), ws=ws)
    assert code == 0 and 'billing/invoice.py does not exist; using pay/invoice.py, the only file named invoice.py' in out
    assert 'def total' in out
    code, out = run('edit.py', *obj(target='billing/invoice.py', old='x', new='y'), ws=ws)  # never for writes
    assert code == 1 and 'not found' in out


def test_outline_target_and_empty_args(ws):
    code, out = run('outline.py', *obj(target='pay/invoice.py'), ws=ws)
    assert code == 0 and out.startswith('NOTE: read target as files: outline.py  files=pay/invoice.py')
    assert 'class Invoice' in out
    run('locate.py', *obj(query='include_router prefix must start with /'), ws=ws, env_extra=FLOW_ON)
    code, out = run('outline.py', *obj(), ws=ws)
    assert code == 0 and out.startswith('NOTE: no files given; read as outline.py  files=pay/routing.py')


def test_search_query_key_and_several_words(ws):
    code, out = run('search.py', *obj(query='self.amount'), ws=ws)
    assert code == 0 and out.startswith('NOTE: read query as text') and 'pay/invoice.py:' in out
    code, out = run('search.py', *obj(query='class Invoice class Nothing'), ws=ws)  # trace fastapi_14372
    assert code == 0 and 'showing matches for any of the words Invoice, Nothing' in out
    assert 'pay/invoice.py:' in out and 'class Invoice' in out


def test_edit_old_with_start_uses_the_old_text(ws):
    n = INVOICE.split('\n').index('        if self.amount == None:') + 1
    code, out = run('edit.py', *obj(target='pay/invoice.py', old='        if self.amount == None:', start=str(n),
                                    new='        if self.amount is None:'), ws=ws)
    assert code == 0 and 'NOTE: old and start/end given; replaced the old text' in out
    assert '        if self.amount is None:' in src(ws)


def test_edit_start_without_end_and_new_suggests_the_statement_end(ws):
    n = INVOICE.split('\n').index('    def total(self, tax=0.1):') + 1
    code, out = run('edit.py', *obj(target='pay/invoice.py', start=str(n)), ws=ws)
    assert code == 2 and 'needs the key end' in out
    assert f'edit.py  target=pay/invoice.py  start={n}  end={n + 3}  new=<new text for lines {n}-{n + 3}>' in out
    assert src(ws) == INVOICE


def test_edit_start_without_end_replaces_one_line(ws):
    """v16 (note 11 L10: 87 "needs the key end" errors): start alone is that one line."""
    n = INVOICE.split('\n').index('    def total(self, tax=0.1):') + 1
    code, out = run('edit.py', *obj(target='pay/invoice.py', start=str(n), new='    def total(self, tax=0.2):'), ws=ws)
    assert code == 0 and out.startswith(f'NOTE: no end given: replaced line {n} only\nEDITED'), out
    assert src(ws) == INVOICE.replace('tax=0.1', 'tax=0.2')


def test_edit_start_without_end_replaces_a_whole_definition_with_a_body(ws):
    lines = INVOICE.split('\n')
    n = lines.index('    def total(self, tax=0.1):') + 1
    new = '    def total(self, tax=0.1):\n        return 0'
    code, out = run('edit.py', *obj(target='pay/invoice.py', start=str(n), new=new), ws=ws)
    assert code == 0 and f'NOTE: no end given: NEW redefines total, so it replaced its lines {n}-{n + 3}' in out, out
    assert src(ws) == '\n'.join(lines[:n - 1] + new.split('\n') + lines[n + 3:])


def test_show_long_class_prints_methods_and_the_related_one(ws):
    commit(ws, {'pay/big.py': BIG_CLASS})
    run('locate.py', *obj(query='Big.apply_discount ignores the rate'), ws=ws, env_extra=FLOW_ON)
    code, out = run('show.py', *obj(target='pay/big.py', symbol='Big'), ws=ws)
    n = len(BIG_CLASS.splitlines())
    assert code == 0 and out.startswith(f'pay/big.py class Big L1-{n} ({n} lines); its methods:')  # v18 format
    assert f'apply_discount L{n - 1}-{n}' in out and 'its first lines:' in out
    assert 'most related to the task: show.py  target=pay/big.py  symbol=Big.apply_discount' in out
    assert len(out.splitlines()) < 80
    code, out = run('show.py', *obj(target='pay/big.py', symbol='Big'), ws=ws, env_extra=FLOW_ON)
    assert 'NEXT: read the definition most related to the issue' in out
    assert 'show.py  target=pay/big.py  symbol=Big.apply_discount' in out.split('NEXT:', 1)[1]


def test_show_long_function_prints_head_and_body(ws):
    commit(ws, {'pay/huge.py': HUGE_FN})
    code, out = run('show.py', *obj(target='pay/huge.py', symbol='huge'), ws=ws)
    lines = HUGE_FN.splitlines()
    body = lines.index('    total = a + b') + 1
    assert code == 0 and out.startswith(f'pay/huge.py huge L1-{len(lines)}')
    # v18: every docstring folded to its first line; the 2-line body follows (no window needed)
    assert '\n1  def huge(a, b=1):\n' in out and f'... (docstring lines 3-{body - 1})' in out
    assert f'\n{body}      total = a + b' in out and 'detail 30' not in out
