"""Unit tests for the repo-tools skill scripts (agent/skills/repo-tools/scripts).

Each test builds a small git repository in a temp dir and runs a script as a subprocess, the way
run_skill_script does, with WORKSPACE pointing at the temp repo. Run inside the sandbox image with
`scripts/test_skills.sh` (Python 3.13, like Kaggle), or with any local pytest.
"""

import glob
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import textwrap

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'agent/skills/repo-tools/scripts'
DATA = pathlib.Path(__file__).resolve().parents[1] / 'data/competition'
RESULTS = pathlib.Path(__file__).resolve().parents[1] / 'results'

MODULE = textwrap.dedent('''\
    """Payment helpers."""

    TEMPLATE = """\\
    <html>
    <head>
    <meta charset="UTF-8">
    </head>
    <body>{body}</body>
    """


    class Invoice:
        def __init__(self, amount):
            self.amount = amount

        def total(self, tax=0.1):
            if self.amount == None:
                return 0
            return self.amount * (1 + tax)


    async def fetch_invoice(invoice_id):
        return Invoice(invoice_id)


    def parse_header(value):
        return value.split(",")
    ''')


def run(script, *args, ws):
    env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_FLOW='off')
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True, env=env,
                       timeout=60)
    return r.returncode, r.stdout


def git(ws, *args):
    subprocess.run(['git', '-C', str(ws), *args], check=True, capture_output=True)


@pytest.fixture
def ws(tmp_path):
    (tmp_path / 'pay').mkdir()
    (tmp_path / 'pay/__init__.py').write_text('')
    (tmp_path / 'pay/invoice.py').write_text(MODULE)
    (tmp_path / 'pay/http.py').write_text('def parse_header(v):\n    return v\n\n\ndef send(req):\n    return req\n')
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/test_invoice.py').write_text('from pay.invoice import Invoice\n\n\ndef test_total():\n    assert Invoice(10).total() == 11\n')
    (tmp_path / 'README.md').write_text('# pay\n')
    git(tmp_path, 'init', '-q')
    git(tmp_path, '-c', 'user.email=t@t', '-c', 'user.name=t', 'add', '-A')
    git(tmp_path, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'baseline')
    return tmp_path


def source(ws, rel='pay/invoice.py'):
    return (ws / rel).read_text()


# ---------------- edit.py ----------------

def test_edit_exact(ws):
    code, out = run('edit.py', 'pay/invoice.py', 'if self.amount == None:', 'if self.amount is None:', ws=ws)
    assert code == 0 and out.startswith('EDITED pay/invoice.py (exact)')
    assert 'if self.amount is None:' in source(ws)
    assert '+        if self.amount is None:' in out  # diff is printed


def test_edit_accepts_workspace_prefixed_path(ws):
    code, _ = run('edit.py', '/workspace/pay/invoice.py', 'if self.amount == None:', 'if self.amount is None:', ws=ws)
    assert code == 0


def test_edit_decodes_escaped_newlines_and_quotes(ws):
    old = 'def total(self, tax=0.1):\\n        if self.amount == None:'
    new = 'def total(self, tax=0.1):\\n        if self.amount is None:'
    code, out = run('edit.py', 'pay/invoice.py', old, new, ws=ws)
    assert code == 0 and 'decoded escapes' in out
    assert 'if self.amount is None:' in source(ws) and '\\n' not in source(ws).split('class Invoice')[1]


def test_edit_decodes_double_escaped_text(ws):
    old = '<meta charset=\\\\\\"UTF-8\\\\\\">'  # two layers: \\\" -> \" -> "
    code, out = run('edit.py', 'pay/invoice.py', old, '<meta charset=\\\\\\"utf-8\\\\\\">', ws=ws)
    assert code == 0, out
    assert '<meta charset="utf-8">' in source(ws)


def test_edit_keeps_literal_backslash_n_inside_strings_when_raw_matches(ws):
    code, out = run('edit.py', 'pay/invoice.py', 'return value.split(",")', 'return value.split("\\n")', ws=ws)
    assert code == 0, out
    assert 'return value.split("\\n")' in source(ws)  # literal backslash-n preserved, file still parses


def test_edit_whitespace_flexible_reindents(ws):
    old = 'if self.amount == None:\n    return 0'  # indentation lost
    new = 'if self.amount is None:\n    return 0.0'
    code, out = run('edit.py', 'pay/invoice.py', old, new, ws=ws)
    assert code == 0 and 'whitespace-flexible' in out
    assert '        if self.amount is None:\n            return 0.0' in source(ws)


def test_edit_rejects_ambiguous_match(ws):
    code, out = run('edit.py', 'pay/invoice.py', 'return', 'yield', ws=ws)
    assert code == 1 and 'occurs' in out and 'times' in out
    assert source(ws) == MODULE


def test_edit_not_found_shows_closest_region(ws):
    code, out = run('edit.py', 'pay/invoice.py', 'if self.amount == Nothing:', 'x', ws=ws)
    assert code == 1 and 'NOT APPLIED' in out and 'closest region' in out
    assert 'if self.amount == None:' in out


def test_edit_refuses_syntax_breaking_change(ws):
    code, out = run('edit.py', 'pay/invoice.py', 'if self.amount == None:', 'if self.amount == None', ws=ws)
    assert code == 1 and 'would not parse' in out
    assert source(ws) == MODULE


def test_edit_refuses_test_and_config_files(ws):
    for rel in ('tests/test_invoice.py', 'conftest.py', 'pyproject.toml', 'pkg/foo_test.py'):
        code, out = run('edit.py', rel, 'a', 'b', ws=ws)
        assert code == 1 and out.startswith('REFUSED'), rel


def test_edit_missing_file(ws):
    code, out = run('edit.py', 'pay/nope.py', 'a', 'b', ws=ws)
    assert code == 1 and 'not found' in out


def test_edit_non_python_file_skips_syntax_check(ws):
    code, _ = run('edit.py', 'README.md', '# pay', '# pay (', ws=ws)
    assert code == 0 and '# pay (' in source(ws, 'README.md')


def test_edit_lines_mode(ws):
    lines = MODULE.split('\n')
    n = lines.index('        if self.amount == None:') + 1
    code, out = run('edit.py', '--lines', 'pay/invoice.py', str(n), str(n + 1),
                    '        if self.amount is None:\\n            return 0.0', ws=ws)
    assert code == 0, out
    assert '        if self.amount is None:\n            return 0.0' in source(ws)


def test_edit_lines_mode_bad_range(ws):
    code, out = run('edit.py', '--lines', 'pay/invoice.py', '500', '501', 'x', ws=ws)
    assert code == 1 and 'bad range' in out


def test_edit_usage(ws):
    code, out = run('edit.py', 'only-one-arg', ws=ws)
    assert code == 2 and out.startswith('USAGE ERROR') and 'edit.py  target=' in out


@pytest.mark.skipif(not glob.glob(str(RESULTS / 'E004_v1_20m/rich_3043/traces/*.json'))
                    or not (DATA / 'snapshots/rich_3043.tgz').exists(), reason='needs local trace + snapshot')
def test_edit_recovers_real_escaped_edits_from_trace(tmp_path):
    """Regression: v1's failed edit_file calls on rich_3043. All escape-caused failures must now apply;
    the remaining ones contain genuinely wrong text and must be refused without touching the file."""
    with tarfile.open(DATA / 'snapshots/rich_3043.tgz') as tf:
        member = tf.getmember('./rich/_export_format.py')
        original = tf.extractfile(member).read().decode()
    trace = json.load(open(glob.glob(str(RESULTS / 'E004_v1_20m/rich_3043/traces/*.json'))[0]))
    fails = [tc['arguments'] for s in trace['steps'] for tc in s.get('tool_calls') or []
             if tc.get('function_name') == 'edit_file' and 'not found' in str((s.get('observation') or {}).get('content'))]
    applied = 0
    for a in fails:
        (tmp_path / 'rich').mkdir(exist_ok=True)
        (tmp_path / 'rich/_export_format.py').write_text(original)
        code, _ = run('edit.py', a['filepath'], a['old_string'], a['new_string'], ws=tmp_path)
        applied += code == 0
        if code != 0:
            assert (tmp_path / 'rich/_export_format.py').read_text() == original
    # v20: + call 2 (OLD wrote `body {` where the f-string has `body {{`): anchored on OLD's first and last lines, NEW's
    # change (`<html>` moved above `<head>`, the actual fix) is applied there
    assert len(fails) == 11 and applied == 9


# ---------------- check_patch.py ----------------

def test_check_patch_no_changes(ws):
    code, out = run('check_patch.py', ws=ws)
    assert code == 1 and 'NO CHANGES' in out


def test_check_patch_ready(ws):
    run('edit.py', 'pay/invoice.py', 'if self.amount == None:', 'if self.amount is None:', ws=ws)
    code, out = run('check_patch.py', ws=ws)
    assert code == 0 and 'READY' in out and 'pay/invoice.py' in out


def test_check_patch_flags_and_cleans_stray_files_and_test_edits(ws):
    run('edit.py', 'pay/invoice.py', 'if self.amount == None:', 'if self.amount is None:', ws=ws)
    (ws / 'repro.py').write_text('print(1)\n')
    (ws / 'tests/test_invoice.py').write_text('# changed\n')
    code, out = run('check_patch.py', ws=ws)
    assert code == 1 and 'repro.py' in out and 'tests/test_invoice.py' in out
    code, out = run('check_patch.py', '--clean', ws=ws)
    assert code == 0 and 'READY' in out
    assert not (ws / 'repro.py').exists()
    assert (ws / 'tests/test_invoice.py').read_text().startswith('from pay.invoice')
    assert 'if self.amount is None:' in source(ws)  # the real fix survives


def test_check_patch_flags_syntax_error(ws):
    (ws / 'pay/http.py').write_text('def broken(:\n')
    code, out = run('check_patch.py', ws=ws)
    assert code == 1 and 'does not compile' in out


def test_check_patch_flags_import_error(ws):
    (ws / 'pay/http.py').write_text('import does_not_exist_anywhere\n')
    code, out = run('check_patch.py', ws=ws)
    assert code == 1 and 'import pay.http fails' in out


# ---------------- locate.py ----------------

def test_locate_prefers_defining_source_file_over_tests(ws):
    code, out = run('locate.py', 'Invoice.total returns wrong value when amount is None', ws=ws)
    assert code == 0
    files = [l.split()[1] for l in out.splitlines() if l[:1].isdigit()]
    assert files[0] == 'pay/invoice.py'
    assert 'Invoice.total' in out
    assert files.index('tests/test_invoice.py') > 0 if 'tests/test_invoice.py' in files else True


def test_locate_path_and_traceback_boosts(ws):
    text = 'Traceback:\n  File "/workspace/pay/http.py", line 2, in parse_header\nValueError'
    code, out = run('locate.py', text, ws=ws)
    first = [l for l in out.splitlines() if l[:1].isdigit()][0]
    assert 'pay/http.py' in first and 'traceback' in first


def test_locate_finds_async_functions(ws):
    code, out = run('locate.py', '`fetch_invoice` should retry', ws=ws)
    assert 'fetch_invoice' in out


def test_locate_shows_best_function_source(ws):
    code, out = run('locate.py', 'Invoice.total returns wrong value when amount is None', ws=ws)
    assert '--- best match: pay/invoice.py Invoice.total' in out
    assert 'if self.amount == None:' in out  # the function body is printed with line numbers
    assert any(l.strip().startswith(str(n)) for l in out.splitlines() for n in range(1, 40))


def test_locate_no_show(ws):
    code, out = run('locate.py', '--no-show', 'Invoice.total returns wrong value', ws=ws)
    assert code == 0 and '--- best match' not in out and 'pay/invoice.py' in out


def test_locate_usage(ws):
    code, out = run('locate.py', ws=ws)
    assert code == 2 and 'usage' in out


# ---------------- outline.py / show.py ----------------

def test_outline_lists_definitions_with_ranges(ws):
    code, out = run('outline.py', 'pay/invoice.py', 'pay/http.py', ws=ws)
    assert code == 0
    assert 'class Invoice' in out and 'def total(self, tax)' in out  # v18: parameter names only
    assert 'async def fetch_invoice(invoice_id)' in out and 'TEMPLATE L3-9' in out
    assert 'pay/http.py' in out


def test_outline_grep(ws):
    code, out = run('outline.py', '--grep', 'total', 'pay/invoice.py', ws=ws)
    assert 'def total' in out and 'fetch_invoice' not in out


def test_show_method_with_line_numbers(ws):
    code, out = run('show.py', 'pay/invoice.py', 'Invoice.total', ws=ws)
    assert code == 0 and 'Invoice.total' in out
    assert 'if self.amount == None:' in out and out.splitlines()[1].strip()[0].isdigit()


def test_show_missing_symbol(ws):
    code, out = run('show.py', 'pay/invoice.py', 'Nope', ws=ws)
    assert code == 1 and 'not found' in out


# ---------------- search.py / run.py / show.py ranges ----------------

def test_search_lists_source_before_tests(ws):
    code, out = run('search.py', 'Invoice', ws=ws)
    lines = out.splitlines()
    assert code == 0 and lines[0].startswith('pay/invoice.py:')
    assert lines.index(next(l for l in lines if l.startswith('tests/'))) > 0


def test_search_caps_output(ws):
    (ws / 'pay/many.py').write_text(''.join(f'x{i} = "needle"\n' for i in range(40)))
    code, out = run('search.py', 'needle', ws=ws)
    assert len([l for l in out.splitlines() if ':' in l and 'needle' in l]) == 10 and 'more matches' in out  # v18


def test_search_path_and_no_match(ws):
    code, out = run('search.py', 'parse_header', 'pay/http.py', ws=ws)
    assert code == 0 and out.splitlines()[0] == 'pay/http.py:1:'  # one hit: with context, the hit marked
    assert '\n1> def parse_header(v):' in out and 'pay/invoice.py' not in out  # v18: unpadded number
    code, out = run('search.py', 'zzz_nothing_zzz', ws=ws)
    assert code == 1 and 'no matches' in out


def test_search_invalid_regex_falls_back_to_literal(ws):
    code, out = run('search.py', 'split(",")', ws=ws)  # literal by default
    assert code == 0 and 'pay/invoice.py' in out


def test_run_snippet_imports_repo_and_leaves_no_files(ws):
    code, out = run('run.py', 'snippet', 'from pay.invoice import Invoice\nprint(Invoice(2).total(0))', ws=ws)
    assert code == 0 and '2' in out and 'exit code: 0' in out
    assert subprocess.run(['git', '-C', str(ws), 'status', '--porcelain'], capture_output=True, text=True).stdout == ''


def test_run_snippet_decodes_escaped_newlines(ws):
    code, out = run('run.py', 'snippet', 'x = 1\\nprint(x + 1)', ws=ws)
    assert 'exit code: 0' in out and '2' in out


def test_run_test_file(ws):
    code, out = run('run.py', 'test', 'tests/test_invoice.py', ws=ws)
    assert 'exit code:' in out  # pytest may be missing outside the sandbox image; inside it this passes
    if 'No module named pytest' not in out:
        assert '1 passed' in out and 'exit code: 0' in out


def test_show_line_range(ws):
    code, out = run('show.py', 'pay/invoice.py', '1', '3', ws=ws)
    assert code == 0 and out.splitlines()[0].startswith('pay/invoice.py L1-3')
    assert '"""Payment helpers."""' in out


def test_search_regex_flag(ws):
    code, out = run('search.py', '-E', r'def (total|parse_header)\(', ws=ws)
    assert code == 0 and 'def total' in out and 'def parse_header' in out
