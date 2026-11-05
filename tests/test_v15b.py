"""agent-v15b: v14 plus v15's cheap edit guards, without v15's test runs (agent/skills/repo-tools/scripts: guards.py,
edit.py, flow.py, run.py). research/notes/10-edit-failures.md lever 2; experiments/LOG.md E034/E035.

- edit.py undoes an edit after which the module no longer imports (it did before), refuses a read-only @property of
  a name the class assigns, and warns about \\uXXXX escapes that Python decodes to the plain character.
- A refused edit gets a targeted footer: property -> show the assignment lines; import -> redo the line range.
- No pytest runs after edits, no REGRESSIONS, no NOT READY gate in check_patch.py (v14 behaviour).
- run.py snippets time out after 30 s (tests 90 s); the flow clock is 270 s (4.5 min).
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
SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'agent/skills/repo-tools/scripts'
sys.path.insert(0, str(SCRIPTS))
import flow  # noqa: E402
import guards  # noqa: E402

INVOICE = textwrap.dedent('''\
    """Invoices."""

    RATE = 0.1


    class Invoice:
        def __init__(self, amount):
            self.amount = amount
            self.paid = False

        def total(self, tax=RATE):
            if self.amount is None:
                return 0
            return self.amount * (1 + tax)

        def label(self):
            return "invoice"
    ''')
EXTRA = 'def double(x):\n    return 2 * x\n'
TEST_INVOICE = textwrap.dedent('''\
    from pay.invoice import Invoice


    def test_total():
        assert Invoice(10).total(0) == 10
    ''')


def git(ws, *args):
    return subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / 'repo'
    (root / 'pay').mkdir(parents=True)
    (root / 'pay/__init__.py').write_text('')
    (root / 'pay/invoice.py').write_text(INVOICE)
    (root / 'pay/extra.py').write_text(EXTRA)
    (root / 'tests').mkdir()
    (root / 'tests/test_invoice.py').write_text(TEST_INVOICE)
    git(root, 'init', '-q')
    git(root, 'add', '-A')
    git(root, 'commit', '-qm', 'baseline')
    return root


def env(ws, flow_on=False):
    e = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'), PYTHONDONTWRITEBYTECODE='1')
    e.pop('REPO_TOOLS_FLOW', None)
    if not flow_on:
        e['REPO_TOOLS_FLOW'] = 'off'
    return e


def run(script, ws, flow_on=False, **args):
    argv = [x for k, v in args.items() for x in ('--' + k, str(v))]
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *argv], capture_output=True, text=True,
                       env=env(ws, flow_on), timeout=180)
    assert 'Traceback' not in r.stderr, r.stderr
    return r.returncode, r.stdout


def status(ws):
    return git(ws, 'status', '--porcelain', '--untracked-files=all', '--ignored').strip()


BREAK_TOTAL = {'target': 'pay/invoice.py', 'old': 'return self.amount * (1 + tax)', 'new': 'return self.amount * (2 + tax)'}


# ---------------------------------------------------------------- no test runs (v15's regression check removed)

def test_edit_runs_no_tests_and_is_fast(ws):
    t0 = time.time()
    code, out = run('edit.py', ws, **BREAK_TOTAL)            # breaks tests/test_invoice.py::test_total
    took = time.time() - t0
    assert code == 0 and out.startswith('EDITED pay/invoice.py'), out
    assert 'regression' not in out.lower() and 'REGRESSIONS' not in out
    assert took < 10, took                                   # parse + import check only
    assert status(ws) == 'M pay/invoice.py'                  # the import check leaves nothing (no __pycache__)


def test_edit_footer_is_v14s(ws):
    code, out = run('edit.py', ws, flow_on=True, **BREAK_TOTAL)
    nxt = [l for l in out.splitlines() if l.startswith('NEXT:')]
    assert nxt[0].startswith('NEXT: verify once: run.py  test=tests/test_invoice.py'), out
    for l in nxt:
        assert_notation(l)


def test_check_patch_has_no_gate(ws):
    """check_patch runs no tests itself. v21 V1: an untested edit is NOT VERIFIED while time is left, and the footer
    names the existing tests to run; with the gate off (no time left) it is READY, as in v14."""
    run('edit.py', ws, flow_on=True, **BREAK_TOTAL)
    code, out = run('check_patch.py', ws, flow_on=True)
    assert code == 1 and 'NOT VERIFIED: nothing ran since your edit at call 1' in out and 'READY' not in out, out
    assert 'NEXT: run the existing tests that use total: run.py  test=tests/test_invoice.py' in out, out
    assert 'passed' not in out and 'failed' not in out, out   # no test run
    os.environ['REPO_TOOLS_VERIFY_MIN_LEFT_S'] = '100000'
    try:
        code, out = run('check_patch.py', ws, flow_on=True, clean='no')
    finally:
        del os.environ['REPO_TOOLS_VERIFY_MIN_LEFT_S']
    assert code == 0 and 'READY: call submit_patch now.' in out and 'NOT READY' not in out, out
    assert out.rstrip().endswith('NEXT: submit_patch (the patch is ready)')


def test_create_runs_no_tests(ws):
    run('edit.py', ws, **BREAK_TOTAL)
    code, out = run('create.py', ws, target='pay/receipts.py', content='def receipt():\n    return 1\n')
    assert code == 0 and 'CREATED pay/receipts.py' in out and 'regression' not in out.lower(), out


def test_flow_ready_parse_needs_line_start():
    f = flow.parse_facts('check_patch.py', [], 'PROBLEMS:\n  - x.py: NOT READY-looking text\n')
    assert f['ready'] is False
    f = flow.parse_facts('check_patch.py', [], 'changed files:\n  M a.py\nREADY: call submit_patch now.\n')
    assert f['ready'] is True


# ---------------------------------------------------------------- import guard

def test_import_guard_undoes_a_breaking_edit(ws):
    (ws / 'pay/invoice.py').write_text(INVOICE + '\n\nDEFAULT = Invoice(0)\n')
    git(ws, 'commit', '-qam', 'default')
    # v17: deleting class Invoice is now caught earlier by the deleted-definition guard (tests/test_v17_edit.py);
    # a new required argument still breaks `DEFAULT = Invoice(0)` at import time
    code, out = run('edit.py', ws, target='pay/invoice.py', start='7', end='7',
                    new='    def __init__(self, amount, currency):')
    assert code == 1 and out.startswith('NOT APPLIED: after this edit `import pay.invoice` fails'), out
    assert "missing 1 required positional argument: 'currency'" in out
    assert out.rstrip().endswith('CLOSEST: pay/invoice.py 7 7')
    assert status(ws) == ''                         # file left unchanged


def test_import_guard_keeps_edit_when_broken_before(ws):
    (ws / 'pay/broken.py').write_text('import not_installed_anywhere\n\nX = 1\n')
    git(ws, 'add', '-A')
    git(ws, 'commit', '-qm', 'broken')
    code, out = run('edit.py', ws, target='pay/broken.py', old='X = 1', new='X = 2')
    assert code == 0 and out.startswith('EDITED'), out


def test_import_guard_footer(ws):
    (ws / 'pay/invoice.py').write_text(INVOICE + '\n\nDEFAULT = Invoice(0)\n')
    git(ws, 'commit', '-qam', 'default')
    code, out = run('edit.py', ws, flow_on=True, target='pay/invoice.py', start='7', end='7',
                    new='    def __init__(self, amount, currency):')  # v17: see test_import_guard_undoes_a_breaking_edit
    nxt = [l for l in out.splitlines() if l.startswith('NEXT:')]
    assert nxt and 'still imports' in nxt[0] and 'start=7  end=7' in nxt[0], out
    for l in nxt:
        assert_notation(l)


# ---------------------------------------------------------------- property guard

PROP_EDIT = {'target': 'pay/invoice.py', 'old': '    def label(self):',
             'new': '    @property\n    def paid(self):\n        return False\n\n    def label(self):'}


def test_property_guard_refuses(ws):
    code, out = run('edit.py', ws, **PROP_EDIT)
    assert code == 1 and out.startswith("NOT APPLIED: the edit makes 'paid' a read-only @property of Invoice"), out
    assert 'Invoice.__init__ assigns self.paid = ... (line 9)' in out
    assert status(ws) == ''


def test_property_guard_footer_points_at_the_assignment(ws):
    code, out = run('edit.py', ws, flow_on=True, **PROP_EDIT)
    nxt = [l for l in out.splitlines() if l.startswith('NEXT:')]
    assert ('show.py', {'target': 'pay/invoice.py', 'start': '1', 'end': '19'}) in calls(nxt[0]), out
    for l in nxt:
        assert_notation(l)


def test_property_guard_allows_setter_and_new_names(ws):
    with_setter = dict(PROP_EDIT, new=PROP_EDIT['new'].replace(
        '    def label(self):', '    @paid.setter\n    def paid(self, value):\n        self._paid = value\n\n'
                               '    def label(self):'))
    code, out = run('edit.py', ws, **with_setter)
    assert code == 0 and out.startswith('EDITED'), out
    git(ws, 'checkout', '-q', '.')
    code, out = run('edit.py', ws, **dict(PROP_EDIT, new=PROP_EDIT['new'].replace('def paid', 'def is_paid')))
    assert code == 0 and out.startswith('EDITED'), out


def test_property_conflict_unit():
    before = 'class C:\n    def __init__(self):\n        self.x = 1\n'
    after = before + '\n    @property\n    def x(self):\n        return 2\n'
    line, msg = guards.property_conflict(before, after)
    assert line == 3 and "'x' a read-only @property of C" in msg
    # assigning to an existing read-only property is caught the other way round too
    b2 = 'class C:\n    @property\n    def x(self):\n        return 2\n\n    def reset(self):\n        pass\n'
    a2 = b2.replace('pass', 'self.x = 0')
    assert guards.property_conflict(b2, a2)[1].startswith("the edit makes 'x' a read-only @property of C, but "
                                                          'C.reset assigns')
    assert guards.property_conflict(a2, a2 + '\n') is None   # pre-existing conflict: not this edit's


# ---------------------------------------------------------------- escape warnings

def test_escape_warning(ws):
    code, out = run('edit.py', ws, target='pay/invoice.py', old='        return "invoice"',
                    new='        return "a<b".replace("<", "\\u003c")')
    assert code == 0
    warns = [l for l in out.splitlines() if l.startswith('WARNING:')]
    assert any("reads as the character '<' itself" in w and '\\\\u003c' in w for w in warns), out
    assert any(".replace('<', '<') replaces a string with itself" in w for w in warns), out


def test_no_escape_warning_for_escaped_backslash_or_raw():
    before = 'X = 1\n'
    assert guards.escape_warnings(before, 'X = "a".replace("<", "\\\\u003c")\n') == []
    assert guards.escape_warnings(before, 'X = r"\\u003c"\n') == []
    assert guards.escape_warnings(before, 'X = "\\u00e9"\n') == []          # not ASCII: intended
    assert guards.escape_warnings(before, 'X = f"{1}\\x3c"\n')            # f-string middle


# ---------------------------------------------------------------- run.py, clock, bytecode

def test_run_snippet_timeout_is_short(ws):
    r = subprocess.run([sys.executable, '-c', 'import sys; sys.dont_write_bytecode = True; sys.path.insert(0, sys.argv[1]); '
                        'import run; assert run.SNIPPET_TIMEOUT == 15 and run.TIMEOUT == 90; run.SNIPPET_TIMEOUT = 1; '
                        'sys.exit(run.main(["--snippet", "import time; time.sleep(5)"]))', str(SCRIPTS)],
                       capture_output=True, text=True, env=env(ws), timeout=60)
    assert 'timed out after 1s: the code does not return' in r.stdout, r.stdout + r.stderr


def test_clock_is_four_and_a_half_minutes():
    assert flow.TIME_BUDGET_S == 220  # v20b: 4.0 min minus ~20 s before the first call
    assert flow._mmss(270) == '4m30s' and flow._mmss(300) == '5m'


def test_no_bytecode_in_scripts_dir():
    assert not list(SCRIPTS.rglob('__pycache__')) and not list(SCRIPTS.rglob('*.pyc'))
