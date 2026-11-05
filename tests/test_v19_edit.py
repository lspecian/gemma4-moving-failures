"""agent-v19 edit.py robustness (scripts/toolbench.py phase 4: start_only 31% ok / 37% silently wrong, indent_minus4
19% silently wrong in agent-v18d).

1. "start" without "end" and a multi-line NEW (infer_end): the end is inferred and applied with a NOTE when NEW ends
   with a distinctive line of the file, when NEW repeats lines START..END and adds new lines (an insertion), or when
   NEW rewrites START..END line for line; otherwise NOT APPLIED with the likely range, and executing the printed call
   gives the intended edit.
2. A line-range NEW shifted as a whole is moved back to the replaced lines' indentation even when it parses as written
   (not when NEW only re-indents, not inside a docstring).
3. Line numbers that no read explains, after an edit moved the lines NEW matches: shifted with a NOTE.
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

APP = textwrap.dedent('''\
    import email.message
    import inspect
    import json


    class Router:
        def __init__(self, on_startup=None, prefix=""):
            # keep the handlers
            self.on_startup = on_startup or []
            super().__init__()
            self.prefix = prefix
            self.routes = []

        def add(self, path, endpoint):
            if not path.startswith("/"):
                raise ValueError(
                    "paths must start with a slash"
                )
            self.routes.append((self.prefix + path, endpoint))
            return endpoint

        def find(self, path):
            for route, endpoint in self.routes:
                if route == path:
                    return endpoint
            return None


    def describe(router):
        """Describe the router.

            Indented example line.
        """
        names = [route for route, _ in router.routes]
        return ", ".join(names)
    ''')
L = APP.split('\n')


def line_of(text, line, after=0):
    return text.split('\n').index(line, after) + 1


def git(ws, *args):
    return subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / 'repo'
    (root / 'pkg').mkdir(parents=True)
    (root / 'pkg/__init__.py').write_text('')
    (root / 'pkg/app.py').write_text(APP)
    git(root, 'init', '-q')
    git(root, 'add', '-A')
    git(root, 'commit', '-qm', 'baseline')
    return root


def run(script, ws, flow_on=False, **args):
    e = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'), PYTHONDONTWRITEBYTECODE='1')
    e['REPO_TOOLS_FLOW'] = 'on' if flow_on else 'off'
    argv = [x for k, v in args.items() for x in ('--' + k, str(v))]
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *argv], capture_output=True, text=True, env=e,
                       timeout=180)
    assert 'Traceback' not in r.stderr, r.stderr
    return r.returncode, r.stdout


def src(ws):
    return (ws / 'pkg/app.py').read_text()


def with_lines(a, b, new):
    """APP with lines A..B replaced by NEW (a list of lines)."""
    return '\n'.join(L[:a - 1] + new + L[b:])


def ready_calls(out):
    return [args for script, args in calls(out) if script == 'edit.py']


def run_ready(ws, out, new, k=0):
    """Execute the K-th printed edit.py call the way a model would: its template filled with the NEW it sent."""
    args = dict(ready_calls(out)[k])
    args = {key: (new if v == '<your NEW unchanged>' else v) for key, v in args.items()}
    return run('edit.py', ws, **args)


# ---------------------------------------------------------------- 1. start without end

def test_start_only_new_ending_with_a_file_line_replaces_up_to_it(ws):
    n = line_of(APP, '        if not path.startswith("/"):')
    new = ['        if not path or not path.startswith("/"):', '            raise ValueError(',
           '                "paths must start with a slash"']
    code, out = run('edit.py', ws, target='pkg/app.py', start=n, new='\n'.join(new))
    assert code == 0, out
    assert out.startswith(f'NOTE: no end given: NEW ends with line {n + 2}, so it replaced lines {n}-{n + 2}\n'
                          f'EDITED pkg/app.py (lines {n}-{n + 2})'), out
    assert src(ws) == with_lines(n, n + 2, new)


def test_start_only_insertion_keeps_the_lines_after_start(ws):
    """fastapi_14873 (gold): NEW = the line at START, then new lines; the code after START stays."""
    n = line_of(APP, '        self.routes = []')
    new = ['        self.routes = []', '        self.on_shutdown = []', '        self.middleware = None']
    code, out = run('edit.py', ws, target='pkg/app.py', start=n, new='\n'.join(new))
    assert code == 0, out
    assert out.startswith(f'NOTE: no end given: NEW repeats line {n} and adds 2 new lines, so it replaced line {n} only;'
                          ' the lines after it are kept\n'), out
    assert src(ws) == with_lines(n, n, new)


def test_start_only_insertion_after_several_repeated_lines(ws):
    """fastapi_14583: NEW = lines 1-3 unchanged plus an import."""
    new = ['import email.message', 'import inspect', 'import json', 'import copy']
    code, out = run('edit.py', ws, target='pkg/app.py', start=1, new='\n'.join(new))
    assert code == 0 and 'NEW repeats lines 1-3 and adds 1 new line, so it replaced lines 1-3 only' in out, out
    assert src(ws) == with_lines(1, 3, new)


def test_start_only_rewrite_line_for_line(ws):
    n = line_of(APP, '            raise ValueError(')
    new = ['            raise TypeError(', '                "paths must start with a slash: " + path',
           '            )']
    code, out = run('edit.py', ws, target='pkg/app.py', start=n, new='\n'.join(new))
    assert code == 0, out
    assert out.startswith(f'NOTE: no end given: NEW rewrites lines {n}-{n + 2} line for line, so it replaced them\n'), out
    assert src(ws) == with_lines(n, n + 2, new)


def test_start_only_insertion_or_rewrite_is_refused_and_the_call_inserts(ws):
    """fastapi_14099 (gold): `import email.message` + `import functools`: a new sibling or a rewrite of the next
    import? Not applied; the first call inserts, the second replaces the next line too."""
    new = 'import email.message\nimport functools'
    code, out = run('edit.py', ws, target='pkg/app.py', start=1, new=new)
    assert code == 2 and src(ws) == APP, out
    assert out.startswith('NOT APPLIED: "start" without "end" and a NEW of 2 lines'), out
    assert 'lines 1-1, NEW repeats line 1 and the rest is new (an insertion after line 1):' in out
    assert [(c['start'], c['end']) for c in ready_calls(out)] == [('1', '1'), ('1', '2')], out
    assert_notation(out)
    code, out = run_ready(ws, out, new)
    assert code == 0 and src(ws) == with_lines(1, 1, new.split('\n')), out


def test_start_only_swap_is_refused_with_the_longer_range(ws):
    """fastapi_14873 v16: NEW moves super().__init__() above the assignment; ending at the assignment would leave
    the old super().__init__() below it (a duplicate)."""
    n = line_of(APP, '        # keep the handlers')
    new = '        super().__init__()\n        self.on_startup = on_startup or []'
    code, out = run('edit.py', ws, target='pkg/app.py', start=n, new=new)
    assert code == 2 and src(ws) == APP, out
    assert ready_calls(out)[0]['end'] == str(n + 2), out
    assert out.rstrip().endswith(f'CLOSEST: pkg/app.py {n} {n + 2}')
    code, out = run_ready(ws, out, new)
    assert code == 0 and src(ws) == with_lines(n, n + 2, new.split('\n')), out


def test_start_only_unclear_rewrite_prints_the_best_range(ws):
    n = line_of(APP, '        for route, endpoint in self.routes:')
    new = '        matches = [e for r, e in self.routes if r == path]\n        return matches[0] if matches else None'
    code, out = run('edit.py', ws, target='pkg/app.py', start=n, new=new)
    assert code == 2 and src(ws) == APP, out
    assert f'lines {n}-{n + 3}, the lines NEW matches best:' in out, out
    code, out = run_ready(ws, out, new)
    assert code == 0 and src(ws) == with_lines(n, n + 3, new.split('\n')), out


def test_start_only_one_line_equal_to_the_next_line_deletes_start(ws):
    n = line_of(APP, '        # keep the handlers')
    code, out = run('edit.py', ws, target='pkg/app.py', start=n, new='        self.on_startup = on_startup or []')
    assert code == 0 and f'NOTE: no end given: NEW is line {n + 1}, so it replaced lines {n}-{n + 1} (line {n} deleted)' \
        in out, out
    assert src(ws) == with_lines(n, n, [])


def test_start_only_one_line_for_a_multiline_statement_is_inferred(ws):
    # v19 refused this with the statement's range; v20 replaces the simple statement that begins at START (E060
    # fastapi_13207: the model's next call used exactly that range)
    n = line_of(APP, '            raise ValueError(')
    new = '            raise ValueError("paths must start with a slash")'
    code, out = run('edit.py', ws, target='pkg/app.py', start=n, new=new)
    assert code == 0, out
    assert out.startswith(f'NOTE: no end given: NEW rewrites the statement at line {n}, lines {n}-{n + 2}\n'), out
    assert src(ws) == with_lines(n, n + 2, [new])


def test_start_only_one_line_for_a_compound_statement_is_still_refused(ws):
    n = line_of(APP, '        for route, endpoint in self.routes:')
    new = '        endpoint = dict(self.routes).get(path)'
    code, out = run('edit.py', ws, target='pkg/app.py', start=n, new=new)
    assert code == 2 and src(ws) == APP, out
    assert out.startswith(f'NOT APPLIED: "start" without "end": NEW is one line, and line {n} alone cannot be replaced '
                          'by it'), out


def test_start_only_one_line_still_replaces_one_line(ws):
    n = line_of(APP, '        self.prefix = prefix')
    code, out = run('edit.py', ws, target='pkg/app.py', start=n, new='        self.prefix = prefix.rstrip("/")')
    assert code == 0 and out.startswith(f'NOTE: no end given: replaced line {n} only\nEDITED'), out


def test_start_only_one_line_without_indentation_still_replaces_one_line(ws):
    """A one-line NEW pasted without indentation: still that one line, re-indented by the edit (not a refusal)."""
    n = line_of(APP, '        self.prefix = prefix')
    code, out = run('edit.py', ws, target='pkg/app.py', start=n, new='self.prefix = prefix.rstrip("/")')
    assert code == 0 and out.startswith(f'NOTE: no end given: replaced line {n} only\nEDITED'), out
    assert src(ws) == with_lines(n, n, ['        self.prefix = prefix.rstrip("/")'])


def test_infer_end_unit():
    n = line_of(APP, '        self.routes = []')
    assert edit.infer_end(APP, n, [L[n - 1], '        self.x = 1'], True)[:2] == (n, True)
    m = line_of(APP, '        if not path.startswith("/"):')
    assert edit.infer_end(APP, m, ['        if not path:', L[m], L[m + 1]], True)[:2] == (m + 2, True)
    # a NEW that would not parse anywhere: never sure
    assert edit.infer_end(APP, m, ['        if not path:', '            raise ('], True)[1] is False


# ---------------------------------------------------------------- 2. indentation shifted as a whole

def test_line_range_new_indented_4_less_is_moved_back(ws):
    """toolbench indent_minus4: the dedented block still parses (it lands in the enclosing block)."""
    a, b = line_of(APP, '        for route, endpoint in self.routes:'), line_of(APP, '        return None')
    want = ['        for route, endpoint in self.routes:', '            if route == path or route == path + "/":',
            '                return endpoint', '        return None']
    new = '\n'.join(l[4:] for l in want)
    code, out = run('edit.py', ws, target='pkg/app.py', start=a, end=b, new=new)
    assert code == 0 and f"EDITED pkg/app.py (lines {a}-{b}) (indentation repaired: +4 to the replaced lines' " \
        'indentation)' in out, out
    assert src(ws) == with_lines(a, b, want)


def test_line_range_new_indented_4_more_is_moved_back(ws):
    n = line_of(APP, '        self.routes.append((self.prefix + path, endpoint))')
    want = ['        self.routes.append((self.prefix + path.lower(), endpoint))', '        return endpoint']
    code, out = run('edit.py', ws, target='pkg/app.py', start=n, end=n + 1, new='\n'.join('    ' + l for l in want))
    assert code == 0 and '(indentation repaired: -4' in out, out
    assert src(ws) == with_lines(n, n + 1, want)


def test_line_range_reindent_only_is_kept_as_written(ws):
    """numpy_10951 (gold): moving `return` into the loop changes only its indentation: deliberate."""
    n = line_of(APP, '        return None')
    code, out = run('edit.py', ws, target='pkg/app.py', start=n, end=n, new='            return None')
    assert code == 0 and 'indentation repaired' not in out, out
    assert src(ws) == with_lines(n, n, ['            return None'])


def test_insertion_at_a_blank_line_dedented_below_the_next_line_is_moved(ws):
    """A blank line replaced by a method written at module level: it would end the class (the next method's body
    would follow it); moved to the next line's indentation."""
    k = line_of(APP, '    def find(self, path):') - 1  # the blank line before find()
    new = '\ndef helper(self):\n    return len(self.routes)\n\n'
    code, out = run('edit.py', ws, target='pkg/app.py', start=k, end=k, new=new)
    assert code == 0 and f"EDITED pkg/app.py (lines {k}-{k}) (indentation repaired: +4 to the replaced lines' " \
        'indentation)' in out, out
    assert src(ws) == with_lines(k, k, ['', '    def helper(self):', '        return len(self.routes)', ''])


def test_no_shift_inside_a_docstring(ws):
    n = line_of(APP, '        Indented example line.')
    code, out = run('edit.py', ws, target='pkg/app.py', start=n, end=n, new='    Example line.')
    assert code == 0 and 'indentation repaired' not in out, out
    assert src(ws) == with_lines(n, n, ['    Example line.'])


def test_uniform_shift_unit():
    region = ['        if a:', '            b()']
    assert edit.uniform_shift('    if a:\n        b(1)', region) == -4
    assert edit.uniform_shift('            if a:\n                b(1)', region) == 4
    assert edit.uniform_shift('    if a:\n        b()', region) == 0                # indentation only: deliberate
    assert edit.uniform_shift('        if a:\n        b(1)', region) == 0            # first line right
    assert edit.uniform_shift('    x = 1\n    if a:\n        b()\n', region) == -4      # a line added, all shifted
    # NEW's first line is new (deeper: a continuation): the line it keeps tells the shift
    assert edit.uniform_shift('            extra=1,\n    )\n    y = 3', ['        )', '        y = 2']) == -4
    assert edit.uniform_shift('            extra=1,\n        )\n        y = 3', ['        )', '        y = 2']) == 0
    # a blank line replaced: compared with the next line
    lines = ['class A:', '    x = 1', '', '    def f(self):', '        pass']
    assert edit.uniform_shift('def g(self):\n    pass', [''], lines, 3) == -4
    assert edit.uniform_shift('    def g(self):\n        pass', [''], lines, 3) == 0
    assert edit.uniform_shift('        y = 2', [''], lines, 3) == 0                 # deeper than the next line: kept


# ---------------------------------------------------------------- 3. stale numbers without a read

def test_stale_numbers_without_a_read_follow_the_moved_lines(ws):
    """toolbench stale_lines: the lines were never shown (show.py truncated the function), an edit above moved them,
    and NEW matches the moved lines, not the ones now at those numbers."""
    a = line_of(APP, '    def add(self, path, endpoint):')
    code, out = run('edit.py', ws, flow_on=True, target='pkg/app.py', start=a, end=a,
                    new='    # adds a route\n    # (one per path)\n' + L[a - 1])
    assert code == 0, out
    n = line_of(APP, '        self.routes.append((self.prefix + path, endpoint))')
    new = '        self.routes.append((self.prefix + path.rstrip("/"), endpoint))'
    code, out = run('edit.py', ws, flow_on=True, target='pkg/app.py', start=n, end=n, new=new)
    assert code == 0 and f'used {n + 2}' in out and f'EDITED pkg/app.py (lines {n + 2}-{n + 2})' in out, out
    assert new + '\n' in src(ws) and '        self.routes.append((self.prefix + path, endpoint))' not in src(ws)


def test_stale_numbers_without_a_read_kept_when_new_fits_the_current_lines(ws):
    a = line_of(APP, '    def add(self, path, endpoint):')
    run('edit.py', ws, flow_on=True, target='pkg/app.py', start=a, end=a,
        new='    # adds a route\n    # (one per path)\n' + L[a - 1])
    n = line_of(APP, '        self.routes.append((self.prefix + path, endpoint))')
    new = '        self.routes.append((self.prefix + path, endpoint, None))'
    code, out = run('edit.py', ws, flow_on=True, target='pkg/app.py', start=n + 2, end=n + 2, new=new)
    assert code == 0 and 'used ' not in out, out


def test_stale_numbers_that_only_parse_where_the_lines_moved_are_shifted(ws):
    """fastapi_12942 (toolbench stale_lines): NEW is about as similar to both numberings, but only the moved lines
    keep the file parsing: the current numbers are used."""
    code, out = run('show.py', ws, flow_on=True, target='pkg/app.py', start=1, end=40)
    assert code == 0, out
    a = line_of(APP, '    def add(self, path, endpoint):')
    code, out = run('edit.py', ws, flow_on=True, target='pkg/app.py', start=a, end=a,
                    new='    # adds a route\n    # (one per path)\n' + L[a - 1])
    assert code == 0, out
    n = line_of(APP, '            raise ValueError(')
    new = '            return None'  # similarity 0.11 to the lines now at n..n+2, 0.19 to the moved ones: unclear
    code, out = run('edit.py', ws, flow_on=True, target='pkg/app.py', start=n, end=n + 2, new=new)
    assert code == 0 and f'used the current numbers {n + 2}-{n + 4}' in out, out
    assert 'ValueError' not in src(ws) and '    def add(self, path, endpoint):\n' in src(ws)


def test_moved_fits_unit():
    lines = APP.split('\n')
    n = line_of(APP, '            raise ValueError(')
    nl = ['            raise TypeError(path)']
    assert edit.moved_fits('pkg/app.py', lines, nl, (n - 1, n + 1), (n, n + 2), True) is True    # only moved parses
    assert edit.moved_fits('x.txt', lines, [L[n - 1], 'x', L[n + 1]], (n - 1, n + 1), (n, n + 2), True) is True  # ends
    assert edit.moved_fits('x.txt', lines, [L[n - 2], 'x', L[n]], (n - 1, n + 1), (n, n + 2), True) is False
    assert edit.moved_fits('x.txt', lines, ['x'], (n - 1, n + 1), (n, n + 2), True) is None


def test_no_syntax_warnings_on_stderr(ws):
    """A file with an invalid escape ("\\d") made every parse print a SyntaxWarning to stderr (14 lines for one v19
    start-only edit), and stderr reaches the model."""
    (ws / 'pkg/esc.py').write_text('import re\nP = re.compile("\\d+")\n\n\ndef f():\n    a = 1\n    b = 2\n    return a + b\n')
    e = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_FLOW='off', PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, str(SCRIPTS / 'edit.py'), '--target', 'pkg/esc.py', '--start', '6', '--new',
                        '    a = 10\n    b = 20'], capture_output=True, text=True, env=e, timeout=180)
    assert r.returncode == 0 and 'NEW rewrites lines 6-7 line for line' in r.stdout, r.stdout
    assert r.stderr == '', r.stderr
