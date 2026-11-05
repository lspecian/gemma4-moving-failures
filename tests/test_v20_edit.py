"""agent-v20 edit.py (research/notes/15-edit-failures.md: 211 failed edit calls of E057-E061 replayed with
scripts/edit_failures.py).

1. Range form: literal \\n outside strings decoded (mixed escapes); the end inference accepts an end where NEW fits
   once moved back by its uniform indentation shift; bare decorators are not "repeated lines"; a NEW that begins with
   the line above START starts there; a simple multi-line statement at START; new code at a blank line is inserted;
   end = start - 1 inserts; no old/start: NEW's first line or the def it redefines locates it; a range whose result
   does not parse is repaired when NEW's own first/last lines name the range; compile errors are refused.
2. Text form drift: blank lines miscounted; OLD anchored on its first and last lines (NEW's changes merged, drifted
   lines kept); NEW already in the file; ambiguous one-line OLD prints a ready multi-edit call.
3. Multi-edit: start2/end2/new2, old2/new2 ... (all hunks against the file before the call, overlap refused, all or
   nothing, one diff, one CHECK line).
4. CHECK line: new undefined names, import status (an import failing without the edit too is not mentioned).
Run with scripts/test_skills.sh (sandbox image, py3.13) or uvx pytest.
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
import argfix  # noqa: E402
import edit  # noqa: E402
import flow  # noqa: E402
import guards  # noqa: E402

APP = textwrap.dedent('''\
    import os
    from typing import (
        Any,
        Dict,
    )


    class FastError(RuntimeError):
        """
        A generic error.
        """


    class ScopeError(FastError):
        pass


    class Router:
        def __init__(self, prefix=""):
            self.prefix = prefix
            self.routes = []

        @property
        def size(self):
            return len(self.routes)

        @property
        def empty(self):
            return not self.routes

        def add(self, path, endpoint):
            if not path:
                raise ValueError("empty path")
            self.routes.append((self.prefix + path, endpoint))
            return endpoint

        def find(self, path):
            for route, endpoint in self.routes:
                if route == path:
                    return endpoint
            return None


    def describe(router):
        names = [
            route
            for route, _ in router.routes
        ]
        return ", ".join(names)


    def render(router):
        text = describe(router)
        return text.upper()
    ''')
L = APP.split('\n')


def n_of(line, after=0):
    return L.index(line, after) + 1


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


def run(ws, flow_on=False, **args):
    e = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'), PYTHONDONTWRITEBYTECODE='1')
    e['REPO_TOOLS_FLOW'] = 'on' if flow_on else 'off'
    argv = [x for k, v in args.items() for x in ('--' + k, str(v))]
    r = subprocess.run([sys.executable, str(SCRIPTS / 'edit.py'), *argv], capture_output=True, text=True, env=e,
                       timeout=180)
    assert 'Traceback' not in r.stderr, r.stderr
    return r.returncode, r.stdout


def src(ws):
    return (ws / 'pkg/app.py').read_text()


def with_lines(a, b, new):
    return '\n'.join(L[:a - 1] + new + L[b:])


# ---------------------------------------------------------------- 1. range form

def test_decode_breaks_only_outside_strings():
    assert edit.decode_breaks('def f(\n    a,\\n    b="x\\ny",\\n):') == 'def f(\n    a,\n    b="x\\ny",\n):'
    assert edit.decode_breaks('sep = "\\n"') is None                       # inside a string: kept
    assert edit.decode_breaks('raise E(\\"x\\")\\nreturn') is None           # escaped quotes: the whole text is escaped
    assert edit.decode_breaks('a = 1  # one\\nb = 2') == 'a = 1  # one\nb = 2'
    assert edit.new_lines('    x = 1\n    y = 2,\\n    z') == ['    x = 1', '    y = 2,', '    z']


def test_start_only_new_indented_two_more_is_inferred_and_moved_back(ws):
    """E058 rich_3471/rich_3772/fastapi_14297: NEW redefines the def at START, indented 2 more than the file."""
    s = n_of('    def find(self, path):')
    new = ['      def find(self, path):', '          for route, endpoint in self.routes:',
           '              if route == path or route == "/" + path:', '                  return endpoint',
           '          return None']
    code, out = run(ws, target='pkg/app.py', start=s, new='\n'.join(new))
    assert code == 0, out
    assert f'NOTE: no end given: NEW redefines find, so it replaced its lines {s}-{s + 4}' in out
    assert src(ws) == with_lines(s, s + 4, [l[2:] for l in new])


def test_start_only_with_mixed_escapes(ws):
    """E058 fastapi_13713: real line breaks and, on one line, literal \\n escapes."""
    s = n_of('    def add(self, path, endpoint):')
    new = ('    def add(self, path, endpoint):\n        if not path:\\n            raise ValueError("no path")\n'
           '        self.routes.append((self.prefix + path, endpoint))\n        return endpoint')
    code, out = run(ws, target='pkg/app.py', start=s, new=new)
    assert code == 0, out
    assert src(ws) == with_lines(s, s + 4, ['    def add(self, path, endpoint):', '        if not path:',
                                            '            raise ValueError("no path")',
                                            '        self.routes.append((self.prefix + path, endpoint))',
                                            '        return endpoint'])


def test_start_only_property_rewrite_ignores_the_next_property_decorator(ws):
    """E058 rich_3675: `@property` right after the end counted as a line NEW repeats (the range ran 2 lines on)."""
    s = n_of('    @property')
    new = ['    @property', '    def size(self):', '        return len(list(self.routes))']
    code, out = run(ws, target='pkg/app.py', start=s, new='\n'.join(new))
    assert code == 0 and src(ws) == with_lines(s, s + 2, new), out


def test_start_on_the_def_under_its_decorator_starts_at_the_decorator(ws):
    """E061 fastapi_9555: start at `def`, NEW begins with the decorator above it (it was duplicated)."""
    s = n_of('    def empty(self):')
    new = ['    @property', '    def empty(self):', '        return len(self.routes) == 0']
    code, out = run(ws, target='pkg/app.py', start=s, new='\n'.join(new))
    assert code == 0, out
    assert f'NEW starts with line {s - 1} (@property), the line above start={s}' in out
    assert src(ws) == with_lines(s - 1, s + 1, new)


def test_start_only_one_line_for_a_multi_line_statement(ws):
    """E060 fastapi_13207: a one-line NEW for a statement spanning several lines."""
    s = n_of('    names = [')
    code, out = run(ws, target='pkg/app.py', start=s, new='    names = [route for route, _ in router.routes]')
    assert code == 0, out
    assert f'NEW rewrites the statement at line {s}, lines {s}-{s + 3}' in out
    assert src(ws) == with_lines(s, s + 3, ['    names = [route for route, _ in router.routes]'])


def test_start_only_new_code_at_a_blank_line_is_inserted(ws):
    """E061 fastapi_14306, E060 fastapi_15280: a new method at the blank line between two methods."""
    s = n_of('', n_of('        return endpoint'))
    new = ['    def clear(self):', '        self.routes = []']
    code, out = run(ws, target='pkg/app.py', start=s, new='\n'.join(new))
    assert code == 0, out
    assert f'NEW is new code and line {s} is blank, so NEW was inserted there' in out
    assert src(ws) == with_lines(s, s, [''] + new + [''])


def test_end_before_start_by_one_inserts(ws):
    """E059 fastapi_15280: start=N end=N-1 (an empty range): NEW goes before line N."""
    s = n_of('def render(router):')
    new = ['def count(router):', '    return len(router.routes)', '', '']
    code, out = run(ws, target='pkg/app.py', start=s, end=s - 1, new='\n'.join(new) + '\n')
    assert code == 0 and f'end = start - 1: NEW was inserted before line {s}' in out, out
    assert src(ws) == with_lines(s, s - 1, new)


def test_no_old_no_start_new_starting_with_a_unique_line(ws):
    """E057/E061 fastapi_14964: NEW = an existing import + a new one, no old/start."""
    code, out = run(ws, target='pkg/app.py', new='import os\nimport sys')
    assert code == 0, out
    assert 'no old or start given; NEW starts with line 1 of pkg/app.py' in out
    assert src(ws) == with_lines(1, 1, ['import os', 'import sys'])


def test_no_old_no_start_new_redefining_a_unique_function(ws):
    """E061 fastapi_13786: NEW redefines a function the file defines once."""
    s = n_of('def render(router):')
    new = ['def render(router):', '    return describe(router).lower()']
    code, out = run(ws, target='pkg/app.py', new='\n'.join(new))
    assert code == 0 and f'no old or start given; NEW starts with line {s} of pkg/app.py' in out, out
    assert 'NEW redefines render, so it replaced its lines' in out
    assert src(ws) == with_lines(s, s + 2, new)
    (ws / 'pkg/app.py').write_text(APP)
    new = ['def render(router, upper=True):', '    return describe(router)']
    code, out = run(ws, target='pkg/app.py', new='\n'.join(new))
    assert code == 0 and f'NEW redefines render, which begins at line {s}' in out, out
    assert src(ws) == with_lines(s, s + 2, new)


def test_no_old_no_start_unknown_place_is_still_a_usage_error(ws):
    code, out = run(ws, target='pkg/app.py', new='x = 1\ny = 2')
    assert code == 2 and out.startswith('USAGE ERROR: edit.py needs the key old'), out
    assert src(ws) == APP


def test_range_repair_end_includes_a_line_new_lacks(ws):
    """E059 fastapi_11355: end on the import list's `)`, NEW without it."""
    code, out = run(ws, target='pkg/app.py', start=3, end=5, new='    Any,\n    Dict,\n    List,')
    assert code == 0, out
    assert 'lines 3-5 as given do not parse with NEW;' in out and 'so the edit replaced lines 3-4' in out
    assert src(ws) == with_lines(3, 4, ['    Any,', '    Dict,', '    List,'])


def test_range_repair_new_includes_the_line_after_end(ws):
    """E059 fastapi_14583: NEW ends with the `)` below END (it would be duplicated)."""
    code, out = run(ws, target='pkg/app.py', start=4, end=4, new='    Dict,\n    List,\n)')
    assert code == 0 and 'NEW ends with line 5, so the edit replaced lines 4-5' in out, out
    assert src(ws) == with_lines(4, 5, ['    Dict,', '    List,', ')'])


def test_range_repair_closing_lines_kept_and_new_inserted_after(ws):
    """E059/E060 fastapi_15030: the range is the `]` of a statement, NEW is a new statement."""
    s = n_of('    ]')
    code, out = run(ws, target='pkg/app.py', start=s, end=s, new='    names = sorted(names)')
    assert code == 0, out
    assert f'so the edit inserted NEW after line {s} and replaced nothing' in out
    assert src(ws) == with_lines(s + 1, s, ['    names = sorted(names)'])


def test_range_that_parses_is_never_repaired(ws):
    s = n_of('    return text.upper()')
    code, out = run(ws, target='pkg/app.py', start=s, end=s, new='    return text.lower()')
    assert code == 0 and 'as given do not parse' not in out, out


def test_unrepairable_range_is_still_refused(ws):
    s = n_of('        for route, endpoint in self.routes:')
    code, out = run(ws, target='pkg/app.py', start=s, end=s, new='        while True')
    assert code == 1 and out.startswith('NOT APPLIED: the result would not parse'), out
    assert src(ws) == APP


def test_compile_error_is_refused_before_writing(ws):
    """E061 fastapi_14266: `keyword argument repeated` parses (ast) but does not compile."""
    s = n_of('            raise ValueError("empty path")')
    code, out = run(ws, target='pkg/app.py', start=s, end=s, new='            raise ValueError(x=1, x=2)')
    assert code == 1 and 'NOT APPLIED: the result would not compile' in out, out
    assert src(ws) == APP


# ---------------------------------------------------------------- 2. text-form drift

def test_old_with_one_blank_line_where_the_file_has_two(ws):
    """E057 fastapi_14372: OLD with 1 blank line between the classes, the file has 2: matched, the 2 kept."""
    old = 'class FastError(RuntimeError):\n    """\n    A generic error.\n    """\n\nclass ScopeError(FastError):'
    new = ('class FastError(RuntimeError):\n    """\n    A generic error.\n    """\n\n'
           'class ScopeError(FastError, KeyError):')
    code, out = run(ws, target='pkg/app.py', old=old, new=new)
    assert code == 0 and '(blank lines ignored)' in out and 'except for the number of blank lines' in out, out
    assert src(ws) == APP.replace('class ScopeError(FastError):', 'class ScopeError(FastError, KeyError):')


def test_blank_line_count_changed_on_purpose_is_kept(ws):
    """E058 requests_7315: OLD 1 blank (file 2), NEW 0 blank: NEW's count wins."""
    old = '    pass\n\nclass Router:'
    new = '    pass\nclass Router:'
    code, out = run(ws, target='pkg/app.py', old=old, new=new)
    assert code == 0, out
    assert '    pass\nclass Router:' in src(ws)


def test_old_with_a_summarised_docstring_is_merged_on_its_anchors(ws):
    """E057 fastapi_14605 / rich_3882: OLD invented `pass` where the class has a docstring; NEW inserts a class."""
    old = 'class FastError(RuntimeError):\n    pass\n\n\nclass ScopeError(FastError):'
    new = ('class FastError(RuntimeError):\n    pass\n\n\nclass FastWarning(UserWarning):\n    pass\n\n\n'
           'class ScopeError(FastError):')
    code, out = run(ws, target='pkg/app.py', old=old, new=new)
    assert code == 0 and "anchored on OLD's first and last lines" in out, out
    assert 'the lines OLD had differently were kept as they are' in out
    assert src(ws) == APP.replace('class ScopeError(FastError):',
                                  'class FastWarning(UserWarning):\n    pass\n\n\nclass ScopeError(FastError):')
    assert '    A generic error.' in src(ws)


def test_anchor_merge_refused_when_a_changed_line_is_not_in_the_file(ws):
    old = 'class FastError(RuntimeError):\n    pass\n\n\nclass ScopeError(FastError):'
    new = 'class FastError(RuntimeError):\n    x = 1\n\n\nclass ScopeError(FastError):'
    code, out = run(ws, target='pkg/app.py', old=old, new=new)
    assert code == 1 and out.startswith('NOT APPLIED: OLD not found'), out
    assert src(ws) == APP


def test_new_already_in_the_file(ws):
    """E058 requests_7309: the model resent an applied edit (OLD = the old code, NEW = what the file has)."""
    old = '    def find(self, path):\n        for route, endpoint in self.ROUTES:\n            if route == path:'
    new = '    def find(self, path):\n        for route, endpoint in self.routes:\n            if route == path:'
    code, out = run(ws, target='pkg/app.py', old=old, new=new)
    assert code == 0 and out.startswith('NOT CHANGED: OLD is not in pkg/app.py, but NEW already is'), out
    assert src(ws) == APP


def test_ambiguous_one_line_old_prints_a_ready_multi_edit_that_works(ws):
    a = n_of('    @property')
    code, out = run(ws, target='pkg/app.py', old='    @property', new='    @property  # read-only')
    assert code == 1 and 'OLD occurs 2 times' in out, out
    ready = [args for script, args in calls(out) if script == 'edit.py' and 'start2' in args]
    assert ready and ready[0]['start'] == str(a) and ready[0]['start2'] == str(n_of('    @property', a)), out
    assert_notation(out)
    code, out = run(ws, **ready[0])
    assert code == 0, out
    assert src(ws) == APP.replace('    @property', '    @property # read-only')  # (notation: one space)


def test_fuzzy_matches_every_decoding(ws):
    """E058 fastapi_14986: a "\\n" inside a string of OLD was decoded with the line breaks before the near-identical
    match ran; each decoding is tried now."""
    text = 'def f():\n    a = 1\n    s = "x\\n"\n    b = 2\n    c = 3\n'
    (ws / 'pkg/f.py').write_text(text)
    old = 'def f():\\n    a = 1\\n    s = "x\\\\n"\\n    b = 2\\n    c = 33'  # one char off, escaped
    found = edit.locate(text, old)
    assert found is not None and 'near-identical' in found[4]


# ---------------------------------------------------------------- 3. multi-edit

def test_multi_edit_three_hunks_mixed(ws):
    s1, s2 = n_of('        self.prefix = prefix'), n_of('        return endpoint')
    code, out = run(ws, target='pkg/app.py', start=s1, end=s1, new='        self.prefix = prefix.rstrip("/")',
                    start2=s2, end2=s2, new2='        return endpoint or None',
                    old3='    return text.upper()', new3='    return text.title()')
    assert code == 0, out
    assert out.startswith('EDITED pkg/app.py (3 hunks: ')
    assert out.count('\n--- a/pkg/app.py') == 1 and out.count('CHECK:') == 1
    want = with_lines(s1, s1, ['        self.prefix = prefix.rstrip("/")'])
    want = '\n'.join(want.split('\n')[:s2 - 1] + ['        return endpoint or None'] + want.split('\n')[s2:])
    assert src(ws) == want.replace('    return text.upper()', '    return text.title()')


def test_multi_edit_line_numbers_refer_to_the_file_before_the_call(ws):
    """Hunk 1 adds lines above hunk 2: hunk 2's numbers are still the old ones."""
    s1, s2 = 1, n_of('def render(router):')
    code, out = run(ws, target='pkg/app.py', start=s1, end=s1, new='import os\nimport re\nimport sys',
                    start2=s2 + 1, end2=s2 + 1, new2='    text = describe(router).strip()')
    assert code == 0, out
    assert '    text = describe(router).strip()' in src(ws) and src(ws).startswith('import os\nimport re\nimport sys\n')


def test_multi_edit_overlap_is_refused_and_nothing_changes(ws):
    s = n_of('        self.prefix = prefix')
    code, out = run(ws, target='pkg/app.py', start=s, end=s + 1, new='        pass',
                    start2=s + 1, end2=s + 1, new2='        self.routes = {}')
    assert code == 1 and 'overlap, so nothing was changed' in out, out
    assert src(ws) == APP


def test_multi_edit_one_failing_hunk_applies_nothing(ws):
    s = n_of('        self.prefix = prefix')
    code, out = run(ws, target='pkg/app.py', start=s, end=s, new='        self.prefix = prefix or "/"',
                    old2='this text is nowhere in the file at all', new2='x = 1')
    assert code == 1 and 'NOT APPLIED: hunk 2 of 2 (old2/new2) cannot be applied, so nothing was changed' in out, out
    assert src(ws) == APP


def test_multi_edit_hunks_that_only_parse_together(ws):
    """Two hunks that each break the syntax alone but fix it together are refused per hunk (each must parse)."""
    s = n_of('    def find(self, path):')
    code, out = run(ws, target='pkg/app.py', start=s, end=s, new='    def find(self, path):\n        try:',
                    start2=s + 4, end2=s + 4, new2='        except KeyError:\n            pass\n        return None')
    assert code == 1 and src(ws) == APP, out


def test_multi_edit_start_without_end_in_a_later_hunk(ws):
    s1, s2 = n_of('        self.prefix = prefix'), n_of('    def find(self, path):')
    new2 = ['    def find(self, path):', '        for route, endpoint in reversed(self.routes):',
            '            if route == path:', '                return endpoint', '        return None']
    code, out = run(ws, target='pkg/app.py', start=s1, end=s1, new='        self.prefix = prefix or ""',
                    start2=s2, new2='\n'.join(new2))
    assert code == 0 and 'hunk 2: no end given: NEW ' in out, out
    assert 'reversed(self.routes)' in src(ws) and 'prefix or ""' in src(ws)


def test_multi_edit_too_many_hunks(ws):
    args = {'target': 'pkg/app.py', 'start': 1, 'end': 1, 'new': 'import os'}
    for k in range(2, 6):
        args.update({f'start{k}': 1, f'end{k}': 1, f'new{k}': 'import os'})
    code, out = run(ws, **{k: v for k, v in args.items() if not k.endswith('5')})
    assert code in (0, 1)
    args['start5'] = 2  # (start5..new5 are not keys at all)
    code, out = run(ws, **args)
    assert code == 2 and out.startswith('USAGE ERROR: unknown key'), out


def test_multi_edit_keys_are_hidden_from_usage_messages():
    assert 'start2' in argfix.KEYS['edit.py'] and 'old4' in argfix.KEYS['edit.py']
    assert argfix.shown_keys('edit.py') == ['target', 'old', 'new', 'start', 'end']
    a = flow.action_key('edit.py', ['--target', 'a.py', '--start', '1', '--end', '1', '--new', 'x'])
    b = flow.action_key('edit.py', ['--target', 'a.py', '--start', '1', '--end', '1', '--new', 'x', '--start2', '3',
                                    '--end2', '3', '--new2', 'y'])
    assert a != b  # a multi-edit is not a repeat of its first hunk


# ---------------------------------------------------------------- 4. CHECK line

def test_check_line_clean_edit(ws):
    s = n_of('    return text.upper()')
    code, out = run(ws, target='pkg/app.py', start=s, end=s, new='    return text.lower()')
    assert code == 0, out
    check = [l for l in out.splitlines() if l.startswith('CHECK:')]
    assert check == ['CHECK: compiles; no new undefined names; import pkg.app OK'], out


def test_check_line_reports_a_new_undefined_name(ws):
    s = n_of('    return text.upper()')
    code, out = run(ws, target='pkg/app.py', start=s, end=s, new='    return textwrap.indent(text, "  ")')
    assert code == 0 and f"CHECK: line {s}: undefined name 'textwrap'" in out, out


def test_check_line_reports_a_removed_import_still_used(ws):
    (ws / 'pkg/b.py').write_text('import json\n\n\ndef dump(x):\n    return json.dumps(x)\n')
    code, out = run(ws, target='pkg/b.py', start=1, end=1, new='import sys')
    assert code == 0, out
    assert "CHECK: line 5: undefined name 'json' (the edit removed its definition or import)" in out, out


def test_check_line_ignores_names_undefined_before_the_edit(ws):
    (ws / 'pkg/c.py').write_text('def f():\n    return magic + 1\n\n\ndef g():\n    return 2\n')
    code, out = run(ws, target='pkg/c.py', start=6, end=6, new='    return magic * 2')
    assert code == 0 and 'no new undefined names' in out, out


def test_check_line_quiet_when_the_import_fails_without_the_edit_too(ws):
    (ws / 'pkg/d.py').write_text('import not_installed_anywhere\n\n\ndef f():\n    return 1\n')
    code, out = run(ws, target='pkg/d.py', start=5, end=5, new='    return 2')
    assert code == 0, out
    assert [l for l in out.splitlines() if l.startswith('CHECK:')] == ['CHECK: compiles; no new undefined names'], out


def test_undefined_names_unit():
    assert guards.undefined_names('import os\n\ndef f(a):\n    b = a\n    return os.sep + b + c\n') == {'c': 5}
    assert guards.undefined_names('def f():\n    return [x for x in range(3)]\n') == {}
    assert guards.undefined_names('class A:\n    X = 1\n    def f(self):\n        return X\n') == {'X': 4}
    assert guards.undefined_names('from m import *\n\ndef f():\n    return q\n') == {}
    assert guards.undefined_names('def f():\n    global G\n    G = 1\n\ndef g():\n    return G\n') == {}
    assert guards.undefined_names('try:\n    import a\nexcept ImportError:\n    a = None\nprint(a, __name__)\n') == {}
    assert guards.undefined_names('def f(:\n') is None
    assert guards.new_undefined_names('x = 1\n', 'x = 1\ny = z\n') == [(2, 'z', True)]


def test_multi_edit_under_flow_records_one_successful_edit(ws):
    s1, s2 = n_of('        self.prefix = prefix'), n_of('    return text.upper()')
    code, out = run(ws, flow_on=True, target='pkg/app.py', start=s1, end=s1, new='        self.prefix = prefix or ""',
                    start2=s2, end2=s2, new2='    return text.title()')
    assert code == 0 and 'EDITED pkg/app.py (2 hunks: ' in out, out
    assert any(l.startswith('NEXT:') for l in out.splitlines()), out
    e = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(ws.parent / 'state'), PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, str(SCRIPTS / 'status.py')], capture_output=True, text=True, env=e, timeout=60)
    assert 'pkg/app.py' in r.stdout and f'L{s1}' in r.stdout and f'L{s2}' in r.stdout, r.stdout  # both ranges
    code, again = run(ws, flow_on=True, target='pkg/app.py', start=s1, end=s1, new='        self.prefix = prefix or ""',
                      start2=s2, end2=s2, new2='    return text.title()')
    assert 'changes nothing' in again or 'ALREADY DONE' in again, again


def test_ambiguous_ready_multi_edit_only_for_a_one_line_new(ws):
    code, out = run(ws, target='pkg/app.py', old='    @property', new='    @property\\n    # read-only')
    assert code == 1 and 'to change all of them in one call' not in out, out
