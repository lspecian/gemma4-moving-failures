"""agent-v17, lever L2 of research/notes/12-failures-eval113.md (agent/skills/repo-tools/scripts: edit.py, guards.py,
check_patch.py).

- start without end and a multi-line NEW is NOT APPLIED (exit 2) unless the lines it rewrites are clear (v19
  infer_end): the message shows the likely range and a ready line-range call (fastapi_14873 v16 x2, rich_3930: one
  line replaced by a block duplicated or orphaned code). When NEW redefines a name defined elsewhere (stale line
  numbers, rich_3052), the call points at that definition.
- A range or old/new edit that deletes a module/class-level def/class/property still used by the file, the package
  or the tests is NOT APPLIED (guard 'definition'); an unused one is deleted with a WARNING (fastapi_14458/14448
  `computed_scope`, rich_3063 `_parse`).
- show.py line-number prefixes are stripped from NEW when every line has one (fastapi_11355 v16_r1).
- check_patch.py warns about definitions present at HEAD, gone from the working tree and still used.
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
import guards  # noqa: E402

# fastapi_14873: APIRouter.__init__ sets on_startup before super().__init__()
ROUTING = textwrap.dedent('''\
    class Router:
        def __init__(self, on_startup=None):
            # keep the handlers
            self.on_startup = on_startup or []
            super().__init__()

        def startup(self):
            for handler in self.on_startup:
                handler()


    class APIRouter(Router):
        def __init__(self, on_startup=None, prefix=""):
            # the prefix must start with a slash
            if prefix:
                assert prefix.startswith("/")
                assert not prefix.endswith("/")
            self.prefix = prefix
            super().__init__(on_startup=on_startup)
    ''')
# fastapi_14458: three properties of Dependant, then computed_scope, which another property uses
MODELS = textwrap.dedent('''\
    import inspect


    class Dependant:
        def __init__(self, call=None, scope=None):
            self.call = call
            self.scope = scope

        @property
        def is_gen_callable(self):
            return inspect.isgeneratorfunction(self.call)

        @property
        def is_coroutine_callable(self):
            return inspect.iscoroutinefunction(self.call)

        @property
        def computed_scope(self):
            return self.scope or "function"

        @property
        def cache_key(self):
            return (self.call, self.computed_scope)
    ''')
# rich_3063: escape() then _parse(), which render() and the tests use
MARKUP = textwrap.dedent('''\
    import re

    RE_TAGS = re.compile(r"\\[(.*?)\\]")


    def escape(markup):
        return markup.replace("[", "\\\\[")


    def _parse(markup):
        for match in RE_TAGS.finditer(markup):
            yield match.start(), match.group(1)


    def render(markup):
        return list(_parse(markup))


    def unused_helper(x):
        return x
    ''')
UTILS = textwrap.dedent('''\
    from pkg.models import Dependant


    def solve(dependant: Dependant):
        return dependant.computed_scope
    ''')
TEST_MARKUP = 'from pkg.markup import _parse, render\n\n\ndef test_parse():\n    assert list(_parse("[b]x"))\n'


def git(ws, *args):
    return subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / 'repo'
    (root / 'pkg').mkdir(parents=True)
    (root / 'pkg/__init__.py').write_text('')
    (root / 'pkg/routing.py').write_text(ROUTING)
    (root / 'pkg/models.py').write_text(MODELS)
    (root / 'pkg/markup.py').write_text(MARKUP)
    (root / 'pkg/utils.py').write_text(UTILS)
    (root / 'tests').mkdir()
    (root / 'tests/test_markup.py').write_text(TEST_MARKUP)
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


def src(ws, rel):
    return (ws / rel).read_text()


def line_of(text, line, after=0):
    return text.split('\n').index(line, after) + 1


def ready_call(out):
    """The ready edit.py call of a refusal, parsed from the notation."""
    found = [args for script, args in calls(out) if script == 'edit.py']
    assert found, out
    return found[0]


# ---------------------------------------------------------------- 1. start without end, multi-line NEW

def test_start_only_multiline_new_on_a_comment_is_refused_with_a_ready_range(ws):
    """fastapi_14873 v16: start on a comment line, NEW moves super().__init__() above the assignment -> nothing written;
    the range includes the old super().__init__() (ending at the assignment would leave it in place, duplicated)."""
    n = line_of(ROUTING, '        # keep the handlers')
    new = '        super().__init__()\n        self.on_startup = on_startup or []'
    code, out = run('edit.py', ws, target='pkg/routing.py', start=n, new=new)
    assert code == 2 and src(ws, 'pkg/routing.py') == ROUTING, out
    assert out.startswith(f'NOT APPLIED: "start" without "end" and a NEW of 2 lines: replacing line {n} alone would '
                          'leave the old code after it in place (duplicated), so nothing was changed.'), out
    assert f'lines {n}-{n + 2}, the lines NEW matches best:' in out
    assert f'{n:5d}          # keep the handlers' in out and f'{n + 2:5d}          super().__init__()' in out
    assert ready_call(out) == {'target': 'pkg/routing.py', 'start': str(n), 'end': str(n + 2),
                               'new': '<your NEW unchanged>'}
    assert out.rstrip().endswith(f'CLOSEST: pkg/routing.py {n} {n + 2}')
    assert_notation(out)


def test_start_only_multiline_new_rewriting_the_lines_line_for_line(ws):
    n = line_of(ROUTING, '        if prefix:')
    code, out = run('edit.py', ws, target='pkg/routing.py', start=n,
                    new='        if prefix:\n            assert prefix.startswith("/"), prefix')
    assert code == 0 and f'NOTE: no end given: NEW rewrites lines {n}-{n + 1} line for line, so it replaced them' \
        in out, out
    assert src(ws, 'pkg/routing.py') == ROUTING.replace('startswith("/")\n', 'startswith("/"), prefix\n')


def test_start_only_on_a_def_line_with_another_def_is_refused(ws):
    """rich_3930 v16_r1: NEW `def graphemes` pasted over the `def cell_len` header orphaned its body."""
    n = line_of(MARKUP, 'def _parse(markup):')
    code, out = run('edit.py', ws, target='pkg/markup.py', start=n, new='def graphemes(text):\n    return list(text)')
    assert code == 2 and src(ws, 'pkg/markup.py') == MARKUP, out
    assert f'lines {n}-{n + 2}, the lines NEW matches best:' in out
    assert 'to insert code, keep the lines that stay in NEW' in out


def test_start_only_redefinition_with_stale_line_points_at_the_definition(ws):
    """rich_3052 v16_r1: start was a blank line two lines above `def ask` (numbers moved after an edit)."""
    n = line_of(MARKUP, 'def render(markup):')
    code, out = run('edit.py', ws, target='pkg/markup.py', start=n - 2,
                    new='def render(markup, strict=False):\n    return list(_parse(markup))')
    assert code == 2 and src(ws, 'pkg/markup.py') == MARKUP, out
    assert f'NEW redefines render, which begins at line {n}, not {n - 2}' in out
    assert ready_call(out)['start'] == str(n) and ready_call(out)['end'] == str(n + 1)


def test_start_only_same_definition_still_replaces_it(ws):
    n = line_of(MARKUP, 'def render(markup):')
    new = 'def render(markup):\n    return list(_parse(markup or ""))'
    code, out = run('edit.py', ws, target='pkg/markup.py', start=n, new=new)
    assert code == 0 and f'NOTE: no end given: NEW rewrites lines {n}-{n + 1} line for line' in out, out
    assert 'return list(_parse(markup or ""))' in src(ws, 'pkg/markup.py')


def test_start_only_single_line_still_applies(ws):
    n = line_of(ROUTING, '        self.on_startup = on_startup or []')
    code, out = run('edit.py', ws, target='pkg/routing.py', start=n, new='        self.on_startup = list(on_startup or [])')
    assert code == 0 and out.startswith(f'NOTE: no end given: replaced line {n} only\nEDITED'), out


def test_start_only_one_line_with_an_escape_in_a_string_is_one_line(ws):
    n = line_of(MARKUP, '    return markup.replace("[", "\\\\[")')
    code, out = run('edit.py', ws, target='pkg/markup.py', start=n, new='    return markup.replace("\\n", " ")')
    assert code == 0 and f'replaced line {n} only' in out, out
    assert '    return markup.replace("\\n", " ")\n' in src(ws, 'pkg/markup.py')


def test_start_only_new_with_literal_newline_escapes_counts_decoded_lines(ws):
    n = line_of(ROUTING, '        # keep the handlers')
    code, out = run('edit.py', ws, target='pkg/routing.py', start=n,
                    new='        super().__init__()\\n        self.on_startup = on_startup or []')
    assert code == 2 and 'a NEW of 2 lines' in out and src(ws, 'pkg/routing.py') == ROUTING, out


def test_start_only_refusal_footer_has_no_text_matching_advice(ws):
    n = line_of(ROUTING, '        # keep the handlers')
    code, out = run('edit.py', ws, flow_on=True, target='pkg/routing.py', start=n, new='        a = 1\n        b = 2')
    assert code == 2 and 'copy the old text exactly' not in out, out
    assert_notation(out)


# ---------------------------------------------------------------- 2. deleted definitions

def test_range_edit_deleting_a_used_property_is_refused(ws):
    """fastapi_14458 v14_r1: a range edit over the three properties ran past is_coroutine_callable and deleted
    computed_scope, which cache_key (same file) and utils.solve use."""
    a = line_of(MODELS, '    @property')
    b = line_of(MODELS, '        return self.scope or "function"')
    new = ('    @property\n    def is_gen_callable(self):\n        if inspect.isclass(self.call):\n'
           '            return False\n        return inspect.isgeneratorfunction(self.call)')
    code, out = run('edit.py', ws, target='pkg/models.py', start=a, end=b, new=new)
    assert code == 1 and src(ws, 'pkg/models.py') == MODELS, out
    d0 = line_of(MODELS, '    def computed_scope(self):') - 1
    assert out.startswith(f'NOT APPLIED: the edit (lines {a}-{b}) deletes `def computed_scope` '
                          f'(Dependant.computed_scope, lines {d0}-{d0 + 2}, a @property), and NEW does not define it, '
                          'but the code still uses it; nothing was changed.'), out
    use = line_of(MODELS, '        return (self.call, self.computed_scope)')
    assert f'  computed_scope is used at: pkg/models.py:{use} (return (self.call, self.computed_scope)), pkg/utils.py:5' \
        in out
    end = d0 - 2  # the blank line before the property is not part of the narrower range
    assert f'to change only the lines before it: edit.py  target=pkg/models.py  start={a}  end={end}  ' in out
    assert f'or keep it: copy lines {d0}-{d0 + 2} (its definition) into NEW unchanged.' in out
    assert out.rstrip().endswith(f'CLOSEST: pkg/models.py {a} {end}')
    assert_notation(out)


def test_deleted_definition_footer_gives_the_narrower_range(ws):
    a = line_of(MODELS, '    @property')
    b = line_of(MODELS, '        return self.scope or "function"')
    code, out = run('edit.py', ws, flow_on=True, target='pkg/models.py', start=a, end=b, new='    pass')
    nxt = [l for l in out.splitlines() if l.startswith('NEXT:')]
    assert code == 1 and nxt and f'start={a}  end={line_of(MODELS, "    def computed_scope(self):") - 3}' in nxt[0], out
    assert_notation(out)


def test_stale_range_deleting_a_def_line_is_refused(ws):
    """rich_3063 v14_r1: a range with pre-edit line numbers took the `def _parse(...)` line with it."""
    a = line_of(MARKUP, '    return markup.replace("[", "\\\\[")')
    b = line_of(MARKUP, 'def _parse(markup):')
    code, out = run('edit.py', ws, target='pkg/markup.py', start=a, end=b,
                    new='    if markup.endswith("\\\\"):\n        markup += "\\\\"\n    return markup.replace("[", "\\\\[")')
    assert code == 1 and src(ws, 'pkg/markup.py') == MARKUP, out
    assert 'deletes `def _parse`' in out and 'tests/test_markup.py:1' in out, out


def test_old_new_edit_deleting_a_def_used_only_by_tests_is_refused(ws):
    (ws / 'pkg/markup.py').write_text(MARKUP.replace('return list(_parse(markup))', 'return []'))
    git(ws, 'commit', '-qam', 'render no longer parses')
    old = 'def _parse(markup):\n    for match in RE_TAGS.finditer(markup):\n        yield match.start(), match.group(1)\n'
    code, out = run('edit.py', ws, target='pkg/markup.py', old=old, new='')
    assert code == 1 and 'deletes `def _parse`' in out and '_parse is used at: tests/test_markup.py:1 (from pkg' in out


def test_deleting_an_unused_def_applies_with_a_warning(ws):
    a = line_of(MARKUP, 'def unused_helper(x):')
    code, out = run('edit.py', ws, target='pkg/markup.py', start=a - 2, end=a + 1, new='')
    assert code == 0 and 'unused_helper' not in src(ws, 'pkg/markup.py'), out
    assert (f'WARNING: the edit deleted `def unused_helper` (lines {a}-{a + 1}); nothing in the package uses it, but if '
            'that was not intended, put it back.') in out


def test_rename_applies_and_names_the_uses_to_update(ws):
    """v17b (httpx_3672: complete() replaced by keep_alive/reset, the callers come next): a deletion that adds new
    definitions is a rename/replacement -> applied with a WARNING naming the remaining uses."""
    n = line_of(MARKUP, 'def _parse(markup):')
    code, out = run('edit.py', ws, target='pkg/markup.py', start=n, end=n, new='def parse_tags(markup):')
    assert code == 0 and 'def parse_tags(markup):' in src(ws, 'pkg/markup.py'), out
    assert 'WARNING: the edit replaced `def _parse`' in out and '(new: parse_tags)' in out
    assert 'still used at: pkg/markup.py:' in out and 'update those uses too, or keep the old name' in out


def test_rename_with_an_alias_has_no_warning(ws):
    n = line_of(MARKUP, 'def _parse(markup):')
    code, out = run('edit.py', ws, target='pkg/markup.py', start=n, end=n + 2,
                    new='def parse_tags(markup):\n    for match in RE_TAGS.finditer(markup):\n'
                        '        yield match.start(), match.group(1)\n\n\n_parse = parse_tags')
    assert code == 0 and 'WARNING: the edit' not in out, out


def test_moving_a_definition_is_not_a_deletion(ws):
    """fastapi_14873 gold: a pure move within the file keeps every name."""
    a = line_of(ROUTING, '        # keep the handlers')
    code, out = run('edit.py', ws, target='pkg/routing.py', start=a, end=a + 2,
                    new='        super().__init__()\n        # keep the handlers\n        self.on_startup = on_startup or []')
    assert code == 0 and 'WARNING: the edit deleted' not in out, out


def test_lost_definitions_unit(tmp_path, monkeypatch):
    monkeypatch.setenv('WORKSPACE', str(tmp_path))  # v19b: without it the root falls back to the cwd (this repo)
    before = 'class C:\n    def m(self):\n        return 1\n\n\ndef f():\n    return C().m()\n'
    lost = guards.lost_definitions(before, 'class C:\n    pass\n\n\ndef f():\n    return C().m()\n', 'pkg/x.py')
    assert [(d['qualname'], d['kind'], d['refs']) for d in lost] == [('C.m', 'def', [('pkg/x.py', 6, 'return C().m()')])]
    # a removed top-level name is not "used" by an unrelated attribute of the same name
    lost = guards.lost_definitions('def m():\n    return 1\n\n\nx = obj.m\n', 'x = obj.m\n', 'pkg/x.py')
    assert [(d['name'], d['refs']) for d in lost] == [('m', [])]
    # re-bound by assignment or import: not lost; a removed class hides its members
    assert guards.lost_definitions('def m():\n    pass\n', 'from y import m\n', 'pkg/x.py') == []
    assert guards.lost_definitions('def m():\n    pass\n', 'm = property(len)\n', 'pkg/x.py') == []
    lost = guards.lost_definitions('class C:\n    def a(self):\n        pass\n', 'X = 1\n', 'pkg/x.py')
    assert [d['qualname'] for d in lost] == ['C']
    # keyword arguments and attribute stores are not uses
    lost = guards.lost_definitions('def g():\n    pass\n', 'f(g=1)\nself.g = 2\n', 'pkg/x.py')
    assert lost[0]['refs'] == []
    # unparsable text: nothing to say
    assert guards.lost_definitions('def g(:\n', '', 'pkg/x.py') == []


# ---------------------------------------------------------------- 3. line-number prefixes in NEW

def test_numbered_new_in_a_range_edit_is_stripped(ws):
    a = line_of(MARKUP, 'def render(markup):')
    new = f'{a:5d}  def render(markup):\n{a + 1:5d}      return list(_parse(markup.strip()))'
    code, out = run('edit.py', ws, target='pkg/markup.py', start=a, end=a + 1, new=new)
    assert code == 0 and out.startswith('NOTE: removed the line-number prefixes copied from show.py output from NEW\n'
                                        'EDITED'), out
    assert '\ndef render(markup):\n    return list(_parse(markup.strip()))\n' in src(ws, 'pkg/markup.py')


def test_numbered_old_and_new_with_literal_newlines_fastapi_11355(ws):
    """fastapi_11355 v16_r1: OLD and NEW copied from show.py with prefixes, as one line with literal \\n; NEW
    renumbered after the inserted line."""
    head = ROUTING.split('\n')
    a = line_of(ROUTING, '    def startup(self):')
    old = '\\n'.join(f'{a + i:5d}  {head[a - 1 + i]}' for i in range(3)) + '\\n'
    new_lines = [head[a - 1], '        handlers = list(self.on_startup)', '        for handler in handlers:',
                 head[a + 1]]
    new = '\\n'.join(f'{a + i:5d}  {l}' for i, l in enumerate(new_lines)) + '\\n'
    code, out = run('edit.py', ws, target='pkg/routing.py', old=old, new=new)
    assert code == 0 and 'NOTE: removed the line-number prefixes' in out, out
    assert '\n'.join(new_lines) in src(ws, 'pkg/routing.py')


def test_numbered_new_with_unnumbered_old_is_stripped(ws):
    code, out = run('edit.py', ws, target='pkg/markup.py', old='        yield match.start(), match.group(1)',
                    new='   12          yield match.start(), match.group(1) or ""')
    assert code == 0 and 'NOTE: removed the line-number prefixes' in out, out
    assert '        yield match.start(), match.group(1) or ""\n' in src(ws, 'pkg/markup.py')


def test_strip_show_prefixes_unit():
    import edit
    assert edit.strip_show_prefixes('  955  x = 1\n  956\n  957      y = 2') == 'x = 1\n\n    y = 2'
    assert edit.strip_show_prefixes('    1  a\n    1  b') is None        # numbers must increase
    assert edit.strip_show_prefixes('x = 1\n  2  y') is None             # every line
    assert edit.strip_show_prefixes('    42') is None                    # a bare number is not a prefixed line
    assert edit.strip_show_prefixes('1 + 2') is None                     # one space: code, not a prefix


# ---------------------------------------------------------------- 4. check_patch

def test_check_patch_warns_about_a_deleted_definition_still_used(ws):
    text = MARKUP.replace('def _parse(markup):\n    for match in RE_TAGS.finditer(markup):\n'
                          '        yield match.start(), match.group(1)\n\n\n', '')
    (ws / 'pkg/markup.py').write_text(text)
    code, out = run('check_patch.py', ws)
    n = line_of(MARKUP, 'def _parse(markup):')
    use = line_of(text, '    return list(_parse(markup))')
    assert (f'WARNING: pkg/markup.py: `def _parse` (lines {n}-{n + 2}) existed at HEAD but the patch deleted it, and it '
            f'is still used at pkg/markup.py:{use} (return list(_parse(markup))), tests/test_markup.py:1: an edit '
            'removed it by mistake? Put the definition back with edit.py.') in out, out


def test_check_patch_quiet_about_an_unused_deleted_definition(ws):
    (ws / 'pkg/markup.py').write_text(MARKUP.replace('\n\ndef unused_helper(x):\n    return x\n', ''))
    code, out = run('check_patch.py', ws)
    assert code == 0 and 'WARNING' not in out and 'READY' in out, out
