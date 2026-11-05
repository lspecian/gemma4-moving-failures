"""agent-v21 edit.py / guards.py (research/notes/16-wrong-patches-real-speed.md lever V4: tool gaps behind wrong
patches at real speed). Each case is reconstructed from a real run (E059-E063: the model's arguments, the file region
it edited):

1. Sibling sites for insertions: the "same code also appears at" note keys on the unchanged neighbour lines of a pure
   insertion that share a name with it (fastapi_5077 E059/E061: the unwrap loop was never put into
   get_typed_return_annotation).
2. Duplicate definitions: start-only NEW that redefines several consecutive definitions replaces all of them
   (fastapi_9555 E059); a name defined twice in one body is named in the CHECK line with the old lines to delete;
   an empty NEW deletes lines (it left one blank line), a range without new says how to delete.
3. CHECK against the baseline commit: names left undefined by an earlier edit stay listed until fixed (rich_4070
   E063: escape/getpass).
4. Doubled backslashes: a run exactly twice the replaced line's at the same place (rich_3063 E063, fastapi_15588 E060),
   or a literal whose halved text is in the file: WARNING + CHECK (warn only, never auto-fixed).
5. A range edit that drops a function's final `return` (fastapi_14482 E060); a function-local import that shadows an
   earlier use (fastapi_14962 E059, rich_4070 E063 call 30).
Run with scripts/test_skills.sh (sandbox image, py3.13) or uvx pytest.
"""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from notation import assert_notation

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'agent/skills/repo-tools/scripts'
sys.path.insert(0, str(SCRIPTS))
import edit  # noqa: E402
import guards  # noqa: E402

# fastapi/dependencies/utils.py at fastapi_5077's base (lines 186-226)
UTILS = textwrap.dedent('''\
    import inspect
    from typing import Any, Callable, Dict

    from pydantic.typing import ForwardRef, evaluate_forwardref


    def get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:
        signature = inspect.signature(call)
        globalns = getattr(call, "__globals__", {})
        typed_params = [
            inspect.Parameter(
                name=param.name,
                kind=param.kind,
                default=param.default,
                annotation=get_typed_annotation(param.annotation, globalns),
            )
            for param in signature.parameters.values()
        ]
        typed_signature = inspect.Signature(typed_params)
        return typed_signature


    def get_typed_annotation(annotation: Any, globalns: Dict[str, Any]) -> Any:
        if isinstance(annotation, str):
            annotation = ForwardRef(annotation)
            annotation = evaluate_forwardref(annotation, globalns, globalns)
            if annotation is type(None):
                return None
        return annotation


    def get_typed_return_annotation(call: Callable[..., Any]) -> Any:
        signature = inspect.signature(call)
        annotation = signature.return_annotation

        if annotation is inspect.Signature.empty:
            return None

        globalns = getattr(call, "__globals__", {})
        return get_typed_annotation(annotation, globalns)
    ''')

# fastapi/dependencies/models.py at fastapi_9555's base (the Dependant properties)
MODELS = textwrap.dedent('''\
    import inspect
    from dataclasses import dataclass
    from functools import cached_property
    from typing import Any, Callable, Optional, Union

    from asyncio import iscoroutinefunction


    @dataclass
    class Dependant:
        call: Optional[Callable[..., Any]] = None
        scope: Union[str, None] = None

        @cached_property
        def is_gen_callable(self) -> bool:
            if inspect.isgeneratorfunction(self.call):
                return True
            dunder_call = getattr(self.call, "__call__", None)  # noqa: B004
            return inspect.isgeneratorfunction(dunder_call)

        @cached_property
        def is_async_gen_callable(self) -> bool:
            if inspect.isasyncgenfunction(self.call):
                return True
            dunder_call = getattr(self.call, "__call__", None)  # noqa: B004
            return inspect.isasyncgenfunction(dunder_call)

        @cached_property
        def is_coroutine_callable(self) -> bool:
            if inspect.isroutine(self.call):
                return iscoroutinefunction(self.call)
            if inspect.isclass(self.call):
                return False
            dunder_call = getattr(self.call, "__call__", None)  # noqa: B004
            return iscoroutinefunction(dunder_call)

        @cached_property
        def computed_scope(self) -> Union[str, None]:
            if self.scope:
                return self.scope
            if self.is_gen_callable or self.is_async_gen_callable:
                return "request"
            return None
    ''')

# E059's NEW (call 18): the three properties, unwrapped
PROPS_NEW = '''    @cached_property
    def is_gen_callable(self) -> bool:
        call = inspect.unwrap(self.call) if self.call is not None else None
        if call and inspect.isgeneratorfunction(call):
            return True
        dunder_call = getattr(call, "__call__", None) if call else None  # noqa: B004
        return inspect.isgeneratorfunction(dunder_call) if dunder_call else False

    @cached_property
    def is_async_gen_callable(self) -> bool:
        call = inspect.unwrap(self.call) if self.call is not None else None
        if call and inspect.isasyncgenfunction(call):
            return True
        dunder_call = getattr(call, "__call__", None) if call else None  # noqa: B004
        return inspect.isasyncgenfunction(dunder_call) if dunder_call else False

    @cached_property
    def is_coroutine_callable(self) -> bool:
        call = inspect.unwrap(self.call) if self.call is not None else None
        if call and inspect.isroutine(call):
            return iscoroutinefunction(call)
        if call and inspect.isclass(call):
            return False
        dunder_call = getattr(call, "__call__", None) if call else None  # noqa: B004
        return iscoroutinefunction(dunder_call) if dunder_call else False
'''

# rich/console.py (rich_4070's base, the parts E063 touched)
CONSOLE = textwrap.dedent('''\
    import inspect
    import os
    from getpass import getpass
    from html import escape
    from inspect import isclass

    from .pretty import Pretty, is_expandable


    class Console:
        def input(self, prompt="", password=False):
            return getpass("", stream=None) if password else input(prompt)

        def export_html(self, text):
            return "<pre>" + escape(text) + "</pre>"

        def _caller_frame_info(self, offset):
            frame = inspect.currentframe()
            return os.path.basename(frame.f_code.co_filename), offset

        def _collect_renderables(self, renderables, append, append_text, check_text):
            for renderable in renderables:
                if hasattr(renderable, "__rich_console__") and not isclass(renderable):
                    append(renderable)
                elif is_expandable(renderable):
                    check_text()
                    append(Pretty(renderable))
                else:
                    append_text(str(renderable))
            check_text()
    ''')

# rich/markup.py at rich_3063's base
MARKUP = textwrap.dedent('''\
    import re
    from typing import Callable, Match

    _ReStringMatch = Match[str]  # regex match object
    _ReSubCallable = Callable[[_ReStringMatch], str]  # Callable invoked by re.sub
    _EscapeSubMethod = Callable[[_ReSubCallable, str], str]  # Sub method of a compiled re


    def escape(
        markup: str,
        _escape: _EscapeSubMethod = re.compile(r"(\\\\*)(\\[[a-z#/@][^[]*?])").sub,
    ) -> str:
        """Escapes text so that it won't be interpreted as markup."""

        def escape_backslashes(match: Match[str]) -> str:
            """Called by re.sub replace matches."""
            backslashes, text = match.groups()
            return f"{backslashes}{backslashes}\\\\{text}"

        markup = _escape(escape_backslashes, markup)
        return markup
    ''')

# fastapi/sse.py at fastapi_15588's base
SSE = textwrap.dedent('''\
    def _check_id_no_null(v: str | None) -> str | None:
        if v is not None and "\\0" in v:
            raise ValueError("SSE 'id' must not contain null characters")
        return v


    def format_lines(lines):
        return "\\n".join(lines).encode("utf-8")
    ''')

# fastapi/_compat/v2.py at fastapi_14482's base
V2 = textwrap.dedent('''\
    from typing import Sequence, Type

    from pydantic import BaseModel, create_model


    def create_body_model(
        *, fields: Sequence, model_name: str
    ) -> Type[BaseModel]:
        field_params = {f.name: (f.field_info.annotation, f.field_info) for f in fields}
        BodyModel: Type[BaseModel] = create_model(model_name, **field_params)  # type: ignore[call-overload]
        return BodyModel
    ''')

# fastapi/routing.py (fastapi_14962's base, get_request_handler's endpoint function, shortened)
ROUTING = textwrap.dedent('''\
    from starlette.responses import JSONResponse, Response


    def get_request_handler(actual_response_class=JSONResponse):
        async def app(request, run_endpoint, serialize):
            raw_response = await run_endpoint(request)
            if isinstance(raw_response, Response):
                return raw_response
            response_args = {"status_code": 200}
            content = await serialize(raw_response)
            response = actual_response_class(content, **response_args)
            return response

        return app
    ''')


def git(ws, *args):
    return subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / 'repo'
    (root / 'pkg').mkdir(parents=True)
    (root / 'pkg/__init__.py').write_text('')
    for name, text in (('utils', UTILS), ('models', MODELS), ('console', CONSOLE), ('markup', MARKUP), ('sse', SSE),
                       ('v2', V2), ('routing', ROUTING), ('pretty', 'def is_expandable(x):\n    return True\n\n\n'
                                                                    'class Pretty:\n    def __init__(self, x):\n'
                                                                    '        self.x = x\n')):
        (root / f'pkg/{name}.py').write_text(text)
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


def n_of(text, line, after=0):
    return text.split('\n').index(line, after) + 1


def check(out):
    lines = [l for l in out.splitlines() if l.startswith('CHECK:')]
    assert len(lines) == 1, out
    return lines[0]


CLEAN = 'CHECK: compiles; no new undefined names'

# ---------------------------------------------------------------- 1. sibling sites of an insertion (fastapi_5077)

SIG = n_of(UTILS, 'def get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:')
RET_GLOBALNS = n_of(UTILS, '    globalns = getattr(call, "__globals__", {})', SIG + 3)
RET_SIG = n_of(UTILS, '    signature = inspect.signature(call)', SIG + 3)
SIG_END = n_of(UTILS, '    return typed_signature')
LOOP = '    while hasattr(call, "__wrapped__"):\n        call = call.__wrapped__\n'


def sig_new(lines):
    """get_typed_signature with LOOP put after its first LINES lines (E059: 2, after `signature = ...`; E061: 1)."""
    body = UTILS.split('\n')[SIG - 1:SIG_END]
    return '\n'.join(body[:lines]) + '\n' + LOOP + '\n'.join(body[lines:]) + '\n'


@pytest.mark.parametrize('lines, at', [(2, RET_GLOBALNS), (1, RET_SIG)])
def test_insertion_names_the_sibling_function(ws, lines, at):
    # E059 call 13 / E061 call 7: a range edit of the whole function whose diff is a pure insertion
    code, out = run(ws, target='pkg/utils.py', start=SIG, end=SIG_END, new=sig_new(lines))
    assert code == 0, out
    assert (f'the same code also appears at: pkg/utils.py:{at + 2} get_typed_return_annotation; if it has the same '
            'bug, fix it too') in out, out


def test_insertion_sibling_drives_flow_next(ws):
    code, out = run(ws, flow_on=True, target='pkg/utils.py', start=SIG, end=SIG_END, new=sig_new(2))
    nexts = [l for l in out.splitlines() if l.startswith('NEXT:')]
    assert nexts and nexts[0].startswith(f'NEXT: if pkg/utils.py:{RET_GLOBALNS + 2} get_typed_return_annotation has '
                                         'the same bug, fix it too: '), out
    for line in nexts:
        assert_notation(line)


def test_insertion_sibling_not_listed_once_it_has_the_insertion(ws):
    code, out = run(ws, target='pkg/utils.py', start=RET_GLOBALNS, end=RET_GLOBALNS,
                    new=LOOP + '    globalns = getattr(call, "__globals__", {})')
    assert code == 0 and f'pkg/utils.py:{SIG + 2} get_typed_signature' in out, out  # the other way round
    code, out = run(ws, target='pkg/utils.py', start=SIG + 2, end=SIG + 2,
                    new=LOOP + '    globalns = getattr(call, "__globals__", {})')
    assert code == 0 and 'also appears at' not in out, out  # both sites have the loop now


def test_insertion_keys_need_a_shared_name():
    b = UTILS.split('\n')
    j = SIG + 1  # insert after `signature = ...` (0-based index of the line below)
    after = b[:j] + ['    logger.debug("typed signature")'] + b[j:]
    assert edit.also_at('pkg/utils.py', UTILS, '\n'.join(after)) == []  # nothing in common with the neighbours
    after = b[:j] + ['    call = inspect.unwrap(call)'] + b[j:]
    assert [s[0] for s in edit.also_at('pkg/utils.py', UTILS, '\n'.join(after))] == [RET_GLOBALNS + 1]


def test_insertion_of_new_definitions_has_no_siblings():
    after = UTILS + '\n\ndef helper(call):\n    return getattr(call, "__globals__", {})\n'
    assert edit.also_at('pkg/utils.py', UTILS, after) == []


# ---------------------------------------------------------------- 2. duplicate definitions (fastapi_9555)

GEN = n_of(MODELS, '    @cached_property')
COR_END = n_of(MODELS, '        return iscoroutinefunction(dunder_call)')


def test_start_only_new_redefining_consecutive_properties_replaces_them_all(ws):
    # E059 call 18: start=78 (the decorator) without end; v18d replaced only is_gen_callable, v20 refused
    code, out = run(ws, target='pkg/models.py', start=GEN, new=PROPS_NEW)
    assert code == 0, out
    assert (f'NOTE: no end given: NEW redefines is_gen_callable, is_async_gen_callable and is_coroutine_callable, so '
            f'it replaced their lines {GEN}-{COR_END}') in out, out
    text = (ws / 'pkg/models.py').read_text()
    assert text.count('def is_async_gen_callable') == 1 and text.count('inspect.unwrap') == 3, text
    assert 'def computed_scope' in text and check(out).startswith(CLEAN), out


def test_start_only_run_stops_at_a_definition_new_does_not_redefine(ws):
    two = PROPS_NEW.split('\n\n    @cached_property\n    def is_coroutine_callable')[0] + '\n'
    code, out = run(ws, target='pkg/models.py', start=GEN, new=two)
    end = n_of(MODELS, '        return inspect.isasyncgenfunction(dunder_call)')
    assert code == 0 and f'so it replaced their lines {GEN}-{end}' in out, out
    assert (ws / 'pkg/models.py').read_text().count('def is_coroutine_callable') == 1


def test_start_only_run_needs_the_alignment_to_agree(ws):
    # toolbench start_only on 9555's gold span: NEW ends inside is_coroutine_callable (its last line is kept)
    gold = PROPS_NEW.replace('        return iscoroutinefunction(dunder_call) if dunder_call else False\n', '')
    code, out = run(ws, target='pkg/models.py', start=GEN, new=gold)
    assert 'NEW redefines is_gen_callable, is_async_gen_callable and is_coroutine_callable' not in out, out
    assert (ws / 'pkg/models.py').read_text().count('        return iscoroutinefunction(dunder_call)\n') == 1


def test_duplicate_definitions_are_named_with_the_lines_to_delete(ws):
    # what v18d did: the three properties over is_gen_callable alone
    gen_end = n_of(MODELS, '        return inspect.isgeneratorfunction(dunder_call)')
    code, out = run(ws, target='pkg/models.py', start=GEN, end=gen_end, new=PROPS_NEW)
    assert code == 0, out
    text = (ws / 'pkg/models.py').read_text()
    old_start = n_of(text, '        if inspect.isasyncgenfunction(self.call):') - 2
    old_end = n_of(text, '        return iscoroutinefunction(dunder_call)')
    assert check(out) == (f'CHECK: Dependant.is_async_gen_callable (and 1 more: is_coroutine_callable) are now defined '
                          f'twice: the old definitions at lines {old_start}-{old_end} come after the new ones and '
                          f'replace them; delete lines {old_start}-{old_end}'), out
    # the deletion the model tried at call 21 (start/end without new): now an empty new deletes them
    code, out = run(ws, target='pkg/models.py', start=old_start, end=old_end)
    assert code == 2 and (f'to delete lines {old_start}-{old_end}, pass an empty new: edit.py  target=pkg/models.py  '
                          f'start={old_start}  end={old_end}  new=') in out, out
    code, out = run(ws, target='pkg/models.py', start=old_start, end=old_end, new='')
    assert code == 0 and f'EDITED pkg/models.py (lines {old_start}-{old_end} deleted)' in out, out
    assert check(out).startswith(CLEAN), out
    text = (ws / 'pkg/models.py').read_text()
    assert text.count('def is_coroutine_callable') == 1 and '\n\n\n    @cached_property\n    def computed_scope' \
        in text  # (the blank line before the deleted block stays: one blank line more than before)


def test_duplicate_definitions_unit():
    a = 'class A:\n    def f(self):\n        return 1\n'
    assert guards.duplicate_definitions(a, a + '\n    def f(self):\n        return 2\n')[0]['lines'] == [2, 5]
    setter = 'class A:\n    @property\n    def x(self):\n        return 1\n'
    assert guards.duplicate_definitions(setter, setter + '\n    @x.setter\n    def x(self, v):\n        pass\n') == []
    over = 'from typing import overload\n\n\n@overload\ndef f(x: int) -> int: ...\n'
    assert guards.duplicate_definitions(over, over + '@overload\ndef f(x: str) -> str: ...\ndef f(x):\n'
                                        '    return x\n') == []
    alt = 'import sys\nif sys.version_info >= (3, 8):\n    def f():\n        return 1\n'
    assert guards.duplicate_definitions(alt, alt + 'else:\n    def f():\n        return 2\n') == []
    pre = 'def f():\n    return 1\n\n\ndef f():\n    return 2\n'  # already twice before the edit: not this edit's
    assert guards.duplicate_definitions(pre, pre.replace('return 2', 'return 3')) == []


def test_empty_new_deletes_lines_in_every_form(ws):
    s = n_of(V2, '    return BodyModel')
    code, out = run(ws, target='pkg/v2.py', start=s, new='')
    assert code == 0 and f'NOTE: no end given: deleted line {s} only' in out, out
    assert (ws / 'pkg/v2.py').read_text() == V2.replace('    return BodyModel\n', '')
    git(ws, 'checkout', '-q', 'pkg/v2.py')
    code, out = run(ws, target='pkg/v2.py', old='    return BodyModel', new='')
    assert code == 0 and '(exact, lines deleted)' in out, out
    assert (ws / 'pkg/v2.py').read_text() == V2.replace('    return BodyModel\n', '')


def test_deletion_that_breaks_syntax_is_refused(ws):
    s = n_of(V2, '    field_params = {f.name: (f.field_info.annotation, f.field_info) for f in fields}')
    code, out = run(ws, target='pkg/v2.py', start=s, end=s + 2, new='')
    assert code == 1 and out.startswith('NOT APPLIED: the result would not parse: SyntaxError line ') and \
        'NEW starts at' not in out and 'of NEW)' not in out, out
    assert (ws / 'pkg/v2.py').read_text() == V2


# ---------------------------------------------------------------- 3. CHECK against the baseline (rich_4070 E063)

def test_names_left_undefined_by_an_earlier_edit_stay_in_check(ws):
    # call 14: the imports removed at the top (the model's "defer imports" step)
    code, out = run(ws, target='pkg/console.py', start=1, end=5, new='import os')
    text = (ws / 'pkg/console.py').read_text()
    g = n_of(text, '        return getpass("", stream=None) if password else input(prompt)')
    assert code == 0 and check(out) == (f"CHECK: line {g}: undefined name 'getpass' (the edit removed its definition "
                                        "or import) (and 3 more: escape, inspect, isclass)"), out
    # call 15: isclass repaired; v20 then said "no new undefined names" and escape/getpass/inspect stayed undefined
    text = (ws / 'pkg/console.py').read_text()
    s = n_of(text, '            if hasattr(renderable, "__rich_console__") and not isclass(renderable):')
    code, out = run(ws, target='pkg/console.py', start=s, end=s,
                    new='            if hasattr(renderable, "__rich_console__") and not isinstance(renderable, type):')
    assert code == 0, out
    text = (ws / 'pkg/console.py').read_text()
    first = min(n_of(text, l) for l in ('        return "<pre>" + escape(text) + "</pre>"',
                                         '        return getpass("", stream=None) if password else input(prompt)'))
    assert check(out) == (f"CHECK: line {first}: undefined name 'getpass' (an earlier edit removed its definition or "
                          "import) (and 2 more: escape, inspect)"), out
    # the deferred imports added: clean again
    code, out = run(ws, target='pkg/console.py', start=1, end=1,
                    new='import inspect\nimport os\nfrom getpass import getpass\nfrom html import escape')
    assert code == 0 and check(out).startswith(CLEAN), out


def test_a_name_never_bound_at_head_is_reported_once(ws):
    # the usual order of a correct multi-hunk fix: a use first, the import in a later edit (toolbench oracle: 102
    # gold edits would have been flagged if every name undefined since an earlier edit were repeated)
    s = n_of(V2, '    return BodyModel')
    code, out = run(ws, target='pkg/v2.py', start=s, end=s, new='    warnings.warn("x")\n    return BodyModel')
    assert "undefined name 'warnings'" in check(out), out
    code, out = run(ws, target='pkg/v2.py', start=s - 1, end=s - 1,
                    new='    BodyModel: Type[BaseModel] = create_model(model_name)')
    assert code == 0 and check(out).startswith(CLEAN), out


def test_untracked_file_has_no_baseline(ws):
    (ws / 'pkg/new.py').write_text('import json\n\n\ndef f():\n    return json.dumps(1)\n')
    run(ws, target='pkg/new.py', start=1, end=1, new='import sys')
    code, out = run(ws, target='pkg/new.py', start=1, end=1, new='import os')
    assert code == 0 and check(out).startswith(CLEAN), out  # no baseline: only names new since the previous edit


# ---------------------------------------------------------------- 4. doubled backslashes (rich_3063, fastapi_15588)

def test_doubled_regex_backslashes_warned_and_in_check(ws):
    # E063 call 17: the doubled `(\\\\*)` copied from show.py's JSON-encoded output, the new `(?<!\\)` written right
    s = n_of(MARKUP, '    _escape: _EscapeSubMethod = re.compile(r"(\\\\*)(\\[[a-z#/@][^[]*?])").sub,')
    new = ('    _escape: _EscapeSubMethod = re.compile(\n        r"(\\\\\\\\*)(\\[[a-z#/@][^[]*?]|(?<!\\\\)$)"\n'
           '    ).sub,')
    code, out = run(ws, target='pkg/markup.py', start=s, end=s, new=new)
    assert code == 0, out
    assert (f'WARNING: line {s + 1} has 4 backslashes in a row where the old line had 2. Results are JSON-encoded, so '
            'every backslash of the file is shown doubled; if you copied them from a result, halve them.') in out, out
    assert check(out) == f'CHECK: line {s + 1}: 4 backslashes in a row where the replaced line had 2 (see WARNING)'
    assert '(\\\\\\\\*)' in (ws / 'pkg/markup.py').read_text()  # warned, not changed


def test_doubled_string_escapes_warned(ws):
    # E060 call 18: "\\0" / "\\n" for "\0" / "\n"
    new = ('def _check_id_no_null(v: str | None) -> str | None:\n    if v is not None:\n        if "\\\\0" in v:\n'
           '            raise ValueError("SSE \'id\' must not contain null characters")\n        if "\\\\n" in v:\n'
           '            raise ValueError("SSE \'id\' must be a single line")\n    return v')
    code, out = run(ws, target='pkg/sse.py', start=1, end=4, new=new)
    assert code == 0, out
    warn = [l for l in out.splitlines() if l.startswith('WARNING: line ')]
    assert warn[0].startswith('WARNING: line 3 has 2 backslashes in a row where the old line had 1.'), out
    assert warn[1].startswith('WARNING: line 5 has 2 backslashes in a row where the same string elsewhere in the file '
                              'has 1.'), out
    assert check(out) == 'CHECK: line 3: 2 backslashes in a row where the replaced line had 1 (see WARNING)'


def test_intended_backslashes_not_flagged():
    # rich_3063's gold: new literals with one and two backslashes, nothing doubled in the replaced lines
    after = MARKUP.replace('    return markup\n', '    if markup.endswith("\\\\") and not markup.endswith("\\\\\\\\"):\n'
                           '        return markup + "\\\\"\n    return markup\n')
    assert guards.doubled_runs(MARKUP, after) == []
    lit = MARKUP + 'SEP = "\\\\"\n'  # the file has the literal with one backslash: the gold line is still not flagged
    assert guards.doubled_runs(lit, lit.replace('    return markup\n', after.split('markup = _escape(escape_backslashes, '
                                                                                  'markup)\n')[1])) == []
    same = MARKUP.replace('[a-z#/@]', '[a-z#/@!]')  # the regex changed elsewhere: runs unchanged
    assert guards.doubled_runs(MARKUP, same) == []
    old = 'X = "a\\\\b"\n'
    assert guards.doubled_runs(old, 'X = "a\\\\\\\\b"\n') == [(1, 4, 2, 'replaced')]
    assert guards.doubled_runs(old, 'X = "a\\\\\\\\\\\\b"\n') == []  # three for one: not a doubling


# ---------------------------------------------------------------- 5. dropped return (14482), local import (14962)

def test_range_edit_that_drops_the_final_return_is_named(ws):
    s = n_of(V2, '    BodyModel: Type[BaseModel] = create_model(model_name, **field_params)  # type: ignore[call-overload]')
    new = ('    BodyModel: Type[BaseModel] = create_model(\n        model_name, __config__={"arbitrary_types_allowed": True}, '
           '**field_params\n    )  # type: ignore[call-overload]')
    code, out = run(ws, target='pkg/v2.py', start=s, end=s + 1, new=new)
    assert code == 0, out
    assert check(out) == ('CHECK: create_body_model() (line 6) no longer ends with `return BodyModel`: the edit removed '
                          'it, so the function now returns None'), out
    code, out = run(ws, target='pkg/v2.py', start=s + 3, end=s + 3,
                    new='    )  # type: ignore[call-overload]\n    return BodyModel')
    assert code == 0 and check(out).startswith(CLEAN), out


def test_dropped_return_unit():
    f = 'def f(x):\n    y = x + 1\n    return y\n'
    assert guards.dropped_returns(f, 'def f(x):\n    y = x + 2\n    return y\n') == []
    assert guards.dropped_returns(f, 'def f(x):\n    if x:\n        return 1\n    else:\n        return 2\n') == []
    assert guards.dropped_returns(f, 'def f(x):\n    raise NotImplementedError\n') == []
    assert guards.dropped_returns(f, 'def f(x):\n    y = x + 1\n') == [('f', 1, 'return y')]
    assert guards.dropped_returns('def f():\n    return None\n', 'def f():\n    pass\n') == []


def test_function_local_import_shadowing_an_earlier_use(ws):
    # E059 call 14: `from starlette.responses import Response` inside the endpoint function
    s = n_of(ROUTING, '        response = actual_response_class(content, **response_args)')
    new = ('        if isinstance(content, bytes):\n            from starlette.responses import Response\n'
           '            response = Response(content=content, media_type="application/json", **response_args)\n'
           '        else:\n            response = actual_response_class(content, **response_args)')
    code, out = run(ws, target='pkg/routing.py', start=s, end=s, new=new)
    assert code == 0, out
    use = n_of(ROUTING, '        if isinstance(raw_response, Response):')
    assert check(out) == (f'CHECK: line {s + 1}: Response is imported inside app(), which makes it local to the whole '
                          f'function, so its use at line {use} raises UnboundLocalError; import it at the top of the '
                          'module'), out


def test_local_import_after_its_removal_at_the_top(ws):
    # rich_4070 E063 calls 23-30: the module-level import removed, then imported under the elif that uses it
    code, out = run(ws, target='pkg/console.py', old='from .pretty import Pretty, is_expandable\n', new='')
    assert code == 0 and "undefined name 'is_expandable'" in check(out), out
    text = (ws / 'pkg/console.py').read_text()
    s = n_of(text, '            elif is_expandable(renderable):')
    code, out = run(ws, target='pkg/console.py', start=s, end=s,
                    new='            elif is_expandable(renderable):\n                from .pretty import Pretty, '
                        'is_expandable')
    assert code == 0, out
    assert (f'line {s + 1}: is_expandable is imported inside _collect_renderables(), which makes it local to the '
            f'whole function, so its use at line {s} raises UnboundLocalError') in check(out), out


def test_local_shadowing_unit():
    before = 'import json\n\n\ndef f(x):\n    a = json.dumps(x)\n    return a\n'
    ok = before.replace('    a = json.dumps(x)', '    import json\n    a = json.dumps(x)')
    assert guards.local_shadowing(before, ok) == []  # imported before its first use
    bad = before.replace('    return a', '    json = None\n    return a')
    assert guards.local_shadowing(before, bad) == [('json', 'f', 6, 5, 'assigned')]
    pre = 'def f(x):\n    if x:\n        return y\n    y = 1\n    return y\n'  # the function bound y before the edit
    assert guards.local_shadowing(pre, pre.replace('y = 1', 'y = 2')) == []
    glob = 'X = 1\n\n\ndef f():\n    global X\n    print(X)\n    X = 2\n'
    assert guards.local_shadowing(glob.replace('X = 2', 'X = 3'), glob) == []
