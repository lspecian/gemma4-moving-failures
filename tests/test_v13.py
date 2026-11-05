"""agent-v13: information per call instead of refusals (agent/skills/repo-tools/scripts: names.py, show.py, search.py,
run.py, flow.py, kb.py).

E025 (v12, 11/36): after a refusal that printed the code to edit, the next call was a read 42 times out of 47 and an
applied edit once. The thinking before the reads asked targeted questions ("where is is_interactive set?", "does
APIRouter have a mount method?", "is FastAPIError imported here?", "where is _SVG_FONT_FAMILY used?"), and the model
even read files through run.py snippets to get around the block. v13 answers those questions inside the read:
  - show.py: "Names used here" under the code (where each repository name the lines use is defined or assigned);
    symbol=Class.attr that is not a def: the assignment, the inherited member, or "has no" + members + bases;
    a module-level name the file imports or another file defines: shown where it is defined.
  - search.py: the enclosing definition of each hit; with <= 3 hits, 2 lines of context; nothing matched an
    identifier: where a definition of that name (an attribute, not a def) or a similar one exists, or that none does.
  - run.py: a snippet that only reads repository source runs, with a NOTE giving the equivalent show.py call.
  - flow.py: reads are never refused (rules 1 and 3 steer the footer); past the budget the first NEXT line is the
    line-range edit of the best region with the number of reads since locate, then a pointer to that code.
Run with scripts/test_skills.sh (sandbox image, py3.13).
"""

import os
import pathlib
import re
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
import names  # noqa: E402

# A small rich-like package: Console sets is_interactive in __init__ (an attribute, not a def), Progress reads it
# through a property, Live and a base class in another file, an imported helper, stdlib and third-party imports.
CONSOLE = textwrap.dedent('''\
    """Console."""
    import os
    import threading
    from typing import Optional

    from ._helpers import detect_width, DEFAULT_WIDTH


    class ConsoleOptions:
        is_terminal: bool = False


    class Console:
        _environ = os.environ

        def __init__(self, force_interactive: Optional[bool] = None, width=None):
            self._lock = threading.RLock()
            self.width = width or detect_width(DEFAULT_WIDTH)
            self.is_interactive = (
                (self.is_terminal and not self.is_dumb_terminal)
                if force_interactive is None
                else force_interactive
            )

        @property
        def is_terminal(self) -> bool:
            """Check if the console is writing to a terminal."""
            return self._environ.get("TERM", "") != ""

        @property
        def is_dumb_terminal(self) -> bool:
            return self._environ.get("TERM", "") in ("dumb", "unknown")

        def print(self, *objects) -> None:
            with self._lock:
                text = " ".join(str(o) for o in objects)
                os.write(1, text.encode())
    ''')
HELPERS = textwrap.dedent('''\
    """Helpers."""
    DEFAULT_WIDTH = 80


    def detect_width(default: int) -> int:
        """Terminal width, or the default."""
        return default
    ''')
LIVE = textwrap.dedent('''\
    """Live display."""
    from .console import Console
    from .base import Renderable


    class Live(Renderable):
        def __init__(self, console=None):
            self.console = console or Console()
            self._started = False

        def start(self) -> None:
            self._started = True
            if self.console.is_interactive:
                self.console.print("start")

        def stop(self) -> None:
            self._started = False
    ''')
BASE = textwrap.dedent('''\
    """Base classes."""
    import abc


    class Renderable(abc.ABC):
        def __init__(self):
            self.style = None

        def render(self, width: int) -> str:
            return " " * width
    ''')
PROGRESS = textwrap.dedent('''\
    """Progress bars."""
    from .console import Console
    from .live import Live


    class Progress:
        def __init__(self, console=None):
            self.live = Live(console=console)

        @property
        def console(self) -> Console:
            return self.live.console

        def stop(self) -> None:
            """Stop the progress display."""
            self.live.stop()
            if not self.console.is_interactive:
                self.console.print()
    ''')
ROUTING = textwrap.dedent('''\
    """Routing (fastapi-like: APIRouter extends a third-party Router)."""
    from starlette import routing

    from .exceptions import FastAPIError


    class APIRouter(routing.Router):
        def __init__(self, prefix=""):
            self.prefix = prefix
            self.routes = []

        def route(self, path):
            return path

        def include_router(self, router):
            if router is self:
                raise FastAPIError("Cannot include a router into itself")
            self.routes.extend(router.routes)
    ''')
EXCEPTIONS = 'class FastAPIError(RuntimeError):\n    """A generic, FastAPI-specific error."""\n'


def git(ws, *args):
    subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *args], check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    ws, state = tmp_path / 'ws', tmp_path / 'state'
    files = {'rich/__init__.py': '', 'rich/console.py': CONSOLE, 'rich/_helpers.py': HELPERS, 'rich/live.py': LIVE,
             'rich/base.py': BASE, 'rich/progress.py': PROGRESS, 'fastapi/__init__.py': '',
             'fastapi/routing.py': ROUTING, 'fastapi/exceptions.py': EXCEPTIONS,
             'tests/test_progress.py': 'from rich.progress import Progress\n\n\ndef test_stop():\n'
                                       '    Progress().stop()\n'}
    for rel, text in files.items():
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text(text)
    git(ws, 'init', '-q')
    git(ws, 'add', '-A')
    git(ws, 'commit', '-qm', 'baseline')
    monkeypatch.setenv('WORKSPACE', str(ws))
    monkeypatch.setenv('REPO_TOOLS_DIR', str(state))
    monkeypatch.setattr(names, 'ROOT', str(ws))
    names._MODS.clear()
    names._STATE.update(files=0, bytes=0, kb=None, kb_tried=False)

    def call(script, _flow=False, **args):
        """One script call in the object form (flow off unless _flow: the scripts' own output)."""
        argv = [x for k, v in args.items() for x in ('--' + k, str(v))]
        env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(state), PYTHONDONTWRITEBYTECODE='1')
        env.pop('REPO_TOOLS_FLOW', None)
        if not _flow:
            env['REPO_TOOLS_FLOW'] = 'off'
        r = subprocess.run([sys.executable, str(SCRIPTS / script), *argv], capture_output=True, text=True, env=env,
                           timeout=120)
        return r.returncode, r.stdout

    call.ws = ws
    return call


def build_kb(repo):
    """The KB that locate.py builds on its first call (show/search use it, never build it)."""
    code, out = repo('kb.py')
    assert code == 0, out


def names_block(out):
    lines = out.splitlines()
    if 'Names used here, defined elsewhere:' not in lines:
        return {}
    i = lines.index('Names used here, defined elsewhere:')
    block = {}
    for line in lines[i + 1:]:
        if not line.startswith('  ') or line.startswith('  ...'):
            break
        name, rest = line.strip().split(None, 1)
        block[name] = rest
    return block


def full_names(repo, rel, symbol=None, start=None, end=None):
    """v18: show.py prints at most 3 task-related names on one `names:` line; the resolution behind it is checked here
    on the full block of names.names_used (same ranking) for the lines a show of the symbol / range prints."""
    import ast
    sys.path.insert(0, str(SCRIPTS))
    import show
    if symbol:
        node, _ = show.find(ast.parse((repo.ws / rel).read_text()), symbol.split('.'))[0]
        start, end = show.def_start(node), node.end_lineno
    names._MODS.clear()
    names._STATE.update(files=0, bytes=0, kb=None, kb_tried=False)
    return names_block('\n'.join(names.names_used(rel, [(int(start), int(end))])))


def names_line(out):
    """{display: rest} of show.py's v18 `names:` line, in order."""
    line = next((l for l in out.splitlines() if l.startswith('names: ')), '')
    return dict(e.split(' ', 1) for e in line[len('names: '):].split(' | ')) if line else {}


def line_of(repo, rel, text):
    return next(i for i, l in enumerate((repo.ws / rel).read_text().splitlines(), 1) if text in l)


# ---------------------------------------------------------------- kb: assignments

def test_kb_indexes_assignments_and_attributes(repo):
    build_kb(repo)
    import kb
    conn = kb.open(str(repo.ws))
    rows = kb.find_defs('is_interactive', conn)
    assert rows == [('rich/console.py', line_of(repo, 'rich/console.py', 'self.is_interactive = ('), 'attribute',
                     'Console.is_interactive', 'Console.__init__')]
    kinds = {r[2] for r in kb.find_defs('is_terminal', conn)}
    assert kinds == {'class attribute', 'def'}  # ConsoleOptions.is_terminal: bool; Console.is_terminal property
    assert ('rich/_helpers.py', 2, 'assignment', 'DEFAULT_WIDTH', '') in kb.find_defs('DEFAULT_WIDTH', conn)
    assert {'is_interactive', 'DEFAULT_WIDTH', 'detect_width', '_lock'} <= kb.all_names(conn)
    assert not kb.find_defs('text', conn)  # a local variable of Console.print is not a definition


# ---------------------------------------------------------------- show.py: names used here

def test_show_lists_where_the_names_used_are_defined(repo):
    build_kb(repo)
    code, out = repo('show.py', target='rich/progress.py', symbol='Progress.stop')
    assert code == 0
    n = line_of(repo, 'rich/console.py', 'self.is_interactive = (')
    assert names_line(out)['self.console.is_interactive'] == f'rich/console.py:{n} = ( [in Console.__init__]', out
    block = full_names(repo, 'rich/progress.py', 'Progress.stop')
    # self.console.is_interactive: Progress.console is a property whose class is unknown statically; the KB finds the
    # attribute Console.__init__ assigns in another file
    assert block['self.console.is_interactive'] == (
        f'rich/console.py:{line_of(repo, "rich/console.py", "self.is_interactive = (")}  self.is_interactive = (  '
        '[in Console.__init__]')
    assert block['self.live'].startswith('rich/progress.py:8  self.live = Live(console=console)  [in Progress.__init__]')
    assert block['self.console'].startswith('rich/progress.py:11  @property def console(self) -> Console:')
    assert block['self.live.stop'].startswith('rich/live.py:16  def stop(self) -> None:')
    assert 'print' not in block and 'self.console.print' in block  # builtins never; Console.print is a method


def test_show_names_skip_locals_builtins_stdlib_and_the_lines_shown(repo):
    build_kb(repo)
    block = full_names(repo, 'rich/console.py', 'Console.__init__')
    assert block['detect_width'].startswith('rich/_helpers.py:5  def detect_width(default: int) -> int:')
    assert block['DEFAULT_WIDTH'].startswith('rich/_helpers.py:2  DEFAULT_WIDTH = 80')
    assert block['self.is_terminal'].startswith('rich/console.py:') and '@property def is_terminal' in \
        block['self.is_terminal']
    for name in ('threading.RLock', 'force_interactive', 'width', 'Optional', 'self.is_interactive', 'self._lock'):
        assert name not in block, name  # stdlib, parameters, annotations, assigned in the lines shown


def test_show_names_resolve_inherited_members_and_module_names(repo):
    build_kb(repo)
    block = full_names(repo, 'rich/console.py', start='35', end='38')  # Console.print body
    assert block['self._lock'].startswith('rich/console.py:17  self._lock = threading.RLock()  [in Console.__init__]')
    (repo.ws / 'rich/live.py').write_text(LIVE + '\n    def width(self):\n        return self.render(self.style)\n')
    block = full_names(repo, 'rich/live.py', 'Live.width')
    assert block['self.render'].startswith('rich/base.py:9  def render(self, width: int) -> str:')
    assert block['self.style'].startswith('rich/base.py:7  self.style = None  [in Renderable.__init__]')


def test_show_names_put_the_issue_names_first(repo):
    build_kb(repo)
    conn = flow.connect()
    flow.set_meta(conn, 'query', 'Add TTY_INTERACTIVE so Live does not animate when the console is not interactive')
    conn.close()
    code, out = repo('show.py', target='rich/live.py', symbol='Live.start')
    order = list(names_line(out))
    assert order[0] == 'self.console.is_interactive', order  # its word "interactive" is in the issue


def test_show_names_without_kb_still_resolve_static_names(repo):
    code, out = repo('show.py', target='rich/progress.py', symbol='Progress.stop')  # no KB built: no build here
    assert 'self.live' in names_line(out) and 'self.console.is_interactive' not in names_line(out)
    block = full_names(repo, 'rich/progress.py', 'Progress.stop')
    assert 'self.live' in block and 'self.console.is_interactive' not in block
    assert not (pathlib.Path(os.environ['REPO_TOOLS_DIR']) / 'kb.sqlite').exists()


def test_show_names_block_is_bounded(repo):
    many = ''.join(f'def f{i}():\n    return {i}\n\n\n' for i in range(30))
    body = '\n'.join(f'    x{i} = f{i}()' for i in range(30))
    (repo.ws / 'rich/many.py').write_text(many + 'def use():\n' + body + '\n')
    code, out = repo('show.py', target='rich/many.py', symbol='use')
    assert 'names:' not in out  # v18: module functions unrelated to the task are never listed
    lines = names.names_used('rich/many.py', [(121, 151)])
    assert len(lines) == names.MAX_NAMES + 2 and lines[-1] == f'  ... {30 - names.MAX_NAMES} more'
    assert all(len(l) <= names.LINE_CHARS for l in lines)


def test_show_names_fast_on_a_large_file(repo):
    """< 0.3 s per call on a 5000-line file (names resolution only; the KB already built by locate)."""
    cls = ['class Big:', '    def __init__(self):'] + [f'        self.a{i} = {i}' for i in range(400)]
    for i in range(300):
        cls += [f'    def m{i}(self, x):', f'        """Method {i}."""', f'        y = self.a{i % 400} + x',
                f'        z = helper{i % 50}(y)', f'        if z > {i}:', f'            return self.m{(i + 1) % 300}(z)',
                '        return z', '']
    funcs = [f'def helper{i}(v):\n    return v + {i}\n' for i in range(50)]
    src = '"""Big."""\n' + '\n'.join(funcs) + '\n\n' + '\n'.join(cls) + '\n'
    src += ''.join(f'\n\ndef tail{i}():\n    return {i}\n' for i in range(5000 - len(src.splitlines())))
    (repo.ws / 'rich/big.py').write_text(src)
    assert len(src.splitlines()) >= 5000
    git(repo.ws, 'add', '-A')
    git(repo.ws, 'commit', '-qm', 'big')
    build_kb(repo)
    lines = src.splitlines()
    start = lines.index('    def m150(self, x):') + 1
    times = []
    for _ in range(3):
        names._MODS.clear()
        names._STATE.update(files=0, bytes=0, kb=None, kb_tried=False)
        t = time.perf_counter()
        block = names.names_used('rich/big.py', [(start, start + 6)])
        times.append(time.perf_counter() - t)
    assert any(l.strip().startswith('self.a150') for l in block) and any('helper0' in l for l in block), block
    assert sorted(times)[1] < 0.3, times
    t = time.perf_counter()
    code, out = repo('show.py', target='rich/big.py', symbol='Big.m150')
    assert code == 0 and 'names: self.a150 ' in out
    assert time.perf_counter() - t < 2.0  # the whole script, process start and KB open included


# ---------------------------------------------------------------- show.py: "does X have Y"

def test_show_instance_attribute_shows_its_assignment(repo):
    """rich_3777: show.py symbol=Console.is_interactive said "not found"; it is assigned in Console.__init__."""
    build_kb(repo)
    code, out = repo('show.py', target='rich/console.py', symbol='Console.is_interactive')
    n = line_of(repo, 'rich/console.py', 'self.is_interactive = (')
    assert code == 0
    assert out.startswith(f'NOTE: Console.is_interactive is an instance attribute, not a def: it is assigned in '
                          f'Console.__init__ at rich/console.py:{n}')
    assert f'\n{n}          self.is_interactive = (' in out and f'\n{n + 4}          )' in out
    assert names_line(out)['self.is_terminal'].endswith(': @property (self) -> bool')


def test_show_inherited_member_is_shown_in_the_base_class(repo):
    code, out = repo('show.py', target='rich/live.py', symbol='Live.render')
    assert code == 0 and out.startswith('NOTE: Live has no render of its own; it inherits Renderable.render from '
                                        'rich/base.py:')
    assert 'rich/base.py Renderable.render L9-10' in out and 'return " " * width' in out


def test_show_missing_member_lists_members_and_external_bases(repo):
    """fastapi_15800: "does APIRouter have a mount method?" (8 searches in E025 r2)."""
    code, out = repo('show.py', target='fastapi/routing.py', symbol='APIRouter.mount')
    assert code == 1
    lines = out.splitlines()
    assert lines[0].startswith('APIRouter.mount not found: APIRouter has no mount (no def, property, class attribute '
                               'or self.mount assignment in APIRouter or its base classes in the repository)')
    assert lines[1] == ('class APIRouter(routing.Router)  fastapi/routing.py:7  bases: routing.Router (from '
                        'starlette.routing, not in the repository)')
    assert 'mount may be inherited from outside the repository: starlette.routing.Router' in out
    assert "run.py  snippet=from starlette.routing import Router; print(hasattr(Router, 'mount'))" in out
    assert 'APIRouter members:' in out and 'route L12' in out and 'prefix L9 (attribute)' in out
    assert_notation(out)


def test_show_missing_member_suggests_close_names(repo):
    code, out = repo('show.py', target='rich/console.py', symbol='Console.is_termnal')
    assert code == 1 and 'close names: Console.is_terminal' in out and 'not found' in out


def test_show_imported_name_is_shown_where_defined(repo):
    """fastapi_14258: "where is FastAPIError imported from?" (4 reads of the import block in E025)."""
    code, out = repo('show.py', target='fastapi/routing.py', symbol='FastAPIError')
    assert code == 0
    assert out.startswith('NOTE: FastAPIError is not defined in fastapi/routing.py: it is imported at line 4 '
                          '(from fastapi.exceptions import FastAPIError); its definition is in fastapi/exceptions.py:')
    assert 'fastapi/exceptions.py FastAPIError L1-2' in out


def test_show_name_defined_in_another_file(repo):
    build_kb(repo)
    code, out = repo('show.py', target='rich/console.py', symbol='Renderable')
    assert code == 0 and out.startswith('NOTE: Renderable is not defined in rich/console.py; it is defined in '
                                        'rich/base.py:')
    code, out = repo('show.py', target='rich/console.py', symbol='Nope')
    assert code == 1 and 'Nope not found in rich/console.py' in out


# ---------------------------------------------------------------- search.py

def test_search_hits_show_their_enclosing_definition(repo):
    code, out = repo('search.py', text='self._')
    assert code == 0
    assert 'rich/console.py:17 Console.__init__: self._lock = threading.RLock()' in out
    assert 'rich/live.py:12 Live.start: self._started = True' in out
    code, out = repo('search.py', text='is_terminal: bool')
    assert out.splitlines()[0] == 'rich/console.py:10 ConsoleOptions:'  # a class body
    code, out = repo('search.py', text='DEFAULT_WIDTH = 80')
    assert out.splitlines()[0] == 'rich/_helpers.py:2:'  # module level: no enclosing name (with context: <= 3 hits)


def test_search_few_hits_come_with_context(repo):
    code, out = repo('search.py', text='is_interactive')
    lines = out.splitlines()
    n = line_of(repo, 'rich/console.py', 'self.is_interactive = (')
    assert lines[0] == f'rich/console.py:{n} Console.__init__:'
    assert lines[1:6] == [f'{i}{">" if i == n else " "} {(repo.ws / "rich/console.py").read_text().splitlines()[i - 1]}'
                          .rstrip() for i in range(n - 2, n + 3)]  # v18: unpadded numbers
    assert 'rich/live.py:13 Live.start:' in lines and 'rich/progress.py:17 Progress.stop:' in lines
    assert len(lines) == 3 * 6


def test_search_many_hits_have_no_context_and_keep_the_cap(repo):
    (repo.ws / 'rich/many.py').write_text('class M:\n    def f(self):\n' + ''.join(f'        x{i} = "needle"\n'
                                                                                  for i in range(40)))
    code, out = repo('search.py', text='needle')
    hits = [l for l in out.splitlines() if 'needle' in l and l.startswith('rich/many.py:')]
    assert len(hits) == 10 and hits[0] == 'rich/many.py:3 M.f: x0 = "needle"' and 'more matches' in out  # v18: 10


def test_search_footer_reads_the_new_hit_format(repo):
    repo('locate.py', _flow=True, query='Progress stop is_interactive console')
    code, out = repo('search.py', _flow=True, text='self._started')
    shows = [a for s, a in calls('\n'.join(nexts(out))) if s == 'show.py']
    assert shows and shows[0]['target'] == 'rich/live.py', out


def test_search_without_match_says_where_the_name_is_defined(repo):
    """rich_3777: `search regex=def is_interactive within=rich/console.py` found nothing: it is an attribute."""
    build_kb(repo)
    code, out = repo('search.py', regex='def is_interactive', within='rich/console.py')
    n = line_of(repo, 'rich/console.py', 'self.is_interactive = (')
    assert code == 1 and "no matches for 'def is_interactive' in rich/console.py (none either" in out
    assert 'is_interactive is defined as:' in out
    assert (f'  Console.is_interactive (an attribute assigned in Console.__init__) rich/console.py:{n}: '
            'show.py  target=rich/console.py  symbol=Console.is_interactive') in out
    code, out = repo('search.py', text='detect_widht')
    assert 'no definition named detect_widht; similar names:' in out
    assert 'show.py  target=rich/_helpers.py  symbol=detect_width' in out
    code, out = repo('search.py', text='def mount', within='fastapi')
    assert 'mount is not defined anywhere in the repository (no def, class or assignment of that name).' in out
    code, out = repo('search.py', text='3104')  # not an identifier: nothing more to say
    assert out.strip() == "no matches for '3104' (none either ignoring case, in non-.py text files)"  # v18
    assert_notation(out)


def test_search_without_kb_does_not_guess(repo):
    code, out = repo('search.py', text='def mount')
    assert code == 1 and 'not defined anywhere' not in out


def test_search_definition_hint_drives_the_footer(repo):
    repo('locate.py', _flow=True, query='Progress stop is_interactive console')
    code, out = repo('search.py', _flow=True, regex='def is_interactive')
    assert 'NEXT: read it' in out and 'show.py  target=rich/console.py  symbol=Console.is_interactive' in out


# ---------------------------------------------------------------- run.py: snippets that only read source

READ_SNIPPET = 'with open("rich/console.py", "r") as f:\n    lines = f.readlines()\n    for i in range(14, 22):\n' \
               '        print(f"{i+1}: {lines[i].strip()}")'


def test_run_read_only_snippet_runs_with_a_show_note(repo):
    code, out = repo('run.py', snippet=READ_SNIPPET)
    assert out.startswith('NOTE: this snippet only reads repository source; show.py does that with line numbers and '
                          'lists where the names in the code are defined: '
                          'show.py  target=rich/console.py  start=15  end=22')
    assert '17: self._lock = threading.RLock()' in out and 'exit code: 0' in out
    code, out = repo('run.py', snippet='print(open("/workspace/rich/live.py").read()[:20])')
    assert 'show.py  target=rich/live.py' in out.splitlines()[0]
    code, out = repo('run.py', snippet='import inspect\nfrom rich.console import Console\n'
                                       'print(inspect.getsource(Console.__init__))')
    assert 'show.py  target=rich/console.py  symbol=Console.__init__' in out.splitlines()[0]
    assert_notation(out.splitlines()[0])


@pytest.mark.parametrize('snippet', [
    'from rich.progress import Progress\nProgress().stop()',                    # runs repository code
    'print(open("rich/console.py").read().count("x"))\nimport os; os.remove("x")',  # does something else too
    'open("rich/console.py", "w").write("")',                                    # writes
    'print(open("/etc/hostname").read())',                                       # not a repository file
    'print(1)'])
def test_run_other_snippets_get_no_note(repo, snippet):
    assert flow.snippet_reads(snippet) is None
    code, out = repo('run.py', snippet=snippet)
    assert 'only reads repository source' not in out


def test_snippet_reads_ranges_and_slices(repo):
    assert flow.snippet_reads(READ_SNIPPET) == [{'file': 'rich/console.py', 'start': 15, 'end': 22}]
    assert flow.snippet_reads('lines = open("rich/live.py").readlines()\nprint("".join(lines[3:9]))') == \
        [{'file': 'rich/live.py', 'start': 4, 'end': 9}]
    assert flow.snippet_reads('import inspect, rich.live\nprint(inspect.getsource(rich.live.Live))') == \
        [{'symbol': 'rich.live.Live', 'module': 'rich'}]
    assert flow.snippet_reads('with open(\\"rich/live.py\\") as f:\\n    print(f.read())') == [{'file': 'rich/live.py'}]


# ---------------------------------------------------------------- flow: soft budget, through the real scripts

def nexts(out):
    return [l for l in out.splitlines() if l.startswith('NEXT:')]


def test_reads_before_locate_run_and_point_to_locate(repo):
    code, out = repo('show.py', _flow=True, target='rich/progress.py', symbol='Progress.stop')
    assert 'def stop(self) -> None:' in out and 'NOT RUN' not in out
    first = calls(nexts(out)[0])[0]
    assert first[0] == 'locate.py' and first[1]['query'].endswith('Progress.stop rich/progress.py')
    code, out = repo('search.py', _flow=True, text='is_interactive')
    assert 'Live.start' in out and calls(nexts(out)[0])[0][0] == 'locate.py'


def test_read_only_snippets_count_as_reads_not_checks(repo):
    repo('locate.py', _flow=True, query='Progress.stop prints a blank line when the console is not interactive '
                                        'Console.is_interactive rich/progress.py')
    assert flow.State(flow.connect()).mode == 'normal'
    for i in range(4):  # never refused
        code, out = repo('run.py', _flow=True, snippet=READ_SNIPPET.replace('range(14, 22)', f'range({i}, {i + 5})'))
        assert 'exit code: 0' in out and 'NOT RUN' not in out, out
    st = flow.State(flow.connect())
    assert st.checks == [] and st.look_since_edit == 4
    assert re.match(r'NEXT: 4 reads since locate: edit now', nexts(out)[0]), out
    for i in range(flow.MAX_RUNS_BEFORE_EDIT):
        repo('run.py', _flow=True, snippet=f'print({i})')
    code, out = repo('run.py', _flow=True, snippet='print("again")')
    assert 'exit code: 0' in out and not out.startswith('NOT RUN')  # v13: reproduction runs are not refused either
    # v16 (L1): three snippets that printed no error: sweep the invariant or edit
    assert re.match(r'NEXT: 3 snippets did not reproduce it: sweep the invariant over many inputs in ONE snippet, or '
                    r'edit the most suspicious line now: edit\.py  target=rich/progress\.py  start=14  end=18', nexts(out)[0]), out


# ---------------------------------------------------------------- locate.py: no names block (v18)

def test_locate_prints_the_best_function_without_a_names_block(repo):
    """v18 (note 13 B): locate's names block was 106 tokens per call and rarely used; show.py keeps its own."""
    code, out = repo('locate.py', query='Progress.stop prints a blank line when the console is not '
                                         'interactive Console.is_interactive rich/progress.py')
    assert code == 0 and '--- best match: rich/progress.py Progress.stop L14-18' in out, out
    assert 'Names used here' not in out and names_block(out) == {}
    assert out.rstrip().splitlines()[-1].startswith('   18  ')
