"""End-to-end tests of the repo-tools scripts with the flow engine ON, as the agent experiences them.

Each test drives a sequence of real script calls (separate processes, like run_skill_script) against a small git
repo and checks the output the model would see: results, refusals, NEXT/BUDGET footers and shared state.

Sequences are written in the OBJECT form of argument schema v2/v3 (research/notes/6-flow-design.md): the model
passes args={"k": "v"}, which run_skill_script turns into argv ["--k", "v", ...]. Array args can hang Kaggle's
vLLM Gemma 4 tool parser, so the object form is what the agent uses. The `agent` fixture runs every sequence
twice: with object args, and translated to the scripts' positional form (still accepted, kept green).
Schema v3: the model-facing key for the repository file is "target" and search.py's directory filter is "within"
("path", "file", "file_path", "dir", "in", ... still work as aliases), and footers suggest calls in the quote-free
notation `show.py  target=a.py  symbol=f` (tests/notation.py), never JSON.
"""

import os
import pathlib
import re
import subprocess
import sys
import textwrap

import pytest

from notation import assert_notation, calls

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'agent/skills/repo-tools/scripts'

INVOICE = textwrap.dedent('''\
    """Payment helpers."""


    class Invoice:
        def __init__(self, amount):
            self.amount = amount

        def total(self, tax=0.1):
            if self.amount == None:
                return 0
            return self.amount * (1 + tax)
    ''')

FILE_SCRIPTS = ('show.py', 'history.py', 'edit.py', 'create.py')  # "target" names the repository file
FILE_ALIASES = ('target', 'path', 'file', 'file_path', 'filepath', 'filename', 'file_name', 'within', 'dir',
                'directory', 'in')
BOTH_FORMS = pytest.mark.parametrize('agent', ['positional', 'object'],
                                     indirect=True)


def positional(script, args):
    """The positional argv for an object-form call (the scripts' old calling convention)."""
    a = dict(args)
    for k in FILE_ALIASES:  # the model-facing key or an alias: FILE of these scripts, PATH of search.py
        if k in a and script in FILE_SCRIPTS + ('search.py',):
            a['file' if script in FILE_SCRIPTS else 'path'] = a.pop(k)
    if script == 'locate.py':
        return [a['query']]
    if script == 'show.py':
        return [a['file'], a['symbol']] if 'symbol' in a else [a['file'], a['start'], a['end']]
    if script == 'outline.py':
        return (['--grep', a['grep']] if 'grep' in a else []) + a['files'].split()
    if script == 'search.py':
        return (['-E', a['regex']] if 'regex' in a else [a['text']]) + ([a['path']] if 'path' in a else [])
    if script in ('callers.py', 'tests_for.py'):
        return [a.get('name', a.get('target'))]
    if script == 'history.py':
        return ['--similar', a['similar']] if 'similar' in a else [a['file']]
    if script == 'edit.py':
        if 'start' in a:
            return ['--lines', a['file'], a['start'], a['end'], a['new']]
        return [a['file'], a['old'], a['new']]
    if script == 'create.py':
        return [a['file'], a['content']]
    if script == 'run.py':
        if 'test' in a:
            return ['test', a['test']] + (['-k', a['k']] if 'k' in a else [])
        return ['snippet', a['snippet']]
    if script == 'check_patch.py':
        return ['--clean'] if a.get('clean') == 'yes' else []
    if script == 'hypothesis.py':
        if 'confirm' in a or 'reject' in a:
            verb = 'confirm' if 'confirm' in a else 'reject'
            return [verb, a[verb], a.get('evidence', '')]
        if 'list' in a:
            return ['list']
        return ['add'] + [v for k, v in a.items() if k.startswith('add')]
    return []


@pytest.fixture
def agent(tmp_path, request):
    form = getattr(request, 'param', 'object')
    ws, state = tmp_path / 'ws', tmp_path / 'state'
    (ws / 'pay').mkdir(parents=True)
    (ws / 'pay/__init__.py').write_text('')
    (ws / 'pay/invoice.py').write_text(INVOICE)
    (ws / 'pay/http.py').write_text('def parse_header(v):\n    return v.split(",")\n')
    (ws / 'tests').mkdir()
    (ws / 'tests/test_invoice.py').write_text(
        'from pay.invoice import Invoice\n\n\ndef test_total():\n    assert Invoice(10).total(0) == 10\n')
    for cmd in (['init', '-q'], ['add', '-A'], ['commit', '-qm', 'baseline']):
        subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *cmd], check=True,
                       capture_output=True)

    def call(script, _flow=True, **args):
        """One run_skill_script call with args={...} (all values strings, like ADK stringifies them).
        _flow=False runs the script with the flow engine off (to check a call's arguments alone)."""
        args = {k: str(v) for k, v in args.items()}
        argv = [x for k, v in args.items() for x in ('--' + k, v)] if form == 'object' else positional(script, args)
        env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(state), PYTHONDONTWRITEBYTECODE='1')
        env.pop('REPO_TOOLS_FLOW', None)
        if not _flow:
            env['REPO_TOOLS_FLOW'] = 'off'
        r = subprocess.run([sys.executable, str(SCRIPTS / script), *argv], capture_output=True, text=True, env=env,
                           timeout=120)
        return r.stdout

    call.ws = ws
    call.form = form
    return call


def next_lines(out):
    return [l for l in out.splitlines() if l.startswith('NEXT:')]


def suggested_calls(out):
    """(script, args dict) for every call suggested in the NEXT lines, in order (tests/notation.py's rule)."""
    return [c for line in next_lines(out) for c in calls(line)]


def assert_object_form_footer(out):
    """NEXT lines use the notation of schema v3 (no JSON, no arrays, two-space separators, valid keys)."""
    for line in next_lines(out):
        assert_notation(line)


def prefilled(out):
    """Suggested calls with args and without <placeholders>: they must work as they stand."""
    return [(s, a) for s, a in suggested_calls(out) if a and not any('<' in v for v in a.values())]


@BOTH_FORMS
def test_look_around_before_locate_runs_and_points_to_locate(agent):
    """v13: reads are never refused; before locate ran, the footer's first call is locate.py."""
    out = agent('show.py', target='pay/invoice.py', symbol='Invoice.total')
    assert 'def total' in out and 'NOT RUN' not in out
    assert suggested_calls(out)[0][0] == 'locate.py'
    assert_object_form_footer(out)


@BOTH_FORMS
def test_happy_path_locate_edit_run_check(agent):
    out = agent('locate.py', query='Invoice.total returns wrong value when amount is None')
    assert 'pay/invoice.py' in out and 'def total' in out  # shortlist + best function source
    assert next_lines(out)  # footer tells the model what to do next
    assert ('edit.py', {'target': 'pay/invoice.py', 'old': '<exact current lines>', 'new': '<new lines>'}) \
        in suggested_calls(out)
    assert_object_form_footer(out)

    out = agent('edit.py', target='pay/invoice.py', old='if self.amount == None:', new='if self.amount is None:')
    assert out.startswith('EDITED pay/invoice.py')
    assert any(s in ('run.py', 'check_patch.py') for s, _ in suggested_calls(out))
    assert_object_form_footer(out)

    out = agent('run.py', test='tests/test_invoice.py')
    assert 'exit code: 0' in out
    assert ('check_patch.py', {}) in suggested_calls(out)

    out = agent('check_patch.py')
    assert 'READY' in out
    assert any('submit_patch' in l for l in next_lines(out))


@BOTH_FORMS
def test_repeat_call_is_not_rerun(agent):
    agent('locate.py', query='Invoice.total amount None')
    first = agent('show.py', target='pay/http.py', symbol='parse_header')
    again = agent('show.py', file='pay/http.py', symbol='parse_header')  # "file" is an alias: the same call
    agent('edit.py', target='pay/invoice.py', old='no such line', new='x')  # never two refusals in a row
    third = agent('show.py', path='pay/http.py', symbol='parse_header')  # so is "path"
    assert 'already' in third.lower()
    assert 'def parse_header' in first
    assert 'already' in again.lower()


@BOTH_FORMS
def test_exploration_budget_pushes_an_edit(agent):
    """v13: past the budget every read still runs; its footer's first call is the line-range edit."""
    agent('locate.py', query='Invoice.total amount None')
    outs = [agent('search.py', text=t) for t in ('self.amount', 'amount', 'total', 'tax', 'Invoice', 'return')]
    assert not any(o.startswith(('NOT RUN', 'ALREADY DONE')) for o in outs), outs
    pushed = [o for o in outs if re.search(r'^NEXT: \d+ reads since locate: edit now', o, re.M)]
    assert len(pushed) >= 3, outs
    first = suggested_calls(pushed[-1])[0]
    assert first[0] == 'edit.py' and first[1]['target'] == 'pay/invoice.py' and 'start' in first[1]
    assert_object_form_footer(pushed[-1])


@BOTH_FORMS
def test_failed_edits_lead_to_a_line_number_edit(agent):
    agent('locate.py', query='Invoice.total amount None')
    agent('edit.py', file='pay/invoice.py', old='if self.amount == Nothing:', new='x')
    out = agent('edit.py', target='pay/invoice.py', old='if self.amount == Nothingness:', new='x')
    edits = [a for s, a in suggested_calls(out) if s == 'edit.py' and 'start' in a]
    assert edits and edits[0]['target'] == 'pay/invoice.py', out
    assert_object_form_footer(out)


@BOTH_FORMS
def test_following_a_prefilled_footer_call_works(agent):
    """A fully pre-filled suggestion (no <placeholders>) is a valid call as it stands."""
    agent('locate.py', query='Invoice.total amount None')
    out = agent('search.py', text='self.amount')
    ready = [(s, a) for s, a in suggested_calls(out) if not any('<' in v for v in a.values())]
    assert ready and ready[0][0] == 'show.py', out
    script, args = ready[0]
    out = agent(script, **args)
    assert 'def total' in out and 'USAGE' not in out and 'NOT RUN' not in out


@BOTH_FORMS
def test_every_prefilled_footer_call_is_valid(agent):
    """Every suggested call without <placeholders> along a run parses and runs without a usage error."""
    outs = [agent('locate.py', query='Invoice.total amount None'), agent('search.py', text='self.amount'),
            agent('show.py', target='pay/invoice.py', start='1', end='12'),
            agent('edit.py', target='pay/invoice.py', old='if self.amount == Nothing:', new='x'),
            agent('edit.py', path='pay/invoice.py', old='if self.amount == Nothingness:', new='x'),
            agent('edit.py', target='pay/invoice.py', old='if self.amount == None:', new='if self.amount is None:'),
            agent('run.py', test='tests/test_invoice.py')]
    ready = []
    for out in outs:
        assert_object_form_footer(out)
        ready += prefilled(out)
    assert {s for s, _ in ready} >= {'show.py', 'run.py'}, ready
    for script, args in ready:
        out = agent(script, _flow=False, **args)
        assert 'USAGE ERROR' not in out and 'unknown key' not in out, (script, args, out)


@pytest.mark.parametrize('agent', ['object'], indirect=True)
def test_adk_parameters_swallowed_into_args_are_ignored(agent):
    """Kaggle's vLLM parser sometimes moves run_skill_script's own skill_name/file_path into args (LOG E023)."""
    agent('locate.py', query='Invoice.total amount None', skill_name='repo-tools')
    out = agent('edit.py', skill_name='repo-tools', file_path='edit.py', target='pay/invoice.py',
                old='if self.amount == None:', new='if self.amount is None:')
    assert out.startswith('EDITED pay/invoice.py'), out
    assert 'if self.amount is None:' in (agent.ws / 'pay/invoice.py').read_text()
    out = agent('show.py', target='pay/invoice.py', symbol='Invoice.total', file_path='show.py')
    assert 'def total' in out and 'USAGE' not in out


@BOTH_FORMS
def test_run_snippet_with_brackets(agent):
    """The payload that hung vLLM's array parser (`]` in an unquoted element) goes through as an object value."""
    agent('locate.py', query='Invoice.total amount None')
    out = agent('run.py', snippet='xs = [1, 2]; print(xs[0], "--k")')
    assert '1 --k' in out and 'exit code: 0' in out


@BOTH_FORMS
def test_create_then_check_patch_keeps_the_new_file(agent):
    agent('locate.py', query='add a receipts module')
    out = agent('create.py', target='pay/receipts.py', content='def receipt(invoice):\n    return f"total={invoice.total()}"\n')
    assert (agent.ws / 'pay/receipts.py').exists(), out
    out = agent('check_patch.py')
    assert 'pay/receipts.py' in out and 'NOT VERIFIED: nothing ran since your edit at call 2' in out, out  # v21 V1
    agent('run.py', snippet='from pay.receipts import receipt\nprint(receipt)')
    out = agent('check_patch.py')
    assert 'pay/receipts.py' in out and 'READY' in out, out


@BOTH_FORMS
def test_no_bytecode_or_state_in_workspace(agent):
    agent('locate.py', query='Invoice.total amount None')
    agent('edit.py', target='pay/invoice.py', old='if self.amount == None:', new='if self.amount is None:')
    status = subprocess.run(['git', '-C', str(agent.ws), 'status', '--porcelain', '--untracked-files=all'],
                            capture_output=True, text=True).stdout
    assert status.strip() == 'M pay/invoice.py'
    assert not list(SCRIPTS.rglob('__pycache__'))


@pytest.mark.parametrize('agent', ['positional', 'object'], indirect=True)
def test_hypothesis_and_status_routine(agent):
    """hypothesis.py and status.py (owned with the flow engine) take the object form in this branch already."""
    out = agent('hypothesis.py', add='pay/invoice.py:Invoice.total returns 0 for None',
                add2='pay/http.py:parse_header splits quoted commas')
    assert 'H1 [untested]' in out and 'H2 [untested]' in out
    assert_object_form_footer(out)
    out = agent('hypothesis.py', reject='2', evidence='parse_header is fine')
    assert 'H2 [rejected] pay/http.py:parse_header splits quoted commas  -- parse_header is fine' in out
    out = agent('hypothesis.py', confirm='1', evidence='printed 0 for [None]')
    assert 'H1 [confirmed]' in out and '-- printed 0 for [None]' in out
    out = agent('hypothesis.py', list='yes')
    assert 'H1 [confirmed]' in out and 'H2 [rejected]' in out
    out = agent('status.py')
    assert out.startswith('MEMO: no edit yet; locate.py not run yet\ncalls so far: 4'), out  # v18c: MEMO first
    assert 'mode: hypothesis' in out
    assert suggested_calls(out)[0] == ('edit.py', {'target': 'pay/invoice.py', 'old': '<exact current lines>',
                                                   'new': '<new lines>'})  # edit where H1 is confirmed
    assert_object_form_footer(out)


# ---------------------------------------------------------------- agent-v12: replay of E024 r1 fastapi_14258

def _routing_py():
    """A fastapi-like routing module: APIRouter.include_router is >150 lines (Annotated parameters with Doc
    strings, a long docstring) and its body, where the fix goes, starts near the end."""
    params = ''.join(f'        p{i}: Annotated[str, Doc("""\n            Parameter {i}.\n            """)] = "",\n'
                     for i in range(50))
    return ('"""Routing."""\nfrom typing import Annotated\n\n\ndef Doc(text):\n    return text\n\n\n'
            'class APIRoute:\n    def __init__(self, path):\n        self.path = path\n\n\n'
            'class APIRouter:\n    def __init__(self):\n        self.routes = []\n\n'
            '    def add_api_route(self, path):\n        self.routes.append(APIRoute(path))\n\n'
            '    def include_router(\n        self,\n        router: Annotated["APIRouter", Doc("The router.")],\n'
            + params + '    ) -> None:\n        """Include another APIRouter in the same app.\n\n'
            + ''.join(f'        Docs line {i}.\n' for i in range(20)) + '        """\n'
            '        for route in router.routes:\n            if isinstance(route, APIRoute):\n'
            '                self.add_api_route(route.path)\n')


def _applications_py():
    methods = ''.join(f'\n    def route_{i}(self, path):\n        """Route {i}."""\n        return self.router\n'
                      for i in range(45))
    return ('"""Applications."""\nfrom fastapi import routing\n\n\nclass FastAPI:\n'
            '    def __init__(self):\n        self.router = routing.APIRouter()\n\n'
            '    def websocket(self, path):\n        def decorator(func):\n            return func\n'
            '        return decorator\n\n'
            '    def include_router(self, router, prefix=""):\n        """Include an APIRouter."""\n'
            '        self.router.include_router(router)\n' + methods)


@pytest.fixture
def fastapi_agent(tmp_path):
    ws, state = tmp_path / 'ws', tmp_path / 'state'
    (ws / 'fastapi').mkdir(parents=True)
    (ws / 'tests').mkdir()
    (ws / 'fastapi/__init__.py').write_text('from .applications import FastAPI\nfrom .routing import APIRouter\n')
    (ws / 'fastapi/routing.py').write_text(_routing_py())
    (ws / 'fastapi/applications.py').write_text(_applications_py())
    (ws / 'tests/test_router.py').write_text('from fastapi import APIRouter\n\n\ndef test_include():\n'
                                             '    APIRouter().include_router(APIRouter())\n')
    for cmd in (['init', '-q'], ['add', '-A'], ['commit', '-qm', 'baseline']):
        subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *cmd], check=True,
                       capture_output=True)

    def call(script, **args):
        argv = [x for k, v in args.items() for x in ('--' + k, str(v))]
        env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(state), PYTHONDONTWRITEBYTECODE='1')
        env.pop('REPO_TOOLS_FLOW', None)
        return subprocess.run([sys.executable, str(SCRIPTS / script), *argv], capture_output=True, text=True,
                              env=env, timeout=120).stdout

    call.ws = ws
    return call


def test_replay_e024_fastapi_14258_r1(fastapi_agent):
    """E024 r1 fastapi_14258 (no edit, timed out): locate ranked fastapi/routing.py first; the model read
    APIRouter.include_router (the right place), then wandered to fastapi/applications.py; from then on 9 consecutive
    refusals whose NEXT pushed an edit of applications.py (the last file viewed). v12 alternated refusals with calls
    that ran. v13: every read runs; from the budget on (3 reads after locate) the first NEXT is the pre-filled
    line-range edit of routing.py's include_router body (locate's top file, read first), never the last-viewed file,
    and the second NEXT points at that code ("shown at call N")."""
    agent = fastapi_agent
    routing = (agent.ws / 'fastapi/routing.py').read_text().splitlines()
    body = routing.index('        for route in router.routes:') + 1
    seq = [('locate.py', dict(query='APIRouter.include_router circular reference: include a router into itself')),
           ('show.py', dict(symbol='APIRouter.include_router', target='fastapi/routing.py')),
           ('show.py', dict(target='fastapi/routing.py', start=body - 3, end=len(routing))),
           ('show.py', dict(symbol='FastAPI.include_router', target='fastapi/applications.py')),
           ('show.py', dict(target='fastapi/applications.py', start=9, end=13)),
           # the 6 reads that followed in the trace
           ('show.py', dict(target='fastapi/applications.py', start=20, end=60)),
           ('show.py', dict(symbol='FastAPI', target='fastapi/applications.py')),
           ('show.py', dict(symbol='FastAPI', target='fastapi/applications.py')),
           ('show.py', dict(target='fastapi/routing.py', start=1, end=100)),
           ('show.py', dict(target='fastapi/applications.py', start=1, end=100)),
           ('show.py', dict(target='fastapi/routing.py', start=1, end=50))]
    outs = [agent(script, **args) for script, args in seq]
    for out in outs:
        assert_object_form_footer(out)
    locate_out = outs[0]
    assert re.search(r'^\d+\.\d+ fastapi/routing\.py', locate_out, re.M).start() < locate_out.index('fastapi/applications.py')
    assert f'\n{body}          for route in router.routes:' in outs[1]  # the long function's body is shown (v18 window)
    assert not any(o.startswith('NOT RUN') for o in outs), [o[:80] for o in outs]
    assert outs[7].startswith('ALREADY DONE at call 7')  # the identical repeat is still replayed
    assert 'class FastAPI' in outs[6]
    reads = 0
    for i, out in enumerate(outs[1:], 1):  # from the 3rd read after locate on, the footer pushes the edit
        if out.startswith('ALREADY DONE'):
            continue
        reads += 1
        if reads < 3:
            continue
        nxt = next_lines(out)
        assert re.match(rf'NEXT: {reads} reads since locate: edit now', nxt[0]), (i, nxt)
        first = suggested_calls(out)[0]
        assert first[0] == 'edit.py' and first[1]['target'] == 'fastapi/routing.py', (i, first)
        a, b = int(first[1]['start']), int(first[1]['end'])
        assert a <= body <= b and first[1]['new'] == f'<new text for lines {a}-{b}>'
        # v17 rule 8b: once the context estimate reaches COMPACT_FROM_TOKENS the old-text tail is dropped
        assert re.match(r'NEXT: the code most likely to change: fastapi/routing\.py APIRouter\.include_router lines '
                        rf'{a}-{b} \(shown at call [23]\)(?:$|; or edit by text: edit\.py  target=fastapi/routing\.py)',
                        nxt[1]), nxt
    # the pre-filled edit works as it stands once the placeholder is replaced
    first = suggested_calls(outs[-1])[0][1]
    a, b = int(first['start']), int(first['end'])
    guard = ['        if router is self:',
             '            raise AssertionError("Cannot include the same APIRouter instance into itself.")']
    new = '\n'.join(routing[a - 1:body - 1] + guard + routing[body - 1:b])
    out = agent('edit.py', target='fastapi/routing.py', start=a, end=b, new=new)
    assert out.startswith('EDITED fastapi/routing.py'), out


# ---------------------------------------------------------------- agent-v13: replay of E025 r1 rich_3777

@pytest.fixture
def rich_agent(tmp_path):
    """A rich-like package (tests/test_v13.py): Console.is_interactive is an attribute set in Console.__init__, read
    by Progress.stop and Live.start through self.console."""
    from test_v13 import BASE, CONSOLE, HELPERS, LIVE, PROGRESS
    ws, state = tmp_path / 'ws', tmp_path / 'state'
    files = {'rich/__init__.py': '', 'rich/console.py': CONSOLE, 'rich/_helpers.py': HELPERS, 'rich/live.py': LIVE,
             'rich/base.py': BASE, 'rich/progress.py': PROGRESS,
             'tests/test_progress.py': 'from rich.progress import Progress\n\n\ndef test_stop():\n'
                                       '    Progress().stop()\n'}
    for rel, text in files.items():
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text(text)
    for cmd in (['init', '-q'], ['add', '-A'], ['commit', '-qm', 'baseline']):
        subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *cmd], check=True,
                       capture_output=True)

    def call(script, **args):
        argv = [x for k, v in args.items() for x in ('--' + k, str(v))]
        env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(state), PYTHONDONTWRITEBYTECODE='1')
        env.pop('REPO_TOOLS_FLOW', None)
        return subprocess.run([sys.executable, str(SCRIPTS / script), *argv], capture_output=True, text=True,
                              env=env, timeout=120).stdout

    call.ws = ws
    return call


def test_replay_e025_rich_3777_r1(rich_agent):
    """E025 r1 rich_3777 (no edit, timed out): the thinking asked "where is is_interactive set? I need to see it
    before adding TTY_INTERACTIVE". show.py symbol=Console.is_interactive said "not found" (it is an attribute set
    in Console.__init__), search regex=def is_interactive found nothing and was refused, a show of Console.__init__
    was refused, and the model read console.py through run.py snippets until the reproduction cap refused those too
    (v13: that cap is soft too).
    v13: every read runs; show.py prints the assignment, search says where the attribute is defined, a show of
    Progress.stop resolves self.console.is_interactive to console.py, the snippets run with a show.py NOTE, and from
    the budget on every footer's first NEXT is the line-range edit."""
    agent = rich_agent
    n = next(i for i, l in enumerate((agent.ws / 'rich/console.py').read_text().splitlines(), 1)
             if 'self.is_interactive = (' in l)
    snippet = 'with open("rich/console.py", "r") as f:\n    lines = f.readlines()\n    for i in range({}, {}):\n' \
              '        print(f"{{i+1}}: {{lines[i].strip()}}")'
    seq = [('locate.py', dict(query='TTY_COMPATIBLE')),
           ('show.py', dict(symbol='Console.is_terminal', target='rich/console.py')),
           ('locate.py', dict(query='rich/progress.py rich/status.py TTY_INTERACTIVE Progress.stop is_interactive')),
           ('show.py', dict(target='rich/progress.py', symbol='Progress.stop')),
           ('show.py', dict(symbol='Console.is_interactive', target='rich/console.py')),
           ('outline.py', dict(files='rich/console.py')),
           ('search.py', dict(regex='def is_interactive', within='rich/console.py')),
           ('search.py', dict(text='is_interactive')),
           ('show.py', dict(target='rich/console.py', start=n)),
           ('run.py', dict(snippet=snippet.format(10, 30))),
           ('run.py', dict(snippet=snippet.format(15, 25))),
           ('show.py', dict(target='rich/console.py', start=16, end=24)),
           ('run.py', dict(snippet=snippet.format(20, 30)))]
    outs = [agent(script, **args) for script, args in seq]
    for out in outs:
        assert_object_form_footer(out)
        assert not out.startswith(('NOT RUN', 'ALREADY DONE')), out[:200]
    # show of Progress.stop: the attribute it reads is resolved to where Console.__init__ assigns it
    assert re.search(rf'^names: (?:.* \| )?self\.console\.is_interactive rich/console\.py:{n} = \( '
                     r'\[in Console\.__init__\]( \||$)', outs[3], re.M), outs[3]  # v18: one names line
    # show symbol=Console.is_interactive: the assignment, not "not found"
    assert outs[4].startswith('NOTE: Console.is_interactive is an instance attribute, not a def: it is assigned in '
                              f'Console.__init__ at rich/console.py:{n}')
    assert f'\n{n}          self.is_interactive = (' in outs[4]
    # search for a def that does not exist: where the attribute is
    assert 'is_interactive is defined as:' in outs[6] and \
        'show.py  target=rich/console.py  symbol=Console.is_interactive' in outs[6]
    # search: enclosing definitions and context
    assert f'rich/console.py:{n} Console.__init__:' in outs[7] and 'rich/progress.py:17 Progress.stop:' in outs[7]
    # the snippets that read console.py run, with the equivalent show.py call
    for out, (a, b) in zip((outs[9], outs[10], outs[12]), ((11, 30), (16, 25), (21, 30))):
        assert f'show.py  target=rich/console.py  start={a}  end={b}' in out.splitlines()[0] and 'exit code: 0' in out
    # from the budget on (3 reads after the first locate), the first NEXT is the line-range edit
    reads_since = 0
    for i, out in enumerate(outs):
        script = seq[i][0]
        if i == 0 or script == 'locate.py':
            continue
        reads_since += 1
        if reads_since >= 3:
            nxt = next_lines(out)
            assert re.match(rf'NEXT: {reads_since} reads since locate: edit now', nxt[0]), (i, nxt)
            first = suggested_calls(out)[0]
            assert first[0] == 'edit.py' and first[1]['target'].startswith('rich/') and 'start' in first[1], first
            assert nxt[1].startswith('NEXT: the code most likely to change: ' + first[1]['target'])
