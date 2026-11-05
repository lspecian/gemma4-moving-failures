"""Tests for object-form arguments (argument schema v2/v3, research/notes/6-flow-design.md).

The model calls run_skill_script with args as an OBJECT, which ADK turns into argv ["--key", "value", ...]
(array-form args can freeze Kaggle's vLLM tool parser). Every script must give the same result for the object form
as for the positional form, pass values through intact (leading --, newlines, brackets, quotes, commas), and answer
unknown or missing keys with a USAGE ERROR that shows an example call.
Schema v3: examples are written in the quote-free notation `show.py  target=a.py  symbol=f` (tests/notation.py), never
JSON; the model-facing key for the repository file is "target" and search.py's directory filter is "within" (probes
showed that ANY args key named "file" or "path" makes the model skip ADK's file_path). "path", "file", "file_path",
"filepath", "filename", "file_name", "dir", "directory", "in" (and "within"/"target") are accepted as aliases, and
ADK parameters misplaced inside args (skill_name, file_path naming the script) are ignored.
Scripts run as subprocesses with WORKSPACE / REPO_TOOLS_DIR pointing at temp dirs and the flow engine off.
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'agent/skills/repo-tools/scripts'
sys.dont_write_bytecode = True  # never leave .pyc files in agent/ (the harness rejects them)
sys.path.insert(0, str(SCRIPTS))
import argfix  # noqa: E402
from notation import SHOWN_KEYS, assert_notation, calls, render  # noqa: E402

INVOICE = textwrap.dedent('''\
    """Payment helpers."""


    class Invoice:
        def __init__(self, amount):
            self.amount = amount

        def total(self, tax=0.1):
            if self.amount == None:
                return 0
            return self.amount * (1 + tax)


    def apply_discount(invoice, pct):
        return invoice.total() * (1 - pct)


    def parse_header(value):
        return value.split(",")
    ''')
CLI = 'FLAG = "-E"\nARGS = ["--verbose", "x[0]"]\n'
API = 'from pay.invoice import Invoice, apply_discount\n\n\ndef handle(x):\n    return apply_discount(Invoice(x), 0.1)\n'
TEST_INVOICE = ('from pay.invoice import Invoice\n\n\ndef test_total():\n    assert round(Invoice(10).total(), 2) == 11\n\n\n'
                'def test_none():\n    assert Invoice(None).total() == 0\n')
LONG = ''.join(f'def f{i}(x):\n    return x + {i}\n\n\n' for i in range(60))


def git(ws, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t',
               GIT_COMMITTER_EMAIL='t@t', GIT_AUTHOR_DATE='2024-01-01T10:00:00 +0000',
               GIT_COMMITTER_DATE='2024-01-01T10:00:00 +0000')
    return subprocess.run(['git', '-C', str(ws), *args], check=True, capture_output=True, text=True, env=env).stdout


def write(ws, rel, text):
    p = ws / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    write(root, 'pay/__init__.py', '')
    write(root, 'pay/invoice.py', INVOICE)
    write(root, 'tests/test_invoice.py', TEST_INVOICE)
    write(root, 'notes.md', ''.join(f'- note {i}\n' for i in range(10)))
    git(root, 'add', '-A')
    git(root, 'commit', '-qm', 'Add invoice module')
    write(root, 'pay/cli.py', CLI)
    write(root, 'pay/api.py', API)
    write(root, 'pay/long.py', LONG)
    write(root, 'CHANGELOG.md', ''.join(f'- change {i}\n' for i in range(150)))
    git(root, 'add', '-A')
    git(root, 'commit', '-qm', 'Fix discount handling in invoice totals')
    return root


def copy(ws, name):
    """A second, identical repository (for scripts that change files)."""
    dst = ws.parent / name
    shutil.copytree(ws, dst, symlinks=True)
    return dst


def run(script, *args, ws, state=None):
    env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(state or ws.parent / f'state-{ws.name}'),
               REPO_TOOLS_FLOW='off', PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *[str(a) for a in args]], capture_output=True,
                       text=True, env=env, timeout=120)
    assert 'Traceback' not in r.stderr, r.stderr
    return r.returncode, r.stdout


def obj(**kw):
    """The argv ADK builds from args={"k": "v", ...}: ["--k", "v", ...] in insertion order."""
    out = []
    for k, v in kw.items():
        out += [f'--{k}', str(v)]
    return out


def same(script, positional, d, ws):
    """Run both forms on the same repository; they must print the same thing and exit the same way."""
    a = run(script, *positional, ws=ws)
    b = run(script, *obj(**d), ws=ws)
    assert a == b, f'\npositional: {a}\nobject:     {b}'
    return b


def src(ws, rel):
    return (ws / rel).read_text()


# ============================== the parser (argfix) ==============================

def test_detects_object_form_and_reads_values_strictly_by_position():
    argv = ['--file', 'a.py', '--old', '--file', '--new', '-x\n]"q", [1, 2]']
    d, note = argfix.object_args(argv, 'edit.py')
    assert d == {'file': 'a.py', 'old': '--file', 'new': '-x\n]"q", [1, 2]'} and note is None
    assert list(d) == ['file', 'old', 'new']  # insertion order kept


def test_empty_argv_is_the_empty_object():
    assert argfix.object_args([], 'check_patch.py') == ({}, None)
    assert argfix.object_args([], 'show.py') == ({}, None)


@pytest.mark.parametrize('script,argv', [
    ('show.py', ['pay/invoice.py', 'Invoice.total']),
    ('show.py', ['pay/invoice.py', '1', '20']),
    ('edit.py', ['--lines', 'a.py', '1', '2', 'x']),
    ('edit.py', ['a.py', 'old', 'new']),
    ('search.py', ['-E', 'a|b']),
    ('search.py', ['--recursive', 'needle']),         # a grep flag of the positional form, not a key
    ('search.py', ['--regex', 'a|b', 'pay/']),        # odd length
    ('locate.py', ['--k', '2', '--show', 'Invoice total']),  # valueless --show, then the query
    ('locate.py', ['--no-show', 'Invoice total']),
    ('outline.py', ['--grep', 'total', 'pay/invoice.py']),
    ('run.py', ['test', 'tests/test_invoice.py', '-k', 'total']),
    ('run.py', ['snippet', 'print(1)']),
    ('check_patch.py', ['--clean']),
    ('history.py', ['--similar', 'a', 'b']),
    ('create.py', ['pay/x.py', '{"a": 1}']),
])
def test_positional_form_is_not_mistaken_for_an_object(script, argv):
    assert argfix.object_args(argv, script) == (None, None)


def test_json_object_packed_into_one_string():
    d, note = argfix.object_args(['{"file": "pay/invoice.py", "start": 3, "end": 9}'], 'show.py')
    assert d == {'file': 'pay/invoice.py', 'start': '3', 'end': '9'} and 'JSON' in note
    assert argfix.object_args(['{"nope": 1}'], 'search.py') == (None, None)  # search text that looks like a dict


def test_unknown_and_repeated_keys_raise_misuse_with_an_object_example():
    with pytest.raises(argfix.Misuse) as e:
        argfix.object_args(['--name', 'a.py', '--symbol', 'f'], 'show.py')
    msg = str(e.value)
    assert 'unknown key name for show.py' in msg and 'valid keys for show.py: target, symbol, start, end' in msg
    assert 'show.py  target=billing/invoice.py  symbol=Invoice.total' in msg
    assert_notation(msg)
    with pytest.raises(argfix.Misuse) as e:
        argfix.object_args(['--file', 'a.py', '--file', 'b.py'], 'show.py')
    assert 'twice' in str(e.value)
    with pytest.raises(argfix.Misuse) as e:  # two aliases of the same key are a repeat too
        argfix.object_args(['--target', 'a.py', '--path', 'b.py'], 'show.py')
    assert 'key target given twice' in str(e.value)
    assert_notation(str(e.value))


FILE_KEYS = ['target', 'path', 'file', 'file_path', 'filepath', 'filename', 'file_name', 'within', 'dir', 'directory',
             'in']


@pytest.mark.parametrize('key', FILE_KEYS)
@pytest.mark.parametrize('script', ['show.py', 'history.py', 'edit.py', 'create.py'])
def test_target_and_its_aliases_name_the_file(script, key):
    rest = {'show.py': {'symbol': 'f'}, 'history.py': {}, 'edit.py': {'old': 'x', 'new': 'y'},
            'create.py': {'content': 'X = 1\n'}}[script]
    argv = [f'--{key}', 'a.py'] + [x for k, v in rest.items() for x in (f'--{k}', v)]
    assert argfix.object_args(argv, script) == ({'file': 'a.py', **rest}, None)


def test_search_within_is_the_directory_filter_and_aliases_map_to_it():
    assert argfix.object_args(['--text', 'x', '--within', 'pay/'], 'search.py') == ({'text': 'x', 'path': 'pay/'}, None)
    for key in ('path', 'dir', 'directory', 'in', 'file', 'file_path', 'filepath', 'filename', 'target'):
        assert argfix.object_args(['--text', 'x', f'--{key}', 'pay/'], 'search.py') == \
            ({'text': 'x', 'path': 'pay/'}, None), key


@pytest.mark.parametrize('stray', [['--skill_name', 'repo-tools'], ['--file_path', 'edit.py'],
                                   ['--file_path', 'scripts/edit.py'], ['--filepath', 'edit.py'],
                                   ['--skill_name', 'repo-tools', '--file_path', 'edit.py']])
def test_adk_parameters_inside_args_are_dropped(stray):
    argv = stray + ['--target', 'a.py', '--old', 'x', '--new', 'y']
    assert argfix.object_args(argv, 'edit.py') == ({'file': 'a.py', 'old': 'x', 'new': 'y'}, None)
    assert argfix.to_positional('edit.py', argv) == ['a.py', 'x', 'y']
    # only the script's own name is dropped: file_path naming a repository file is the path
    assert argfix.object_args(['--file_path', 'a.py', '--old', 'x', '--new', 'y'], 'edit.py')[0]['file'] == 'a.py'
    # nothing but stray keys is the empty object
    assert argfix.object_args(['--skill_name', 'repo-tools'], 'check_patch.py') == ({}, None)


def test_keys_are_case_insensitive():
    assert argfix.object_args(['--File', 'a.py', '--Symbol', 'f'], 'show.py')[0] == {'file': 'a.py', 'symbol': 'f'}
    assert argfix.object_args(['--Path', 'a.py', '--Symbol', 'f'], 'show.py')[0] == {'file': 'a.py', 'symbol': 'f'}
    assert argfix.object_args(['--Target', 'a.py', '--Symbol', 'f'], 'show.py')[0] == {'file': 'a.py', 'symbol': 'f'}
    assert argfix.object_args(['--text', 'x', '--Within', 'a/'], 'search.py')[0] == {'text': 'x', 'path': 'a/'}


@pytest.mark.parametrize('script,positional', [
    ('locate.py', ['Invoice total is wrong']),
    ('locate.py', ['--k', '3', '--no-show', 'Invoice total']),
    ('show.py', ['pay/invoice.py', 'Invoice.total']),
    ('show.py', ['pay/invoice.py', '10', '20']),
    ('outline.py', ['pay/invoice.py', 'pay/api.py']),
    ('outline.py', ['--grep', 'total', 'pay/invoice.py']),
    ('search.py', ['def total']),
    ('search.py', ['-E', 'def (a|b)', 'pay/']),
    ('callers.py', ['apply_discount']),
    ('tests_for.py', ['pay/invoice.py']),
    ('history.py', ['pay/invoice.py']),
    ('history.py', ['--similar', 'invoice totals']),
    ('edit.py', ['pay/invoice.py', 'a ]', 'b\n"c"']),
    ('edit.py', ['--lines', 'pay/invoice.py', '3', '4', 'x = [1, 2]']),
    ('create.py', ['pay/new.py', 'X = 1\n']),
    ('run.py', ['test', 'tests/test_invoice.py', '-k', 'total']),
    ('run.py', ['snippet', 'print([1][0])']),
    ('check_patch.py', []),
    ('check_patch.py', ['--clean']),
])
def test_to_object_and_to_positional_round_trip(script, positional):
    d = argfix.to_object(script, positional)
    assert d is not None
    assert argfix.to_positional(script, obj(**d)) == positional
    assert argfix.to_positional(script, positional) == positional  # the positional form is left alone


def test_call_renders_the_object_form():
    assert argfix.call('show.py', 'a.py', '1', '100') == 'show.py  target=a.py  start=1  end=100'
    assert argfix.call('search.py', '-E', 'a|b') == 'search.py  regex=a|b'
    assert argfix.call('search.py', 'x', 'pay/') == 'search.py  text=x  within=pay/'  # search's directory filter
    assert argfix.call('run.py', 'test', 't.py', '-k', 'x') == 'run.py  test=t.py  k=x'
    assert argfix.call('check_patch.py') == 'check_patch.py'  # no args: the script name alone
    line = argfix.call('edit.py', 'a.py', 'x = [1]', 'y = "]"')
    assert line == 'edit.py  target=a.py  old=x = [1]  new=y = "]"'
    assert calls(line) == [('edit.py', {'target': 'a.py', 'old': 'x = [1]', 'new': 'y = "]"'})]


def test_text_notation():
    assert argfix.text('show.py', {'file': 'a.py', 'symbol': 'f'}) == render('show.py', target='a.py', symbol='f')
    assert argfix.text('check_patch.py', {}) == argfix.text('check_patch.py') == 'check_patch.py'
    # values never contain two spaces or newlines: whitespace is collapsed, so the separators stay unambiguous
    t = argfix.text('edit.py', {'file': 'a.py', 'old': 'if  x:\n    y', 'new': '\tz  '})
    assert t == 'edit.py  target=a.py  old=if x: y  new=z'
    assert calls(t + '  or  ' + argfix.text('check_patch.py')) == [
        ('edit.py', {'target': 'a.py', 'old': 'if x: y', 'new': 'z'}), ('check_patch.py', {})]
    assert argfix.shown_keys('edit.py') == ['target', 'old', 'new', 'start', 'end']
    assert argfix.shown_keys('search.py') == ['text', 'regex', 'within']
    assert argfix.shown_keys('tests_for.py') == ['target']  # its own key, not an alias
    assert argfix.text('search.py', {'regex': 'a|b', 'path': 'pay/'}) == render('search.py', regex='a|b', within='pay/')
    for script in argfix.KEYS:  # no model-facing key is named file or path
        assert not {'file', 'path'} & set(argfix.shown_keys(script)), script


def test_every_script_has_a_schema_and_examples():
    in_scope = {p.name for p in SCRIPTS.glob('*.py')} - {'argfix.py', 'kb.py', 'flow.py', 'status.py',
                                                         'hypothesis.py', 'names.py', 'guards.py',
                                                         # v18 aliases of run.py snippet for invented names
                                                         'repro.py', 'reproduce_issue.py', 'reproduce_bug.py'}
    assert in_scope == set(argfix.KEYS)
    for script, keys in argfix.KEYS.items():
        assert set(argfix.REQUIRED[script]) <= set(keys)
        for ex in argfix.EXAMPLES[script]:
            assert set(ex) <= set(keys)
        found = assert_notation(argfix.examples(script))
        assert [s for s, _ in found] == [script] * len(argfix.EXAMPLES[script])
        doc = (SCRIPTS / script).read_text().split('"""')[1]
        # object form first, in the notation: `usage: show.py  target=...` (`check_patch.py   (no args)`)
        assert re.search(r'^usage: ' + re.escape(script) + r'(  [a-z_]+=|   \(no args\))', doc, re.M), script
    for p in SCRIPTS.glob('*.py'):  # no usage line anywhere shows JSON args
        usage = [l for l in p.read_text().split('"""')[1].splitlines() if l.startswith(('usage:', '       '))]
        assert not any('args=' in l or '{"' in l for l in usage), (p.name, usage)
        assert not any(re.search(r'(?<![\w-])(?:file|path)=', l) for l in usage), (p.name, usage)


# ============================== same result in both forms ==============================

def test_locate_same(ws):
    b = same('locate.py', ['Invoice.total returns wrong value with apply_discount'],
             {'query': 'Invoice.total returns wrong value with apply_discount'}, ws)
    assert b[0] == 0 and 'pay/invoice.py' in b[1] and '--- best match' in b[1]
    b = same('locate.py', ['--k', '2', '--no-show', 'Invoice total'], {'query': 'Invoice total', 'k': '2', 'show': 'no'},
             ws)
    assert b[0] == 0 and '--- best match' not in b[1]
    assert len([ln for ln in b[1].splitlines() if ln[:1].isdigit()]) == 2


def test_show_same(ws):
    assert same('show.py', ['pay/invoice.py', 'Invoice.total'], {'target': 'pay/invoice.py', 'symbol': 'Invoice.total'},
                ws)[1].startswith('pay/invoice.py Invoice.total L8-')
    assert 'pay/long.py L10-20 of 240' in same('show.py', ['pay/long.py', '10', '20'],
                                               {'target': 'pay/long.py', 'start': '10', 'end': '20'}, ws)[1]
    same('show.py', ['pay/long.py', '42'], {'file': 'pay/long.py', 'start': '42'}, ws)  # "file": an alias of path
    same('show.py', ['/workspace/pay/invoice.py', 'total'], {'target': '/workspace/pay/invoice.py', 'symbol': 'total'}, ws)
    same('show.py', ['pay/long.py'], {'target': 'pay/long.py'}, ws)  # FILE only: an outline, exit 2
    same('show.py', ['pay/long.py', 'START', 'END'], {'target': 'pay/long.py', 'start': 'START', 'end': 'END'}, ws)


def test_outline_same(ws):
    b = same('outline.py', ['pay/invoice.py', 'pay/api.py'], {'files': 'pay/invoice.py pay/api.py'}, ws)
    assert b[0] == 0 and 'class Invoice' in b[1] and 'def handle' in b[1]
    b = same('outline.py', ['--grep', 'total', 'pay/invoice.py'], {'files': 'pay/invoice.py', 'grep': 'total'}, ws)
    assert 'def total' in b[1] and 'parse_header' not in b[1]
    same('outline.py', ['pay/'], {'files': 'pay/'}, ws)


def test_search_same(ws):
    b = same('search.py', ['def total'], {'text': 'def total'}, ws)
    assert b[0] == 0 and 'pay/invoice.py:' in b[1]
    b = same('search.py', ['-E', 'def (total|apply_discount)', 'pay/'],
             {'regex': 'def (total|apply_discount)', 'within': 'pay/'}, ws)
    assert b[1].count('pay/invoice.py:') == 2
    same('search.py', ['no_such_thing_xyz'], {'text': 'no_such_thing_xyz'}, ws)


def test_callers_tests_for_history_same(ws):
    assert 'pay/api.py' in same('callers.py', ['apply_discount'], {'name': 'apply_discount'}, ws)[1]
    b = same('tests_for.py', ['pay/invoice.py'], {'target': 'pay/invoice.py'}, ws)
    assert 'run one of these:\n  run.py  test=tests/test_invoice.py\n' in b[1]  # a file: whole test files
    assert ('run.py', {'test': 'tests/test_invoice.py'}) in assert_notation(b[1].split('run one of these:')[1])
    b = same('tests_for.py', ['Invoice.total'], {'target': 'Invoice.total'}, ws)
    runs = assert_notation(b[1].split('run one of these:')[1])
    assert runs and all(s == 'run.py' and a['test'] == 'tests/test_invoice.py' and a['k'].startswith('test_')
                        for s, a in runs), b[1]
    assert 'Add invoice module' in same('history.py', ['pay/invoice.py'], {'target': 'pay/invoice.py'}, ws)[1]
    assert 'Add invoice module' in same('history.py', ['pay/invoice.py'], {'file_path': 'pay/invoice.py'}, ws)[1]
    assert 'Add invoice module' in same('history.py', ['pay/invoice.py'], {'path': 'pay/invoice.py'}, ws)[1]
    b = same('history.py', ['--similar', 'discount handling'], {'similar': 'discount handling'}, ws)
    assert 'Fix discount handling' in b[1]


def test_edit_same(ws):
    ws2 = copy(ws, 'repo2')
    a = run('edit.py', 'pay/invoice.py', 'if self.amount == None:', 'if self.amount is None:', ws=ws)
    b = run('edit.py', *obj(target='pay/invoice.py', old='if self.amount == None:', new='if self.amount is None:'),
            ws=ws2)
    assert a == b and a[0] == 0 and src(ws, 'pay/invoice.py') == src(ws2, 'pay/invoice.py')
    a = run('edit.py', '--lines', 'pay/invoice.py', '14', '15', 'def apply_discount(invoice, pct=0):\n'
            '    return invoice.total() * (1 - pct)', ws=ws)
    b = run('edit.py', *obj(target='pay/invoice.py', start='14', end='15', new='def apply_discount(invoice, pct=0):\n'
                            '    return invoice.total() * (1 - pct)'), ws=ws2)
    assert a == b and a[0] == 0 and 'pct=0' in src(ws2, 'pay/invoice.py')
    assert src(ws, 'pay/invoice.py') == src(ws2, 'pay/invoice.py')
    a = run('edit.py', 'pay/invoice.py', 'if self.amount == Nothing:\n    return 0', 'x', ws=ws)
    b = run('edit.py', *obj(file='pay/invoice.py', old='if self.amount == Nothing:\n    return 0', new='x'), ws=ws2)
    assert a == b and a[0] == 1 and 'CLOSEST: pay/invoice.py' in a[1]


@pytest.mark.parametrize('key', ['target', 'path', 'file', 'file_path', 'filepath', 'dir', 'in'])
def test_show_and_edit_accept_target_and_its_aliases(key, ws):
    ref = run('show.py', 'pay/invoice.py', 'Invoice.total', ws=ws)
    assert ref[0] == 0 and ref[1].startswith('pay/invoice.py Invoice.total L8-')
    assert run('show.py', *obj(**{key: 'pay/invoice.py', 'symbol': 'Invoice.total'}), ws=ws) == ref
    code, out = run('show.py', *obj(**{key: 'pay/long.py', 'start': '10', 'end': '20'}), ws=ws)
    assert code == 0 and 'pay/long.py L10-20 of 240' in out
    code, out = run('edit.py', *obj(**{key: 'pay/invoice.py', 'old': 'if self.amount == None:',
                                       'new': 'if self.amount is None:'}), ws=ws)
    assert code == 0 and out.startswith('EDITED pay/invoice.py') and 'is None' in src(ws, 'pay/invoice.py')
    code, out = run('edit.py', *obj(**{key: 'pay/invoice.py', 'start': '14', 'end': '14',
                                       'new': 'def apply_discount(invoice, pct=0):'}), ws=ws)
    assert code == 0 and 'pct=0' in src(ws, 'pay/invoice.py')


@pytest.mark.parametrize('stray', [{'skill_name': 'repo-tools'}, {'file_path': 'edit.py'},
                                   {'skill_name': 'repo-tools', 'file_path': 'scripts/edit.py'}])
def test_edit_ignores_adk_parameters_inside_args(stray, ws):
    """The vLLM parser sometimes swallows run_skill_script's own skill_name/file_path into args (LOG E023)."""
    ws2 = copy(ws, 'repo2')
    a = run('edit.py', *obj(target='pay/invoice.py', old='if self.amount == None:', new='if self.amount is None:'),
            ws=ws)
    b = run('edit.py', *obj(**stray, target='pay/invoice.py', old='if self.amount == None:',
                            new='if self.amount is None:'), ws=ws2)
    assert a == b and b[0] == 0 and 'USAGE' not in b[1] and src(ws2, 'pay/invoice.py') == src(ws, 'pay/invoice.py')
    assert 'if self.amount is None:' in src(ws2, 'pay/invoice.py')
    # stray keys after the real ones (the order the parser swallowed them in) work too
    code, out = run('edit.py', *obj(target='pay/invoice.py', old='return 0', new='return 0.0', **stray), ws=ws2)
    assert code == 0 and 'return 0.0' in src(ws2, 'pay/invoice.py')


def test_search_within_key_is_a_directory_filter(ws):
    write(ws, 'other/invoice_copy.py', 'def total():\n    return 1\n')
    code, everywhere = run('search.py', *obj(text='def total'), ws=ws)
    assert code == 0 and 'other/invoice_copy.py:' in everywhere and 'pay/invoice.py:' in everywhere
    ref = run('search.py', *obj(text='def total', within='pay/'), ws=ws)
    assert ref[0] == 0 and 'pay/invoice.py:' in ref[1] and 'other/' not in ref[1]
    assert run('search.py', 'def total', 'pay/', ws=ws) == ref  # the positional PATH
    for key in ('path', 'dir', 'directory', 'in', 'file_path', 'file', 'filepath'):
        assert run('search.py', *obj(text='def total', **{key: 'pay/'}), ws=ws) == ref, key


def test_create_same(ws):
    ws2 = copy(ws, 'repo2')
    a = run('create.py', 'pay/currency.py', 'RATES = {"EUR": 1.0}\n', ws=ws)
    b = run('create.py', *obj(target='pay/currency.py', content='RATES = {"EUR": 1.0}\n'), ws=ws2)
    assert a == b and a[0] == 0 and src(ws2, 'pay/currency.py') == 'RATES = {"EUR": 1.0}\n'


def test_run_same(ws):
    a = run('run.py', 'test', 'tests/test_invoice.py', '-k', 'total', ws=ws)
    b = run('run.py', *obj(test='tests/test_invoice.py', k='total'), ws=ws)
    norm = lambda r: (r[0], re.sub(r'in [\d.]+s', 'in Xs', r[1]))  # noqa: E731
    assert norm(a) == norm(b) and 'exit code: 0' in b[1] and '1 passed, 1 deselected' in b[1]
    same('run.py', ['snippet', 'import pay.invoice as m; print(m.Invoice(2).total())'],
         {'snippet': 'import pay.invoice as m; print(m.Invoice(2).total())'}, ws)


def test_check_patch_same(ws):
    ws2 = copy(ws, 'repo2')
    for w in (ws, ws2):
        write(w, 'pay/invoice.py', INVOICE.replace('== None', 'is None'))
        write(w, 'tests/test_invoice.py', TEST_INVOICE + '\n# touched\n')
        write(w, 'repro.py', 'print(1)\n')
    a = run('check_patch.py', ws=ws)
    b = run('check_patch.py', ws=ws2)  # {} is the empty argv
    assert a == b and a[0] == 1 and 'PROBLEMS' in a[1] and '(to remove it: check_patch.py  clean=yes)' in a[1]
    b = run('check_patch.py', *obj(clean='no'), ws=ws2)
    assert b == a and (ws2 / 'repro.py').exists()
    a = run('check_patch.py', '--clean', ws=ws)
    b = run('check_patch.py', *obj(clean='yes'), ws=ws2)
    assert a == b and a[0] == 0 and 'READY' in a[1] and not (ws2 / 'repro.py').exists()
    assert git(ws2, 'status', '--porcelain') == git(ws, 'status', '--porcelain') == ' M pay/invoice.py\n'


# ============================== values pass through intact ==============================

HARD = '--not-a-key, "quoted", \'single\', [bracket] ] and a\nsecond line, x[0]'


def test_create_keeps_hard_content_exactly(ws):
    code, out = run('create.py', *obj(target='docs/hard.txt', content=HARD + '\n'), ws=ws)
    assert code == 0 and src(ws, 'docs/hard.txt') == HARD + '\n'


def test_edit_values_with_leading_dashes_and_brackets(ws):
    code, out = run('edit.py', *obj(target='notes.md', old='- note 3\n- note 4', new='--- replaced, [x] "q"\n]'), ws=ws)
    assert code == 0, out
    assert '- note 2\n--- replaced, [x] "q"\n]\n- note 5\n' in src(ws, 'notes.md')
    code, out = run('edit.py', *obj(target='pay/cli.py', old='ARGS = ["--verbose", "x[0]"]',
                                    new='ARGS = ["--verbose", "x[0]", "--k", "a,b"]  # ]'), ws=ws)
    assert code == 0 and src(ws, 'pay/cli.py') == 'FLAG = "-E"\nARGS = ["--verbose", "x[0]", "--k", "a,b"]  # ]\n'


def test_edit_new_that_looks_like_a_key(ws):
    code, out = run('edit.py', *obj(target='notes.md', old='- note 7', new='--file'), ws=ws)
    assert code == 0 and '\n--file\n' in src(ws, 'notes.md')


def test_search_text_that_looks_like_a_flag(ws):
    code, out = run('search.py', *obj(text='-E'), ws=ws)  # positionally, -E would be the regex flag
    assert code == 0 and 'pay/cli.py:1:\n1> FLAG = "-E"' in out  # <= 3 hits: with context (v18: unpadded)
    code, out = run('search.py', *obj(text='"--verbose", "x[0]"'), ws=ws)
    assert code == 0 and 'pay/cli.py:2:' in out


def test_run_snippet_with_brackets_quotes_and_newlines(ws):
    code, out = run('run.py', *obj(snippet='x = [1, 2]\nprint("a,b", x[0], \'--k\')\nprint("]")'), ws=ws)
    assert code == 0 and 'a,b 1 --k\n]\nexit code: 0' in out


def test_locate_query_with_leading_dashes(ws):
    code, out = run('locate.py', *obj(query='--k Invoice.total fails on x[0], "None"', show='no'), ws=ws)
    assert code == 0 and 'pay/invoice.py' in out and '--- best match' not in out


def test_show_symbol_and_outline_files_with_extra_spaces(ws):
    code, out = run('outline.py', *obj(files='  pay/invoice.py,  pay/api.py '), ws=ws)
    assert code == 0 and 'class Invoice' in out and 'def handle' in out


# ============================== usage errors ==============================

@pytest.mark.parametrize('script', sorted(argfix.KEYS))
def test_unknown_key_is_a_usage_error(script, ws):
    code, out = run(script, *obj(bogus='x'), ws=ws)
    assert code == 2 and out.startswith(f'USAGE ERROR: unknown key bogus for {script}; ')
    assert f'valid keys for {script}: ' + ', '.join(SHOWN_KEYS[script]) + '. E.g. ' in out
    msg = out.split('\nusage:')[0]  # the correction (the script's usage text follows it)
    shown = [{{'file': 'target', 'path': 'within'}.get(k, k): ' '.join(v.split()) for k, v in ex.items()}
             for ex in argfix.EXAMPLES[script][:1]]  # v18: one example
    assert assert_notation(msg)[-1] == (script, shown[0])  # the example, parsed (the last call in the message)


@pytest.mark.parametrize('script,d,missing', [
    ('locate.py', {}, 'query'),
    ('locate.py', {'show': 'no'}, 'query'),
    ('locate.py', {'k': '3'}, 'query'),
    ('show.py', {}, 'target'),
    ('show.py', {'start': '1', 'end': '5'}, 'target'),  # (a symbol alone is looked up; nothing viewed yet here)
    ('show.py', {'target': 'pay/long.py', 'end': '20'}, 'start'),
    ('outline.py', {}, 'files'),
    ('outline.py', {'grep': 'total'}, 'files'),
    ('search.py', {}, 'text'),
    ('search.py', {'within': 'pay/'}, 'text'),
    ('search.py', {'dir': 'pay/'}, 'text'),
    ('callers.py', {}, 'name'),
    ('tests_for.py', {}, 'target'),
    ('history.py', {}, 'target'),
    ('history.py', {'filename': ' '}, 'target'),                # an alias of target, blank
    ('edit.py', {}, 'target'),
    ('edit.py', {'path': ' ', 'old': 'x', 'new': 'y'}, 'target'),
    ('edit.py', {'target': 'pay/invoice.py', 'old': 'x'}, 'new'),
    ('edit.py', {'target': 'pay/invoice.py', 'new': 'x'}, 'old'),
    ('edit.py', {'target': 'pay/invoice.py', 'end': '3', 'new': 'x'}, 'start'),   # v16: start alone is one line
    ('edit.py', {'target': 'pay/invoice.py', 'start': '3'}, 'end'),
    ('edit.py', {'target': 'pay/invoice.py', 'start': '3', 'end': '4'}, 'new'),
    ('create.py', {}, 'target'),
    ('create.py', {'target': 'pay/x.py'}, 'content'),
    ('create.py', {'file': 'pay/x.py'}, 'content'),
    ('run.py', {}, 'test'),
    ('run.py', {'k': 'total'}, 'test'),
])
def test_missing_key_is_a_usage_error(script, d, missing, ws):
    code, out = run(script, *obj(**d), ws=ws)
    assert code == 2 and out.startswith('USAGE ERROR:'), out
    msg = out.split('\nusage:')[0]
    assert f'needs the key {missing} (valid keys: ' + ', '.join(SHOWN_KEYS[script]) + '). E.g. ' in msg, out
    found = assert_notation(msg)
    assert found and found[0][0] == script and all(set(a) <= set(SHOWN_KEYS[script]) for _, a in found), out


@pytest.mark.parametrize('script,d', [  # (show's symbol+start and edit's old+start are read as symbol / old: v12)
    ('search.py', {'text': 'a', 'regex': 'b'}),
    ('run.py', {'test': 'tests/test_invoice.py', 'snippet': 'print(1)'}),
    ('history.py', {'target': 'pay/invoice.py', 'similar': 'x'}),
])
def test_conflicting_keys_are_a_usage_error(script, d, ws):
    code, out = run(script, *obj(**d), ws=ws)
    assert code == 2 and 'USAGE ERROR' in out and 'not together' in out


@pytest.mark.parametrize('script,d,needle', [
    ('show.py', {'target': 'pay/invoice.py', 'start': 'total'}, '"start" must be a line number'),
    ('edit.py', {'target': 'pay/invoice.py', 'start': 'a', 'end': '4', 'new': 'x'}, 'must be line numbers'),
    ('edit.py', {'target': 'pay/invoice.py', 'start': '9', 'end': '4', 'new': 'x'}, 'is before'),
    ('locate.py', {'query': 'Invoice', 'k': 'many'}, '"k" must be a number'),
    ('locate.py', {'query': 'Invoice', 'show': 'maybe'}, 'show must be yes or no, got maybe'),
    ('check_patch.py', {'clean': 'maybe'}, 'clean must be yes or no, got maybe'),
    ('run.py', {'test': '-k total'}, 'goes in "k"'),
    ('run.py', {'snippet': '<python code>'}, 'placeholder'),
])
def test_bad_values_are_usage_errors_with_object_examples(script, d, needle, ws):
    code, out = run(script, *obj(**d), ws=ws)
    assert code == 2 and 'USAGE ERROR' in out and needle in out, out
    found = assert_notation(out.split('\nusage:')[0])
    assert any(s == script and a for s, a in found), out  # an example call with args, in the notation


@pytest.mark.parametrize('d', [{}, {'target': ''}, {'target': '  '}, {'old': 'x', 'new': 'y'},
                               {'path': ' ', 'old': 'x', 'new': 'y'},
                               {'start': '3', 'end': '4', 'new': 'x'}, {'target': ' ', 'start': '3', 'end': '4'}])
def test_edit_without_a_file_suggests_an_example_file(d, ws):
    """A blank or missing file key: the example names billing/invoice.py, never `.` (clean_path('') is '.')."""
    code, out = run('edit.py', *obj(**d), ws=ws)
    assert code == 2 and out.startswith('USAGE ERROR: edit.py needs the key'), out
    msg = out.split('\nusage:')[0]
    examples = [a for s, a in assert_notation(msg) if s == 'edit.py' and a]
    assert examples and all(a.get('target') == 'billing/invoice.py' for a in examples), msg
    assert 'target=.' not in msg and src(ws, 'pay/invoice.py') == INVOICE


def test_empty_object_where_valid(ws):
    code, out = run('check_patch.py', ws=ws)  # {} -> no argv
    assert code == 1 and 'NO CHANGES' in out
    code, out = run('show.py', ws=ws)
    assert code == 2 and 'USAGE ERROR' in out and 'show.py  target=billing/invoice.py  symbol=Invoice.total' in out


def test_positional_corrections_show_the_object_form(ws):
    code, out = run('search.py', 'Invoice', 'total', ws=ws)
    assert code == 2 and 'search.py  regex=Invoice|total' in out
    assert_notation(out.split('\nusage:')[0])
    code, out = run('run.py', 'pytest', ws=ws)
    assert 'run.py  test=' in out or 'run.py  snippet=' in out or 'exit code' in out
    assert 'args=' not in out
    code, out = run('show.py', 'pay/long.py', 'START', 'END', ws=ws)
    assert code == 2 and 'show.py  target=pay/long.py  start=1  end=100' in out
    assert '["' not in out  # no array-form examples left
    assert_notation(out.split('\nusage:')[0])


def test_no_bytecode_left_in_agent():
    assert not list(SCRIPTS.rglob('__pycache__')) and not list(SCRIPTS.rglob('*.pyc'))
