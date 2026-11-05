"""Repository code with invalid escapes ("\\d") must not make the read scripts print Python's SyntaxWarnings to stderr
(flow.py silences them process-wide; rft3: 56% of the oracle locate results carried them, status "warning"), while
run.py's snippet subprocess still reports the snippet's own warning."""

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'agent/skills/repo-tools/scripts'
ESC = 'import re\n\nPATTERN = re.compile("\\d+")\n\n\ndef parse_count(text):\n    """Count \\d digits."""\n' \
      '    return len(PATTERN.findall(text))\n'


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / 'ws'
    (w / 'pkg').mkdir(parents=True)
    (w / 'pkg' / '__init__.py').write_text('')
    (w / 'pkg' / 'counter.py').write_text(ESC)
    for n in range(320):  # > kb.PARALLEL_MIN_FILES: a forked child parses half of the files
        (w / 'pkg' / f'mod{n}.py').write_text(f'import re\nR{n} = re.compile("\\w+\\s")\n\n\ndef f{n}():\n    return R{n}\n')
    for cmd in (['init', '-q'], ['add', '-A'], ['-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'base']):
        subprocess.run(['git', *cmd], cwd=w, check=True, capture_output=True)
    return w


def run(ws, tmp_path, script, **args):
    env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(tmp_path / 'state'), PYTHONDONTWRITEBYTECODE='1')
    env.pop('REPO_TOOLS_FLOW', None)
    env.pop('PYTHONWARNINGS', None)
    argv = [sys.executable, str(SCRIPTS / script)] + [x for k, v in args.items() for x in (f'--{k}', v)]
    return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=180, cwd=tmp_path)


def test_the_file_does_warn_when_parsed(ws):
    r = subprocess.run([sys.executable, '-W', 'default', '-c', f'import ast; ast.parse(open({str(ws / "pkg/counter.py")!r}).read())'],
                       capture_output=True, text=True)
    assert 'invalid escape sequence' in r.stderr  # (the fixture really has the problem)


def test_read_scripts_print_no_syntax_warnings(ws, tmp_path):
    for script, args in (('locate.py', {'query': 'parse_count counts digits wrong PATTERN'}),
                         ('show.py', {'target': 'pkg/counter.py', 'symbol': 'parse_count'}),
                         ('search.py', {'text': 'PATTERN'}),
                         ('outline.py', {'files': 'pkg/counter.py'})):
        r = run(ws, tmp_path, script, **args)
        assert r.returncode == 0, (script, r.stdout, r.stderr)
        assert 'counter.py' in r.stdout, (script, r.stdout)
        assert r.stderr == '', (script, r.stderr)


def test_run_snippet_still_reports_its_own_warning(ws, tmp_path):
    r = run(ws, tmp_path, 'run.py', snippet='import re\nprint(len(re.findall("\\d", "a1b2")))')
    assert '2' in r.stdout
    assert 'invalid escape sequence' in r.stdout + r.stderr, (r.stdout, r.stderr)
