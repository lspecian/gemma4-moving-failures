"""Run a quick check: one test file, or a short Python snippet. Bounded time and output.

usage: run.py  test=tests/test_invoice.py  k=total   one test file with pytest -x -q (k optional)
       run.py  snippet=<python code>                     code, with the repository importable
       (positional form, still accepted: run.py test FILE [-k EXPR]  |  run.py snippet "<python code>")

Runs in /workspace with a timeout (tests 90 s, snippets 60 s). Prints the exit status and the last 25 lines of output;
on a timeout also where the code was (file:line function: source), most recent call first.
Tests run with --tb=short; pytest's separator lines, progress dots and warnings summary are dropped, and runs of
library frames (site-packages, the standard library) are collapsed to one line (v18, note 13).
Snippets run from a temp file outside /workspace, so nothing is left in the patch. A snippet that printed nothing
shows the repr() of its last expression (like the interactive interpreter); a snippet whose output equals an
earlier snippet's says "same output as call N" with its last line instead of repeating it (v18).
A snippet that only reads repository source (open() of a repository file, inspect.getsource) still runs, with a NOTE
giving the equivalent show.py call (line numbers, and where the names in the code are defined).
Alias scripts (repro.py, reproduce_issue.py, reproduce_bug.py: names the model invents) run their code like
run.py snippet (alias_main); they are not listed in SKILL.md.
"""

import ast
import os
import re
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402  (flow engine: refusals + NEXT/BUDGET footers)
import argfix  # noqa: E402
import names  # noqa: E402
from argfix import Misuse  # noqa: E402

ROOT = (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))  # overridable for local testing
TIMEOUT = 90           # run.py test
SNIPPET_TIMEOUT = 60   # run.py snippet (v15: 90 -> 30 s; v16: 15 s, note 11 L8: hangs cost rich_3480/4006 90-180 s;
                       # v23: 60 s with the 8-minute budget, note 18: unfamiliar repos' examples can be slow)
HARD_EXTRA = 10        # the subprocess is killed this long after the timeout if the wrapper could not exit
STUCK_FRAMES = 8       # frames shown of where a timed-out run was
TAIL = 25
# The wrapper every run goes through: faulthandler prints every thread's stack at the timeout and exits, so a
# TIMEOUT shows where the code loops. Its own frames are dropped from the output (clean_traceback).
BOOT = '''import faulthandler, os, runpy, sys
faulthandler.dump_traceback_later(float(sys.argv[1]), exit=True, file=os.fdopen(os.dup(2), "w"))
del sys.argv[:2]
if sys.argv[0] == "-m":
    del sys.argv[0]
    runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
else:
    sys.path[0] = os.path.dirname(os.path.abspath(sys.argv[0]))
    runpy.run_path(sys.argv[0], run_name="__main__")
'''
TIMEOUT_MARK = re.compile(r'^Timeout \((?:\d+:)?\d+:\d+(?:\.\d+)?\)!$|^Timeout \(wrapper\)!$', re.M)
DUMP_FRAME = re.compile(r'^\s+File "(.+)", line (\d+) in (\S+)\s*$')
WRAPPER_FRAME = re.compile(r'^\s*File "(?:<string>|<frozen runpy>)", line \d+')
ESCAPES = {'n': '\n', 't': '\t', '"': '"', "'": "'", '\\': '\\'}
USAGE = __doc__.strip().split('\n\n')[1]
PYTEST_FLAGS = {'-v', '-vv', '-s', '-x', '-q', '-l', '-rA', '-ra', '--no-header'}
EXAMPLE_TEST = 'tests/test_console.py'


LAST_MARK = '\x1eLAST-EXPR\x1e'   # the instrumented snippet prints its last expression's repr after this mark
LAST_VAR = '__snippet_last__'
REPR_CHARS = 1500
SAME_MIN_CHARS = 160               # shorter repeated outputs are printed again (cheaper than a reference)
SAME_LINE_CHARS = 160


def show(proc_out, code, test=False):
    """Print the cleaned output (pytest's noise dropped for tests) and the exit code."""
    for line in render(proc_out, code, test):
        print(line)


def render(proc_out, code, test=False):
    lines = clean_traceback(proc_out.rstrip().splitlines())
    if test:
        lines = clean_pytest(lines)
    out = []
    if len(lines) > TAIL:
        out.append(f'... ({len(lines) - TAIL} earlier lines omitted)')
    out += [line[:300] for line in lines[-TAIL:]]
    return out + [f'exit code: {code}']


SEPARATOR = re.compile(r'^\s*([=_!*-])\1{2,}(?:\s+(.*?)\s+\1{2,})?\s*$')
PROGRESS = re.compile(r'^(?:\S+\.py\s+)?[.sFExXp]+\s*(?:\[\s*\d+%\])?\s*$')
COUNTS = re.compile(r'\b\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)\b|no tests ran')
TB_FRAME = re.compile(r'^(\S.*?):(\d+): in \S+')          # --tb=short: path:line: in func
PY_FRAME = re.compile(r'^\s*File "([^"]+)", line \d+')     # a Python traceback (collection errors, crashes)


def _library(path):
    """A frame outside the repository: site-packages, the standard library, pytest's own modules."""
    if '<frozen' in path or 'site-packages' in path or 'dist-packages' in path:
        return True
    if path.startswith('/'):
        return not path.startswith(ROOT.rstrip('/') + '/')
    return False


def clean_pytest(lines):
    """pytest output without what the model never needs (v18, note 13: ~1,379 -> ~650 characters on a failure):
    separator lines (a section title such as a test name is kept as `--- name`, the final counts as plain text),
    progress dots, the warnings summary, and runs of library frames (one `... (N library frames)` line)."""
    out, skip, lib, lib_src = [], False, 0, False

    def flush():
        nonlocal lib
        if lib:
            out.append(f'  ... ({lib} library frame{"s" if lib > 1 else ""})')
            lib = 0

    for line in lines:
        m = SEPARATOR.match(line)
        if m:
            title = (m.group(2) or '').strip()
            flush()
            skip = 'warnings summary' in title
            lib_src = False
            if COUNTS.search(title):
                out.append(title)
            elif m.group(1) in '_-' and title:  # a test's name, a Captured stdout section
                out.append(f'--- {title}')
            continue
        if skip:
            continue
        if PROGRESS.match(line) and line.strip():
            continue
        fm = TB_FRAME.match(line) or PY_FRAME.match(line)
        if fm:
            if _library(fm.group(1)):
                lib += 1
                lib_src = True
                continue
            flush()
            lib_src = False
        elif lib_src and line.startswith((' ', '\t')) and not line.lstrip().startswith(('E ', 'E\t')):
            continue  # the source lines of a dropped library frame
        else:
            lib_src = False
            flush()
        out.append(line)
    flush()
    while out and not out[-1].strip():
        out.pop()
    return out


def clean_traceback(lines):
    """Drop the frames of the timeout wrapper (BOOT: <string>, <frozen runpy>) and their source lines."""
    out, skipping = [], False
    for line in lines:
        if WRAPPER_FRAME.match(line):
            skipping = True
            continue
        if skipping and line.startswith('    '):
            continue
        skipping = False
        out.append(line)
    return out


def stuck_at(err, snippet_path=None):
    """(the frames faulthandler printed at the timeout, the rest of stderr). Frames are most recent first, wrapper
    frames dropped, at most STUCK_FRAMES, each with its source line when the file is the snippet or in /workspace."""
    lines = err.splitlines()
    i = next((k for k, l in enumerate(lines) if TIMEOUT_MARK.match(l)), None)
    if i is None:
        return [], err
    frames, inside = [], False
    for line in lines[i + 1:]:
        m = DUMP_FRAME.match(line)
        if not m or m.group(1).startswith('<'):
            continue
        path, n, fn = m.group(1), int(m.group(2)), m.group(3)
        name = 'snippet.py' if snippet_path and path == snippet_path else \
            os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path
        src = ''
        if name == 'snippet.py' or not name.startswith('/'):
            try:
                with open(path, encoding='utf-8', errors='replace') as fh:
                    src = fh.read().split('\n')[n - 1].strip()
            except (OSError, IndexError):
                pass
        ours = name == 'snippet.py' or not name.startswith('/')
        if not ours and (inside or len(frames) >= 3):
            continue  # library frames: only the innermost few (where it loops), not the callers (pytest, pluggy)
        inside = inside or ours
        frames.append(f'  {name}:{n} {fn}' + (f': {src[:160]}' if src else ''))
    return frames[:STUCK_FRAMES], '\n'.join(lines[:i])


def run(cmd, timeout=None, snippet_path=None, test=False, finish=None):
    """Run cmd ([script] or ['-m', module, ...] for python) under BOOT: at the timeout faulthandler prints where the
    code was and the process exits; that place is shown as the TIMEOUT result (v16, note 11 L8). The timeout is
    TIMEOUT for tests, SNIPPET_TIMEOUT for a snippet (snippet_path). test: pytest output (clean_pytest). finish:
    a function of the output lines of a completed run (exit code line last) returning the lines to print."""
    if timeout is None:
        timeout = SNIPPET_TIMEOUT if snippet_path else TIMEOUT
    path = os.pathsep.join([ROOT, os.path.join(ROOT, 'src')] + [os.environ.get('PYTHONPATH', '')])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=path, PYTHONUNBUFFERED='1')
    full = [sys.executable, '-c', BOOT, str(timeout)] + cmd
    try:
        p = subprocess.run(full, cwd=ROOT, capture_output=True, text=True, timeout=timeout + HARD_EXTRA, env=env)
        out, err, code = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:  # the wrapper did not get to exit (stuck in C code)
        out = (e.stdout or '') + (e.stderr or '') if isinstance(e.stdout, str) else ''
        out, err, code = out, 'Timeout (wrapper)!', None
    frames, rest = stuck_at(err, snippet_path)
    last = None
    if snippet_path:  # the temp file's name means nothing to the model; nor does the last-expression variable
        out, err, rest = (unmark(t.replace(snippet_path, 'snippet.py')) for t in (out, err, rest))
        frames = [unmark(f) for f in frames]
        out, last = take_last(out)
    if code is not None and not TIMEOUT_MARK.search(err):
        lines = render(out + err, code, test)
        if snippet_path and last is not None and not (out + err).strip():
            lines = [f'nothing printed; the last expression is: {last}'] + lines[-1:]
        for line in (finish(lines) if finish else lines):
            print(line)
        return 0
    show(out + rest, 'TIMEOUT', test)
    if frames:
        print(f'still running after {timeout}s at (most recent call first):')
        for f in frames:
            print(f)
    print(f'timed out after {timeout}s: ' + ('pick a smaller test (-k) or skip verification' if not snippet_path
                                             else 'the code does not return; the lines above show where it loops'
                                             if frames else 'the code does not return; check a smaller case or skip it'))
    return 1


CARETS = re.compile(r'^ *[~^]+ *$')


def unmark(text):
    """TEXT without the LAST_VAR assignment with_last_expr() inserted into a source line (and the traceback's caret
    line under it shifted back by as much)."""
    mark = LAST_VAR + ' = '
    if mark not in text:
        return text
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if mark in line:
            lines[i] = line.replace(mark, '')
            if i + 1 < len(lines) and CARETS.match(lines[i + 1]) and lines[i + 1].startswith(' ' * len(mark)):
                lines[i + 1] = lines[i + 1][len(mark):]
    return '\n'.join(lines)


def take_last(out):
    """(OUT without the last-expression line the instrumented snippet printed, that repr or None)."""
    i = out.rfind(LAST_MARK)
    if i < 0:
        return out, None
    j = out.find('\n', i)
    value = out[i + len(LAST_MARK):j if j >= 0 else len(out)]
    head = out[:i]
    return head + (out[j + 1:] if j >= 0 else ''), value


def with_last_expr(code):
    """CODE whose last statement, when it is an expression (not a print() call), is assigned to LAST_VAR, followed by
    a line that prints LAST_MARK + its repr. Only text inside the last statement's first line changes and one line is
    appended, so traceback line numbers stay right. CODE unchanged when it does not parse or ends otherwise."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    node = tree.body[-1]
    v = node.value
    if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ('print', 'exit', 'quit'):
        return code
    if isinstance(v, ast.Constant) and isinstance(v.value, str):  # a docstring-like string, not a question
        return code
    lines = code.split('\n')
    raw = lines[node.lineno - 1].encode('utf-8')
    lines[node.lineno - 1] = (raw[:node.col_offset] + (LAST_VAR + ' = ').encode() + raw[node.col_offset:]).decode('utf-8')
    tail = ('import sys as __snippet_sys; __snippet_sys.stdout.write("\\n\\x1eLAST-EXPR\\x1e" + '
            f'repr({LAST_VAR})[:{REPR_CHARS}] + "\\n")')
    new = '\n'.join(lines).rstrip('\n') + '\n' + tail + '\n'
    try:
        compile(new, '<snippet>', 'exec')
    except (SyntaxError, ValueError):
        return code
    return new


def _body(text):
    """A run's output for comparison: without the scripts' own NOTE lines."""
    return '\n'.join(l for l in str(text or '').splitlines() if not l.startswith(('NOTE:', 'note:'))).strip()


def same_output(lines):
    """The lines to print for a snippet: 'same output as call N' (with the last line of output and the exit code)
    when an earlier snippet of this task printed exactly this and it is long enough to be worth a reference; the
    lines themselves otherwise. Uses the flow engine's stored outputs (nothing when flow is off)."""
    body = '\n'.join(lines).strip()
    if len(body) < SAME_MIN_CHARS or os.environ.get('REPO_TOOLS_FLOW', 'on') == 'off':
        return lines
    st = flow.peek()
    if st is None:
        return lines
    for n, script, key, status, summary in reversed(st.actions):
        if script != 'run.py' or status in ('refused', 'usage') or _body(summary) != body:
            continue
        if not st.conn.execute("SELECT 1 FROM checks WHERE n=? AND kind IN ('snippet', 'read')", (n,)).fetchone():
            continue  # a test run
        edits = [e[0] for e in st.edits if e[2] and e[0] > n]
        last = next((l for l in reversed(lines[:-1]) if l.strip()), '')
        if len(last) > SAME_LINE_CHARS:
            last = last[:SAME_LINE_CHARS - 1] + '…'
        head = f'same output as call {n}' + (f' (before your edit at call {edits[-1]})' if edits else '')
        return [head + (f'; its last line: {last}' if last else '')] + lines[-1:]
    return lines


def test_files():
    return [f for f in argfix.all_files() if f.endswith('.py') and re.search(r'(^|/)test_[^/]*\.py$|_test\.py$', f)]


def check_test_path(target):
    """Return (target, note) with FILE[::node] resolved to an existing file, or raise Misuse."""
    file_part, sep, node = argfix.clean_path(target).partition('::')
    if argfix.exists(file_part):
        return file_part + sep + node, None
    if not argfix.is_placeholder(file_part):
        tail = file_part.rstrip('/')
        hits = [f for f in test_files() if f == tail or f.endswith('/' + tail)]
        if len(hits) == 1:
            return hits[0] + sep + node, f'{file_part} is not at the repository root; using {hits[0]}'
        stem = os.path.basename(tail)[:-3] if tail.endswith('.py') else os.path.basename(tail)
        near = [f for f in test_files() if stem and stem.replace('test_', '') in f][:5]
        raise Misuse(f'test file {file_part} not found' + (f'; test files like it: {", ".join(near)}' if near
                                                           else '') + '. E.g. '
                     + argfix.call('run.py', 'test', near[0] if near else EXAMPLE_TEST, '-k', 'name'))
    raise Misuse(f'{file_part!r} is a placeholder: pass a real test file, e.g. '
                 + argfix.call('run.py', 'test', EXAMPLE_TEST, '-k', 'html'))


def parse_test(rest, notes):
    """rest = the arguments after 'test'. Returns the pytest argument list."""
    if len(rest) == 1 and ' ' in rest[0].strip():
        rest = argfix.split(rest[0])
        notes.append('split the test arguments into separate strings')
    flat = []
    for a in rest:  # "-k foo" glued together
        if a.startswith('-k ') or a.startswith('-m '):
            flat += [a[:2], a[3:].strip().strip('"\'')]
        else:
            flat.append(a)
    rest = flat
    if not rest or not rest[0].strip():
        raise Misuse('run.py test needs a test FILE, e.g. ' + argfix.call('run.py', 'test', EXAMPLE_TEST, '-k', 'html'))
    targets, extra, i = [], [], 0
    while i < len(rest):
        a = rest[i]
        if a in ('-k', '-m'):
            if i + 1 >= len(rest):
                raise Misuse(f'{a} needs an expression, e.g. ' + argfix.call('run.py', 'test', rest[0], '-k', 'html'))
            extra += [a, rest[i + 1]]
            i += 2
            continue
        if a in PYTEST_FLAGS or a.startswith('--tb='):
            extra.append(a)
        elif a.startswith('-'):
            notes.append(f'ignored the pytest option {a}')
        elif not targets or argfix.looks_like_path(a):
            target, note = check_test_path(a)
            targets.append(target)
            if note:
                notes.append(note)
        elif '-k' not in extra:
            extra += ['-k', a]
            notes.append(f'read {a!r} as -k {a!r}; write it as '
                         + argfix.obj('run.py', {'test': targets[0] if targets else 'FILE', 'k': a}))
        else:
            raise Misuse(f'unexpected argument {a!r}. E.g. ' + argfix.call('run.py', 'test', targets[0], '-k', 'name'))
        i += 1
    return targets + extra


def strip_snippet_prefix(s):
    """'snippet\\ncode', 'snippet,\\ncode', 'snippet`, `code', 'snippet: code' -> code (or None)."""
    m = re.match(r'\s*snippet(?=[\s,:`\'"])[\s,:`\'"]*', s)
    if not m:
        return None
    code = s[m.end():]
    head = s[:m.end()]
    for q in '`\'"':
        if q in head and code.rstrip().endswith(q) and code.count(q) % 2 == 1:
            code = code.rstrip()[:-1]
    return code


def parse_object(d, notes):
    """The object form: {"test"} (+ "k") or {"snippet"}. Values are used as given."""
    argfix.exclusive(d, 'run.py', ('test', 'k'), ('snippet',))
    if 'snippet' in d:
        code = d['snippet']
        if not code.strip():
            argfix.require(d, 'run.py', 'snippet', example={'snippet': 'from billing.invoice import Invoice'})
        if argfix.is_placeholder(code) or code.strip() in ('<python code>', '<code>'):
            raise Misuse(f'{code!r} is a placeholder; pass real code, e.g. '
                         + argfix.obj('run.py', {'snippet': 'from billing.invoice import Invoice; print(Invoice)'}))
        return ['snippet', code], notes
    argfix.require(d, 'run.py', 'test', example={'test': 'tests/test_invoice.py', **({'k': d['k']} if 'k' in d else {})})
    if d['test'].strip().startswith('-'):
        raise Misuse(f'"test" is the test FILE, got {d["test"]!r}; the test-name filter goes in "k", e.g. '
                     + argfix.obj('run.py', {'test': 'tests/test_invoice.py', 'k': 'total'}))
    args = parse_test([d['test']], notes)
    if d.get('k', '').strip():
        args += ['-k', d['k']]
    return ['test'] + args, notes


def parse(argv):
    d, note = argfix.object_args(argv, 'run.py')
    if d is not None:
        return parse_object(d, [note] if note else [])
    argv, note = argfix.unpack(argv, 'run.py')
    notes = [note] if note else []
    if argv and argv[0] not in ('test', 'snippet'):
        first = argv[0]
        code = strip_snippet_prefix(first)
        if code is not None and code.strip():
            argv = ['snippet', code] + argv[1:]
            notes.append('separated "snippet" from the code; pass the code as the value of the key snippet')
        elif re.match(r'\s*test\s+\S', first):
            argv = ['test'] + argfix.split(first.strip()[4:]) + argv[1:]
            notes.append('split "test FILE" into separate strings')
        elif first in ('pytest', 'py.test') or argv[:3] == ['python', '-m', 'pytest'] \
                or argv[:3] == ['python3', '-m', 'pytest'] or re.match(r'\s*(python3? -m )?pytest\s', first):
            parts = argfix.split(first) + argv[1:] if ' ' in first.strip() else argv
            parts = parts[parts.index('pytest') + 1:] if 'pytest' in parts else parts[1:]
            argv = ['test'] + parts
            notes.append('read the pytest command as run.py test')
        elif first in ('python', 'python3') and len(argv) == 3 and argv[1] == '-c':
            argv = ['snippet', argv[2]]
            notes.append('read python -c as run.py snippet')
        elif len(argv) == 1 and argfix.looks_like_path(first) and re.search(r'test[^/]*\.py(::|$)', first):
            argv = ['test', first]
            notes.append('read the test file path as run.py test FILE')
        elif len(argv) == 1 and ('\n' in first or '(' in first) and compiles(first):
            argv = ['snippet', first]
            notes.append('read the code as run.py snippet')
    if not argv or argv[0] not in ('test', 'snippet'):
        raise Misuse('the first argument must be "test" or "snippet"; e.g. '
                     + argfix.call('run.py', 'snippet', 'import rich; print(rich.__file__)'))
    if argv[0] == 'test':
        return ['test'] + parse_test(argv[1:], notes), notes
    rest = argv[1:]
    if not rest or not ''.join(rest).strip():
        raise Misuse('run.py snippet needs the code, e.g. ' + argfix.call('run.py', 'snippet', 'import rich; print(1)'))
    if len(rest) > 1:
        raise Misuse('pass the whole snippet as ONE string (lines joined with \\n), e.g. '
                     + argfix.call('run.py', 'snippet', 'x = 1\nprint(x)'))
    if argfix.is_placeholder(rest[0]) or rest[0].strip() in ('<python code>', '<code>'):
        raise Misuse(f'{rest[0]!r} is a placeholder; pass real code, e.g. '
                     + argfix.call('run.py', 'snippet', 'from rich.console import Console; print(Console)'))
    return ['snippet', rest[0]], notes


def show_equivalent(code):
    """The show.py call(s) that read what a source-reading snippet reads ('' for any other snippet)."""
    reads = flow.snippet_reads(code) or []
    out = []
    for r in reads:
        if 'file' in r:
            d = {'file': r['file']}
            if r.get('start'):
                d.update(start=str(r['start']), end=str(min(r['end'], r['start'] + 99)))
            out.append(argfix.obj('show.py', d))
        elif r.get('symbol'):
            full = (r['module'] + r['symbol'][len(r['symbol'].split('.')[0]):]) if r.get('module') else r['symbol']
            parts = full.split('.')
            for i in range(len(parts) - 1, 0, -1):
                rel = names.module_file('.'.join(parts[:i]))
                if rel:
                    out.append(argfix.obj('show.py', {'file': rel, 'symbol': '.'.join(parts[i:])}))
                    break
            else:
                out.append(argfix.obj('show.py', {'symbol': r['symbol']}))
    return list(dict.fromkeys(out))


def compiles(code):
    try:
        compile(code, '<snippet>', 'exec')
        return True
    except (SyntaxError, ValueError):
        return False


def main(argv):
    try:
        args, notes = parse(argv)
    except Misuse as e:
        print(f'USAGE ERROR: {e}')  # v18: the error carries one example; the usage block repeated it
        return 2
    for note in notes:
        print(f'NOTE: {note}')
    if args[0] == 'test':
        tb = [] if any(a.startswith('--tb') for a in args[1:]) else ['--tb=short']
        return run(['-m', 'pytest', *args[1:], '-x', '-q', *tb, '-p', 'no:cacheprovider'], test=True)
    code = args[1]
    shows = show_equivalent(code)
    if shows:
        print('NOTE: this snippet only reads repository source; show.py does that with line numbers and lists where '
              'the names in the code are defined: ' + '  or  '.join(shows[:2]))
    if not compiles(code):
        # arguments sometimes arrive with literal escape sequences
        code = re.sub(r'\\(.)', lambda m: ESCAPES.get(m.group(1), m.group(0)), code)
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(with_last_expr(code))
    try:
        return run([f.name], snippet_path=f.name, finish=same_output)
    finally:
        os.unlink(f.name)


# ---------------------------------------------------------------- aliases (v18, note 13 audit #7)
# Names the model invents for a reproduction script (24 ADK-rejected calls in 452 runs, args {content} or {code}).
# Each alias file calls alias_main(); the call is recorded and budgeted as run.py snippet (flow.main('run.py')).
ALIASES = ('repro.py', 'reproduce_issue.py', 'reproduce_bug.py')
CODE_KEYS = ('snippet', 'code', 'content', 'script', 'source', 'python', 'text', 'program', 'command', 'cmd')


def alias_code(argv):
    """The code an alias call carries: the value of its first code-like key (object form), or its one positional
    string; None when there is none."""
    argv = [str(a) for a in (argv or [])]
    if len(argv) >= 2 and len(argv) % 2 == 0 and all(re.fullmatch(r'--\w+', k) for k in argv[0::2]):
        d = {k[2:].lower(): v for k, v in zip(argv[0::2], argv[1::2])}
        for k in CODE_KEYS:
            if str(d.get(k, '')).strip():
                return d[k]
        rest = [v for k, v in d.items() if k not in ('skill_name', 'file_path', 'target', 'file', 'path')]
        return rest[0] if len(rest) == 1 and rest[0].strip() else None
    if len(argv) == 2 and argv[0] == 'snippet':
        return argv[1]
    if len(argv) == 1 and argv[0].strip():
        return argv[0]
    return None


def run_as_snippet(code, note):
    """Run CODE as `run.py snippet=CODE` under the flow engine (recorded, budgeted and refused like one), printing
    NOTE first."""
    def go(argv):
        print(f'NOTE: {note}')
        return main(argv)
    return flow.main('run.py', go, argv=['--snippet', code])


def alias_main(name, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    code = alias_code(argv)
    if code is None:
        def usage(_):
            print(f'USAGE ERROR: {name} is not one of the scripts; run code with '
                  + argfix.obj('run.py', {'snippet': 'from billing.invoice import Invoice; print(Invoice(1).total())'}))
            return 2
        return flow.main('run.py', usage, argv=argv)
    return run_as_snippet(code, f'{name} is not one of the scripts; ran the code as run.py snippet=<the code> '
                                '(call run.py next time)')


if __name__ == '__main__':
    sys.exit(flow.main('run.py', main))
