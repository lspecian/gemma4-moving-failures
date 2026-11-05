"""Create a NEW file in the repository (for example a new module, an example or documentation file, or a release script).

usage: create.py  target=billing/currency.py  content=<the whole file text>
       (positional form, still accepted: create.py FILE CONTENT)

Paths are relative to /workspace; missing directories are created. The whole file content goes in ONE string
(any length). Literal escapes (\\n \\t \\" \\') are decoded when the text as given does not parse, like edit.py;
a .py file must parse, otherwise nothing is written. Existing files are refused (change them with edit.py),
except an untracked file (e.g. one this script created earlier), which is replaced. Test files, test config and scratch files
(repro*, scratch*, tmp*) are refused: the grader discards tests and scratch files pollute the patch. v18: a scratch
.py file (repro.py, ...) or a call without a target whose content is Python code is a reproduction script: nothing is
created and the content runs as run.py snippet (recorded and budgeted as run.py), with a NOTE.
Created paths are recorded so check_patch.py keeps them in the patch.
"""

import ast
import json
import os
import re
import subprocess
import sys
import textwrap

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402  (flow engine: refusals + NEXT/BUDGET footers)
import argfix  # noqa: E402
from argfix import Misuse  # noqa: E402

ROOT = (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))  # overridable for local testing
STATE = (os.environ.get('REPO_TOOLS_DIR')  # v19b: per task, as flow.state_dir()
         or (os.path.join(ROOT, '.git', 'repo_tools') if os.path.isdir(os.path.join(ROOT, '.git')) else '/tmp/repo_tools'))
CREATED = os.path.join(STATE, 'created.txt')
PROTECTED = re.compile(
    r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)(conftest\.py|pytest\.ini|pyproject\.toml|'
    r'setup\.cfg|tox\.ini|\.pytest\.ini|sitecustomize\.py)$')
SCRATCH = re.compile(r'(^|/)(repro|reproduce|scratch|tmp|temp)([_\-.\d][^/]*)?$|\.(log|orig|rej|bak|swp|tmp|diff|patch)$',
                     re.I)
ESCAPES = {'n': '\n', 't': '\t', '"': '"', "'": "'", '\\': '\\'}
USAGE = __doc__.strip().split('\n\n')[1]
PREVIEW_DEFS = 12
EXAMPLE = ('pkg/receipts.py', 'def receipt(invoice):\n    return str(invoice)\n')  # v23: neutral (was a fastapi docs_src path)


def decode(s):
    return re.sub(r'\\(.)', lambda m: ESCAPES.get(m.group(1), m.group(0)), s)


def decode_quotes(s):
    return re.sub(r'\\(["\'])', r'\1', s)


def strip_fence(s):
    """```python ... ``` around the whole content."""
    m = re.fullmatch(r'\s*```[\w+-]*[ \t]*\n(.*?)\n?```\s*', s, re.S)
    return m.group(1) + '\n' if m else None


def py_candidates(text):
    """(how, text) in order of preference: as given, escape-decoded, quotes-only, twice decoded; each also
    dedented and without a markdown code fence."""
    bases = [text]
    fenced = strip_fence(text)
    if fenced is not None:
        bases.append(fenced)
    for b in bases:
        fence = ' (code fence removed)' if b is not text else ''
        once = decode(b)
        seq = [('', b), (' (decoded escapes)', once), (' (decoded escaped quotes)', decode_quotes(b)),
               (' (decoded escapes twice)', decode(once))]
        for how, t in seq:
            yield how + fence, t
            d = textwrap.dedent(t)
            if d != t:
                yield how + fence + ' (dedented)', d


def choose_python(text):
    """Return (content, how, None) for the first candidate that parses, else (None, failed text, SyntaxError)."""
    errors, seen = [], set()
    for how, t in py_candidates(text):
        if t in seen:
            continue
        seen.add(t)
        try:
            ast.parse(t)
            return t, how, None
        except SyntaxError as e:
            errors.append((e, t, how))
        except ValueError as e:  # null bytes
            errors.append((SyntaxError(str(e)), t, how))
    e, t, _ = max(errors, key=lambda x: (x[0].lineno or 0, x[1].count('\n')))
    return None, t, e


def choose_text(rel, text):
    """Non-Python files: decode escapes only when the text has none of its own newlines (or JSON needs it)."""
    fenced = None if rel.endswith(('.md', '.rst', '.txt')) else strip_fence(text)
    if fenced is not None:
        text = fenced
    if rel.endswith('.json'):
        for how, t in (('', text), (' (decoded escapes)', decode(text))):
            try:
                json.loads(t)
                return t, how
            except ValueError:
                continue
    if '\n' not in text.rstrip('\n') and re.search(r'\\n', text):
        once = decode(text)
        if '\n' not in once.rstrip('\n') and re.search(r'\\n', once):
            return decode(once), ' (decoded escapes twice)'
        return once, ' (decoded escapes)'
    return text, ''


def created_list():
    try:
        return [l.strip() for l in open(CREATED, encoding='utf-8') if l.strip()]
    except OSError:
        return []


def record(rel):
    if rel in created_list():
        return
    os.makedirs(STATE, exist_ok=True)
    with open(CREATED, 'a', encoding='utf-8') as fh:
        fh.write(rel + '\n')


def git(*args):
    try:
        return subprocess.run(['git', '-C', ROOT, *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None


def tracked(rel):
    r = git('ls-files', '--error-unmatch', '--', rel)
    return r is not None and r.returncode == 0


def parse(argv):
    d, note = argfix.object_args(argv, 'create.py')
    if d is not None:
        argfix.require(d, 'create.py', 'file', example={'file': EXAMPLE[0], 'content': EXAMPLE[1]})
        if 'content' not in d:
            argfix.require(d, 'create.py', 'content', example={'file': argfix.clean_path(d['file']),
                                                               'content': EXAMPLE[1]})
        return check_values(d['file'], d['content'], [note] if note else [])
    argv, note = argfix.unpack(argv, 'create.py')
    notes = [note] if note else []
    if len(argv) == 1 and '\n' in argv[0]:  # "path/to/file.py\n<content>"
        head, _, body = argv[0].partition('\n')
        if argfix.looks_like_path(head.strip()) and body.strip():
            argv = [head.strip(), body]
            notes.append('split FILE from the content; pass path and content as two keys of args')
    m = re.match(r'\s*([\w./-]+\.\w+)[ \t]*\\n', argv[0]) if len(argv) == 1 else None
    if m:  # "file.py\\n<content>" with a literal backslash-n
        if argfix.looks_like_path(m.group(1)):
            argv = [m.group(1), argv[0][m.end():]]
            notes.append('split FILE from the content; pass path and content as two keys of args')
    if not argv:
        raise Misuse('create.py needs FILE and CONTENT, e.g. ' + argfix.call('create.py', *EXAMPLE))
    if len(argv) == 1:
        raise Misuse(f'create.py needs the file CONTENT after FILE, as one string, e.g. '
                     + argfix.call('create.py', argfix.clean_path(argv[0]), EXAMPLE[1]))
    if len(argv) > 2:
        raise Misuse(f'create.py takes exactly FILE and CONTENT (got {len(argv)} arguments): pass the whole content '
                     'as ONE string with its lines joined by \\n, e.g. ' + argfix.call('create.py', *EXAMPLE))
    return check_values(*argv, notes)


def check_values(raw_path, content, notes):
    if argfix.is_placeholder(raw_path) or not raw_path.strip():
        raise Misuse(f'{raw_path!r} is a placeholder: pass the real path of the new file, e.g. '
                     + argfix.call('create.py', *EXAMPLE))
    stripped = content.strip()
    if argfix.is_placeholder(stripped) or re.fullmatch(r'<[^<>\n]{1,40}>', stripped):
        raise Misuse(f'CONTENT {stripped!r} is a placeholder: pass the full text of the new file, e.g. '
                     + argfix.call('create.py', *EXAMPLE))
    return raw_path, content, notes


def check_path(raw):
    """Return the repo-relative path, or print why it is refused and return None."""
    p = str(raw).strip().strip('`"\'').strip()
    if p.startswith('/') and not (p.startswith(ROOT.rstrip('/') + '/') or p.startswith('/workspace/')):
        print(f'REFUSED: {p} is outside /workspace; give a path inside the repository, e.g. {EXAMPLE[0]}')
        return None
    rel = argfix.clean_path(p)
    norm = os.path.normpath(rel)
    if norm.startswith('..') or norm in ('.', '') or rel.endswith('/'):
        print(f'REFUSED: {p!r} is not a file path inside /workspace, e.g. {EXAMPLE[0]}')
        return None
    rel = norm.replace(os.sep, '/')
    if rel.split('/')[0] == '.git' or '/.git/' in rel:
        print(f'REFUSED: {rel} is inside .git')
        return None
    if PROTECTED.search(rel):
        print(f'REFUSED: {rel} is a test or test-config file. The grader replaces the tests with its own; '
              'fix or add source files only.')
        return None
    if SCRATCH.search(rel):
        print(f'REFUSED: {rel} looks like a scratch file, and every new file ends up in the patch. To try code, use '
              'run.py snippet, e.g. ' + argfix.obj('run.py', {'snippet': '<python code>'})
              + ' (it runs from a temp file outside the repository).')
        return None
    return rel


def summary_defs(text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = 'class' if isinstance(n, ast.ClassDef) else 'def'
            out.append(f'{kind} {n.name} L{n.lineno}-{n.end_lineno}')
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            out += [f'{t.id} L{n.lineno}' for t in targets if isinstance(t, ast.Name)]
    return out


def main(argv):
    try:
        raw_path, content, notes = parse(argv)
    except Misuse as e:
        print(f'USAGE ERROR: {e}')  # v18: the error carries one example; the usage block repeated it
        return 2
    for note in notes:
        print(f'NOTE: {note}')
    rel = check_path(raw_path)
    if rel is None:
        return 1
    path = os.path.join(ROOT, rel)
    replacing = False
    if os.path.isdir(path):
        print(f'REFUSED: {rel} is an existing directory; give a file path inside it')
        return 1
    if os.path.exists(path):
        r = git('check-ignore', '-q', '--', rel)
        if not tracked(rel) and not (r is not None and r.returncode == 0):
            replacing = True  # made earlier by create.py or a snippet: not part of the repository
        else:
            n = sum(1 for _ in open(path, 'rb'))
            print(f'REFUSED: {rel} already exists ({n} lines); create.py only makes new files. Change it with '
                  + argfix.obj('edit.py', {'file': rel, 'old': '<old text>', 'new': '<new text>'})
                  + '  or replace a range with '
                  + argfix.obj('edit.py', {'file': rel, 'start': 'START', 'end': 'END', 'new': '<new text>'}))
            return 1
    parent = os.path.dirname(rel)
    blocker = next((parent.split('/')[:i] for i in range(1, len(parent.split('/')) + 1)
                    if parent and os.path.isfile(os.path.join(ROOT, *parent.split('/')[:i]))), None)
    if blocker:
        print(f'REFUSED: {"/".join(blocker)} is a file, so {rel} cannot be created under it')
        return 1

    if rel.endswith('.py'):
        text, how, err = choose_python(content)
        if text is None:
            lines = how.split('\n')  # the failed candidate that got furthest
            print(f'NOT CREATED: {rel} would not parse: SyntaxError line {err.lineno}: {err.msg}')
            if err.lineno:
                for j in range(max(1, err.lineno - 3), min(len(lines), err.lineno + 2) + 1):
                    print(f'{">>" if j == err.lineno else "  "}{j:4d}  {lines[j - 1][:200]}')
            return 1
    else:
        text, how = choose_text(rel, content)
    if text and not text.endswith('\n'):
        text += '\n'

    new_dirs = []
    d = parent
    while d and not os.path.isdir(os.path.join(ROOT, d)):
        new_dirs.append(d)
        d = os.path.dirname(d)
    os.makedirs(os.path.dirname(path) or ROOT, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    record(rel)

    n = text.count('\n')
    print(f'{"REPLACED" if replacing else "CREATED"} {rel} ({n} lines{", python syntax OK" if rel.endswith(".py") else ""})'
          f'{how}')
    if new_dirs:
        print(f'new directories: {", ".join(reversed(new_dirs))}')
        top = new_dirs[-1]
        up = os.path.dirname(top)
        sib_pkgs = [s for s in (os.listdir(os.path.join(ROOT, up)) if os.path.isdir(os.path.join(ROOT, up)) else [])
                    if os.path.isfile(os.path.join(ROOT, up, s, '__init__.py'))]
        if rel.endswith('.py') and sib_pkgs and not rel.endswith('__init__.py'):
            print(f'note: sibling packages in {up or "."}/ have __init__.py; if this must be importable as a regular '
                  f'package, also create {top}/__init__.py')
        shadow = os.path.join(ROOT, top + '.py')
        if os.path.isfile(shadow):
            print(f'note: {top}.py also exists; a package directory {top}/ with __init__.py would shadow it')
    defs = summary_defs(text) if rel.endswith('.py') else []
    if defs:
        more = f' (+{len(defs) - PREVIEW_DEFS} more)' if len(defs) > PREVIEW_DEFS else ''
        print('defines: ' + ', '.join(defs[:PREVIEW_DEFS]) + more)
    else:
        for i, line in enumerate(text.split('\n')[:5], 1):
            print(f'{i:5d}  {line[:160]}')
    r = git('check-ignore', '-q', '--', rel)
    if r is not None and r.returncode == 0:
        print(f'WARNING: {rel} matches .gitignore, so it will NOT be part of the patch; choose another path')
    return 0


def snippet_instead(argv):
    """(code, note) when the call is a reproduction script in disguise: a scratch .py target (repro.py,
    reproduce_issue.py, ...) or no target at all, with content that parses as Python (v18, note 13: 29 scratch
    refusals and 25 missing-target errors in 452 runs, nearly all repro scripts); None otherwise."""
    try:
        d, _ = argfix.object_args(argv, 'create.py')
        if d is None:
            raw, content, _ = parse(argv)
        else:
            raw, content = d.get('file', ''), d.get('content', '')
    except Misuse:
        return None
    if not str(content).strip():
        return None
    rel = argfix.clean_path(raw) if str(raw).strip() else ''
    if rel and not (rel.endswith('.py') and SCRATCH.search(rel) and not PROTECTED.search(rel)):
        return None
    text, _, _ = choose_python(str(content))
    if text is None or not text.strip():
        return None
    if rel:
        return text, (f'{rel} looks like a scratch file and every new file ends up in the patch, so nothing was created; '
                      'its content ran as run.py snippet instead')
    return text, ('no target given and the content is Python code, so nothing was created; it ran as run.py snippet '
                  'instead (to add a repository file, give target=<path>)')


if __name__ == '__main__':
    alt = snippet_instead(flow.strip_separator(sys.argv[1:]))
    if alt is not None:
        import run  # noqa: E402
        sys.exit(run.run_as_snippet(*alt))
    sys.exit(flow.main('create.py', main))
