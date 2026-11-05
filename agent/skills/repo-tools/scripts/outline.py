"""Print the structure of one or more Python files: classes, functions and top-level names with line ranges.

usage: outline.py  files=billing/invoice.py billing/tax.py   (space-separated; + optional grep=total)
       (positional form, still accepted: outline.py [--grep TEXT] FILE [FILE ...])

Paths are relative to /workspace. With "grep", only definitions whose name or signature contains TEXT
(case-insensitive) are shown. A directory lists its Python files with line counts. Output is capped so it stays small.
"target" is read as files (NOTE line); no files at all: the file viewed last, else locate's top file.
v18 (the largest output per call in E041): functions as `L12-30 def name(a, b, *, c)` (parameter names only, at most
SIG_CHARS characters), a one-line range as `L12`, the module-level names on wrapped `names:` lines, at most
MAX_DEFS definitions per file without grep (more: each class's methods as one wrapped list of names), and at most
CAP characters in all.
"""

import ast
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402  (flow engine: refusals + NEXT/BUDGET footers)
import argfix  # noqa: E402
from argfix import Misuse  # noqa: E402

ROOT = (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))  # overridable for local testing
MAX_LINES = 120
MAX_DEFS = 40          # v18: definitions listed per file (without grep)
SIG_CHARS = 60         # v18: characters of a function's parameter list
CAP = 3500             # v18: characters of output at most
MAX_DIR_FILES = 40     # v18: files listed for a directory
USAGE = '\n'.join(__doc__.strip().splitlines()[2:4])


def signature(node):
    """'def name(a, b, *, c, **kw)': parameter names only, cut at SIG_CHARS characters."""
    a = node.args
    params = [x.arg for x in a.posonlyargs] + (['/'] if a.posonlyargs else []) + [x.arg for x in a.args]
    if a.vararg:
        params.append('*' + a.vararg.arg)
    elif a.kwonlyargs:
        params.append('*')
    params += [x.arg for x in a.kwonlyargs] + (['**' + a.kwarg.arg] if a.kwarg else [])
    args = ', '.join(params)
    if len(args) > SIG_CHARS:
        args = args[:SIG_CHARS - 4].rsplit(', ', 1)[0] + ', ...'
    prefix = 'async def' if isinstance(node, ast.AsyncFunctionDef) else 'def'
    return f'{prefix} {node.name}({args})'


def span(node):
    end = getattr(node, 'end_lineno', None) or node.lineno
    return f'L{node.lineno}' if end == node.lineno else f'L{node.lineno}-{end}'


def count_lines(path):
    try:
        with open(path, 'rb') as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def list_dir(rel):
    base = os.path.join(ROOT, rel)
    py, other = [], []
    for d, dirs, files in os.walk(base):
        dirs[:] = sorted(x for x in dirs if x not in argfix.SKIP_DIRS and not x.startswith('.'))
        for f in sorted(files):
            if not f.startswith('.') or rel.startswith('.'):
                p = os.path.join(d, f)
                (py if f.endswith('.py') else other).append(f'  {os.path.relpath(p, ROOT)} ({count_lines(p)} lines)')
    if py:
        more = [f'  ... {len(py) - MAX_DIR_FILES} more'] if len(py) > MAX_DIR_FILES else []
        return [f'{rel.rstrip("/")}/ has {len(py)} Python files; definitions: '
                + argfix.call('outline.py', py[0].split(' (')[0].strip())] + py[:MAX_DIR_FILES] + more
    more = [f'  ... {len(other) - MAX_DIR_FILES} more'] if len(other) > MAX_DIR_FILES else []
    return [f'{rel.rstrip("/")}/ has no Python files; {len(other)} other files; read one with '
            + argfix.call('show.py', (other[0].split(' (')[0].strip() if other else 'FILE'), 1, 100)] \
        + other[:MAX_DIR_FILES] + more


def outline(rel, grep):
    rel = argfix.clean_path(rel)
    path = os.path.join(ROOT, rel)
    if os.path.isdir(path):
        return list_dir(rel)
    if not os.path.exists(path):
        try:
            rel, note = argfix.resolve(rel, 'outline.py', nearby=True)
            path = os.path.join(ROOT, rel)
        except Misuse as e:
            return [str(e)]
        prefix = [f'NOTE: {note}']
    else:
        prefix = []
    if not rel.endswith('.py'):
        n = count_lines(path)
        return prefix + [f'{rel} is not a Python file ({n} lines); read it with '
                         + argfix.call('show.py', rel, 1, min(n, 100))]
    try:
        src = open(path, encoding='utf-8', errors='replace').read()
        tree = ast.parse(src)
    except SyntaxError as e:
        return prefix + [f'{rel}: SyntaxError line {e.lineno}: {e.msg}']
    out = prefix + [f'{rel} ({len(src.splitlines())} lines)']
    defs, assigns = [], []  # defs: (depth, line, [method entries 'name L1-9'] of a class)

    def walk(node, depth):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                line = f'class {ch.name}'
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                line = signature(ch)
            elif depth == 0 and isinstance(ch, (ast.Assign, ast.AnnAssign)):
                targets = ch.targets if isinstance(ch, ast.Assign) else [ch.target]
                for t in targets:
                    if isinstance(t, ast.Name) and (not grep or grep in t.id.lower()):
                        assigns.append(f'{t.id} {span(ch)}')
                continue
            else:
                continue
            if not grep or grep in line.lower():
                short = f'{ch.name} {span(ch)}' if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else ''
                defs.append((depth, f'{"  " * (depth + 1)}{span(ch)}  {line}', short))
            if isinstance(ch, ast.ClassDef) or (grep and isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef))):
                walk(ch, depth + 1)

    walk(tree, 0)
    if assigns:
        out += names_lines(assigns, '  names: ')
    if grep or len(defs) <= MAX_DEFS:
        return out + [d[1] for d in defs]
    # v18: too many definitions: the methods of each class as one wrapped list of names (no signatures)
    i = 0
    while i < len(defs):
        depth, line, _ = defs[i]
        out.append(line)
        j = i + 1
        while j < len(defs) and defs[j][0] > depth:
            j += 1
        methods = [d[2] for d in defs[i + 1:j] if d[0] == depth + 1 and d[2]]
        if methods:
            out += names_lines(methods, '    methods: ')
        i = j
    return out


def names_lines(items, head, width=110):
    """Items as wrapped lines after head ('  names: ')."""
    lines, cur, indent = [], head, ' ' * (len(head) - len(head.lstrip()) + 2)
    for it in items:
        piece = it if cur == head else ', ' + it
        if len(cur) + len(piece) > width and cur != head:
            lines.append(cur + ',')
            cur = indent + it
        else:
            cur += piece
    return lines + [cur]


def parse_object(d, notes):
    """The object form: {"files": "a.py b.py"} (+ "grep"). "target" (or another file alias) is read as files; no
    files at all: the file the model is working on (flow.current_file), when there is one."""
    if not d.get('files', '').strip():
        cur = flow.current_file()
        if cur:
            d = dict(d, files=cur)
            notes.append('no files given; read as ' + argfix.obj('outline.py', d))
    argfix.require(d, 'outline.py', 'files')
    files = argfix.split_list(d['files'])
    grep = d.get('grep', '').strip() or None
    return check_files(files, grep, notes)


def parse(argv):
    d, note = argfix.object_args(argv, 'outline.py')
    if d is not None:
        return parse_object(d, [note] if note else [])
    argv, note = argfix.unpack(argv, 'outline.py')
    notes = [note] if note else []
    if len(argv) == 1 and (' ' in argv[0].strip() or '\n' in argv[0].strip()):
        parts = argfix.split(argv[0])
        if parts and (parts[0] == '--grep' or all(argfix.looks_like_path(p) for p in parts)):
            argv = parts
            notes.append('split the single string into separate arguments')
    grep = None
    if '--grep' in argv:
        i = argv.index('--grep')
        if i + 1 >= len(argv):
            raise Misuse('--grep needs TEXT, e.g. ' + argfix.call('outline.py', '--grep', 'export', 'rich/console.py'))
        grep = argv[i + 1].strip().strip('"\'')
        argv = argv[:i] + argv[i + 2:]
        if i:
            notes.append('put --grep TEXT before the files next time')
        if grep and argfix.looks_like_path(grep) and argfix.exists(argfix.clean_path(grep)) and argv \
                and not argfix.exists(argfix.clean_path(argv[0])):
            grep, argv = argv[0], [grep] + argv[1:]  # --grep FILE TEXT
            notes.append('swapped TEXT and FILE after --grep')
    elif argv and argv[0].startswith('--grep'):
        grep, argv = argv[0][6:].lstrip('=').strip(), argv[1:]
    return check_files([a for a in argv if a.strip()], grep, notes)


def check_files(files, grep, notes):
    if not files:
        raise Misuse('outline.py needs at least one FILE, e.g. ' + argfix.call('outline.py', 'rich/console.py'))
    for f in files:
        if argfix.is_placeholder(f) and not argfix.exists(argfix.clean_path(f)):
            raise Misuse(f'{f!r} is a placeholder; pass real paths, e.g. ' + argfix.call('outline.py', 'rich/console.py'))
    return files, (grep.lower() if grep else None), notes


def main(argv):
    try:
        files, grep, notes = parse(argv)
    except Misuse as e:
        print(f'USAGE ERROR: {e}')
        print('usage: ' + USAGE.split('usage: ', 1)[-1])
        return 2
    lines = [f'NOTE: {n}' for n in notes]
    flow.fact(paths=[argfix.clean_path(f) for f in files if f.endswith('.py')] or None)
    for rel in files:
        lines += outline(rel, grep)
    size = 0
    for i, line in enumerate(lines[:MAX_LINES]):
        if size + len(line) + 1 > CAP - 80:
            print(f'... cut at {CAP} characters ({len(lines) - i} more lines); narrow with grep= or fewer files')
            return 0
        print(line)
        size += len(line) + 1
    if len(lines) > MAX_LINES:
        print(f'... {len(lines) - MAX_LINES} more lines; narrow with grep= or fewer files')
    return 0


if __name__ == '__main__':
    sys.exit(flow.main('outline.py', main))
