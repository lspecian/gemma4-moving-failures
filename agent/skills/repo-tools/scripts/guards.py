"""Edit guards for edit.py (a helper module, not a command). agent-v15b, research/notes/10-edit-failures.md
(lever 2): cheap checks of one edit, no test runs.

- import_error(): the edited module is imported by dotted name in a subprocess (IMPORT_TIMEOUT s cap, cwd a temp
  dir, PYTHONDONTWRITEBYTECODE), so nothing is written to /workspace. edit.py undoes an edit after which the module
  no longer imports when it did before.
- property_conflict(): a read-only @property X of a class whose methods assign self.X = ... (new with this edit);
  edit.py refuses such an edit.
- escape_warnings(): \\uXXXX / \\xNN escapes in string literals on changed lines that Python decodes to a printable
  ASCII character (e.g. '\\u003c' is '<' itself), and str.replace calls whose two literals are then equal;
  v16: a changed line with twice the backslashes in a row of the line it replaced (doubled_backslashes()).
- infinite_loops() (v16, used by check_patch.py): `while True:` loops added by the patch with no break, return or
  raise in their body (a hang in the hidden tests costs the task and wall clock).
- unreachable() (v18, note 13 P6): statements an edit put after a return/raise/continue/break of the same block
  (dead code); edit.py refuses an indentation repair that does this and warns when NEW as written does.
- lost_definitions() (v17, research/notes/12-failures-eval113.md L2): module- and class-level def/class/property names
  that an edit removes (gone from the file, not re-bound anywhere at module/class level) with the places that still
  use them: the file itself, else a time-capped (REF_SECONDS) scan of the package and the tests directory.
  edit.py refuses such an edit when the name is still used and warns when it is not; check_patch.py compares HEAD
  with the working tree (fastapi_14458/14448 lost `computed_scope`, rich_3063 `_parse`, rich_3930 `cell_len`).
Stdlib only.
"""

import ast
import difflib
import io
import os
import re
import subprocess
import sys
import tempfile
import time
import tokenize

sys.dont_write_bytecode = True

IMPORT_TIMEOUT = 10
TEST_FILE = re.compile(r'(^|/)test_[^/]*\.py$|(^|/)[^/]*_test\.py$')
NO_IMPORT = re.compile(r'(^|/)(scripts|benchmarks|tools)/')


def root():
    return (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))


def module_name(rel):
    if not rel.endswith('.py'):
        return None
    parts = rel[:-3].split('/')
    if parts and parts[0] == 'src':
        parts = parts[1:]
    if parts and parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts) if parts and all(p.isidentifier() for p in parts) else None


def _changed_lines(before, after):
    """(old line numbers, new line numbers) changed by the edit, 1-based."""
    a, b = before.split('\n'), after.split('\n')
    old, new = set(), set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        old.update(range(i1 + 1, i2 + 1))
        new.update(range(j1 + 1, j2 + 1))
        if i1 == i2:  # pure insertion: the line it follows tells where
            old.add(max(1, i1))
    return old, new


def import_error(rel, timeout=IMPORT_TIMEOUT):
    """The error printed when importing REL's module fails ([lines]), None when it imports, times out or is not an
    importable module."""
    mod = module_name(rel)
    if not mod or NO_IMPORT.search(rel) or TEST_FILE.search(rel):
        return None
    ws = root()
    path = os.pathsep.join([ws, os.path.join(ws, 'src')] + ([os.environ['PYTHONPATH']] if os.environ.get('PYTHONPATH')
                                                             else []))
    with tempfile.TemporaryDirectory() as tmp:  # cwd outside the workspace: nothing written there
        try:
            r = subprocess.run([sys.executable, '-c', f'import {mod}'], cwd=tmp, capture_output=True, text=True,
                               timeout=timeout, stdin=subprocess.DEVNULL,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=path))
        except subprocess.TimeoutExpired:
            return None
        except OSError:
            return None
    if r.returncode == 0:
        return None
    lines = [l.rstrip() for l in r.stderr.splitlines() if l.strip()]
    where = [l.strip() for l in lines if l.strip().startswith('File ') and rel in l]
    return (where[-1:] if where else []) + (lines[-1:] or ['import failed'])


def import_status(rel, timeout=IMPORT_TIMEOUT):
    """(status, error lines, seconds) of importing REL's module in a subprocess (as import_error): status 'ok',
    'fail' (lines: where and the error), 'timeout', or 'skip' (not an importable module, or no interpreter)."""
    mod = module_name(rel)
    if not mod or NO_IMPORT.search(rel) or TEST_FILE.search(rel):
        return 'skip', None, 0.0
    ws = root()
    path = os.pathsep.join([ws, os.path.join(ws, 'src')] + ([os.environ['PYTHONPATH']] if os.environ.get('PYTHONPATH')
                                                             else []))
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            r = subprocess.run([sys.executable, '-c', f'import {mod}'], cwd=tmp, capture_output=True, text=True,
                               timeout=timeout, stdin=subprocess.DEVNULL,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=path))
        except subprocess.TimeoutExpired:
            return 'timeout', None, time.time() - t0
        except OSError:
            return 'skip', None, 0.0
    secs = time.time() - t0
    if r.returncode == 0:
        return 'ok', None, secs
    lines = [l.rstrip() for l in r.stderr.splitlines() if l.strip()]
    where = [l.strip() for l in lines if l.strip().startswith('File ') and rel in l]
    return 'fail', (where[-1:] if where else []) + (lines[-1:] or ['import failed']), secs


IMPLICIT = {'__file__', '__name__', '__doc__', '__spec__', '__loader__', '__package__', '__builtins__', '__path__',
            '__annotations__', '__dict__', '__module__', '__qualname__', '__class__', '__debug__', 'reveal_type',
            '__getattr__', 'WindowsError', 'unicode', 'long', 'basestring', 'xrange', 'raw_input', 'unichr', 'file',
            'reduce', 'buffer', 'cmp', 'execfile', 'intern', 'apply', 'coerce'}
DYNAMIC = re.compile(r'\b(?:globals|locals|vars)\(\)|\bexec\(|__builtins__|\bbuiltins\.')


def undefined_names(text):
    """{name: first line} of the names TEXT loads where nothing binds them (stdlib symtable + ast, pyflakes-like): not
    local to the function or class scope (parameters, assignments, imports, def/class, for/with/except/match targets,
    comprehension variables), not bound at module level (also through `global X` in a function), not a builtin.
    Empty when the module has a star import or touches globals dynamically; None when it does not parse."""
    import builtins
    import symtable
    try:
        top = symtable.symtable(text, 'x.py', 'exec')
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError, TypeError):
        return None
    if DYNAMIC.search(text) or any(isinstance(n, ast.ImportFrom) and any(a.name == '*' for a in n.names)
                                   for n in ast.walk(tree)):
        return {}
    bound = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    known = bound | set(dir(builtins)) | IMPLICIT
    loose = set()
    stack = [top]
    while stack:
        t = stack.pop()
        stack.extend(t.get_children())
        for s in t.get_symbols():
            name = s.get_name()
            if t is not top and s.is_declared_global() and s.is_assigned():
                known.add(name)
            if not s.is_referenced() or name in known:
                continue
            if t is top:
                if not (s.is_assigned() or s.is_imported() or s.is_namespace()):
                    loose.add(name)
            elif s.is_global() and not s.is_declared_global():
                loose.add(name)
            elif s.is_declared_global():
                loose.add(name)
    loose -= known
    if not loose:
        return {}
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in loose:
            if n.id not in out or n.lineno < out[n.id]:
                out[n.id] = n.lineno
    return out


def new_undefined_names(before, after):
    """[(line, name, on_changed_line)] of names undefined in AFTER (undefined_names) that were not undefined in BEFORE,
    first by line; [] when either cannot be analysed (None when AFTER cannot)."""
    a = undefined_names(after)
    if a is None:
        return None
    b = undefined_names(before)
    if b is None or not a:
        return []
    _, changed = _changed_lines(before, after)
    out = []
    tree = ast.parse(after)
    for name in sorted(set(a) - set(b)):
        lines = sorted({n.lineno for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name
                        and isinstance(n.ctx, ast.Load)})
        on = [l for l in lines if l in changed]
        out.append((on[0] if on else lines[0], name, bool(on)))
    return sorted(out)


def _class_props(cls):
    """(read-only property names {name: line}, setter names, {name: (line, method)} of self.NAME stores)."""
    props, setters, stores = {}, set(), {}
    for item in cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in item.decorator_list:
            if isinstance(d, ast.Name) and d.id == 'property':
                props[item.name] = item.lineno
            elif isinstance(d, ast.Attribute) and d.attr in ('setter', 'deleter') and isinstance(d.value, ast.Name):
                setters.add(d.value.id)
    for item in cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        stack = list(item.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            flat = []
            while targets:
                t = targets.pop()
                if isinstance(t, (ast.Tuple, ast.List)):
                    targets.extend(t.elts)
                else:
                    flat.append(t)
            for t in flat:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self':
                    if t.attr not in stores or node.lineno < stores[t.attr][0]:
                        stores[t.attr] = (node.lineno, item.name)
            stack.extend(ast.iter_child_nodes(node))
    return props, setters, stores


def _conflicts(text):
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    out = {}
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        props, setters, stores = _class_props(cls)
        for name, pline in props.items():
            if name in stores and name not in setters:
                out[(cls.name, name)] = (pline, stores[name][0], stores[name][1])
    return out


def property_conflict(before, after):
    """(line of the assignment, refusal text) when the edit creates a read-only @property X in a class whose methods
    assign self.X, else None. Line numbers are the current file's where the assignment already exists there."""
    new = _conflicts(after)
    if not new:
        return None
    old = _conflicts(before)
    for (cls, name), (pline, aline, method) in new.items():
        if (cls, name) in old:
            continue
        b_stores = {}
        try:
            for node in ast.walk(ast.parse(before)):
                if isinstance(node, ast.ClassDef) and node.name == cls:
                    b_stores = _class_props(node)[2]
                    break
        except (SyntaxError, ValueError):
            pass
        line = b_stores[name][0] if name in b_stores else aline
        where = f'line {line}' if name in b_stores else f'line {aline} after the edit'
        return line, (f'the edit makes {name!r} a read-only @property of {cls}, but {cls}.{method} assigns '
                f'self.{name} = ... ({where}) and there is no @{name}.setter: every {cls}(...) would fail with '
                f'AttributeError: property {name!r} of {cls!r} object has no setter. Change the value where '
                f'{method} assigns it ({where}) instead, or give the property another name, or add a setter.')
    return None


ESC = re.compile(r'(\\+)(u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|U[0-9a-fA-F]{8})')


def _string_tokens(text):
    """[(line, token text, is_raw)] of string literal pieces (STRING and f-string middles)."""
    out = []
    try:
        raw_stack = []
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            name = tokenize.tok_name.get(tok.type, '')
            if name == 'STRING':
                prefix = re.match(r'[A-Za-z]*', tok.string).group(0).lower()
                out.append((tok.start[0], tok.string, 'r' in prefix))
            elif name == 'FSTRING_START':
                raw_stack.append('r' in tok.string.lower())
            elif name == 'FSTRING_END':
                if raw_stack:
                    raw_stack.pop()
            elif name == 'FSTRING_MIDDLE':
                out.append((tok.start[0], tok.string, bool(raw_stack and raw_stack[-1])))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        pass
    return out


def escape_warnings(before, after):
    """Warnings about \\uXXXX / \\xNN escapes in string literals on the lines the edit changed that Python decodes to a
    printable ASCII character: the model usually meant the escape TEXT (one backslash level got lost in the JSON
    arguments), e.g. .replace('<', '\\u003c') replaces '<' with '<' and does nothing."""
    _, new = _changed_lines(before, after)
    if not new:
        return []
    warnings, seen = [], set()
    for line, s, raw in _string_tokens(after):
        if raw or line not in new:
            continue
        for m in ESC.finditer(s):
            if len(m.group(1)) % 2 == 0:
                continue  # an escaped backslash: the text \\u003c, as intended
            seq = m.group(2)
            ch = chr(int(seq[1:], 16))
            if not (0x20 <= ord(ch) < 0x7f) or (line, seq) in seen:
                continue
            seen.add((line, seq))
            warnings.append(f'WARNING: line {line}: the string literal contains \\{seq}, which Python reads as the '
                            f'character {ch!r} itself (not the {len(seq) + 1} characters \\{seq}). If the output '
                            f'must contain the text \\{seq}, the file needs two backslashes there: \\\\{seq}.')
    try:
        tree = ast.parse(after)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'replace' \
                    and len(node.args) >= 2 and all(isinstance(a, ast.Constant) and isinstance(a.value, str)
                                                    for a in node.args[:2]) \
                    and node.args[0].value == node.args[1].value and node.lineno in new:
                warnings.append(f'WARNING: line {node.lineno}: .replace({node.args[0].value!r}, '
                                f'{node.args[1].value!r}) replaces a string with itself and changes nothing.')
    except (SyntaxError, ValueError):
        pass
    return (warnings + doubled_backslashes(before, after))[:4]


def _max_run(line):
    return max((len(m) for m in re.findall(r'\\+', line)), default=0)


def doubled_backslashes(before, after):
    """v16 (note 11 L2, rich_3063 r5): a changed line whose longest backslash run is at least twice that of the line it
    replaced. Tool results are JSON-encoded, so the model sees every file backslash doubled and sometimes writes the
    doubled form back (`(\\\\*)` became `(\\\\\\\\*)`)."""
    a, b = before.split('\n'), after.split('\n')
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != 'replace':
            continue
        for j in range(j1, j2):
            run = _max_run(b[j])
            if run < 2:
                continue
            best = max(a[i1:i2], key=lambda o: difflib.SequenceMatcher(None, o.strip(), b[j].strip()).ratio())
            old = _max_run(best)
            if old and run >= 2 * old and difflib.SequenceMatcher(None, best.strip(), b[j].strip()).ratio() >= 0.6:
                out.append(f'WARNING: line {j + 1} has {run} backslashes in a row where the old line had {old}. Results '
                           'are JSON-encoded, so every backslash of the file is shown doubled; if you copied them from '
                           'a result, halve them.')
    return out[:2]


LOOPS = (ast.For, ast.AsyncFor, ast.While)
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
EXITS = {'exit', '_exit', 'quit'}


def _leaves(body):
    """Whether a loop body can end the loop: break (not inside a nested loop), return, raise, yield, await, or an
    exit() call (not inside a nested def)."""
    stack = [(n, False) for n in body]
    while stack:
        node, nested = stack.pop()
        if isinstance(node, SCOPES):
            continue
        if isinstance(node, (ast.Return, ast.Raise, ast.Yield, ast.YieldFrom, ast.Await)):
            return True
        if isinstance(node, ast.Break) and not nested:
            return True
        if isinstance(node, ast.Call):
            f = node.func
            if (f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)) in EXITS:
                return True
        inner = nested or isinstance(node, LOOPS)
        stack.extend((ch, inner) for ch in ast.iter_child_nodes(node))
    return False


def infinite_loops(text, lines=None):
    """[line] of `while True:` / `while 1:` loops whose header is on one of LINES (all when None) and whose body
    cannot end the loop (_leaves)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) \
                and isinstance(node.test.value, (bool, int)) and node.test.value \
                and (lines is None or node.lineno in lines) and not _leaves(node.body):
            out.append(node.lineno)
    return sorted(out)


TERMINALS = (ast.Return, ast.Raise, ast.Continue, ast.Break)
TERMINAL_WORD = {ast.Return: 'return', ast.Raise: 'raise', ast.Continue: 'continue', ast.Break: 'break'}


def _stmt_lists(tree):
    for node in ast.walk(tree):
        for field in ('body', 'orelse', 'finalbody'):
            lst = getattr(node, field, None)
            if isinstance(lst, list) and lst and isinstance(lst[0], ast.stmt):
                yield lst
        for h in getattr(node, 'handlers', []) or []:
            yield h.body
        for c in getattr(node, 'cases', []) or []:
            yield c.body


def unreachable(before, after):
    """[(first, last, terminal line, 'return'|'raise'|...)] (AFTER's 1-based lines) of statements that follow a
    return/raise/continue/break in the same block, where the dead statements or the terminal one are on lines the edit
    changed (dead code that was already there is not this edit's). [] when AFTER does not parse."""
    try:
        tree = ast.parse(after)
    except (SyntaxError, ValueError):
        return []
    changed = _changed_lines(before, after)[1]
    out = []
    for lst in _stmt_lists(tree):
        for i, st in enumerate(lst[:-1]):
            if not isinstance(st, TERMINALS):
                continue
            dead = lst[i + 1:]
            first, last = dead[0].lineno, max(d.end_lineno or d.lineno for d in dead)
            if changed & set(range(first, last + 1)) or changed & set(range(st.lineno, (st.end_lineno or st.lineno) + 1)):
                out.append((first, last, st.lineno, TERMINAL_WORD[type(st)]))
            break
    return sorted(out)


REF_SECONDS = 0.5      # cap on the package scan for uses of a removed name
REF_MAX = 3            # uses listed per removed name
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', 'site-packages', '.tox', '.venv', 'venv', 'build', 'dist',
             'docs', 'docs_src', '.mypy_cache', '.pytest_cache'}
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
BLOCKS = (ast.If, ast.Try, ast.With) + ((ast.TryStar,) if hasattr(ast, 'TryStar') else ())


def _bodies(node):
    """The statement lists of a module-level block (if/try/with) whose definitions still count as module-level."""
    out = [node.body, getattr(node, 'orelse', []), getattr(node, 'finalbody', [])]
    out += [h.body for h in getattr(node, 'handlers', [])]
    return out


def _kind(node):
    if isinstance(node, ast.ClassDef):
        return 'class'
    for d in node.decorator_list:
        name = d.attr if isinstance(d, ast.Attribute) else getattr(d, 'id', '')
        if name in ('property', 'cached_property'):
            return 'property'
    return 'def'


def definitions(text):
    """{qualname: dict(name, qualname, kind, start, end, cls)} of module- and class-level def/class nodes (start
    includes decorators; the first definition of a name wins, e.g. a property before its setter), None when TEXT
    does not parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    out = {}

    def visit(body, prefix, cls):
        for node in body:
            if isinstance(node, DEFS):
                q = prefix + node.name
                if q not in out:
                    out[q] = dict(name=node.name, qualname=q, kind=_kind(node), cls=cls, end=node.end_lineno,
                                  start=min([d.lineno for d in node.decorator_list] + [node.lineno]))
                if isinstance(node, ast.ClassDef):
                    visit(node.body, q + '.', node.name)
            elif isinstance(node, BLOCKS):
                for b in _bodies(node):
                    visit(b, prefix, cls)
    visit(tree.body, '', None)
    return out


def bound_names(text):
    """Names bound at module or class level (defs, classes, assignments, imports; inside if/try/with too), None when
    TEXT does not parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    out = set()

    def targets(t):
        if isinstance(t, ast.Name):
            out.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                targets(e)

    def visit(body):
        for node in body:
            if isinstance(node, DEFS):
                out.add(node.name)
                if isinstance(node, ast.ClassDef):
                    visit(node.body)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    targets(t)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets(node.target)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    out.add(a.asname or a.name.split('.')[0])
            elif isinstance(node, BLOCKS):
                for b in _bodies(node):
                    visit(b)
    visit(tree.body)
    return out


def _file_refs(text, name, member=False):
    """[(line, text)] where TEXT uses NAME: a name token (not a keyword argument or assignment target `name=`; an
    attribute `.name` only when MEMBER, i.e. NAME was a class member) or a string literal equal to it (__all__,
    getattr)."""
    lines, out = text.split('\n'), []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return []
    for i, tok in enumerate(toks):
        hit = False
        if tok.type == tokenize.NAME and tok.string == name:
            nxt = next((t for t in toks[i + 1:] if t.type not in (tokenize.NL, tokenize.COMMENT)), None)
            prev = toks[i - 1].string if i else ''
            hit = not (nxt is not None and nxt.type == tokenize.OP and nxt.string == '=') \
                and prev not in ('def', 'class') and (member or prev != '.')
        elif tok.type == tokenize.STRING and tok.string.strip('\'"') == name and re.fullmatch(r'[\'"]\w+[\'"]', tok.string):
            hit = True
        if hit and (not out or out[-1][0] != tok.start[0]):
            out.append((tok.start[0], lines[tok.start[0] - 1].strip()))
    return out


def _scan_dirs(rel):
    """Directories (relative) worth scanning for uses of a name defined in REL: its top package, then tests."""
    parts = rel.split('/')
    pkg = '/'.join(parts[:2]) if parts[0] == 'src' and len(parts) > 2 else (parts[0] if len(parts) > 1 else '')
    dirs = [pkg] if pkg else []
    dirs += [d for d in ('tests', 'test', 'testing') if os.path.isdir(os.path.join(root(), d))]
    return dirs or ['']


def _repo_refs(rel, d, deadline):
    """[(path, line, text)] of uses of D's name in other .py files of the package and the tests: a top-level name
    imported (`import ... name` lines, or a bare `name,` line of a multi-line import) or as `module.name`; a class
    member as `.name` in a file that mentions the class. Stops at DEADLINE (time.monotonic())."""
    name = d['name']
    stem = rel.rsplit('/', 1)[-1][:-3] if rel.endswith('.py') else rel
    if stem == '__init__':
        stem = rel.rsplit('/', 2)[-2] if '/' in rel else stem
    n = re.escape(name)
    if d['cls']:
        pat = re.compile(r'\.' + n + r'\b')
        need = re.compile(r'\b' + re.escape(d['cls']) + r'\b')
    else:
        pat = re.compile(r'(?:\bimport\b.*\b' + n + r'\b|\b' + re.escape(stem) + r'\.' + n + r'\b|^\s*' + n
                         + r'\s*,?\s*(?:#.*)?$)')
        need = None
    out, ws = [], root()
    for top in _scan_dirs(rel):
        for dirpath, dirnames, filenames in os.walk(os.path.join(ws, top)):
            dirnames[:] = sorted(x for x in dirnames if x not in SKIP_DIRS and not x.startswith('.'))
            for fn in sorted(filenames):
                if time.monotonic() > deadline:
                    return out
                if not fn.endswith('.py'):
                    continue
                path = os.path.join(dirpath, fn)
                other = os.path.relpath(path, ws).replace(os.sep, '/')
                if other == rel:
                    continue
                try:
                    text = open(path, encoding='utf-8', errors='replace').read()
                except OSError:
                    continue
                if name not in text or (need and not need.search(text)):
                    continue
                for i, line in enumerate(text.split('\n'), 1):
                    if name in line and not line.lstrip().startswith('#') and pat.search(line):
                        out.append((other, i, line.strip()))
                        if len(out) >= REF_MAX:
                            return out
        if not top:
            break
    return out


def _line_map(before, after):
    """{after line: before line} for the lines the edit kept (1-based)."""
    a, b = before.split('\n'), after.split('\n')
    out = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == 'equal':
            out.update({j + 1: i1 + (j - j1) + 1 for j in range(j1, j2)})
    return out


def lost_definitions(before, after, rel, seconds=REF_SECONDS, number_before=False):
    """[dict(name, qualname, kind, start, end, cls, refs=[(path, line, text)])] of module/class-level def/class/property
    names in BEFORE that AFTER no longer binds at module or class level (dunders skipped; a removed class hides its
    members), with up to REF_MAX places that still use each name: AFTER itself (at most REF_MAX - 1), then the
    package and tests (_repo_refs, all names together capped at SECONDS). [] when either text does not parse. start/end
    are BEFORE's line numbers; uses in REL are AFTER's, or with NUMBER_BEFORE BEFORE's (a use on a line the edit wrote
    reads 'NEW')."""
    old, now = definitions(before), bound_names(after)
    if not old or now is None:
        return []
    lost = [d for d in old.values() if d['name'] not in now and not (d['name'].startswith('__')
                                                                    and d['name'].endswith('__'))]
    lost = [d for d in lost if not any(d['qualname'].startswith(o['qualname'] + '.') for o in lost)]
    deadline = time.monotonic() + seconds
    lmap = _line_map(before, after) if number_before and lost else None
    for d in lost:
        refs = [(rel, n, t) for n, t in _file_refs(after, d['name'], bool(d['cls']))][:REF_MAX - 1]
        if lmap is not None:
            refs = [(p, lmap.get(n, 'NEW'), t) for p, n, t in refs]
        d['refs'] = refs + _repo_refs(rel, d, deadline)[:REF_MAX - len(refs)]
    return sorted(lost, key=lambda d: (not d['refs'], d['start']))


def describe(d):
    """`def computed_scope` (Dependant.computed_scope, a @property, lines 120-128)."""
    kind = 'class' if d['kind'] == 'class' else 'def'
    what = ', a @property' if d['kind'] == 'property' else ''
    q = f'{d["qualname"]}, ' if d['qualname'] != d['name'] else ''
    span = f'line {d["start"]}' if d['start'] == d['end'] else f'lines {d["start"]}-{d["end"]}'
    return f'`{kind} {d["name"]}` ({q}{span}{what})'


def uses(refs):
    """'a.py:12 (x = f()), b.py:3' (the first use with its text; a use in the edit's own text: 'NEW (...)')."""
    return ', '.join((f'{p}:{n}' if n != 'NEW' else 'NEW') + (f' ({t[:80]})' if i == 0 else '')
                     for i, (p, n, t) in enumerate(refs))
