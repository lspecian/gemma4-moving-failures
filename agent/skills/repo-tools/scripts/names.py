"""Where names are defined: shared helpers of show.py and search.py (a module, not a script to call).

E025 (v12): before a read the model's thinking asks targeted questions: "where is is_interactive set?", "does
APIRouter have a mount method?", "is FastAPIError imported here?", "where is _SVG_FONT_FAMILY used?". These helpers
answer them inside the read that is already happening:

  names_used(rel, ranges)     the "Names used here" block (locate.py): the repository names the shown lines use but
                              do not define (self.attr, obj.attr, Class.method, functions, constants), each with
                              where it is defined or assigned and its signature or assignment line. Builtins, the
                              standard library and third-party packages are skipped.
  names_line(rel, ranges)     v18: show.py's one-line form, at most 3 names that relate to the task (issue
                              identifiers, words of the task, self.attr), never one already printed or read.
  member(rel, cls, attr)      Class.attr that is not a def: an attribute assigned in a method (self.X = ...), a member
                              inherited from a base class in the repository, or "has no" with the class's members and
                              base classes.
  enclosing(lines, n)         the qualname of the def/class a line is in (search.py hits), from indentation alone.
  similar(word)               defined names close to word (the KB), for searches that found nothing.

Resolution is static and cheap: one ast parse of the shown file, the files its imports and base classes lead to (at
most MAX_FILES of at most MAX_PARSE_BYTES together, and at most MAX_USES names per call: count caps, not a clock, so
the same call always prints the same names), and the KB ($REPO_TOOLS_DIR/kb.sqlite, only when locate.py already built it)
for attributes of objects whose class is unknown. Stdlib only.
"""

import ast
import builtins
import difflib
import os
import re
import sys

sys.dont_write_bytecode = True

ROOT = (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))
MAX_NAMES = 8           # lines in the "Names used here" block (v17 L1: was 12)
LINE_CHARS = 150        # chars per line of the block
TEXT_CHARS = 90         # chars of the signature / assignment shown per name
LINE_NAMES = 3          # v18: names in show.py's one-line names block
LINE_TEXT_CHARS = 60    # v18: chars of the signature / assignment per name in that line
MAX_FILES = 8           # other files parsed per call (imports, base classes)
MAX_BYTES = 600_000     # larger files are not parsed
MAX_PARSE_BYTES = 1_500_000  # v19: total size of the other files parsed per call
MAX_USES = 200          # v19: distinct names resolved per call, in source order (max on the 112 local gold functions:
                        # 216; ~2-20 ms with the KB open). v18's 0.2 s clock made the names line depend on machine load
KB_MAX_DEFS = 2         # obj.attr with an unknown obj: resolved through the KB only when defined at most this often
MEMBERS_MAX = 60        # members listed by member() for a class that lacks the requested one
BUILTINS = set(dir(builtins)) | {'self', 'cls', '__class__', '__name__', '__file__', '__doc__'}
FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
COMMENT = re.compile(r'\s*#[^"\']*$')
DEF_RE = re.compile(r'^(\s*)(?:async\s+def|def|class)\s+(\w+)')
AUX = re.compile(r'(^|/)(tests?|testing|docs|docs_src|examples?|benchmarks)/|(^|/)test_[^/]*\.py$|_test\.py$|'
                 r'(^|/)conftest\.py$')

_MODS = {}               # rel -> Module or None (per process)
_STATE = {'files': 0, 'bytes': 0, 'kb': None, 'kb_tried': False}


# ---------------------------------------------------------------- files and modules

def _root():
    return os.environ.get('WORKSPACE', ROOT)


def _isfile(rel):
    return bool(rel) and os.path.isfile(os.path.join(_root(), rel))


def module_file(mod):
    """The repository file of a dotted module name ('rich.console' -> 'rich/console.py'), or None."""
    if not mod or mod.startswith('.'):
        return None
    base = mod.replace('.', '/')
    for pre in ('', 'src/', 'lib/'):
        for cand in (pre + base + '.py', pre + base + '/__init__.py'):
            if _isfile(cand):
                return cand
    return None


def _relative(rel, level, mod):
    """The dotted (path-based) module of `from <level dots><mod> import ...` inside file rel."""
    parts = rel.split('/')[:-1]
    if level > 1:
        parts = parts[:len(parts) - (level - 1)] if level - 1 <= len(parts) else []
    return '.'.join(parts + ([mod] if mod else []))


class Module:
    """One parsed repository file: its lines, tree, module-level names, imports and classes."""

    def __init__(self, rel, src=None, tree=None):
        self.rel = rel
        if src is None:
            with open(os.path.join(_root(), rel), encoding='utf-8', errors='replace') as fh:
                src = fh.read()
        self.lines = src.split('\n')
        self.tree = tree if tree is not None else ast.parse(src)
        self.top = {}        # name -> statement defining it at module level
        self.imports = {}    # local name -> (dotted module, imported attribute or None, line)
        self.classes = {}    # qualname -> ClassDef
        self._members = {}
        self._locals = {}
        self._scan(self.tree.body, top=True)
        stack = [(n, n.name) for n in self.tree.body if isinstance(n, ast.ClassDef)]
        while stack:
            node, qn = stack.pop()
            self.classes[qn] = node
            stack += [(c, f'{qn}.{c.name}') for c in node.body if isinstance(c, ast.ClassDef)]

    def _scan(self, body, top):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.top.setdefault(node.name, node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in tgts:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            self.top.setdefault(n.id, node)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        self.imports.setdefault(a.asname, (a.name, None, node.lineno))
                    else:
                        head = a.name.split('.')[0]
                        self.imports.setdefault(head, (head, None, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                mod = _relative(self.rel, node.level, node.module) if node.level else (node.module or '')
                for a in node.names:
                    if a.name != '*':
                        self.imports.setdefault(a.asname or a.name, (mod, a.name, node.lineno))
            elif isinstance(node, (ast.If, ast.Try, ast.With)) or type(node).__name__ == 'TryStar':
                for field in ('body', 'orelse', 'finalbody', 'handlers'):
                    for sub in getattr(node, field, None) or ():
                        self._scan(sub.body if isinstance(sub, ast.ExceptHandler) else [sub], top)

    def text(self, node, owner=None):
        """A definition's signature line(s), or an assignment's first line, trimmed."""
        i = node.lineno
        s = self.lines[i - 1].strip() if i - 1 < len(self.lines) else ''
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            s = COMMENT.sub('', s)
            j = i
            while not s.rstrip().endswith(':') and j < min(len(self.lines), i + 5):
                s += ' ' + COMMENT.sub('', self.lines[j].strip())
                j += 1
        s = ' '.join(s.split())
        if len(s) > TEXT_CHARS:
            s = s[:TEXT_CHARS - 1] + '…'
        return s + (f'  [in {owner}]' if owner else '')

    def members(self, cls):
        """{name: (node, kind, owner)} of a ClassDef: methods ('def'/'property'), class-level assignments
        ('class attribute') and self.X stores in its methods ('attribute', owner = the method's qualname; the store in
        __init__ wins, else the first)."""
        key = id(cls)
        if key in self._members:
            return self._members[key]
        out, attrs = {}, {}
        qn = next((q for q, c in self.classes.items() if c is cls), cls.name)
        for node in cls.body:
            if isinstance(node, FUNCS):
                deco = {getattr(d, 'id', getattr(d, 'attr', '')) for d in node.decorator_list}
                kind = 'property' if deco & {'property', 'cached_property'} else 'def'
                if node.name not in out or kind == 'property':
                    out.setdefault(node.name, (node, kind, ''))
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Store) \
                            and isinstance(sub.value, ast.Name) and sub.value.id in ('self', 'cls'):
                        stmt = sub
                        prev = attrs.get(sub.attr)
                        if prev is None or (node.name == '__init__' and prev[2] != f'{qn}.__init__'):
                            attrs[sub.attr] = (stmt, 'attribute', f'{qn}.{node.name}')
            elif isinstance(node, ast.ClassDef):
                out.setdefault(node.name, (node, 'class', ''))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in tgts:
                    if isinstance(t, ast.Name):
                        out.setdefault(t.id, (node, 'class attribute', ''))
        for name, v in attrs.items():
            out.setdefault(name, v)
        self._members[key] = out
        return out

    def qualname(self, cls):
        return next((q for q, c in self.classes.items() if c is cls), cls.name)

    def locals_of(self, fn):
        """Names local to a function (parameters, stores, imports, nested defs): never resolved globally."""
        key = id(fn)
        if key in self._locals:
            return self._locals[key]
        out = set()
        a = fn.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
            out.add(arg.arg)
        glob = set()
        for n in ast.walk(fn):
            if n is fn:
                continue
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                out.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                out |= {(x.asname or x.name).split('.')[0] for x in n.names}
            elif isinstance(n, ast.ExceptHandler) and n.name:
                out.add(n.name)
            elif isinstance(n, ast.arg):
                out.add(n.arg)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                glob |= set(n.names)
        out -= glob
        self._locals[key] = out
        return out


def load(rel, src=None, tree=None):
    """The Module of a repository file (cached per process); None when unreadable, unparsable or over budget."""
    if rel in _MODS:
        return _MODS[rel]
    mod = None
    try:
        full = os.path.join(_root(), rel)
        size = os.path.getsize(full) if src is None else 0
        if src is not None or (size <= MAX_BYTES and _budget_ok(size)):
            mod = Module(rel, src, tree)
    except (OSError, SyntaxError, ValueError, RecursionError, MemoryError):
        mod = None
    _MODS[rel] = mod
    return mod


def _budget_ok(size):
    """Room to parse one more file of `size` bytes in this call (MAX_FILES, MAX_PARSE_BYTES); counts it."""
    if _STATE['files'] >= MAX_FILES or _STATE.get('bytes', 0) + size > MAX_PARSE_BYTES:
        return False
    _STATE['files'] += 1
    _STATE['bytes'] = _STATE.get('bytes', 0) + size
    return True


def _kb():
    """The KB connection when locate.py already built it (never built here: too slow for a read), else None."""
    if not _STATE['kb_tried']:
        _STATE['kb_tried'] = True
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import kb
            if os.path.exists(kb.db_path()):
                _STATE['kb'] = kb.open(_root())
        except Exception:
            _STATE['kb'] = None
    return _STATE['kb']


# ---------------------------------------------------------------- definitions

class Def:
    """Where a name is defined: file, line, text (signature or assignment line), kind; node/module for classes and
    modules (to resolve their members)."""

    def __init__(self, rel, line, text, kind, node=None, module=None, qualname=''):
        self.rel, self.line, self.text, self.kind = rel, line, text, kind
        self.node, self.module, self.qualname = node, module, qualname


def _def_of(mod, name, node, kind=None, owner=None, qualname=None):
    if isinstance(node, ast.ClassDef):
        kind = 'class'
    elif isinstance(node, FUNCS):
        kind = kind or 'def'
    text = mod.text(node, owner)
    if kind == 'property':
        text = '@property ' + text
    return Def(mod.rel, node.lineno, text, kind or 'assignment', node, mod, qualname or name)


def resolve_name(mod, name, depth=0):
    """A module-level name of mod: Def (in the repository), ('external', 'module.name') or None."""
    if mod is None or depth > 4:
        return None
    node = mod.top.get(name)
    if node is not None:
        return _def_of(mod, name, node)
    imp = mod.imports.get(name)
    if not imp:
        return None
    m, attr, line = imp
    if attr is None:
        f = module_file(m)
        if f:
            return Def(f, 1, f'module {m}', 'module', None, None, m)
        return ('external', m)
    f = module_file(m)
    if f:
        target = load(f)
        d = resolve_name(target, attr, depth + 1) if target else None
        if isinstance(d, Def) or (isinstance(d, tuple) and d):
            return d
    sub = module_file(f'{m}.{attr}')
    if sub:
        return Def(sub, 1, f'module {m}.{attr}', 'module', None, None, f'{m}.{attr}')
    if f:
        return None
    return ('external', f'{m}.{attr}')


def resolve_expr(mod, expr, depth=0):
    """A Name / dotted Attribute expression (a base class, `routing.Router`) -> Def, ('external', dotted) or None."""
    if isinstance(expr, ast.Name):
        return resolve_name(mod, expr.id, depth)
    if isinstance(expr, ast.Attribute):
        base = resolve_expr(mod, expr.value, depth)
        if isinstance(base, tuple):
            return ('external', f'{base[1]}.{expr.attr}')
        if isinstance(base, Def):
            return attribute_of(base, expr.attr, depth + 1)
    return None


def attribute_of(d, attr, depth=0):
    """attr of a resolved Def: a module's top-level name, or a class's member (inherited ones included)."""
    if d is None or depth > 6:
        return None
    if d.kind == 'module':
        m = load(d.rel)
        r = resolve_name(m, attr, depth + 1) if m else None
        if r is None:
            sub = module_file(f'{d.qualname}.{attr}')
            if sub:
                return Def(sub, 1, f'module {d.qualname}.{attr}', 'module', None, None, f'{d.qualname}.{attr}')
        return r
    if d.kind == 'class' and d.module is not None:
        r = find_member(d.module, d.node, attr, depth + 1)
        return r[0] if r else None
    return None


def bases(mod, cls, depth=0):
    """[(base text, Def | ('external', dotted) | None)] of a ClassDef."""
    out = []
    for b in cls.bases:
        try:
            text = ast.unparse(b)
        except Exception:
            text = '?'
        out.append((text, resolve_expr(mod, b, depth)))
    return out


def find_member(mod, cls, attr, depth=0):
    """(Def, defining class qualname) of attr on a class or its repository base classes, or None."""
    if depth > 6 or mod is None:
        return None
    mem = mod.members(cls).get(attr)
    cq = mod.qualname(cls)
    if mem:
        node, kind, owner = mem
        return _def_of(mod, attr, node, kind, owner, f'{cq}.{attr}'), cq
    for text, b in bases(mod, cls, depth):
        if isinstance(b, Def) and b.kind == 'class' and b.module is not None:
            r = find_member(b.module, b.node, attr, depth + 1)
            if r:
                return r
    return None


# ---------------------------------------------------------------- the "Names used here" block

def _chain(node):
    """['self', 'console', 'is_interactive'] for self.console.is_interactive; None when the root is not a Name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id] + parts[::-1]
    return None


def _overlaps(node, a, b):
    s = getattr(node, 'lineno', None)
    if s is None:
        return True
    e = getattr(node, 'end_lineno', None) or s
    return s <= b and a <= e


def _in(line, ranges):
    return any(a <= line <= b for a, b in ranges)


def _uses(mod, ranges):
    """[(display, kind, payload, scopes, line)] for every load in the shown ranges, in source order. kind: 'name'
    (payload: id), 'self' (payload: attr chain after self), 'chain' (payload: full chain), 'attr' (payload: attr of
    an object that is not a plain name chain)."""
    out = []
    lo, hi = min(a for a, _ in ranges), max(b for _, b in ranges)

    def visit(node, scopes):
        for field, value in ast.iter_fields(node):
            if field in ('annotation', 'returns', 'decorator_list', 'type_comment', 'type_params'):
                continue  # type hints and decorators: names the reader rarely asks about
            items = value if isinstance(value, list) else [value]
            for ch in items:
                if not isinstance(ch, ast.AST) or not _overlaps(ch, lo, hi):
                    continue
                line = getattr(ch, 'lineno', None)
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    visit(ch, scopes + [ch])
                    continue
                if isinstance(ch, ast.Attribute) and isinstance(ch.ctx, ast.Load):
                    c = _chain(ch)
                    if c and line and _in(line, ranges):
                        if c[0] in ('self', 'cls'):
                            out.append(('.'.join(c), 'self', c, scopes, line))
                            if len(c) > 2:  # self.live.stop(): self.live too
                                out.append(('.'.join(c[:2]), 'self', c[:2], scopes, line))
                        else:
                            out.append(('.'.join(c), 'chain', c, scopes, line))
                        continue
                    if c is None and line and _in(line, ranges):
                        out.append(('.' + ch.attr, 'attr', [ch.attr], scopes, line))
                    visit(ch, scopes)
                    continue
                if isinstance(ch, ast.Name) and isinstance(ch.ctx, ast.Load):
                    if line and _in(line, ranges) and ch.id not in BUILTINS:
                        out.append((ch.id, 'name', [ch.id], scopes, line))
                    continue
                visit(ch, scopes)

    visit(mod.tree, [])
    return out


def _is_local(mod, name, scopes):
    for s in scopes:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if isinstance(s, ast.Lambda):
                if name in {a.arg for a in s.args.args}:
                    return True
            elif name in mod.locals_of(s):
                return True
    return False


def _kb_defs(attr):
    """Definitions of attr anywhere in the repository (KB), source files first: [(rel, line, kind, qualname, owner)];
    [] when unknown, ambiguous (> KB_MAX_DEFS) or the KB is not built."""
    conn = _kb()
    if conn is None or len(attr) < 3 or attr.startswith('__'):
        return []
    try:
        import kb
        rows = [r for r in kb.find_defs(attr, conn) if not AUX.search(r[0])]
    except Exception:
        return []
    owners = {(r[0], r[3]) for r in rows}
    return rows if 0 < len(owners) <= KB_MAX_DEFS else []


def _kb_def(attr, obj=None):
    """Def of attr from the KB. obj (the name before .attr, e.g. console) picks among two owners (Console)."""
    rows = _kb_defs(attr)
    if not rows:
        return None
    if obj:
        o = obj.lower().strip('_')

        def score(r):
            owner_cls = r[3].split('.')[-2].lower() if '.' in r[3] else ''
            return (owner_cls != o, o not in owner_cls if owner_cls else True)
        rows = sorted(rows, key=score)
    rel, line, kind, qn, owner = rows[0]
    m = load(rel)
    text = ''
    if m is not None and 0 < line <= len(m.lines):
        text = ' '.join(COMMENT.sub('', m.lines[line - 1]).split())
        if kind in ('def', 'async def') and line > 1 and m.lines[line - 2].strip() in ('@property', '@cached_property'):
            text = '@property ' + text
        if len(text) > TEXT_CHARS:
            text = text[:TEXT_CHARS - 1] + '…'
        if owner:
            text += f'  [in {owner}]'
    return Def(rel, line, text, kind, qualname=qn)


def _resolve_use(mod, kind, chain, scopes, cls):
    """Def of one use, or None (local, builtin, external, unknown)."""
    if kind == 'name':
        if _is_local(mod, chain[0], scopes):
            return None
        d = resolve_name(mod, chain[0])
        return d if isinstance(d, Def) and d.kind != 'module' else None
    if kind == 'self':
        if cls is None or len(chain) < 2:
            return None
        if len(chain) == 2:
            r = find_member(mod, cls, chain[1])
            return r[0] if r else None
        r = find_member(mod, cls, chain[1])  # self.a.b: b of whatever a is (the KB when a's class is unknown)
        if r and r[0].kind == 'class':
            return attribute_of(r[0], chain[-1]) if len(chain) == 3 else None
        return _kb_def(chain[-1], chain[-2])
    if kind == 'chain':
        if _is_local(mod, chain[0], scopes):
            return _kb_def(chain[-1], chain[-2])
        d = resolve_name(mod, chain[0])
        if isinstance(d, tuple):
            return None  # an external package
        if d is None:
            return _kb_def(chain[-1], chain[-2])
        for attr in chain[1:]:
            d = attribute_of(d, attr)
            if d is None:
                return None
        return d if d.kind != 'module' else None
    if kind == 'attr':
        return _kb_def(chain[-1])
    return None


def _query_words(q=None):
    """Identifiers and words of the task: q, else the locate query in the flow state."""
    if q is None:
        try:
            import flow
            st = flow.peek()
            q = flow.get_meta(st.conn, 'query', '') if st else ''
        except Exception:
            q = ''
    toks = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', q or '')
    idents = {w.lower() for w in toks if '_' in w.strip('_') or re.search(r'[a-z][A-Z]', w)}  # not prose words
    words = set()
    for w in set(toks):
        words |= _words_of(w)
    return idents, words


def _words_of(name):
    return {p for p in re.split(r'_+', re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()) if len(p) >= 4}


def names_used(rel, ranges, src=None, tree=None, limit=MAX_NAMES, query=None):
    """The "Names used here" lines for the shown line ranges of rel ([] when nothing resolves). query: the task text
    that orders them (default: the locate query in the flow state). Never raises."""
    try:
        return _names_used(rel, ranges, src, tree, limit, query)
    except Exception:
        return []


def _found(rel, ranges, src, tree, query=None):
    """[(prio, -shared, order, display, Def)] of the repository names the shown ranges use but do not define, best
    first: prio 0 an issue identifier, 1 shares a word with the task, 2 self.attr of the class shown, 3 other."""
    if not rel.endswith('.py') or not ranges:
        return []
    _STATE['files'] = _STATE['bytes'] = 0
    mod = load(rel, src, tree)
    if mod is None:
        return []
    uses = _uses(mod, ranges)
    if not uses:
        return []
    shown_cls = None
    for c in mod.classes.values():
        if any(c.lineno <= a and b <= c.end_lineno for a, b in ranges):
            if shown_cls is None or c.lineno >= shown_cls.lineno:
                shown_cls = c
    idents, words = _query_words(query)
    seen, found = set(), []
    for order, (display, kind, chain, scopes, line) in enumerate(uses):
        if display in seen:
            continue
        if len(seen) >= MAX_USES:
            break
        seen.add(display)
        cls = next((s for s in reversed(scopes) if isinstance(s, ast.ClassDef)), None) or shown_cls
        d = _resolve_use(mod, kind, chain, scopes, cls)
        if not isinstance(d, Def) or d.line is None:
            continue
        if d.rel == rel and _in(d.line, ranges):
            continue  # defined in the lines shown
        last = chain[-1]
        shared = len(set().union(*(_words_of(p) for p in chain if p not in ('self', 'cls'))) & words)
        if last.lower() in idents:
            prio = 0
        elif shared:
            prio = 1
        elif kind == 'self':
            prio = 2
        else:
            prio = 3
        found.append((prio, -shared, order, display, d))
    found.sort(key=lambda f: f[:3])
    return found


def _names_used(rel, ranges, src, tree, limit, query=None):
    found = _found(rel, ranges, src, tree, query)
    if not found:
        return []
    out = ['Names used here, defined elsewhere:']
    width = min(34, max(len(f[3]) for f in found[:limit]))
    for prio, _, order, display, d in found[:limit]:
        line = f'  {display:<{width}}  {d.rel}:{d.line}  {d.text}'
        out.append(line if len(line) <= LINE_CHARS else line[:LINE_CHARS - 1] + '…')
    if len(found) > limit:
        out.append(f'  ... {len(found) - limit} more')
    return out


def _short_text(display, text):
    """A definition's text without the name the line already shows: 'def size(self)' -> '(self)',
    'self.x = 1  [in C.__init__]' -> '= 1 [in C.__init__]'."""
    text = ' '.join(text.split()).rstrip(':').rstrip()
    last = display.split('.')[-1]
    m = re.match(r'^(@property )?(async )?def ' + re.escape(last) + r'(?=\()', text)
    if m:
        return (m.group(1) or '') + (m.group(2) or '') + text[m.end():]
    for pre in (display, 'self.' + last, 'cls.' + last, last):
        if text.startswith(pre) and text[len(pre):len(pre) + 1] in (' ', ':', '='):
            return text[len(pre):].strip() if text[len(pre):len(pre) + 1] != ':' else text[len(pre):]
    return text


def names_line(rel, ranges, src=None, tree=None, limit=LINE_NAMES, skip=(), seen_ranges=(), query=None):
    """v18: show.py's one-line names block: ('names: a L12: def a(x) | ...', [displays]) of at most `limit` names the
    shown lines use (issue identifiers, names sharing a word with the task, self.attr of the class shown; never
    other names), skipping the displays in `skip` (printed earlier in the task) and names whose definition line lies
    in seen_ranges [(rel, start, end)] (already read). ('', []) when nothing qualifies. Never raises."""
    try:
        picked = []
        for prio, _, _, display, d in _found(rel, ranges, src, tree, query):
            if prio > 2 or display in skip:
                continue
            if any(p == d.rel and a is not None and b is not None and a <= d.line <= b for p, a, b in seen_ranges):
                continue
            picked.append((display, d))
            if len(picked) >= limit:
                break
        if not picked:
            return '', []
        parts = []
        for display, d in picked:
            where = f'L{d.line}' if d.rel == rel else f'{d.rel}:{d.line}'
            text = _short_text(display, d.text)
            if len(text) > LINE_TEXT_CHARS:
                text = text[:LINE_TEXT_CHARS - 1] + '…'
            parts.append(f'{display} {where}' + (text if text[:1] == ':' else ' ' + text if text[:1] in '=(' else
                                                 ': ' + text))
        return 'names: ' + ' | '.join(parts), [p[0] for p in picked]
    except Exception:
        return '', []


# ---------------------------------------------------------------- "Class has no attr", members, bases

def describe_bases(mod, cls):
    """'routing.Router (from starlette.routing, not in the repository)' / 'Base (pkg/base.py:12)' per base."""
    parts = []
    for text, b in bases(mod, cls):
        if isinstance(b, Def):
            parts.append(f'{text} ({b.rel}:{b.line})')
        elif isinstance(b, tuple):
            parts.append(f'{text} (from {b[1].rsplit(".", 1)[0] if "." in b[1] else b[1]}, not in the repository)')
        else:
            parts.append(text)
    return parts


def member_list(mod, cls, near=None):
    """'name L12, prop L30 (property), attr L41 (attribute), ...' of a class, closest to `near` first."""
    mem = mod.members(cls)
    items = sorted(mem.items(), key=lambda kv: kv[1][0].lineno)
    close = difflib.get_close_matches(near, list(mem), n=4, cutoff=0.7) if near else []
    if near:
        close += [n for n in mem if n not in close and _words_of(n) & _words_of(near)][:4]
    order = [kv for kv in items if kv[0] in close] + [kv for kv in items if kv[0] not in close]
    out = []
    for name, (node, kind, owner) in order[:MEMBERS_MAX]:
        tag = '' if kind in ('def',) else f' ({kind})'
        out.append(f'{name} L{node.lineno}{tag}')
    more = len(order) - MEMBERS_MAX
    return out, close, max(0, more)


def external_bases(mod, cls, depth=0):
    """Dotted names of base classes outside the repository, through repository base classes too."""
    out = []
    for text, b in bases(mod, cls, depth):
        if isinstance(b, tuple):
            out.append(b[1])
        elif isinstance(b, Def) and b.kind == 'class' and b.module is not None and depth < 4:
            out += external_bases(b.module, b.node, depth + 1)
    return out


def wrap(items, width=110, indent='  '):
    lines, cur = [], indent
    for it in items:
        piece = it if cur == indent else ', ' + it
        if len(cur) + len(piece) > width and cur != indent:
            lines.append(cur + ',')
            cur = indent + it
        else:
            cur += piece
    if cur.strip():
        lines.append(cur)
    return lines


# ---------------------------------------------------------------- search helpers

def enclosing(lines, n):
    """Qualname of the def/class line n (1-based) is in, from indentation ('' at module level). A def/class line
    itself belongs to its parent."""
    if not 0 < n <= len(lines):
        return ''
    cur = lines[n - 1]
    if not cur.strip():
        return ''
    indent = len(cur) - len(cur.lstrip())
    names = []
    for i in range(n - 2, -1, -1):
        if indent == 0:
            break
        m = DEF_RE.match(lines[i])
        if m and len(m.group(1)) < indent:
            names.append(m.group(2))
            indent = len(m.group(1))
    return '.'.join(reversed(names))


def similar(word, k=3):
    """Definitions named like word: exact (other kinds / case) first, then close names. [(rel, line, kind, qualname,
    owner)], at most k; [] without a KB."""
    conn = _kb()
    if conn is None or not word:
        return []
    try:
        import kb
        exact = [r for r in kb.find_defs(word, conn) if not AUX.search(r[0])]
        if exact:
            return exact[:k]
        names = kb.all_names(conn)
        low = word.lower()
        cands = [n for n in names if n.lower() == low]
        pool = [n for n in names if abs(len(n) - len(word)) <= 4]
        cands += [n for n in difflib.get_close_matches(word, pool, n=k, cutoff=0.75) if n not in cands]
        out = []
        for n in cands:
            rows = [r for r in kb.find_defs(n, conn) if not AUX.search(r[0])]
            out += rows[:2]
            if len(out) >= k:
                break
        return out[:k]
    except Exception:
        return []
