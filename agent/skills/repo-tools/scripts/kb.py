"""Code knowledge base for the repository: one sqlite file, built lazily, reused by every later script call.

usage: kb.py [--rebuild] [--stats]      (debugging aid; the other scripts import this module)

The KB lives in $REPO_TOOLS_DIR/kb.sqlite (default /tmp/repo_tools). It is keyed by the repo root and
`git rev-parse HEAD`; on each open, changed/new/removed .py files are re-indexed (stat-based), and the whole KB is
rebuilt when HEAD moved or many files changed. Schema v1:
  files(id, path UNIQUE, lang, lines, is_aux, is_test, module, mtime, size, ntok, indexed)
  symbols(id, path, qualname, name, kind[class|def|async def], start, end, signature, doc, parent, ntok)
  refs(src_symbol, path, name, line, call)     names/attributes loaded inside each symbol (NULL = module level)
  assigns(src_symbol, path, name, line, self)  names stored at module/class level (self=0; src_symbol = the class or
                                               NULL), and self.X / cls.X stores inside methods (self=1; src_symbol =
                                               the method)
  imports(path, module, name, line)            module resolved for relative imports
  tests(test_symbol, test_path, target_name)   source names referenced by each test function
  fts(path, qualname, body, doc, strings)      FTS5, one row per symbol (rowid = symbols.id), pre-tokenized
  fts_files(path, body, strings)               FTS5, one row per indexed .py file (rowid = files.id)
  commits(sha, subject, "when", nfiles), commit_files(sha, path), cochange(a, b, n), fts_commits(subject)
API: open(root) -> conn, find_symbol(name), find_defs(name), all_names(), callers(name), tests_for(path_or_symbol),
cochanged(path), similar_commits(text), search_text(query, k), commits_for(path), bm25(conn, 'files'|'defs', tokens).
Every API function takes an optional conn=...; without it the KB of $WORKSPACE (default /workspace) is used.
Stdlib only.
"""

import ast
import builtins
import collections
import hashlib
import io
import math
import os
import pickle
import re
import sqlite3
import subprocess
import sys
import time

sys.dont_write_bytecode = True

SCHEMA = '1'
try:  # a changed kb.py (new parsing rules) invalidates existing KBs
    with io.open(os.path.abspath(__file__), 'rb') as _fh:
        CODE_VERSION = hashlib.sha1(_fh.read()).hexdigest()[:10]
except (NameError, OSError):
    CODE_VERSION = SCHEMA
ROOT = (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))
MAX_BYTES = 400_000          # larger .py files are listed but not indexed (generated code, vendored data)
MAX_PY_FILES = 6000          # safety cap for huge repos (tests/docs are the first to be left unindexed)
MAX_COMMITS = 2000           # history depth read from git log
MAX_COCHANGE_FILES = 20      # commits touching more .py files than this are sweeping changes: no co-change pairs
MAX_STRINGS = 3000           # chars of string literals kept per symbol
PARALLEL_MIN_FILES = 300     # fork a second parser process above this many .py files
REINDEX_MAX = 60             # more changed files than this (or 20%) -> full rebuild instead of incremental
TOKENIZER = "unicode61 tokenchars '_'"

STOP = set("""a an the and or not no is are was were be been being to of in on at by for with from as it its
this that these those if then else when while do does did done can could should would will shall may might must
i we you he she they me my our your their us them there here what which who whom how why where all any some
each every more most other such only own same so than too very just also but into out up down over under again
further once about above below between both few nor off per via using use used get set add fix fixes fixed
support allow make new now issue bug error errors problem when work works working expected actual behavior
self cls def return none true false import class pass raise try except lambda yield async await print
http https www com github org py python""".split())

IDENT_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
CAMEL = re.compile(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+')
AUX_PARTS = ('/tests/', '/test/', '/test_', '_test.py', '/docs/', '/docs_src/', '/examples/', '/example/',
             '/benchmarks/', '/scripts/', 'conftest.py', '/setup.py', '/noxfile', '/tools/')
TEST_DIRS = ('/tests/', '/test/', '/testing/')
LANGS = {'py': 'python', 'pyi': 'python-stub', 'md': 'markdown', 'rst': 'rst', 'txt': 'text', 'toml': 'toml',
         'cfg': 'ini', 'ini': 'ini', 'yml': 'yaml', 'yaml': 'yaml', 'json': 'json', 'js': 'javascript',
         'ts': 'typescript', 'html': 'html', 'css': 'css', 'c': 'c', 'h': 'c', 'cpp': 'cpp', 'rs': 'rust',
         'go': 'go', 'sh': 'shell', 'pyx': 'cython'}
BUILTIN_NAMES = set(dir(builtins)) | {'self', 'cls'}
DEFS = {ast.FunctionDef: 'def', ast.AsyncFunctionDef: 'async def', ast.ClassDef: 'class'}
BLOCK_FIELDS = ('body', 'orelse', 'finalbody', 'handlers', 'cases')  # statement lists that may hold defs

SCHEMA_SQL = f"""
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, lang TEXT, lines INTEGER, is_aux INTEGER,
                   is_test INTEGER, module TEXT, mtime INTEGER, size INTEGER, ntok INTEGER, indexed INTEGER);
CREATE TABLE symbols(id INTEGER PRIMARY KEY, path TEXT, qualname TEXT, name TEXT, kind TEXT, start INTEGER,
                     "end" INTEGER, signature TEXT, doc TEXT, parent INTEGER, ntok INTEGER);
CREATE TABLE refs(src_symbol INTEGER, path TEXT, name TEXT, line INTEGER, call INTEGER);
CREATE TABLE assigns(src_symbol INTEGER, path TEXT, name TEXT, line INTEGER, self INTEGER);
CREATE TABLE imports(path TEXT, module TEXT, name TEXT, line INTEGER);
CREATE TABLE tests(test_symbol INTEGER, test_path TEXT, target_name TEXT);
CREATE TABLE commits(id INTEGER PRIMARY KEY, sha TEXT, subject TEXT, "when" INTEGER, nfiles INTEGER);
CREATE TABLE commit_files(sha TEXT, path TEXT);
CREATE TABLE cochange(a TEXT, b TEXT, n INTEGER);
CREATE VIRTUAL TABLE fts USING fts5(path, qualname, body, doc, strings, tokenize="{TOKENIZER}");
CREATE VIRTUAL TABLE fts_files USING fts5(path, body, strings, tokenize="{TOKENIZER}");
CREATE VIRTUAL TABLE fts_commits USING fts5(subject, tokenize="{TOKENIZER}");
CREATE VIRTUAL TABLE fts_v USING fts5vocab(fts, 'instance');
CREATE VIRTUAL TABLE fts_files_v USING fts5vocab(fts_files, 'instance');
"""
INDEX_SQL = """
CREATE INDEX symbols_name ON symbols(name);
CREATE INDEX symbols_path ON symbols(path);
CREATE INDEX symbols_qualname ON symbols(qualname);
CREATE INDEX refs_name ON refs(name);
CREATE INDEX refs_src ON refs(src_symbol);
CREATE INDEX refs_path ON refs(path);
CREATE INDEX assigns_name ON assigns(name);
CREATE INDEX assigns_path ON assigns(path);
CREATE INDEX imports_module ON imports(module);
CREATE INDEX imports_path ON imports(path);
CREATE INDEX tests_target ON tests(target_name);
CREATE INDEX commit_files_path ON commit_files(path);
CREATE INDEX commit_files_sha ON commit_files(sha);
CREATE INDEX commits_sha ON commits(sha);
CREATE INDEX cochange_a ON cochange(a);
"""


# ---------------------------------------------------------------- tokenizer (shared with locate.py)

_TOK = {}


def subtoks(ident):
    out = []
    for part in ident.split('_'):
        if part:
            out += [p.lower() for p in CAMEL.findall(part)]
    return out


def ident_tokens(ident):
    """Search tokens of one identifier: the full identifier (if compound) plus its snake/camel parts,
    stopwords dropped, light plural stemming. Memoized: identifiers repeat a lot."""
    r = _TOK.get(ident)
    if r is None:
        r = []
        parts = subtoks(ident)
        low = ident.lower()
        if len(parts) > 1 and low not in STOP:
            r.append(low)
        for p in parts:
            if len(p) > 1 and p not in STOP:
                r.append(p[:-1] if len(p) > 4 and p.endswith('s') else p)
        _TOK[ident] = r
    return r


def tokenize(text):
    toks = []
    get = _TOK.get
    for m in IDENT_RE.findall(text):
        r = get(m)
        toks += ident_tokens(m) if r is None else r
    return toks


_LINE_TOK = {}


def _line_tokens(line):
    """tokenize() of one source line, memoized (blank lines, `return x`, `)` ... repeat a lot). Read-only result."""
    r = _LINE_TOK.get(line)
    if r is None:
        r = tokenize(line)
        if len(line) < 120:
            if len(_LINE_TOK) > 300_000:
                _LINE_TOK.clear()
            _LINE_TOK[line] = r
    return r


# ---------------------------------------------------------------- paths

def state_dir():
    """$REPO_TOOLS_DIR, else <workspace>/.git/repo_tools (v19b: per task by construction - the harness rebuilds
    /workspace for every task, and .git is never part of the patch), else /tmp/repo_tools. A shared /tmp let the
    working memory of one task leak into the next under the subprocess sandbox (L4 probe: call 52/~20 at a task's
    first call, every read refused as 'context nearly full')."""
    d = os.environ.get('REPO_TOOLS_DIR')
    if d:
        return d
    git = os.path.join((os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd()))), '.git')
    return os.path.join(git, 'repo_tools') if os.path.isdir(git) else '/tmp/repo_tools'


def db_path():
    return os.path.join(state_dir(), 'kb.sqlite')


def rel_path(p, root=None):
    """Normalize a path the model passed: strip /workspace/ (or the root) and leading ./"""
    p = p.strip().strip('`"\'')
    for pre in filter(None, (root, ROOT, '/workspace')):
        pre = pre.rstrip('/') + '/'
        if p.startswith(pre):
            p = p[len(pre):]
    while p.startswith('./'):
        p = p[2:]
    return p.lstrip('/')


def is_aux(path):
    p = '/' + path.lower()
    return any(s in p for s in AUX_PARTS)


def is_test(path):
    p = '/' + path.lower()
    base = p.rsplit('/', 1)[-1]
    return base.startswith('test_') or base.endswith('_test.py') or base == 'conftest.py' or any(
        d in p for d in TEST_DIRS)


def module_name(rel):
    if not rel.endswith('.py'):
        return None
    m = rel[:-3].replace('/', '.')
    if m == '__init__':
        return ''
    if m.endswith('.__init__'):
        m = m[:-9]
    for pre in ('src.', 'lib.'):
        if m.startswith(pre):
            return m[len(pre):]
    return m


def lang_of(path):
    base = path.rsplit('/', 1)[-1]
    return LANGS.get(base.rsplit('.', 1)[-1].lower(), '') if '.' in base else ''


GIT = ['git', '-c', 'safe.directory=*']  # the workspace may be owned by another uid


def git(root, *args, timeout=20):
    try:
        p = subprocess.run(GIT + ['-C', root, *args], capture_output=True, text=True, timeout=timeout,
                           errors='replace')
        return p.stdout if p.returncode == 0 else ''
    except (OSError, subprocess.SubprocessError):
        return ''


def git_head(root):
    return git(root, 'rev-parse', 'HEAD').strip()


def list_files(root):
    """Tracked + untracked-not-ignored files (relative paths). Falls back to a directory walk."""
    out = git(root, 'ls-files', '-co', '--exclude-standard', '-z')
    if out:
        return sorted(set(p for p in out.split('\0') if p))
    res = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if not x.startswith('.') and x not in (
            'node_modules', 'venv', 'build', 'dist', '__pycache__', 'site-packages'))
        res += [os.path.relpath(os.path.join(d, f), root) for f in files]
    return sorted(res)


# ---------------------------------------------------------------- parsing one file

def _signature(lines, start):
    sig = lines[start - 1].strip() if start - 1 < len(lines) else ''
    i = start
    while not sig.endswith(':') and i < len(lines) and i < start + 6:
        sig += ' ' + lines[i].strip()
        i += 1
    return re.sub(r'\s+', ' ', sig)[:200]


def _relative_module(rel, level, mod):
    base = (module_name(rel) or '').split('.') if rel else []
    if not rel.endswith('__init__.py'):
        base = base[:-1]
    if level > 1:
        base = base[:len(base) - (level - 1)] if level - 1 <= len(base) else []
    return '.'.join([p for p in base if p] + ([mod] if mod else []))


def _is_main_guard(node):
    t = getattr(node, 'test', None)
    return (isinstance(node, ast.If) and isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
            and t.left.id == '__name__')


def parse_file(root, rel):
    """Everything the KB stores about one .py file, as plain data (picklable, built in a worker process)."""
    full = os.path.join(root, rel)
    try:
        st = os.stat(full)
        with io.open(full, encoding='utf-8', errors='replace') as fh:
            src = fh.read()
    except (OSError, ValueError):
        return None
    rec = {'path': rel, 'mtime': st.st_mtime_ns, 'size': st.st_size, 'lines': src.count('\n') + (
        1 if src and not src.endswith('\n') else 0), 'syms': [], 'refs': [], 'assigns': [], 'imports': [], 'body': '',
        'ptoks': '', 'strings': '', 'ntok': 0, 'indexed': 0}
    if len(src) > MAX_BYTES:
        return rec
    lines = src.split('\n')  # the same line numbering as ast (splitlines() also splits on \f, \x1c, ...)
    if lines and lines[-1] == '':
        lines.pop()
    line_toks = [_line_tokens(line) for line in lines]  # identifiers never span lines
    toks = [t for lt in line_toks for t in lt]
    ptoks = tokenize(rel.replace('/', ' '))
    rec.update(body=' '.join(toks), ptoks=' '.join(ptoks), ntok=len(toks) + 3 * len(ptoks), indexed=1)
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return rec
    n = len(lines)
    syms, docids = [], set()

    def collect(nodes, prefix, parent):
        for node in nodes:
            kind = DEFS.get(type(node))
            if kind:
                qn = prefix + node.name
                end = min(getattr(node, 'end_lineno', None) or node.lineno, n)
                lid = len(syms)
                doc = ''
                body0 = node.body[0] if node.body else None
                if isinstance(body0, ast.Expr) and isinstance(getattr(body0, 'value', None), ast.Constant) \
                        and isinstance(body0.value.value, str):
                    doc = body0.value.value
                    docids.add(id(body0.value))
                syms.append([lid, parent, qn, node.name, kind, node.lineno, end, doc])
                collect(node.body, qn + '.', lid)
            elif hasattr(node, 'body') and not _is_main_guard(node):  # if/try/with/for/match blocks
                for field in BLOCK_FIELDS:                             # (demo code under __main__: skipped)
                    sub = getattr(node, field, None)
                    if sub:
                        collect(sub, prefix, parent)

    collect(tree.body, '', -1)
    mod_doc = ast.get_docstring(tree, clean=False)
    if mod_doc and tree.body:
        docids.add(id(tree.body[0].value))
    owner = [-1] * (n + 2)
    for s in syms:  # pre-order: inner symbols overwrite their parents
        lid, start, end = s[0], s[5], s[6]
        if start <= end:
            owner[start:end + 1] = [lid] * (end + 1 - start)

    refs, seen, callee = [], set(), set()
    stores, seen_store = [], set()  # (owner, name, line, is_self): see the assigns table
    strs = collections.defaultdict(list)
    imports = []
    Name, Attribute, Constant, Call, Load = ast.Name, ast.Attribute, ast.Constant, ast.Call, ast.Load
    AST = ast.AST
    stack = [tree]
    pop, push = stack.pop, stack.append
    while stack:  # iterative walk (ast.walk is ~2x slower); parents before children, ctx nodes skipped
        node = pop()
        for f in node._fields:
            if f == 'ctx':
                continue
            v = getattr(node, f, None)
            if type(v) is list:
                for x in v:
                    if isinstance(x, AST):
                        push(x)
            elif isinstance(v, AST):
                push(v)
        t = type(node)
        if t is Call:
            callee.add(id(node.func))
        elif t is Name:
            ctx = type(node.ctx)
            if ctx is Load and node.id not in BUILTIN_NAMES:
                ln = node.lineno
                key = (owner[ln] if ln <= n else -1, node.id, ln)
                if key not in seen:
                    seen.add(key)
                    refs.append(key + (int(id(node) in callee),))
            elif ctx is ast.Store:
                ln = node.lineno
                own = owner[ln] if ln <= n else -1
                if (own == -1 or syms[own][4] == 'class') and (own, node.id) not in seen_store:
                    seen_store.add((own, node.id))
                    stores.append((own, node.id, ln, 0))
        elif t is Attribute:
            ctx = type(node.ctx)
            if ctx is Load:
                ln = node.lineno
                key = (owner[ln] if ln <= n else -1, node.attr, ln)
                if key not in seen:
                    seen.add(key)
                    refs.append(key + (int(id(node) in callee),))
            elif ctx is ast.Store and type(node.value) is Name and node.value.id in ('self', 'cls'):
                ln = node.lineno
                own = owner[ln] if ln <= n else -1
                if own != -1 and (own, node.attr) not in seen_store:
                    seen_store.add((own, node.attr))
                    stores.append((own, node.attr, ln, 1))
        elif t is Constant:
            v = node.value
            if type(v) is str and len(v) >= 4 and id(node) not in docids:
                strs[owner[node.lineno] if node.lineno <= n else -1].append(v)
        elif t is ast.Import:
            for a in node.names:
                imports.append((a.name, '', node.lineno))
        elif t is ast.ImportFrom:
            mod = _relative_module(rel, node.level, node.module) if node.level else (node.module or '')
            for a in node.names:
                imports.append((mod, a.name, node.lineno))

    # per-symbol token documents
    child_ranges = collections.defaultdict(list)
    for s in syms:
        if s[1] >= 0:
            child_ranges[s[1]].append((s[5], s[6]))
    out_syms = []
    for lid, parent, qn, name, kind, start, end, doc in syms:
        if kind == 'class':  # class-level lines only; methods have their own rows
            keep = [True] * (end - start + 1)
            for a, b in child_ranges.get(lid, ()):
                for i in range(max(a, start), min(b, end) + 1):
                    keep[i - start] = False
            btoks = [t for i, k in enumerate(keep) if k for t in line_toks[start - 1 + i]]
        else:
            btoks = [t for lt in line_toks[start - 1:end] for t in lt]
        ntoks = subtoks(name)
        sstr = ' '.join(tokenize(' | '.join(strs.get(lid, ()))[:MAX_STRINGS]))
        out_syms.append((lid, parent, qn, name, kind, start, end, _signature(lines, start),
                         doc.strip().split('\n')[0][:200], ' '.join(ntoks), ' '.join(btoks),
                         ' '.join(tokenize(doc[:500])), sstr, len(btoks) + 3 * len(ntoks)))
    allstr = []
    for v in strs.values():
        allstr += v
    rec.update(syms=out_syms, refs=refs, imports=imports, assigns=stores,
               strings=' '.join(tokenize(' | '.join(allstr)[:20 * MAX_STRINGS])))
    return rec


def _stub(root, rel):
    """A listed-but-not-indexed .py file (beyond MAX_PY_FILES)."""
    try:
        st = os.stat(os.path.join(root, rel))
    except OSError:
        return None
    return {'path': rel, 'mtime': st.st_mtime_ns, 'size': st.st_size, 'lines': None, 'syms': [], 'refs': [],
            'assigns': [], 'imports': [], 'body': '', 'ptoks': '', 'strings': '', 'ntok': 0, 'indexed': 0}


def _parse_list(root, rels):
    return [r for r in (parse_file(root, rel) for rel in rels) if r is not None]


def parse_many(root, rels):
    """Parse files; with 2+ CPUs and many files, a forked child parses half of them."""
    if len(rels) < PARALLEL_MIN_FILES or (os.cpu_count() or 1) < 2 or not hasattr(os, 'fork'):
        return _parse_list(root, rels)
    sizes = {}
    for r in rels:
        try:
            sizes[r] = os.path.getsize(os.path.join(root, r))
        except OSError:
            sizes[r] = 0
    order = sorted(rels, key=lambda r: -sizes[r])
    mine, theirs = order[0::2], order[1::2]
    out_file = os.path.join(state_dir(), f'parse.{os.getpid()}.pkl')
    try:
        pid = os.fork()
    except OSError:
        return _parse_list(root, rels)
    if pid == 0:  # child
        code = 1
        try:
            data = _parse_list(root, theirs)
            with io.open(out_file + '.part', 'wb') as fh:
                pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(out_file + '.part', out_file)
            code = 0
        finally:
            os._exit(code)
    res = _parse_list(root, mine)
    _, status = os.waitpid(pid, 0)
    other = None
    if status == 0:
        try:
            with io.open(out_file, 'rb') as fh:
                other = pickle.load(fh)
        except Exception:
            other = None
    for f in (out_file, out_file + '.part'):
        try:
            os.remove(f)
        except OSError:
            pass
    if other is None:
        other = _parse_list(root, theirs)
    return res + other


# ---------------------------------------------------------------- writing

def _insert(conn, recs, others, next_file, next_sym):
    frows, ftsf, srows, fts, rrows, irows, arows = [], [], [], [], [], [], []
    for rec in recs:
        rel = rec['path']
        fid = next_file
        next_file += 1
        frows.append((fid, rel, 'python', rec['lines'], int(is_aux(rel)), int(is_test(rel)), module_name(rel),
                      rec['mtime'], rec['size'], rec['ntok'], rec['indexed']))
        if rec['indexed']:
            ftsf.append((fid, rec['ptoks'], rec['body'], rec['strings']))
        idmap = {}
        for (lid, parent, qn, name, kind, start, end, sig, doc, ntoks, btoks, dtoks, stoks, ntok) in rec['syms']:
            gid = next_sym
            next_sym += 1
            idmap[lid] = gid
            srows.append((gid, rel, qn, name, kind, start, end, sig, doc, idmap.get(parent), ntok))
            fts.append((gid, rec['ptoks'], ntoks, btoks, dtoks, stoks))
        for lid, name, line, call in rec['refs']:
            rrows.append((idmap.get(lid), rel, name, line, call))
        for mod, name, line in rec['imports']:
            irows.append((rel, mod, name, line))
        for lid, name, line, is_self in rec.get('assigns', ()):
            arows.append((idmap.get(lid), rel, name, line, is_self))
    for rel in others:
        frows.append((next_file, rel, lang_of(rel), None, int(is_aux(rel)), int(is_test(rel)), None, None, None,
                      0, 0))
        next_file += 1
    conn.executemany('INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?)', frows)
    conn.executemany('INSERT INTO fts_files(rowid, path, body, strings) VALUES (?,?,?,?)', ftsf)
    conn.executemany('INSERT INTO symbols VALUES (?,?,?,?,?,?,?,?,?,?,?)', srows)
    conn.executemany('INSERT INTO fts(rowid, path, qualname, body, doc, strings) VALUES (?,?,?,?,?,?)', fts)
    conn.executemany('INSERT INTO refs VALUES (?,?,?,?,?)', rrows)
    conn.executemany('INSERT INTO imports VALUES (?,?,?,?)', irows)
    conn.executemany('INSERT INTO assigns VALUES (?,?,?,?,?)', arows)


def _compute_tests(conn):
    conn.execute('DELETE FROM tests')
    conn.execute("""
        INSERT INTO tests(test_symbol, test_path, target_name)
        SELECT DISTINCT s.id, s.path, r.name
        FROM symbols s JOIN files f ON f.path = s.path AND f.is_test = 1
        JOIN refs r ON r.src_symbol = s.id
        WHERE s.kind != 'class' AND s.name LIKE 'test%'
          AND r.name IN (SELECT DISTINCT x.name FROM symbols x JOIN files g ON g.path = x.path AND g.is_test = 0)
    """)


def _git_log_cmd(root):
    return GIT + ['-C', root, 'log', '-n', str(MAX_COMMITS), '--no-renames', '--name-only',
                  '--format=%x00%H%x1f%at%x1f%s']


def _ingest_git(conn, raw):
    commits, cfiles, pairs = [], [], collections.Counter()
    for i, chunk in enumerate(raw.split('\0')[1:]):
        head, _, rest = chunk.partition('\n')
        parts = head.split('\x1f', 2)
        if len(parts) != 3:
            continue
        sha, when, subject = parts
        files = [f for f in rest.split('\n') if f]
        commits.append((i + 1, sha, subject, int(when) if when.isdigit() else 0, len(files)))
        cfiles += [(sha, f) for f in files]
        py = sorted(f for f in files if f.endswith('.py'))
        if 1 < len(py) <= MAX_COCHANGE_FILES:
            for j, a in enumerate(py):
                for b in py[j + 1:]:
                    pairs[(a, b)] += 1
    conn.executemany('INSERT INTO commits VALUES (?,?,?,?,?)', commits)
    conn.executemany('INSERT INTO fts_commits(rowid, subject) VALUES (?,?)',
                     [(c[0], ' '.join(tokenize(c[2]))) for c in commits])
    conn.executemany('INSERT INTO commit_files VALUES (?,?)', cfiles)
    conn.executemany('INSERT INTO cochange VALUES (?,?,?)',
                     [(a, b, k) for (a, b), k in pairs.items()] + [(b, a, k) for (a, b), k in pairs.items()])


def build(root, path=None):
    """Full build into a temp file, then atomically replace the KB. Returns the open connection."""
    t0 = time.time()
    root = os.path.abspath(root)
    path = path or db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp{os.getpid()}'
    if os.path.exists(tmp):
        os.remove(tmp)
    log_file = tmp + '.log'
    proc = None
    head = git_head(root)
    if head:
        try:
            with io.open(log_file, 'wb') as fh:
                proc = subprocess.Popen(_git_log_cmd(root), stdout=fh, stderr=subprocess.DEVNULL)
        except OSError:
            proc = None
    listing = list_files(root)
    pys = [p for p in listing if p.endswith('.py')]
    if len(pys) > MAX_PY_FILES:  # huge repo: index source files first, list the rest without indexing
        pys.sort(key=lambda p: (is_test(p), is_aux(p)))
        stubs = (_stub(root, p) for p in pys[MAX_PY_FILES:])
        recs = parse_many(root, pys[:MAX_PY_FILES]) + [r for r in stubs if r]
    else:
        recs = parse_many(root, pys)
    got = {r['path'] for r in recs}
    others = [p for p in listing if not p.endswith('.py') and p not in got]
    conn = sqlite3.connect(tmp)
    conn.execute('PRAGMA journal_mode=OFF')
    conn.execute('PRAGMA synchronous=OFF')
    conn.executescript(SCHEMA_SQL)
    _insert(conn, recs, others, 1, 1)
    raw = ''
    if proc is not None:
        try:
            proc.wait(timeout=30)
            with io.open(log_file, encoding='utf-8', errors='replace') as fh:
                raw = fh.read()
        except (subprocess.SubprocessError, OSError):
            proc.kill()
    try:
        os.remove(log_file)
    except OSError:
        pass
    _ingest_git(conn, raw)
    conn.executescript(INDEX_SQL)
    _compute_tests(conn)
    meta = {'schema': SCHEMA, 'code': CODE_VERSION, 'root': root, 'head': head,
            'built_at': str(int(time.time())), 'build_seconds': f'{time.time() - t0:.2f}', 'py_files': str(len(recs))}
    conn.executemany('INSERT INTO meta VALUES (?,?)', list(meta.items()))
    conn.commit()
    conn.close()
    os.replace(tmp, path)
    return _connect(path)


def _connect(path):
    conn = sqlite3.connect(path, timeout=10)
    conn.execute('PRAGMA synchronous=OFF')
    return conn


def _meta(conn):
    try:
        return dict(conn.execute('SELECT key, value FROM meta'))
    except sqlite3.DatabaseError:
        return {}


def refresh(conn, root):
    """Re-index .py files whose mtime/size changed, new ones and removed ones. Returns the count, or None when a
    full rebuild is needed instead."""
    listing = list_files(root)
    known = {p: (m, s) for p, m, s in conn.execute("SELECT path, mtime, size FROM files WHERE lang = 'python'")}
    other_known = {p for (p,) in conn.execute("SELECT path FROM files WHERE lang != 'python' OR lang IS NULL")}
    changed, current = [], set()
    for rel in listing:
        if not rel.endswith('.py'):
            continue
        try:
            st = os.stat(os.path.join(root, rel))
        except OSError:
            continue
        current.add(rel)
        if known.get(rel) != (st.st_mtime_ns, st.st_size):
            changed.append(rel)
    removed = [p for p in known if p not in current]
    new_other = [p for p in listing if not p.endswith('.py') and p not in other_known]
    n = len(changed) + len(removed)
    if n > max(REINDEX_MAX, len(known) // 5):
        return None
    if not n and not new_other:
        return 0
    stale = [p for p in changed if p in known] + removed
    for rel in stale:
        ids = [i for (i,) in conn.execute('SELECT id FROM symbols WHERE path = ?', (rel,))]
        conn.executemany('DELETE FROM fts WHERE rowid = ?', [(i,) for i in ids])
        fid = conn.execute('SELECT id FROM files WHERE path = ?', (rel,)).fetchone()
        if fid:
            conn.execute('DELETE FROM fts_files WHERE rowid = ?', fid)
        for table in ('symbols', 'refs', 'imports', 'assigns'):
            conn.execute(f'DELETE FROM {table} WHERE path = ?', (rel,))
        conn.execute('DELETE FROM files WHERE path = ?', (rel,))
    recs = _parse_list(root, changed)
    nf = conn.execute('SELECT coalesce(max(id), 0) + 1 FROM files').fetchone()[0]
    ns = conn.execute('SELECT coalesce(max(id), 0) + 1 FROM symbols').fetchone()[0]
    _insert(conn, recs, new_other, nf, ns)
    _compute_tests(conn)
    conn.commit()
    return n


_OPEN = {}


def open(root=None, rebuild=False):  # noqa: A001 - the documented API name
    """Connection to the up-to-date KB for root (default $WORKSPACE). Builds or refreshes as needed."""
    root = os.path.abspath(root or ROOT)
    path = db_path()
    key = (root, path)
    if key in _OPEN and not rebuild:
        return _OPEN[key]
    conn = None
    if os.path.exists(path) and not rebuild:
        try:
            conn = _connect(path)
            m = _meta(conn)
            stale = (m.get('schema'), m.get('code'), m.get('root'), m.get('head')) != (
                SCHEMA, CODE_VERSION, root, git_head(root))
            if stale or refresh(conn, root) is None:
                conn.close()
                conn = None
        except sqlite3.DatabaseError:
            conn = None
    if conn is None:
        conn = build(root, path)
    _OPEN[key] = conn
    return conn


def _c(conn):
    return conn if conn is not None else open()


# ---------------------------------------------------------------- queries

def _clean_name(name):
    name = name.strip().strip('`"\'').replace('()', '')
    if '(' in name:
        name = name.split('(')[0]
    if ':' in name:  # path:Symbol
        name = name.rsplit(':', 1)[1]
    return name.strip()


def find_symbol(name, conn=None):
    """Definitions of name ('Class.method', 'method', 'module.func'): [(path, qualname, kind, start, end, signature)],
    source files first."""
    c = _c(conn)
    name = _clean_name(name)
    q = ('SELECT s.path, s.qualname, s.kind, s.start, s."end", s.signature FROM symbols s JOIN files f ON f.path = '
         's.path WHERE {} ORDER BY f.is_aux, f.is_test, length(s.qualname), s.path, s.start')
    rows = []
    if '.' in name:
        rows = c.execute(q.format("s.qualname = ?1 OR s.qualname LIKE '%.' || ?1"), (name,)).fetchall()
        if not rows:  # dotted module path: pkg.mod.func or pkg.mod.Class.method
            parts = name.split('.')
            for i in range(1, len(parts)):
                mod, qn = '.'.join(parts[:i]), '.'.join(parts[i:])
                rows = c.execute(q.format('s.qualname = ? AND f.module LIKE ?'), (qn, '%' + mod)).fetchall()
                if rows:
                    break
        if not rows:
            name = name.split('.')[-1]
    if not rows:
        rows = c.execute(q.format('s.name = ?'), (name,)).fetchall()
    return rows


def find_defs(name, conn=None, k=20):
    """Every place name is defined: defs/classes (symbols) and assignments (assigns), source files first:
    [(path, line, kind, qualname, owner)] where kind is 'class', 'def', 'async def', 'attribute' (self.X / cls.X
    stored in a method; qualname Class.X, owner the method's qualname), 'class attribute' (Class.X = ... in a class
    body) or 'assignment' (module level); owner is '' except for attributes."""
    c = _c(conn)
    name = _clean_name(name).split('.')[-1]
    out = [(p, s, kind, qn, '') for p, s, kind, qn in c.execute(
        'SELECT s.path, s.start, s.kind, s.qualname FROM symbols s JOIN files f ON f.path = s.path WHERE s.name = ? '
        'ORDER BY f.is_aux, f.is_test, s.path, s.start LIMIT ?', (name, k))]
    try:
        rows = c.execute("""SELECT a.path, a.line, a.self, m.qualname, m.kind, cl.qualname FROM assigns a
                            LEFT JOIN symbols m ON m.id = a.src_symbol LEFT JOIN symbols cl ON cl.id = m.parent
                            JOIN files f ON f.path = a.path WHERE a.name = ?
                            ORDER BY f.is_aux, f.is_test, a.path, a.line LIMIT ?""", (name, k)).fetchall()
    except sqlite3.DatabaseError:  # a KB built by an older kb.py has no assigns table
        rows = []
    for path, line, is_self, mq, mkind, cq in rows:
        if is_self:
            out.append((path, line, 'attribute', f'{cq}.{name}' if cq else name, mq or ''))
        elif mq:
            out.append((path, line, 'class attribute', f'{mq}.{name}', ''))
        else:
            out.append((path, line, 'assignment', name, ''))
    rank = {p: (is_aux(p), is_test(p)) for p in {o[0] for o in out}}
    return sorted(out, key=lambda o: (rank[o[0]], o[0], o[1]))[:k]


def all_names(conn=None):
    """Every defined name in the repository (defs, classes, assigned names and attributes)."""
    c = _c(conn)
    names = {n for (n,) in c.execute('SELECT DISTINCT name FROM symbols')}
    try:
        names |= {n for (n,) in c.execute('SELECT DISTINCT name FROM assigns')}
    except sqlite3.DatabaseError:
        pass
    return names


def callers(name, conn=None):
    """Load sites of name (last component of a dotted name): [(path, enclosing qualname or '', line, is_call)],
    source files first."""
    c = _c(conn)
    last = _clean_name(name).split('.')[-1]
    return c.execute("""
        SELECT r.path, coalesce(s.qualname, ''), r.line, r.call FROM refs r
        LEFT JOIN symbols s ON s.id = r.src_symbol JOIN files f ON f.path = r.path
        WHERE r.name = ? ORDER BY f.is_aux, f.is_test, r.path, r.line""", (last,)).fetchall()


def is_file_target(target):
    t = rel_path(target)
    return t.endswith('.py') or '/' in t


def tests_for(target, conn=None, k=12):
    """Tests most likely to exercise a source file or symbol: [(test_path, test_name or None)], best first.
    Test functions referencing the names defined in the file (rare names count more), test files importing its
    module, and test files named after it."""
    c = _c(conn)
    t = rel_path(target)
    score = collections.Counter()
    file_bonus = collections.Counter()
    if is_file_target(t):
        names = [n for (n,) in c.execute('SELECT DISTINCT name FROM symbols WHERE path = ?', (t,))
                 if not n.startswith('__')]
        mod = module_name(t) or ''
        stem = t.rsplit('/', 1)[-1][:-3] if t.endswith('.py') else ''
        if mod:
            pkg, _, last = mod.rpartition('.')
            for (tp,) in c.execute("""SELECT DISTINCT i.path FROM imports i JOIN files f ON f.path = i.path
                                      WHERE f.is_test = 1 AND (i.module = ? OR i.module LIKE ? OR
                                      (i.module = ? AND i.name = ?))""", (mod, mod + '.%', pkg, last)):
                file_bonus[tp] += 1.0
        if stem and stem != '__init__':
            for (tp,) in c.execute("SELECT path FROM files WHERE is_test = 1 AND (path LIKE ? OR path LIKE ?)",
                                   ('%/test_' + stem + '.py', 'test_' + stem + '.py')):
                file_bonus[tp] += 1.5
    else:
        names = [_clean_name(target).split('.')[-1]]
    if names:
        ph = ','.join('?' * len(names))
        ndef = dict(c.execute(f"""SELECT s.name, count(DISTINCT s.path) FROM symbols s JOIN files f ON
                                 f.path = s.path AND f.is_test = 0 WHERE s.name IN ({ph}) GROUP BY s.name""", names))
        ntest = dict(c.execute(f'SELECT target_name, count(*) FROM tests WHERE target_name IN ({ph}) '
                               'GROUP BY target_name', names))
        for tp, tn, target_name in c.execute(f"""SELECT t.test_path, s.name, t.target_name FROM tests t
                                                JOIN symbols s ON s.id = t.test_symbol
                                                WHERE t.target_name IN ({ph})""", names):
            score[(tp, tn)] += 1.0 / max(1, ndef.get(target_name, 1)) / math.log2(1 + ntest.get(target_name, 1))
        if len(names) == 1:
            low = names[0].lower()
            for tp, tn in c.execute("""SELECT s.path, s.name FROM symbols s JOIN files f ON f.path = s.path
                                      AND f.is_test = 1 WHERE s.name LIKE 'test%' AND s.kind != 'class'
                                      AND lower(s.name) LIKE ?""", ('%' + low + '%',)):
                score[(tp, tn)] += 1.0
    for (tp, tn) in list(score):
        score[(tp, tn)] += 0.3 * file_bonus.get(tp, 0)
    ranked = [key for key, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]
    have = {tp for tp, _ in ranked}
    extra = [(tp, None) for tp, _ in sorted(file_bonus.items(), key=lambda kv: (-kv[1], kv[0])) if tp not in have]
    return (ranked + extra)[:k]


def cochanged(path, conn=None, k=10):
    """Files that changed in the same commits as path: [(path, n)]."""
    return _c(conn).execute('SELECT b, n FROM cochange WHERE a = ? ORDER BY n DESC, b LIMIT ?',
                            (rel_path(path), k)).fetchall()


def commits_for(path, conn=None, k=10):
    """Most recent commits touching path: [(sha, subject, when)]."""
    return _c(conn).execute("""SELECT c.sha, c.subject, c."when" FROM commit_files cf JOIN commits c ON c.sha = cf.sha
                               WHERE cf.path = ? ORDER BY c."when" DESC, c.id LIMIT ?""",
                            (rel_path(path), k)).fetchall()


def _match_query(tokens):
    uniq = list(dict.fromkeys(t for t in tokens if t))
    return ' OR '.join(f'"{t}"' for t in uniq[:64])


def similar_commits(text, conn=None, k=8):
    """Past commits whose subject resembles text: [(sha, subject, [files])], best first."""
    c = _c(conn)
    q = _match_query(tokenize(text))
    if not q:
        return []
    rows = c.execute("""SELECT c.sha, c.subject FROM fts_commits JOIN commits c ON c.id = fts_commits.rowid
                        WHERE fts_commits MATCH ? ORDER BY bm25(fts_commits) LIMIT ?""", (q, k)).fetchall()
    return [(sha, subj, [p for (p,) in c.execute('SELECT path FROM commit_files WHERE sha = ?', (sha,))])
            for sha, subj in rows]


def search_text(query, k=10, conn=None):
    """Ranked symbol search (FTS5 bm25 over names, bodies, docstrings, string literals):
    [(path, qualname, start, end, score)], higher score = better."""
    c = _c(conn)
    q = _match_query(tokenize(query))
    if not q:
        return []
    return c.execute("""SELECT s.path, s.qualname, s.start, s."end", -bm25(fts, 0.5, 3.0, 1.0, 1.0, 1.0) AS sc
                        FROM fts JOIN symbols s ON s.id = fts.rowid WHERE fts MATCH ? ORDER BY sc DESC LIMIT ?""",
                     (q, k)).fetchall()


def bm25(conn, which, qtokens, k1=1.2, b=0.75):
    """Lucene-style BM25 of every indexed .py file ('files': body + 3x path tokens) or every function
    ('defs': body + 3x name tokens) for the query tokens (repeated query tokens weigh 1 + log(count)).
    Returns ({doc id: score}, {doc id: ntok}) for docs with a positive score."""
    if which == 'files':
        vocab, boosted = 'fts_files_v', 'path'
        docs = dict(conn.execute('SELECT id, ntok FROM files WHERE indexed = 1'))
    else:
        vocab, boosted = 'fts_v', 'qualname'
        docs = dict(conn.execute("SELECT id, ntok FROM symbols WHERE kind != 'class'"))
    n = len(docs)
    scores = collections.defaultdict(float)
    if not n:
        return scores, docs
    avg = sum(docs.values()) / n or 1
    for t, w in collections.Counter(qtokens).items():
        tf = {}
        for doc, col, cnt in conn.execute(f'SELECT doc, col, count(*) FROM {vocab} WHERE term = ? '
                                          'GROUP BY doc, col', (t,)):
            if col == 'body':
                tf[doc] = tf.get(doc, 0) + cnt
            elif col == boosted:
                tf[doc] = tf.get(doc, 0) + 3 * cnt
        tf = {d: f for d, f in tf.items() if d in docs}
        if not tf:
            continue
        idf = math.log(1 + (n - len(tf) + 0.5) / (len(tf) + 0.5))
        qw = idf * (1 + math.log(w))
        for d, f in tf.items():
            scores[d] += qw * f * (k1 + 1) / (f + k1 * (1 - b + b * docs[d] / avg))
    return scores, docs


def main(argv):
    t0 = time.time()
    conn = open(rebuild='--rebuild' in argv)
    m = _meta(conn)
    print(f"kb: {db_path()} root={m.get('root')} head={m.get('head', '')[:10]} built in {m.get('build_seconds')}s"
          f' (opened in {time.time() - t0:.2f}s)')
    if '--stats' in argv:
        for t in ('files', 'symbols', 'refs', 'assigns', 'imports', 'tests', 'commits', 'commit_files', 'cochange'):
            print(f'{t}: {conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
