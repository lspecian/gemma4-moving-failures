"""Print one function or class, or a line range, with line numbers.

usage: show.py  target=billing/invoice.py  symbol=Invoice.total
       show.py  target=billing/invoice.py  start=140  end=200   (at most 100 lines)
       (positional form, still accepted: show.py FILE SYMBOL  |  show.py FILE START END)

Paths are relative to /workspace. "symbol" is a name or dotted qualname (Class.method); module-level and class-level
assignments (CONSTANT = ...) work too. Code lines print as `123  code` (the number, two spaces, the line: edit.py
strips this prefix when it is copied into OLD/NEW).
v18 (research/notes/13-tool-efficiency.md 4B, lean output under Kaggle's 14,336-token compaction):
- An explicit start/end range prints every line asked for (E039), except a docstring longer than DOC_KEEP lines that
  lies entirely inside it, folded to its first line plus `... (docstring lines a-b)`.
- Every other read (symbol, start alone, a short file whole) folds every docstring that way, and a signature longer
  than SIG_MAX lines to the def line plus one line of parameter names with a ready call that shows it whole.
- A symbol read of a function longer than LONG_FN lines prints its head (signature, docstring line) and a window of
  FN_WINDOW lines around the lines that match the issue's terms (the locate query), then `rest: show.py ...`.
- "start" without "end": the definition around the line when it is at most 2*LINE_HALF+1 lines, else its def line
  and LINE_HALF lines on each side of the line (from the line itself when it continues the previous read); outside
  any definition LINE_HALF lines on each side. The window always contains the line.
- A class longer than LONG_CLASS lines prints its methods with line ranges and its first lines.
- Under the code at most NAMES_MAX names on one `names:` line: names the shown lines use that are defined elsewhere
  and relate to the task (issue identifiers, words of the task, self.attr), never one printed earlier in the task
  (sqlite meta 'show:names') or defined in lines already read; none for files outside locate's candidates.
- Output is capped at CAP characters (EXPLICIT_CAP for an explicit range), with a ready call for the rest.
Calls whose intent is clear are run with one NOTE line saying how they were read: a symbol without a target (the
symbol is looked up in the repository), start/end without a target (the file viewed last, else locate's top
file), symbol together with start/end (the symbol), a target alone (its outline; a non-Python file: its first 100
lines), a target that does not exist but whose file name is unique in the repository.
symbol=Class.attr that is not a def answers the question behind it: an attribute assigned in a method
(self.attr = ...) is shown at its assignment; a member inherited from a base class in the repository is shown there;
otherwise "Class has no attr" with the class's members and base classes (and where those are defined). A
module-level name that the file imports, or that is defined in another file, is shown where it is defined.
Compact mode (the context is large: flow.context_tokens() >= flow.COMPACT_FROM_TOKENS): narrower windows, fewer
names; one line at the end says so (an explicit start/end range is never shortened).
"""

import ast
import contextlib
import difflib
import io
import json
import keyword
import os
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402  (flow engine: refusals + NEXT/BUDGET footers)
import argfix  # noqa: E402
import names  # noqa: E402  (where the names in the shown code are defined)
from argfix import Misuse  # noqa: E402

ROOT = (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))  # overridable for local testing
MAX_LINES = 150        # a symbol printed whole at most this long (longer: truncated with a ready call)
RANGE_LINES = 100
EXPLICIT_RANGE_LINES = RANGE_LINES  # v17b: an explicit start/end range keeps the full limit in compact mode
OUTLINE_LINES = 25
LONG_CLASS = 100       # v18: a class symbol read longer than this: its methods and first lines
LONG_HEAD = 20         # a long class: its first lines printed under its method list
METHODS_MAX = 40       # methods listed for a long class
LONG_FN = 60           # v18: a function symbol read longer than this: head + a window around the issue's terms
FN_WINDOW = 40         # ... the window's lines
LINE_HALF = 20         # v18: start without end: this many lines on each side of the line
SIG_MAX = 8            # v18: signatures longer than this are folded (symbol and start-only reads)
DOC_KEEP = 8           # v18: an explicit range keeps the docstrings inside it of at most this many lines
NAMES_MAX = 3          # v18: names on the one `names:` line
CAP = 3500             # v18: characters of output (the harness cuts results at 5,000; flow appends the footer)
EXPLICIT_CAP = 4200    # ... for an explicit start/end range (E039: shortening asked-for ranges cost re-reads)
NAMES_META = 'show:names'  # flow meta key: JSON {name: call} of the names printed on names lines in this task
# compact mode per level: FN_WINDOW, LINE_HALF, LONG_HEAD, METHODS_MAX, OUTLINE_LINES, NAMES_MAX
COMPACT = {1: (30, 15, 12, 20, 12, 2), 2: (24, 10, 6, 12, 8, 1)}
COMPACT_LINE = 'compact: context is large; ask for a narrower range if needed'
LEVEL = 0              # compact level of this call
CUT = []               # what compact mode left out in this call (non-empty: print COMPACT_LINE)
LISTING = {}           # the code listing being printed: rel, end (for the ready call after a cut at CAP)
USAGE = '\n'.join(__doc__.strip().splitlines()[2:5])
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
BLOCKS = (ast.If, ast.Try, ast.ExceptHandler, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While) \
    + ((ast.TryStar,) if hasattr(ast, 'TryStar') else ())  # statements whose bodies hold module/class-level defs
STOP = {'the', 'and', 'for', 'with', 'not', 'this', 'that', 'from', 'import', 'when', 'into', 'should', 'self',
        'def', 'class', 'return', 'none', 'true', 'false', 'are', 'was', 'were', 'has', 'have', 'but', 'also',
        'use', 'used', 'using', 'which', 'what', 'there', 'their', 'then', 'than', 'they', 'like', 'some', 'can',
        'will', 'would', 'could', 'does', 'only', 'other', 'more', 'fix', 'add', 'adds', 'issue', 'error', 'bug',
        'file', 'files', 'value', 'values', 'make', 'need', 'via', 'its', 'all', 'any', 'one', 'two', 'new', 'old'}


def compact_level():
    """v17 L1: 0 normally; 1 from flow.COMPACT_FROM_TOKENS of context; 2 from ~5k tokens more. 0 whenever flow does
    not provide the estimate (or it fails)."""
    try:
        fn, lo = getattr(flow, 'context_tokens', None), getattr(flow, 'COMPACT_FROM_TOKENS', None)
        if fn is None or not lo:
            return 0
        t = int(fn() or 0)
        hi = getattr(flow, 'COMPACT_STRONG_TOKENS', lo + 5000)
        return 2 if t >= hi else 1 if t >= lo else 0
    except Exception:
        return 0


def set_level(level):
    """Shrink the limits for compact level 1 or 2 (level 0 keeps the module constants as they are)."""
    global LEVEL, FN_WINDOW, LINE_HALF, LONG_HEAD, METHODS_MAX, OUTLINE_LINES, NAMES_MAX
    LEVEL = level
    del CUT[:]
    LISTING.clear()
    if level in COMPACT:
        FN_WINDOW, LINE_HALF, LONG_HEAD, METHODS_MAX, OUTLINE_LINES, NAMES_MAX = COMPACT[level]


def cut(what):
    """Compact mode left something out (the COMPACT_LINE is printed at the end)."""
    if LEVEL:
        CUT.append(what)


def numbered(i, text):
    """One code line: the number, two spaces, the line (edit.py's SHOW_PREFIX strips exactly this form); a blank
    line: the number alone."""
    return f'{i}  {text}' if text.strip() else f'{i}'


def code(lines, a, b, folds=()):
    """Print lines a..b (1-based, inclusive) with line numbers; a fold (first, last, text) starting inside a..b
    prints as its one text line instead of its lines."""
    i = a
    while i <= b:
        f = next((f for f in folds if f[0] == i), None)
        if f:
            print(f[2])
            i = max(i, f[1]) + 1
            continue
        print(numbered(i, lines[i - 1]))
        i += 1
    if LISTING:
        LISTING['last'] = b


def doc_range(node):
    """(first, last) lines of a module/def/class docstring, or None."""
    body = getattr(node, 'body', None)
    if not body:
        return None
    d = body[0]
    if isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant) and isinstance(d.value.value, str):
        return d.lineno, d.end_lineno or d.lineno
    return None


def doc_fold(lines, a, b):
    """The fold of the docstring at lines a..b: (first, last, text) of the lines after its first line of text, or
    None when that leaves fewer than 2 lines."""
    keep = a
    if lines[a - 1].strip().lstrip('rRbBuU') in ('"""', "'''") and a < b:
        keep = a + 1  # the opening quotes alone on their line: keep the summary line too
    if b - keep < 2:
        return None
    return keep + 1, b, f'... (docstring lines {keep + 1}-{b})'


def header_end(node, lines):
    """Last line of a def's signature (the line ending with ':' before its body)."""
    first = node.body[0]
    body = min([getattr(d, 'lineno', first.lineno) for d in getattr(first, 'decorator_list', [])] + [first.lineno])
    if body <= node.lineno:
        return node.lineno
    end = body - 1
    while end > node.lineno and (not lines[end - 1].strip() or lines[end - 1].strip().startswith('#')):
        end -= 1
    return end


def param_names(node):
    a = node.args
    out = [x.arg for x in a.posonlyargs] + (['/'] if a.posonlyargs else []) + [x.arg for x in a.args]
    if a.vararg:
        out.append('*' + a.vararg.arg)
    elif a.kwonlyargs:
        out.append('*')
    out += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        out.append('**' + a.kwarg.arg)
    return out


def sig_fold(rel, node, lines):
    """The fold of a signature longer than SIG_MAX lines: (def line + 1, its last line, one line of parameter names
    with a ready call that shows it whole), or None."""
    end = header_end(node, lines)
    if end - node.lineno + 1 <= SIG_MAX:
        return None
    full = argfix.call('show.py', rel, node.lineno, end)
    return (node.lineno + 1, end, f'... (signature lines {node.lineno + 1}-{end} folded; parameters: '
                                  f'{", ".join(param_names(node))}) full: {full}')


def folds_of(rel, tree, lines, keep=None, signatures=True):
    """Every docstring fold (and, with signatures, every long-signature fold) of the tree, except folds containing
    the line `keep` (the line asked for is never folded)."""
    out = []
    for node in ast.walk(tree) if tree is not None else ():
        if isinstance(node, (ast.Module,) + DEFS):
            r = doc_range(node)
            f = doc_fold(lines, *r) if r else None
            if f and not (keep is not None and r[0] <= keep <= r[1]):
                out.append(f)
        if signatures and isinstance(node, FUNCS):
            f = sig_fold(rel, node, lines)
            if f and not (keep is not None and node.lineno <= keep <= f[1]):
                out.append(f)
    return sorted(out)


def explicit_folds(tree, lines, start, end):
    """An explicit range folds only the docstrings longer than DOC_KEEP lines that lie entirely inside it."""
    out = []
    for node in ast.walk(tree) if tree is not None else ():
        if isinstance(node, (ast.Module,) + DEFS):
            r = doc_range(node)
            if r and start <= r[0] and r[1] <= end and r[1] - r[0] + 1 > DOC_KEEP:
                f = doc_fold(lines, *r)
                if f:
                    out.append(f)
    return sorted(out)


def parse_tree(rel, src):
    if not rel.endswith('.py'):
        return None
    try:
        return ast.parse(src)
    except (SyntaxError, ValueError):
        return None


def targets(node):
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def definitions(tree):
    """(node, qualname parts) for every def/class, plus module- and class-level assignments; at module and class
    level also those inside if/try/with/for/while blocks (v19: `try: ... except ImportError: class EmailStr` was
    listed by locate and then "not found" here)."""
    out, stack = [], [(tree, [], True)]
    while stack:
        node, prefix, assigns_ok = stack.pop()
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, DEFS):
                out.append((ch, prefix + [ch.name]))
                stack.append((ch, prefix + [ch.name], isinstance(ch, ast.ClassDef)))
            elif assigns_ok:
                for name in targets(ch):
                    out.append((ch, prefix + [name]))
                if isinstance(ch, BLOCKS):
                    stack.append((ch, prefix, True))
    return out


def find(tree, parts):
    defs = definitions(tree)
    while parts:
        matches = [(n, '.'.join(q)) for n, q in defs if q[-len(parts):] == parts]
        if matches:
            return sorted(matches, key=lambda m: m[0].lineno)
        parts = parts[1:]  # rich.console.Console.export_html -> console.Console.export_html -> ...
    return []


def clean_symbol(s):
    s = s.strip().strip('`"\'').strip()
    for kw in ('async def ', 'def ', 'class '):
        if s.startswith(kw):
            s = s[len(kw):].strip()
    if '(' in s:
        s = s[:s.index('(')]
    return s.replace('::', '.').replace('#', '.').rstrip(':').strip('.')


def read(rel):
    return open(os.path.join(ROOT, rel), encoding='utf-8', errors='replace').read()


def brief_outline(rel, src):
    """Top-level definitions with line ranges, for correction messages."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    for ch in tree.body:
        if isinstance(ch, ast.ClassDef):
            n = sum(isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) for c in ch.body)
            out.append(f'  L{ch.lineno}-{ch.end_lineno}  class {ch.name} ({n} methods)')
        elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f'  L{ch.lineno}-{ch.end_lineno}  def {ch.name}')
        else:
            for name in targets(ch):
                if ch.end_lineno - ch.lineno >= 2:
                    out.append(f'  L{ch.lineno}-{ch.end_lineno}  {name} = ...')
    if len(out) > OUTLINE_LINES:
        out = out[:OUTLINE_LINES] + [f'  ... {len(out) - OUTLINE_LINES} more; outline.py {rel} lists them all']
    return out


def need_location(rel, why):
    """FILE given without a usable SYMBOL / range: a correction that shows where things are."""
    src = read(rel)
    n = len(src.splitlines())
    msg = [f'{why} {rel} has {n} lines; pick a SYMBOL or a range of at most {RANGE_LINES} lines, e.g. '
           + argfix.call('show.py', rel, '1', str(min(n, RANGE_LINES)))]
    if rel.endswith('.py'):
        outline = brief_outline(rel, src)
        if outline:
            msg.append(f'definitions in {rel}; show one with ' + argfix.call('show.py', rel, 'NAME'))
            msg += outline
    return Misuse('\n'.join(msg))


def print_names(rel, ranges, src=None, tree=None):
    """v18: one `names:` line under the printed code: at most NAMES_MAX names related to the task, never one printed
    earlier in the task (flow meta NAMES_META: {display: call}) or defined in lines already read, while that result
    is still in the context (flow's State.visible_from: results before it are presumed gone after a compaction);
    nothing for files outside locate's candidates (and the files edited). With REPO_TOOLS_FLOW=off (a script's own
    unit tests: no task) nothing is remembered."""
    try:
        st = flow.peek()
    except Exception:
        st = None
    conn = getattr(st, 'conn', None) if os.environ.get('REPO_TOOLS_FLOW', 'on') != 'off' else None
    cands = {c[0] for c in getattr(st, 'candidates', None) or []}
    if cands and rel not in cands | set(getattr(st, 'edited_paths', None) or []):
        return
    visible = int(getattr(st, 'visible_from', 0) or 0)
    try:
        printed = json.loads(flow.get_meta(conn, NAMES_META, '{}') or '{}') if conn is not None else {}
        printed = printed if isinstance(printed, dict) else {}
    except Exception:
        printed = {}
    skip = {k for k, n in printed.items() if int(n or 0) >= visible}
    seen = [(v[0], v[1], v[2]) for v in (getattr(st, 'viewed', None) or []) if conn is not None
            and (len(v) < 4 or int(v[3] or 0) >= visible)]
    line, shown = names.names_line(rel, ranges, src, tree, limit=NAMES_MAX, skip=skip, seen_ranges=seen)
    if not line:
        return
    print(line)
    if conn is not None:
        try:
            n = int(getattr(st, 'n', 0) or 0) + 1  # this call
            printed.update({k: n for k in shown})
            flow.set_meta(conn, NAMES_META, json.dumps(printed))
        except Exception:
            pass


def show_range(rel, start, end, label=None, explicit=False, keep=None):
    """Lines start..end (at most RANGE_LINES). explicit (the model asked for this range): every line except long
    docstrings entirely inside it; otherwise every docstring and long signature folded (except the line keep)."""
    src = read(rel)
    lines = src.splitlines()
    if start > len(lines):
        print(f'{rel} has only {len(lines)} lines; e.g. ' + argfix.call('show.py', rel, max(1, len(lines) - 40), len(lines)))
        return 1
    start, end = max(1, start), min(len(lines), end)
    limit = EXPLICIT_RANGE_LINES if explicit else RANGE_LINES  # E039: shortening asked-for ranges cost re-reads
    stop = min(end, start + limit - 1)
    tree = parse_tree(rel, src)
    folds = explicit_folds(tree, lines, start, stop) if explicit else folds_of(rel, tree, lines, keep)
    LISTING.update(rel=rel, end=end, explicit=explicit)
    print(f'{rel}{" " + label if label else ""} L{start}-{stop} of {len(lines)}')
    code(lines, start, stop, folds)
    if stop < end:
        print(f'... truncated at {limit} lines; rest: ' + argfix.call('show.py', rel, stop + 1, end))
    flow.fact(path=rel, start=start, end=stop, label=label or '')
    print_names(rel, [(start, stop)], src, tree)
    return 0


def show_line(rel, line):
    """start without end: the innermost definition containing LINE when it is at most 2*LINE_HALF+1 lines; a longer
    one: its def line and LINE_HALF lines on each side of LINE (from LINE itself when the previous read of the file
    ended just before it: a continuation); outside any definition: LINE_HALF lines on each side. The window always
    contains LINE (v17 L8, rich_3777)."""
    src = read(rel)
    lines = src.splitlines()
    if line > len(lines):
        return show_range(rel, line, line)
    tree = parse_tree(rel, src)
    best = innermost(tree, line)
    if best is None:
        return show_range(rel, max(1, line - LINE_HALF), line + LINE_HALF, keep=line)
    node, qn = best
    start, end = def_start(node), node.end_lineno
    width = 2 * LINE_HALF + 1
    folds = folds_of(rel, tree, lines, keep=line)
    LISTING.update(rel=rel, end=end, explicit=False)
    if end - start + 1 <= width:
        print(f'{rel} {qn} L{start}-{end} of {len(lines)}')
        code(lines, start, end, folds)
        flow.fact(path=rel, start=start, end=end, label=qn)
        print_names(rel, [(start, end)], src, tree)
        return 0
    lo = line if continues(rel, line) else line - LINE_HALF
    lo = max(start, min(lo, end - width + 1))
    hi = min(end, lo + width - 1)
    print(f'{rel} {qn} L{lo}-{hi} of {len(lines)}')
    if lo <= node.lineno + 2:  # the window reaches the def line: one block from the start
        lo = start
        code(lines, lo, hi, folds)
        ranges = [(lo, hi)]
    else:
        code(lines, node.lineno, node.lineno)
        print(f'... (lines {node.lineno + 1}-{lo - 1} not shown; {qn} spans L{start}-{end})')
        code(lines, lo, hi, folds)
        ranges = [(node.lineno, node.lineno), (lo, hi)]
    if hi < end:
        print(f'... {qn} continues to line {end}; rest: '
              + argfix.call('show.py', rel, hi + 1, min(end, hi + RANGE_LINES)))
    cut('definition')
    flow.fact(path=rel, start=lo, end=hi, label=qn)
    print_names(rel, ranges, src, tree)
    return 0


def continues(rel, line):
    """The latest read of rel ended just before line (start=N right after a window that ended at N-1)."""
    try:
        st = flow.peek()
        v = next((v for v in st.viewed if v[0] == rel), None) if st is not None else None
        return v is not None and v[2] is not None and line - 3 <= int(v[2]) < line
    except Exception:
        return False


def class_named(tree, parts):
    """(ClassDef, qualname) of the class whose qualname ends with parts, or (None, None)."""
    hits = [(n, '.'.join(q)) for n, q in definitions(tree) if isinstance(n, ast.ClassDef) and q[-len(parts):] == parts]
    return hits[0] if hits else (None, None)


def show_symbol(rel, symbol, depth=0):
    src = read(rel)
    if not rel.endswith('.py'):
        raise need_location(rel, f'{rel} is not a Python file, so SYMBOL lookup does not work;')
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f'{rel}: SyntaxError line {e.lineno}: {e.msg}; show a line range instead, e.g. '
              + argfix.call('show.py', rel, max(1, (e.lineno or 1) - 20), (e.lineno or 1) + 20))
        return 1
    parts = clean_symbol(symbol).split('.')
    matches = find(tree, parts)
    if len(parts) >= 2 and not argfix.is_placeholder(symbol):
        cls, cqn = class_named(tree, parts[:-1])
        if cls is not None and not any(q == f'{cqn}.{parts[-1]}' for _, q in matches):
            return show_member(rel, src, tree, cls, cqn, parts[-1], depth)
    if not matches:
        if argfix.is_placeholder(symbol):
            raise need_location(rel, f'{symbol!r} is a placeholder, not a name in the file.')
        if depth == 0:
            r = show_elsewhere(rel, src, tree, parts)
            if r is not None:
                return r
        qnames = sorted({'.'.join(q) for _, q in definitions(tree)})
        close = difflib.get_close_matches(clean_symbol(symbol), qnames, n=5, cutoff=0.6)
        close += [q for q in qnames if q.split('.')[-1].lower() == clean_symbol(symbol).split('.')[-1].lower()
                  and q not in close]
        hint = f' Close names: {", ".join(close[:5])}.' if close else ''
        print(f'{symbol} not found in {rel}.{hint} Use outline.py {rel} to list definitions.')
        return 1
    lines = src.splitlines()
    node, qn = matches[0]
    start = def_start(node) if isinstance(node, DEFS) else node.lineno
    end = node.end_lineno
    if len(matches) > 1:
        print(f'{len(matches)} matches: ' + ', '.join(f'{q} L{n.lineno}' for n, q in matches[:8]) + f'; showing {qn}')
    if end - start + 1 > LONG_CLASS and isinstance(node, ast.ClassDef):
        return show_long_class(rel, src, lines, tree, node, qn, start, end)
    if end - start + 1 > LONG_FN and isinstance(node, FUNCS):
        return show_function_window(rel, src, lines, tree, node, qn, start, end)
    LISTING.update(rel=rel, end=end, explicit=False)
    print(f'{rel} {qn} L{start}-{end}')
    stop = min(end, start + MAX_LINES - 1)
    code(lines, start, stop, folds_of(rel, tree, lines))
    if end > stop:
        print('... truncated; rest: ' + argfix.call('show.py', rel, stop + 1, min(end, stop + RANGE_LINES)))
        cut('symbol')
    flow.fact(path=rel, start=start, end=stop, label=qn)
    print_names(rel, [(start, stop)], src, tree)
    return 0


def statement_of(tree, line):
    """The innermost statement starting at line (for an attribute assignment: the whole statement), or None."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and node.lineno == line and not isinstance(node, DEFS):
            if best is None or (node.end_lineno or line) < (best.end_lineno or line):
                best = node
    return best


def show_member(rel, src, tree, cls, cqn, attr, depth=0):
    """symbol=Class.attr where attr is not a def of the class: where it is assigned, inherited from, or that the
    class has no such member (with its members and base classes)."""
    mod = names.load(rel, src, tree)
    hit = names.find_member(mod, cls, attr) if mod else None
    if hit:
        d, owner_cls = hit
        if d.kind == 'attribute':
            return show_assignment(d, f'{cqn}.{attr}', owner_cls)
        if d.rel != rel or owner_cls != cqn:  # inherited from a base class in the repository
            print(f'NOTE: {cqn} has no {attr} of its own; it inherits {owner_cls}.{attr} from {d.rel}:')
            if depth < 2:
                return show_symbol(d.rel, f'{owner_cls}.{attr}', depth + 1)
        if d.kind == 'class attribute':
            return show_assignment(d, f'{cqn}.{attr}', owner_cls)
    bases = names.describe_bases(mod, cls) if mod else []
    head = f'class {cqn}' + (f'({", ".join(b.split(" (")[0] for b in bases)})' if bases else '')
    print(f'{cqn}.{attr} not found: {cqn} has no {attr} (no def, property, class attribute or self.{attr} assignment '
          f'in {cqn}' + (' or its base classes in the repository).' if bases else ').'))
    print(f'{head}  {rel}:{cls.lineno}' + ('  bases: ' + '; '.join(bases) if bases else ''))
    ext = names.external_bases(mod, cls) if mod else []
    if ext:
        first = ext[0]
        m, _, c = first.rpartition('.')
        check = argfix.obj('run.py', {'snippet': f'from {m} import {c}; print(hasattr({c}, {attr!r}))'}) if m else ''
        print(f'{attr} may be inherited from outside the repository: ' + ', '.join(dict.fromkeys(ext))
              + (f' (not in this repository; to check: {check})' if check else ''))
    if mod:
        members, close, more = names.member_list(mod, cls, attr)
        if close:
            print('close names: ' + ', '.join(f'{cqn}.{c}' for c in close))
        print(f'{cqn} members:')
        for line in names.wrap(members + ([f'... {more} more'] if more else [])):
            print(line)
    return 1


def show_assignment(d, asked, owner_cls):
    """An attribute assigned in a method (or a class attribute): the statement with 2 lines of context."""
    src = read(d.rel)
    lines = src.splitlines()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None
    stmt = statement_of(tree, d.line) if tree is not None else None
    a = d.line
    b = (stmt.end_lineno if stmt is not None and stmt.end_lineno else a)
    b = min(b, a + RANGE_LINES - 6)
    start, stop = max(1, a - 2), min(len(lines), b + 2)
    where = d.text.rsplit('[in ', 1)[-1].rstrip(']') if '[in ' in d.text else owner_cls
    kind = 'an instance attribute' if d.kind == 'attribute' else 'a class attribute'
    print(f'NOTE: {asked} is {kind}, not a def: it is assigned in {where} at {d.rel}:{d.line}')
    print(f'{d.rel} {d.qualname} L{start}-{stop}')
    code(lines, start, stop)
    flow.fact(path=d.rel, start=start, end=stop, label=d.qualname)
    print_names(d.rel, [(start, stop)], src, tree)
    return 0


def show_elsewhere(rel, src, tree, parts):
    """A module-level name that rel does not define: imported (shown where it is defined) or defined in another
    file (shown there; several: ready calls). None when nothing is found."""
    name = parts[0] if len(parts) == 1 else None
    mod = names.load(rel, src, tree)
    if name and mod is not None and name in mod.imports:
        m, attr, line = mod.imports[name]
        d = names.resolve_name(mod, name)
        src_txt = f'from {m} import {attr}' if attr else f'import {m}'
        if isinstance(d, names.Def) and d.kind == 'module':
            print(f'{name} is the module {d.rel} (imported at line {line}: {src_txt}); list its definitions with '
                  + argfix.obj('outline.py', {'files': d.rel}))
            return 1
        if isinstance(d, names.Def):
            print(f'NOTE: {name} is not defined in {rel}: it is imported at line {line} ({src_txt}); '
                  f'its definition is in {d.rel}:')
            return show_symbol(d.rel, d.qualname, 1)
        where = 'outside the repository' if isinstance(d, tuple) else 'from a module not found in the repository'
        print(f'{name} is not defined in {rel}: it is imported at line {line} ({src_txt}), {where}.')
        return 1
    last = parts[-1]
    try:
        import kb
        rows = [r for r in kb.find_defs(last, kb.open(ROOT)) if not kb.is_aux(r[0])]
    except Exception:
        rows = []
    if len(parts) >= 2:
        want = '.'.join(parts)
        rows = [r for r in rows if r[3] == want or r[3].endswith('.' + want)] or \
            [r for r in rows if r[3].split('.')[-2:] == parts[-2:]]
    uniq = list(dict.fromkeys((r[0], r[3]) for r in rows))
    if len(uniq) == 1:
        p, qn = uniq[0]
        print(f'NOTE: {".".join(parts)} is not defined in {rel}; it is defined in {p}:')
        return show_symbol(p, qn, 1)
    if len(uniq) > 1:
        print(f'{".".join(parts)} is not defined in {rel}; it is defined in {len(uniq)} places:')
        for p, qn in uniq[:6]:
            print('  ' + argfix.obj('show.py', {'file': p, 'symbol': qn}))
        return 1
    return None


def show_long_class(rel, src, lines, tree, node, qn, start, end):
    """A class too long to print: its methods with line ranges, its first LONG_HEAD lines, and the method most
    related to the task as the next call (flow fact next_show)."""
    methods = [(c, f'{qn}.{c.name}') for c in node.body if isinstance(c, FUNCS)]
    print(f'{rel} class {qn} L{start}-{end} ({end - start + 1} lines); its methods:')
    for line in names.wrap([f'{c.name} L{c.lineno}-{c.end_lineno}' for c, _ in methods[:METHODS_MAX]]
                           + ([f'... {len(methods) - METHODS_MAX} more'] if len(methods) > METHODS_MAX else [])):
        print(line)
    if len(methods) > METHODS_MAX:
        cut('class')
    stop = min(end, start + LONG_HEAD - 1)
    first = next((c.lineno for c, _ in methods if c.lineno > start), None)
    if first:
        stop = min(stop, def_start(next(c for c, _ in methods if c.lineno == first)) - 1)
    print(f'its first lines:')
    LISTING.update(rel=rel, end=stop, explicit=False)
    code(lines, start, stop, folds_of(rel, tree, lines))
    best = flow.related_symbol(flow.peek(), rel, [mq for _, mq in methods])
    if best:
        print('most related to the task: ' + argfix.obj('show.py', {'file': rel, 'symbol': best}))
        flow.fact(next_show=(rel, best))
    return 0


def show_outline(rel):
    """A target without symbol or range: a short file whole, a Python file's outline (with a ready call for the
    definition most related to the task), another file's first RANGE_LINES lines."""
    n = len(read(rel).splitlines())
    if n <= RANGE_LINES:
        print('NOTE: the whole file (it is short)')
        return show_range(rel, 1, RANGE_LINES)
    if not rel.endswith('.py'):
        print(f'NOTE: no start/end given; the first {RANGE_LINES} lines of {n}')
        return show_range(rel, 1, RANGE_LINES)
    try:
        tree = ast.parse(read(rel))
    except SyntaxError:
        print(f'NOTE: no symbol or start/end given; the first {RANGE_LINES} lines of {n}')
        return show_range(rel, 1, RANGE_LINES)
    print(f'NOTE: {rel} has {n} lines; its outline:')
    found = []
    for ch in tree.body:
        if isinstance(ch, ast.ClassDef):
            print(f'  L{ch.lineno}-{ch.end_lineno} class {ch.name}')
            for c in ch.body:
                if isinstance(c, FUNCS):
                    found.append(f'{ch.name}.{c.name}')
                    if len(found) <= OUTLINE_LINES * 2:
                        print(f'    L{c.lineno}-{c.end_lineno} {c.name}')
        elif isinstance(ch, FUNCS):
            found.append(ch.name)
            if len(found) <= OUTLINE_LINES * 2:
                print(f'  L{ch.lineno}-{ch.end_lineno} def {ch.name}')
    if len(found) > OUTLINE_LINES * 2:
        print(f'  ... {len(found) - OUTLINE_LINES * 2} more definitions; ' + argfix.call('outline.py', rel)
              + ' lists them all')
        cut('outline')
    best = flow.related_symbol(flow.peek(), rel, found) or (found[0] if found else None)
    if best:
        print('e.g. ' + argfix.obj('show.py', {'file': rel, 'symbol': best}))
        flow.fact(next_show=(rel, best))
    else:
        flow.fact(outlined=rel)
    return 0


def innermost(tree, line):
    """(node, qualname) of the innermost def/class containing line (decorators included), or None."""
    best = None
    for node, q in definitions(tree) if tree is not None else ():
        if isinstance(node, DEFS) and def_start(node) <= line <= node.end_lineno:
            if best is None or node.lineno >= best[0].lineno:
                best = (node, '.'.join(q))
    return best


def def_start(node):
    return min([d.lineno for d in getattr(node, 'decorator_list', [])] + [node.lineno])


def issue_terms():
    """{term: weight} of the task's terms (the locate query in the flow state): identifiers (snake_case, CamelCase,
    CONSTANTS, dunders, names with digits) weigh 3, other words of 5+ letters 1; {} without a query."""
    try:
        st = flow.peek()
        q = flow.get_meta(st.conn, 'query', '') if st is not None else ''
    except Exception:
        q = ''
    out = {}
    for t in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', q or ''):
        dunder = len(t) > 4 and t.startswith('__') and t.endswith('__')
        low = t.lower() if dunder else t.lower().strip('_')
        if len(low) < 3 or low in STOP or keyword.iskeyword(low):
            continue
        core = t.strip('_')
        ident = dunder or '_' in core or bool(re.search(r'[a-z][A-Z]', core)) or (core.isupper() and len(core) >= 4) \
            or any(c.isdigit() for c in core)
        if ident:
            out[low] = 3
        elif len(low) >= 5:
            out.setdefault(low, 1)
    return out


def best_window(lines, first, last, width, terms):
    """Start line of the window of `width` lines within first..last with the most issue-term weight (comment lines
    count half), with at least 5 lines above its first matching line; first when nothing matches."""
    hi_start = max(first, last - width + 1)
    if not terms:
        return first
    score = {}
    for i in range(first, last + 1):
        low = lines[i - 1].lower()
        w = sum(v for t, v in terms.items() if t in low)
        if w:
            score[i] = w / 2 if low.lstrip().startswith('#') else w
    if not score:
        return first
    best, best_s = first, -1
    for m in sorted(score):
        s = max(first, min(m - width // 4, hi_start))
        total = sum(v for i, v in score.items() if s <= i < s + width)
        if total > best_s:
            best, best_s = s, total
    top = min(i for i in score if best <= i < best + width)
    return max(first, min(best, top - 5))  # at least 5 lines above the first matching line in the window


def show_function_window(rel, src, lines, tree, node, qn, start, end):
    """v18: a function longer than LONG_FN lines: its head (signature, folded when long; the docstring's first line)
    and a window of FN_WINDOW lines around the lines that match the issue's terms, then a ready call for the rest."""
    folds = folds_of(rel, tree, lines)
    doc = doc_range(node)
    head_end = max(header_end(node, lines), doc[1] if doc else 0)
    body_first = min(head_end + 1, end)
    lo = best_window(lines, body_first, end, FN_WINDOW, issue_terms())
    hi = min(end, lo + FN_WINDOW - 1)
    LISTING.update(rel=rel, end=end, explicit=False)
    print(f'{rel} {qn} L{start}-{end} of {len(lines)}')
    if lo <= head_end + 3:  # the window starts at the body: one block
        lo = start
        code(lines, start, hi, folds)
        ranges = [(start, hi)]
    else:
        code(lines, start, head_end, folds)
        print(f'... (lines {head_end + 1}-{lo - 1} not shown)')
        code(lines, lo, hi, folds)
        ranges = [(start, head_end), (lo, hi)]
    if hi < end:
        print(f'... {qn} continues to line {end}; rest: '
              + argfix.call('show.py', rel, hi + 1, min(end, hi + RANGE_LINES)))
    elif lo > head_end + 1:
        print(f'rest: ' + argfix.call('show.py', rel, head_end + 1, min(lo - 1, head_end + RANGE_LINES)))
    cut('function')
    flow.fact(path=rel, start=lo, end=hi, label=qn)
    print_names(rel, ranges, src, tree)
    return 0


def parse(argv):
    """Return (action, args, notes) where action is 'symbol', 'range', 'line' or 'file'."""
    d, note = argfix.object_args(argv, 'show.py')
    if d is not None:
        return parse_object(d, [note] if note else [])
    argv, note = argfix.unpack(argv, 'show.py')
    notes = [note] if note else []
    if argv and argv[0] == '--lines':  # edit.py's syntax used for show.py
        argv = argv[1:]
        notes.append('show.py takes no --lines flag')
    if len(argv) == 1:
        a = argv[0].strip()
        if ':' in a and ' ' not in a:
            head, _, tail = a.replace('::', ':').partition(':')
            argv = [head] + argfix.split(tail.replace(':', ' '))
        elif ' ' in a or '\n' in a:
            argv = argfix.split(a)
        if len(argv) > 1:
            notes.append('split the single argument into FILE and SYMBOL/START END')
    if not argv:
        raise Misuse('show.py needs FILE and SYMBOL, or FILE START END.\n' + USAGE)
    return parse_rest(argv[0], [a for a in argv[1:] if a.strip().strip('`').strip()], notes)


def find_symbol(symbol):
    """(rel, qualname) of the definition of SYMBOL in the repository (a call without a target), or Misuse listing
    the candidates as ready calls. Source files first; several matches: the exact qualname, else the one in
    locate's candidate files, when unique."""
    name = clean_symbol(symbol)
    if not name or argfix.is_placeholder(symbol):
        raise Misuse('show.py needs the key target with symbol, e.g. '
                     + argfix.obj('show.py', {'file': 'billing/invoice.py', 'symbol': 'Invoice.total'}))
    import kb  # the code knowledge base (built by locate.py; imported only when needed)
    rows = kb.find_symbol(name, kb.open(ROOT))
    rows = [r for r in rows if not kb.is_aux(r[0])] or rows
    if not rows:
        raise Misuse(f'{name} is not defined in the repository. Search for it: '
                     + argfix.obj('search.py', {'text': name.split('.')[-1]}))
    uniq = list(dict.fromkeys((r[0], r[1]) for r in rows))
    if len(uniq) > 1:
        exact = [u for u in uniq if u[1] == name]
        st = flow.peek()
        cands = {c[0] for c in st.candidates} if st else set()
        in_cands = [u for u in uniq if u[0] in cands]
        if len(exact) == 1:
            uniq = exact
        elif len(in_cands) == 1:
            uniq = in_cands
    if len(uniq) == 1:
        return uniq[0]
    ready = '\n'.join('  ' + argfix.obj('show.py', {'file': p, 'symbol': q}) for p, q in uniq[:6])
    raise Misuse(f'{name} is defined in {len(uniq)} places; add the target, e.g. one of:\n' + ready)


def heal(d, notes):
    """Complete a call whose intent is clear (see the module docstring); notes say how it was read."""
    d = {k: v for k, v in d.items() if str(v).strip() or k in ('symbol', 'start', 'end')}
    if not str(d.get('file', '')).strip():
        if str(d.get('symbol', '')).strip():
            rel, qn = find_symbol(d['symbol'])
            d = {'file': rel, 'symbol': qn}
            notes.append('no target given; read as ' + argfix.obj('show.py', d))
        elif str(d.get('start', '')).strip() and argfix.line_number(d['start']) is not None:
            cur = flow.current_file()
            if cur:
                d['file'] = cur
                notes.append('no target given; read as ' + argfix.obj('show.py', {k: d[k] for k in ('file', 'start', 'end') if k in d}))
    if str(d.get('symbol', '')).strip() and ('start' in d or 'end' in d):
        d = {k: v for k, v in d.items() if k not in ('start', 'end')}
        notes.append('symbol and start/end given; showing the symbol')
    return d


def parse_object(d, notes):
    """The object form: {"file", "symbol"} or {"file", "start", "end"} ("start" alone: the definition there)."""
    d = heal(d, notes)
    argfix.require(d, 'show.py', 'file')
    argfix.exclusive(d, 'show.py', ('symbol',), ('start', 'end'))
    if 'end' in d and 'start' not in d:
        argfix.require(d, 'show.py', 'start', example={'file': d['file'], 'start': '1', 'end': d['end']})
    if 'symbol' in d:
        argfix.require(d, 'show.py', 'symbol', example={'file': d['file'], 'symbol': 'Invoice.total'})
    for k in ('start', 'end'):
        v = d.get(k, '')
        if v.strip() and argfix.line_number(v) is None and argfix.line_range(v) is None \
                and not argfix.is_placeholder(v):
            raise Misuse(f'"{k}" must be a line number, got {v!r}. E.g. '
                         + argfix.obj('show.py', {'file': d['file'], 'start': '120', 'end': '180'})
                         + (' (a name goes in "symbol")' if k == 'start' else ''))
    rest = [d[k] for k in ('symbol', 'start', 'end') if k in d and d[k].strip()]
    return parse_rest(d['file'], rest, notes)


def parse_rest(file_arg, rest, notes):
    rel, notes_path = argfix.resolve(file_arg, 'show.py', ('Invoice.total',), nearby=True)
    if notes_path:
        notes.append(notes_path)
    if os.path.isdir(os.path.join(ROOT, rel)):
        raise Misuse(f'{rel} is a directory. List its files with outline.py {rel}, then e.g. '
                     + argfix.call('show.py', rel.rstrip('/') + '/<file>.py', 'NAME'))
    if len(rest) >= 2:
        s, e = argfix.line_number(rest[0]), argfix.line_number(rest[1])
        if len(rest) > 2 and s is not None and e is not None:
            notes.append(f'ignored extra arguments {rest[2:]!r}'[:120])
        s_ph = s is None and argfix.is_placeholder(rest[0])
        e_ph = e is None and argfix.is_placeholder(rest[1])
        if (s is None and not s_ph) or (e is None and not e_ph):
            if s is None and e is None and len(rest) == 2:
                raise Misuse(f'show.py takes one SYMBOL or two line numbers, got {rest!r}. E.g. '
                             + argfix.call('show.py', rel, clean_symbol(rest[0]) or 'NAME'))
            raise Misuse(f'line numbers must be integers, got {rest[:2]!r}. E.g. '
                         + argfix.call('show.py', rel, '120', '180'))
        if s_ph and e_ph:
            return 'file', (rel, f'{rest[0]!r} and {rest[1]!r} are placeholders, not line numbers.'), notes
        if s_ph or e_ph:
            s, e = (1, e) if s_ph else (s, len(read(rel).splitlines()))
            notes.append(f'read {rest[0] if s_ph else rest[1]!r} as the {"start" if s_ph else "end"} of the file: '
                         + argfix.call('show.py', rel, s, e))
        if e < s:
            s, e = e, s
            notes.append('swapped START and END')
        return 'range', (rel, s, e), notes
    if len(rest) == 1:
        r = argfix.line_range(rest[0])
        if r:
            notes.append(f'read {rest[0]!r} as the range {r[0]} {r[1]}')
            return 'range', (rel, min(r), max(r)), notes
        n = argfix.line_number(rest[0])
        if n is not None:
            return 'line', (rel, n), notes
        return 'symbol', (rel, rest[0]), notes
    return 'file', (rel, 'show.py needs a SYMBOL or START END after FILE;'), notes


def main(argv):
    set_level(compact_level())
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = run(argv)
            if CUT:
                print(COMPACT_LINE)
    finally:
        out = buf.getvalue()
        sys.stdout.write(capped(out, EXPLICIT_CAP if LISTING.get('explicit') else CAP))
    return rc


CODE_LINE = re.compile(r'^(\d+)(?:  |$)')


def capped(out, cap):
    """v18: output longer than cap characters: without its names line, then cut at a line with a ready call for the
    rest of the listing (flow's fact end follows the last line printed)."""
    if len(out) <= cap:
        return out
    lines = [ln for ln in out.rstrip('\n').split('\n') if not ln.startswith('names: ')]
    if sum(len(ln) + 1 for ln in lines) <= cap:
        return '\n'.join(lines) + '\n'
    keep, size = [], 0
    for ln in lines:
        if size + len(ln) + 1 > cap - 170:
            break
        keep.append(ln)
        size += len(ln) + 1
    last = next((int(m.group(1)) for m in map(CODE_LINE.match, reversed(keep)) if m), None)
    rest = f'... cut at {cap} characters'
    if last is not None and LISTING.get('rel'):
        end = max(int(LISTING.get('end') or last), last + 1)
        rest += '; rest: ' + argfix.call('show.py', LISTING['rel'], last + 1, min(end, last + RANGE_LINES))
        flow.fact(end=last)
    return '\n'.join(keep + [rest]) + '\n'


def run(argv):
    try:
        action, args, notes = parse(argv)
        for note in notes:
            print(f'NOTE: {note}')
        if action == 'file':
            rel, why = args
            if why.startswith('show.py needs a SYMBOL'):  # only a target: its outline
                return show_outline(rel)
            if len(read(rel).splitlines()) <= RANGE_LINES:
                print('NOTE: the whole file (it is short)')
                return show_range(rel, 1, RANGE_LINES)
            raise need_location(rel, why)
        if action == 'range':
            return show_range(*args, explicit=True)
        if action == 'line':
            return show_line(*args)
        return show_symbol(*args)
    except Misuse as e:
        print(f'USAGE ERROR: {e}')
        if 'e.g.' not in str(e).lower():
            print(USAGE)
        return 2


if __name__ == '__main__':
    sys.exit(flow.main('show.py', main))
