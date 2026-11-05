"""Rank the files and functions most likely to need changing for an issue.

usage: locate.py  query=<issue title + identifiers + error text>   (+ optional k=4, show=no)
       (positional form, still accepted: locate.py [--k N] [--no-show] <issue text>)

Stdlib only, on the code knowledge base (kb.py: built on first use, reused afterwards). Scores every .py file with
BM25 over identifiers, plus its best function's BM25, a BM25 of the query over the file's own path (v19: file and
package names), and boosts for files that define identifiers named in the issue, mentioned paths (a bare file name
counts less the more files have it) and modules, traceback frames, error/quoted text found in string literals, and
files changed by past commits whose subject resembles the issue. Tests, docs, examples and vendored code are
down-weighted (v19: not a non-test file the issue mentions by path, nor docs/examples when the issue is about them;
a package's own tools/ or scripts/ directory is source). Within a file, functions the issue names (dotted qualname,
or a code-like identifier; v18, P1 of note 13) come before similarity guesses, the most similar first (v19); a def
nested in a def counts as its outer def.
Output (v18 "lean", note 13 B): the ranked list (k files x up to 3 functions with line ranges; every function the
issue names in a listed file), one meta line (low confidence, a similar past commit, the tests to run), then a
preview of the best function, only when confident (the top file clearly leads, and the best function is named in
the issue or clearly leads its file): the whole function when it fits PREVIEW_LINES lines, else its def line plus
the window of body lines that mention the issue's terms; docstrings and long signatures are folded to one marker
line. The facts for the flow engine (candidates, identifier count, confidence, thin issue, the range shown) go
through flow.fact, not the text.
Measured with scripts/locate_bench.py (v18d -> v19): gold file first on 299 SWE-bench-train tasks 48.5% -> 53.8%,
on the 112 local tasks 54.5% -> 58.0% (queries built from the statement) and 66.1% -> 68.8% (the model's queries);
the preview is shown less often (64% -> 55% external) and right more often (42% -> 51% of the previews).
"""

import ast
import collections
import math
import os
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402  (flow engine: refusals + NEXT/BUDGET footers)
import argfix  # noqa: E402
import kb  # noqa: E402

ROOT = (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))  # overridable for local testing
K_DEFAULT = 4           # files listed (v18: 5 -> 4)
FNS_PER_FILE = 3        # functions listed per file ...
NAMED_PER_FILE = 5      # ... or up to this many when the issue names them
NAMED_MAX_FILES = 3     # an issue identifier names a function only when at most this many source files define it
PREVIEW_LINES = 25      # v18: the best match is printed in at most this many lines (markers included)
SIG_LINES = 3           # a longer signature is folded to its first line plus a marker
FN_WEIGHT = 0.5         # best function BM25 (normalized) added to the file score
PATH_WEIGHT = 0.5       # BM25 of the query over the file's path (last PATH_PARTS components), normalized
PATH_PARTS = 2
MENTION_WEIGHT = 2.0    # a mentioned path, divided by the number of files it can name (main.py: many)
DEFINES_WEIGHT = 0.6    # per mentioned identifier the file defines, divided by the number of defining files
AUX_FACTOR = 0.3        # tests/docs/examples/scripts/vendored code (see rank() for the exceptions)
HISTORY_WEIGHT = 0.5    # files touched by past commits with a similar subject (normalized to max 1)
HISTORY_COMMITS = 10
HISTORY_MAX_FILES = 10  # ignore sweeping commits
FILE_EXTS = {'py', 'pyi', 'pyx', 'build', 'txt', 'md', 'rst', 'cfg', 'ini', 'toml', 'yaml', 'yml', 'json', 'sql', 'sh',
             'html', 'js', 'ts', 'css', 'c', 'h', 'cc', 'cpp', 'go', 'exe', 'whl', 'tgz', 'gz', 'zip', 'log', 'csv',
             'conf', 'xml', 'lock', 'in', 'tmpl', 'jpg', 'png', 'ipynb'}
STRING_TOKENS = 5       # error/quoted text of the issue: its first tokens as a phrase among the string literals,
STRING_MAX_FILES = 5    # worth 1 / the number of files containing it; ignored in more files than this
TITLE_WEIGHT = 0.8      # a file defining a function named by consecutive title words (title_defs)
TITLE_MAX_DEFS = 3      # ... only when the title words name at most this many functions
TITLE_GENERIC = {'fix', 'fixed', 'fixes', 'fixing', 'add', 'added', 'adds', 'adding', 'support', 'allow', 'allows',
                 'make', 'makes', 'use', 'uses', 'update', 'bug', 'issue', 'error', 'handle', 'remove', 'improve',
                 'new', 'feat', 'feature', 'refactor', 'docs', 'test', 'tests', 'correct', 'properly'}
LOW_CONF_RATIO = 1.2
FN_MARGIN = 1.1         # preview an unnamed best function only when it scores this much above the next in its file
THIN_MIN_WORDS = 5      # fewer distinct query words -> thin issue
THIN_WORDS = 12         # ... or no identifiers/paths/tracebacks and fewer distinct words than this
IDENT_RE = kb.IDENT_RE
STOP = kb.STOP
tokenize = kb.tokenize
subtoks = kb.subtoks
DOCS_DIR = re.compile(r'(^|/)(docs|docs_src|examples?)/')
VENDORED = ('/_vendor/', '/vendor/', '/vendored/', '/third_party/', '/thirdparty/', '/externals/', '/extern/')


def is_aux(path):
    """Tests, docs, examples, scripts (kb.is_aux), Django-style tests.py modules and vendored third-party code. A
    tools/ or scripts/ directory below the top level is a package's own code (conan/tools/, mesonbuild/scripts/)."""
    p = '/' + path.lower()
    if p.endswith(('/tests.py', '/test.py')) or any(v in p for v in VENDORED):
        return True
    parts = path.split('/')
    inner = [x for x in parts[1:-1] if x.lower() not in ('tools', 'scripts')]
    return kb.is_aux('/'.join(parts[:1] + inner + parts[-1:]) if len(parts) > 1 else path)


def parse_issue(text):
    q = {'tokens': tokenize(text), 'dotted': set()}
    code = ' '.join(re.findall(r'`([^`\n]+)`', text)) + ' ' + ' '.join(re.findall(r'```.*?```', text, re.S))
    idents = set(IDENT_RE.findall(code))
    for m in IDENT_RE.findall(text):  # code-looking identifiers anywhere
        if '_' in m.strip('_') or re.search(r'[a-z][A-Z]', m) or re.match(r'[A-Z][a-z]+[A-Z]', m):
            idents.add(m)
    for m in re.findall(r'\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\b', text):  # dotted names
        idents.update(m.split('.'))
        q['dotted'].add(m)
    idents.update(re.findall(r'(\w+)\(', text))
    q['idents'] = {i for i in idents if len(i) > 2 and i.lower() not in STOP}
    q['paths'] = set(re.findall(r'([\w./-]+\.py)\b', text))
    q['quoted'] = [a or b for a, b in re.findall(r'"([^"\n]{12,160})"|\'([^\'\n]{12,160})\'', text)
                   if ' ' in (a or b)]
    q['frames'] = re.findall(r'File "([^"]+)", line (\d+), in (\w+)', text)
    q['strings'] = re.findall(r'(?:Error|Exception|Warning)\w*: (.{12,80})', text)
    # thin issue: nothing code-like to anchor on and little text (e.g. a bare PR title linking an issue)
    words = len(set(q['tokens']))
    q['thin'] = words < THIN_MIN_WORDS or (not (q['idents'] or q['paths'] or q['frames']) and words < THIN_WORDS)
    return q


def history_scores(conn, text, paths):
    """File -> share of similar past commits (by subject) that touched it, normalized to max 1."""
    hist = collections.defaultdict(float)
    evidence = {}
    query = kb._match_query(tokenize(text))
    if not query:
        return hist, evidence
    try:
        rows = conn.execute("""SELECT c.sha, c.subject, -bm25(fts_commits) FROM fts_commits JOIN commits c
                               ON c.id = fts_commits.rowid WHERE fts_commits MATCH ?
                               ORDER BY bm25(fts_commits) LIMIT 60""", (query,)).fetchall()
    except Exception:
        return hist, evidence
    used = 0
    top = None
    for sha, subject, s in rows:
        files = [p for (p,) in conn.execute('SELECT path FROM commit_files WHERE sha = ?', (sha,))
                 if p in paths and not is_aux(p)]
        if not files or len(files) > HISTORY_MAX_FILES:
            continue
        top = top or s
        for p in files:
            hist[p] += (s / top) / len(files)
            evidence.setdefault(p, subject)
        used += 1
        if used >= HISTORY_COMMITS:
            break
    mx = max(hist.values(), default=0)
    if mx > 0:
        for p in hist:
            hist[p] /= mx
    return hist, evidence


def _name_parts(name):
    """Lower-case words of an identifier: _get_padding_width -> [get, padding, width]; PaddingWidth -> [padding,
    width]."""
    return [w.lower() for w in re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).split('_') if w]


def title_defs(conn, text, paths):
    """{path: (name, (path, qualname, start, end))}: functions whose name is two or three consecutive plain words of
    the issue title plus at most one more word ("fix for padding width" -> Table._get_padding_width); stop words,
    generic verbs and identifiers in the title are skipped, dunders never match.
    Only when such a run names at most TITLE_MAX_DEFS functions outside tests/docs (rich_3935, note 11 L7)."""
    title = text.strip().split('\n', 1)[0]
    plain = [t.strip('.,:;!?()"\'') for t in title.split()]  # identifiers (path_url, a.b) are not title words
    words = [w.lower() for w in plain if re.fullmatch(r'[A-Za-z]+', w)][:16]
    words = [w for w in words if len(w) >= 3 and w not in STOP and w not in TITLE_GENERIC]
    runs = [tuple(words[i:i + n]) for n in (3, 2) for i in range(len(words) - n + 1)]
    for run in runs:
        rows = conn.execute("SELECT path, qualname, name, start, \"end\" FROM symbols WHERE kind != 'class' AND "
                            "lower(name) LIKE ?", ('%' + '%'.join(run) + '%',)).fetchall()
        hits = []
        for path, qn, name, start, end in rows:
            parts = _name_parts(name)
            if name.startswith('__') or len(parts) - len(run) > 1:  # dunders; the words must be most of the name
                continue
            if path in paths and not is_aux(path) and any(tuple(parts[i:i + len(run)]) == run
                                                          for i in range(len(parts) - len(run) + 1)):
                hits.append((path, qn, name, start, end))
        if hits and len(hits) <= TITLE_MAX_DEFS:
            out = {}
            for path, qn, name, start, end in hits:
                out.setdefault(path, (name, (path, qn, start, end)))
            return out
    return {}


def named_functions(conn, text, q, paths):
    """{path: [(path, qualname, start, end), ...]}: functions/methods the issue names, in the order the issue first
    mentions them (q['named_kind'][(path, qualname)]: 0 dotted, 1 bare). A dotted name matching a qualname's tail, or
    ending in a Class.method qualname (Console.is_terminal, rich.console.Console.is_terminal, also a dunder:
    Text.__add__) always counts; a bare identifier counts when it looks like code (is_gen_callable, getAttr, or in
    backticks: v19, a plain word the reporter's own code calls, np.array( or display(, names nothing), is not a dunder
    and at most NAMED_MAX_FILES source files define it (a common name such as get or run says nothing about where).
    Tests/docs/examples are skipped."""
    dotted = {d for d in q['dotted'] if d.count('.') >= 1}
    idents = {i for i in q['idents'] if not (i.startswith('__') and i.endswith('__'))}
    tails = {d.split('.')[-1] for d in dotted}
    names = sorted(idents | tails)
    if not names:
        return {}
    ph = ','.join('?' * len(names))
    rows = [r for r in conn.execute(f"SELECT path, qualname, name, start, \"end\" FROM symbols WHERE kind != 'class' "
                                    f"AND name IN ({ph})", names) if r[0] in paths and not is_aux(r[0])]
    nfiles = collections.defaultdict(set)
    for path, qn, name, *_ in rows:
        nfiles[name].add(path)

    ticked = set(IDENT_RE.findall(' '.join(re.findall(r'`([^`\n]+)`', text))))

    def code_like(name):  # is_gen_callable, getAttr, `prepare`; not a plain word the reporter's code calls (array)
        return '_' in name.strip('_') or bool(re.search(r'[a-z0-9][A-Z]', name)) or name in ticked

    def first_pos(*words):
        pos = [m.start() for w in words if w for m in [re.search(r'(?<!\w)' + re.escape(w) + r'\b', text)] if m]
        return min(pos) if pos else len(text)

    out = collections.defaultdict(list)
    for path, qn, name, start, end in rows:
        full = '.' + (kb.module_name(path) or '') + '.' + qn  # .rich.console.Console.is_terminal
        hit = next((d for d in dotted if ('.' + qn).endswith('.' + d)  # Console.is_terminal, pkg.mod.Cls.meth
                    or ('.' in qn and ('.' + d).endswith('.' + qn))
                    or full.endswith('.' + d)), None)  # v19: console.is_terminal, jax.random.fold_in
        if hit:
            out[path].append((0, first_pos(hit), (path, qn, start, end)))
        elif name in idents and len(nfiles[name]) <= NAMED_MAX_FILES and code_like(name):
            out[path].append((1, first_pos(name), (path, qn, start, end)))
    q['named_kind'] = {(p, x[2][1]): x[0] for p, v in out.items() for x in v}
    return {p: [fn for *_, fn in sorted(v, key=lambda x: (x[0], x[1], x[2][2]))] for p, v in out.items()}


def path_matches(mentioned, path):
    """A path the issue mentions (relative, absolute, ./x, a bare file name) names this repository path."""
    m = mentioned.replace('\\', '/')
    while m.startswith('./'):
        m = m[2:]
    return m == path or m.endswith('/' + path) or path.endswith('/' + m)


def path_scores(paths, qtokens, k1=1.2, b=0.75):
    """{path: BM25 of the query over the path's last PATH_PARTS components} normalized to max 1: the file and package
    names (rich/file_proxy.py for "proxy isatty"; __init__.py stands for its package). Rare components weigh most."""
    docs = {}
    for p in paths:
        parts = p[:-3].split('/')
        if parts[-1] == '__init__' and len(parts) > 1:
            parts = parts[:-1]
        docs[p] = tokenize(' '.join(parts[-PATH_PARTS:]))
    df = collections.Counter(t for toks in docs.values() for t in set(toks))
    n = len(docs)
    avg = sum(len(t) for t in docs.values()) / max(1, n) or 1
    qs = set(qtokens)
    out = {}
    for p, toks in docs.items():
        tf = collections.Counter(t for t in toks if t in qs)
        if tf:
            out[p] = sum(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * f * (k1 + 1)
                         / (f + k1 * (1 - b + b * len(toks) / avg)) for t, f in tf.items())
    mx = max(out.values(), default=0) or 1
    return {p: v / mx for p, v in out.items()}


def rank(root, text, conn=None, history=True):
    """(ranked paths, best_fn {path: [(score, (path, qualname, start, end))]}, reasons, parsed query, score). Each
    file's score is the sum of its weighted signals (q['features'][path]: {signal: weighted value}), times
    AUX_FACTOR for tests/docs/examples/vendored code."""
    q = parse_issue(text)
    conn = conn or kb.open(root)
    files = conn.execute('SELECT id, path FROM files WHERE indexed = 1 ORDER BY path').fetchall()
    paths = [p for _, p in files]
    if not paths:
        return [], {}, {}, q, {}
    score = dict.fromkeys(paths, 0.0)
    feat = collections.defaultdict(dict)
    reasons = collections.defaultdict(list)

    def add(p, name, value, why=None):
        if p in score:
            score[p] += value
            feat[p][name] = feat[p].get(name, 0.0) + value
            if why:
                reasons[p].append(why)

    fscore, _ = kb.bm25(conn, 'files', q['tokens'])
    mx = max(fscore.values(), default=0) or 1
    for i, p in files:
        if fscore.get(i):
            add(p, 'bm25', fscore[i] / mx)
    fnscore, _ = kb.bm25(conn, 'defs', q['tokens'])
    fmx = max(fnscore.values(), default=0) or 1
    fold = {}  # v19: (path, qualname) of a def nested in a def -> its outermost enclosing def (path, qn, start, end):
    best = {}  # ranked, listed and previewed as that def (the unit the model reads and edits; toolbench's gold too)
    syms = {sid: (parent, kind, path, qn, start, end) for sid, parent, kind, path, qn, start, end in conn.execute(
        'SELECT id, parent, kind, path, qualname, start, "end" FROM symbols')}
    for sid, (parent, kind, path, qn, start, end) in syms.items():
        if kind == 'class':
            continue
        fn, up = (path, qn, start, end), parent
        while up is not None and up in syms:
            u = syms[up]
            if u[1] != 'class':
                fn = (u[2], u[3], u[4], u[5])
            up = u[0]
        if fn[1] != qn:
            fold[(path, qn)] = fn
        best[fn] = max(best.get(fn, 0.0), fnscore.get(sid, 0.0) / fmx)  # the outer def: its best member's score
    best_fn = collections.defaultdict(list)
    for fn, sc in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        best_fn[fn[0]].append((sc, fn))
    q['fold'] = fold
    for p in paths:
        best_fn[p].sort(key=lambda x: -x[0])
        if best_fn[p] and best_fn[p][0][0]:
            add(p, 'fn', FN_WEIGHT * best_fn[p][0][0])
    for p, v in path_scores(paths, q['tokens']).items():
        add(p, 'path', PATH_WEIGHT * v)
    if q['idents']:
        idents = sorted(q['idents'])
        ph = ','.join('?' * len(idents))
        defs = collections.defaultdict(set)
        for name, path in conn.execute(f'SELECT DISTINCT name, path FROM symbols WHERE name IN ({ph})', idents):
            defs[name].add(path)
        for name, ps in defs.items():
            if name.startswith('__') and name.endswith('__'):  # __init__ (from __init__.py): defined everywhere
                continue
            for p in ps:  # the evidence names only a rare definition (main, run in 30 files: a weight, no reason)
                add(p, 'defines', DEFINES_WEIGHT / len(ps), 'defines ' + name if len(ps) <= NAMED_MAX_FILES else None)
    for qp in q['paths']:  # a bare file name (main.py, __init__.py) mentioned counts less the more files have it
        hits = [p for p in paths if path_matches(qp, p)]
        for p in hits:
            add(p, 'mention', MENTION_WEIGHT / len(hits), 'path mentioned')
    dotted = [d for d in q['dotted'] if d.rsplit('.', 1)[-1].lower() not in FILE_EXTS]  # not meson.build, setup.py
    for p in paths:
        mod = p[:-3].replace('/', '.')
        for d in dotted:
            if mod.endswith(d) or (d.count('.') >= 1 and ('.' + d).startswith('.' + mod.split('.src.')[-1] + '.')):
                add(p, 'module', 1.0, 'module ' + d)  # once per file: jax.random.fold_in, jax.random.PRNGKey
                break
    for fp, ln, fn in q['frames']:
        for p in paths:
            if path_matches(fp, p):
                add(p, 'frame', 1.5, f'traceback {fn}:{ln}')
    byid = dict(files)
    seen = set()
    for s in q['strings'] + q['quoted']:  # message text the code contains as a string literal
        toks = tokenize(s)[:STRING_TOKENS]
        if len(toks) < 2 or tuple(toks) in seen:
            continue
        seen.add(tuple(toks))
        try:
            hits = [byid.get(i) for (i,) in conn.execute('SELECT rowid FROM fts_files WHERE fts_files MATCH ?',
                                                        ('strings : "' + ' '.join(toks) + '"',))]
        except Exception:
            hits = []
        hits = [p for p in hits if p in score]
        for p in hits if len(hits) <= STRING_MAX_FILES else ():
            add(p, 'error_text', 1.0 / len(hits), 'contains error text')
    for p, (name, fn) in title_defs(conn, text, set(paths)).items():  # v16: "padding width" -> _get_padding_width
        fn = fold.get(fn[:2], fn)
        add(p, 'title', TITLE_WEIGHT, 'title names ' + name)
        top = best_fn[p][0][0] if best_fn[p] else 0.0
        best_fn[p] = [(top + 0.01, fn)] + [x for x in best_fn[p] if x[1][1] != fn[1]]
    q['named'] = named_functions(conn, text, q, set(paths))  # v18 P1: named functions first within their file
    for p, fns in q['named'].items():  # v19: a nested def stands for its outer def (the unit the model reads, edits)
        out = []
        for fn in fns:
            f = fold.get(fn[:2], fn)
            if f[1] != fn[1]:
                kinds = q.setdefault('named_kind', {})
                kinds[(p, f[1])] = min(kinds.get((p, f[1]), 1), kinds.get((p, fn[1]), 1))
            if f not in out:
                out.append(f)
        q['named'][p] = out
    for p, fns in q['named'].items():
        keys = {fn[1] for fn in fns}
        top = best_fn[p][0][0] if best_fn[p] else 0.0
        own = {x[1][1]: x[0] for x in best_fn[p]}  # v19: dotted names first, then the most similar to the issue
        fns = sorted(fns, key=lambda fn: (q.get('named_kind', {}).get((p, fn[1]), 1), -own.get(fn[1], 0.0)))
        best_fn[p] = [(top + 0.02, fn) for fn in fns] + [x for x in best_fn[p] if x[1][1] not in keys]
    q['history'] = {}
    if history:
        hist, evidence = history_scores(conn, text, set(paths))
        for p, h in hist.items():
            if h > 0:
                add(p, 'history', HISTORY_WEIGHT * h, 'similar past commits' if h >= 0.5 else None)
        q['history'] = evidence
    # not demoted: a non-test file the issue mentions by path (setup.py: ...), docs/examples when the title is about
    # them ("Update docs ...", a gitmoji memo)
    docs = re.search(r'\b(docs?|documentation|examples?|tutorials?)\b|\U0001f4dd', text.strip().split('\n', 1)[0], re.I)
    for p in paths:
        if is_aux(p) and not (not kb.is_test(p) and ('mention' in feat[p] or docs and DOCS_DIR.search(p))):
            score[p] *= AUX_FACTOR
            feat[p]['aux'] = AUX_FACTOR
    q['features'] = feat
    return sorted(paths, key=lambda p: -score[p]), best_fn, reasons, q, score


def tests_hint(conn, path):
    tests = kb.tests_for(path, conn=conn, k=6)
    if not tests:
        return None
    return 'tests: ' + argfix.obj('run.py', {'test': tests[0][0]})


def low_confidence(ranked, score):
    """The top two files score close (or nothing scored at all)."""
    if not ranked:
        return True
    s1, s2 = score[ranked[0]], score[ranked[1]] if len(ranked) > 1 else 0
    return bool(s2 and s1 / s2 < LOW_CONF_RATIO) or s1 <= 0


def pick_best(ranked, best_fn, q, low):
    """(best function (path, qualname, start, end) or None, confident). The best function is the top file's first;
    it is previewed only when confident: the top file clearly leads (not low) and the issue names the function or it
    clearly leads the file's next one (FN_MARGIN). On 421 real first-locate queries (E040/E041) the preview was a
    gold function in 38% of runs when always printed; with the v18 gate it was printed in ~49% and right in ~58%
    (named: 77%). v19: a named function in a top file that does not clearly lead is no longer previewed (external
    tasks: right in 17% of those previews; overall preview precision 42% -> 47%)."""
    top = ranked[0] if ranked else None
    if not top or not best_fn.get(top):
        return None, False
    fns = best_fn[top]
    fn = fns[0][1]
    if any(fn[1] == f[1] for f in q.get('named', {}).get(top, ())):
        return fn, not low
    f2 = fns[1][0] if len(fns) > 1 else 0.0
    return fn, bool(not low and fns[0][0] > 0 and fns[0][0] >= FN_MARGIN * f2)


def listed(ranked, best_fn, q, k):
    """[(path, [fn, ...])] for the top k files: the functions the issue names first (up to NAMED_PER_FILE), then the
    best similarity matches up to FNS_PER_FILE."""
    out = []
    for p in ranked[:k]:
        n = len(q.get('named', {}).get(p, ()))
        out.append((p, [f for _, f in best_fn[p][:max(FNS_PER_FILE, min(n, NAMED_PER_FILE))]]))
    return out


def evidence(reasons):
    """The bracket after a file: 'defines X' for one issue name, 'defines N issue names' for several."""
    rs = list(dict.fromkeys(reasons))
    defs = [r for r in rs if r.startswith('defines ')]
    rest = [r for r in rs if not r.startswith('defines ')]
    head = defs if len(defs) <= 1 else [f'defines {len(defs)} issue names']
    return '; '.join(head + rest)[:90]


def _terms(q):
    """Lower-case identifiers and words of the issue that mark the interesting lines of a long function."""
    idents = {i.lower() for i in q.get('idents', ())}
    words = {t for t in q.get('tokens', ()) if len(t) > 2}
    return idents, words


def _line_weight(line, idents, words):
    code = line.split('#', 1)[0] if '#' in line and '"' not in line and "'" not in line else line
    w = 0
    for m in set(IDENT_RE.findall(code)):
        low = m.lower()
        if low in idents:
            w += 3
        elif low in words or any(s in words for s in subtoks(m) if len(s) > 2):
            w += 1
    return w


def _def_node(tree, start):
    for node in ast.walk(tree) if tree is not None else ():
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.lineno == start:
            return node
    return None


def preview_lines(lines, tree, start, end, q, limit=PREVIEW_LINES):
    """[(line number or None, text)] of the best function in at most `limit` lines: the whole function when it fits
    (docstring and a long signature folded), else the def line plus the window of body lines that mention the issue's
    terms most. Markers (number None) say what is left out and where."""
    end = min(end, len(lines))
    node = _def_node(tree, start)
    body0 = node.body[0] if node is not None and node.body else None
    first_body = body0.lineno if body0 is not None and body0.lineno > start else start + 1
    sig_end = max(start, first_body - 1)
    while sig_end > start and not lines[sig_end - 1].strip():
        sig_end -= 1
    doc = None
    if body0 is not None and isinstance(body0, ast.Expr) and isinstance(getattr(body0, 'value', None), ast.Constant) \
            and isinstance(body0.value.value, str) and body0.lineno > start:
        doc = (body0.lineno, body0.end_lineno or body0.lineno)
    head = []
    if sig_end - start + 1 > SIG_LINES:
        head.append((start, lines[start - 1]))
        head.append((None, f'... (L{start + 1}-{sig_end}: signature)'))
    else:
        head += [(i, lines[i - 1]) for i in range(start, sig_end + 1)]
    body_from = sig_end + 1
    if doc:  # part of the head; 3+ lines: one marker with the docstring's first line
        head += [(i, lines[i - 1]) for i in range(body_from, doc[0])]
        if doc[1] - doc[0] >= 2:
            first = next((s.strip() for s in (body0.value.value or '').strip().split('\n') if s.strip()), '')
            first = first[:80] + ('...' if len(first) > 80 else '')
            head.append((None, f'... (L{doc[0]}-{doc[1]}: docstring "{first}")'))
        else:
            head += [(i, lines[i - 1]) for i in range(doc[0], doc[1] + 1)]
        body_from = doc[1] + 1
    body = list(range(body_from, end + 1))
    if len(head) + len(body) <= limit:
        return head + [(i, lines[i - 1]) for i in body]
    room = limit - len(head) - 2  # markers before and after the window
    idents, words = _terms(q)
    weights = [_line_weight(lines[i - 1], idents, words) for i in body]
    best, best_w = 0, -1
    run = sum(weights[:room])
    for j in range(len(body) - room + 1):
        if j:
            run += weights[j + room - 1] - weights[j - 1]
        if run > best_w:
            best, best_w = j, run
    if best_w <= 0:  # nothing mentions the issue: the start of the body
        best = 0
    a, b = body[best], body[min(best + room, len(body)) - 1]

    def total(a, b):
        return len(head) + (a > body_from) + (b - a + 1) + (b < end)

    while True:  # a marker for one line costs that line: print the line; fill the lines a missing marker frees
        if a - body_from == 1:
            a = body_from
        if end - b == 1:
            b = end
        if total(a, b) >= limit:
            break
        if b < end:
            b += 1
        elif a > body_from:
            a -= 1
        else:
            break
    out = list(head)
    if a > body_from:
        out.append((None, f'... (L{body_from}-{a - 1})'))
    out += [(i, lines[i - 1]) for i in range(a, b + 1)]
    if b < end:
        out.append((None, f'... (L{b + 1}-{end})'))
    return out


USAGE = '\n'.join(__doc__.strip().splitlines()[2:4])


def parse_object(d):
    """The object form: {"query"} (+ "k", "show"). Returns (k, show, text)."""
    argfix.require(d, 'locate.py', 'query')
    k = K_DEFAULT
    if d.get('k', '').strip():
        n = argfix.line_number(d['k'])
        if n is None:
            raise argfix.Misuse(f'"k" must be a number, got {d["k"]!r}. E.g. '
                                + argfix.obj('locate.py', {'query': d['query'][:60], 'k': '5'}))
        k = max(1, min(10, n))
    return k, argfix.flag(d, 'show', 'locate.py') if 'show' in d else True, d['query']


def main(argv):
    try:
        d, note = argfix.object_args(argv, 'locate.py')
        if d is not None:
            k, show, query = parse_object(d)
            if note:
                print(f'NOTE: {note}')
            return locate(k, show, [query])
    except argfix.Misuse as e:
        print(f'USAGE ERROR: {e}')
        print(USAGE)
        return 2
    k, show = K_DEFAULT, True
    while argv and argv[0] in ('--k', '--no-show', '--show'):
        if argv[0] == '--k':
            try:
                k = max(1, min(10, int(argv[1])))
            except (IndexError, ValueError):
                print('usage: ' + USAGE.split('usage: ', 1)[-1])
                return 2
            argv = argv[2:]
        else:
            show, argv = argv[0] == '--show', argv[1:]
    return locate(k, show, argv)


def locate(k, show, argv):
    # Arguments may arrive with literal escape sequences; they only add noise to the query.
    text = ' '.join(argv).replace('\\n', '\n').replace('\\"', '"')
    if not text.strip():
        print('usage: ' + USAGE.split('usage: ', 1)[-1])
        return 2
    conn = kb.open(ROOT)
    ranked, best_fn, reasons, q, score = rank(ROOT, text, conn=conn)
    if not ranked:
        print('no Python files found under /workspace')
        return 1
    cands = []
    for p, fns in listed(ranked, best_fn, q, k):
        why = evidence(reasons[p])
        print(f'{score[p]:.2f} {p}' + (f'  [{why}]' if why else ''))
        for f in fns:
            print(f'      {f[1]}  L{f[2]}-{f[3]}')
            cands.append((p, f[1], f[2], f[3], round(score[p], 3), why))
        if not fns:
            cands.append((p, '', None, None, round(score[p], 3), why))
    low = low_confidence(ranked, score)
    meta = []
    if low:
        meta.append('low confidence: the top files score close')
    hist = [p for p in ranked[:k] if p in q['history']]
    if hist:
        subj = q['history'][hist[0]]
        meta.append(f'past commit "{subj[:50]}{"..." if len(subj) > 50 else ""}" touched {hist[0]}')
    hint = tests_hint(conn, ranked[0])
    if hint:
        meta.append(hint)  # last: a call, so nothing follows its value
    if meta:
        print(' | '.join(meta))
    ids = len(q['idents'])
    flow.fact(candidates=cands, identifier_count=ids, low_confidence=low,
              thin=bool(q['thin'] and (ids == 0 or (low and ids <= 1))))
    best, confident = pick_best(ranked, best_fn, q, low)
    if not (show and best and confident):
        return 0
    rel, qn, start, end = best
    try:
        with open(os.path.join(ROOT, rel), encoding='utf-8', errors='replace') as fh:
            src = fh.read()
    except OSError:
        return 0
    lines = src.split('\n')  # ast line numbering
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        tree = None
    rows = preview_lines(lines, tree, start, end, q)
    nums = [i for i, _ in rows if i]
    if not nums:
        return 0
    named = any(qn == f[1] for f in q.get('named', {}).get(rel, ()))
    print(f'--- best match: {rel} {qn} L{start}-{end}' + (' (named in the issue)' if named else ''))
    for i, line in rows:
        print(f'{i:5d}  {line}' if i else f'       {line}')
    # the last run of consecutive printed lines: the code the model has read in full (flow: viewed, edit early); the
    # whole function (start-end) only when nothing was folded
    segs = [[]]
    for i, _ in rows:
        if i is None:
            segs.append([])
        else:
            segs[-1].append(i)
    seg = [s for s in segs if s][-1]
    flow.fact(shown=(rel, seg[0], seg[-1]))
    return 0


if __name__ == '__main__':
    sys.exit(flow.main('locate.py', main))
