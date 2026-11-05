"""toolbench: an LLM-free benchmark of the agent's repo-tools scripts (how well their output guides toward the fix).

Purpose: decide when the tools' model-facing format can be frozen (before a LoRA is trained on it). Minutes, CPU, no
model. Every script runs as ADK's run_skill_script runs it (argv `--key str(value)`, runpy from a temp cwd with a
scripts/ symlink, WORKSPACE / REPO_TOOLS_DIR, flow state on) on a simulated clock (call n at T0 + n x --call-seconds;
the flow's BUDGET clock sees Kaggle-like elapsed time), with the interpreter --python (default python3.13 like the
sandbox; no repository dependencies are installed, so import guards / run.py see a bare environment: env-dependent
results are marked, run.py tests are never run).

Task sources
  local     data/competition/tasks.jsonl + snapshots/<task>.tgz (default list experiments/eval113.txt); each job extracts
            the snapshot to a temp dir and sets up git like the harness (replay_tools.prepare).
  external  SWE-bench train sample (experiments/toolbench_external.txt, built by scripts/toolbench_data.py select/setup;
            cached workspaces data/toolbench/ws/<id>, reset to the baseline commit before and after each job).
Gold = the .py non-test files the gold patch modifies, its hunks (the changed span of each hunk, in base line numbers)
and the functions containing changed lines (Class.method; nested defs fold into their outer def). New-file-only gold
patches are skipped (as eval_locate.py).

Query (what the agent sends as locate.py query=, system.md: "the issue title verbatim + every identifier, path and error
message in the statement"): build_query(): the statement's first line, then .py paths, error lines (`KeyError: 'x'`,
the last line of a traceback), then identifiers in order of appearance (backticked names, dotted names, CamelCase,
snake_case with an underscore, names called with `(`), URLs removed, Python keywords/builtins dropped, at most
QUERY_IDENTS identifiers / QUERY_CHARS characters.

Phases, per task x agent (all phases start from the same locate call; its flow state + KB are copied):
  1. locate: gold file rank among the printed files (@1/@3/printed), gold function in the printed list, best match
     (`--- best match:` preview) = gold function, output size.
  2. NEXT-follower bot: after locate, repeatedly run the first call of the first NEXT line that has one (notation:
     script + two-space key=value pairs, tests/notation.py), up to BOT_MAX_CALLS calls including locate. Templates are
     filled by fill_call() (target -> the file in focus, symbol -> its locate function, snippet -> a print that
     reproduces nothing, hypothesis texts -> a guess naming the focus function, ...). Stops at an edit.py/create.py
     suggestion (never applied; its range is checked against the gold spans), check_patch.py, a run.py test (env),
     an unfillable template or no NEXT call. Metrics: a gold span (+-SHOW_PAD lines) printed as numbered source lines
     (show.py / locate preview / edit context), at which call and estimated prompt, whether the estimated prompt had
     crossed the compaction threshold by then (BASE_TOKENS + sum(PER_CALL_TOKENS + json chars / 3.2)), the first edit
     suggestion's range vs the gold spans.
  2b. Explore bot (same, but edit/create suggestions are skipped; the first one is judged): it follows the first
     other NEXT call (the read suggestions), i.e. do the footers' reads lead to the gold code when locate missed it?
  3. Oracle: locate -> show.py symbol= of each gold function (<= 3; a range read for spans outside functions or not
     printed) -> edit.py line-range form per gold span (bottom-up, so base numbers stay valid; pure insertions and
     deletions are anchored on a neighbouring line because NEW cannot be empty) -> check_patch.py. Metrics: calls,
     tokens, edits applied first try, final .py files == gold (exact / modulo trailing whitespace), extra changes.
  4. Edit robustness: for <= ROBUST_SPANS gold spans, after a show of the span, the model's typical mistakes (notes
     12/13, edit.py tests): stale_lines (a 2-line insertion above the function first, then the span with the old
     numbers), start_only (multi-line NEW without end), json_old / json_both (OLD, or OLD and NEW, as the JSON result
     showed them: \\n, \\", doubled backslashes), indent_plus4 / indent_minus4 (line-range NEW re-indented), line_prefix
     (show.py's `123  ` prefixes copied into NEW). Outcome: intended (the file is exactly the intended result), wrong
     (applied, but not the intended text: a silent corruption), refused with a ready repair call (an edit.py call in
     the output; templates filled with the mistaken call's own values) and whether EXECUTING it gives the intended
     result (repair_intended / repair_wrong / repair_refused), or refused without a repair. `equivalent`: the file
     differs from the intended text but parses to the same AST (indentation inside brackets): cosmetic. `noop`:
     reported as applied, nothing changed.
  5. Output cost: per script call raw chars, JSON chars as the model sees them, est. tokens (json chars / 3.2), footer
     share (BUDGET/NEXT/MEMO/note lines).

Usage
    .venv/bin/python scripts/toolbench.py --agent agent-v18d [--agent WORKTREE] [--source local,external] [--workers 4]
        [--tasks ID,ID | --tasks-file F] [--limit N] [--label NAME] [--queries recorded.json] [--call-seconds 20]
    .venv/bin/python scripts/toolbench.py --format-diff agent-v18d WORKTREE [--source ...]   # model-facing diff
    .venv/bin/python scripts/toolbench.py --report results/toolbench/<label>                # re-print the scorecard
--agent: a git ref | WORKTREE | dir (scripts/materialize.sh), LABEL=SPEC to name it; the first agent is the reference
for deltas. Outputs: results/toolbench/<label>/<agent>.jsonl (one record per task, with every call's output),
scorecard.txt, summary.json. --format-diff runs agent A's bot + oracle call sequences, replays exactly those calls
through B, and counts outputs whose text differs (clock, temp paths and timings normalised), body vs footer.
"""

import argparse
import ast
import collections
import concurrent.futures
import json
import keyword
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

COMPACT_AT = 14336         # the scorer's compaction threshold (HARNESS_README 7.2)
BASE_TOKENS = 5200         # base prompt (system + task + tree + tools), audit / LOG 2026-10-04
PER_CALL_TOKENS = 336      # per call before its output: call text, args, overhead (replay_tools K_CALLS)
CHARS_PER_TOKEN = 3.2
BOT_MAX_CALLS = 10
SHOW_PAD = 5
ROBUST_SPANS = 2
SHOW_FUNCS = 3
CALL_TIMEOUT = 120
T0 = 1_800_000_000.0       # simulated clock origin
QUERY_IDENTS, QUERY_CHARS = 25, 500
OK_OUTCOMES = ('intended', 'repair_intended', 'equivalent', 'repair_equivalent')
MISTAKES_V19 = ('stale_lines', 'start_only', 'json_old', 'json_both', 'indent_plus4', 'indent_minus4', 'line_prefix')
DRIFT = ('old_blank_drift', 'old_skip_line', 'old_wrong_line')   # v20: OLD with small drift (edit_failures cases)
MISTAKES = MISTAKES_V19 + DRIFT
DRIFT_CONTEXT = 3          # context lines OLD/NEW carry above and below the gold span in the drift mistakes
RESULTS = os.path.join(ROOT, 'results', 'toolbench')
TB = os.path.join(ROOT, 'data', 'toolbench')
SNAPSHOTS = os.path.join(ROOT, 'data', 'competition', 'snapshots')
TEST_PATH = re.compile(r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$')
BASELINE = '_swegemma_baseline'

# ------------------------------------------------------------------ query (pure)

STOP = set(keyword.kwlist) | {'self', 'cls', 'None', 'True', 'False', 'print', 'len', 'str', 'int', 'dict', 'list',
                              'set', 'tuple', 'type', 'object', 'e.g', 'i.e', 'etc', 'vs', 'pip', 'python', 'Python',
                              'http', 'https', 'www', 'com', 'org', 'github', 'README', 'TODO', 'PR', 'API'}
URL = re.compile(r'https?://\S+|www\.\S+')
CODE_SPAN = re.compile(r'`([^`\n]{1,80})`')
PY_PATH = re.compile(r'(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.py)\b')
ERROR_LINE = re.compile(r'\b([A-Z]\w*(?:Error|Exception|Warning|Exit|Interrupt))\b(?::[ \t]*([^\n]{0,100}))?')
IDENT = re.compile(r'(?<![\w.])([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(\()?')


def is_identifier_like(tok, called):
    """A token worth sending to locate: dotted, CamelCase, snake_case with '_', or called with '('."""
    if tok in STOP or len(tok) < 3 or tok.split('.')[0] in STOP:
        return False
    if '.' in tok:
        parts = tok.split('.')
        return all(parts) and not tok.endswith(('.com', '.org', '.io', '.html', '.md', '.txt', '.rst'))
    if '_' in tok.strip('_') or (tok.startswith('_') and len(tok) > 3):
        return True
    if re.match(r'^[A-Z][a-z0-9]+[A-Z]', tok) or re.match(r'^[a-z]+[A-Z]', tok):
        return True
    return bool(called)


def build_query(statement):
    """The locate.py query an agent following system.md would send (see the module docstring)."""
    text = URL.sub(' ', statement or '').replace('\r', '')
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    title = re.sub(r'^#+\s*', '', lines[0]) if lines else ''
    parts, seen = [title], set(re.findall(r'[\w.]+', title))

    def add(tok):
        if tok and tok not in seen:
            seen.add(tok)
            parts.append(tok)

    for m in PY_PATH.finditer(text):
        add(m.group(1))
    for m in ERROR_LINE.finditer(text):
        msg = (m.group(2) or '').strip()
        add(f'{m.group(1)}: {msg}'.strip(': ') if msg and len(msg) > 2 else m.group(1))
    idents = []
    for m in CODE_SPAN.finditer(text):
        for t in IDENT.finditer(m.group(1)):
            if is_identifier_like(t.group(1), True):
                idents.append(t.group(1))
    for m in IDENT.finditer(text):
        if is_identifier_like(m.group(1), m.group(2)):
            idents.append(m.group(1))
    n = 0
    for t in idents:
        if n >= QUERY_IDENTS:
            break
        if t not in seen:
            add(t)
            n += 1
    q = ' '.join(p for p in parts if p)
    q = re.sub(r'\s+', ' ', q).strip()
    return q[:QUERY_CHARS].rsplit(' ', 1)[0] if len(q) > QUERY_CHARS else q


# ------------------------------------------------------------------ gold (pure)

def parse_patch(patch):
    """{path: {'kind': added|deleted|modified, 'hunks': [[(tag, text, old_no, new_no)]]}} of a unified diff (with or
    without `diff --git` headers). old_no is the base line number of ' '/'-' lines (for '+' the next base line)."""
    files, cur, hunk, o, n = {}, None, None, 0, 0
    lines = patch.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('diff --git '):
            cur, hunk = None, None
        elif line.startswith('--- ') and i + 1 < len(lines) and lines[i + 1].startswith('+++ '):
            a, b = line[4:].split('\t')[0].strip(), lines[i + 1][4:].split('\t')[0].strip()
            path = (b[2:] if b.startswith('b/') else b) if b != '/dev/null' else (a[2:] if a.startswith('a/') else a)
            kind = 'added' if a == '/dev/null' else 'deleted' if b == '/dev/null' else 'modified'
            cur = files.setdefault(path, {'kind': kind, 'hunks': []})
            hunk = None
            i += 2
            continue
        else:
            m = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if m and cur is not None:
                o, n = int(m.group(1)), int(m.group(3))
                if m.group(2) == '0':
                    o += 1  # "-12,0": insertion after line 12
                hunk = []
                cur['hunks'].append(hunk)
            elif hunk is not None and line[:1] in (' ', '-', '+') and not line.startswith(('--- ', '+++ ')):
                tag = line[0]
                hunk.append((tag, line[1:], o, n))
                if tag != '+':
                    o += 1
                if tag != '-':
                    n += 1
            elif hunk is not None and line == '' and i + 1 < len(lines) and lines[i + 1][:1] in (' ', '-', '+', '@'):
                hunk.append((' ', '', o, n))  # a blank context line whose leading space was stripped
                o += 1
                n += 1
        i += 1
    return files


def spans_of(path, hunks):
    """The change span of each hunk: {file, a, b, old, new} with a..b the base lines replaced (b = a - 1: a pure
    insertion before base line a), old = those base lines, new = their replacement."""
    out = []
    for h in hunks:
        idx = [k for k, x in enumerate(h) if x[0] != ' ']
        if not idx:
            continue
        seg = h[idx[0]:idx[-1] + 1]
        old = [t for tag, t, _, _ in seg if tag != '+']
        new = [t for tag, t, _, _ in seg if tag != '-']
        a = seg[0][2]
        out.append({'file': path, 'a': a, 'b': a + len(old) - 1, 'old': old, 'new': new})
    return out


def apply_spans(text, spans):
    """TEXT with SPANS (same file, base numbering) applied bottom-up."""
    lines = text.split('\n')
    for s in sorted(spans, key=lambda s: -s['a']):
        lines[s['a'] - 1:s['b']] = s['new']
    return '\n'.join(lines)


def def_ranges(src):
    """[(start, end, qualname)] of every def (decorators included in start); nested defs fold into the outer def."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    out = []

    def visit(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                visit(ch, prefix + [ch.name])
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([d.lineno for d in ch.decorator_list] + [ch.lineno])
                out.append((start, ch.end_lineno, '.'.join(prefix + [ch.name]), ch.lineno))
            elif not isinstance(ch, ast.Lambda):
                visit(ch, prefix)
    visit(tree, [])
    return out


def enclosing(defs, line):
    """The innermost recorded def containing LINE: (start, end, qualname, def line) or None."""
    best = None
    for d in defs:
        if d[0] <= line <= d[1] and (best is None or d[1] - d[0] < best[1] - best[0]):
            best = d
    return best


def gold_of(patch, read):
    """Gold of a task: {'files': [...], 'spans': [...], 'functions': [[file, qual]], 'added': [...]} from its patch;
    read(path) -> base text or None. Only .py non-test files the patch modifies count."""
    pf = parse_patch(patch)
    g = {'files': [], 'spans': [], 'functions': [], 'added': [p for p, f in pf.items() if f['kind'] == 'added'],
         'unparsable': []}
    for path, f in pf.items():
        if f['kind'] != 'modified' or not path.endswith('.py') or TEST_PATH.search(path):
            continue
        src = read(path)
        if src is None:
            continue
        sp = spans_of(path, f['hunks'])
        if not sp:
            continue
        g['files'].append(path)
        try:
            ast.parse(src)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            g['unparsable'].append(path)  # e.g. Python 2 code in old SWE-bench commits
        defs = def_ranges(src)
        for s in sp:
            lines = list(range(s['a'], s['b'] + 1)) or [max(1, s['a'] - 1)]
            ds = [enclosing(defs, l) for l in lines]
            d0 = next((d for d in ds if d), None)
            s['def'] = list(d0) if d0 else None
            g['spans'].append(s)
            for d in ds:
                if d and [path, d[2]] not in g['functions']:
                    g['functions'].append([path, d[2]])
    return g


def span_region(s, pad=0):
    """(lo, hi) base lines of a span (an insertion covers its neighbours), widened by pad."""
    lo, hi = (s['a'], s['b']) if s['b'] >= s['a'] else (s['a'] - 1, s['a'])
    return max(1, lo - pad), hi + pad


# ------------------------------------------------------------------ output parsing (pure)

LOC_FILE = re.compile(r'^(\d+\.\d+) (\S+)')
LOC_FN = re.compile(r'^\s{4,}(\S+)  L(\d+)-(\d+)')
LOC_BEST = re.compile(r'^--- best match: (\S+) (\S+) L(\d+)-(\d+)')
NUMBERED = re.compile(r'^\s*(\d+)(?:  |\t|$)')
FOOTER = re.compile(r'^(BUDGET:|NEXT:|MEMO:|\[\d+/\d+|note: results are JSON-encoded)')


def parse_locate(out):
    files, fns, best = [], [], None
    cur = None
    for line in (out or '').splitlines():
        m = LOC_FILE.match(line)
        if m:
            cur = m.group(2)
            files.append(cur)
            continue
        m = LOC_BEST.match(line)
        if m:
            best = [m.group(1), m.group(2), int(m.group(3)), int(m.group(4))]
            continue
        m = LOC_FN.match(line)
        if m and cur:
            fns.append([cur, m.group(1), int(m.group(2)), int(m.group(3))])
    return {'files': files, 'functions': fns, 'best': best}


def shown_lines(out, is_file):
    """{path: set(line numbers)} printed as numbered source lines; the file is the last header line whose first token
    (or the token after `--- best match:` / `EDITED`) is a workspace file (is_file(path))."""
    shown, cur = collections.defaultdict(set), None
    for line in (out or '').splitlines():
        m = NUMBERED.match(line)
        if m and cur:
            shown[cur].add(int(m.group(1)))
            continue
        toks = line.split()
        if not toks:
            continue
        cands = [toks[0]]
        if line.startswith('--- best match:') and len(toks) > 3:
            cands.append(toks[3])
        if toks[0] in ('EDITED', 'REPLACED') and len(toks) > 1:
            cands.append(toks[1])
        for t in cands:
            t = t.rstrip(':,')
            if ('/' in t or t.endswith('.py')) and is_file(t):
                cur = t
                break
    return shown


def split_footer(out):
    """(body, footer) of a script output: the footer is every BUDGET/NEXT/MEMO/[n/N prefix/backslash-note line."""
    body, foot = [], []
    for line in (out or '').split('\n'):
        (foot if FOOTER.match(line) else body).append(line)
    return '\n'.join(body), '\n'.join(foot)


SCRIPTS = ('locate.py', 'show.py', 'outline.py', 'search.py', 'callers.py', 'tests_for.py', 'history.py', 'edit.py',
           'create.py', 'run.py', 'check_patch.py', 'hypothesis.py', 'status.py')
SCRIPT_RE = re.compile(r'(?<![\w/.-])(' + '|'.join(re.escape(s) for s in SCRIPTS) + r')(?![\w/])')
PAIR = re.compile(r'([a-z_0-9]+)=(.*)', re.S)
TEMPLATE = re.compile(r'<[^<>]{1,100}>')


def calls_in(text):
    """[(script, {key: value})] written in the call notation (tests/notation.py): a script token, then two-space
    separated key=value parts, stopping at the first part that is not key=value."""
    out = []
    for line in str(text or '').splitlines():
        pos = 0
        while True:
            m = SCRIPT_RE.search(line, pos)
            if not m:
                break
            args, end, rest = {}, m.end(), line[m.end():]
            if rest.startswith('  '):
                for part in rest[2:].split('  '):
                    kv = PAIR.fullmatch(part)
                    if not kv:
                        break
                    args[kv.group(1)] = kv.group(2)
                    end += 2 + len(part)
            out.append((m.group(1), args))
            pos = end
    return out


def next_lines(out):
    return [l for l in (out or '').splitlines() if l.startswith('NEXT:')]


def first_next_call(out, skip=()):
    """(script, args, line index among NEXT lines, the line) of the first call in the NEXT lines whose script is not
    in SKIP (default: the first NEXT line holding a call, its first call), else None."""
    for k, line in enumerate(next_lines(out)):
        for script, args in calls_in(line[5:]):
            if script not in skip:
                return script, args, k, line
    return None


def fill_call(script, args, ctx):
    """(args with templates filled, None) or (None, stop reason) for the bot (see the module docstring)."""
    if script in ('edit.py', 'create.py'):
        return None, 'edit_suggested' if script == 'edit.py' else 'create_suggested'
    if script == 'check_patch.py':
        return None, 'check_patch'
    if script == 'run.py' and 'test' in args:
        return None, 'run_test'
    out = {}
    for k, v in args.items():
        if not TEMPLATE.search(v):
            out[k] = v
            continue
        fn = ctx.get('focus_fn') or ''
        if k in ('target', 'file', 'files', 'within', 'path'):
            if not ctx.get('focus_file'):
                return None, 'unfillable'
            out[k] = ctx['focus_file']
        elif k == 'symbol':
            if not fn:
                return None, 'unfillable'
            out[k] = fn
        elif k == 'name':
            if not fn:
                return None, 'unfillable'
            out[k] = fn.split('.')[-1]
        elif k in ('text', 'regex', 'grep'):
            if not ctx.get('idents'):
                return None, 'unfillable'
            out[k] = ctx['idents'][0]
        elif k in ('query', 'similar'):
            out[k] = ctx.get('query') or 'issue'
        elif k == 'snippet':
            out[k] = 'print("toolbench: no repro")'
        elif k in ('add', 'add2', 'add3'):
            n = {'add': 0, 'add2': 1, 'add3': 2}[k]
            cand = (ctx.get('candidates') or [])[n:n + 1]
            where = f'{cand[0][0]}:{cand[0][1]}' if cand else f'{ctx.get("focus_file")}:{fn}'
            out[k] = f'{where} mishandles the case the issue describes'
        elif k == 'evidence':
            out[k] = 'toolbench: the output was inspected'
        else:
            return None, 'unfillable'
    return out, None


# ------------------------------------------------------------------ mistakes (pure)

def json_escape(text):
    return json.dumps(text, ensure_ascii=False)[1:-1]


def reindent(lines, delta):
    out = []
    for l in lines:
        if not l.strip():
            out.append(l)
        elif delta > 0:
            out.append(' ' * delta + l)
        else:
            out.append(l[-delta:] if l[:-delta].strip() == '' else l)
    return out


def anchored(span, base_lines):
    """(start, end, new lines) of a line-range edit that makes SPAN's change. Pure insertions and deletions include a
    neighbouring line (NEW cannot be empty), and a NEW ending in a blank line takes the following lines up to a
    non-blank one along (edit.py drops one trailing newline of NEW)."""
    a, b, new = span['a'], span['b'], list(span['new'])
    if b < a:  # insertion before base line a
        if a - 1 >= 1:
            a, b, new = a - 1, a - 1, [base_lines[a - 2]] + new
        else:
            a, b, new = a, a, new + [base_lines[a - 1]]
    elif not new:  # deletion of a..b
        if a - 1 >= 1:
            a, new = a - 1, [base_lines[a - 2]]
        elif b < len(base_lines):
            b, new = b + 1, [base_lines[b]]
    while new and not new[-1].strip() and b < len(base_lines) - 1:
        b += 1
        new = new + [base_lines[b - 1]]
    return a, b, new


def merge_spans(spans, base_lines):
    """One span covering SPANS (same file, ascending, base numbering): the lines between them kept."""
    a, pos, new = spans[0]['a'], spans[0]['a'], []
    for s in spans:
        new += base_lines[pos - 1:s['a'] - 1] + list(s['new'])
        pos = s['b'] + 1
    b = max(s['b'] for s in spans)
    return {'file': spans[0]['file'], 'a': a, 'b': b, 'old': base_lines[a - 1:b], 'new': new}


def same_code(x, y):
    """True when the two texts parse to the same AST (a cosmetic difference: indentation inside brackets, ...)."""
    try:
        return ast.dump(ast.parse(x)) == ast.dump(ast.parse(y))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def mistakes(span, base_lines, strings=frozenset()):
    """[(name, [edit.py args dicts in order], expected-text transform)] for one gold span. The transform maps the base
    text to the intended text. Pure insertions/deletions use the anchored line-range form. No indentation mistakes
    inside multi-line strings (STRINGS: their base lines), where indentation is content."""
    f = span['file']
    a, b, new = anchored(span, base_lines)
    new_text = '\n'.join(new)
    out = []
    goal = lambda text: apply_spans(text, [span])  # noqa: E731
    d = span.get('def')
    if d:
        ds = d[0]
        ind = re.match(r'\s*', base_lines[ds - 1]).group(0)
        shift = {'target': f, 'start': str(ds), 'end': str(ds),
                 'new': f'{ind}# toolbench: shifted 1\n{ind}# toolbench: shifted 2\n' + base_lines[ds - 1]}
        shift_span = {'file': f, 'a': ds, 'b': ds - 1, 'new': [f'{ind}# toolbench: shifted 1',
                                                               f'{ind}# toolbench: shifted 2']}
        if ds < a:
            out.append(('stale_lines', [shift, {'target': f, 'start': str(a), 'end': str(b), 'new': new_text}],
                        lambda text: apply_spans(text, [span, shift_span])))
    if b > a or len(new) > 1:
        out.append(('start_only', [{'target': f, 'start': str(a), 'new': new_text}], goal))
    if span['old']:
        old_text = '\n'.join(span['old'])
        if json_escape(old_text) != old_text:
            out.append(('json_old', [{'target': f, 'old': json_escape(old_text), 'new': '\n'.join(span['new'])}],
                        goal))
            if json_escape('\n'.join(span['new'])) != '\n'.join(span['new']):
                out.append(('json_both', [{'target': f, 'old': json_escape(old_text),
                                           'new': json_escape('\n'.join(span['new']))}], goal))
    if any(l.strip() for l in new) and not any(x in strings for x in range(a, b + 2)):
        out.append(('indent_plus4', [{'target': f, 'start': str(a), 'end': str(b),
                                      'new': '\n'.join(reindent(new, 4))}], goal))
        if all(l[:4] == '    ' for l in new if l.strip()):
            out.append(('indent_minus4', [{'target': f, 'start': str(a), 'end': str(b),
                                           'new': '\n'.join(reindent(new, -4))}], goal))
    out.append(('line_prefix', [{'target': f, 'start': str(a), 'end': str(b),
                                 'new': '\n'.join(f'{a + i}  {l}' for i, l in enumerate(new))}], goal))
    out += drift_mistakes(span, base_lines, goal)
    return out


def drift_mistakes(span, base_lines, goal):
    """v20: text-form edits whose OLD (and NEW, copied from it) drifted from the file in lines the change does not
    touch (E057-E061 failures, research/notes/15-edit-failures.md): OLD/NEW = the gold span with DRIFT_CONTEXT lines
    of context each side, then old_blank_drift: one blank run in the context one line longer; old_skip_line: one
    interior context line left out; old_wrong_line: one interior context line replaced by `pass` (an invented line).
    The intended result is the gold change; the drifted context line stays as the file has it."""
    f, a, b = span['file'], span['a'], span['b']
    lo, hi = max(1, a - DRIFT_CONTEXT), min(len(base_lines), max(a, b) + DRIFT_CONTEXT)
    above = base_lines[lo - 1:a - 1]
    below = base_lines[max(a - 1, b):hi]
    old = above + list(span['old']) + below
    new = above + list(span['new']) + below
    if sum(1 for l in old if l.strip()) < 4 or old == new:
        return []
    ctx = [i for i in range(1, len(above))] + [len(above) + len(span['old']) + i for i in range(len(below) - 1)]
    out = []
    blank = next((i for i in ctx if not old[i].strip() and old[i - 1].strip()), None)
    if blank is not None:
        k = blank - len(above) - len(span['old']) + len(above) + len(span['new']) if blank >= len(above) else blank
        o2, n2 = old[:blank] + [''] + old[blank:], new[:k] + [''] + new[k:]
        out.append(('old_blank_drift', [{'target': f, 'old': '\n'.join(o2), 'new': '\n'.join(n2)}], goal))
    inner = [i for i in ctx if old[i].strip() and 0 < i < len(old) - 1 and old[i].strip() not in ('', ')', ']', '}')]
    if inner:
        i = inner[len(inner) // 2]
        k = i - len(above) - len(span['old']) + len(above) + len(span['new']) if i >= len(above) else i
        out.append(('old_skip_line', [{'target': f, 'old': '\n'.join(old[:i] + old[i + 1:]),
                                       'new': '\n'.join(new[:k] + new[k + 1:])}], goal))
        ind = re.match(r'\s*', old[i]).group(0)
        if old[i].strip() != 'pass':
            out.append(('old_wrong_line', [{'target': f, 'old': '\n'.join(old[:i] + [ind + 'pass'] + old[i + 1:]),
                                            'new': '\n'.join(new[:k] + [ind + 'pass'] + new[k + 1:])}], goal))
    return out


# ------------------------------------------------------------------ normalisation for --format-diff (pure)

CLOCK = re.compile(r'\b\d+m\d\ds\b|\b\d+:\d\d/|\b\d+(?:\.\d+)?\s?s\b(?= (?:left|used|elapsed))')
TMP_PATH = re.compile(r'(?:/private)?/(?:var/folders|tmp)/[^\s\'"]+')
TIMING = re.compile(r'\b\d+\.\d+s\b|\bin \d+(?:\.\d+)? ?(?:ms|s)\b')


def normalise(text):
    return TIMING.sub('<t>', TMP_PATH.sub('<tmp>', CLOCK.sub('T', text or '')))


def est_tokens(json_chars):
    return json_chars / CHARS_PER_TOKEN


# ------------------------------------------------------------------ executing scripts

WRAPPER = r'''
import os, runpy, sys, time
_a, _b = os.environ.pop('REPLAY_NOW', ''), os.environ.pop('REPLAY_END', '')
if _a:
    _real = time.time
    _clk = [float(_a), _real()]
    time.time = lambda: _clk[0] + (_real() - _clk[1])
    if _b:
        sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
        try:
            import flow as _flow
            _after = _flow.after
            def _after_at_end(*a, **k):
                _clk[0], _clk[1] = float(_b), _real()
                return _after(*a, **k)
            _flow.after = _after_at_end
        except Exception:
            pass
        sys.path.pop(0)
sys.argv = %r
runpy.run_path(sys.argv[0], run_name='__main__')
'''


def default_python():
    for p in ('/opt/homebrew/bin/python3.13', shutil.which('python3.13') or ''):
        if p and os.path.exists(p):
            return p
    return sys.executable


class Session:
    """One flow session (state dir) of one agent on one workspace; call() runs a script as ADK does."""

    def __init__(self, agent, ws, state, python, call_seconds, start_index=1, hashseed='0'):
        self.agent, self.ws, self.state, self.python = agent, ws, state, python
        self.cs, self.i, self.hashseed = call_seconds, start_index - 1, hashseed

    def call(self, script, args, phase):
        self.i += 1
        now = T0 + self.cs * (self.i - 1)
        argv = [f'scripts/{script}'] + [x for k, v in args.items() for x in (f'--{k}', str(v))]
        env = dict(os.environ, WORKSPACE=self.ws, REPO_TOOLS_DIR=self.state, PYTHONDONTWRITEBYTECODE='1',
                   REPLAY_NOW=repr(now), REPLAY_END=repr(now + 1.0), GIT_TERMINAL_PROMPT='0')
        for k in ('REPO_TOOLS_FLOW', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'PYTHONHASHSEED'):
            env.pop(k, None)
        if self.hashseed != 'random':  # str hashing (set order) is randomised per process unless fixed
            env['PYTHONHASHSEED'] = str(self.hashseed)
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix='adk_') as td:
            os.symlink(self.agent['scripts'], os.path.join(td, 'scripts'))
            try:
                r = subprocess.run([self.python, '-c', WRAPPER % (argv,)], cwd=td, env=env, capture_output=True,
                                   text=True, timeout=CALL_TIMEOUT, errors='replace')
                out, err, code = r.stdout, r.stderr, r.returncode
            except subprocess.TimeoutExpired:
                out, err, code = '', f'Command timed out after {CALL_TIMEOUT}s (exit_code=124)', 'timeout'
        ws = self.ws.rstrip('/')
        fix = lambda s: s.replace(ws + '/', '/workspace/').replace(ws, '/workspace')  # noqa: E731
        out, err = fix(out), fix(err)
        res = {'skill_name': self.agent['skill'], 'file_path': script, 'stdout': out, 'stderr': err,
               'status': 'success' if not err else ('error' if not out else 'warning')}
        jc = len(json.dumps(res))
        body, foot = split_footer(out)
        rec = {'phase': phase, 'i': self.i, 'script': script, 'args': args, 'code': code,
               'secs': round(time.time() - t0, 2), 'chars': len(out) + len(err), 'json_chars': jc,
               'arg_chars': len(json.dumps(args, ensure_ascii=False)), 'tokens': round(est_tokens(jc), 1),
               'footer_chars': len(foot), 'out': out, 'err': err[-2000:]}
        if 'Traceback (most recent call last)' in err:
            rec['crash'] = True
        return rec


def cp_tree(src, dst):
    """Copy-on-write copy of a directory (APFS clonefile via cp -c; falls back to a plain copy)."""
    if subprocess.run(['cp', '-cR', src, dst], capture_output=True).returncode:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=True)


def git(ws, *a, check=False):
    r = subprocess.run(['git', '-c', 'core.quotepath=false', '-C', ws, *a], capture_output=True, text=True,
                       errors='replace')
    if check and r.returncode:
        raise RuntimeError(f'git {a}: {r.stderr[-300:]}')
    return r.stdout


def reset_ws(ws):
    git(ws, 'reset', '-q', '--hard', BASELINE)
    git(ws, 'clean', '-qfd')


def read_file(ws, rel):
    try:
        with open(os.path.join(ws, rel), encoding='utf-8', errors='replace', newline='') as f:
            return f.read()
    except OSError:
        return None


def changed_files(ws):
    git(ws, 'add', '-N', '.')
    return [l for l in git(ws, 'diff', '--name-only', BASELINE).splitlines() if l]


def edit_status(out):
    t = (out or '').lstrip()
    head = t.split('\n--- a/')[0]
    if t.startswith('ALREADY DONE'):
        return 'already_done'
    if re.search(r'^(EDITED|CREATED|REPLACED) ', head, re.M):
        return 'applied'
    if 'USAGE ERROR' in t[:200]:
        return 'usage'
    if re.search(r'NOT APPLIED|NOT CREATED|REFUSED|NOT RUN', t[:800]):
        return 'refused'
    return 'other'


def repair_call(out):
    """The first ready edit.py call (with a start or old value) printed in an edit result, else None."""
    for sc, a in calls_in(out):
        if sc == 'edit.py' and ('start' in a or 'old' in a):
            return a
    return None


def fill_repair(args, mistaken, path):
    """A repair call as the model would send it: templates (<your NEW unchanged>, ...) filled with the mistaken
    call's own values; the target defaults to the file."""
    out = dict(args)
    for k, v in args.items():
        if TEMPLATE.search(v) and k in ('new', 'old', 'target'):
            out[k] = mistaken.get(k, v)
    if 'target' not in out and 'file' not in out:
        out['target'] = path
    return out


def string_lines(src):
    """Base lines inside a multi-line string literal (indentation there is content, not code)."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.end_lineno and n.end_lineno > n.lineno:
            out.update(range(n.lineno + 1, n.end_lineno + 1))
    return out


# ------------------------------------------------------------------ phases

class TaskRun:
    def __init__(self, task, agent, ws, opt, workdir):
        self.task, self.agent, self.ws, self.opt, self.workdir = task, agent, ws, opt, workdir
        self.gold = task['gold']
        self.calls = []
        self._n = 0
        self.base = {f: read_file(ws, f) for f in self.gold['files']}

    def new_state(self, src=None):
        self._n += 1
        d = os.path.join(self.workdir, f'st_{self.agent["label"]}_{self._n}')
        if src:
            cp_tree(src, d)
        else:
            os.makedirs(d)
        return d

    def session(self, state, start_index):
        return Session(self.agent, self.ws, state, self.opt['python'], self.opt['call_seconds'], start_index,
                       self.opt.get('hashseed', '0'))

    def is_file(self, rel):
        return os.path.isfile(os.path.join(self.ws, rel))

    def gold_hit(self, out):
        """Gold spans (indices) printed as numbered lines (+-SHOW_PAD) in OUT."""
        sh = shown_lines(out, self.is_file)
        hit = set()
        for k, s in enumerate(self.gold['spans']):
            lo, hi = span_region(s, SHOW_PAD)
            if any(lo <= x <= hi for x in sh.get(s['file'], ())):
                hit.add(k)
        return hit

    def record(self, rec):
        self.calls.append(rec)
        return rec

    # 1. locate
    def locate(self):
        reset_ws(self.ws)
        st = self.new_state()
        rec = self.record(self.session(st, 1).call('locate.py', {'query': self.task['query']}, 'locate'))
        self.template = st
        p = parse_locate(rec['out'])
        g = self.gold
        gfiles = set(g['files'])
        gfn = {tuple(x) for x in g['functions']}
        gfn_short = {(f, q.split('.')[-1]) for f, q in gfn}
        rank = next((k + 1 for k, f in enumerate(p['files']) if f in gfiles), None)
        in_list = any((f, q) in gfn or (f, q.split('.')[-1]) in gfn_short for f, q, _, _ in p['functions'])
        best = p['best']
        best_gold = bool(best and ((best[0], best[1]) in gfn or (best[0], best[1].split('.')[-1]) in gfn_short))
        self.loc = p
        return {'query': self.task['query'], 'rank': rank, 'printed': len(p['files']), 'fn_in_list': in_list,
                'has_fn': bool(gfn), 'best': best, 'best_gold': best_gold, 'best_shown': best is not None,
                'gold_shown': bool(self.gold_hit(rec['out'])), 'tokens': rec['tokens'], 'chars': rec['chars'],
                'secs': rec['secs']}

    def ctx(self, focus_file):
        fns = [x for x in self.loc['functions'] if x[0] == focus_file]
        idents = [t for t in re.findall(r'[A-Za-z_][\w.]*', self.task['query']) if is_identifier_like(t, False)]
        return {'focus_file': focus_file, 'focus_fn': fns[0][1] if fns else None, 'query': self.task['query'],
                'idents': idents, 'candidates': [(x[0], x[1]) for x in self.loc['functions']]}

    # 2. NEXT-follower bot
    def bot(self, explore=False):
        """explore=False: follow the first NEXT call, stop at an edit suggestion. explore=True: skip edit/create
        suggestions (the first one is judged) and follow the first other NEXT call: do the read suggestions lead to
        the gold code?"""
        phase = 'explore' if explore else 'bot'
        st = self.new_state(self.template)
        sess = self.session(st, 2)
        loc = self.calls[0]
        outs = [loc]
        focus = self.loc['files'][0] if self.loc['files'] else None
        res = {'stop': None, 'calls': 1, 'gold_at': None, 'gold_ctx': None, 'edit': None, 'path': ['locate.py'],
               'first_next': None, 'first_next_line': 0}
        ctx_tok = BASE_TOKENS + PER_CALL_TOKENS + loc['tokens']
        if self.gold_hit(loc['out']):
            res['gold_at'], res['gold_ctx'] = 1, round(ctx_tok)
        seen = set()
        while True:
            last = outs[-1]
            nx = first_next_call(last['out'])
            if explore and nx and nx[0] in ('edit.py', 'create.py'):
                if res['edit'] is None and nx[0] == 'edit.py':
                    res['edit'] = self.judge_edit(nx[1], len(outs), last['out'])
                if res['first_next'] is None:
                    res['first_next'], res['first_next_line'] = nx[0], nx[2]
                nx = first_next_call(last['out'], ('edit.py', 'create.py'))
            if nx is None:
                res['stop'] = 'submit' if any('submit_patch' in l for l in next_lines(last['out'])) else 'no_next'
                break
            script, args, k, line = nx
            if res['first_next'] is None:
                res['first_next'], res['first_next_line'] = script, k
            filled, stop = fill_call(script, args, self.ctx(focus))
            if stop:
                res['stop'] = stop
                if script == 'edit.py' and res['edit'] is None:
                    res['edit'] = self.judge_edit(args, len(outs), last['out'])
                break
            if len(outs) >= BOT_MAX_CALLS:
                res['stop'] = 'max_calls'
                break
            key = (script, json.dumps(filled, sort_keys=True))
            res['repeat'] = res.get('repeat', 0) + (key in seen)
            seen.add(key)
            rec = self.record(sess.call(script, filled, phase))
            outs.append(rec)
            res['path'].append(script)
            ctx_tok += PER_CALL_TOKENS + rec['tokens']
            if res['gold_at'] is None and self.gold_hit(rec['out']):
                res['gold_at'], res['gold_ctx'] = len(outs), round(ctx_tok)
            t = filled.get('target') or filled.get('files')
            if t and self.is_file(t):
                focus = t
        res['calls'] = len(outs)
        res['tokens'] = round(sum(o['tokens'] for o in outs))
        res['ctx_end'] = round(ctx_tok)
        res['crossed_before_gold'] = (res['gold_ctx'] or ctx_tok) >= COMPACT_AT if res['gold_at'] else None
        return res

    def judge_edit(self, args, at_call, last_out):
        """The suggested edit's file/range vs the gold spans. An old-text suggestion (old=<exact current lines>) is
        taken to target the lines of that file the suggesting output printed."""
        t = args.get('target') or args.get('file') or ''
        e = {'at': at_call, 'target': t, 'form': 'range' if 'start' in args else 'old', 'gold_file': t in self.gold[
            'files'], 'overlap': None, 'near': None}
        try:
            a = int(args['start'])
            b = int(args.get('end') or a)
        except (KeyError, ValueError):
            ls = shown_lines(last_out, self.is_file).get(t)
            if not ls:
                return e
            a, b = min(ls), max(ls)
            e['form'] = 'old (printed lines)'
        e['range'] = [a, b]
        sp = [s for s in self.gold['spans'] if s['file'] == t]
        e['overlap'] = any(a <= span_region(s)[1] and span_region(s)[0] <= b for s in sp)
        e['near'] = any(a <= span_region(s, SHOW_PAD)[1] and span_region(s, SHOW_PAD)[0] <= b for s in sp)
        return e

    # 3. oracle
    def oracle(self):
        reset_ws(self.ws)
        st = self.new_state(self.template)
        sess = self.session(st, 2)
        recs = [self.calls[0]]
        shown = collections.defaultdict(set)

        def note(rec):
            recs.append(self.record(rec))
            for f, ls in shown_lines(rec['out'], self.is_file).items():
                shown[f] |= ls

        for f, ls in shown_lines(recs[0]['out'], self.is_file).items():
            shown[f] |= ls
        for f, q in self.gold['functions'][:SHOW_FUNCS]:
            note(sess.call('show.py', {'target': f, 'symbol': q}, 'oracle'))
        extra = 0
        for s in self.gold['spans']:
            lo, hi = span_region(s)
            if not all(x in shown[s['file']] for x in range(lo, hi + 1)):
                extra += 1
                note(sess.call('show.py', {'target': s['file'], 'start': str(max(1, lo - SHOW_PAD)),
                                           'end': str(hi + SHOW_PAD)}, 'oracle'))
        first_try, anchors, statuses, merges = True, 0, [], 0
        for f in self.gold['files']:
            base_lines = self.base[f].split('\n')
            fs = sorted([s for s in self.gold['spans'] if s['file'] == f], key=lambda s: s['a'])
            i = len(fs) - 1
            while i >= 0:  # bottom-up; a refused edit is retried merged with the span above (an interdependent change)
                j = i
                while True:
                    span = fs[i] if j == i else merge_spans(fs[j:i + 1], base_lines)
                    a, b, new = anchored(span, base_lines)
                    anchors += (a, b) != (span['a'], span['b'])
                    rec = sess.call('edit.py', {'target': f, 'start': str(a), 'end': str(b), 'new': '\n'.join(new)},
                                    'oracle')
                    note(rec)
                    stt = edit_status(rec['out'])
                    statuses.append(stt)
                    first_try &= stt == 'applied'
                    if stt == 'applied' or j == 0:
                        break
                    j -= 1
                    merges += 1
                i = j - 1
        note(sess.call('check_patch.py', {}, 'oracle'))
        exact = ws_eq = equiv = True
        for f in self.gold['files']:
            want = apply_spans(self.base[f], [s for s in self.gold['spans'] if s['file'] == f])
            got = read_file(self.ws, f) or ''
            exact &= got == want
            ws_eq &= [l.rstrip() for l in got.split('\n')] == [l.rstrip() for l in want.split('\n')]
            equiv &= got == want or same_code(got, want)
        other = [p for p in changed_files(self.ws) if p not in self.gold['files']]
        cp = recs[-1]['out']
        res = {'calls': len(recs), 'tokens': round(sum(r['tokens'] for r in recs)),
               'ctx_end': round(BASE_TOKENS + sum(PER_CALL_TOKENS + r['tokens'] for r in recs)),
               'extra_reads': extra, 'anchored_edits': anchors, 'edits': len(statuses), 'edit_status': statuses,
               'first_try': first_try, 'merged_retries': merges, 'exact': exact, 'equal_ws': ws_eq,
               'same_code': equiv, 'other_files': other,
               'check_patch': (cp.strip().splitlines() or [''])[0][:160],
               'check_ready': bool(re.search(r'^READY\b', cp, re.M))}
        probs = [l.strip()[2:] for l in cp.splitlines() if l.startswith('  - ')]
        env = [x for x in probs if re.search(r'import \S+ fails: (ModuleNotFoundError|ImportError)', x)]
        res['check_problems'] = [x[:200] for x in probs if x not in env][:4]
        res['check_ok_ignoring_env'] = res['check_ready'] or bool(probs) and len(env) == len(probs)
        reset_ws(self.ws)
        return res

    # 4. edit robustness
    def robust(self):
        out = []
        spans = []
        for f in self.gold['files']:
            fs = [s for s in self.gold['spans'] if s['file'] == f]
            spans += ([s for s in fs if s['old'] and s['new']] or fs)[:1]
        for s in self.gold['spans']:
            if len(spans) >= ROBUST_SPANS:
                break
            if s not in spans:
                spans.append(s)
        for s in spans[:ROBUST_SPANS]:
            reset_ws(self.ws)
            base_text = self.base[s['file']]
            base_lines = base_text.split('\n')
            viewed = self.new_state(self.template)
            sess = self.session(viewed, 2)
            if s.get('def'):
                self.record(sess.call('show.py', {'target': s['file'], 'symbol': s['def'][2]}, 'robust'))
            lo, hi = span_region(s)
            if s.get('def') is None or not (s['def'][0] <= lo and hi <= s['def'][1]):
                self.record(sess.call('show.py', {'target': s['file'], 'start': str(max(1, lo - SHOW_PAD)),
                                                  'end': str(hi + SHOW_PAD)}, 'robust'))
            start_i = sess.i + 1
            for name, edits, goal in mistakes(s, base_lines, string_lines(base_text)):
                reset_ws(self.ws)
                st = self.new_state(viewed)
                ss = self.session(st, start_i)
                want = goal(base_text)
                r = {'mistake': name, 'span': [s['file'], s['a'], s['b']]}
                for e in edits[:-1]:
                    pre = self.record(ss.call('edit.py', e, 'robust'))
                    r['pre'] = edit_status(pre['out'])
                pre_text = read_file(self.ws, s['file'])
                rec = self.record(ss.call('edit.py', edits[-1], 'robust'))
                got = read_file(self.ws, s['file'])
                stt = edit_status(rec['out'])
                r['status'] = stt
                r['tokens'] = rec['tokens']
                r['first_line'] = (rec['out'].strip().splitlines() or [''])[0][:200]
                r['args'] = {k: (v if len(v) < 300 else v[:297] + '...') for k, v in edits[-1].items()}
                r['out_head'] = rec['out'][:1200]
                r['warned'] = 'WARNING' in rec['out']
                if got == want:
                    r['outcome'] = 'intended'
                elif got != pre_text:
                    r['outcome'] = 'equivalent' if same_code(got, want) else 'wrong'
                elif stt not in ('refused', 'usage'):
                    r['outcome'] = 'noop'  # reported as done, nothing changed
                else:
                    rep = repair_call(rec['out'])
                    if rep is None:
                        r['outcome'] = 'refused_no_repair'
                    else:
                        args = fill_repair(rep, edits[-1], s['file'])
                        r['repair_args'] = {k: (v if len(v) < 120 else v[:117] + '...') for k, v in args.items()}
                        rec2 = self.record(ss.call('edit.py', args, 'robust'))
                        got2 = read_file(self.ws, s['file'])
                        r['repair_status'] = edit_status(rec2['out'])
                        r['repair_out_head'] = rec2['out'][:800]
                        r['outcome'] = 'repair_intended' if got2 == want else 'repair_refused' if got2 == pre_text \
                            else 'repair_equivalent' if same_code(got2, want) else 'repair_wrong'
                out.append(r)
        reset_ws(self.ws)
        return out


# ------------------------------------------------------------------ tasks

def load_local(ids):
    tasks = {}
    for line in open(os.path.join(ROOT, 'data', 'competition', 'tasks.jsonl')):
        d = json.loads(line)
        tasks[d['instance_id']] = d
    return [dict(tasks[i], source='local') for i in ids if i in tasks]


def load_external(ids=None):
    import toolbench_data
    recs = toolbench_data.load_records(set(ids) if ids else None)
    return [dict(r, source='external') for r in recs]


def task_ids(path):
    return [l.split()[0] for l in open(path) if l.strip() and not l.startswith('#')]


def prepare_local(task, workdir):
    import replay_tools
    ws = os.path.join(workdir, 'ws')
    replay_tools.prepare(os.path.join(SNAPSHOTS, f'{task["instance_id"]}.tgz'), ws)
    return ws


def work(job):
    """Run every phase for every agent on one task (one workspace)."""
    task, agents, opt = job
    t0 = time.time()
    workdir = tempfile.mkdtemp(prefix=f'tb_{task["instance_id"]}_', dir=opt.get('workroot'))
    out = []
    try:
        if task['source'] == 'local':
            ws = prepare_local(task, workdir)
        else:
            ws = os.path.join(TB, 'ws', task['instance_id'])
            if not os.path.exists(os.path.join(ws, '.git', 'toolbench_ok')):
                raise RuntimeError('workspace not set up (scripts/toolbench_data.py setup)')
            reset_ws(ws)
        task = dict(task, query=(opt.get('queries') or {}).get(task['instance_id'])
                    or build_query(task['problem_statement']))
        task['gold'] = gold_of(task['patch'], lambda p: read_file(ws, p))
        if not task['gold']['spans']:
            return task['instance_id'], [{'task': task['instance_id'], 'source': task['source'],
                                          'repo': task['repo'], 'agent': a['label'], 'skip': 'no modified .py gold'}
                                         for a in agents], time.time() - t0
        if opt.get('format_diff'):
            return task['instance_id'], format_diff_task(task, agents, ws, opt, workdir), time.time() - t0
        for agent in agents:
            tr = TaskRun(task, agent, ws, opt, workdir)
            rec = {'task': task['instance_id'], 'source': task['source'], 'repo': task['repo'], 'agent': agent['label'],
                   'gold': {k: task['gold'][k] for k in ('files', 'functions', 'added', 'unparsable')},
                   'spans': [[s['file'], s['a'], s['b']] for s in task['gold']['spans']]}
            try:
                rec['locate'] = tr.locate()
                rec['bot'] = tr.bot()
                rec['explore'] = tr.bot(explore=True)
                rec['oracle'] = tr.oracle()
                rec['robust'] = tr.robust()
            except Exception:  # noqa: BLE001 - one broken task must not stop the batch
                import traceback
                rec['error'] = traceback.format_exc()[-1500:]
                reset_ws(ws)
            rec['calls'] = [{k: v for k, v in c.items() if k not in ('err',) or c.get('crash')} for c in tr.calls]
            out.append(rec)
    except Exception as e:  # noqa: BLE001
        out = [{'task': task['instance_id'], 'source': task['source'], 'repo': task.get('repo'), 'agent': a['label'],
                'error': f'setup: {e!r}'[:500]} for a in agents]
    finally:
        if task['source'] == 'external':
            try:
                reset_ws(os.path.join(TB, 'ws', task['instance_id']))
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(workdir, ignore_errors=True)
    return task['instance_id'], out, time.time() - t0


# ------------------------------------------------------------------ format diff

def format_diff_task(task, agents, ws, opt, workdir):
    """Agent A's locate, bot and oracle call sequences, replayed through B (same args, same clock, B's own flow state
    after B's locate): per call, is the normalised output text the same (body / footer)?"""
    a, b = agents[:2]
    tr = TaskRun(task, a, ws, opt, workdir)
    tr.locate()
    tr.bot()
    tr.bot(explore=True)
    tr.oracle()
    reset_ws(ws)
    tb = TaskRun(task, b, ws, opt, workdir)
    tb.locate()
    pairs = [(tr.calls[0], tb.calls[0])]
    for phase in ('bot', 'explore', 'oracle'):
        reset_ws(ws)
        sess = tb.session(tb.new_state(tb.template), 2)
        pairs += [(c, sess.call(c['script'], c['args'], phase)) for c in tr.calls if c['phase'] == phase]
    reset_ws(ws)
    rows = []
    for ca, cb in pairs:
        na, nb = normalise(ca['out']), normalise(cb['out'])
        ba, fa = split_footer(na)
        bb, fb = split_footer(nb)
        rows.append({'task': task['instance_id'], 'source': task['source'], 'agent': f'{a["label"]}->{b["label"]}',
                     'phase': ca['phase'], 'i': ca['i'], 'script': ca['script'], 'same': na == nb,
                     'body_same': ba == bb, 'footer_same': fa == fb, 'same_ex_volatile': devolatile(na) == devolatile(nb),
                     'diff': None if na == nb else first_diff(na, nb)})
    return rows


VOLATILE = re.compile(r'^names: .*(?:\n|$)', re.M)


def devolatile(text):
    """Without the lines whose content depended on machine speed: show.py's `names:` line (up to agent-v18d names.py
    resolved names within a 0.2 s clock, so a loaded machine printed fewer; v19 uses count caps)."""
    return VOLATILE.sub('', text)


def first_diff(x, y):
    xs, ys = x.split('\n'), y.split('\n')
    for k in range(max(len(xs), len(ys))):
        lx = xs[k] if k < len(xs) else '<missing>'
        ly = ys[k] if k < len(ys) else '<missing>'
        if lx != ly:
            return f'line {k + 1}: A `{lx[:150]}` | B `{ly[:150]}`'
    return None


def format_report(rows):
    rows = [r for r in rows if 'same' in r]  # skipped tasks / errors carry no comparison
    lines = []
    p = lines.append
    for src in sorted({r['source'] for r in rows}):
        rs = [r for r in rows if r['source'] == src]
        p(f'=== {src}: {len({r["task"] for r in rs})} tasks, {len(rs)} calls compared ({rs[0]["agent"]})')
        p(f'identical {sum(r["same"] for r in rs)}, body differs {sum(not r["body_same"] for r in rs)}, footer '
          f'differs {sum(not r["footer_same"] for r in rs)}; identical apart from the time-limited `names:` line '
          f'{sum(r.get("same_ex_volatile", r["same"]) for r in rs)}')
        by = collections.defaultdict(list)
        for r in rs:
            by[r['script']].append(r)
        p('per script (differ/calls): ' + ', '.join(f'{s} {sum(not r["same"] for r in x)}/{len(x)}'
                                                    for s, x in sorted(by.items(), key=lambda kv: -len(kv[1]))))
        for r in sorted([r for r in rs if not r['same']], key=lambda r: r.get('same_ex_volatile', False))[:8]:
            p(f'  {r["task"]} {r["phase"]} call {r["i"]} {r["script"]}: {r["diff"]}')
    return '\n'.join(lines)


# ------------------------------------------------------------------ scorecard

def pct(xs):
    xs = [bool(x) for x in xs if x is not None]
    return 100.0 * sum(xs) / len(xs) if xs else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def metrics(recs):
    """{metric: value} over task records of one agent and one source."""
    ok = [r for r in recs if 'locate' in r]
    m = collections.OrderedDict()
    m['tasks'] = len(ok)
    m['errors'] = sum(1 for r in recs if r.get('error'))
    L = [r['locate'] for r in ok]
    m['locate file@1 %'] = pct(l['rank'] == 1 for l in L)
    m['locate file@3 %'] = pct(l['rank'] is not None and l['rank'] <= 3 for l in L)
    m['locate file printed %'] = pct(l['rank'] is not None for l in L)
    m['locate gold fn in list %'] = pct(l['fn_in_list'] for l in L if l['has_fn'])
    m['locate best match = gold fn %'] = pct(l['best_gold'] for l in L if l['has_fn'])
    m['locate preview shown %'] = pct(l['best_shown'] for l in L)
    m['locate prints gold span %'] = pct(l['gold_shown'] for l in L)
    m['locate tokens (mean)'] = mean(l['tokens'] for l in L)
    B = [r['bot'] for r in ok if 'bot' in r]
    m['bot shows gold span %'] = pct(b['gold_at'] is not None for b in B)
    m['bot shows gold after locate %'] = pct(b['gold_at'] is not None and b['gold_at'] > 1 for b in B)
    m['bot calls to gold (median)'] = med(b['gold_at'] for b in B)
    m['bot prompt at gold (median tok)'] = med(b['gold_ctx'] for b in B)
    m['bot gold only past 14,336 %'] = pct(b['crossed_before_gold'] for b in B if b['gold_at'])
    m['bot reaches edit suggestion %'] = pct(b['stop'] == 'edit_suggested' for b in B)
    E = [b['edit'] for b in B if b.get('edit')]
    m['bot edit suggestion on gold span % (all tasks)'] = pct(bool(b.get('edit') and b['edit'].get('overlap'))
                                                              for b in B)
    m['bot edit suggestion on gold span % (of judgeable)'] = pct(e.get('overlap') for e in E)
    m['bot edit suggestion near gold (+-5) % (judgeable)'] = pct(e.get('near') for e in E)
    m['bot edit suggestion in a gold file % (of all sugg.)'] = pct(e['gold_file'] for e in E)
    m['bot calls (mean)'] = mean(b['calls'] for b in B)
    m['bot tokens (mean)'] = mean(b['tokens'] for b in B)
    m['bot repeated calls (mean)'] = mean(b.get('repeat', 0) for b in B)
    X = [r['explore'] for r in ok if 'explore' in r]
    m['explore shows gold span %'] = pct(b['gold_at'] is not None for b in X)
    m['explore shows gold via a NEXT read %'] = pct(b['gold_at'] is not None and b['gold_at'] > 1 for b in X)
    m['explore NEXT read finds gold (of locate misses) %'] = pct(b['gold_at'] is not None for b, r in zip(X, ok)
                                                                  if not r['locate']['gold_shown'])
    m['explore calls to gold (median)'] = med(b['gold_at'] for b in X)
    m['explore prompt at gold (median tok)'] = med(b['gold_ctx'] for b in X)
    m['explore gold only past 14,336 %'] = pct(b['crossed_before_gold'] for b in X if b['gold_at'])
    m['explore calls (mean)'] = mean(b['calls'] for b in X)
    m['explore tokens (mean)'] = mean(b['tokens'] for b in X)
    m['explore repeated calls (mean)'] = mean(b.get('repeat', 0) for b in X)
    for k, v in collections.Counter(b['stop'] for b in X).most_common(4):
        m[f'  explore stop: {k} %'] = 100.0 * v / max(1, len(X))
    O = [r['oracle'] for r in ok if 'oracle' in r]
    m['oracle final == gold %'] = pct(o['exact'] for o in O)
    m['oracle == gold modulo trailing ws %'] = pct(o['equal_ws'] for o in O)
    m['oracle == gold as code (AST) %'] = pct(o.get('same_code') for o in O)
    m['oracle merged retries (tasks %)'] = pct(o.get('merged_retries', 0) > 0 for o in O)
    m['oracle all edits applied first try %'] = pct(o['first_try'] for o in O)
    m['oracle check_patch READY %'] = pct(o['check_ready'] for o in O)
    m['oracle check_patch OK ignoring missing deps %'] = pct(o.get('check_ok_ignoring_env') for o in O)
    m['oracle calls (median)'] = med(o['calls'] for o in O)
    m['oracle tokens (median)'] = med(o['tokens'] for o in O)
    m['oracle prompt at end (median)'] = med(o['ctx_end'] for o in O)
    m['oracle past 14,336 %'] = pct(o['ctx_end'] >= COMPACT_AT for o in O)
    R = [x for r in ok for x in r.get('robust', [])]
    m['robust cases'] = len(R)
    m['robust intended (direct) %'] = pct(x['outcome'] == 'intended' for x in R)
    m['robust intended (direct or via repair) %'] = pct(x['outcome'] in OK_OUTCOMES for x in R)
    m['robust cosmetic-only difference %'] = pct(x['outcome'] in ('equivalent', 'repair_equivalent') for x in R)
    m['robust silent wrong edit %'] = pct(x['outcome'] == 'wrong' for x in R)
    m['robust silent wrong edit % (v19 mistakes)'] = pct(x['outcome'] == 'wrong' for x in R
                                                         if x['mistake'] in MISTAKES_V19)
    m['robust intended (direct or via repair) % (v19 mistakes)'] = pct(x['outcome'] in OK_OUTCOMES for x in R
                                                                       if x['mistake'] in MISTAKES_V19)
    m['  of which flagged by a WARNING %'] = pct(x.get('warned') for x in R if x['outcome'] == 'wrong')
    m['robust repair call -> wrong edit %'] = pct(x['outcome'] == 'repair_wrong' for x in R)
    m['robust refused, no repair call %'] = pct(x['outcome'] == 'refused_no_repair' for x in R)
    m['robust silent no-op %'] = pct(x['outcome'] == 'noop' for x in R)
    m['robust repair call executed -> intended %'] = pct(x['outcome'] == 'repair_intended' for x in R
                                                         if x['outcome'].startswith('repair'))
    for name in MISTAKES:
        xs = [x for x in R if x['mistake'] == name]
        if xs:
            m[f'  {name} ok % / wrong % / n'] = (
                pct(x['outcome'] in OK_OUTCOMES for x in xs),
                pct(x['outcome'] in ('wrong', 'repair_wrong') for x in xs), len(xs))
    C = [c for r in ok for c in r.get('calls', [])]
    m['calls total'] = len(C)
    m['footer share of output %'] = 100.0 * sum(c['footer_chars'] for c in C) / max(1, sum(c['chars'] for c in C))
    m['script crashes'] = sum(1 for c in C if c.get('crash'))
    m['script timeouts'] = sum(1 for c in C if c['code'] == 'timeout')
    by = collections.defaultdict(list)
    for c in C:
        by[c['script']].append(c)
    for s, cs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        m[f'  {s} tok/call / footer % / calls'] = (
            mean(c['tokens'] for c in cs), 100.0 * sum(c['footer_chars'] for c in cs) / max(1, sum(c['chars']
                                                                                              for c in cs)), len(cs))
    return m


def fmt(v):
    if v is None:
        return '-'
    if isinstance(v, tuple):
        return '/'.join(fmt(x) for x in v)
    if isinstance(v, float):
        return f'{v:.1f}'
    return str(v)


def delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = b - a
        return f'{d:+.1f}' if isinstance(d, float) else f'{d:+d}'
    if isinstance(a, tuple) and isinstance(b, tuple):
        return '/'.join(delta(x, y) for x, y in zip(a, b))
    return ''


def scorecard(recs):
    agents = list(dict.fromkeys(r['agent'] for r in recs))
    lines, summary = [], {}
    for src in ('local', 'external'):
        rs = [r for r in recs if r['source'] == src]
        if not rs:
            continue
        ms = {a: metrics([r for r in rs if r['agent'] == a]) for a in agents}
        summary[src] = {a: {k: v for k, v in m.items()} for a, m in ms.items()}
        skipped = sorted({r['task'] for r in rs if r.get('skip')})
        lines.append(f'=== {src}: {ms[agents[0]]["tasks"]} tasks' + (f' ({len(skipped)} skipped: no modified .py gold)'
                                                                       if skipped else ''))
        keys = list(ms[agents[0]].keys())
        for a in agents[1:]:
            keys += [k for k in ms[a] if k not in keys]
        w = max(len(k) for k in keys) + 2
        head = f'{"metric":<{w}}' + ''.join(f'{a[:16]:>17}' for a in agents) + ('  delta' if len(agents) > 1 else '')
        lines.append(head)
        for k in keys:
            vals = [ms[a].get(k) for a in agents]
            row = f'{k:<{w}}' + ''.join(f'{fmt(v):>17}' for v in vals)
            if len(agents) > 1:
                row += '  ' + delta(vals[0], vals[-1])
            lines.append(row)
        lines.append('')
    return '\n'.join(lines), summary


# ------------------------------------------------------------------ main

def materialize(spec, dest):
    label, _, s = spec.partition('=') if '=' in spec else (None, '', spec)
    label = label or os.path.basename(s.rstrip('/')) or s
    out = os.path.join(dest, '_agents', label)
    subprocess.run(['bash', os.path.join(ROOT, 'scripts/materialize.sh'), s, out], cwd=ROOT, check=True,
                   capture_output=True)
    import replay_tools
    skill, scripts = replay_tools.skill_info(out)
    return {'label': label, 'dir': out, 'skill': skill, 'scripts': os.path.abspath(scripts)}


def select_tasks(a):
    srcs = a.source.split(',')
    tasks = []
    only = None
    if a.tasks:
        only = set(a.tasks.split(','))
    if 'local' in srcs:
        ids = task_ids(a.tasks_file or os.path.join(ROOT, 'experiments', 'eval113.txt'))
        tasks += load_local([i for i in ids if not only or i in only])
    if 'external' in srcs:
        import toolbench_data
        ext = [t for t in load_external() if not only or t['instance_id'] in only]
        tasks += [t for t in ext if toolbench_data.ws_ready(t['instance_id'])]
    if a.limit:
        per = collections.defaultdict(list)
        for t in tasks:
            per[t['source']].append(t)
        tasks = [t for ts in per.values() for t in ts[:a.limit]]
    return tasks


def run_jobs(tasks, agents, opt, workers, log_path):
    recs = []
    t0 = time.time()
    jobs = [(t, agents, opt) for t in tasks]
    with open(log_path, 'w') as log, concurrent.futures.ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for k, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            tid, out, secs = fut.result()
            for r in out:
                log.write(json.dumps(r, ensure_ascii=False) + '\n')
            log.flush()
            recs += out
            err = [r for r in out if r.get('error')]
            if k % 10 == 0 or err or k == len(jobs):
                print(f'[{k}/{len(jobs)}] {tid} {secs:.0f}s' + (f' ERROR {err[0]["error"][-200:]!r}' if err else '')
                      + f'  ({time.time() - t0:.0f}s)', flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--agent', action='append', default=[], help='git ref | WORKTREE | dir, optionally LABEL=SPEC')
    ap.add_argument('--format-diff', nargs=2, metavar=('A', 'B'), help='compare the model-facing text of A and B')
    ap.add_argument('--source', default='local,external')
    ap.add_argument('--tasks', default=None, help='comma-separated task ids')
    ap.add_argument('--tasks-file', default=None, help='local task list (default experiments/eval113.txt)')
    ap.add_argument('--limit', type=int, default=0, help='first N tasks per source')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--label', default=None)
    ap.add_argument('--python', default=None, help='interpreter for the scripts (default python3.13)')
    ap.add_argument('--call-seconds', type=float, default=20.0, help='simulated seconds per call (flow clock)')
    ap.add_argument('--queries', default=None, help='JSON {task id: locate query} overriding build_query (e.g. the '
                    'queries the model sent in recorded runs)')
    ap.add_argument('--hashseed', default='0', help="PYTHONHASHSEED of the script processes (default 0: "
                    "reproducible; 'random' as on Kaggle: some outputs then vary run to run)")
    ap.add_argument('--report', default=None, help='re-print the scorecard of an output dir')
    a = ap.parse_args()
    if a.report:
        recs = []
        for fn in sorted(os.listdir(a.report)):
            if fn.endswith('.jsonl') and fn != 'format_diff.jsonl':
                recs += [json.loads(l) for l in open(os.path.join(a.report, fn)) if l.strip()]
        text = scorecard(recs)[0]
        if recs:
            open(os.path.join(a.report, 'scorecard.txt'), 'w').write(text + '\n')
        print(text)
        fd = os.path.join(a.report, 'format_diff.jsonl')
        if os.path.exists(fd):
            print(format_report([json.loads(l) for l in open(fd) if l.strip()]))
        return
    specs = list(a.format_diff) if a.format_diff else a.agent
    if not specs:
        ap.error('--agent or --format-diff is required')
    label = a.label or time.strftime('tb_%Y%m%d_%H%M%S')
    out = os.path.join(RESULTS, label)
    os.makedirs(out, exist_ok=True)
    agents = [materialize(s, out) for s in specs]
    opt = {'python': a.python or default_python(), 'call_seconds': a.call_seconds, 'format_diff': bool(a.format_diff),
           'queries': json.load(open(a.queries)) if a.queries else None, 'hashseed': a.hashseed}
    tasks = select_tasks(a)
    print(f'{len(tasks)} tasks ({collections.Counter(t["source"] for t in tasks)}), agents '
          f'{[x["label"] for x in agents]}, python {opt["python"]}, workers {a.workers} -> {out}', flush=True)
    t0 = time.time()
    if a.format_diff:
        rows = run_jobs(tasks, agents, opt, a.workers, os.path.join(out, 'format_diff.jsonl'))
        rows = [r for r in rows if 'same' in r]
        text = format_report(rows)
        open(os.path.join(out, 'format_diff.txt'), 'w').write(text + '\n')
        print(text)
        print(f'done in {time.time() - t0:.0f}s')
        return
    recs = run_jobs(tasks, agents, opt, a.workers, os.path.join(out, 'records.jsonl'))
    for ag in agents:
        with open(os.path.join(out, f'{ag["label"]}.jsonl'), 'w') as f:
            for r in recs:
                if r['agent'] == ag['label']:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.remove(os.path.join(out, 'records.jsonl'))
    text, summary = scorecard(recs)
    open(os.path.join(out, 'scorecard.txt'), 'w').write(text + '\n')
    json.dump({'agents': [x['label'] for x in agents], 'python': opt['python'], 'call_seconds': a.call_seconds,
               'secs': round(time.time() - t0), 'summary': summary}, open(os.path.join(out, 'summary.json'), 'w'),
              indent=1, default=list)
    print(text)
    print(f'done in {time.time() - t0:.0f}s -> {out}')


if __name__ == '__main__':
    main()
