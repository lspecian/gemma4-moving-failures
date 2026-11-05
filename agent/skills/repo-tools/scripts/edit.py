"""Replace text in a source file, tolerating escaped text, and refuse edits that break Python syntax.

usage: edit.py  target=billing/invoice.py  old=<exact current text>  new=<replacement>
       edit.py  target=billing/invoice.py  start=151  end=153  new=<text for lines 151-153>
       edit.py  target=billing/invoice.py  start=151  end=151  new=<...>  start2=190  end2=192  new2=<...>
       (several places of one file in one call, v20: start2 end2 new2 or old2 new2, ... up to 4 hunks)
       (positional form, still accepted: edit.py FILE OLD NEW  |  edit.py --lines FILE START END NEW)

The first form replaces the unique occurrence of OLD with NEW; the second replaces lines START..END (1-based,
inclusive) with NEW.

Paths are relative to /workspace. Matching order: exact text, then with literal escapes (\\n \\t \\" \\')
decoded (up to two layers), then line by line ignoring indentation differences (NEW is re-indented), then
line-number prefixes copied from show.py removed, then a near-identical block (every line nearly equal).
OLD must match exactly one place; when it matches several, each is listed with its enclosing function and a ready
line-range edit. After an edit, other places in the file that still contain a changed line are named ("the same code
also appears at"). "start" without "end" replaces that one line when NEW is one line; for a multi-line NEW the end is
inferred when clear (v19, below), else NOTHING is applied and the likely range is printed with a ready line-range
call. Line-number prefixes copied from show.py (`  955  code`) are removed from NEW when every line has one. A .py file
must still parse after the edit, otherwise nothing is written; when NEW's indentation is off by a consistent amount,
the indentation that parses is used, and a line-range NEW shifted as a whole is moved back even when it parses. A .py
edit is also undone when the module no longer imports (it did before), refused when it makes a read-only @property of
a name the class assigns (self.X = ...), and refused when it deletes a module- or class-level def/class/property
that the file, its package or the tests still use (a WARNING when nothing uses it). A WARNING flags \\uXXXX / \\xNN
escapes that Python reads as the plain character.
Test files and test-runner config are refused: the grader discards changes to them. New files: create.py.
On NOT APPLIED the last line is `CLOSEST: FILE START END` when one region is the likely target.

v18 (research/notes/13-tool-efficiency.md, code-review P4-P6, audit #8):
- No target: the file the model last read or edited (flow state) is used with a NOTE when it is unambiguous: OLD is
  found in exactly one of the recently read files, or START..END lies in exactly one file's read ranges.
- Stale line numbers: a line range read before an edit of the same file that moved it (and not seen since) is
  shifted by the edit's offset with a NOTE when NEW matches the shifted lines at least as well; refused with the
  current numbers when unclear or when the edit changed those lines. Ranges read are flow's `viewed` rows plus the
  ranges edit.py printed itself (meta 'edit:shown'); every successful edit's hunks are kept (meta 'edit:hunks').
  After an edit following an ambiguous-OLD list, the remaining sites are reprinted with their current lines.
- An indentation repair is anchored on NEW lines that equal replaced lines, and a repair that leaves code after a
  return/raise/continue/break of the same block (dead code) is refused, naming the repair; NEW as written that does
  so gets a WARNING.
- The closest region of an OLD not found is printed only from similarity CLOSEST_MIN (fact similarity=<ratio>);
  below it: "no similar lines". Ambiguous OLD: the sites once, one line each. Usage errors: one example. A syntax
  error prints NEW's lines as placed (numbered, the error line marked >>) and the expected indentation.

v19 (toolbench edit robustness: start_only 31% ok / 37% wrong, indent_minus4 19% silently wrong):
- "start" without "end" (infer_end): the lines after START are aligned with NEW (equal lines, similar lines, lines
  NEW leaves out); an end must keep the file parsing without dead code and must not leave a line NEW repeats right
  after it. Applied with a NOTE when NEW ends with a distinctive line of the file there, when NEW starts with lines
  START..END unchanged and the rest is new (an insertion), or when NEW rewrites START..END line for line - each only
  when the alignment agrees. A one-line NEW equal to a later line replaces START..that line. Otherwise NOT APPLIED with
  the best-aligned range (and the other likely one) as ready calls.
- A line-range NEW indented D more or less than the lines it replaces as a whole (uniform_shift: its first line, or
  the lines it keeps, and its least indented line are all D off; for a blank line replaced: below the next line) is
  moved back with "(indentation repaired: ...)" even when it parses as written; not when NEW changes only the
  indentation, nor inside a multi-line string.
- Stale line numbers that no read explains: shifted with a NOTE only when NEW clearly matches the moved lines.

v20 (research/notes/15-edit-failures.md: 211 failed edit calls of E057-E061 replayed with scripts/edit_failures.py; 55
of 160 failed in E057/E058). Every inference below prints a NOTE saying what it did:
- NEW decoding: literal \\n outside string literals are line breaks also when NEW has real ones (decode_breaks; a \\n
  inside a string stays). The near-identical match tries every decoding of OLD.
- start without end: an end fits when NEW fits there moved back by its uniform indentation shift (NEW indented 2 or 4
  more than the file was refused 12 times); a bare decorator (@property) after the range is not a repeated line; NEW
  beginning with the line above START (its decorator) starts there; one line, or several, for a multi-line simple
  statement (assignment, call, return, import, raise) that begins at START replaces that statement; new code (no line
  like the next ones) at a blank line is inserted there, the blank line kept.
- end = start - 1: NEW is inserted before line START. Neither old nor start: NEW's first line, when it stands once in
  the file, or the def/class NEW redefines, when the file has one, is START (then as start without end).
- A line range whose result does not parse is repaired when exactly one reading of NEW's own first/last lines parses
  (range_repair): NEW starts with the line above START, ends with the line below END or with a line inside the
  range, the start-only inference, or the range is only closing brackets NEW lacks (NEW goes after them).
- OLD not found: blank lines miscounted (blank_insensitive; NEW's blank runs that copy OLD's take the file's count),
  then OLD's first and last lines as anchors with NEW's changes merged into the block (anchor_merge: every changed
  OLD line must be in the block, at least half of OLD's lines; the lines OLD had wrong are kept as the file has them).
  Several candidates: the sites are listed (viewed_choice, the one inside the code read last, is off: PREFER_VIEWED).
  NEW already in the file (once): NOT CHANGED, counted as done. Exact duplicates are listed as before; a one-line
  OLD/NEW gets a ready multi-edit call.
- Compile errors beyond syntax (a repeated keyword argument) are refused before writing.
- Multi-edit: hunks 2..MAX_HUNKS as start2 end2 new2 or old2 new2 (mixed is fine); all are resolved against the file
  before this call (its line numbers), overlaps refused, applied bottom-up all or nothing; one diff, one CHECK line.
- CHECK line after every .py edit (check_line): new undefined names (guards.new_undefined_names: symtable + ast, only
  names not undefined before), the import status (the import guard's own run: OK / timed out; an import that fails
  without the edit too is not mentioned and is cached per workspace, meta 'edit:import_env'). The import's timeout
  shrinks near the task's end (import_budget) but the guard always runs.
"""

import ast
import difflib
import json
import os
import re
import sys
import warnings
from collections import Counter

sys.dont_write_bytecode = True
warnings.filterwarnings('ignore', category=SyntaxWarning)  # the file's own invalid escapes, once per parse:
# stderr noise in every result (v19 parses each candidate range); guards.escape_warnings reports NEW's
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402  (flow engine: refusals + NEXT/BUDGET footers)
import argfix  # noqa: E402
import guards  # noqa: E402  (import / property / escape guards, agent-v15b)
import names  # noqa: E402  (enclosing definitions of the sites of an ambiguous OLD)
from argfix import Misuse  # noqa: E402

ROOT = (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))  # overridable for local testing
PROTECTED = re.compile(
    r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)(conftest\.py|pytest\.ini|pyproject\.toml|'
    r'setup\.cfg|tox\.ini|\.pytest\.ini|sitecustomize\.py)$')
ESCAPES = {'n': '\n', 't': '\t', '"': '"', "'": "'", '\\': '\\'}
USAGE = __doc__.strip().split('\n\n')[1]
LINE_PREFIX = re.compile(r'^\s*\d+(?: {2}|\t|: ?|\| ?)')
ELISION = re.compile(  # '# ... existing code ...', '# ...', '# (rest of the method)', '# omitted for brevity'
    r'^[ \t]*(?:#|//)[ \t]*(?:(?:\.\.\.|…)[ \t]*$|(?:\.\.\.|…)[^\n]*?\b(?:rest|existing|remaining|unchanged|same|'
    r'previous|original|other|code|omitted)\b|\(?[ \t]*rest of (?:the )?(?:code|function|method|file|class|body|'
    r'implementation)\b|[^\n]*\bomitted for brevity\b)', re.I | re.M)
ANGLE = re.compile(r'^\s*<\s*(new|old|the|your|replacement|updated|fixed|modified|code|text|content)[\w\s-]{0,30}>\s*$',
                   re.I)
FUZZY_MIN, FUZZY_LINE_MIN, FUZZY_RIVAL, FUZZY_MIN_LINES = 0.95, 0.8, 0.9, 3
SITES_MAX = 5          # sites listed for an ambiguous OLD, and other places that still have a changed line
SITE_MAX_CHANGED = 8   # edits that change more distinctive lines than this are rewrites: no "also appears at"
SITE_MIN_CHARS = 12    # a changed line this long (stripped) or longer is distinctive enough to look for elsewhere
TRIVIAL = re.compile(r'(?:return|pass|else:|try:|finally:|break|continue|raise|\)|\]|\}|[)\]}],?|#.*|"""|\'\'\'|'
                     r'return (?:None|True|False|self|result)|except Exception:)$')
SCORE_LINES = 25
CLOSEST_MIN = 0.6      # v18 P5: below this similarity no closest region is printed (0.55 sent rich to escape_text)
RECENT_FILES = 4       # files the model read or edited last, searched for OLD when no target is given
MEM_MAX = 40           # entries kept per edit.py memory list (meta 'edit:...')
SHIFT_MARGIN = 0.15    # stale numbers: NEW must match one numbering this much better than the other to decide
NEW_SHOW_MAX = 16      # NEW lines printed with a syntax error (longer: head and the error window)
CHECK_MIN_LEFT_S = 35  # v20: the import check's timeout shrinks as flow's clock nears this many seconds left
IMPORT_MIN_S = 3       # v20: ... but never below this
MAX_HUNKS = 4          # v20 multi-edit: hunks per call (start/end/new, start2/end2/new2, ... or old/new, old2/new2, ...)


# ---------------------------------------------------------------- flow state (v18)

def flow_on():
    return os.environ.get('REPO_TOOLS_FLOW', 'on') != 'off'


_STATE = []


def state():
    """flow's State for this call (None when flow is off or unavailable), read once."""
    if not _STATE:
        st = None
        if flow_on():
            try:
                st = flow.peek()
            except Exception:
                st = None
        _STATE.append(st)
    return _STATE[0]


def call_n():
    """The number this call will get in flow's record (0 when flow is off)."""
    st = state()
    return st.n + 1 if st is not None else 0


def mem_get(name):
    st = state()
    if st is None:
        return []
    try:
        return json.loads(flow.get_meta(st.conn, 'edit:' + name) or '[]') or []
    except Exception:
        return []


def mem_set(name, value):
    st = state()
    if st is None:
        return
    try:
        flow.set_meta(st.conn, 'edit:' + name, json.dumps(value[-MEM_MAX:]))
    except Exception:
        pass


def remember_shown(rel, *ranges):
    """Line ranges of REL this call printed (they are what the model's next line numbers come from)."""
    if state() is None or not ranges:
        return
    mem_set('shown', mem_get('shown') + [[call_n(), rel, int(a), int(b)] for a, b in ranges])


def file_hunks(before, after):
    """[[i1, i2, j1, j2]] of the changed line blocks (0-based, difflib opcodes other than 'equal')."""
    a, b = before.split('\n'), after.split('\n')
    return [[i1, i2, j1, j2] for tag, i1, i2, j1, j2 in
            difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if tag != 'equal']


HUNK_OLD_MAX = 300     # old lines kept per edit (to rebuild what the model read before it)


def with_old_lines(hunks, old):
    """HUNKS with the old lines each replaced appended ([i1, i2, j1, j2, [lines]]; None past HUNK_OLD_MAX lines)."""
    total = sum(i2 - i1 for i1, i2, _, _ in hunks)
    return [[i1, i2, j1, j2, old[i1:i2] if total <= HUNK_OLD_MAX else None] for i1, i2, j1, j2 in hunks]


def text_before(lines, edits):
    """LINES (the file now) as it was before EDITS ([n, rel, hunks] in order), or None when an edit kept no old
    lines."""
    lines = list(lines)
    for _, _, hs in reversed(edits):
        for h in sorted(hs, key=lambda h: -h[2]):
            if len(h) < 5 or h[4] is None:
                return None
            lines[h[2]:h[3]] = h[4]
    return lines


def map_range(s, e, hunks):
    """Lines S..E (1-based, before an edit with HUNKS) after it: (s, e), or None when the edit changed them."""
    delta = 0
    for i1, i2, j1, j2, *_ in hunks:
        if i1 == i2:  # an insertion after old line i1
            if i1 < s:
                delta += j2 - j1
            elif i1 < e:
                return None
        elif i2 < s:
            delta += (j2 - j1) - (i2 - i1)
        elif i1 + 1 <= e:
            return None
    return s + delta, e + delta


def similarity(a_lines, b_lines):
    a = '\n'.join(l.strip() for l in a_lines if l.strip())
    b = '\n'.join(l.strip() for l in b_lines if l.strip())
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a or b else 1.0


class Stale(Exception):
    """A line range from before an edit that changed or moved it: printed instead of editing (exit 1)."""

    def __init__(self, text, rel, start, end, current=False):
        super().__init__(text)
        self.text, self.rel, self.start, self.end, self.current = text, rel, start, end, current


def seen_since(st, rel, s, e, n0):
    """Whether lines S..E of REL were shown after call N0: a read (flow's viewed), a range edit.py printed, or a
    search hit / traceback line / diff of a later result naming one of the lines."""
    for p, a, b, n, _ in st.viewed:
        if n > n0 and p == rel and a and b and a <= s and e <= b:
            return True
    for n, p, a, b in mem_get('shown'):
        if n >= n0 and p == rel and a <= s and e <= b:
            return True
    nums = '|'.join(str(x) for x in {s, e})
    mark = re.compile(r'(?:' + re.escape(rel) + r':|' + re.escape(os.path.basename(rel)) + r'", line )(?:' + nums
                      + r')(?!\d)')
    return any(n > n0 and mark.search(summary or '') for n, _, _, _, summary in st.actions)


def moved_fits(rel, clines, nl, given, moved, ranged):
    """Which numbering NEW was written for when similarity does not tell (v19): True (MOVED) when only the moved range
    keeps a .py file parsing with NEW there (a range given; the other way round proves little: the indentation is
    repaired later), or NEW's first and last lines equal the moved range's ends more often than the given range's;
    False when they equal the given range's more often; None when neither tells."""
    if not nl:
        return None
    if ranged and rel.endswith('.py') and parses(rel, '\n'.join(clines))[0]:
        ok = [parses(rel, '\n'.join(clines[:a - 1] + nl + clines[b:]))[0] for a, b in (given, moved)]
        if ok[1] and not ok[0]:
            return True
    ends = [(nl[0].strip() == clines[a - 1].strip()) + (nl[-1].strip() == clines[b - 1].strip()) for a, b in (given, moved)]
    return None if ends[0] == ends[1] else ends[1] > ends[0]


def fix_stale(rel, s, e, new, notes):
    """(s, e) for a line-range edit of REL, shifted when the numbers come from a read made before an edit that moved
    them (P4, fastapi_14458: the model used 136-137 from a site list printed before its edit at 112 added 2 lines).
    e None: start only (treated as one line). Raises Stale when the numbers are stale but the edit changed those
    lines, or when NEW does not tell which numbering it meant."""
    _STALE_DONE.append(True)
    st = state()
    if st is None:
        return s, e
    hunks = sorted((h for h in mem_get('hunks') if h[1] == rel), key=lambda h: h[0])
    if not hunks:
        return s, e
    e_ = e if e is not None else s
    last = hunks[-1][0]
    if seen_since(st, rel, s, e_, last):
        return s, e
    sources = [n for p, a, b, n, _ in st.viewed if p == rel and a and b and a <= s and e_ <= b and n < last]
    sources += [n for n, p, a, b in mem_get('shown') if p == rel and a <= s and e_ <= b and n < last]
    src = max(sources) if sources else None  # None: no read of these lines (truncated show, numbers from elsewhere)
    later = [h for h in hunks if src is None or h[0] > src]
    try:
        clines = open(os.path.join(ROOT, rel), encoding='utf-8').read().split('\n')
    except OSError:
        return s, e
    cur, movers, toucher = (s, e_), [], None
    for n, _, hs in later:
        nxt = map_range(cur[0], cur[1], hs) if cur else None
        if nxt is not None and nxt != cur:
            movers.append(n)
        if nxt is None and cur is not None:
            toucher = n
        cur = nxt
    if cur == (s, e_):
        return s, e
    nl = new_lines(new)
    r_given = similarity(nl, clines[s - 1:e_]) if e_ <= len(clines) else 0.0
    if src is None:  # v19: no read to date the numbers by: shift only when NEW clearly matches the moved lines
        if cur is None or cur[1] > len(clines) or e_ > len(clines):
            return s, e
        r_moved = similarity(nl, clines[cur[0] - 1:cur[1]])
        if r_moved < r_given + SHIFT_MARGIN and not (r_moved >= r_given and
                                                     moved_fits(rel, clines, nl, (s, e_), cur, e is not None)):
            return s, e
        s2, e2 = cur
        many = e_ > s
        span, now = (f'{s}-{e_}', f'{s2}-{e2}') if many else (f'{s}', f'{s2}')
        notes.append(f'NEW matches line{"s" if many else ""} {now}, where your edit{"s" if len(movers) > 1 else ""} at '
                     f'call{"s" if len(movers) > 1 else ""} {", ".join(map(str, movers))} moved line{"s" if many else ""}'
                     f' {span}, not the current line{"s" if many else ""} {span}; used {now}')
        remember_shown(rel, (s2, e2))
        return s2, (e2 if e is not None else None)
    if cur is None:  # the edit changed those lines: compare NEW with what the model read there
        before = text_before(clines, later)
        r_read = similarity(nl, before[s - 1:e_]) if before is not None and e_ <= len(before) else 1.0
        if r_given >= r_read + SHIFT_MARGIN:
            return s, e  # NEW fits the lines as numbered now: the numbers are current (counted from its own edit)
        raise Stale(f'NOT APPLIED: lines {s}-{e_} are numbered as read at call {src}, and your edit at call '
                    f'{toucher} changed those lines since, so nothing was changed. Read them again: '
                    + argfix.obj('show.py', {'file': rel, 'start': max(1, s - 5), 'end': e_ + 10}), rel, s, e_)
    s2, e2 = cur
    r_read = similarity(nl, clines[s2 - 1:e2]) if e2 <= len(clines) else 0.0  # the lines read, now at s2..e2
    if r_given >= r_read + SHIFT_MARGIN:
        return s, e  # NEW fits the lines as numbered now: the numbers came from somewhere flow did not see
    many = e_ > s
    by = (f'your edit at call {movers[0]}' if len(movers) == 1 else
          'your edits at calls ' + ', '.join(map(str, movers[:-1])) + f' and {movers[-1]}')
    why = (f'{by} {"added" if s2 > s else "removed"} {abs(s2 - s)} line{"s" if abs(s2 - s) > 1 else ""} above '
           f'{"them" if many else "it"}')
    span, now = (f'{s}-{e_}', f'{s2}-{e2}') if many else (f'{s}', f'{s2}')
    tie = None if r_read >= r_given + SHIFT_MARGIN or e2 > len(clines) else \
        moved_fits(rel, clines, nl, (s, e_), (s2, e2), e is not None)
    if tie is False:
        return s, e
    if tie or r_read >= r_given + SHIFT_MARGIN:
        notes.append(f'line{"s" if many else ""} {span} {"are" if many else "is"} numbered as read at call {src}, before '
                     f'{why}; used the current number{"s" if many else ""} {now}')
        remember_shown(rel, (s2, e2))
        return s2, (e2 if e is not None else None)
    lines = [f'NOT APPLIED: line{"s" if many else ""} {span} {"are" if many else "is"} numbered as read at call {src}, '
             f'before {why}; {"they are" if many else "it is"} now {now}, and NEW does not tell which you meant, so '
             'nothing was changed. Lines now:']
    for j in range(s2, min(e2, len(clines)) + 1)[:SHOW_MAX]:
        lines.append(f'{j:5d}  {clines[j - 1][:200]}')
    lines.append('if these are the lines: ' + argfix.obj('edit.py', {'file': rel, 'start': s2, 'end': e2,
                                                                       'new': '<your NEW unchanged>'}))
    raise Stale('\n'.join(lines), rel, s2, e2, current=True)


def recent_files(st):
    """Source files the model read or edited, most recent first."""
    order = [(n, p) for p, a, b, n, _ in st.viewed] + [(e[0], e[1]) for e in st.edits]
    files = []
    for _, p in sorted(order, key=lambda x: -x[0]):
        p = argfix.clean_path(p) if p else ''
        if p and p not in files and not PROTECTED.search(p) and os.path.isfile(os.path.join(ROOT, p)):
            files.append(p)
    return files


def infer_target(d):
    """(file, note) for an edit call without a target: the file the model was working on, when unambiguous
    (audit #8: 28 'needs the key target' errors); None otherwise."""
    st = state()
    if st is None:
        return None
    files = recent_files(st)
    if not files:
        return None
    if str(d.get('old', '')).strip():
        hits = []
        for f in files[:RECENT_FILES]:
            try:
                found = locate(open(os.path.join(ROOT, f), encoding='utf-8').read(), d['old'])
            except (OSError, ValueError):
                continue
            if found is not None:
                hits.append(f)
        if len(hits) == 1:
            return hits[0], f'no target given; OLD is in {hits[0]}, a file you read, so edited that'
        if not hits and len(files) == 1:
            return files[0], f'no target given; used {files[0]}, the only file you read'
        return None
    s = argfix.line_number(d.get('start', ''))
    e = argfix.line_number(d.get('end', '')) or s
    if s is not None:
        cover = list(dict.fromkeys(argfix.clean_path(p) for p, a, b, n, _ in st.viewed
                                   if a and b and a <= s and e <= b))
        cover = [f for f in cover if f in files]
        if len(cover) == 1:
            return cover[0], f'no target given; lines {s}-{e} are in {cover[0]}, the file you read them in'
        if cover:
            return None
    if len(files) == 1:
        return files[0], f'no target given; used {files[0]}, the only file you read'
    return None


def decode(s):
    return re.sub(r'\\(.)', lambda m: ESCAPES.get(m.group(1), m.group(0)), s)


def decode_breaks(s):
    """S with the literal \\n (and \\t) that stand OUTSIDE string literals turned into line breaks (tabs), or None when
    there are none. v20 (E058 fastapi_13713, rich_3052): NEW arrived with real newlines and, on some lines, literal \\n
    escapes (`*,\\n        title: str,`): a backslash-n outside a string is never valid Python, while decode() would
    also turn the \\n inside string literals (`"...\\n"`) into line breaks. An escaped quote or backslash outside a
    string means the whole text is escaped (decode() handles that): None."""
    if '\\' not in s:
        return None
    out, i, n, quote, changed = [], 0, len(s), None, False
    while i < n:
        ch = s[i]
        if quote:  # inside a string literal: copy as is, escapes included
            if ch == '\\' and i + 1 < n:
                out.append(s[i:i + 2])
                i += 2
                continue
            if s.startswith(quote, i):
                out.append(quote)
                i += len(quote)
                quote = None
                continue
            if ch == '\n' and len(quote) == 1:  # an unterminated one-line string: stop treating it as a string
                quote = None
            out.append(ch)
            i += 1
            continue
        if ch == '#':  # a comment runs to the line break (real, or a literal \n)
            j = i
            while j < n and s[j] != '\n' and not s.startswith('\\n', j):
                j += 1
            out.append(s[i:j])
            i = j
            continue
        if ch in '"\'':
            quote = s[i:i + 3] if s.startswith(ch * 3, i) else ch
            out.append(quote)
            i += len(quote)
            continue
        if ch == '\\' and i + 1 < n:
            nxt = s[i + 1]
            if nxt in 'nt':
                out.append('\n' if nxt == 'n' else '\t')
                changed = True
                i += 2
                continue
            if nxt in '"\'\\':
                return None
        out.append(ch)
        i += 1
    return ''.join(out) if changed else None


def variants(s):
    """The text as given, then (v20) with only the literal \\n outside string literals decoded (decode_breaks), then
    decoded once and twice (deduplicated), each tagged with its level."""
    out, cur = [(0, s)], s
    smart = decode_breaks(s)
    if smart is not None:
        out.append((1, smart))
    for level in (1, 2):
        nxt = decode(cur)
        if nxt == cur:
            break
        if all(nxt != v for _, v in out):
            out.append((level, nxt))
        cur = nxt
    return out


def parses(path, text):
    if not path.endswith('.py'):
        return True, None
    try:
        ast.parse(text)
        return True, None
    except SyntaxError as e:
        return False, e


def line_starts(text, idx):
    return text.count('\n', 0, idx) + 1


def indent(line):
    return len(line) - len(line.lstrip())


def trim_blank(lines):
    lines = list(lines)
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    return lines


def flexible_match(content, old):
    """Match OLD line by line ignoring leading/trailing whitespace; return [(start, end, indent_delta)]."""
    olines = trim_blank(old.split('\n'))
    if not olines:
        return []
    clines = content.split('\n')
    want = [l.strip() for l in olines]
    hits = []
    for i in range(len(clines) - len(olines) + 1):
        if clines[i].strip() == want[0] and [l.strip() for l in clines[i:i + len(olines)]] == want:
            hits.append((i, i + len(olines), indent(clines[i]) - indent(olines[0])))
    return hits


def reindent(text, delta, first_delta=None):
    """Shift every line by DELTA spaces (the first line by FIRST_DELTA when given)."""
    out = []
    for n, line in enumerate(text.split('\n')):
        d = first_delta if (n == 0 and first_delta is not None) else delta
        if not line.strip() or d == 0:
            out.append(line)
        elif d > 0:
            out.append(' ' * d + line)
        else:
            out.append(line[min(-d, indent(line)):])
    return '\n'.join(out)


SHOW_PREFIX = re.compile(r'^ *(\d+)(  | ?$)')  # show.py prints f'{n:5d}  {line}' (a blank line: the number alone)


def strip_show_prefixes(text):
    """TEXT without the line-number prefixes show.py prints, when every non-blank line has one, the numbers increase
    and at least one line has code after its number; else None."""
    out, nums, code = [], [], False
    for line in text.split('\n'):
        if not line.strip():
            out.append(line)
            continue
        m = SHOW_PREFIX.match(line)
        if not m:
            return None
        nums.append(int(m.group(1)))
        code = code or bool(m.group(2))
        out.append(line[m.end():])
    if not code or any(b <= a for a, b in zip(nums, nums[1:])):
        return None
    return '\n'.join(out)


def unnumbered(new):
    """(NEW without show.py line-number prefixes, True) when some decoding of NEW has them on every line (the decoding
    with the most lines wins: NEW often arrives as one line with literal \\n), else (NEW, False). v17 (note 12 L2:
    fastapi_11355 lost its last minute to `   7      Any,` lines)."""
    best = None
    for _, v in variants(str(new)):
        stripped = strip_show_prefixes(v)
        if stripped is not None and (best is None or v.count('\n') > best[0]):
            best = (v.count('\n'), stripped)
    return (best[1], True) if best else (new, False)


NUMBERED_NOTE = 'removed the line-number prefixes copied from show.py output from NEW'


def strip_line_numbers(text):
    """Remove 'NNNN  ' prefixes copied from show.py output, if every non-blank line has one."""
    lines = text.split('\n')
    body = [l for l in lines if l.strip()]
    if not body or not all(LINE_PREFIX.match(l) for l in body):
        return None
    return '\n'.join(LINE_PREFIX.sub('', l, count=1) if l.strip() else l for l in lines)


def uniform_shift(new, region, lines=None, end_l=None):
    """D when NEW, written for the line range REGION, is indented D spaces more (D < 0: less) than the lines it
    replaces as a whole (v19: re-indented NEW often still parses, and 19% of the toolbench's indent_minus4 edits were
    applied silently wrong): NEW's least indented line is D off from REGION's, and so is its first line or every line
    NEW keeps from REGION (equal but for indentation). REGION all blank (an insertion; LINES and END_L, its last line,
    given): NEW starts below the next line's indentation (it would swallow that line). 0 when none of these holds,
    and when NEW changes nothing but the indentation (a deliberate re-nesting, numpy_10951)."""
    nb = [l for l in new.split('\n') if l.strip()]
    rb = [l for l in region if l.strip()]
    if not nb or [l.strip() for l in new.split('\n')] == [l.strip() for l in region]:
        return 0
    low = min(map(indent, nb))
    if not rb:
        nxt = next((l for l in lines[end_l:] if l.strip()), None) if lines is not None and end_l else None
        d = indent(nb[0]) - indent(nxt) if nxt is not None else 0
        return d if d < 0 and low - indent(nxt) == d else 0
    d = indent(nb[0]) - indent(rb[0])
    if d and low - min(map(indent, rb)) == d:
        return d
    kept = set()
    for l in nb:
        hits = [r for r in rb if r.strip() == l.strip()]
        if len(hits) == 1:
            kept.add(indent(l) - indent(hits[0]))
    if len(kept) == 1:
        d = kept.pop()
        if d and low - min(map(indent, rb)) == d:
            return d
    return 0


def indent_candidates(new, region, delta=0, old_lines=None, shift=0):
    """NEW as re-indented today, then repairs for a consistent indentation slip (tried only if the result does
    not parse): the body shift that makes NEW's lines equal to the replaced lines they repeat (v18 P6: anchored on
    the lines NEW keeps, not on its first line), shift to the region's first-line indentation, fix a first line
    pasted without indentation. Yields (note, text); a repair's note says how it shifted. SHIFT (uniform_shift of a
    line-range NEW): NEW moved back to the replaced lines' indentation comes first, even when NEW as written parses."""
    if shift:
        yield f' (indentation repaired: {-shift:+d} to the replaced lines\' indentation)', reindent(new, -shift)
    yield '', reindent(new, delta)
    nl, rl = new.split('\n'), trim_blank(region)
    nb = [l for l in nl if l.strip()]
    if not nb or not rl:
        return
    fi, ni = indent(rl[0]), indent(nb[0])
    if old_lines:  # per-line deltas between OLD and the file tell how NEW's body lines are off
        pairs = [indent(f) - indent(o) for o, f in zip(old_lines[1:], region[1:]) if o.strip() and f.strip()]
        if pairs:
            body = Counter(pairs).most_common(1)[0][0]
            if body != delta:
                yield f' (indentation repaired: {body:+d})', reindent(new, body, first_delta=delta)
                yield f' (indentation repaired: {body:+d})', reindent(new, body)
    first_i = next(i for i, l in enumerate(nl) if l.strip())
    anchors = []  # (shift, file line) for NEW lines after the first that repeat exactly one replaced line
    for line in nl[first_i + 1:]:
        same = [j for j, r in enumerate(region) if line.strip() and r.strip() == line.strip()]
        if len(same) == 1:
            anchors.append((indent(region[same[0]]) - indent(line), same[0]))
    if anchors:
        body = Counter(s for s, _ in anchors).most_common(1)[0][0]
        first = fi - ni + delta if not old_lines else delta
        if body != delta or first != delta:
            yield f' (indentation repaired: {body:+d} to match the lines NEW keeps)', \
                reindent(new, body, first_delta=first)
    if fi != ni:
        yield f' (indentation repaired: {fi - ni:+d} to the first replaced line)', reindent(new, fi - ni)
    rest = nb[1:]
    if rest:  # first line to the region's indentation; the rest as a block under or beside it
        head = nl[:first_i] + [' ' * fi + nl[first_i].lstrip()]
        tail = '\n'.join(nl[first_i + 1:])
        yield ' (indentation repaired: first line)', '\n'.join(head + [tail])
        # the least indented of the other lines is a sibling of the first line; failing that (a block opener
        # whose lines are all inside the block) it is one level deeper
        low = min(indent(l) for l in rest)
        for target in (fi, fi + 4) if nb[0].rstrip().endswith(':') else (fi,):
            yield f' (indentation repaired: {target - low:+d} after the first line)', \
                '\n'.join(head + [reindent(tail, target - low)])


def in_string(text, line):
    """Whether LINE (1-based) of TEXT is inside a multi-line string literal (its indentation is content)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    return any(isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)) and n.lineno < line <= n.end_lineno
               for n in ast.walk(tree))


def choose_new(path, content, start, end, new_variants, preferred_level, region=None, delta=0, old_lines=None,
               lines_mode=None):
    """Pick the NEW variant that keeps the file parseable, preferring the same escape level as OLD.
    Returns (after, level, note, errors, dead): errors [(SyntaxError, text, level, candidate, note)] when nothing
    parses; dead [(note, text, unreachable)] of repairs refused because they made code unreachable (v18 P6).
    LINES_MODE (the file's lines, the range's last line): a line-range edit of a file that parses, not inside a
    multi-line string: a uniformly shifted NEW is moved back first (uniform_shift, v19)."""
    ordered = sorted(new_variants, key=lambda v: (v[0] != preferred_level, v[0]))
    errors, dead = [], []
    for level, new in ordered:
        shift = uniform_shift(new, region, *lines_mode) if lines_mode and region else 0
        cands = indent_candidates(new, region, delta, old_lines, shift) if (region and path.endswith('.py')) \
            else [('', reindent(new, delta))]
        seen = set()
        for note, cand in cands:
            if cand in seen:
                continue
            seen.add(cand)
            after = content[:start] + cand + content[end:]
            ok, err = parses(path, after)
            if ok and note:
                bad = guards.unreachable(content, after)
                if bad:
                    dead.append((note, after, bad))
                    continue
            if ok:
                return after, level, note, None, []
            errors.append((err, after, level, cand, note))
    return None, None, None, errors, dead


def show_numbered(lines, nums, mark=None):
    """Print LINES[n-1] for n in NUMS (ascending; a gap prints '...'), the line MARK with '>>'."""
    prev = None
    for j in nums:
        if prev is not None and j > prev + 1:
            print('  ...')
        print(f'{">>" if j == mark else "  "}{j:5d}  {lines[j - 1][:200]}')
        prev = j


def placed_lines(a, b, el, n):
    """Line numbers to print for NEW placed at A..B with an error at EL in an N-line text: one line of context each
    side, the error line with its neighbours, at most NEW_SHOW_MAX lines of NEW (head and the error window)."""
    nums = set(range(max(1, a - 1), min(n, b + 1) + 1))
    if b - a + 1 > NEW_SHOW_MAX:
        nums = set(range(max(1, a - 1), a + 4)) | set(range(max(1, b - 1), min(n, b + 1) + 1))
    if el:
        nums |= set(range(max(1, el - 2), min(n, el + 2) + 1))
    return sorted(nums)


def report_syntax_error(errors, start_line, region):
    """NOT APPLIED for a NEW that would not parse: NEW's lines as they would be placed (numbered, the error line marked
    >>), and the indentation the replaced lines have (v18, audit #8: 61 syntax failures in 452 runs). The attempt
    shown is NEW as written (the repairs are guesses), the one that got furthest."""
    plain = [x for x in errors if not x[4]] or errors
    e, text, _, cand, _ = max(plain, key=lambda x: (x[0].lineno or 0, x[2]))
    lines = text.split('\n')
    a = start_line
    b = a + cand.count('\n')
    el = e.lineno
    where = f' (line {el - a + 1} of NEW)' if el and a <= el <= b else ''
    print(f'NOT APPLIED: the result would not parse: SyntaxError line {el}{where}: {e.msg}')
    if el or cand.strip():
        show_numbered(lines, placed_lines(a, b, el, len(lines)), el)
    body = [l for l in (region or []) if l.strip()]
    first = next((l for l in cand.split('\n') if l.strip()), '')
    if body:
        want, got = indent(body[0]), indent(first)
        print(f'expected indentation: the replaced lines start at {want} spaces; NEW starts at {got}.' if got != want
              else f'indentation: NEW starts at {want} spaces like the replaced lines.')


def report_dead(rel, dead, start_line, end_line, region):
    """NOT APPLIED for a NEW whose only parsing re-indentation leaves code after a return/raise (v18 P6:
    fastapi_14458 r1 call 10 wrote a check after `return True`, and the run failed)."""
    note, text, bad = dead[0]
    first, last, term, word = bad[0]
    lines = text.split('\n')
    how = note.strip().strip('()').replace('indentation repaired: ', '')
    print(f'NOT APPLIED: NEW is not indented like the lines it replaces, and the only re-indentation that parses '
          f'({how}) puts lines {first}-{last} after the `{word}` at line {term}, where they never run; nothing was '
          'changed.')
    show_numbered(lines, list(range(max(1, term - 1), min(len(lines), last) + 1))[:12], term)
    body = [l for l in (region or []) if l.strip()]
    if body:
        print(f'Write NEW with the file\'s indentation: line {start_line} starts at {indent(body[0])} spaces.')
    print(f'CLOSEST: {rel} {start_line} {end_line}')
    flow.fact(ok=False, path=rel, closest=(rel, start_line, end_line))
    remember_shown(rel, (start_line, end_line))
    return 1


_SCORES = {}


def window_scores(clines, old):
    """Similarity of OLD (stripped lines) to every window of the same length, best first: ([(ratio, start)], n).
    Windows are ranked on at most SCORE_LINES lines so long OLD texts stay fast on long files."""
    key = (len(clines), old)
    if key in _SCORES:
        return _SCORES[key]
    olines = old.strip('\n').split('\n')
    n = max(1, len(olines))
    k = min(n, SCORE_LINES)
    target = '\n'.join(l.strip() for l in olines[:k])
    stripped = [l.strip() for l in clines]
    scores = []
    for i in range(0, max(1, len(clines) - n + 1)):
        sm = difflib.SequenceMatcher(None, target, '\n'.join(stripped[i:i + k]))
        scores.append((sm.quick_ratio(), i, sm))
    best = sorted(scores, key=lambda s: -s[0])[:40]
    if k < n:  # re-rank the finalists on the whole text
        full = '\n'.join(l.strip() for l in olines)
        best = [(q, i, difflib.SequenceMatcher(None, full, '\n'.join(stripped[i:i + n]))) for q, i, _ in best]
    _SCORES[key] = sorted(((sm.ratio(), i) for q, i, sm in best), reverse=True), n
    return _SCORES[key]


def fuzzy_match(content, old):
    """A block whose lines each nearly equal OLD's lines (a dropped quote, a typo). Unique or nothing."""
    olines = old.strip('\n').split('\n')
    if sum(1 for l in olines if l.strip()) < FUZZY_MIN_LINES:
        return None
    clines = content.split('\n')
    scores, n = window_scores(clines, old)
    if not scores or scores[0][0] < FUZZY_MIN:
        return None
    ratio, i = scores[0]
    if any(r >= FUZZY_RIVAL and abs(j - i) >= n for r, j in scores[1:]):
        return None
    for o, f in zip(olines, clines[i:i + n]):
        if bool(o.strip()) != bool(f.strip()):
            return None
        if o.strip() and difflib.SequenceMatcher(None, o.strip(), f.strip()).ratio() < FUZZY_LINE_MIN:
            return None
    first = next(k for k, l in enumerate(olines) if l.strip())
    return i, i + n, indent(clines[i + first]) - indent(olines[first]), ratio


def closest(rel, content, old):
    """Print the file region most similar to OLD and return (start, end, similarity) 1-based; below CLOSEST_MIN
    print that nothing is similar and return None (v18 P5: a 0.55 region sent the model to unrelated code)."""
    clines = content.split('\n')
    scores, n = window_scores(clines, old)
    best, best_i = scores[0] if scores else (0.0, 0)
    if best < CLOSEST_MIN:
        print(f'no similar lines in {rel} either (best similarity {best:.2f}): read the code you mean to change with '
              'show.py and copy OLD from it.')
        flow.fact(ok=False, path=rel, similarity=round(best, 2))
        return None
    lo, hi = best_i, min(len(clines), best_i + n + 2)
    print(f'closest region (similarity {best:.2f}), lines {lo + 1}-{hi}:')
    for j in range(lo, hi):
        print(f'{j + 1:5d}  {clines[j][:200]}')
    start, end = best_i + 1, min(len(clines), best_i + n)
    flow.fact(ok=False, path=rel, closest=(rel, start, end), similarity=round(best, 2))
    remember_shown(rel, (lo + 1, hi))
    return start, end, best


def show_diff(rel, before, after):
    diff = list(difflib.unified_diff(before.split('\n'), after.split('\n'), f'a/{rel}', f'b/{rel}', n=1, lineterm=''))
    for line in diff[:40]:
        print(line)
    if len(diff) > 40:
        print(f'... {len(diff) - 40} more diff lines')


def closest_footer(rel, start, end):
    print('to replace those lines instead: '
          + argfix.obj('edit.py', {'file': rel, 'start': start, 'end': end, 'new': f'<new text for lines {start}-{end}>'}))
    print(f'CLOSEST: {rel} {start} {end}')
    remember_shown(rel, (start, end))


EXAMPLE_NEW = '        return 0 if self.amount is None else self.amount'


def statement_at(text, start):
    """The outermost statement (a def/class counts from its first decorator) whose first line is START, or None."""
    try:
        nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.stmt)
                 and min([d.lineno for d in getattr(n, 'decorator_list', [])] + [n.lineno]) == start]
    except (SyntaxError, ValueError):
        return None
    return max(nodes, key=lambda n: n.end_lineno) if nodes else None


def statement_end(rel, start, new):
    """A plausible END for a line-range edit given only START: the last line of the statement (or definition)
    that begins at START, else START + the number of lines of NEW - 1. For correction messages only."""
    try:
        node = statement_at(open(os.path.join(ROOT, rel), encoding='utf-8').read(), start)
        if node is not None:
            return node.end_lineno
    except (OSError, ValueError):
        pass
    return start + max(1, len(new_lines(new))) - 1


def _parses_alone(line):
    t = line.strip()
    for cand in (t, t + '\n    pass'):
        try:
            ast.parse(cand)
            return True
        except (SyntaxError, ValueError):
            pass
    return False


def new_lines(new):
    """NEW's lines as the edit would write them: show.py prefixes removed; a one-line NEW with literal \\n escapes
    is decoded unless it parses as it is (`sep = "\\n"`); blank lines at both ends dropped."""
    text, numbered = unnumbered(str(new))
    smart = decode_breaks(text) if not numbered else None
    if smart is not None:  # v20: literal \n outside strings (also next to real line breaks) are line breaks
        text = smart
    elif not numbered and '\n' not in text:
        decoded = variants(text)[-1][1]
        if '\n' in decoded.strip('\n') and not _parses_alone(text):
            text = decoded
    return trim_blank(text.split('\n'))


DEF_HEAD = re.compile(r'\s*(?:@[\w.]+.*|(?:async\s+)?def\s+(\w+)|class\s+(\w+))')


NO_END = -1


def start_above(rel, start, new, notes, key='start'):
    """START - 1 for a start-only edit whose NEW begins with line START - 1 (kept above START) followed by a line like
    line START: the model numbered from the def under its decorator (v20, E061 fastapi_9555: start=79 at
    `def is_gen_callable`, NEW = `@cached_property` + the def: the decorator was duplicated). Else START."""
    try:
        lines = open(os.path.join(ROOT, argfix.clean_path(rel)), encoding='utf-8').read().split('\n')
    except (OSError, ValueError):
        return start
    nl = new_lines(new)
    if not 2 <= start <= len(lines) or len(nl) < 2:
        return start
    above, here = lines[start - 2].strip(), lines[start - 1].strip()
    if above and nl[0].strip() == above and nl[0].strip() != here and not PLAIN.match(above) and \
            _pair(nl[1].strip(), here) >= ALIGN_T:
        notes.append(f'NEW starts with line {start - 1} ({above[:40]}), the line above {key}={start}, so the edit '
                     f'starts at line {start - 1}')
        return start - 1
    return start


def default_end(rel, start, new):
    """END for a line-range edit given only START (v19, infer_end): START itself when NEW is one line that fits there;
    (end, note) when the lines NEW rewrites are clear; NO_END otherwise (refused with the likely range: replacing one
    line with a block duplicated code in fastapi_14873 and orphaned rich_3930's `cell_len` body). None when the file
    cannot be read or START is outside it (the usual error message then applies)."""
    try:
        text = open(os.path.join(ROOT, argfix.clean_path(rel)), encoding='utf-8').read()
    except (OSError, ValueError):
        return None
    lines = text.split('\n')
    if not 1 <= start <= len(lines):
        return None
    nl = new_lines(new)
    if not nl:
        return start
    end, sure, why, _ = infer_end(text, start, nl, rel.endswith('.py'), lead_blank=lead_blank(new))
    if sure and why == BLANK_INSERT:  # keep line START (blank) above NEW and a blank line below it
        body = '\n'.join(nl)
        below = '\n\n' if start < len(lines) and lines[start].strip() else ''  # (the edit drops one final newline)
        return start, (f'no end given: NEW is new code and line {start} is blank, so NEW was inserted there (the blank '
                       'line kept)'), '\n' + body + below
    if sure and end == start and len(nl) == 1:
        return start
    if sure:
        return end, f'no end given: {why}'
    return NO_END


# ---------------------------------------------------------------- start without end: the lines NEW rewrites (v19)

ALIGN_T = 0.6        # two different lines pair up (a modified line) from this similarity
ALIGN_DEL = 0.6      # cost of a replaced line NEW does not account for (x its weight)
ALIGN_FUZZY = 0.8    # a similar pair scores this x similarity (an equal pair: 1)
ALIGN_DUP = 1.0      # cost of each NEW line that repeats one of the lines kept right after the range
END_WINDOW = 10      # the end is searched in START..START + len(NEW) + END_WINDOW
ALIGN_MAX = 300      # longer NEWs are not aligned (the refusal shows the statement at START)
DUP_AFTER = 3        # lines after the range checked for a line NEW repeats
INSERT_SIM = 0.85    # a NEW line this similar to a line after START..END rewrites it: not an insertion
PLAIN = re.compile(r'(?:return|pass|else:|try:|finally:|break|continue|raise|[)\]}][)\]},:;]*|"""|\'\'\'|'
                   r'return (?:None|True|False|self|result)|except Exception:)$')
BARE_DECORATOR = re.compile(r'@[\w.]+$')


def _weight(t):
    return 0.4 if PLAIN.match(t) or len(t) < 4 else 1.0


def _anchor(t):
    """A line distinctive enough to locate NEW's end by (comments count)."""
    return len(t) >= 6 and not PLAIN.match(t)


def _pair(a, b):
    """Similarity of two stripped lines, 0.0 below ALIGN_T."""
    if a == b:
        return 1.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if sm.real_quick_ratio() < ALIGN_T or sm.quick_ratio() < ALIGN_T:
        return 0.0
    r = sm.ratio()
    return r if r >= ALIGN_T else 0.0


def align_prefixes(nl, rl):
    """How well all of NEW's lines NL rewrite each prefix RL[:j] of the lines after START (stripped, non-blank): an
    alignment in order where an equal pair scores 1, a similar pair ALIGN_FUZZY x similarity, a line of RL[:j] left
    unpaired -ALIGN_DEL (lines are weighted: brackets, else:, return... count less), a NEW line left unpaired 0.
    [(score, pairs)] for j = 0..len(RL); pairs = the RL lines paired with a NEW line."""
    prev = [(0.0, 0)]
    for r in rl:
        prev.append((prev[-1][0] - ALIGN_DEL * _weight(r), 0))
    for a in nl:
        cur = [prev[0]]
        for j, r in enumerate(rl, 1):
            best = max(prev[j], (cur[j - 1][0] - ALIGN_DEL * _weight(r), cur[j - 1][1]))
            s = _pair(a, r)
            if s:
                best = max(best, (prev[j - 1][0] + (1.0 if s == 1.0 else ALIGN_FUZZY * s) * _weight(r),
                                  prev[j - 1][1] + 1))
            cur.append(best)
        prev = cur
    return prev


def lead_blank(new):
    """Whether NEW, as the edit writes it, starts with a blank line."""
    text, _ = unnumbered(str(new))
    if '\n' not in text:
        text = variants(text)[-1][1]
    return bool(text.split('\n')[0].strip() == '' and text.strip())


def infer_end(text, start, nl, py, lead_blank=True):
    """infer_end_ (below); a range from a blank line START that NEW does not start with is never sure (rich_3052: the
    blank lines above a def would be deleted)."""
    end, sure, why, alt = infer_end_(text, start, nl, py)
    if sure and end > start and not lead_blank and not text.split('\n')[start - 1].strip():
        return end, False, f'line {start} is blank and NEW does not start with a blank line', alt
    return end, sure, why, alt


def infer_end_(text, start, nl, py):
    """(end, sure, why, alternative end or None) for a line-range edit given START and NEW's lines NL but no END.
    A candidate END must keep the file parsing (a .py file that parses) with no code after a return/raise, and NEW
    must not repeat the lines kept right after it (DUP_AFTER). Sure, in this order:
    - NEW's last line is a distinctive line of the file at END (once in the window), the alignment agrees
      (fastapi_14583/rich_4079: NEW = the lines up to END, some of them rewritten);
    - NEW starts with lines START..END unchanged and none of its other lines is in the window: an insertion after END
      (fastapi_14873: `)` + the new lines), the alignment agrees;
    - NEW rewrites START..END line for line (each pair equal or similar), clearly better than any other END
      (fastapi_13207: a 3-line assignment);
    - NEW redefines the def/class that begins at START, and the alignment ends where it does;
    - one-line NEW: equal to a later line END (lines START..END-1 deleted), the alignment agrees; else START when it
      fits there (at that line's indentation).
    Otherwise not sure: the best-aligned END (or a rule's END that the alignment does not confirm) for the refusal."""
    lines = text.split('\n')
    hi = min(len(lines), start + len(nl) + END_WINDOW)
    check = py and parses('x.py', text)[0]
    N = [l.strip() for l in nl if l.strip()]

    def fits(k):
        if not check:
            return True
        after = '\n'.join(lines[:start - 1] + nl + lines[k:])
        if parses('x.py', after)[0] and not guards.unreachable(text, after):
            return True
        # v20 (E058 rich_3471/3772/2725, fastapi_14297: NEW indented 2 or 4 more than the file): the edit moves a NEW
        # shifted as a whole back (uniform_shift), so an end fits when NEW fits there moved back
        d = uniform_shift('\n'.join(nl), lines[start - 1:k], lines, k)
        if not d:
            return False
        after = '\n'.join(lines[:start - 1] + reindent('\n'.join(nl), -d).split('\n') + lines[k:])
        return parses('x.py', after)[0] and not guards.unreachable(text, after)

    def dup(k):  # (v20: a bare decorator such as @property starts most methods: not a repeated line, rich_3675)
        kept = {l.strip() for l in lines[k:k + DUP_AFTER] if l.strip()}
        return sum(1 for x in N if x in kept and _anchor(x) and not BARE_DECORATOR.match(x))

    if not lines[start - 1].strip() and len(N) >= 2 and fits(start) and not any(
            _anchor(x) and not BARE_DECORATOR.match(x) and (x == r or _pair(x, r) >= INSERT_SIM) for x in N
            for r in (l.strip() for l in lines[start:hi] if l.strip())):
        # v20 (E061 fastapi_14306, E060 fastapi_15280, E059 fastapi_15030, E058 rich_3777): new code at a blank line
        return start, True, BLANK_INSERT, None
    idx = [k for k in range(start, hi + 1) if lines[k - 1].strip()]
    cands = []
    if len(N) <= ALIGN_MAX:
        scores = align_prefixes(N, [lines[k - 1].strip() for k in idx])
        if not lines[start - 1].strip():
            cands.append((scores[0][0] - ALIGN_DUP * dup(start), start, 0, 0))
        cands += [(scores[j][0] - ALIGN_DUP * dup(k), k, scores[j][1], j) for j, k in enumerate(idx, 1)]
        cands.sort(key=lambda c: (-c[0], c[1]))
    ranked = [c for c in cands[:6] if fits(c[1])] or cands
    if not ranked:  # NEW too long to align: the statement at START, else as many lines as NEW has
        node = statement_at(text, start) if py else None
        k = node.end_lineno if node is not None else min(len(lines), start + len(nl) - 1)
        return k, False, f'the {"statement" if node is not None else "lines"} at line {start}', None
    guess = ranked[0][1]
    ok = lambda k: fits(k) and not dup(k)  # noqa: E731
    last = N[-1]
    if len(N) == 1:
        pos = [k for k in range(start, hi + 1) if lines[k - 1].strip() == last]
        if _anchor(last) and len(pos) == 1 and pos[0] > start and pos[0] == guess and ok(pos[0]):
            return pos[0], True, (f'NEW is line {pos[0]}, so it replaced lines {start}-{pos[0]} (lines {start}-'
                                  f'{pos[0] - 1} deleted)' if pos[0] - 1 > start else
                                  f'NEW is line {pos[0]}, so it replaced lines {start}-{pos[0]} (line {start} deleted)'), None
        here = ' ' * indent(lines[start - 1]) + nl[0].lstrip() if lines[start - 1].strip() else nl[0]
        if fits(start) or guess == start or (check and parses('x.py', '\n'.join(lines[:start - 1] + [here] +
                                                                              lines[start:]))[0]):
            return start, True, '', None  # (a one-line NEW indented wrongly: the edit repairs it)
        k = simple_statement_end(text, start, N, check)
        if k and ok(k):  # v20 (E060 fastapi_13207: a one-line assignment for a 3-line one)
            return k, True, f'NEW rewrites the statement at line {start}, lines {start}-{k}', None
        return guess, False, 'the lines NEW matches best', None
    pos = [k for k in range(start, hi + 1) if lines[k - 1].strip() == last]
    if _anchor(last) and len(pos) == 1 and N.count(last) == 1 and fits(pos[0]):
        k = pos[0]
        if k == guess and not dup(k):
            return k, True, f'NEW ends with line {k}, so it replaced lines {start}-{k}', None
        if dup(k) and guess > k:  # NEW also repeats a line kept after K (fastapi_14873: a swap): the longer range
            return guess, False, 'the lines NEW matches best', k
        return k, False, f'NEW ends with line {k}', (guess if guess != k else None)
    m = 0
    while m < len(N) and start + m <= len(lines) and lines[start - 1 + m].strip() == N[m]:
        m += 1
    insert = None
    if m and m < len(N):
        k = start + m - 1
        later = {lines[j - 1].strip() for j in range(k + 1, hi + 1) if lines[j - 1].strip()}
        nxt = [lines[j - 1].strip() for j in range(k + 1, min(len(lines), k + len(N) - m) + 1)]
        modified = any(_anchor(x) and _pair(x, r) >= INSERT_SIM for x in N[m:] for r in nxt)  # a rewrite, not new
        if not modified and not any(x in later and not PLAIN.match(x) for x in N[m:]) and ok(k):
            span = f'line {start}' if k == start else f'lines {start}-{k}'
            if k == guess:
                return k, True, (f'NEW repeats {span} and adds {len(N) - m} new line{"s" if len(N) - m > 1 else ""}, '
                                 f'so it replaced {span} only; the lines after {"it" if k == start else "them"} '
                                 'are kept'), None
            insert = (k, False, f'NEW repeats {span} and the rest is new (an insertion after line {k})', guess)
    score, k, pairs, j = ranked[0]
    if insert is None and pairs == j == len(N) and ok(k):  # (an insertion that also reads as a rewrite: unsure)
        rivals = [c[0] for c in ranked[1:] if c[1] != k]
        if not rivals or rivals[0] < score - 0.5:
            return k, True, f'NEW rewrites lines {start}-{k} line for line, so it replaced them', None
    if insert:
        return insert
    k = simple_statement_end(text, start, N, check)
    if k and k == guess and ok(k):  # v20 (E060 fastapi_14791: a 14-line dict assignment rewritten in 16 lines)
        return k, True, f'NEW rewrites the statement at line {start}, lines {start}-{k}', None
    body = [l for l in N if not l.startswith('@')]
    head = DEF_HEAD.match(body[0]) if len(body) > 1 else None  # a def with a body, not just a new header line
    name = head and (head.group(1) or head.group(2))
    if name and check:  # v17: NEW redefines the def/class at START: the whole definition, when the alignment agrees
        for n in ast.walk(ast.parse(text)):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name and \
                    start in (n.lineno, min([x.lineno for x in n.decorator_list] + [n.lineno])):
                if n.end_lineno == guess and ok(guess):
                    return guess, True, f'NEW redefines {name}, so it replaced its lines {start}-{guess}', None
                break
    return guess, False, 'the lines NEW matches best', None


SIMPLE = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Return, ast.Raise, ast.Assert, ast.Import,
          ast.ImportFrom, ast.Delete)
BLANK_INSERT = 'BLANK_INSERT'   # infer_end_'s why for new code at a blank line (default_end turns it into an insertion)


def simple_statement_end(text, start, N, check):
    """The last line of the multi-line simple statement (an assignment, call, return, import...: no block) that begins
    at line START, when NEW's first line pairs with line START; else None (v20)."""
    if not check or not N:
        return None
    node = statement_at(text, start)
    if node is None or not isinstance(node, SIMPLE) or node.end_lineno <= start:
        return None
    first = text.split('\n')[start - 1].strip()
    head = lambda t: re.match(r'[^(\[{=]*[(\[{=]', t)  # noqa: E731  (the target / callee up to its first bracket)
    same_head = head(N[0]) and head(first) and len(head(N[0]).group(0)) >= 4 and \
        head(N[0]).group(0).replace(' ', '') == head(first).group(0).replace(' ', '')
    return node.end_lineno if same_head or _pair(N[0], first) >= ALIGN_T else None


class NoEnd(Exception):
    """start without end and a multi-line NEW that does not redefine the def/class at START."""

    def __init__(self, rel, start, new):
        super().__init__(rel)
        self.rel, self.start, self.new = rel, start, new


SHOW_MAX = 14  # lines printed by the start-without-end refusal (longer statements: head and tail)


def report_no_end(e):
    """NOT APPLIED for start without end and a NEW whose range is unclear (infer_end): the lines it most likely
    replaces and the ready call, plus the other likely range when two readings compete (exit 2: the call was
    incomplete, nothing was matched)."""
    rel, start = e.rel, e.start
    text = open(os.path.join(ROOT, rel), encoding='utf-8').read()
    lines = text.split('\n')
    nl = new_lines(e.new) or ['']
    end = alt = why = None
    body = [l for l in nl if l.strip() and not l.strip().startswith('@')]
    m = DEF_HEAD.match(body[0]) if body else None
    name = m and (m.group(1) or m.group(2))
    if name and rel.endswith('.py'):  # rich_3052: NEW redefines X, but START is not X's line (numbers moved)
        try:
            defs = [n for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                             ast.ClassDef)) and n.name == name]
        except (SyntaxError, ValueError):
            defs = []
        near = min(defs, key=lambda n: abs(n.lineno - start)) if defs else None
        if near is not None and start not in (near.lineno, min([d.lineno for d in near.decorator_list] + [near.lineno])):
            start = min([d.lineno for d in near.decorator_list] + [near.lineno])
            end = near.end_lineno
            why = (f'NEW redefines {name}, which begins at line {start}, not {e.start}: line numbers move after every '
                   'edit, read them again')
    if end is None:
        end, _, why, alt = infer_end(text, start, nl, rel.endswith('.py'), lead_blank=lead_blank(e.new))
        why = why.split(', so ')[0]
    end = max(start, min(end, len(lines)))
    if len(nl) == 1:
        print(f'NOT APPLIED: "start" without "end": NEW is one line, and line {e.start} alone cannot be replaced by it '
              f'(the code there continues to line {end}), so nothing was changed. Give "end", the last line NEW '
              'replaces.')
    else:
        print(f'NOT APPLIED: "start" without "end" and a NEW of {len(nl)} lines: replacing line {e.start} alone would '
              'leave the old code after it in place (duplicated), so nothing was changed. Give "end", the last line NEW '
              'replaces.')
    print(f'lines {start}-{end}, {why}:')
    span = list(range(start, end + 1))
    if len(span) > SHOW_MAX:
        span = span[:SHOW_MAX - 4] + [None] + span[-3:]
    for j in span:
        print(f'  ... ({end - start + 1 - (SHOW_MAX - 1)} more lines)' if j is None else f'{j:5d}  {lines[j - 1][:200]}')
    print('if NEW replaces exactly these lines: '
          + argfix.obj('edit.py', {'file': rel, 'start': start, 'end': end, 'new': '<your NEW unchanged>'}))
    if alt and alt != end:
        print(f'if it replaces lines {start}-{alt} ({lines[alt - 1].strip()[:60]!r} is line {alt}): '
              + argfix.obj('edit.py', {'file': rel, 'start': start, 'end': alt, 'new': '<your NEW unchanged>'}))
    print('otherwise set end to the last line NEW replaces (to insert code, keep the lines that stay in NEW).')
    print(f'CLOSEST: {rel} {start} {end}')
    flow.fact(ok=False, path=rel, closest=(rel, start, end))
    remember_shown(rel, (start, max(end, alt or end)))
    return 2


def parse_object(d, notes):
    """The object form: {"file", "old", "new"} or {"file", "start", "end", "new"}. Values are used as given.
    "old" together with start/end: the old text is used (start/end ignored, NOTE line). v20: the keys of further hunks
    (start2 end2 new2, old2 new2, ... up to MAX_HUNKS) are set aside in _GROUPS (extra_hunks() reads them)."""
    groups = {}
    for k in list(d):
        m = EXTRA_KEY.match(k)
        if m:
            groups.setdefault(int(m.group(2)), {})[m.group(1)] = d.pop(k)
    _GROUPS.clear()
    _GROUPS.update(groups)
    if 'old' in d and ('start' in d or 'end' in d) and d['old'].strip():
        d = {k: v for k, v in d.items() if k not in ('start', 'end')}
        notes.append('old and start/end given; replaced the old text (start/end ignored)')
    argfix.exclusive(d, 'edit.py', ('old',), ('start', 'end'))
    if not str(d.get('file', '')).strip() and any(k in d for k in ('old', 'start', 'end')) and 'new' in d:
        guess = infer_target(d)  # v18: the file the model is working on, when unambiguous
        if guess:
            d = dict(d, file=guess[0])
            notes.append(guess[1])
    inferred = False
    if 'new' in d and str(d.get('file', '')).strip() and not any(k in d for k in ('old', 'start', 'end')):
        guess = anchor_start(argfix.clean_path(d['file']), d['new'])  # v20: where NEW belongs, when it tells
        if guess:
            d = dict(d, start=str(guess[0]))
            notes.append(guess[1])
            inferred = True
    if 'start' in d or 'end' in d:
        rel = argfix.clean_path(d['file']) if d.get('file', '').strip() else 'billing/invoice.py'
        example = {'file': rel, 'start': d.get('start') or '120', 'end': d.get('end') or '122', 'new': EXAMPLE_NEW}
        s0 = argfix.line_number(d.get('start', ''))
        if s0 is not None and not str(d.get('end', '')).strip() and d.get('file', '').strip():
            if 'new' in d and not inferred:
                s0 = fix_stale(rel, s0, None, d['new'], notes)[0]
                s0 = start_above(rel, s0, d['new'], notes)
                d = dict(d, start=str(s0))
            elif inferred:
                _STALE_DONE.append(True)  # (the number was found in the file as it is now)
            e0 = default_end(rel, s0, d.get('new', ''))  # v16: start alone is one line (note 11: 87 USAGE ERRORs)
            if e0 == NO_END and 'new' in d:
                raise NoEnd(rel, s0, d['new'])
            if isinstance(e0, tuple) and 'new' in d:  # v19: the lines NEW rewrites are clear (infer_end)
                _AUTO_END.append((rel, s0, d['new']))
                d = dict(d, end=str(e0[0]))
                if len(e0) > 2:  # v20: an insertion at a blank line: NEW with the blank lines around it
                    d['new'] = e0[2]
                notes.append(e0[1])
            elif e0 is not None and e0 != NO_END and 'new' in d:
                d = dict(d, end=str(e0))
                notes.append(f'no end given: replaced line {s0} only')
            else:
                e0 = statement_end(rel, s0, d.get('new', ''))
                example.update(start=str(s0), end=str(e0), new=f'<new text for lines {s0}-{e0}>')
        argfix.require(d, 'edit.py', 'file', 'start', 'end', example=example)
        if 'new' not in d:
            argfix.require(d, 'edit.py', 'new', example=example)
        start, end = argfix.line_number(d['start']), argfix.line_number(d['end'])
        if start is None or end is None:
            if argfix.is_placeholder(d['start']) or argfix.is_placeholder(d['end']):
                raise Misuse(f'{d["start"]!r} {d["end"]!r} are placeholders: pass real line numbers from show.py, e.g. '
                             + argfix.obj('edit.py', {**example, 'start': '120', 'end': '122'}))
            bad = [v for v, n in ((d['start'], start), (d['end'], end)) if n is None]
            raise Misuse(f'"start" and "end" must be line numbers, got {bad!r}; find them with show.py first, e.g. '
                         + argfix.obj('edit.py', {**example, 'start': '120', 'end': '122'}))
        if end == start - 1 and str(d.get('new', '')).strip():  # v20: an empty range: insert NEW before line START
            ins = insert_before(rel, start, d['new'], notes)
            if ins:
                return 'lines', d['file'], ins[0], ins[0], ins[1], notes
        if end < start:
            raise Misuse(f'"end" ({end}) is before "start" ({start}); e.g. '
                         + argfix.obj('edit.py', {**example, 'start': str(end), 'end': str(start)}))
        return 'lines', d['file'], start, end, d['new'], notes
    example = {'file': argfix.clean_path(d['file']) if d.get('file', '').strip() else 'billing/invoice.py',
               'old': 'if self.amount == None:', 'new': 'if self.amount is None:'}
    argfix.require(d, 'edit.py', 'file', example=example)
    if 'old' in d or 'new' in d:  # echo what was given (if short), a placeholder for the rest
        example = {'file': example['file']}
        for key, hole in (('old', '<exact current text>'), ('new', '<replacement text>')):
            example[key] = d[key] if key in d and len(d[key]) <= 200 else hole
    for key in ('old', 'new'):  # "new" may be empty (a deletion) but must be given
        if key not in d:
            raise Misuse(f'edit.py needs the key {argfix.SHOWN_KEY.get(key, key)} (valid keys: '
                         f'{", ".join(argfix.shown_keys("edit.py"))}). E.g. ' + argfix.obj('edit.py', example))
    return 'text', d['file'], d['old'], d['new'], None, notes


ANCHOR_MIN_CHARS = 8   # v20: NEW's first line this long (stripped) locates an edit given without old or start


def anchor_start(rel, new):
    """(line, note) where an edit given NEW but neither old nor start belongs (v20: 7 USAGE ERRORs `needs the key old`
    in E057-E061, e.g. fastapi_14964 sent `from typing import Any` + a new import): NEW's first line, when it stands
    exactly once in the file; else the def/class NEW (re)defines, when the file defines that name exactly once. The
    start-only inference then decides the end (or refuses with the likely range). None otherwise."""
    try:
        text = open(os.path.join(ROOT, rel), encoding='utf-8').read()
    except (OSError, ValueError):
        return None
    nl = new_lines(new)
    if not nl:
        return None
    lines = text.split('\n')
    first = nl[0].strip()
    if len(first) >= ANCHOR_MIN_CHARS and not PLAIN.match(first) and not BARE_DECORATOR.match(first):
        hits = [i + 1 for i, l in enumerate(lines) if l.strip() == first]
        if len(hits) == 1:
            return hits[0], f'no old or start given; NEW starts with line {hits[0]} of {rel}, so the edit starts there'
    body = [l for l in nl if l.strip() and not l.strip().startswith('@')]
    m = DEF_HEAD.match(body[0]) if len(body) > 1 else None
    name = m and (m.group(1) or m.group(2))
    if name and rel.endswith('.py'):
        try:
            defs = [n for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                             ast.ClassDef)) and n.name == name]
        except (SyntaxError, ValueError):
            defs = []
        if len(defs) == 1:
            n = defs[0]
            at = min([x.lineno for x in n.decorator_list] + [n.lineno])
            static = any(isinstance(x, ast.Name) and x.id in ('staticmethod', 'classmethod') for x in n.decorator_list)
            if not nl[0].strip().startswith('@') and not static:
                at = n.lineno  # NEW without decorators: keep the def's own (a static/class method's change it)
            return at, f'no old or start given; NEW redefines {name}, which begins at line {at}, so the edit starts there'
    return None


def insert_before(rel, start, new, notes):
    """(line, NEW for that line) inserting NEW before line START (end = start - 1: an empty range, v20 E059
    fastapi_15280), or None when START is not a line of the file."""
    try:
        lines = open(os.path.join(ROOT, rel), encoding='utf-8').read().split('\n')
    except (OSError, ValueError):
        return None
    body = str(new)[:-1] if str(new).endswith('\n') else str(new)  # (one final newline dropped, as for every NEW)
    if 1 <= start <= len(lines):
        notes.append(f'end = start - 1: NEW was inserted before line {start} (nothing replaced)')
        return start, body + '\n' + lines[start - 1] + '\n'
    if start == len(lines) + 1 and lines:
        notes.append(f'end = start - 1: NEW was added after the last line, {len(lines)}')
        return len(lines), lines[-1] + '\n' + body + '\n'
    return None


_AUTO_END = []      # (rel, start, new) when parse_object inferred the end of a start-only edit (v19)
AUTO_NOTE = 'no end given: NEW '  # its note, printed after the edit succeeded
_STALE_DONE = []    # fix_stale already ran for this call


def parse(argv):
    """Return (mode, rel, a, b, c, notes): ('lines', rel, start, end, new) or ('text', rel, old, new, None); a line
    range with stale numbers is shifted (fix_stale). Further hunks of a multi-edit go to _EXTRA (v20)."""
    mode, rel, a, b, c, notes = _parse(argv)
    if mode == 'lines' and not _STALE_DONE:
        a, b = fix_stale(argfix.clean_path(rel), a, b, c, notes)
    if _GROUPS:
        extra_hunks(argfix.clean_path(rel), notes)
    return mode, rel, a, b, c, notes


EXTRA_KEY = re.compile(r'(start|end|new|old)([2-9])$')
_GROUPS = {}        # {n: {start/end/new/old: value}} of hunks 2.. of a multi-edit call (parse_object)
_EXTRA = []         # their Hunks (extra_hunks)


def extra_hunks(rel, notes):
    """Hunks 2.. of a multi-edit call (v20): start{n} end{n} new{n} (line numbers of the file before this call, as
    hunk 1's) or old{n} new{n}. start{n} without end{n}: the end is inferred as for hunk 1 when it is clear (a NOTE),
    else a USAGE ERROR naming the likely end."""
    for n in sorted(_GROUPS):
        g = _GROUPS[n]
        ex = argfix.obj('edit.py', {'file': rel, 'start': '120', 'end': '121', 'new': EXAMPLE_NEW, f'start{n}': '140',
                                    f'end{n}': '140', f'new{n}': '        return total'})
        if n > MAX_HUNKS:
            raise Misuse(f'at most {MAX_HUNKS} hunks per call (start..start{MAX_HUNKS}, or old..old{MAX_HUNKS}); '
                         'send the rest in another call')
        if 'new' not in g:
            raise Misuse(f'hunk {n} needs new{n} (its new text), e.g. {ex}')
        if str(g.get('old', '')).strip():
            if 'start' in g or 'end' in g:
                notes.append(f'old{n} and start{n}/end{n} given; replaced the old{n} text')
            _EXTRA.append(Hunk('text', g['old'], g['new'], None, None, f'old{n}/new{n}'))
            continue
        s = argfix.line_number(g.get('start', ''))
        e = argfix.line_number(g.get('end', ''))
        if s is None:
            raise Misuse(f'hunk {n} needs start{n} and end{n} (line numbers) or old{n}, e.g. {ex}')
        if e is None and str(g.get('end', '')).strip():
            raise Misuse(f'end{n} must be a line number, got {g["end"]!r}; e.g. {ex}')
        if e is None:
            s = fix_stale(rel, s, None, g['new'], notes)[0]
            s = start_above(rel, s, g['new'], notes, f'start{n}')
            e0 = default_end(rel, s, g['new'])
            if e0 == NO_END:
                try:
                    text = open(os.path.join(ROOT, rel), encoding='utf-8').read()
                    guess = infer_end(text, s, new_lines(g['new']), rel.endswith('.py'),
                                      lead_blank=lead_blank(g['new']))[0]
                except (OSError, ValueError):
                    guess = s
                raise Misuse(f'start{n}={s} without end{n} and a new{n} of several lines: give end{n}, the last line '
                             f'new{n} replaces (lines {s}-{guess} look like it)')
            if isinstance(e0, tuple):
                e = e0[0]
                if len(e0) > 2:
                    g['new'] = e0[2]
                notes.append(f'hunk {n}: {e0[1]}')
            else:
                e = e0 if e0 is not None else s
        else:
            s, e = fix_stale(rel, s, e, g['new'], notes)
        if e < s:
            raise Misuse(f'end{n} ({e}) is before start{n} ({s})')
        _EXTRA.append(Hunk('lines', s, e, g['new'], None, f'start{n}/end{n}/new{n}'))


def _parse(argv):
    d, note = argfix.object_args(argv, 'edit.py')
    if d is not None:
        return parse_object(d, [note] if note else [])
    argv, note = argfix.unpack(argv, 'edit.py')
    notes = [note] if note else []
    if len(argv) == 1 and argv[0].lstrip().startswith('--lines'):  # "--lines FILE 5 9\n<new text>"
        head, _, new = argv[0].lstrip().partition('\n')
        argv = argfix.split(head) + ([new] if new or len(argfix.split(head)) == 4 else [])
        notes.append('split "--lines FILE START END" and NEW into separate strings')
    example_new = '        if self.amount is None:\n            return 0'
    if argv and argv[0] == '--lines':
        rel = argfix.clean_path(argv[1]) if len(argv) > 1 else 'rich/console.py'
        if len(argv) == 4 and argfix.line_range(argv[2]):  # --lines FILE 5-9 NEW
            r = argfix.line_range(argv[2])
            argv = argv[:2] + [str(r[0]), str(r[1])] + argv[3:]
        if len(argv) != 5:
            if len(argv) > 5 and argfix.line_number(argv[2]) is not None \
                    and argfix.line_number(argv[3]) is not None:
                joined = '\n'.join(argv[4:])
                raise Misuse('NEW must be ONE string (join its lines with \\n), e.g. '
                             + argfix.call('edit.py', '--lines', rel, argv[2], argv[3], joined[:120]))
            raise Misuse('--lines needs FILE START END NEW, e.g. '
                         + argfix.call('edit.py', '--lines', rel, '120', '122', example_new))
        start, end = argfix.line_number(argv[2]), argfix.line_number(argv[3])
        if start is None or end is None:
            bad = [a for a, v in ((argv[2], start), (argv[3], end)) if v is None]
            raise Misuse(f'START and END must be line numbers, got {bad!r}; find them with show.py first, e.g. '
                         + argfix.call('edit.py', '--lines', rel, '120', '122', example_new))
        if end < start:
            raise Misuse(f'END ({end}) is before START ({start}); e.g. '
                         + argfix.call('edit.py', '--lines', rel, str(end), str(start), example_new))
        return 'lines', argv[1], start, end, argv[4], notes
    if len(argv) == 4 and argfix.line_number(argv[1]) is not None and argfix.line_number(argv[2]) is not None:
        notes.append('read FILE START END NEW as --lines FILE START END NEW')
        return 'lines', argv[0], argfix.line_number(argv[1]), argfix.line_number(argv[2]), argv[3], notes
    if len(argv) == 4 and (argfix.is_placeholder(argv[1]) or argfix.is_placeholder(argv[2])):
        raise Misuse(f'{argv[1]!r} {argv[2]!r} are placeholders: to replace lines, pass real line numbers from '
                     'show.py, e.g. ' + argfix.call('edit.py', '--lines', argfix.clean_path(argv[0]), '120', '122',
                                                   example_new))
    if len(argv) != 3:
        if len(argv) < 3:
            raise Misuse(f'edit.py needs FILE OLD NEW (got {len(argv)} argument{"s" if len(argv) != 1 else ""}), e.g. '
                         + argfix.call('edit.py', 'rich/console.py', 'if self.amount == None:', 'if self.amount is None:'))
        raise Misuse(f'edit.py takes exactly FILE OLD NEW (got {len(argv)} arguments); multi-line text goes in ONE '
                     'string with \\n, e.g. '
                     + argfix.call('edit.py', argfix.clean_path(argv[0]), 'def total(self):\n        return 0',
                                   'def total(self):\n        return 1'))
    return 'text', argv[0], argv[1], argv[2], None, notes


def main(argv):
    try:
        mode, rel_arg, a, b, c, notes = parse(argv)
        rel = argfix.clean_path(rel_arg)
        if PROTECTED.search(rel):
            print(f'REFUSED: {rel} is a test or test-config file. The grader resets these; fix the source code instead.')
            return 1
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            if os.path.isdir(path):
                print(f'{rel} is a directory; edit.py needs a file (list files with outline.py {rel})')
                return 1
            try:
                found_rel, note = argfix.resolve(rel, 'edit.py', ('if x == None:', 'if x is None:'))
            except Misuse as e:
                if argfix.is_placeholder(rel):
                    raise
                print(f'{e}\nTo add a new file use ' + argfix.obj('create.py', {'file': rel, 'content': '<file content>'}))
                return 1
            if PROTECTED.search(found_rel) or not os.path.isfile(os.path.join(ROOT, found_rel)):
                print(f'{rel}: not found (paths are relative to /workspace)')
                return 1
            rel, path = found_rel, os.path.join(ROOT, found_rel)
            notes.append(note)
        new = c if mode == 'lines' else b
        if ANGLE.match(new):
            raise Misuse(f'NEW {new!r} is a placeholder: pass the actual replacement text, e.g. '
                         + argfix.call('edit.py', rel, 'if self.amount == None:', 'if self.amount is None:'))
    except NoEnd as e:
        if PROTECTED.search(e.rel):
            print(f'REFUSED: {e.rel} is a test or test-config file. The grader resets these; fix the source code instead.')
            return 1
        return report_no_end(e)
    except Stale as e:
        if PROTECTED.search(e.rel):
            print(f'REFUSED: {e.rel} is a test or test-config file. The grader resets these; fix the source code instead.')
            return 1
        print(e.text)
        if e.current:  # the lines' current numbers are known
            print(f'CLOSEST: {e.rel} {e.start} {e.end}')
            flow.fact(ok=False, path=e.rel, closest=(e.rel, e.start, e.end))
            remember_shown(e.rel, (e.start, e.end))
        else:
            flow.fact(ok=False, path=e.rel)
        return 1
    except Misuse as e:
        print(f'USAGE ERROR: {e}')  # v18: the error carries one example; the usage block repeated it
        return 2
    flow.fact(path=rel)  # also when the target was inferred (v18) or resolved from a partial path
    auto = next((x for x in _AUTO_END if x[0] == rel), None) if mode == 'lines' else None
    for note in notes:
        if auto is None or not note.startswith(AUTO_NOTE):
            print(f'NOTE: {note}')
    content = open(path, encoding='utf-8').read()
    original_parses = parses(path, content)[0]
    clines = content.split('\n')
    specs = [Hunk(mode, a, b, c, auto, 'start/end/new' if mode == 'lines' else 'old/new')] + list(_EXTRA)
    plans = []
    for k, h in enumerate(specs):
        if len(specs) > 1:
            h.label = f'hunk {k + 1} of {len(specs)} ({h.keys})'
        p = plan_hunk(rel, path, content, clines, original_parses, h)
        if isinstance(p, int):
            return p
        plans.append(p)
    if len(plans) == 1:
        after = plans[0].after
    else:
        after = combine(rel, content, plans, original_parses)
        if isinstance(after, int):
            return after
    target = (min(p.target[0] for p in plans), max(p.target[1] for p in plans))
    fix = ''.join(p.fix or '' for p in plans)
    if after == content:
        print('NOT APPLIED: the edit changes nothing; NEW must differ from the current text.')
        return 1
    is_py = rel.endswith('.py')
    if is_py and original_parses:
        bad = compile_error(content, after)
        if bad:  # v20 (E061 fastapi_14266: `keyword argument repeated` parses but does not compile)
            print(f'NOT APPLIED: the result would not compile: line {bad.lineno}: {bad.msg}; nothing was changed.')
            lines = after.split('\n')
            if bad.lineno:
                show_numbered(lines, list(range(max(1, bad.lineno - 2), min(len(lines), bad.lineno + 2) + 1)),
                              bad.lineno)
            print(f'CLOSEST: {rel} {target[0]} {target[1]}')
            flow.fact(ok=False, path=rel, closest=(rel, target[0], target[1]))
            remember_shown(rel, target)
            return 1
        conflict = guards.property_conflict(content, after)
        if conflict:
            print('NOT APPLIED: ' + conflict[1])
            flow.fact(ok=False, path=rel, guard='property', guard_line=conflict[0])
            return 1
    lost = guards.lost_definitions(content, after, rel, number_before=True) if is_py and original_parses else []
    # v17b: an edit that also adds definitions replaces or renames (httpx_3672: complete() -> keep_alive/reset, the
    # callers are the next edits): apply it and name the callers; only a deletion with nothing new is refused.
    replaced_by = added_definitions(content, after) if lost else []
    if any(d['refs'] for d in lost) and (not replaced_by or auto is not None):
        if auto is not None and len(plans) == 1:  # the guessed statement range deleted a definition: the v17b refusal
            return report_no_end(NoEnd(rel, auto[1], auto[2]))
        return report_lost(rel, clines, after, [d for d in lost if d['refs']], target)
    open(path, 'w', encoding='utf-8').write(after)
    imp = None
    if is_py:
        imp = import_broken(rel, path, content, after, target)
        if imp is True:
            return 1
    if auto is not None:
        print('NOTE: ' + next(n for n in notes if n.startswith(AUTO_NOTE)))
    for p in plans:
        if p.drift_note:
            print(f'NOTE: {p.label + ": " if p.label else ""}{p.drift_note}')
    if len(plans) == 1:
        print(f'EDITED {rel} ({plans[0].strategy}){fix}')
    else:
        print(f'EDITED {rel} ({len(plans)} hunks: ' + '; '.join(p.strategy + (p.fix or '').replace('(', '[').replace(
            ')', ']') for p in sorted(plans, key=lambda p: p.start)) + ')')
    show_diff(rel, content, after)
    hunks = file_hunks(content, after)
    if state() is not None:  # (the diff's own lines are not 'shown': a unified diff prints no line numbers per line)
        mem_set('hunks', mem_get('hunks') + [[call_n(), rel, with_old_lines(hunks, clines)]])
    for p in plans:
        if p.mode == 'lines' and p.b >= p.a:
            boundary_warning(clines, p.a, p.b, after_hunk(content, p))
    for d in lost[:2]:
        if d['refs']:
            print(f'WARNING: the edit replaced {guards.describe(d)} (new: {", ".join(replaced_by[:3])}), but it is still '
                  f'used at: {guards.uses(d["refs"])}; update those uses too, or keep the old name.')
        else:
            print(f'WARNING: the edit deleted {guards.describe(d)}; nothing in the package uses it, but if that was '
                  'not intended, put it back.')
    if is_py:
        for w in guards.escape_warnings(content, after):
            print(w)
        if not fix:  # NEW as written (a repair that does this is refused above)
            for first, last, term, word in guards.unreachable(content, after)[:1]:
                print(f'WARNING: lines {first}-{last} come after the `{word}` at line {term} in the same block, so they '
                      'never run; if that is not intended, fix the indentation with edit.py.')
    others = also_at(rel, content, after)
    if others:
        print('the same code also appears at: ' + ', '.join(f'{rel}:{a}' + (f' {f}' if f else '') for a, _, f in others)
              + '; if it has the same bug, fix it too')
        flow.fact(also=[(rel, a, b, f) for a, b, f in others])
        remember_shown(rel, *[(a, b) for a, b, _ in others])
    if state() is not None:
        remaining_sites(rel, hunks, others)
    if is_py:
        check = check_line(rel, content, after, imp)
        if check:
            print(check)
    return 0


class Hunk:
    """One replacement of an edit.py call: ('lines', start, end, NEW) or ('text', OLD, NEW, None); AUTO: the start-only
    end inference of the first hunk (parse_object); KEYS: the argument names it came from (multi-edit, v20)."""

    def __init__(self, mode, a, b, c, auto=None, keys=''):
        self.mode, self.a, self.b, self.c, self.auto, self.keys = mode, a, b, c, auto, keys
        self.label = ''


class Plan:
    """A hunk resolved against the file: AFTER (the whole file with only this hunk applied), the replaced characters
    START..END of the original, FIX (indentation repair note), STRATEGY, TARGET (1-based lines), DRIFT_NOTE."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def after_hunk(content, p):
    """The text a plan puts in place of characters p.start..p.end of CONTENT."""
    return p.after[p.start:len(p.after) - (len(content) - p.end)]


def hunk_refusal(h):
    """Before a multi-edit hunk's refusal: which hunk it is (the call applies all or nothing)."""
    if h.label:
        print(f'NOT APPLIED: {h.label} cannot be applied, so nothing was changed (all hunks or none):')


def plan_hunk(rel, path, content, clines, original_parses, h):
    """Resolve hunk H against CONTENT: a Plan, or the exit code after printing why it cannot be applied."""
    mode, a, b, new, auto = h.mode, h.a, h.b, (h.c if h.mode == 'lines' else h.b), h.auto
    drift_note = None
    if mode == 'lines':
        start_l, end_l = a, b
        if not (1 <= start_l <= end_l <= len(clines)):
            hunk_refusal(h)
            print(f'bad range {start_l}-{end_l}; file has {len(clines)} lines')
            return 1
        region = clines[start_l - 1:end_l]
        if elided(new, '\n'.join(region)):
            return 2
        start = sum(len(l) + 1 for l in clines[:start_l - 1])
        end = sum(len(l) + 1 for l in clines[:end_l]) - 1
        new, numbered = unnumbered(new)
        if numbered:
            print(f'NOTE: {NUMBERED_NOTE}')
        new_variants = [(lv, n[:-1] if n.endswith('\n') and len(n) > 1 else n) for lv, n in variants(new)]
        after, level, fix, errors, dead = choose_new(path, content, start, end, new_variants, 0, region=region,
                                                     lines_mode=(clines, end_l) if original_parses and not
                                                     in_string(content, start_l) else None)
        strategy = f'lines {start_l}-{end_l}'
        target = (start_l, end_l)
        if after is None and original_parses and auto is None and rel.endswith('.py'):
            rep = range_repair(path, content, clines, start_l, end_l, new)  # v20
            if rep:
                after, fix, (s2, e2), why, ins = rep
                drift_note = (f'lines {start_l}-{end_l} as given do not parse with NEW; {why}, so the edit '
                              + (f'inserted NEW after line {e2} and replaced nothing' if ins else
                                 f'replaced lines {s2}-{e2}'))
                region = clines[s2 - 1:e2]
                start = sum(len(l) + 1 for l in clines[:s2 - 1])
                end = sum(len(l) + 1 for l in clines[:e2]) - 1
                strategy = f'inserted after line {e2}' if ins else f'lines {s2}-{e2}'
                target = (s2, e2)
                a, b = (e2 + 1, e2) if ins else (s2, e2)  # (boundary_warning checks the range as repaired)
    else:
        old = a
        if not old:
            hunk_refusal(h)
            print('OLD is empty; give the exact existing text to replace.')
            return 1
        if old == new:
            hunk_refusal(h)
            print('NOT APPLIED: OLD and NEW are identical; NEW must contain the changed code.')
            return 1
        if elided(new, old):
            return 2
        found = locate(content, old)
        if isinstance(found, Ambiguous):
            hunk_refusal(h)
            report_sites(rel, clines, found, old, new)
            return 1
        if found is None:
            if argfix.is_placeholder(old) or ANGLE.match(old):
                print(f'USAGE ERROR: OLD {old!r} is a placeholder: copy the exact current lines from show.py, e.g. '
                      + argfix.call('edit.py', rel, 'if self.amount == None:', 'if self.amount is None:'))
                return 2
            drift = drift_match(rel, content, old, new)  # v20
            if isinstance(drift, Ambiguous):
                hunk_refusal(h)
                report_sites(rel, clines, drift, old, new)
                return 1
            if drift is None:
                hunk_refusal(h)
                there = new_already_there(content, new)
                if there:
                    code = report_new_there(rel, clines, there)
                    if h.label:  # (a multi-edit applies all hunks or none)
                        flow.fact(ok=False, path=rel)
                        return 1
                    return code
                print('NOT APPLIED: OLD not found (also tried decoded escapes, ignoring indentation and blank lines, '
                      'and OLD\'s first and last lines as anchors).')
                near = closest(rel, content, variants(old)[-1][1])
                if near:
                    closest_footer(rel, near[0], near[1])
                return 1
            s0 = sum(len(l) + 1 for l in clines[:drift.i0])
            found = (s0, s0 + len('\n'.join(clines[drift.i0:drift.i1])), drift.level, drift.delta, drift.strategy,
                     (drift.i0, drift.i1), drift.old_lines)
            drift_note = drift.note
            if drift.merged:  # NEW is the block with NEW's changes, in the file's indentation
                new = drift.new_text
        start, end, level, delta, strategy, lines_span, old_lines = found
        nv = None
        if strategy.startswith('line numbers') or not any(strip_show_prefixes(v) is not None for _, v in variants(old)):
            nv, numbered = unnumbered(new)  # (OLD with prefixes that matched as is: the file has them, keep NEW's)
            if numbered:
                print(f'NOTE: {NUMBERED_NOTE}')
            else:
                nv = strip_line_numbers(new) if strategy.startswith('line numbers') else None
        if drift_note:
            new_variants = [(level, drift.new_text)]
        else:
            new_variants = variants(nv if nv is not None else new)
        region = clines[lines_span[0]:lines_span[1]] if lines_span else None
        after, _, fix, errors, dead = choose_new(path, content, start, end, new_variants, level,
                                                 region=region, delta=delta, old_lines=old_lines)
        target = (lines_span[0] + 1, lines_span[1]) if lines_span else \
            (line_starts(content, start), line_starts(content, end))

    if after is None:
        if auto is not None and original_parses:  # v18: the guessed statement range did not work: the v17b refusal
            hunk_refusal(h)
            return report_no_end(NoEnd(rel, auto[1], auto[2]))
        if dead and original_parses:
            hunk_refusal(h)
            return report_dead(rel, dead, target[0], target[1], region)
        if original_parses:
            hunk_refusal(h)
            report_syntax_error(errors, line_starts(content, start), region)
            print(f'CLOSEST: {rel} {target[0]} {target[1]}')
            remember_shown(rel, target)
            return 1
        after = errors[0][1]  # file did not parse before either; don't block on syntax
        fix = ''
    if after == content and h.label:
        hunk_refusal(h)
        print('NOT APPLIED: this hunk changes nothing; its NEW must differ from the current text.')
        return 1
    return Plan(after=after, start=start, end=end, fix=fix, strategy=strategy, target=target, region=region,
                mode=mode, a=a, b=b, drift_note=drift_note, label=h.label)


def range_repair(path, content, clines, s, e, new):
    """(after, fix, (start, end), why) for a line-range edit whose result does not parse, when NEW's own first/last
    lines say which lines it was written for (v20, E059-E061: 8 such refusals): NEW starts with the line above START;
    NEW ends with the line below END, or with a line inside the range (the rest of the range is not NEW's: it was
    deleted); the start-only inference from START; or the range is only closing brackets NEW does not have (NEW
    goes after them). Only when exactly one of these parses (no dead code); end < start means an insertion after
    END. None otherwise."""
    nl = new_lines(new)
    N = [l.strip() for l in nl if l.strip()]
    if not N:
        return None
    text, n = content, len(clines)
    cands = []
    if s >= 2 and N[0] == clines[s - 2].strip() and N[0] != clines[s - 1].strip() and not PLAIN.match(N[0]):
        cands.append((s - 1, e, f'NEW starts with line {s - 1}'))
    if e < n and N[-1] == clines[e].strip() and N[-1] != clines[e - 1].strip():
        cands.append((s, e + 1, f'NEW ends with line {e + 1}'))
    ks = [k for k in range(s, e + 1) if clines[k - 1].strip() == N[-1]]
    if len(ks) == 1 and ks[0] < e and (_anchor(N[-1]) or ks[0] == e - 1):
        cands.append((s, ks[0], f'NEW ends with line {ks[0]}'))
    try:
        k, sure, why, _ = infer_end(text, s, nl, True, lead_blank=lead_blank(new))
        if sure and k != e and why != BLANK_INSERT:
            cands.append((s, k, why.split(', so ')[0]))
    except Exception:
        pass
    closers = [l.strip() for l in clines[s - 1:e] if l.strip()]
    if closers and all(re.fullmatch(r'[)\]}][)\]},:;]*', c) for c in closers) and not any(c in N for c in closers):
        cands.append((e + 1, e, f'lines {s}-{e} only close brackets and NEW does not'))
    results = {}
    for s2, e2, why in cands:
        if e2 >= s2 and N[0] != clines[s2 - 1].strip() and _pair(N[0], clines[s2 - 1].strip()) < ALIGN_T:
            continue  # NEW does not start like the range: the numbers are off by more than the boundary
        if e2 < s2:  # an insertion after line E2: keep line E2 first (its indentation anchors NEW's repair)
            a = sum(len(l) + 1 for l in clines[:e2 - 1])
            b = a + len(clines[e2 - 1])
            nv = [(0, clines[e2 - 1] + '\n' + '\n'.join(nl))]
            region = clines[e2 - 1:e2]
            lm = (clines, e2)
        else:
            a = sum(len(l) + 1 for l in clines[:s2 - 1])
            b = sum(len(l) + 1 for l in clines[:e2]) - 1
            nv = [(0, '\n'.join(nl))]
            region = clines[s2 - 1:e2]
            lm = (clines, e2)
        after, _, fix, _, _ = choose_new(path, content, a, b, nv, 0, region=region, lines_mode=lm)
        if after is not None and after not in results:
            results[after] = (fix, (s2, e2), why)
    if len(results) != 1:
        return None
    after, (fix, (s2, e2), why) = next(iter(results.items()))
    return (after, fix, (e2, e2), why, True) if e2 < s2 else (after, fix, (s2, e2), why, False)


def combine(rel, content, plans, original_parses):
    """The file with every plan applied (bottom-up; each plan's numbers and text refer to CONTENT, the file before this
    call), or the exit code after printing why not: two hunks touch the same lines, or the hunks parse alone but not
    together."""
    order = sorted(plans, key=lambda p: (p.start, p.end))
    for p, q in zip(order, order[1:]):
        if q.start <= p.end:
            print(f'NOT APPLIED: {p.label} (lines {p.target[0]}-{p.target[1]}) and {q.label} (lines {q.target[0]}-'
                  f'{q.target[1]}) overlap, so nothing was changed; merge them into one hunk, or give each hunk its own '
                  'lines (all line numbers refer to the file before this call).')
            print(f'CLOSEST: {rel} {p.target[0]} {q.target[1]}')
            flow.fact(ok=False, path=rel, closest=(rel, p.target[0], q.target[1]))
            return 1
    after = content
    for p in reversed(order):
        after = after[:p.start] + after_hunk(content, p) + after[p.end:]
    ok, err = parses(rel, after)
    if original_parses and not ok:
        lines = after.split('\n')
        el = err.lineno or 1
        print(f'NOT APPLIED: each hunk parses on its own, but together they do not: SyntaxError line {el}: {err.msg}; '
              'nothing was changed.')
        show_numbered(lines, list(range(max(1, el - 3), min(len(lines), el + 2) + 1)), el)
        flow.fact(ok=False, path=rel)
        return 1
    return after


def import_budget():
    """Seconds an import check may take: less near the task's end (flow's clock must leave room for the harness, whose
    clock runs ~20-30 s ahead of flow's and which kills a call at the task's end: E057 fastapi_14605's last edit timed
    out after 2 s and its output was lost), never under IMPORT_MIN_S: the guard always runs (an edit that breaks the
    import must not be kept at the end, E061 fastapi_14266)."""
    st = state()
    if st is None:
        return guards.IMPORT_TIMEOUT
    try:
        left = flow.time_left(st)
    except Exception:
        return guards.IMPORT_TIMEOUT
    return max(IMPORT_MIN_S, min(guards.IMPORT_TIMEOUT, (left - CHECK_MIN_LEFT_S) / 2))


def import_broken(rel, path, before, after, target):
    """After writing AFTER: if the module no longer imports but did with BEFORE, restore BEFORE, print NOT APPLIED
    and return True. A module that did not import before either keeps the edit (the failure is not this edit's).
    Otherwise (v20) the import's status for the CHECK line: ('ok', module, seconds), ('env', module) when it fails
    without this edit too (cached per workspace: later edits of that module skip the import), ('timeout', module),
    ('time', None) when the task's time is nearly up (not checked), or None (not an importable module)."""
    mod = guards.module_name(rel)
    if not mod or guards.NO_IMPORT.search(rel) or guards.TEST_FILE.search(rel):
        return None
    budget = import_budget()
    if mod in mem_get('import_env'):
        return ('env', mod)
    status, err, secs = guards.import_status(rel, timeout=budget)
    if status == 'ok':
        return ('ok', mod, secs)
    if status != 'fail':
        return (status, mod)
    open(path, 'w', encoding='utf-8').write(before)
    if guards.import_status(rel, timeout=budget)[0] != 'ok':  # broken before the edit too: not this edit's fault
        open(path, 'w', encoding='utf-8').write(after)
        mem_set('import_env', mem_get('import_env') + [mod])
        return ('env', mod)
    removed = before.count('\n') - after.count('\n')
    size = f' (it removed {removed} lines net)' if removed > 3 else ''
    print(f'NOT APPLIED: after this edit `import {mod}` fails (it worked before), so the file was left unchanged'
          f'{size}:')
    for line in err:
        print('  ' + line[:300])
    print('Redo the edit without breaking the module: keep every definition the rest of the file uses.')
    print(f'CLOSEST: {rel} {target[0]} {target[1]}')
    flow.fact(ok=False, path=rel, guard='import', closest=(rel, target[0], target[1]))
    remember_shown(rel, target)
    return True


def compile_error(before, after):
    """The SyntaxError compile() raises for AFTER beyond ast.parse's (a repeated keyword argument, `return` outside a
    function, a `nonlocal` with no binding, ...) when BEFORE compiled; else None."""
    try:
        compile(after, 'x.py', 'exec', dont_inherit=True)
        return None
    except SyntaxError as e:
        try:
            compile(before, 'x.py', 'exec', dont_inherit=True)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return None
        return e
    except (ValueError, RecursionError, MemoryError):
        return None


def check_line(rel, before, after, imp):
    """v20 diagnostics after a successful .py edit, one line: `CHECK: compiles; no new undefined names; import X OK`,
    or the first problem with its line: a name the edit leaves undefined (loaded where nothing binds it: not local, not a parameter, not a module-level binding or
    import, not a builtin; only names that were not undefined before the edit), an import that timed out. An import
    that fails without the edit too (the environment) is not mentioned."""
    problems = []  # (a compile error is refused before writing: compile_error)
    new_undef = guards.new_undefined_names(before, after)
    for line, name, changed in new_undef[:1]:
        more = f' (and {len(new_undef) - 1} more: ' + ', '.join(sorted({n for _, n, _ in new_undef[1:]})[:4]) + ')' \
            if len(new_undef) > 1 else ''
        problems.append(f"line {line}: undefined name '{name}'" + ('' if changed else
                                                                     ' (the edit removed its definition or import)')
                        + more)
    if imp and imp[0] == 'timeout':
        problems.append(f'import {imp[1]} timed out')
    if problems:
        flow.fact(check=problems[0])
        return 'CHECK: ' + '; '.join(problems)
    parts = ['compiles', 'no new undefined names' if new_undef is not None else None]
    if imp and imp[0] == 'ok':
        parts.append(f'import {imp[1]} OK')
    return 'CHECK: ' + '; '.join(p for p in parts if p)


def added_definitions(before, after):
    """Module/class-level def/class/property names AFTER defines and BEFORE did not (dunders skipped)."""
    old, new = guards.definitions(before) or {}, guards.definitions(after) or {}
    names = {d['name'] for d in old.values()}
    return sorted({d['name'] for d in new.values() if d['name'] not in names and not d['name'].startswith('__')})


def report_lost(rel, clines, after, used, target):
    """NOT APPLIED for an edit that deletes definitions the code still uses (guards.lost_definitions): what is lost,
    where it is used, and a narrower range that keeps it (v17, note 12 L2)."""
    s, e = target
    print(f'NOT APPLIED: the edit ({f"line {s}" if s == e else f"lines {s}-{e}"}) deletes ' + ' and '.join(guards.describe(d) for d in used[:3])
          + f', and NEW does not define {"it" if len(used) == 1 else "them"}, but the code still uses '
          + f'{"it" if len(used) == 1 else "them"}; nothing was changed.')
    for d in used[:3]:
        print(f'  {d["name"]} is used at: {guards.uses(d["refs"])}')
    first, last = min(d['start'] for d in used), max(d['end'] for d in used)
    cs, ce = s, e
    if first > s:
        ce = first - 1
        while ce > s and not clines[ce - 1].strip():
            ce -= 1
        print('to change only the lines before it: '
              + argfix.obj('edit.py', {'file': rel, 'start': cs, 'end': ce, 'new': f'<new text for lines {cs}-{ce}>'}))
    elif last < e:
        cs = last + 1
        while cs < e and not clines[cs - 1].strip():
            cs += 1
        print('to change only the lines after it: '
              + argfix.obj('edit.py', {'file': rel, 'start': cs, 'end': ce, 'new': f'<new text for lines {cs}-{ce}>'}))
    d = used[0]
    added = sorted(set(guards.bound_names(after) or ()) - set(guards.bound_names('\n'.join(clines)) or ()))
    added = [n for n in added if not n.startswith('__')]
    print(f'or keep it: copy lines {d["start"]}-{d["end"]} (its definition) into NEW unchanged'
          + (f'; to rename it to {added[0]}, also keep the old name: {d["name"]} = {added[0]}' if len(added) == 1
             and len(used) == 1 and d['kind'] != 'property' else '') + '.')
    print(f'CLOSEST: {rel} {cs} {ce}')
    flow.fact(ok=False, path=rel, guard='definition', closest=(rel, cs, ce))
    remember_shown(rel, (cs, ce))
    return 1


def boundary_warning(clines, start_l, end_l, inserted):
    """NEW that repeats the line just outside the replaced range usually means START/END were off by one."""
    new_lines = [l.strip() for l in inserted.split('\n') if l.strip()]
    if not new_lines:
        return
    after_l = clines[end_l].strip() if end_l < len(clines) else ''
    before_l = clines[start_l - 2].strip() if start_l >= 2 else ''
    if len(after_l) > 3 and new_lines[-1] == after_l:
        print(f'WARNING: the last line of NEW repeats line {end_l + 1}, which was kept, so it is now duplicated; '
              f'if that is wrong, fix it with edit.py (END should have been {end_l + 1})')
    if len(before_l) > 3 and new_lines[0] == before_l:
        print(f'WARNING: the first line of NEW repeats line {start_l - 1}, which was kept, so it is now duplicated; '
              f'if that is wrong, fix it with edit.py (START should have been {start_l - 1})')


def elided(new, old):
    """Refuse NEW that elides code with a '# ... rest unchanged' style comment the old text does not have."""
    m = ELISION.search(new)
    if m and not ELISION.search(old):
        print(f'USAGE ERROR: NEW contains the placeholder comment {m.group(0).strip()!r}. edit.py writes NEW literally, '
              'so the elided code would be deleted. Replace a smaller OLD region and write NEW out in full.')
        return True
    return False


class Ambiguous:
    """OLD matched several places: what (the message head) and the 1-based (first, last) line of each match."""

    def __init__(self, what, spans):
        self.what, self.spans = what, spans


def read_at(rel, a, b):
    """The call at which the model last read lines A..B of REL (flow's viewed rows), or None."""
    st = state()
    if st is None:
        return None
    return max((n for p, x, y, n, _ in st.viewed if argfix.clean_path(p or '') == rel and x and y and x <= a and b <= y),
               default=None)


def one_line(text):
    """TEXT as one line when it is one line fully decoded (blank lines around it dropped), else None (toolbench
    fastapi_5077: an escaped two-line NEW was printed as one line with literal \\n in the ready call)."""
    lines = trim_blank(variants(str(text))[-1][1].split('\n'))
    return lines[0] if len(lines) == 1 else None


def report_sites(rel, clines, amb, old=None, new=None):
    """NOT APPLIED for an ambiguous OLD: every match with its enclosing function, once, one line each (v18: the ready
    call for the first is the footer's; v16, note 11 L4: rich_3480 fixed Text.append but not Text.append_text).
    v20: a site the model read is marked with the call; for a one-line OLD and NEW at 2..MAX_HUNKS one-line sites, a
    ready multi-edit call changes them all at once (E057 fastapi_14605: the gold patch changed all 4)."""
    sites = [(a, b, names.enclosing(clines, a)) for a, b in amb.spans[:SITES_MAX]]
    print(f'NOT APPLIED: {amb.what} in {rel}; edit each one that has the bug by its lines (they move by the lines '
          'each edit adds):')
    for a, b, func in sites:
        n = read_at(rel, a, b)
        print(f'  {a}-{b}' + (f' {func}' if func else '') + (f' (you read it at call {n})' if n else ''))
    if len(amb.spans) > SITES_MAX:
        print(f'  ... {len(amb.spans) - SITES_MAX} more')
    o, nw = (one_line(old), one_line(new)) if old is not None and new is not None else (None, None)
    if o and nw is not None and 1 < len(amb.spans) <= MAX_HUNKS and all(a == b for a, b in amb.spans):
        d = {'file': rel}
        for k, (a, _) in enumerate(amb.spans, 1):
            line = clines[a - 1]
            sfx = '' if k == 1 else str(k)
            d.update({'start' + sfx: a, 'end' + sfx: a,
                      'new' + sfx: line.replace(o.strip(), nw.strip(), 1) if o.strip() in line else nw})
        print('to change all of them in one call: ' + argfix.obj('edit.py', d))
    a, b, _ = sites[0]
    flow.fact(ok=False, path=rel, closest=(rel, a, b), sites=[(rel, a, b, f) for a, b, f in sites])
    remember_shown(rel, *[(a, b) for a, b, _ in sites])
    if state() is not None:
        mem_set('sites', [[call_n(), rel, [[a, b, f or ''] for a, b, f in sites]]])


def report_new_there(rel, clines, there):
    """OLD not found, but NEW's lines are already in the file (once): the change was made, nothing to do (v20, E058
    requests_7309: the model resent an edit that had been applied; "OLD not found" sent it nowhere and the run ended).
    Counted as a successful edit by flow (the change is in place), exit 0."""
    a, b = there
    print(f'NOT CHANGED: OLD is not in {rel}, but NEW already is, at line{"s" if b > a else ""} '
          f'{a}{"-" + str(b) if b > a else ""} (an earlier edit made this change), so there was nothing to do. Lines now:')
    show_numbered(clines, list(range(a, min(b, a + 7) + 1)))
    if b > a + 7:
        print(f'  ... ({b - a - 7} more lines)')
    flow.fact(ok=True, path=rel)
    remember_shown(rel, (a, b))
    return 0


def remaining_sites(rel, hunks, others):
    """After a successful edit of REL: the sites of the last ambiguous-OLD list this edit did not touch, with their
    current lines (P4: the model reused the old numbers of the list), unless also_at already named them."""
    saved = mem_get('sites')
    if not saved or saved[-1][1] != rel:
        return
    left = []
    for a, b, f in saved[-1][2]:
        m = map_range(a, b, hunks)
        if m is not None:
            left.append((m[0], m[1], f))
    mem_set('sites', [[call_n(), rel, [list(x) for x in left]]] if left else [])
    show = [x for x in left if not any(o[0] <= x[1] and x[0] <= o[1] for o in others)]  # also_at named the rest
    if show:
        print('remaining sites of that OLD, current lines: ' + ', '.join(f'{a}-{b}' + (f' {f}' if f else '')
                                                                       for a, b, f in show))
        remember_shown(rel, *[(a, b) for a, b, _ in show])


def also_at(rel, before, after):
    """[(line, last line, function)] of other places in the edited file that still contain a line this edit changed
    (stripped, distinctive lines only), e.g. the same loop in a sibling method; nearest to the edit first, at most
    SITES_MAX."""
    a, b = before.split('\n'), after.split('\n')
    changed, new_lines = set(), set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ('replace', 'delete'):
            changed.update(l.strip() for l in a[i1:i2])
        if tag in ('replace', 'insert'):
            new_lines.update(range(j1, j2))
    changed = {l for l in changed if len(l) >= SITE_MIN_CHARS and not TRIVIAL.match(l)}
    if len(changed) > SITE_MAX_CHANGED:  # a rewrite, not a small fix that may have a sibling
        return []
    out = []
    for j, line in enumerate(b):
        if j in new_lines or line.strip() not in changed:
            continue
        if out and out[-1][1] == j:  # consecutive lines: one site
            out[-1] = (out[-1][0], j + 1, out[-1][2])
            continue
        out.append((j + 1, j + 1, names.enclosing(b, j + 1)))
    at = min(new_lines, default=0) + 1
    return sorted(out, key=lambda s: abs(s[0] - at))[:SITES_MAX]  # nearest first: a sibling method usually


def locate(content, old):
    """Find OLD. Returns (start, end, level, delta, strategy, (line0, line1), old_lines), None, or an error text."""
    clines = content.split('\n')

    def span(i0, i1):
        start = sum(len(l) + 1 for l in clines[:i0])
        return start, start + len('\n'.join(clines[i0:i1]))

    def by_text(text_variants, label):
        for level, o in text_variants:
            count = content.count(o)
            if count == 1:
                i = content.index(o)
                i0, i1 = line_starts(content, i) - 1, line_starts(content, i + len(o))
                whole = (i == 0 or content[i - 1] == '\n') and content[i + len(o):i + len(o) + 1] in ('', '\n')
                return (i, i + len(o), level, 0, label.format(' (decoded escapes)' if level else ''),
                        (i0, i1) if whole else None, o.split('\n'))
            if count > 1:
                spans = [(line_starts(content, m.start()), line_starts(content, max(m.start(), m.end() - 1)))
                         for m in re.finditer(re.escape(o), content)]
                return Ambiguous(f'OLD occurs {count} times', spans)
        return None

    def by_lines(text_variants, label):
        for level, o in text_variants:
            hits = flexible_match(content, o)
            if len(hits) == 1:
                i0, i1, delta = hits[0]
                start, end = span(i0, i1)
                return start, end, level, delta, label, (i0, i1), trim_blank(o.split('\n'))
            if len(hits) > 1:
                return Ambiguous(f'OLD matches {len(hits)} places ignoring indentation', [(h[0] + 1, h[1]) for h in hits])
        return None

    vs = variants(old)
    found = by_text(vs, 'exact{}') or by_lines(vs, 'whitespace-flexible')
    if found is None:
        stripped = strip_line_numbers(vs[-1][1])
        if stripped is not None:
            sv = variants(stripped)
            found = by_text(sv, 'line numbers removed{}') or by_lines(sv, 'line numbers removed, whitespace-flexible')
    if found is None:  # v20: every decoding (E058 fastapi_14986: only the most decoded one was tried, and decoding
        for level, o in reversed(vs):  # turned the "\n" of a string literal into a line break)
            fz = fuzzy_match(content, o)
            if fz:
                i0, i1, delta, ratio = fz
                start, end = span(i0, i1)
                olines = o.strip('\n').split('\n')
                found = (start, end, level, delta, f'near-identical block, similarity {ratio:.2f}', (i0, i1), olines)
                break
    return found


# ---------------------------------------------------------------- v20: OLD that drifted from the file

MERGE_MIN_LINES = 3      # block anchors need an OLD of at least this many non-blank lines (OpenCode, Hermes)
MERGE_MIN_EQUAL = 0.5    # share of OLD's non-blank lines that must equal a line of the block (the rest drifted)
MERGE_SPAN = 2           # the block is at most MERGE_SPAN x OLD's lines + MERGE_SLACK long
MERGE_SLACK = 10
NEW_THERE_MIN_CHARS = 20  # a one-line NEW this long (stripped) is distinctive enough to say "already in the file"


class Drift:
    """A drifted OLD resolved to file lines I0..I1 (0-based, end exclusive): NEW_TEXT replaces them (written for the
    file's indentation when MERGED, else re-indented by DELTA like a whitespace-flexible match); NOTE says how."""

    def __init__(self, i0, i1, level, delta, strategy, new_text, old_lines, note, merged=False):
        self.i0, self.i1, self.level, self.delta, self.strategy = i0, i1, level, delta, strategy
        self.new_text, self.old_lines, self.note, self.merged = new_text, old_lines, note, merged


def blank_insensitive(clines, olines):
    """[(i0, i1)] of file blocks whose non-blank lines equal OLD's non-blank lines (stripped) in order, any number of
    blank lines between them (v20, E057 fastapi_14372 / E058 requests_7315: OLD had 1 blank line where the file has
    2, or 2 where it has 3)."""
    want = [l.strip() for l in olines if l.strip()]
    if len(want) < 2:
        return []
    nb = [(i, l.strip()) for i, l in enumerate(clines) if l.strip()]
    hits = []
    for k in range(len(nb) - len(want) + 1):
        if nb[k][1] == want[0] and all(nb[k + m][1] == want[m] for m in range(1, len(want))):
            hits.append((nb[k][0], nb[k + len(want) - 1][0] + 1))
    return hits


def _gaps(lines):
    """(leading blank lines, [[line, blank lines after it]]) of LINES (lines as given)."""
    lead, out = 0, []
    for l in lines:
        if l.strip():
            out.append([l, 0])
        elif out:
            out[-1][1] += 1
        else:
            lead += 1
    return lead, out


def keep_blank_runs(new, olines, flines):
    """NEW with the blank lines after each line NEW keeps from OLD set to the file's count when NEW has OLD's count
    there: OLD's blank lines were miscounted, not changed on purpose (NEW's own count wins where it differs from
    OLD's). Lines pair up by position (difflib on the stripped non-blank lines: a repeated docstring quote pairs
    right)."""
    _, o = _gaps(olines)
    _, f = _gaps(flines)
    lead, nn = _gaps(new.split('\n'))
    if len(o) != len(f) or not nn:
        return new
    sm = difflib.SequenceMatcher(None, [x[0].strip() for x in o], [x[0].strip() for x in nn], autojunk=False)
    for k0, j0, size in sm.get_matching_blocks():
        for t in range(size):
            k, j = k0 + t, j0 + t
            if j < len(nn) - 1 and k < len(o) - 1 and nn[j][1] == o[k][1]:
                nn[j][1] = f[k][1]
    out = [''] * lead
    for line, blanks in nn:
        out.append(line)
        out.extend([''] * blanks)
    return '\n'.join(out)


def anchor_merge(clines, olines, nlines):
    """[(i0, i1, new block lines, equal, total)] for each file block that OLD's first and last non-blank lines anchor
    (stripped, the nearest last line after the first, within MERGE_SPAN x OLD + MERGE_SLACK lines) where NEW's changes
    to OLD can be applied: every OLD line NEW replaces or deletes equals a line of the block (in order, those lines
    contiguous but for blank lines), every insertion point is next to such a line, and at least MERGE_MIN_EQUAL of
    OLD's non-blank lines equal block lines. The block keeps its own lines where OLD drifted from it (a docstring OLD
    summarised, an invented `pass`, E057 fastapi_14605 / rich_3882): only NEW's changes are applied (a 3-way merge,
    like `patch` with fuzz)."""
    O = [l.strip() for l in olines]
    nb_o = [k for k, l in enumerate(O) if l]
    if len(nb_o) < MERGE_MIN_LINES:
        return []
    # the anchors are OLD's first and last DISTINCTIVE lines (a `},` or `)` is in every file many times: toolbench
    # fastapi_14791 anchored on the wrong `},`); the lines before/after them must equal the file's neighbours
    anc = [k for k in nb_o if _anchor(O[k]) and not BARE_DECORATOR.match(O[k])]
    if len(anc) < 2:
        return []
    fa, la = anc[0], anc[-1]
    # NEW's changes to OLD keep indentation (toolbench fastapi_14986: a line NEW only re-indented counted as unchanged)
    ops = [op for op in difflib.SequenceMatcher(None, [l.rstrip() for l in olines], [l.rstrip() for l in nlines],
                                                autojunk=False).get_opcodes() if op[0] != 'equal']
    if not ops:
        return []
    S = [l.strip() for l in clines]
    span = MERGE_SPAN * len(O) + MERGE_SLACK
    tail = len(O) - 1 - la
    out = []
    for i0 in range(len(S)):
        if S[i0] != O[fa]:
            continue
        j0 = next((j for j in range(i0 + 1, min(len(S), i0 + span)) if S[j] == O[la]), None)
        if j0 is None:
            continue
        i, j = i0 - fa, j0 + tail
        if i < 0 or j >= len(S) or any(S[i + t] != O[t] for t in range(fa)) or \
                any(S[j0 + 1 + t] != O[la + 1 + t] for t in range(tail)):
            continue
        block = S[i:j + 1]
        m = {}
        for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(None, O, block, autojunk=False).get_opcodes():
            if tag == 'equal':
                for d in range(a2 - a1):
                    m[a1 + d] = b1 + d
        equal = sum(1 for k in nb_o if k in m)
        if equal == len(nb_o) and len(block) == len(O):
            continue  # OLD is there as it is (other matchers decide)
        if equal < MERGE_MIN_EQUAL * len(nb_o) or fa not in m or la not in m:
            continue
        edits, ok = [], True
        for tag, a1, a2, b1, b2 in ops:
            if a2 > a1:  # replaced/deleted OLD lines: each non-blank one is in the block, contiguous
                pos = [m.get(k) for k in range(a1, a2) if O[k]]
                if any(p is None for p in pos):
                    ok = False
                    break
                lo = min((m[k] for k in range(a1, a2) if k in m), default=None)
                hi = max((m[k] for k in range(a1, a2) if k in m), default=None)
                if lo is None or any(block[p] and p not in pos for p in range(lo, hi + 1)):
                    ok = False
                    break
                ref = next(k for k in range(a1, a2) if k in m)
                edits.append((lo, hi + 1, b1, b2, ref))
            else:  # an insertion before OLD line a1: next to a line of the block
                before, after_ = m.get(a1 - 1) if a1 > 0 else None, m.get(a1) if a1 < len(O) else None
                if before is not None and (after_ is None or after_ == before + 1 or
                                           all(not x for x in block[before + 1:after_])):
                    edits.append((before + 1, before + 1, b1, b2, a1 - 1))
                elif after_ is not None and before is None:
                    edits.append((after_, after_, b1, b2, a1))
                else:
                    ok = False
                    break
        if not ok:
            continue
        lines = list(clines[i:j + 1])
        anchors = sorted((k for k in m if O[k]), key=lambda k: k)
        for lo, hi, b1, b2, ref in sorted(edits, key=lambda e: -e[0]):
            r = min(anchors, key=lambda k: (abs(k - ref), k))  # the nearest OLD line found in the block
            d = indent(clines[i + m[r]]) - indent(olines[r])  # the block's indentation where NEW changes it
            lines[lo:hi] = reindent('\n'.join(nlines[b1:b2]), d).split('\n') if b2 > b1 else []
        out.append((i, j + 1, lines, equal, len(nb_o)))
    return out


PREFER_VIEWED = False  # v20: off. Of the 19 ambiguous-OLD refusals of E057-E061, 13 have a known intent (the model's
# next call or the gold patch): "the site inside the code read last" was right 6 times and wrong 7 times (rich_2943,
# requests_6644 x2, fastapi_13207 x2, fastapi_14485, fastapi_5624), so a guess would often edit the wrong site
# silently; the sites are listed instead (with a ready multi-edit call for a one-line OLD/NEW).


def viewed_choice(rel, spans):
    """The index of the one span (1-based first, last) inside the region of REL the model read most recently among
    those containing any span, with that read's call number; None when no read contains one, or it contains several,
    or PREFER_VIEWED is off."""
    st = state()
    if st is None or not PREFER_VIEWED:
        return None
    for p, a, b, n, _ in sorted(st.viewed, key=lambda v: -v[3]):
        if argfix.clean_path(p or '') != rel or not a or not b:
            continue
        inside = [k for k, (s, e) in enumerate(spans) if a <= s and e <= b]
        if inside:
            return (inside[0], n) if len(inside) == 1 else None
    return None


def drift_match(rel, content, old, new):
    """A Drift for an OLD that no exact / decoded / indentation-free / near-identical match found (v20): its lines with
    blank lines miscounted (blank_insensitive), else NEW's changes applied where OLD's first and last lines anchor them
    (anchor_merge). Several places: Ambiguous (viewed_choice when PREFER_VIEWED).
    None when nothing fits."""
    clines = content.split('\n')
    nvs = variants(new)
    for level, o in variants(old):
        olines = trim_blank(o.split('\n'))
        hits = blank_insensitive(clines, olines)
        pick = None
        if len(hits) > 1:
            pick = viewed_choice(rel, [(a + 1, b) for a, b in hits])
            if pick is None:
                return Ambiguous(f'OLD matches {len(hits)} places ignoring blank lines', [(a + 1, b) for a, b in hits])
        if hits:
            i0, i1 = hits[pick[0] if pick else 0]
            nv = next((v for lv, v in nvs if lv == level), new)
            raw = o.split('\n')  # (OLD's own blank lines at both ends are outside the block: so are NEW's copies)
            lead_o = next((k for k, l in enumerate(raw) if l.strip()), 0)
            trail_o = next((k for k, l in enumerate(reversed(raw)) if l.strip()), 0)
            nraw = nv.split('\n')
            while lead_o and nraw and not nraw[0].strip():
                nraw, lead_o = nraw[1:], lead_o - 1
            while trail_o and nraw and not nraw[-1].strip():
                nraw, trail_o = nraw[:-1], trail_o - 1
            new_text = keep_blank_runs('\n'.join(nraw), olines, clines[i0:i1])
            first = next(k for k in range(i0, i1) if clines[k].strip())
            note = (f'OLD matched lines {i0 + 1}-{i1} except for the number of blank lines'
                    + (f'; it also matches {len(hits) - 1} other place{"s" if len(hits) > 2 else ""}, this one is in '
                       f'the code you read at call {pick[1]}' if pick else ''))
            return Drift(i0, i1, level, indent(clines[first]) - indent(olines[0]), 'blank lines ignored', new_text,
                         olines, note)
    for level, o in variants(old):
        olines = trim_blank(o.split('\n'))
        nv = next((v for lv, v in nvs if lv == level), None)
        if nv is None:
            continue
        hits = anchor_merge(clines, olines, trim_blank(nv.split('\n')))
        pick = None
        if len(hits) > 1:
            pick = viewed_choice(rel, [(h[0] + 1, h[1]) for h in hits])
            if pick is None:
                return Ambiguous(f'OLD\'s first and last lines enclose {len(hits)} places', [(h[0] + 1, h[1]) for h in hits])
        if hits:
            i0, i1, lines, equal, total = hits[pick[0] if pick else 0]
            note = (f'OLD did not match exactly: lines {i0 + 1}-{i1} begin and end with OLD\'s first and last lines '
                    f'and contain {equal} of its {total} lines; NEW\'s changes were applied there and the lines OLD '
                    'had differently were kept as they are'
                    + (f' (this place is in the code you read at call {pick[1]})' if pick else ''))
            return Drift(i0, i1, level, 0, 'anchored on OLD\'s first and last lines', '\n'.join(lines), None, note,
                         merged=True)
    return None


def new_already_there(content, new):
    """(first, last) 1-based lines where NEW's non-blank lines already stand (stripped, once, consecutively): OLD is
    gone because an earlier edit made this change (v20, E058 requests_7309: the model resent its applied edit). None
    when NEW is not there, is there several times, or is too short to tell."""
    clines = content.split('\n')
    for _, v in variants(new):
        nl = [l for l in trim_blank(v.split('\n'))]
        want = [l.strip() for l in nl if l.strip()]
        if not want or (len(want) == 1 and len(want[0]) < NEW_THERE_MIN_CHARS):
            continue
        hits = blank_insensitive(clines, nl) if len(want) > 1 else \
            [(i, i + 1) for i, l in enumerate(clines) if l.strip() == want[0]]
        if len(hits) == 1:
            return hits[0][0] + 1, hits[0][1]
    return None


if __name__ == '__main__':
    sys.exit(flow.main('edit.py', main))
