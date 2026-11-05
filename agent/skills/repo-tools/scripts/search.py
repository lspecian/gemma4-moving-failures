"""Search the repository for a literal string or regex; bounded output.

usage: search.py  text=def apply_discount                    literal text
       search.py  regex=def (total|subtotal)  within=billing/   Python regex (+ optional within)
       (positional form, still accepted: search.py [-E] [-i] TEXT [PATH])

Searches *.py files under /workspace (or "path": a directory, file or glob inside it) for the literal "text", or
for the Python regular expression "regex" (positional: -E makes TEXT a regex; -i ignores case). Source files are
listed before tests/docs/examples. "query" is read as "text". Prints at most 10 matching lines as
`file:line Enclosing.def: text` (the function or class the line is in; `file:line: text` at module level), then the
remaining matches per file with counts (source files first). With at most 3 matches, each comes with 2 lines of
context before and after (the matching line marked with >).
v18: when nothing matches, the fallbacks run in the same call and the first that matches is printed, labelled in
a NOTE: the slips (quotes around the text, regex syntax without -E, TEXT and PATH in one string), then ignoring
case, then an identifier split into its words (isCoroutineCallable -> is, coroutine, callable: first the words in
order with any separator, then lines with all of them), then lines with any of several words, then non-.py text
files. Output is capped at CAP characters.
Hits are ranked: source files before tests/docs; within those, regex literals (re.compile(...), r"...") first, then
other code where the text is a whole word, then code where it is part of a longer word, then comments and docstrings.
When no .py file matches, other text files (.md .txt .rst .toml .cfg .ini .yaml .yml .html .js) are searched too and
their matches labelled. When nothing matches an identifier, says where a definition of that name, or of a similar
one, exists (as ready show.py calls), or that the repository defines no such name.
Compact mode (the context is large, see flow.context_tokens): fewer hits and context lines, one line says so.
"""

import ast
import contextlib
import glob
import io
import keyword
import os
import re
import sys
import time
import tokenize

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402  (flow engine: refusals + NEXT/BUDGET footers)
import argfix  # noqa: E402
import names  # noqa: E402  (enclosing definitions, similar names)
from argfix import Misuse  # noqa: E402

ROOT = (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))  # overridable for local testing
MAX_HITS = 10          # v18: was 15 (search output ~1,100 characters typical)
CONTEXT_HITS = 3       # this many hits or fewer: each with CONTEXT lines around it
CONTEXT = 2
CONTEXT_CHARS = 140    # chars of a context line
OVERFLOW_FILES = 5     # files named (with hit counts) in the line about the matches not printed
SIMILAR = 3            # definitions listed when nothing matches an identifier
HIT_CHARS = 110        # chars of a hit line (v18: was 160)
CAP = 3500             # v18: characters of output at most (the harness cuts results at 5,000; flow adds the footer)
# v17 L1 compact mode: (MAX_HITS, CONTEXT_HITS, CONTEXT, OVERFLOW_FILES, SIMILAR, HIT_CHARS) per level
COMPACT = {1: (8, 2, 1, 4, 2, 100), 2: (6, 1, 1, 3, 1, 90)}
COMPACT_LINE = 'compact: context is large; fewer hits shown, narrow with within= if needed'
# v17 L9: when no .py file matches, these text files are searched too (fastapi_14492: docs/ has only .md files)
TEXT_EXT = ('.md', '.txt', '.rst', '.toml', '.cfg', '.ini', '.yaml', '.yml', '.html', '.js')
TEXT_MAX_FILES = 8000
TEXT_MAX_BYTES = 400_000
TEXT_SECONDS = 3.0
TEXT_HITS = 10         # matching lines printed from text files (prose lines are long and less often the answer)
TEXT_CHARS = 120
RANK_FILES = 60        # .py files with hits whose comments/docstrings are told apart by tokenizing (the rest: guess)
REGEX_LIT = re.compile(r'\bre\.\w+\(|\(\?P<|(?<![\w])(?:[rR][bBfF]?|[bBfF][rR])["\']')
LOCALE = re.compile(r'(^|/)(?!en/)[a-z]{2}(?:-[a-zA-Z]{2,4})?/|\.(?!en\.)[a-z]{2}(?:-[a-zA-Z]{2,4})?\.(md|rst|txt)$')
IDENT = re.compile(r'\s*(?:(?:async\s+)?def\s+|class\s+|self\.|\.)?([A-Za-z_][A-Za-z0-9_]{2,})\s*(?:\\?\(|:|=)?\s*')
AUX = re.compile(r'(^|/)(tests?|testing|docs|docs_src|examples?|benchmarks|scripts)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$')
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.tox', '.mypy_cache'}
USAGE = __doc__.strip().split('\n\n')[1]
FLAGS = {'-E': 'regex', '--regex': 'regex', '-P': 'regex', '-i': 'icase', '--ignore-case': 'icase', '-F': 'literal',
         '-w': 'word'}
IGNORED = {'-r', '-R', '-n', '-rn', '-nr', '-rni', '-I', '-H', '--recursive', '--line-number'}
REGEX_HINT = re.compile(r'\\[.()\[\]{}sdwbS+*?|]|\.\*|\|')


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
    """Shrink the limits for compact level 1 or 2; level -1: matches from non-.py text files (fewer, shorter)."""
    global MAX_HITS, CONTEXT_HITS, CONTEXT, OVERFLOW_FILES, SIMILAR, HIT_CHARS
    if level in COMPACT:
        MAX_HITS, CONTEXT_HITS, CONTEXT, OVERFLOW_FILES, SIMILAR, HIT_CHARS = COMPACT[level]
    elif level == -1:
        MAX_HITS, HIT_CHARS = min(MAX_HITS, TEXT_HITS), min(HIT_CHARS, TEXT_CHARS)


def iter_py(base):
    if os.path.isfile(base):
        yield base
        return
    for d, dirs, files in os.walk(base):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith('.'))
        for f in sorted(files):
            if f.endswith('.py') and not f.startswith('.'):  # skips the harness's .adk_exec_*.py
                yield os.path.join(d, f)


def iter_text(base):
    """Non-Python text files under base (bounded; a file given as base was already searched as it is)."""
    if os.path.isfile(base):
        return
    for d, dirs, files in os.walk(base):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith('.'))
        for f in sorted(files):
            if f.endswith(TEXT_EXT) and not f.startswith('.') and not f.endswith('.min.js'):
                p = os.path.join(d, f)
                try:
                    if os.path.getsize(p) > TEXT_MAX_BYTES:
                        continue
                except OSError:
                    continue
                yield p


def parse_object(d, notes):
    """The object form: {"text"} or {"regex"}, + optional "path". Values are used as given."""
    argfix.exclusive(d, 'search.py', ('text',), ('regex',))
    key = 'regex' if 'regex' in d else 'text'
    argfix.require(d, 'search.py', key)
    text, opts = d[key], ({'regex'} if key == 'regex' else set())
    path = None
    if d.get('path', '').strip():
        a1 = argfix.clean_path(d['path'])
        if argfix.exists(a1) or any(c in a1 for c in '*?['):
            path = a1
        else:
            path, n = argfix.resolve(d['path'], 'search.py')
            notes.append(n)
    else:
        path_as_text(text)
    return text, path, opts, notes


def path_as_text(text):
    """TEXT that is an existing file or directory (and no PATH): the model meant to look inside it."""
    if argfix.looks_like_path(text) and argfix.exists(argfix.clean_path(text)) and argfix.clean_path(text) != '.':
        rel = argfix.clean_path(text)
        kind = 'a directory' if os.path.isdir(os.path.join(ROOT, rel)) else 'a file'
        raise Misuse(f'{text!r} is {kind}, not search text. Search inside it with '
                     + argfix.call('search.py', 'def export_html', rel)
                     + f'  or list its files or definitions with ' + argfix.call('outline.py', rel)
                     + '  To search for the path string itself, add a path: ' + argfix.call('search.py', rel, '.'))


def parse(argv):
    d, note = argfix.object_args(argv, 'search.py')
    if d is not None:
        return parse_object(d, [note] if note else [])
    argv, note = argfix.unpack(argv, 'search.py')
    notes = [note] if note else []
    opts = set()
    # flags glued to their value in one string: "-E foo|bar", "-E \"def mount\""
    if argv and argv[0].startswith('-') and (' ' in argv[0].strip()) and argv[0].split()[0] in {**FLAGS, **dict.fromkeys(IGNORED)}:
        argv = argfix.split(argv[0]) + argv[1:]
        notes.append('split the flag from its value; pass "-E" and the pattern as separate strings')
    while argv and (argv[0] in FLAGS or argv[0] in IGNORED):
        flag = argv.pop(0)
        if flag in FLAGS:
            opts.add(FLAGS[flag])
        else:
            notes.append(f'ignored the grep flag {flag} (search.py is always recursive with line numbers)')
    if len(argv) == 1 and argv[0][:1] in '"\'' and ' ' in argv[0]:
        parts = argfix.split(argv[0])  # '"is_terminal" rich/' -> TEXT and PATH
        if len(parts) == 2 and argfix.looks_like_path(parts[1]) and argfix.exists(argfix.clean_path(parts[1])):
            argv = parts
            notes.append('split TEXT and PATH into two strings')
    if not argv or not argv[0]:
        raise Misuse('search.py needs the TEXT to look for, e.g. ' + argfix.call('search.py', 'def export_html'))
    if len(argv) > 2:
        extra = argv[2:]
        raise Misuse(f'search.py takes one TEXT and at most one PATH; got {len(argv)} arguments. For several words '
                     'use a regex: ' + argfix.call('search.py', '-E', '|'.join([argv[0]] + [argv[1]] + extra)[:60]))
    text = argv[0]
    path = None
    if len(argv) == 2:
        a0, a1 = argfix.clean_path(argv[0]), argfix.clean_path(argv[1])
        p0 = argfix.looks_like_path(argv[0]) and argfix.exists(a0)
        p1 = argfix.exists(a1) or any(c in a1 for c in '*?[')
        if p0 and not p1:
            text, path = argv[1], a0
            notes.append(f'swapped the arguments: TEXT comes first, then PATH (search.py {text!r} {a0})')
        elif p1:
            path = a1
        else:
            try:
                path, n = argfix.resolve(argv[1], 'search.py')
                notes.append(n)
            except Misuse:
                raise Misuse(f'PATH {argv[1]!r} does not exist. search.py takes one TEXT and an optional PATH; '
                             f'to find either word use a regex: '
                             + argfix.call('search.py', '-E', f'{argv[0]}|{argv[1]}'))
    else:
        path_as_text(text)
    return text, path, opts, notes


KIND_TEXT = {'attribute': 'an attribute assigned in {owner}', 'class attribute': 'a class attribute',
             'assignment': 'a module-level name', 'class': 'a class', 'def': 'a function', 'async def': 'a function'}


def print_similar(word):
    """Nothing matched the identifier word: where a definition of it (another kind: an attribute, not a def) or of a
    similar name exists, as ready show.py calls; or that the repository defines no such name."""
    rows = names.similar(word, SIMILAR)
    if rows is None or names._kb() is None:
        return
    if not rows:
        print(f'{word} is not defined anywhere in the repository (no def, class or assignment of that name).')
        return
    exact = rows[0][3].split('.')[-1] == word
    print(f'{word} is defined as:' if exact else f'no definition named {word}; similar names:')
    for rel, line, kind, qn, owner in rows:
        what = KIND_TEXT.get(kind, kind).format(owner=owner or '?')
        if kind in ('def', 'async def') and '.' in qn:
            what = 'a method'
        print(f'  {qn} ({what}) {rel}:{line}: ' + argfix.obj('show.py', {'file': rel, 'symbol': qn}))
    flow.fact(next_show=(rows[0][0], rows[0][3]))


def compile_pattern(text, regex, icase, word):
    src = text if regex else re.escape(text)
    if word:
        src = rf'\b{src}\b'
    return re.compile(src, re.I if icase else 0)


def run_search(pat, bases, text_files=False):
    hits, files, t0 = [], 0, time.perf_counter()
    for base in bases:
        for path in (iter_text(base) if text_files else iter_py(base)):
            if text_files:
                files += 1
                if files > TEXT_MAX_FILES or time.perf_counter() - t0 > TEXT_SECONDS:
                    return hits
            rel = os.path.relpath(path, ROOT)
            try:
                with open(path, encoding='utf-8', errors='replace') as fh:
                    for n, line in enumerate(fh, 1):
                        if pat.search(line):
                            hits.append((bool(AUX.search(rel)), rel, n, line.strip()))
            except OSError:
                continue
    return hits


def ident_words(text):
    """The words of one identifier ('isCoroutineCallable', 'TTY_INTERACTIVE', 'dunder-call') in lower case;
    [] when text is not a single identifier."""
    t = text.strip().strip('`"\'')
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*', t):
        return []
    t = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', t))
    return [w for w in re.split(r'[_-]+', t.lower()) if w]


def retries_for(text, regex, opts, path):
    """[(pattern, bases or None, why)] to try when the text matched nothing: common slips (quotes around the text,
    regex syntax without -E, wrong case, TEXT and PATH in one string, several words)."""
    retries = []
    stripped = text.strip()
    if len(stripped) > 2 and stripped[0] == stripped[-1] and stripped[0] in '"\'`':
        retries.append((stripped[1:-1], regex, 'icase' in opts, 'without the surrounding quotes'))
    if not regex and REGEX_HINT.search(text):
        retries.append((text, True, 'icase' in opts, 'as a regex (pass regex syntax as "regex", not "text")'))
    if 'icase' not in opts:
        retries.append((text, regex, True, 'ignoring case'))
    split = ident_words(text) if not regex else []
    if len(split) >= 2:  # v18: an identifier split into its words
        retries.append((r'[\W_]*'.join(re.escape(w) for w in split), True, True,
                        'for the words ' + ' '.join(split) + ' in this order (any case and separator)'))
        long_words = [w for w in split if len(w) >= 3]
        if len(long_words) >= 2:
            retries.append((''.join(f'(?=.*{re.escape(w)})' for w in long_words), True, True,
                            'for lines with all of ' + ', '.join(long_words) + ' (any case)'))
    parts = text.split()
    split_bases = None
    if not path and len(parts) == 2:  # "TEXT PATH" or "PATH TEXT" packed into one string
        for t, p in ((parts[0], parts[1]), (parts[1], parts[0])):
            p = argfix.clean_path(p)
            if argfix.looks_like_path(p) and argfix.exists(p) and not argfix.looks_like_path(t):
                split_bases = [os.path.join(ROOT, p)]
                retries.insert(0, (t.strip('"\''), regex, 'icase' in opts, f'for {t!r} in {p} (pass "text" and '
                                                                          '"path" as two keys)'))
                break
    words = [w for w in dict.fromkeys(re.findall(r'[A-Za-z_][A-Za-z0-9_]{2,}', text))
             if not keyword.iskeyword(w)]
    if not regex and len(text.split()) > 1 and 1 < len(words) <= 6:  # several names: lines with any of them
        retries.append(('|'.join(re.escape(w) for w in words), True, 'icase' in opts,
                        'for any of the words ' + ', '.join(words)))
    out = []
    for t, rx, ic, why in retries:
        try:
            out.append((compile_pattern(t, rx, ic, 'word' in opts),
                        split_bases if split_bases and why.startswith('for ') else None, why))
        except re.error:
            continue
    return out


def main(argv):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return search(argv)
    finally:
        out = buf.getvalue()
        if len(out) > CAP:  # v18: cut at a line; the hits fact keeps what was printed
            keep, size = [], 0
            for ln in out.split('\n'):
                if size + len(ln) + 1 > CAP - 80:
                    break
                keep.append(ln)
                size += len(ln) + 1
            out = '\n'.join(keep) + f'\n... cut at {CAP} characters; narrow with within=<file or dir>\n'
        sys.stdout.write(out)


def search(argv):
    try:
        text, path, opts, notes = parse(argv)
    except Misuse as e:
        print(f'USAGE ERROR: {e}')
        print(USAGE)
        return 2
    level = compact_level()
    set_level(level)
    if path and any(c in path for c in '*?['):
        bases = sorted(glob.glob(os.path.join(ROOT, path), recursive=True))
    else:
        bases = [os.path.join(ROOT, path) if path else ROOT]
    regex = 'regex' in opts
    try:
        pat = compile_pattern(text, regex, 'icase' in opts, 'word' in opts)
    except re.error as e:
        notes.append(f'invalid regex ({e}); searched for the literal text instead')
        regex = False
        pat = compile_pattern(text, False, 'icase' in opts, 'word' in opts)
    hits, used, other, retries = run_search(pat, bases), pat, False, []
    if not hits:
        retries = retries_for(text, regex, opts, path)
        for p, b, why in retries:
            hits = run_search(p, b or bases)
            if hits:
                notes.append(f'no matches for {text!r}; showing matches {why}')
                used = p
                break
        if not hits:  # v17 L9: no .py file matches: other text files (docs, config), labelled as such
            for p, b, why in [(pat, None, '')] + [r for r in retries if not r[2].startswith('for any of')]:
                hits = run_search(p, b or bases, text_files=True)
                if hits:
                    if why:
                        notes.append(f'no matches for {text!r}; showing matches {why}')
                    used, other = p, True
                    break
    for note in notes:
        print(f'NOTE: {note}')
    if not hits:
        word = IDENT.fullmatch(text.replace('\\b', ''))
        if argfix.is_placeholder(text):
            print(f'USAGE ERROR: {text!r} is a placeholder; pass the text to look for, e.g. '
                  + argfix.call('search.py', 'def export_html', 'rich/'))
            return 2
        if argfix.looks_like_path(text) and not path:
            print(f'USAGE ERROR: no matches for {text!r}, which looks like a file path: search.py searches file '
                  'contents, not file names. List files with outline.py DIR (e.g. outline.py rich/), or search '
                  'code: ' + argfix.call('search.py', 'def export_html', 'rich/'))
            return 2
        tried = [w for w in ('ignoring case' if any(r[2] == 'ignoring case' for r in retries) else '',
                             'as separate words' if any(r[2].startswith('for the words') for r in retries) else '',
                             'in non-.py text files') if w]
        print(f'no matches for {text!r}' + (f' in {path}' if path else '') + ' (none either ' + ', '.join(tried) + ')')
        if word:
            print_similar(word.group(1))
        return 1
    cache = {}

    def lines_of(rel):
        if rel not in cache:
            try:
                with open(os.path.join(ROOT, rel), encoding='utf-8', errors='replace') as fh:
                    cache[rel] = fh.read().split('\n')
            except OSError:
                cache[rel] = []
        return cache[rel]
    hits = rank(hits, used, lines_of, other)
    if other:
        set_level(-1)
        exts = ' '.join(sorted({os.path.splitext(h[1])[1] for h in hits}))
        print(f'no .py file matches{" in " + path if path else ""}; {len(hits)} '
              f'match{"es" if len(hits) != 1 else ""} in other files ({exts}):')
    for aux, rel, n, line in hits[:MAX_HITS]:
        lines = lines_of(rel)
        encl = names.enclosing(lines, n) if rel.endswith('.py') else ''
        if len(hits) <= CONTEXT_HITS and lines:
            print(f'{rel}:{n}' + (f' {encl}' if encl else '') + ':')
            for i in range(max(1, n - CONTEXT), min(len(lines), n + CONTEXT) + 1):
                print(f'{i}{">" if i == n else " "} {lines[i - 1][:CONTEXT_CHARS]}'.rstrip())
        else:
            print(f'{rel}:{n}' + (f' {encl}' if encl else '') + f': {line[:HIT_CHARS]}')
    flow.fact(hits=[(rel, n) for _, rel, n, _ in hits[:MAX_HITS]])  # also for the context form 'file:line:'
    if len(hits) > MAX_HITS:
        print(overflow_line(hits[MAX_HITS:]))
    if level and (len(hits) > MAX_HITS or CONTEXT_HITS < len(hits) <= 3):
        print(COMPACT_LINE)
    return 0


def _comment_or_doc(src):
    """(line numbers of docstrings / bare string statements, {line: column} of comments) of Python source."""
    docs, comments = set(), {}
    try:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                docs.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    except (SyntaxError, ValueError):
        pass
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                comments[tok.start[0]] = tok.start[1]
    except (tokenize.TokenError, SyntaxError):
        pass
    return docs, comments


def _whole(raw, s, e):
    """The match raw[s:e] is not part of a longer word (an alphanumeric character on neither side)."""
    left = s == 0 or not raw[s].isalnum() or not raw[s - 1].isalnum()
    right = e >= len(raw) or not raw[e - 1].isalnum() or not raw[e].isalnum()
    return left and right


def rank(hits, pat, lines_of, other=False):
    """v17 L9 (rich_3067: the url regex in rich/highlighter.py was cut off behind emoji-table and docs hits): source
    files first; then regex literals, other code with the text as a whole word, code with it inside a longer word,
    comments and docstrings; then path and line. Text files: translations (docs/xx/, README.xx.md) last."""
    if other:
        return sorted(hits, key=lambda h: (h[0], bool(LOCALE.search(h[1])), h[1], h[2]))
    info, t0 = {}, time.perf_counter()

    def tier(h):
        aux, rel, n, line = h
        lines = lines_of(rel)
        raw = lines[n - 1] if 0 < n <= len(lines) else line
        if rel not in info:
            ok = len(info) < RANK_FILES and time.perf_counter() - t0 < 1.5
            info[rel] = _comment_or_doc('\n'.join(lines)) if ok else None
        m = pat.search(raw)
        s, e = (m.start(), m.end()) if m else (0, 0)
        if info[rel] is not None:
            docs, comments = info[rel]
            if n in docs or (n in comments and s >= comments[n]):
                return 3
        elif raw.lstrip().startswith(('#', '"""', "'''")):
            return 3
        if REGEX_LIT.search(raw):
            return 0
        return 1 if not m or _whole(raw, s, e) else 2
    return sorted(hits, key=lambda h: (h[0], tier(h), h[1], h[2]))


def overflow_line(rest):
    """The hits not printed, per file with counts (source files first, then by count), so the model can narrow the
    search with within= (v16, rich_3067: the one source hit was cut off and never seen)."""
    count, aux = {}, {}
    for a, rel, n, line in rest:
        count[rel] = count.get(rel, 0) + 1
        aux[rel] = a
    files = sorted(count, key=lambda r: (aux[r], -count[r], r))
    shown = ', '.join(f'{r} ({count[r]})' for r in files[:OVERFLOW_FILES])
    more = f' and {len(files) - OVERFLOW_FILES} more files' if len(files) > OVERFLOW_FILES else ''
    return (f'... {len(rest)} more matches in {len(files)} file{"s" if len(files) != 1 else ""}: {shown}{more}; '
            'narrow with within=<file or dir>')


if __name__ == '__main__':
    sys.exit(flow.main('search.py', main))
