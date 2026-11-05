"""Shared argument repair for the repo-tools scripts (a helper module imported by them, not a command).

Arguments reach the scripts through the model's JSON tool calls and often arrive in the wrong shape: a JSON list
packed into one string, the whole command in one string, a flag glued to its value, placeholders copied from a
usage line, /workspace/ prefixes. The scripts repair what is unambiguous (and print one NOTE line saying how the
call was read) and raise Misuse with a corrected example call for the rest; the caller prints it and exits 2.
Placeholder detection only runs after the literal reading has failed, so a legitimate call is never blocked.

The canonical call passes args as an OBJECT (argv of "--key value" pairs, see object_args below); the scripts read
it strictly, and fall back to the positional form with the repairs above. Correction examples (call) are written in
the object form, in the quote-free notation of text() (never JSON: see NOTATION below).
"""

import ast
import json
import os
import re
import shlex

ROOT = (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))  # overridable for local testing
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist', '.tox', '.mypy_cache',
             '.pytest_cache', '.eggs'}
MAX_WALK = 60000

PLACEHOLDER = re.compile(
    r'^\s*(?:<[^<>\n]{1,40}>|\{[A-Za-z _-]{1,30}\}|\.\.\.|…|'
    r'(?:START|END|FILE|PATH|SYMBOL|TEXT|OLD|NEW|NUM|LINE|LINENO|FROM|TO|CONTENT|CODE|EXPR|REGEX|PATTERN|N|M)'
    r'(?:_[A-Z]{1,12})?|'
    r'(?:path/to/)?(?:file|FILE|your_file|some_file|filename|module)(?:\.py)?|path/to/\S*)\s*$')


class Misuse(Exception):
    """Wrong arguments. str(exc) is the correction to print; the script then exits with code 2."""


def is_placeholder(s):
    return bool(PLACEHOLDER.match(str(s)))


def quote(s):
    """Show an argument the way the model should write it in a list of strings."""
    return json.dumps(str(s), ensure_ascii=False)


# NOTATION (argument schema v3). Everything the model reads shows calls as
#     show.py  path=billing/invoice.py  symbol=Invoice.total
# i.e. the script, then key=value pairs separated by two spaces, values unquoted. Never JSON: when the model copies
# JSON quotes ("key": "value") into its tool call instead of Gemma's own string delimiters, Kaggle's vLLM parser
# either hangs (list args) or, on an unbalanced brace in the value, swallows file_path/skill_name into args
# (LOG 2026-09-30, E023). Model-facing key for the repository file is "target" and search's directory filter is
# "within": with a key named "file" or "path" inside args the model skips ADK's file_path (probe P003/P004: the
# edit call became args:{new,old,path,skill_name} with no file_path). Internally the scripts keep d["file"] and
# search's d["path"]; every old name stays accepted (alias_pairs).
SHOWN_KEY = {'file': 'target', 'path': 'within'}     # internal key -> key the model sees
FILE_ALIASES = ('target', 'path', 'within', 'file_path', 'filepath', 'filename', 'file_name', 'dir', 'directory', 'in')
STRAY_KEYS = ('skill_name',)                          # ADK's own parameters, misplaced inside args


def text(script, d=None):
    """A call in the notation the model reads: `show.py  path=a.py  symbol=f` (`check_patch.py` alone: no args)."""
    pairs = [f'{SHOWN_KEY.get(k, k)}={" ".join(str(v).split())}' for k, v in (d or {}).items()]
    return '  '.join([script] + pairs)


KEY_ALIASES = {'search.py': {'query': 'text'}}   # other names the model used for a key (flow.KEY_ALIASES)


def file_key(keys):
    """The internal key naming the repository file(s) of a script: file, path (search.py) or files (outline.py)."""
    return next((k for k in ('file', 'path', 'files') if k in keys), None)


def alias_pairs(script, pairs, notes=None):
    """Map model-facing/alias keys onto the script's internal keys; drop ADK parameters misplaced inside args
    (skill_name, or file_path naming the script itself). pairs: [(name, value)] -> [(name, value)].
    Renames the model would not expect (target -> files, query -> text) are reported in `notes`."""
    keys = KEYS.get(script, ())
    target = file_key(keys)
    extra = KEY_ALIASES.get(script, {})
    names = [x for x, _ in pairs]
    out = []
    for n, v in pairs:
        if n in STRAY_KEYS:
            continue
        if n in ('file_path', 'filepath') and re.fullmatch(r'(?:scripts/)?' + re.escape(script), str(v).strip()):
            continue
        if target and n in FILE_ALIASES + ('file',) and n != target and target not in names:
            if target == 'files' and notes is not None:
                notes.append(f'read {n} as files: ' + text(script, {'files': v}))
            n = target
        elif n in extra and n not in keys and extra[n] not in names and 'regex' not in names:
            if notes is not None:
                notes.append(f'read {n} as {extra[n]}')
            n = extra[n]
        out.append((n, v))
    return out


def call(script, *args):
    """A corrected example call, written in the object form (see to_object) whenever it can be expressed so."""
    d = to_object(script, [str(a) for a in args])
    if d is not None:
        return obj(script, d)
    return '  '.join([script] + [' '.join(str(a).split()) for a in args])


# ---------------------------------------------------------------------------------------------------------------
# Object-form arguments (argument schema v2, research/notes/6-flow-design.md).
# Array-form tool-call arguments can freeze Kaggle's vLLM Gemma 4 tool parser (an unquoted array element that
# contains ']'), so the canonical call passes args as an OBJECT. ADK's run_skill_script turns args={"k": "v", ...}
# into argv ["--k", "v", ...] (insertion order, values stringified). The positional form keeps working.
# Detection (object_args): argv is the object form when it is empty, when it is one string holding a JSON object
# whose keys are all keys of the script, or when it has an even length and every even-indexed token is "--name"
# (name = identifier, case-insensitive). The values (odd-indexed tokens) are taken strictly by position, so a value
# may start with "-"/"--", contain newlines, brackets, quotes or commas. Two exceptions keep old positional calls
# working: no name is a key but one is a flag of the positional form (search.py ["--recursive", "x"]), or every
# name is such a flag, none is a required key and a valueless flag is followed by something other than yes/no
# (locate.py ["--k", "2", "--show", "<query>"]). Any other
# unknown or repeated key is a USAGE ERROR listing the valid keys with an example.

EDIT_HUNK_KEYS = tuple(f'{k}{n}' for n in range(2, 5) for k in ('start', 'end', 'new', 'old'))  # v20 multi-edit
NUMBERED_KEY = re.compile(r'[a-z]+[2-9]$')
KEYS = {
    'locate.py': ('query', 'k', 'show'),
    'show.py': ('file', 'symbol', 'start', 'end'),
    'outline.py': ('files', 'grep'),
    'search.py': ('text', 'regex', 'path'),
    'callers.py': ('name',),
    'tests_for.py': ('target',),
    'history.py': ('file', 'similar'),
    'edit.py': ('file', 'old', 'new', 'start', 'end') + EDIT_HUNK_KEYS,
    'create.py': ('file', 'content'),
    'run.py': ('test', 'k', 'snippet'),
    'check_patch.py': ('clean',),
}
EXAMPLES = {
    'locate.py': [{'query': 'Invoice.total returns None when the amount is missing'}],
    'show.py': [{'file': 'billing/invoice.py', 'symbol': 'Invoice.total'},
                {'file': 'billing/invoice.py', 'start': '140', 'end': '200'}],
    'outline.py': [{'files': 'billing/invoice.py billing/tax.py'}, {'files': 'billing/invoice.py', 'grep': 'total'}],
    'search.py': [{'text': 'def apply_discount'}, {'regex': 'def (total|subtotal)', 'path': 'billing/'}],
    'callers.py': [{'name': 'apply_discount'}],
    'tests_for.py': [{'target': 'billing/invoice.py'}, {'target': 'Invoice.total'}],
    'history.py': [{'file': 'billing/invoice.py'}, {'similar': 'Invoice total ignores discounts'}],
    'edit.py': [{'file': 'billing/invoice.py', 'old': 'if self.amount == None:', 'new': 'if self.amount is None:'},
                {'file': 'billing/invoice.py', 'start': '151', 'end': '151',
                 'new': '        if self.amount is None:'}],
    'create.py': [{'file': 'billing/currency.py', 'content': 'RATES = {}\n'}],
    'run.py': [{'test': 'tests/test_invoice.py', 'k': 'total'}, {'snippet': 'from billing.invoice import Invoice'}],
    'check_patch.py': [{}, {'clean': 'yes'}],
}
REQUIRED = {  # at least one of these keys makes a complete call (check_patch.py: none needed)
    'locate.py': ('query',), 'show.py': ('file',), 'outline.py': ('files',), 'search.py': ('text', 'regex'),
    'callers.py': ('name',), 'tests_for.py': ('target',), 'history.py': ('file', 'similar'), 'edit.py': ('file',),
    'create.py': ('file',), 'run.py': ('test', 'snippet'), 'check_patch.py': (),
}
POSITIONAL_FLAGS = {'locate.py': {'k', 'show'}, 'search.py': {'regex', 'recursive'}, 'outline.py': {'grep'},
                    'history.py': {'similar'}, 'check_patch.py': {'clean'}, 'edit.py': {'lines'},
                    'show.py': {'lines'}}
BARE_FLAGS = {'show', 'clean', 'recursive'}  # positional flags that take no value
KEY_TOKEN = re.compile(r'--([A-Za-z_][A-Za-z0-9_]*)')
YES = {'yes', 'y', 'true', '1', 'on'}
NO = {'no', 'n', 'false', '0', 'off', '', 'none'}


def obj(script, d):
    """An object-form call as the model should read it: `show.py  path=a.py  symbol=f` (see text())."""
    return text(script, d)


def examples(script):
    return '\n  or  '.join(obj(script, d) for d in EXAMPLES.get(script, [{}]))


def first_example(script, key=None):
    """One example call (v18, note 13: usage errors carry a single example): the first, or the first using KEY."""
    exs = EXAMPLES.get(script, [{}])
    return obj(script, next((d for d in exs if key in d), exs[0]) if key else exs[0])


def shown_keys(script):
    """The keys a usage message names (v20: not edit.py's keys of further hunks, start2 ... old4)."""
    return [SHOWN_KEY.get(k, k) for k in KEYS[script] if not (script == 'edit.py' and NUMBERED_KEY.match(k))]


def usage_line(script):
    return f'valid keys for {script}: ' + ', '.join(shown_keys(script)) + '. E.g. ' + first_example(script)


def _scalar(v):
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if isinstance(v, (list, tuple)):
        return ' '.join(str(x) for x in v)
    return '' if v is None else str(v)


def _as_dict(s, script):
    """A JSON object packed into one string, when every key is one of the script's keys; else None."""
    t = str(s).strip()
    if not (t.startswith('{') and t.endswith('}')):
        return None
    for load in (lambda x: json.loads(x, strict=False), ast.literal_eval):
        try:
            v = load(t)
        except Exception:
            continue
        if isinstance(v, dict) and all(isinstance(k, str) and k.lower() in KEYS[script] for k in v):
            return {k.lower(): _scalar(x) for k, x in v.items()}
    return None


def object_args(argv, script):
    """Read the object form. Returns (dict, note) for the object form, (None, None) for the positional form.
    Raises Misuse for unknown or repeated keys."""
    argv = [str(a) for a in (argv or [])]
    keys = KEYS[script]
    if not argv:
        return {}, None
    if len(argv) == 1:
        d = _as_dict(argv[0], script)
        if d is not None:
            return d, 'the arguments arrived as one JSON-encoded string; pass args as an object'
        return None, None
    if len(argv) % 2:
        return None, None
    names = []
    for tok in argv[0::2]:
        m = KEY_TOKEN.fullmatch(tok)
        if not m:
            return None, None
        names.append(m.group(1).lower())
    renamed = []
    pairs = alias_pairs(script, list(zip(names, argv[1::2])), renamed)
    if not pairs:
        return {}, None
    names, argv = [n for n, _ in pairs], [x for n, v in pairs for x in ('--' + n, v)]
    flags = POSITIONAL_FLAGS.get(script, set())
    if not any(n in keys for n in names) and any(n in flags for n in names):
        return None, None  # e.g. search.py ["--recursive", "x"]: the positional form with a grep flag
    if REQUIRED[script] and not any(n in REQUIRED[script] for n in names) and all(n in flags for n in names) \
            and any(n in BARE_FLAGS and v.strip().lower() not in YES | NO for n, v in zip(names, argv[1::2])):
        return None, None  # locate.py ["--k", "2", "--show", "<query>"]: a valueless flag, then the query
    d = {}
    for n, v in zip(names, argv[1::2]):
        if n not in keys:
            bad = [x for x in names if x not in keys]
            raise Misuse(f'unknown key{"s" if len(bad) > 1 else ""} ' + ', '.join(bad)
                         + f' for {script}; ' + usage_line(script))
        if n in d:
            raise Misuse(f'key {SHOWN_KEY.get(n, n)} given twice; ' + usage_line(script))
        d[n] = v
    return d, ('; '.join(renamed) or None)


def require(d, script, *keys, example=None):
    """Raise Misuse naming the missing keys (with an example) unless all KEYS are present and non-blank."""
    missing = [k for k in keys if not str(d.get(k, '')).strip()]
    if missing:
        ex = obj(script, example) if example is not None else first_example(script)
        raise Misuse(f'{script} needs the key{"s" if len(missing) > 1 else ""} '
                     + ', '.join(SHOWN_KEY.get(k, k) for k in missing)
                     + f' (valid keys: {", ".join(shown_keys(script))}). E.g. ' + ex)


def exclusive(d, script, *groups):
    """Raise Misuse when keys from more than one GROUP (tuples of keys) are present."""
    present = [g for g in groups if any(k in d for k in g)]
    if len(present) > 1:
        names = ' and '.join('/'.join(SHOWN_KEY.get(k, k) for k in g if k in d) for g in present)
        raise Misuse(f'{script} takes {names} in different calls, not together. E.g. ' + first_example(script))


def flag(d, key, script):
    """A yes/no value."""
    v = str(d.get(key, 'no')).strip().strip('"\'').lower()
    if v in YES:
        return True
    if v in NO:
        return False
    raise Misuse(f'{key} must be yes or no, got {d[key]}. E.g. ' + first_example(script, key))


def to_positional(script, argv):
    """The positional argv equivalent of a call in either form (for identity/comparison, e.g. the flow engine).
    A positional call, or an object call with bad keys, is returned unchanged."""
    argv = [str(a) for a in (argv or [])]
    try:
        d, _ = object_args(argv, script)
    except Misuse:
        return argv
    if d is None:
        return argv
    g = d.get
    if script == 'locate.py':
        out = (['--k', g('k')] if 'k' in d else []) + (['--no-show'] if g('show', 'yes').lower() in NO else [])
        return out + ([g('query')] if 'query' in d else [])
    if script == 'show.py':
        return [x for x in (g('file'), g('symbol'), g('start'), g('end')) if x is not None]
    if script == 'outline.py':
        return (['--grep', g('grep')] if 'grep' in d else []) + split_list(g('files', ''))
    if script == 'search.py':
        return (['-E', g('regex')] if 'regex' in d else []) + ([g('text')] if 'text' in d else []) \
            + ([g('path')] if 'path' in d else [])
    if script == 'history.py':
        return ['--similar', g('similar')] if 'similar' in d else [x for x in (g('file'),) if x is not None]
    if script == 'edit.py':
        if 'start' in d or 'end' in d:
            return ['--lines'] + [x for x in (g('file'), g('start'), g('end'), g('new')) if x is not None]
        return [x for x in (g('file'), g('old'), g('new')) if x is not None]
    if script == 'run.py':
        if 'snippet' in d:
            return ['snippet', g('snippet')]
        return (['test', g('test')] if 'test' in d else []) + (['-k', g('k')] if 'k' in d else [])
    if script == 'check_patch.py':
        return ['--clean'] if 'clean' in d and g('clean').strip().lower() in YES else []
    return [d[k] for k in KEYS[script] if k in d]  # callers, tests_for, create: keys in positional order


def to_object(script, args):
    """The object form of a positional call (for correction examples), or None if it cannot be expressed."""
    a = [str(x) for x in args]
    if script not in KEYS:
        return None
    if script == 'locate.py':
        d = {}
        while a and a[0] in ('--k', '--no-show', '--show'):
            if a[0] == '--k' and len(a) > 1:
                d['k'], a = a[1], a[2:]
            else:
                d['show'], a = ('no' if a[0] == '--no-show' else 'yes'), a[1:]
        return {'query': ' '.join(a), **d} if a else None
    if script == 'show.py':
        if len(a) == 2:
            n = line_number(a[1])
            return {'file': a[0], 'start': a[1]} if n is not None else {'file': a[0], 'symbol': a[1]}
        if len(a) == 3:
            return {'file': a[0], 'start': a[1], 'end': a[2]}
        return {'file': a[0]} if len(a) == 1 else None
    if script == 'outline.py':
        d = {}
        if len(a) >= 2 and a[0] == '--grep':
            d['grep'], a = a[1], a[2:]
        return {'files': ' '.join(a), **d} if a and not any(x.startswith('-') for x in a) else None
    if script == 'search.py':
        key = 'text'
        if a and a[0] in ('-E', '--regex', '-P'):
            key, a = 'regex', a[1:]
        if not a or len(a) > 2 or any(x.startswith('-') and len(x) <= 3 for x in a[:1]):
            return None
        return {key: a[0], **({'path': a[1]} if len(a) == 2 else {})}
    if script == 'callers.py':
        return {'name': a[0]} if len(a) == 1 else None
    if script == 'tests_for.py':
        return {'target': a[0]} if len(a) == 1 else None
    if script == 'history.py':
        if a and a[0] == '--similar':
            return {'similar': ' '.join(a[1:])} if len(a) > 1 else None
        return {'file': a[0]} if len(a) == 1 else None
    if script == 'edit.py':
        if len(a) == 5 and a[0] == '--lines':
            return {'file': a[1], 'start': a[2], 'end': a[3], 'new': a[4]}
        return {'file': a[0], 'old': a[1], 'new': a[2]} if len(a) == 3 else None
    if script == 'create.py':
        return {'file': a[0], 'content': a[1]} if len(a) == 2 else None
    if script == 'run.py':
        if len(a) == 2 and a[0] == 'snippet':
            return {'snippet': a[1]}
        if len(a) in (2, 4) and a[0] == 'test' and (len(a) == 2 or a[2] == '-k'):
            return {'test': a[1], **({'k': a[3]} if len(a) == 4 else {})}
        return None
    if script == 'check_patch.py':
        return {'clean': 'yes'} if a == ['--clean'] else ({} if not a else None)
    return None


def split_list(s):
    """'a.py b.py', 'a.py, b.py', '["a.py", "b.py"]' -> ['a.py', 'b.py']."""
    lst = _as_list(s) if str(s).strip().startswith('[') else None
    if lst is not None:
        return [x for x in lst if x.strip()]
    return [x for x in re.split(r'[\s,]+', str(s)) if x]


def _as_list(s):
    """A JSON/Python list that was packed into one string ('["a.py", "1", "9"]' or 'a.py", "1", "9')."""
    t = s.strip()
    cands = [(t, False)] if t.startswith('[') and t.endswith(']') else []
    m = re.match(r'''\[?\s*["']?([^\s"',]+)["'], ["']''', t)  # first item must be a path, flag or word
    if m:
        body = t.strip('[]').strip()
        if not body.startswith(('"', "'")):
            body = '"' + body
        if not body.endswith(('"', "'")):
            body += '"'
        cands.append(('[' + body + ']', True))
    for c, fragment in cands:
        for load in (lambda x: json.loads(x, strict=False), ast.literal_eval):
            try:
                v = load(c)
            except Exception:
                continue
            if isinstance(v, (list, tuple)) and v and all(isinstance(x, (str, int, float)) for x in v) \
                    and (not fragment or len(v) >= 2):
                return [str(x) for x in v]
    return None


def split(s):
    """Shell-like split that never fails (falls back to whitespace)."""
    try:
        return shlex.split(s)
    except ValueError:
        return s.split()


def unpack(argv, script):
    """Undo JSON-list packing and a leading script name. Returns (argv, note-or-None)."""
    argv = [str(a) for a in (argv or [])]
    note = None
    if len(argv) == 1:
        lst = _as_list(argv[0])
        if lst is not None:
            argv, note = lst, 'the arguments arrived as one JSON-encoded string; pass args as an object'
    name = re.compile(r'(?:python3?\s+)?(?:scripts/)?' + re.escape(script))
    if argv and name.fullmatch(argv[0].strip()) and not exists(clean_path(argv[0])):
        argv, note = argv[1:], note or f'dropped the script name {script} from the arguments'
    elif argv and name.match(argv[0].strip()) and re.match(name.pattern + r'[\s,]', argv[0].strip()):
        rest = argv[0].strip()[name.match(argv[0].strip()).end():].strip().lstrip(',').strip()
        lst = _as_list(rest)
        argv = (lst if lst is not None else split(rest)) + argv[1:]
        note = f'dropped the script name {script} and split the command into arguments'
    return argv, note


def clean_path(p):
    """Normalise a path argument: quotes, backticks, /workspace/ and ./ prefixes, trailing ':'."""
    p = str(p).strip().strip('`"\'').strip()
    for prefix in (ROOT.rstrip('/') + '/', '/workspace/'):
        if p.startswith(prefix):
            p = p[len(prefix):]
    if p in (ROOT.rstrip('/'), '/workspace'):
        p = '.'
    p = p.lstrip('/')
    while p.startswith('./'):
        p = p[2:]
    return p.rstrip(':') or '.'


def exists(rel):
    return os.path.exists(os.path.join(ROOT, rel))


def looks_like_path(s):
    s = str(s).strip()
    return bool(s) and ' ' not in s and '\n' not in s and (
        '/' in s or bool(re.search(r'\.(py|md|txt|toml|cfg|ini|ya?ml|json|rst|html|js|css)$', s)))


def all_files():
    out = []
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith('.'))
        rel_d = os.path.relpath(d, ROOT)
        for f in sorted(files):
            if not f.startswith('.'):
                out.append(f if rel_d == '.' else f'{rel_d}/{f}')
        if len(out) > MAX_WALK:
            break
    return out


def resolve(rel, script, example_args=(), nearby=False):
    """Return (rel, note) for an existing path; repair a/ b/ diff prefixes or a unique path suffix; else Misuse.
    nearby=True (read-only scripts): a path that does not exist but whose file name exists exactly once in the
    repository (outside tests/docs) is read as that file."""
    rel = clean_path(rel)
    if exists(rel):
        return rel, None
    for pre in ('a/', 'b/'):
        if rel.startswith(pre) and exists(rel[2:]):
            return rel[2:], f'read {rel} as {rel[2:]}'
    tail = rel.rstrip('/')
    hits = [f for f in all_files() if f == tail or f.endswith('/' + tail)] if tail else []
    if len(hits) == 1:
        return hits[0], f'{rel} is not at the repository root; using {hits[0]}'
    if is_placeholder(rel):
        raise Misuse(f'{rel!r} is a placeholder: pass a real path relative to /workspace, e.g. '
                     + call(script, 'rich/console.py', *example_args))
    if hits:
        raise Misuse(f'{rel}: not found; did you mean one of: ' + ', '.join(hits[:6]))
    base = os.path.basename(tail)
    near = [f for f in all_files() if os.path.basename(f) == base][:6] if base else []
    if nearby and base and '.' in base:
        src = [f for f in near if not re.search(r'(^|/)(tests?|testing|docs|docs_src|examples?)/', f)]
        if len(src) == 1:
            return src[0], f'{rel} does not exist; using {src[0]}, the only file named {base}'
    raise Misuse(f'{rel}: not found (paths are relative to /workspace)'
                 + ('; files with that name: ' + ', '.join(near) if near else ''))


def line_number(tok):
    """'120', 'L120', 'line 120', '`120`', '120,' -> 120; anything else -> None."""
    t = str(tok).strip().strip('`"\'').strip().rstrip(',:;').strip()
    m = re.fullmatch(r'(?:[Ll](?:ine)?\s*)?(\d{1,7})', t)
    return int(m.group(1)) if m else None


def line_range(tok):
    """'10-20', '10:20', 'L10-L20', '10..20' -> (10, 20); else None."""
    t = str(tok).strip().strip('`"\'').strip()
    m = re.fullmatch(r'[Ll]?(\d{1,7})\s*(?:-|:|\.\.|,|to)\s*[Ll]?(\d{1,7})', t)
    return (int(m.group(1)), int(m.group(2))) if m else None


if __name__ == '__main__':
    print('argfix.py is a helper module used by the other repo-tools scripts; it is not a command.')
