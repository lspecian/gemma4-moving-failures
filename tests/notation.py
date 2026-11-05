"""The call notation of argument schema v3, as the tests read it (written independently of argfix.text/flow.cmd).

Everything the model reads (NEXT footers, usage lines, Misuse corrections) writes a call as the script followed by
key=value pairs separated by TWO spaces, values unquoted, never JSON:
    show.py  target=billing/invoice.py  symbol=Invoice.total
    check_patch.py                       (no args: the script name alone)
Text after a call is separated from its last value by two spaces. Parse rule: find a script token; split what
follows on two spaces; consume consecutive parts matching ^[a-z_0-9]+= as key=value; stop at the first part that
does not. The model-facing key for the repository file is "target" (the scripts read it as "file" internally) and
search.py's directory filter is "within" (internally "path"): probes showed that ANY args key named "file" or "path"
makes the model skip ADK's own file_path parameter, so no suggested call may use those names (they still work as
aliases when the model sends them).
"""

import re

SHOWN_KEYS = {  # the keys the model may see / use, per script
    'locate.py': ('query', 'k', 'show'),
    'show.py': ('target', 'symbol', 'start', 'end'),
    'outline.py': ('files', 'grep'),
    'search.py': ('text', 'regex', 'within'),
    'callers.py': ('name',),
    'tests_for.py': ('target',),
    'history.py': ('target', 'similar'),
    'edit.py': ('target', 'old', 'new', 'start', 'end'),
    'create.py': ('target', 'content'),
    'run.py': ('test', 'k', 'snippet'),
    'check_patch.py': ('clean',),
    'hypothesis.py': ('add', 'add2', 'add3', 'list', 'confirm', 'reject', 'evidence'),
    'status.py': (),
}
MORE_KEYS = {  # keys a call may use that usage messages do not list (v20 multi-edit: further hunks)
    'edit.py': tuple(f'{k}{n}' for n in range(2, 5) for k in ('start', 'end', 'new', 'old')),
}
SCRIPT = re.compile(r'(?<![\w/.-])(' + '|'.join(re.escape(s) for s in SHOWN_KEYS) + r')(?![\w/])')
PAIR = re.compile(r'([a-z_0-9]+)=(.*)', re.S)


def render(script, **args):
    """The exact rendering of a call: `show.py  target=a.py  symbol=f`; `check_patch.py` without args."""
    return '  '.join([script] + [f'{k}={v}' for k, v in args.items()])


def calls(text):
    """[(script, {key: value})] for every call written in the notation, in order of appearance."""
    out = []
    for line in str(text).splitlines():
        pos = 0
        while True:
            m = SCRIPT.search(line, pos)
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


def argv(args):
    """The argv ADK builds from object args {k: v}: ['--k', 'v', ...] in insertion order."""
    return [x for k, v in args.items() for x in ('--' + k, str(v))]


def assert_notation(text):
    """No JSON or old renderings anywhere in TEXT, and every call in it parses with valid keys and two-space
    separators. Returns the parsed calls."""
    text = str(text)
    for bad in ('args=', 'run_skill_script(', '{"', "{'", 'skill_name', 'file_path='):
        assert bad not in text, (bad, text)
    assert not re.search(r'"\w+":', text), text                 # no JSON-quoted key
    assert not re.search(r'\.py \[', text), text                # no array args
    assert not re.search(r'\.py [a-z_0-9]+=', text), text       # key after a single space
    assert not re.search(r'(?<![\w-])(?:file|path)=', text), text  # the names that make the model skip file_path
    found = calls(text)
    for script, args in found:
        keys = SHOWN_KEYS[script] + MORE_KEYS.get(script, ())
        assert set(args) <= set(keys), (script, args, text)
        for k, v in args.items():
            assert '  ' not in v and '\n' not in v, (script, k, v)
            if keys:  # a following key glued on with a single space would end up inside the value
                assert not re.search(r'(?:^| )(?:' + '|'.join(keys) + r')=', v), (script, k, v)
    return found
