"""Track 2-3 guesses about where and why the bug is, for issues that name few identifiers.

usage: hypothesis.py  add=<guess>  add2=<guess 2>  add3=<guess 3>   e.g. hypothesis.py  add=rich/console.py:Console.print drops the style
       hypothesis.py  list=yes
       hypothesis.py  confirm=1  evidence=<what the run showed>        e.g. hypothesis.py  confirm=1  evidence=printed None
       hypothesis.py  reject=2  evidence=<what the run showed>
       (positional form, still accepted: add "<guess>" ... | list | confirm ID "<evidence>" | reject ID "<evidence>")

Routine: add 2-3 concrete guesses (name the file and function), test each with run.py snippet, mark it confirmed
or rejected with what the output showed, then edit where a hypothesis is confirmed. State lives in
/tmp/repo_tools/state.sqlite and is shared with the other scripts' NEXT hints.
"""

import os
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402

MAX_TEXT = 300
MAX_HYPOTHESES = 6
KEYS = flow.KEYS['hypothesis.py']
ADD = re.compile(r'add(\d*)$')


def usage(error=None):
    if error:
        print(f'USAGE ERROR: {error}')
    print('\n'.join(__doc__.strip().splitlines()[2:7]))
    return 2


def show(conn):
    rows = conn.execute('SELECT id, text, status, evidence FROM hypotheses ORDER BY id').fetchall()
    if not rows:
        print('no hypotheses yet')
    for i, text, status, ev in rows:
        print(f'H{i} [{status}] {text}' + (f'  -- {ev}' if ev else ''))


def parse_id(s):
    s = str(s).strip().strip('"\'').lstrip('#').upper().lstrip('H')
    return int(s) if s.isdigit() else None


def parse(argv):
    """(command, texts or id, evidence) from either form, or (None, error message, None)."""
    args = flow.norm_args(argv)
    obj = flow.object_args(args)
    if obj is not None:
        bad = [k for k in obj if k not in KEYS and not ADD.match(k)]
        if bad:
            return None, (f'unknown key{"s" if len(bad) > 1 else ""} {", ".join(bad)}; valid keys: '
                          'add (add2, add3), list, confirm, reject, evidence. '
                          'Example: hypothesis.py  add=pkg/mod.py:func does X wrong because Y'), None
    d = flow.canon('hypothesis.py', args)
    adds = sorted(((int(ADD.match(k).group(1) or 1), v) for k, v in d.items() if ADD.match(k)))
    verdicts = [k for k in ('confirm', 'reject') if k in d]
    if adds:
        return 'add', [v.strip() for _, v in adds if v.strip()], None
    if len(verdicts) == 1:
        return verdicts[0], d[verdicts[0]], d.get('evidence', '').strip()
    if len(verdicts) > 1:
        return None, 'give either "confirm" or "reject", not both', None
    if 'list' in d:
        return 'list', None, None
    return None, None, None


def main(argv):
    cmd, val, ev = parse(argv)
    if cmd is None:
        return usage(val)
    conn = flow.connect()
    try:
        if cmd == 'add':
            if not val:
                return usage('"add" needs the text of a guess')
            have = conn.execute('SELECT COUNT(*) FROM hypotheses').fetchone()[0]
            for text in val[:max(0, MAX_HYPOTHESES - have)]:
                conn.execute("INSERT INTO hypotheses(text, status, evidence) VALUES(?, 'untested', '')",
                             (text[:MAX_TEXT],))
            if have + len(val) > MAX_HYPOTHESES:
                print(f'kept the first {MAX_HYPOTHESES} hypotheses; test those before adding more')
            flow.set_meta(conn, 'mode', 'hypothesis')
        elif cmd in ('confirm', 'reject'):
            hid = parse_id(val)
            if hid is None:
                return usage(f'"{cmd}" needs a hypothesis number, e.g. hypothesis.py  {cmd}=1  evidence=<what the run showed>')
            row = conn.execute('SELECT id FROM hypotheses WHERE id=?', (hid,)).fetchone()
            if not row:
                print(f'no hypothesis H{hid}; current list:')
                show(conn)
                return 1
            conn.execute('UPDATE hypotheses SET status=?, evidence=? WHERE id=?',
                         ('confirmed' if cmd == 'confirm' else 'rejected', (ev or '')[:MAX_TEXT], hid))
        show(conn)
        return 0
    finally:
        conn.close()


if __name__ == '__main__':
    sys.exit(flow.main('hypothesis.py', main))
