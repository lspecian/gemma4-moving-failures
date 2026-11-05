"""Summarise progress on the task so far and recommend one next action.

usage: status.py   (no arguments; any arguments given, in object or positional form, are ignored)

Reads the working memory the other repo-tools scripts keep in /tmp/repo_tools/state.sqlite: calls made, the
locate shortlist, code already viewed, edits, checks and hypotheses, and the budget used (calls and time since
the first call). Ends with a single NEXT line. v18c: starts with the MEMO line the footers print around a context
compaction (the code to change or the edited ranges with current line numbers, the last repro/test verdict, whether
the patch is READY).
"""

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402

MAX_ITEMS = 4


def report(st):
    refused = sum(1 for a in st.actions if a[3] == 'refused')
    ok_edits = sum(1 for e in st.edits if e[2])
    lines = [f'calls so far: {st.n} ({refused} refused) | mode: {st.mode} | edits: {ok_edits} ok, '
             f'{len(st.edits) - ok_edits} failed | checks since last edit: {len(st.checks_since_edit)} | '
             f'look-around since last edit: {st.look_since_edit}/{st.explore_budget}']
    if not st.located:
        lines.append('locate: not run yet')
    else:
        seen, tops = set(), []
        for path, symbol, start, end, score, ev in st.candidates:
            if path in seen:
                continue
            seen.add(path)
            tops.append(f'{path}' + (f' {symbol} L{start}-{end}' if symbol else ''))
        lines.append('locate top: ' + ('; '.join(tops[:MAX_ITEMS]) or '(no candidates)'))
    if st.viewed:
        lines.append('viewed: ' + flow._viewed_summary(st))
    for n, path, ok, msg in st.edits[-MAX_ITEMS:]:
        lines.append(f'edit call {n}: {path} ' + ('applied' if ok else 'FAILED' + (f' ({msg[:80]})' if msg else '')))
    for n, kind, target, code in st.checks[-MAX_ITEMS:]:
        lines.append(f'check call {n}: {kind} {target or ""} exit {code}'.replace('  ', ' '))
    for i, text, status, ev in st.hypotheses:
        lines.append(f'H{i} [{status}] {text[:120]}' + (f' -- {ev[:80]}' if ev else ''))
    return lines


def main(argv):
    conn = flow.connect()
    try:
        st = flow.State(conn)
        print(flow.memo(st) or 'MEMO: nothing recorded yet')
        for line in report(st):
            print(line)
        phase, bline = flow.budget(st)
        if bline:
            print(bline)
        facts = {'ready': st.patch_ready} if st.last and st.last[1] == 'check_patch.py' else {}
        script = 'check_patch.py' if st.last and st.last[1] == 'check_patch.py' else 'status.py'
        nxt = flow.escalate(st, script, flow.suggest(st, script, facts), phase)
        print(flow.format_footer(nxt[:1]) if nxt else 'NEXT: ' + flow.CHECK + ', then ' + flow.SUBMIT)
        return 0
    finally:
        conn.close()


if __name__ == '__main__':
    # status.py prints its own NEXT line; it is still recorded as a call (after() output is discarded).
    try:
        code = main(sys.argv[1:])
    except Exception as e:  # never leave the model without a next step
        print(f'status unavailable ({type(e).__name__}); NEXT: ' + flow.CHECK + ', then ' + flow.SUBMIT)
        code = 0
    try:
        flow.after('status.py', sys.argv[1:], code, '')
    except Exception:
        pass
    sys.exit(code)
