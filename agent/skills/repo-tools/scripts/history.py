"""What does git history say about a file, or about an issue?

usage: history.py  target=billing/invoice.py     recent commits touching the file + files changed with it
       history.py  similar=<issue title>              past commits whose subject resembles the text + their files
       (positional form, still accepted: history.py FILE  |  history.py --similar "<text>")

Uses the code knowledge base (kb.py; last 2000 commits).
"""

import datetime
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402  (flow engine: refusals + NEXT/BUDGET footers)
import argfix  # noqa: E402
import kb  # noqa: E402

ROOT = (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))
MAX_COMMITS = 10
MAX_COCHANGED = 10
MAX_SIMILAR = 8
MAX_FILES_SHOWN = 4


def day(ts):
    try:
        return datetime.datetime.fromtimestamp(int(ts), datetime.timezone.utc).strftime('%Y-%m-%d')
    except (TypeError, ValueError, OverflowError, OSError):
        return '?'


def similar(conn, text):
    rows = kb.similar_commits(text, conn=conn, k=MAX_SIMILAR)
    if not rows:
        print('no similar past commits.')
        return 1
    print('past commits with a similar subject (best first):')
    for sha, subject, files in rows:
        print(f'  {sha[:8]} {subject[:90]}')
        src = [f for f in files if f.endswith('.py') and not kb.is_aux(f)]
        rest = [f for f in files if f not in src]
        shown = (src + rest)[:MAX_FILES_SHOWN]
        if shown:
            more = len(files) - len(shown)
            print('      files: ' + ', '.join(shown) + (f' (+{more} more)' if more > 0 else ''))
    return 0


def file_history(conn, target):
    path = kb.rel_path(target)
    commits = kb.commits_for(path, conn=conn, k=MAX_COMMITS)
    if not commits:
        known = conn.execute('SELECT 1 FROM files WHERE path = ?', (path,)).fetchone()
        print(f'{path}: ' + ('no commits in the indexed history.' if known else 'not a file in the repository.'))
        return 1
    total = conn.execute('SELECT count(*) FROM commit_files WHERE path = ?', (path,)).fetchone()[0]
    print(f'{path}: {total} commit(s) in the last {kb.MAX_COMMITS}; most recent:')
    for sha, subject, when in commits:
        print(f'  {sha[:8]} {day(when)} {subject[:90]}')
    co = kb.cochanged(path, conn=conn, k=MAX_COCHANGED)
    if co:
        print('usually changed together with (commits in common):')
        for other, n in co:
            print(f'  {other} ({n})')
    return 0


USAGE = '\n'.join(__doc__.strip().splitlines()[2:5])


def main(argv):
    try:
        d, note = argfix.object_args(argv, 'history.py')
        if d is not None:
            argfix.exclusive(d, 'history.py', ('file',), ('similar',))
            argfix.require(d, 'history.py', 'similar' if 'similar' in d else 'file')
    except argfix.Misuse as e:
        print(f'USAGE ERROR: {e}')
        print(USAGE)
        return 2
    if d is not None:
        if note:
            print(f'NOTE: {note}')
        args = ['--similar', d['similar']] if 'similar' in d else [d['file']]
    else:
        args = [a for a in argv if a.strip()]
    if not args:
        print('usage: ' + USAGE.split('usage: ', 1)[-1])
        return 2
    conn = kb.open(ROOT)
    if args[0] == '--similar':
        text = ' '.join(args[1:]).replace('\\n', '\n').replace('\\"', '"')
        if not text.strip():
            print('usage: ' + USAGE.split('usage: ', 1)[-1])
            return 2
        return similar(conn, text)
    return file_history(conn, args[0])


if __name__ == '__main__':
    sys.exit(flow.main('history.py', main))
