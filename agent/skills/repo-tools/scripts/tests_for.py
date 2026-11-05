"""Which tests exercise a source file or symbol? Prints ready run.py arguments.

usage: tests_for.py  target=billing/invoice.py   or   tests_for.py  target=Invoice.total
       (positional form, still accepted: tests_for.py FILE|SYMBOL)

Ranks test functions that reference the names defined in the file (rare names count more) or the symbol, plus
test files that import the module or are named after it. Prints the run.py args (object form) for the best ones.
Uses the code knowledge base (kb.py).
"""

import collections
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402  (flow engine: refusals + NEXT/BUDGET footers)
import argfix  # noqa: E402
import kb  # noqa: E402

ROOT = (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))
MAX_FILES = 3
MAX_NAMES = 3


USAGE = '\n'.join(__doc__.strip().splitlines()[2:4])


def main(argv):
    try:
        d, note = argfix.object_args(argv, 'tests_for.py')
        if d is not None:
            argfix.require(d, 'tests_for.py', 'target')
    except argfix.Misuse as e:
        print(f'USAGE ERROR: {e}')
        print(USAGE)
        return 2
    if d is not None:
        args = [d['target']]
        if note:
            print(f'NOTE: {note}')
    else:
        args = [a for a in argv if a.strip()]
        if not args or args[0].startswith('-'):
            print('usage: ' + USAGE.split('usage: ', 1)[-1])
            return 2
    target = args[0].replace('\\n', ' ').split()[0]
    conn = kb.open(ROOT)
    tests = kb.tests_for(target, conn=conn, k=30)
    if not tests:
        print(f'no tests found for {target} (no test references it). Check with search.py, or verify with '
              'run.py snippet.')
        return 1
    by_file = collections.OrderedDict()
    for path, name in tests:
        by_file.setdefault(path, [])
        if name and name not in by_file[path]:
            by_file[path].append(name)
    print(f'tests for {kb.rel_path(target)} (best first):')
    for path, names in list(by_file.items())[:MAX_FILES + 3]:
        shown = ', '.join(names[:5]) + (f' (+{len(names) - 5} more)' if len(names) > 5 else '')
        print(f'  {path}' + (f': {shown}' if shown else ' (imports the module)'))
    print('run one of these:')
    whole_file = kb.is_file_target(target)  # for a file: run whole test files; for a symbol: select tests with -k
    for path, names in list(by_file.items())[:MAX_FILES]:
        cmd = {'test': path, **({'k': ' or '.join(names[:MAX_NAMES])} if names and not whole_file else {})}
        print('  ' + '  '.join(['run.py'] + [f'{k}={v}' for k, v in cmd.items()]))
    return 0


if __name__ == '__main__':
    sys.exit(flow.main('tests_for.py', main))
