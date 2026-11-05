"""Where is a function/class/attribute defined and used?

usage: callers.py  name=apply_discount     (a name, Class.method, or module.func)
       (positional form, still accepted: callers.py SYMBOL)

Prints the definitions (file, line range, signature) and the places that reference the name (calls marked with
"call"), source files before tests, with the source line. Matching is by name (no type inference): for a method,
references to any attribute with that name are listed. Uses the code knowledge base (kb.py).
v18: an attribute with no def (self.X = ... in a method, a class attribute, a module-level name) lists its
assignments as definitions (code-review P8); at most MAX_SITES references (tests/docs: at most AUX_SITES; the rest
counted per file), CAP characters.
"""

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402  (flow engine: refusals + NEXT/BUDGET footers)
import argfix  # noqa: E402
import kb  # noqa: E402

ROOT = (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))
MAX_DEFS = 5
MAX_SITES = 15         # v18: was 25
AUX_SITES = 5          # v18: references in tests/docs listed at most
SRC_CHARS = 80         # chars of a reference's source line
CAP = 3500             # v18: characters of output at most


def source_line(cache, path, line):
    if path not in cache:
        try:
            with open(os.path.join(ROOT, path), encoding='utf-8', errors='replace') as fh:
                cache[path] = fh.read().split('\n')  # ast line numbering
        except OSError:
            cache[path] = []
    lines = cache[path]
    return lines[line - 1].strip()[:SRC_CHARS] if 0 < line <= len(lines) else ''


USAGE = '\n'.join(__doc__.strip().splitlines()[2:4])


def main(argv):
    try:
        d, note = argfix.object_args(argv, 'callers.py')
        if d is not None:
            argfix.require(d, 'callers.py', 'name')
    except argfix.Misuse as e:
        print(f'USAGE ERROR: {e}')
        print(USAGE)
        return 2
    if d is not None:
        args = [d['name']]
        if note:
            print(f'NOTE: {note}')
    else:
        args = [a for a in argv if a.strip()]
        if not args or args[0].startswith('-'):
            print('usage: ' + USAGE.split('usage: ', 1)[-1])
            return 2
    name = kb._clean_name(args[0].replace('\\n', ' ').split()[0])
    conn = kb.open(ROOT)
    defs = kb.find_symbol(name, conn=conn)
    sites = kb.callers(name, conn=conn)
    cache = {}
    assigns = []
    if not defs:  # v18 (P8): an attribute or variable: where it is assigned
        try:
            assigns = [r for r in kb.find_defs(name, conn) if r[2] in ('attribute', 'class attribute', 'assignment')]
        except Exception:
            assigns = []
    if not defs and not sites and not assigns:
        print(f'{name}: no definition or reference found (names only; try search.py for text)')
        return 1
    out = [f'{name}: {len(defs) + len(assigns)} definition(s), {len(sites)} reference(s) '
           f'({sum(s[3] for s in sites)} calls) in {len({s[0] for s in sites})} file(s)']
    for path, qn, kind, start, end, sig in defs[:MAX_DEFS]:
        out.append(f'  def {path}:{start}-{end} {qn}  {sig[:80]}')
    for path, line, kind, qn, owner in assigns[:MAX_DEFS]:
        where = f' in {owner}' if owner else ''
        out.append(f'  {kind} {path}:{line} {qn}{where}  | {source_line(cache, path, line)}')
    if len(defs) > MAX_DEFS or len(assigns) > MAX_DEFS:
        out.append(f'  ... {max(len(defs), len(assigns)) - MAX_DEFS} more definitions')
    src_sites = [x for x in sites if not kb.is_aux(x[0])]
    aux_sites = [x for x in sites if kb.is_aux(x[0])]
    shown = src_sites[:MAX_SITES] + aux_sites[:AUX_SITES]
    for path, qn, line, call in shown:
        where = f' in {qn}' if qn else ''
        out.append(f"  {path}:{line}{where}{' (call)' if call else ''}  | {source_line(cache, path, line)}")
    rest = [x for x in sites if x not in shown]
    if rest:
        count = {}
        for x in rest:
            count[x[0]] = count.get(x[0], 0) + 1
        files = ', '.join(f'{p} ({n})' for p, n in list(count.items())[:5]) + (' ...' if len(count) > 5 else '')
        out.append(f'  ... {len(rest)} more references: {files}')
    size = 0
    for line in out:
        if size + len(line) + 1 > CAP - 60:
            print(f'... cut at {CAP} characters')
            break
        print(line)
        size += len(line) + 1
    return 0


if __name__ == '__main__':
    sys.exit(flow.main('callers.py', main))
