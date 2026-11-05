"""B2 screen (LLM-free): does check_patch.module_name(rel) give the importable module name of the gold files in the
299 external toolbench workspaces? True name = the path from the parent of the top-most package dir (a chain of
dirs with __init__.py) to the file. Also flags layouts with no __init__.py chain (scripts, namespace packages)."""
import json, os, re, collections
WS = 'data/toolbench/ws'

def derived(rel):
    parts = rel[:-3].split('/')
    if parts and parts[0] == 'src':
        parts = parts[1:]
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts) if parts and all(p.isidentifier() for p in parts) else None

def true_name(ws, rel):
    parts = rel[:-3].split('/')
    d = parts[:-1]
    i = len(d)
    while i > 0 and os.path.exists(os.path.join(ws, *d[:i], '__init__.py')):
        i -= 1
    pkg = d[i:]                      # the __init__ chain ending at the file's dir
    if not pkg and parts[-1] != '__init__':
        return None, '/'.join(d[:i]) or '.'   # a module outside any package: importable only from its own dir
    mod = pkg + ([] if parts[-1] == '__init__' else [parts[-1]])
    return '.'.join(mod), '/'.join(d[:i]) or '.'

rows = [json.loads(l) for l in open('data/toolbench/external.jsonl')]
stats = collections.Counter(); roots = collections.Counter(); ex = []
for r in rows:
    ws = os.path.join(WS, r['instance_id'])
    if not os.path.isdir(ws):
        stats['no workspace'] += 1; continue
    files = [m for m in re.findall(r'^\+\+\+ b/(\S+\.py)$', r['patch'], re.M)
             if not re.search(r'(^|/)(tests?|testing)/|test_[^/]*\.py$|_test\.py$|conftest\.py$', m)]
    for rel in files:
        if not os.path.exists(os.path.join(ws, rel)):
            stats['new file'] += 1; continue
        t, root = true_name(ws, rel); d = derived(rel)
        stats['files'] += 1
        roots['root=' + ('repo' if root == '.' else ('src' if root == 'src' else 'other'))] += 1
        if t is None:
            stats['not in a package'] += 1
            if len(ex) < 6: ex.append(('nopkg', r['instance_id'], rel))
        elif t == d:
            stats['match'] += 1
        else:
            stats['MISMATCH'] += 1
            if len(ex) < 14: ex.append(('mismatch', r['instance_id'], rel, d, t))
print(dict(stats)); print(dict(roots))
for e in ex: print(' ', e)
