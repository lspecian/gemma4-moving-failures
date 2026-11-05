"""locate_bench: a fast, in-process benchmark of locate.py's ranking (the toolbench's locate phase, minus the flow).

    .venv/bin/python scripts/locate_bench.py [--agent agent/ | REF | DIR] [--source external,local]
        [--queries built,recorded] [--workers 8] [--label NAME] [--tasks ID,ID] [--show N]
    .venv/bin/python scripts/locate_bench.py --compare results/locate_bench/A results/locate_bench/B

Each task's workspace is prepared once into results/locate_bench/ws/<id> and reused:
  local     data/competition/snapshots/<id>.tgz, extracted and committed like the harness (replay_tools.prepare);
            task list experiments/eval113.txt.
  external  the toolbench workspaces data/toolbench/ws/<id> are shared and READ-ONLY here: a `git clone --shared`
            of them checked out at their _swegemma_baseline commit (same HEAD, history and tracked files).
The KB of each task is cached in results/locate_bench/kb/<hash of kb.py>/<id> (REPO_TOOLS_DIR), so a pass over the
411 tasks only opens the KBs and ranks. The agent's locate.py is imported in each worker; locate.locate() runs on
the workspace with stdout captured (k=4, preview on, as the agent calls it), the flow footer is not produced. The
query is toolbench.build_query() of the statement ("built") or, for local tasks, the query the model sent in
recorded runs (data/toolbench/queries_v18c_recorded.json; "recorded", missing ones fall back to built).
Gold and metrics as scripts/toolbench.py's locate phase: file@1/@3/printed (first gold file among the printed files),
gold fn in the listed functions, best match (`--- best match:` preview) = gold fn, preview shown, preview precision
(best = gold when shown), gold span printed, output tokens (JSON-encoded body chars / 3.2; the toolbench's figure also
counts the NEXT/BUDGET footer), plus the gold file's rank in the full ranking (MRR, @10) and the locate time.
Outputs: results/locate_bench/<label>/<source>_<queries>.jsonl (one record per task: ranking top 10 with scores and
evidence, gold, output) and summary.txt.
"""

import argparse
import collections
import concurrent.futures
import contextlib
import hashlib
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import toolbench as tb  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'locate_bench')
WS = os.path.join(OUT, 'ws')
RECORDED = os.path.join(ROOT, 'data', 'toolbench', 'queries_v18c_recorded.json')
PY313 = '/opt/homebrew/bin/python3.13'

# ------------------------------------------------------------------ workspaces


def ws_dir(task):
    return os.path.join(WS, task['instance_id'])


def ws_ok(task):
    return os.path.exists(os.path.join(ws_dir(task), '.git', 'locate_bench_ok'))


def prepare(task):
    """The task's cached workspace (created on first use); see the module docstring."""
    ws = ws_dir(task)
    if ws_ok(task):
        return ws
    shutil.rmtree(ws, ignore_errors=True)
    os.makedirs(WS, exist_ok=True)
    if task['source'] == 'local':
        import replay_tools
        replay_tools.prepare(os.path.join(tb.SNAPSHOTS, f'{task["instance_id"]}.tgz'), ws)
    else:
        src = os.path.join(tb.TB, 'ws', task['instance_id'])
        sha = subprocess.run(['git', '-C', src, 'rev-parse', tb.BASELINE + '^{commit}'], capture_output=True,
                             text=True, check=True).stdout.strip()
        subprocess.run(['git', 'clone', '-q', '--shared', '--no-checkout', src, ws], check=True, capture_output=True)
        subprocess.run(['git', '-C', ws, '-c', 'advice.detachedHead=false', 'checkout', '-q', '-B', 'main', sha],
                       check=True, capture_output=True)
    open(os.path.join(ws, '.git', 'locate_bench_ok'), 'w').write(task['instance_id'] + '\n')
    return ws


# ------------------------------------------------------------------ worker

_AG = {}


def init_worker(scripts, overrides=()):
    sys.dont_write_bytecode = True
    import warnings
    warnings.simplefilter('ignore')  # SyntaxWarnings of the repositories' code (ast.parse)
    os.environ.pop('REPO_TOOLS_FLOW', None)
    sys.path.insert(0, scripts)
    import locate  # noqa: F401
    _AG['locate'] = sys.modules['locate']
    _AG['kb'] = sys.modules['kb']
    for kv in overrides:  # --set NAME=VALUE: a locate.py constant, for quick ablations
        name, _, val = kv.partition('=')
        setattr(_AG['locate'], name, type(getattr(_AG['locate'], name))(val) if hasattr(_AG['locate'], name) else val)
    _AG['kbhash'] = hashlib.sha1(open(os.path.join(scripts, 'kb.py'), 'rb').read()).hexdigest()[:10]


def run_task(job):
    task, queries = job
    loc, kb = _AG['locate'], _AG['kb']
    t0 = time.time()
    try:
        ws = prepare(task)
    except Exception as e:  # noqa: BLE001
        return [{'task': task['instance_id'], 'source': task['source'], 'repo': task['repo'],
                 'error': f'prepare: {e!r}'[:300]}]
    gold = tb.gold_of(task['patch'], lambda p: tb.read_file(ws, p))
    prep = time.time() - t0
    os.environ['REPO_TOOLS_DIR'] = os.path.join(OUT, 'kb', _AG['kbhash'], task['instance_id'])
    os.environ['WORKSPACE'] = ws
    loc.ROOT = kb.ROOT = ws
    t1 = time.time()
    conn = kb.open(ws)
    kb_secs = time.time() - t1
    out = []
    for qname in queries:
        rq = None
        if qname == 'recorded':
            rq = _AG.setdefault('recorded', json.load(open(RECORDED)) if os.path.exists(RECORDED) else {}).get(
                task['instance_id'])
            if task['source'] != 'local':
                continue
        query = rq or tb.build_query(task['problem_statement'])
        rec = {'task': task['instance_id'], 'source': task['source'], 'repo': task['repo'], 'queries': qname,
               'query': query, 'gold': {k: gold[k] for k in ('files', 'functions')}}
        if not gold['spans']:
            rec['skip'] = 'no modified .py gold'
            out.append(rec)
            continue
        real_rank = loc.rank
        cache = {}

        def rank_once(*a, **k):
            if 'r' not in cache:
                cache['r'] = real_rank(*a, **k)
            return cache['r']

        loc.rank = rank_once
        buf = io.StringIO()
        t2 = time.time()
        try:
            with contextlib.redirect_stdout(buf):
                loc.locate(loc.K_DEFAULT, True, [query])
        except Exception as e:  # noqa: BLE001
            import traceback
            rec['error'] = traceback.format_exc()[-1500:]
        finally:
            loc.rank = real_rank
        rec['secs'] = round(time.time() - t2, 3)
        rec['kb_secs'], rec['prep_secs'] = round(kb_secs, 3), round(prep, 3)
        text = buf.getvalue()
        rec['out'] = text
        if 'r' in cache:
            ranked, best_fn, reasons, q, score = cache['r']
            gf = set(gold['files'])
            rec['full_rank'] = next((i + 1 for i, p in enumerate(ranked) if p in gf), None)
            rec['top'] = [[p, round(score[p], 3), loc.evidence(reasons[p])] for p in ranked[:10]]
            rec['gold_scores'] = {p: [round(score.get(p, 0), 3), (ranked.index(p) + 1) if p in ranked else None,
                                      loc.evidence(reasons.get(p, []))] for p in gold['files']}
            rec['named'] = {p: [f[1] for f in v] for p, v in q.get('named', {}).items()}
            rec['gold_fn_pos'] = {}
            for gp in gold['files']:
                quals = [f[1][1] for f in best_fn.get(gp, [])]
                rec['gold_fn_pos'][gp] = {
                    'top': [[round(x[0], 3), x[1][1]] for x in best_fn.get(gp, [])[:5]],
                    'gold': {gq: (quals.index(gq) + 1 if gq in quals else None)
                             for f, gq in gold['functions'] if f == gp}}
            feats = q.get('features') or {}
            rec['features'] = {p: {k: round(v, 4) for k, v in feats.get(p, {}).items()}
                               for p in list(ranked[:20]) + gold['files']}
        rec.update(metrics_of(text, gold, ws))
        out.append(rec)
    for key in [k for k in kb._OPEN if k[0] == os.path.abspath(ws)]:
        kb._OPEN.pop(key).close()
    return out


def metrics_of(text, gold, ws):
    """The toolbench's locate metrics (TaskRun.locate) of one output."""
    p = tb.parse_locate(text)
    gfiles = set(gold['files'])
    gfn = {tuple(x) for x in gold['functions']}
    gfn_short = {(f, q.split('.')[-1]) for f, q in gfn}
    rank = next((k + 1 for k, f in enumerate(p['files']) if f in gfiles), None)
    in_list = any((f, q) in gfn or (f, q.split('.')[-1]) in gfn_short for f, q, _, _ in p['functions'])
    best = p['best']
    best_gold = bool(best and ((best[0], best[1]) in gfn or (best[0], best[1].split('.')[-1]) in gfn_short))
    sh = tb.shown_lines(text, lambda rel: os.path.isfile(os.path.join(ws, rel)))
    shown = False
    for s in gold['spans']:
        lo, hi = tb.span_region(s, tb.SHOW_PAD)
        if any(lo <= x <= hi for x in sh.get(s['file'], ())):
            shown = True
    res = {'skill_name': 'repo-tools', 'file_path': 'locate.py', 'stdout': text, 'stderr': '', 'status': 'success'}
    return {'rank': rank, 'printed': len(p['files']), 'fn_in_list': in_list, 'has_fn': bool(gfn), 'best': best,
            'best_gold': best_gold, 'best_shown': best is not None, 'gold_shown': shown,
            'tokens': round(len(json.dumps(res)) / tb.CHARS_PER_TOKEN, 1), 'lines': len(text.splitlines())}


# ------------------------------------------------------------------ report


def pct(xs):
    xs = [bool(x) for x in xs]
    return 100.0 * sum(xs) / len(xs) if xs else float('nan')


def summarize(recs):
    ok = [r for r in recs if 'rank' in r]
    fn = [r for r in ok if r['has_fn']]
    shown = [r for r in fn if r['best_shown']]
    secs = sorted(r['secs'] for r in ok) or [0]
    fr = [r.get('full_rank') for r in ok]
    return collections.OrderedDict([
        ('n', len(ok)),
        ('@1', pct(r['rank'] == 1 for r in ok)),
        ('@3', pct(r['rank'] is not None and r['rank'] <= 3 for r in ok)),
        ('printed', pct(r['rank'] is not None for r in ok)),
        ('@10', pct(x is not None and x <= 10 for x in fr)),
        ('MRR', 100.0 * statistics.mean([1.0 / x if x else 0.0 for x in fr]) if fr else float('nan')),
        ('fn_list', pct(r['fn_in_list'] for r in fn)),
        ('best=gold', pct(r['best_gold'] for r in fn)),
        ('shown', pct(r['best_shown'] for r in ok)),
        ('prec', pct(r['best_gold'] for r in shown)),
        ('span', pct(r['gold_shown'] for r in ok)),
        ('tok', statistics.mean(r['tokens'] for r in ok) if ok else float('nan')),
        ('sec_med', statistics.median(secs)),
        ('sec_p95', secs[min(len(secs) - 1, int(0.95 * len(secs)))]),
    ])


def table(groups, title):
    lines = [title]
    keys = None
    for name, recs in groups:
        m = summarize(recs)
        if keys is None:
            keys = list(m)
            lines.append(f'{"":<34}' + ''.join(f'{k:>9}' for k in keys))
        lines.append(f'{name[:34]:<34}' + ''.join(f'{v:>9.1f}' if isinstance(v, float) else f'{v:>9}'
                                                  for v in m.values()))
    return '\n'.join(lines)


def report(recs, per_repo=True):
    out = []
    for key in sorted({(r['source'], r['queries']) for r in recs}):
        rs = [r for r in recs if (r['source'], r['queries']) == key]
        groups = [(f'{key[0]}/{key[1]} ALL', rs)]
        if per_repo:
            for repo in sorted({r['repo'] for r in rs}):
                groups.append(('  ' + repo, [r for r in rs if r['repo'] == repo]))
        out.append(table(groups, f'=== {key[0]} ({key[1]} queries)'))
        errs = [r for r in rs if r.get('error')]
        if errs:
            out.append(f'ERRORS: {len(errs)}: ' + '; '.join(f'{r["task"]}: {r["error"][-160:]!r}' for r in errs[:5]))
    return '\n\n'.join(out)


def load_dir(d):
    recs = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith('.jsonl'):
            recs += [json.loads(l) for l in open(os.path.join(d, fn)) if l.strip()]
    return recs


def compare(a, b):
    ra, rb = load_dir(a), load_dir(b)
    ia = {(r['task'], r['queries']): r for r in ra if 'rank' in r}
    ib = {(r['task'], r['queries']): r for r in rb if 'rank' in r}
    common = sorted(set(ia) & set(ib))
    out = []
    for key in sorted({(ia[k]['source'], k[1]) for k in common}):
        ks = [k for k in common if (ia[k]['source'], k[1]) == key]
        groups = []
        for repo in [None] + sorted({ia[k]['repo'] for k in ks}):
            kk = [k for k in ks if repo is None or ia[k]['repo'] == repo]
            ma, mb = summarize([ia[k] for k in kk]), summarize([ib[k] for k in kk])
            groups.append((repo or 'ALL', ma, mb))
        cols = ['@1', '@3', 'MRR', 'fn_list', 'best=gold', 'shown', 'prec', 'tok']
        out.append(f'=== {key[0]} ({key[1]}): {len(ks)} tasks, A={os.path.basename(a)} B={os.path.basename(b)} '
                   '(B, delta)')
        out.append(f'{"":<34}' + ''.join(f'{c:>15}' for c in cols))
        for name, ma, mb in groups:
            out.append(f'{name[:34]:<34}' + ''.join(f'{mb[c]:>7.1f} {mb[c] - ma[c]:>+6.1f}' for c in cols))
        won = [k[0] for k in ks if ib[k]['rank'] == 1 and ia[k]['rank'] != 1]
        lost = [k[0] for k in ks if ia[k]['rank'] == 1 and ib[k]['rank'] != 1]
        out.append(f'file@1 won {len(won)}: {" ".join(won[:30])}')
        out.append(f'file@1 lost {len(lost)}: {" ".join(lost[:30])}')
        out.append('')
    return '\n'.join(out)


# ------------------------------------------------------------------ main


def resolve_agent(spec):
    """(label, scripts dir): a directory (agent/ or a materialized one) or a git ref (materialized once)."""
    if os.path.isdir(spec):
        d = os.path.abspath(spec)
        label = os.path.basename(d.rstrip('/'))
    else:
        d = os.path.join(OUT, '_agents', spec)
        if not os.path.isdir(d):
            subprocess.run(['bash', os.path.join(ROOT, 'scripts/materialize.sh'), spec, d], cwd=ROOT, check=True,
                           capture_output=True)
        label = spec
    sc = os.path.join(d, 'skills', 'repo-tools', 'scripts')
    if not os.path.isdir(sc):
        sc = d  # a scripts dir itself
    return label, sc


def select(sources, only):
    tasks = []
    if 'local' in sources:
        tasks += tb.load_local(tb.task_ids(os.path.join(ROOT, 'experiments', 'eval113.txt')))
    if 'external' in sources:
        import toolbench_data
        tasks += [t for t in tb.load_external() if toolbench_data.ws_ready(t['instance_id'])]
    if only:
        tasks = [t for t in tasks if t['instance_id'] in only]
    return tasks


def main():
    if os.environ.get('PYTHONHASHSEED') != '0' or (os.path.exists(PY313) and sys.version_info[:2] != (3, 13)):
        exe = PY313 if os.path.exists(PY313) else sys.executable
        os.execve(exe, [exe] + sys.argv, dict(os.environ, PYTHONHASHSEED='0'))  # as the toolbench runs the scripts
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--agent', default=os.path.join(ROOT, 'agent'))
    ap.add_argument('--source', default='external,local')
    ap.add_argument('--queries', default='built,recorded')
    ap.add_argument('--tasks', default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--label', default=None)
    ap.add_argument('--compare', nargs=2, default=None)
    ap.add_argument('--set', action='append', default=[], help='NAME=VALUE: override a locate.py constant')
    ap.add_argument('--no-repo', action='store_true', help='no per-repo breakdown')
    a = ap.parse_args()
    if a.compare:
        print(compare(*a.compare))
        return
    label, scripts = resolve_agent(a.agent)
    tasks = select(a.source.split(','), set(a.tasks.split(',')) if a.tasks else None)
    queries = a.queries.split(',')
    out = os.path.join(OUT, a.label or label)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    recs = []
    # big repos first: the slowest tasks start early
    tasks.sort(key=lambda t: (t['source'] != 'external', t['repo'] not in ('huggingface/transformers', 'apache/airflow',
                                                                            'pandas-dev/pandas', 'ray-project/ray')))
    with concurrent.futures.ProcessPoolExecutor(a.workers, initializer=init_worker, initargs=(scripts, a.set)) as ex:
        futs = [ex.submit(run_task, (t, queries)) for t in tasks]
        for k, f in enumerate(concurrent.futures.as_completed(futs), 1):
            recs += f.result()
            if k % 50 == 0:
                print(f'[{k}/{len(tasks)}] {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
    by = collections.defaultdict(list)
    for r in recs:
        by[f'{r["source"]}_{r["queries"] if "queries" in r else "x"}'].append(r)
    for name, rs in by.items():
        with open(os.path.join(out, name + '.jsonl'), 'w') as f:
            for r in sorted(rs, key=lambda r: r['task']):
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    text = report([r for r in recs if 'queries' in r], per_repo=not a.no_repo)
    open(os.path.join(out, 'summary.txt'), 'w').write(text + '\n')
    print(text)
    print(f'{len(tasks)} tasks in {time.time() - t0:.0f}s -> {out} (agent {scripts})')


if __name__ == '__main__':
    main()
