"""Failed edit.py calls of real runs, as re-runnable cases (v20, research/notes/15-edit-failures.md).

collect: replays every recorded call of each run that has a failed edit.py call through the RECORDING's own agent
(results/<exp>/_agent) on the task snapshot (scripts/replay_tools.py's workspace setup, host python, run.py calls
skipped: they only feed flow's check counters), and at every failed edit.py call (NOT APPLIED / USAGE ERROR /
REFUSED / ADK rejection / timeout) saves a case: the call's args, the recorded output, the replayed output (sanity),
the working-tree diff before the call, the flow state dir, the edited file's content before the call, the next call
the model made, and the gold patch's hunks of that file.

run: re-runs each case's call through another agent's edit.py on the same workspace (reset, diff re-applied, flow
state restored and re-keyed) and records the output and the resulting diff of the target file.

    .venv/bin/python scripts/edit_failures.py collect --runs results/E057_v18d_kaggleslow,results/E058_v19_kaggleslow \\
        --out results/EF_v20
    .venv/bin/python scripts/edit_failures.py run --cases results/EF_v20 --agent WORKTREE --label v20
    .venv/bin/python scripts/edit_failures.py show --cases results/EF_v20 [--label v20] [--case N]
    .venv/bin/python scripts/edit_failures.py report --cases results/EF_v20,results/EF_v20b --label v20 [-v]
"""

import argparse
import concurrent.futures
import difflib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_tools as rt  # noqa: E402

ROOT = rt.ROOT
WORKROOT = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'edit_failures_ws')


def edit_failed(c):
    """Why a recorded edit.py call failed ('' when it applied)."""
    kind = rt.recorded_kind(c)
    if kind != 'ran':
        return kind
    status, _, _, _ = rt.edit_outcome(rt.recorded_stdout(c))
    return '' if status == 'applied' else status


def is_edit(c):
    a = c['args'] if isinstance(c['args'], dict) else {}
    return c['name'] == 'run_skill_script' and str(a.get('file_path', '')).strip().endswith('edit.py')


def gold_patches():
    out = {}
    for line in open(os.path.join(ROOT, 'data/competition/tasks.jsonl')):
        d = json.loads(line)
        out[d['instance_id']] = d['patch']
    return out


def file_gold(patch, rel):
    """The gold patch's part for REL ('' when it does not touch REL)."""
    parts = re.split(r'(?m)^(?=diff --git |--- a/)', patch)
    keep = [p for p in parts if re.search(r'(?m)^\+\+\+ b/' + re.escape(rel) + r'$', p)]
    return ''.join(keep)


def target_of(args):
    sa = args.get('args') if isinstance(args.get('args'), dict) else {}
    for k in ('target', 'file', 'path', 'file_path', 'filepath', 'filename'):
        v = sa.get(k)
        if v and not str(v).endswith('edit.py'):
            return str(v).strip().replace('/workspace/', '')
    return None


def call_brief(c):
    if c is None:
        return 'END'
    if c['name'] != 'run_skill_script':
        return c['name']
    a = c['args'] if isinstance(c['args'], dict) else {}
    sa = a.get('args')
    keys = ','.join(sa.keys()) if isinstance(sa, dict) else str(type(sa).__name__)
    ok = ''
    if is_edit(c):
        ok = ' ok' if not edit_failed(c) else ' FAILED'
    return f'{a.get("file_path")}({keys}){ok}'


def task_ws(task):
    return os.path.join(WORKROOT, task, 'ws')


def ensure_ws(task):
    """The task's workspace (extracted once, kept for the run phase), its pristine ignored paths."""
    ws = task_ws(task)
    meta = os.path.join(WORKROOT, task, 'pristine.json')
    if os.path.exists(meta) and os.path.isdir(os.path.join(ws, '.git')):
        return ws, set(json.load(open(meta)))
    shutil.rmtree(os.path.join(WORKROOT, task), ignore_errors=True)
    pristine = rt.prepare(os.path.join(ROOT, 'data/competition/snapshots', f'{task}.tgz'), ws)
    json.dump(sorted(pristine), open(meta, 'w'))
    return ws, pristine


def collect_task(job):
    task, runs, out_dir = job
    ws, pristine = ensure_ws(task)
    gold = gold_patches().get(task, '')
    cases = []
    for run in runs:
        calls = rt.recorded_calls(run['trace'])
        if not any(is_edit(c) and edit_failed(c) for c in calls):
            continue
        agent = rt.agent_record('rec', os.path.join(os.path.dirname(run['dir']).rsplit('/r', 1)[0], '_agent'))
        rt.reset(ws, pristine)
        state = tempfile.mkdtemp(prefix='state_', dir=os.path.join(WORKROOT, task))
        last_ts = None
        for i, c in enumerate(calls):
            ts = c['ts'] or ((last_ts + 1) if last_ts else None)
            last_ts = ts
            args = c['args'] if isinstance(c['args'], dict) else {}
            fp = str(args.get('file_path') or '')
            script = fp[len('scripts/'):] if fp.startswith('scripts/') else fp
            why = edit_failed(c) if is_edit(c) else ''
            if why:
                cid = f'{run["exp"]}__{run["id"].replace("/", "_")}__{i + 1}'
                cdir = os.path.join(out_dir, 'cases', cid)
                os.makedirs(cdir, exist_ok=True)
                rel = target_of(args)
                before = None
                if rel and os.path.isfile(os.path.join(ws, rel)):
                    before = open(os.path.join(ws, rel), encoding='utf-8', errors='replace').read()
                shutil.copytree(state, os.path.join(cdir, 'state'), dirs_exist_ok=True)
                open(os.path.join(cdir, 'pre.diff'), 'w').write(rt.patch_diff(ws))
                rt.git(ws, 'reset', '-q', check=False)  # (patch_diff's add -N)
                nxt = calls[i + 1] if i + 1 < len(calls) else None
                rest = [call_brief(x) for x in calls[i + 1:i + 4]]
                case = {'id': cid, 'exp': run['exp'], 'run': run['id'], 'task': task, 'resolved': run['resolved'],
                        'call': i + 1, 'ncalls': len(calls), 'why': why, 'args': args, 'ts': ts, 'end_ts': c.get('end_ts'),
                        'recorded': rt.recorded_stdout(c) if rt.recorded_kind(c) != 'noresult' else None,
                        'recorded_raw': c['content'], 'target': rel, 'before': before,
                        'gold': file_gold(gold, rel) if rel else '', 'gold_files': re.findall(r'(?m)^\+\+\+ b/(\S+)', gold),
                        'next': call_brief(nxt), 'next3': rest, 'ended': nxt is None,
                        'next_args': (nxt['args'] if nxt else None)}
                if rt.recorded_kind(c) != 'rejected' and script:
                    o, e, code, _ = rt.run_script(agent['scripts'], script, args, ws, state, ts, 60, None, c.get('end_ts'))
                    case['replayed'] = o
                    case['replayed_err'] = e[-2000:]
                    case['after_ref'] = open(os.path.join(ws, rel), encoding='utf-8', errors='replace').read() \
                        if rel and os.path.isfile(os.path.join(ws, rel)) else None
                json.dump(case, open(os.path.join(cdir, 'case.json'), 'w'), indent=1, ensure_ascii=False)
                cases.append(cid)
                continue
            if c['name'] != 'run_skill_script' or rt.recorded_kind(c) == 'rejected' or not script:
                continue
            if rt.adk_check(args, agent['skill'], agent['scripts']):
                continue
            if script == 'run.py':
                continue
            rt.run_script(agent['scripts'], script, args, ws, state, ts, 60, None, c.get('end_ts'))
        shutil.rmtree(state, ignore_errors=True)
    rt.reset(ws, pristine)
    return task, cases


def collect(a):
    runs = rt.load_runs(a.runs.split(','), tasks=set(a.tasks.split(',')) if a.tasks else None)
    by_task = {}
    for r in runs:
        calls = rt.recorded_calls(r['trace'])
        if any(is_edit(c) and edit_failed(c) for c in calls):
            by_task.setdefault(r['task'], []).append(r)
    os.makedirs(os.path.join(a.out, 'cases'), exist_ok=True)
    print(f'{sum(map(len, by_task.values()))} runs with a failed edit, {len(by_task)} tasks', flush=True)
    with concurrent.futures.ProcessPoolExecutor(a.workers) as ex:
        for task, cases in ex.map(collect_task, [(t, rs, a.out) for t, rs in by_task.items()]):
            print(f'{task}: {len(cases)} cases', flush=True)


def load_cases(d):
    out = []
    for cid in sorted(os.listdir(os.path.join(d, 'cases'))):
        p = os.path.join(d, 'cases', cid, 'case.json')
        if os.path.exists(p):
            out.append(json.load(open(p)))
    return out


def rekey(state_dir, ws):
    """Point the saved flow state at the workspace as it is now (flow resets state on a path/HEAD change)."""
    db = os.path.join(state_dir, 'state.sqlite')
    if not os.path.exists(db):
        return
    sys.path.insert(0, os.path.join(ROOT, 'agent/skills/repo-tools/scripts'))
    real = os.path.realpath(ws)
    head = rt.git(ws, 'rev-parse', 'HEAD', check=False).strip()
    conn = sqlite3.connect(db)
    conn.execute("INSERT OR REPLACE INTO meta VALUES('workspace', ?)", (real + '@' + head,))
    conn.commit()
    conn.close()


def run_case(job):
    case, cdir, scripts, label = job
    task = case['task']
    ws, pristine = ensure_ws(task)
    rt.reset(ws, pristine)
    pre = open(os.path.join(cdir, 'pre.diff')).read()
    if pre.strip():
        r = subprocess.run(['git', 'apply', '--whitespace=nowarn', '-'], cwd=ws, input=pre.encode(), capture_output=True)
        if r.returncode:
            return case['id'], {'error': 'pre.diff did not apply: ' + r.stderr.decode()[:300]}
    state = tempfile.mkdtemp(prefix='state_', dir=os.path.join(WORKROOT, task))
    shutil.copytree(os.path.join(cdir, 'state'), state, dirs_exist_ok=True)
    rekey(state, ws)
    args = case['args']
    res = {}
    if rt.adk_check(args, 'repo-tools', scripts):
        res = {'out': 'ADK REJECTED', 'status': 'rejected'}
    else:
        fp = str(args.get('file_path'))
        script = fp[len('scripts/'):] if fp.startswith('scripts/') else fp
        o, e, code, secs = rt.run_script(scripts, script, args, ws, state, case['ts'], 60, None, case.get('end_ts'))
        status, _, _, first = rt.edit_outcome(o)
        rel = case['target']
        after = open(os.path.join(ws, rel), encoding='utf-8', errors='replace').read() \
            if rel and os.path.isfile(os.path.join(ws, rel)) else None
        diff = ''
        if case['before'] is not None and after is not None and after != case['before']:
            diff = '\n'.join(difflib.unified_diff(case['before'].split('\n'), after.split('\n'), 'a/' + rel, 'b/' + rel,
                                                  n=1, lineterm=''))
        res = {'out': o, 'err': e[-1500:], 'code': code, 'secs': round(secs, 2), 'status': status, 'first': first,
               'diff': diff}
    shutil.rmtree(state, ignore_errors=True)
    rt.reset(ws, pristine)
    return case['id'], res


def run(a):
    if a.agent == 'WORKTREE':
        scripts = os.path.join(ROOT, 'agent/skills/repo-tools/scripts')
    elif os.path.isdir(a.agent):
        scripts = rt.skill_info(a.agent)[1]
    else:
        dest = os.path.join(a.cases, '_agents', a.label)
        subprocess.run(['bash', os.path.join(ROOT, 'scripts/materialize.sh'), a.agent, dest], cwd=ROOT, check=True,
                       capture_output=True)
        scripts = rt.skill_info(dest)[1]
    cases = load_cases(a.cases)
    if a.case:
        cases = [c for c in cases if a.case in c['id']]
    by_task = {}
    for c in cases:
        by_task.setdefault(c['task'], []).append(c)
    results = {}

    def task_job(cs):
        return [run_case((c, os.path.join(a.cases, 'cases', c['id']), scripts, a.label)) for c in cs]

    with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:  # one task's cases in order (shared workspace)
        for lst in ex.map(task_job, by_task.values()):
            for cid, res in lst:
                results[cid] = res
    out = os.path.join(a.cases, f'run_{a.label}.json')
    json.dump(results, open(out, 'w'), indent=1, ensure_ascii=False)
    n = sum(1 for r in results.values() if r.get('status') == 'applied')
    print(f'{len(results)} cases, {n} applied -> {out}')


def show(a):
    cases = load_cases(a.cases)
    res = json.load(open(os.path.join(a.cases, f'run_{a.label}.json'))) if a.label else {}
    for c in cases:
        if a.case and a.case not in c['id']:
            continue
        print('=' * 100)
        print(f"{c['id']}  why={c['why']}  resolved={c['resolved']}  next={c['next']}  ended={c['ended']}")
        sa = c['args'].get('args')
        print('ARGS:', json.dumps(sa, ensure_ascii=False)[:3000])
        print('RECORDED:', (c['recorded'] or '')[:2500])
        if a.verbose:
            print('REPLAYED:', (c.get('replayed') or '')[:1500])
            print('GOLD:', c['gold'][:2500])
        if c['id'] in res:
            r = res[c['id']]
            print(f"--- {a.label}: {r.get('status')}  {r.get('secs')}s")
            print(r.get('out', '')[:2500])


CATEGORIES = [  # (key, group, description), report order
    ('adk_rejected', 'format', 'ADK rejected the call (skill_name/file_path inside args, quoted skill name)'),
    ('args_mangled', 'format', 'args mangled by the tool-call parser (keys split at `:`/`,`, `key=<|"|>value` keys)'),
    ('new_outside_args', 'format', 'NEW (or start) given next to args instead of inside it: the script never saw it'),
    ('time', 'time', 'the task\'s time ran out during the call (no result / killed): the edit itself applies'),
    ('start_only_fixed', 'tool', 'start without end, the range was clear (NEW indented off, mixed \\n escapes, a '
                                 'decorator above start, @property in the duplicate check): v20 infers it'),
    ('old_drift_fixed', 'tool', 'OLD drifted (blank lines miscounted, docstring summarised, `{` for `{{`): v20 '
                                'blank-insensitive / anchored match'),
    ('range_fixed', 'tool', 'start/end off: NEW\'s own first/last lines name the range (the given one does not parse): '
                            'v20 range repair'),
    ('location_inferred', 'tool', 'no old/start (NEW starts with a unique line or redefines a unique def), or end = '
                                  'start - 1: v20 locates it'),
    ('stale_fixed', 'tool', 'stale line numbers: NEW now tells which numbering it meant (decoded NEW): shifted'),
    ('new_already_there', 'tool', 'OLD gone, NEW already in the file (the model resent an applied edit): v20 says so'),
    ('compile_error', 'tool', 'parses but does not compile (repeated keyword): v20 refuses before writing'),
    ('ambiguous', 'model', 'OLD occurs in several places (v20: a ready multi-edit call when one-line)'),
    ('start_only_unclear', 'model', 'start without end and the range is genuinely unclear (refused with ready calls)'),
    ('old_invented', 'model', 'OLD not in the file (invented / wrong lines, anchors do not match)'),
    ('new_breaks_syntax', 'model', 'NEW does not parse / unrepairable indentation / dead code'),
    ('guard', 'model', 'a guard refused it: import breaks, read-only property, deletes a used definition'),
    ('stale_numbers', 'model', 'line numbers from before an edit that changed those lines'),
    ('test_file', 'model', 'edit of a test file'),
    ('noop_or_repeat', 'model', 'changes nothing / identical repeat (ALREADY DONE)'),
    ('missing_location', 'model', 'no old/start (or a bad range): nowhere to put NEW'),
    ('other', 'model', 'other'),
]


def categorize(c, r):
    """The category key of a failed call (case C) given the v20 result R of the same call."""
    rec = c['recorded'] or ''
    a = c['args']
    sa = a.get('args') if isinstance(a.get('args'), dict) else {}
    status = (r or {}).get('status')
    out = (r or {}).get('out') or ''
    if c['why'] == 'rejected':
        return 'adk_rejected'
    if c['why'] in ('noresult', 'harness_timeout'):
        return 'time'
    if c['why'] == 'already_done':
        return 'noop_or_repeat'
    if 'takes exactly FILE OLD NEW' in rec or any('<|' in k or '\n' in k or '"' in k for k in sa):
        return 'args_mangled'
    if ('needs the key' in rec or 'needs the key' in out) and any(k in a for k in ('new', 'old', 'start', 'end')):
        return 'new_outside_args'
    if status == 'applied':
        if 'numbered as read at call' in out:
            return 'stale_fixed'
        if 'blank lines ignored' in out or 'anchored on OLD' in out:
            return 'old_drift_fixed'
        if 'as given do not parse with NEW' in out:
            return 'range_fixed'
        if 'no old or start given' in out or 'end = start - 1' in out:
            return 'location_inferred'
        if 'no end given' in out or 'NEW starts with line' in out:
            return 'start_only_fixed'
        return 'start_only_fixed' if 'start' in sa and 'end' not in sa else 'old_drift_fixed'
    if out.startswith('NOT CHANGED') or 'NOT CHANGED: OLD is not in' in out:
        return 'new_already_there'
    if 'would not compile' in out:
        return 'compile_error'
    if 'test or test-config' in out:
        return 'test_file'
    if 'changes nothing' in out or 'identical' in out[:200]:
        return 'noop_or_repeat'
    if 'occurs' in out[:120] or 'matches 2 places' in out[:200] or re.search(r'matches \d+ places|enclose \d+ places',
                                                                             out[:200]):
        return 'ambiguous'
    if '"start" without "end"' in out or 'without end' in out:
        return 'start_only_unclear'
    if 'OLD not found' in out:
        return 'old_invented'
    if 'would not parse' in out or 'not indented like' in out or 'do not' in out[:120]:
        return 'new_breaks_syntax'
    if '`import ' in out or 'read-only @property' in out or 'deletes `' in out or 'deletes ' in out[:200]:
        return 'guard'
    if 'numbered as read' in out:
        return 'stale_numbers'
    if 'needs the key' in out or 'bad range' in out or 'is before' in out:
        return 'missing_location'
    return 'other'


def report(a):
    import collections
    rows = []
    for d in a.cases.split(','):
        res = json.load(open(os.path.join(d, f'run_{a.label}.json')))
        for c in load_cases(d):
            rows.append((c, res.get(c['id']), categorize(c, res.get(c['id']))))
    cnt = collections.Counter(k for _, _, k in rows)
    ended = collections.Counter(k for c, _, k in rows if c['ended'])
    again = collections.Counter(k for c, _, k in rows if c['next'].startswith('edit.py') or 'edit.py' in c['next'][:12])
    print(f'{len(rows)} failed edit calls; v20 ({a.label}): '
          f'{sum(1 for _, r, _ in rows if (r or {}).get("status") == "applied")} applied')
    print(f'{"category":20s} {"n":>4s} {"run ended":>9s} {"edit next":>9s}  group  description')
    for key, group, desc in CATEGORIES:
        if cnt[key]:
            print(f'{key:20s} {cnt[key]:4d} {ended[key]:9d} {again[key]:9d}  {group:6s} {desc}')
    if a.verbose:
        for c, r, k in sorted(rows, key=lambda x: x[2]):
            print(k, c['id'], ((r or {}).get('first') or '')[:90])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--runs', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--tasks', default=None)
    c.add_argument('--workers', type=int, default=6)
    r = sub.add_parser('run')
    r.add_argument('--cases', required=True)
    r.add_argument('--agent', required=True)
    r.add_argument('--label', required=True)
    r.add_argument('--case', default=None)
    r.add_argument('--workers', type=int, default=6)
    s = sub.add_parser('show')
    s.add_argument('--cases', required=True)
    s.add_argument('--label', default=None)
    s.add_argument('--case', default=None)
    s.add_argument('-v', '--verbose', action='store_true')
    rp = sub.add_parser('report')
    rp.add_argument('--cases', required=True, help='case dirs, comma-separated')
    rp.add_argument('--label', required=True)
    rp.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()
    {'collect': collect, 'run': run, 'show': show, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()
