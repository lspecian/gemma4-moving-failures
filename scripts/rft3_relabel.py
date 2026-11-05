"""rft3 part 2: relabelled decision states from our own runs, in a frozen agent format (agent-v19, agent-v20; CPU only,
no model).

    .venv/bin/python scripts/rft3_relabel.py jobs        # raw captures + traces -> data/rft3/relabel_jobs.jsonl
    .venv/bin/python scripts/rft3_relabel.py [--version v20] replay [--workers 6]   # docker (sandbox image, task env)
    .venv/bin/python scripts/rft3_relabel.py [--version v20] collect     # -> data/rft3/states[_v20]/relabel.jsonl

--version: the agent whose scripts replay the prefix and the continuation (v19 default: relabel_raw.jsonl,
relabel_report.json, states/; v20: relabel_raw_v20.jsonl, relabel_report_v20.json, states_v20/). The jobs are shared.
v20 continuation: the gold spans of one file (all printed at the state, by definition) are ONE multi-edit call
(rft3_oracle.Builder.hunk_groups / multi_args), kept only when it gives exactly the gold (else the file and the flow
state are restored and single edits follow, as v19).

DAgger-style relabelling (research/notes/14-oracle-trajectories.md, recipe D): in our own Kaggle-like runs, find the
first state before the first edit where every gold span had ALREADY been printed as numbered lines in an earlier tool
output, but the model made another read/search; label the oracle continuation there.

jobs: runs with raw request/response captures (data/raw/<run>, scripts/slow_proxy.py --save) and traces
(results/<run>): RUNS below (E048 v17b, E049/E053 v18c, E054 v18c+b256, E057 v18d, E058 v19; local eval113 tasks).
The 37 held-out tasks of experiments/lora2_heldout.txt (our test set; all of E055/E056) never contribute. Episodes
(scripts/rft2_build.py grouping: run x task, a restarted task keeps its last attempt) are cut at the first compaction
(prefix break between consecutive requests): later states are skipped and counted. Per call: the model's own call
(name + arguments as sent), id, thinking (`reasoning`), visible text (`content`), the recorded result, and the
recorded clock from the trace (results/<run>/r1/c*/traces: call start and end timestamps, aligned by order and
arguments). Gold = data/competition/tasks.jsonl patch (toolbench.gold_of). Episodes calling tools other than
run_skill_script before the state (load_skill, get_status, ...) are skipped (counted; almost none).

replay (inside the sandbox image like scripts/replay_tools.py: stock site-packages + the task's per-task environment
on PYTHONPATH, i.e. TASK_ENV=1 runs): per task the snapshot is extracted and set up as the harness does (git exclude,
pytest.ini / conftest.py, baseline commit); per episode the workspace is reset and the recorded calls are re-executed
IN ORDER with agent-v19's scripts on a fresh flow state and the recorded clock (time.time = the recorded call start,
then the recorded end for flow.after, as replay_tools), so every earlier output is v19 format; ADK's checks emulated
(rejected calls get v19's ADK error object). After each call: which gold spans' edit ranges (toolbench.anchored) are
printed (toolbench.shown_lines on the v19 outputs). The state is the first k >= 1 where all are printed, no edit
attempt happened before, and the model's recorded call k is a read/search (show/outline/search/callers/tests_for/
history/locate/status); run.py calls are passed over (a repro before editing is what the prompt asks). An edit attempt
first -> no state ("edited"); end of the pre-compaction prefix first -> none.
At the state: turns[:k] = the model's own calls, thinking and visible text (label "context") with the v19 outputs;
then the oracle continuation (label "train"): edit.py line-range per gold span, bottom-up on the base numbering (no
edit happened yet; checked: the gold files are unchanged), check_patch.py, submit_patch, on the same workspace and
flow state, clock = the recorded time of call k + 20 s per call. Final files == gold is checked.
Stats: runs/tasks qualifying, the call index of the state, the turn that first printed the gold, the reads wasted
(recorded calls from the state to the model's first edit attempt, or to the end of the episode).

collect: request = rft3_states.build_request (agent-v19 system prompt, the captured user message's tree); est. prompt
tokens (June template + 110/turn for null thinking): >= 14,336 is dropped (counted).
"""

import argparse
import collections
import concurrent.futures
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rft3_states as S  # noqa: E402
import toolbench as tb  # noqa: E402

OUT = os.path.join(ROOT, 'data', 'rft3')
JOBS = os.path.join(OUT, 'relabel_jobs.jsonl')


def configure(version=None):
    """Paths of agent VERSION (rft3_states.VERSIONS): v19 data/rft3/relabel_raw.jsonl, relabel_report.json,
    states/relabel.jsonl (the step A build); v20 relabel_raw_v20.jsonl, relabel_report_v20.json,
    states_v20/relabel.jsonl. The jobs (captures + traces) do not depend on the version."""
    global RAW, REPORT
    if version:
        S.use_version(version)
    RAW = S.suffixed(os.path.join(OUT, 'relabel_raw.jsonl'))
    REPORT = S.suffixed(os.path.join(OUT, 'relabel_report.json'))


configure()
RUNS = ['E058_v19_kaggleslow', 'E057_v18d_kaggleslow', 'E054_v18c_b256_kaggleslow', 'E053_v18c_kaggleslow',
        'E049_v18c_kaggleslow', 'E048_v17b_kaggleslow']
HELDOUT = os.path.join(ROOT, 'experiments', 'lora2_heldout.txt')
IMAGE = 'swebench-sandbox:latest'
SP_DIR = '/usr/local/lib/python3.13/site-packages'
SP_CACHE = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'swegemma_sp_cache_v8')
CALL_SECONDS = 20.0
CALL_TIMEOUT = 180
READS = {'show.py', 'outline.py', 'search.py', 'callers.py', 'tests_for.py', 'history.py', 'locate.py', 'status.py',
         'hypothesis.py'}
EDIT_SCRIPTS = {'edit.py', 'create.py'}


# ------------------------------------------------------------------ jobs (host)

def heldout():
    return {l.strip() for l in open(HELDOUT) if l.strip() and not l.startswith('#')}


def trace_calls(run, task):
    import replay_tools
    fs = glob.glob(os.path.join(ROOT, 'results', run, 'r*', 'c*', 'traces', f'trace_{task}.json')) + \
        glob.glob(os.path.join(ROOT, 'results', run, 'r*', 'traces', f'trace_{task}.json'))
    return replay_tools.recorded_calls(sorted(fs)[-1]) if fs else None


def results_of(run):
    res = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'results', run, 'r*', 'c*', 'task_results.jsonl')) +
                    glob.glob(os.path.join(ROOT, 'results', run, 'r*', 'task_results.jsonl'))):
        for line in open(f):
            if line.strip():
                x = json.loads(line)
                res[x['instance_id']] = x
    return res


def make_jobs():
    held = heldout()
    stats = collections.Counter()
    n = 0
    with open(JOBS, 'w') as out:
        for run in RUNS:
            groups, tasks = S.captured_episodes(run)
            results = results_of(run)
            for tid, reqs in sorted(groups.items()):
                stats['episodes'] += 1
                if tid in held:
                    stats['skipped: held-out task'] += 1
                    continue
                pre = [reqs[0]]
                for x, y in zip(reqs, reqs[1:]):
                    if y['request']['messages'][:len(x['request']['messages'])] != x['request']['messages']:
                        break
                    pre.append(y)
                results_by_id = {}
                for d in reqs:
                    for m in d['request']['messages']:
                        if m.get('role') == 'tool':
                            results_by_id.setdefault(m['tool_call_id'], m['content'])
                turns = []
                for d in reqs:  # every call of the episode (the prefix limit is stored separately)
                    msg = d['response']['choices'][0]['message']
                    calls = msg.get('tool_calls') or []
                    if len(calls) != 1:
                        break
                    c = calls[0]
                    try:
                        args = json.loads(c['function']['arguments'])
                    except ValueError:
                        args = None
                    turns.append({'id': c['id'], 'name': c['function']['name'], 'args': args,
                                  'thinking': msg.get('reasoning') or msg.get('reasoning_content') or None,
                                  'visible': msg.get('content') or None, 'result': results_by_id.get(c['id'])})
                tc = trace_calls(run, tid)
                aligned = 0
                if tc:
                    for t, x in zip(turns, tc):
                        if x['name'] != t['name'] or (x['args'] or {}) != (t['args'] or {}):
                            break
                        t['ts'], t['end_ts'] = x['ts'], x['end_ts']
                        aligned += 1
                stats['turns'] += len(turns)
                stats['turns_with_clock'] += aligned
                task = tasks[tid]
                user = pre[0]['request']['messages'][1]['content']
                job = {'run': run, 'task': tid, 'repo': task['repo'], 'problem_statement': task['problem_statement'],
                       'patch': task['patch'], 'tree': S.tree_of_user(user), 'turns': turns,
                       'precompaction_calls': min(len(pre), len(turns)), 'compacted': len(pre) < len(reqs),
                       'resolved': bool(results.get(tid, {}).get('resolved')), 'aligned': aligned}
                out.write(json.dumps(job, ensure_ascii=False) + '\n')
                n += 1
    print(f'{n} jobs -> {JOBS}; ' + ', '.join(f'{k} {v}' for k, v in stats.items()))


# ------------------------------------------------------------------ replay (inside the sandbox image)

def prepare_ws(tgz, ws):
    """Snapshot -> workspace as the harness sets it up (replay_tools.prepare + setup_workspace_test_config)."""
    import replay_tools
    import rft3_oracle
    os.makedirs(ws, exist_ok=True)
    subprocess.run(['tar', '--no-same-owner', '-xzf', tgz, '-C', ws], check=True, capture_output=True)
    real = os.path.realpath(ws)
    for dp, dns, fns in os.walk(ws):
        for nm in dns + fns:
            p = os.path.join(dp, nm)
            if os.path.islink(p):
                t = os.path.realpath(p)
                if t == real or real.startswith(t.rstrip('/') + '/'):
                    os.unlink(p)
    if not os.path.isdir(os.path.join(ws, '.git')):
        replay_tools.git(ws, 'init', '-q')
    os.makedirs(os.path.join(ws, '.git', 'info'), exist_ok=True)
    with open(os.path.join(ws, '.git', 'info', 'exclude'), 'a') as f:
        f.write(replay_tools.GIT_EXCLUDE)
    rft3_oracle.harness_test_config(ws)
    replay_tools.git(ws, 'add', '-A')
    replay_tools.git(ws, 'commit', '-m', 'baseline', '--allow-empty', '-q', '--no-verify')
    replay_tools.git(ws, 'tag', '-f', replay_tools.BASELINE)
    return set(replay_tools.ignored(ws))


def call_kind(name, args, scripts):
    """edit | read | run | check | submit | rejected | other (as ADK would handle the call with v19's scripts)."""
    import next_step_probe as nsp
    if name == 'submit_patch':
        return 'submit'
    label, intent = nsp.classify_call(name, args if isinstance(args, dict) else {}, scripts)
    if label in ('edit', 'edit_badshape') or intent in EDIT_SCRIPTS:
        return 'edit'
    if label == 'malformed':
        # a garbled call that is not an edit (e.g. file_path `` with target/symbol args): the model chose to read on
        inner = args.get('args') if isinstance(args, dict) and isinstance(args.get('args'), dict) else {}
        if intent in READS or (intent is None and not {'snippet', 'test', 'new', 'old', 'content'} & set(inner)):
            return 'read'
        return 'rejected'
    if label in ('read', 'search', 'locate'):
        return 'read'
    if label in ('snippet', 'test'):
        return 'run'
    if label == 'check':
        return 'check'
    if intent in ('status.py', 'hypothesis.py') or (isinstance(args, dict) and str(args.get('file_path', '')).endswith(
            ('status.py', 'hypothesis.py'))):
        return 'read'
    return 'other'


class Replayer:
    def __init__(self, job, agent, ws, state, env_path):
        self.job, self.agent, self.ws, self.state, self.env_path = job, agent, ws, state, env_path
        self.outs, self.results = [], []
        self.is_file = lambda rel: os.path.isfile(os.path.join(ws, rel))  # noqa: E731

    def script(self, file_path, args, now, end):
        import replay_tools
        name = file_path[len('scripts/'):] if file_path.startswith('scripts/') else file_path
        out, err, code, _ = replay_tools.run_script(self.agent['scripts'], name, {'args': args} if args else {},
                                                    self.ws, self.state, now, CALL_TIMEOUT, self.env_path, end)
        return out, S.adk_result(file_path, out, err, code, timeout=CALL_TIMEOUT), code

    def recorded(self, t, k):
        """Re-execute recorded call k with v19 (ADK checks emulated): (stdout, result)."""
        import replay_tools
        if t['name'] != 'run_skill_script':
            return None, None
        a = t['args'] if isinstance(t['args'], dict) else None
        err = replay_tools.adk_check(a, self.agent['skill'], self.agent['scripts']) if a is not None else \
            {'error': "Argument 'skill_name' is required.\nArgument 'file_path' is required.",
             'error_code': 'INVALID_ARGUMENTS'}
        if err:
            return '', json.dumps(err, ensure_ascii=False)
        sa = a.get('args')
        if isinstance(sa, list) or a.get('short_options') or a.get('positional_args'):
            name = str(a['file_path'])
            name = name[len('scripts/'):] if name.startswith('scripts/') else name
            out, er, code, _ = replay_tools.run_script(self.agent['scripts'], name, a, self.ws, self.state,
                                                       t.get('ts'), CALL_TIMEOUT, self.env_path, t.get('end_ts'))
            return out, S.adk_result(a['file_path'], out, er, code, timeout=CALL_TIMEOUT)
        out, res, _ = self.script(str(a['file_path']), sa or {}, t.get('ts'), t.get('end_ts'))
        return out, res


def relabel_one(job, agent, ws, pristine, workdir, env_path):
    """(record or None, why, info) for one episode."""
    import replay_tools
    import rft3_oracle
    replay_tools.reset(ws, pristine)
    state = tempfile.mkdtemp(prefix='state_', dir=workdir)
    read = lambda rel: tb.read_file(ws, rel)  # noqa: E731
    gold = tb.gold_of(job['patch'], read)
    info = {'run': job['run'], 'task': job['task'], 'resolved': job['resolved']}
    if not gold['spans']:
        return None, 'no modified .py gold span', info
    base = {f: read(f) for f in gold['files']}
    spans = [(s, *tb.anchored(s, base[s['file']].split('\n'))[:2]) for s in gold['spans']]
    R = Replayer(job, agent, ws, state, env_path)
    shown = collections.defaultdict(set)
    turns = job['turns']
    limit = job['precompaction_calls']
    gold_at, fn_at = None, None
    state_k = None
    last_ts = [None]
    names = {n for n in os.listdir(agent['scripts']) if n.endswith('.py')}

    def all_printed():
        return all(all(x in shown[s['file']] for x in range(a, b + 1)) for s, a, b in spans)

    for k, t in enumerate(turns):
        kind = call_kind(t['name'], t['args'], names)
        if k >= 1 and all_printed() and kind == 'read':
            if k >= limit:
                return None, 'state past compaction', dict(info, k=k)
            state_k = k
            break
        if kind == 'edit':
            return None, 'edited before a wasted read' if gold_at is not None else 'edit before the gold was printed', \
                dict(info, k=k, gold_at=gold_at)
        if kind == 'submit':
            return None, 'submitted without edit', dict(info, k=k)
        if t['name'] != 'run_skill_script':
            return None, f'other tool in the prefix: {t["name"]}', dict(info, k=k)
        if k >= limit:
            return None, 'no state before compaction' if gold_at is None else 'state past compaction', \
                dict(info, k=k, gold_at=gold_at)
        if not t.get('ts'):  # no aligned trace clock: 20 s after the previous call
            t = dict(t, ts=(last_ts[0] or 1_800_000_000.0) + CALL_SECONDS)
            t['end_ts'] = t['ts'] + 1.0
        last_ts[0] = float(t.get('end_ts') or t['ts'])
        out, res = R.recorded(t, k)
        R.outs.append(out or '')
        R.results.append(res)
        for f, ls in tb.shown_lines(out or '', R.is_file).items():
            shown[f] |= ls
        if gold_at is None and all_printed():
            gold_at = k
        if fn_at is None and any(f in (out or '') for f in gold['files']) and \
                any(q.split('.')[-1] in (out or '') for _, q in gold['functions']):
            fn_at = k
    if state_k is None:
        return None, 'gold never printed before the end of the episode' if gold_at is None else \
            'episode ended right after the gold was printed', dict(info, gold_at=gold_at)
    # wasted reads: recorded calls from the state to the first edit attempt (or the end of the episode)
    first_edit = next((j for j in range(state_k, len(turns)) if call_kind(turns[j]['name'], turns[j]['args'], names)
                       == 'edit'), None)
    info.update(k=state_k, gold_at=gold_at, fn_at=fn_at, first_edit=first_edit,
                wasted=(first_edit if first_edit is not None else len(turns)) - state_k,
                next_recorded=S.script_of({'arguments': turns[state_k]['args'] or {}}))
    for f in gold['files']:
        if read(f) != base[f]:
            return None, 'gold file changed before the state', info
    # continuation: oracle edits (bottom-up, base numbering), check_patch, submit
    t_last = last_ts[0] or 1_800_000_000.0
    clock = [float(t_last)]

    def call(script, args):
        clock[0] += CALL_SECONDS
        out, res, code = R.script(script, args, clock[0], clock[0] + 1.0)
        return out, res, code

    new = []
    printed_at = {}
    for s, a, b in spans:
        acc = set()
        for j, o in enumerate(R.outs):
            acc |= tb.shown_lines(o, R.is_file).get(s['file'], set())
            if all(x in acc for x in range(a, b + 1)):
                printed_at[(s['file'], a, b)] = j
                break
    statuses = []
    multi = 'start2' in open(os.path.join(agent['scripts'], 'edit.py')).read()  # v20 edit.py: multi-edit

    def where_printed(f, a, b):
        pk = printed_at.get((f, a, b))
        if pk is None:
            pk = next((q for q in range(len(R.outs)) if all(
                x in set().union(*[tb.shown_lines(o, R.is_file).get(f, set()) for o in R.outs[:q + 1]])
                for x in range(a, b + 1))), None)
        return pk

    def recorded_kind(pk):
        return S.script_of({'arguments': turns[pk]['args'] or {}}) or turns[pk]['name']

    for f in gold['files']:
        base_lines = base[f].split('\n')
        fs = sorted([s for s in gold['spans'] if s['file'] == f], key=lambda s: s['a'])
        groups = rft3_oracle.Builder.hunk_groups(fs, base_lines) if multi and len(fs) >= 2 else []
        if 2 <= len(groups) <= rft3_oracle.MAX_HUNKS:
            # v20: every span of the file is printed at the state: ONE multi-edit call (numbers as read); kept only
            # when applied with exactly the gold result, else the file and the flow state are restored (single edits)
            pks = [where_printed(f, a, b) for a, b, _, _ in groups]
            if None not in pks:
                before = read(f)
                saved = state + '.multi'
                shutil.rmtree(saved, ignore_errors=True)
                shutil.copytree(state, saved)
                args = rft3_oracle.Builder.multi_args(f, groups)
                out, res, _ = call('edit.py', args)
                want = tb.apply_spans(base[f], fs)
                got = read(f) or ''
                if tb.edit_status(out) == 'applied' and (got == want or [x.rstrip() for x in got.split('\n')] ==
                                                          [x.rstrip() for x in want.split('\n')]):
                    shutil.rmtree(saved, ignore_errors=True)
                    statuses.append('applied')
                    new.append({'call': S.script_call('edit.py', args), 'result': res, 'kind': 'edit',
                                'meta': {'hunks': len(groups)},
                                'grounding': f'{f}: ' + '; '.join(
                                    f'lines {a}-{b} printed numbered at turn {pk} ({recorded_kind(pk)})'
                                    for (a, b, _, _), pk in zip(groups, pks))})
                    continue
                with open(os.path.join(ws, f), 'w', newline='') as fh:
                    fh.write(before)
                shutil.rmtree(state, ignore_errors=True)
                os.rename(saved, state)
                clock[0] -= CALL_SECONDS
                info.setdefault('multi_refused', []).append(f)
        i = len(fs) - 1
        while i >= 0:
            j = i
            while True:
                span = fs[i] if j == i else tb.merge_spans(fs[j:i + 1], base_lines)
                a, b, nw = tb.anchored(span, base_lines)
                pk = where_printed(f, a, b)
                if pk is None:
                    return None, 'merged edit range not printed', info
                args = {'target': f, 'start': str(a), 'end': str(b), 'new': '\n'.join(nw)}
                out, res, _ = call('edit.py', args)
                stt = tb.edit_status(out)
                statuses.append(stt)
                new.append({'call': S.script_call('edit.py', args), 'result': res, 'kind': 'edit',
                            'grounding': f'lines {a}-{b} of {f} printed numbered at turn {pk} ({recorded_kind(pk)})'})
                if stt == 'applied' or j == 0:
                    break
                j -= 1
            i = j - 1
    exact = True
    for f in gold['files']:
        want = tb.apply_spans(base[f], [s for s in gold['spans'] if s['file'] == f])
        got = read(f) or ''
        exact &= got == want or [l.rstrip() for l in got.split('\n')] == [l.rstrip() for l in want.split('\n')]
    if not exact:
        return None, 'final != gold', dict(info, statuses=statuses)
    out, res, _ = call('check_patch.py', {})
    ready = bool(re.search(r'^READY\b', out, re.M))
    problems = [l.strip()[2:] for l in out.splitlines() if l.startswith('  - ')]
    new.append({'call': S.script_call('check_patch.py', {}), 'result': res, 'kind': 'check',
                'grounding': 'after the edits (system prompt: check_patch.py, then submit_patch)'})
    replay_tools.git(ws, 'add', '-N', '.', check=False)
    diff = replay_tools.git(ws, 'diff', '--binary', replay_tools.BASELINE, check=False)
    new.append({'call': S.submit_call(), 'result': S.submit_result(diff), 'kind': 'submit',
                'grounding': f'check_patch at turn {state_k + len(new) - 1}: ' + ('READY' if ready else
                                                                                  'printed problems')})
    info.update(check_ready=ready, check_problems=problems[:2], statuses=statuses)
    if not ready and problems:
        return None, 'check_patch problems', info
    rec_turns = []
    for k in range(state_k):
        t = turns[k]
        rec_turns.append({'id': t['id'], 'call': {'name': t['name'], 'arguments': t['args']},
                          'result': R.results[k], 'label': 'context', 'thinking': t['thinking'],
                          'visible': t['visible'], 'grounding': None,
                          'kind': call_kind(t['name'], t['args'], names)})
    for m, t in enumerate(new):
        rec_turns.append({'id': S.call_id('rft3relabel', job['run'], job['task'], m), 'call': t['call'],
                          'result': t['result'], 'label': 'train', 'thinking': None, 'visible': None,
                          'grounding': t['grounding'], 'kind': t['kind'], **({'meta': t['meta']} if t.get('meta')
                                                                             else {})})
    rec = {'id': f'relabel_{job["run"]}_{job["task"]}', 'source': 'relabel', 'task': job['task'], 'repo': job['repo'],
           'turns': rec_turns,
           'gold': {'files': gold['files'], 'functions': gold['functions'],
                    'spans': [{'file': s['file'], 'a': s['a'], 'b': s['b']} for s in gold['spans']]},
           'meta': {'run': job['run'], 'state_call': state_k, 'gold_printed_at': gold_at, 'gold_fn_named_at': fn_at,
                    'recorded_next': info['next_recorded'], 'recorded_first_edit': first_edit,
                    'wasted_reads': info['wasted'], 'resolved_in_run': job['resolved'], 'check_ready': ready,
                    'edit_statuses': statuses, 'clock_aligned': job['aligned'] >= state_k,
                    'recorded_replay_same': sum(1 for k in range(state_k) if turns[k].get('result') is not None and
                                                unclock_result(turns[k]['result']) == unclock_result(R.results[k]))}}
    return rec, None, info


CLOCK = re.compile(r'\b\d+m\d\ds\b|\b\d+:\d\d/')


def unclock_result(res):
    try:
        d = json.loads(res)
    except (TypeError, ValueError):
        return res
    if isinstance(d, dict) and 'stdout' in d:
        d['stdout'] = CLOCK.sub('T', d['stdout'] or '')
    return json.dumps(d, sort_keys=True)


def work_task(job_group):
    import replay_tools
    task, jobs, opt = job_group
    workdir = tempfile.mkdtemp(prefix=f'rl_{task}_', dir=opt['workroot'])
    ws = os.path.join(workdir, 'ws')
    out = []
    t0 = time.time()
    try:
        pristine = prepare_ws(os.path.join(opt['snapshots'], f'{task}.tgz'), ws)
        env_path = replay_tools.task_env_path(task, workdir, opt['manifest'], opt['spcache'])
        agent = {'scripts': opt['scripts'], 'skill': S.SKILL}
        for job in jobs:
            try:
                rec, why, info = relabel_one(job, agent, ws, pristine, workdir, env_path)
            except Exception:  # noqa: BLE001
                rec, why, info = None, 'error: ' + traceback.format_exc()[-400:].replace('\n', ' | '), \
                    {'run': job['run'], 'task': task}
            out.append({'run': job['run'], 'task': task, 'record': rec, 'discard': why, 'info': info})
    except Exception:  # noqa: BLE001
        for job in jobs:
            out.append({'run': job['run'], 'task': task, 'record': None,
                        'discard': 'setup: ' + traceback.format_exc()[-300:].replace('\n', ' | '), 'info': {}})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return task, out, time.time() - t0


def replay_inside(a):
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    if a.tasks:
        jobs = [j for j in jobs if j['task'] in set(a.tasks.split(','))]
    by = collections.defaultdict(list)
    for j in jobs:
        by[j['task']].append(j)
    opt = {'snapshots': a.snapshots, 'manifest': a.manifest, 'spcache': a.spcache, 'workroot': a.workroot,
           'scripts': a.scripts}
    n = 0
    with open(a.out_file, 'w') as f, concurrent.futures.ProcessPoolExecutor(a.workers) as ex:
        for task, out, secs in ex.map(work_task, [(t, js, opt) for t, js in sorted(by.items())]):
            n += 1
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + '\n')
            f.flush()
            print(f'[{n}/{len(by)}] {task}: {sum(1 for o in out if o["record"])}/{len(out)} states, {secs:.0f}s',
                  flush=True)


def replay(workers, tasks=None):
    """Run replay_inside in the sandbox image (python3.13, linux/amd64), the repository mounted read-only at its own
    path (the worktree's data/ and results/ symlinks resolve), data/rft3 writable."""
    import shlex
    main_repo = os.path.realpath(os.path.join(ROOT, 'data', '..'))
    real_data = os.path.realpath(os.path.join(ROOT, 'data'))
    main_root = os.path.dirname(real_data)
    agent = S.Agent()
    scripts = os.path.realpath(agent.scripts)
    out_dir = os.path.realpath(OUT)
    inner = ['python', os.path.join(ROOT, 'scripts', 'rft3_relabel.py'), 'inside', '--jobs', os.path.realpath(JOBS),
             '--out-file', os.path.join(out_dir, os.path.basename(RAW)), '--workers', str(workers),
             '--snapshots', os.path.join(real_data, 'competition', 'snapshots'),
             '--manifest', os.path.join(real_data, 'competition', 'wheels_local', 'task_env.json'),
             '--spcache', '/spcache', '--workroot', '/tmp', '--scripts', scripts] + \
        (['--tasks', tasks] if tasks else [])
    setup = f'tar --no-same-owner -xf /spcache/sp_base.tar -C {SP_DIR} && ' if \
        os.path.exists(os.path.join(SP_CACHE, 'sp_base.tar')) else ''
    mounts = ['-v', f'{main_root}:{main_root}:ro', '-v', f'{out_dir}:{out_dir}', '-v', f'{SP_CACHE}:/spcache:ro']
    if not ROOT.startswith(main_root + os.sep):
        mounts += ['-v', f'{ROOT}:{ROOT}:ro']
    cmd = ['docker', 'run', '--rm', '--platform', 'linux/amd64', '--network', 'none', *mounts, '-w', '/tmp',
           '-e', 'PYTHONDONTWRITEBYTECODE=1', IMAGE, 'bash', '-c', setup + ' '.join(shlex.quote(x) for x in inner)]
    del main_repo
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f'replay done in {time.time() - t0:.0f}s')


# ------------------------------------------------------------------ collect (host)

def collect(max_prompt=S.COMPACT_AT):
    import rft3_oracle
    agent = S.Agent()
    jobs = {(j['run'], j['task']): j for j in (json.loads(l) for l in open(JOBS) if l.strip())}
    rows = [json.loads(l) for l in open(RAW) if l.strip()]
    held = heldout()
    discards = collections.Counter()
    kept = []
    for r in rows:
        if r['record'] is None:
            discards[re.sub(r'\d+', 'N', (r['discard'] or '?').split(' | ')[0][:70])] += 1
            continue
        rec = r['record']
        assert rec['task'] not in held
        job = jobs[(r['run'], r['task'])]
        rec['request'] = S.build_request(agent, job['repo'], job['problem_statement'], job['tree'])
        rec = {k: rec[k] for k in ('id', 'source', 'task', 'repo', 'request', 'turns', 'gold', 'meta')}
        if rft3_oracle.check_not_clean(rec):  # v20: the gold leaves a CHECK warning (see rft3_oracle.collect)
            discards['CHECK line not clean after the last gold edit of a file'] += 1
            continue
        n, est = S.prompt_tokens(rec)
        rec['meta']['rendered_prompt_tokens'] = n
        rec['meta']['est_prompt_tokens'] = est
        if est >= max_prompt:
            discards['estimated prompt >= 14,336 (compaction)'] += 1
            continue
        kept.append(rec)
    os.makedirs(S.STATES_DIR, exist_ok=True)
    with open(os.path.join(S.STATES_DIR, 'relabel.jsonl'), 'w') as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    infos = [r['info'] for r in rows]
    gold_seen = [i for i in infos if i.get('gold_at') is not None]
    ks = [r['meta']['state_call'] for r in kept]
    wasted = [r['meta']['wasted_reads'] for r in kept]
    rep = {'episodes': len(rows), 'kept': len(kept), 'tasks': len({r['task'] for r in kept}),
           'by_run': dict(collections.Counter(r['meta']['run'] for r in kept)),
           'by_repo': dict(collections.Counter(r['repo'] for r in kept)),
           'discards': dict(discards.most_common()),
           'episodes_where_gold_got_printed_before_any_edit': len(gold_seen),
           'state_call': {'median': statistics.median(ks) if ks else None, 'hist': dict(sorted(
               collections.Counter(ks).items()))},
           'gold_printed_at': dict(sorted(collections.Counter(r['meta']['gold_printed_at'] for r in kept).items())),
           'wasted_reads_after_gold_visible': {'mean': round(statistics.mean(wasted), 2) if wasted else None,
                                               'median': statistics.median(wasted) if wasted else None,
                                               'hist': dict(sorted(collections.Counter(wasted).items()))},
           'never_edited_after_state': sum(r['meta']['recorded_first_edit'] is None for r in kept),
           'resolved_in_run': sum(bool(r['meta']['resolved_in_run']) for r in kept),
           'recorded_next_at_state': dict(collections.Counter(r['meta']['recorded_next'] for r in kept)),
           'prefix_outputs_identical_mod_clock_to_recording': {
               r['meta']['run']: 0 for r in kept}}
    same = collections.defaultdict(lambda: [0, 0])
    for r in kept:
        same[r['meta']['run']][0] += r['meta']['recorded_replay_same']
        same[r['meta']['run']][1] += r['meta']['state_call']
    rep['prefix_outputs_identical_mod_clock_to_recording'] = {k: f'{a}/{b}' for k, (a, b) in same.items()}
    rep['edit_calls'] = dict(collections.Counter(
        'multi' if (t.get('meta') or {}).get('hunks', 1) > 1 else 'single' for r in kept for t in r['turns']
        if t['label'] == 'train' and t.get('kind') == 'edit'))
    rep['edit_check_lines'] = dict(collections.Counter(
        rft3_oracle.check_status(json.loads(t['result']).get('stdout')) for r in kept for t in r['turns']
        if t['label'] == 'train' and t.get('kind') == 'edit'))
    rep['multi_refused'] = sum(bool(r['info'].get('multi_refused')) for r in rows)
    json.dump(rep, open(REPORT, 'w'), indent=1)
    print(json.dumps(rep, indent=1))
    if kept:
        print(S.stats([os.path.join(S.STATES_DIR, 'relabel.jsonl')]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--version', default=None, help='agent version (rft3_states.VERSIONS: v19 default, v20)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('jobs')
    r = sub.add_parser('replay')
    r.add_argument('--workers', type=int, default=6)
    r.add_argument('--tasks', default=None)
    sub.add_parser('collect')
    i = sub.add_parser('inside')
    for k in ('jobs', 'out-file', 'snapshots', 'manifest', 'spcache', 'workroot', 'scripts'):
        i.add_argument(f'--{k}', required=True)
    i.add_argument('--workers', type=int, default=6)
    i.add_argument('--tasks', default=None)
    a = ap.parse_args()
    if a.version:
        configure(a.version)
    if a.cmd == 'jobs':
        make_jobs()
    elif a.cmd == 'replay':
        replay(a.workers, a.tasks)
    elif a.cmd == 'inside':
        replay_inside(a)
    else:
        collect()


if __name__ == '__main__':
    main()
