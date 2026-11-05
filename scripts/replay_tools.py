"""LLM-free regression replay: re-run recorded tool calls through another agent version's scripts.

Takes recorded runs (results/<exp>/r<k>/c<j>/: traces/trace_<task>.json, task_results.jsonl, patches/<task>.patch),
and replays every recorded call of each run, in order, through one or more agent versions' repo-tools scripts on a
fresh workspace (data/competition/snapshots/<task>.tgz, git-initialized like the harness: .git/info/exclude, a
`baseline` commit tagged _swegemma_baseline) with a fresh REPO_TOOLS_DIR. No model calls: the model's calls are
fixed, only the tools change. Reports per run and in aggregate:

1. Edit fidelity (the main regression signal). Every edit.py/create.py call that was applied in the recording
   (EDITED/CREATED/REPLACED): is it still applied, with the same file and line range? NOT APPLIED / refusals are
   listed with their first line; edits the new version applies that the recording did not are "newly applied".
   The final patch (the last submit_patch diff, else the working-tree diff at the end, as the harness grades it)
   is compared with the recorded patch: identical / different / empty. For runs that PASSED, a different final
   diff is a red flag. With several agents, the first agent is the reference: the workspace fingerprint after
   every call is compared and the first divergent call is named.
2. Script health: crashes (non-zero exit or a Python traceback on stderr), timeouts.
3. Context cost: characters per call as the model sees them (json.dumps of the function response, like
   observation.content), recorded vs replayed, and the estimated prompt before each model call with a fixed fit
   prompt ~ base + 0.2953*chars + 336*calls (chars = results + call arguments): the model call at which 14,336
   tokens (the scorer's compaction threshold) is first reached, recorded vs replayed (and the recording's actual
   prompt_tokens). base: the agent's flow.CTX_BASE_TOKENS when its flow has the sawtooth fit (CTX_POST_BASE_TOKENS,
   v18+), else 5200 (LOG 2026-10-04 refit for v17b's prompt); --base LABEL=N overrides.
4. Non-script calls: submit_patch takes the diff (as the harness does, `git add -N .` + diff vs the baseline);
   get_status and other tools are no-ops. Calls ADK would reject for the replayed version (missing skill_name or
   file_path, wrong skill name, unknown script, args not an object/list) are skipped and counted; calls rejected
   in the recording but accepted by the new version (e.g. v18's repro.py alias) run and are counted as
   "newly accepted".

Faithfulness notes:
- Arguments become argv exactly as ADK's run_skill_script does: --key str(value) (Python str(), so 12 -> "12",
  True -> "True"); scripts run through runpy.run_path('scripts/<name>.py') with cwd = a temp dir (as ADK's wrapper).
- Time: flow.py reads time.time() (TIME_BUDGET clock, footer prefix `m:ss/4:30`). The replay wrapper patches
  time.time() in the script process to the recorded call's timestamp (+ real time elapsed inside the call), so the
  flow clock sees the recorded elapsed times. Child processes (run.py snippets/tests, import guards) see real time.
- Workspace paths: the replay workspace is not /workspace; its path is replaced by /workspace in all outputs.
- run.py (snippets/tests) and the edit import guard depend on the Python environment. With --env task (default)
  the container gets the stock base site-packages (sp_base.tar) plus the task's per-task environment
  (scripts/task_env.py tars, on PYTHONPATH ahead of site-packages), close to TASK_ENV=1 runs. run.py results are
  still marked env-dependent: they are excluded from the fidelity verdict, but they run (their exit codes change
  flow state) and their agreement with the recording (exit code line) is reported.
- A call the harness killed ("Command timed out after Ns") is replayed with the same N-second timeout. A call with
  no recorded result (session timeout during the call) is replayed and flagged.

Usage (from the repo root; runs inside the sandbox image via docker by default, like scripts/test_skills.sh):
    .venv/bin/python scripts/replay_tools.py --runs results/E041_v17b_eval113,results/E043_v17b_kaggleslow \\
        --agent agent-v17b --agent agent-v18 --passing --out results/R001_replay_v18 [--workers 6]
    # --agent: a git ref (materialized with scripts/materialize.sh), a directory (e.g. results/E044_x/_agent),
    #          or WORKTREE; LABEL=SPEC names it. The first agent is the reference for per-call divergence.
    # --tasks fastapi_14258,httpx_3672   only these tasks;  --limit N  first N runs;  --host  no docker
    python3 scripts/replay_tools.py --report results/R001_replay_v18    # re-print the summary
Outputs: <out>/runs.jsonl (one record per run x agent, every call, with each replayed stdout and the final patch),
<out>/summary.txt (printed), and <out>/grade_<agent>_<k>.jsonl: tasks files whose `patch` is the replayed final patch
of each passing run with a changed diff, to grade with the hidden tests (the printed scripts/verify_gold.py command).
A changed diff is not a failure by itself: the model's later calls were made against the recorded outputs, so when
the new version accepts an edit the recording refused, the recorded retry applies a second time (a "cascade").
"""

import argparse
import ast
import concurrent.futures
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE = 'swebench-sandbox:latest'
SP_DIR = '/usr/local/lib/python3.13/site-packages'
SP_CACHE = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'swegemma_sp_cache_v8')
COMPACT_AT = 14336
K_CHARS, K_CALLS, BASE_V17B = 0.2953, 336, 5200
CALL_TIMEOUT = 180            # eval_config timeout_seconds
BASELINE = '_swegemma_baseline'
GIT_EXCLUDE = '__pycache__/\n*.pyc\n.pytest_cache/\n*.egg-info/\nbuild/\ndist/\n.coverage\n.adk_exec_*.py\n'
EDIT_SCRIPTS = ('edit.py', 'create.py')
READ_ONLY = {'locate.py', 'show.py', 'search.py', 'outline.py', 'callers.py', 'tests_for.py', 'history.py',
             'status.py', 'hypothesis.py'}
APPLIED = re.compile(r'^(EDITED|CREATED|REPLACED) (\S+)(?: \(([^)]*)\))?', re.M)
RANGE = re.compile(r'lines? (\d+)(?:-(\d+))?')
EXIT_LINE = re.compile(r'^exit code: (\S+)', re.M)

# The script process: argv as ADK builds it, runpy as ADK's wrapper, and time.time() on the recorded clock: from the
# recorded call time while flow.before() and the script run, from the recorded result time once flow.after() starts
# (it writes the action's ts and the footer's elapsed time), each plus the real time elapsed since.
WRAPPER = r'''
import os, runpy, sys, time
_a, _b = os.environ.pop('REPLAY_NOW', ''), os.environ.pop('REPLAY_END', '')
if _a:
    _real = time.time
    _clk = [float(_a), _real()]
    time.time = lambda: _clk[0] + (_real() - _clk[1])
    if _b:
        sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
        try:
            import flow as _flow
            _after = _flow.after
            def _after_at_end(*a, **k):
                _clk[0], _clk[1] = float(_b), _real()
                return _after(*a, **k)
            _flow.after = _after_at_end
        except Exception:
            pass
        sys.path.pop(0)
sys.argv = %r
runpy.run_path(sys.argv[0], run_name='__main__')
'''


# ------------------------------------------------------------------ recorded runs

def load_runs(dirs, passing=False, tasks=None):
    """[{id, exp, dir, task, resolved, trace, patch}] for every task_results.jsonl line under the results dirs."""
    runs = []
    for d in dirs:
        d = d.rstrip('/')
        for tr in sorted(glob.glob(f'{d}/r*/c*/task_results.jsonl') + glob.glob(f'{d}/r*/task_results.jsonl')):
            rd = os.path.dirname(tr)
            for line in open(tr):
                if not line.strip():
                    continue
                r = json.loads(line)
                task = r['instance_id']
                if passing and not r.get('resolved'):
                    continue
                if tasks and task not in tasks:
                    continue
                trace = os.path.join(rd, 'traces', f'trace_{task}.json')
                if not os.path.exists(trace):
                    continue
                runs.append({'id': os.path.relpath(os.path.join(rd, task), os.path.dirname(d)),
                             'exp': os.path.basename(d), 'dir': rd, 'task': task, 'resolved': bool(r.get('resolved')),
                             'trace': trace, 'patch': os.path.join(rd, 'patches', f'{task}.patch')})
    return runs


def _extra(x):
    return x if isinstance(x, dict) else {}


def recorded_calls(trace_file):
    """Every model tool call of a trace, in order: name, arguments, recorded result content, clock, prompt size."""
    steps = json.load(open(trace_file))['steps']
    calls = []
    for s in steps:
        if s.get('source') != 'agent' or not s.get('tool_calls'):
            continue
        obs = s.get('observation')
        content = obs.get('content') if isinstance(obs, dict) else None
        pt = _extra(s.get('metrics')).get('prompt_tokens')
        for tc in s['tool_calls']:
            ex = _extra(tc.get('extra')) or _extra(s.get('extra'))
            ox = obs.get('extra') if isinstance(obs, dict) else None
            if isinstance(ox, str):   # a dict's repr in ATIF traces
                try:
                    ox = ast.literal_eval(ox)
                except (ValueError, SyntaxError):
                    ox = None
            calls.append({'name': tc.get('function_name'), 'args': tc.get('arguments') or {}, 'content': content,
                          'ts': ex.get('timestamp'), 'end_ts': _extra(ox).get('timestamp'),
                          'elapsed': ex.get('elapsed_s'), 'prompt_tokens': pt})
    return calls


def recorded_kind(c):
    """ran | rejected | harness_timeout | noresult | tool (non-script tool)."""
    if c['name'] != 'run_skill_script':
        return 'tool'
    if c['content'] is None:
        return 'noresult'
    try:
        d = json.loads(c['content'])
    except (TypeError, ValueError):
        return 'ran'
    if isinstance(d, dict) and d.get('error_code'):
        return 'rejected'
    if isinstance(d, dict) and str(d.get('stderr') or '').startswith('Command timed out after'):
        return 'harness_timeout'
    return 'ran'


def recorded_stdout(c):
    try:
        d = json.loads(c['content'])
    except (TypeError, ValueError):
        return c['content'] or ''
    return (d.get('stdout') or d.get('error') or '') if isinstance(d, dict) else str(d)


# ------------------------------------------------------------------ ADK's run_skill_script, for one agent version

def skill_info(agent_dir):
    """(skill name, scripts dir) of the agent's repo-tools skill (SKILL.md front matter `name:`)."""
    skills = sorted(glob.glob(os.path.join(agent_dir, 'skills', '*', 'SKILL.md')))
    for md in skills:
        m = re.search(r'^name:\s*(\S+)', open(md).read(), re.M)
        if m and os.path.isdir(os.path.join(os.path.dirname(md), 'scripts')):
            return m.group(1), os.path.join(os.path.dirname(md), 'scripts')
    raise SystemExit(f'no skill with scripts/ in {agent_dir}')


def adk_check(args, skill_name, scripts):
    """None when ADK would run the call, else its error object (run_skill_script.run_async's checks)."""
    sn, fp, sa = args.get('skill_name'), args.get('file_path'), args.get('args')
    so, pa = args.get('short_options'), args.get('positional_args')
    errors = []
    if not sn:
        errors.append("Argument 'skill_name' is required.")
    if not fp:
        errors.append("Argument 'file_path' is required.")
    if sa is not None and not isinstance(sa, (dict, list)):
        errors.append(f"'args' must be a JSON object (dict) or a list of strings, got {type(sa).__name__}.")
    if so is not None and not isinstance(so, dict):
        errors.append(f"'short_options' must be a JSON object (dict), got {type(so).__name__}.")
    if pa is not None and not isinstance(pa, list):
        errors.append(f"'positional_args' must be a list of strings, got {type(pa).__name__}.")
    if isinstance(sa, list) and (so or pa):
        errors.append("Cannot specify 'short_options' or 'positional_args' when 'args' is a list.")
    if errors:
        return {'error': '\n'.join(errors), 'error_code': 'INVALID_ARGUMENTS'}
    if sn != skill_name:
        return {'error': f"Skill '{sn}' not found.", 'error_code': 'SKILL_NOT_FOUND'}
    name = fp[len('scripts/'):] if str(fp).startswith('scripts/') else str(fp)
    if '/' in name or not os.path.isfile(os.path.join(scripts, name)):
        return {'error': f"Script '{fp}' not found in skill '{sn}'.", 'error_code': 'SCRIPT_NOT_FOUND'}
    if not name.endswith('.py'):
        return {'error': f"Unsupported script type for '{fp}'.", 'error_code': 'UNSUPPORTED_SCRIPT_TYPE'}
    return None


def script_argv(name, args):
    """sys.argv as ADK's wrapper builds it for a .py script."""
    argv = [f'scripts/{name}']
    sa = args.get('args')
    if isinstance(sa, list):
        return argv + [str(v) for v in sa]
    for k, v in (sa or {}).items():
        argv += [f'--{k}', str(v)]
    for k, v in (args.get('short_options') or {}).items():
        argv += [f'-{k}', str(v)]
    if args.get('positional_args'):
        argv += ['--'] + [str(v) for v in args['positional_args']]
    return argv


def tool_result(file_path, out, err, skill='repo-tools'):
    """The function response ADK returns for a .py script (status from stdout/stderr, as run_skill_script)."""
    status = 'success' if not err else ('error' if not out else 'warning')
    return {'skill_name': skill, 'file_path': file_path, 'stdout': out, 'stderr': err, 'status': status}


def run_script(scripts, name, args, ws, state, now, timeout, env_path=None, end=None):
    """(stdout, stderr, exit code or 'timeout', seconds) of one script call, outputs with the ws path -> /workspace."""
    env = dict(os.environ, WORKSPACE=ws, REPO_TOOLS_DIR=state, PYTHONDONTWRITEBYTECODE='1')
    env.pop('REPO_TOOLS_FLOW', None)
    env.pop('PYTHONPATH', None)
    if env_path:
        env['PYTHONPATH'] = env_path
    if now:
        env['REPLAY_NOW'] = repr(float(now))
        if end and float(end) >= float(now):
            env['REPLAY_END'] = repr(float(end))
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix='adk_') as td:
        os.symlink(os.path.abspath(scripts), os.path.join(td, 'scripts'))
        try:
            r = subprocess.run([sys.executable, '-c', WRAPPER % (script_argv(name, args),)], cwd=td, env=env,
                               capture_output=True, text=True, timeout=timeout, errors='replace')
            out, err, code = r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            out, err, code = '', f'Command timed out after {timeout}s (exit_code=124)', 'timeout'
    fix = lambda s: s.replace(ws.rstrip('/') + '/', '/workspace/').replace(ws.rstrip('/'), '/workspace')  # noqa
    return fix(out), fix(err), code, time.time() - t0


# ------------------------------------------------------------------ workspace

def git(ws, *a, check=True, inp=None):
    env = dict(os.environ, GIT_AUTHOR_NAME='replay', GIT_AUTHOR_EMAIL='replay@local', GIT_COMMITTER_NAME='replay',
               GIT_COMMITTER_EMAIL='replay@local')
    r = subprocess.run(['git', '-c', 'core.quotepath=false', '-c', 'commit.gpgsign=false', '-c',
                        f'safe.directory={ws}', '--no-pager', *a], cwd=ws, env=env, capture_output=True, input=inp)
    if check and r.returncode:
        raise RuntimeError(f'git {a}: {r.stderr.decode(errors="replace")[:400]}')
    return r.stdout.decode(errors='replace')


def prepare(tgz, ws):
    """Extract the snapshot and set up git as the harness does; returns the ignored paths of the pristine tree."""
    os.makedirs(ws, exist_ok=True)
    subprocess.run(['tar', '--no-same-owner', '-xzf', tgz, '-C', ws], check=True, capture_output=True)
    real = os.path.realpath(ws)
    for dp, dns, fns in os.walk(ws):  # circular symlinks (harness: removed before setup)
        for n in dns + fns:
            p = os.path.join(dp, n)
            if os.path.islink(p):
                t = os.path.realpath(p)
                if t == real or real.startswith(t.rstrip('/') + '/'):
                    os.unlink(p)
    if not os.path.isdir(os.path.join(ws, '.git')):
        git(ws, 'init', '-q')
    os.makedirs(os.path.join(ws, '.git', 'info'), exist_ok=True)
    with open(os.path.join(ws, '.git', 'info', 'exclude'), 'a') as f:
        f.write(GIT_EXCLUDE)
    git(ws, 'add', '-A')
    git(ws, 'commit', '-m', 'baseline', '--allow-empty', '-q', '--no-verify')
    git(ws, 'tag', '-f', BASELINE)
    return set(ignored(ws))


def ignored(ws):
    return [p for p in git(ws, 'ls-files', '-o', '-i', '--exclude-standard', '--directory').splitlines() if p]


def reset(ws, pristine_ignored):
    git(ws, 'reset', '-q', '--hard', BASELINE)
    git(ws, 'clean', '-qffd')
    for p in ignored(ws):
        if p not in pristine_ignored:
            full = os.path.join(ws, p)
            shutil.rmtree(full, ignore_errors=True) if os.path.isdir(full) and not os.path.islink(full) \
                else os.path.exists(full) and os.unlink(full)


def patch_diff(ws):
    """What submit_patch (and the harness fallback) captures."""
    git(ws, 'add', '-N', '.', check=False)
    return git(ws, 'diff', '--binary', BASELINE, check=False)


def fingerprint(ws):
    """Hash of the working tree's changes (tracked diff + untracked files), without touching the index."""
    h = hashlib.sha1(git(ws, 'diff', '--binary', BASELINE, check=False).encode())
    for p in sorted(git(ws, 'ls-files', '-o', '--exclude-standard').splitlines()):
        h.update(p.encode())
        try:
            with open(os.path.join(ws, p), 'rb') as f:
                h.update(hashlib.sha1(f.read()).digest())
        except OSError:
            pass
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ per-call classification

def edit_outcome(out):
    """(applied | refused | already_done | usage | other, file, line range or detail, first line) of an edit result."""
    text = out or ''
    head = text.split('\n--- a/')[0].split('\n--- /dev/null')[0]
    first = next((l for l in text.splitlines() if l.strip() and not l.startswith('NOTE')), '')[:200]
    if text.lstrip().startswith('ALREADY DONE'):
        return 'already_done', None, None, first
    m = APPLIED.search(head)
    if m:
        return 'applied', m.group(2), m.group(3), m.group(0)[:200]
    if 'USAGE ERROR' in text[:200]:
        return 'usage', None, None, first
    if re.search(r'NOT APPLIED|NOT CREATED|REFUSED|NOT RUN', text[:600]):
        return 'refused', None, None, first
    return 'other', None, None, first


def refusal(out):
    """The flow refusal a result starts with (NOT RUN / ALREADY DONE), else None."""
    t = (out or '').lstrip()
    return 'NOT RUN' if t.startswith('NOT RUN') else 'ALREADY DONE' if t.startswith('ALREADY DONE') else None


CLOCK = re.compile(r'\b\d+m\d\ds\b|\b\d+:\d\d/')


def unclock(text):
    """Output without the flow clock (BUDGET `1m23s`, prefix `1:23/`), which the recorded clock reproduces to ~3 s."""
    return CLOCK.sub('T', text or '')


def same_range(a, b):
    ra, rb = RANGE.search(a or ''), RANGE.search(b or '')
    if ra and rb:
        return ra.groups() == rb.groups() or (ra.group(1), ra.group(2) or ra.group(1)) == \
            (rb.group(1), rb.group(2) or rb.group(1))
    return True


def est_crossing(chars, base):
    """1-based model call whose prompt first reaches COMPACT_AT, with flow.py's fit (prompt for model call n: base +
    K_CHARS * raw characters of the script calls run before it (output + argv, as flow counts them) + K_CALLS * their
    number; rejected and non-script calls are absorbed in the per-call term, as in the fit). `chars`: per model
    call, the raw characters of a script call that ran, or None."""
    total = ran = 0
    for n, c in enumerate(chars, 1):
        if base + K_CHARS * total + K_CALLS * ran >= COMPACT_AT:
            return n
        if c is not None:
            total += c
            ran += 1
    n = len(chars) + 1
    return n if base + K_CHARS * total + K_CALLS * ran >= COMPACT_AT else None


def flow_base(agent_dir, scripts):
    try:
        src = open(os.path.join(scripts, 'flow.py')).read()
    except OSError:
        return BASE_V17B
    m = re.search(r'^CTX_BASE_TOKENS\s*=\s*(\d+)', src, re.M)
    return int(m.group(1)) if m and re.search(r'^CTX_POST_BASE_TOKENS\s*=', src, re.M) else BASE_V17B


# ------------------------------------------------------------------ replay one run through one agent

def replay_run(run, calls, agent, ws, state_root, env_path, recorded_patch, ref=None):
    """Replay `calls` through `agent` ({label, dir, scripts, skill}) on the reset workspace `ws`."""
    state = tempfile.mkdtemp(prefix='state_', dir=state_root)
    rec_out = {'run': run['id'], 'exp': run['exp'], 'task': run['task'], 'resolved': run['resolved'],
               'agent': agent['label'], 'calls': []}
    submitted, last_ts = None, None
    for i, c in enumerate(calls, 1):
        kind = recorded_kind(c)
        ts = c['ts'] or ((last_ts + 1) if last_ts else None)
        last_ts = ts
        args = c['args'] if isinstance(c['args'], dict) else {}
        fp = str(args.get('file_path') or '')
        script = fp[len('scripts/'):] if fp.startswith('scripts/') else fp
        rc = {'i': i, 'tool': c['name'], 'script': script if c['name'] == 'run_skill_script' else None,
              'rec_kind': kind, 'rec_chars': len(c['content'] or ''),
              'arg_chars': len(json.dumps(c['args'], ensure_ascii=False)),
              'argv_chars': sum(len(x) for x in script_argv(script, args)[1:]) if c['name'] == 'run_skill_script' else 0}
        if kind == 'ran':
            rc['rec_raw'] = len(recorded_stdout(c)) + len(str((json.loads(c['content']) or {}).get('stderr') or ''))
        if c['name'] != 'run_skill_script':
            if c['name'] == 'submit_patch':
                submitted = patch_diff(ws)
                content = json.dumps({'status': 'ok', 'patch_size': len(submitted),
                                      'files_changed': submitted.count('diff --git')})
            else:
                content = c['content'] or ''
            rc.update(new_kind='tool', new_chars=len(content))
            rec_out['calls'].append(rc)
            continue
        err = adk_check(args, agent['skill'], agent['scripts'])
        if err:
            content = c['content'] if kind == 'rejected' else json.dumps(err)
            rc.update(new_kind='rejected', error_code=err['error_code'], new_chars=len(content))
            rec_out['calls'].append(rc)
            continue
        timeout = CALL_TIMEOUT
        if kind == 'harness_timeout':
            m = re.search(r'after (\d+)s', recorded_stdout(c) + str(json.loads(c['content']).get('stderr')))
            timeout = int(m.group(1)) if m else CALL_TIMEOUT
        out, errout, code, secs = run_script(agent['scripts'], script, args, ws, state, ts, timeout, env_path,
                                             c.get('end_ts'))
        content = json.dumps(tool_result(fp, out, errout))
        rc.update(new_kind='ran', new_chars=len(content), new_raw=len(out) + len(errout), code=code, secs=round(secs, 2),
                  newly_accepted=kind == 'rejected')
        if code == 'timeout':
            rc['timeout'] = True
        elif 'Traceback (most recent call last)' in errout or code not in (0, 1, 2) or (code and not out.strip()):
            rc['crash'] = (errout or out)[-600:]   # exit 1/2 with output: a refusal / no hit / usage (normal)
        rc['out'] = out if len(out) < 20000 else out[:20000] + '...'
        rc['refusal'] = refusal(out)
        if kind == 'ran':
            rc['rec_refusal'] = refusal(recorded_stdout(c))
            rc['same_stdout'] = out == recorded_stdout(c)
            rc['same_mod_clock'] = rc['same_stdout'] or unclock(out) == unclock(recorded_stdout(c))
        if script in EDIT_SCRIPTS:
            o_status, o_file, o_det, o_first = edit_outcome(recorded_stdout(c)) if kind == 'ran' else (kind, None,
                                                                                                         None, '')
            n_status, n_file, n_det, n_first = edit_outcome(out)
            rc['edit'] = {'rec': o_status, 'new': n_status, 'rec_line': o_first, 'new_line': n_first}
            if o_status == 'applied':
                if n_status != 'applied':
                    rc['edit']['verdict'] = 'NOT_APPLIED'
                elif n_file != o_file or not same_range(o_det, n_det):
                    rc['edit']['verdict'] = 'changed_range'
                else:
                    rc['edit']['verdict'] = 'same'
            elif n_status == 'applied':
                rc['edit']['verdict'] = 'applied_no_record' if kind in ('noresult', 'harness_timeout') \
                    else 'newly_applied'
        if script == 'run.py':
            ro, no = EXIT_LINE.search(recorded_stdout(c)), EXIT_LINE.search(out)
            rc['env_dependent'] = True
            rc['run_agree'] = (ro.group(1) if ro else None) == (no.group(1) if no else None) if kind == 'ran' else None
        if script not in READ_ONLY:
            rc['fp'] = fingerprint(ws)
        rec_out['calls'].append(rc)
    final = submitted if submitted is not None else patch_diff(ws)
    if not final.strip():
        final = ''
    rec_out['final_patch'] = final
    rec_out['diff'] = 'empty' if not final else ('identical' if final == recorded_patch else 'different')
    rec_out['recorded_patch_empty'] = not recorded_patch.strip()
    rec_out['actual_cross'] = next((c_['i'] for c_, x in zip(rec_out['calls'], calls)
                                    if x['prompt_tokens'] and int(x['prompt_tokens']) >= COMPACT_AT), None)
    shutil.rmtree(state, ignore_errors=True)
    return rec_out


def compare_to_ref(rec, ref):
    """First call whose workspace fingerprint differs from the reference agent's replay; final patch equality."""
    rec['diff_vs_ref'] = 'identical' if rec['final_patch'] == ref['final_patch'] else 'different'
    rf = {c['i']: c.get('fp') for c in ref['calls']}
    last_r = last_n = None
    for c in rec['calls']:
        if c['i'] in rf and rf[c['i']] is not None:
            last_r = rf[c['i']]
        if c.get('fp') is not None:
            last_n = c['fp']
        if (c.get('fp') is not None or rf.get(c['i']) is not None) and last_r != last_n:
            rec['first_divergence'] = c['i']
            return
    rec['first_divergence'] = None


# ------------------------------------------------------------------ worker (inside the sandbox image)

def task_env_path(task, workdir, manifest, spcache):
    """Extract the task's per-task environment (scripts/task_env.py) once; its dir goes on PYTHONPATH."""
    if not manifest or not os.path.exists(manifest):
        return None
    entry = json.load(open(manifest))['tasks'].get(task)
    if not entry or not entry.get('install'):
        return None
    key = hashlib.sha1(json.dumps([entry['remove'], entry['install']]).encode()).hexdigest()[:16]
    tar = os.path.join(spcache, f'task_env_{key}.tar')
    if not os.path.exists(tar):
        return None
    d = os.path.join(workdir, 'env')
    os.makedirs(d, exist_ok=True)
    subprocess.run(['tar', '--no-same-owner', '-xf', tar, '-C', d], check=True, capture_output=True)
    return d


def work_task(job):
    """Replay every run of one task through every agent (one extraction per task)."""
    task, runs, agents, opt = job
    t0 = time.time()
    workdir = tempfile.mkdtemp(prefix=f'replay_{task}_', dir=opt['workroot'])
    ws = os.path.join(workdir, 'ws')
    out = []
    try:
        pristine = prepare(os.path.join(opt['snapshots'], f'{task}.tgz'), ws)
        env_path = task_env_path(task, workdir, opt.get('manifest'), opt.get('spcache')) \
            if opt.get('env') == 'task' else None
        for run in runs:
            calls = recorded_calls(run['trace'])
            recorded_patch = open(run['patch'], errors='replace').read() if os.path.exists(run['patch']) else ''
            ref = None
            for agent in agents:
                reset(ws, pristine)
                try:
                    rec = replay_run(run, calls, agent, ws, workdir, env_path, recorded_patch)
                except Exception as e:  # noqa: BLE001 - one broken run must not stop the batch
                    rec = {'run': run['id'], 'exp': run['exp'], 'task': task, 'resolved': run['resolved'],
                           'agent': agent['label'], 'calls': [], 'error': repr(e)[:500], 'diff': 'error',
                           'final_patch': ''}
                if ref is None:
                    ref = rec
                else:
                    compare_to_ref(rec, ref)
                out.append(rec)
    except Exception as e:  # noqa: BLE001
        for run in runs:
            for agent in agents:
                out.append({'run': run['id'], 'exp': run['exp'], 'task': task, 'resolved': run['resolved'],
                            'agent': agent['label'], 'calls': [], 'error': f'setup: {e!r}'[:500], 'diff': 'error',
                            'final_patch': ''})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return task, out, time.time() - t0


def run_inside(a, agents):
    runs = load_runs(a.runs.split(','), a.passing, set(a.tasks.split(',')) if a.tasks else None)
    if a.limit:
        runs = runs[:a.limit]
    by_task = {}
    for r in runs:
        by_task.setdefault(r['task'], []).append(r)
    opt = {'snapshots': a.snapshots, 'workroot': a.workroot or tempfile.gettempdir(), 'env': a.env,
           'manifest': a.manifest, 'spcache': a.spcache}
    os.makedirs(a.out, exist_ok=True)
    # biggest tasks first: fewer stragglers
    jobs = sorted(((t, rs, agents, opt) for t, rs in by_task.items()),
                  key=lambda j: -os.path.getsize(os.path.join(a.snapshots, f'{j[0]}.tgz'))
                  if os.path.exists(os.path.join(a.snapshots, f'{j[0]}.tgz')) else 0)
    print(f'{len(runs)} runs, {len(by_task)} tasks, agents {[x["label"] for x in agents]}, workers {a.workers}',
          flush=True)
    with open(os.path.join(a.out, 'runs.jsonl'), 'w') as log:
        done = 0
        with concurrent.futures.ProcessPoolExecutor(a.workers) as ex:
            for task, recs, secs in ex.map(work_task, jobs):
                done += 1
                for rec in recs:
                    log.write(json.dumps(rec, ensure_ascii=False) + '\n')
                log.flush()
                flags = ' '.join(f"{r['agent']}:{r['diff']}" for r in recs)
                print(f'[{done}/{len(jobs)}] {task} {len(recs)} replays {secs:.0f}s  {flags}', flush=True)


# ------------------------------------------------------------------ report

def pct(a, b):
    return f'{a}/{b} ({100 * a / b:.0f}%)' if b else f'{a}/0'


def summarize(out_dir, bases=None):
    recs = [json.loads(l) for l in open(os.path.join(out_dir, 'runs.jsonl')) if l.strip()]
    meta = json.load(open(os.path.join(out_dir, 'meta.json'))) if os.path.exists(os.path.join(out_dir,
                                                                                               'meta.json')) else {}
    bases = bases or meta.get('bases', {})
    rec_base = meta.get('recorded_bases', {})
    lines = []
    p = lines.append
    agents = list(dict.fromkeys(r['agent'] for r in recs))
    p(f'replay of {len({r["run"] for r in recs})} runs ({sum(1 for r in recs if r["agent"] == agents[0] and r["resolved"])} '
      f'passing) through {", ".join(agents)}; reference agent: {agents[0]}')
    p(f'context fit: prompt ~ base + {K_CHARS}*chars + {K_CALLS}*calls; bases {bases}; recorded {rec_base}')
    flagged = []
    for ag in agents:
        rs = [r for r in recs if r['agent'] == ag]
        calls = [c for r in rs for c in r['calls']]
        p('')
        p(f'=== {ag}: {len(rs)} runs, {len(calls)} calls'
          + (f', {sum(1 for r in rs if r.get("error"))} replay errors' if any(r.get('error') for r in rs) else ''))
        script_calls = [c for c in calls if c['tool'] == 'run_skill_script']
        p(f'calls: run {sum(c["new_kind"] == "ran" for c in script_calls)}, ADK-rejected (skipped) '
          f'{sum(c["new_kind"] == "rejected" for c in script_calls)} (recorded {sum(c["rec_kind"] == "rejected" for c in script_calls)}), '
          f'newly accepted {sum(bool(c.get("newly_accepted")) for c in script_calls)}, non-script '
          f'{sum(c["tool"] != "run_skill_script" for c in calls)}, no recorded result '
          f'{sum(c["rec_kind"] == "noresult" for c in calls)}, harness-killed {sum(c["rec_kind"] == "harness_timeout" for c in calls)}')
        # 1. edit fidelity
        eds = [c for c in calls if c.get('edit')]
        was = [c for c in eds if c['edit']['rec'] == 'applied']
        v = {}
        for c in was:
            v[c['edit'].get('verdict')] = v.get(c['edit'].get('verdict'), 0) + 1
        p(f'edits applied in the recording: {len(was)}; still applied, same range {v.get("same", 0)}; '
          f'changed range {v.get("changed_range", 0)}; NOT APPLIED {v.get("NOT_APPLIED", 0)}; newly applied '
          f'{sum(1 for c in eds if c["edit"].get("verdict") == "newly_applied")} (of {len(eds) - len(was)} '
          f'recorded not applied)')
        for label, sub in (('passing', [r for r in rs if r['resolved']]), ('failing', [r for r in rs if not r['resolved']])):
            if not sub:
                continue
            d = {k: sum(r['diff'] == k for r in sub) for k in ('identical', 'different', 'empty', 'error')}
            p(f'final diff vs recorded patch, {label} runs ({len(sub)}): identical {d["identical"]}, different '
              f'{d["different"]}, empty {d["empty"]}' + (f', error {d["error"]}' if d['error'] else '')
              + (f'; vs {agents[0]} replay: identical {sum(r.get("diff_vs_ref") == "identical" for r in sub)}, '
                 f'different {sum(r.get("diff_vs_ref") == "different" for r in sub)}' if ag != agents[0] else ''))
        # 2. health
        crashes = [c for c in calls if c.get('crash')]
        tos = [c for c in calls if c.get('timeout')]
        by = {}
        for c in crashes:
            by[c['script']] = by.get(c['script'], 0) + 1
        p(f'script health: crashes {len(crashes)} {by or ""}, timeouts {len(tos)} '
          f'{sorted({c["script"] for c in tos}) or ""}')
        runpy_ = [c for c in calls if c.get('run_agree') is not None]
        p(f'run.py (env-dependent, excluded from the verdict): exit-code line agrees with the recording '
          f'{pct(sum(c["run_agree"] for c in runpy_), len(runpy_))}')
        both = [c for c in calls if 'rec_refusal' in c]
        p('flow refusals (calls run in both): ' + ', '.join(
            f'{k} {sum(c["rec_refusal"] == k for c in both)} -> {sum(c["refusal"] == k for c in both)}'
            for k in ('NOT RUN', 'ALREADY DONE')))
        same = [c for c in calls if 'same_stdout' in c and c['script'] != 'run.py']
        p(f'stdout identical to the recording (non-run.py script calls): {pct(sum(c["same_stdout"] for c in same), len(same))}'
          f', ignoring the flow clock {pct(sum(c.get("same_mod_clock", False) for c in same), len(same))}')
        # 3. context cost
        per = {}
        for c in calls:
            if c['tool'] != 'run_skill_script' or c['new_kind'] != 'ran' or c['rec_kind'] != 'ran':
                continue
            per.setdefault(c['script'], []).append((c['rec_chars'], c['new_chars']))
        tot_o = sum(c['rec_chars'] + c['arg_chars'] for c in calls)
        tot_n = sum(c['new_chars'] + c['arg_chars'] for c in calls)
        p(f'context, JSON-escaped chars per run (results + call arguments, as the model sees them): recorded {tot_o / max(1, len(rs)):.0f} -> replayed {tot_n / max(1, len(rs)):.0f} '
          f'({100 * (tot_n - tot_o) / max(1, tot_o):+.0f}%)')
        p('  per script (calls run in both): ' + ', '.join(
            f'{s} {sum(o for o, _ in x) / len(x):.0f}->{sum(n for _, n in x) / len(x):.0f}'
            for s, x in sorted(per.items(), key=lambda kv: -len(kv[1]))))
        b_new = bases.get(ag, BASE_V17B)
        cross_o, cross_n = [], []
        for r in rs:
            exp = r['exp']
            co = est_crossing([c['rec_raw'] + c['argv_chars'] if 'rec_raw' in c else None for c in r['calls']],
                              rec_base.get(exp, BASE_V17B))
            cn = est_crossing([c['new_raw'] + c['argv_chars'] if 'new_raw' in c else None for c in r['calls']], b_new)
            r['cross_rec'], r['cross_new'] = co, cn
            cross_o.append(co)
            cross_n.append(cn)
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
        no, nn = [x for x in cross_o if x], [x for x in cross_n if x]
        p(f'14,336-token crossing (estimate): recorded {len(no)}/{len(rs)} runs (median model call {med(no)}), '
          f'replayed {len(nn)}/{len(rs)} (median {med(nn)}); later/never in {sum(1 for o, n in zip(cross_o, cross_n) if o and (not n or n > o))}, '
          f'earlier in {sum(1 for o, n in zip(cross_o, cross_n) if n and (not o or n < o))}')
        acts = [r.get('actual_cross') for r in rs]
        if acts:
            p(f'  recorded actual prompt_tokens >= 14,336: {sum(1 for x in acts if x)}/{len(acts)} runs (median call '
              f'{med([x for x in acts if x])})')
        for r in rs:
            if r['resolved'] and r['diff'] != 'identical':
                flagged.append((ag, r))
    # flagged runs
    p('')
    p('=== passing runs whose final diff is not the recorded one')
    for ag, r in flagged:
        lines.extend(explain(ag, r))
    if not flagged:
        p('(none)')
    return '\n'.join(lines)


CAUSES = {'newly_applied': 'cascade: applies an edit the recording refused, so the recorded retry applies again',
          'NOT_APPLIED': 'refused an edit the recording applied', 'changed_range': 'applied at another range',
          'applied_no_record': 'applied an edit whose recorded result is missing'}


def cause(r):
    """The first per-call difference that explains a changed final diff."""
    for c in r['calls']:
        v = (c.get('edit') or {}).get('verdict')
        if v in CAUSES:
            return f'call {c["i"]} {c["script"]}: {CAUSES[v]}'
        if c.get('crash') or c.get('timeout'):
            return f'call {c["i"]} {c["script"]}: {"crash" if c.get("crash") else "timeout"}'
        if c['new_kind'] == 'rejected' and c['rec_kind'] != 'rejected':
            return f'call {c["i"]} {c["script"]}: now rejected by ADK'
        if c.get('newly_accepted') and c['script'] not in READ_ONLY:
            return f'call {c["i"]} {c["script"]}: newly accepted'
    if r.get('first_divergence'):
        return f'call {r["first_divergence"]}: the workspace first differs here (env-dependent run.py?)'
    return 'unknown (no per-call difference; env-dependent run.py or the recorded patch itself)'


def explain(ag, r):
    out = [f'- [{ag}] {r["run"]}: diff {r["diff"]}; cause: {cause(r)}' + (f' (vs reference: {r.get("diff_vs_ref")}, first divergent '
                                                         f'call {r.get("first_divergence")})' if 'diff_vs_ref' in r else '')
           + (f' ERROR {r["error"]}' if r.get('error') else '')]
    for c in r['calls']:
        e = c.get('edit')
        if e and e.get('verdict') in ('NOT_APPLIED', 'changed_range', 'newly_applied'):
            out.append(f'    call {c["i"]} {c["script"]} {e["verdict"]}: recorded `{e["rec_line"][:110]}` -> '
                       f'`{e["new_line"][:110]}`')
        elif c.get('crash'):
            out.append(f'    call {c["i"]} {c["script"]} CRASH: {c["crash"][-160:]!r}')
        elif c.get('timeout'):
            out.append(f'    call {c["i"]} {c["script"]} TIMEOUT')
        elif c['new_kind'] == 'rejected' and c['rec_kind'] != 'rejected':
            out.append(f'    call {c["i"]} {c["script"]} now ADK-rejected ({c.get("error_code")})')
        elif c.get('newly_accepted'):
            out.append(f'    call {c["i"]} {c["script"]} newly accepted (rejected in the recording)')
        elif c['rec_kind'] == 'noresult':
            out.append(f'    call {c["i"]} {c["script"]} had no recorded result (session timeout during the call)')
    return out


# ------------------------------------------------------------------ host side

def materialize(spec, dest):
    """LABEL=SPEC or SPEC (git ref, WORKTREE or directory) -> (label, dir under dest)."""
    label, _, s = spec.partition('=') if '=' in spec else (None, '', spec)
    label = label or os.path.basename(s.rstrip('/')) or s
    if os.path.isdir(s) and label in ('_agent', 'agent'):
        label = os.path.basename(os.path.dirname(os.path.abspath(s))) + '_' + label.strip('_')
    out = os.path.join(dest, '_agents', label)
    subprocess.run(['bash', os.path.join(ROOT, 'scripts/materialize.sh'), s, out], cwd=ROOT, check=True,
                   capture_output=True)
    return label, out


def agent_record(label, d):
    skill, scripts = skill_info(d)
    return {'label': label, 'dir': d, 'skill': skill, 'scripts': scripts}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs', help='results dirs (comma-separated), e.g. results/E041_v17b_eval113')
    ap.add_argument('--agent', action='append', default=[], help='git ref | WORKTREE | dir, optionally LABEL=SPEC')
    ap.add_argument('--out', help='output dir (e.g. results/R001_replay)')
    ap.add_argument('--passing', action='store_true', help='only runs that resolved')
    ap.add_argument('--tasks', default=None, help='comma-separated task ids')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--env', default='task', choices=('task', 'base', 'none'),
                    help='python env for run.py/import guards: base site-packages + per-task env (default), base, none')
    ap.add_argument('--host', action='store_true', help='run here instead of in the sandbox image')
    ap.add_argument('--base', action='append', default=[], help='LABEL=TOKENS context base override')
    ap.add_argument('--report', default=None, help='only print the summary of an existing output dir')
    # internal (inside the container)
    ap.add_argument('--inside', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--agent-dirs', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--snapshots', default=os.path.join(ROOT, 'data/competition/snapshots'))
    ap.add_argument('--manifest', default=os.path.join(ROOT, 'data/competition/wheels_local/task_env.json'))
    ap.add_argument('--spcache', default=SP_CACHE)
    ap.add_argument('--workroot', default=None)
    a = ap.parse_args()

    if a.report:
        print(summarize(a.report))
        return
    if a.inside:
        agents = [agent_record(*x.split('=', 1)) for x in a.agent_dirs.split(',')]
        run_inside(a, agents)
        return
    if not (a.runs and a.agent and a.out):
        ap.error('--runs, --agent and --out are required')
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    agents = [materialize(s, out) for s in a.agent]
    bases = {label: flow_base(d, skill_info(d)[1]) for label, d in agents}
    for b in a.base:
        k, v = b.split('=')
        bases[k] = int(v)
    rec_bases = {}
    for d in a.runs.split(','):
        ad = os.path.join(d.rstrip('/'), '_agent')
        rec_bases[os.path.basename(d.rstrip('/'))] = flow_base(ad, skill_info(ad)[1]) if os.path.isdir(ad) \
            else BASE_V17B
    json.dump({'agents': {l: d for l, d in agents}, 'runs': a.runs, 'bases': bases, 'recorded_bases': rec_bases,
               'env': a.env, 'passing': a.passing}, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    passthru = ['--runs', a.runs, '--workers', str(a.workers), '--env', a.env]
    passthru += (['--passing'] if a.passing else []) + (['--tasks', a.tasks] if a.tasks else []) + \
        (['--limit', str(a.limit)] if a.limit else [])
    t0 = time.time()
    if a.host:
        cmd = [sys.executable, os.path.abspath(__file__), '--inside', '--out', out, '--agent-dirs',
               ','.join(f'{l}={d}' for l, d in agents), '--spcache', a.spcache] + passthru
        subprocess.run(cmd, cwd=ROOT, check=True)
    else:
        if not out.startswith(ROOT + os.sep):
            raise SystemExit('--out must be inside the repository (it is mounted into the container)')
        rel = lambda p: '/repo/' + os.path.relpath(p, ROOT)  # noqa: E731
        runs_in = ','.join(rel(os.path.abspath(d)) for d in a.runs.split(','))
        passthru[1] = runs_in
        inner = ['python', '/repo/scripts/replay_tools.py', '--inside', '--out', '/out', '--agent-dirs',
                 ','.join(f'{l}=/out/{os.path.relpath(d, out)}' for l, d in agents),
                 '--snapshots', '/repo/data/competition/snapshots',
                 '--manifest', '/repo/data/competition/wheels_local/task_env.json', '--spcache', '/spcache',
                 '--workroot', '/tmp'] + passthru
        setup = ''
        if a.env in ('task', 'base') and os.path.exists(os.path.join(a.spcache, 'sp_base.tar')):
            setup = f'tar --no-same-owner -xf /spcache/sp_base.tar -C {SP_DIR} && '
        cmd = ['docker', 'run', '--rm', '--platform', 'linux/amd64', '--network', 'none',
               '-v', f'{ROOT}:/repo:ro', '-v', f'{out}:/out', '-v', f'{a.spcache}:/spcache:ro', '-w', '/tmp',
               '-e', 'PYTHONDONTWRITEBYTECODE=1', IMAGE, 'bash', '-c',
               setup + ' '.join(__import__('shlex').quote(x) for x in inner)]
        subprocess.run(cmd, check=True)
    print(f'replay done in {time.time() - t0:.0f}s', flush=True)
    text = summarize(out)
    open(os.path.join(out, 'summary.txt'), 'w').write(text + '\n')
    print(text)
    grade_tasks(out)


def grade_tasks(out):
    """tasks.jsonl batches (one instance per task each) whose `patch` is the replayed final patch of every passing run
    with a changed diff, to grade with the hidden tests: scripts/verify_gold.py --tasks <file> ..."""
    tasks = {}
    for line in open(os.path.join(ROOT, 'data/competition/tasks.jsonl')):
        d = json.loads(line)
        tasks[d['instance_id']] = d
    recs = [json.loads(l) for l in open(os.path.join(out, 'runs.jsonl')) if l.strip()]
    for ag in dict.fromkeys(r['agent'] for r in recs):
        batches = []
        for r in recs:
            if r['agent'] != ag or not r['resolved'] or r['diff'] not in ('different', 'empty') or r['task'] not in tasks:
                continue
            b = next((b for b in batches if r['task'] not in {t['instance_id'] for t, _ in b}), None)
            if b is None:
                batches.append(b := [])
            b.append((dict(tasks[r['task']], patch=r['final_patch']), r['run']))
        for k, b in enumerate(batches):
            path = os.path.join(out, f'grade_{ag}_{k}.jsonl')
            with open(path, 'w') as f:
                f.writelines(json.dumps(t) + '\n' for t, _ in b)
            print(f'grade {ag} batch {k} ({", ".join(run for _, run in b)}):\n  .venv/bin/python scripts/verify_gold.py '
                  f'--tasks {os.path.relpath(path, ROOT)} --snapshots-dir data/competition/snapshots --results-dir '
                  f'{os.path.relpath(os.path.join(out, f"grade_{ag}_{k}"), ROOT)} --submission-dir '
                  f'{os.path.relpath(os.path.join(out, "_agents", ag), ROOT)} --sandbox docker --concurrency 2')


if __name__ == '__main__':
    main()
