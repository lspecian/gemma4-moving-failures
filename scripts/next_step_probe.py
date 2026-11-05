"""Next-step probe: does an agent variant make the model EDIT (or read on) at the same pre-edit decision points?

Full Kaggle-speed runs take hours (E044: v18 edited in 15/56 runs vs v17b 23/56). This probe predicts such behaviour
shifts in minutes: it takes recorded decision points before the first edit, rebuilds each one as the variant would
have shown it, samples only the NEXT model turn N times and classifies the action.

States (select_states): from recorded traces, every point after the first call (k = calls in the prefix: k=1 is
"after locate") up to and including the call that was the run's first edit attempt; prefixes with an edit/submit are
never used, and the prompt must stay below --max-prompt tokens (14,000: no compaction in the prefix; Kaggle compacts
from 14,336 -- compaction states are future work). --max-states picks a subset balanced over k buckets and
recordings, at most --per-task states per task (then 2x when buckets run dry).

Per variant (--agent DIR, repeatable; a materialized agent dir such as results/E043_v17b_kaggleslow/_agent):
  - system prompt = its prompts/system.md with {problem_description} + ADK's agent suffix (its agent.yaml name and
    description); the harness's user message as recorded; tool schemas from data/probe/adk_request.json;
  - tool OUTPUTS as the variant's own scripts print them: the recorded prefix calls are re-executed in a sandbox
    container of the task (swebench-sandbox image, snapshot, per-task env as in TASK_ENV=1 runs) with the variant's
    scripts, a fresh flow state and the recorded clock (time.time shifted per call, so BUDGET/elapsed footers show
    the recording's timeline). ADK's run_skill_script validation is emulated (SKILL_NOT_FOUND, SCRIPT_NOT_FOUND,
    INVALID_ARGUMENTS), so e.g. repro.py exists for v18 and not for v17b. Other tools keep their recorded result.
  - past thinking: the recording's own text as `reasoning` (adk_submission 0.2.12 sends it), i.e. produced under the
    recorded version's prompt and outputs. With --recorded-outputs the recorded tool results are used (no sandbox).
Sampling: vLLM chat endpoint (--server, Kaggle's June template on the rented server) with the variant's
configs/sampling.yaml (default temperature 0.2, top_p 0.95, thinking_token_budget 512, max tokens 1536), seeds
1000+s (paired across variants). Each sample is classified like ADK would see it (classify_response): edit (edit.py /
create.py of a real file, valid shape), edit_badshape, read (show/outline/callers/tests_for/history), search, locate,
snippet / test (run.py and the repro aliases; create.py of a scratch file), check (check_patch), submit, other,
malformed (ADK rejects it: bracketed or delimiter-mangled names, file_path/skill_name nested into args, unknown
script or tool, unparsable arguments, a raw call the server's parser did not extract), no_call (a text-only answer:
ADK ends the run). Completion tokens, thinking chars and the prompt tokens are recorded.

    .venv/bin/python scripts/next_step_probe.py --out results/N001 \
        --agent results/E043_v17b_kaggleslow/_agent --agent results/E044_v18_kaggleslow/_agent \
        --agent results/E045_v18tools_v17prompt/_agent --workers 16     # free server; default 2 when shared
    .venv/bin/python scripts/next_step_probe.py --out results/N001 --report     # report from samples.jsonl

Validation (N001, 2026-10-04: 60 states x 4 samples, edit share 0.25; v17b vs v18 vs v18 tools + v17b prompt): replays
with a recording's own scripts reproduced 283/285 recorded outputs (clock +-1 s); P(edit) 0.150 / 0.113 / 0.117
(E044's direction: v18 edits less; v18-tools+v17b-prompt = v18, so the tools/outputs, not the prompt cut, drive it).
All the signal sat at the states where the recording edited next (0.57 / 0.42 / 0.43; elsewhere P(edit) = 0.017 for
all three), hence the defaults: --edit-share 0.6, --max-states 80, --n 3. Throughput on the rented GPU: ~10
requests/min with --workers 16 (KV cache ~90% full); replays ~7 s per task (cached in --out/replays).

Outputs in --out: states.json, replays/<variant>/<trace>.json (cached sandbox outputs), samples.jsonl (one line per
sample; reruns skip samples already there), report.json + the printed report: per variant P(edit) overall and by k,
P(malformed), P(no_call), the action mix, tokens; paired comparisons on the same states (mean difference of the
per-state P(edit), bootstrap 95% CI, sign test).
"""

import argparse
import concurrent.futures
import glob
import hashlib
import json
import math
import os
import pathlib
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

DEFAULT_TRACES = ('results/E043_v17b_kaggleslow', 'results/E042_v17b_kagglelike', 'results/E044_v18_kaggleslow')
MODEL = 'gemma-4-31b-it-qat-w4a16-ct'
SKILL = 'repo-tools'

# ---------------------------------------------------------------- classification (pure: tests/test_next_step_probe.py)

READ_SCRIPTS = {'show.py', 'outline.py', 'callers.py', 'tests_for.py', 'history.py'}
RUN_ALIASES = {'repro.py', 'reproduce_issue.py', 'reproduce_bug.py'}
OTHER_TOOLS = {'get_status', 'list_skills', 'load_skill', 'load_skill_resource'}
SCRIPT_NAME = re.compile(r'(?:scripts/)?([A-Za-z0-9_\-]+\.py)')
SCRATCH = re.compile(r'(^|/)(repro|reproduce|scratch|debug|tmp|temp|test_repro|check|try|demo)[\w\-]*\.py$')
EDIT_GROUP = ('edit', 'edit_badshape')
EXPLORE = ('read', 'search', 'locate')
RUN = ('snippet', 'test')


def adk_error(fargs, scripts):
    """The error run_skill_script returns before running anything (ADK skill_toolset.RunSkillScriptTool.run_async), or
    None when the script would run. `scripts`: the variant's script file names."""
    if not isinstance(fargs, dict):
        return {'error': "Argument 'skill_name' is required.\nArgument 'file_path' is required.",
                'error_code': 'INVALID_ARGUMENTS'}
    skill, fp = fargs.get('skill_name'), fargs.get('file_path')
    sa, so, pa = fargs.get('args'), fargs.get('short_options'), fargs.get('positional_args')
    errors = []
    if not skill:
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
    if skill != SKILL:
        return {'error': f"Skill '{skill}' not found.", 'error_code': 'SKILL_NOT_FOUND'}
    name = str(fp)[len('scripts/'):] if str(fp).startswith('scripts/') else str(fp)
    if name not in scripts:
        return {'error': f"Script '{fp}' not found in skill '{skill}'.", 'error_code': 'SCRIPT_NOT_FOUND'}
    return None


def _intent(fargs):
    """The script a malformed call meant (e.g. 'edit.py' from 'edit.py`,skill_name:' or a nested file_path)."""
    if not isinstance(fargs, dict):
        return None
    for v in (fargs.get('file_path'), (fargs.get('args') or {}).get('file_path') if isinstance(fargs.get('args'), dict)
              else None):
        m = SCRIPT_NAME.search(str(v or ''))
        if m:
            return m.group(1)
    return None


def _script_args(fargs):
    sa = fargs.get('args')
    pos = list(sa) if isinstance(sa, list) else list(fargs.get('positional_args') or [])
    return (sa if isinstance(sa, dict) else {}), pos


def is_scratch(target):
    t = str(target or '').removeprefix('/workspace/').lstrip('./')
    return bool(SCRATCH.search(t)) or ('/' not in t and t.endswith('.py') and not t.startswith('setup'))


def script_label(script, fargs):
    """Action label of a run_skill_script call that ADK would run."""
    sa, pos = _script_args(fargs)
    if script == 'edit.py':
        ok = ('new' in sa and any(k in sa for k in ('old', 'start', 'line', 'lines'))) or len(pos) >= 2 \
            or (sa.get('new') is not None and len(pos) >= 1)
        return 'edit' if ok else 'edit_badshape'
    if script == 'create.py':
        target = sa.get('target') or sa.get('path') or sa.get('file') or (pos[0] if pos else None)
        content = sa.get('content') if 'content' in sa else (pos[1] if len(pos) > 1 else None)
        if target and is_scratch(target) or not target and content is not None:
            return 'snippet'   # a scratch/repro file (v17b refuses it, v18 runs it as a snippet): a run attempt
        return 'edit' if target and content is not None else 'edit_badshape'
    if script in RUN_ALIASES:
        return 'snippet'
    if script == 'run.py':
        return 'test' if any(k in sa for k in ('test', 'tests', 'k')) else 'snippet'
    if script in READ_SCRIPTS:
        return 'read'
    if script == 'search.py':
        return 'search'
    if script == 'locate.py':
        return 'locate'
    if script == 'check_patch.py':
        return 'check'
    return 'other'   # status.py, hypothesis.py, kb.py, ...


def classify_call(name, fargs, scripts):
    """(label, intent) of one function call as ADK would handle it; scripts: the variant's script file names."""
    if name == 'submit_patch':
        return 'submit', None
    if name in OTHER_TOOLS:
        return 'other', None
    if name != 'run_skill_script':
        m = SCRIPT_NAME.search(str(name))
        return 'malformed', (m.group(1) if m else None)
    if adk_error(fargs, scripts):
        return 'malformed', _intent(fargs)
    fp = str(fargs['file_path'])
    script = fp[len('scripts/'):] if fp.startswith('scripts/') else fp
    return script_label(script, fargs), script


def classify_response(msg, scripts):
    """(label, intent, info) of a sampled chat-completion message (the server's own tool parser already ran)."""
    calls = msg.get('tool_calls') or []
    content = msg.get('content') or ''
    if not calls:
        if '<|tool_call>' in content or re.search(r'\bcall:\s*\w+\{', content):
            return 'malformed', None, {'why': 'raw call not parsed'}
        return 'no_call', None, {}
    f = calls[0].get('function') or {}
    raw = f.get('arguments')
    try:
        fargs = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except ValueError:
        return 'malformed', None, {'why': 'arguments not JSON', 'name': f.get('name')}
    label, intent = classify_call(f.get('name'), fargs, scripts)
    return label, intent, {'name': f.get('name'), 'args': fargs, 'n_calls': len(calls)}


# ---------------------------------------------------------------- states (pure)

def agent_calls(steps):
    """[(step index, step)] of the agent steps that made tool calls."""
    return [(i, s) for i, s in enumerate(steps) if s.get('source') == 'agent' and s.get('tool_calls')]


def prompt_tokens(step):
    try:
        return int((step.get('metrics') or {}).get('prompt_tokens') or 0)
    except (TypeError, ValueError):
        return 0


def recorded_label(step, scripts):
    tc = step['tool_calls'][0]
    return classify_call(tc.get('function_name'), tc.get('arguments') or {}, scripts)


def select_states(steps, scripts, max_prompt=14000):
    """Decision points before the first edit: [{'k', 'step', 'next', 'prompt_tokens'}]. k = calls in the prefix (k=1:
    after the first call, normally locate.py); 'step' is the agent step whose call is resampled; 'next' the recorded
    label of that call. Stops at the first edit attempt (included as the state before it: its prefix has none), at a
    submit, and once any prompt so far reached max_prompt (a compaction may follow)."""
    calls = agent_calls(steps)
    out, peak = [], 0
    for c, (i, s) in enumerate(calls):
        peak = max(peak, prompt_tokens(s))
        if peak >= max_prompt:
            break
        label, intent = recorded_label(s, scripts)
        if c >= 1:
            out.append({'k': c, 'step': i, 'next': label, 'prompt_tokens': prompt_tokens(s)})
        if label in EDIT_GROUP or label == 'submit' or intent in ('edit.py', 'create.py') and label == 'malformed':
            break
    return out


def k_bucket(k):
    return str(k) if k <= 4 else ('5-6' if k <= 6 else '7+')


def _round_robin(cands, n, bucket, per, per_task, picked, rng):
    """Add up to n of cands to picked, round-robin over bucket(c) and over recordings within a bucket, at most per_task
    states per task in a first pass, then 2*per_task. per: task -> count so far (shared across calls)."""
    buckets = {}
    for c in cands:
        buckets.setdefault(bucket(c), {}).setdefault(c['source'], []).append(c)
    for srcs in buckets.values():
        for lst in srcs.values():
            rng.shuffle(lst)
    order = sorted(buckets, key=lambda b: (len(b) > 1, b))   # '1'..'4' then '5-6', '7+'
    seen = {(c['trace'], c['step']) for c in picked}
    added = 0
    for cap in (per_task, 2 * per_task):
        progress = True
        while added < n and progress:
            progress = False
            for b in order:
                if added >= n:
                    break
                srcs = buckets[b]
                for src in sorted(srcs, key=lambda s: (sum(1 for p in picked if p['source'] == s), s)):
                    lst = srcs[src]
                    j = next((i for i, c in enumerate(lst) if per.get(c['task'], 0) < cap
                              and (c['trace'], c['step']) not in seen), None)
                    if j is None:
                        continue
                    c = lst.pop(j)
                    seen.add((c['trace'], c['step']))
                    per[c['task']] = per.get(c['task'], 0) + 1
                    picked.append(c)
                    added += 1
                    progress = True
                    break
    return picked


def select_balanced(cands, max_states, per_task=1, seed=0, edit_share=0.0):
    """Pick up to max_states states: round(edit_share * max_states) of them among the states whose recorded next call
    was an edit attempt (the decision points where the recording did edit), the rest round-robin over k buckets; both
    balanced over recordings, at most per_task per task (then 2x). cands: dicts with 'task', 'source', 'k', 'next',
    'trace', 'step'."""
    rng = random.Random(seed)
    picked, per = [], {}
    n_edit = round(max_states * edit_share)
    if n_edit:
        _round_robin([c for c in cands if c['next'] in EDIT_GROUP], n_edit, lambda c: 'edit', per, per_task, picked,
                     rng)
    _round_robin([c for c in cands if c['next'] not in EDIT_GROUP], max_states - len(picked),
                 lambda c: k_bucket(c['k']), per, per_task, picked, rng)
    return picked


# ---------------------------------------------------------------- statistics (pure)

def sign_test(wins, losses):
    """Two-sided exact binomial p for wins vs losses (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def bootstrap_ci(diffs, reps=2000, seed=0):
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


# ---------------------------------------------------------------- variants

class Variant:
    def __init__(self, agent):
        self.dir = pathlib.Path(agent).resolve()
        self.prompt = (self.dir / 'prompts/system.md').read_text()
        y = (self.dir / 'agent.yaml').read_text()
        self.name = re.search(r'^name:\s*(.+)$', y, re.M).group(1).strip()
        self.description = re.search(r'^description:\s*(.+)$', y, re.M).group(1).strip()
        self.scripts_dir = self.dir / 'skills' / SKILL / 'scripts'
        self.scripts = {p.name for p in self.scripts_dir.glob('*.py')}
        self.sampling = {'temperature': 0.2, 'top_p': 0.95, 'max_output_tokens': 1536, 'thinking_budget': 512}
        sp = self.dir / 'configs/sampling.yaml'
        if sp.exists():
            t = sp.read_text()
            for k in self.sampling:
                m = re.search(rf'^\s*{k}:\s*([\d.]+)\s*$', t, re.M)
                if m:
                    self.sampling[k] = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))
        h = hashlib.sha1()
        for p in sorted(self.scripts_dir.glob('*.py')):
            h.update(p.name.encode() + p.read_bytes())
        self.scripts_hash = h.hexdigest()[:10]
        label = self.dir.name if self.dir.name != '_agent' else self.dir.parent.name
        self.id = label
        self.label = (self.dir.with_suffix('.version').read_text().strip()
                      if self.dir.with_suffix('.version').exists() else label)

    def system(self, problem, suffix):
        s = suffix.replace('"swe_agent_v10"', f'"{self.name}"', 1)
        s = re.sub(r'The description about you is ".*?"\.', lambda _: f'The description about you is '
                   f'"{self.description}".', s, count=1)
        return self.prompt.replace('{problem_description}', problem.strip()) + s


# ---------------------------------------------------------------- sandbox replay

DRIVER = r'''
import json, os, subprocess, sys, tempfile, time, uuid
job = json.load(sys.stdin)
subprocess.run('cd /workspace && git reset -q --hard && git clean -fdq', shell=True)
state = tempfile.mkdtemp(prefix='rt_')
env = dict(os.environ, REPO_TOOLS_DIR=state)
out = []
for c in job['calls']:
    files = {}
    for n in os.listdir(job['scripts']):
        if n.endswith('.py'):
            files['scripts/' + n] = open(os.path.join(job['scripts'], n)).read()
    fp = c['file_path'] if c['file_path'].startswith('scripts/') else 'scripts/' + c['file_path']
    argv = [fp]
    sa = c.get('args')
    if isinstance(sa, list):
        argv += [str(v) for v in sa]
    else:
        for k, v in (sa or {}).items():
            argv += ['--' + str(k), str(v)]
        for k, v in (c.get('short_options') or {}).items():
            argv += ['-' + str(k), str(v)]
        if c.get('positional_args'):
            argv += ['--'] + [str(v) for v in c['positional_args']]
    off = job['base'] + c['t'] - time.time()
    code = '\n'.join([
        'import time as _tm', '_rt = _tm.time', f'_tm.time = lambda: _rt() + {off!r}',
        'import os', 'import tempfile', 'import sys', 'import runpy', f'_files = {files!r}',
        'def _materialize_and_run():', '  _orig_cwd = os.getcwd()',
        '  with tempfile.TemporaryDirectory() as td:',
        '    for rel_path, content in _files.items():',
        '      full_path = os.path.join(td, rel_path)',
        '      os.makedirs(os.path.dirname(full_path), exist_ok=True)',
        '      with open(full_path, "w") as f:', '        f.write(content)',
        '    os.chdir(td)', '    try:',
        f'      sys.argv = {argv!r}', '      try:', f'        runpy.run_path({fp!r}, run_name="__main__")',
        '      except SystemExit as e:', '        if e.code is not None and e.code != 0:', '          raise e',
        '    finally:', '      os.chdir(_orig_cwd)', '_materialize_and_run()'])
    path = f'/workspace/.adk_exec_{uuid.uuid4().hex[:8]}.py'
    open(path, 'w').write(code)
    t0 = time.time()
    try:
        r = subprocess.run(['python3', path], cwd='/workspace', env=env, capture_output=True, text=True,
                           timeout=job.get('timeout', 300))
        so, se, rc = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        so, se, rc = (e.stdout or b'').decode(errors='replace') if isinstance(e.stdout, bytes) else (e.stdout or ''), '', 124
    os.unlink(path)
    if not se and rc != 0:
        se = f'Command failed with status=error, exit_code={rc}'
    status = 'error' if se and not so else ('warning' if se else 'success')
    out.append({'skill_name': c['skill_name'], 'file_path': c['file_path'], 'stdout': so, 'stderr': se,
                'status': status, '_secs': round(time.time() - t0, 2)})
print(json.dumps(out))
'''


class TaskSandbox:
    """One swebench-sandbox container per task, set up like the agent's Container A (snapshot, editable install,
    task env, baseline commit); replays run serially in it (workspace reset + fresh flow state each time)."""

    def __init__(self, task_id):
        import task_env
        from swegemma.harness import agent_runner as ar
        from swegemma.sandbox.docker import ContainerConfig, ContainerManager
        self.task_id = task_id
        self.lock = threading.Lock()
        self.dm = ContainerManager(ContainerConfig(image='swebench-sandbox:latest'))
        self.cid = self.dm.start()
        cfg = argparse.Namespace(tasks_path=ROOT / 'data/competition/tasks.jsonl',
                                 snapshots_dir=ROOT / 'data/competition/snapshots', submission_dir=None,
                                 wheels_dir=None, sandbox='docker')
        repo = task_repo(task_id)
        snap = next(iter(glob.glob(str(ROOT / f'data/competition/snapshots/{task_id}.t*gz'))))
        ar.setup_container_wheels(self.dm, self.cid, cfg)
        ar.extract_snapshot(self.dm, self.cid, pathlib.Path(snap))
        ar.setup_git_exclude(self.dm, self.cid)
        ar.install_editable_package(self.dm, self.cid)
        ar.install_test_dependencies(self.dm, self.cid, repo, config=cfg)
        if os.environ.get('SWEGEMMA_TASK_ENV', '1') != '0':
            task_env.apply_task_env(self.dm, self.cid, task_id)
        ar.setup_workspace_test_config(self.dm, self.cid, repo=repo)
        ar.setup_baseline_commit(self.dm, self.cid, 'baseline')
        self.copied = set()
        self._put_text(DRIVER, '/probe/driver.py')

    def _put_text(self, text, dest):
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(text)
        subprocess.run(['docker', 'exec', self.cid, 'mkdir', '-p', os.path.dirname(dest)], check=True)
        subprocess.run(['docker', 'cp', f.name, f'{self.cid}:{dest}'], check=True, capture_output=True)
        os.unlink(f.name)

    def replay(self, variant, calls, base, timeout=300):
        dest = f'/probe/{variant.scripts_hash}'
        with self.lock:
            if dest not in self.copied:
                subprocess.run(['docker', 'exec', self.cid, 'mkdir', '-p', dest], check=True)
                subprocess.run(['docker', 'cp', f'{variant.scripts_dir}/.', f'{self.cid}:{dest}/'], check=True,
                               capture_output=True)
                self.copied.add(dest)
            job = {'scripts': dest, 'calls': calls, 'base': base, 'timeout': timeout}
            r = subprocess.run(['docker', 'exec', '-i', self.cid, 'python3', '/probe/driver.py'],
                               input=json.dumps(job), capture_output=True, text=True,
                               timeout=timeout * max(1, len(calls)) + 60)
            if r.returncode != 0:
                raise RuntimeError(f'driver failed: {r.stderr[-800:]}')
            return json.loads(r.stdout.strip().splitlines()[-1])

    def close(self):
        try:
            self.dm.stop(self.cid)
        except Exception:  # noqa: BLE001
            subprocess.run(['docker', 'rm', '-f', self.cid], capture_output=True)


CLOCK = re.compile(r'\d+m\d+s|\d+:\d\d/')
TASKS = {}


TASKS_LOCK = threading.Lock()


def _tasks():
    with TASKS_LOCK:   # replay threads call this concurrently: never hand out a half-loaded dict
        if not TASKS:
            loaded = {}
            for line in open(ROOT / 'data/competition/tasks.jsonl'):
                d = json.loads(line)
                loaded[d['instance_id']] = d
            TASKS.update(loaded)
    return TASKS


def task_of(trace_file):
    return os.path.basename(trace_file)[len('trace_'):-len('.json')]


def task_repo(task_id):
    return _tasks()[task_id]['repo']


def problem_statement(task_id):
    return _tasks()[task_id]['problem_statement']


def step_time(step):
    """When the step's tool started (the model event's timestamp); the replayed script's clock then advances in real
    time, so a slow call (a 15 s snippet timeout) ends as late as it did in the recording."""
    return float((step.get('extra') or {}).get('timestamp') or 0)


def replay_trace(trace, steps, upto_call, variant, sandbox, cache_dir):
    """Tool contents {f'{step}_{j}': content} for the first upto_call agent calls of a trace, as the variant prints
    them. Cached on disk per (variant scripts, trace, upto)."""
    cache = pathlib.Path(cache_dir) / variant.scripts_hash / f'{task_of(trace)}__{hashlib.sha1(trace.encode()).hexdigest()[:8]}.json'
    if cache.exists():
        d = json.loads(cache.read_text())
        if d['upto'] >= upto_call:
            return d['contents']
    calls = agent_calls(steps)[:upto_call]
    t0 = step_time(calls[0][1]) if calls else 0
    jobs, where, contents = [], [], {}
    for i, s in calls:
        for j, tc in enumerate(s['tool_calls']):
            key = f'{i}_{j}'
            name, fargs = tc.get('function_name'), tc.get('arguments') or {}
            if name == 'run_skill_script':
                err = adk_error(fargs, variant.scripts)
                if err:
                    contents[key] = json.dumps(err)
                    continue
                jobs.append({'skill_name': fargs['skill_name'], 'file_path': str(fargs['file_path']),
                             'args': fargs.get('args'), 'short_options': fargs.get('short_options'),
                             'positional_args': fargs.get('positional_args'), 't': step_time(s) - t0})
                where.append(key)
            else:   # other tools / unknown names: the recorded result (version-independent except load_skill)
                obs = s.get('observation') or {}
                contents[key] = obs.get('content', '') if isinstance(obs, dict) else ''
    if jobs:
        res = sandbox().replay(variant, jobs, base=1.8e9)
        for key, r in zip(where, res):
            r.pop('_secs', None)
            contents[key] = json.dumps(r)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({'trace': trace, 'upto': upto_call, 'contents': contents}))
    return contents


def recorded_contents(steps, upto_call):
    out = {}
    for i, s in agent_calls(steps)[:upto_call]:
        obs = s.get('observation') or {}
        for j, _ in enumerate(s['tool_calls']):
            out[f'{i}_{j}'] = obs.get('content', '') if isinstance(obs, dict) else ''
    return out


def build_messages(steps, upto_call, variant, task_id, suffix, contents):
    """The chat request messages (OpenAI wire format) before agent call number upto_call."""
    msgs = [{'role': 'system', 'content': variant.system(problem_statement(task_id), suffix)},
            {'role': 'user', 'content': steps[1]['message']}]
    for i, s in agent_calls(steps)[:upto_call]:
        calls, results = [], []
        for j, tc in enumerate(s['tool_calls']):
            cid = tc.get('tool_call_id') or f'call_{i}_{j}'
            calls.append({'id': cid, 'type': 'function', 'function': {
                'name': tc['function_name'], 'arguments': json.dumps(tc.get('arguments') or {}, ensure_ascii=False)}})
            results.append({'role': 'tool', 'tool_call_id': cid, 'content': contents[f'{i}_{j}']})
        m = {'role': 'assistant', 'content': '', 'tool_calls': calls}
        if s.get('message'):
            m['reasoning'] = s['message']   # adk_submission 0.2.12: past thinking reaches the model
        msgs.append(m)
        msgs.extend(results)
    return msgs


def shorten(x, cap=300):
    """The call arguments with long string values cut (keys and short values intact: enough to re-classify)."""
    if isinstance(x, dict):
        return {k: shorten(v, cap) for k, v in x.items()}
    if isinstance(x, list):
        return [shorten(v, cap) for v in x]
    return x[:cap] if isinstance(x, str) else x


def reclassify(samples, scripts_of):
    """Re-apply classify_call to stored samples (after a classifier change), where the call was stored."""
    for s in samples:
        if s.get('name') and s['variant'] in scripts_of:
            s['label'], s['intent'] = classify_call(s['name'], s.get('fargs'), scripts_of[s['variant']])
        elif s.get('intent') == 'create.py' and s.get('label') == 'edit_badshape' and '"content"' in (s.get('call') or '') \
                and '"target"' not in (s.get('call') or ''):
            s['label'] = 'snippet'   # samples stored before fargs: create.py content without a target
    return samples


def chat(server, msgs, tools, sampling, seed, timeout=900):
    body = {'model': MODEL, 'messages': msgs, 'tools': tools,
            'max_completion_tokens': int(sampling['max_output_tokens']), 'temperature': sampling['temperature'],
            'top_p': sampling['top_p'], 'seed': seed, 'chat_template_kwargs': {'enable_thinking': True}}
    if sampling.get('thinking_budget'):
        body['thinking_token_budget'] = int(sampling['thinking_budget'])
    req = urllib.request.Request(server.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


# ---------------------------------------------------------------- report

def report(samples, variants_order=None):
    vs = variants_order or sorted({s['variant'] for s in samples})
    by = {v: [s for s in samples if s['variant'] == v and s['label'] != 'request_error'] for v in vs}
    lines, rep = [], {'variants': {}, 'pairs': []}
    labels = ['edit', 'edit_badshape', 'read', 'search', 'locate', 'snippet', 'test', 'check', 'submit', 'other',
              'malformed', 'no_call']
    kbs = ['1', '2', '3', '4', '5-6', '7+']
    lines.append(f"{'variant':38s} {'n':>4s} {'P(edit)':>8s} {'edit+try':>8s} {'malf':>6s} {'nocall':>6s} "
                 f"{'explore':>7s} {'run':>5s} {'tok':>5s} {'think':>6s} {'prompt':>6s}")
    for v in vs:
        S = by[v]
        n = len(S) or 1

        def p(cond):
            return sum(1 for s in S if cond(s)) / n
        r = {'n': len(S), 'p_edit': p(lambda s: s['label'] == 'edit'),
             'p_edit_attempt': p(lambda s: s['label'] in EDIT_GROUP or s.get('intent') in ('edit.py', 'create.py')
                                 and s['label'] == 'malformed'),
             'p_malformed': p(lambda s: s['label'] == 'malformed'), 'p_no_call': p(lambda s: s['label'] == 'no_call'),
             'p_explore': p(lambda s: s['label'] in EXPLORE), 'p_run': p(lambda s: s['label'] in RUN),
             'mix': {lab: sum(1 for s in S if s['label'] == lab) for lab in labels},
             'tokens': sum(s.get('tokens') or 0 for s in S) / n, 'think_chars': sum(s.get('think_chars') or 0
                                                                                   for s in S) / n,
             'prompt_tokens': sum(s.get('prompt_tokens') or 0 for s in S) / n, 'by_k': {}, 'by_source': {}}
        for kb in kbs:
            K = [s for s in S if k_bucket(s['k']) == kb]
            if K:
                r['by_k'][kb] = (sum(1 for s in K if s['label'] == 'edit') / len(K), len(K))
        for grp, cond in (('recorded edit', lambda s: s.get('recorded_next') in EDIT_GROUP),
                          ('recorded other', lambda s: s.get('recorded_next') not in EDIT_GROUP)):
            K = [s for s in S if cond(s)]
            if K:
                r.setdefault('by_recorded', {})[grp] = (sum(1 for s in K if s['label'] == 'edit') / len(K), len(K),
                                                        sum(1 for s in K if s['label'] in RUN) / len(K))
        for src in sorted({s['source'] for s in S}):
            K = [s for s in S if s['source'] == src]
            r['by_source'][src] = (sum(1 for s in K if s['label'] == 'edit') / len(K), len(K))
        rep['variants'][v] = r
        lines.append(f"{v[:38]:38s} {r['n']:4d} {r['p_edit']:8.3f} {r['p_edit_attempt']:8.3f} {r['p_malformed']:6.3f} "
                     f"{r['p_no_call']:6.3f} {r['p_explore']:7.3f} {r['p_run']:5.3f} {r['tokens']:5.0f} "
                     f"{r['think_chars']:6.0f} {r['prompt_tokens']:6.0f}")
    lines.append('\nP(edit) by k (prefix calls; n samples):')
    lines.append(f"{'variant':38s} " + ' '.join(f'{kb:>11s}' for kb in kbs))
    for v in vs:
        bk = rep['variants'][v]['by_k']
        lines.append(f'{v[:38]:38s} ' + ' '.join(f'{bk[kb][0]:6.2f} ({bk[kb][1]:3d})' if kb in bk else ' ' * 11
                                                 for kb in kbs))
    lines.append('\nP(edit) by recording:')
    for v in vs:
        lines.append(f'{v[:38]:38s} ' + '  '.join(f'{src}: {pe:.2f} ({n})'
                                                  for src, (pe, n) in rep['variants'][v]['by_source'].items()))
    lines.append('\nP(edit) / P(run) at states where the recording edited next vs the others:')
    for v in vs:
        lines.append(f'{v[:38]:38s} ' + '  '.join(f'{g}: {pe:.2f} / {pr:.2f} ({n})' for g, (pe, n, pr)
                                                  in rep['variants'][v].get('by_recorded', {}).items()))
    lines.append('\naction mix (counts):')
    lines.append(f"{'variant':38s} " + ' '.join(f'{lab[:6]:>6s}' for lab in labels))
    for v in vs:
        lines.append(f'{v[:38]:38s} ' + ' '.join(f"{rep['variants'][v]['mix'][lab]:6d}" for lab in labels))

    def per_state(v, cond):
        d = {}
        for s in by[v]:
            d.setdefault((s['trace'], s['step']), []).append(cond(s))
        return {k: sum(x) / len(x) for k, x in d.items()}
    lines.append('\npaired on the same states (A - B; mean of per-state differences, bootstrap 95% CI, sign test '
                 'over states with A != B):')
    metrics = (('edit', lambda s: s['label'] == 'edit'), ('explore', lambda s: s['label'] in EXPLORE),
               ('malformed', lambda s: s['label'] == 'malformed'))
    for a in range(len(vs)):
        for b in range(a + 1, len(vs)):
            for mname, cond in metrics:
                A, B = per_state(vs[a], cond), per_state(vs[b], cond)
                common = sorted(set(A) & set(B))
                diffs = [A[k] - B[k] for k in common]
                if not diffs:
                    continue
                w, lo_ = sum(1 for d in diffs if d > 0), sum(1 for d in diffs if d < 0)
                ci = bootstrap_ci(diffs)
                mean = sum(diffs) / len(diffs)
                pv = sign_test(w, lo_)
                rep['pairs'].append({'a': vs[a], 'b': vs[b], 'metric': mname, 'states': len(diffs), 'mean_diff': mean,
                                     'ci95': ci, 'a_higher': w, 'b_higher': lo_, 'sign_p': pv})
                lines.append(f'{mname:9s} {vs[a][:28]:28s} - {vs[b][:28]:28s}: {mean:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] '
                             f'on {len(diffs)} states; A>B {w}, B>A {lo_}, sign p={pv:.3f}')
    return '\n'.join(lines), rep


# ---------------------------------------------------------------- main

def load_states(trace_dirs, scripts, max_prompt):
    cands = []
    for d in trace_dirs:
        files = sorted(glob.glob(f'{d}/r*/traces/trace_*.json') + glob.glob(f'{d}/r*/c*/traces/trace_*.json'))
        for f in files:
            try:
                steps = json.load(open(f))['steps']
            except (ValueError, KeyError, OSError):
                continue
            for st in select_states(steps, scripts, max_prompt):
                cands.append(dict(st, trace=f, task=task_of(f), source=os.path.basename(d.rstrip('/'))))
    return cands


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--agent', action='append', default=[], help='materialized agent dir (repeatable)')
    ap.add_argument('--traces', default=','.join(DEFAULT_TRACES), help='results dirs, comma-separated')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=3, help='samples per state and variant')
    ap.add_argument('--max-states', type=int, default=80)
    ap.add_argument('--per-task', type=int, default=1)
    ap.add_argument('--edit-share', type=float, default=0.6,
                    help='share of states whose recorded next call was an edit attempt (the rest by k bucket)')
    ap.add_argument('--max-prompt', type=int, default=14000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--server', default='http://127.0.0.1:8002/v1')
    ap.add_argument('--workers', type=int, default=2, help='concurrent model requests')
    ap.add_argument('--replay-workers', type=int, default=3, help='concurrent sandbox replays (one container/task)')
    ap.add_argument('--recorded-outputs', action='store_true', help='use the recorded tool results (no sandbox)')
    ap.add_argument('--max-requests', type=int, default=0, help='stop after this many model requests (0: no cap)')
    ap.add_argument('--report', action='store_true', help='only print the report from --out/samples.jsonl')
    ap.add_argument('--dry-run', action='store_true', help='states + replays; write the requests to --out/messages/, '
                                                            'no model calls')
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    samples_path = out / 'samples.jsonl'

    if a.report:
        samples = [json.loads(x) for x in open(samples_path)]
        order = list(dict.fromkeys(s['variant'] for s in samples))
        cfg = out / 'config.json'
        if cfg.exists():
            vs = {k: Variant(d) for k, d in json.loads(cfg.read_text())['variants'].items()}
            reclassify(samples, {k: v.scripts for k, v in vs.items()})
        text, rep = report(samples, order)
        print(text)
        (out / 'report.json').write_text(json.dumps(rep, indent=1))
        (out / 'report.txt').write_text(text + '\n')
        return

    variants = [Variant(x) for x in a.agent]
    if not variants:
        sys.exit('give at least one --agent')
    all_scripts = set.union(*(v.scripts for v in variants))
    cfg_path = out / 'config.json'
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {'variants': {}}
    cfg['variants'].update({v.id: str(v.dir) for v in variants})
    cfg_path.write_text(json.dumps(cfg, indent=1))
    req = json.load(open(ROOT / 'data/probe/adk_request.json'))
    sysmsg = req['messages'][0]['content']
    suffix = sysmsg[sysmsg.index('\n\nYou are an agent.'):]
    tools = req['tools']

    states_path = out / 'states.json'
    if states_path.exists():
        picked = json.loads(states_path.read_text())
        print(f'{len(picked)} states from {states_path}')
    else:
        cands = load_states([x for x in a.traces.split(',') if x], all_scripts, a.max_prompt)
        picked = select_balanced(cands, a.max_states, a.per_task, a.seed, a.edit_share)
        states_path.write_text(json.dumps(picked, indent=0))
        print(f'{len(cands)} candidate states, picked {len(picked)}')
    kc = {}
    for s in picked:
        kc[k_bucket(s['k'])] = kc.get(k_bucket(s['k']), 0) + 1
    print('k buckets:', dict(sorted(kc.items())), ' recorded next:',
          {lab: sum(1 for s in picked if s['next'] == lab) for lab in sorted({s['next'] for s in picked})})

    # ---- replays: one per (trace, variant), up to the deepest selected state of that trace
    steps_of = {f: json.load(open(f))['steps'] for f in {s['trace'] for s in picked}}
    upto = {}
    for s in picked:
        upto[s['trace']] = max(upto.get(s['trace'], 0), s['k'])
    contents = {}
    live, live_lock = {}, threading.Lock()   # task -> open sandbox (closed by the finally below on errors)

    def do_task(item):
        """All replays of one task (every trace and variant) in one container, closed when done."""
        task, todo = item
        res, holder = [], [None]

        def sandbox():
            if holder[0] is None:
                t = time.time()
                holder[0] = TaskSandbox(task)
                with live_lock:
                    live[task] = holder[0]
                print(f'  sandbox {task} ready in {time.time() - t:.0f}s', flush=True)
            return holder[0]
        try:
            for f, v in todo:
                if a.recorded_outputs:
                    res.append((f, v, recorded_contents(steps_of[f], upto[f]), None))
                    continue
                try:
                    res.append((f, v, replay_trace(f, steps_of[f], upto[f], v, sandbox, out / 'replays'), None))
                except Exception as e:  # noqa: BLE001
                    res.append((f, v, None, repr(e)[:400]))
        finally:
            if holder[0] is not None:
                holder[0].close()
                with live_lock:
                    live.pop(task, None)
        return res

    per_task = {}
    for f in sorted(upto):
        per_task.setdefault(task_of(f), []).extend((f, v) for v in variants)
    t0 = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(a.replay_workers) as ex:
            for res in ex.map(do_task, sorted(per_task.items())):
                for f, v, c, err in res:
                    if err:
                        print(f'REPLAY FAILED {task_of(f)} {v.id}: {err}', flush=True)
                    else:
                        contents[(f, v.id)] = c
    finally:
        for sb in list(live.values()):
            sb.close()
    n_jobs = sum(len(x) for x in per_task.values())
    if not a.recorded_outputs:   # replays with the recording's own scripts must reproduce the recorded outputs
        same, total, src_hash = 0, 0, {}
        for f in upto:
            src = pathlib.Path(f).parents[3] if re.fullmatch(r'c\d+', pathlib.Path(f).parents[1].name) else \
                pathlib.Path(f).parents[2]   # <results>/r1/c0/traces/x.json or <results>/r1/traces/x.json
            if src not in src_hash:
                src_hash[src] = Variant(src / '_agent').scripts_hash if (src / '_agent').is_dir() else None
            for v in variants:
                if v.scripts_hash == src_hash[src] and (f, v.id) in contents:
                    rec = recorded_contents(steps_of[f], upto[f])
                    for k, c in contents[(f, v.id)].items():
                        total += 1
                        same += CLOCK.sub('T', c) == CLOCK.sub('T', rec[k])
                    break
        if total:
            print(f'replay fidelity: {same}/{total} outputs identical to the recording when replayed with its own '
                  f'scripts (clock masked)', flush=True)
    print(f'replays done in {time.time() - t0:.0f}s ({len(contents)}/{n_jobs})', flush=True)

    # ---- sampling
    done = set()
    if samples_path.exists():
        for x in open(samples_path):
            d = json.loads(x)
            if d['label'] != 'request_error':
                done.add((d['variant'], d['trace'], d['step'], d['seed']))
    work = []
    for s in picked:
        for v in variants:
            if (s['trace'], v.id) not in contents:
                continue
            msgs = build_messages(steps_of[s['trace']], s['k'], v, s['task'], suffix, contents[(s['trace'], v.id)])
            for seed in range(a.n):
                if (v.id, s['trace'], s['step'], seed) not in done:
                    work.append((s, v, seed, msgs))
    if a.dry_run:
        md = out / 'messages'
        md.mkdir(exist_ok=True)
        for s in picked:
            for v in variants:
                if (s['trace'], v.id) in contents:
                    m = build_messages(steps_of[s['trace']], s['k'], v, s['task'], suffix, contents[(s['trace'], v.id)])
                    (md / f"{v.id}__{s['task']}_{s['step']}.json").write_text(json.dumps(m, indent=1,
                                                                                           ensure_ascii=False))
        print(f'dry run: requests in {md}')
        return
    # interleave variants and states so a partial run is still paired
    work.sort(key=lambda w: (w[2], picked.index(w[0]), [x.id for x in variants].index(w[1].id)))
    if a.max_requests:
        work = work[:a.max_requests]
    print(f'{len(work)} model requests ({len(done)} already done)', flush=True)

    def run(w):
        s, v, seed, msgs = w
        t = time.time()
        try:
            return w, chat(a.server, msgs, tools, v.sampling, 1000 + seed), None, time.time() - t
        except Exception as e:  # noqa: BLE001
            return w, None, repr(e)[:300], time.time() - t

    log = open(samples_path, 'a')
    with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
        for (s, v, seed, _), resp, err, secs in ex.map(run, work):
            rec = {'variant': v.id, 'version': v.label, 'trace': s['trace'], 'task': s['task'], 'source': s['source'],
                   'step': s['step'], 'k': s['k'], 'recorded_next': s['next'], 'seed': seed, 'secs': round(secs, 1)}
            if err:
                rec.update(label='request_error', error=err)
            else:
                msg = resp['choices'][0]['message']
                label, intent, info = classify_response(msg, v.scripts)
                usage = resp.get('usage') or {}
                rec.update(label=label, intent=intent, tokens=usage.get('completion_tokens'),
                           prompt_tokens=usage.get('prompt_tokens'),
                           think_chars=len(msg.get('reasoning') or msg.get('reasoning_content') or ''),
                           finish=resp['choices'][0].get('finish_reason'),
                           call=json.dumps({'name': info.get('name'), 'args': info.get('args')},
                                           ensure_ascii=False)[:600] if info.get('name') else None,
                           name=info.get('name'), fargs=shorten(info.get('args')),
                           content=(msg.get('content') or '')[:400] or None)
            log.write(json.dumps(rec, ensure_ascii=False) + '\n')
            log.flush()
            print(f"{v.id[:24]:24s} {s['task']:16s} k={s['k']:2d} seed {seed}: {rec['label']:13s} "
                  f"{rec.get('intent') or '':14s} {rec.get('tokens') or '':>5} tok {secs:5.1f}s "
                  f"(recorded: {s['next']})", flush=True)
    log.close()
    samples = reclassify([json.loads(x) for x in open(samples_path)], {v.id: v.scripts for v in variants})
    text, rep = report(samples, [v.id for v in variants])
    print('\n' + text)
    (out / 'report.json').write_text(json.dumps(rep, indent=1))
    (out / 'report.txt').write_text(text + '\n')


if __name__ == '__main__':
    main()
