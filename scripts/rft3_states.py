"""rft3 decision states and trajectories in a frozen agent format (agent-v19; agent-v20 with --version v20 /
RFT3_VERSION, data/rft3/states_v20): schema, request builder, reader/validator.

Shared by scripts/rft3_oracle.py (part 1: oracle trajectories from SWE-bench train) and scripts/rft3_relabel.py
(part 2: relabelled decision states from our own runs). Step B sends each record's request (+ turns) to vLLM to sample
Gemma's thinking; step D renders training sequences. research/notes/14-oracle-trajectories.md has the plan.

Record (one JSONL line per trajectory, data/rft3/states/<source>.jsonl):
  id, source ("oracle_clean" | "oracle_recovery" | "oracle_repair" | "relabel"), task, repo,
  request   the exact chat-completions body the harness sends for the FIRST assistant turn (key order as captured):
            messages [system = agent-v19 prompts/system.md with {problem_description} replaced by the raw problem
            statement (ADK's state injection) + ADK's agent suffix; user = swegemma build_agent_prompt() with the
            Kaggle-like config (budget 4.5 min / 40 calls, 300 s commands, output 5000 chars, 150 lines, 10000 chars,
            code-intelligence block, no hints, the `find . -maxdepth 3 ... | sort | head -150` tree)], model,
            max_completion_tokens, temperature, tools (ADK's 6 tool schemas), top_p, chat_template_kwargs,
            thinking_token_budget (agent-v19 configs/sampling.yaml). The proxy's logprobs fields are not part of it.
  turns     [{id (tool-call id), call {name, arguments}, result (the tool message content exactly as ADK/LiteLLM send
            it: json.dumps(..., ensure_ascii=False) of the function response), label ("train" | "context"),
            thinking (real text for relabel context turns, else null: step B fills it), visible (text before the
            call: real for context turns, else null), grounding (why the action is justified by earlier outputs),
            kind (locate/show/outline/search/edit/edit_mistake/check/submit/... ), meta (optional)}]
  gold      {files, functions, spans [{file, a, b}]}
  meta      {...: per source; always est_prompt_tokens (estimated prompt of the last request, see estimate below)}

Message rebuild (messages_at(rec, k): the request messages before assistant turn k, 0-based), as ADK + LiteLLM send
them back (checked on every captured request of E058, `validate`):
  assistant: {"role": "assistant", ["content": visible,] "tool_calls": [{"type": "function", "id": id, "function":
             {"name": n, "arguments": json.dumps(arguments, ensure_ascii=False)}}], ["reasoning_content": thinking,
             "reasoning": thinking]}   (content only when the model wrote text, reasoning only when non-empty)
  tool:      {"role": "tool", "tool_call_id": id, "content": result}
Call arguments: the model writes keys in alphabetical order (args, file_path, skill_name; inner args sorted in 98% of
captured calls); oracle calls use sorted keys.
Tool results: run_skill_script -> {"skill_name", "file_path", "stdout", "stderr", "status"} with ADK's executor rules
(a non-zero exit with empty stderr gets "Command failed with status=error, exit_code=N"; status success / warning /
error); submit_patch -> {"result": "{\"status\": \"ok\", \"patch_size\": N, \"files_changed\": K}"}; rejected calls ->
{"error", "error_code"}.

v20 (agent-v20): edit.py calls may carry several hunks (start2/end2/new2 ... start4, old2/new2: string values like every
arg; turn meta {hunks: n}); every successful .py edit result has a `CHECK:` line. No v20 run is captured yet: the
rebuild is checked against requests the harness's own code builds (scripts/rft3_adkcheck.py --version v20).

Usage
    .venv/bin/python scripts/rft3_states.py validate [--runs E058_v19_kaggleslow]   # rebuild vs captured requests
    .venv/bin/python scripts/rft3_states.py stats data/rft3/states/*.jsonl [--tokens]
    .venv/bin/python scripts/rft3_states.py show data/rft3/states/oracle_clean.jsonl [--id ID | --n 1] [--full]
    .venv/bin/python scripts/rft3_states.py check data/rft3/states/*.jsonl            # schema + rebuild sanity
    .venv/bin/python scripts/rft3_states.py --version v20 check data/rft3/states_v20/*.jsonl
"""

import argparse
import collections
import copy
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# Agent versions the rft3 data is built in (v19: data/rft3/states, the step A/B build; v20: multi-edit + CHECK line,
# data/rft3/states_v20). The version is process-wide: RFT3_VERSION in the environment (so worker processes started
# with spawn see it) or use_version() (sets the environment too); every script has --version.
# ref: what scripts/materialize.sh exports (a tag or commit); v20 = agent-v20 + flow.py's SyntaxWarning filter
# (d67f7bc: repository parse warnings no longer reach stderr; nothing else model-facing changed)
VERSIONS = {'v19': {'tag': 'agent-v19', 'ref': 'agent-v19', 'agent': '_agent_v19', 'states': 'states'},
            'v20': {'tag': 'agent-v20', 'ref': 'd67f7bc', 'agent': '_agent_v20', 'states': 'states_v20'}}
VERSION = os.environ.get('RFT3_VERSION') or 'v19'
AGENT_DIR = os.path.join(ROOT, 'data', 'rft3', VERSIONS[VERSION]['agent'])
AGENT_TAG = VERSIONS[VERSION]['tag']
STATES_DIR = os.path.join(ROOT, 'data', 'rft3', VERSIONS[VERSION]['states'])


def use_version(version):
    """Switch this process (and its future children) to agent VERSION (v19 | v20)."""
    global VERSION, AGENT_DIR, AGENT_TAG, STATES_DIR
    if version not in VERSIONS:
        raise SystemExit(f'unknown version {version}; known: {sorted(VERSIONS)}')
    os.environ['RFT3_VERSION'] = VERSION = version
    AGENT_DIR = os.path.join(ROOT, 'data', 'rft3', VERSIONS[version]['agent'])
    AGENT_TAG = VERSIONS[version]['tag']
    STATES_DIR = os.path.join(ROOT, 'data', 'rft3', VERSIONS[version]['states'])


def suffixed(path):
    """PATH with the version's suffix before the extension for every version but v19 (oracle_raw -> oracle_raw_v20,
    relabel_raw.jsonl -> relabel_raw_v20.jsonl); v19 keeps the original names (the step A/B build)."""
    if VERSION == 'v19':
        return path
    root, ext = os.path.splitext(path)
    return f'{root}_{VERSION}{ext}'
ADK_REQUEST = os.path.join(ROOT, 'data', 'probe', 'adk_request.json')
SKILL = 'repo-tools'
COMPACT_AT = 14336
PROXY_KEYS = ('logprobs', 'top_logprobs', 'return_token_ids', 'seed')
# estimated thinking + visible tokens per earlier assistant turn (adk_submission 0.2.12 renders past thinking back;
# median thinking 95-110 tokens at budget 256, LOG E057/E058), added to the rendered prompt without thinking
THINK_EST = 110

# Kaggle-like harness config as the local Kaggle-like runs (E048-E058) used it (checked against their captures)
HARNESS = {'time_minutes': 4.5, 'tool_calls': 40, 'turns': None, 'cost_usd': None, 'command_timeout_seconds': 300,
           'max_stdout_chars': 5000, 'max_file_lines': 150, 'max_file_chars': 10000, 'graph': True,
           'enable_sandbox_testing': True}
TREE_CMD = ('find . -maxdepth 3 -not -path "./.git/*" -not -name "*.pyc" -not -name "__pycache__" | LC_ALL=C sort '
            '| head -150')


# ------------------------------------------------------------------ agent + request

class Agent:
    """The frozen agent version (materialized AGENT_TAG, e.g. agent-v19 in data/rft3/_agent_v19): prompt, ADK suffix,
    sampling, scripts."""

    def __init__(self, d=None):
        d = d or AGENT_DIR
        ref = VERSIONS[VERSION]['ref']
        stamp = d.rstrip('/') + '.version'
        if not os.path.isdir(d) or (d == AGENT_DIR and os.path.exists(stamp) and
                                    open(stamp).read().split()[:1] != [ref]):
            subprocess.run(['bash', os.path.join(ROOT, 'scripts', 'materialize.sh'), ref, d], cwd=ROOT,
                           check=True, capture_output=True)
        self.dir = d
        self.prompt = open(os.path.join(d, 'prompts', 'system.md')).read()
        y = open(os.path.join(d, 'agent.yaml')).read()
        self.name = re.search(r'^name:\s*(.+)$', y, re.M).group(1).strip()
        self.description = re.search(r'^description:\s*(.+)$', y, re.M).group(1).strip()
        self.model = re.search(r'^model:\s*(.+)$', y, re.M).group(1).strip()
        s = open(os.path.join(d, 'configs', 'sampling.yaml')).read()
        num = lambda k: re.search(rf'^\s*{k}:\s*([\d.]+)\s*$', s, re.M).group(1)  # noqa: E731
        self.sampling = {'temperature': float(num('temperature')), 'top_p': float(num('top_p')),
                         'max_output_tokens': int(num('max_output_tokens')),
                         'include_thoughts': re.search(r'^\s*include_thoughts:\s*true', s, re.M) is not None,
                         'thinking_budget': int(num('thinking_budget'))}
        self.scripts = os.path.join(d, 'skills', SKILL, 'scripts')
        self.script_names = {n for n in os.listdir(self.scripts) if n.endswith('.py')}
        # v20: edit.py takes several hunks of one file in one call (start2/end2/new2 ... start4, old2/new2 ...)
        self.multi_edit = 'start2' in open(os.path.join(self.scripts, 'edit.py')).read()
        if os.path.exists(ADK_REQUEST):
            req = json.load(open(ADK_REQUEST))
            sysmsg = req['messages'][0]['content']
            suffix = sysmsg[sysmsg.index('\n\nYou are an agent.'):]
            self.tools = req['tools']
        else:  # inside the sandbox container the probe file is mounted with the repo
            raise SystemExit(f'missing {ADK_REQUEST}')
        suffix = re.sub(r'Your internal name is "[^"]*"\.', lambda _: f'Your internal name is "{self.name}".', suffix,
                        count=1)
        self.suffix = re.sub(r'The description about you is ".*?"\.',
                             lambda _: f'The description about you is "{self.description}".', suffix, count=1)

    def system(self, problem):
        # ADK's inject_session_state replaces {problem_description} with str(state value): no stripping
        return self.prompt.replace('{problem_description}', problem) + self.suffix


def user_prompt(repo, problem, tree, hints='', cfg=HARNESS):
    """swegemma.harness.agent_runner.build_agent_prompt with the Kaggle-like config (validated on E058)."""
    budget = []
    if cfg['time_minutes'] is not None:
        budget.append(f'- Time allowance: {cfg["time_minutes"]} minutes')
    if cfg['tool_calls'] is not None:
        budget.append(f'- Tool calls allowance: {cfg["tool_calls"]} calls')
    if cfg['turns'] is not None:
        budget.append(f'- Max loop iterations: {cfg["turns"]} turns')
    if cfg['cost_usd'] is not None:
        budget.append(f'- Cost budget: ${cfg["cost_usd"]:.2f} USD')
    harness = [f'- Single command timeout: {cfg["command_timeout_seconds"]} seconds (commands exceeding this fail '
               f'without ending the session)',
               f'- Command output limit: {cfg["max_stdout_chars"]} characters',
               f'- File view limit: {cfg["max_file_lines"]} lines per read_file call',
               f'- File character limit: {cfg["max_file_chars"]} characters per read_file call',
               '- Environment is offline (no network/PyPI access). All repository and test dependencies are ALREADY '
               'pre-installed. Do NOT attempt to run pip install or download packages.']
    parts = [f'You are evaluating a software engineering task for repository {repo}.\n\nProblem Statement:\n'
             f'{problem}\n']
    if hints and hints.strip():
        parts.append(f'## Hints:\n{hints.strip()}\n')
    parts.append('## Task Budget (Session terminates when any budget is exhausted)\n' + '\n'.join(budget) + '\n')
    parts.append('## Execution Environment Rules\n' + '\n'.join(harness) + '\n')
    lines = ['## Instructions:',
             '0. All source code is under `/workspace`. Do NOT search outside `/workspace` (e.g. `/usr/`, `/wheels/`, '
             'system site-packages). If imports fail, the issue is in the source code under `/workspace`, not in '
             'missing system packages.',
             '1. Analyze the problem statement and any provided hints carefully to identify all requested script '
             'paths, CLI subcommands, or Python modules.',
             '2. Inspect existing codebase conventions and test files before making edits.',
             '3. Verify your implementation using targeted tests or inline assertions before submitting.',
             '4. Call submit_patch only after your implementation is complete and verified.',
             '5. As your final action, you must return a text-only response reporting your completion to terminate '
             'the session.']
    parts.append('\n'.join(lines))
    if cfg['graph']:
        parts.append('## Code Intelligence Tools\n'
                     'This repository has pre-built code graph and embedding data. Use these tools for fast, targeted '
                     'navigation:\n'
                     '- `search_similar_code(query)`: Find semantically similar functions/classes by keyword.\n'
                     '- `get_code_neighbors(node)`: Find callers, callees, and definitions related to a symbol.\n'
                     '- `get_code_subgraph(nodes)`: Get the induced subgraph for a set of symbols.\n')
    if tree:
        parts.append('## Workspace Layout\nThe repository is located at `/workspace`. Here is the directory tree (up '
                     'to 3 levels):\n```\n' + tree + '\n```\n')
    return '\n'.join(parts)


def tree_of_user(user):
    """The workspace tree embedded in a captured user message (None if absent)."""
    head = '(up to 3 levels):\n```\n'
    i = user.rfind(head)
    if i < 0 or not user.endswith('\n```\n'):
        return None
    return user[i + len(head):-len('\n```\n')]


def workspace_tree(ws):
    """The harness's tree command (agent_runner.run_agent_sandbox), run in ws (C-locale sort as in the container)."""
    r = subprocess.run(TREE_CMD, shell=True, cwd=ws, capture_output=True, text=True, errors='replace')
    return r.stdout.strip() if r.returncode == 0 else ''


def build_request(agent, repo, problem, tree, hints=''):
    s = agent.sampling
    req = {'messages': [{'role': 'system', 'content': agent.system(problem)},
                        {'role': 'user', 'content': user_prompt(repo, problem, tree, hints)}],
           'model': agent.model, 'max_completion_tokens': s['max_output_tokens'], 'temperature': s['temperature'],
           'tools': copy.deepcopy(agent.tools), 'top_p': s['top_p'],
           'chat_template_kwargs': {'enable_thinking': bool(s['include_thoughts'])}}
    if s.get('thinking_budget'):
        req['thinking_token_budget'] = s['thinking_budget']
    return req


# ------------------------------------------------------------------ calls, results, messages

def call_id(*parts):
    """A deterministic tool-call id in vLLM's format (chatcmpl-tool- + 16 hex)."""
    return 'chatcmpl-tool-' + hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:16]


def sort_keys(x):
    if isinstance(x, dict):
        return {k: sort_keys(x[k]) for k in sorted(x)}
    return x


def script_call(script, args=None):
    """A run_skill_script call as the model writes it (keys sorted; every arg value a string)."""
    a = {'args': {k: str(v) for k, v in (args or {}).items()}, 'file_path': script, 'skill_name': SKILL}
    if not args:
        a.pop('args')
        # the model sends no args key for no-arg scripts (check_patch.py) in most captured calls; keep {} out
    return {'name': 'run_skill_script', 'arguments': sort_keys(a)}


def submit_call():
    return {'name': 'submit_patch', 'arguments': {}}


def adk_result(file_path, out, err, code, skill=SKILL, timeout=None):
    """The run_skill_script function response as the model sees it (AdkSandboxCodeExecutor + ADK + LiteLLM)."""
    if not err:
        if code == 'timeout':
            err = f'Command timed out after {timeout}s (exit_code=124)'
        elif code not in (0, None):
            err = f'Command failed with status=error, exit_code={code}'
    status = 'success' if not err else ('error' if not out else 'warning')
    return json.dumps({'skill_name': skill, 'file_path': file_path, 'stdout': out, 'stderr': err, 'status': status},
                      ensure_ascii=False)


def submit_result(diff):
    inner = json.dumps({'status': 'ok', 'patch_size': len(diff), 'files_changed': diff.count('diff --git')})
    return json.dumps({'result': inner}, ensure_ascii=False)


def assistant_message(turn):
    m = {'role': 'assistant'}
    if turn.get('visible'):
        m['content'] = turn['visible']
    m['tool_calls'] = [{'type': 'function', 'id': turn['id'],
                        'function': {'name': turn['call']['name'],
                                     'arguments': json.dumps(turn['call']['arguments'], ensure_ascii=False)}}]
    if turn.get('thinking'):
        m['reasoning_content'] = turn['thinking']
        m['reasoning'] = turn['thinking']
    return m


def tool_message(turn):
    return {'role': 'tool', 'tool_call_id': turn['id'], 'content': turn['result']}


def messages_at(rec, k):
    """The request messages before assistant turn k (0-based; k = len(turns) gives the prompt after the last result)."""
    msgs = copy.deepcopy(rec['request']['messages'])
    for t in rec['turns'][:k]:
        msgs.append(assistant_message(t))
        msgs.append(tool_message(t))
    return msgs


def request_at(rec, k):
    req = {}
    for key, v in rec['request'].items():
        req[key] = messages_at(rec, k) if key == 'messages' else copy.deepcopy(v)
    return req


def script_of(call):
    a = call.get('arguments') if isinstance(call.get('arguments'), dict) else {}
    fp = str(a.get('file_path') or '')
    return fp[len('scripts/'):] if fp.startswith('scripts/') else fp


# ------------------------------------------------------------------ captured runs (validation)

def load_raw(raw_dir):
    import rft2_build
    return rft2_build.load_run(raw_dir)


def captured_episodes(run, raw_root=None, tasks=None):
    """{task: [captured agent requests in order]} of one raw run (rft2_build grouping) + the task table."""
    import rft2_build
    raw_root = raw_root or os.path.join(ROOT, 'data', 'raw')
    tasks = tasks or {json.loads(l)['instance_id']: json.loads(l) for l in open(os.path.join(ROOT, 'data',
                                                                                             'competition',
                                                                                             'tasks.jsonl'))}
    recs = rft2_build.load_run(os.path.join(raw_root, run))
    stats = collections.Counter()
    groups, _ = rft2_build.group_run(recs, rft2_build.TaskFinder({k: v['problem_statement'].strip()
                                                                  for k, v in tasks.items()}), stats)
    return groups, tasks


def record_from_capture(agent, task, reqs):
    """A record built from one captured episode (requests before the first compaction): request rebuilt with our
    builder from the captured tree, turns from the captured responses and the tool results of later requests."""
    first = reqs[0]['request']
    tree = tree_of_user(first['messages'][1]['content'])
    rec = {'id': f'capture_{task["instance_id"]}', 'source': 'capture', 'task': task['instance_id'],
           'repo': task['repo'], 'request': build_request(agent, task['repo'], task['problem_statement'], tree),
           'turns': []}
    results = {}
    for d in reqs:
        for m in d['request']['messages']:
            if m.get('role') == 'tool':
                results.setdefault(m['tool_call_id'], m['content'])
    for d in reqs:
        msg = d['response']['choices'][0]['message']
        calls = msg.get('tool_calls') or []
        if len(calls) != 1:
            break
        c = calls[0]
        try:
            args = json.loads(c['function']['arguments'])
        except ValueError:
            break
        rec['turns'].append({'id': c['id'], 'call': {'name': c['function']['name'], 'arguments': args},
                             'result': results.get(c['id']), 'label': 'context',
                             'thinking': msg.get('reasoning') or msg.get('reasoning_content') or None,
                             'visible': msg.get('content') or None, 'grounding': None})
    return rec


def strip_proxy(req):
    return {k: v for k, v in req.items() if k not in PROXY_KEYS}


def normalise_ids(msgs):
    """Messages with tool-call ids replaced by their order of appearance (ids are not compared)."""
    ids = {}
    out = copy.deepcopy(msgs)
    for m in out:
        for c in m.get('tool_calls') or []:
            c['id'] = ids.setdefault(c['id'], f'id{len(ids)}')
        if m.get('tool_call_id'):
            m['tool_call_id'] = ids.setdefault(m['tool_call_id'], f'id{len(ids)}')
    return out


def first_diff(a, b):
    sa, sb = json.dumps(a, ensure_ascii=False), json.dumps(b, ensure_ascii=False)
    k = next((i for i in range(min(len(sa), len(sb))) if sa[i] != sb[i]), min(len(sa), len(sb)))
    return f'at char {k}: rebuilt …{sa[max(0, k - 80):k + 80]!r} | captured …{sb[max(0, k - 80):k + 80]!r}'


def validate(runs, raw_root=None, verbose=0):
    agent = Agent()
    tot = collections.Counter()
    examples = []
    for run in runs:
        groups, tasks = captured_episodes(run, raw_root)
        c = collections.Counter()
        for tid, reqs in sorted(groups.items()):
            task = tasks[tid]
            # requests before the first compaction (prefix break)
            pre = [reqs[0]]
            for x, y in zip(reqs, reqs[1:]):
                if y['request']['messages'][:len(x['request']['messages'])] != x['request']['messages']:
                    break
                pre.append(y)
            rec = record_from_capture(agent, task, pre)
            c['episodes'] += 1
            ok_ep = True
            first = strip_proxy(pre[0]['request'])
            c['user_equal'] += rec['request']['messages'][1] == first['messages'][1]
            c['system_equal'] += rec['request']['messages'][0] == first['messages'][0]
            c['params_equal'] += {k: v for k, v in rec['request'].items() if k != 'messages'} == \
                {k: v for k, v in first.items() if k != 'messages'}
            c['key_order_equal'] += list(rec['request']) == list(first)
            for k, d in enumerate(pre):
                if k > len(rec['turns']):
                    break
                cap = strip_proxy(d['request'])
                got = request_at(rec, k)
                c['requests'] += 1
                same = list(got) == list(cap) and all(
                    (normalise_ids(got[key]) == normalise_ids(cap[key])) if key == 'messages' else got[key] == cap[key]
                    for key in cap)
                same_bytes = same and json.dumps(normalise_ids(got['messages']), ensure_ascii=False) == \
                    json.dumps(normalise_ids(cap['messages']), ensure_ascii=False)
                c['requests_equal'] += same
                c['requests_equal_bytes'] += same_bytes
                if not same_bytes:
                    ok_ep = False
                    if len(examples) < 6:
                        examples.append(f'{run}/{tid} request {k}: ' + first_diff(normalise_ids(got['messages']),
                                                                                   normalise_ids(cap['messages'])))
            c['episodes_all_equal'] += ok_ep
            c['requests_after_compaction_skipped'] += len(reqs) - len(pre)
        print(f'{run}: {dict(c)}')
        tot.update(c)
    print(f'TOTAL: requests rebuilt == captured (ids normalised): {tot["requests_equal"]}/{tot["requests"]} '
          f'(json bytes {tot["requests_equal_bytes"]}); episodes all equal {tot["episodes_all_equal"]}/'
          f'{tot["episodes"]}; first request: system {tot["system_equal"]}, user {tot["user_equal"]}, params '
          f'{tot["params_equal"]}, key order {tot["key_order_equal"]}; skipped after compaction '
          f'{tot["requests_after_compaction_skipped"]}')
    for e in examples:
        print('  MISMATCH', e[:700])
    return tot


# ------------------------------------------------------------------ token estimate

_RENDERER = []


def renderer():
    if not _RENDERER:
        import rft_build
        _RENDERER.append(rft_build.Renderer(os.path.join(ROOT, 'data', 'rft2', 'tokenizer_june')))
    return _RENDERER[0]


def prompt_tokens(rec, k=None, think_est=THINK_EST):
    """(rendered tokens, estimate) of the prompt before turn k (default: the last turn = the largest prompt the
    trajectory sends): June template, rendered with the record's own thinking (null for synthetic turns), plus
    think_est tokens for every earlier turn whose thinking is still null (step B fills it)."""
    R = renderer()
    k = len(rec['turns']) - 1 if k is None else k
    msgs = messages_at(rec, k)
    n = len(R.ids(R.text(msgs, rec['request'].get('tools'), True, True)))
    missing = sum(1 for t in rec['turns'][:k] if not t.get('thinking'))
    return n, n + think_est * missing


# ------------------------------------------------------------------ reading, checking, printing

def read_records(paths):
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


REQUIRED = ('id', 'source', 'task', 'repo', 'request', 'turns', 'gold', 'meta')
TURN_KEYS = ('id', 'call', 'result', 'label', 'thinking', 'visible', 'grounding')


def check_record(rec):
    """Problems of one record ([] when fine): schema, labels, rebuildable messages, JSON tool results, args objects."""
    p = [f'missing {k}' for k in REQUIRED if k not in rec]
    if p:
        return p
    if rec['source'] not in ('oracle_clean', 'oracle_recovery', 'oracle_repair', 'relabel'):
        p.append(f'source {rec["source"]}')
    ids = set()
    for k, t in enumerate(rec['turns']):
        p += [f'turn {k} missing {x}' for x in TURN_KEYS if x not in t]
        if t.get('label') not in ('train', 'context'):
            p.append(f'turn {k} label {t.get("label")}')
        if t.get('id') in ids:
            p.append(f'turn {k} duplicate id')
        ids.add(t.get('id'))
        try:
            json.loads(t.get('result') or '')
        except ValueError:
            p.append(f'turn {k} result is not JSON')
        a = (t.get('call') or {}).get('arguments')
        if t.get('label') == 'train':
            if not isinstance(a, dict):
                p.append(f'turn {k} arguments not an object')
            elif t['call']['name'] == 'run_skill_script':
                if not isinstance(a.get('args', {}), dict):
                    p.append(f'turn {k} args not an object')
                elif any(not isinstance(v, str) for v in a.get('args', {}).values()):
                    p.append(f'turn {k} non-string arg value')
                if a.get('skill_name') != SKILL or not re.fullmatch(r'[\w-]+\.py', str(a.get('file_path'))):
                    p.append(f'turn {k} malformed script call')
    if not any(t.get('label') == 'train' for t in rec['turns']):
        p.append('no trained turn')
    msgs = messages_at(rec, len(rec['turns']))
    if len(msgs) != 2 + 2 * len(rec['turns']):
        p.append('rebuild length')
    return p


def short(v, n=90):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = s.replace('\n', '\\n')
    return s if len(s) <= n else s[:n - 3] + '...'


def describe_call(call):
    if call['name'] != 'run_skill_script':
        return call['name']
    a = call['arguments'] if isinstance(call['arguments'], dict) else {}
    inner = a.get('args') if isinstance(a.get('args'), dict) else {}
    return f'{script_of(call) or a}  ' + '  '.join(f'{k}={short(v, 70)}' for k, v in inner.items())


def show_record(rec, full=False, out_lines=6):
    lines = [f'### {rec["id"]}  [{rec["source"]}]  task {rec["task"]} ({rec["repo"]})',
             f'gold: {rec["gold"].get("functions")} spans '
             f'{[(s["file"], s["a"], s["b"]) for s in rec["gold"].get("spans", [])]}',
             f'meta: {short(rec.get("meta"), 400)}']
    for k, t in enumerate(rec['turns']):
        lines.append(f'[{k}] {t["label"]:7s} {describe_call(t["call"])}')
        if t.get('thinking'):
            lines.append(f'      thinking: {short(t["thinking"], 160)}')
        if t.get('grounding'):
            lines.append(f'      grounding: {t["grounding"]}')
        try:
            res = json.loads(t['result'])
            text = res.get('stdout') or res.get('result') or res.get('error') or ''
            if res.get('stderr'):
                text += f'\n[stderr] {res["stderr"]}'
        except (ValueError, AttributeError):
            text = str(t['result'])
        rl = text.splitlines()
        for l in (rl if full else rl[:out_lines]):
            lines.append(f'      | {l[:150]}')
        if not full and len(rl) > out_lines:
            lines.append(f'      | ... ({len(rl) - out_lines} more lines)')
    return '\n'.join(lines)


def stats(paths, tokens=False):
    recs = list(read_records(paths))
    by = collections.defaultdict(list)
    for r in recs:
        by[r['source']].append(r)
    out = []
    for src, rs in sorted(by.items()):
        calls = [len(r['turns']) for r in rs]
        trained = [sum(t['label'] == 'train' for t in r['turns']) for r in rs]
        repos = collections.Counter(r['repo'] for r in rs)
        est = [r['meta'].get('est_prompt_tokens') for r in rs if r['meta'].get('est_prompt_tokens')]
        out.append(f'== {src}: {len(rs)} trajectories, {len({r["task"] for r in rs})} tasks, {len(repos)} repos; '
                   f'calls/trajectory mean {statistics.mean(calls):.1f} median {statistics.median(calls)} '
                   f'max {max(calls)}; trained turns mean {statistics.mean(trained):.1f}')
        if est:
            out.append(f'   est. prompt tokens of the last request: median {statistics.median(est):.0f}, p90 '
                       f'{sorted(est)[int(0.9 * len(est))]:.0f}, max {max(est):.0f}')
        kinds = collections.Counter(t.get('kind') or script_of(t['call']) or t['call']['name'] for r in rs
                                    for t in r['turns'])
        out.append('   turn kinds: ' + ', '.join(f'{k} {v}' for k, v in kinds.most_common()))
        out.append('   per repo: ' + ', '.join(f'{k} {v}' for k, v in sorted(repos.items(), key=lambda kv: -kv[1])))
    if tokens:
        for src, rs in sorted(by.items()):
            ns = [prompt_tokens(r) for r in rs]
            out.append(f'== {src} rendered prompt before the last turn (June template): no-thinking median '
                       f'{statistics.median(n for n, _ in ns):.0f}; with {THINK_EST}/turn thinking allowance median '
                       f'{statistics.median(e for _, e in ns):.0f}, max {max(e for _, e in ns)}; over {COMPACT_AT}: '
                       f'{sum(e >= COMPACT_AT for _, e in ns)}')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--version', default=None, help='agent version of the data (v19 default | v20)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('validate')
    v.add_argument('--runs', default='E058_v19_kaggleslow')
    v.add_argument('--raw-root', default=os.path.join(ROOT, 'data', 'raw'))
    s = sub.add_parser('stats')
    s.add_argument('paths', nargs='+')
    s.add_argument('--tokens', action='store_true')
    sh = sub.add_parser('show')
    sh.add_argument('path')
    sh.add_argument('--id', default=None)
    sh.add_argument('--n', type=int, default=1)
    sh.add_argument('--full', action='store_true')
    c = sub.add_parser('check')
    c.add_argument('paths', nargs='+')
    a = ap.parse_args()
    if a.version:
        use_version(a.version)
    if a.cmd == 'validate':
        validate(a.runs.split(','), a.raw_root)
    elif a.cmd == 'stats':
        print(stats(a.paths, a.tokens))
    elif a.cmd == 'show':
        n = 0
        for r in read_records([a.path]):
            if a.id and r['id'] != a.id:
                continue
            print(show_record(r, a.full))
            n += 1
            if n >= a.n:
                break
    elif a.cmd == 'check':
        bad = collections.Counter()
        n = 0
        for r in read_records(a.paths):
            n += 1
            for p in check_record(r):
                bad[p.split(' ')[0] if p.startswith('turn') else p] += 1
                bad[f'{r["source"]}: {re.sub(r"turn [0-9]+ ", "turn k ", p)}'] += 1
        print(f'{n} records; problems: {dict(bad) or "none"}')


if __name__ == '__main__':
    main()
