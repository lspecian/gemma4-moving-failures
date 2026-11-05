"""Call-validity probe: does a prompt variant make the model write well-formed tool calls?

Rebuilds real conversation states from eval traces (the first call of each task, and the call just before each
edit), swaps in a variant's system prompt (and its argument dialect in the history), renders them with Gemma 4's
chat template exactly as vLLM would, samples the raw next turn from a llama.cpp server, and parses it with the
Kaggle vLLM gemma4 tool parser. No harness, no sandbox, no GPU: a few seconds per sample.

    .venv/bin/python scripts/call_probe.py --agent results/E023_v10_vllm/_agent --traces results/E023_v10_vllm \
        --out results/P001_v10 [--n 2] [--states first,edit,mid] [--max-states 24]

Each sample also records the generated tokens and the length of the thinking (speed: on Kaggle every token is
~0.08 s), so prompt variants can be compared for time per call as well as for validity.

Sub-agent probes (variant hyper: main agent + writer + scout agent_tools; experiments/variants/hyper/_README.md):
    --role main     the main agent at the edit states with the variant's prompt/tools (does it call writer, well-formed,
                    or draft code in its thinking?); history footers rewritten by the role's phase.probe_rewrite
    --role writer   writer contexts built from edit states: the request (target/start/end from the agent's edit,
                    change from its thinking) -> kinds wshow (first turn) and wedit (after the real show.py output);
                    each sampled edit.py call is applied to the snapshot and checked (EDITED / NOT APPLIED)
    --role scout    scout chains from mid states: the question from the agent's thinking, up to --steps turns with
                    the real scripts on the snapshot; records loops, answers and pointer lines
    --request data/probe/hyper/<role>_request.json   (from experiments/variants/hyper/_adk_capture.py --save):
                    the role's exact system suffix, tools, thinking flag and token cap.

Server: llama-server on :8001 (scripts/serve_local.sh, --server http://127.0.0.1:8001) or the rented vLLM
(--server http://127.0.0.1:8002/v1 --workers 6: Kaggle's exact stack). Needs the chat template (fetched once into data/),
a captured ADK request for the tool schemas (data/probe/adk_request.json), the vLLM wheel in data/wheelhouse.
"""

import argparse
import glob
import importlib.util
import json
import os
import pathlib
import random
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.request
import zipfile

import jinja2

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = ROOT / 'data/probe'
TEMPLATE_URL = 'https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/raw/main/chat_template.jinja'
SCRIPTS = {'locate', 'show', 'outline', 'search', 'callers', 'tests_for', 'history', 'edit', 'create', 'run',
           'check_patch', 'hypothesis', 'status'}
D = '<|"|>'


def kaggle_parser():
    """_parse_gemma4_args and the call regex from the scorer's own vLLM wheel (not redistributed: read in place)."""
    whl = next(iter(glob.glob(str(ROOT / 'data/wheelhouse/vllm-*.whl'))))
    src = zipfile.ZipFile(whl).read('vllm/tool_parsers/gemma4_tool_parser.py').decode()
    delim = re.search(r'^STRING_DELIM\s*=\s*(.+)$', src, re.M).group(1)
    helpers = src[src.index('def _parse_gemma4_value'):src.index('# Parser\n')]
    ns = {}
    exec('import json\nSTRING_DELIM = ' + delim + '\n' + helpers, ns)

    def parse(args_str):  # the array parser can spin forever on some inputs (the bug we hit): cap it
        if threading.current_thread() is not threading.main_thread():  # worker threads: no SIGALRM
            box = {}

            def go():
                try:
                    box['v'] = ns['_parse_gemma4_args'](args_str)
                except Exception as e:  # noqa: BLE001
                    box['e'] = e
            t = threading.Thread(target=go, daemon=True)
            t.start()
            t.join(5)
            if t.is_alive():
                raise TimeoutError('parser hang')
            if 'e' in box:
                raise box['e']
            return box['v']
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('parser hang')))
        signal.alarm(5)
        try:
            return ns['_parse_gemma4_args'](args_str)
        finally:
            signal.alarm(0)
    return parse, re.compile(r'<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>', re.S)


def template():
    PROBE.mkdir(parents=True, exist_ok=True)
    p = PROBE / 'chat_template.jinja'
    if not p.exists():
        p.write_bytes(urllib.request.urlopen(TEMPLATE_URL, timeout=60).read())
    env = jinja2.Environment(extensions=['jinja2.ext.loopcontrols'])
    env.globals['raise_exception'] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
    return env.from_string(p.read_text())


# ------------------------------------------------------------------ variant dialect

def load_flow(agent, scripts=None):
    """The variant's flow module (for rendering suggested calls in its own notation), or None. Its sibling phase.py, if
    any, is imported by flow and left in sys.modules['phase']."""
    d = pathlib.Path(scripts) if scripts else pathlib.Path(agent) / 'skills/repo-tools/scripts'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(d))
    for m in ('flow', 'argfix', 'phase'):
        sys.modules.pop(m, None)
    spec = importlib.util.spec_from_file_location('flow', d / 'flow.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['flow'] = mod   # phase.py imports flow
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


OLD_CALL = re.compile(r'run_skill_script\(skill_name="repo-tools", file_path="([\w.]+)", args=')


def rewrite_text(text, flow, renames):
    """Re-render every v10-style suggested call in `text` with the variant's own cmd()."""
    out, i, dec = [], 0, json.JSONDecoder()
    for m in OLD_CALL.finditer(text):
        if m.start() < i:
            continue
        try:
            args, end = dec.raw_decode(text, m.end())
        except ValueError:
            continue
        if not text.startswith(')', end):
            continue
        args = {renames.get(k, k): v for k, v in args.items()}
        out.append(text[i:m.start()] + flow.cmd(m.group(1), args))
        i = end + 1
    return ''.join(out) + text[i:]


class Variant:
    def __init__(self, agent, prompt=None, scripts=None):
        self.agent = pathlib.Path(agent)
        self.prompt = pathlib.Path(prompt or self.agent / 'prompts/system.md').read_text()
        self.flow = load_flow(agent, scripts)
        phase = getattr(self.flow, 'phase', None)
        self.rewrite = getattr(phase, 'probe_rewrite', None)  # multi-agent variants: footers as this agent sees them
        self.renames = dict(getattr(self.flow, 'PROBE_RENAMES', {}))  # history arg keys to rename, e.g. file->path
        self.native = self.flow.cmd('show.py', {'path': 'a.py'}).startswith('run_skill_script(skill_name="repo')

    def call_args(self, a):
        a = dict(a)
        if isinstance(a.get('args'), dict):
            a['args'] = {self.renames.get(k, k): v for k, v in a['args'].items()}
        return a

    def tool_content(self, content):
        if self.native:
            return content
        try:
            d = json.loads(content)
        except ValueError:
            return self._text(content)
        for k in ('stdout', 'stderr', 'error'):
            if isinstance(d.get(k), str):
                d[k] = self._text(d[k])
        return json.dumps(d, ensure_ascii=False)

    def _text(self, text):
        text = rewrite_text(text, self.flow, self.renames)
        return self.rewrite(text) if self.rewrite else text


# ------------------------------------------------------------------ states from traces

TASKS = {}


def problem_statement(trace_file):
    """ADK fills {problem_description} with the task's problem_statement (traces keep the raw template)."""
    if not TASKS:
        for line in open(ROOT / 'data/competition/tasks.jsonl'):
            d = json.loads(line)
            TASKS[d['instance_id']] = d['problem_statement']
    tid = os.path.basename(trace_file)[len('trace_'):-len('.json')]
    return next(v for k, v in TASKS.items() if k == tid or k.endswith(tid) or tid.endswith(k))


def system_message(trace_file, variant, adk_suffix):
    return variant.prompt.replace('{problem_description}', problem_statement(trace_file).strip()) + adk_suffix


def states(trace_file, kinds):
    """(kind, index of the agent step whose call we resample) for one trace."""
    steps = json.load(open(trace_file))['steps']
    agent = [i for i, s in enumerate(steps) if s.get('source') == 'agent' and s.get('tool_calls')]
    out = []
    if 'first' in kinds and agent:
        out.append(('first', agent[0]))
    if 'mid' in kinds and len(agent) >= 4:
        out.append(('mid', agent[len(agent) // 2]))  # mid-exploration: where thinking runs longest
    if 'look' in kinds:  # the first lookup the scout would take over (search/outline/callers/tests_for/history)
        for i in agent[1:]:
            fp = str((steps[i]['tool_calls'][0].get('arguments') or {}).get('file_path', ''))
            if fp.split('.')[0] in ('search', 'outline', 'callers', 'tests_for', 'history'):
                out.append(('look', i))
                break
    if 'edit' in kinds:
        for i in agent:
            a = steps[i]['tool_calls'][0]['arguments']
            if 'edit' in str(a.get('file_path', '')) or (isinstance(a.get('args'), dict) and 'new' in a['args']):
                if i != agent[0]:
                    out.append(('edit', i))
                break  # the first edit attempt of the run
    return steps, out


PAST_THINKING = [True]   # --past-thinking


def past_thinking(text):
    """The thinking of a past assistant turn, under both keys: adk_submission 0.2.12 (the scorer since Sep 30) sends
    `reasoning` = reasoning_content and vLLM 0.19.1 reads only `reasoning`, so the model sees its own past thinking;
    0.2.11 sent only reasoning_content, which vLLM dropped. Our chat-template render reads either. --past-thinking off
    drops both (the 0.2.11 behaviour on vLLM)."""
    if not PAST_THINKING[0] or not text:
        return {}
    return {'reasoning_content': text, 'reasoning': text}


def messages(trace_file, steps, upto, variant, adk_suffix):
    msgs = [{'role': 'system', 'content': system_message(trace_file, variant, adk_suffix)},
            {'role': 'user', 'content': steps[1]['message']}]
    for s in steps[2:upto]:
        if s.get('source') != 'agent' or not s.get('tool_calls'):
            continue
        calls, results = [], []
        for j, tc in enumerate(s['tool_calls']):
            cid = f'call_{len(msgs)}_{j}'
            calls.append({'id': cid, 'type': 'function',
                          'function': {'name': tc['function_name'], 'arguments': variant.call_args(tc['arguments'])}})
            obs = s.get('observation') or {}
            results.append({'role': 'tool', 'tool_call_id': cid,
                            'content': variant.tool_content(obs.get('content', '') if isinstance(obs, dict) else '')})
        msgs.append({'role': 'assistant', 'content': '', **past_thinking(s.get('message') or ''), 'tool_calls': calls})
        msgs.extend(results)
    return msgs


# ------------------------------------------------------------------ sampling + classification

MODEL = 'gemma-4-31b-it-qat-w4a16-ct'


def chat_tokens(server, msgs, tools, thinking=True):
    """Token ids of the chat prompt exactly as vLLM's /v1/chat/completions builds it (BOS included)."""
    wire = []
    for m in msgs:  # OpenAI wire format: tool-call arguments as JSON strings
        m = dict(m)
        if m.get('tool_calls'):
            m['tool_calls'] = [dict(tc, function=dict(tc['function'], arguments=json.dumps(tc['function']['arguments'],
                                                                                          ensure_ascii=False)))
                               for tc in m['tool_calls']]
        wire.append(m)
    body = {'model': MODEL, 'messages': wire, 'tools': tools, 'add_generation_prompt': True,
            'chat_template_kwargs': {'enable_thinking': thinking}}
    req = urllib.request.Request(server.rstrip('/').removesuffix('/v1') + '/tokenize', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=300))['tokens']


def sample(server, prompt, n_predict, seed):
    """Raw next turn, special tokens kept. llama.cpp: /completion; vLLM (the Kaggle stack): /v1/completions."""
    if server.rstrip('/').endswith('/v1'):
        body = {'model': MODEL, 'prompt': prompt, 'max_tokens': n_predict, 'temperature': 0.2, 'top_p': 0.95,
                'seed': seed, 'skip_special_tokens': False, 'spaces_between_special_tokens': False}
        req = urllib.request.Request(server.rstrip('/') + '/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        d = json.load(urllib.request.urlopen(req, timeout=900))
        return d['choices'][0]['text'], (d.get('usage') or {}).get('completion_tokens')
    body = json.dumps({'prompt': prompt, 'n_predict': n_predict, 'temperature': 0.2, 'top_p': 0.95,
                       'cache_prompt': True, 'seed': seed}).encode()
    req = urllib.request.Request(server + '/completion', data=body, headers={'Content-Type': 'application/json'})
    d = json.load(urllib.request.urlopen(req, timeout=900))
    return d['content'], d.get('tokens_predicted')


def classify(raw, parse, call_re):
    m = call_re.search(raw)
    if not m:
        return 'no_call', None, None
    name, args_str = m.group(1), m.group(2)
    try:
        a = parse(args_str)
    except TimeoutError:
        return 'PARSER_HANG', name, None
    except Exception as e:  # noqa: BLE001
        return 'parse_error', name, repr(e)
    json_quotes = bool(re.search(r'[{,]\s*"?\w+"?\s*:\s*"', args_str))
    if name != 'run_skill_script':
        return ('other_tool_jsonq' if json_quotes else 'other_tool'), name, a
    fp = str(a.get('file_path', ''))
    ok = a.get('skill_name') == 'repo-tools' and fp.endswith('.py') and fp[:-3] in SCRIPTS \
        and isinstance(a.get('args', {}), dict)
    if not ok:
        return ('bad_envelope_jsonq' if json_quotes else 'bad_envelope'), name, a
    return ('ok_jsonq' if json_quotes else 'ok'), name, a


# ------------------------------------------------------------------ sub-agent probes (--role main|writer|scout)

AGENT_TOOLS = ('writer', 'scout')
CRASH = ('unknown_tool', 'bad_agent_args')   # the harness run raises: the task ends with an empty patch


def classify_role(label, name, a, tool_names, role_scripts):
    """Refine classify() for a multi-agent role: agent_tool calls, undeclared tools, scripts this role lacks."""
    if name is None or label in ('PARSER_HANG', 'parse_error'):
        return label
    if name not in tool_names:
        return 'unknown_tool'            # ADK raises ValueError: Tool '<name>' not found -> run aborted
    if name in AGENT_TOOLS:
        ok = isinstance(a, dict) and isinstance(a.get('request'), str) and a['request'].strip()
        return f'ok_{name}' if ok else 'bad_agent_args'   # AgentTool.run_async: args['request'] -> KeyError
    if label.startswith('ok') and name == 'run_skill_script' and str(a.get('file_path')) not in role_scripts:
        return 'missing_script'          # SCRIPT_NOT_FOUND: no crash, no budget, a wasted turn
    return label


CODE_HINT = re.compile(r'\n|\bdef |\breturn\b|\braise\b|==|!=|\w\(.*\)\s*$|^\s*(if|for|while) .*:$|[{}]', re.M)


def request_shape(req):
    """Which writer-request fields are present, the change length, whether the change looks like code."""
    req = str(req or '')
    m = re.search(r'change=(.*)$', req, re.S)
    change = m.group(1).strip() if m else ''
    return {k: (k + '=') in req for k in ('target', 'start', 'end', 'change')} | {
        'change_chars': len(change), 'change_code': bool(CODE_HINT.search(change)), 'request_chars': len(req)}


def task_of(trace_file):
    return os.path.basename(trace_file)[len('trace_'):-len('.json')]


WS_LOCK = threading.Lock()


def snapshot(task):
    """data/probe/ws/<task>: the task's repository snapshot, extracted once."""
    d = PROBE / 'ws' / task
    with WS_LOCK:
        if not (d / '.git').exists():
            tgz = next(iter(glob.glob(str(ROOT / f'data/competition/snapshots/{task}.tgz'))))
            d.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tgz) as t:
                t.extractall(d, **({"filter": "fully_trusted"} if hasattr(tarfile, "data_filter") else {}))
    return d


def run_script(scripts, script, args, ws, state, timeout=120):
    """Run one skill script locally as the sandbox would (argv --key value), with flow on: stdout + footer."""
    argv = []
    for k, v in (args or {}).items():
        argv += [f'--{k}', v if isinstance(v, str) else json.dumps(v)]
    env = dict(os.environ, WORKSPACE=str(ws), REPO_TOOLS_DIR=str(state), PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory() as cwd:
        try:
            r = subprocess.run([sys.executable, str(pathlib.Path(scripts).resolve() / script), *argv], cwd=cwd, env=env,
                               capture_output=True, text=True, timeout=timeout)
            return r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            return '', 'timeout', -1


def tool_result(script, out, err, code):
    """The function response as ADK serializes a run_skill_script result."""
    return json.dumps({'skill_name': 'repo-tools', 'file_path': script, 'stdout': out, 'stderr': err,
                       'status': 'success' if code == 0 else 'error'})


def first_locate(steps):
    for s in steps:
        for tc in s.get('tool_calls') or []:
            a = tc.get('arguments') or {}
            if str(a.get('file_path', '')).startswith('locate') and isinstance(a.get('args'), dict):
                return a['args']
    return None


def base_state(agent, steps, ws, tmp):
    """A flow state dir as after the main agent's own locate.py (candidates, query), from the trace's call."""
    st = pathlib.Path(tempfile.mkdtemp(dir=tmp))
    loc = first_locate(steps)
    main_scripts = pathlib.Path(agent) / 'skills/main/repo-tools/scripts'
    if loc and (main_scripts / 'locate.py').exists():
        q = {k: v for k, v in loc.items() if k in ('query', 'within')}
        run_script(main_scripts, 'locate.py', q or loc, ws, st)
    return st


def _decode(text):
    return re.sub(r'\\(.)', lambda m: {'n': '\n', 't': '\t', '"': '"', "'": "'", '\\': '\\'}.get(m.group(1),
                                                                                              m.group(0)), text)


def edit_region(ws, args, pad=3):
    """(target, start, end) of the agent's edit (line form, or its old text found in the file), widened by pad."""
    if not isinstance(args, dict):
        return None
    target = args.get('target') or args.get('path') or args.get('file') or args.get('file_path')
    if not target:
        return None
    try:
        lines = (pathlib.Path(ws) / str(target).removeprefix('/workspace/')).read_text(errors='replace').split('\n')
    except OSError:
        return None
    a = b = None
    if args.get('start'):
        try:
            a = int(str(args['start']).strip())
            b = int(str(args.get('end') or a).strip())
        except ValueError:
            a = None
    elif args.get('old'):
        text = '\n'.join(lines)
        for old in (str(args['old']), _decode(str(args['old']))):
            k = text.find(old)
            if k < 0:
                first = old.strip().split('\n')[0].strip()
                hit = [n for n, line in enumerate(lines) if first and line.strip() == first]
                if len(hit) == 1:
                    a = hit[0] + 1
                    b = a + old.strip().count('\n')
                    break
                continue
            a = text[:k].count('\n') + 1
            b = a + old.rstrip('\n').count('\n')
            break
    if not a:
        return None
    return str(target).removeprefix('/workspace/'), max(1, a - pad), min(len(lines), b + pad)


ACTION = re.compile(r'\b(add|remov|replac|chang|mov|check|rais|return|us(e|ing)|set|ignor|skip|strip|insert|delet|'
                    r'renam|wrap|call|pass|fix|instead|should|need|must|only|default)\w*', re.I)
NOISE = re.compile(r'^(wait|let me|let\'s|now|ok|okay|yes|hmm|so|actually|i will apply|i\'ll apply|i\'ll use|'
                   r'i will use|first)\b', re.I)
SIDE = re.compile(r'\btests?\b|pre-commit|changelog|\bI should (?:probably )?(?:check|make sure|verify)|'
                  r'\bI will now apply|\bapply the fix', re.I)
PYKW = {'self', 'None', 'True', 'False', 'return', 'raise', 'import', 'from', 'class', 'def', 'pass', 'elif', 'else',
        'not', 'and', 'for', 'while', 'with', 'try', 'except', 'lambda', 'yield', 'async', 'await', 'continue', 'break'}


def change_text(thinking, args, cap=300):
    """change= derived deterministically from the agent's own thinking at its edit step: the last two sentences that
    describe an action (code blocks, numbered source lines and filler dropped), plus up to 5 names/strings that the
    agent's new text adds (the request is told to carry exact names and values)."""
    t = re.sub(r'```.*?```', ' ', str(thinking or ''), flags=re.S)
    keep = []
    for line in t.split('\n'):
        if re.match(r'^\s{4,}|\s*\d+\s{2}', line):
            continue
        for sent in re.split(r'(?<=[.!?])\s+', line.strip()):
            sent = sent.strip()
            if 20 <= len(sent) <= 250 and ACTION.search(sent) and not NOISE.match(sent) and sent.count('`') <= 6 \
                    and not SIDE.search(sent):
                keep.append(sent)
    change = ' '.join(keep[-2:]) or 'Make the change the issue asks for in these lines.'
    old, new = str(args.get('old') or ''), str(args.get('new') or '')
    tok = re.compile(r'"[^"\n]{1,60}"|\'[^\'\n]{1,60}\'|[A-Za-z_][A-Za-z0-9_]{2,}')
    added = [x for x in dict.fromkeys(tok.findall(new)) if x not in old and x not in PYKW and x not in change
             and (x[0] in '"\'' or re.search(r'[_A-Z0-9]', x) or re.search(re.escape(x) + r'\(', new))]
    if added:
        change += ' Names: ' + ', '.join(added[:5]) + '.'
    return change[:cap]


_SENT = r'(?:[^?!\n.]|\.(?=\S)){8,200}'   # a sentence: dots inside names/paths (rich/console.py) do not end it
QUESTION = re.compile(r'(?:^|(?<=[.!?\n]))\s*((?:where|which|what|does|do|is|are|how)\b' + _SENT + r'\?|'
                      r'(?:I (?:need|want|have) to|I\'ll|I will|Let me|Let\'s) (?:find|check|see|know|understand|look at|'
                      r'figure out|search for|look for|locate)\b' + _SENT + ')', re.I)


def scout_question(steps, i):
    """The lookup question behind a mid-exploration call: the last question (or 'I need to find ...') in the agent's
    thinking at that step; else built from the call's own arguments."""
    qs = QUESTION.findall(str(steps[i].get('message') or ''))
    if qs:
        q = qs[-1].strip()
        return re.sub(r'^(?:I (?:need|want|have) to|I\'ll|I will|Let me|Let\'s) ', '', q, flags=re.I)[:200]
    a = steps[i]['tool_calls'][0].get('arguments') or {}
    args = a.get('args') if isinstance(a.get('args'), dict) else {}
    for k in ('symbol', 'text', 'regex', 'name'):
        if args.get(k):
            return f'where is {str(args[k])[:80]} defined and where is it used'
    if args.get('files') or args.get('target'):
        return f'which classes and functions in {str(args.get("files") or args.get("target"))[:80]} matter for this issue'
    return None


POINTER = re.compile(r'([\w./-]+\.(?:py|pyi|html|cfg|toml|txt|md|rst|j2|jinja2?)):(\d+)(?:-(\d+))?')


def grounded(answer, ws):
    """(pointer lines, of which point at a real file and line in the snapshot)."""
    ptrs = POINTER.findall(answer or '')
    ok = 0
    for path, a, b in ptrs:
        try:
            n = len((pathlib.Path(ws) / path).read_text(errors='replace').split('\n'))
            ok += int(1 <= int(a) <= n)
        except OSError:
            pass
    return len(ptrs), ok


def visible(raw):
    text = raw.split('<channel|>')[-1]
    return re.sub(r'<\|?[a-z_"]+\|?>', '', text).strip()


def assistant_call(script, args, cid):
    return {'role': 'assistant', 'content': '',
            'tool_calls': [{'id': cid, 'type': 'function', 'function': {
                'name': 'run_skill_script', 'arguments': {'args': args, 'file_path': script,
                                                          'skill_name': 'repo-tools'}}}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', required=True)
    ap.add_argument('--traces', required=True, help='results dir(s), comma-separated')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--states', default=None, help='first,mid,edit,look; default first,edit; --role writer: edit; '
                                                    'scout: look (the first search/outline/callers/tests_for/history)')
    ap.add_argument('--max-states', type=int, default=24)
    ap.add_argument('--server', default='http://127.0.0.1:8001')
    ap.add_argument('--n-predict', type=int, default=None, help='default 1536, or the --request cap for a role')
    ap.add_argument('--workers', type=int, default=1, help='parallel requests (vLLM: 4-6)')
    ap.add_argument('--role', default='single', choices=('single', 'main', 'writer', 'scout'))
    ap.add_argument('--request', default=None, help='captured ADK request (default data/probe/adk_request.json; '
                                                    'roles: data/probe/hyper/<role>_request.json)')
    ap.add_argument('--prompt', default=None, help='system prompt (default prompts/system.md; roles writer/scout: '
                                                   'prompts/<role>.md)')
    ap.add_argument('--skills', default=None, help='scripts dir of the role (default skills/<role>/repo-tools/scripts)')
    ap.add_argument('--thinking', default=None, choices=('on', 'off'), help='default: from --request')
    ap.add_argument('--steps', type=int, default=3, help='scout: script calls simulated before the answer')
    ap.add_argument('--past-thinking', default='on', choices=('on', 'off'),
                    help='past assistant thinking in the history (on = the scorer with adk_submission 0.2.12)')
    a = ap.parse_args()
    PAST_THINKING[0] = a.past_thinking == 'on'
    role = a.role
    agent = pathlib.Path(a.agent).resolve()  # absolute: scripts run with cwd = a temp dir

    parse, call_re = kaggle_parser()
    tpl = template()
    req_path = a.request or (PROBE / 'adk_request.json' if role == 'single' else PROBE / 'hyper' / f'{role}_request.json')
    req = json.load(open(req_path))
    sysmsg = req['messages'][0]['content']
    adk_suffix = sysmsg[sysmsg.index('\n\nYou are an agent.'):]
    thinking = (a.thinking == 'on') if a.thinking else (
        True if role == 'single' else (req.get('chat_template_kwargs') or {}).get('enable_thinking', True))
    n_predict = a.n_predict or (1536 if role == 'single' else
                                req.get('max_completion_tokens') or req.get('max_tokens') or 1536)
    scripts = a.skills or (None if role == 'single' else agent / f'skills/{role}/repo-tools/scripts')
    prompt = a.prompt or (agent / f'prompts/{role}.md' if role in ('writer', 'scout') else None)
    variant = Variant(a.agent, prompt, scripts)
    tools = req['tools']
    tool_names = {t['function']['name'] for t in tools}
    role_scripts = {f.name for f in pathlib.Path(scripts).glob('*.py')} & {x + '.py' for x in SCRIPTS} \
        if scripts else {x + '.py' for x in SCRIPTS}
    kinds = (a.states or {'single': 'first,edit', 'main': 'edit', 'writer': 'edit', 'scout': 'look'}[role]).split(',')

    todo = []
    for d in a.traces.split(','):
        for f in sorted(glob.glob(f'{d}/r*/traces/trace_*.json')):
            steps, st = states(f, kinds)
            todo += [(f, steps, k, i) for k, i in st]
    random.Random(0).shuffle(todo)
    per_kind = {}
    picked = []
    for t in todo:  # balance kinds
        if per_kind.get(t[2], 0) < a.max_states // len(kinds):
            per_kind[t[2]] = per_kind.get(t[2], 0) + 1
            picked.append(t)

    os.makedirs(a.out, exist_ok=True)
    json.dump({'role': role, 'agent': str(agent), 'request': str(req_path), 'thinking': thinking,
               'n_predict': n_predict, 'kinds': kinds, 'n': a.n, 'server': a.server,
               'past_thinking': PAST_THINKING[0]},
              open(os.path.join(a.out, 'config.json'), 'w'), indent=1)
    log = open(os.path.join(a.out, 'samples.jsonl'), 'a')
    tally = {}
    vllm = a.server.rstrip('/').endswith('/v1')

    def render(msgs):
        if vllm:  # exact prompt: vLLM's own chat rendering + BOS, as token ids (its template copy may differ)
            return chat_tokens(a.server, msgs, tools, thinking)
        return tpl.render(messages=msgs, tools=tools, add_generation_prompt=True, bos_token='',
                          enable_thinking=thinking)

    def system_for(f):
        return variant.prompt.replace('{problem_description}', problem_statement(f).strip()) + adk_suffix

    def record(rec, label, kind):
        tally.setdefault(kind, {}).setdefault(label, 0)
        tally[kind][label] += 1
        log.write(json.dumps(rec, ensure_ascii=False) + '\n')
        log.flush()

    def describe(raw, ntok, parsed):
        think = raw.split('<channel|>')[0] if '<channel|>' in raw else ''
        return {'tokens': ntok, 'think_chars': len(think), 'args': parsed if isinstance(parsed, dict) else str(parsed),
                'raw_call': raw[raw.find('<|tool_call>'):][:1500] if '<|tool_call>' in raw else raw[-600:]}

    import concurrent.futures
    tmp = pathlib.Path(tempfile.mkdtemp(prefix='call_probe_'))
    jobs = []
    if role in ('single', 'main'):
        for f, steps, kind, i in picked:
            msgs = messages(f, steps, i, variant, adk_suffix)
            prompt_ = render(msgs)
            jobs += [(f, kind, i, s, prompt_, None) for s in range(a.n)]
    elif role == 'writer':
        for f, steps, kind, i in picked:
            ws = snapshot(task_of(f))
            args = (steps[i]['tool_calls'][0].get('arguments') or {}).get('args')
            reg = edit_region(ws, args)
            if not reg:
                print(f'skip {task_of(f)} step {i}: edit region not found ({str(args)[:80]})', flush=True)
                continue
            target, s0, e0 = reg
            request = f'target={target}  start={s0}  end={e0}  change={change_text(steps[i].get("message"), args)}'
            state = base_state(agent, steps, ws, tmp)
            base = [{'role': 'system', 'content': system_for(f)}, {'role': 'user', 'content': request}]
            show_args = {'target': target, 'start': str(s0), 'end': str(e0)}
            out, err, code = run_script(scripts, 'show.py', show_args, ws, state)
            with_show = base + [assistant_call('show.py', show_args, 'call_w1'),
                                {'role': 'tool', 'tool_call_id': 'call_w1', 'content': tool_result('show.py', out, err,
                                                                                                    code)}]
            ctx = {'ws': str(ws), 'state': str(state), 'request': request}
            for kind_, msgs in (('wshow', base), ('wedit', with_show)):
                p_ = render(msgs)
                jobs += [(f, kind_, i, s, p_, ctx) for s in range(a.n)]

    def run(job):
        f, kind, i, s, prompt_, ctx = job
        return job, sample(a.server, prompt_, n_predict, seed=1000 + s)

    if role in ('single', 'main', 'writer'):
        with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
            for (f, kind, i, s, _, ctx), (raw, ntok) in ex.map(run, jobs):
                label, name, parsed = classify(raw, parse, call_re)  # parsing stays on the main thread (SIGALRM)
                if role != 'single':
                    label = classify_role(label, name, parsed, tool_names, role_scripts)
                rec = {'trace': f, 'kind': kind, 'step': i, 'seed': s, 'label': label, 'tool': name}
                rec.update(describe(raw, ntok, parsed))
                extra = ''
                if label == 'ok_writer':
                    rec['request'] = parsed['request'][:600]
                    rec['shape'] = request_shape(parsed['request'])
                    extra = f"  change {rec['shape']['change_chars']} chars{' CODE' if rec['shape']['change_code'] else ''}"
                if ctx:
                    rec['writer_request'] = ctx['request']
                    fp = str(parsed.get('file_path', '')) if isinstance(parsed, dict) else ''
                    rec['script'] = fp
                    if kind == 'wedit' and label.startswith('ok') and fp in ('edit.py', 'create.py') \
                            and isinstance(parsed.get('args'), dict):
                        ea = parsed['args']
                        tgt = pathlib.Path(ctx['ws']) / str(ea.get('target') or '').removeprefix('/workspace/')
                        before = tgt.read_bytes() if tgt.is_file() else None
                        st2 = pathlib.Path(tempfile.mkdtemp(dir=tmp))
                        shutil.copytree(ctx['state'], st2, dirs_exist_ok=True)
                        out, err, code = run_script(scripts, fp, ea, ctx['ws'], st2)
                        rec['applied'] = out.lstrip().startswith(('EDITED', 'CREATED')) or 'EDITED' in out[:400]
                        rec['edit_out'] = (out + err)[:1200]
                        if before is None:
                            tgt.unlink(missing_ok=True)
                        else:
                            tgt.write_bytes(before)
                        extra = f"  {'APPLIED' if rec['applied'] else 'NOT APPLIED'}"
                record(rec, label, kind)
                print(f'{kind:5s} {task_of(f):16s} step {i:3d} seed {s}: {label} {rec.get("script", "")} '
                      f'{ntok} tok{extra}', flush=True)
    else:  # scout chains
        chains = []
        for f, steps, kind, i in picked:
            q = scout_question(steps, i)
            if not q:
                continue
            ws = snapshot(task_of(f))
            state = base_state(agent, steps, ws, tmp)
            chains += [(f, i, s, q, ws, state) for s in range(a.n)]

        def chain(c):
            f, i, s, q, ws, state0 = c
            state = pathlib.Path(tempfile.mkdtemp(dir=tmp))
            shutil.copytree(state0, state, dirs_exist_ok=True)
            msgs = [{'role': 'system', 'content': system_for(f)}, {'role': 'user', 'content': q}]
            turns, seen, outcome, answer, loops = [], set(), 'no_answer', '', 0
            for t in range(a.steps + 2):   # steps calls + the answer (+1: a refused call is answered by "answer now")
                raw, ntok = sample(a.server, render(msgs), n_predict, seed=1000 + s + 100 * t)
                label, name, parsed = classify(raw, parse, call_re)
                label = classify_role(label, name, parsed, tool_names, role_scripts)
                turn = {'label': label, 'tokens': ntok, 'tool': name}
                turns.append(turn)
                if label == 'no_call':
                    answer = visible(raw)
                    outcome = 'answered'
                    break
                if not label.startswith('ok'):
                    outcome = 'bad_call'
                    turn['raw'] = raw[-600:]
                    break
                fp, args = parsed['file_path'], parsed.get('args') if isinstance(parsed.get('args'), dict) else {}
                key = fp + json.dumps(args, sort_keys=True)
                loops += key in seen
                seen.add(key)
                turn.update(script=fp, args=args)
                out, err, code = run_script(scripts, fp, args, ws, state)
                turn['out_head'] = out[:200]
                cid = f'call_s{t}'
                msgs += [assistant_call(fp, args, cid),
                         {'role': 'tool', 'tool_call_id': cid, 'content': tool_result(fp, out, err, code)}]
            n_ptr, ok_ptr = grounded(answer, ws)
            return c, {'trace': f, 'kind': 'scout', 'step': i, 'seed': s, 'question': q, 'label': outcome,
                       'turns': turns, 'calls': sum(1 for x in turns if x.get('script')), 'repeats': loops,
                       'answer': answer[:2000], 'pointers': n_ptr, 'pointers_ok': ok_ptr,
                       'not_found': answer.upper().startswith('NOT FOUND'),
                       'tokens': sum(x['tokens'] or 0 for x in turns)}

        with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
            for c, rec in ex.map(chain, chains):
                record(rec, rec['label'], 'scout')
                print(f"scout {task_of(c[0]):16s} step {c[1]:3d} seed {c[2]}: {rec['label']} after {rec['calls']} "
                      f"calls ({' '.join(x.get('script') or x['label'] for x in rec['turns'])}), repeats "
                      f"{rec['repeats']}, pointers {rec['pointers_ok']}/{rec['pointers']}  Q: {rec['question'][:70]}",
                      flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(tally, indent=1))
    json.dump(tally, open(os.path.join(a.out, 'tally.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
