"""rft3: check the record -> request rebuild (rft3_states.request_at) against requests the harness's own code builds,
for an agent version that has no captured runs yet (v20). No GPU, no sandbox, no model.

    .venv/bin/python scripts/rft3_adkcheck.py --version v20 [--n 40] [--files data/rft3/states_v20/*.jsonl]

The materialized agent (rft3_states.Agent) is compiled with adk_submission.compile_submission and run by ADK's Runner,
exactly as swegemma's harness wires it (as experiments/variants/hyper/_adk_capture.py): the model is a fake
OpenAI-compatible server that replays one record's turns (tool calls with their ids, thinking as
reasoning_content, visible text as content; then a final text reply), the sandbox is a fake that returns each
turn's recorded stdout/stderr/exit code through the real AdkSandboxCodeExecutor (whose rules write the "Command failed
..." stderr), submit_patch is a stub returning the recorded result, and the user message is the record's. Every
request the fake server receives (k = 0 .. len(turns)) is compared with request_at(record, k) as JSON (key order
included, the proxy's logprobs fields aside): system message (prompt + ADK suffix with the version's name and
description), tools, sampling params, and every assistant/tool message rebuilt from the turns.
Records checked: a seeded sample of every file given, preferring multi-edit, repair and relabel records and records
with thinking (subset900_filled.jsonl).
"""

import argparse
import asyncio
import http.server
import json
import os
import random
import re
import sys
import threading
import warnings

sys.dont_write_bytecode = True
warnings.filterwarnings('ignore')
os.environ.setdefault('LITELLM_LOG', 'ERROR')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rft3_states as S  # noqa: E402

REQUESTS = []
REPLIES = []
OUTPUTS = []
FAILED = re.compile(r'^Command failed with status=(\w+), exit_code=(-?\d+)$')
TIMED_OUT = re.compile(r'^Command timed out after \d+s \(exit_code=(-?\d+)\)$')


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        REQUESTS.append(body)
        out = REPLIES.pop(0) if REPLIES else reply('(script exhausted)')
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def reply(content=None, call=None, cid=None, reasoning=None):
    msg = {'role': 'assistant', 'content': content}
    if reasoning:
        msg['reasoning_content'] = reasoning
    finish = 'stop'
    if call:
        msg['tool_calls'] = [{'id': cid, 'type': 'function', 'function': {
            'name': call['name'], 'arguments': json.dumps(call['arguments'], ensure_ascii=False)}}]
        finish = 'tool_calls'
    return {'id': 'x', 'object': 'chat.completion', 'created': 0, 'model': 'gemma-4-31b-it-qat-w4a16-ct',
            'choices': [{'index': 0, 'message': msg, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}


class FakeRes:
    def __init__(self, stdout, stderr='', exit_code=0, status='ok'):
        self.stdout, self.stderr, self.exit_code, self.status = stdout, stderr, exit_code, status


class FakeSandbox:
    work_dir = '/tmp'

    def write_file(self, name, code):
        pass

    def run_command(self, cmd, timeout=None):
        if cmd.startswith('rm -f'):
            return FakeRes('')
        return OUTPUTS.pop(0) if OUTPUTS else FakeRes('(no output scripted)')


def sandbox_result(result):
    """What the sandbox returned for a recorded run_skill_script result (inverse of the executor's stderr rules)."""
    d = json.loads(result)
    out, err = d.get('stdout') or '', d.get('stderr') or ''
    m = FAILED.match(err)
    if m:
        return FakeRes(out, '', int(m.group(2)), m.group(1))
    m = TIMED_OUT.match(err)
    if m:
        return FakeRes(out, '', int(m.group(1)), 'timeout')
    return FakeRes(out, err, 0, 'ok')


def problem_of(agent, rec):
    """The raw problem statement substituted into the system prompt (ADK state injection)."""
    pre, post = agent.prompt.split('{problem_description}', 1)
    sysmsg = rec['request']['messages'][0]['content']
    assert sysmsg.startswith(pre) and sysmsg.endswith(post + agent.suffix), 'system message is not prompt + suffix'
    return sysmsg[len(pre):len(sysmsg) - len(post + agent.suffix)]


async def run_record(agent, rec, port):
    from adk_eval_core.sandbox.base import AdkSandboxCodeExecutor
    from adk_submission import compile_submission
    from google.adk.apps.app import App
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from swegemma.config import build_submission_limits
    from swegemma.context import SwegemmaContext
    from swegemma.models import setup_gemma_model_registry
    problem = problem_of(agent, rec)
    REQUESTS.clear()
    REPLIES.clear()
    OUTPUTS.clear()
    submits = []
    for t in rec['turns']:
        REPLIES.append(reply(t.get('visible'), t['call'], t['id'], t.get('thinking')))
        if t['call']['name'] == 'run_skill_script' and 'stdout' in json.loads(t['result']):
            OUTPUTS.append(sandbox_result(t['result']))  # (a call ADK rejects never reaches the sandbox)
        elif t['call']['name'] == 'submit_patch':
            submits.append(json.loads(t['result'])['result'])
    REPLIES.append(reply('Done.'))
    ctx = SwegemmaContext(problem_statement=problem, max_tool_calls=60, max_time_minutes=30)
    ex = AdkSandboxCodeExecutor(sandbox=FakeSandbox(), timeout_seconds=180, budget_check_fn=ctx.check_budget)
    limits, gen = build_submission_limits()

    def get_status() -> str:
        """Check active budget details (time, tool calls) and active patch details."""
        return '{"status": "ok"}'

    def submit_patch() -> str:
        """Capture the current working tree modifications as the agent's submission.

        This generates a unified git diff against the baseline commit. The agent
        can submit multiple times; only the final submission is evaluated.
        """
        return submits.pop(0) if submits else '{"status": "ok"}'
    tools = {'get_status': get_status, 'submit_patch': submit_patch}
    root = compile_submission(submission_dir=agent.dir, tool_registry=tools,
                              model_registry=setup_gemma_model_registry(api_base=f'http://127.0.0.1:{port}/v1',
                                                                        num_retries=0),
                              code_executor=ex, limits=limits, generation_constraints=gen)
    ss = InMemorySessionService()
    s = await ss.create_session(app_name='swegemma_eval', user_id='u', state={'problem_description': problem})
    runner = Runner(app=App(name='swegemma_eval', root_agent=root), session_service=ss)
    ctx.start_agent_session()
    async for _ in runner.run_async(user_id='u', session_id=s.id, new_message=types.Content(
            role='user', parts=[types.Part(text=rec['request']['messages'][1]['content'])])):
        pass
    return list(REQUESTS)


def compare(rec, captured):
    """(requests equal as JSON bytes, total compared, first mismatch)."""
    eq, first = 0, None
    n = min(len(captured), len(rec['turns']) + 1)
    for k in range(n):
        got = S.request_at(rec, k)
        cap = S.strip_proxy(captured[k])
        a, b = json.dumps(got, ensure_ascii=False), json.dumps(cap, ensure_ascii=False)
        if a == b:
            eq += 1
        elif first is None:
            i = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
            first = f'request {k} at char {i}: rebuilt …{a[max(0, i - 100):i + 100]!r} | harness …' \
                    f'{b[max(0, i - 100):i + 100]!r}'
    if len(captured) != len(rec['turns']) + 1 and first is None:
        first = f'{len(captured)} requests for {len(rec["turns"])} turns'
    return eq, n, first


def pick(paths, n, seed=0):
    recs = []
    for p in paths:
        recs += list(S.read_records([p]))
    rng = random.Random(seed)

    def weight(r):
        multi = any((t.get('meta') or {}).get('hunks', 1) > 1 for t in r['turns'])
        think = any(t.get('thinking') for t in r['turns'])
        return (multi, r['source'] in ('relabel', 'oracle_repair'), think, rng.random())
    recs.sort(key=weight, reverse=True)
    top = recs[:n // 2]
    rest = recs[n // 2:]
    rng.shuffle(rest)
    return top + rest[:n - len(top)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--version', default='v20')
    ap.add_argument('--files', nargs='*', default=None)
    ap.add_argument('--n', type=int, default=40)
    a = ap.parse_args()
    S.use_version(a.version)
    agent = S.Agent()
    paths = a.files or [p for p in (os.path.join(S.STATES_DIR, f'{src}.jsonl') for src in (
        'oracle_clean', 'oracle_recovery', 'oracle_repair', 'relabel')) if os.path.exists(p)]
    recs = pick(paths, a.n)
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tot = eq = ok_recs = 0
    for rec in recs:
        captured = asyncio.run(run_record(agent, rec, port))
        e, n, first = compare(rec, captured)
        tot += n
        eq += e
        ok_recs += first is None
        multi = sum((t.get('meta') or {}).get('hunks', 1) > 1 for t in rec['turns'])
        print(f'{rec["id"]} [{rec["source"]}, {len(rec["turns"])} turns, multi-edit {multi}, thinking '
              f'{sum(bool(t.get("thinking")) for t in rec["turns"])}]: {e}/{n} requests equal'
              + (f'\n    MISMATCH {first[:900]}' if first else ''), flush=True)
    srv.shutdown()
    print(f'TOTAL ({S.AGENT_TAG}, harness code: adk_submission.compile_submission + ADK Runner + LiteLLM): '
          f'{eq}/{tot} requests byte-equal; records all equal {ok_recs}/{len(recs)}')
    return 0 if eq == tot else 1


if __name__ == '__main__':
    sys.exit(main())
