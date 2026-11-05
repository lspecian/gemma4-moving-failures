"""LLM-free harness replay: an OpenAI-compatible server that answers with a recorded run's model responses.

Point the real harness at it (OPENAI_BASE_URL) and every task makes the same tool calls as in the recording, through
the real sandbox, skill wrapper and per-task environment, with no GPU. Use: exact tool timings, script crashes, any
harness-side behaviour of a recorded run. Responses come from slow_proxy.py --save directories (data/raw/<run>).

- Each recorded request is assigned to its task by the issue text in its first two messages (the system instruction
  carries the issue, so this survives compaction) or "for repository <org/repo>" in the task prompt.
- A task's responses are served in recorded order, whatever the request says. Compaction summary requests ("The
  following is a conversation history...") and health checks get a short generic reply and do not advance the queue.
- When a task's queue is empty the server answers with plain text (no tool call), which ends the agent's turn.
- Model time is ~0: the run's durations are tool, sandbox and harness time only.

Handoff mode (--handoff-upstream URL): each task gets the recorded responses only up to (not including) the first one
that edits (edit.py/create.py, edit_file/write_file); every later request, and every compaction request, is forwarded
to the upstream model (e.g. Gemma behind slow_proxy.py). The upstream model then continues from the recorded model's
exploration: the same calls and results in its context, in the real sandbox state. Requests are mapped to tasks by
the issue text or by the ids of tool calls this server returned (so compaction dropping the issue does not matter).

    python3 scripts/replay_server.py --raw data/raw/E076_neb_v24_unseen --tasks data/unseen/tasks.jsonl --port 18090
    python3 scripts/replay_server.py --raw data/raw/E084_x --tasks data/competition/tasks.jsonl --port 18120 \
        --handoff-upstream http://127.0.0.1:18020 --log logs/R_handoff.jsonl
"""

import argparse
import collections
import gzip
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMPACTION = 'The following is a conversation history between a user and an AI agent'


def norm(text):
    return ' '.join(text.split())


def head_text(messages):
    parts = []
    for m in messages[:2]:
        c = m.get('content')
        if isinstance(c, list):
            c = ' '.join(str(x.get('text', '')) for x in c if isinstance(x, dict))
        parts.append(str(c or ''))
    return norm(' '.join(parts))


class Index:
    def __init__(self, tasks_path):
        tasks = [json.loads(line) for line in open(tasks_path)]
        self.sig = {t['instance_id']: norm(t['problem_statement'])[:150] for t in tasks if t.get('problem_statement')}
        per_repo = collections.defaultdict(list)
        for t in tasks:
            per_repo[t['repo']].append(t['instance_id'])
        # "for repository <org/repo>" names a task only when the set has one task of that repository (unseen100)
        self.repo = {f"repository {r}": ids[0] for r, ids in per_repo.items() if len(ids) == 1}

    def task_of(self, messages):
        txt = head_text(messages)
        hits = {tid for tid, s in self.sig.items() if s in txt}
        if len(hits) != 1:
            hits = {tid for k, tid in self.repo.items() if k in txt}
        return hits.pop() if len(hits) == 1 else None


EDIT_SCRIPTS = ('edit.py', 'create.py')
EDIT_TOOLS = ('edit_file', 'write_file')


def edits(resp):
    """True when a response's tool calls change a file (our edit/create scripts or the stock edit tools)."""
    for tc in ((resp.get('choices') or [{}])[0].get('message') or {}).get('tool_calls') or []:
        fn = tc.get('function') or {}
        if fn.get('name') in EDIT_TOOLS:
            return True
        if fn.get('name') == 'run_skill_script':
            try:
                if (json.loads(fn.get('arguments') or '{}').get('file_path') or '').split('/')[-1] in EDIT_SCRIPTS:
                    return True
            except ValueError:
                pass
    return False


def call_ids(resp):
    return [tc.get('id') for tc in ((resp.get('choices') or [{}])[0].get('message') or {}).get('tool_calls') or []
            if tc.get('id')]


def load(raw_dir, index):
    queues = collections.defaultdict(list)
    for f in sorted(os.listdir(raw_dir)):
        if not f.endswith('.json.gz'):
            continue
        r = json.load(gzip.open(os.path.join(raw_dir, f)))
        if r.get('path') != '/v1/chat/completions' or not isinstance(r.get('response'), dict):
            continue
        tid = index.task_of(r['request'].get('messages', []))
        if tid:
            queues[tid].append((r.get('t0', 0), r['response']))
    return {tid: collections.deque(resp for _, resp in sorted(q, key=lambda x: x[0])) for tid, q in queues.items()}


def text_reply(model, text):
    return {'id': f'replay-{time.time_ns()}', 'object': 'chat.completion', 'created': int(time.time()), 'model': model,
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': text}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='slow_proxy.py --save directory of the recorded run')
    ap.add_argument('--tasks', required=True, help='tasks.jsonl of the task set')
    ap.add_argument('--port', type=int, default=18090)
    ap.add_argument('--log', default=None, help='JSON lines: task, served index, remaining')
    ap.add_argument('--handoff-upstream', default='', help='after the recorded prefix (up to the first edit), forward here')
    ap.add_argument('--strip-reasoning', action='store_true',
                    help='serve recorded responses without their reasoning and visible text (tool calls only): the '
                         'information a model gathered without its analysis')
    a = ap.parse_args()
    index = Index(a.tasks)
    queues = load(a.raw, index)
    if a.handoff_upstream:  # keep the exploration only: the responses before the first edit
        for tid, q in queues.items():
            k = next((i for i, r in enumerate(q) if edits(r)), len(q))
            queues[tid] = collections.deque(list(q)[:k])
    if a.strip_reasoning:
        for q in queues.values():
            for resp in q:
                for ch in resp.get('choices') or []:
                    msg = ch.get('message') or {}
                    for k in ('reasoning_content', 'reasoning', 'thinking'):
                        msg.pop(k, None)
                    if msg.get('tool_calls'):
                        msg['content'] = None
    id2task = {}
    handed = set()
    print(f'replay: {sum(len(q) for q in queues.values())} responses for {len(queues)} tasks', file=sys.stderr)
    served = collections.Counter()
    lock = threading.Lock()
    logf = open(a.log, 'a') if a.log else None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip('/').endswith('/models'):
                return self._send({'object': 'list', 'data': [{'id': 'gemma-4-31b-it-qat-w4a16-ct', 'object': 'model'}]})
            self._send({'error': 'not found'}, 404)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}'
            req = json.loads(raw)
            model = req.get('model', 'replay')
            messages = req.get('messages', [])
            txt = head_text(messages)
            if txt.startswith(COMPACTION) or len(messages) < 2:
                if a.handoff_upstream:
                    return self._forward(raw)[0]
                return self._send(text_reply(model, 'Summary: the agent is working on the issue.'))
            tid = index.task_of(messages)
            if tid is None:  # compaction dropped the issue: the tool calls this server returned name the task
                ids = [tc.get('id') for m in messages for tc in (m.get('tool_calls') or [])] + \
                      [m.get('tool_call_id') for m in messages if m.get('tool_call_id')]
                tid = next((id2task[i] for i in reversed(ids) if i in id2task), None)
            with lock:
                q = queues.get(tid)
                resp = q.popleft() if q else None
                served[tid] += 1
                n, left = served[tid], (len(q) if q else 0)
                first_handoff = resp is None and a.handoff_upstream and tid not in handed
                if first_handoff:
                    handed.add(tid)
            if resp is None and a.handoff_upstream:
                _, resp = self._forward(raw)
                src = 'upstream'
            else:
                self._send(resp if resp is not None else text_reply(model, 'Done.'))
                src = 'recorded' if resp is not None else 'done'
            if isinstance(resp, dict):
                with lock:
                    for i in call_ids(resp):
                        id2task[i] = tid
            if logf:
                logf.write(json.dumps({'task': tid, 'n': n, 'left': left, 'src': src,
                                       'handoff': bool(first_handoff)}) + '\n')
                logf.flush()

        def _forward(self, raw):
            """Send the request upstream unchanged; answer with the upstream reply. Returns (None, parsed reply)."""
            r = urllib.request.Request(a.handoff_upstream.rstrip('/') + self.path, data=raw, method='POST',
                                       headers={'Content-Type': 'application/json',
                                                'Authorization': self.headers.get('Authorization', 'Bearer EMPTY')})
            try:
                with urllib.request.urlopen(r, timeout=1800) as up:
                    status, body = up.status, up.read()
            except urllib.error.HTTPError as e:
                status, body = e.code, e.read()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            try:
                return None, json.loads(body)
            except ValueError:
                return None, None

    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
