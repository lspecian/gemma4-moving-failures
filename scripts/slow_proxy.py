"""Latency proxy: makes a fast model server answer at Kaggle's 4×L4 speed (research/notes/13-tool-efficiency.md §1).

Each chat completion is forwarded upstream; before answering, the proxy sleeps until the request has taken as long as
it would on the target hardware:
    target = uncached_prompt_tokens / PREFILL + completion_tokens / DECODE
(uncached = prompt tokens minus usage.prompt_tokens_details.cached_tokens when the server reports it; else the
prompt tokens beyond the longest prefix this proxy has seen for the same first message — vLLM's prefix cache).
Concurrent requests are delayed independently (Kaggle scores tasks one at a time, so per-stream speed is what counts).

    python3 scripts/slow_proxy.py --port 8004 --upstream http://127.0.0.1:8002 --decode 12 --prefill 1500
    OPENAI_BASE_URL=http://127.0.0.1:8004/v1 scripts/run_chunked.sh ...
Every request is logged to --log as one JSON line (tokens, upstream seconds, target seconds); with --save DIR the full
request and response are also kept (gzipped JSON per call: exact replays, response caching, LoRA training data);
--logprobs K adds per-token log-probabilities (top K alternatives) and the generated token ids to every reply (raw
output before vLLM's tool parser: detokenize token_ids via /detokenize), for decision analysis.

Hosted APIs as upstream (a teacher model in our harness, research/notes/23): --api-key-env NAME sends
`Authorization: Bearer $NAME` (the client's header is dropped), --model rewrites the request's model, --strip removes
vLLM-only request fields. Cached tokens are read from usage.prompt_tokens_details.cached_tokens or DeepSeek's
usage.prompt_cache_hit_tokens.

    python3 scripts/slow_proxy.py --port 18030 --upstream https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY \
        --model deepseek-v4-pro --decode 35 --prefill 1400 --log logs/E082_proxy.jsonl --save data/raw/E082
"""

import argparse
import gzip
import http.server
import itertools
import os
import json
import threading
import time
import urllib.error
import urllib.request

LOCK = threading.Lock()
COUNTER = itertools.count(1)
SEEN = {}  # first message text -> longest prompt (tokens) seen with it (prefix-cache approximation)


def make_handler(a):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def _send(self, status, headers, data):
            self.send_response(status)
            for k, v in headers:
                if k.lower() not in ('transfer-encoding', 'content-length', 'connection'):
                    self.send_header(k, v)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _upstream(self, body=None):
            # no Accept-Encoding: a compressed reply cannot be read for pacing (hosted APIs gzip when asked)
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ('host', 'content-length', 'connection', 'accept-encoding')}
            if a.api_key_env:
                headers = {k: v for k, v in headers.items() if k.lower() != 'authorization'}
                headers['Authorization'] = 'Bearer ' + os.environ[a.api_key_env]
                headers.setdefault('Content-Type', 'application/json')
            req = urllib.request.Request(a.upstream + self.path, data=body, method=self.command, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    return r.status, r.headers.items(), r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.headers.items(), e.read()

        def do_GET(self):
            self._send(*self._upstream())

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            if (a.model or a.strip) and self.path.endswith('/chat/completions'):  # hosted API upstream
                try:
                    req = json.loads(body)
                    if a.model:
                        req['model'] = a.model
                    for k in a.strip:
                        req.pop(k, None)
                    body = json.dumps(req).encode()
                except ValueError:
                    pass
            if a.logprobs and self.path.endswith('/chat/completions'):  # dev only: token probabilities in the reply
                try:
                    req = json.loads(body)
                    req.setdefault('logprobs', True)
                    req.setdefault('top_logprobs', a.logprobs)
                    req.setdefault('return_token_ids', True)  # the raw generated tokens (tool-call syntax included)
                    body = json.dumps(req).encode()
                except ValueError:
                    pass
            t0 = time.time()
            status, headers, data = self._upstream(body)
            up = time.time() - t0
            rec = {'path': self.path, 'status': status, 'upstream_s': round(up, 2)}
            try:
                req, resp = json.loads(body), json.loads(data)
                usage = resp.get('usage') or {}
                pt, ct = usage.get('prompt_tokens') or 0, usage.get('completion_tokens') or 0
                cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
                if cached is None:
                    cached = usage.get('prompt_cache_hit_tokens')  # DeepSeek's API
                if cached is None:
                    msgs = req.get('messages') or []
                    key = json.dumps(msgs[:1])[:4000]
                    with LOCK:
                        cached = min(SEEN.get(key, 0), pt)
                        SEEN[key] = max(SEEN.get(key, 0), pt)
                target = (pt - cached) / a.prefill + ct / a.decode
                rec.update(prompt=pt, cached=cached, completion=ct, target_s=round(target, 2))
                if status == 200 and target > up:
                    time.sleep(target - up)
            except (ValueError, AttributeError, TypeError):
                pass
            rec['total_s'] = round(time.time() - t0, 2)
            if a.save:  # the full exchange, for exact replays, caching and training data
                with LOCK:
                    n = next(COUNTER)
                try:
                    full = {'path': self.path, 'status': status, 't0': t0, 'timing': rec,
                            'request': json.loads(body), 'response': json.loads(data)}
                except ValueError:
                    full = {'path': self.path, 'status': status, 't0': t0, 'timing': rec,
                            'request_raw': body.decode('utf-8', 'replace'), 'response_raw': data.decode('utf-8', 'replace')}
                with gzip.open(os.path.join(a.save, f'{int(t0 * 1000)}_{n:06d}.json.gz'), 'wt') as f:
                    json.dump(full, f, ensure_ascii=False)
            if a.log:
                with LOCK, open(a.log, 'a') as f:
                    f.write(json.dumps(rec) + '\n')
            self._send(status, headers, data)

        def log_message(self, *args):
            pass

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8004)
    ap.add_argument('--upstream', default='http://127.0.0.1:8002')
    ap.add_argument('--decode', type=float, default=12.0, help='target decode tokens/s per stream')
    ap.add_argument('--prefill', type=float, default=1500.0, help='target prefill tokens/s for uncached prompt')
    ap.add_argument('--log', default='')
    ap.add_argument('--save', default='', help='directory: every request and response as <epoch_ms>_<n>.json.gz')
    ap.add_argument('--logprobs', type=int, default=0,
                    help='add logprobs with this many top alternatives to every chat request (does not change sampling)')
    ap.add_argument('--api-key-env', default='', help='env var holding the upstream API key (hosted APIs)')
    ap.add_argument('--model', default='', help='rewrite the request model (hosted APIs)')
    ap.add_argument('--strip', default='', help='comma-separated request fields to drop (hosted APIs)')
    a = ap.parse_args()
    a.upstream = a.upstream.rstrip('/')
    a.strip = [x for x in a.strip.split(',') if x]
    if a.save:
        os.makedirs(a.save, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', a.port), make_handler(a))
    print(f'slow proxy :{a.port} -> {a.upstream} at decode {a.decode} tok/s, prefill {a.prefill} tok/s', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
