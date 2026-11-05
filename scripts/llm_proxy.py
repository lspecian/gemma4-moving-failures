"""Logging proxy between the harness and the model server: records every chat request and its response.

Used to collect training data with the exact messages/tools the harness sends (adk_submission 0.2.12 serialization,
past thinking included), instead of reconstructing them from traces.

    python3 scripts/llm_proxy.py --port 8003 --upstream http://127.0.0.1:8002 --out data/rft/raw/R001
    OPENAI_BASE_URL=http://127.0.0.1:8003/v1 swegemma eval ...

Each POST is written as <out>/<epoch_ms>_<n>.json = {"path", "request", "response", "seconds"}.
"""

import argparse
import http.server
import itertools
import json
import os
import threading
import time
import urllib.request

COUNTER = itertools.count(1)
LOCK = threading.Lock()


def make_handler(upstream, out):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def _forward(self, body=None):
            req = urllib.request.Request(upstream + self.path, data=body, method=self.command,
                                         headers={k: v for k, v in self.headers.items()
                                                  if k.lower() not in ('host', 'content-length', 'connection')})
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    status, headers, data = r.status, r.headers.items(), r.read()
            except urllib.error.HTTPError as e:
                status, headers, data = e.code, e.headers.items(), e.read()
            self.send_response(status)
            for k, v in headers:
                if k.lower() not in ('transfer-encoding', 'content-length', 'connection'):
                    self.send_header(k, v)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return status, data

        def do_GET(self):
            self._forward()

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            t0 = time.time()
            status, data = self._forward(body)
            with LOCK:
                n = next(COUNTER)
            rec = {'path': self.path, 'status': status, 'seconds': round(time.time() - t0, 3)}
            try:
                rec['request'] = json.loads(body)
                rec['response'] = json.loads(data)
            except ValueError:
                rec['request_raw'] = body.decode('utf-8', 'replace')[:2000]
                rec['response_raw'] = data.decode('utf-8', 'replace')[:2000]
            with open(os.path.join(out, f'{int(t0 * 1000)}_{n:06d}.json'), 'w') as f:
                json.dump(rec, f, ensure_ascii=False)

        def log_message(self, *a):
            pass

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8003)
    ap.add_argument('--upstream', default='http://127.0.0.1:8002')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', a.port), make_handler(a.upstream.rstrip('/'), a.out))
    print(f'proxy :{a.port} -> {a.upstream}, logging to {a.out}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
