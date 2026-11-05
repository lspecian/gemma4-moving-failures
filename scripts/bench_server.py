"""Benchmark an OpenAI-compatible server on an agent-shaped workload.

Turn 1: ~12k-token prompt (a real source file) -> measures cold prefill + generation at long context.
Turn 2: same history + reply + ~1k new tokens -> measures prefix-cache reuse, like an agent's next step.

    python3 scripts/bench_server.py http://127.0.0.1:8001/v1
"""

import json
import pathlib
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = 'gemma-4-31b-it-qat-w4a16-ct'
SRC = pathlib.Path('/tmp/bench_src.py')


def stream(messages, max_tokens=300):
    body = json.dumps({
        'model': MODEL, 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0,
        'stream': True, 'stream_options': {'include_usage': True},
    }).encode()
    req = urllib.request.Request(f'{BASE}/chat/completions', body, {'Content-Type': 'application/json'})
    t0 = time.time()
    first = None
    pieces = 0
    text = []
    usage = {}
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith('data:') or line.endswith('[DONE]'):
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get('usage') or usage
            for ch in chunk.get('choices') or []:
                d = ch.get('delta') or {}
                piece = (d.get('content') or '') + (d.get('reasoning_content') or d.get('reasoning') or '')
                if piece:
                    first = first or time.time()
                    pieces += 1
                    text.append(d.get('content') or '')
    end = time.time()
    gen_tokens = usage.get('completion_tokens') or pieces
    return {
        'ttft_s': round(first - t0, 1),
        'prompt_tokens': usage.get('prompt_tokens'),
        'gen_tokens': gen_tokens,
        'gen_tok_s': round(gen_tokens / max(end - first, 1e-6), 1),
        'total_s': round(end - t0, 1),
    }, ''.join(text)


if __name__ == '__main__':
    src = SRC.read_text()
    msgs = [{'role': 'user', 'content': f'Here is a file:\n```python\n{src}\n```\nList its 5 most important classes and why.'}]
    r1, reply = stream(msgs)
    print('turn1 (cold, ~12k ctx):', r1)
    msgs += [
        {'role': 'assistant', 'content': reply or 'ok'},
        {'role': 'user', 'content': 'Now explain how rendering works, step by step. ' + src[:3000]},
    ]
    r2, _ = stream(msgs)
    print('turn2 (cached prefix):  ', r2)
