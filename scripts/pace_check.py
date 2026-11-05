"""Load check before a Kaggle-speed eval: N concurrent single-stream requests; prints per-stream tok/s and exits 1 when
below --min (the slow proxy can only slow a server down: a server slower than the target speed skews the eval).

    python3 scripts/pace_check.py --server http://127.0.0.1:8002 --streams 8 --min 14
"""
import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--server', default='http://127.0.0.1:8002')
ap.add_argument('--streams', type=int, default=8)
ap.add_argument('--tokens', type=int, default=300)
ap.add_argument('--min', type=float, default=14.0)
ap.add_argument('--context', type=int, default=10000, help='prompt tokens per request (agent prompts are 5-14k; '
                'short prompts overstate the speed: E048 L40S 19.7 tok/s on short prompts, ~10 in the eval)')
a = ap.parse_args()


def one(i):
    filler = ' '.join(f'line{i}_{k} = compute(value_{k}, factor={k % 7})' for k in range(a.context // 9))
    b = {'model': 'gemma-4-31b-it-qat-w4a16-ct', 'messages': [{'role': 'user', 'content': filler + f'\n\nWrite {a.tokens} '
         f'words about river number {i}.'}], 'max_tokens': a.tokens, 'temperature': 0.7, 'ignore_eos': True,
         'chat_template_kwargs': {'enable_thinking': False}}
    post = lambda body: json.load(urllib.request.urlopen(urllib.request.Request(  # noqa: E731
        a.server + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}),
        timeout=600))
    post(dict(b, max_tokens=1))   # warm the prefix cache: in an agent run most of each prompt is cached already
    t = time.time()
    r = post(b)
    return r['usage']['completion_tokens'] / (time.time() - t)


with concurrent.futures.ThreadPoolExecutor(a.streams) as ex:
    rates = list(ex.map(one, range(a.streams)))
low = min(rates)
print(f'{a.streams} streams: per-stream tok/s min {low:.1f} mean {sum(rates) / len(rates):.1f}')
sys.exit(0 if low >= a.min else 1)
