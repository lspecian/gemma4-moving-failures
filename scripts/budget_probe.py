"""Thinking-budget probe: does a hard thinking budget stop runaway thinking at the edit step?

Resamples real trace states (same reconstruction as call_probe.py) through vLLM's CHAT endpoint with
`thinking_token_budget` (vLLM ends the thinking after N tokens and the model must answer). Needs a vLLM server started
with --reasoning-config '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}' (otherwise vLLM
rejects the field). adk_submission 0.2.12 forwards thinking_config.thinking_budget as this field.

    .venv/bin/python scripts/budget_probe.py --agent results/E027_v13_vllm/_agent --traces results/E025_v12_vllm,... \
        --out results/P016_budget512 --budget 512 [--states edit] [--n 3] [--max-states 40]
"""

import argparse
import concurrent.futures
import glob
import json
import os
import random
import sys
import urllib.request

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import call_probe as cp  # noqa: E402

SCRIPTS = cp.SCRIPTS


def chat(server, msgs, tools, budget, max_tokens, seed):
    body = {'model': cp.MODEL, 'messages': msgs, 'tools': tools, 'max_completion_tokens': max_tokens,
            'temperature': 0.2, 'top_p': 0.95, 'seed': seed, 'chat_template_kwargs': {'enable_thinking': True}}
    if budget:
        body['thinking_token_budget'] = budget
    req = urllib.request.Request(server.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=900))


def wire(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if m.get('tool_calls'):
            m['tool_calls'] = [dict(tc, function=dict(tc['function'], arguments=json.dumps(
                tc['function']['arguments'], ensure_ascii=False))) for tc in m['tool_calls']]
        if m.get('reasoning_content') and not m.get('reasoning'):
            m['reasoning'] = m['reasoning_content']  # adk_submission 0.2.12: past thinking reaches the model
        out.append(m)
    return out


def label(resp):
    msg = resp['choices'][0]['message']
    calls = msg.get('tool_calls') or []
    if not calls:
        return 'no_call', None
    f = calls[0]['function']
    try:
        a = json.loads(f['arguments'])
    except ValueError:
        return 'bad_args_json', f['name']
    if f['name'] != 'run_skill_script':
        return 'other_tool', f['name']
    fp = str(a.get('file_path', ''))
    ok = a.get('skill_name') == 'repo-tools' and fp.endswith('.py') and fp[:-3] in SCRIPTS
    return ('ok' if ok else 'bad_envelope'), fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', required=True)
    ap.add_argument('--traces', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--budget', type=int, default=0, help='thinking_token_budget (0 = none)')
    ap.add_argument('--max-tokens', type=int, default=1536)
    ap.add_argument('--states', default='edit')
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--max-states', type=int, default=40)
    ap.add_argument('--server', default='http://127.0.0.1:8002/v1')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()

    req = json.load(open(cp.PROBE / 'adk_request.json'))
    sysmsg = req['messages'][0]['content']
    adk_suffix = sysmsg[sysmsg.index('\n\nYou are an agent.'):]
    variant = cp.Variant(a.agent)
    kinds = a.states.split(',')
    todo = []
    for d in a.traces.split(','):
        for f in sorted(glob.glob(f'{d}/r*/traces/trace_*.json')):
            steps, st = cp.states(f, kinds)
            todo += [(f, steps, k, i) for k, i in st]
    random.Random(0).shuffle(todo)
    todo = todo[:a.max_states]
    jobs = [(f, k, i, s, wire(cp.messages(f, steps, i, variant, adk_suffix)))
            for f, steps, k, i in todo for s in range(a.n)]

    def run(job):
        f, k, i, s, msgs = job
        try:
            return job, chat(a.server, msgs, req['tools'], a.budget, a.max_tokens, 1000 + s), None
        except Exception as e:  # noqa: BLE001
            return job, None, repr(e)[:300]

    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, 'samples.jsonl'), 'a')
    tally = {}
    with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
        for (f, k, i, s, _), resp, err in ex.map(run, jobs):
            if err:
                lab, tool, toks, think = 'request_error', None, None, 0
            else:
                lab, tool = label(resp)
                toks = (resp.get('usage') or {}).get('completion_tokens')
                msg = resp['choices'][0]['message']
                think = len(msg.get('reasoning') or msg.get('reasoning_content') or '')
            tally[lab] = tally.get(lab, 0) + 1
            log.write(json.dumps({'trace': f, 'kind': k, 'step': i, 'seed': s, 'label': lab, 'tool': tool,
                                  'tokens': toks, 'think_chars': think, 'error': err}) + '\n')
            log.flush()
            print(f'{k} {os.path.basename(f)[6:-5]:15s} step {i:3d} seed {s}: {lab} {tool or ""} {toks} tok', flush=True)
    toks = sorted(json.loads(x)['tokens'] or 0 for x in open(os.path.join(a.out, 'samples.jsonl')))
    print(json.dumps(tally), 'median tokens', toks[len(toks) // 2] if toks else '-')


if __name__ == '__main__':
    main()
