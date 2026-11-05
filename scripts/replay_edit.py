"""Counterfactual replay: rebuild a trace's conversation up to its first failed edit_file,
then resample that step N times under different system prompts and count escaped edits.

    python3 scripts/replay_edit.py results/E004_v1_20m/rich_3043 submissions/v1 submissions/v2 -n 6
"""

import argparse
import glob
import json
import pathlib
import re
import urllib.request

BASE = 'http://127.0.0.1:8001/v1'
TOOL_SCHEMAS = {
    'run_command': {'command': {'type': 'string'}},
    'read_file': {'filepath': {'type': 'string'}, 'start_line': {'type': 'integer'}, 'end_line': {'type': 'integer'}},
    'edit_file': {'filepath': {'type': 'string'}, 'old_string': {'type': 'string'}, 'new_string': {'type': 'string'}},
    'get_status': {},
    'submit_patch': {},
    'search_similar_code': {'query': {'type': 'string'}},
    'get_code_neighbors': {'node': {'type': 'string'}},
}


def tools():
    return [{'type': 'function', 'function': {'name': n, 'description': n, 'parameters': {
        'type': 'object', 'properties': p, 'required': [k for k in p if k in ('command', 'filepath', 'old_string', 'new_string', 'query', 'node')]}}}
        for n, p in TOOL_SCHEMAS.items()]


def rebuild(trace_path):
    """Messages up to (not including) the first failed edit_file, plus the user prompt text."""
    t = json.load(open(trace_path))
    msgs, n = [], 0
    for s in t['steps']:
        src = s.get('source')
        if src == 'user':
            msgs.append({'role': 'user', 'content': str(s.get('message'))})
        for tc in s.get('tool_calls') or []:
            obs = str((s.get('observation') or {}).get('content'))
            if tc.get('function_name') == 'edit_file' and 'not found' in obs:
                return msgs
            n += 1
            cid = f'c{n}'
            msgs.append({'role': 'assistant', 'content': '', 'tool_calls': [{'id': cid, 'type': 'function', 'function': {
                'name': tc.get('function_name'), 'arguments': json.dumps(tc.get('arguments') or {})}}]})
            msgs.append({'role': 'tool', 'tool_call_id': cid, 'content': obs})
    raise SystemExit('no failed edit in trace')


def sample(system, msgs, seed):
    body = {'model': 'gemma-4-31b-it-qat-w4a16-ct', 'messages': [{'role': 'system', 'content': system}] + msgs,
            'tools': tools(), 'temperature': 0.2, 'top_p': 0.95, 'seed': seed, 'max_tokens': 6000}
    req = urllib.request.Request(f'{BASE}/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    m = json.load(urllib.request.urlopen(req, timeout=1200))['choices'][0]['message']
    calls = m.get('tool_calls') or []
    if not calls:
        return 'no-call'
    fn = calls[0]['function']
    if fn['name'] != 'edit_file':
        return f"other:{fn['name']}"
    a = json.loads(fn['arguments'])
    old = a.get('old_string', '')
    return 'ESCAPED' if re.search(r'\\n|\\"', old) else 'clean'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('task_dir')
    ap.add_argument('submissions', nargs='+')
    ap.add_argument('-n', type=int, default=6)
    args = ap.parse_args()
    trace = glob.glob(f'{args.task_dir}/traces/trace_*.json')[0]
    msgs = rebuild(trace)
    print(f'replaying {trace}: {len(msgs)} messages before the failed edit')
    for sub in args.submissions:
        system = (pathlib.Path(sub) / 'prompts/system.md').read_text()
        res = [sample(system, msgs, seed) for seed in range(args.n)]
        print(f'{sub:28} {res}')
