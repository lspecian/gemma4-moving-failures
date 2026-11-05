"""Mine the raw request/response logs of slow_proxy.py --save (--logprobs adds token_ids): where generation time goes.

Per run directory (data/raw/<run>/*.json.gz) it reports:
  - agent calls vs ADK compaction-summary calls (requests without tools whose prompt starts like LlmEventSummarizer's)
    and their time at the target speed (the proxy's target_s);
  - per agent call, the decoded raw output split into thinking (<|channel>...<channel|>), visible text, and the tool
    call (<|tool_call>...<tool_call|>), in tokens; how often thinking hits the budget (thinking_token_budget) and how
    often the reply stops at max_tokens (finish_reason length);
  - raw tool calls that ADK would reject: brackets around names, file_path/skill_name inside args, unknown scripts —
    shown as the model wrote them (before vLLM's parser);
  - the summaries' lengths and a sample.

    .venv/bin/python scripts/raw_report.py data/raw/E048_v17b_kaggleslow data/raw/E049_v18c_kaggleslow [--examples 5]
"""

import argparse
import collections
import glob
import gzip
import json
import os
import re
import statistics as S

TOKENIZER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'rft', 'tokenizer', 'tokenizer.json')
SUMMARY_HEAD = 'The following is a conversation history between a user and an AI agent'
SCRIPTS = {'locate.py', 'show.py', 'search.py', 'outline.py', 'callers.py', 'tests_for.py', 'history.py', 'edit.py',
           'create.py', 'run.py', 'check_patch.py', 'status.py', 'hypothesis.py', 'repro.py', 'reproduce_issue.py',
           'reproduce_bug.py'}


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOKENIZER)


def segments(text):
    """(thinking, visible, calls) of a raw generation."""
    think = ''.join(re.findall(r'<\|channel>(.*?)(?:<channel\|>|$)', text, re.S))
    calls = re.findall(r'<\|tool_call>(.*?)(?:<tool_call\|>|$)', text, re.S)
    visible = re.sub(r'<\|channel>.*?(?:<channel\|>|$)|<\|tool_call>.*?(?:<tool_call\|>|$)|<\|tool_response>|<turn\|>',
                     '', text, flags=re.S)
    return think, visible, calls


def _top_keys(call):
    """Top-level keys of the run_skill_script argument object and the args object's own keys (strings blanked)."""
    body = re.sub(r'<\|"\|>.*?<\|"\|>', '""', call, flags=re.S)
    i = body.find('{')
    if i < 0:
        return [], []
    depth, key, top, inner, cur = 0, '', [], [], None
    for ch in body[i:]:
        if ch == '{':
            depth += 1
            if depth == 2 and key.strip().endswith('args'):
                cur = inner
            key = ''
            continue
        if ch == '}':
            depth -= 1
            if depth == 1:
                cur = None
            key = ''
            continue
        if ch == ',':
            key = ''
            continue
        if ch == ':':
            k = key.strip()
            if depth == 1:
                top.append(k)
            elif depth == 2 and cur is not None:
                inner.append(k)
            key = ''
            continue
        key += ch
    return top, inner


def call_problem(call):
    """Why ADK would reject a raw call (Gemma 4 syntax: call:run_skill_script{args:{...},file_path:<|"|>x<|"|>,...})."""
    if not call.startswith('call:run_skill_script'):
        name = call.split('{', 1)[0].replace('call:', '')
        return None if name in ('submit_patch', 'get_status') else f'tool {name}'
    if '\u300c' in call or '\u300d' in call:
        return 'brackets'
    top, inner = _top_keys(call)
    if 'file_path' in inner or 'skill_name' in inner:
        return 'nested in args'
    if 'file_path' not in top or 'skill_name' not in top:
        return 'missing file_path/skill_name'
    sk = re.search(r'skill_name:(?:<\|"\|>)?([^,}<]*)', call)
    fp = re.search(r'file_path:(?:<\|"\|>)?([^,}<]*)', call)
    if sk and sk.group(1).strip() != 'repo-tools':
        return f'skill {sk.group(1).strip()[:30]!r}'
    if fp and fp.group(1).strip() not in SCRIPTS:
        return f'script {fp.group(1).strip()[:30]!r}'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--examples', type=int, default=4)
    a = ap.parse_args()
    tok = load_tokenizer()
    for d in a.dirs:
        files = sorted(glob.glob(os.path.join(d, '*.json.gz')))
        agent, summ = [], []
        problems = collections.Counter()
        examples = collections.defaultdict(list)
        for f in files:
            try:
                r = json.load(gzip.open(f, 'rt'))
            except (OSError, ValueError):
                continue
            req, resp, timing = r.get('request') or {}, r.get('response') or {}, r.get('timing') or {}
            msgs = req.get('messages') or []
            first = str((msgs[0] or {}).get('content') if msgs else '')
            ch = (resp.get('choices') or [{}])[0]
            usage = resp.get('usage') or {}
            if not req.get('tools'):
                if first.startswith(SUMMARY_HEAD) or any(SUMMARY_HEAD in str(m.get('content')) for m in msgs):
                    out = (ch.get('message') or {}).get('content') or ''
                    summ.append({'prompt': usage.get('prompt_tokens'), 'out': usage.get('completion_tokens'),
                                 't': timing.get('total_s'), 'text': out})
                continue
            ids = ch.get('token_ids') or []
            raw = tok.decode(ids, skip_special_tokens=False) if ids else ''
            think, visible, calls = segments(raw)
            n = lambda s: len(tok.encode(s, add_special_tokens=False).ids) if s else 0  # noqa: E731
            rec = {'out': usage.get('completion_tokens') or len(ids), 'think': n(think), 'visible': n(visible.strip()),
                   'call': sum(n(c) for c in calls), 'finish': ch.get('finish_reason'), 't': timing.get('total_s'),
                   'budget': req.get('thinking_token_budget'), 'max': req.get('max_tokens') or req.get('max_completion_tokens'),
                   'prompt': usage.get('prompt_tokens')}
            agent.append(rec)
            for c in calls:
                p = call_problem(c)
                if p:
                    key = p.split(' ')[0] if p.startswith(('tool', 'skill', 'script')) else p
                    problems[key] += 1
                    if len(examples[key]) < a.examples:
                        examples[key].append(c[:220])
            if not calls and ids:
                problems['no call (text only)'] += 1
        print(f'\n===== {d}: {len(agent)} agent calls, {len(summ)} compaction summaries')
        if agent:
            tt = sum(x['t'] or 0 for x in agent)
            print(f'agent calls: {tt / 60:.1f} min total at the proxy pace; per call median out {S.median(x["out"] for x in agent)}'
                  f' tok (thinking {S.median(x["think"] for x in agent)}, visible {S.median(x["visible"] for x in agent)},'
                  f' call {S.median(x["call"] for x in agent)}); mean out {S.mean(x["out"] for x in agent):.0f}')
            share = sum(x['think'] for x in agent) / max(sum(x['out'] for x in agent), 1)
            print(f'thinking share of generated tokens: {share:.0%}; thinking at/over the budget '
                  f'(≥{(agent[0]["budget"] or 0) - 8} tok): {sum(1 for x in agent if x["budget"] and x["think"] >= x["budget"] - 8)}'
                  f' / {len(agent)}; finish=length: {sum(1 for x in agent if x["finish"] == "length")}')
            for cap in (384, 256, 128):
                saved = sum(max(0, x['think'] - cap) for x in agent)
                print(f'  thinking cap {cap}: -{saved} tokens ({saved / max(sum(x["out"] for x in agent), 1):.0%} of all generated)')
            print(f'generated tokens by thinking length: <128: {sum(1 for x in agent if x["think"] < 128)}, '
                  f'128-384: {sum(1 for x in agent if 128 <= x["think"] < 384)}, ≥384: {sum(1 for x in agent if x["think"] >= 384)}')
        if summ:
            print(f'summaries: median prompt {S.median(s["prompt"] or 0 for s in summ)} tok, median out '
                  f'{S.median(s["out"] or 0 for s in summ)} tok, median {S.median(s["t"] or 0 for s in summ):.0f} s each, '
                  f'{sum(s["t"] or 0 for s in summ) / 60:.1f} min total')
            print('  sample summary:', summ[0]['text'][:600].replace('\n', ' '))
        print('raw calls ADK would reject:', dict(problems))
        for k, ex in examples.items():
            print(f'  [{k}]')
            for e in ex:
                print('    ', e.replace('\n', '\\n'))


if __name__ == '__main__':
    main()
