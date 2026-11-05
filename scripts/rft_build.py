"""Stage-1 self-RFT dataset (research/notes/9-lora-plan.md §3, §6): captured agent requests -> episodes -> selected
passing episodes -> token ids + loss masks rendered exactly as the scorer's vLLM renders a chat request.

    .venv/bin/python scripts/rft_build.py                      # all steps, defaults below
    .venv/bin/python scripts/rft_build.py --selftest-only      # only the rendering self-test

Inputs:  data/rft/raw/R001/*.json (scripts/llm_proxy.py captures), results/R001_rft/r*/task_results.jsonl,
         data/competition/tasks.jsonl, experiments/rft_pool.txt, data/rft/tokenizer/ (HF tokenizer files of
         google/gemma-4-31B-it-qat-w4a16-ct, no weights).
Outputs: data/rft/episodes.jsonl      every episode (task x round) with metadata and its full transcript
         data/rft/train.jsonl         the selected episodes (<= --per-task passing runs per task, fewest tool calls)
         data/rft/train_tokens.jsonl  one row per training sequence: input_ids, labels (-100 = masked), task, ...
         data/rft/build_report.json   the numbers printed at the end

Rendering = vLLM 0.19.1 (Kaggle's wheel) for /v1/chat/completions with an HF tokenizer:
  1. messages -> "conversation" (vllm/entrypoints/chat_utils.py _parse_chat_message_content + _postprocess_messages):
     content str -> [{"type": "text", "text": ...}] because the content format is auto-detected as "openai" for this
     template (it loops over message['content']); None -> []; assistant keeps tool_calls and `reasoning` (copied to
     reasoning_content; a message with only reasoning_content loses it); tool keeps tool_call_id; arguments json.loads.
     The "openai" format is where the extra space before <|tool> comes from: the template renders a list-valued
     system content as `item['text'] | trim + ' '`.
  2. tools -> ChatCompletionToolsParam.model_dump(): {"type", "function": {"name", "description", "parameters"}}.
  3. tokenizer.apply_chat_template(conversation, tools, tokenize=False, add_generation_prompt=True,
     continue_final_message=False, enable_thinking=...) -> text (BOS comes from the template), then
     tokenizer(text, add_special_tokens=False) (vLLM's chat default).

Training sequences and masks:
  The template renders a whole agent episode as ONE model turn: thinking (<|channel>thought\n...\n<channel|>),
  <|tool_call>call:...<tool_call|>, then the tool responses <|tool_response>...<tool_response|> inside the same turn.
  For request k let P_k = its prompt (as rendered for the request) and F_k = render(messages_k + response_k, no
  generation prompt). The model's turn T_k = F_k[len(P_k):] (through the stop token: <|tool_response> after a tool
  call, <turn|> after a text answer). A sequence S (= F_j of some later request j) trains turn k iff S starts with
  P_k + T_k, i.e. the model saw exactly that prefix at inference and its turn is rendered in S as it will appear in its
  own history. Past thinking is dropped by the template before the last user message, so a harness nudge (an extra
  user message) breaks the prefix: turns before it are trained in an earlier sequence (F of the last request before
  the nudge). Masked (kept as context, no loss): turns cut by max tokens (finish_reason length), turns whose call
  was rejected by ADK (tool result {"error", "error_code"}: garbled skill names, missing args), turns without
  thinking (their sampled tokens differ from the canonical render), and everything that is not a model turn.
"""

import argparse
import collections
import copy
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT_PART = 'text'


# ------------------------------------------------------------------ vLLM-faithful rendering

def detect_content_format(template):
    """vllm/renderers/hf.py _detect_content_format: 'openai' if the template loops over message['content']."""
    import jinja2.nodes as N
    from transformers.utils.chat_template_utils import _compile_jinja_template

    ast = _compile_jinja_template(template).environment.parse(template)

    def var(n, name):
        return isinstance(n, N.Name) and n.ctx == 'load' and n.name == name

    def attr(n, name, key):
        if isinstance(n, N.Getitem):
            return var(n.node, name) and isinstance(n.arg, N.Const) and n.arg.value == key
        if isinstance(n, N.Getattr):
            return var(n.node, name) and n.attr == key
        return False

    def var_or_elems(n, name, key=None):
        if isinstance(n, N.Filter):
            return n.node is not None and var_or_elems(n.node, name, key)
        if isinstance(n, N.Test):
            return var_or_elems(n.node, name, key)
        if isinstance(n, N.Getitem) and isinstance(n.arg, N.Slice):
            return var_or_elems(n.node, name, key)
        return attr(n, name, key) if key else var(n, name)

    names, queue = ['messages'], collections.deque(['messages'])
    while queue:
        cur = queue.popleft()
        for a in ast.find_all(N.Assign):
            if var_or_elems(a.node, cur) and isinstance(a.target, N.Name):
                names.append(a.target.name)
                if a.target.name != cur:
                    queue.append(a.target.name)
    msg_vars = []
    for loop in ast.find_all(N.For):
        if any(var_or_elems(loop.iter, v) for v in names) and isinstance(loop.target, N.Name):
            msg_vars.append(loop.target.name)
    for loop in ast.find_all(N.For):
        if any(var_or_elems(loop.iter, v, 'content') for v in msg_vars):
            return 'openai'
    return 'string'


class Renderer:
    def __init__(self, tok_dir):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(tok_dir))
        self.template = self.tok.get_chat_template(None, tools=[{}])
        self.fmt = detect_content_format(self.template)

    def conversation(self, messages):
        conv = []
        for m in messages:
            role, content, reasoning = m['role'], m.get('content'), m.get('reasoning')
            parts = [] if content is None else [{'type': 'text', 'text': content}] if isinstance(content, str) else \
                [{'type': 'text', 'text': p['text']} if isinstance(p, dict) else {'type': 'text', 'text': p}
                 for p in content if isinstance(p, str) or (p.get('type') == 'text' and p.get('text') is not None)]
            if self.fmt == 'openai':
                out = {'role': role, 'content': parts}
            else:
                out = {'role': role, 'content': '\n'.join(p['text'] for p in parts)}
            if role == 'assistant':
                if m.get('tool_calls') is not None:
                    out['tool_calls'] = copy.deepcopy(list(m['tool_calls']))
                if reasoning is not None:
                    out['reasoning'] = reasoning
                    out['reasoning_content'] = reasoning
            elif role == 'tool' and 'tool_call_id' in m:
                out['tool_call_id'] = m['tool_call_id']
            if isinstance(m.get('name'), str):
                out['name'] = m['name']
            conv.append(out)
        for m in conv:  # _postprocess_messages
            if m['role'] == 'assistant' and 'tool_calls' in m and isinstance(m['tool_calls'], list):
                if not m['tool_calls']:
                    m.pop('tool_calls')
                    continue
                for item in m['tool_calls']:
                    a = item['function'].get('arguments')
                    if a:
                        if not isinstance(a, (dict, list)):
                            item['function']['arguments'] = json.loads(a)
                    else:
                        item['function']['arguments'] = {}
        return conv

    @staticmethod
    def tool_dicts(tools):
        if tools is None:
            return None
        out = []
        for t in tools:
            f = t['function']
            fd = {'name': f['name'], 'description': f.get('description'), 'parameters': f.get('parameters')}
            fd.update({k: v for k, v in f.items() if k not in fd})
            d = {'type': t.get('type', 'function'), 'function': fd}
            d.update({k: v for k, v in t.items() if k not in d})
            out.append(d)
        return out

    def text(self, messages, tools, gen, enable_thinking=True):
        return self.tok.apply_chat_template(self.conversation(messages), tools=self.tool_dicts(tools),
                                            chat_template=self.template, tokenize=False, add_generation_prompt=gen,
                                            continue_final_message=False, enable_thinking=enable_thinking)

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)['input_ids']


def response_message(resp):
    """The assistant message the harness sends back for this response (verified identical in R001)."""
    m = resp['choices'][0]['message']
    out = {'role': 'assistant', 'content': m.get('content')}
    if m.get('tool_calls'):
        out['tool_calls'] = [{'type': 'function', 'id': c['id'],
                              'function': {'name': c['function']['name'], 'arguments': c['function']['arguments']}}
                             for c in m['tool_calls']]
    r = m.get('reasoning') or m.get('reasoning_content')
    if r:
        out['reasoning'] = out['reasoning_content'] = r
    return out


# ------------------------------------------------------------------ loading and grouping

def load_requests(raw_dir):
    recs = []
    for f in sorted(Path(raw_dir).glob('*.json')):
        d = json.loads(f.read_text())
        ms, n = f.stem.split('_')
        d['t'], d['n'], d['file'] = int(ms) / 1000, int(n), f.name
        recs.append(d)
    recs.sort(key=lambda d: (d['t'], d['n']))
    return recs


def round_ends(results_dir, rounds, override):
    if override:
        return [datetime.fromisoformat(x).timestamp() for x in override.split(',')]
    return [os.path.getmtime(Path(results_dir) / f'r{r}' / 'task_results.jsonl') for r in rounds]


def build_episodes(a):
    pool = [x.strip() for x in open(a.pool) if x.strip() and not x.startswith('#')]
    tasks = {}
    for line in open(a.tasks):
        t = json.loads(line)
        tasks[t['instance_id']] = t['problem_statement'].strip()
    rounds = sorted(int(p.name[1:]) for p in Path(a.results).glob('r[0-9]*') if p.is_dir())
    ends = round_ends(a.results, rounds, a.round_ends)
    results = {}
    for r in rounds:
        for line in open(Path(a.results) / f'r{r}' / 'task_results.jsonl'):
            x = json.loads(line)
            results[(r, x['instance_id'])] = x

    def task_of(d):
        u = next((m['content'] for m in d['request']['messages'] if m['role'] == 'user'), '') or ''
        s = d['request']['messages'][0].get('content') or ''
        hits = [i for i in pool if tasks[i] in u or tasks[i] in s] or \
               [i for i in tasks if tasks[i] in u or tasks[i] in s]
        return max(hits, key=lambda i: len(tasks[i])) if hits else None

    def round_of(t):
        return next((r for r, e in zip(rounds, ends) if t <= e + a.round_slack), None)

    recs = load_requests(a.raw)
    groups, unassigned = collections.defaultdict(list), 0
    for d in recs:
        if 'request' not in d:
            unassigned += 1
            continue
        k = (round_of(d['t']), task_of(d))
        if None in k:
            unassigned += 1
            continue
        groups[k].append(d)
    return groups, results, unassigned, len(recs), pool


def norm_asst(m):
    tc = [(c['function']['name'], json.loads(c['function'].get('arguments') or '{}')) for c in m.get('tool_calls') or []]
    return tc, m.get('reasoning') or None, m.get('content') or None


def analyse_episode(reqs, R, stats):
    """Message-level prefix check, turn list, and sequence assignment (see module doc)."""
    ok = [d for d in reqs if d['status'] == 200 and d['response'].get('choices')]
    stats['requests_not_200'] += len(reqs) - len(ok)
    for x, y in zip(ok, ok[1:]):  # message-level prefix property
        ma, mb = x['request']['messages'], y['request']['messages']
        if mb[:len(ma)] != ma:
            stats['msg_prefix_broken'] += 1
        elif len(mb) <= len(ma) or norm_asst(mb[len(ma)]) != norm_asst(response_message(x['response'])):
            stats['msg_response_mismatch'] += 1
        else:
            stats['msg_prefix_ok'] += 1
            if any(m['role'] == 'user' for m in mb[len(ma) + 1:]):
                stats['nudges'] += 1
    turns = []
    for k, d in enumerate(ok):
        req, resp = d['request'], d['response']
        msgs, tools = req['messages'], req.get('tools')
        think = (req.get('chat_template_kwargs') or {}).get('enable_thinking', True)
        P = R.text(msgs, tools, True, think)
        asst = response_message(resp)
        F = R.text(msgs + [asst], tools, False, think)
        ch = resp['choices'][0]
        t = {'k': k, 'file': d['file'], 'P': P, 'F': F, 'finish': ch.get('finish_reason'),
             'prompt_tokens': resp['usage']['prompt_tokens'], 'completion_tokens': resp['usage']['completion_tokens'],
             'n_calls': len(asst.get('tool_calls') or []), 'has_thinking': bool(asst.get('reasoning')),
             'calls': [c['function']['name'] for c in asst.get('tool_calls') or []]}
        if F.startswith(P):
            T = F[len(P):]
            if not asst.get('tool_calls') and T.endswith('<turn|>\n'):
                T = T[:-1]
            t['T'] = T
        else:
            t['T'] = None
        # ADK-rejected call: the next request's tool results for this turn's calls
        t['adk_error'] = False
        if k + 1 < len(ok) and asst.get('tool_calls'):
            ids = {c['id'] for c in asst['tool_calls']}
            for m in ok[k + 1]['request']['messages'][len(msgs) + 1:]:
                if m['role'] == 'tool' and m.get('tool_call_id') in ids:
                    try:
                        j = json.loads(m['content']) if isinstance(m['content'], str) else None
                    except ValueError:
                        j = None
                    if isinstance(j, dict) and 'error' in j and 'error_code' in j:
                        t['adk_error'] = True
        turns.append(t)
    return ok, turns


def mask_reason(t):
    if t['T'] is None or not t['has_thinking']:
        return 'no_thinking'
    if t['finish'] == 'length':
        return 'length'
    if t['adk_error']:
        return 'adk_error'
    return None


def assign_sequences(turns):
    """Sequences (indices j: S = F_j) covering every trainable turn, latest-first; returns {j: [turn ks]}."""
    seqs = {}
    for t in reversed(turns):
        if mask_reason(t):
            continue
        target = t['P'] + t['T']
        j = next((j for j in sorted(seqs, reverse=True) if turns[j]['F'].startswith(target)), None)
        if j is None:
            j = next(j for j in range(len(turns) - 1, t['k'] - 1, -1) if turns[j]['F'].startswith(target))
            seqs[j] = []
        seqs[j].append(t['k'])
    return {j: sorted(ks) for j, ks in seqs.items()}


def tokenize_sequence(R, S, spans):
    """Token ids for S and labels: tokens inside the (char) spans are trained. Segments are tokenized separately
    (boundaries sit at special tokens / turn starts); the concatenation is checked against tokenizing S whole."""
    ids, labels, pos = [], [], 0
    for s, e in sorted(spans):
        if s > pos:
            x = R.ids(S[pos:s])
            ids += x
            labels += [-100] * len(x)
        x = R.ids(S[s:e])
        ids += x
        labels += x
        pos = e
    if pos < len(S):
        x = R.ids(S[pos:])
        ids += x
        labels += [-100] * len(x)
    whole = R.ids(S)
    return ids, labels, ids == whole


# ------------------------------------------------------------------ steps

def selftest(R, groups, a, report):
    """Every captured 200 request re-rendered: token count must equal the server's usage.prompt_tokens."""
    n = exact = 0
    worst = []
    t0 = time.time()
    for k, reqs in sorted(groups.items()):
        for d in reqs:
            if d['status'] != 200 or not d['response'].get('choices'):
                continue
            req = d['request']
            think = (req.get('chat_template_kwargs') or {}).get('enable_thinking', True)
            ids = R.ids(R.text(req['messages'], req.get('tools'), True, think))
            pt = d['response']['usage']['prompt_tokens']
            n += 1
            exact += len(ids) == pt
            if len(ids) != pt:
                worst.append((len(ids) - pt, d['file']))
            if a.selftest_limit and n >= a.selftest_limit:
                break
        if a.selftest_limit and n >= a.selftest_limit:
            break
    report['selftest'] = {'requests': n, 'exact_token_count': exact, 'mismatches': len(worst),
                          'mismatch_examples': sorted(worst, key=lambda x: -abs(x[0]))[:10],
                          'content_format': R.fmt, 'seconds': round(time.time() - t0)}
    print(f'self-test: {exact}/{n} requests re-rendered to exactly usage.prompt_tokens '
          f'(content format {R.fmt!r}, {time.time() - t0:.0f}s)')
    for diff, f in worst[:10]:
        print(f'   mismatch {diff:+d} tokens: {f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', default=str(ROOT / 'data/rft/raw/R001'))
    ap.add_argument('--results', default=str(ROOT / 'results/R001_rft'))
    ap.add_argument('--tasks', default=str(ROOT / 'data/competition/tasks.jsonl'))
    ap.add_argument('--pool', default=str(ROOT / 'experiments/rft_pool.txt'))
    ap.add_argument('--tokenizer', default=str(ROOT / 'data/rft/tokenizer'))
    ap.add_argument('--out', default=str(ROOT / 'data/rft'))
    ap.add_argument('--round-ends', default='', help='comma-separated ISO times; default: task_results.jsonl mtimes')
    ap.add_argument('--round-slack', type=float, default=0.0, help='seconds added to each round end')
    ap.add_argument('--per-task', type=int, default=3)
    ap.add_argument('--max-len', type=int, default=32768)
    ap.add_argument('--selftest-limit', type=int, default=0, help='0 = all requests')
    ap.add_argument('--selftest-only', action='store_true')
    ap.add_argument('--no-selftest', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    groups, results, unassigned, n_req, pool = build_episodes(a)
    print(f'{n_req} requests -> {len(groups)} episodes ({unassigned} unassigned)')
    R = Renderer(a.tokenizer)
    if not a.no_selftest:
        selftest(R, groups, a, report)
    if a.selftest_only:
        json.dump(report, open(out / 'build_report.json', 'w'), indent=1)
        return

    # ---- step 1: episodes
    stats = collections.Counter()
    episodes = []
    for (r, task), reqs in sorted(groups.items()):
        ok, turns = analyse_episode(reqs, R, stats)
        res = results.get((r, task), {})
        if not ok:
            stats['episodes_without_200'] += 1
            continue
        last = ok[-1]
        transcript = last['request']['messages'] + [response_message(last['response'])]
        masks = collections.Counter(mask_reason(t) or 'trained' for t in turns)
        stats.update({f'turns_{k}': v for k, v in masks.items()})
        if not res:
            stats['episodes_without_result'] += 1
        ep = {'id': f'R001_r{r}_{task}', 'round': r, 'task': task, 'resolved': res.get('resolved'),
              'tool_calls': res.get('tool_calls'), 'total_llm_calls': res.get('total_llm_calls'),
              'duration_seconds': res.get('duration_seconds'), 'error': res.get('error'),
              'requests': len(reqs), 'requests_ok': len(ok), 'turns': len(turns),
              'user_messages': sum(m['role'] == 'user' for m in transcript),
              'turn_info': [{k: t[k] for k in ('file', 'finish', 'prompt_tokens', 'completion_tokens', 'n_calls',
                                              'calls', 'has_thinking', 'adk_error')} | {'mask': mask_reason(t)}
                            for t in turns],
              'last_prompt_tokens': last['response']['usage']['prompt_tokens'],
              'final_tokens': last['response']['usage']['total_tokens'],
              'tools': last['request'].get('tools'),
              'chat_template_kwargs': last['request'].get('chat_template_kwargs'),
              'messages': transcript}
        ep['_turns'] = turns
        episodes.append(ep)
    with open(out / 'episodes.jsonl', 'w') as f:
        for ep in episodes:
            f.write(json.dumps({k: v for k, v in ep.items() if k != '_turns'}, ensure_ascii=False) + '\n')

    by_round = collections.defaultdict(lambda: [0, 0])
    by_task = collections.defaultdict(lambda: [0, 0])
    for ep in episodes:
        by_round[ep['round']][0] += 1
        by_round[ep['round']][1] += bool(ep['resolved'])
        by_task[ep['task']][0] += 1
        by_task[ep['task']][1] += bool(ep['resolved'])
    npass = sum(bool(e['resolved']) for e in episodes)
    print(f'\n== episodes: {len(episodes)}, passing {npass} ({npass / max(len(episodes), 1):.0%})')
    print('   per round (episodes/passing): ' + ', '.join(f'r{r} {n}/{p}' for r, (n, p) in sorted(by_round.items())))
    solved = sorted(t for t, (n, p) in by_task.items() if p)
    print(f'   tasks with >= 1 pass: {len(solved)}/{len(by_task)}; per task (passing/episodes): ' +
          ', '.join(f'{t} {p}/{n}' for t, (n, p) in sorted(by_task.items())))
    print('   checks: ' + ', '.join(f'{k} {v}' for k, v in sorted(stats.items())))
    mism = [e['id'] for e in episodes if e['tool_calls'] is not None and
            e['tool_calls'] != sum(t['n_calls'] for t in e['_turns'])]
    print(f'   tool_calls (task_results) != calls in captured responses: {len(mism)} episodes')
    report['episodes'] = {'n': len(episodes), 'passing': npass, 'unassigned_requests': unassigned,
                          'per_round': {r: {'episodes': n, 'passing': p} for r, (n, p) in sorted(by_round.items())},
                          'per_task': {t: {'episodes': n, 'passing': p} for t, (n, p) in sorted(by_task.items())},
                          'checks': dict(stats), 'tool_call_count_mismatch': mism}

    # ---- step 2: selection
    sel = []
    for task in sorted(by_task):
        cand = [e for e in episodes if e['task'] == task and e['resolved']]
        cand.sort(key=lambda e: (e['tool_calls'] if e['tool_calls'] is not None else 10 ** 6,
                                 e['duration_seconds'] or 10 ** 6))
        sel += cand[:a.per_task]
    with open(out / 'train.jsonl', 'w') as f:
        for ep in sel:
            f.write(json.dumps({k: v for k, v in ep.items() if k != '_turns'}, ensure_ascii=False) + '\n')

    # ---- step 3: tokens + masks
    rows, seq_stats = [], collections.Counter()
    for ep in sel:
        turns = ep['_turns']
        seqs = assign_sequences(turns)
        seq_stats['sequences'] += len(seqs)
        seq_stats['extra_sequences_from_nudges'] += len(seqs) - 1
        for j, ks in sorted(seqs.items()):
            S = turns[j]['F']
            spans = [(len(turns[k]['P']), len(turns[k]['P']) + len(turns[k]['T'])) for k in ks]
            ids, labels, same = tokenize_sequence(R, S, spans)
            seq_stats['segment_tokenization_equal'] += same
            # trained-token count vs. the server's completion_tokens (re-rendered canonical form vs. sampled)
            for k, (s, e) in zip(ks, spans):
                seq_stats['turn_tokens_canonical'] += len(R.ids(S[s:e]))
                seq_stats['turn_tokens_sampled'] += turns[k]['completion_tokens']
            if len(ids) > a.max_len:
                seq_stats['dropped_too_long'] += 1
                continue
            rows.append({'id': f'{ep["id"]}_s{j}', 'episode': ep['id'], 'task': ep['task'], 'round': ep['round'],
                         'input_ids': ids, 'labels': labels, 'n_tokens': len(ids),
                         'n_trained': sum(x != -100 for x in labels), 'trained_turns': len(ks),
                         'segment_tokenization_equal': same})
    with open(out / 'train_tokens.jsonl', 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')

    turns_all = sum(len(e['_turns']) for e in sel)
    trained_turns = sum(r['trained_turns'] for r in rows)
    lens = [r['n_tokens'] for r in rows]
    tr = [r['n_trained'] for r in rows]
    mask_counts = collections.Counter(mask_reason(t) or 'trained' for e in sel for t in e['_turns'])
    print(f'\n== selection: {len(sel)} episodes over {len({e["task"] for e in sel})} tasks '
          f'(<= {a.per_task} passing per task, fewest tool calls, ties: shorter duration)')
    print(f'   assistant turns {turns_all}: ' + ', '.join(f'{k} {v}' for k, v in mask_counts.most_common()))
    print(f'   tool calls per selected episode: median {statistics.median([e["tool_calls"] for e in sel])}, '
          f'range {min(e["tool_calls"] for e in sel)}-{max(e["tool_calls"] for e in sel)}')
    if rows:
        print(f'\n== tokens: {len(rows)} sequences ({seq_stats["extra_sequences_from_nudges"]} extra from nudges, '
              f'{seq_stats["dropped_too_long"]} dropped > {a.max_len}), trained turns {trained_turns}')
        print(f'   length max {max(lens)}, mean {statistics.mean(lens):.0f}, median {statistics.median(lens):.0f}; '
              f'total {sum(lens)} tokens, trained {sum(tr)} ({sum(tr) / sum(lens):.1%})')
        print(f'   segment-wise tokenization == whole-text tokenization: '
              f'{seq_stats["segment_tokenization_equal"]}/{seq_stats["sequences"]}')
        print(f'   trained-turn tokens (canonical re-render) {seq_stats["turn_tokens_canonical"]} vs sampled '
              f'completion_tokens {seq_stats["turn_tokens_sampled"]}')
    report['selection'] = {'episodes': len(sel), 'tasks': len({e['task'] for e in sel}), 'per_task': a.per_task,
                           'ids': [e['id'] for e in sel], 'assistant_turns': turns_all,
                           'turn_masks': dict(mask_counts)}
    report['tokens'] = {'sequences': len(rows), 'trained_turns': trained_turns,
                        'max_len': max(lens) if lens else 0, 'mean_len': statistics.mean(lens) if lens else 0,
                        'total_tokens': sum(lens), 'trained_tokens': sum(tr), **dict(seq_stats)}
    json.dump(report, open(out / 'build_report.json', 'w'), indent=1)
    print(f'\nwrote {out}/episodes.jsonl, train.jsonl, train_tokens.jsonl, build_report.json')


if __name__ == '__main__':
    sys.exit(main())
