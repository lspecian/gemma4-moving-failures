"""rft3 step D: the rft3 records (scripts/rft3_states.py schema, thinking filled by step B / scripts/rft3_carry.py)
-> token sequences + loss masks for scripts/lora_train.py, rendered as the Kaggle scorer's vLLM renders the agent's
requests (Kaggle's June chat template, data/rft2/tokenizer_june; research/notes/14-oracle-trajectories.md §5).

    .venv/bin/python scripts/rft3_build.py                       # build with the defaults below (one command, rerun
                                                                 # it when data/rft3/sampled_v20/ has the GPU export)
    .venv/bin/python scripts/rft3_build.py --remainder none      # without the call-only oracle remainder
    .venv/bin/python scripts/rft3_build.py --verify 0            # prefix-consistency check on every record (slower)

Inputs (agent-v20 format, data/rft3/states_v20):
  subset900_filled.jsonl  899 trajectories (442 clean, 225 recovery, 102 repair, 130 relabel); train turns carry step
                          B's thinking (sampling.carried) or wait for the GPU pass (sampling.needs_sampling).
  data/rft3/sampled_v20/  the GPU pass over the not-carried turns (scripts/rft3_sample.py --version v20 sample
                          --records <filled>): its per-turn lines <source>.jsonl (preferred: final = the fallback's
                          kept thinking, else the bootstrap's) or the export <source>.records.jsonl. Absent / partial:
                          those turns stay thinking-masked (call-only), counted as "pending" in the report.
  relabel.jsonl           134 relabel states (130 in the subset + 4 that multi-edit made grounded: call-only).
  oracle_*.jsonl          the other oracle trajectories (no thinking) = the "remainder" (--remainder).
  data/rft2/train_tokens_all.jsonl  rft2's 168 rows (65 episode sequences + 103 format-repair rows), copied as they
                          are: they were rendered from v17b/v18c/v18d runs (older system prompt, scripts and outputs),
                          not v20.

Sequence (one per trajectory). P_k = render(rft3_states.messages_at(rec, k)) with the generation prompt = the exact
request text vLLM builds for assistant turn k (checked: P_k is a string prefix of P_k+1 for every k, and token-wise
when --verify covers the record). Turn k's block P_k+1[len(P_k):] is, in the template's history form,
    [<|channel>thought\\n THINKING \\n<channel|>] <|tool_call>call:...<tool_call|> <|tool_response> RESPONSE
    <tool_response|> [VISIBLE <turn|>\\n]
(the template renders an assistant message's text AFTER its tool responses and closes the turn there). The model
generates a turn as
    [<|channel>thought\\n THINKING <channel|>] [VISIBLE] <|tool_call>call:...<tool_call|><|tool_response>
(no newline before <channel|>, text before the call; checked against the sampled token ids of captured runs:
--raw-check). Context turns (relabel's real earlier turns, repair mistakes) are written in history form; train turns
in the generated form, followed by their RESPONSE (and, if the turn has text, the history trailer VISIBLE <turn|>\\n,
so later turns see what vLLM shows them). Consequence (documented, measured in the report): the context of a later
turn differs from its real request by one newline token per earlier train turn with thinking and by an extra copy of
the visible sentence inside each earlier train turn with text; train turns before the first such turn see their
exact request (prefix-exact). Turns whose thinking is null render without a thought block, exactly as messages_at
(and the GPU pass's contexts) have them.

Loss mask per train turn (context turns and every non-assistant token: no loss):
  thinking trained   the thinking is kept (bootstrap = the policy's own sample, rationalized = verified) and shorter
                     than --think-cap - 8 tokens (rft2's rule: 248 at budget 256) and not sampling.mask_thinking:
                     <|channel>thought ... <channel|> VISIBLE call <|tool_response> are all labelled.
  call (+ text)      thinking present but masked (at the budget): VISIBLE + call + <|tool_response> labelled.
  call body          no thinking (null: not matched by step B, or the GPU pass pending, or remainder): the call from
                     `call:` on is labelled; <|tool_call> itself is not (at that position the model would open its
                     thought channel: training it would teach skipping the thought). The model's own form when it
                     does not think is <|channel>thought\\n<channel|> (seen in captures); it is never trained.
Compaction: the prompt of the last request (P_n-1) must stay under --max-prompt (14,336 = ADK's compaction point):
records at or above it are dropped (counted).

Mix (defaults per note 14 and the step D plan): the 899 subset trajectories once, the 134 relabel states x2
(--relabel-copies), rft2's rows x1 (--rft2-copies), and the oracle remainder (call-only, --remainder):
  none   not used
  mix    (default) all remainder recovery + repair trajectories, plus remainder clean ones (sha1 order) until clean
         trajectories are --clean-share (0.55) of all oracle trajectories in the train split
  all    every remainder trajectory
Validation (lora_train.py --val-tasks): whole repos of the oracle data (--val-repos, ~5%) + rft2's val tasks
(rich_2943, requests_7309; relabel states of those tasks too). Val rows are written once (no copies); remainder
trajectories of val repos are left out (they feed the probe set, scripts/rft3_probe.py).

Outputs (data/rft3/):
  train_tokens.jsonl   one row per sequence: id, task, repo, source, kind, copy, input_ids, labels (-100 = no loss),
                       n_tokens, n_trained (= n_trained_tokens), trained_turns, think_trained_turns, prompt_tokens
                       (last request), segment_tokenization_equal. lora_train.py splits val rows by task.
  val_tasks.txt        the val task ids (lora_train.py --val-tasks "$(paste -sd, data/rft3/val_tasks.txt)")
  build_index.jsonl    per row: id, split, source, per-turn kind/label/mode/think tokens/thinking source (no tokens)
  build_report.json    the numbers printed at the end
Training (H200): python3 scripts/lora_train.py --data data/rft3/train_tokens.jsonl --epochs 1 \\
                     --val-tasks "$(paste -sd, data/rft3/val_tasks.txt)" ... (scripts/lora_train_h200.sh)
"""

import argparse
import collections
import copy
import gzip
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rft3_states as S  # noqa: E402
import rft_build as rb  # noqa: E402  (stdlib at import; transformers only in Renderer)

SOURCES = ('oracle_clean', 'oracle_recovery', 'oracle_repair', 'relabel')
ORACLE = ('oracle_clean', 'oracle_recovery', 'oracle_repair')
THINK_OPEN, THINK_END = '<|channel>thought\n', '<channel|>'
CALL_OPEN, CALL_END = '<|tool_call>', '<tool_call|>'
RESP_OPEN, RESP_END = '<|tool_response>', '<tool_response|>'
TURN_END = '<turn|>\n'
VAL_REPOS = 'ipython/ipython,scipy/scipy'
RFT2_VAL_TASKS = 'rich_2943,requests_7309'
TOK_DIR = os.path.join(ROOT, 'data', 'rft2', 'tokenizer_june')
TEMPLATE = os.path.join(ROOT, 'data', 'probe', 'chat_template_kaggle_june.jinja')
TOK_S = (470, 1000, 1400)  # training throughput (tokens/s): rft2 measured 470; lora_train's speed flags promise 2-3.5x


# ------------------------------------------------------------------ pure-Python core (unit-tested with a fake renderer)

def strip_thinking(text):
    """The template's strip_thinking macro: drop <|channel>...<channel|> parts, trim."""
    out = ''
    for part in (text or '').split(THINK_END):
        out += part.split('<|channel>')[0] if '<|channel>' in part else part
    return out.strip()


class BlockError(ValueError):
    pass


def split_block(block, thinking, visible):
    """One turn's history-form text (P_k+1[len(P_k):]) -> {think, call, resp, trailer}: the template's thought block
    (with its newline before <channel|>), the call(s) <|tool_call>...<tool_call|>, the response text after
    <|tool_response> through <tool_response|>, and the visible-text trailer (VISIBLE <turn|>\\n or ''). Raises
    BlockError when the block does not have exactly that shape."""
    think = f'{THINK_OPEN}{thinking}\n{THINK_END}' if thinking else ''
    if not block.startswith(think):
        raise BlockError('thought block')
    rest = block[len(think):]
    if not rest.startswith(CALL_OPEN):
        raise BlockError('no call')
    j = rest.find(CALL_END + RESP_OPEN)
    if j < 0:
        raise BlockError('call not followed by a response')
    call = rest[:j + len(CALL_END)]
    rest = rest[j + len(CALL_END) + len(RESP_OPEN):]
    v = strip_thinking(visible)
    trailer = v + TURN_END if v else ''
    if not rest.endswith(trailer) or not rest[:len(rest) - len(trailer)].endswith(RESP_END):
        raise BlockError('response / trailer')
    resp = rest[:len(rest) - len(trailer)]
    if think + call + RESP_OPEN + resp + trailer != block:
        raise BlockError('reassembly')
    return {'think': think, 'call': call, 'resp': resp, 'trailer': trailer}


def generated_turn(call, thinking, visible):
    """(thought, text, call part) of a turn as the model generates it; the call part ends with <|tool_response> (the
    stop token after a call)."""
    th = f'{THINK_OPEN}{thinking}{THINK_END}' if thinking else ''
    return th, strip_thinking(visible), call + RESP_OPEN


def turn_mode(turn, think_tokens, think_cap):
    """'full' (thinking + text + call trained), 'call' (thinking present, masked: text + call), 'call_body' (no
    thinking: the call from `call:` on) for a train turn."""
    s = turn.get('sampling') or {}
    if not turn.get('thinking'):
        return 'call_body'
    if s.get('mask_thinking') or (think_cap and think_tokens >= think_cap - 8):
        return 'call'
    return 'full'


def train_span(mode, th, v):
    """(start offset inside the generated turn) of the labelled part."""
    if mode == 'full':
        return 0
    if mode == 'call':
        return len(th)
    return len(th) + len(v) + len(CALL_OPEN)


def assemble(prompts, turns, modes):
    """prompts = [P_0 .. P_n] (history form, P_k a prefix of P_k+1), turns = the record's turns, modes = {k: mode} of
    the train turns -> (sequence text, [(start, end, k)] labelled char spans, [(seq offset, history offset, k)] of
    each turn, history text P_n). Context turns keep their history block; train turns are written as generated."""
    for k in range(len(turns)):
        if not prompts[k + 1].startswith(prompts[k]):
            raise BlockError(f'P_{k} is not a prefix of P_{k + 1}')
    out = [prompts[0]]
    pos = len(prompts[0])
    spans, offsets = [], []
    for k, t in enumerate(turns):
        block = prompts[k + 1][len(prompts[k]):]
        offsets.append((pos, len(prompts[k]), k))
        if k not in modes:
            out.append(block)
            pos += len(block)
            continue
        pc = split_block(block, t.get('thinking') or '', t.get('visible') or '')
        th, v, cl = generated_turn(pc['call'], t.get('thinking') or '', t.get('visible') or '')
        g = th + v + cl
        spans.append((pos + train_span(modes[k], th, v), pos + len(g), k))
        piece = g + pc['resp'] + pc['trailer']
        out.append(piece)
        pos += len(piece)
    return ''.join(out), spans, offsets, prompts[-1]


def deviation(turns, modes, k):
    """What separates train turn k's context in the sequence from its real request: (earlier train turns with
    thinking = one newline token each, earlier train turns with visible text = one extra copy each)."""
    nl = sum(1 for i in modes if i < k and turns[i].get('thinking'))
    vis = sum(1 for i in modes if i < k and strip_thinking(turns[i].get('visible') or ''))
    return nl, vis


def token_diff(a, b):
    """Tokens that differ between two token lists (difflib: the larger side of every non-equal block)."""
    import difflib
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in
               difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op != 'equal')


def clean_quota(n_clean, n_other, share):
    """Clean trajectories to ADD so that clean / (clean + other) ~= share."""
    if share >= 1:
        return 10 ** 9
    want = round(share / (1 - share) * n_other)
    return max(0, want - n_clean)


def sha_order(ids):
    return sorted(ids, key=lambda x: hashlib.sha1(x.encode()).hexdigest())


# ------------------------------------------------------------------ records: merge carried + GPU-sampled thinking

def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def carried(turn):
    return bool((turn.get('sampling') or {}).get('carried'))


def overlay_sampled(recs, sampled_dir, stats):
    """Fill the not-carried train turns of RECS (in place) from the GPU pass in SAMPLED_DIR: per-turn lines
    (<source>.jsonl, rft3_sample.Store; final = fallback kept, else sample kept) or, without them, the export
    (<source>.records.jsonl). Turns without a line stay thinking-masked (counted as pending)."""
    import rft3_sample as RS
    lines = {}
    if sampled_dir and os.path.isdir(sampled_dir):
        for src in SOURCES:
            p = os.path.join(sampled_dir, f'{src}.jsonl')
            if os.path.exists(p):
                for line in open(p):
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    lines.setdefault((d['id'], d['k']), {})[d['pass']] = d
        exported = {}
        if not lines:
            for src in SOURCES:
                p = os.path.join(sampled_dir, f'{src}.records.jsonl')
                if os.path.exists(p):
                    for r in read_jsonl(p):
                        exported[r['id']] = r
        stats['sampled_lines'] = len(lines)
        stats['sampled_export_records'] = len(exported)
    else:
        exported = {}
    for rec in recs:
        ex = exported.get(rec['id'])
        for k, t in enumerate(rec['turns']):
            if t['label'] != 'train' or carried(t):
                continue
            if ex is not None and k < len(ex['turns']) and ex['turns'][k]['id'] == t['id']:
                rec['turns'][k] = copy.deepcopy(ex['turns'][k])
                rec['turns'][k].setdefault('sampling', {})['merged'] = 'export'
                stats['gpu_turns_merged'] += 1
                continue
            ps = lines.get((rec['id'], k))
            if not ps:
                stats['gpu_turns_pending'] += 1
                continue
            kept, how = RS.final_of(ps.get('sample'), ps.get('fallback'))
            RS.apply_kept(t, kept)
            base = ps.get('sample') or ps.get('fallback') or {}
            t['sampling'] = {'from': how, 'think_tokens': kept['think_tokens'] if kept else None,
                             'mask_thinking': not kept or bool(kept.get('at_budget')),
                             'rule': kept.get('rule') if kept else None,
                             'call_from': kept.get('call_from') if kept else None, 'ctx': base.get('ctx'),
                             'merged': 'lines'}
            stats['gpu_turns_merged'] += 1
            stats[f'gpu_turns_{how or "masked"}'] += 1
    return recs


def load_records(a, stats):
    """(subset records with thinking merged, relabel records not in the subset, remainder oracle records)."""
    filled = list(read_jsonl(a.filled))
    overlay_sampled(filled, a.sampled, stats)
    have = {r['id'] for r in filled}
    oracle_tasks = {r['task'] for r in filled if r['source'] in ORACLE}
    extra_relabel = [r for r in read_jsonl(os.path.join(a.states, 'relabel.jsonl')) if r['id'] not in have]
    remainder = []
    for src in ORACLE:
        for r in read_jsonl(os.path.join(a.states, f'{src}.jsonl')):
            if r['id'] not in have and r['task'] not in oracle_tasks:
                remainder.append(r)
    return filled, extra_relabel, remainder


# ------------------------------------------------------------------ rendering (real tokenizer)

def make_renderer(tok_dir=TOK_DIR, template=TEMPLATE):
    import rft2_build
    return rft2_build.make_renderer(tok_dir, template if os.path.exists(template) else None)


class Builder:
    def __init__(self, R, think_cap=256, max_prompt=S.COMPACT_AT):
        self.R, self.think_cap, self.max_prompt = R, think_cap, max_prompt
        self._tok = {}

    def ntok(self, text):
        if not text:
            return 0
        if text not in self._tok:
            self._tok[text] = len(self.R.ids(text))
        return self._tok[text]

    def prompts(self, rec):
        req = rec['request']
        think = (req.get('chat_template_kwargs') or {}).get('enable_thinking', True)
        return [self.R.text(S.messages_at(rec, k), req.get('tools'), True, think)
                for k in range(len(rec['turns']) + 1)]

    def build(self, rec, verify=False):
        """One record -> row dict (input_ids, labels, stats) or {'drop': reason}."""
        turns = rec['turns']
        modes, info = {}, []
        for k, t in enumerate(turns):
            nt = self.ntok(t.get('thinking') or '')
            if t['label'] == 'train':
                modes[k] = turn_mode(t, nt, self.think_cap)
            info.append({'k': k, 'kind': t.get('kind'), 'label': t['label'], 'mode': modes.get(k),
                         'think_tokens': nt, 'visible': bool(strip_thinking(t.get('visible') or '')),
                         'from': (t.get('sampling') or {}).get('from') if t['label'] == 'train' else
                         ('real' if t.get('thinking') else None)})
        if not modes:
            return {'drop': 'no train turn'}
        P = self.prompts(rec)
        last_prompt = len(self.R.ids(P[-2]))
        if last_prompt >= self.max_prompt:
            return {'drop': 'prompt >= max', 'prompt_tokens': last_prompt}
        try:
            text, spans, offsets, hist = assemble(P, turns, modes)
        except BlockError as e:
            return {'drop': f'render: {e}'}
        ids, labels, same = rb.tokenize_sequence(self.R, text, [(s, e) for s, e, _ in spans])
        row = {'input_ids': ids, 'labels': labels, 'n_tokens': len(ids), 'segment_tokenization_equal': same,
               'n_trained': sum(x != -100 for x in labels), 'trained_turns': len(modes),
               'think_trained_turns': sum(m == 'full' for m in modes.values()), 'prompt_tokens': last_prompt,
               'turns': info}
        row['n_trained_tokens'] = row['n_trained']
        if verify:
            row['verify'] = self.verify(rec, P, text, spans, offsets, hist, ids, modes)
        return row

    def verify(self, rec, P, text, spans, offsets, hist, ids, modes):
        """Prefix consistency, token level: (1) every request P_k tokenizes to a prefix of the history-form
        trajectory's tokens; (2) for every train turn k, the sequence's tokens before it equal tokens(P_k) when no
        earlier train turn deviates (deviation() == (0, 0)), else the token count differs by exactly the expected
        newlines / text copies is reported; (3) each generated train turn tokenizes standalone to the same ids it has
        inside the sequence (its labels are what the model emits)."""
        R = self.R
        out = collections.Counter()
        hist_ids = R.ids(hist)
        for k in range(len(P) - 1):
            pk = R.ids(P[k])
            out['requests'] += 1
            out['request_prefix_of_history'] += hist_ids[:len(pk)] == pk
        off_of = {k: s for s, _, k in offsets}
        for s, e, k in spans:
            pk = R.ids(P[k])
            pre = R.ids(text[:off_of[k]])
            nl, vis = deviation(rec['turns'], modes, k)
            out['train_turns'] += 1
            if (nl, vis) == (0, 0):
                out['train_turns_exact_context_expected'] += 1
                out['train_turns_exact_context'] += pre == pk
            else:
                out['train_turns_deviating_context'] += 1
                out['deviation_tokens'] += token_diff(pre, pk)
                out['deviation_newlines'] += nl
                out['deviation_visible_copies'] += vis
            # the generated turn (from the turn start) standalone == inside the sequence
            t0 = off_of[k]
            g_end = e
            g_ids = R.ids(text[t0:g_end])
            n_pre = len(pre)
            out['generated_turn_tokens_equal'] += ids[n_pre:n_pre + len(g_ids)] == g_ids
        return dict(out)


# ------------------------------------------------------------------ raw-capture check of the generated form

def raw_check(R, runs, limit, stats):
    """The generated form (generated_turn on split_block of the template's own rendering) vs the sampled token ids of
    captured responses (data/raw/<run>, --logprobs captures): exact for well-formed single calls with sorted argument
    keys, no second thought channel and no thinking cut by the budget."""
    import rft2_build
    for run in runs:
        d = os.path.join(ROOT, 'data', 'raw', run)
        if not os.path.isdir(d):
            stats[f'{run}: missing'] += 1
            continue
        n = 0
        for f in sorted(os.listdir(d)):
            if not f.endswith('.json.gz') or n >= limit:
                continue
            try:
                x = json.load(gzip.open(os.path.join(d, f), 'rt'))
            except (OSError, ValueError):
                continue
            req = x.get('request') or {}
            ch = (x.get('response') or {}).get('choices') or []
            if not req.get('tools') or not ch or not ch[0].get('token_ids'):
                continue
            asst = rb.response_message(x['response'])
            calls = asst.get('tool_calls') or []
            if len(calls) != 1:
                continue
            n += 1
            raw = R.tok.decode(ch[0]['token_ids'], skip_special_tokens=False)
            th = asst.get('reasoning') or ''
            vis = asst.get('content') or ''
            tool = {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': '{"stdout": "x"}'}
            think = (req.get('chat_template_kwargs') or {}).get('enable_thinking', True)
            P0 = R.text(req['messages'], req['tools'], True, think)
            P1 = R.text(req['messages'] + [asst, tool], req['tools'], True, think)
            try:
                pc = split_block(P1[len(P0):], th, vis)
            except BlockError as e:
                stats[f'split error: {e}'] += 1
                continue
            g = ''.join(generated_turn(pc['call'], th, vis))
            same = R.ids(g) == ch[0]['token_ids']
            budget = req.get('thinking_token_budget') or 0
            at_budget = bool(budget) and len(R.ids(th)) >= budget - 8
            keys_sorted = True
            try:
                a = json.loads(calls[0]['function']['arguments'])
                inner = a.get('args') if isinstance(a.get('args'), dict) else {}
                keys_sorted = list(a) == sorted(a) and list(inner) == sorted(inner)
            except (ValueError, AttributeError):
                keys_sorted = False
            raw_calls = re.findall(r'<\|tool_call>(.*?)<tool_call\|>', raw, re.S)
            raw_sorted = bool(raw_calls) and raw_calls == re.findall(r'<\|tool_call>(.*?)<tool_call\|>', g, re.S)
            clean = (rft2_build.well_formed(calls[0]) and raw.count('<|channel>') <= 1 and not at_budget and not vis
                     and raw_sorted and bool(th))
            stats['checked'] += 1
            stats['equal'] += same
            stats['clean'] += clean
            stats['clean_equal'] += clean and same
            stats['clean_text_equal'] += clean and g == raw
            stats['empty_thought_channel'] += raw.startswith(THINK_OPEN + THINK_END)
            stats['keys_unsorted'] += not keys_sorted
            if clean and not same and len(stats.setdefault('_examples', [])) < 5:
                stats['_examples'].append(f'{run}/{f}')
    return stats


# ------------------------------------------------------------------ main

def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', default='v20')
    ap.add_argument('--filled', default=None, help='default <states>/subset900_filled.jsonl')
    ap.add_argument('--sampled', default=None, help='GPU pass output (default data/rft3/sampled_v20)')
    ap.add_argument('--states', default=None, help='default data/rft3/states_v20')
    ap.add_argument('--rft2', default=os.path.join(ROOT, 'data', 'rft2', 'train_tokens_all.jsonl'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'rft3'))
    ap.add_argument('--remainder', choices=['none', 'mix', 'all'], default='mix')
    ap.add_argument('--clean-share', type=float, default=0.55)
    ap.add_argument('--relabel-copies', type=int, default=2)
    ap.add_argument('--rft2-copies', type=int, default=1)
    ap.add_argument('--subset-copies', type=int, default=1)
    ap.add_argument('--val-repos', default=VAL_REPOS)
    ap.add_argument('--rft2-val-tasks', default=RFT2_VAL_TASKS)
    ap.add_argument('--think-cap', type=int, default=256, help='thinking of >= cap-8 tokens is not trained; 0 = off')
    ap.add_argument('--max-prompt', type=int, default=S.COMPACT_AT)
    ap.add_argument('--max-len', type=int, default=32768)
    ap.add_argument('--verify', type=int, default=150,
                    help='records (random, all sources) given the token-level prefix-consistency check; 0 = all')
    ap.add_argument('--raw-check', default='E060_v19b_real', help='captured runs for the generated-form check')
    ap.add_argument('--raw-limit', type=int, default=1500)
    ap.add_argument('--limit', type=int, default=0, help='first N records per group (dev)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    S.use_version(a.version)
    a.states = a.states or S.STATES_DIR
    a.filled = a.filled or os.path.join(a.states, 'subset900_filled.jsonl')
    a.sampled = a.sampled or S.suffixed(os.path.join(ROOT, 'data', 'rft3', 'sampled'))
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    stats = collections.Counter()
    report = {'args': vars(a)}

    filled, extra_relabel, remainder = load_records(a, stats)
    val_repos = {x for x in a.val_repos.split(',') if x}
    rft2_val = {x for x in a.rft2_val_tasks.split(',') if x}
    print(f'records: subset {len(filled)} ({dict(collections.Counter(r["source"] for r in filled))}); GPU pass '
          f'{a.sampled}: lines {stats["sampled_lines"]}, export records {stats["sampled_export_records"]}, turns merged '
          f'{stats["gpu_turns_merged"]}, pending (thinking-masked) {stats["gpu_turns_pending"]}; relabel not in the '
          f'subset {len(extra_relabel)}; remainder {len(remainder)} '
          f'({dict(collections.Counter(r["source"] for r in remainder))})', flush=True)

    # ---- selection
    def split_of(r):
        return 'val' if (r['repo'] in val_repos and r['source'] in ORACLE) or r['task'] in rft2_val else 'train'

    groups = [('relabel' if r['source'] == 'relabel' else 'subset_' + r['source'], r) for r in filled]
    groups += [('relabel_extra', r) for r in extra_relabel]  # (group, record)
    rem_train = [r for r in remainder if split_of(r) == 'train']
    rem_sel = []
    if a.remainder == 'all':
        rem_sel = rem_train
    elif a.remainder == 'mix':
        other = [r for r in rem_train if r['source'] != 'oracle_clean']
        n_clean = sum(1 for r in filled if r['source'] == 'oracle_clean' and split_of(r) == 'train')
        n_other = len(other) + sum(1 for r in filled if r['source'] in ('oracle_recovery', 'oracle_repair')
                                   and split_of(r) == 'train')
        quota = clean_quota(n_clean, n_other, a.clean_share)
        clean_by_id = {r['id']: r for r in rem_train if r['source'] == 'oracle_clean'}
        rem_sel = other + [clean_by_id[i] for i in sha_order(clean_by_id)[:quota]]
        report['remainder_mix'] = {'subset_clean': n_clean, 'other': n_other, 'clean_added': min(quota,
                                                                                                  len(clean_by_id))}
    groups += [('remainder_' + r['source'], r) for r in rem_sel]
    if a.limit:
        per = collections.Counter()
        keep = []
        for g, r in groups:
            if per[g] < a.limit:
                keep.append((g, r))
                per[g] += 1
        groups = keep

    # ---- render + tokenize
    R = make_renderer()
    B = Builder(R, a.think_cap, a.max_prompt)
    rng = random.Random(a.seed)
    ver_ids = {r['id'] for _, r in groups} if a.verify == 0 else \
        set(rng.sample(sorted(r['id'] for _, r in groups), min(a.verify, len(groups))))
    rows, index, drops = [], [], collections.Counter()
    ver = collections.Counter()
    t0 = time.time()
    for n, (g, r) in enumerate(groups):
        res = B.build(r, verify=r['id'] in ver_ids)
        if 'drop' in res:
            drops[f'{g}: {res["drop"]}'] += 1
            continue
        if res['n_tokens'] > a.max_len:
            drops[f'{g}: longer than --max-len'] += 1
            continue
        ver.update(res.pop('verify', {}))
        split = split_of(r)
        copies = 1 if split == 'val' else (a.relabel_copies if r['source'] == 'relabel' else
                                           a.subset_copies if not g.startswith('remainder') else 1)
        turns = res.pop('turns')
        index.append({'id': r['id'], 'group': g, 'split': split, 'copies': copies, 'source': r['source'],
                      'task': r['task'], 'repo': r['repo'], 'n_tokens': res['n_tokens'], 'n_trained': res['n_trained'],
                      'prompt_tokens': res['prompt_tokens'], 'turns': turns})
        for c in range(copies):
            row = dict(res, id=r['id'] if c == 0 else f'{r["id"]}_copy{c}', task=r['task'], repo=r['repo'],
                       source=r['source'], kind=g, copy=c, split=split)
            rows.append(row)
        if (n + 1) % 200 == 0:
            print(f'  [{n + 1}/{len(groups)}] {time.time() - t0:.0f}s', flush=True)
    stats['segment_tokenization_equal'] = sum(r['segment_tokenization_equal'] for r in rows)

    # ---- rft2 rows
    rft2_rows = []
    for x in read_jsonl(a.rft2):
        split = 'val' if x['task'] in rft2_val else 'train'
        copies = 1 if split == 'val' else a.rft2_copies
        for c in range(copies):
            y = dict(x, id=x['id'] if c == 0 else f'{x["id"]}_copy{c}', repo=x['task'].rsplit('_', 1)[0],
                     source='rft2', kind='rft2_' + x.get('kind', 'episode'), copy=c, split=split)
            rft2_rows.append(y)
    rows += rft2_rows
    for x in rft2_rows:
        if x['copy'] == 0:
            index.append({'id': x['id'], 'group': x['kind'], 'split': x['split'], 'copies': 1 if x['split'] == 'val'
                          else a.rft2_copies, 'source': 'rft2', 'task': x['task'], 'repo': x['repo'],
                          'n_tokens': x['n_tokens'], 'n_trained': x['n_trained']})

    # ---- write
    val_tasks = sorted({r['task'] for r in rows if r['split'] == 'val'})
    with open(os.path.join(a.out, 'train_tokens.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items()}) + '\n')
    with open(os.path.join(a.out, 'val_tasks.txt'), 'w') as f:
        f.write('\n'.join(val_tasks) + '\n')
    with open(os.path.join(a.out, 'build_index.jsonl'), 'w') as f:
        for x in index:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')

    # ---- report
    def agg(rs):
        lens = [r['n_tokens'] for r in rs]
        return {'rows': len(rs), 'tokens': sum(lens), 'labelled': sum(r['n_trained'] for r in rs),
                'len_median': pct(lens, .5), 'len_max': max(lens, default=0)}
    by_kind = collections.defaultdict(list)
    for r in rows:
        by_kind[(r['split'], r['kind'])].append(r)
    report['rows'] = {f'{s} {k}': agg(v) for (s, k), v in sorted(by_kind.items())}
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'val']
    report['train'] = agg(train)
    report['val'] = agg(val) | {'tasks': len(val_tasks)}
    lens = [r['n_tokens'] for r in train]
    report['train_length'] = {q: pct(lens, v) for q, v in (('min', 0), ('p10', .1), ('p25', .25), ('median', .5),
                                                            ('p75', .75), ('p90', .9), ('max', 1))}
    modes = collections.Counter()
    modes_kind = collections.defaultdict(collections.Counter)
    think_from = collections.Counter()
    vis_trained = 0
    for x in index:
        if x['split'] != 'train' or x['source'] == 'rft2':
            continue
        for t in x['turns']:
            if t['mode']:
                w = x['copies']
                modes[t['mode']] += w
                modes_kind[t['kind']][t['mode']] += w
                if t['mode'] == 'full':
                    think_from[t['from'] or '?'] += w
                vis_trained += w * (t['visible'] and t['mode'] in ('full', 'call'))
    tot = sum(modes.values())
    report['masking'] = {'train_turns (with copies)': tot, 'modes': dict(modes),
                         'share_thinking_trained': round(modes['full'] / tot, 3) if tot else 0,
                         'thinking_trained_from': dict(think_from), 'visible_text_trained_turns': vis_trained,
                         'by_kind': {k: dict(v) for k, v in sorted(modes_kind.items())}}
    report['drops'] = dict(drops)
    report['merge'] = {k: v for k, v in stats.items() if k.startswith(('gpu_', 'sampled_'))}
    report['verify'] = dict(ver)
    tt = report['train']['tokens']
    report['h200_hours'] = {f'{s} tok/s': {'1 epoch': round(tt / s / 3600, 2), '2 epochs': round(2 * tt / s / 3600, 2)}
                            for s in TOK_S}
    if a.raw_check:
        rc = raw_check(R, [x for x in a.raw_check.split(',') if x], a.raw_limit, collections.Counter())
        report['raw_check'] = dict(rc)
    report['seconds'] = round(time.time() - t_start)
    json.dump(report, open(os.path.join(a.out, 'build_report.json'), 'w'), indent=1)

    print(f'\n== rows (split kind: rows, tokens, labelled, median len, max len)')
    for k, v in report['rows'].items():
        print(f'   {k:32s} {v["rows"]:5d} {v["tokens"]:10d} {v["labelled"]:8d} {v["len_median"]:6d} {v["len_max"]:6d}')
    print(f'   TRAIN {report["train"]}\n   VAL   {report["val"]}')
    print(f'== train sequence length: {report["train_length"]}')
    m = report['masking']
    print(f'== masking (train turns of rft3 rows, copies counted): {m["train_turns (with copies)"]} turns: '
          f'{m["modes"]}; thinking trained {m["share_thinking_trained"]:.1%} (from {m["thinking_trained_from"]}); '
          f'visible text trained in {m["visible_text_trained_turns"]} turns')
    for k, v in m['by_kind'].items():
        print(f'     {k}: {v}')
    print(f'== drops: {report["drops"] or "none"}')
    print(f'== GPU merge: {report["merge"]}')
    print(f'== verify (token level, {len(ver_ids)} records): {report["verify"]}')
    print(f'   segment-wise tokenization == whole: {stats["segment_tokenization_equal"]}/{len(rows) - len(rft2_rows)}')
    if a.raw_check:
        print(f'== generated form vs sampled token ids ({a.raw_check}): '
              f'{ {k: v for k, v in report["raw_check"].items() if not k.startswith("_")} }')
    print(f'== H200 time for {tt} train tokens: ' + '; '.join(
        f'{s} tok/s: {v["1 epoch"]} h (1 ep), {v["2 epochs"]} h (2 ep)' for s, v in
        ((s, report['h200_hours'][f'{s} tok/s']) for s in TOK_S)))
    print(f'\nwrote {a.out}/train_tokens.jsonl ({len(rows)} rows), val_tasks.txt ({len(val_tasks)} tasks), '
          f'build_index.jsonl, build_report.json in {report["seconds"]}s')


if __name__ == '__main__':
    sys.exit(main())
