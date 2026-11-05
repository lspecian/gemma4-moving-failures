"""Stage-2 self-RFT dataset (research/notes/9-lora-plan.md; LOG 2026-10-05): Kaggle-like raw captures -> episodes ->
repaired, selected passing episodes -> token ids + loss masks rendered as the scorer's vLLM renders them.

    .venv/bin/python scripts/rft2_build.py                    # all steps, defaults below
    .venv/bin/python scripts/rft2_build.py --selftest-only    # only the rendering self-test

Inputs:  data/raw/<run>/*.json.gz (scripts/slow_proxy.py --save [--logprobs]: {path, status, t0, timing, request,
         response}), results/<run>/r1/c*/task_results.jsonl, data/competition/tasks.jsonl, experiments/eval113.txt.
         Default runs, in selection-priority order: E054 (v18c + thinking budget 256 = agent-v18d), E053 (v18c),
         E049 (v18c), E048 (v17b); all eval113 x 1 round, concurrency 4, compaction on, Kaggle's June template, ~10-12
         tok/s. --r001 adds data/rft/raw/R001 (v13, no compaction, July template; rendered with the July tokenizer).
Tokenizer: data/rft2/tokenizer_june/ = the HF tokenizer of google/gemma-4-31B-it-qat-w4a16-ct (data/rft/tokenizer,
         which carries HF's 2026-07-15 chat template) with Kaggle's June files swapped in
         (data/probe/chat_template_kaggle_june.jinja, tokenizer_config_kaggle_june.json; created on first run).
Outputs (data/rft2/):
         episodes.jsonl      every episode (run x task): result, compactions, per-turn info, repairs (no transcripts)
         train.jsonl         the selected episodes with each training sequence's (repaired) messages + tools
         train_tokens.jsonl  one row per sequence: id, episode, task, run, input_ids, labels (-100 = no loss),
                             n_tokens, n_trained_tokens, trained_turns, repaired (scripts/lora_train.py --data input)
         build_report.json   the numbers printed at the end
         experiments/lora2_heldout.txt / lora2_train_tasks.txt (fixed once written; --resplit to redo)

Steps:
 1. Load captures; keep agent calls (chat completions with tools); count ADK compaction-summary calls (no tools, prompt
    starts "The following is a conversation history between a user and an AI agent") and other requests (health
    checks). Group by run and task: the task is the tasks.jsonl problem statement contained in the first user message;
    after a compaction that message is replaced by ADK's "For context:" summary, so such a request is linked to its
    episode by the tool-call ids of the retained events. A task started twice in one run keeps its last attempt.
 2. Self-test: every agent request re-rendered (rft_build.Renderer = vLLM 0.19.1's chat path) must tokenize to exactly
    usage.prompt_tokens. Also: canonical re-render of each model turn vs the sampled token_ids (--logprobs captures).
 3. Sequences: rft_build's prefix rule (a turn is trained in a sequence S = F_j iff S starts with P_k + T_k). A
    compaction rewrites history (the prompt shrinks), so turns before it are trained in the last pre-compaction
    sequence and the post-compaction requests start a new one; the retained events re-appear there as context only.
 4. Repair of ADK-rejected calls (tool result {"error", "error_code"}: 「repo-tools」, a backtick in place of <|"|>,
    skill_name ":" ...). A maximal chain of rejected single-call turns k..k+m-1 followed by an accepted call r=k+m in
    the same context (no compaction in between):
      - merge (r runs a script the garbled call named): the chain and its error results are removed and r's thinking
        is replaced by turn k's -> the trained turn is "the model's original reasoning + the correctly written call".
        Simply dropping the chain and training r as is would be wrong: r's own thinking is nearly always about the
        error ("My apologies, I used Japanese brackets", "Ah, I used create.py as skill_name"), checked on all runs;
      - none (otherwise): no change; the chain stays masked, r is trained in its real context (a recovery).
    Removals are applied to every later request of the episode, the transcript is re-rendered (it is no longer what
    the model saw: such sequences carry repaired=true). Rejected turns without a successful retry stay masked.
 5. Masks (context, no loss): turns cut by max_tokens (finish length), ADK-rejected calls not repaired, text-only
    turns (final summaries: the June template drops their thinking), turns without thinking, calls whose canonical
    render would still be rejected (raw_report.call_problem), and - for conciseness - the thinking part (not the call)
    of turns whose thinking reached --think-cap tokens (cut by the budget or longer than v18d's budget).
 6. Selection: passing episodes only; per task up to --per-task, by run priority, then fewer tool calls, then less
    total thinking. Tasks in the held-out list never contribute.
 7. Held-out split: a fixed repo-stratified third of experiments/eval113.txt (sha1 order per repo).

Training (on a GPU host, see scripts/lora_train.py; its loader reads id/task/input_ids/labels):
    python3 scripts/lora_train.py --model /models/gemma-4-31b-it-qat-w4a16-ct --data data/rft2/train_tokens.jsonl \
        --out /root/lora_out --name rft2 [--val-tasks <a few train tasks>]
"""

import argparse
import collections
import copy
import gzip
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rft_build as rb  # noqa: E402  (stdlib-only at import time; transformers is imported by Renderer)
from raw_report import SUMMARY_HEAD, call_problem  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ['E054_v18c_b256_kaggleslow', 'E053_v18c_kaggleslow', 'E049_v18c_kaggleslow', 'E048_v17b_kaggleslow']
SKILL = 'repo-tools'
SCRIPT_RE = re.compile(r'([A-Za-z_][\w-]*\.py)')


# ------------------------------------------------------------------ pure-Python core (unit-tested, no tokenizer)

def call_ids(m):
    return [c.get('id') for c in m.get('tool_calls') or []]


def tool_error(content):
    """The ADK rejection object {"error", "error_code"} of a tool result, else None."""
    try:
        j = json.loads(content) if isinstance(content, str) else content
    except ValueError:
        return None
    return j if isinstance(j, dict) and 'error' in j and 'error_code' in j else None


def call_args(call):
    a = call['function'].get('arguments')
    if isinstance(a, dict):
        return a
    try:
        a = json.loads(a or '{}')
    except ValueError:
        return {}
    return a if isinstance(a, dict) else {}


def intended_scripts(call):
    """The scripts a (possibly garbled) run_skill_script call may have meant: every *.py name in a value whose key
    mentions file_path or skill_name, at the top level (garbled keys such as '},file_path' included) or nested in
    args (the model sometimes writes a script as skill_name and a file as file_path). Other tools: their name."""
    a = call_args(call)
    if call['function'].get('name') != 'run_skill_script':
        return [call['function'].get('name')]
    inner = a.get('args') if isinstance(a.get('args'), dict) else {}
    keys = [(a, 'file_path'), (a, 'skill_name'), (inner, 'file_path'), (inner, 'skill_name')]
    keys += [(a, k) for k in sorted(a) if k not in ('file_path', 'skill_name') and
             ('file_path' in k or 'skill_name' in k)]
    out = []
    for d, key in keys:
        v = d.get(key)
        if isinstance(v, str):
            out += [m.split('/')[-1] for m in SCRIPT_RE.findall(v) if m.split('/')[-1] not in out]
    return out


def well_formed(call):
    """A run_skill_script call ADK accepts as written: skill_name exactly repo-tools, file_path a bare script name,
    neither nested inside args. Other tools pass."""
    if call['function'].get('name') != 'run_skill_script':
        return True
    a = call_args(call)
    fp, args = a.get('file_path'), a.get('args')
    if a.get('skill_name') != SKILL or not isinstance(fp, str) or not re.fullmatch(r'[\w-]+\.py', fp):
        return False
    return not (isinstance(args, dict) and ('skill_name' in args or 'file_path' in args))


def turn_records(reqs):
    """reqs: one episode's agent calls in time order, each {'messages', 'response' (assistant message as the harness
    sends it back), 'finish'}. Returns per-turn dicts: k, call ids, rejection (from the tool results in ANY later
    request: compaction can drop them from the next one), error text, thinking/text-only flags, and whether the turn's
    assistant message reaches the next request (same context: no compaction in between)."""
    out = []
    results = {}
    for d in reqs:
        for m in d['messages']:
            if m.get('role') == 'tool' and m.get('tool_call_id'):
                results.setdefault(m['tool_call_id'], m.get('content'))
    for k, d in enumerate(reqs):
        r = d['response']
        ids = call_ids(r)
        errs = [tool_error(results.get(i)) for i in ids]
        nxt = reqs[k + 1]['messages'] if k + 1 < len(reqs) else []
        out.append({'k': k, 'ids': ids, 'n_calls': len(ids), 'finish': d.get('finish'),
                    'rejected': any(errs), 'error': next((e['error'] for e in errs if e), None),
                    'has_result': all(i in results for i in ids),
                    'reasoning': r.get('reasoning') or r.get('reasoning_content') or '',
                    'well_formed': all(well_formed(c) for c in r.get('tool_calls') or []),
                    'text_only': not ids, 'scripts': intended_scripts(r['tool_calls'][0]) if ids else [],
                    'args': call_args(r['tool_calls'][0]) if ids else None,
                    'in_next': bool(ids) and any(m.get('role') == 'assistant' and call_ids(m) == ids for m in nxt)})
    return out


def plan_repairs(turns):
    """Repairs for chains of rejected turns (module doc, step 4). Returns (drop_ids, reasoning_override {call id of
    the retry: thinking of the chain's first turn}, repairs [dicts], unrepaired rejected turn indices)."""
    drop, override, repairs, unrepaired = set(), {}, [], []
    k = 0
    while k < len(turns):
        if not turns[k]['rejected']:
            k += 1
            continue
        j = k
        while j < len(turns) and turns[j]['rejected']:
            j += 1
        chain = list(range(k, j))
        rep = {'chain': chain, 'retry': j if j < len(turns) else None, 'mode': 'none', 'why': ''}
        r = turns[j] if j < len(turns) else None
        if r is None:
            rep['why'] = 'no retry (episode ended)'
        elif r['text_only'] or r['finish'] == 'length':
            rep['why'] = 'retry is not a call'
        elif not r['has_result'] or not r['well_formed']:
            rep['why'] = 'retry not confirmed (no tool result, or garbled too)'
        elif any(turns[c]['n_calls'] != 1 for c in chain) or r['n_calls'] != 1:
            rep['why'] = 'parallel calls'
        elif not all(turns[c]['in_next'] for c in chain):
            rep['why'] = 'compaction between rejection and retry'
        elif not turns[k]['reasoning']:
            rep['why'] = 'rejected turn has no thinking'
        elif r['scripts'] and r['scripts'][0] in turns[k]['scripts']:
            rep['mode'] = 'merge'
            rep['same_args'] = _norm_args(turns[k]['args']) == _norm_args(r['args'])
        else:
            rep['why'] = 'the retry calls a different script'
        if rep['mode'] == 'none':
            unrepaired += chain
        else:
            for c in chain:
                drop.update(turns[c]['ids'])
            if rep['mode'] == 'merge':
                override[r['ids'][0]] = turns[k]['reasoning']
        repairs.append(rep)
        k = j
    return drop, override, repairs, unrepaired


def _norm_args(a):
    """The args object of a call with the garbling stripped (for the same-arguments statistic only)."""
    a = dict(a or {})
    inner = a.get('args') if isinstance(a.get('args'), dict) else {}
    inner = {k: v for k, v in inner.items() if k not in ('skill_name', 'file_path')}
    return json.dumps(inner, sort_keys=True)


def apply_repairs(messages, drop_ids, override):
    """messages with the dropped assistant calls and their tool results removed and the overridden thinking set."""
    out = []
    for m in messages:
        ids = call_ids(m)
        if m.get('role') == 'assistant' and ids and all(i in drop_ids for i in ids):
            continue
        if m.get('role') == 'tool' and m.get('tool_call_id') in drop_ids:
            continue
        if m.get('role') == 'assistant' and ids and ids[0] in override:
            m = dict(m)
            m['reasoning'] = m['reasoning_content'] = override[ids[0]]
        out.append(m)
    return out


def split_heldout(tasks, frac=1 / 3):
    """Fixed repo-stratified held-out set: per repo (id prefix before '_'), tasks in sha1 order, the first
    round(n * frac)."""
    by_repo = collections.defaultdict(list)
    for t in tasks:
        by_repo[t.rsplit('_', 1)[0]].append(t)
    held = []
    for repo, ts in sorted(by_repo.items()):
        ts.sort(key=lambda t: hashlib.sha1(t.encode()).hexdigest())
        held += ts[:round(len(ts) * frac)]
    return sorted(held)


def assign_sequences(turns, masked):
    """rft_build.assign_sequences with an explicit mask predicate: {j: [trained turn ks]}, S = F_j."""
    seqs = {}
    for t in reversed(turns):
        if masked(t):
            continue
        target = t['P'] + t['T']
        j = next((j for j in sorted(seqs, reverse=True) if turns[j]['F'].startswith(target)), None)
        if j is None:
            j = next(j for j in range(len(turns) - 1, t['i'] - 1, -1) if turns[j]['F'].startswith(target))
            seqs[j] = []
        seqs[j].append(t['i'])
    return {j: sorted(ks) for j, ks in seqs.items()}


# ------------------------------------------------------------------ loading and grouping

def load_run(raw_dir):
    recs = []
    for f in sorted(Path(raw_dir).glob('*.json.gz')):
        try:
            d = json.load(gzip.open(f, 'rt'))
        except (OSError, ValueError):
            continue
        ch = (d.get('response') or {}).get('choices') or []
        if ch:
            ch[0].pop('logprobs', None)  # keep token_ids, drop the (large) per-token alternatives
        d['file'] = f.name
        d['t'] = d.get('t0') or int(f.name.split('_')[0]) / 1000
        recs.append(d)
    recs.sort(key=lambda d: (d['t'], d['file']))
    return recs


def kind(d):
    req = d.get('request') or {}
    msgs = req.get('messages') or []
    if not str(d.get('path', '')).endswith('chat/completions') or d.get('status') != 200:
        return 'other'
    if not req.get('tools'):
        first = str(msgs[0].get('content') if msgs else '')
        return 'summary' if SUMMARY_HEAD in first else 'other'
    return 'agent' if (d.get('response') or {}).get('choices') else 'other'


def text_of(content):
    if isinstance(content, list):
        return '\n'.join(p.get('text', '') if isinstance(p, dict) else str(p) for p in content)
    return content or ''


class TaskFinder:
    def __init__(self, tasks):
        self.tasks = tasks
        self.cache = {}

    def __call__(self, text):
        key = hash(text)
        if key not in self.cache:
            hits = [i for i, ps in self.tasks.items() if ps and ps in text]
            self.cache[key] = max(hits, key=lambda i: len(self.tasks[i])) if hits else None
        return self.cache[key]


def group_run(recs, find_task, stats):
    """{task: [agent requests]} for one run (module doc, step 1) and {task: compaction summaries}."""
    groups, summaries, id2task = collections.defaultdict(list), collections.Counter(), {}
    for d in recs:
        kd = kind(d)
        stats[f'requests_{kd}'] += 1
        msgs = (d.get('request') or {}).get('messages') or []
        if kd == 'summary':
            t = find_task(text_of(msgs[0].get('content')))
            summaries[t] += 1
            stats['summaries_unlinked'] += t is None
            continue
        if kd != 'agent':
            continue
        user = next((m for m in msgs if m['role'] == 'user'), {})
        task = find_task(user['content']) if isinstance(user.get('content'), str) else None
        if task is None:
            votes = collections.Counter(id2task[i] for m in msgs for i in
                                        call_ids(m) + ([m['tool_call_id']] if m.get('tool_call_id') else [])
                                        if i in id2task)
            task = votes.most_common(1)[0][0] if votes else None
            stats['requests_linked_by_call_ids'] += task is not None
        if task is None:
            stats['requests_unassigned'] += 1
            continue
        for m in msgs:
            for i in call_ids(m):
                id2task.setdefault(i, task)
        for i in call_ids(d['response']['choices'][0]['message']):
            id2task.setdefault(i, task)
        groups[task].append(d)
    out = {}
    for task, reqs in groups.items():  # a task started twice (fresh [system, user] request): keep the last attempt
        starts = [n for n, d in enumerate(reqs) if not any(m['role'] in ('assistant', 'tool')
                                                           for m in d['request']['messages'])]
        if len(starts) > 1:
            stats['episodes_restarted'] += 1
        out[task] = reqs[starts[-1]:] if starts else reqs
    return out, summaries


def load_results(results_dir):
    res = {}
    for f in sorted(Path(results_dir).glob('r*/c*/task_results.jsonl')) + \
            sorted(Path(results_dir).glob('r*/task_results.jsonl')):
        for line in open(f):
            x = json.loads(line)
            res[x['instance_id']] = x
    return res


# ------------------------------------------------------------------ rendering

def ensure_june_tokenizer(src, out, template, config):
    """data/rft2/tokenizer_june: the HF tokenizer files with Kaggle's June chat template and tokenizer_config."""
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(src).iterdir():
        if f.name in ('chat_template.jinja', 'tokenizer_config.json') or f.name.startswith('.'):
            continue
        dst = out / f.name
        if not dst.exists():
            os.symlink(os.path.relpath(f.resolve(), out), dst)
    for s, name in ((template, 'chat_template.jinja'), (config, 'tokenizer_config.json')):
        if not (out / name).exists() or (out / name).read_bytes() != Path(s).read_bytes():
            shutil.copyfile(s, out / name)
    return out


def make_renderer(tok_dir, template_file=None):
    R = rb.Renderer(tok_dir)
    if template_file:
        want = Path(template_file).read_text()
        if R.template != want:
            raise SystemExit(f'{tok_dir}: loaded chat template differs from {template_file}')
    return R


def as_req(d):
    ch = d['response']['choices'][0]
    return {'messages': d['request']['messages'], 'response': rb.response_message(d['response']),
            'finish': ch.get('finish_reason')}


def selftest(R, items, report, key):
    """items: [(request dict, run)] -> exact token-count matches against usage.prompt_tokens."""
    n = exact = 0
    worst = []
    t0 = time.time()
    for d, run in items:
        req = d['request']
        think = (req.get('chat_template_kwargs') or {}).get('enable_thinking', True)
        ids = R.ids(R.text(req['messages'], req.get('tools'), True, think))
        pt = d['response']['usage']['prompt_tokens']
        n += 1
        exact += len(ids) == pt
        if len(ids) != pt:
            worst.append((len(ids) - pt, f'{run}/{d["file"]}'))
    report[key] = {'requests': n, 'exact_token_count': exact, 'mismatches': len(worst),
                   'mismatch_examples': sorted(worst, key=lambda x: -abs(x[0]))[:10], 'content_format': R.fmt,
                   'seconds': round(time.time() - t0)}
    print(f'self-test ({key}): {exact}/{n} requests re-rendered to exactly usage.prompt_tokens '
          f'(content format {R.fmt!r}, {time.time() - t0:.0f}s)')
    for diff, f in worst[:10]:
        print(f'   mismatch {diff:+d} tokens: {f}')


def think_tokens(R, text, cache={}):  # noqa: B006 (memo)
    if not text:
        return 0
    if text not in cache:
        cache[text] = len(R.ids(text))
    return cache[text]


def render_turn(R, ep, k):
    """Turn k of an episode rendered from its repaired request: P (prompt), F (prompt + turn), T (the turn), or None
    if a repair removed the turn."""
    q, d = ep['_reqs'][k], ep['_raw'][k]
    drop, override = ep['_drop'], ep['_override']
    resp = q['response']
    if resp.get('tool_calls') and all(i in drop for i in call_ids(resp)):
        return None
    msgs = apply_repairs(q['messages'], drop, override)
    asst = apply_repairs([resp], set(), override)[0]
    req = d['request']
    think = (req.get('chat_template_kwargs') or {}).get('enable_thinking', True)
    P = R.text(msgs, req.get('tools'), True, think)
    F = R.text(msgs + [asst], req.get('tools'), False, think)
    T = F[len(P):] if F.startswith(P) else None
    if T is not None and not asst.get('tool_calls') and T.endswith('<turn|>\n'):
        T = T[:-1]
    merged = bool(asst.get('tool_calls')) and call_ids(asst)[0] in override
    return {'k': k, 'P': P, 'F': F, 'T': T, 'asst': asst, 'msgs': msgs, 'info': ep['turns'][k], 'merged': merged,
            'repaired': msgs != q['messages'] or merged, 'tools': req.get('tools')}


def sampled_form(T):
    """A rendered model turn as the model writes it: the template puts '\n' before <channel|> (after the thinking
    text, which already ends in the model's own newline, if any); the sampled tokens do not have it."""
    return T.replace('\n<channel|>', '<channel|>', 1)


def trained_text(S, items, sampled_end):
    """S with each trained turn (s, e, call_only) optionally rewritten in its sampled form; returns (text, spans)."""
    out, spans, pos = [], [], 0
    for s, e, call_only in sorted(items):
        out.append(S[pos:s])
        T = sampled_form(S[s:e]) if sampled_end else S[s:e]
        base = sum(len(x) for x in out)
        start = base
        if call_only:
            c = T.find('<channel|>')
            start += c + len('<channel|>') if c >= 0 else 0
        spans.append((start, base + len(T)))
        out.append(T)
        pos = e
    out.append(S[pos:])
    return ''.join(out), spans


def render_episode(R, ep, a, stats):
    """Turns with P/F/T rendered from the repaired requests, masks, sequences -> token rows."""
    turns = []
    for k in range(len(ep['_reqs'])):
        t = render_turn(R, ep, k)
        if t is None:
            continue
        t['i'] = len(turns)
        asst = t['asst']
        t['mask'] = mask_reason(t, asst, ep, a)
        t['think_tokens'] = think_tokens(R, asst.get('reasoning') or '')
        t['think_masked'] = (t['mask'] is None and a.think_cap > 0 and t['think_tokens'] >= a.think_cap - 8)
        # the trained form vs the sampled tokens (--logprobs captures carry token_ids; unchanged call turns only)
        ids = ep['_raw'][k]['response']['choices'][0].get('token_ids')
        if ids and t['T'] and not t['merged'] and asst.get('tool_calls'):
            T = sampled_form(t['T']) if a.sampled_end else t['T']
            stats['trained_form_vs_sampled_checked'] += 1
            stats['trained_form_vs_sampled_equal'] += R.ids(T) == ids
        turns.append(t)
    seqs = assign_sequences(turns, lambda t: t['mask'] is not None)
    rows = []
    for j, ks in sorted(seqs.items()):
        items = [(len(turns[i]['P']), len(turns[i]['P']) + len(turns[i]['T']), turns[i]['think_masked'])
                 for i in ks]
        S, spans = trained_text(turns[j]['F'], items, a.sampled_end)
        row = token_row(R, S, spans, a, stats)
        if row is None:
            continue
        row.update({'id': f'{ep["id"]}_s{turns[j]["k"]}', 'episode': ep['id'], 'task': ep['task'], 'run': ep['run'],
                    'kind': 'episode', 'trained_turns': len(ks), 'repaired': any(turns[i]['repaired'] for i in ks),
                    '_msgs': turns[j]['msgs'] + [turns[j]['asst']], '_tools': turns[j]['tools']})
        rows.append(row)
    return turns, seqs, rows


def token_row(R, S, spans, a, stats):
    ids, labels, same = rb.tokenize_sequence(R, S, spans)
    stats['segment_tokenization_equal'] += same
    stats['sequences_built'] += 1
    if len(ids) > a.max_len:
        stats['dropped_too_long'] += 1
        return None
    n = sum(x != -100 for x in labels)
    return {'input_ids': ids, 'labels': labels, 'n_tokens': len(ids), 'n_trained': n, 'n_trained_tokens': n,
            'segment_tokenization_equal': same}


def format_rows(R, ep, a, stats):
    """Call-format rows from merged repairs (any episode outside the selection, passing or not): context = the
    repaired prompt of the retry, trained = only the corrected call (after <channel|>), not the thinking."""
    rows = []
    for rp in ep['repairs']:
        if rp['mode'] != 'merge':
            continue
        t = render_turn(R, ep, rp['retry'])
        if t is None or mask_reason(t, t['asst'], ep, a):
            stats['format_rows_skipped'] += 1
            continue
        S, spans = trained_text(t['P'] + t['T'], [(len(t['P']), len(t['P']) + len(t['T']), True)], a.sampled_end)
        row = token_row(R, S, spans, a, stats)
        if row is None:
            continue
        row.update({'id': f'{ep["id"]}_f{t["k"]}', 'episode': ep['id'], 'task': ep['task'], 'run': ep['run'],
                    'kind': 'format_repair', 'trained_turns': 1, 'repaired': True})
        rows.append(row)
    return rows


def mask_reason(t, asst, ep, a):
    info = ep['turns'][t['k']]
    if t['T'] is None:
        return 'render'
    if info['finish'] == 'length':
        return 'length'
    if not asst.get('tool_calls'):
        return 'text_only'
    if info['rejected']:
        return 'adk_error'
    if not asst.get('reasoning') or '<|channel>' not in t['T']:
        return 'no_thinking'
    for c in asst['tool_calls']:
        if not well_formed(c):
            return 'bad_call'
    for seg in re.findall(r'<\|tool_call>(.*?)<tool_call\|>', t['T'], re.S):
        if call_problem(seg) and not call_problem(seg).startswith('script'):
            return 'bad_call'
    return None


# ------------------------------------------------------------------ main

def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def think_stats(xs):
    return {'turns': len(xs), 'mean': round(statistics.mean(xs), 1) if xs else 0, 'median': pct(xs, .5),
            'p90': pct(xs, .9), 'max': max(xs) if xs else 0, 'ge_248': sum(x >= 248 for x in xs),
            'ge_504': sum(x >= 504 for x in xs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default=','.join(RUNS), help='comma-separated, in selection-priority order')
    ap.add_argument('--raw-root', default=str(ROOT / 'data/raw'))
    ap.add_argument('--results-root', default=str(ROOT / 'results'))
    ap.add_argument('--tasks', default=str(ROOT / 'data/competition/tasks.jsonl'))
    ap.add_argument('--eval', default=str(ROOT / 'experiments/eval113.txt'))
    ap.add_argument('--heldout', default=str(ROOT / 'experiments/lora2_heldout.txt'))
    ap.add_argument('--train-tasks', default=str(ROOT / 'experiments/lora2_train_tasks.txt'))
    ap.add_argument('--resplit', action='store_true', help='rewrite the held-out split even if the files exist')
    ap.add_argument('--hf-tokenizer', default=str(ROOT / 'data/rft/tokenizer'))
    ap.add_argument('--template', default=str(ROOT / 'data/probe/chat_template_kaggle_june.jinja'))
    ap.add_argument('--tokenizer-config', default=str(ROOT / 'data/probe/tokenizer_config_kaggle_june.json'))
    ap.add_argument('--r001', action='store_true', help='also use data/rft/raw/R001 (v13; July template)')
    ap.add_argument('--out', default=str(ROOT / 'data/rft2'))
    ap.add_argument('--per-task', type=int, default=3)
    ap.add_argument('--think-cap', type=int, default=256,
                    help='thinking of >= cap-8 tokens is not trained (the call still is); 0 = off')
    ap.add_argument('--canonical-thinking-end', dest='sampled_end', action='store_false',
                    help="train the template's '\\n<channel|>' instead of the model's own thinking end")
    ap.add_argument('--max-len', type=int, default=32768)
    ap.add_argument('--selftest-only', action='store_true')
    ap.add_argument('--no-selftest', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {'args': vars(a)}
    runs = [r for r in a.runs.split(',') if r]
    rank = {r: n for n, r in enumerate(runs)}

    # ---- held-out split
    eval_tasks = [x.strip() for x in open(a.eval) if x.strip() and not x.startswith('#')]
    if Path(a.heldout).exists() and not a.resplit:
        held = [x.strip() for x in open(a.heldout) if x.strip() and not x.startswith('#')]
    else:
        held = split_heldout(eval_tasks)
        Path(a.heldout).write_text('# LoRA stage 2 held-out tasks (scripts/rft2_build.py split_heldout: repo-stratified '
                                   'third of eval113, sha1 order). No training sequence comes from these.\n' +
                                   '\n'.join(held) + '\n')
    train_tasks = sorted(set(eval_tasks) - set(held))
    Path(a.train_tasks).write_text('# LoRA stage 2 training tasks = eval113 minus lora2_heldout.txt\n' +
                                   '\n'.join(train_tasks) + '\n')
    held_set = set(held)

    # ---- step 1: load + group
    tasks = {}
    for line in open(a.tasks):
        t = json.loads(line)
        tasks[t['instance_id']] = t['problem_statement'].strip()
    find_task = TaskFinder(tasks)
    stats = collections.Counter()
    episodes, all_items = [], []
    for run in runs:
        recs = load_run(Path(a.raw_root) / run)
        results = load_results(Path(a.results_root) / run)
        rs = collections.Counter()
        groups, summaries = group_run(recs, find_task, rs)
        stats.update(rs)
        print(f'{run}: {len(recs)} requests: agent {rs["requests_agent"]}, compaction summaries '
              f'{rs["requests_summary"]}, other {rs["requests_other"]}; {len(groups)} episodes; linked by call ids '
              f'{rs["requests_linked_by_call_ids"]}, unassigned {rs["requests_unassigned"]}, restarted '
              f'{rs["episodes_restarted"]}; results for {len(results)} tasks')
        for task, reqs in sorted(groups.items()):
            all_items += [(d, run) for d in reqs]
            res = results.get(task, {})
            ep = {'id': f'{run}_{task}', 'run': run, 'task': task, 'repo': task.rsplit('_', 1)[0],
                  'heldout': task in held_set, 'resolved': res.get('resolved'), 'tool_calls': res.get('tool_calls'),
                  'total_llm_calls': res.get('total_llm_calls'), 'duration_seconds': res.get('duration_seconds'),
                  'error': res.get('error'), 'agent_patch_size': res.get('agent_patch_size'),
                  'requests': len(reqs), 'compactions': summaries.get(task, 0),
                  '_raw': reqs, '_reqs': [as_req(d) for d in reqs], '_renderer': 'june'}
            episodes.append(ep)
        stats['episodes_without_result'] += sum(1 for t in groups if t not in results)
        stats['results_without_episode'] += sum(1 for t in results if t not in groups)
    if a.r001:
        ns = argparse.Namespace(raw=str(ROOT / 'data/rft/raw/R001'), results=str(ROOT / 'results/R001_rft'),
                                tasks=a.tasks, pool=str(ROOT / 'experiments/rft_pool.txt'), round_ends='',
                                round_slack=0.0)
        groups, results, unassigned, n_req, _ = rb.build_episodes(ns)
        print(f'R001: {n_req} requests -> {len(groups)} episodes ({unassigned} unassigned)')
        for (r, task), reqs in sorted(groups.items()):
            reqs = [d for d in reqs if d['status'] == 200 and d['response'].get('choices')]
            res = results.get((r, task), {})
            episodes.append({'id': f'R001_r{r}_{task}', 'run': f'R001_r{r}', 'task': task,
                             'repo': task.rsplit('_', 1)[0], 'heldout': task in held_set,
                             'resolved': res.get('resolved'), 'tool_calls': res.get('tool_calls'),
                             'total_llm_calls': res.get('total_llm_calls'),
                             'duration_seconds': res.get('duration_seconds'), 'error': res.get('error'),
                             'agent_patch_size': res.get('agent_patch_size'), 'requests': len(reqs),
                             'compactions': 0, '_raw': reqs, '_reqs': [as_req(d) for d in reqs],
                             '_renderer': 'july'})
            rank.setdefault(f'R001_r{r}', len(runs))

    # ---- step 2: rendering self-test
    tok_dir = ensure_june_tokenizer(a.hf_tokenizer, out / 'tokenizer_june', a.template, a.tokenizer_config)
    R = make_renderer(tok_dir, a.template)
    renderers = {'june': R}
    if a.r001:
        renderers['july'] = make_renderer(a.hf_tokenizer)
    if not a.no_selftest:
        selftest(R, all_items, report, 'selftest')
        if a.r001:
            selftest(renderers['july'], [(d, e['run']) for e in episodes if e['_renderer'] == 'july'
                                         for d in e['_raw']], report, 'selftest_r001_july')
    if a.selftest_only:
        json.dump(report, open(out / 'build_report.json', 'w'), indent=1)
        return

    # ---- step 3/4: turns, compaction points, repairs (all episodes; pure Python)
    rep_stats = collections.Counter()
    for ep in episodes:
        reqs = ep['_reqs']
        ep['turns'] = turn_records(reqs)
        # prefix property between consecutive requests: compaction (prompt rewritten) vs append-only
        breaks = 0
        for x, y in zip(reqs, reqs[1:]):
            ma, mb = x['messages'], y['messages']
            if mb[:len(ma)] != ma:
                breaks += 1
            elif len(mb) <= len(ma) or rb.norm_asst(mb[len(ma)]) != rb.norm_asst(x['response']):
                stats['msg_response_mismatch'] += 1
        ep['prefix_breaks'] = breaks
        stats['prefix_breaks'] += breaks
        stats['episodes_with_compaction'] += ep['compactions'] > 0
        drop, override, repairs, unrep = plan_repairs(ep['turns'])
        ep['_drop'], ep['_override'], ep['repairs'] = drop, override, repairs
        ep['rejected_turns'] = sum(t['rejected'] for t in ep['turns'])
        for rp in repairs:
            rep_stats[f'chains_{rp["mode"]}'] += 1
            rep_stats[f'rejected_turns_{rp["mode"]}'] += len(rp['chain'])
            if rp['mode'] == 'merge':
                rep_stats['merge_same_args'] += rp['same_args']
            if rp['why']:
                rep_stats[f'unrepaired: {rp["why"]}'] += len(rp['chain'])
            if ep['resolved']:
                rep_stats[f'passing_rejected_turns_{rp["mode"]}'] += len(rp['chain'])
        rep_stats['rejected_turns'] += ep['rejected_turns']
        rep_stats['turns'] += len(ep['turns'])
        rep_stats['length_cut_turns'] += sum(t['finish'] == 'length' for t in ep['turns'])
        for t in ep['turns']:
            if t['rejected']:
                rep_stats['error: ' + re.sub(r"'[^']*'", "'…'", t['error'] or '').replace('\n', ' | ')[:80]] += 1
    n_breaks = stats['prefix_breaks']
    print(f'\n== episodes: {len(episodes)}; prefix breaks (compaction rewrites) {n_breaks} in '
          f'{sum(e["prefix_breaks"] > 0 for e in episodes)} episodes; compaction summaries linked '
          f'{sum(e["compactions"] for e in episodes)}; response not echoed in next request: '
          f'{stats["msg_response_mismatch"]}')
    print(f'   rejected turns {rep_stats["rejected_turns"]} of {rep_stats["turns"]} '
          f'({rep_stats["rejected_turns"] / max(rep_stats["turns"], 1):.1%}); repaired: merge '
          f'{rep_stats["rejected_turns_merge"]} turns ({rep_stats["chains_merge"]} chains, same args '
          f'{rep_stats["merge_same_args"]}), not repaired (masked) '
          f'{rep_stats["rejected_turns_none"]}; cut by max_tokens {rep_stats["length_cut_turns"]}')
    for k, v in sorted(rep_stats.items()):
        if k.startswith(('unrepaired', 'error:')):
            print(f'     {k}: {v}')

    # thinking lengths over all turns
    all_think = [think_tokens(renderers[e['_renderer']], t['reasoning']) for e in episodes for t in e['turns']
                 if not t['text_only']]

    per_run = collections.defaultdict(collections.Counter)
    for e in episodes:
        per_run[e['run']]['episodes'] += 1
        per_run[e['run']]['passing'] += bool(e['resolved'])
        per_run[e['run']]['compacted'] += e['compactions'] > 0
    for r, c in per_run.items():
        print(f'   {r}: {c["episodes"]} episodes, passing {c["passing"]}, compacted {c["compacted"]}')
    passing = [e for e in episodes if e['resolved']]
    side = {'train': [e for e in passing if not e['heldout']], 'heldout': [e for e in passing if e['heldout']]}
    split_rep = {k: {'passing_episodes': len(v), 'tasks_with_pass': len({e['task'] for e in v}),
                     'tasks': len(held) if k == 'heldout' else len(train_tasks)} for k, v in side.items()}
    print(f'   held-out split: {len(held)} held-out / {len(train_tasks)} train tasks; passing episodes '
          f'train {split_rep["train"]["passing_episodes"]} ({split_rep["train"]["tasks_with_pass"]} tasks), held-out '
          f'{split_rep["heldout"]["passing_episodes"]} ({split_rep["heldout"]["tasks_with_pass"]} tasks)')

    # ---- step 6: selection
    def total_think(e):
        return sum(think_tokens(renderers[e['_renderer']], t['reasoning']) for t in e['turns'])

    sel = []
    for task in sorted({e['task'] for e in side['train']}):
        cand = [e for e in side['train'] if e['task'] == task]
        cand.sort(key=lambda e: (rank.get(e['run'], 99), e['tool_calls'] if e['tool_calls'] is not None else 10 ** 6,
                                 total_think(e)))
        sel += cand[:a.per_task]
    assert not any(e['task'] in held_set for e in sel)

    # ---- step 7: render, mask, tokenize
    rows, rstats = [], collections.Counter()
    masks = collections.Counter()
    sel_think, trained_think = [], []
    for ep in sel:
        turns, seqs, erows = render_episode(renderers[ep['_renderer']], ep, a, rstats)
        ep['sequences'] = len(seqs)
        rstats['episode_sequences'] += len(seqs)
        masks['removed_by_repair'] += len(ep['turns']) - len(turns)
        for t in turns:
            masks[t['mask'] or ('trained_call_only' if t['think_masked'] else 'trained')] += 1
            if t['info']['text_only']:
                continue
            sel_think.append(t['think_tokens'])
            if t['mask'] is None and not t['think_masked']:
                trained_think.append(t['think_tokens'])
            rstats['merged_turns_trained'] += t['merged'] and t['mask'] is None
            rstats['repaired_context_turns_trained'] += t['repaired'] and t['mask'] is None
        rows += erows
    assert not any(r['task'] in held_set for r in rows)
    # call-format rows from merged repairs in the episodes not selected (train-side tasks only)
    sel_ids = {e['id'] for e in sel}
    fstats = collections.Counter()
    frows = [r for e in episodes if e['id'] not in sel_ids and not e['heldout']
             for r in format_rows(renderers[e['_renderer']], e, a, fstats)]
    assert not any(r['task'] in held_set for r in frows)

    with open(out / 'episodes.jsonl', 'w') as f:
        for ep in episodes:
            d = {k: v for k, v in ep.items() if not k.startswith('_')}
            d['turns'] = [{k: v for k, v in t.items() if k not in ('reasoning', 'args')} |
                          {'think_chars': len(t['reasoning'])} for t in ep['turns']]
            d['selected'] = ep in sel
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    by_ep = collections.defaultdict(list)
    for r in rows:
        by_ep[r['episode']].append({'id': r['id'], 'repaired': r['repaired'], 'messages': r['_msgs'],
                                    'tools': r['_tools']})
    with open(out / 'train.jsonl', 'w') as f:
        for ep in sel:
            d = {k: v for k, v in ep.items() if not k.startswith('_') and k != 'turns'}
            d['sequences'] = by_ep[ep['id']]
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    with open(out / 'train_tokens.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith('_')}) + '\n')
    with open(out / 'train_tokens_format.jsonl', 'w') as f:
        for r in frows:
            f.write(json.dumps(r) + '\n')

    lens = [r['n_tokens'] for r in rows]
    tr = [r['n_trained'] for r in rows]
    print(f'\n== selection: {len(sel)} passing episodes over {len({e["task"] for e in sel})} train tasks '
          f'(<= {a.per_task} per task; run priority {runs}, then fewer tool calls, then less thinking); by run: ' +
          ', '.join(f'{r} {sum(e["run"] == r for e in sel)}' for r in rank if any(e['run'] == r for e in sel)))
    print('   turns: ' + ', '.join(f'{k} {v}' for k, v in masks.most_common()))
    print(f'   trained turns with a merged repair (original thinking + corrected call) {rstats["merged_turns_trained"]}; '
          f'trained turns in a repaired context {rstats["repaired_context_turns_trained"]}')
    if rows:
        print(f'== tokens: {len(rows)} sequences ({sum(r["repaired"] for r in rows)} repaired, '
              f'{rstats["dropped_too_long"]} dropped > {a.max_len}), trained turns '
              f'{sum(r["trained_turns"] for r in rows)}; length max {max(lens)}, mean {statistics.mean(lens):.0f}; '
              f'total {sum(lens)} tokens, trained {sum(tr)} ({sum(tr) / sum(lens):.1%})')
        print(f'   segment-wise tokenization == whole: {rstats["segment_tokenization_equal"]}/'
              f'{rstats["sequences_built"]}; trained form ({"sampled" if a.sampled_end else "canonical"} thinking '
              f'end) == sampled token_ids: {rstats["trained_form_vs_sampled_equal"]}/'
              f'{rstats["trained_form_vs_sampled_checked"]} unrepaired call turns')
    fl = [r['n_tokens'] for r in frows]
    print(f'== format rows (train_tokens_format.jsonl; merged repairs outside the selection, call tokens only): '
          f'{len(frows)} rows from {len({r["episode"] for r in frows})} episodes, {sum(fl)} tokens, trained '
          f'{sum(r["n_trained"] for r in frows)}; skipped {fstats["format_rows_skipped"]}')
    ts = {'all_turns': think_stats(all_think), 'selected_episodes_turns': think_stats(sel_think),
          'trained_thinking_turns': think_stats(trained_think)}
    for k, v in ts.items():
        print(f'   thinking tokens, {k}: {v}')

    report['episodes'] = {'n': len(episodes), 'passing': len(passing), 'per_run': {r: dict(c) for r, c in
                                                                                  per_run.items()},
                          'checks': dict(stats)}
    report['repairs'] = dict(rep_stats)
    report['split'] = split_rep | {'heldout_file': a.heldout, 'train_tasks_file': a.train_tasks}
    report['selection'] = {'episodes': len(sel), 'tasks': len({e['task'] for e in sel}), 'ids': [e['id'] for e in sel],
                           'turn_masks': dict(masks), 'by_run': dict(collections.Counter(e['run'] for e in sel))}
    report['tokens'] = {'sequences': len(rows), 'repaired_sequences': sum(r['repaired'] for r in rows),
                        'trained_turns': sum(r['trained_turns'] for r in rows), 'max_len': max(lens, default=0),
                        'mean_len': statistics.mean(lens) if lens else 0, 'total_tokens': sum(lens),
                        'trained_tokens': sum(tr), **dict(rstats)}
    report['thinking'] = ts
    report['format_rows'] = {'rows': len(frows), 'episodes': len({r['episode'] for r in frows}), 'tokens': sum(fl),
                             'trained_tokens': sum(r['n_trained'] for r in frows), **dict(fstats)}
    json.dump(report, open(out / 'build_report.json', 'w'), indent=1)
    print(f'\nwrote {out}/episodes.jsonl, train.jsonl, train_tokens.jsonl, train_tokens_format.jsonl, '
          f'build_report.json; {a.heldout}, '
          f'{a.train_tasks}')


if __name__ == '__main__':
    sys.exit(main())
