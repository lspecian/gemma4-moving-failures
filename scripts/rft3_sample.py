"""rft3 step B: fill Gemma 4's own thinking (+ visible text) into the synthetic trajectories by sampling the policy at
every trained state (research/notes/14-oracle-trajectories.md §2 and §5 "Thinking, per synthetic turn").

    # a measurement pass on a fixed random subset first (writes data/rft3/sampled/dev/), then the report
    .venv/bin/python scripts/rft3_sample.py sample --dev 50 --k 8 --concurrency 24
    .venv/bin/python scripts/rft3_sample.py report --dev
    # the full pass (resumable: rerun the same command after an interruption), then fallback as a second pass
    .venv/bin/python scripts/rft3_sample.py sample --k 8 --k-kind locate=4,check=4,submit=2 --concurrency 24
    .venv/bin/python scripts/rft3_sample.py sample --fallback --concurrency 24
    .venv/bin/python scripts/rft3_sample.py report
    .venv/bin/python scripts/rft3_sample.py export            # -> data/rft3/sampled/<source>.records.jsonl (step D)
    .venv/bin/python scripts/rft3_sample.py show --id ID [--dev]

Input: data/rft3/states/<source>.jsonl (scripts/rft3_states.py schema; train turns have thinking/visible = null).
--version v20: data/rft3/states_v20 and output data/rft3/sampled_v20. --records FILE: the records from that file (e.g.
data/rft3/states_v20/subset900_filled.jsonl from scripts/rft3_carry.py): train turns with sampling.carried keep their
thinking and are neither sampled nor exported again (only the rest is GPU work).
Server: --server (default http://127.0.0.1:8002/v1 = the vast.ai tunnel of scripts/vast_up.sh: Kaggle's vLLM 0.19.1,
June chat template, gemma4 tool + reasoning parsers, thinking_token_budget). The model id comes from /v1/models
(--model; with an adapter served via LORA=name=dir, auto picks the adapter). llama.cpp (scripts/serve_local.sh, :8001)
works for smoke tests: no `n` (K sequential requests), no thinking budget, no token ids, its own (July) template.

1. Bootstrap (UI-TARS-style): per record, the train turns IN ORDER (turn k's context carries the thinking/visible
   kept for the earlier turns; context turns keep their real text): the exact agent request = the record's request
   params (agent-v19: temperature 0.2, top_p 0.95, max_completion_tokens 1536, enable_thinking, thinking_token_budget
   256) with rft3_states.messages_at(record, k), sent once with n=K (vLLM) or K times (fallback). No hint: the model
   does not see the oracle call.
2. Parse each sample: thinking (message.reasoning / reasoning_content), visible text (message.content), the server's
   parsed tool_calls; with --token-ids also the raw generation (decoded token ids, raw_report.segments: thinking
   <|channel>...<channel|>, text, <|tool_call>...<tool_call|>) for exact token counts and raw_report.call_problem.
   Problems (never a match): length (finish=length), no_call (text only), multi_call, bad_json, unknown tool,
   malformed (rft2_build.well_formed: skill_name exactly repo-tools, file_path a bare script name, not nested in args),
   raw:<call_problem>, unknown script. Thinking tokens: from the raw tokens, else the tokenizer on the thinking text;
   "at budget" = >= thinking_token_budget - 8 (raw_report's rule: vLLM cut it).
3. Match rule vs the oracle call (match()):
   - submit_patch: the same tool. check_patch.py, locate.py: the same script (locate's query wording differs; the
     oracle call is kept).
   - show.py: the same target file AND (symbol equal to the oracle's symbol (Class.method == method unless the last
     part is a dunder or the classes differ) or to a gold function of that file) OR (a line range - start..end, start
     alone = start-20..start+20 as show.py reads it - covering every gold line the oracle read printed; when the oracle
     read printed no gold line, overlapping at least half of the oracle's printed lines).
   - outline.py: its files include the oracle's file(s). history.py: the same mode (similar=... any wording;
     target=: the same file).
   - search.py: the oracle's identifier occurs as a whole word in the sample's text/regex (retry_counter does not
     contain retry_count), or the sample's text (>= 4 characters) is a substring of the oracle's identifier (a broader
     search: its hits include the oracle's). (Whether a different search would show the same
     gold file is unknowable without running it; this is the "argument appears in the oracle args" rule.)
   - any other script: the same script and the same normalized args.
   - edit.py (strict): the same target file; the sample's lines (start..end; start alone with one line = that line,
     with N lines = start..start+N-1 as an approximation of edit.py's infer_end; old=: the old text found once in the
     numbered lines printed earlier for that file, ignoring indentation) overlap the oracle's start..end; every line of
     the union window was printed earlier (else unknowable); and the change is close: both edits applied to the
     window, the multisets of inserted/removed code tokens (difflib over \\w+|punctuation tokens, indentation ignored)
     have F1 >= --edit-threshold (default 0.8; 1.0 = identical change). The F1 of every same-file sample is stored, so
     the report shows the distribution and the match rate at other thresholds without resampling. A different but also
     correct fix cannot be known: it does not match. matched_other: a sample that matches another (later) oracle edit
     of the same record (hunk order) is counted, not kept. v20 multi-edit calls (either side): every hunk resolved as
     above (edit_hunks); the windows of overlapping hunks of both calls must all be printed; results and F1 over all
     windows together (a one-hunk call of a two-hunk change scores low).
   Selection among matches: natural ending first (thinking below the budget), then a non-empty visible text, then the
   shortest thinking (tokens). The kept turn stores the sample's thinking and visible text; its call is the SAMPLE's
   call only when identical to the oracle call after normalization (same script, same args; values stripped, NEW/OLD
   up to a trailing newline), else the oracle call (the recorded result belongs to the oracle call). Note: the
   agent-v19 policy almost never writes visible text (7 of 391 E058 replies, and those were leaked thinking), so a
   non-empty visible text is a preference, not a requirement (--require-visible makes it one).
4. Fallback (--fallback; for turns without a match; inline for new turns, a second pass for turns sampled earlier):
   answer-anchored rationalization. Request A (thinking off, tool_choice none, n=--fb-n): the same context plus a user
   message "your next call will be X; write the brief reasoning (<= ~80 words) you would have had before deciding it
   yourself and the one sentence you would write before it, citing only facts visible above" -> THINKING:/SENTENCE:.
   Checks per candidate (shortest first): (a) no leak phrase ("I was told", "the given call", ...); (b) grounding:
   every path, code identifier (dotted, snake_case, camelCase, backticked) and number of >= 2 digits it mentions
   occurs in the context (system, issue, earlier calls, results and thinking; for an edit also the oracle NEW text,
   which the thinking may draft); (c) consistency: the June-template prompt (rft_build.Renderer on
   data/rft2/tokenizer_june, as vLLM renders it) + "<|channel>thought\\n" + thinking + "<channel|>" is continued
   WITHOUT the hint (/v1/completions; llama.cpp /completion) --verify-n times; every continuation's call must match
   the oracle call (rule 3). Pass -> kept with from=rationalized. Fail -> thinking stays null and the turn is labelled
   mask_thinking (step D trains the call only).
   Context caveat of the second pass: turns after a newly rationalized turn were sampled with that turn's thinking
   absent; each line records the thinking source of every earlier turn of its context (ctx) so step D can see it.
5. Output (append-only JSONL per source, one line per turn and pass; resume skips turns done; the last line wins):
   data/rft3/sampled/<source>.jsonl (--dev N: data/rft3/sampled/dev/), lines {id, source, k, kind, pass
   (sample|fallback), cfg, samples [{thinking, visible, call, problem, finish, think_tokens, tokens, at_budget, raw?}],
   matches [{ok, rule, why, sim, identical, other}], kept {thinking, visible, call, from, think_tokens, at_budget,
   call_from} | null, usage, seconds, ctx}.
6. report: per source and kind: match rate at 1/2/4/K samples, kept thinking tokens, share kept at the budget,
   problem shares, visible-text share, edit F1 distribution and the match rate per threshold, fallback outcomes;
   tokens generated, and the GPU time of the full set at --tok-s aggregate decode (+ prefill at --prefill-tok-s).
7. export: the states records with thinking/visible/call filled per trained turn and turn['sampling'] = {from:
   bootstrap|rationalized|null, mask_thinking (no thinking, or thinking at the budget: rft2 masks those), think_tokens,
   rule}.
8. selfcheck (no model): every oracle call matches itself through its rule, every oracle edit window is in the
   printed lines, one-token changes of oracle edits score below 1, a templated thought naming the call is grounded.
Tests: tests/test_rft3_sample.py (raw parser, sample parsing, match rules, grounding, one record on a fake server).

Smoke test (llama.cpp, 3 records, K=2, 2026-10-07): locate/show/search/submit match readily;
after the edits the policy runs run.py rather than check_patch.py (the prompt's "edit, then run.py, check_patch.py";
the oracle path has no run.py), and at the edit state it reads more (outline/run.py) - the expected low decisiveness.
Line F1 is strict on reformatting (the same change wrapped differently scores ~0.3); token F1 is lenient on one-token
changes (80% of one-token-perturbed oracle edits keep token F1 >= 0.8, 37% keep line F1 >= 0.8): the report shows
both per threshold.
"""

import argparse
import builtins
import collections
import concurrent.futures
import copy
import difflib
import json
import keyword
import os
import random
import re
import statistics
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import zlib

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rft3_states as S  # noqa: E402
from raw_report import call_problem, segments  # noqa: E402

STATES = os.path.join(ROOT, 'data', 'rft3', 'states')
OUT = os.path.join(ROOT, 'data', 'rft3', 'sampled')
RECORDS = None   # --records FILE: the records to sample (e.g. data/rft3/states_v20/subset900_filled.jsonl)


def configure(version=None, records=None):
    """--version: the agent version's states (rft3_states.VERSIONS) and output dir (v19 data/rft3/sampled, v20
    data/rft3/sampled_v20); --records: read the records from that file instead of <states>/<source>.jsonl."""
    global STATES, OUT, RECORDS, AGENT_SCRIPTS
    if version:
        S.use_version(version)
        STATES = S.STATES_DIR
        OUT = S.suffixed(os.path.join(ROOT, 'data', 'rft3', 'sampled'))
        AGENT_SCRIPTS = None
    if records:
        RECORDS = records


def carried(turn):
    """A train turn whose thinking was carried over from an earlier sampling pass (scripts/rft3_carry.py): it is
    not sampled again, its thinking/visible/sampling stay as they are."""
    return bool((turn.get('sampling') or {}).get('carried'))
SOURCES = ('oracle_clean', 'oracle_recovery', 'oracle_repair', 'relabel')
TOKENIZER = os.path.join(ROOT, 'data', 'rft', 'tokenizer', 'tokenizer.json')
DEFAULT_SERVER = 'http://127.0.0.1:8002/v1'
TOOLS = {'get_status', 'submit_patch', 'list_skills', 'load_skill', 'load_skill_resource', 'run_skill_script'}
STR = '<|"|>'
DEV_SEED = 0
SHOW_HALF = 20  # show.py start without end: +-LINE_HALF lines (v19 show.py)
KIND_OF = {'show_range': 'show', 'recovery_outline': 'outline', 'recovery_search': 'search',
           'recovery_history': 'history', 'edit_repair': 'edit_repair'}
LEAK = re.compile(r"\b(i was|i've been|i have been|was) (told|given|asked)\b|\bthe (given|provided|specified|"
                  r"suggested|requested) call\b|\b(next call is|next call will be|hint|instructions? say)\b|"
                  r"\bas (instructed|requested|specified)\b", re.I)

AGENT_SCRIPTS = None


def agent_scripts():
    global AGENT_SCRIPTS
    if AGENT_SCRIPTS is None:
        d = os.path.join(S.AGENT_DIR, 'skills', S.SKILL, 'scripts')
        AGENT_SCRIPTS = {n for n in os.listdir(d) if n.endswith('.py')} if os.path.isdir(d) else set()
    return AGENT_SCRIPTS


# ------------------------------------------------------------------ tokens

_TOK = []
_TOK_LOCK = threading.Lock()


def tokenizer():
    with _TOK_LOCK:
        if not _TOK:
            from tokenizers import Tokenizer
            _TOK.append(Tokenizer.from_file(TOKENIZER))
    return _TOK[0]


def n_tokens(text):
    return len(tokenizer().encode(text, add_special_tokens=False).ids) if text else 0


_REND = []
_REND_LOCK = threading.Lock()


def prefill_ids(messages, tools, thinking):
    """Token ids of the generation prompt as vLLM renders it with Kaggle's June template (BOS included), followed by
    the model's thinking block as it writes it: <|channel>thought\\n THINKING <channel|> (no newline before the end)."""
    with _REND_LOCK:
        R = S.renderer()
        ids = R.ids(R.text(messages, tools, True, True))
    return ids + tokenizer().encode('<|channel>thought\n' + thinking + '<channel|>', add_special_tokens=False).ids


# ------------------------------------------------------------------ Gemma 4 raw tool-call syntax

def parse_gemma_value(s, i):
    """(value, next index) of one Gemma 4 tool-call value at s[i]: <|"|>string<|"|>, {object}, [list], or a bare
    number / true / false / null / word."""
    if s.startswith(STR, i):
        j = s.find(STR, i + len(STR))
        if j < 0:
            raise ValueError('unterminated string')
        return s[i + len(STR):j], j + len(STR)
    if i < len(s) and s[i] == '{':
        return parse_gemma_object(s, i)
    if i < len(s) and s[i] == '[':
        out, i = [], i + 1
        while i < len(s) and s[i] != ']':
            v, i = parse_gemma_value(s, i)
            out.append(v)
            if i < len(s) and s[i] == ',':
                i += 1
        if i >= len(s):
            raise ValueError('unterminated list')
        return out, i + 1
    m = re.compile(r'[^,}\]]*').match(s, i)
    tok = m.group(0).strip()
    j = m.end()
    if tok in ('true', 'false'):
        return tok == 'true', j
    if tok == 'null':
        return None, j
    try:
        return (float(tok) if re.search(r'[.eE]', tok) else int(tok)), j
    except ValueError:
        return tok, j


def parse_gemma_object(s, i):
    if s[i] != '{':
        raise ValueError('expected {')
    out, i = {}, i + 1
    while i < len(s) and s[i] != '}':
        j = s.find(':', i)
        if j < 0:
            raise ValueError('key without value')
        key = s[i:j].strip()
        if not key or STR in key or '{' in key:
            raise ValueError(f'bad key {key[:30]!r}')
        v, i = parse_gemma_value(s, j + 1)
        out[key] = v
        if i < len(s) and s[i] == ',':
            i += 1
        elif i < len(s) and s[i] != '}':
            raise ValueError(f'unexpected {s[i]!r}')
    if i >= len(s):
        raise ValueError('unterminated object')
    return out, i + 1


def parse_gemma_call(text):
    """{'name', 'arguments'} of one raw call body 'call:NAME{...}' (the text between <|tool_call> and <tool_call|>)."""
    m = re.match(r'\s*call:([\w\-.]+)\s*', text)
    if not m:
        raise ValueError('no call:NAME')
    args, end = parse_gemma_object(text, m.end())
    if text[end:].strip():
        raise ValueError('trailing text after the arguments')
    return {'name': m.group(1), 'arguments': args}


# ------------------------------------------------------------------ samples

def well_formed(call):
    """rft2_build.well_formed on a parsed call {name, arguments}."""
    if call['name'] != 'run_skill_script':
        return True
    a = call['arguments']
    fp, args = a.get('file_path'), a.get('args')
    if a.get('skill_name') != S.SKILL or not isinstance(fp, str) or not re.fullmatch(r'[\w-]+\.py', fp):
        return False
    return not (isinstance(args, dict) and ('skill_name' in args or 'file_path' in args))


def classify_call(call, raw_call=None):
    """Problem of a parsed call (None when ADK would run it)."""
    if call['name'] not in TOOLS:
        return f'unknown tool {call["name"][:30]}'
    if not isinstance(call['arguments'], dict):
        return 'bad_json'
    if raw_call is not None:
        p = call_problem(raw_call)
        if p and not p.startswith('tool '):
            return f'raw: {p}'
    if not well_formed(call):
        return 'malformed'
    if call['name'] == 'run_skill_script':
        if 'args' in call['arguments'] and not isinstance(call['arguments']['args'], dict):
            return 'malformed'
        if agent_scripts() and call['arguments']['file_path'] not in agent_scripts():
            return f'unknown script {call["arguments"]["file_path"][:30]}'
    return None


def finish_sample(s, budget):
    s['think_tokens'] = s.get('think_tokens') if s.get('think_tokens') is not None else n_tokens(s['thinking'])
    s['at_budget'] = bool(budget) and s['think_tokens'] >= budget - 8
    if s.get('tokens') is None:
        s['tokens'] = s['think_tokens'] + n_tokens(s['visible']) + \
            (n_tokens(json.dumps(s['call'], ensure_ascii=False)) if s['call'] else 0)
    return s


def parse_raw(text, finish=None, budget=None, keep_raw=False):
    """A sample from a raw generation (special tokens kept): thinking, visible text, the call (our parser)."""
    think, visible, calls = segments(text)
    visible = visible.replace('<eos>', '')
    if think.startswith('thought\n'):
        think = think[len('thought\n'):]
    s = {'thinking': think, 'visible': visible.strip(), 'call': None, 'problem': None, 'finish': finish}
    if keep_raw:
        s['raw'] = text
    if finish == 'length':
        s['problem'] = 'length'
    elif not calls:
        s['problem'] = 'no_call'
    elif len(calls) > 1:
        s['problem'] = 'multi_call'
    else:
        try:
            s['call'] = parse_gemma_call(calls[0])
            s['problem'] = classify_call(s['call'], calls[0])
        except ValueError as e:
            s['problem'] = f'unparsable: {e}'
    if s['problem'] == 'length':
        s['call'] = None
    tok = n_tokens(think) if think else 0
    s['think_tokens'] = tok
    return finish_sample(s, budget)


def parse_choice(ch, budget=None, keep_raw=False):
    """A sample from one chat-completions choice (the server's parsed message; raw tokens when returned)."""
    msg = ch.get('message') or {}
    ids = ch.get('token_ids')
    raw = tokenizer().decode(ids, skip_special_tokens=False) if ids else None
    s = {'thinking': msg.get('reasoning') or msg.get('reasoning_content') or '',
         'visible': (msg.get('content') or '').strip(), 'call': None, 'problem': None,
         'finish': ch.get('finish_reason')}
    raw_calls = segments(raw)[2] if raw else None
    if keep_raw and raw is not None:
        s['raw'] = raw
    if raw is not None:
        rthink, _, _ = segments(raw)
        s['think_tokens'] = n_tokens(rthink[len('thought\n'):] if rthink.startswith('thought\n') else rthink)
        s['tokens'] = len(ids)
    elif ch.get('_completion_tokens'):  # one sample per request: the server's count
        s['tokens'] = ch['_completion_tokens']
    calls = msg.get('tool_calls') or []
    if s['finish'] == 'length':
        s['problem'] = 'length'
    elif not calls:
        s['problem'] = 'no_call'
    elif len(calls) > 1:
        s['problem'] = 'multi_call'
    else:
        f = calls[0].get('function') or {}
        a = f.get('arguments')
        try:
            a = json.loads(a) if isinstance(a, str) else a
        except ValueError:
            a = None
        if not isinstance(a, dict):
            s['problem'] = 'bad_json'
        else:
            s['call'] = {'name': f.get('name') or '', 'arguments': a}
            s['problem'] = classify_call(s['call'], raw_calls[0] if raw_calls and len(raw_calls) == 1 else None)
    return finish_sample(s, budget)


# ------------------------------------------------------------------ call helpers

def norm_path(p):
    p = str(p or '').strip().strip('`\'"')
    for pre in ('/workspace/', 'workspace/', './'):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


def script_of(call):
    if not call:
        return None
    if call['name'] != 'run_skill_script':
        return call['name']
    return S.script_of(call)


def args_of(call):
    a = (call or {}).get('arguments') or {}
    inner = a.get('args') if isinstance(a.get('args'), dict) else {}
    return {k: v if isinstance(v, str) else json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in inner.items()}


def target_of(args):
    return norm_path(args.get('target') or args.get('file') or args.get('path') or '')


def to_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def norm_args(call):
    out = {}
    for k, v in args_of(call).items():
        out[k] = v.rstrip('\n') if k in ('new', 'old', 'content', 'snippet') else v.strip()
        if k in ('target', 'file', 'path'):
            out[k] = norm_path(v)
    return out


def identical(a, b):
    return bool(a and b) and script_of(a) == script_of(b) and norm_args(a) == norm_args(b)


def call_text(call):
    """A call in the system prompt's notation (show.py  target=x  symbol=y)."""
    if call['name'] != 'run_skill_script':
        return call['name']
    parts = [script_of(call)]
    long = []
    for k, v in args_of(call).items():
        if '\n' in v or len(v) > 120:
            parts.append(f'{k}=<{k} below>')
            long.append(f'{k}:\n```\n{v}\n```')
        else:
            parts.append(f'{k}={v}')
    return '  '.join(parts) + ('\n' + '\n'.join(long) if long else '')


# ------------------------------------------------------------------ context

NUMBERED = re.compile(r'^\s*(\d+)(?:  (.*)|\s*)$')
HUNK = re.compile(r'^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@')
EDITED = re.compile(r'^EDITED (\S+) \(lines (\d+)-(\d+)\)')
EDITED_MULTI = re.compile(r'^EDITED (\S+) \(\d+ hunks: lines (\d+)-(\d+)')  # v20 multi-edit (first hunk's lines)


def result_text(result):
    try:
        d = json.loads(result)
    except (TypeError, ValueError):
        return str(result or '')
    if not isinstance(d, dict):
        return str(d)
    return '\n'.join(str(d.get(k) or '') for k in ('stdout', 'stderr', 'result', 'error') if d.get(k))


def known_lines(turns):
    """{file: {line number: text}} of the numbered source lines printed in these turns' results (show.py's "N  text";
    the file is the last header line naming a .py file, as toolbench.shown_lines). An EDITED result drops its line
    range and shifts the lines below by the edit's line delta (later prints win)."""
    known = collections.defaultdict(dict)
    for t in turns:
        out = result_text(t.get('result'))
        lines = out.splitlines()
        head = next((x for x in lines[:12] if x.startswith('EDITED ')), None)  # (v20: NOTE lines may come first)
        mm = EDITED_MULTI.match(head) if head else None
        m = EDITED.match(head) if head and not mm else None
        if mm:  # a multi-edit: every known line from its first hunk on is dropped (later prints win)
            f, a = norm_path(mm.group(1)), int(mm.group(2))
            known[f] = {n: x for n, x in known.get(f, {}).items() if n < a}
        elif m:
            f, a, b = norm_path(m.group(1)), int(m.group(2)), int(m.group(3))
            delta = 0
            for line in lines:
                h = HUNK.match(line)
                if h:
                    delta += int(h.group(2) or 1) - int(h.group(1) or 1)
            old = known.get(f, {})
            known[f] = {(n + delta if n > b else n): x for n, x in old.items() if not a <= n <= b}
        cur = None
        for line in lines:
            m = NUMBERED.match(line)
            if m and cur:
                known[cur][int(m.group(1))] = m.group(2) or ''
                continue
            toks = line.split()
            if not toks:
                continue
            cands = [toks[0]]
            if line.startswith('--- best match:') and len(toks) > 3:
                cands.append(toks[3])
            if toks[0] in ('EDITED', 'REPLACED') and len(toks) > 1:
                cands.append(toks[1])
            for c in cands:
                c = c.rstrip(':,')
                if c.endswith('.py'):
                    cur = norm_path(c)
                    break
    return {f: d for f, d in known.items() if d}


def printed_lines(result):
    """{file: set(line numbers)} printed in one result."""
    return {f: set(d) for f, d in known_lines([{'result': result}]).items()}


def gold_lines(gold, f):
    out = set()
    for sp in gold.get('spans') or []:
        if norm_path(sp['file']) != f:
            continue
        a, b = sp['a'], sp['b']
        out |= set(range(a, b + 1)) if b >= a else {max(1, a - 1), a}
    return out


# ------------------------------------------------------------------ match rules

def sym_eq(a, b):
    a, b = str(a or '').strip(), str(b or '').strip()
    if not a or not b:
        return False
    if a == b:
        return True
    pa, pb = a.split('.'), b.split('.')
    if pa[-1] != pb[-1] or pa[-1].startswith('__'):
        return False
    return len(pa) == 1 or len(pb) == 1 or pa[-2] == pb[-2]


def m(ok, rule, why='', **kw):
    return dict({'ok': bool(ok), 'rule': rule, 'why': why}, **kw)


def match_show(sa, oa, oturn, gold):
    f = target_of(oa)
    if target_of(sa) != f:
        return m(False, 'show', 'other file')
    o_printed = printed_lines(oturn.get('result')).get(f, set())
    needed = o_printed & gold_lines(gold, f)
    if sa.get('symbol'):
        if sym_eq(sa['symbol'], oa.get('symbol')):
            return m(True, 'show symbol')
        if any(norm_path(gf) == f and sym_eq(sa['symbol'], q) for gf, q in gold.get('functions') or []):
            return m(True, 'show gold function')
        return m(False, 'show', 'other symbol')
    a, b = to_int(sa.get('start')), to_int(sa.get('end'))
    if a is None:
        return m(False, 'show', 'no symbol or range')
    if b is None:
        a, b = max(1, a - SHOW_HALF), a + SHOW_HALF
    rng = set(range(a, b + 1))
    if needed:
        return m(needed <= rng, 'show range', '' if needed <= rng else 'misses gold lines')
    if not o_printed:
        oa_a, oa_b = to_int(oa.get('start')), to_int(oa.get('end'))
        o_printed = set(range(oa_a, (oa_b or oa_a) + 1)) if oa_a else set()
    ok = bool(o_printed) and len(o_printed & rng) * 2 >= len(o_printed)
    return m(ok, 'show range', '' if ok else 'little overlap')


def words(text):
    return set(re.findall(r'[A-Za-z_]\w*', text or ''))


def match_search(sa, oa):
    ovals = [v for k, v in oa.items() if k in ('text', 'regex') and v]
    svals = [v for k, v in sa.items() if k in ('text', 'regex') and v]
    if not svals:
        return m(False, 'search', 'no text')
    for o in ovals:
        o = o.strip()
        for s in svals:
            s = s.strip()
            if re.search(r'(?<!\w)' + re.escape(o) + r'(?!\w)', s):
                return m(True, 'search contains oracle text')
            if len(s) >= 4 and s in o:  # a broader text: its hits include the oracle's
                return m(True, 'search text in oracle text')
    return m(False, 'search', 'other text')


def split_lines(text):
    return text.split('\n')


def strip_prefixes(lines):
    """edit.py removes show.py's line-number prefixes from NEW when every non-blank line has one."""
    nb = [l for l in lines if l.strip()]
    if nb and all(re.match(r'^\s*\d+  ', l) for l in nb):
        return [re.sub(r'^\s*\d+  ', '', l) if l.strip() else '' for l in lines]
    return lines


def find_old(old, flines):
    """(start, end) of OLD's lines in the known lines of the file, ignoring indentation (unique match only)."""
    want = [l.strip() for l in old.strip('\n').split('\n')]
    if not want or not any(want):
        return None
    hits = []
    for a in sorted(flines):
        if all(flines.get(a + i, None) is not None and flines[a + i].strip() == w for i, w in enumerate(want)):
            hits.append((a, a + len(want) - 1))
    return hits[0] if len(hits) == 1 else None


def edit_of(args, flines):
    """(start, end, new lines, note) of an edit.py call on the known lines, or (None, None, None, why)."""
    if 'new' not in args:
        return None, None, None, 'no new'
    new = args['new']
    if new.endswith('\n'):
        new = new[:-1]  # edit.py drops one trailing newline of NEW
    new_lines = strip_prefixes(split_lines(new))
    a, b = to_int(args.get('start')), to_int(args.get('end'))
    note = ''
    if a is None and args.get('old'):
        old = args['old']
        r = find_old(old, flines)
        if r is None and ('\\n' in old or '\\"' in old):
            r = find_old(old.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'"), flines)
        if r is None:
            return None, None, None, 'old not found in printed lines'
        a, b = r
        note = 'old'
    elif a is None:
        return None, None, None, 'no start or old'
    elif b is None:
        b = a + max(0, len(new_lines) - 1)
        note = 'start only' if len(new_lines) > 1 else ''
    if b < a:
        return None, None, None, 'end before start'
    return a, b, new_lines, note


HUNK_KEY = re.compile(r'^(start|end|new|old)([2-9])$')


def edit_hunks(args, flines):
    """([(start, end, new lines)] ascending, note) of an edit.py call: the first hunk (start/end/new or old/new) and,
    v20 multi-edit, hunks 2.. (start2/end2/new2, old2/new2, ...; numbers of the file before the call); (None, why)
    when a hunk is not evaluable or two overlap."""
    a, b, new, note = edit_of({k: v for k, v in args.items() if not HUNK_KEY.match(k)}, flines)
    if a is None:
        return None, note
    hunks, notes = [(a, b, new)], [note] if note else []
    extra = collections.defaultdict(dict)
    for k, v in args.items():
        mk = HUNK_KEY.match(k)
        if mk:
            extra[int(mk.group(2))][mk.group(1)] = v
    for n in sorted(extra):
        a, b, new, note = edit_of(extra[n], flines)
        if a is None:
            return None, f'hunk {n}: {note}'
        hunks.append((a, b, new))
        if note:
            notes.append(f'hunk {n}: {note}')
    hunks.sort()
    if any(y[0] <= x[1] for x, y in zip(hunks, hunks[1:])):
        return None, 'overlapping hunks'
    return hunks, '; '.join(notes)


def windows(intervals):
    """Merged (lo, hi) of overlapping line intervals, ascending."""
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def apply_hunks(flines, lo, hi, hunks):
    """Lines lo..hi of the known file with the hunks inside that window applied (bottom-up)."""
    out = [flines[n] for n in range(lo, hi + 1)]
    for a, b, new in sorted((h for h in hunks if lo <= h[0] and h[1] <= hi), reverse=True):
        out[a - lo:b - lo + 1] = new
    return out


TOKEN_RE = re.compile(r'\w+|[^\w\s]')


def code_tokens(lines):
    return TOKEN_RE.findall('\n'.join(l.strip() for l in lines))


def change_counter(base, after):
    """Counter of ('-', token) / ('+', token) of the token-level diff base -> after (indentation ignored)."""
    bt, at = code_tokens(base), code_tokens(after)
    c = collections.Counter()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, bt, at, autojunk=False).get_opcodes():
        if op in ('replace', 'delete'):
            c.update(('-', t) for t in bt[i1:i2])
        if op in ('replace', 'insert'):
            c.update(('+', t) for t in at[j1:j2])
    return c


def norm_line(line):
    return re.sub(r'\s+', ' ', line.strip())


def line_change_counter(base, after):
    """Counter of ('-', line) / ('+', line) of the line diff base -> after (lines whitespace-normalized; blank lines
    ignored)."""
    bl = [norm_line(x) for x in base if x.strip()]
    al = [norm_line(x) for x in after if x.strip()]
    c = collections.Counter()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, bl, al, autojunk=False).get_opcodes():
        if op in ('replace', 'delete'):
            c.update(('-', x) for x in bl[i1:i2])
        if op in ('replace', 'insert'):
            c.update(('+', x) for x in al[j1:j2])
    return c


def change_f1(co, cs):
    if not co and not cs:
        return 1.0
    inter = sum((co & cs).values())
    return 2 * inter / (sum(co.values()) + sum(cs.values()))


def match_edit(sa, oa, known, threshold):
    """Edits (single or v20 multi-edit, either side) are compared as one change: per window of overlapping hunks of
    both calls (every line printed earlier), both results; then the result rule and the change F1 over all windows."""
    f = target_of(oa)
    if target_of(sa) != f:
        return m(False, 'edit', 'other file')
    flines = known.get(f, {})
    oh, _ = edit_hunks(oa, flines)
    if oh is None:
        return m(False, 'edit', 'oracle edit not evaluable')
    sh, note = edit_hunks(sa, flines)
    if sh is None:
        return m(False, 'edit', note)
    if not any(a <= ob and oa_ <= b for a, b, _ in sh for oa_, ob, _ in oh):
        return m(False, 'edit', 'other lines', note=note)
    wins = windows([(a, b) for a, b, _ in oh + sh])
    if any(n not in flines for lo, hi in wins for n in range(lo, hi + 1)):
        return m(False, 'edit', 'unseen lines', note=note)
    base, after_o, after_s = [], [], []
    for lo, hi in wins:
        base += [flines[n] for n in range(lo, hi + 1)]
        after_o += apply_hunks(flines, lo, hi, oh)
        after_s += apply_hunks(flines, lo, hi, sh)
    if [norm_line(x) for x in after_o if x.strip()] == [norm_line(x) for x in after_s if x.strip()]:
        return m(True, 'edit', 'same result', sim=1.0, tsim=1.0, note=note)
    sim = round(change_f1(line_change_counter(base, after_o), line_change_counter(base, after_s)), 3)
    tsim = round(change_f1(change_counter(base, after_o), change_counter(base, after_s)), 3)
    return m(sim >= threshold, 'edit', '' if sim >= threshold else 'different change', sim=sim, tsim=tsim,
             note=note)


def match(scall, oturn, gold, known, threshold=0.8):
    """Does a sampled call (parsed, well-formed) lead to the same action as the oracle turn's call? (module doc §3)"""
    ocall = oturn['call']
    if scall is None:
        return m(False, 'none', 'no call')
    if scall['name'] != ocall['name']:
        return m(False, 'tool', f'other tool {scall["name"]}')
    if ocall['name'] != 'run_skill_script':
        return m(True, ocall['name'], identical=True)
    so, ss = script_of(ocall), script_of(scall)
    same = identical(scall, ocall)
    if ss != so:
        return m(False, 'script', f'other script {ss}')
    sa, oa = args_of(scall), args_of(ocall)
    if same:
        r = m(True, so, 'identical')
    elif so in ('locate.py', 'check_patch.py'):
        r = m(True, so)
    elif so == 'show.py':
        r = match_show(sa, oa, oturn, gold)
    elif so == 'outline.py':
        sf = {norm_path(x) for x in re.split(r'[\s,]+', sa.get('files') or sa.get('target') or '') if x}
        of = {norm_path(x) for x in re.split(r'[\s,]+', oa.get('files') or oa.get('target') or '') if x}
        r = m(bool(of) and of <= sf, 'outline', '' if of <= sf else 'other files')
    elif so == 'history.py':
        if oa.get('similar'):
            r = m(bool(sa.get('similar')), 'history similar', '' if sa.get('similar') else 'other mode')
        else:
            r = m(target_of(sa) == target_of(oa), 'history target')
    elif so == 'search.py':
        r = match_search(sa, oa)
    elif so == 'edit.py':
        r = match_edit(sa, oa, known, threshold)
    else:
        r = m(False, so, 'args differ')
    r['identical'] = same
    return r


def choose(samples, matches, require_visible=False):
    """Index of the kept sample: natural ending first, then non-empty visible text, then the shortest thinking."""
    ok = [i for i, (s, x) in enumerate(zip(samples, matches)) if x['ok'] and (s['visible'] or not require_visible)]
    if not ok:
        return None
    return min(ok, key=lambda i: (samples[i]['at_budget'], not samples[i]['visible'], samples[i]['think_tokens'], i))


# ------------------------------------------------------------------ grounding (fallback)

# a path: slashes and a file extension in the last part (not "and/or"), or a bare *.py name
PATH_RE = re.compile(r'(?<![\w/.-])(?:/?[\w.-]+/)+[\w-]+\.\w+|(?<![\w/.-])[\w-]+\.py\b')
IDENT_RE = re.compile(r'(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|(?<![\w.])\w*_\w*|(?<![\w.])[a-z]+[A-Z]\w*|'
                      r'(?<![\w.])[A-Z][a-z0-9]+[A-Z]\w*')
NUM_RE = re.compile(r'(?<![\w.])\d{2,}(?![\w])')
BUILTIN = set(dir(builtins)) | set(keyword.kwlist) | {'self', 'cls'}


def call_allowance(call):
    """Text a rationalized thought may cite besides the context: the call's own argument values (the oracle's args
    are grounded by step A's rules; a show range is placed near a printed line reference, not on it; an edit's NEW
    is code the thought may draft)."""
    return '\n'.join(args_of(call).values())


def context_text(messages):
    parts = []
    for msg in messages:
        if isinstance(msg.get('content'), str):
            parts.append(msg['content'])
        for k in ('reasoning', 'reasoning_content'):
            if msg.get(k):
                parts.append(msg[k])
        for c in msg.get('tool_calls') or []:
            parts.append(c['function'].get('arguments') or '')
    text = '\n'.join(parts)
    return text + '\n' + text.replace('\\n', '\n').replace('\\"', '"')


def ungrounded(thinking, ctx, allowed=''):
    """Mentions in THINKING (paths, code identifiers, numbers >= 2 digits) found neither in CTX nor in ALLOWED."""
    w_ctx = words(ctx) | words(allowed)
    nums = set(re.findall(r'\d+', ctx)) | set(re.findall(r'\d+', allowed))
    bad = []
    text = thinking
    for p in PATH_RE.findall(text):
        q = norm_path(p.rstrip('.,:;)'))
        if q.startswith(('http', 'www.')) or '/' not in q and not q.endswith('.py'):
            continue
        if q not in ctx and q not in allowed:
            bad.append(q)
    text_np = PATH_RE.sub(' ', text)
    for code in re.findall(r'`([^`\n]+)`', text_np):
        for wd in re.findall(r'[A-Za-z_]\w*', code):
            if len(wd) > 2 and wd not in w_ctx and wd not in BUILTIN:
                bad.append(wd)
    for ident in IDENT_RE.findall(re.sub(r'`[^`\n]+`', ' ', text_np)):
        parts = [x for x in ident.split('.') if x]
        if all(len(x) <= 1 for x in parts):  # e.g, i.e
            continue
        if parts and not all(x in w_ctx or x in BUILTIN for x in parts) and len(ident) > 2:
            bad.append(ident)
    for n in NUM_RE.findall(text_np):
        if n not in nums:
            bad.append(n)
    return sorted(set(bad))


HINT = """(This message is a data-generation request, not part of the task.) Your next call will be:
{call}
Write the short reasoning you would think right before deciding on this call yourself, based only on the results \
above: at most 80 words, first person, plain, in the style of your own thinking (for example: "The locate output \
lists X in Y. I'll show Z to see how it builds W."). Mention only files, names, line numbers and facts that appear \
above{extra}; do not mention this message or that the call was given to you. End the reasoning with what you \
will do next, naming its exact target (the script and the file, symbol, line range or text it uses). Then write the \
one short sentence you would write before the call (what the last result told you and what this call is for).
Answer exactly in this form:
THINKING: <the reasoning>
SENTENCE: <the sentence>"""


def parse_rationale(text):
    mt = re.search(r'THINKING:\s*(.*?)\s*(?:\n\s*)?SENTENCE:\s*(.*)', text or '', re.S)
    if not mt:
        return None, None
    think = mt.group(1).strip().strip('"').strip()
    sent = mt.group(2).strip().split('\n')[0].strip().strip('"').strip()
    return (think or None), (sent or None)


# ------------------------------------------------------------------ server

class Server:
    def __init__(self, url, model=None, timeout=1800, retries=4):
        self.url, self.timeout, self.retries = url.rstrip('/'), timeout, retries
        d = self.get('/models')
        data = d.get('data') or []
        ids = [x['id'] for x in data]
        self.kind = 'llamacpp' if any(x.get('owned_by') == 'llamacpp' for x in data) else 'vllm'
        if model:
            self.model = model
        elif len(ids) > 1:  # vLLM lists the base model first, then the served LoRA modules: sample the adapter
            self.model = ids[-1]
        else:
            self.model = ids[0]
        self.models = ids
        self.n_ok = self.kind == 'vllm'
        self.lock = threading.Lock()

    def get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=60) as r:
            return json.load(r)

    def post(self, url, body):
        err = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                msg = e.read().decode('utf-8', 'replace')[:600]
                if e.code < 500:
                    raise RuntimeError(f'HTTP {e.code}: {msg}') from None
                err = f'HTTP {e.code}: {msg}'
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                err = repr(e)
            time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError(f'request failed after {self.retries} attempts: {err}')

    def chat(self, body, n, seed=None):
        """(choices, usage, requests): n samples in one request (vLLM) or n sequential requests."""
        body = dict(body, model=self.model)
        if n > 1 and self.n_ok:
            try:
                d = self.post(self.url + '/chat/completions', dict(body, n=n, **({'seed': seed} if seed is not None
                                                                                  else {})))
                return d['choices'], d.get('usage') or {}, 1
            except RuntimeError as e:
                if "'n'" not in str(e) and ' n ' not in str(e) and 'n must' not in str(e):
                    raise
                with self.lock:
                    self.n_ok = False
        choices, usage = [], collections.Counter()
        for i in range(n):
            b = dict(body, **({'seed': seed + i} if seed is not None else {}))
            d = self.post(self.url + '/chat/completions', b)
            ch = d['choices'][0]
            ch['index'] = i
            u = d.get('usage') or {}
            ch['_completion_tokens'] = u.get('completion_tokens')
            choices.append(ch)
            usage['prompt_tokens'] += u.get('prompt_tokens') or 0
            usage['completion_tokens'] += u.get('completion_tokens') or 0
            usage['cached_tokens'] += ((u.get('prompt_tokens_details') or {}).get('cached_tokens') or
                                       (d.get('timings') or {}).get('cache_n') or 0)
        return choices, dict(usage), n

    def complete(self, prompt_ids, max_tokens, temperature, top_p, n=1, seed=None):
        """[(raw text, finish)], usage: raw continuations of a token prompt (special tokens kept)."""
        if self.kind == 'vllm':
            body = {'model': self.model, 'prompt': prompt_ids, 'max_tokens': max_tokens, 'temperature': temperature,
                    'top_p': top_p, 'n': n, 'skip_special_tokens': False, 'spaces_between_special_tokens': False}
            if seed is not None:
                body['seed'] = seed
            d = self.post(self.url + '/completions', body)
            return [(c['text'], c.get('finish_reason')) for c in d['choices']], d.get('usage') or {}
        base = self.url[:-3] if self.url.endswith('/v1') else self.url
        out, usage = [], collections.Counter()
        for i in range(n):
            body = {'prompt': prompt_ids, 'n_predict': max_tokens, 'temperature': temperature, 'top_p': top_p,
                    'cache_prompt': True}
            if seed is not None:
                body['seed'] = seed + i
            d = self.post(base + '/completion', body)
            fin = 'length' if d.get('stop_type') == 'limit' or d.get('truncated') else 'stop'
            out.append((d.get('content') or '', fin))
            usage['prompt_tokens'] += d.get('tokens_evaluated') or 0
            usage['completion_tokens'] += d.get('tokens_predicted') or 0
        return out, dict(usage)


def add_usage(acc, u):
    acc['prompt_tokens'] = acc.get('prompt_tokens', 0) + (u.get('prompt_tokens') or 0)
    acc['completion_tokens'] = acc.get('completion_tokens', 0) + (u.get('completion_tokens') or 0)
    c = u.get('cached_tokens')
    if c is None:
        c = (u.get('prompt_tokens_details') or {}).get('cached_tokens')
    acc['cached_tokens'] = acc.get('cached_tokens', 0) + (c or 0)
    return acc


# ------------------------------------------------------------------ store

class Store:
    """Append-only JSONL per source; {(id, k, pass): last line}."""

    def __init__(self, out_dir):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.lines = {}
        self.lock = threading.Lock()
        for src in SOURCES:
            p = self.path(src)
            if os.path.exists(p):
                for line in open(p):
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue  # a line cut by a crash
                    self.lines[(d['id'], d['k'], d['pass'])] = d

    def path(self, src):
        return os.path.join(self.dir, f'{src}.jsonl')

    def get(self, rid, k, p):
        return self.lines.get((rid, k, p))

    def put(self, d):
        with self.lock:
            with open(self.path(d['source']), 'a') as f:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
            self.lines[(d['id'], d['k'], d['pass'])] = d


def final_of(sample_line, fb_line):
    """(kept dict or None, 'bootstrap'|'rationalized'|None) of a turn."""
    if fb_line and fb_line.get('kept'):
        return fb_line['kept'], 'rationalized'
    if sample_line and sample_line.get('kept'):
        return sample_line['kept'], 'bootstrap'
    return None, None


def apply_kept(turn, kept):
    if kept:
        turn['thinking'] = kept.get('thinking') or None
        turn['visible'] = kept.get('visible') or None
        turn['call'] = kept.get('call') or turn['call']


# ------------------------------------------------------------------ per turn

class Cfg(argparse.Namespace):
    def k_for(self, kind):
        return self.k_kind.get(coarse_kind(kind), self.k)


def coarse_kind(kind, call=None):
    k = KIND_OF.get(kind, kind)
    if k == 'recovery_next':
        sc = script_of(call) if call else None
        return (sc or 'read').replace('.py', '')
    return k


def fb_wanted(cfg, oturn):
    """--fb-kinds: rationalize only these coarse kinds (empty: every unmatched turn)."""
    kinds = getattr(cfg, 'fb_kinds', None)
    return not kinds or coarse_kind(oturn.get('kind'), oturn['call']) in kinds


def turn_seed(cfg, rid, k, salt=0):
    if cfg.seed is None:
        return None
    return (zlib.crc32(f'{rid}|{k}|{salt}'.encode()) + cfg.seed) % (2 ** 31 - 1000)


def sample_turn(rec, turns, k, cfg, srv):
    t0 = time.time()
    oturn = rec['turns'][k]
    work = {'request': rec['request'], 'turns': turns}
    body = {key: (S.messages_at(work, k) if key == 'messages' else copy.deepcopy(v))
            for key, v in rec['request'].items()}
    if cfg.temperature is not None:
        body['temperature'] = cfg.temperature
    if cfg.token_ids:
        body['return_token_ids'] = True
    budget = rec['request'].get('thinking_token_budget')
    K = cfg.k_for(oturn.get('kind'))
    choices, usage, nreq = srv.chat(body, K, turn_seed(cfg, rec['id'], k))
    samples = [parse_choice(c, budget, keep_raw=cfg.keep_raw) for c in choices]
    ct, est = usage.get('completion_tokens'), sum(s['tokens'] for s in samples)
    if nreq == 1 and len(samples) > 1 and ct and est and not cfg.token_ids:
        for s in samples:  # n samples in one request: scale our estimates to the server's total
            s['tokens'] = round(s['tokens'] * ct / est)
    known = known_lines(turns[:k])
    matches = []
    for s in samples:
        if s['problem']:
            matches.append(m(False, 'problem', s['problem']))
            continue
        x = match(s['call'], oturn, rec['gold'], known, cfg.edit_threshold)
        if not x['ok'] and script_of(oturn['call']) == 'edit.py' and script_of(s['call']) == 'edit.py':
            for j in range(k + 1, len(rec['turns'])):  # another oracle edit of this record (hunk order)?
                ot = rec['turns'][j]
                if ot['label'] == 'train' and script_of(ot['call']) == 'edit.py' and \
                        match(s['call'], ot, rec['gold'], known, cfg.edit_threshold)['ok']:
                    x['other'] = j
                    break
        matches.append(x)
    i = choose(samples, matches, cfg.require_visible)
    kept = None
    if i is not None:
        s = samples[i]
        same = bool(matches[i].get('identical'))
        kept = {'thinking': s['thinking'], 'visible': s['visible'], 'call': s['call'] if same else oturn['call'],
                'call_from': 'sample' if same else 'oracle', 'from': 'bootstrap', 'sample': i,
                'think_tokens': s['think_tokens'], 'at_budget': s['at_budget'], 'rule': matches[i]['rule']}
    return {'id': rec['id'], 'source': rec['source'], 'k': k, 'kind': oturn.get('kind'),
            'ckind': coarse_kind(oturn.get('kind'), oturn['call']), 'pass': 'sample',
            'cfg': {'K': K, 'temperature': body.get('temperature'), 'top_p': body.get('top_p'), 'model': srv.model,
                    'server': srv.kind, 'budget': budget, 'edit_threshold': cfg.edit_threshold},
            'samples': samples, 'matches': matches, 'kept': kept, 'usage': add_usage({}, usage), 'requests': nreq,
            'seconds': round(time.time() - t0, 1), 'ctx': ctx_sources(rec, turns, k)}


def ctx_sources(rec, turns, k):
    return ['real' if t['label'] == 'context' and t.get('thinking') else
            ('none' if not t.get('thinking') else t.get('_src', '?')) for t in turns[:k]]


def fallback_turn(rec, turns, k, cfg, srv):
    """Answer-anchored rationalization + grounding + unhinted consistency check (module doc §4)."""
    t0 = time.time()
    oturn = rec['turns'][k]
    work = {'request': rec['request'], 'turns': turns}
    msgs = S.messages_at(work, k)
    budget = rec['request'].get('thinking_token_budget')
    is_edit = script_of(oturn['call']) == 'edit.py'
    extra = ' (and the new code of the call, which you are about to write)' if is_edit else ''
    body = {key: copy.deepcopy(v) for key, v in rec['request'].items() if key not in ('messages',
                                                                                      'thinking_token_budget')}
    body['messages'] = msgs + [{'role': 'user', 'content': HINT.format(call=call_text(oturn['call']), extra=extra)}]
    body['chat_template_kwargs'] = {'enable_thinking': False}
    body['tool_choice'] = 'none'
    body['max_completion_tokens'] = 400
    body['temperature'] = cfg.fb_temperature
    usage = {}
    out = {'id': rec['id'], 'source': rec['source'], 'k': k, 'kind': oturn.get('kind'),
           'ckind': coarse_kind(oturn.get('kind'), oturn['call']), 'pass': 'fallback', 'candidates': [],
           'kept': None, 'label': 'mask_thinking', 'ctx': ctx_sources(rec, turns, k)}
    try:
        choices, u, _ = srv.chat(body, cfg.fb_n, turn_seed(cfg, rec['id'], k, 1))
    except RuntimeError as e:
        out.update(error=str(e)[:300], usage=usage, seconds=round(time.time() - t0, 1))
        return out
    add_usage(usage, u)
    ctx = context_text(msgs)
    allowed = call_allowance(oturn['call'])
    cands = []
    for ch in choices:
        msg = ch.get('message') or {}
        text = msg.get('content') or ''
        think, sent = parse_rationale(text)
        c = {'text': text[:2000], 'thinking': think, 'sentence': sent}
        if not think:
            c['fail'] = 'unparsable'
        elif LEAK.search(think) or (sent and LEAK.search(sent)):
            c['fail'] = 'leak'
        else:
            c['ungrounded'] = ungrounded(think + '\n' + (sent or ''), ctx, allowed)
            if c['ungrounded']:
                c['fail'] = 'ungrounded'
            else:
                c['think_tokens'] = n_tokens(think)
        cands.append(c)
    known = known_lines(turns[:k])
    order = sorted((c for c in cands if 'fail' not in c), key=lambda c: c['think_tokens'])
    for c in order[:cfg.fb_verify_max]:
        ids = prefill_ids(msgs, rec['request'].get('tools'), c['thinking'])
        try:
            outs, u = srv.complete(ids, rec['request'].get('max_completion_tokens') or 1536,
                                   cfg.temperature if cfg.temperature is not None else rec['request']['temperature'],
                                   rec['request'].get('top_p', 1.0), n=cfg.verify_n,
                                   seed=turn_seed(cfg, rec['id'], k, 2))
        except RuntimeError as e:
            c['fail'] = f'verify error {str(e)[:120]}'
            continue
        add_usage(usage, u)
        ver = [parse_raw(text, fin, None) for text, fin in outs]
        vm = [match(v['call'], oturn, rec['gold'], known, cfg.edit_threshold) if not v['problem'] else
              m(False, 'problem', v['problem']) for v in ver]
        c['verify'] = [{'call': v['call'], 'problem': v['problem'], 'visible': v['visible'][:300], 'ok': x['ok'],
                        'why': x['why'], 'sim': x.get('sim')} for v, x in zip(ver, vm)]
        if all(x['ok'] for x in vm):
            c['ok'] = True
            out['kept'] = {'thinking': c['thinking'], 'visible': c['sentence'] or '', 'call': oturn['call'],
                           'call_from': 'oracle', 'from': 'rationalized', 'think_tokens': c['think_tokens'],
                           'at_budget': False, 'rule': vm[0]['rule']}
            out['label'] = 'rationalized'
            break
        c['fail'] = 'verify mismatch'
    out['candidates'] = cands
    out.update(usage=usage, seconds=round(time.time() - t0, 1))
    return out


# ------------------------------------------------------------------ per record

def run_record(rec, st, cfg, srv, stats):
    turns = copy.deepcopy(rec['turns'])
    for t in turns:
        t['_src'] = 'real' if t['label'] == 'context' and t.get('thinking') else None
    for k, t in enumerate(rec['turns']):
        if t['label'] != 'train':
            continue
        if carried(t):
            turns[k]['_src'] = t['sampling'].get('from')
            continue
        sl = st.get(rec['id'], k, 'sample')
        if sl is None:
            sl = sample_turn(rec, turns, k, cfg, srv)
            st.put(sl)
            stats['sampled'] += 1
            stats['matched'] += sl['kept'] is not None
        fb = st.get(rec['id'], k, 'fallback')
        if cfg.fallback and sl['kept'] is None and fb is None and fb_wanted(cfg, rec['turns'][k]):
            fb = fallback_turn(rec, turns, k, cfg, srv)
            st.put(fb)
            stats['fallback'] += 1
            stats['fallback_ok'] += fb['kept'] is not None
        kept, src = final_of(sl, fb)
        apply_kept(turns[k], kept)
        turns[k]['_src'] = src
    stats['records'] += 1


def load_records(sources, only=None):
    out = []
    for src in sources:
        if RECORDS:
            rs = (r for r in S.read_records([RECORDS]) if r['source'] == src)
        else:
            rs = S.read_records([os.path.join(STATES, f'{src}.jsonl')])
        for r in rs:
            if only and r['id'] not in only:
                continue
            out.append(r)
    return out


def dev_ids(n, sources=SOURCES):
    """A fixed random subset of n trajectories over all sources (seed 0, sorted ids)."""
    ids = sorted(r['id'] for r in load_records(sources))
    return set(random.Random(DEV_SEED).sample(ids, min(n, len(ids))))


def parse_k_kind(s):
    out = {}
    for part in (s or '').split(','):
        if part.strip():
            k, v = part.split('=')
            out[k.strip()] = int(v)
    return out


def cmd_sample(a):
    only = set(a.only.split(',')) if a.only else None
    if a.dev:
        d = dev_ids(a.dev)
        only = d if only is None else only & d
    recs = load_records(a.sources.split(','), only)
    if a.limit:
        per = collections.defaultdict(int)
        keep = []
        for r in recs:
            if per[r['source']] < a.limit:
                keep.append(r)
                per[r['source']] += 1
        recs = keep
    srv = Server(a.server, a.model)
    out_dir = a.out or (os.path.join(OUT, 'dev') if a.dev else OUT)
    st = Store(out_dir)
    cfg = Cfg(**vars(a))
    cfg.k_kind = parse_k_kind(a.k_kind)
    cfg.fb_kinds = {x.strip() for x in (a.fb_kinds or '').split(',') if x.strip()}
    conc = 1 if srv.kind == 'llamacpp' else a.concurrency
    todo = []
    for r in recs:
        ks = [k for k, t in enumerate(r['turns']) if t['label'] == 'train' and not carried(t)]
        need = any(st.get(r['id'], k, 'sample') is None for k in ks) or (
            a.fallback and any((st.get(r['id'], k, 'sample') or {}).get('kept') is None and
                               st.get(r['id'], k, 'fallback') is None and fb_wanted(cfg, r['turns'][k]) for k in ks))
        if need:
            todo.append(r)
    print(f'server {srv.kind} {a.server} model {srv.model} (listed: {srv.models}); {len(recs)} records, '
          f'{len(todo)} with work; out {out_dir}; K={a.k} {cfg.k_kind or ""} concurrency {conc}'
          f'{" fallback on" if a.fallback else ""}', flush=True)
    stats = collections.Counter()
    t0 = time.time()
    lock = threading.Lock()

    def go(r):
        try:
            run_record(r, st, cfg, srv, stats)
        except Exception as e:  # noqa: BLE001 (one record's failure must not stop the pass)
            with lock:
                stats['errors'] += 1
                print(f'ERROR {r["id"]}: {e!r}\n{traceback.format_exc()[-800:]}', flush=True)
        with lock:
            n = stats['records'] + stats['errors']
            if n % a.progress == 0 or n == len(todo):
                el = time.time() - t0
                print(f'[{n}/{len(todo)} records, {el / 60:.1f} min] turns sampled {stats["sampled"]}, matched '
                      f'{stats["matched"]}; fallback {stats["fallback"]} ok {stats["fallback_ok"]}; errors '
                      f'{stats["errors"]}', flush=True)

    with concurrent.futures.ThreadPoolExecutor(conc) as ex:
        list(ex.map(go, todo))
    print(f'done: {dict(stats)} in {(time.time() - t0) / 60:.1f} min')


# ------------------------------------------------------------------ report / export / show

def load_lines(out_dir):
    st = Store(out_dir)
    by = collections.defaultdict(dict)
    for (rid, k, p), d in st.lines.items():
        by[(rid, k)][p] = d
    return by


def fmt_rate(a, b):
    return f'{a / b:.0%}' if b else '-'


def train_turn_counts(sources=SOURCES):
    """{coarse kind: number of train turns} over the full states set."""
    c = collections.Counter()
    for r in load_records(sources):
        for t in r['turns']:
            if t['label'] == 'train':
                c[coarse_kind(t.get('kind'), t['call'])] += 1
    return c


def report(out_dir, a):
    by = load_lines(out_dir)
    if not by:
        return f'no sampled lines in {out_dir}'
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    tot = collections.Counter()
    tok_by_kind = collections.defaultdict(list)
    sims, edit_turns = [], []
    fb = collections.Counter()
    fb_fail = collections.Counter()
    final = collections.Counter()
    Kmax = 1
    for (rid, k), ps in by.items():
        sl = ps.get('sample')
        if not sl:
            continue
        src, ck = sl['source'], sl['ckind']
        Kmax = max(Kmax, len(sl['samples']))
        oks = [x['ok'] for x in sl['matches']]
        for key in (src, 'ALL'):
            for kk in (ck, 'all'):
                R = rows[(key, kk)]
                R['n'].append(1)
                for j in (1, 2, 4, 8):
                    R[f'@{j}'].append(any(oks[:j]))
                R['@K'].append(any(oks))
                seen = set()
                for s in sl['samples']:
                    R['dup'].append((s['thinking'], json.dumps(s['call'], sort_keys=True)) in seen)
                    seen.add((s['thinking'], json.dumps(s['call'], sort_keys=True)))
                if sl['kept']:
                    R['kept_think'].append(sl['kept']['think_tokens'])
                    R['kept_budget'].append(sl['kept']['at_budget'])
                    R['kept_identical'].append(sl['kept']['call_from'] == 'sample')
                for s in sl['samples']:
                    R['problem'].append(bool(s['problem']) and s['problem'] not in ('no_call', 'length'))
                    R['no_call'].append(s['problem'] == 'no_call')
                    R['length'].append(s['problem'] == 'length')
                    R['visible'].append(bool(s['visible']))
                    R['think'].append(s['think_tokens'])
                    R['at_budget'].append(s['at_budget'])
                R['other_edit'].append(any(x.get('other') is not None for x in sl['matches']))
        for s in sl['samples']:
            tok_by_kind[ck].append(s['tokens'])
            tot['samples'] += 1
            tot['sample_tokens'] += s['tokens']
            if s['problem'] and s['problem'] not in ('no_call', 'length'):
                tot[f'problem: {s["problem"].split(" ")[0]}'] += 1
        add_usage(tot, sl.get('usage') or {})
        tot['requests'] += sl.get('requests') or 1
        tot['seconds'] += sl.get('seconds') or 0
        if ck in ('edit', 'edit_repair'):
            same = [x.get('sim') for x in sl['matches'] if x.get('sim') is not None]
            sims.append(max(same) if same else None)
            edit_turns.append(sl)
        f = ps.get('fallback')
        kept, how = final_of(sl, f)
        final[how or ('masked' if f else 'unmatched')] += 1
        if f:
            fb['attempted'] += 1
            fb['ok'] += f['kept'] is not None
            add_usage(tot, f.get('usage') or {})
            tot['fallback_completion'] += (f.get('usage') or {}).get('completion_tokens') or 0
            for c in f['candidates']:
                fb_fail[c.get('fail', 'ok' if c.get('ok') else 'not tried')] += 1
            if f.get('error'):
                fb_fail['request error'] += 1
    lines = [f'== {out_dir}: {sum(1 for p in by.values() if "sample" in p)} sampled turns, '
             f'{len({rid for rid, _ in by})} trajectories; K up to {Kmax}']
    hdr = (f'{"source":16s} {"kind":12s} {"turns":>5s} {"@1":>5s} {"@2":>5s} {"@4":>5s} {"@8":>5s} {"@K":>5s} '
           f'{"thinkK":>6s} {"bud%":>5s} {"ident":>5s} | {"think":>5s} {"atbud":>5s} {"bad":>5s} {"nocall":>6s} '
           f'{"len":>4s} {"vis":>4s} {"dup":>4s} {"othEd":>5s}')
    lines.append('per turn: match rate when sampling 1/2/4/8/K; kept sample: mean thinking tokens, share at the budget,'
                 ' call identical to the oracle; per sample: mean thinking tokens, at budget, malformed, text only, '
                 'cut by max tokens, visible text, duplicate of an earlier sample of the turn; turns where a sample '
                 'matched another oracle edit (hunk order)')
    lines.append(hdr)
    for (src, ck), R in sorted(rows.items(), key=lambda kv: (kv[0][0] == 'ALL', kv[0][0], kv[0][1] == 'all',
                                                           kv[0][1])):
        n = len(R['n'])
        mean = lambda xs: statistics.mean(xs) if xs else 0  # noqa: E731
        lines.append(f'{src:16s} {ck:12s} {n:5d} ' + ' '.join(f'{fmt_rate(sum(R[f"@{j}"]), n):>5s}'
                                                             for j in (1, 2, 4, 8)) +
                     f' {fmt_rate(sum(R["@K"]), n):>5s}'
                     f' {mean(R["kept_think"]):6.0f} {fmt_rate(sum(R["kept_budget"]), len(R["kept_budget"])):>5s}'
                     f' {fmt_rate(sum(R["kept_identical"]), len(R["kept_identical"])):>5s} |'
                     f' {mean(R["think"]):5.0f} {fmt_rate(sum(R["at_budget"]), len(R["at_budget"])):>5s}'
                     f' {fmt_rate(sum(R["problem"]), len(R["problem"])):>5s}'
                     f' {fmt_rate(sum(R["no_call"]), len(R["no_call"])):>6s}'
                     f' {fmt_rate(sum(R["length"]), len(R["length"])):>4s}'
                     f' {fmt_rate(sum(R["visible"]), len(R["visible"])):>4s}'
                     f' {fmt_rate(sum(R["dup"]), len(R["dup"])):>4s}'
                     f' {fmt_rate(sum(R["other_edit"]), n):>5s}')
    if edit_turns:
        vals = [s for s in sims if s is not None]
        bins = [(0, .2), (.2, .4), (.4, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.0), (1.0, 1.01)]
        lines.append(f'edit turns {len(edit_turns)}: best change-F1 among same-file overlapping samples (None = no '
                     f'such sample: {sum(s is None for s in sims)}): ' +
                     ', '.join(f'[{lo:.1f},{min(hi, 1):.1f}{")" if hi <= 1 else "]"} {sum(lo <= v < hi for v in vals)}'
                               for lo, hi in bins))
        for th in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            hit = sum(any((x.get('sim') or 0) >= th for x in sl['matches']) for sl in edit_turns)
            thit = sum(any((x.get('tsim') or 0) >= th for x in sl['matches']) for sl in edit_turns)
            lines.append(f'   match rate at threshold {th:.1f}: line F1 {fmt_rate(hit, len(edit_turns))}, token F1 '
                         f'{fmt_rate(thit, len(edit_turns))}')
        why = collections.Counter(x['why'] for sl in edit_turns for x in sl['matches'] if not x['ok'])
        lines.append('   edit non-match reasons (samples): ' + ', '.join(f'{k} {v}' for k, v in why.most_common(10)))
    probs = {k: v for k, v in tot.items() if k.startswith('problem:')}
    lines.append(f'malformed samples by type: {probs or "none"}')
    if fb['attempted']:
        lines.append(f'fallback: {fb["attempted"]} turns, ok {fb["ok"]} ({fmt_rate(fb["ok"], fb["attempted"])}); '
                     f'candidates: ' + ', '.join(f'{k} {v}' for k, v in fb_fail.most_common()))
    lines.append('final thinking source per sampled turn: ' + ', '.join(f'{k} {v}' for k, v in final.most_common()))
    ns = max(tot['samples'], 1)
    mean_tok = tot['sample_tokens'] / ns
    lines.append(f'tokens: {tot["samples"]} samples, generated {tot["completion_tokens"]} (server usage; per sample '
                 f'{tot["sample_tokens"] / ns:.0f} by our count), prompt {tot["prompt_tokens"]} (cached '
                 f'{tot["cached_tokens"]}), requests {tot["requests"]}, wall {tot["seconds"] / 3600:.2f} h of '
                 f'request time')
    # estimate for the full set
    counts = train_turn_counts()
    kk = parse_k_kind(a.k_kind)
    gen = 0
    for ck, n in counts.items():
        t = tok_by_kind.get(ck)
        gen += n * kk.get(ck, a.k) * (statistics.mean(t) if t else mean_tok)
    n_turns = sum(counts.values())
    unmatched = sum(1 for p in by.values() if p.get('sample') and not p['sample']['kept'])
    fb_share = unmatched / max(1, sum(1 for p in by.values() if p.get('sample')))
    fb_tok = n_turns * fb_share * (a.fb_n * 150 + a.verify_n * 120)
    recs = load_records(SOURCES)
    prefill = sum(r['meta'].get('est_prompt_tokens') or 0 for r in recs)
    sys_tok = 4000 * len(recs)  # the system prompt + tools are shared by all records (prefix cache)
    lines.append(f'full set: {len(recs)} trajectories, {n_turns} train turns ' +
                 ', '.join(f'{k} {v}' for k, v in counts.most_common()) +
                 f'; K={a.k} {kk or ""}: ~{gen / 1e6:.1f}M tokens generated (measured tokens/sample per kind), '
                 f'fallback ~{fb_tok / 1e6:.1f}M (unmatched share {fb_share:.0%}); prefill '
                 f'~{(prefill - sys_tok) / 1e6:.0f}M uncached tokens (each trajectory once with prefix caching)')
    for tps in a.tok_s:
        h = (gen + fb_tok) / tps / 3600 + (prefill - sys_tok) / a.prefill_tok_s / 3600
        lines.append(f'   at {tps} tok/s aggregate decode + {a.prefill_tok_s} tok/s prefill: ~{h:.1f} GPU hours')
    return '\n'.join(lines)


def cmd_export(a):
    out_dir = a.out or (os.path.join(OUT, 'dev') if a.dev else OUT)
    by = load_lines(out_dir)
    n = collections.Counter()
    for src in a.sources.split(','):
        p = os.path.join(out_dir, f'{src}.records.jsonl')
        with open(p, 'w') as f:
            for r in load_records([src]):
                ks = [k for k, t in enumerate(r['turns']) if t['label'] == 'train']
                if not all('sample' in by.get((r['id'], k), {}) for k in ks if not carried(r['turns'][k])):
                    n['incomplete'] += 1
                    continue
                r = copy.deepcopy(r)
                n['carried'] += sum(carried(r['turns'][k]) for k in ks)
                for k in ks:
                    if carried(r['turns'][k]):
                        continue
                    ps = by[(r['id'], k)]
                    kept, how = final_of(ps.get('sample'), ps.get('fallback'))
                    t = r['turns'][k]
                    apply_kept(t, kept)
                    t['sampling'] = {'from': how, 'think_tokens': kept['think_tokens'] if kept else None,
                                     'mask_thinking': not kept or bool(kept.get('at_budget')),
                                     'rule': kept.get('rule') if kept else None,
                                     'call_from': kept.get('call_from') if kept else None,
                                     'ctx': ps['sample'].get('ctx')}
                    n[how or 'masked'] += 1
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                n['records'] += 1
        print(f'wrote {p}')
    print(dict(n))


def selfcheck(a):
    """The match rules and the grounding check on the real states, no model: every oracle call must match itself
    through its rule (not the identical shortcut), every oracle edit window must be in the printed lines, a one-token
    change of an oracle edit must score below 1, and a templated thought naming the call's target must be grounded."""
    c = collections.Counter()
    bad = collections.defaultdict(list)
    for r in load_records(a.sources.split(',')):
        for k, t in enumerate(r['turns']):
            if t['label'] != 'train':
                continue
            sc, oa = script_of(t['call']), args_of(t['call'])
            known = known_lines(r['turns'][:k])
            ctx = context_text(S.messages_at(r, k))
            c[f'turns {sc}'] += 1
            if sc == 'show.py':
                x = match_show(oa, oa, t, r['gold'])
            elif sc == 'search.py':
                x = match_search(oa, oa)
            elif sc == 'edit.py':
                x = match_edit(oa, oa, known, 0.8)
                toks = TOKEN_RE.findall(oa.get('new', ''))
                words_ = [w for w in toks if re.fullmatch(r'[A-Za-z_]\w*', w)]
                if words_:
                    pert = dict(oa, new=re.sub(r'\b' + re.escape(words_[-1]) + r'\b', words_[-1] + 'X',
                                               oa['new'], count=1))
                    y = match_edit(pert, oa, known, 0.8)
                    c['edit one-token change: sim<1'] += (y.get('sim') or 0) < 1
                    c['edit one-token change: matched at 0.8 (line F1)'] += y['ok']
                    c['edit one-token change: token F1 >= 0.8'] += (y.get('tsim') or 0) >= 0.8
                    c['edit one-token change: evaluated'] += 1
            else:
                x = match(t['call'], t, r['gold'], known)
            c[f'self-match {sc}'] += x['ok']
            if not x['ok'] and len(bad[sc]) < 5:
                bad[sc].append(f'{r["id"]} turn {k}: {x["why"]}')
            if sc in ('show.py', 'edit.py', 'outline.py'):
                f = target_of(oa) or (oa.get('files') or '')
                thought = f'The output above shows {f}. I\'ll ' + (
                    f'read `{oa["symbol"]}` there.' if oa.get('symbol') else
                    f'look at lines {oa.get("start")}-{oa.get("end")}.' if oa.get('start') else 'outline it.')
                ug = ungrounded(thought, ctx, call_allowance(t['call']))
                c[f'templated thought grounded {sc}'] += not ug
                c[f'templated thought grounded without the call allowance {sc}'] += not ungrounded(thought, ctx)
                if ug and len(bad['grounding']) < 8:
                    bad['grounding'].append(f'{r["id"]} turn {k} {sc}: {ug}')
    for k, v in sorted(c.items()):
        print(f'{k}: {v}')
    for k, v in bad.items():
        print(f'-- failures {k}:')
        for x in v:
            print('   ', x)


def short(s, n=160):
    s = (s or '').replace('\n', '\\n')
    return s if len(s) <= n else s[:n - 3] + '...'


def cmd_show(a):
    out_dir = a.out or (os.path.join(OUT, 'dev') if a.dev else OUT)
    by = load_lines(out_dir)
    ids = [a.id] if a.id else sorted({rid for rid, _ in by})[:a.n]
    recs = {r['id']: r for r in load_records(SOURCES, set(ids))}
    for rid in ids:
        r = recs[rid]
        print(f'### {rid}  gold {r["gold"].get("functions")}')
        for k, t in enumerate(r['turns']):
            ps = by.get((rid, k), {})
            print(f'[{k}] {t["label"]:7s} {t.get("kind")}: {S.describe_call(t["call"])[:150]}')
            sl = ps.get('sample')
            if not sl:
                continue
            for i, (s, x) in enumerate(zip(sl['samples'], sl['matches'])):
                mark = '*' if sl['kept'] and sl['kept'].get('sample') == i else ' '
                call = S.describe_call(s['call'])[:110] if s['call'] else '-'
                print(f'   {mark}s{i} {"MATCH" if x["ok"] else "no   "} {x["rule"]}/{x["why"]}'
                      f'{" sim=%s" % x["sim"] if x.get("sim") is not None else ""} think {s["think_tokens"]}'
                      f'{" BUDGET" if s["at_budget"] else ""}{" problem=" + s["problem"] if s["problem"] else ""}'
                      f' | {call}')
                if a.thinking or mark == '*':
                    print(f'        thinking: {short(s["thinking"], a.width)}')
                    if s['visible']:
                        print(f'        visible: {short(s["visible"], a.width)}')
            f = ps.get('fallback')
            if f:
                print(f'   fallback: {f["label"]}')
                for c in f['candidates']:
                    print(f'      cand fail={c.get("fail")} ungrounded={c.get("ungrounded")} think: '
                          f'{short(c.get("thinking"), a.width)}')
                    for v in c.get('verify') or []:
                        what = S.describe_call(v['call'])[:100] if v['call'] else v['problem']
                        print(f'         verify ok={v["ok"]} {v["why"]} {what}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--version', default=None, help='agent version of the states (v19 default | v20)')
    common.add_argument('--records', default=None, help='records file instead of <states>/<source>.jsonl (carried '
                                                         'turns, scripts/rft3_carry.py, are not sampled again)')
    common.add_argument('--sources', default=','.join(SOURCES))
    common.add_argument('--dev', type=int, default=0, help='a fixed random subset of N trajectories (out: .../dev)')
    common.add_argument('--out', default=None, help='output dir (default data/rft3/sampled[/dev])')
    common.add_argument('--k', type=int, default=8, help='samples per train turn')
    common.add_argument('--k-kind', default='', help='per kind K, e.g. locate=4,check=4,submit=2')
    common.add_argument('--fb-n', type=int, default=2, help='fallback: rationalization candidates per turn')
    common.add_argument('--verify-n', type=int, default=1, help='fallback: unhinted continuations that must match')
    s = sub.add_parser('sample', parents=[common])
    s.add_argument('--server', default=DEFAULT_SERVER)
    s.add_argument('--model', default=None, help='model id (default: from /v1/models; a served adapter if any)')
    s.add_argument('--concurrency', type=int, default=24, help='records in flight (each sends n=K per request)')
    s.add_argument('--limit', type=int, default=0, help='first N records per source')
    s.add_argument('--only', default='', help='comma-separated record ids')
    s.add_argument('--temperature', type=float, default=None, help="default: the record's (agent-v19: 0.2)")
    s.add_argument('--edit-threshold', type=float, default=0.8)
    s.add_argument('--require-visible', action='store_true')
    s.add_argument('--token-ids', action='store_true', help='vLLM return_token_ids: raw text, exact counts')
    s.add_argument('--keep-raw', action='store_true', help='store the decoded raw generation per sample')
    s.add_argument('--seed', type=int, default=None,
                   help='per-turn seeds derived from it (default none: seeded sampling is slower in vLLM)')
    s.add_argument('--fallback', action='store_true', help='rationalize unmatched turns (inline + second pass)')
    s.add_argument('--fb-kinds', default='', help='fallback only for these coarse kinds, e.g. edit,edit_repair')
    s.add_argument('--fb-temperature', type=float, default=0.7)
    s.add_argument('--fb-verify-max', type=int, default=2, help='candidates verified at most per turn')
    s.add_argument('--progress', type=int, default=10)
    r = sub.add_parser('report', parents=[common])
    r.add_argument('--tok-s', type=lambda x: [int(v) for v in x.split(',')], default=[1000, 1500, 2000])
    r.add_argument('--prefill-tok-s', type=int, default=2500)
    sub.add_parser('export', parents=[common])
    sub.add_parser('selfcheck', parents=[common])
    sh = sub.add_parser('show', parents=[common])
    sh.add_argument('--id', default=None)
    sh.add_argument('--n', type=int, default=1)
    sh.add_argument('--thinking', action='store_true', help='print every sample thinking')
    sh.add_argument('--width', type=int, default=300)
    a = ap.parse_args()
    configure(a.version, a.records)
    if a.cmd == 'sample':
        cmd_sample(a)
    elif a.cmd == 'selfcheck':
        selfcheck(a)
    elif a.cmd == 'report':
        print(report(a.out or (os.path.join(OUT, 'dev') if a.dev else OUT), a))
    elif a.cmd == 'export':
        cmd_export(a)
    elif a.cmd == 'show':
        cmd_show(a)


if __name__ == '__main__':
    main()
