"""DeepSeek -> Gemma SFT dataset (LOG 2026-10-15 "LoRA plan"; research/notes/23-training-revisited.md): DeepSeek V4.1
Flash's passing trajectories with the stock tools (E085) rewritten as agent-v30 (agents/stock) requests and rendered
exactly as the Kaggle scorer's vLLM renders a Gemma 4 31B chat request, with DeepSeek's reasoning_content as Gemma's
thinking. No GPU, no API.

    .venv/bin/python scripts/ds_sft_build.py                     # defaults below -> data/ds_sft/
    .venv/bin/python scripts/ds_sft_build.py --max-calls 40      # cap on the recorded tool calls (default 28)
    .venv/bin/python scripts/ds_sft_build.py --selftest-only     # only the renderer / parser self-tests
    (from a worktree without data/: --data-root <main checkout>/data --results-root <main checkout>/results)

Inputs:  data/raw/E085_flash_bundle_generous/*.json.gz (scripts/slow_proxy.py --save: {path, status, t0, timing,
         request, response}; request = ADK/LiteLLM's OpenAI request for the public Budget-Fit bundle with prompt budgets
         scaled to 100 calls / 30 min (results/E085_flash_bundle_generous/_agent); response = DeepSeek's reply: message
         content, tool_calls, reasoning_content), results/E085_flash_bundle_generous/r1/c*/task_results.jsonl,
         data/competition/tasks.jsonl, agents/stock (agent-v30: agent.yaml, prompts/system.md).
Renderer: scripts/rft_build.Renderer (vLLM 0.19.1's chat path) with data/rft2/tokenizer_june (Kaggle's June template).
Parser:  vLLM 0.19.1's gemma4 tool-call argument parser, read in place from data/wheelhouse/vllm-0.19.1-*.whl
         (scripts/call_probe.py does the same), with the non-streaming glue of chat_completion/serving.py (reasoning
         parser first, tool parser on the remaining content) re-implemented in Gemma4Parser below.

Steps:
 1. Episodes (scripts/rft2_build.group_run): agent requests grouped by task (problem statement in the first user
    message; after a compaction the request is linked by the tool-call ids of its retained events); a task started
    twice (the fastapi_11194 smoke run of E085's proxy log) keeps its last attempt, which is checked against the
    recorded result (total_llm_calls == requests). ADK compaction-summary requests are not used. Kept: resolved tasks
    whose recorded tool_calls <= --max-calls and that never call a graph tool.
 2. Requests -> v30 requests: the system message becomes v30's instruction + ADK's identity suffix ('\\n\\nYou are an
    agent. Your internal name is "<name>".', checked on E085's own system message); the graph tools are dropped from
    the tools; the harness's user prompt gets v30's budget lines (Time allowance / Tool calls allowance / Max loop
    iterations from agents/stock/agent.yaml) and loses the "## Code Intelligence Tools" section (harness 0.2.11 adds it
    only for agents that declare graph tools; checked against swegemma 0.2.11's own build_agent_prompt when
    .venv-0211 exists); ADK's post-compaction "For context:" summaries (DeepSeek-written) get their budget numbers
    rewritten (30 min / 100 tool calls / 300 turns -> v30's); enable_thinking = true.
 3. Turns: each DeepSeek response is an assistant turn: visible content, tool calls, reasoning_content as thinking.
    Thinking longer than --think-cap - --think-margin Gemma tokens (the agent's thinking_token_budget 4,096; at
    >= cap-8 a thought counts as cut by the budget, rft2's rule) is truncated at a paragraph or sentence boundary
    outside code fences; the truncated thinking replaces the original in every later request (history), as Gemma
    would have produced it.
 4. Sequences (rft2's prefix rule): P_k = request k rendered with the generation prompt, B_k = the turn's history
    block (render of request k + the turn + its tool results, minus P_k). A segment is a run of requests with
    P_k+1 == P_k + B_k (append-only); a compaction (or any other rewrite) starts a new segment and a new sequence,
    where the retained events are context only. One sequence per segment: P_first, then per turn the history block
    (context turns) or the GENERATED form (trained turns) + its tool responses + the history trailer, ending after
    the last trained turn. Generated form (scripts/rft3_build.py, checked on Gemma's sampled token ids, self-test e):
        <|channel>thought\\n THINKING <channel|> [VISIBLE] <|tool_call>call:...<tool_call|>[...] <|tool_response>
    vs the template's history form <|channel>thought\\n THINKING \\n<channel|> <|tool_call>... <|tool_response>
    RESPONSES [VISIBLE <turn|>\\n]. So a trained turn's context differs from its real request by one newline per
    earlier trained turn with thinking and one extra copy of each earlier trained turn's visible text (measured:
    report['deviation']); the first trained turn of each sequence sees its exact request.
 5. Loss: thinking + visible text + calls + the stop token <|tool_response> of each trained turn ('full'); a turn
    without reasoning_content trains only its calls from `call:` on ('call_body', rft3's rule: the model would open its
    thought channel at <|tool_call>; --no-thinking mask drops them instead). Masked (context only): the final text-only
    turn, turns cut by max_tokens, turns with a failed edit_file/write_file call, turns whose texts contain Gemma
    control-token strings, and turns whose rendered form does not parse back into DeepSeek's calls (self-test b).
    Everything else (system, user, tool responses, context turns) is -100.

Self-tests (printed and in build_report.json['selftest']):
 (a) renderer round trip: captured Gemma requests re-rendered with their original system prompt and tools must
     tokenize to exactly usage.prompt_tokens: rft2's captures (E054, v18c), the stock bundle on Gemma (E080, harness
     0.2.11) and the handoff runs (E089: Gemma requests whose history holds DeepSeek's E085 turns with
     reasoning_content). Plus: the v30 rewrite reproduces E085's system message from E085's own prompt, and the
     rewritten user prompt equals swegemma 0.2.11's build_agent_prompt for v30.
 (b) every trained turn, decoded from its tokens in the sequence, parses (vLLM gemma4 reasoning + tool parser) into
     DeepSeek's tool names and JSON arguments, its (truncated) thinking and its visible text.
 (c) every labelled token lies inside a trained assistant turn (labels == input_ids there; each labelled run decodes
     to one turn: starts with <|channel>thought or call:, ends with its only <|tool_response>, no turn/response markers).
 (d) every sequence <= --max-len (32,768) tokens (longer ones are dropped and counted).
 (e) the generated form vs Gemma's sampled token_ids on captured stock-tool turns (E080, --logprobs captures).

Outputs (data/ds_sft/):
    episodes.jsonl      every E085 episode: result, kept / drop reason, segments, per-turn info (no transcripts)
    train_tokens.jsonl  one row per sequence (scripts/lora_train.py --data): id, episode, task, run, kind, segment,
                        input_ids, labels (-100 = no loss), n_tokens, n_trained (= n_trained_tokens), trained_turns,
                        think_trained_turns, max_prompt_tokens, segment_tokenization_equal, split
    build_report.json   the numbers printed at the end

Caveats (also in the report): DeepSeek's thinking is longer than Gemma's (estimated decode time per episode at 35 tok/s
vs the 8-minute budget is reported); E085 compacted at 14,336 DeepSeek tokens, so some contexts are longer in Gemma
tokens than Gemma would ever see; DeepSeek makes parallel calls (several calls per turn) and sometimes writes visible
text with its calls, which Gemma in v30 runs never does; reasoning that mentions E085's budgets is not rewritten.
"""

import argparse
import ast
import collections
import glob
import gzip
import json
import os
import random
import re
import statistics
import sys
import time
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import rft_build as rb  # noqa: E402  (stdlib at import; transformers only in Renderer)
import rft2_build as r2  # noqa: E402

RUN = 'E085_flash_bundle_generous'
GRAPH_TOOLS = ('get_code_neighbors', 'search_similar_code', 'get_code_subgraph')
THINK_OPEN, THINK_END = '<|channel>thought\n', '<channel|>'
CALL_OPEN, CALL_END = '<|tool_call>', '<tool_call|>'
RESP_OPEN, RESP_END = '<|tool_response>', '<tool_response|>'
TURN_END = '<turn|>\n'
FORBIDDEN_IN_TURN = ('<|turn>', '<turn|>', RESP_END, '<|tool>', '<tool|>', '<|think|>')
EDIT_TOOLS = ('edit_file', 'write_file')
TOK_S = (470, 1000)   # training throughput (tokens/s): rft2 measured 470; lora_train's speed flags ~1,000
GEMMA_DECODE_TOK_S = 35   # Kaggle's decode speed for Gemma 4 31B (LOG: paced runs at 35 tok/s)
SELFTEST_RUNS = 'E054_v18c_b256_kaggleslow:40,E080_bundle_unseen:60,E089_handoff_stock:0'
RAW_CHECK_RUN = 'E080_bundle_unseen'


# ------------------------------------------------------------------ pure-Python core (unit-tested, no tokenizer)

def read_agent(agent_dir):
    """agents/<x>/agent.yaml (regex: its `!include` tags are not plain YAML) + eval_config.yaml (the four budget keys
    Kaggle's scorer reads) -> name, description, tools, budget, instruction text."""
    d = Path(agent_dir)
    y = (d / 'agent.yaml').read_text()

    def field(k):
        m = re.search(rf'^{k}:\s*(.+)$', y, re.M)
        return m.group(1).strip() if m else None
    inc = re.search(r'^instruction:\s*!include\s+(\S+)', y, re.M)
    tools = re.search(r'^tools:\s*\n((?:[ \t]+-[ \t]*\S+[ \t]*\n?)+)', y, re.M)
    ev = (d / 'eval_config.yaml').read_text() if (d / 'eval_config.yaml').exists() else ''

    def num(k):
        m = re.search(rf'^\s+{k}:\s*([\d.]+)\s*$', ev, re.M)
        return m.group(1) if m else None
    return {'name': field('name'), 'description': field('description'),
            'tools': re.findall(r'-[ \t]*(\S+)', tools.group(1)) if tools else [],
            'instruction': (d / inc.group(1)).read_text() if inc else '',
            'max_tool_calls': num('max_tool_calls'), 'max_time_minutes': num('max_time_minutes'),
            'max_turns': num('max_turns'), 'timeout_seconds': num('timeout_seconds')}


def adk_system(instruction, name, description=None):
    """The system message ADK builds: the instruction + '\\n\\n' + its identity line (checked on E085 and v27)."""
    s = f'{instruction}\n\nYou are an agent. Your internal name is "{name}".'
    if description:
        s += f' The description about you is "{description}".'
    return s


BUDGET_LINES = (('time', r'^- Time allowance: (\S+) minutes$', '- Time allowance: {} minutes'),
                ('calls', r'^- Tool calls allowance: (\S+) calls$', '- Tool calls allowance: {} calls'),
                ('turns', r'^- Max loop iterations: (\S+) turns$', '- Max loop iterations: {} turns'))


def prompt_budget(text):
    """{'time', 'calls', 'turns'} as written in a harness user prompt (strings), missing keys absent."""
    out = {}
    for k, pat, _ in BUDGET_LINES:
        m = re.search(pat, text, re.M)
        if m:
            out[k] = m.group(1)
    return out


def rewrite_user_prompt(text, budget):
    """The harness's first user message for another agent: budget lines set to `budget` ({'time': '8.0', 'calls':
    '28', 'turns': '80'}) and the '## Code Intelligence Tools' section removed (harness 0.2.11 adds it only when the
    agent declares a graph tool; build_agent_prompt joins its parts with '\\n'). Returns (text, changes)."""
    n = 0
    for k, pat, rep in BUDGET_LINES:
        if k in budget:
            text, c = re.subn(pat, rep.format(budget[k]).replace('\\', r'\\'), text, count=1, flags=re.M)
            n += c
    i = text.find('\n## Code Intelligence Tools\n')
    if i >= 0:
        j = text.find('\n## Workspace Layout\n', i + 1)
        text = text[:i] + (text[j:] if j >= 0 else '')
        n += 1
    return text, n


def rewrite_budget_mentions(text, old, new):
    """Budget numbers in free text (DeepSeek's compaction summaries: '30 min, 100 tool calls, 300 loop iterations',
    '30-minute time budget', 'budgets are 30 min / 100 tool calls / 300 iterations') -> the new budget. old/new:
    {'time', 'calls', 'turns'} as numbers in strings ('30.0', '100', '300'). Returns (text, replacements)."""
    def intpart(x):
        return str(x).split('.')[0]
    n = 0
    ot, nt = re.escape(intpart(old['time'])), intpart(new['time'])
    text, c = re.subn(rf'\b{ot}(\.0)?(\s?-?\s?)(minutes?|mins?)\b', lambda m: nt + (m.group(1) or '') + m.group(2) +
                      m.group(3), text)
    n += c
    oc = re.escape(intpart(old['calls']))
    text, c = re.subn(rf'\b{oc}(\s+|-)(tool[\s-]calls?|calls?)\b', lambda m: intpart(new['calls']) + m.group(1) +
                      m.group(2), text)
    n += c
    ou = re.escape(intpart(old['turns']))
    text, c = re.subn(rf'\b{ou}(\s+|-)(loop iterations?|iterations?|turns|loop turns|loops?)\b',
                      lambda m: intpart(new['turns']) + m.group(1) + m.group(2), text)
    return text, n + c


def boundaries(text):
    """Cut positions (prefix ends) at paragraph breaks and sentence ends, with whether each lies inside a ``` code
    fence. A cut at p keeps text[:p]."""
    cuts = {m.start() for m in re.finditer(r'\n\s*\n', text)}
    cuts |= {m.end() for m in re.finditer(r'[.!?:;](?=\s)', text)}
    cuts |= {m.start() for m in re.finditer(r'\n', text)}
    out = []
    for p in sorted(cuts):
        if 0 < p < len(text):
            out.append((p, text[:p].count('```') % 2 == 1))
    return out


def truncate_thinking(text, ntok, limit):
    """(text, how): unchanged (how None) when ntok(text) <= limit, else the longest prefix that ends at a paragraph /
    sentence / line boundary outside a code fence (how 'boundary'), inside a fence if nothing else fits
    ('in_fence'), else at a word boundary ('hard'); trailing whitespace stripped; always ntok <= limit."""
    if ntok(text) <= limit:
        return text, None
    for how, ok in (('boundary', lambda f: not f), ('in_fence', lambda f: True)):
        cands = [p for p, f in boundaries(text) if ok(f)]
        lo, hi, best = 0, len(cands) - 1, None
        while lo <= hi:  # token count grows with the prefix: binary search for the last fitting cut
            mid = (lo + hi) // 2
            if ntok(text[:cands[mid]].rstrip()) <= limit:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        if best is not None:
            cut = text[:cands[best]].rstrip()
            if cut:
                return cut, how
    words = [m.start() for m in re.finditer(r'\s', text)]
    lo, hi, best = 0, len(words) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if ntok(text[:words[mid]].rstrip()) <= limit:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return (text[:words[best]].rstrip() if best is not None else ''), 'hard'


def strip_thinking(text):
    """The template's strip_thinking macro: drop <|channel>...<channel|> parts, trim (as rft3_build)."""
    out = ''
    for part in (text or '').split(THINK_END):
        out += part.split('<|channel>')[0] if '<|channel>' in part else part
    return out.strip()


class BlockError(ValueError):
    pass


def split_block(block, thinking, visible):
    """A turn's history block (P_k+1[len(P_k):]) -> {think, calls, resp, trailer} (rft3_build.split_block for one or
    more calls): the template's thought block (with its newline before <channel|>), the calls
    <|tool_call>...<tool_call|> (concatenated), the responses after the first <|tool_response> through the last
    <tool_response|>, and the visible-text trailer (VISIBLE <turn|>\\n or ''). Raises BlockError otherwise."""
    think = f'{THINK_OPEN}{thinking}\n{THINK_END}' if thinking else ''
    if not block.startswith(think):
        raise BlockError('thought block')
    rest = block[len(think):]
    if not rest.startswith(CALL_OPEN):
        raise BlockError('no call')
    j = rest.find(CALL_END + RESP_OPEN)
    if j < 0:
        raise BlockError('calls not followed by a response')
    calls = rest[:j + len(CALL_END)]
    rest = rest[j + len(CALL_END) + len(RESP_OPEN):]
    v = strip_thinking(visible)
    trailer = v + TURN_END if v else ''
    if not rest.endswith(trailer) or not rest[:len(rest) - len(trailer)].endswith(RESP_END):
        raise BlockError('response / trailer')
    resp = rest[:len(rest) - len(trailer)]
    if think + calls + RESP_OPEN + resp + trailer != block:
        raise BlockError('reassembly')
    return {'think': think, 'calls': calls, 'resp': resp, 'trailer': trailer}


def generated_turn(calls, thinking, visible):
    """(thought, text, call part) of a turn as Gemma generates it: no newline before <channel|>, the visible text
    before the calls (vLLM's tool parser: content = text before the first <|tool_call>), the call part ending with the
    stop token <|tool_response>."""
    th = f'{THINK_OPEN}{thinking}{THINK_END}' if thinking else ''
    return th, strip_thinking(visible), calls + RESP_OPEN


def label_offset(mode, th, v):
    """Start of the labelled part inside a generated turn: 'full' from the thought channel, 'call_body' from `call:`
    after the first <|tool_call> (rft3_build.train_span)."""
    return 0 if mode == 'full' else len(th) + len(v) + len(CALL_OPEN)


def assemble(prefix, pieces):
    """prefix = P of the segment's first request; pieces = [(k, history_block, generated|None, split|None)] in order,
    generated = (th, v, cl) for trained turns. -> (text, {k: (turn start, label start, generated end)}). The
    sequence ends after the last trained turn's generated part."""
    last = max((i for i, p in enumerate(pieces) if p[2] is not None), default=-1)
    out, pos, spans = [prefix], len(prefix), {}
    for i, (k, block, gen, sp) in enumerate(pieces[:last + 1]):
        if gen is None:
            out.append(block)
            pos += len(block)
            continue
        th, v, cl = gen
        g = th + v + cl
        spans[k] = (pos, pos + len(g))
        piece = g if i == last else g + sp['resp'] + sp['trailer']
        out.append(piece)
        pos += len(piece)
    return ''.join(out), spans


def call_args_obj(call):
    a = call['function'].get('arguments')
    if isinstance(a, dict):
        return a
    return json.loads(a or '{}')


def runs_of(labels):
    """[(start, end)] maximal runs of labelled (!= -100) positions."""
    out, s = [], None
    for i, x in enumerate(labels):
        if x != -100 and s is None:
            s = i
        elif x == -100 and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(labels)))
    return out


# ------------------------------------------------------------------ vLLM 0.19.1 gemma4 parsers

class Gemma4Parser:
    """vLLM 0.19.1's non-streaming chat path for Gemma 4 output text: reasoning parser (gemma4: BaseThinkingReasoning
    extract_reasoning = partition on <|channel> then <channel|>, strip the 'thought\\n' label) and then the gemma4 tool
    parser on the content (regex over <|tool_call>call:NAME{ARGS}<tool_call|>, ARGS by the wheel's own
    _parse_gemma4_args, content = the text before the first <|tool_call>, stripped, None if empty)."""

    def __init__(self, wheel):
        src = zipfile.ZipFile(wheel).read('vllm/tool_parsers/gemma4_tool_parser.py').decode()
        delim = re.search(r'^STRING_DELIM\s*=\s*(.+)$', src, re.M).group(1)
        helpers = src[src.index('def _parse_gemma4_value'):src.index('# Parser\n')]
        ns = {}
        exec('import json\nSTRING_DELIM = ' + delim + '\n' + helpers, ns)  # noqa: S102 (the scorer's own code)
        self.parse_args = ns['_parse_gemma4_args']
        pat = re.search(r'self\.tool_call_regex = re\.compile\(\s*r"(.+?)",\s*re\.DOTALL', src, re.S).group(1)
        self.call_re = re.compile(pat, re.S)
        rsrc = zipfile.ZipFile(wheel).read('vllm/reasoning/gemma4_reasoning_parser.py').decode()
        self.thought_prefix = ast.literal_eval(re.search(r'^_THOUGHT_PREFIX = (.+)$', rsrc, re.M).group(1))

    def __call__(self, text):
        if text.endswith(RESP_OPEN):  # the stop token after a call (not part of vLLM's output text)
            text = text[:-len(RESP_OPEN)]
        if '<|channel>' not in text and THINK_END not in text:
            reasoning, content = None, text
        else:
            a, sep, b = text.partition('<|channel>')
            rest = b if sep else a
            if THINK_END not in rest:
                reasoning, content = rest, None
            else:
                reasoning, _, content = rest.partition(THINK_END)
                content = content or None
            if reasoning is not None and reasoning.startswith(self.thought_prefix):
                reasoning = reasoning[len(self.thought_prefix):]
        calls = []
        if content and CALL_OPEN in content:
            calls = [(name, self.parse_args(args)) for name, args in self.call_re.findall(content)]
            if calls:
                end = content.find(CALL_OPEN)
                content = (content[:end].strip() if end > 0 else None) or None
        return {'reasoning': reasoning, 'content': content, 'calls': calls}


def parse_matches(parsed, thinking, visible, calls):
    """'' if the parsed turn equals DeepSeek's (thinking or None, visible text or None, [(name, args dict)]), else
    the first difference."""
    want_v = strip_thinking(visible) or None
    if (parsed['reasoning'] or None) != (thinking or None):
        return 'reasoning'
    if parsed['content'] != want_v:
        return 'content'
    if len(parsed['calls']) != len(calls):
        return 'call count'
    for (n1, a1), (n2, a2) in zip(parsed['calls'], calls):
        if n1 != n2:
            return 'tool name'
        if a1 != a2:
            return 'arguments'
    return ''


# ------------------------------------------------------------------ loading

def find_wheel(data_root):
    w = sorted(glob.glob(str(Path(data_root) / 'wheelhouse' / 'vllm-0.19.1-*.whl')))
    if not w:
        raise SystemExit(f'no vllm 0.19.1 wheel in {data_root}/wheelhouse (needed for the gemma4 parser)')
    return w[0]


def load_build_agent_prompt(venv_glob):
    """swegemma 0.2.11's build_agent_prompt (harness/agent_runner.py of .venv-0211), compiled alone; None if absent."""
    hits = sorted(glob.glob(venv_glob))
    if not hits:
        return None
    src = Path(hits[0]).read_text()
    tree = ast.parse(src)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_agent_prompt'), None)
    if fn is None:
        return None
    code = 'from __future__ import annotations\n' + ast.get_source_segment(src, fn)
    ns = {'Path': Path}
    exec(compile(code, hits[0], 'exec'), ns)  # noqa: S102 (the harness's own code)
    return ns['build_agent_prompt']


def iter_captures(run_dir, seed=0):
    """Agent requests (tools, status 200, usage) of a capture dir, lazily, in a seeded order."""
    files = sorted(Path(run_dir).glob('*.json.gz'))
    random.Random(seed).shuffle(files)
    for f in files:
        try:
            d = json.load(gzip.open(f, 'rt'))
        except (OSError, ValueError):
            continue
        rq = d.get('request') or {}
        resp = d.get('response') or {}
        if d.get('status') != 200 or not rq.get('tools') or not resp.get('choices') or not resp.get('usage'):
            continue
        d['file'] = f.name
        yield d


def sample_captures(run_dir, n, seed=0):
    """Up to n agent requests from a capture dir (0 = all)."""
    out = []
    for d in iter_captures(run_dir, seed):
        out.append(d)
        if n and len(out) >= n:
            break
    return out


# ------------------------------------------------------------------ rendering helpers

class Ctx:
    def __init__(self, R, a, v30, old_budget, new_budget):
        self.R, self.a, self.v30 = R, a, v30
        self.old_budget, self.new_budget = old_budget, new_budget
        self.system = adk_system(v30['instruction'], v30['name'], v30['description'])
        self.summaries = {}
        self._ntok = {}
        added = R.tok.get_added_vocab() if hasattr(R.tok, 'get_added_vocab') else {}
        self.control = sorted({t for t in added if ('<|' in t or '|>' in t) and len(t) > 3}, key=len, reverse=True)

    def ntok(self, text):
        if not text:
            return 0
        if text not in self._ntok:
            self._ntok[text] = len(self.R.ids(text))
        return self._ntok[text]

    def has_control(self, *texts):
        return any(c in t for t in texts if t for c in self.control)


def to_v30_tools(tools, v30_tools):
    keep = [t for t in tools if t['function']['name'] not in GRAPH_TOOLS]
    names = [t['function']['name'] for t in keep]
    if names != list(v30_tools):
        raise SystemExit(f'tools after dropping the graph tools {names} != agents/stock tools {v30_tools}')
    return keep


def transform_messages(msgs, ctx, override, stats):
    """A captured E085 request's messages as v30 would have sent them (module doc, step 2)."""
    out = []
    for i, m in enumerate(msgs):
        m = dict(m)
        if m['role'] == 'system' and i == 0:
            m['content'] = ctx.system
        elif m['role'] == 'user':
            c = m.get('content')
            if isinstance(c, str):
                m['content'], n = rewrite_user_prompt(c, ctx.new_budget)
                stats['user_prompt_rewrites'] += n > 0
            elif isinstance(c, list):
                parts, n = [], 0
                for p in c:
                    if isinstance(p, dict) and p.get('type') == 'text' and isinstance(p.get('text'), str):
                        t, k = rewrite_budget_mentions(p['text'], ctx.old_budget, ctx.new_budget)
                        n += k
                        p = dict(p, text=t)
                    parts.append(p)
                m['content'] = parts
                ctx.summaries[json.dumps(c)] = n  # distinct summaries (each later request repeats its summary)
        elif m['role'] == 'assistant':
            ids = [c.get('id') for c in m.get('tool_calls') or []]
            hit = next((override[x] for x in ids if x in override), None)
            if hit is not None:
                m['reasoning'] = m['reasoning_content'] = hit
        out.append(m)
    return out


# ------------------------------------------------------------------ episode -> turns -> sequences

RECALL = re.compile(r"\b(I recall|recall that|recall the|from memory|I remember|upstream fix|actual (upstream )?(fix|PR|commit)|"
                    r"the real fix|original PR|the merged PR|merged fix|in the upstream|(fastapi|rich|requests|httpx)[/ ]"
                    r"?(PR|pull|#) ?#?\d+|PR #?\d+|the fix (was|is) (in|from) (PR|commit|version))", re.I)


def turn_mode(t, a):
    """(mode, mask reason) of a turn: mode 'full' / 'call_body' when trained (mask None), else (None, reason)."""
    if t['text_only']:
        return None, 'text_only'
    if t['finish'] == 'length':
        return None, 'length'
    if t['graph']:
        return None, 'graph_tool'
    if t['edit_failed']:
        return None, 'edit_failed'
    if t['control']:
        return None, 'control_tokens'
    if a.mask_recall and RECALL.search(t['thinking'] or ''):  # the teacher recalling the public fix: not for private repos
        return None, 'recall'
    if not t['thinking']:
        return (None, 'no_thinking') if a.no_thinking == 'mask' else ('call_body', None)
    return 'full', None


def build_episode(ep, ctx, parser, stats):
    """One kept episode -> (turn info list, rows, per-turn check records)."""
    R, a = ctx.R, ctx.a
    raw = ep['_raw']
    resp = [rb.response_message(d['response']) for d in raw]
    finish = [d['response']['choices'][0].get('finish_reason') for d in raw]
    results = {}
    for d in raw:
        for m in d['request']['messages']:
            if m.get('role') == 'tool' and m.get('tool_call_id'):
                results.setdefault(m['tool_call_id'], m)
    # ---- turns: thinking truncation (override by call id, applied to every later request), masks
    override, turns = {}, []
    limit = a.think_cap - a.think_margin
    for k, asst in enumerate(resp):
        calls = asst.get('tool_calls') or []
        ids = [c['id'] for c in calls]
        think0 = asst.get('reasoning') or ''
        think, how = truncate_thinking(think0, ctx.ntok, limit) if think0 else ('', None)
        if how:
            for i in ids:
                override[i] = think
            asst = dict(asst, reasoning=think, reasoning_content=think)
            if not think:
                asst.pop('reasoning')
                asst.pop('reasoning_content')
            resp[k] = asst
        res = []
        for c in calls:
            r = results.get(c['id'])
            try:
                j = json.loads(r['content']) if r and isinstance(r.get('content'), str) else None
            except ValueError:
                j = None
            res.append(j if isinstance(j, dict) else None)
        try:
            args = [call_args_obj(c) for c in calls]
        except ValueError:
            args = None
        names = [c['function']['name'] for c in calls]
        t = {'k': k, 'ids': ids, 'names': names, 'n_calls': len(calls), 'text_only': not calls,
             'finish': finish[k], 'graph': any(n in GRAPH_TOOLS for n in names),
             'edit_failed': any(n in EDIT_TOOLS and (r is None or r.get('status') != 'ok') for n, r in zip(names, res)),
             'thinking': think, 'think_tokens_raw': ctx.ntok(think0), 'think_tokens': ctx.ntok(think),
             'truncated': how, 'visible': asst.get('content') or '', 'args': args,
             'control': ctx.has_control(think, asst.get('content') or '',
                                        *[c['function'].get('arguments') or '' for c in calls]),
             'has_results': all(i in results for i in ids)}
        if args is None:
            t['control'] = True  # unparsable JSON arguments: cannot be rendered
        t['mode'], t['mask'] = turn_mode(t, a)
        turns.append(t)
    # ---- v30 requests, P_k, history blocks B_k, segments
    tools = to_v30_tools(raw[0]['request']['tools'], ctx.v30['tools'])
    msgs = [transform_messages(d['request']['messages'], ctx, override, stats) for d in raw]
    P = [R.text(m, tools, True, True) for m in msgs]
    B = []
    for k, t in enumerate(turns):
        asst = resp[k]
        if t['text_only']:
            F = R.text(msgs[k] + [asst], tools, False, True)
            B.append(F[len(P[k]):] if F.startswith(P[k]) else None)
            continue
        tr = [results.get(i) or {'role': 'tool', 'tool_call_id': i, 'content': '{"status": "ok"}'} for i in t['ids']]
        F = R.text(msgs[k] + [asst] + tr, tools, True, True)
        B.append(F[len(P[k]):] if F.startswith(P[k]) else None)
    seg = [0]
    breaks = collections.Counter()
    for k in range(1, len(turns)):
        prev, cur = msgs[k - 1], msgs[k]
        if B[k - 1] is not None and P[k] == P[k - 1] + B[k - 1]:
            seg.append(seg[-1])
            continue
        seg.append(seg[-1] + 1)
        breaks['compaction' if cur[:len(prev)] != prev else 'render_mismatch'] += 1
    # ---- generated forms + the parse check (b, pre-tokenization: masks turns that would not parse back)
    gens, splits = {}, {}
    for k, t in enumerate(turns):
        t['segment'] = seg[k]
        t['prompt_tokens'] = None
        t['gen_tokens'] = ctx.ntok(t['thinking']) + ctx.ntok(t['visible'])  # refined below when the block splits
        if t['text_only']:
            continue
        try:
            sp = split_block(B[k] or '', t['thinking'], t['visible'])
        except BlockError as e:
            if t['mode'] is not None:
                t['mode'], t['mask'] = None, f'render: {e}'
            continue
        th, v, cl = generated_turn(sp['calls'], t['thinking'], t['visible'])
        g = th + v + cl
        g_ids = R.ids(g)
        t['gen_tokens'] = len(g_ids)
        if t['mode'] is None:
            continue
        parsed = parser(R.tok.decode(g_ids, skip_special_tokens=False))
        want = [(n, x) for n, x in zip(t['names'], t['args'])]
        diff = parse_matches(parsed, t['thinking'] if t['mode'] == 'full' else '', t['visible'], want)
        if diff:
            t['mode'], t['mask'] = None, f'parse: {diff}'
            continue
        gens[k], splits[k] = (th, v, cl), sp
    # ---- one sequence per segment
    rows, checks = [], []
    for s in sorted(set(seg)):
        ks = [k for k in range(len(turns)) if seg[k] == s]
        if not any(k in gens for k in ks):
            continue
        pieces = [(k, B[k], gens.get(k), splits.get(k)) for k in ks]
        text, spans = assemble(P[ks[0]], pieces)
        cuts = set()
        lab = {}
        for k, (g0, g1) in spans.items():
            th, v, cl = gens[k]
            l0 = g0 + label_offset(turns[k]['mode'], th, v)
            lab[k] = (l0, g1)
            cuts |= {g0, l0, g1}
        ids, at, same = tokenize_cuts(R, text, cuts)
        labels = [-100] * len(ids)
        for k, (l0, g1) in lab.items():
            for i in range(at[l0], at[g1]):
                labels[i] = ids[i]
        n_tr = sum(x != -100 for x in labels)
        prompt_toks = []
        for k in ks:
            if k in gens:
                turns[k]['prompt_tokens'] = ctx.ntok(P[k])
                prompt_toks.append(turns[k]['prompt_tokens'])
        row = {'id': f'{ep["id"]}_s{s}', 'episode': ep['id'], 'task': ep['task'], 'run': ep['run'],
               'kind': 'ds_episode', 'segment': s, 'split': 'train', 'input_ids': ids, 'labels': labels,
               'n_tokens': len(ids), 'n_trained': n_tr, 'n_trained_tokens': n_tr, 'trained_turns': len(spans),
               'think_trained_turns': sum(turns[k]['mode'] == 'full' for k in spans),
               'max_prompt_tokens': max(prompt_toks), 'segment_tokenization_equal': same}
        rows.append(row)
        # context deviation (module doc, step 4): the first trained turn of the sequence sees its exact request
        first = min(spans)
        for k in sorted(spans):
            ctx_ids = R.ids(text[:spans[k][0]])
            pk = R.ids(P[k])
            checks.append({'row': row['id'], 'k': k, 'first': k == first, 'exact': ctx_ids == pk,
                           'len_diff': len(ctx_ids) - len(pk), 'tok_span': (at[spans[k][0]], at[spans[k][1]]),
                           'lab_span': (at[lab[k][0]], at[lab[k][1]]), 'mode': turns[k]['mode'],
                           'thinking': turns[k]['thinking'] if turns[k]['mode'] == 'full' else '',
                           'visible': turns[k]['visible'], 'calls': list(zip(turns[k]['names'], turns[k]['args']))})
    ep['segments'] = len(set(seg))
    ep['segment_breaks'] = dict(breaks)
    return turns, rows, checks


def tokenize_cuts(R, text, cuts):
    """Token ids of text tokenized piecewise at the char positions `cuts` (special-token boundaries), the token index
    of each cut, and whether the piecewise ids equal the whole-text ids."""
    ids, at, pos = [], {}, 0
    for c in sorted(set(cuts) | {len(text)}):
        if c > pos:
            ids += R.ids(text[pos:c])
        at[c] = len(ids)
        pos = c
    at.setdefault(0, 0)
    return ids, at, ids == R.ids(text)


# ------------------------------------------------------------------ self-tests on the built rows

def check_rows(R, parser, rows, checks):
    """(b) parse each trained turn from its tokens in the sequence, (c) labels only inside trained turns."""
    by_id = {r['id']: r for r in rows}
    b = collections.Counter()
    for c in checks:
        r = by_id[c['row']]
        t0, t1 = c['tok_span']
        text = R.tok.decode(r['input_ids'][t0:t1], skip_special_tokens=False)
        diff = parse_matches(parser(text), c['thinking'], c['visible'], c['calls'])
        b['turns'] += 1
        b['parse_equal'] += not diff
        if diff:
            b[f'parse_diff: {diff}'] += 1
    cc = collections.Counter()
    for r in rows:
        ids, labels = r['input_ids'], r['labels']
        mine = sorted(c['lab_span'] for c in checks if c['row'] == r['id'])
        allowed = set()
        for s, e in mine:
            allowed.update(range(s, e))
        lab = [i for i, x in enumerate(labels) if x != -100]
        cc['sequences'] += 1
        cc['labelled_tokens'] += len(lab)
        cc['labelled_outside_turns'] += sum(i not in allowed for i in lab)
        cc['label_ne_input'] += sum(labels[i] != ids[i] for i in lab)
        runs = runs_of(labels)
        cc['runs'] += len(runs)
        cc['runs_ne_trained_turns'] += len(runs) != r['trained_turns']
        for s, e in runs:
            t = R.tok.decode(ids[s:e], skip_special_tokens=False)
            ok = (t.startswith(THINK_OPEN) or t.startswith('call:')) and t.endswith(RESP_OPEN) and \
                t.count(RESP_OPEN) == 1 and not any(x in t for x in FORBIDDEN_IN_TURN) and \
                (t.count('<|channel>') <= 1) and 'response:' not in t.split(CALL_END)[-1]
            cc['runs_well_formed'] += ok
        cc['too_long'] += r['n_tokens'] > 32768
    return dict(b), dict(cc)


def raw_check(R, run_dir, limit):
    """(e) The generated form (from the template's own history block) vs Gemma's sampled token_ids, as rft3_build's
    raw_check, for one or more calls. 'clean' = thinking present and below the budget, no visible text, argument keys
    sorted in the raw output (the template sorts them), a single thought channel."""
    st = collections.Counter()
    ex = []
    for d in iter_captures(run_dir):
        if st['checked'] >= limit:
            break
        ch = d['response']['choices'][0]
        if not ch.get('token_ids'):
            continue
        asst = rb.response_message(d['response'])
        calls = asst.get('tool_calls') or []
        if not calls:
            continue
        req = d['request']
        raw = R.tok.decode(ch['token_ids'], skip_special_tokens=False)
        th, vis = asst.get('reasoning') or '', asst.get('content') or ''
        tr = [{'role': 'tool', 'tool_call_id': c['id'], 'content': '{"stdout": "x"}'} for c in calls]
        P0 = R.text(req['messages'], req['tools'], True, True)
        P1 = R.text(req['messages'] + [asst] + tr, req['tools'], True, True)
        try:
            sp = split_block(P1[len(P0):], th, vis)
        except BlockError as e:
            st[f'split error: {e}'] += 1
            continue
        g = ''.join(generated_turn(sp['calls'], th, vis))
        same = R.ids(g) == ch['token_ids']
        budget = req.get('thinking_token_budget') or 0
        at_budget = bool(budget) and len(R.ids(th)) >= budget - 8
        raw_calls = re.findall(r'<\|tool_call>.*?<tool_call\|>', raw, re.S)
        clean = bool(th) and not at_budget and not vis and raw.count('<|channel>') <= 1 and \
            ''.join(raw_calls) == sp['calls']
        st['checked'] += 1
        st['equal'] += same
        st['clean'] += clean
        st['clean_equal'] += clean and same
        st['clean_text_equal'] += clean and g == raw  # (the model sometimes samples a non-canonical tokenization)
        st['multi_call'] += len(calls) > 1
        if clean and not same and len(ex) < 5:
            ex.append(d['file'])
    out = dict(st)
    out['examples_clean_not_equal'] = ex
    return out


# ------------------------------------------------------------------ main

def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def dist(xs):
    return {'n': len(xs), 'mean': round(statistics.mean(xs), 1) if xs else 0, 'median': pct(xs, .5),
            'p90': pct(xs, .9), 'max': max(xs) if xs else 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default=RUN)
    ap.add_argument('--data-root', default=str(ROOT / 'data'))
    ap.add_argument('--results-root', default=str(ROOT / 'results'))
    ap.add_argument('--agent', default=str(ROOT / 'agents' / 'stock'), help='the target agent (v30)')
    ap.add_argument('--out', default=None, help='default <data-root>/ds_sft')
    ap.add_argument('--max-calls', type=int, default=28, help="keep passing episodes with recorded tool_calls <= this")
    ap.add_argument('--think-cap', type=int, default=4096, help="the agent's thinking_token_budget")
    ap.add_argument('--think-margin', type=int, default=8,
                    help='thinking is truncated to <= cap - margin tokens (>= cap-8 = cut by the budget, rft2)')
    ap.add_argument('--no-thinking', choices=['call_body', 'mask'], default='call_body',
                    help='turns without reasoning_content: train the call from `call:` on (rft3) or mask them')
    ap.add_argument('--time-minutes', default=None,
                    help="v30's time budget as the harness prints it (default: agent.yaml's max_time_minutes as a "
                         "float, '8.0', as local swegemma runs print it)")
    ap.add_argument('--max-len', type=int, default=32768)
    ap.add_argument('--no-mask-recall', dest='mask_recall', action='store_false',
                    help='train turns whose thinking recalls the upstream fix (default: masked, context only; 2026-10-15: '
                         '70 of 373 thinking turns in 26 of 41 episodes)')
    ap.add_argument('--selftest-runs', default=SELFTEST_RUNS, help='run:n,... (n = sampled requests, 0 = all)')
    ap.add_argument('--raw-check-run', default=RAW_CHECK_RUN)
    ap.add_argument('--raw-check-limit', type=int, default=400)
    ap.add_argument('--harness-src', default=str(ROOT / '.venv-0211/lib/python3*/site-packages/swegemma/harness/'
                                                        'agent_runner.py'))
    ap.add_argument('--selftest-only', action='store_true')
    a = ap.parse_args()
    data = Path(a.data_root)
    out = Path(a.out or data / 'ds_sft')
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    report = {'args': vars(a)}
    st = report['selftest'] = {}

    v30 = read_agent(a.agent)
    tok_dir = r2.ensure_june_tokenizer(data / 'rft' / 'tokenizer', data / 'rft2' / 'tokenizer_june',
                                       data / 'probe' / 'chat_template_kaggle_june.jinja',
                                       data / 'probe' / 'tokenizer_config_kaggle_june.json')
    R = r2.make_renderer(tok_dir, data / 'probe' / 'chat_template_kaggle_june.jinja')
    parser = Gemma4Parser(find_wheel(data))
    print(f'target agent {v30["name"]}: tools {v30["tools"]}, budget {v30["max_tool_calls"]} calls / '
          f'{v30["max_time_minutes"]} min / {v30["max_turns"]} turns; renderer content format {R.fmt!r}')

    # ---- (a) renderer round trip on captured Gemma requests (original system prompt and tools)
    for spec in [x for x in a.selftest_runs.split(',') if x]:
        run, n = (spec.split(':') + ['0'])[:2]
        d = data / 'raw' / run
        if not d.is_dir():
            st[f'a_roundtrip_{run}'] = 'missing'
            continue
        items = [(x, run) for x in sample_captures(d, int(n))]
        r2.selftest(R, items, st, f'a_roundtrip_{run}')
        if run.startswith('E089'):
            st[f'a_roundtrip_{run}']['requests_with_deepseek_reasoning'] = sum(
                any(m.get('reasoning_content') for m in x['request']['messages'] if m['role'] == 'assistant')
                for x, _ in items)
    if a.raw_check_run and (data / 'raw' / a.raw_check_run).is_dir():
        st['e_generated_form_vs_sampled'] = raw_check(R, data / 'raw' / a.raw_check_run, a.raw_check_limit)
        print(f'self-test (e): generated form vs Gemma sampled token_ids ({a.raw_check_run}): '
              f'{ {k: v for k, v in st["e_generated_form_vs_sampled"].items() if k != "examples_clean_not_equal"} }')
    if a.selftest_only:
        json.dump(report, open(out / 'build_report.json', 'w'), indent=1)
        return

    # ---- step 1: episodes
    tasks, task_rows = {}, {}
    for line in open(data / 'competition' / 'tasks.jsonl'):
        t = json.loads(line)
        tasks[t['instance_id']] = t['problem_statement'].strip()
        task_rows[t['instance_id']] = t
    stats = collections.Counter()
    recs = r2.load_run(data / 'raw' / a.run)
    groups, _summ = r2.group_run(recs, r2.TaskFinder(tasks), stats)
    results = r2.load_results(Path(a.results_root) / a.run)
    print(f'{a.run}: {len(recs)} captures: agent {stats["requests_agent"]}, compaction summaries '
          f'{stats["requests_summary"]}, other {stats["requests_other"]}; {len(groups)} episodes (restarted '
          f'{stats["episodes_restarted"]}, linked by call ids {stats["requests_linked_by_call_ids"]}, unassigned '
          f'{stats["requests_unassigned"]}); results for {len(results)} tasks')
    first_user = next(m['content'] for d in recs if r2.kind(d) == 'agent'
                      for m in d['request']['messages'] if m['role'] == 'user' and isinstance(m['content'], str))
    old_budget = prompt_budget(first_user)
    new_budget = {'time': a.time_minutes or str(float(v30['max_time_minutes'])), 'calls': v30['max_tool_calls'],
                  'turns': v30['max_turns']}
    ctx = Ctx(R, a, v30, old_budget, new_budget)
    print(f'budget in the prompts: {old_budget} -> {new_budget}')

    # (a) the system-message rule on E085's own prompt; the rewritten user prompt vs the harness's own builder
    e085_agent = read_agent(Path(a.results_root) / a.run / '_agent')
    sys0 = next(d['request']['messages'][0]['content'] for d in recs if r2.kind(d) == 'agent')
    st['a_system_rule_reproduces_E085'] = adk_system(e085_agent['instruction'], e085_agent['name'],
                                                     e085_agent['description']) == sys0
    print(f'self-test (a): ADK system-message rule reproduces E085\'s system message: '
          f'{st["a_system_rule_reproduces_E085"]}')

    episodes = []
    passing = {t for t, r in results.items() if r.get('resolved')}
    for task in sorted(set(groups) | set(results)):
        reqs = groups.get(task, [])
        res = results.get(task, {})
        ep = {'id': f'{a.run}_{task}', 'run': a.run, 'task': task, 'resolved': res.get('resolved'),
              'tool_calls': res.get('tool_calls'), 'total_llm_calls': res.get('total_llm_calls'),
              'duration_seconds': res.get('duration_seconds'), 'requests': len(reqs),
              'captured_calls': sum(len(d['response']['choices'][0]['message'].get('tool_calls') or [])
                                    for d in reqs),
              'graph_calls': sum(c['function']['name'] in GRAPH_TOOLS for d in reqs
                                 for c in d['response']['choices'][0]['message'].get('tool_calls') or []),
              '_raw': reqs}
        if not reqs:
            ep['drop'] = 'no captures'
        elif not res:
            ep['drop'] = 'no result'
        elif not res.get('resolved'):
            ep['drop'] = 'not resolved'
        elif res.get('tool_calls') is None or res['tool_calls'] > a.max_calls:
            ep['drop'] = f'tool_calls > {a.max_calls}'
        elif ep['graph_calls']:
            ep['drop'] = 'graph tool call'
        elif res.get('total_llm_calls') != len(reqs):
            ep['drop'] = 'captured requests != recorded total_llm_calls'
        else:
            ep['drop'] = None
        episodes.append(ep)
    caps = {c: sum(1 for t in passing if (results[t].get('tool_calls') or 10 ** 9) <= c) for c in (28, 40)}
    kept = [e for e in episodes if e['drop'] is None]
    print(f'episodes: {len(episodes)}, passing {len(passing)} (tool_calls <= 28: {caps[28]}, <= 40: {caps[40]}); '
          f'kept {len(kept)} (cap {a.max_calls}); dropped: '
          f'{dict(collections.Counter(e["drop"] for e in episodes if e["drop"]))}')

    # (a) user prompt vs swegemma 0.2.11's build_agent_prompt for v30
    bap = load_build_agent_prompt(a.harness_src)
    if bap is not None:
        from types import SimpleNamespace as NS
        tm = float(new_budget['time']) if '.' in str(new_budget['time']) else int(new_budget['time'])
        cfg = NS(budget=NS(time_minutes=tm, tool_calls=int(new_budget['calls']), turns=int(new_budget['turns']),
                           cost_usd=None),
                 harness=NS(command_timeout_seconds=int(v30['timeout_seconds']), max_stdout_chars=5000,
                            max_file_lines=150, max_file_chars=10000),
                 enable_sandbox_testing=True, graph_dir=None, embeddings_dir=None)
        n_eq = n = 0
        for ep in kept:
            u = ep['_raw'][0]['request']['messages'][1]['content']
            tree_head = '(up to 3 levels):\n```\n'
            i = u.rfind(tree_head)
            tree = u[i + len(tree_head):-len('\n```\n')] if i >= 0 and u.endswith('\n```\n') else ''
            tr = task_rows[ep['task']]
            want = bap(NS(repo=tr['repo'], problem_statement=tr['problem_statement'], hints_text=tr.get('hints_text'),
                          base_commit=tr.get('base_commit')), cfg, tree, declared_tools=v30['tools'])
            n += 1
            n_eq += rewrite_user_prompt(u, new_budget)[0] == want
        st['a_user_prompt_equals_harness_0211'] = {'episodes': n, 'equal': n_eq}
        print(f'self-test (a): rewritten user prompt == swegemma 0.2.11 build_agent_prompt(v30): {n_eq}/{n}')
    else:
        st['a_user_prompt_equals_harness_0211'] = 'harness source not found'

    # ---- steps 2-5
    rows, checks, all_turns = [], [], []
    for n, ep in enumerate(kept):
        turns, erows, echecks = build_episode(ep, ctx, parser, stats)
        ep['_turns'] = turns
        rows += erows
        checks += echecks
        all_turns += [(ep, t) for t in turns]
        if (n + 1) % 10 == 0:
            print(f'  [{n + 1}/{len(kept)}] {time.time() - t_start:.0f}s', flush=True)
    too_long = [r for r in rows if r['n_tokens'] > a.max_len]
    rows = [r for r in rows if r['n_tokens'] <= a.max_len]
    checks = [c for c in checks if c['row'] in {r['id'] for r in rows}]

    # ---- self-tests b, c, d on the written rows
    b, c = check_rows(R, parser, rows, checks)
    st['b_trained_turns_parse'] = b
    st['c_labels_inside_assistant_turns'] = c
    st['d_max_len'] = {'max_len': a.max_len, 'longest': max((r['n_tokens'] for r in rows), default=0),
                       'dropped_longer': len(too_long), 'dropped_ids': [r['id'] for r in too_long]}
    dev_first = [x for x in checks if x['first']]
    dev_later = [x for x in checks if not x['first']]
    report['deviation'] = {
        'first_trained_turn_exact_context': f'{sum(x["exact"] for x in dev_first)}/{len(dev_first)}',
        'later_trained_turns': len(dev_later),
        'later_context_token_diff': dist([abs(x['len_diff']) for x in dev_later]),
        'note': 'later trained turns see earlier trained turns in generated form (no newline before <channel|>, '
                'visible text before the calls): one token per earlier thinking turn, + visible text copies'}

    # ---- outputs
    with open(out / 'train_tokens.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    with open(out / 'episodes.jsonl', 'w') as f:
        for ep in episodes:
            d = {k: v for k, v in ep.items() if not k.startswith('_')}
            d['kept'] = ep['drop'] is None
            if '_turns' in ep:
                d['turns'] = [{k: t[k] for k in ('k', 'names', 'n_calls', 'mode', 'mask', 'segment', 'think_tokens_raw',
                                                  'think_tokens', 'truncated', 'prompt_tokens', 'finish')} |
                              {'visible_chars': len(t['visible'])} for t in ep['_turns']]
                d['sequences'] = sum(r['episode'] == ep['id'] for r in rows)
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # ---- report
    trained = [(ep, t) for ep, t in all_turns if t['mode']]
    masks = collections.Counter(t['mask'] or f'trained_{t["mode"]}' for _, t in all_turns)
    mix = collections.Counter(n for _, t in trained for n in t['names'])
    per_turn = collections.Counter(t['n_calls'] for _, t in trained)
    tot = sum(r['n_tokens'] for r in rows)
    trn = sum(r['n_trained'] for r in rows)
    think_tr = [t['think_tokens'] for _, t in trained if t['mode'] == 'full']
    think_raw = [t['think_tokens_raw'] for _, t in all_turns if t['think_tokens_raw']]
    trunc = collections.Counter(t['truncated'] for _, t in all_turns if t['truncated'])
    # generation time per episode at Gemma's decode speed (all of DeepSeek's turns, rendered in Gemma tokens)
    gen_min = []
    for ep in kept:
        toks = sum(t['gen_tokens'] for t in ep['_turns'] if not t['text_only'])
        gen_min.append(round(toks / GEMMA_DECODE_TOK_S / 60, 2))
    prompts = [t['prompt_tokens'] for _, t in trained if t['prompt_tokens']]
    first_edit = []  # 1-based index of the first edit_file/write_file call (v30's prompt: edit by the 12th call)
    for ep in kept:
        n = 0
        for t in ep['_turns']:
            hit = next((n + i + 1 for i, x in enumerate(t['names']) if x in EDIT_TOOLS), None)
            if hit:
                first_edit.append(hit)
                break
            n += t['n_calls']
    report['episodes'] = {
        'total': len(episodes), 'passing': len(passing), 'passing_tool_calls_le_28': caps[28],
        'passing_tool_calls_le_40': caps[40], 'kept': len(kept), 'max_calls': a.max_calls,
        'dropped': dict(collections.Counter(e['drop'] for e in episodes if e['drop'])),
        'kept_tasks': [e['task'] for e in kept], 'grouping': dict(stats),
        'graph_tool_calls_all_episodes': sum(e['graph_calls'] for e in episodes),
        'episodes_calling_graph_tools': sum(e['graph_calls'] > 0 for e in episodes),
        'mean_recorded_tool_calls': round(statistics.mean(e['tool_calls'] for e in kept), 2) if kept else 0,
        'mean_captured_calls': round(statistics.mean(e['captured_calls'] for e in kept), 2) if kept else 0,
        'mean_llm_turns': round(statistics.mean(e['requests'] for e in kept), 2) if kept else 0,
        'first_edit_call_index': dict(dist(first_edit), after_12th_call=sum(x > 12 for x in first_edit),
                                      no_edit_tool=len(kept) - len(first_edit)),
        'segments': sum(e['segments'] for e in kept),
        'segment_breaks': dict(sum((collections.Counter(e['segment_breaks']) for e in kept), collections.Counter())),
        'episodes_with_compaction': sum(e['segments'] > 1 for e in kept)}
    report['turns'] = {'total': len(all_turns), 'masks': dict(masks), 'trained': len(trained),
                       'trained_full': sum(t['mode'] == 'full' for _, t in trained),
                       'trained_call_body': sum(t['mode'] == 'call_body' for _, t in trained),
                       'trained_with_visible_text': sum(bool(strip_thinking(t['visible'])) for _, t in trained),
                       'tool_call_mix': dict(mix.most_common()), 'calls_per_trained_turn': dict(sorted(per_turn.items())),
                       'prompt_tokens_of_trained_turns': dist(prompts),
                       'trained_turns_prompt_over_14336': sum(p > 14336 for p in prompts)}
    report['thinking'] = {'cap': a.think_cap, 'limit': a.think_cap - a.think_margin,
                          'turns_with_thinking': len(think_raw), 'truncated_turns': sum(trunc.values()),
                          'truncated_by': dict(trunc), 'raw_tokens': dist(think_raw), 'trained_tokens': dist(think_tr),
                          'compaction_summaries': len(ctx.summaries),
                          'summaries_with_budget_rewritten': sum(n > 0 for n in ctx.summaries.values()),
                          'summary_budget_mentions_rewritten': sum(ctx.summaries.values()),
                          'reasoning_mentioning_e085_budget': sum(
                              bool(re.search(r'\b(100 (tool )?calls|30 min|300 turns)', t['thinking']))
                              for _, t in all_turns)}
    report['tokens'] = {'sequences': len(rows), 'total_tokens': tot, 'trained_tokens': trn,
                        'trained_share': round(trn / tot, 3) if tot else 0,
                        'length': dist([r['n_tokens'] for r in rows]),
                        'segment_tokenization_equal': f'{sum(r["segment_tokenization_equal"] for r in rows)}/{len(rows)}'}
    report['gemma_decode_minutes_per_episode_at_35_tok_s'] = dict(dist(gen_min), over_8_min=sum(x > 8 for x in gen_min))
    report['train_time_hours'] = {f'{s} tok/s': {'1 epoch': round(tot / s / 3600, 2), '2 epochs': round(2 * tot / s / 3600, 2),
                                                 '3 epochs': round(3 * tot / s / 3600, 2)} for s in TOK_S}
    report['seconds'] = round(time.time() - t_start)
    json.dump(report, open(out / 'build_report.json', 'w'), indent=1)

    e, tu, th, tk = report['episodes'], report['turns'], report['thinking'], report['tokens']
    print(f'\n== episodes: kept {e["kept"]} of {e["passing"]} passing (cap {a.max_calls}; <= 28: '
          f'{e["passing_tool_calls_le_28"]}, <= 40: {e["passing_tool_calls_le_40"]}); dropped {e["dropped"]}')
    print(f'   mean recorded tool_calls {e["mean_recorded_tool_calls"]}, captured calls {e["mean_captured_calls"]}, '
          f'LLM turns {e["mean_llm_turns"]}; segments {e["segments"]} ({e["episodes_with_compaction"]} episodes '
          f'compacted; breaks {e["segment_breaks"]}); first edit call {e["first_edit_call_index"]}')
    print(f'== turns: {tu["total"]}: {tu["masks"]}')
    print(f'   trained {tu["trained"]} (full {tu["trained_full"]}, call_body {tu["trained_call_body"]}, with visible '
          f'text {tu["trained_with_visible_text"]}); calls per trained turn {tu["calls_per_trained_turn"]}')
    print(f'   tool-call mix (trained turns): {tu["tool_call_mix"]}')
    print(f'   prompt tokens of trained turns {tu["prompt_tokens_of_trained_turns"]}; over 14,336: '
          f'{tu["trained_turns_prompt_over_14336"]}')
    print(f'== thinking: truncated {th["truncated_turns"]} turns {th["truncated_by"]} (limit {th["limit"]}); raw '
          f'{th["raw_tokens"]}; trained {th["trained_tokens"]}')
    print(f'   compaction summaries {th["compaction_summaries"]}, {th["summaries_with_budget_rewritten"]} with budget '
          f'numbers rewritten ({th["summary_budget_mentions_rewritten"]} mentions); reasoning mentioning E085 budgets '
          f'{th["reasoning_mentioning_e085_budget"]}')
    print(f'== tokens: {tk["sequences"]} sequences, {tk["total_tokens"]} tokens, trained {tk["trained_tokens"]} '
          f'({tk["trained_share"]:.1%}); length {tk["length"]}; segment-wise tokenization == whole '
          f'{tk["segment_tokenization_equal"]}')
    print(f'== Gemma decode time per episode at 35 tok/s: {report["gemma_decode_minutes_per_episode_at_35_tok_s"]}')
    print(f'== training time: {report["train_time_hours"]}')
    print(f'== self-test (b) trained turns parse back: {b}')
    print(f'== self-test (c) labels: {c}')
    print(f'== self-test (d) max len: {st["d_max_len"]}')
    print(f'== context deviation: {report["deviation"]}')
    print(f'\nwrote {out}/episodes.jsonl, train_tokens.jsonl ({len(rows)} rows), build_report.json in '
          f'{report["seconds"]}s')


if __name__ == '__main__':
    sys.exit(main())
