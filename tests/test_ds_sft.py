"""Unit tests for scripts/ds_sft_build.py (DeepSeek trajectories -> v30-format Gemma SFT sequences).

Pure parts (no data): the v30 rewrite of the harness prompt and of compaction summaries, ADK's system message, the
thinking truncation, the history-block split and the generated form. The conversion of one DeepSeek turn end to end
(render with Kaggle's June template through scripts/rft_build.Renderer, generated form, tokens, decode, vLLM 0.19.1's
gemma4 parsers from the wheel) runs when the June tokenizer and the vLLM wheel exist: data/ of this checkout, or
DS_SFT_DATA_ROOT=<main checkout>/data from a worktree. Host only: .venv/bin/python -m pytest tests/test_ds_sft.py
"""

import json
import os
import pathlib
import sys

import pytest

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import ds_sft_build as D  # noqa: E402

DATA = pathlib.Path(os.environ.get('DS_SFT_DATA_ROOT', ROOT / 'data'))
Q = '<|"|>'

USER = ('You are evaluating a software engineering task for repository o/r.\n\nProblem Statement:\nFix it\n\n'
        '## Task Budget (Session terminates when any budget is exhausted)\n- Time allowance: 30.0 minutes\n'
        '- Tool calls allowance: 100 calls\n- Max loop iterations: 300 turns\n\n## Execution Environment Rules\n'
        '- Single command timeout: 240 seconds (commands exceeding this fail without ending the session)\n\n'
        '## Instructions:\n0. All source code is under `/workspace`.\n'
        '## Code Intelligence Tools\nThis repository has pre-built code graph data:\n- `search_similar_code(query)`: x\n'
        '\n## Workspace Layout\nThe repository is located at `/workspace`. Here is the directory tree (up to 3 levels):'
        '\n```\n./a.py\n```\n')
NEW = {'time': '8.0', 'calls': '28', 'turns': '80'}
OLD = {'time': '30.0', 'calls': '100', 'turns': '300'}


def test_user_prompt_rewrite():
    out, n = D.rewrite_user_prompt(USER, NEW)
    assert n == 4
    assert '- Time allowance: 8.0 minutes\n- Tool calls allowance: 28 calls\n- Max loop iterations: 80 turns\n' in out
    assert 'Code Intelligence' not in out and 'search_similar_code' not in out
    # build_agent_prompt joins its parts with '\n': the instructions are followed directly by the layout part
    assert '0. All source code is under `/workspace`.\n## Workspace Layout\n' in out
    assert D.prompt_budget(USER) == OLD


def test_summary_budget_rewrite():
    text = ('- Budget: 30 min, 100 tool calls, 300 loop iterations. Constraints: 30-minute time budget; budgets are '
            '30 min / 100 tool calls / 300 iterations; read lines 300-330 of a 100-line file in 30 seconds')
    out, n = D.rewrite_budget_mentions(text, OLD, NEW)
    assert out.startswith('- Budget: 8 min, 28 tool calls, 80 loop iterations. Constraints: 8-minute time budget;')
    assert 'budgets are 8 min / 28 tool calls / 80 iterations' in out
    assert out.endswith('read lines 300-330 of a 100-line file in 30 seconds')  # other numbers untouched
    assert n == 7


def test_adk_system():
    assert D.adk_system('Be good.\n', 'swe_agent_stock_v30') == \
        'Be good.\n\n\nYou are an agent. Your internal name is "swe_agent_stock_v30".'


def test_truncate_thinking_sentence_boundary():
    ntok = lambda s: len(s.split())  # noqa: E731 (toy tokenizer: words)
    text = 'One two three. Four five six.\n\n```py\nx = 1\ny = 2\n```\nSeven eight nine ten. Eleven twelve.'
    assert D.truncate_thinking(text, ntok, 100) == (text, None)
    out, how = D.truncate_thinking(text, ntok, 8)
    assert (out, how) == ('One two three. Four five six.', 'boundary')  # not inside the code fence
    out, how = D.truncate_thinking(text, ntok, 18)
    assert how == 'boundary' and out.endswith('```\nSeven eight nine ten.') and ntok(out) == 18
    out, how = D.truncate_thinking(text, ntok, 17)  # the fence closes at 14 words: cut after it, never inside
    assert how == 'boundary' and out.endswith('y = 2\n```') and ntok(out) == 14


def test_split_block_and_generated_turn():
    calls = (f'<|tool_call>call:read_file{{filepath:{Q}a.py{Q}}}<tool_call|>'
             f'<|tool_call>call:run_command{{command:{Q}ls{Q}}}<tool_call|>')
    resp = f'response:read_file{{value:{Q}x{Q}}}<tool_response|><|tool_response>response:run_command{{value:{Q}y{Q}}}'
    block = f'<|channel>thought\nLook.\n<channel|>{calls}<|tool_response>{resp}<tool_response|>Reading.<turn|>\n'
    sp = D.split_block(block, 'Look.', ' Reading. ')
    assert sp['calls'] == calls and sp['trailer'] == 'Reading.<turn|>\n'
    th, v, cl = D.generated_turn(sp['calls'], 'Look.', ' Reading. ')
    assert th + v + cl == f'<|channel>thought\nLook.<channel|>Reading.{calls}<|tool_response>'
    assert D.label_offset('full', th, v) == 0
    assert (th + v + cl)[D.label_offset('call_body', th, v):].startswith('call:read_file{')
    with pytest.raises(D.BlockError):
        D.split_block(block, 'Other thought.', ' Reading. ')


def test_assemble_and_runs():
    sp = {'resp': 'R1<tool_response|>', 'trailer': ''}
    pieces = [(0, 'B0', None, None), (1, 'B1', ('T', '', 'C1'), sp), (2, 'B2', None, None), (3, 'B3', ('U', '', 'C3'),
                                                                                        sp), (4, 'B4', None, None)]
    text, spans = D.assemble('P', pieces)
    assert text == 'PB0TC1R1<tool_response|>B2UC3'  # ends after the last trained turn
    assert {k: text[s:e] for k, (s, e) in spans.items()} == {1: 'TC1', 3: 'UC3'}
    assert D.runs_of([-100, 5, 6, -100, 7]) == [(1, 3), (4, 5)]


# ------------------------------------------------------------------ one DeepSeek turn end to end (real template)

def _have_data():
    return (DATA / 'rft2' / 'tokenizer_june' / 'tokenizer.json').exists() and \
        list((DATA / 'wheelhouse').glob('vllm-0.19.1-*.whl'))


@pytest.mark.skipif(not _have_data(), reason='needs data/rft2/tokenizer_june and the vLLM 0.19.1 wheel')
def test_one_deepseek_turn_end_to_end():
    pytest.importorskip('transformers')
    import rft2_build as r2
    R = r2.make_renderer(DATA / 'rft2' / 'tokenizer_june', DATA / 'probe' / 'chat_template_kaggle_june.jinja')
    parser = D.Gemma4Parser(D.find_wheel(DATA))
    tools = [{'type': 'function', 'function': {
        'name': 'edit_file', 'description': 'Replace text.', 'parameters': {
            'type': 'object', 'properties': {'filepath': {'type': 'string'}, 'old_string': {'type': 'string'},
                                             'new_string': {'type': 'string'}},
            'required': ['filepath', 'old_string', 'new_string']}}},
        {'type': 'function', 'function': {'name': 'run_command', 'description': 'Run.', 'parameters': {
            'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]
    msgs = [{'role': 'system', 'content': D.adk_system('Fix bugs.\n', 'swe_agent_stock_v30')},
            {'role': 'user', 'content': D.rewrite_user_prompt(USER, NEW)[0]},
            {'role': 'assistant', 'content': None, 'reasoning': 'Read first.', 'reasoning_content': 'Read first.',
             'tool_calls': [{'id': 'call_00_a', 'type': 'function', 'function': {
                 'name': 'run_command', 'arguments': json.dumps({'command': 'cat a.py'})}}]},
            {'role': 'tool', 'tool_call_id': 'call_00_a', 'content': '{"status": "ok", "stdout": "x = \\"1\\"\\n"}'}]
    # DeepSeek's reply as the proxy recorded it: reasoning_content, visible text, two parallel calls (JSON arguments
    # with a newline, quotes, a backslash, non-ASCII)
    new = 'x = "2"  # é\nprint(r"\\d")'
    ds = {'choices': [{'message': {
        'role': 'assistant', 'content': 'Fixing the value.',
        'reasoning_content': 'The value must be "2".\n\nSo edit a.py, then run it.',
        'tool_calls': [
            {'id': 'call_00_b', 'type': 'function', 'function': {'name': 'edit_file', 'arguments': json.dumps(
                {'filepath': 'a.py', 'old_string': 'x = "1"', 'new_string': new})}},
            {'id': 'call_01_c', 'type': 'function', 'function': {'name': 'run_command', 'arguments': json.dumps(
                {'command': 'python3 a.py'})}}]}, 'finish_reason': 'tool_calls'}]}
    asst = D.rb.response_message(ds)
    results = [{'role': 'tool', 'tool_call_id': 'call_00_b', 'content': '{"status": "ok"}'},
               {'role': 'tool', 'tool_call_id': 'call_01_c', 'content': '{"status": "ok", "stdout": "2"}'}]
    P = R.text(msgs, tools, True, True)
    F = R.text(msgs + [asst] + results, tools, True, True)
    assert F.startswith(P)
    sp = D.split_block(F[len(P):], asst['reasoning'], asst['content'])
    th, v, cl = D.generated_turn(sp['calls'], asst['reasoning'], asst['content'])
    g = th + v + cl
    assert g.startswith('<|channel>thought\nThe value must be "2".') and '<channel|>Fixing the value.<|tool_call>' in g
    assert g.endswith('<tool_call|><|tool_response>') and g.count('<|tool_call>') == 2
    # the trained turn, tokenized and decoded as the model would emit it, parses back into DeepSeek's turn
    parsed = parser(R.tok.decode(R.ids(g), skip_special_tokens=False))
    calls = [(c['function']['name'], json.loads(c['function']['arguments'])) for c in asst['tool_calls']]
    assert D.parse_matches(parsed, asst['reasoning'], asst['content'], calls) == ''
    assert parsed['calls'][0][1]['new_string'] == new
    # the sequence: P + generated turn + its responses + the history trailer, labels exactly on the generated turn
    text, spans = D.assemble(P, [(0, F[len(P):], (th, v, cl), sp)])
    assert text == P + g
    s, e = spans[0]
    ids, at, same = D.tokenize_cuts(R, text, {s, e})
    assert same and ids[:at[s]] == R.ids(P)
    assert R.tok.decode(ids[at[s]:at[e]], skip_special_tokens=False) == g
    # history form differs from the generated form only by the newline before <channel|> and the text's position
    assert sp['think'] == '<|channel>thought\n' + asst['reasoning'] + '\n<channel|>'
    assert sp['trailer'] == 'Fixing the value.<turn|>\n'
