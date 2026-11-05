"""Unit tests for scripts/rft2_build.py's pure-Python parts: ADK-rejection detection, the intended script of a garbled
call, repair planning (merge / none) and its application to messages, the held-out split, sequence assignment and the
sampled-form thinking end. Synthetic messages only: no tokenizer, no captures."""

import json
import pathlib
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
import rft2_build as r2  # noqa: E402

SYS = {'role': 'system', 'content': 'You are an expert software engineer.'}
USER = {'role': 'user', 'content': 'Problem Statement:\nfix it'}


def call(cid, file_path='show.py', skill='repo-tools', args=None, **extra):
    a = {'args': args if args is not None else {'target': 'a.py'}}
    if file_path is not None:
        a['file_path'] = file_path
    if skill is not None:
        a['skill_name'] = skill
    a.update(extra)
    return {'type': 'function', 'id': cid, 'function': {'name': 'run_skill_script', 'arguments': json.dumps(a)}}


def asst(c, thinking):
    return {'role': 'assistant', 'content': None, 'tool_calls': [c], 'reasoning': thinking,
            'reasoning_content': thinking}


def result(cid, error=None):
    body = {'error': error, 'error_code': 'SKILL_NOT_FOUND'} if error else {'skill_name': 'repo-tools',
                                                                             'stdout': 'ok'}
    return {'role': 'tool', 'tool_call_id': cid, 'content': json.dumps(body)}


def episode(turns):
    """turns: [(assistant message, tool result or None)] -> requests as rft2_build.as_req builds them: request k has
    the history up to turn k, its response is turn k's message; a last request carries the last result and gets a
    text answer."""
    reqs, hist = [], [SYS, USER]
    for m, res in turns:
        reqs.append({'messages': list(hist), 'response': m, 'finish': 'tool_calls'})
        hist = hist + [m] + ([res] if res else [])
    reqs.append({'messages': list(hist), 'response': {'role': 'assistant', 'content': 'done'}, 'finish': 'stop'})
    return reqs, hist


# ---------------------------------------------------------------- call shapes

def test_tool_error():
    assert r2.tool_error('{"error": "Skill x not found.", "error_code": "SKILL_NOT_FOUND"}')['error_code']
    assert r2.tool_error('{"skill_name": "repo-tools", "stdout": "error: x"}') is None
    assert r2.tool_error('not json') is None
    assert r2.tool_error(None) is None


def test_intended_scripts_of_garbled_calls():
    assert r2.intended_scripts(call('c', '「show.py」', '「repo-tools」')) == ['show.py']
    assert r2.intended_scripts(call('c', 'show.py`,skill_name:', None)) == ['show.py']  # backtick swallow
    nested = call('c', None, None, args={'file_path': 'outline.py', 'skill_name': 'repo-tools', 'files': 'a.py'})
    assert r2.intended_scripts(nested) == ['outline.py']
    assert r2.intended_scripts(call('c', 'repro.py', 'create.py')) == ['repro.py', 'create.py']
    assert r2.intended_scripts(call('c', 'ls -R rich/')) == []
    garbled_key = call('c', None, 'repo-tools', **{'},file_path': 'edit.py'})
    assert r2.intended_scripts(garbled_key) == ['edit.py']
    other = {'type': 'function', 'id': 'x', 'function': {'name': 'submit_patch', 'arguments': '{}'}}
    assert r2.intended_scripts(other) == ['submit_patch']


def test_well_formed():
    assert r2.well_formed(call('c'))
    assert not r2.well_formed(call('c', skill='「repo-tools」'))
    assert not r2.well_formed(call('c', skill='":"'))
    assert not r2.well_formed(call('c', 'show.py`,skill_name:', None))
    assert not r2.well_formed(call('c', args={'target': 'a.py', 'file_path': 'show.py'}))
    assert not r2.well_formed(call('c', 'rich/table.py'))


# ---------------------------------------------------------------- repairs

def test_merge_repair_keeps_original_thinking_and_corrected_call():
    turns = [(asst(call('c0', 'locate.py'), 'find it'), result('c0')),
             (asst(call('c1', '「show.py」', '「repo-tools」'), 'read a.py next'),
              result('c1', "Skill '「repo-tools」' not found.")),
             (asst(call('c2'), 'My apologies, I used Japanese brackets.'), result('c2')),
             (asst(call('c3', 'edit.py', args={'target': 'a.py', 'old': 'x', 'new': 'y'}), 'edit now'), result('c3'))]
    reqs, final = episode(turns)
    recs = r2.turn_records(reqs)
    assert [t['rejected'] for t in recs] == [False, True, False, False, False]
    assert recs[-1]['text_only']
    assert recs[1]['in_next'] and recs[1]['scripts'] == ['show.py']
    drop, override, repairs, unrepaired = r2.plan_repairs(recs)
    assert drop == {'c1'} and override == {'c2': 'read a.py next'} and unrepaired == []
    assert repairs == [{'chain': [1], 'retry': 2, 'mode': 'merge', 'why': '', 'same_args': True}]
    fixed = r2.apply_repairs(final, drop, override)
    assert [m['role'] for m in fixed] == ['system', 'user', 'assistant', 'tool', 'assistant', 'tool', 'assistant',
                                          'tool']
    assert not any('not found' in str(m.get('content')) for m in fixed)
    retry = fixed[4]
    assert retry['tool_calls'][0]['id'] == 'c2'
    assert retry['reasoning'] == retry['reasoning_content'] == 'read a.py next'
    assert final[6]['reasoning'].startswith('My apologies')  # the input is not modified
    # a request in the middle: the rejected turn disappears from its history too
    mid = r2.apply_repairs(reqs[2]['messages'], drop, override)
    assert mid == reqs[1]['messages']


def test_chain_of_rejections_merges_into_the_first_thinking():
    turns = [(asst(call('c0', '「show.py」'), 'first idea'), result('c0', 'Skill not found.')),
             (asst(call('c1', 'show.py`,skill_name:', None), 'again'), result('c1', "Argument 'skill_name' is required.")),
             (asst(call('c2'), 'Sorry.'), result('c2'))]
    reqs, final = episode(turns)
    drop, override, repairs, _ = r2.plan_repairs(r2.turn_records(reqs))
    assert drop == {'c0', 'c1'} and override == {'c2': 'first idea'}
    assert repairs[0]['chain'] == [0, 1] and repairs[0]['mode'] == 'merge'
    fixed = r2.apply_repairs(final, drop, override)
    assert [m['role'] for m in fixed] == ['system', 'user', 'assistant', 'tool']


def test_no_repair_when_the_retry_runs_another_script():
    turns = [(asst(call('c0', 'ls -R rich/'), 'list files'), result('c0', "Script 'ls -R rich/' not found.")),
             (asst(call('c1', 'search.py', args={'text': 'Progress'}), 'I should use search.py instead.'),
              result('c1'))]
    reqs, final = episode(turns)
    drop, override, repairs, unrepaired = r2.plan_repairs(r2.turn_records(reqs))
    assert drop == set() and override == {} and unrepaired == [0]
    assert repairs[0]['mode'] == 'none' and 'different script' in repairs[0]['why']
    assert r2.apply_repairs(final, drop, override) == final


def test_no_repair_without_a_confirmed_retry():
    # rejected, then the episode ends with a call whose result never came back (timeout) and that is garbled too
    turns = [(asst(call('c0', '「show.py」'), 'x'), result('c0', 'Skill not found.')),
             (asst(call('c1', args={'target': 'a.py', 'file_path': 'show.py'}, file_path=None, skill=None), 'y'),
              None)]
    reqs, _ = episode(turns)
    recs = r2.turn_records(reqs)
    assert not recs[1]['has_result'] and not recs[1]['well_formed']
    drop, _, repairs, unrepaired = r2.plan_repairs(recs)
    assert drop == set() and unrepaired == [0] and repairs[0]['why'].startswith('retry not confirmed')
    # rejected, then only a text answer
    reqs, _ = episode(turns[:1])
    _, _, repairs, unrepaired = r2.plan_repairs(r2.turn_records(reqs))
    assert unrepaired == [0] and repairs[0]['why'] == 'retry is not a call'
    # a garbled last call has no result to show the rejection: not 'rejected', but well_formed catches it (masked)
    last = r2.turn_records(reqs[:1])[0]
    assert not last['rejected'] and not last['has_result'] and not last['well_formed']


def test_no_repair_across_a_compaction():
    turns = [(asst(call('c0', '「show.py」'), 'x'), result('c0', 'Skill not found.')),
             (asst(call('c1'), 'Sorry.'), result('c1'))]
    reqs, _ = episode(turns)
    summary = {'role': 'user', 'content': [{'type': 'text', 'text': 'For context:'}]}
    reqs[1]['messages'] = [SYS, summary, result('c0', 'Skill not found.')]  # history rewritten by compaction
    recs = r2.turn_records(reqs)
    assert recs[0]['rejected'] and not recs[0]['in_next']
    drop, _, repairs, _ = r2.plan_repairs(recs)
    assert drop == set() and 'compaction' in repairs[0]['why']


def test_rejection_found_in_a_later_request():
    turns = [(asst(call('c0', '「show.py」'), 'x'), result('c0', 'Skill not found.')),
             (asst(call('c1'), 'Sorry.'), result('c1')),
             (asst(call('c2', 'edit.py'), 'z'), result('c2'))]
    reqs, _ = episode(turns)
    reqs[1]['messages'] = reqs[1]['messages'][:-1]  # the error result only appears in request 2
    assert r2.turn_records(reqs)[0]['rejected']


# ---------------------------------------------------------------- split, sequences, trained form

def test_split_heldout_is_stratified_and_fixed():
    tasks = [f'fastapi_{i}' for i in range(61)] + [f'rich_{i}' for i in range(46)] + \
        [f'requests_{i}' for i in range(5)] + ['httpx_1']
    held = r2.split_heldout(tasks)
    assert held == r2.split_heldout(list(reversed(tasks)))
    count = lambda p: sum(t.startswith(p + '_') for t in held)  # noqa: E731
    assert (count('fastapi'), count('rich'), count('requests'), count('httpx')) == (20, 15, 2, 0)


def test_assign_sequences_splits_at_a_prefix_break():
    def t(i, P, T, F=None, masked=False):
        return {'i': i, 'P': P, 'T': T, 'F': F if F is not None else P + T, 'm': masked}

    turns = [t(0, 'S', 'a'), t(1, 'Sa|', 'b'), t(2, 'Sab|', 'c', masked=True),  # 'Sab|' != 'Sa|b': no prefix
             t(3, 'X', 'd')]  # compaction: a new history
    seqs = r2.assign_sequences(turns, lambda x: x['m'])
    assert seqs == {1: [0, 1], 3: [3]}


def test_sampled_form_and_trained_text_spans():
    T = '<|channel>thought\nlook\n\n<channel|><|tool_call>call:x{}<tool_call|><|tool_response>'
    assert r2.sampled_form(T) == T.replace('\n\n<channel|>', '\n<channel|>')
    S = 'PROMPT' + T + 'RESULT'
    text, spans = r2.trained_text(S, [(6, 6 + len(T), False)], True)
    assert text == 'PROMPT' + r2.sampled_form(T) + 'RESULT'
    assert text[spans[0][0]:spans[0][1]] == r2.sampled_form(T)
    text, spans = r2.trained_text(S, [(6, 6 + len(T), True)], True)  # call only
    assert text[spans[0][0]:spans[0][1]] == '<|tool_call>call:x{}<tool_call|><|tool_response>'
    text, spans = r2.trained_text(S, [(6, 6 + len(T), False)], False)
    assert text == S and text[spans[0][0]:spans[0][1]] == T
