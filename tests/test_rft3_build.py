"""Unit tests for scripts/rft3_build.py (rft3 step D: records -> token sequences + loss masks) and the pure parts of
scripts/rft3_probe.py. A fake renderer reproduces the Gemma 4 June template for the message shapes the agent sends
(system + user, assistant turns with thinking / text / one tool call, tool results), with a toy tokenizer (special
tokens, words, single characters): prefix consistency, the turn split, the generated form, the masks, the GPU-pass
merge and the mix quota. When transformers, the June tokenizer (data/rft2/tokenizer_june) and the v20 records exist
(the local .venv, not the sandbox image), the same checks run on real records with the real template."""

import collections
import json
import pathlib
import re
import sys

import pytest

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import rft3_build as B  # noqa: E402
import rft3_probe as P  # noqa: E402
import rft3_states as S  # noqa: E402

Q = '<|"|>'


# ------------------------------------------------------------------ fake June template + toy tokenizer

def fmt_value(v):
    if isinstance(v, str):
        return Q + v + Q
    if isinstance(v, dict):
        return '{' + ','.join(f'{k}:{fmt_value(v[k])}' for k in sorted(v)) + '}'
    return str(v)


class FakeRenderer:
    """The June template's rendering of agent requests (see data/probe/chat_template_kaggle_june.jinja): thinking of
    assistant turns after the last user message as <|channel>thought\\n...\\n<channel|>, the call, the following tool
    results inside the same model turn, the assistant's text after its results + <turn|>, no <|turn>model between
    consecutive assistant messages, a generation prompt only after a user/system turn."""

    SPECIAL = re.compile(r'<\|[a-z_"]*\|?>|<[a-z_]+\|>|\w+|\s|.', re.S)

    def text(self, messages, tools, gen, enable_thinking=True):
        out = ['<bos><|turn>system\n<|think|>\n', messages[0]['content'].strip()]
        out += [f'<|tool>declaration:{t["function"]["name"]}<tool|>' for t in tools or []]
        out.append('<turn|>\n')
        rest = messages[1:]
        last_type = 'tool'
        prev_role = None
        for i, m in enumerate(rest):
            if m['role'] == 'tool':
                continue
            if m['role'] == 'user':
                out.append('<|turn>user\n' + m['content'].strip() + '<turn|>\n')
                last_type, prev_role = None, 'user'
                continue
            if prev_role != 'assistant':
                out.append('<|turn>model\n')
            th = m.get('reasoning') or m.get('reasoning_content')
            if th and m.get('tool_calls'):
                out.append('<|channel>thought\n' + th + '\n<channel|>')
            for c in m.get('tool_calls') or []:
                a = c['function']['arguments']
                a = json.loads(a) if isinstance(a, str) else a
                out.append('<|tool_call>call:' + c['function']['name'] + '{' +
                           ','.join(f'{k}:{fmt_value(a[k])}' for k in sorted(a)) + '}<tool_call|>')
                last_type = 'tool_call'
            responded = False
            for f in rest[i + 1:]:
                if f['role'] != 'tool':
                    break
                name = next(c['function']['name'] for c in m['tool_calls'] if c['id'] == f['tool_call_id'])
                out.append(f'<|tool_response>response:{name}{{value:{Q}{f["content"]}{Q}}}<tool_response|>')
                responded, last_type = True, 'tool_response'
            content = B.strip_thinking(m.get('content') or '')
            out.append(content)
            if last_type == 'tool_call' and not responded:
                out.append('<|tool_response>')
            elif not (responded and not content):
                out.append('<turn|>\n')
            prev_role = 'assistant'
        if gen and last_type not in ('tool_response', 'tool_call'):
            out.append('<|turn>model\n')
        return ''.join(out)

    def ids(self, text):
        return self.SPECIAL.findall(text)


TOOLS = [{'type': 'function', 'function': {'name': 'run_skill_script'}},
         {'type': 'function', 'function': {'name': 'submit_patch'}}]


def turn(k, script, args, label='train', thinking=None, visible=None, result='ok', kind=None, src=None, **sampling):
    call = S.script_call(script, args) if script != 'submit' else S.submit_call()
    t = {'id': f'call{k}', 'call': call, 'result': json.dumps({'stdout': f'{result} {k}'}), 'label': label,
         'thinking': thinking, 'visible': visible, 'grounding': None, 'kind': kind or script.replace('.py', '')}
    if label == 'train':
        t['sampling'] = dict({'from': src or ('bootstrap' if thinking else None), 'mask_thinking': not thinking},
                             **sampling)
    return t


def record(turns, source='oracle_clean', rid='r1'):
    return {'id': rid, 'source': source, 'task': 'org__repo-1', 'repo': 'org/repo',
            'request': {'messages': [{'role': 'system', 'content': 'You fix bugs. '},
                                     {'role': 'user', 'content': 'Problem: x is wrong'}],
                        'model': 'm', 'tools': TOOLS, 'chat_template_kwargs': {'enable_thinking': True}},
            'turns': turns, 'gold': {'files': ['a.py'], 'spans': [{'file': 'a.py', 'a': 3, 'b': 3}]}, 'meta': {}}


def sample_record():
    return record([
        turn(0, 'show.py', {'target': 'a.py'}, label='context', thinking='I read a.py first.\n'),
        turn(1, 'show.py', {'target': 'a.py', 'symbol': 'f'}, thinking='Now f.\n'),
        turn(2, 'edit.py', {'target': 'a.py', 'start': '3', 'end': '3', 'new': 'x = 1'},
             thinking='Line 3 is wrong; fix it.', visible='Line 3 sets x wrong, so I fix it.', src='rationalized'),
        turn(3, 'check_patch.py', {}),
        turn(4, 'submit', None, thinking='x' * 40, mask_thinking=True),
    ])


def build(rec, cap=256, verify=True):
    return B.Builder(FakeRenderer(), cap, max_prompt=10 ** 9).build(rec, verify=verify)


def labelled_text(row):
    """The labelled tokens as text runs (consecutive labelled tokens joined)."""
    runs, cur = [], []
    for x in row['labels']:
        if x == -100:
            if cur:
                runs.append(''.join(cur))
                cur = []
        else:
            cur.append(x)
    if cur:
        runs.append(''.join(cur))
    return runs


# ------------------------------------------------------------------ turn split / generated form

def test_split_block_and_generated_form():
    rec = sample_record()
    R = FakeRenderer()
    Pk = [R.text(S.messages_at(rec, k), TOOLS, True) for k in range(len(rec['turns']) + 1)]
    b2 = Pk[3][len(Pk[2]):]
    t = rec['turns'][2]
    pc = B.split_block(b2, t['thinking'], t['visible'])
    assert pc['think'] == '<|channel>thought\nLine 3 is wrong; fix it.\n<channel|>'
    assert pc['call'].startswith('<|tool_call>call:run_skill_script{') and pc['call'].endswith('<tool_call|>')
    assert pc['resp'].endswith('<tool_response|>') and pc['trailer'] == 'Line 3 sets x wrong, so I fix it.<turn|>\n'
    th, v, cl = B.generated_turn(pc['call'], t['thinking'], t['visible'])
    # the model's own order: thought (no newline before <channel|>), text, call, stop token
    assert th + v + cl == ('<|channel>thought\nLine 3 is wrong; fix it.<channel|>Line 3 sets x wrong, so I fix it.'
                           + pc['call'] + '<|tool_response>')
    b3 = Pk[4][len(Pk[3]):]
    pc3 = B.split_block(b3, None, None)
    assert pc3['think'] == '' and pc3['trailer'] == ''
    with pytest.raises(B.BlockError):
        B.split_block(b2, 'other thinking', t['visible'])
    with pytest.raises(B.BlockError):
        B.split_block(b2, t['thinking'], None)  # the text trailer is not accounted for


def test_strip_thinking_like_the_template():
    assert B.strip_thinking('  a<|channel>hidden<channel|>b  ') == 'ab'
    assert B.strip_thinking(None) == ''


def test_turn_modes():
    t = {'thinking': 'abc', 'sampling': {'mask_thinking': False}}
    assert B.turn_mode(t, 100, 256) == 'full'
    assert B.turn_mode(t, 248, 256) == 'call'          # rft2's cap: >= budget - 8 is not trained
    assert B.turn_mode(t, 248, 0) == 'full'
    assert B.turn_mode(dict(t, sampling={'mask_thinking': True}), 10, 256) == 'call'
    assert B.turn_mode({'thinking': None, 'sampling': {}}, 0, 256) == 'call_body'
    assert B.turn_mode({'thinking': '', 'sampling': {'from': 'bootstrap'}}, 0, 256) == 'call_body'


# ------------------------------------------------------------------ prefix consistency + masks (fake template)

def test_prefix_consistency_fake_template():
    rec = sample_record()
    row = build(rec)
    v = row['verify']
    n = len(rec['turns'])
    assert v['requests'] == n and v['request_prefix_of_history'] == n
    assert v['train_turns'] == 4 and v['generated_turn_tokens_equal'] == 4
    # turn 1 (first train turn) sees its exact request; later ones deviate by the documented newline / text copy
    assert v['train_turns_exact_context_expected'] == 1 and v['train_turns_exact_context'] == 1
    assert v['train_turns_deviating_context'] == 3
    assert B.deviation(rec['turns'], {1: 'full', 2: 'full', 3: 'call_body', 4: 'call'}, 3) == (2, 1)
    assert row['segment_tokenization_equal']


def test_sequence_text_is_history_with_generated_train_turns():
    rec = sample_record()
    R = FakeRenderer()
    Pk = [R.text(S.messages_at(rec, k), TOOLS, True) for k in range(len(rec['turns']) + 1)]
    modes = {1: 'full', 2: 'full', 3: 'call_body', 4: 'call'}
    text, spans, offsets, hist = B.assemble(Pk, rec['turns'], modes)
    assert hist == Pk[-1]
    assert text.startswith(Pk[1])                                            # context turn 0 in history form
    # turn 2 in the sequence: generated form, then its result, then the history trailer
    s2 = next(s for s, e, k in spans if k == 2)
    assert text[s2:].startswith('<|channel>thought\nLine 3 is wrong; fix it.<channel|>Line 3 sets x wrong, so I fix '
                                'it.<|tool_call>call:run_skill_script{')
    assert 'ok 2"}' in text and 'so I fix it.<turn|>\n<|tool_call>call:run_skill_script{file_path' in text
    # removing the deviations gives back the history text exactly
    undo = text.replace('Now f.\n<channel|>', 'Now f.\n\n<channel|>').replace(
        'fix it.<channel|>Line 3 sets x wrong, so I fix it.<|tool_call>', 'fix it.\n<channel|><|tool_call>').replace(
        'x' * 40 + '<channel|>', 'x' * 40 + '\n<channel|>')
    assert undo == hist


def test_masks():
    rec = sample_record()
    row = build(rec)
    runs = labelled_text(row)
    assert len(runs) == 4  # turns 1-4; context turn 0, all tool results and prompts unlabelled
    # turn 1: thinking + call + stop token
    assert runs[0].startswith('<|channel>thought\nNow f.\n<channel|><|tool_call>call:run_skill_script{args:')
    assert runs[0].endswith('<tool_call|><|tool_response>')
    # turn 2: thinking, the visible sentence before the call, the call
    assert runs[1].startswith('<|channel>thought\nLine 3 is wrong; fix it.<channel|>Line 3 sets x wrong')
    # turn 3: no thinking: the call body only, <|tool_call> itself unlabelled
    assert runs[2].startswith('call:run_skill_script{file_path:') and runs[2].endswith('<|tool_response>')
    # turn 4: thinking present but masked (mask_thinking): the call only
    assert runs[3] == '<|tool_call>call:submit_patch{}<tool_call|><|tool_response>'
    assert not any('I read a.py first' in r for r in runs)
    assert not any('ok 1' in r or '<tool_response|>' in r for r in runs)
    assert row['trained_turns'] == 4 and row['think_trained_turns'] == 2
    assert [t['mode'] for t in row['turns']] == [None, 'full', 'full', 'call_body', 'call']


def test_think_cap_masks_long_thinking():
    rec = sample_record()
    rec['turns'][1]['thinking'] = ' '.join(['w'] * 200)  # 399 toy tokens >= 248
    row = build(rec)
    assert row['turns'][1]['mode'] == 'call'
    assert labelled_text(row)[0].startswith('<|tool_call>call:run_skill_script{args:')


def test_drop_over_compaction_point():
    rec = sample_record()
    row = B.Builder(FakeRenderer(), 256, max_prompt=50).build(rec)
    assert row['drop'] == 'prompt >= max'
    assert B.Builder(FakeRenderer(), 256).build(record([turn(0, 'show.py', {'target': 'a'}, label='context')]))[
        'drop'] == 'no train turn'


def test_no_trailer_without_text_and_first_turn_after_generation_prompt():
    rec = record([turn(0, 'locate.py', {'query': 'q'}, thinking='Find it.'), turn(1, 'submit', None)])
    row = build(rec)
    runs = labelled_text(row)
    assert runs[0].startswith('<|channel>thought\nFind it.<channel|><|tool_call>')
    text = ''.join(row['input_ids'])
    assert text.count('<|turn>model\n') == 1 and '<turn|>\n<|tool_call>' not in text
    assert row['verify']['train_turns_exact_context'] == 1  # submit after one thinking turn deviates
    assert row['verify']['train_turns_deviating_context'] == 1


# ------------------------------------------------------------------ merge, mix

def test_overlay_sampled(tmp_path):
    rec = sample_record()
    rec['turns'][3]['sampling'] = {'from': None, 'needs_sampling': True, 'mask_thinking': True}
    rec['turns'][4]['sampling'] = {'from': None, 'needs_sampling': True, 'mask_thinking': True}
    rec['turns'][1]['sampling']['carried'] = 'v19'
    rec['turns'][2]['sampling']['carried'] = 'v19'
    lines = [
        {'id': 'r1', 'source': 'oracle_clean', 'k': 3, 'pass': 'sample', 'ctx': ['real', 'bootstrap', 'rationalized'],
         'kept': {'thinking': 'Check it.', 'visible': '', 'call': rec['turns'][3]['call'], 'from': 'bootstrap',
                  'think_tokens': 3, 'at_budget': False, 'rule': 'check_patch.py', 'call_from': 'sample'}},
        {'id': 'r1', 'source': 'oracle_clean', 'k': 4, 'pass': 'sample', 'kept': None, 'ctx': []},
        {'id': 'r1', 'source': 'oracle_clean', 'k': 4, 'pass': 'fallback', 'ctx': [],
         'kept': {'thinking': 'Submit now.', 'visible': 'Done.', 'call': rec['turns'][4]['call'],
                  'from': 'rationalized', 'think_tokens': 3, 'at_budget': False, 'rule': 'submit_patch',
                  'call_from': 'oracle'}},
    ]
    (tmp_path / 'oracle_clean.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in lines))
    stats = collections.Counter()
    B.overlay_sampled([rec], str(tmp_path), stats)
    t3, t4 = rec['turns'][3], rec['turns'][4]
    assert t3['thinking'] == 'Check it.' and t3['sampling']['from'] == 'bootstrap'
    assert not t3['sampling']['mask_thinking']
    assert t4['thinking'] == 'Submit now.' and t4['visible'] == 'Done.' and t4['sampling']['from'] == 'rationalized'
    assert rec['turns'][2]['thinking'] == 'Line 3 is wrong; fix it.'  # carried turns are never touched
    assert stats['gpu_turns_merged'] == 2 and stats['gpu_turns_pending'] == 0
    # a turn without lines stays pending (thinking-masked)
    rec2 = sample_record()
    rec2['id'] = 'r2'
    rec2['turns'][3]['sampling'] = {'from': None, 'needs_sampling': True, 'mask_thinking': True}
    for k in (1, 2, 4):
        rec2['turns'][k]['sampling']['carried'] = 'v19'
    stats = collections.Counter()
    B.overlay_sampled([rec2], str(tmp_path), stats)
    assert stats['gpu_turns_pending'] == 1 and rec2['turns'][3]['thinking'] is None
    # no directory at all: nothing changes
    stats = collections.Counter()
    B.overlay_sampled([rec2], str(tmp_path / 'absent'), stats)
    assert stats['gpu_turns_pending'] == 1


def test_overlay_from_export_records(tmp_path):
    rec = sample_record()
    rec['turns'][3]['sampling'] = {'from': None, 'needs_sampling': True, 'mask_thinking': True}
    for k in (1, 2, 4):
        rec['turns'][k]['sampling']['carried'] = 'v19'
    ex = json.loads(json.dumps(rec))
    ex['turns'][3]['thinking'] = 'Check.'
    ex['turns'][3]['sampling'] = {'from': 'bootstrap', 'mask_thinking': False}
    (tmp_path / 'oracle_clean.records.jsonl').write_text(json.dumps(ex) + '\n')
    stats = collections.Counter()
    B.overlay_sampled([rec], str(tmp_path), stats)
    assert rec['turns'][3]['thinking'] == 'Check.' and stats['gpu_turns_merged'] == 1


def test_clean_quota_and_order():
    assert B.clean_quota(442, 700, 0.55) == round(0.55 / 0.45 * 700) - 442
    assert B.clean_quota(2000, 10, 0.55) == 0
    ids = ['a', 'b', 'c', 'd']
    assert sorted(B.sha_order(ids)) == ids and B.sha_order(ids) == B.sha_order(list(reversed(ids)))


# ------------------------------------------------------------------ probe (pure parts)

def test_probe_visibility_and_classes():
    show = json.dumps({'stdout': 'a.py f L1-4\n1  def f():\n2      y = 0\n3      x = 2\n4      return x\n'})
    msgs = [{'role': 'system', 'content': 's'}, {'role': 'tool', 'tool_call_id': 'c', 'content': show}]
    gold = {'files': ['a.py'], 'spans': [{'file': 'a.py', 'a': 3, 'b': 3}]}
    known = P.known_of_messages(msgs)
    assert P.visible_files(gold, known) == ['a.py']
    assert P.visible_files({'files': ['b.py'], 'spans': [{'file': 'b.py', 'a': 3, 'b': 3}]}, known) == []
    edit = S.script_call('edit.py', {'target': 'a.py', 'start': '3', 'end': '3', 'new': '    x = 1'})
    q = P.edit_quality(edit, gold, known, {'call': edit})
    assert q == {'resolvable': True, 'overlaps_gold': True, 'matches_oracle': True}
    far = S.script_call('edit.py', {'target': 'a.py', 'start': '1', 'end': '1', 'new': 'def g():'})
    q = P.edit_quality(far, gold, known, {'call': edit})
    assert q['resolvable'] and not q['overlaps_gold'] and not q['matches_oracle']
    assert P.classify({'problem': None, 'call': edit}) == 'edit'
    assert P.classify({'problem': None, 'call': S.script_call('show.py', {'target': 'a.py'})}) == 'read'
    assert P.classify({'problem': 'no_call', 'call': None}) == 'malformed'
    assert P.classify({'problem': 'length', 'call': None}) == 'length'
    assert P.classify({'problem': None, 'call': S.submit_call()}) == 'submit'


def test_probe_summary():
    rs = [{'group': 'oracle_heldout', 'kind': 'S1', 'samples': [
        {'action': 'edit', 'resolvable': True, 'overlaps_gold': True, 'matches_oracle': True, 'think_tokens': 50,
         'at_budget': False},
        {'action': 'read', 'think_tokens': 250, 'at_budget': True}]}]
    s = P.summarize(rs)['oracle_heldout S1']
    assert s['P(edit)'] == 0.5 and s['P(edit matches oracle)'] == 0.5 and s['think at budget'] == 0.5
    assert s['states with >=1 edit'] == 1.0


# ------------------------------------------------------------------ real template + tokenizer (local .venv only)

def _real_available():
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return (ROOT / 'data' / 'rft2' / 'tokenizer_june' / 'tokenizer.json').exists() and \
        (ROOT / 'data' / 'rft3' / 'states_v20' / 'subset900_filled.jsonl').exists()


@pytest.mark.skipif(not _real_available(), reason='needs transformers + data/rft2/tokenizer_june + rft3 v20 records')
def test_prefix_consistency_real_records():
    R = B.make_renderer()
    builder = B.Builder(R)
    recs = []
    with open(ROOT / 'data' / 'rft3' / 'states_v20' / 'subset900_filled.jsonl') as f:
        for line in f:
            r = json.loads(line)
            if any(t.get('visible') for t in r['turns']) or r['source'] == 'relabel' or len(recs) < 3:
                recs.append(r)
            if len(recs) >= 8:
                break
    for rec in recs:
        row = builder.build(rec, verify=True)
        v = row['verify']
        assert v['request_prefix_of_history'] == v['requests'] == len(rec['turns'])
        assert v['generated_turn_tokens_equal'] == v['train_turns']
        assert v['train_turns_exact_context'] == v['train_turns_exact_context_expected']
        assert row['segment_tokenization_equal']
        assert sum(x != -100 for x in row['labels']) == row['n_trained'] > 0
