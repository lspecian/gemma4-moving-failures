"""Unit tests for scripts/rft3_sample.py (rft3 step B): the Gemma 4 raw call parser, sample parsing (server message
and raw text), the match rules vs the oracle call (show / search / outline / history / edit), the selection, the
printed-lines context, the grounding check of rationalized thoughts, and one record end to end on a fake server
(no model, no tokenizer: token counts are stubbed)."""

import json
import pathlib
import sys

import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
import rft3_sample as R  # noqa: E402
import rft3_states as S  # noqa: E402

Q = '<|"|>'


@pytest.fixture(autouse=True)
def stub_tokens(monkeypatch):
    monkeypatch.setattr(R, 'n_tokens', lambda t: len(t.split()) if t else 0)
    monkeypatch.setattr(R, 'AGENT_SCRIPTS', {'locate.py', 'show.py', 'search.py', 'outline.py', 'history.py',
                                             'edit.py', 'check_patch.py', 'run.py'})


def script(name, **args):
    return S.script_call(name, args)


def adk(name, out):
    return S.adk_result(name, out, '', 0)


# ------------------------------------------------------------------ raw parser

def test_parse_gemma_call_nested_and_types():
    raw = (f'call:run_skill_script{{args:{{end:{Q}888{Q},new:{Q}a = {{1: [2]}}\nb{Q},start:875}},'
           f'file_path:{Q}edit.py{Q},skill_name:{Q}repo-tools{Q}}}')
    c = R.parse_gemma_call(raw)
    assert c == {'name': 'run_skill_script', 'arguments': {
        'args': {'end': '888', 'new': 'a = {1: [2]}\nb', 'start': 875}, 'file_path': 'edit.py',
        'skill_name': 'repo-tools'}}
    assert R.parse_gemma_call('call:submit_patch{}') == {'name': 'submit_patch', 'arguments': {}}
    with pytest.raises(ValueError):
        R.parse_gemma_call(f'call:run_skill_script{{args:{{query:{Q}x}}')


def raw_turn(call_body, think='I will look.', visible=''):
    return f'<|channel>thought\n{think}<channel|>{visible}<|tool_call>{call_body}<tool_call|><|tool_response>'


SHOW = f'call:run_skill_script{{args:{{symbol:{Q}A.f{Q},target:{Q}pkg/a.py{Q}}},file_path:{Q}show.py{Q},' \
       f'skill_name:{Q}repo-tools{Q}}}'


def test_parse_raw_ok_and_problems():
    s = R.parse_raw(raw_turn(SHOW, 'The locate output names A.f.'), 'stop', 256)
    assert s['problem'] is None and s['thinking'] == 'The locate output names A.f.' and s['visible'] == ''
    assert s['call']['arguments']['args'] == {'symbol': 'A.f', 'target': 'pkg/a.py'}
    assert not s['at_budget']
    brackets = SHOW.replace(f'{Q}repo-tools{Q}', '「repo-tools」')
    assert R.parse_raw(raw_turn(brackets), 'stop')['problem'].startswith('raw: brackets')
    nested = f'call:run_skill_script{{args:{{file_path:{Q}show.py{Q},target:{Q}a.py{Q}}},skill_name:{Q}repo-tools{Q}}}'
    assert R.parse_raw(raw_turn(nested), 'stop')['problem'].startswith('raw: ')   # call_problem flags it
    assert R.parse_raw('<|channel>thought\nhm<channel|>Done.', 'stop')['problem'] == 'no_call'
    two = raw_turn(SHOW) + '<|tool_call>call:submit_patch{}<tool_call|>'
    assert R.parse_raw(two, 'stop')['problem'] == 'multi_call'
    assert R.parse_raw(raw_turn(SHOW), 'length')['problem'] == 'length'
    # a continuation after a prefilled thinking block: no thinking of its own, the call parsed
    s = R.parse_raw(f'<|tool_call>{SHOW}<tool_call|>', 'stop')
    assert s['problem'] is None and s['thinking'] == ''
    long = R.parse_raw(raw_turn(SHOW, ' '.join(['w'] * 250)), 'stop', 256)
    assert long['at_budget']


def choice(args=None, name='run_skill_script', reasoning='Think.', content=None, finish='tool_calls', raw_args=None):
    tc = [] if name is None else [{'type': 'function', 'id': 'x', 'function': {
        'name': name, 'arguments': raw_args if raw_args is not None else json.dumps(args)}}]
    return {'index': 0, 'finish_reason': finish,
            'message': {'role': 'assistant', 'content': content, 'reasoning': reasoning, 'tool_calls': tc}}


def test_parse_choice():
    ok = choice({'args': {'query': 'x'}, 'file_path': 'locate.py', 'skill_name': 'repo-tools'}, content='Locate it.')
    s = R.parse_choice(ok, 256)
    assert s['problem'] is None and s['visible'] == 'Locate it.' and s['thinking'] == 'Think.'
    assert R.parse_choice(choice({'args': {}, 'file_path': 'locate.py', 'skill_name': '「repo-tools」'})
                          )['problem'] == 'malformed'
    assert R.parse_choice(choice({'args': {'file_path': 'x.py'}, 'file_path': 'show.py', 'skill_name': 'repo-tools'})
                          )['problem'] == 'malformed'
    assert R.parse_choice(choice({'file_path': 'nope.py', 'skill_name': 'repo-tools'}))['problem'].startswith(
        'unknown script')
    assert R.parse_choice(choice({}, name='write_file'))['problem'].startswith('unknown tool')
    assert R.parse_choice(choice(raw_args='{"args": '))['problem'] == 'bad_json'
    assert R.parse_choice(choice(name=None, content='All done.'))['problem'] == 'no_call'
    assert R.parse_choice(choice({}, name='submit_patch', finish='length'))['problem'] == 'length'


# ------------------------------------------------------------------ a synthetic trajectory

FILE = 'pkg/a.py'
SHOW_OUT = (f'{FILE} A.f L10-16\n10      def f(self, x):\n11          y = x + 1\n12\n13          if y > 2:\n'
            f'14              return y\n15          return 0\n16      # end\nNEXT: edit.py ...\n')
LOCATE_OUT = (f'2.1 {FILE}\n      A.f  L10-16\n      B.g  L30-40\n--- best match: {FILE} A.f L10-16\n'
              f'  10      def f(self, x):\n       ... (L11-16)\nNEXT: show.py  target={FILE}  symbol=A.f\n')


def rec():
    turns = []
    calls = [('locate', script('locate.py', query='f returns wrong value'), LOCATE_OUT),
             ('show', script('show.py', symbol='A.f', target=FILE), SHOW_OUT),
             ('edit', script('edit.py', end='11', new='        y = x + 2', start='11', target=FILE),
              f'EDITED {FILE} (lines 11-11)\n--- a/{FILE}\n+++ b/{FILE}\n@@ -11,1 +11,1 @@\n-        y = x + 1\n'
              f'+        y = x + 2\n'),
             ('check', script('check_patch.py'), 'READY: call submit_patch now.\n'),
             ('submit', S.submit_call(), S.submit_result('diff --git a/x b/x\n'))]
    for n, (kind, c, out) in enumerate(calls):
        res = adk(S.script_of(c), out) if c['name'] == 'run_skill_script' else out
        turns.append({'id': S.call_id('t', n), 'call': c, 'result': res, 'label': 'train', 'thinking': None,
                      'visible': None, 'grounding': '', 'kind': kind})
    return {'id': 'oracle_clean_t-1', 'source': 'oracle_clean', 'task': 't-1', 'repo': 'o/t',
            'request': {'messages': [{'role': 'system', 'content': 'You fix issues. f returns wrong value.'},
                                     {'role': 'user', 'content': 'Problem: A.f returns x+1, should be x+2.'}],
                        'model': 'm', 'max_completion_tokens': 1536, 'temperature': 0.2, 'tools': [], 'top_p': 0.95,
                        'chat_template_kwargs': {'enable_thinking': True}, 'thinking_token_budget': 256},
            'turns': turns, 'gold': {'files': [FILE], 'functions': [[FILE, 'A.f']],
                                     'spans': [{'file': FILE, 'a': 11, 'b': 11}]}, 'meta': {}}


def mt(r, k, call):
    known = R.known_lines(r['turns'][:k])
    return R.match(call, r['turns'][k], r['gold'], known, 0.8)


def test_known_lines_and_edit_shift():
    r = rec()
    kn = R.known_lines(r['turns'][:2])
    assert kn[FILE][10] == '    def f(self, x):' and kn[FILE][12] == '' and kn[FILE][14] == '            return y'
    out = (f'EDITED {FILE} (lines 11-11)\n--- a/{FILE}\n+++ b/{FILE}\n@@ -11,1 +11,2 @@\n-        y = x + 1\n'
           f'+        y = x\n+        y += 2\n')
    kn = R.known_lines(r['turns'][:2] + [{'result': adk('edit.py', out)}])
    assert 11 not in kn[FILE] and kn[FILE][15] == '            return y' and kn[FILE][10].strip().startswith('def')


def test_match_locate_check_submit():
    r = rec()
    assert mt(r, 0, script('locate.py', query='other words entirely'))['ok']
    assert not mt(r, 0, script('search.py', text='f'))['ok']
    assert mt(r, 3, script('check_patch.py', clean='yes'))['ok']
    assert mt(r, 4, S.submit_call())['ok'] and mt(r, 4, S.submit_call())['identical']
    assert not mt(r, 4, {'name': 'get_status', 'arguments': {}})['ok']
    assert not mt(r, 3, script('run.py', test='tests/test_a.py'))['ok']


def test_match_show():
    r = rec()
    assert mt(r, 1, script('show.py', symbol='A.f', target=FILE))['identical']
    assert mt(r, 1, script('show.py', symbol='f', target='/workspace/' + FILE))['ok']
    assert not mt(r, 1, script('show.py', symbol='B.f', target=FILE))['ok']
    assert not mt(r, 1, script('show.py', symbol='A.f', target='pkg/b.py'))['ok']
    assert mt(r, 1, script('show.py', start='5', end='20', target=FILE))['ok']        # covers gold line 11
    assert not mt(r, 1, script('show.py', start='12', end='20', target=FILE))['ok']   # misses it
    assert mt(r, 1, script('show.py', start='25', target=FILE))['ok']                 # 5..45 as show.py reads it
    assert not mt(r, 1, script('show.py', target=FILE))['ok']                         # an outline, not a read


def test_sym_eq():
    assert R.sym_eq('A.f', 'f') and R.sym_eq('f', 'A.f') and R.sym_eq('pkg.A.f', 'A.f')
    assert not R.sym_eq('A.__init__', '__init__') and not R.sym_eq('A.f', 'B.f') and not R.sym_eq('', 'f')


def test_match_search_outline_history():
    o = {'call': script('search.py', text='retry_count'), 'result': adk('search.py', '')}
    g = {'spans': [], 'functions': []}
    assert R.match(script('search.py', text='retry_count'), o, g, {})['identical']
    assert R.match(script('search.py', regex=r'def .*retry_count\('), o, g, {})['ok']
    assert R.match(script('search.py', text='retry'), o, g, {})['ok']            # a word of the oracle identifier
    assert not R.match(script('search.py', text='retry_counter'), o, g, {})['ok']
    assert not R.match(script('search.py', text='ret'), o, g, {})['ok']          # too short
    o = {'call': script('outline.py', files=FILE), 'result': adk('outline.py', '')}
    assert R.match(script('outline.py', files=f'{FILE}, pkg/b.py'), o, g, {})['ok']
    assert not R.match(script('outline.py', files='pkg/b.py'), o, g, {})['ok']
    o = {'call': script('history.py', similar='f returns wrong value'), 'result': adk('history.py', '')}
    assert R.match(script('history.py', similar='wrong value of f'), o, g, {})['ok']
    assert not R.match(script('history.py', target=FILE), o, g, {})['ok']


def test_match_edit():
    r = rec()
    assert mt(r, 2, r['turns'][2]['call'])['identical']
    x = mt(r, 2, script('edit.py', start='11', end='11', new='    y = x + 2\n', target=FILE))   # indentation off
    assert x['ok'] and x['why'] == 'same result'
    x = mt(r, 2, script('edit.py', start='11', new='        y = x + 2', target=FILE))            # start alone
    assert x['ok']
    x = mt(r, 2, script('edit.py', old='y = x + 1', new='y = x + 2', target=FILE))               # old= form
    assert x['ok'] and x['note'] == 'old'
    x = mt(r, 2, script('edit.py', start='10', end='15', target=FILE,                          # wider, same change
                        new='    def f(self, x):\n        y = x + 2\n\n        if y > 2:\n            return y\n'
                            '        return 0'))
    assert x['ok'] and x['sim'] == 1.0
    x = mt(r, 2, script('edit.py', start='11', end='11', new='        y = x + 3', target=FILE))  # another fix
    assert not x['ok'] and x['sim'] == 0.5 and x['tsim'] < 1   # the same line removed, another line added
    assert mt(r, 2, script('edit.py', start='15', end='15', new='        return 1', target=FILE))['why'] == \
        'other lines'
    assert mt(r, 2, script('edit.py', start='11', end='20', new='        y = x + 2', target=FILE))['why'] == \
        'unseen lines'
    assert mt(r, 2, script('edit.py', old='z = 1', new='z = 2', target=FILE))['why'] == \
        'old not found in printed lines'
    assert not mt(r, 2, script('edit.py', start='11', end='11', new='y = x + 2', target='pkg/b.py'))['ok']


def test_change_f1_threshold():
    base = ['a = 1', 'b = 2', 'c = 3', 'd = 4', 'e = 5']
    o = ['a = 10', 'b = 20', 'c = 30', 'd = 40', 'e = 50']
    s = ['a = 10', 'b = 20', 'c = 30', 'd = 40', 'e = 5']   # 4 of 5 changed lines
    f1 = R.change_f1(R.line_change_counter(base, o), R.line_change_counter(base, s))
    assert 0.8 < f1 < 1
    assert R.change_f1(R.line_change_counter(base, base), R.line_change_counter(base, base)) == 1.0


def test_choose():
    s = [{'at_budget': True, 'visible': '', 'think_tokens': 10},
         {'at_budget': False, 'visible': '', 'think_tokens': 50},
         {'at_budget': False, 'visible': 'One sentence.', 'think_tokens': 90},
         {'at_budget': False, 'visible': '', 'think_tokens': 20}]
    ok = [{'ok': True}] * 3 + [{'ok': False}]
    assert R.choose(s, ok) == 2
    assert R.choose(s, [{'ok': True}, {'ok': True}, {'ok': False}, {'ok': True}]) == 3
    assert R.choose(s, [{'ok': True}, {'ok': False}, {'ok': False}, {'ok': False}]) == 0
    assert R.choose(s, [{'ok': True}, {'ok': True}, {'ok': False}, {'ok': True}], require_visible=True) is None


def test_grounding():
    r = rec()
    ctx = R.context_text(S.messages_at(r, 2))
    assert R.ungrounded('The show output of pkg/a.py shows `A.f` adds 1 at line 11; it should add 2.', ctx) == []
    assert R.ungrounded('I will patch `compute_total` in pkg/billing.py near line 77.', ctx) == \
        ['77', 'compute_total', 'pkg/billing.py']
    assert R.ungrounded('Use `len` and `self`, and/or check e.g. the return value.', ctx) == []
    assert R.ungrounded('Set `y_new` here.', ctx, allowed='y_new = x + 2') == []
    assert R.LEAK.search('As instructed, I call show.py') and R.LEAK.search('The next call is edit.py')
    assert not R.LEAK.search('The locate output lists A.f, so I will read it.')


def test_parse_rationale():
    t, s = R.parse_rationale('THINKING: The show output has y = x + 1.\nSENTENCE: Fix the increment.')
    assert t == 'The show output has y = x + 1.' and s == 'Fix the increment.'
    assert R.parse_rationale('no form') == (None, None)


def test_call_text():
    assert R.call_text(script('show.py', symbol='A.f', target=FILE)) == f'show.py  symbol=A.f  target={FILE}'
    assert 'new=<new below>' in R.call_text(script('edit.py', new='a\nb', start='1', target=FILE))


# ------------------------------------------------------------------ end to end on a fake server

class FakeServer:
    kind, model, models = 'fake', 'm', ['m']

    def __init__(self):
        self.bodies = []

    def chat(self, body, n, seed=None):
        self.bodies.append(body)
        msgs = body['messages']
        if msgs[-1]['role'] == 'user' and 'data-generation request' in msgs[-1]['content']:
            ch = choice(name=None, reasoning=None,
                        content='THINKING: The show output of pkg/a.py has y = x + 1 at line 11; it must add 2.\n'
                                'SENTENCE: Fix the increment.')
            return [ch] * n, {'completion_tokens': 30}, 1
        k = (len(msgs) - 2) // 2
        if k == 0:
            c = [choice({'args': {'query': 'f wrong'}, 'file_path': 'locate.py', 'skill_name': 'repo-tools'},
                        reasoning=f'Locate first {i}.' + ' x' * (5 - i)) for i in range(n)]
        elif k == 1:
            c = [choice({'args': {'symbol': 'A.f', 'target': FILE}, 'file_path': 'show.py', 'skill_name': 'repo-tools'},
                        reasoning='Read A.f.')] * n
        elif k == 2:  # never the right edit: fallback
            c = [choice({'args': {'end': '11', 'new': '        y = x', 'start': '11', 'target': FILE},
                         'file_path': 'edit.py', 'skill_name': 'repo-tools'}, reasoning='Hmm.')] * n
        elif k == 3:
            c = [choice({'file_path': 'check_patch.py', 'skill_name': 'repo-tools'}, reasoning='Check.')] * n
        else:
            c = [choice({}, name='submit_patch', reasoning='Submit.')] * n
        return [json.loads(json.dumps(x)) for x in c], {'completion_tokens': 10 * n}, 1

    def complete(self, ids, max_tokens, temperature, top_p, n=1, seed=None):
        body = (f'call:run_skill_script{{args:{{end:{Q}11{Q},new:{Q}        y = x + 2{Q},start:{Q}11{Q},'
                f'target:{Q}{FILE}{Q}}},file_path:{Q}edit.py{Q},skill_name:{Q}repo-tools{Q}}}')
        return [(f'<|tool_call>{body}<tool_call|>', 'stop')] * n, {'completion_tokens': 40}


def test_record_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'prefill_ids', lambda msgs, tools, thinking: [1, 2, 3])
    r = rec()
    st = R.Store(str(tmp_path))
    cfg = R.Cfg(k=3, k_kind={'submit': 1}, temperature=None, token_ids=False, keep_raw=False, edit_threshold=0.8,
                require_visible=False, seed=None, fallback=True, fb_temperature=0.7, fb_n=1, fb_verify_max=2,
                verify_n=1)
    srv = FakeServer()
    stats = __import__('collections').Counter()
    R.run_record(r, st, cfg, srv, stats)
    kept = {k: st.get(r['id'], k, 'sample')['kept'] for k in range(5)}
    assert kept[0]['thinking'] == 'Locate first 2. x x x' or kept[0]['think_tokens'] == min(
        s['think_tokens'] for s in st.get(r['id'], 0, 'sample')['samples'])
    assert kept[0]['call_from'] == 'oracle'      # the query differs: the oracle call (its result) is kept
    assert kept[1]['call_from'] == 'sample'      # identical show call
    assert kept[2] is None
    fb = st.get(r['id'], 2, 'fallback')
    assert fb['label'] == 'rationalized' and fb['kept']['thinking'].startswith('The show output')
    assert len(st.get(r['id'], 4, 'sample')['samples']) == 1
    # the context of later turns carries the kept thinking (turn 3 sees turn 2's rationalized thinking)
    body3 = [b for b in srv.bodies if len(b['messages']) == 8 and b['messages'][-1]['role'] == 'tool'][0]
    assert body3['messages'][6].get('reasoning') == fb['kept']['thinking']
    assert body3['messages'][2].get('reasoning') == kept[0]['thinking']
    assert st.get(r['id'], 3, 'sample')['ctx'] == ['bootstrap', 'bootstrap', 'rationalized']
    # resume: a second run sends nothing
    srv2 = FakeServer()
    R.run_record(r, R.Store(str(tmp_path)), cfg, srv2, stats)
    assert srv2.bodies == []


# ------------------------------------------------------------------ v20: multi-edit calls, carried turns

def multi_rec():
    """rec() with the edit turn a v20 multi-edit of lines 11 and 14 (both printed by the show turn)."""
    r = rec()
    r['turns'][2]['call'] = script('edit.py', end='11', new='        y = x + 2', start='11', end2='14',
                                   new2='            return y * 2', start2='14', target=FILE)
    r['turns'][2]['result'] = adk('edit.py', f'NOTE: something inferred\nEDITED {FILE} (2 hunks: lines 11-11; lines '
                                  f'14-14)\n--- a/{FILE}\n+++ b/{FILE}\n')
    r['gold']['spans'].append({'file': FILE, 'a': 14, 'b': 14})
    return r


def test_edit_hunks_and_multi_match():
    r = multi_rec()
    kn = R.known_lines(r['turns'][:2])[FILE]
    hunks, note = R.edit_hunks(R.args_of(r['turns'][2]['call']), kn)
    assert [(a, b) for a, b, _ in hunks] == [(11, 11), (14, 14)] and note == ''
    assert R.edit_hunks({'start': '11', 'end': '13', 'new': 'x', 'start2': '12', 'end2': '14', 'new2': 'y'},
                        kn) == (None, 'overlapping hunks')
    assert mt(r, 2, r['turns'][2]['call'])['ok']
    # the same change as one wide line-range edit, or with the hunks in the other order: the same result
    wide = script('edit.py', start='11', end='14', target=FILE,
                  new='        y = x + 2\n\n        if y > 2:\n            return y * 2')
    assert mt(r, 2, wide)['ok']
    swapped = script('edit.py', start='14', end='14', new='            return y * 2', start2='11', end2='11',
                     new2='        y = x + 2', target=FILE)
    assert mt(r, 2, swapped)['ok']
    # only one of the two hunks (lines 11 / 14 are 3 lines apart: separate windows)
    one = script('edit.py', end='11', new='        y = x + 2', start='11', target=FILE)
    x = mt(r, 2, one)
    assert not x['ok'] and x['sim'] < 0.8
    # a single-hunk oracle is matched as before
    assert mt(rec(), 2, script('edit.py', start='11', new='        y = x + 2', target=FILE))['ok']


def test_known_lines_after_a_multi_edit_and_notes_first():
    r = multi_rec()
    kn = R.known_lines(r['turns'][:3])
    assert set(kn[FILE]) == {10}  # every line from the first hunk on is dropped (NOTE lines before EDITED)
    out = (f'NOTE: the end was inferred\nEDITED {FILE} (lines 11-11)\n--- a/{FILE}\n+++ b/{FILE}\n'
           f'@@ -11,1 +11,2 @@\n-        y = x + 1\n+        y = x\n+        y += 2\n')
    kn = R.known_lines(r['turns'][:2] + [{'result': adk('edit.py', out)}])
    assert 11 not in kn[FILE] and kn[FILE][15] == '            return y'


def test_carried_turns_are_not_sampled(monkeypatch, tmp_path):
    r = rec()
    for k in (0, 1):
        r['turns'][k].update(thinking=f'carried {k}', sampling={'from': 'bootstrap', 'carried': 'v19'})
    asked = []
    monkeypatch.setattr(R, 'sample_turn', lambda rec_, turns, k, cfg, srv: asked.append(
        (k, [t.get('thinking') for t in turns[:k]])) or {'kept': None})
    st = type('St', (), {'get': lambda self, *a: None, 'put': lambda self, d: None})()
    cfg = type('C', (), {'fallback': False})()
    import collections
    R.run_record(r, st, cfg, None, collections.Counter())
    assert [k for k, _ in asked] == [2, 3, 4]
    assert asked[0][1] == ['carried 0', 'carried 1']  # the carried thinking is in the context of later turns
