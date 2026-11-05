"""Unit tests for scripts/next_step_probe.py: call classification (as ADK would handle a sampled call), state selection
from traces, balanced sampling and the paired statistics. No network, no sandbox, no traces needed."""

import json
import pathlib
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
import next_step_probe as nsp  # noqa: E402

V17 = {'locate.py', 'show.py', 'outline.py', 'search.py', 'callers.py', 'tests_for.py', 'history.py', 'edit.py',
       'create.py', 'run.py', 'check_patch.py', 'status.py', 'hypothesis.py', 'flow.py', 'argfix.py'}
V18 = V17 | {'repro.py', 'reproduce_issue.py', 'reproduce_bug.py'}


def rss(fp, args=None, skill='repo-tools', **extra):
    d = {'skill_name': skill, 'file_path': fp}
    if args is not None:
        d['args'] = args
    d.update(extra)
    return d


# ---------------------------------------------------------------- classify_call

def test_reads_searches_locate_check():
    assert nsp.classify_call('run_skill_script', rss('show.py', {'target': 'a.py', 'start': 1}), V17) == \
        ('read', 'show.py')
    for s in ('outline.py', 'callers.py', 'tests_for.py', 'history.py'):
        assert nsp.classify_call('run_skill_script', rss(s, {'target': 'a.py'}), V17)[0] == 'read'
    assert nsp.classify_call('run_skill_script', rss('search.py', {'text': 'x'}), V17)[0] == 'search'
    assert nsp.classify_call('run_skill_script', rss('locate.py', {'query': 'q'}), V17)[0] == 'locate'
    assert nsp.classify_call('run_skill_script', rss('check_patch.py', {}), V17)[0] == 'check'
    assert nsp.classify_call('run_skill_script', rss('scripts/show.py', {'target': 'a.py'}), V17)[0] == 'read'
    assert nsp.classify_call('run_skill_script', rss('status.py'), V17)[0] == 'other'


def test_edit_shapes():
    ok = [{'target': 'a.py', 'start': 3, 'end': 4, 'new': 'x'}, {'target': 'a.py', 'old': 'a', 'new': 'b'},
          {'old': 'a', 'new': 'b'}]   # v18 infers a missing target
    for a in ok:
        assert nsp.classify_call('run_skill_script', rss('edit.py', a), V17) == ('edit', 'edit.py')
    assert nsp.classify_call('run_skill_script', rss('edit.py', {'target': 'a.py', 'start': 3}), V17)[0] == \
        'edit_badshape'
    assert nsp.classify_call('run_skill_script', rss('edit.py', {'target': 'a.py', 'new': 'x'}), V17)[0] == \
        'edit_badshape'
    assert nsp.classify_call('run_skill_script', rss('edit.py', positional_args=['a.py', 'old', 'new']), V17)[0] == \
        'edit'


def test_create_real_file_is_edit_scratch_is_snippet():
    assert nsp.classify_call('run_skill_script', rss('create.py', {'target': 'pkg/new_mod.py', 'content': 'x = 1'}),
                             V17)[0] == 'edit'
    for t in ('repro.py', 'reproduce_issue.py', 'debug_x.py', 'tests/test_repro_x.py', 'scratch.py'):
        assert nsp.classify_call('run_skill_script', rss('create.py', {'target': t, 'content': 'print(1)'}),
                                 V17)[0] == 'snippet', t
    assert nsp.classify_call('run_skill_script', rss('create.py', {'target': 'pkg/x.py'}), V17)[0] == 'edit_badshape'
    # content without a target: v18 runs it as a snippet, v17b rejects it; the intent is a run either way
    assert nsp.classify_call('run_skill_script', rss('create.py', {'content': 'print(1)'}), V17)[0] == 'snippet'


def test_reclassify_stored_samples():
    long = 'x' * 1000
    fargs = nsp.shorten(rss('create.py', {'content': long, 'target': 'pkg/mod.py'}))
    assert len(fargs['args']['content']) == 300 and fargs['args']['target'] == 'pkg/mod.py'
    s = [{'variant': 'A', 'name': 'run_skill_script', 'fargs': fargs, 'label': 'edit_badshape', 'intent': 'create.py'},
         {'variant': 'A', 'label': 'edit_badshape', 'intent': 'create.py', 'call': '{"args": {"content": "x"}}'}]
    assert [x['label'] for x in nsp.reclassify(s, {'A': V17})] == ['edit', 'snippet']


def test_run_snippet_test_and_aliases():
    assert nsp.classify_call('run_skill_script', rss('run.py', {'snippet': 'print(1)'}), V17)[0] == 'snippet'
    assert nsp.classify_call('run_skill_script', rss('run.py', {'test': 'tests/test_a.py', 'k': 'x'}), V17)[0] == 'test'
    # repro.py is an alias script in v18, an unknown script (ADK SCRIPT_NOT_FOUND) in v17b
    assert nsp.classify_call('run_skill_script', rss('repro.py', {'snippet': 'x'}), V18) == ('snippet', 'repro.py')
    assert nsp.classify_call('run_skill_script', rss('repro.py', {'snippet': 'x'}), V17) == ('malformed', 'repro.py')


def test_malformed_shapes_seen_in_traces():
    # delimiter leaked into file_path, skill_name swallowed
    assert nsp.classify_call('run_skill_script', {'args': {'regex': 'x'}, 'file_path': 'search.py`,skill_name:'},
                             V17) == ('malformed', 'search.py')
    # CJK brackets around both names
    assert nsp.classify_call('run_skill_script', rss('「show.py」', {'target': 'a.py'},
                                                     skill='「repo-tools」'), V17) == ('malformed', 'show.py')
    # skill_name/file_path nested into args
    assert nsp.classify_call('run_skill_script', {'args': {'file_path': 'outline.py', 'skill_name': 'repo-tools'},
                                                  'positional_args': ['a.py']}, V17) == ('malformed', 'outline.py')
    # a repository file as the script, an unknown tool name, an edit that the parser mangled
    assert nsp.classify_call('run_skill_script', rss('rich/cells.py', {'start': 1}), V17)[0] == 'malformed'
    assert nsp.classify_call('show.py', {'target': 'a.py'}, V17) == ('malformed', 'show.py')
    assert nsp.classify_call('run_skill_script', {'args': {'new': 'x', 'start': 3}, 'file_path': 'edit.py`,skill_name:'},
                             V17) == ('malformed', 'edit.py')
    assert nsp.classify_call('run_skill_script', rss('show.py', 'a.py'), V17)[0] == 'malformed'   # args a string


def test_other_tools():
    assert nsp.classify_call('submit_patch', {}, V17) == ('submit', None)
    assert nsp.classify_call('get_status', {}, V17) == ('other', None)
    assert nsp.classify_call('load_skill', {'skill_name': 'repo-tools'}, V17) == ('other', None)


def test_adk_error_texts():
    assert nsp.adk_error(rss('show.py', {}), V17) is None
    assert nsp.adk_error({'file_path': 'show.py'}, V17) == {'error': "Argument 'skill_name' is required.",
                                                           'error_code': 'INVALID_ARGUMENTS'}
    assert nsp.adk_error(rss('nope.py'), V17) == {'error': "Script 'nope.py' not found in skill 'repo-tools'.",
                                                  'error_code': 'SCRIPT_NOT_FOUND'}
    assert nsp.adk_error(rss('show.py', skill='x'), V17)['error_code'] == 'SKILL_NOT_FOUND'
    assert nsp.adk_error(rss('show.py', ['a'], positional_args=['b']), V17)['error_code'] == 'INVALID_ARGUMENTS'


# ---------------------------------------------------------------- classify_response

def resp(name=None, args=None, content=''):
    if name is None:
        return {'content': content}
    return {'content': content, 'tool_calls': [{'function': {'name': name, 'arguments': args}}]}


def test_classify_response():
    m = resp('run_skill_script', json.dumps(rss('edit.py', {'target': 'a.py', 'start': 1, 'new': 'x'})))
    assert nsp.classify_response(m, V17)[:2] == ('edit', 'edit.py')
    assert nsp.classify_response(resp(content='I am done.'), V17)[0] == 'no_call'
    raw = '<|tool_call>call:run_skill_script{file_path:<|"|>show.py<|"|>,args:[...]}<tool_call|>'
    assert nsp.classify_response(resp(content=raw), V17)[0] == 'malformed'
    assert nsp.classify_response(resp('run_skill_script', '{not json'), V17)[0] == 'malformed'
    assert nsp.classify_response(resp('run_skill_script', rss('show.py', {'target': 'a.py'})), V17)[0] == 'read'


# ---------------------------------------------------------------- states

def step(fp, args, pt, tc_name='run_skill_script'):
    return {'source': 'agent', 'message': 'thinking', 'metrics': {'prompt_tokens': pt},
            'tool_calls': [{'function_name': tc_name, 'arguments': rss(fp, args) if tc_name == 'run_skill_script'
                            else {}}],
            'observation': {'content': '{}'}}


HEAD = [{'source': 'system', 'message': 'sys'}, {'source': 'user', 'message': 'issue'}]


def test_select_states_stops_at_first_edit():
    steps = HEAD + [step('locate.py', {'query': 'q'}, 5000), step('show.py', {'target': 'a.py'}, 7000),
                    step('search.py', {'text': 'x'}, 8000), step('edit.py', {'target': 'a.py', 'old': 'a', 'new': 'b'},
                                                                 9000),
                    step('run.py', {'snippet': 'x'}, 9500)]
    st = nsp.select_states(steps, V17)
    assert [(s['k'], s['step'], s['next']) for s in st] == [(1, 3, 'read'), (2, 4, 'search'), (3, 5, 'edit')]


def test_select_states_prompt_cap_and_submit():
    steps = HEAD + [step('locate.py', {'query': 'q'}, 5000), step('show.py', {'target': 'a.py'}, 12000),
                    step('show.py', {'target': 'b.py'}, 14500), step('show.py', {'target': 'c.py'}, 6000)]
    assert [s['k'] for s in nsp.select_states(steps, V17, max_prompt=14000)] == [1]   # 14500: a compaction follows
    steps = HEAD + [step('locate.py', {'query': 'q'}, 5000), step(None, None, 6000, tc_name='submit_patch'),
                    step('show.py', {'target': 'a.py'}, 7000)]
    assert [(s['k'], s['next']) for s in nsp.select_states(steps, V17)] == [(1, 'submit')]


def test_select_states_malformed_edit_counts_as_edit_attempt():
    bad = step('edit.py', {'old': 'a', 'new': 'b'}, 7000)
    bad['tool_calls'][0]['arguments'] = {'args': {'old': 'a', 'new': 'b'}, 'file_path': 'edit.py`,skill_name:'}
    steps = HEAD + [step('locate.py', {'query': 'q'}, 5000), bad, step('show.py', {'target': 'a.py'}, 8000)]
    assert [(s['k'], s['next']) for s in nsp.select_states(steps, V17)] == [(1, 'malformed')]


def cands(n_tasks=20, per=6, edit_every=3):
    out = []
    for t in range(n_tasks):
        for k in range(1, per + 1):
            out.append({'task': f't{t}', 'trace': f'src{t % 2}/t{t}', 'step': k + 1, 'k': k,
                        'source': f'src{t % 2}', 'next': 'edit' if k % edit_every == 0 else 'read'})
    return out


def test_select_balanced():
    c = cands()
    p = nsp.select_balanced(c, 12, per_task=1, seed=0, edit_share=0.25)
    assert len(p) == 12
    assert len({x['task'] for x in p}) == 12                     # one state per task in the first pass
    assert sum(1 for x in p if x['next'] == 'edit') == 3         # edit_share
    assert {x['source'] for x in p} == {'src0', 'src1'}
    assert abs(sum(1 for x in p if x['source'] == 'src0') - 6) <= 1
    assert p == nsp.select_balanced(cands(), 12, per_task=1, seed=0, edit_share=0.25)   # deterministic
    p = nsp.select_balanced(cands(n_tasks=3), 6, per_task=1)       # second pass: up to 2 per task
    assert len(p) == 6 and max(sum(1 for x in p if x['task'] == t) for t in ('t0', 't1', 't2')) == 2
    c = cands(edit_every=99)
    assert {nsp.k_bucket(x['k']) for x in nsp.select_balanced(c, 40, per_task=2)} == {'1', '2', '3', '4', '5-6'}


# ---------------------------------------------------------------- statistics / report

def test_sign_test_and_bootstrap():
    assert nsp.sign_test(0, 0) == 1.0
    assert abs(nsp.sign_test(6, 0) - 2 / 64) < 1e-12
    assert abs(nsp.sign_test(3, 3) - 1.0) < 1e-12
    lo, hi = nsp.bootstrap_ci([0.5] * 10)
    assert lo == hi == 0.5
    lo, hi = nsp.bootstrap_ci([0.0, 1.0] * 20)
    assert lo < 0.5 < hi


def test_report_pairs():
    samples = []
    for st in range(6):
        for v, lab in (('A', 'edit'), ('B', 'read')):
            for seed in range(2):
                samples.append({'variant': v, 'trace': f't{st}', 'step': 3, 'k': 2, 'source': 's', 'seed': seed,
                                'label': lab, 'tokens': 100, 'think_chars': 10, 'prompt_tokens': 5000})
    text, rep = nsp.report(samples, ['A', 'B'])
    assert rep['variants']['A']['p_edit'] == 1.0 and rep['variants']['B']['p_edit'] == 0.0
    pair = next(p for p in rep['pairs'] if p['metric'] == 'edit')
    assert pair['mean_diff'] == 1.0 and pair['a_higher'] == 6 and abs(pair['sign_p'] - 2 / 64) < 1e-12
    assert 'P(edit) by k' in text


def test_variant_system_prompt(tmp_path):
    d = tmp_path / 'agent'
    (d / 'prompts').mkdir(parents=True)
    (d / 'skills/repo-tools/scripts').mkdir(parents=True)
    (d / 'configs').mkdir()
    (d / 'prompts/system.md').write_text('Fix it.\n## The issue\n{problem_description}')
    (d / 'agent.yaml').write_text('name: swe_agent_v99\ndescription: Narrow agent.\n')
    (d / 'configs/sampling.yaml').write_text('temperature: 0.3\nthinking_config:\n  thinking_budget: 256\n')
    (d / 'skills/repo-tools/scripts/show.py').write_text('print(1)\n')
    v = nsp.Variant(d)
    suffix = '\n\nYou are an agent. Your internal name is "swe_agent_v10". The description about you is "Old.".\n\nMore.'
    s = v.system('  The bug.  ', suffix)
    assert s.startswith('Fix it.\n## The issue\nThe bug.\n\nYou are an agent.')
    assert '"swe_agent_v99"' in s and '"Narrow agent.".' in s and 'Old.' not in s
    assert v.sampling['temperature'] == 0.3 and v.sampling['thinking_budget'] == 256
    assert v.scripts == {'show.py'}
