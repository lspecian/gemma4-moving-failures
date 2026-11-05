"""scripts/replay_tools.py: replay one synthetic recorded trajectory through the current agent/ scripts.

A tiny repository snapshot, a results dir in the harness layout (task_results.jsonl, traces/, patches/) and a trace
with a locate, an ADK-rejected call (wrong skill name), a line-range edit, a run.py snippet and submit_patch.
"""

import json
import os
import pathlib
import subprocess
import sys
import tarfile
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.dont_write_bytecode = True
import replay_tools as rt  # noqa: E402

INVOICE = textwrap.dedent('''\
    """Payment helpers."""


    class Invoice:
        def __init__(self, amount):
            self.amount = amount

        def total(self, tax=0.1):
            if self.amount == None:
                return 0
            return self.amount * (1 + tax)
    ''')
T0 = 1_800_000_000.0


def _git(ws, *a):
    subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', *a], check=True,
                   capture_output=True)


def _step(i, script, args, content, t, end, skill='repo-tools'):
    return {'step_id': i, 'source': 'agent', 'message': 'thinking',
            'tool_calls': [{'tool_call_id': f'call_{i}', 'function_name': 'run_skill_script' if script else 'submit_patch',
                            'arguments': {'args': args, 'file_path': script, 'skill_name': skill} if script else {},
                            'extra': {'timestamp': T0 + t, 'elapsed_s': t}}],
            'observation': {'content': content, 'extra': repr({'timestamp': T0 + end, 'elapsed_s': end})},
            'metrics': {'prompt_tokens': 4000 + 1000 * i}}


def _result(script, stdout):
    return json.dumps({'skill_name': 'repo-tools', 'file_path': script, 'stdout': stdout, 'stderr': '',
                       'status': 'success'})


@pytest.fixture
def recorded(tmp_path):
    """(results dir, snapshots dir, the expected patch) for task tiny_1."""
    src = tmp_path / 'src'
    (src / 'pay').mkdir(parents=True)
    (src / 'pay/__init__.py').write_text('')
    (src / 'pay/invoice.py').write_text(INVOICE)
    (src / 'tests').mkdir()
    (src / 'tests/test_invoice.py').write_text('from pay.invoice import Invoice\n\n\ndef test_total():\n'
                                               '    assert Invoice(10).total(0) == 10\n')
    snaps = tmp_path / 'snapshots'
    snaps.mkdir()
    with tarfile.open(snaps / 'tiny_1.tgz', 'w:gz') as t:
        t.add(src, arcname='.')
    # the expected patch: the same edit made by hand, diffed like the harness
    ref = tmp_path / 'ref'
    subprocess.run(['cp', '-R', str(src), str(ref)], check=True)
    _git(ref, 'init', '-q')
    _git(ref, 'add', '-A')
    _git(ref, 'commit', '-qm', 'baseline')
    (ref / 'pay/invoice.py').write_text(INVOICE.replace('== None', 'is None'))
    _git(ref, 'add', '-N', '.')
    patch = subprocess.run(['git', '-C', str(ref), 'diff', '--binary', 'HEAD'], capture_output=True,
                           text=True).stdout
    assert patch
    run = tmp_path / 'results/E999_test/r1/c0'
    (run / 'traces').mkdir(parents=True)
    (run / 'patches').mkdir()
    (run / 'task_results.jsonl').write_text(json.dumps({'instance_id': 'tiny_1', 'resolved': True}) + '\n')
    (run / 'patches/tiny_1.patch').write_text(patch)
    line = INVOICE.splitlines().index('        if self.amount == None:') + 1
    steps = [{'step_id': 1, 'source': 'system', 'message': 'sys'}, {'step_id': 2, 'source': 'user', 'message': 'u'},
             _step(3, 'locate.py', {'query': 'Invoice.total returns wrong value when amount is None'},
                   _result('locate.py', 'pay/invoice.py ...'), 10, 12),
             _step(4, 'show.py', {'target': 'pay/invoice.py'},
                   json.dumps({'error': "Skill '「repo-tools」' not found.", 'error_code': 'SKILL_NOT_FOUND'}), 30, 30,
                   skill='「repo-tools」'),
             _step(5, 'edit.py', {'target': 'pay/invoice.py', 'start': line, 'end': line,
                                  'new': '        if self.amount is None:'},
                   _result('edit.py', f'EDITED pay/invoice.py (lines {line}-{line})\n--- a/pay/invoice.py\n'), 100, 107),
             _step(6, 'run.py', {'snippet': 'print(6 * 7)'}, _result('run.py', '42\nexit code: 0\n'), 120, 122),
             _step(7, None, None, json.dumps({'status': 'ok', 'patch_size': len(patch), 'files_changed': 1}),
                   130, 130)]
    (run / 'traces/trace_tiny_1.json').write_text(json.dumps({'schema_version': 'ATIF-v1.7', 'steps': steps}))
    return tmp_path / 'results/E999_test', snaps, patch


def _replay(results, snaps, tmp_path):
    runs = rt.load_runs([str(results)], passing=True)
    assert [r['task'] for r in runs] == ['tiny_1']
    agent = rt.agent_record('worktree', str(ROOT / 'agent'))
    opt = {'snapshots': str(snaps), 'workroot': str(tmp_path), 'env': 'none'}
    task, recs, _ = rt.work_task(('tiny_1', runs, [agent, dict(agent, label='again')], opt))
    return recs


def test_replay_reproduces_the_recorded_patch(recorded, tmp_path):
    results, snaps, patch = recorded
    recs = _replay(results, snaps, tmp_path)
    assert len(recs) == 2 and not any(r.get('error') for r in recs), [r.get('error') for r in recs]
    rec = recs[0]
    assert rec['diff'] == 'identical', rec['final_patch']
    kinds = [c['new_kind'] for c in rec['calls']]
    assert kinds == ['ran', 'rejected', 'ran', 'ran', 'tool']
    assert rec['calls'][1]['error_code'] == 'SKILL_NOT_FOUND'
    edit = rec['calls'][2]
    assert edit['edit']['verdict'] == 'same', edit
    assert not any(c.get('crash') or c.get('timeout') for c in rec['calls'])
    assert rec['calls'][3]['env_dependent'] and rec['calls'][3]['run_agree'] is True
    # the flow clock follows the recording: first action at 12 s, the edit's result at 107 s -> 1:35 elapsed
    # (v18c: v17b's BUDGET line starts at call 3, the run at 122 s -> 1m50s elapsed)
    assert any(x in edit['out'] for x in ('1:34/', '1:35/', '1:36/', '1m34s', '1m35s', '1m36s')) or \
        any(x in rec['calls'][3]['out'] for x in ('1m49s', '1m50s', '1m51s')), (edit['out'], rec['calls'][3]['out'])
    # actual crossing from prompt_tokens (4000 + 1000*step >= 14336 never happens here)
    assert rec['actual_cross'] is None
    # the second agent (same scripts) matches the reference call by call
    assert recs[1]['diff_vs_ref'] == 'identical' and recs[1]['first_divergence'] is None
    # summary renders
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'runs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in recs))
    text = rt.summarize(str(out))
    assert 'still applied, same range 1' in text and 'identical 1' in text


def test_replay_flags_a_different_patch(recorded, tmp_path):
    results, snaps, patch = recorded
    (results / 'r1/c0/patches/tiny_1.patch').write_text(patch.replace('is None', 'is  None'))
    rec = _replay(results, snaps, tmp_path)[0]
    assert rec['diff'] == 'different'
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'runs.jsonl').write_text(json.dumps(rec) + '\n')
    assert 'E999_test/r1/c0/tiny_1: diff different' in rt.summarize(str(out))


def test_adk_check_and_argv():
    scripts = str(ROOT / 'agent/skills/repo-tools/scripts')
    assert rt.adk_check({'skill_name': 'repo-tools', 'file_path': 'show.py', 'args': {}}, 'repo-tools', scripts) is None
    assert rt.adk_check({'skill_name': 'repo-tools', 'file_path': 'scripts/show.py'}, 'repo-tools', scripts) is None
    assert rt.adk_check({'file_path': 'show.py'}, 'repo-tools', scripts)['error_code'] == 'INVALID_ARGUMENTS'
    assert rt.adk_check({'skill_name': 'repo-tools', 'file_path': 'rich/cells.py'}, 'repo-tools',
                        scripts)['error_code'] == 'SCRIPT_NOT_FOUND'
    assert rt.adk_check({'skill_name': 'repo-tools', 'file_path': 'show.py', 'args': 'x'}, 'repo-tools',
                        scripts)['error_code'] == 'INVALID_ARGUMENTS'
    # ADK stringifies values with str(): 12 -> "12", True -> "True"
    assert rt.script_argv('show.py', {'args': {'start': 12, 'all': True}}) == \
        ['scripts/show.py', '--start', '12', '--all', 'True']


def test_crossing_estimate():
    # base 14000: the 2nd model call's prompt (one 1,000-char call before it) is 14000 + 295 + 336 >= 14336
    assert rt.est_crossing([1000, 1000, 1000], 14000) == 2
    assert rt.est_crossing([None, None], 14000) is None    # rejected calls are absorbed in the per-call term
    assert rt.est_crossing([10, 10], 5000) is None
