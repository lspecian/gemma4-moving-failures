"""Unit tests for scripts/rft3_oracle.py's grounding rules (pure parts; a tiny temp workspace, no scripts run):
a method counts as named only with its class, a bare dunder never; search identifiers come from real code (not
docstrings or comments) and never from the project's own name; per-file attribution of names and line references."""

import os
import pathlib
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
import rft3_oracle as O  # noqa: E402

TRAINER = '''class Trainer:
    def __init__(self, gpus=None):
        """Set up. See https://github.com/PyTorchLightning/pytorch-lightning and data_parallel_device_ids."""
        # CUDA_VISIBLE_DEVICES is read elsewhere
        self.gpus = gpus
        self.root_gpu = pick_gpu(gpus)

    def fit(self, model):
        return model
'''


def builder(tmp_path, outs, ps='', repo='Lightning-AI/lightning', files=None):
    ws = tmp_path / 'ws'
    for rel, text in (files or {'pl/trainer.py': TRAINER, 'pl/other.py': 'def helper():\n    pass\n'}).items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    b = object.__new__(O.Builder)
    b.ws, b.ps, b.outs, b.turns = str(ws), ps, list(outs), [{'kind': 'locate', 'call': {}} for _ in outs]
    b.task = {'repo': repo, 'problem_statement': ps}
    b.base = {rel: text for rel, text in (files or {'pl/trainer.py': TRAINER}).items()}
    span = {'file': 'pl/trainer.py', 'a': 6, 'b': 6, 'def': [2, 6, 'Trainer.__init__', 2]}
    b.gold = {'files': list(b.base), 'spans': [span], 'functions': [['pl/trainer.py', 'Trainer.__init__']]}
    return b


# ---------------------------------------------------------------- sym_in

def test_dunder_never_named_by_its_bare_name():
    assert not O.sym_in('pl/trainer.py:5 def __init__(self, gpus=None):', 'Trainer.__init__')
    assert not O.sym_in('Trainer\n    __init__  L2-6', 'Trainer.__init__')
    assert O.sym_in('pl/trainer.py Trainer.__init__ L2-6', 'Trainer.__init__')  # the qualname itself counts


def test_method_needs_its_class():
    assert not O.sym_in('pl/trainer.py:8 in fit: def fit(self, model):', 'Trainer.fit')
    assert O.sym_in('L1-9  class Trainer\n    L8-9  def fit(self, model)', 'Trainer.fit')
    assert O.sym_in('see `Trainer` and its fit(model)', 'Trainer.fit')
    assert not O.sym_in('Trainerish.fit(model)', 'Trainer.fit')  # the class as a whole word


def test_function_by_last_part_in_code_position():
    assert O.sym_in('    helper  L1-2', 'helper')
    assert O.sym_in('calls helper() twice', 'helper')
    assert not O.sym_in('a helper for this', 'helper')  # prose, not code
    assert not O.sym_in('anything', None)


# ---------------------------------------------------------------- code names and project words

def test_code_names_skip_strings_docstrings_and_comments():
    import textwrap
    names = O.code_names(textwrap.dedent(TRAINER.split('def fit')[0].split('\n', 1)[1]))
    assert 'pick_gpu' in names and 'root_gpu' in names
    assert 'PyTorchLightning' not in names and 'data_parallel_device_ids' not in names
    assert 'CUDA_VISIBLE_DEVICES' not in names


def test_code_names_fallback_on_fragments():
    names = O.code_names('    x = call_it("PyTorchLightning")  # CUDA_VISIBLE_DEVICES\n  bad_indent = 1\n')
    assert 'call_it' in names and 'bad_indent' in names
    assert 'PyTorchLightning' not in names and 'CUDA_VISIBLE_DEVICES' not in names


def test_project_words():
    assert O.project_words('Lightning-AI/lightning') == {'lightning'}
    assert O.project_words('google/jax') == {'google'}
    assert 'pandas' in O.project_words('pandas-dev/pandas')


def test_search_candidates_exclude_docstring_and_project_identifiers(tmp_path):
    ps = ('cuda runtime error: PyTorchLightning sets CUDA_VISIBLE_DEVICES; data_parallel_device_ids wrong; '
          'pick_gpu returns the wrong root_gpu')
    b = builder(tmp_path, [], ps=ps)
    cands = [c for c, _ in b.search_candidates('pl/trainer.py', 'Trainer.__init__')]
    assert 'PyTorchLightning' not in cands  # project name (and only in a docstring)
    assert 'CUDA_VISIBLE_DEVICES' not in cands and 'data_parallel_device_ids' not in cands  # comment / docstring
    assert 'pick_gpu' in cands and 'root_gpu' in cands


# ---------------------------------------------------------------- per-file grounding

def test_symbol_grounded_requires_class_in_the_files_section(tmp_path):
    search_out = ('pl/__init__.py:8: __homepage__ = "https://github.com/PyTorchLightning"\n'
                  'pl/trainer.py:3 in __init__: """Set up. See https://github.com/PyTorchLightning/..."""\n')
    b = builder(tmp_path, [search_out])
    assert b.symbol_grounded('pl/trainer.py', 'Trainer.__init__') is None
    b2 = builder(tmp_path / 'b', ['0.9 pl/trainer.py\n      Trainer.__init__  L2-6\n      Trainer.fit  L8-9\n'])
    assert 'turn 0' in b2.symbol_grounded('pl/trainer.py', 'Trainer.__init__')
    assert b2.symbol_grounded('pl/trainer.py', 'Trainer.fit')


def test_symbol_listed_under_another_file_does_not_count(tmp_path):
    out = ('1.2 pl/other.py\n      helper  L1-2\n      Trainer.fit  L8-9\n'
           '0.4 pl/trainer.py\n      Trainer.__init__  L2-6\n')
    b = builder(tmp_path, [out])
    assert b.symbol_grounded('pl/trainer.py', 'Trainer.fit') is None
    assert b.symbol_grounded('pl/other.py', 'helper')


def test_issue_mention_of_a_method_needs_the_class(tmp_path):
    b = builder(tmp_path, ['0.9 pl/trainer.py\n'], ps='calling fit() crashes')
    assert b.symbol_grounded('pl/trainer.py', 'Trainer.fit') is None
    b2 = builder(tmp_path / 'b', ['0.9 pl/trainer.py\n'], ps='`Trainer.fit()` crashes')
    assert b2.symbol_grounded('pl/trainer.py', 'Trainer.fit')


def test_line_references_attributed_per_file(tmp_path):
    out = '1.2 pl/other.py\n      helper  L1-2\n0.4 pl/trainer.py\n      Trainer.__init__  L2-6\n'
    b = builder(tmp_path, [out])
    refs = b.refs_by_file(out)
    assert refs['pl/other.py'] == [(1, 2)] and refs['pl/trainer.py'] == [(2, 6)]
    assert b.def_range_grounding('pl/trainer.py', 6, 6)
    assert b.def_range_grounding('pl/other.py', 2, 2)
    assert os.path.isfile(os.path.join(b.ws, 'pl/trainer.py'))


# ---------------------------------------------------------------- v20 multi-edit path

AGENT_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1] / 'agent' / 'skills' / 'repo-tools' / 'scripts')
MOD = '''import os


def helper(x):
    return x + 1


def main(v):
    return helper(v)
'''
MOD_GOLD = MOD.replace('import os\n', 'import os\nimport sys\n').replace('return helper(v)',
                                                                         'return helper(v) or sys.argv')
SPAN_IMPORT = {'file': 'pkg/mod.py', 'a': 1, 'b': 1, 'old': ['import os'], 'new': ['import os', 'import sys'],
               'def': None}
SPAN_USE = {'file': 'pkg/mod.py', 'a': 9, 'b': 9, 'old': ['    return helper(v)'],
            'new': ['    return helper(v) or sys.argv'], 'def': [8, 9, 'main', 8]}
SPAN_OTHER = {'file': 'pkg/mod.py', 'a': 5, 'b': 5, 'old': ['    return x + 1'], 'new': ['    return x + 2'],
              'def': [4, 5, 'helper', 4]}


class FakeAgent:
    scripts = AGENT_SCRIPTS
    multi_edit = True


def edit_builder(tmp_path, spans=(SPAN_IMPORT, SPAN_USE), text=MOD, iid='org__repo-1'):
    """A Builder on a tiny repository whose gold lines were all printed (numbered) at turn 0, with a real Runner
    (the agent's own edit.py of this checkout, v20, run as ADK runs it)."""
    import collections
    import subprocess
    ws = tmp_path / 'ws'
    (ws / 'pkg').mkdir(parents=True)
    (ws / 'pkg' / '__init__.py').write_text('')
    (ws / 'pkg' / 'mod.py').write_text(text)
    for cmd in (['init', '-q'], ['add', '-A'], ['-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'base']):
        subprocess.run(['git', *cmd], cwd=ws, check=True, capture_output=True)
    b = object.__new__(O.Builder)
    b.task = {'instance_id': iid, 'repo': 'org/repo', 'problem_statement': ''}
    b.agent, b.ws, b.ps = FakeAgent(), str(ws), ''
    b.state = str(tmp_path / 'state')
    os.makedirs(b.state)
    b.R = O.Runner(b.agent, b.ws, b.state, sys.executable)
    b.R.i = 1
    shown = 'pkg/mod.py L1-9\n' + '\n'.join(f'{n:>5}  {line}' for n, line in enumerate(text.split('\n')[:9], 1))
    b.turns = [{'kind': 'show', 'call': {}}]
    b.outs = [shown]
    b.shown = collections.defaultdict(set, {'pkg/mod.py': set(range(1, 10))})
    b.notes, b.extra, b._strings = [], 0, {}
    b.base = {'pkg/mod.py': text}
    b.gold = {'files': ['pkg/mod.py'], 'spans': [dict(s) for s in spans], 'functions': [['pkg/mod.py', 'main']]}
    return b


def stdout_of(turn):
    import json
    return json.loads(turn['result'])['stdout']


def new_st():
    return {'statuses': [], 'merges': 0, 'repair': None, 'forms': []}


def test_hunk_groups_and_multi_args():
    lines = MOD.split('\n')
    groups = O.Builder.hunk_groups([SPAN_IMPORT, SPAN_USE], lines)
    assert [(a, b) for a, b, _, _ in groups] == [(1, 1), (9, 9)]
    args = O.Builder.multi_args('pkg/mod.py', groups)
    assert args == {'target': 'pkg/mod.py', 'start': '1', 'end': '1', 'new': 'import os\nimport sys',
                    'start2': '9', 'end2': '9', 'new2': '    return helper(v) or sys.argv'}
    # an insertion anchored on the line another span changes: one hunk (edit.py refuses overlapping hunks)
    ins = {'file': 'pkg/mod.py', 'a': 2, 'b': 1, 'old': [], 'new': ['import re'], 'def': None}
    groups = O.Builder.hunk_groups([SPAN_IMPORT, ins, SPAN_USE], lines)
    assert [(a, b) for a, b, _, _ in groups] == [(1, 1), (9, 9)]
    assert groups[0][2] == ['import os', 'import sys', 'import re'] and len(groups[0][3]) == 2


def test_single_selected_is_seeded_and_about_a_quarter():
    ids = [f'org__repo-{n}' for n in range(2000)]
    share = sum(O.single_selected(i, 'a.py') for i in ids) / len(ids)
    assert 0.2 < share < 0.3
    assert O.single_selected('org__repo-7', 'a.py') == O.single_selected('org__repo-7', 'a.py')


def test_multi_edit_one_call_gold_result_clean_check(tmp_path):
    b = edit_builder(tmp_path)
    form = b.multi_edit('pkg/mod.py', [SPAN_IMPORT, SPAN_USE], MOD.split('\n'), new_st())
    assert form == 'multi'
    assert (tmp_path / 'ws' / 'pkg' / 'mod.py').read_text() == MOD_GOLD
    t = b.turns[-1]
    assert t['kind'] == 'edit' and t['meta'] == {'hunks': 2} and t['label'] == 'train'
    args = t['call']['arguments']['args']
    assert list(args) == sorted(args) and args['start2'] == '9'  # keys as the model writes them; base numbers
    out = stdout_of(t)
    assert out.startswith('EDITED pkg/mod.py (2 hunks'), out
    assert O.check_status(out) == 'clean', out
    assert 'printed numbered at turn 0' in t['grounding']


def test_multi_edit_not_gold_restores_and_falls_back(tmp_path):
    b = edit_builder(tmp_path)
    bad = dict(SPAN_USE, new=['    return helper(v) or sys.argv)'])  # does not parse: edit.py refuses the call
    assert b.multi_edit('pkg/mod.py', [SPAN_IMPORT, bad], MOD.split('\n'), new_st()) is None
    assert (tmp_path / 'ws' / 'pkg' / 'mod.py').read_text() == MOD and len(b.turns) == 1 and b.R.i == 1
    assert b.notes and 'single edits' in b.notes[0]


def test_bottom_up_single_edits_show_the_transient_undefined_name(tmp_path):
    b = edit_builder(tmp_path)
    assert b.single_edits('pkg/mod.py', [SPAN_IMPORT, SPAN_USE], MOD.split('\n'), new_st()) is None
    assert (tmp_path / 'ws' / 'pkg' / 'mod.py').read_text() == MOD_GOLD
    checks = [O.check_status(stdout_of(t)) for t in b.turns[1:]]
    assert checks == ['undefined', 'clean']  # the use first (bottom-up), then the import


def test_single_order_imports_first_when_numbers_allow(tmp_path):
    lines = MOD.split('\n')
    b = edit_builder(tmp_path)
    # the import adds a line: applying it first would move line 9 -> no allowed order avoids the transient
    assert b.single_order('pkg/mod.py', [SPAN_IMPORT, SPAN_USE], lines) == (None, True)
    # a line-count-neutral import change: import first, the use keeps its base number
    neutral = {'file': 'pkg/mod.py', 'a': 1, 'b': 1, 'old': ['import os'], 'new': ['import os, sys'], 'def': None}
    order, transient = b.single_order('pkg/mod.py', [neutral, SPAN_USE], lines)
    assert not transient and [s['a'] for s in order] == [1, 9]
    # independent hunks: bottom-up as before
    use2 = dict(SPAN_USE, new=['    return helper(v) * 2'])
    assert b.single_order('pkg/mod.py', [SPAN_OTHER, use2], lines) == (None, False)


def test_edits_multi_unless_selected_and_independent(tmp_path):
    ids = [f'org__repo-{n}' for n in range(200)]
    multi_id = next(i for i in ids if not O.single_selected(i, 'pkg/mod.py'))
    single_id = next(i for i in ids if O.single_selected(i, 'pkg/mod.py'))
    b = edit_builder(tmp_path / 'a', iid=multi_id)
    res, why = b.edits()
    assert why is None and res['forms'] == ['multi'] and len(b.turns) == 2
    # selected for single edits, but the use needs the import another hunk adds: the multi-edit after all
    res, why = edit_builder(tmp_path / 'b', iid=single_id).edits()
    assert why is None and res['forms'] == ['multi_dependent']
    # selected and independent: single edits, bottom-up
    use2 = dict(SPAN_USE, new=['    return helper(v) * 2'])
    b = edit_builder(tmp_path / 'c', spans=(SPAN_OTHER, use2), iid=single_id)
    res, why = b.edits()
    assert why is None and res['forms'] == ['single_selected'] and len(b.turns) == 3
    assert [t['call']['arguments']['args']['start'] for t in b.turns[1:]] == ['9', '5']
    # an agent without multi-edit (v19): single edits
    b = edit_builder(tmp_path / 'd', iid=multi_id)
    b.agent = type('A', (), {'scripts': AGENT_SCRIPTS, 'multi_edit': False})()
    res, why = b.edits()
    assert why is None and res['forms'] == ['single_v19'] and len(b.turns) == 3


def test_check_status():
    assert O.check_status('EDITED x\nCHECK: compiles; no new undefined names; import x OK\n') == 'clean'
    assert O.check_status("CHECK: line 3: undefined name 'sys'") == 'undefined'
    assert O.check_status('CHECK: import x timed out') == 'timeout'
    assert O.check_status('EDITED x (README.md)') == 'none'
