"""scripts/toolbench.py: the pure parts (query, gold, output parsing, bot template filling, edit mistakes, scorecard)
and one end-to-end run of every phase on a tiny repository with the current agent/ scripts."""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.dont_write_bytecode = True
import toolbench as tb  # noqa: E402

PATCH = textwrap.dedent('''\
    diff --git a/billing/invoice.py b/billing/invoice.py
    index 1111111..2222222 100644
    --- a/billing/invoice.py
    +++ b/billing/invoice.py
    @@ -7,6 +7,8 @@ class Invoice:
             self.amount = amount

         def total(self, tax=0.1):
    -        if self.amount == None:
    +        if self.amount is None:
                 return 0
    +        if tax < 0:
    +            raise ValueError("negative tax")
             return self.amount * (1 + tax)
    diff --git a/tests/test_invoice.py b/tests/test_invoice.py
    --- a/tests/test_invoice.py
    +++ b/tests/test_invoice.py
    @@ -1,1 +1,2 @@
     import billing
    +import pytest
    diff --git a/billing/new.py b/billing/new.py
    new file mode 100644
    --- /dev/null
    +++ b/billing/new.py
    @@ -0,0 +1,1 @@
    +X = 1
    ''')

INVOICE = textwrap.dedent('''\
    """Payment helpers."""


    class Invoice:
        def __init__(self, amount):
            # the amount
            self.amount = amount

        def total(self, tax=0.1):
            if self.amount == None:
                return 0
            return self.amount * (1 + tax)
    ''')


# ------------------------------------------------------------------ query

def test_build_query_title_paths_errors_identifiers():
    st = ('Invoice.total ignores the tax\n\nCalling `Invoice(10).total(tax=-1)` in billing/invoice.py raises\n'
          'Traceback (most recent call last):\nKeyError: \'rate\'\nSee https://example.com/x_y for apply_discount.')
    q = tb.build_query(st)
    assert q.startswith('Invoice.total ignores the tax')
    assert 'billing/invoice.py' in q and "KeyError: 'rate'" in q and 'apply_discount' in q
    assert 'example.com' not in q and 'x_y' not in q          # URLs removed
    assert q.count('Invoice.total') == 1                       # nothing repeated from the title


def test_build_query_caps_length():
    st = 'Title\n' + ' '.join(f'name_{i}' for i in range(200))
    q = tb.build_query(st)
    assert len(q) <= tb.QUERY_CHARS and q.count('name_') == tb.QUERY_IDENTS


# ------------------------------------------------------------------ gold

def test_parse_patch_and_spans():
    pf = tb.parse_patch(PATCH)
    assert pf['billing/new.py']['kind'] == 'added' and pf['billing/invoice.py']['kind'] == 'modified'
    sp = tb.spans_of('billing/invoice.py', pf['billing/invoice.py']['hunks'])
    assert len(sp) == 1
    s = sp[0]
    assert (s['a'], s['b']) == (10, 11)
    assert s['old'] == ['        if self.amount == None:', '            return 0']
    assert s['new'][0] == '        if self.amount is None:' and len(s['new']) == 4


def test_parse_patch_without_git_headers_and_insertion():
    p = '--- a/m.py\n+++ b/m.py\n@@ -2,0 +3,1 @@\n+x = 1\n'
    s = tb.spans_of('m.py', tb.parse_patch(p)['m.py']['hunks'])[0]
    assert (s['a'], s['b'], s['old'], s['new']) == (3, 2, [], ['x = 1'])   # insertion before base line 3
    assert tb.apply_spans('a\nb\nc\n', [s]) == 'a\nb\nx = 1\nc\n'


def test_gold_of_skips_tests_and_new_files_and_finds_functions():
    g = tb.gold_of(PATCH, lambda p: INVOICE if p == 'billing/invoice.py' else None)
    assert g['files'] == ['billing/invoice.py'] and g['added'] == ['billing/new.py']
    assert g['functions'] == [['billing/invoice.py', 'Invoice.total']]
    want = tb.apply_spans(INVOICE, g['spans'])
    assert 'if self.amount is None:' in want and 'raise ValueError("negative tax")' in want


def test_def_ranges_fold_nested_and_include_decorators():
    src = 'class A:\n    @property\n    def f(self):\n        def g():\n            return 1\n        return g\n'
    assert tb.def_ranges(src) == [(2, 6, 'A.f', 3)]
    assert tb.enclosing(tb.def_ranges(src), 5)[2] == 'A.f'


def test_anchored_insertion_deletion_and_trailing_blank():
    base = ['a', 'b', '', 'c', '']
    assert tb.anchored({'a': 2, 'b': 1, 'new': ['x']}, base) == (1, 1, ['a', 'x'])        # insert before b
    assert tb.anchored({'a': 1, 'b': 0, 'new': ['x']}, base) == (1, 1, ['x', 'a'])        # insert at the top
    assert tb.anchored({'a': 2, 'b': 2, 'new': []}, base) == (1, 2, ['a'])               # delete b
    assert tb.anchored({'a': 2, 'b': 2, 'new': ['B', '']}, base) == (2, 4, ['B', '', '', 'c'])


def test_merge_spans_keeps_lines_between():
    base = ['l1', 'l2', 'l3', 'l4', 'l5']
    m = tb.merge_spans([{'file': 'f', 'a': 2, 'b': 2, 'new': ['X']}, {'file': 'f', 'a': 4, 'b': 4, 'new': ['Y']}],
                       base)
    assert (m['a'], m['b'], m['new']) == (2, 4, ['X', 'l3', 'Y'])


# ------------------------------------------------------------------ output parsing

LOCATE_OUT = textwrap.dedent('''\
    3.30 billing/invoice.py  [defines 2 issue names]
          Invoice.total  L9-12
          Invoice.__init__  L5-7
    0.80 billing/util.py
          helper  L1-4
    --- best match: billing/invoice.py Invoice.total L9-12 (named in the issue)
        9      def total(self, tax=0.1):
       10          if self.amount == None:
    NEXT: if the function shown above is the place, edit it: edit.py  target=billing/invoice.py  old=<exact current lines>  new=<new lines>
    NEXT: otherwise read ONE other candidate: show.py  target=billing/util.py  symbol=helper
    ''')


def test_parse_locate():
    p = tb.parse_locate(LOCATE_OUT)
    assert p['files'] == ['billing/invoice.py', 'billing/util.py']
    assert p['functions'][0] == ['billing/invoice.py', 'Invoice.total', 9, 12]
    assert p['best'] == ['billing/invoice.py', 'Invoice.total', 9, 12]


def test_shown_lines_attributes_numbered_lines_to_files():
    files = {'billing/invoice.py', 'billing/util.py'}
    sh = tb.shown_lines(LOCATE_OUT, files.__contains__)
    assert sh == {'billing/invoice.py': {9, 10}}
    show = 'billing/util.py helper L1-4 of 4\n1  def helper():\n2      return 1\nNEXT: x'
    assert tb.shown_lines(show, files.__contains__) == {'billing/util.py': {1, 2}}
    assert tb.shown_lines('1  def helper():', files.__contains__) == {}     # no file header: not attributed


def test_split_footer():
    body, foot = tb.split_footer('x\nnote: results are JSON-encoded, so\nBUDGET: call 3\nNEXT: a\nNEXT: b')
    assert body == 'x' and foot.count('\n') == 3


def test_calls_in_and_first_next_call():
    cs = tb.calls_in('NEXT: then edit: edit.py  target=a.py  start=3  end=4  new=<new text>  (the change)')
    assert cs == [('edit.py', {'target': 'a.py', 'start': '3', 'end': '4', 'new': '<new text>'})]
    out = 'NEXT: edit where H1 is confirmed; candidates: a.py\nNEXT: or show.py  target=a.py  symbol=f'
    assert tb.first_next_call(out)[:3] == ('show.py', {'target': 'a.py', 'symbol': 'f'}, 1)
    assert tb.first_next_call('no footer') is None


def test_fill_call():
    ctx = {'focus_file': 'a.py', 'focus_fn': 'A.f', 'query': 'q', 'idents': ['A.f'], 'candidates': [('a.py', 'A.f')]}
    assert tb.fill_call('edit.py', {'target': 'a.py'}, ctx) == (None, 'edit_suggested')
    assert tb.fill_call('run.py', {'test': 'tests/t.py'}, ctx) == (None, 'run_test')
    assert tb.fill_call('check_patch.py', {}, ctx) == (None, 'check_patch')
    assert tb.fill_call('show.py', {'target': '<file.py>', 'symbol': '<Class.method>'}, ctx) == \
        ({'target': 'a.py', 'symbol': 'A.f'}, None)
    args, stop = tb.fill_call('hypothesis.py', {'add': '<guess>', 'add2': '<second guess>'}, ctx)
    assert stop is None and args['add'].startswith('a.py:A.f')
    assert tb.fill_call('run.py', {'snippet': '<code>'}, ctx)[0] == {'snippet': 'print("toolbench: no repro")'}
    assert tb.fill_call('outline.py', {'files': 'a.py', 'grep': '<x>'}, dict(ctx, idents=[]))[1] == 'unfillable'


def test_repair_call_needs_a_range_or_old():
    assert tb.repair_call('NEXT: copy the old text exactly and retry edit.py') is None
    out = 'if NEW replaces exactly these lines: edit.py  target=a.py  start=2  end=3  new=<your NEW unchanged>'
    rep = tb.repair_call(out)
    assert rep == {'target': 'a.py', 'start': '2', 'end': '3', 'new': '<your NEW unchanged>'}
    assert tb.fill_repair(rep, {'new': 'x\ny'}, 'a.py')['new'] == 'x\ny'


# ------------------------------------------------------------------ mistakes

def test_mistakes_cover_the_model_errors():
    g = tb.gold_of(PATCH, lambda p: INVOICE)
    span = g['spans'][0]
    ms = {name: (edits, goal) for name, edits, goal in tb.mistakes(span, INVOICE.split('\n'))}
    assert set(ms) == {'stale_lines', 'start_only', 'json_old', 'json_both', 'indent_plus4', 'indent_minus4',
                       'line_prefix', 'old_blank_drift', 'old_skip_line', 'old_wrong_line'}  # + v20 drift
    assert '\n\n\n' in ms['old_blank_drift'][0][0]['old']  # one blank line more than the file has there
    for name in ('old_skip_line', 'old_wrong_line'):  # OLD and NEW carry context; the drifted line is not changed
        e = ms[name][0][0]
        assert e['old'].count('\n') >= 4 and 'self.amount' in e['old'] and set(e) == {'target', 'old', 'new'}
    assert '        pass' in ms['old_wrong_line'][0][0]['old'] and '        pass' in ms['old_wrong_line'][0][0]['new']
    pre, stale = ms['stale_lines'][0]
    assert pre['start'] == pre['end'] == '9' and stale['start'] == '10'          # shift above the def, old numbers
    assert '# toolbench: shifted 1' in ms['stale_lines'][1](INVOICE)
    assert 'end' not in ms['start_only'][0][0]
    assert '\\n' in ms['json_old'][0][0]['old'] and '\n' in ms['json_old'][0][0]['new']
    assert ms['line_prefix'][0][0]['new'].startswith('10          if self.amount is None:')
    assert ms['indent_plus4'][0][0]['new'].startswith(' ' * 12 + 'if')
    assert ms['start_only'][1](INVOICE) == tb.apply_spans(INVOICE, [span])


def test_no_indent_mistakes_inside_strings():
    src = 'X = """\n<a>\n<b>\n"""\n'
    span = {'file': 'f.py', 'a': 3, 'b': 3, 'old': ['<b>'], 'new': ['  <b>'], 'def': None}
    names = [n for n, _, _ in tb.mistakes(span, src.split('\n'), tb.string_lines(src))]
    assert 'indent_plus4' not in names and 'line_prefix' in names


def test_normalise_clock_and_tmp_paths():
    a = 'BUDGET: call 3 of ~20 | 0m40s of 4m30s used\n/private/var/folders/x/y/z.py\n[9/20 2:10/4:30 | e]'
    b = 'BUDGET: call 3 of ~20 | 1m05s of 4m30s used\n/tmp/abc/z.py\n[9/20 3:10/4:30 | e]'
    assert tb.normalise(a) == tb.normalise(b)


def test_devolatile_drops_names_lines():
    a = 'f.py L1-2 of 9\n1  x = g()\nnames: g f.py:7 ()\nNEXT: y'
    assert tb.devolatile(a) == tb.devolatile('f.py L1-2 of 9\n1  x = g()\nNEXT: y')


def test_format_report_skips_rows_without_a_comparison():
    """--report on a format-diff dir: skipped tasks (no modified .py gold) have no 'same' (was a KeyError)."""
    row = {'task': 't1', 'source': 'local', 'agent': 'a->b', 'phase': 'bot', 'i': 2, 'script': 'show.py',
           'same': False, 'body_same': True, 'footer_same': False, 'same_ex_volatile': False, 'diff': 'line 3: ...'}
    text = tb.format_report([{'task': 't0', 'source': 'local', 'agent': 'a', 'skip': 'no modified .py gold'}, row])
    assert 'local: 1 tasks, 1 calls compared (a->b)' in text and 'footer differs 1' in text


def test_same_code():
    assert tb.same_code('f(a,\n      b)\n', 'f(a,\n  b)\n') and not tb.same_code('x = 1\n', 'x = 2\n')


# ------------------------------------------------------------------ scorecard

def test_metrics_and_scorecard_with_two_agents():
    def rec(agent, rank, gold_at):
        return {'task': 't', 'source': 'local', 'repo': 'r', 'agent': agent,
                'locate': {'rank': rank, 'fn_in_list': True, 'has_fn': True, 'best_gold': rank == 1,
                           'best_shown': True, 'gold_shown': rank == 1, 'tokens': 500.0},
                'bot': {'gold_at': gold_at, 'gold_ctx': 6000 if gold_at else None, 'stop': 'edit_suggested',
                        'edit': {'overlap': True, 'near': True, 'gold_file': True}, 'calls': 2, 'tokens': 900,
                        'crossed_before_gold': False if gold_at else None},
                'oracle': {'exact': True, 'equal_ws': True, 'same_code': True, 'first_try': True, 'check_ready': True,
                           'calls': 5, 'tokens': 2000, 'ctx_end': 9000},
                'robust': [{'mistake': 'start_only', 'outcome': 'intended'}, {'mistake': 'json_old', 'outcome': 'wrong'}],
                'calls': [{'script': 'locate.py', 'tokens': 500.0, 'chars': 1600, 'footer_chars': 160, 'code': 0}]}
    text, summary = tb.scorecard([rec('A', 2, None), rec('B', 1, 1)])
    assert summary['local']['A']['locate file@1 %'] == 0 and summary['local']['B']['locate file@1 %'] == 100
    assert summary['local']['B']['robust silent wrong edit %'] == 50
    line = next(l for l in text.splitlines() if l.startswith('locate file@1 %'))
    assert line.rstrip().endswith('+100.0')


# ------------------------------------------------------------------ end to end (current agent/ scripts)

def _git(ws, *a):
    subprocess.run(['git', '-C', str(ws), '-c', 'user.email=t@t', '-c', 'user.name=t', '-c', 'commit.gpgsign=false',
                    *a], check=True, capture_output=True)


def test_all_phases_on_a_tiny_repo(tmp_path):
    ws = tmp_path / 'ws'
    (ws / 'billing').mkdir(parents=True)
    (ws / 'billing' / '__init__.py').write_text('')
    (ws / 'billing' / 'invoice.py').write_text(INVOICE)
    (ws / 'tests').mkdir()
    (ws / 'tests' / 'test_invoice.py').write_text('import billing\n')
    _git(ws, 'init', '-q')
    _git(ws, 'add', '-A')
    _git(ws, 'commit', '-q', '-m', 'baseline')
    _git(ws, 'tag', tb.BASELINE)
    agent = {'label': 'cur', 'scripts': str(ROOT / 'agent/skills/repo-tools/scripts'), 'skill': 'repo-tools'}
    task = {'instance_id': 'tiny', 'source': 'local', 'repo': 'x/tiny',
            'query': tb.build_query('Invoice.total compares None with ==\nUse `is None` in billing/invoice.py')}
    task['gold'] = tb.gold_of(PATCH, lambda p: tb.read_file(str(ws), p))
    opt = {'python': sys.executable, 'call_seconds': 20.0}
    work = tmp_path / 'work'
    work.mkdir()
    tr = tb.TaskRun(task, agent, str(ws), opt, str(work))
    loc = tr.locate()
    assert loc['rank'] == 1 and loc['fn_in_list']
    bot = tr.bot()
    assert bot['stop'] and 1 <= bot['calls'] <= tb.BOT_MAX_CALLS
    ex = tr.bot(explore=True)
    assert ex['stop'] and ex['calls'] >= bot['calls'] and ex['stop'] != 'edit_suggested'
    ora = tr.oracle()
    assert ora['exact'] and ora['first_try'], ora
    rob = tr.robust()
    assert {r['mistake'] for r in rob} >= {'start_only', 'json_old', 'line_prefix'}
    assert all(r['outcome'] for r in rob)
    assert tb.read_file(str(ws), 'billing/invoice.py') == INVOICE          # workspace reset after the phases
    json.dumps(tr.calls)                                                    # records are JSON-serialisable
    assert all(c['tokens'] > 0 for c in tr.calls)
    assert not any(c.get('crash') for c in tr.calls), [c['err'] for c in tr.calls if c.get('crash')]
    assert os.path.isdir(tr.template)
