"""rft3: carry step B's sampled thinking (v19 records) over to the same trajectories rebuilt in a newer agent version
(v20: multi-edit + CHECK line), so only the turns whose state changed need a new (small) GPU sampling pass.

    .venv/bin/python scripts/rft3_carry.py [--version v20] [--subset data/rft3/subset900.txt]

Inputs: the v19 subset (data/rft3/subset900.txt: ids), the v19 states (data/rft3/states/<source>.jsonl: the oracle
calls and outputs the sampling saw), step B's export (data/rft3/sampled/<source>.records.jsonl: thinking/visible/call
and turn['sampling'] per train turn) and the new version's states (data/rft3/states_v20/<source>.jsonl).

Matching: an oracle trajectory by task (its source may change, e.g. repair -> clean when the new edit.py applies the
mistake instead of refusing it), a relabel state by its id (run + task).

Per train turn k of the new record, the v19 turn k is REUSED when
  - the call is identical (same tool, script and args: rft3_sample.identical),
  - every earlier turn j < k has the identical call and the identical tool result except for `CHECK:` lines (v20's
    edit.py diagnostics) and Python's SyntaxWarning/DeprecationWarning lines in stderr (v19 printed them while
    parsing the repository's files, in a run-dependent order; v20 since d67f7bc silences them) together with the
    "warning" status they alone caused, clock jitter up to 3 s in BUDGET footers
    (the replay clock adds the script's real run time: 0m39s vs 0m40s) and pytest durations - the system prompt may
    differ (v20 lists fewer scripts and shows the multi-edit) -, and the
    same thinking/visible text for context turns (relabel prefixes: the model's own).
Then the turn gets v19's kept thinking/visible/call and sampling (+ sampling.carried = "v19"), including turns where
step B kept nothing (from null, mask_thinking: sampled once already, the call trains alone). Every other train turn
gets thinking/visible null and sampling {from: null, needs_sampling: true, mask_thinking: true, plan}, plan = the
subset plan of step B (LOG 2026-10-07): bootstrap K=2 for locate/show/submit, the edit fallback (rationalization) for
edit/edit_repair, none (call only, no GPU) for check/outline/search/history.
Note: a reused turn's later context differs from step B's where an earlier turn is resampled (as step B's own second
pass; sampling.ctx records the sources).

Outputs (data/rft3/states_v20/): subset900_filled.jsonl (the records, every source; rft3_sample.py --version v20
sample --records <it> samples exactly the turns not carried), subset900_todo.jsonl ({id, source, k, kind, plan} per
turn that needs GPU work) and subset900_carry_report.json (reuse by kind, divergence reasons, dropped records, the GPU
estimate from step B's measured wall time per turn: 158.0 min for 2,806 bootstrap turns and 309.2 min for 1,381 edit
fallback turns at concurrency 6 on one RTX 6000 Ada with the rft2 LoRA, ~260 tok/s aggregate decode).
"""

import argparse
import collections
import copy
import json
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rft3_states as S  # noqa: E402

OUT = os.path.join(ROOT, 'data', 'rft3')
SOURCES = ('oracle_clean', 'oracle_recovery', 'oracle_repair', 'relabel')
PLAN = {'locate': 'bootstrap', 'show': 'bootstrap', 'show_range': 'bootstrap', 'submit': 'bootstrap',
        'edit': 'fallback', 'edit_repair': 'fallback'}
# step B, subset 900 (logs/rft3B_sub900_boot.log, logs/rft3B_sub900_fb.log): wall minutes per turn at concurrency 6
BOOT_MIN_PER_TURN = 158.0 / 2806
FB_MIN_PER_TURN = 309.2 / 1381
SERVER_SETUP_MIN = 25  # vast instance start + model and adapter load (step B / E059: ~20-30 min)
CHECK_LINE = re.compile(r'^CHECK: .*\n?', re.M)
CLOCK = re.compile(r'\b(\d+)m(\d\d)s\b')
DURATION = re.compile(r'\bin \d+\.\d+s\b')    # pytest's "98 passed in 1.96s" (run.py)
PARSE_WARNING = re.compile(r'^\S*:\d+: (?:SyntaxWarning|DeprecationWarning): ')
CLOCK_TOL = 3   # seconds: the replay clock (relabel: the recorded call times + the script's real run time) jitters


def norm_result(result):
    """(the tool result as comparable text, [clock values in seconds]): v20's CHECK lines removed from stdout, BUDGET
    clocks taken out (compared with CLOCK_TOL), test durations blanked, stderr lines sorted."""
    try:
        d = json.loads(result)
    except (TypeError, ValueError):
        return result, []
    clocks = []

    def take(m):
        clocks.append(int(m.group(1)) * 60 + int(m.group(2)))
        return 'T'
    if isinstance(d, dict) and isinstance(d.get('stdout'), str):
        d['stdout'] = DURATION.sub('in Xs', CLOCK.sub(take, CHECK_LINE.sub('', d['stdout'])))
    if isinstance(d, dict) and isinstance(d.get('stderr'), str):
        # v19 (step B's contexts) printed Python's warnings about the repository's code while parsing it (in a
        # run-dependent order); v20 since d67f7bc does not: those lines, and the "warning"/"error" status they alone
        # caused, are ignored
        rest = [x for x in d['stderr'].splitlines() if not PARSE_WARNING.match(x)]
        d['stderr'] = sorted(rest)
        if 'status' in d:
            d['status'] = 'success' if not rest else ('error' if not d.get('stdout') else 'warning')
    return json.dumps(d, sort_keys=True, ensure_ascii=False), clocks


def same_result(a, b):
    (ta, ca), (tb, cb) = norm_result(a), norm_result(b)
    return ta == tb and len(ca) == len(cb) and all(abs(x - y) <= CLOCK_TOL for x, y in zip(ca, cb))


def norm_call(call):
    a = call.get('arguments')
    if call['name'] != 'run_skill_script' or not isinstance(a, dict):
        return json.dumps(call, sort_keys=True)
    args = a.get('args') if isinstance(a.get('args'), dict) else a.get('args')
    if isinstance(args, dict):
        args = {k: str(v).strip() for k, v in args.items()}
    return json.dumps({'name': call['name'], 'file_path': S.script_of(call), 'skill_name': a.get('skill_name'),
                       'args': args}, sort_keys=True)


def diverge_reason(old, new):
    """Why turn `new` (v20) differs from `old` (v19) as context for later turns, or None."""
    if norm_call(old['call']) != norm_call(new['call']):
        return 'call'
    if old['result'] != new['result'] and not same_result(old['result'], new['result']):
        return 'output'
    if old['label'] == 'context' and ((old.get('thinking') or None) != (new.get('thinking') or None) or
                                      (old.get('visible') or None) != (new.get('visible') or None)):
        return 'context text'
    return None


def kind_of(t):
    return t.get('kind') or S.script_of(t['call']) or t['call']['name']


def load(dirname, suffix=''):
    out = {}
    for src in SOURCES:
        p = os.path.join(dirname, f'{src}{suffix}.jsonl')
        if os.path.exists(p):
            for r in S.read_records([p]):
                out[r['id']] = r
    return out


def key_of(rec):
    return rec['id'] if rec['source'] == 'relabel' else ('oracle', rec['task'])


def carry(v19_states, v19_sampled, new_states, subset):
    """(filled records, todo rows, report)."""
    old_by_key = {key_of(r): r for rid, r in v19_states.items() if rid in subset}
    new_by_key = {key_of(r): r for r in new_states.values()}
    filled, todo = [], []
    rep = collections.Counter()
    by_kind = collections.defaultdict(collections.Counter)
    reasons = collections.Counter()
    first_diff_kind = collections.Counter()
    src_change = collections.Counter()
    for key, old in sorted(old_by_key.items(), key=lambda kv: str(kv[0])):
        new = new_by_key.get(key)
        if new is None:
            rep[f'dropped in {S.VERSION}: {old["source"]}'] += 1
            continue
        if new['source'] != old['source']:
            src_change[f'{old["source"]} -> {new["source"]}'] += 1
        sampled = v19_sampled[old['id']]
        rec = copy.deepcopy(new)
        div_at, why = None, None
        for k, t in enumerate(rec['turns']):
            # the earliest j < k whose call/output differ breaks reuse for every turn after j
            if div_at is None and k > 0:
                j = k - 1
                if j >= len(old['turns']):
                    div_at, why = j, 'longer'
                else:
                    r_ = diverge_reason(old['turns'][j], rec['turns'][j])
                    if r_:
                        div_at, why = j, r_
                        reasons[f'{r_} differs ({kind_of(rec["turns"][j])})'] += 1
                        first_diff_kind[kind_of(rec['turns'][j])] += 1
            if t['label'] != 'train':
                continue
            kind = kind_of(t)
            ok = div_at is None and k < len(old['turns']) and old['turns'][k]['label'] == 'train' and \
                norm_call(old['turns'][k]['call']) == norm_call(t['call'])
            if ok:
                st = sampled['turns'][k]
                t['thinking'], t['visible'] = st.get('thinking'), st.get('visible')
                a_ = (st['call'].get('arguments') or {}).get('args')
                if not (isinstance(a_, dict) and any(not isinstance(v, str) for v in a_.values())):
                    t['call'] = st['call']  # (step B kept 5 sampled calls with int values: the oracle's strings)
                t['sampling'] = dict(st.get('sampling') or {}, carried='v19')
                got = 'reused: thinking' if st.get('thinking') else 'reused: no thinking (masked before)'
            else:
                plan = PLAN.get(kind, 'none')
                t['thinking'] = t['visible'] = None
                t['sampling'] = {'from': None, 'needs_sampling': True, 'mask_thinking': True, 'plan': plan,
                                 'think_tokens': None, 'rule': None, 'call_from': None}
                got = f'new: {plan}'
                if plan != 'none':
                    todo.append({'id': rec['id'], 'source': rec['source'], 'k': k, 'kind': kind, 'plan': plan,
                                 'hunks': (t.get('meta') or {}).get('hunks', 1)})
            by_kind[kind][got] += 1
            by_kind['ALL'][got] += 1
        rec['meta']['carry'] = {'from': old['id'], 'diverged_at': div_at, 'why': why}
        filled.append(rec)
    n_boot = sum(1 for x in todo if x['plan'] == 'bootstrap')
    n_fb = sum(1 for x in todo if x['plan'] == 'fallback')
    gpu_min = n_boot * BOOT_MIN_PER_TURN + n_fb * FB_MIN_PER_TURN
    report = {'subset': len(subset), 'records_filled': len(filled), 'records': dict(rep),
              'source_changes': dict(src_change),
              'filled_by_source': dict(collections.Counter(r['source'] for r in filled)),
              'train_turns_by_kind': {k: dict(v) for k, v in sorted(by_kind.items())},
              'first_difference': dict(reasons.most_common()),
              'todo': {'turns': len(todo), 'records': len({x['id'] for x in todo}), 'bootstrap_K2': n_boot,
                       'edit_fallback': n_fb, 'multi_edit_fallback': sum(1 for x in todo if x['hunks'] > 1)},
              'gpu_estimate': {
                  'wall_minutes_sampling': round(gpu_min, 1),
                  'wall_minutes_with_setup': round(gpu_min + SERVER_SETUP_MIN, 1),
                  'basis': 'step B subset-900 wall time per turn at concurrency 6, RTX 6000 Ada + rft2 LoRA '
                           f'(~260 tok/s aggregate decode, prompt cache ~0): bootstrap {BOOT_MIN_PER_TURN * 60:.1f} s,'
                           f' edit fallback {FB_MIN_PER_TURN * 60:.1f} s per turn; + ~{SERVER_SETUP_MIN} min setup'}}
    return filled, todo, report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--version', default='v20')
    ap.add_argument('--subset', default=os.path.join(OUT, 'subset900.txt'))
    ap.add_argument('--name', default='subset900')
    a = ap.parse_args()
    v19_states = load(os.path.join(OUT, 'states'))
    v19_sampled = load(os.path.join(OUT, 'sampled'), '.records')
    S.use_version(a.version)
    new_states = load(S.STATES_DIR)
    subset = {l.strip() for l in open(a.subset) if l.strip() and not l.startswith('#')}
    missing = subset - set(v19_sampled)
    if missing:
        raise SystemExit(f'{len(missing)} subset ids have no step B export, e.g. {sorted(missing)[:3]}')
    filled, todo, report = carry(v19_states, v19_sampled, new_states, subset)
    out = os.path.join(S.STATES_DIR, f'{a.name}_filled.jsonl')
    with open(out, 'w') as f:
        for r in filled:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(os.path.join(S.STATES_DIR, f'{a.name}_todo.jsonl'), 'w') as f:
        for x in todo:
            f.write(json.dumps(x) + '\n')
    json.dump(report, open(os.path.join(S.STATES_DIR, f'{a.name}_carry_report.json'), 'w'), indent=1)
    print(json.dumps(report, indent=1))
    print(f'wrote {out} ({len(filled)} records), {a.name}_todo.jsonl ({len(todo)} turns)')


if __name__ == '__main__':
    main()
