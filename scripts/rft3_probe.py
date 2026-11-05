"""rft3 offline decision-state probe (research/notes/14-oracle-trajectories.md §6 "What to measure offline"): states
where the gold code is on screen (S1: expected action edit) and where it is not (S2: expected action a read), each
with the exact request body the agent sends there, and a scorer for a served model (vLLM: base, rft2, rft3 adapters).

    .venv/bin/python scripts/rft3_probe.py build                       # -> data/rft3/probe_states.jsonl
    .venv/bin/python scripts/rft3_probe.py score --server http://127.0.0.1:8002/v1 [--model rft3] [--k 8] \\
        --out results/probe/rft3.jsonl                                 # samples + per-group summary
    .venv/bin/python scripts/rft3_probe.py report results/probe/base.jsonl results/probe/rft3.jsonl

States (build):
  oracle_heldout  the oracle trajectories of the held-out repos (rft3_build.py --val-repos: never trained; agent-v20
                  format; the subset's ones with the policy's own carried/sampled thinking in earlier turns, the rest
                  without thinking). S1 = the request before the first edit (oracle_repair: before the scripted mistake):
                  grounded, i.e. the lines it edits were printed; the oracle edit call is stored. S2 = the latest
                  request before it (after >= 1 result) where no gold file's gold lines are all printed yet and the
                  oracle reads. At most --max-s1 / --max-s2 (sha1 order of the task).
  real_heldout_<format>  the captured requests (data/raw/<run>, scripts/slow_proxy.py --save) of our own runs on the 37
                  held-out eval tasks (experiments/lora2_heldout.txt; never trained; --runs, default the real-speed
                  runs E060 v19b + E059 v18d at thinking budget 256; their format is v19b/v18d, i.e. an older system
                  prompt and scripts than v20: the score summary reports the groups separately), before
                  the first compaction: S1 = the first request where every gold line of some gold file has been printed
                  (rft3_sample.known_lines over the tool results; gold = the task patch's .py non-test hunks) and no
                  edit was attempted; S2 = the first request after a result where no gold file is printed yet. One S1
                  and one S2 per (run, task). The model's recorded call at that state is kept (recorded).
Line: {id, group, kind (S1|S2), expected (edit|read), task, repo, source, run, format, k, request (body to send:
messages, tools, sampling params, chat_template_kwargs, thinking_token_budget), gold {files, spans}, oracle_call
(oracle_heldout S1), recorded (real_heldout: the recorded next call's script), prompt_chars}.

Score (score): K samples per state (n=K in one vLLM request, the request's own temperature unless --temperature),
parsed with rft3_sample.parse_choice (server's tool parser + raw token ids). Per sample: action = edit (edit.py) /
read (show, search, outline, locate, callers, tests_for, history, status) / run / check / submit / create / other /
malformed (rft3_sample problems: no_call, multi_call, bad_json, unknown tool or script, malformed, raw parser problems)
/ length; for edits: resolvable on the printed lines (rft3_sample.edit_hunks) and overlapping the gold lines, and for
oracle states the step B match rule vs the oracle edit (rft3_sample.match_edit, change F1 >= 0.8). Summary per group
and kind: P(edit), P(edit resolvable & overlaps gold), P(edit matches oracle), P(read), malformed rate, length cuts,
thinking tokens (median, share at the budget).
"""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import statistics
import sys
import threading
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rft3_states as S  # noqa: E402
import rft3_sample as RS  # noqa: E402

READS = {'show.py', 'search.py', 'outline.py', 'locate.py', 'callers.py', 'tests_for.py', 'history.py', 'status.py'}
EDITS = {'edit.py'}
RUNS = 'E060_v19b_real,E059_v18d_real'  # real speed (~35 tok/s), thinking budget 256 (E061: 512)
OUT = os.path.join(ROOT, 'data', 'rft3', 'probe_states.jsonl')


# ------------------------------------------------------------------ gold visibility (pure)

def known_of_messages(messages):
    """{file: {line: text}} printed by the tool results of a request's messages (in order)."""
    return RS.known_lines([{'result': m.get('content')} for m in messages if m.get('role') == 'tool'])


def visible_files(gold, known):
    """Gold files whose gold lines are all printed in KNOWN."""
    out = []
    for f in gold.get('files') or sorted({sp['file'] for sp in gold.get('spans') or []}):
        need = RS.gold_lines(gold, RS.norm_path(f))
        if need and need <= set(known.get(RS.norm_path(f), {})):
            out.append(f)
    return out


def is_edit_call(call):
    return RS.script_of(call) in EDITS or RS.script_of(call) == 'create.py'


def classify(sample):
    """Action class of a parsed sample (rft3_sample.parse_choice)."""
    if sample['problem'] == 'length':
        return 'length'
    if sample['problem']:
        return 'malformed'
    sc = RS.script_of(sample['call'])
    if sc in EDITS:
        return 'edit'
    if sc in READS:
        return 'read'
    return {'run.py': 'run', 'check_patch.py': 'check', 'submit_patch': 'submit', 'create.py': 'create'}.get(sc, 'other')


def edit_quality(call, gold, known, oracle_turn=None, threshold=0.8):
    """{resolvable, overlaps_gold, matches_oracle} of a sampled edit.py call at a state."""
    args = RS.args_of(call)
    f = RS.target_of(args)
    hunks, _ = RS.edit_hunks(args, known.get(f, {}))
    out = {'resolvable': hunks is not None, 'overlaps_gold': False, 'matches_oracle': None}
    if hunks:
        gl = RS.gold_lines(gold, f)
        out['overlaps_gold'] = any(gl & set(range(a, b + 1)) for a, b, _ in hunks) if gl else False
    if oracle_turn is not None:
        out['matches_oracle'] = bool(RS.match_edit(args, RS.args_of(oracle_turn['call']), known, threshold)['ok'])
    return out


# ------------------------------------------------------------------ build

def gold_of_patch(patch):
    import toolbench as tb
    spans = []
    files = []
    for path, f in tb.parse_patch(patch).items():
        if f['kind'] != 'modified' or not path.endswith('.py') or tb.TEST_PATH.search(path):
            continue
        sp = tb.spans_of(path, f['hunks'])
        if sp:
            files.append(path)
            spans += [{'file': s['file'], 'a': s['a'], 'b': s['b']} for s in sp]
    return {'files': files, 'spans': spans}


def sha(x):
    return hashlib.sha1(x.encode()).hexdigest()


def oracle_states(recs, max_s1, max_s2):
    s1, s2 = [], []
    for rec in sorted(recs, key=lambda r: sha(r['task'])):
        turns = rec['turns']
        ke = next((k for k, t in enumerate(turns) if t.get('kind') in ('edit', 'edit_mistake', 'edit_repair')), None)
        if ke is None:
            continue
        base = {'task': rec['task'], 'repo': rec['repo'], 'source': rec['source'], 'run': None, 'format': 'v20',
                'gold': {'files': rec['gold'].get('files'), 'spans': rec['gold'].get('spans')},
                'ctx_thinking': sum(1 for t in turns[:ke] if t.get('thinking'))}
        known = RS.known_lines(turns[:ke])
        s1.append(dict(base, id=f'{rec["id"]}@{ke}', group='oracle_heldout', kind='S1', expected='edit', k=ke,
                       request=S.request_at(rec, ke), oracle_call=turns[ke]['call'],
                       visible=visible_files(rec['gold'], known)))
        for k in range(ke - 1, 0, -1):
            if not visible_files(rec['gold'], RS.known_lines(turns[:k])) and \
                    RS.script_of(turns[k]['call']) in READS:
                s2.append(dict(base, id=f'{rec["id"]}@{k}', group='oracle_heldout', kind='S2', expected='read', k=k,
                               request=S.request_at(rec, k), oracle_call=turns[k]['call'], visible=[]))
                break
    return s1[:max_s1], s2[:max_s2]


def real_states(runs, held, tasks, stats):
    out = []
    for run in runs:
        if not os.path.isdir(os.path.join(ROOT, 'data', 'raw', run)):
            stats[f'{run}: no captures'] += 1
            continue
        groups, _ = S.captured_episodes(run, tasks=tasks)
        for tid, reqs in sorted(groups.items()):
            if tid not in held:
                continue
            gold = gold_of_patch(tasks[tid]['patch'])
            if not gold['spans']:
                stats['no .py gold'] += 1
                continue
            pre = [reqs[0]]
            for x, y in zip(reqs, reqs[1:]):
                if y['request']['messages'][:len(x['request']['messages'])] != x['request']['messages']:
                    break
                pre.append(y)
            found = {}
            for k, d in enumerate(pre):
                msgs = d['request']['messages']
                calls = [c for m in msgs if m.get('role') == 'assistant' for c in m.get('tool_calls') or []]
                if any(is_edit_call({'name': c['function']['name'], 'arguments': _args(c)}) for c in calls):
                    break
                if not any(m.get('role') == 'tool' for m in msgs):
                    continue
                vis = visible_files(gold, known_of_messages(msgs))
                kind = 'S1' if vis else 'S2'
                if kind in found:
                    continue
                resp = (d['response'].get('choices') or [{}])[0].get('message') or {}
                rc = (resp.get('tool_calls') or [None])[0]
                found[kind] = dict(id=f'{run}_{tid}@{k}', group='real_heldout_' + run.split('_')[1], kind=kind,
                                   expected='edit' if vis else 'read', task=tid, repo=tasks[tid]['repo'],
                                   source='capture', run=run, format=run.split('_')[1], k=k,
                                   request=S.strip_proxy(d['request']), gold=gold, oracle_call=None, visible=vis,
                                   recorded=RS.script_of({'name': rc['function']['name'], 'arguments': _args(rc)})
                                   if rc else 'text')
                if len(found) == 2:
                    break
            out += [found[k] for k in ('S1', 'S2') if k in found]
            stats['real episodes'] += 1
    return out


def _args(c):
    try:
        a = json.loads(c['function'].get('arguments') or '{}')
    except (ValueError, TypeError):
        return {}
    return a if isinstance(a, dict) else {}


def cmd_build(a):
    import rft3_build as B
    S.use_version('v20')
    val_repos = {x for x in a.val_repos.split(',') if x}
    stats = collections.Counter()
    ns = argparse.Namespace(filled=os.path.join(S.STATES_DIR, 'subset900_filled.jsonl'), states=S.STATES_DIR,
                            sampled=a.sampled or S.suffixed(os.path.join(ROOT, 'data', 'rft3', 'sampled')))
    filled, _, remainder = B.load_records(ns, stats)
    recs = [r for r in filled + remainder if r['repo'] in val_repos and r['source'] in B.ORACLE]
    s1, s2 = oracle_states(recs, a.max_s1, a.max_s2)
    tasks = {json.loads(l)['instance_id']: json.loads(l) for l in open(os.path.join(ROOT, 'data', 'competition',
                                                                                  'tasks.jsonl'))}
    held = {x.strip() for x in open(a.heldout) if x.strip() and not x.startswith('#')}
    real = real_states([x for x in a.runs.split(',') if x], held, tasks, stats) if a.runs else []
    lines = s1 + s2 + real
    for x in lines:
        x['prompt_chars'] = sum(len(m.get('content') or '') for m in x['request']['messages'])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        for x in lines:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    c = collections.Counter((x['group'], x['kind']) for x in lines)
    print(f'oracle records of {sorted(val_repos)}: {len(recs)} ({sum(1 for r in recs if r in filled)} with sampled '
          f'thinking); states: ' + ', '.join(f'{g} {k} {n}' for (g, k), n in sorted(c.items())))
    rec = collections.Counter((x['kind'], x.get('recorded')) for x in real)
    print(f'real held-out: {dict(stats)}; recorded action at the state: {dict(rec)}')
    print(f'wrote {a.out} ({len(lines)} states)')


# ------------------------------------------------------------------ score

def score_state(x, srv, a):
    body = {k: v for k, v in x['request'].items()}
    if a.temperature is not None:
        body['temperature'] = a.temperature
    if a.token_ids:
        body['return_token_ids'] = True
    t0 = time.time()
    choices, usage, _ = srv.chat(body, a.k, a.seed)
    budget = x['request'].get('thinking_token_budget')
    known = known_of_messages(x['request']['messages'])
    oracle_turn = {'call': x['oracle_call']} if x.get('oracle_call') and x['kind'] == 'S1' else None
    out = []
    for ch in choices:
        s = RS.parse_choice(ch, budget)
        act = classify(s)
        d = {'action': act, 'problem': s['problem'], 'think_tokens': s['think_tokens'], 'at_budget': s['at_budget'],
             'visible': bool(s['visible']), 'script': RS.script_of(s['call']) if s['call'] else None}
        if act == 'edit':
            d.update(edit_quality(s['call'], x['gold'], known, oracle_turn))
        if a.keep_calls:
            d['call'] = s['call']
            d['thinking'] = s['thinking']
        out.append(d)
    return {'id': x['id'], 'group': x['group'], 'kind': x['kind'], 'expected': x['expected'], 'task': x['task'],
            'samples': out, 'usage': usage, 'seconds': round(time.time() - t0, 1), 'model': srv.model}


def summarize(results):
    by = collections.defaultdict(list)
    for r in results:
        by[(r['group'], r['kind'])].append(r)
        by[('all', r['kind'])].append(r)
    out = {}
    for key, rs in sorted(by.items()):
        ss = [s for r in rs for s in r['samples']]
        n = len(ss) or 1
        acts = collections.Counter(s['action'] for s in ss)
        edits = [s for s in ss if s['action'] == 'edit']
        th = [s['think_tokens'] for s in ss if s.get('think_tokens') is not None]
        mo = [s for s in edits if s.get('matches_oracle') is not None]
        out[f'{key[0]} {key[1]}'] = {
            'states': len(rs), 'samples': len(ss),
            'P(edit)': round(acts['edit'] / n, 3),
            'P(edit resolvable & overlaps gold)': round(sum(1 for s in edits if s.get('resolvable') and
                                                            s.get('overlaps_gold')) / n, 3),
            'P(edit matches oracle)': round(sum(1 for s in mo if s['matches_oracle']) / n, 3) if mo or
            key[0] == 'oracle_heldout' else None,
            'P(read)': round(acts['read'] / n, 3),
            'malformed': round(acts['malformed'] / n, 3), 'length': round(acts['length'] / n, 3),
            'states with >=1 edit': round(sum(1 for r in rs if any(s['action'] == 'edit' for s in r['samples'])) /
                                          max(len(rs), 1), 3),
            'think median': statistics.median(th) if th else None,
            'think at budget': round(sum(1 for s in ss if s.get('at_budget')) / n, 3),
            'actions': dict(acts.most_common())}
    return out


def cmd_score(a):
    srv = RS.Server(a.server, a.model)
    states = [json.loads(l) for l in open(a.probe)]
    if a.only_group:
        states = [x for x in states if x['group'] == a.only_group]
    if a.limit:
        states = states[:a.limit]
    done = {}
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            done[d['id']] = d
    todo = [x for x in states if x['id'] not in done]
    print(f'server {srv.kind} model {srv.model}; {len(states)} states, {len(todo)} to score, K={a.k}', flush=True)
    lock = threading.Lock()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def go(x):
        try:
            r = score_state(x, srv, a)
        except Exception as e:  # noqa: BLE001 (one state's failure must not stop the pass)
            print(f'ERROR {x["id"]}: {e!r}', flush=True)
            return
        with lock:
            with open(a.out, 'a') as f:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            done[r['id']] = r
            if len(done) % 10 == 0:
                print(f'  {len(done)}/{len(states)}', flush=True)

    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(go, todo))
    summ = summarize([done[x['id']] for x in states if x['id'] in done])
    json.dump(summ, open(os.path.splitext(a.out)[0] + '.summary.json', 'w'), indent=1)
    print_summary({os.path.basename(a.out): summ})


def print_summary(summs):
    keys = sorted({k for s in summs.values() for k in s})
    cols = ['states', 'P(edit)', 'P(edit resolvable & overlaps gold)', 'P(edit matches oracle)', 'P(read)',
            'malformed', 'length', 'states with >=1 edit', 'think median', 'think at budget']
    for k in keys:
        print(f'== {k}')
        print('   ' + ' | '.join(['model'.ljust(24)] + [c[:22] for c in cols]))
        for name, s in summs.items():
            v = s.get(k)
            if v:
                print('   ' + ' | '.join([name[:24].ljust(24)] + [str(v.get(c)) for c in cols]))


def cmd_report(a):
    summs = {}
    for p in a.results:
        rs = [json.loads(l) for l in open(p) if l.strip()]
        summs[os.path.basename(p)] = summarize(rs)
    print_summary(summs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', default=OUT)
    b.add_argument('--val-repos', default='ipython/ipython,scipy/scipy', help='= rft3_build.py --val-repos')
    b.add_argument('--sampled', default=None, help='GPU pass dir (thinking of the subset; default sampled_v20)')
    b.add_argument('--max-s1', type=int, default=75)
    b.add_argument('--max-s2', type=int, default=75)
    b.add_argument('--runs', default=RUNS, help='captured runs for real held-out states ("" = none)')
    b.add_argument('--heldout', default=os.path.join(ROOT, 'experiments', 'lora2_heldout.txt'))
    s = sub.add_parser('score')
    s.add_argument('--server', default=RS.DEFAULT_SERVER)
    s.add_argument('--model', default=None, help='model id (default: the last listed = a served adapter)')
    s.add_argument('--probe', default=OUT)
    s.add_argument('--out', required=True)
    s.add_argument('--k', type=int, default=8)
    s.add_argument('--temperature', type=float, default=None, help="default: the request's own (agent: 0.2)")
    s.add_argument('--seed', type=int, default=None)
    s.add_argument('--concurrency', type=int, default=6)
    s.add_argument('--token-ids', action='store_true', help='vLLM return_token_ids (raw-format problems, exact counts)')
    s.add_argument('--keep-calls', action='store_true', help='store every sample call and thinking')
    s.add_argument('--only-group', default=None)
    s.add_argument('--limit', type=int, default=0)
    r = sub.add_parser('report')
    r.add_argument('results', nargs='+')
    a = ap.parse_args()
    {'build': cmd_build, 'score': cmd_score, 'report': cmd_report}[a.cmd](a)


if __name__ == '__main__':
    main()
