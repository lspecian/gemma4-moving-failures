"""LLM-free benchmark: does kb.tests_for find the test file the hidden tests will extend? (research/notes/23 follow-up)

E083-E085: before its first edit, a strong model read a file that the hidden tests modify in 19/37 successful runs on
tasks Gemma never solves; Gemma did so in 14/92 runs (v24, with its read-a-test rule: 7/46). If our tools can name
that file (and its tests of the target) with zero extra calls, the model sees the repository's own conventions.

For every task: extract the snapshot (replay_tools.prepare), build the KB, run kb.tests_for on each gold source file
(the .py files the gold patch changes, tests excluded), and rank the distinct test files. "Hidden test files" = the
files the task's test_patch modifies that exist in the snapshot (new test files cannot be found). Reports the rank of
the first hidden test file: @1 / @3 / @5 / found / not found, and how many tasks only add new test files.

    python3 scripts/testfile_bench.py --tasks data/competition/tasks.jsonl --snapshots data/competition/snapshots \
        --list experiments/eval113.txt [--agent agent] [--out results/_analysis/testfile/eval113.json]
"""

import argparse
import json
import os
import re
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay_tools  # noqa: E402

TEST_RE = re.compile(r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$')


def files_of(patch):
    return sorted(set(re.findall(r'^(?:diff --git a/|\+\+\+ b/)(\S+)', patch or '', re.M)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--snapshots', required=True)
    ap.add_argument('--list', required=True)
    ap.add_argument('--agent', default=os.path.join(HERE, '..', 'agent'))
    ap.add_argument('--out', default='')
    ap.add_argument('--queries', default='', help='results dir of a recorded run: BM25 also with its first locate query')
    a = ap.parse_args()
    recorded = {}
    if a.queries:
        import ast
        import glob
        for f in glob.glob(os.path.join(a.queries, 'r1', '*', 'traces', 'trace_*.json')):
            tid = os.path.basename(f)[len('trace_'):-len('.json')]
            for st in json.load(open(f))['steps']:
                tc = st.get('tool_calls')
                calls = (ast.literal_eval(tc) if isinstance(tc, str) else tc) or []
                q = next(((c.get('arguments') or {}).get('args', {}).get('query') for c in calls
                          if (c.get('arguments') or {}).get('file_path') == 'locate.py'
                          and isinstance((c.get('arguments') or {}).get('args'), dict)), None)
                if q:
                    recorded[tid] = q
                    break
    sys.path.insert(0, os.path.join(a.agent, 'skills', 'repo-tools', 'scripts'))
    tasks = {json.loads(line)['instance_id']: json.loads(line) for line in open(a.tasks)}
    ids = [x.strip() for x in open(a.list) if x.strip() and not x.startswith('#')]
    rows = []
    for tid in ids:
        t = tasks[tid]
        gold = [p for p in files_of(t['patch']) if p.endswith('.py') and not TEST_RE.search(p)]
        hidden = [p for p in files_of(t.get('test_patch')) if p.endswith('.py')]
        with tempfile.TemporaryDirectory() as td:
            ws, state = os.path.join(td, 'ws'), os.path.join(td, 'state')
            os.makedirs(state)
            replay_tools.prepare(os.path.join(a.snapshots, f'{tid}.tgz'), ws)
            existing = [p for p in hidden if os.path.exists(os.path.join(ws, p))]
            os.environ['WORKSPACE'], os.environ['REPO_TOOLS_DIR'] = ws, state
            import importlib
            import kb
            importlib.reload(kb)
            kb._OPEN.clear()
            conn = kb.open(ws)
            ranked = []
            for g in gold:
                for tp, _ in kb.tests_for(g, conn, k=30):
                    if tp not in ranked:
                        ranked.append(tp)
            # signals for tuning (saved): tests_for from locate's top source file (the realistic target), and BM25 of
            # the issue text over test files
            import locate
            importlib.reload(locate)
            lranked = locate.rank(ws, t['problem_statement'], conn)[0]
            top = next((p for p in lranked if p.endswith('.py') and not TEST_RE.search(p)), None)
            from_top = []
            if top:
                for tp, _ in kb.tests_for(top, conn, k=30):
                    if tp not in from_top:
                        from_top.append(tp)
            fs, _ = kb.bm25(conn, 'files', kb.tokenize(t['problem_statement']))
            tests = dict(conn.execute('SELECT id, path FROM files WHERE indexed = 1 AND is_test = 1'))
            bm = sorted(((sc, tests[i]) for i, sc in fs.items() if i in tests), reverse=True)[:30]
            rq = recorded.get(tid)
            fq, _ = kb.bm25(conn, 'files', kb.tokenize(rq)) if rq else ({}, None)
            bmq = sorted(((sc, tests[i]) for i, sc in fq.items() if i in tests), reverse=True)[:30]
            conn.close()
            kb._OPEN.clear()
        rank = next((i + 1 for i, p in enumerate(ranked) if p in existing), None)
        rows.append({'task': tid, 'gold': gold, 'hidden_test_files': hidden, 'existing': existing,
                     'ranked': ranked[:30], 'rank': rank, 'top_file': top, 'from_top': from_top[:30],
                     'bm25_tests': [(round(sc, 3), p) for sc, p in bm], 'query': rq,
                     'bm25_tests_query': [(round(sc, 3), p) for sc, p in bmq]})
        print(f"{tid:45} gold {len(gold)} hidden {len(hidden)} (existing {len(existing)}) rank {rank}", flush=True)
    n = len(rows)
    new_only = sum(1 for r in rows if r['hidden_test_files'] and not r['existing'])
    findable = [r for r in rows if r['existing']]
    def at(k):
        return sum(1 for r in findable if r['rank'] and r['rank'] <= k)
    print(f"\n{n} tasks; hidden tests only in new files: {new_only}; findable {len(findable)}: "
          f"@1 {at(1)}, @3 {at(3)}, @5 {at(5)}, found {at(10 ** 6)}, not found {len(findable) - at(10 ** 6)}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
