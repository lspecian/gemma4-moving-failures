"""Read size vs compaction (LOG 2026-10-29): per arm, when the first compaction hits (prompt tokens drop >20%), how
many runs compact, tool-output tokens by tool, wall time, empty patches, solves; time split prefill/decode/rest.

    python3 results/_analysis/reads/compaction.py results/E080_bundle_unseen results/E103_v32 results/E104_v33b results/E105_v34
"""
import ast
import collections
import glob
import json
import statistics as st
import sys


def arm(run):
    res = {}
    for f in glob.glob(f'{run}/r1/*/task_results.jsonl'):
        for line in open(f):
            r = json.loads(line)
            res[r['instance_id']] = r
    first, n_comp, tools, pre, dec = [], 0, collections.Counter(), 0, 0
    for f in glob.glob(f'{run}/r1/*/traces/*.json'):
        t = json.load(open(f))
        last, calls, comp = 0, 0, None
        for s in t['steps']:
            m = s.get('metrics')
            if m:
                m = m if isinstance(m, dict) else ast.literal_eval(m)
                p = m.get('prompt_tokens', 0)
                calls += 1
                pre += max(0, p - (m.get('cached_tokens') or 0))
                dec += m.get('completion_tokens', 0)
                if comp is None and last and p < last * 0.8:
                    comp = calls
                last = p
            tc, ob = s.get('tool_calls'), s.get('observation')
            if tc and ob:
                tc = tc if isinstance(tc, list) else ast.literal_eval(tc)
                tools[tc[0].get('function_name')] += len(str(ob)) / 4
        if comp:
            n_comp += 1
            first.append(comp)
    rs = list(res.values())
    wall = sum(r['duration_seconds'] for r in rs)
    tot = sum(tools.values()) or 1
    print(f'{run}: {len(rs)} runs, pass {sum(r["resolved"] for r in rs)}, empty '
          f'{sum(1 for r in rs if not r.get("agent_patch_size"))}, median wall {st.median(r["duration_seconds"] for r in rs):.0f} s')
    print(f'  compaction in {n_comp}/{len(rs)} runs, first at model call median {st.median(first) if first else "-"}')
    print(f'  tool-output tokens per run {tot / max(1, len(rs)):.0f}: ' +
          ', '.join(f'{k} {v / tot:.0%}' for k, v in tools.most_common(4)))
    print(f'  time: prefill {pre / 1400 / wall:.0%}, decode {dec / 35 / wall:.0%}, rest {1 - (pre / 1400 + dec / 35) / wall:.0%}')


for run in sys.argv[1:]:
    arm(run)
