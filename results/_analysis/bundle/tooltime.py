"""Where agent time goes on unseen100: model time (35 tok/s decode + uncached prefill at 1,400 tok/s, from the trace's
token counts) vs everything else (tool execution, sandbox, overhead) = last step's elapsed_s minus model time."""
import json, glob, ast, statistics as st
def run(name):
    rows = []
    for f in glob.glob(f'results/{name}/r1/*/traces/*.json'):
        steps = json.load(open(f))['steps']; model = 0.0; last = 0.0; n = 0
        for s in steps:
            ex = s.get('extra'); ex = ast.literal_eval(ex) if isinstance(ex, str) else (ex or {})
            last = max(last, float(ex.get('elapsed_s', 0) or 0))
            m = s.get('metrics')
            if not m: continue
            m = ast.literal_eval(m) if isinstance(m, str) else m
            model += m.get('completion_tokens', 0) / 35 + max(0, m.get('prompt_tokens', 0) - (m.get('cached_tokens') or 0)) / 1400
            n += 1
        if last: rows.append((last, model, last - model, n))
    tot = sum(r[0] for r in rows); mod = sum(r[1] for r in rows)
    print(f"{name:24} tasks {len(rows)}  agent time mean {tot/len(rows):.0f}s  model {mod/len(rows):.0f}s  "
          f"other {(tot-mod)/len(rows):.0f}s ({(tot-mod)/tot:.0%})  other/call {(tot-mod)/sum(r[3] for r in rows):.1f}s  "
          f"median other {st.median(r[2] for r in rows):.0f}s")
for n in ('E076_neb_v24_unseen', 'E077_neb_v25_unseen', 'E080_bundle_unseen'): run(n)
