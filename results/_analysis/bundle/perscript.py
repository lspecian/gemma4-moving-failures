"""Per-tool 'other' time (step delta minus estimated model time of the NEXT model call is not separable, so we attribute
each step's delta minus its own model estimate to the tool it called; the model call that produced the tool call is
counted in the same step)."""
import json, glob, ast, collections, sys
name = sys.argv[1]
agg = collections.defaultdict(list)
for f in glob.glob(f'results/{name}/r1/*/traces/*.json'):
    prev = None
    for s in json.load(open(f))['steps']:
        ex = s.get('extra'); ex = ast.literal_eval(ex) if isinstance(ex, str) else (ex or {})
        t = float(ex.get('elapsed_s', 0) or 0)
        m = s.get('metrics'); m = ast.literal_eval(m) if isinstance(m, str) and m else (m or {})
        model = m.get('completion_tokens', 0) / 35 + max(0, m.get('prompt_tokens', 0) - (m.get('cached_tokens') or 0)) / 1400 if m else 0
        tc = s.get('tool_calls')
        if prev is not None and tc:
            calls = ast.literal_eval(tc) if isinstance(tc, str) else tc
            c = calls[0]; key = c['function_name']
            if key == 'run_skill_script': key = (c.get('arguments') or {}).get('file_path', '?')
            if key == 'run_command': key = 'run_command'
            agg[key].append(t - prev - model)
        prev = t
rows = sorted(((k, len(v), sum(v)) for k, v in agg.items()), key=lambda r: -r[2])
tasks = len(glob.glob(f'results/{name}/r1/*/traces/*.json'))
for k, n, s in rows[:12]:
    print(f"  {k:22} calls {n:5}  other time per call {s/n:6.1f}s  per task {s/tasks:6.1f}s")
