"""Paper F2: ds1 (LoRA on DeepSeek trajectories) vs v30 on unseen100 [E095/E096], on the tasks where both runs got
model responses before the vLLM server hung (~09:20 UTC): edits, shell commands per run, solves.

    python3 results/_analysis/paper/ds1.py
"""

import glob
import json
import os

RUNS = {'v30': 'results/E095_v30_unseen', 'v30+ds1': 'results/E096_v30ds1_unseen'}


def load(run):
    out = {}
    for f in glob.glob(f'{run}/r1/*/traces/*'):
        t = json.load(open(f))
        tid = os.path.basename(f)[len('trace_'):-len('.json')]
        responses = rc = 0
        edit = False
        for st in t['steps']:
            if st.get('source') == 'agent' and (st.get('message') or st.get('tool_calls')):
                responses += 1
            for tc in st.get('tool_calls') or []:
                nm = tc.get('function_name') or tc.get('name')
                rc += nm == 'run_command'
                edit |= nm in ('edit_file', 'write_file')
        out[tid] = (responses, rc, edit)
    for f in glob.glob(f'{run}/r1/*/task_results.jsonl'):
        for line in open(f):
            r = json.loads(line)
            if r['instance_id'] in out:
                out[r['instance_id']] += (bool(r['resolved']),)
    return out


def main():
    d = {k: load(v) for k, v in RUNS.items()}
    common = sorted(t for t in d['v30'] if all(x.get(t, (0,))[0] > 0 for x in d.values()))
    print(f'{len(common)} tasks with model responses in both runs')
    for k, x in d.items():
        print(f'{k:8} edited {sum(x[t][2] for t in common):3}  run_command/run {sum(x[t][1] for t in common) / len(common):5.1f}'
              f'  solved {sum(x[t][3] for t in common):3}')


if __name__ == '__main__':
    main()
