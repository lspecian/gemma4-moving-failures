"""Compare call_probe.py runs: per state kind, how many sampled calls were well-formed.

    python3 scripts/probe_report.py results/P002_v10_vllm results/P003_v11_vllm [-v]

Labels: ok (valid run_skill_script call), ok_jsonq (valid, but the raw call used JSON quotes somewhere),
bad_envelope (skill_name/file_path missing or wrong), other_tool (a non-script tool: submit_patch, list_skills...),
no_call (text only), PARSER_HANG (Kaggle's parser would freeze the server).
Sub-agent roles (call_probe.py --role main|writer|scout): ok_writer / ok_scout (a well-formed agent_tool call),
bad_agent_args (writer/scout without a string `request`: AgentTool raises KeyError, the run aborts with an empty
patch), unknown_tool (an undeclared tool name: ADK raises, same), missing_script (a script this agent does not have).
Extra lines per role: main = writer-request shape; writer wedit = edits applied to the snapshot; scout = chain outcomes.
"""

import collections
import json
import sys

FAIL = ('bad_envelope', 'bad_envelope_jsonq', 'no_call', 'PARSER_HANG', 'parse_error', 'unknown_tool', 'bad_agent_args',
        'missing_script', 'bad_call', 'no_answer')


def load(d):
    return [json.loads(line) for line in open(f'{d}/samples.jsonl')]


def extra(r, kind, ss):
    """Role-specific lines (call_probe.py --role)."""
    name = r.split('/')[-1]
    w = [s for s in ss if s.get('shape')]
    if w:
        n = len(w)
        print(f'   {name:22s} writer requests {n}: all of target/start/end/change '
              f'{sum(all(s["shape"][k] for k in ("target", "start", "end", "change")) for s in w)}/{n}, change looks like '
              f'code {sum(s["shape"]["change_code"] for s in w)}/{n}, change chars median '
              f'{sorted(s["shape"]["change_chars"] for s in w)[n // 2]}')
    crash = sum(s['label'] in ('unknown_tool', 'bad_agent_args') for s in ss)
    if crash:
        print(f'   {name:22s} RUN-ABORTING calls: {crash}')
    scripts = collections.Counter(s.get('script') for s in ss if s.get('script') is not None)
    if scripts:
        print(f'   {name:22s} scripts called: {dict(scripts)}')
    ed = [s for s in ss if 'applied' in s]
    if ed:
        print(f'   {name:22s} edits applied to the snapshot: {sum(s["applied"] for s in ed)}/{len(ed)}')
    ch = [s for s in ss if s.get('kind') == 'scout' or 'turns' in s]
    if ch:
        n = len(ch)
        print(f'   {name:22s} scout chains {n}: answered {sum(s["label"] == "answered" for s in ch)}, calls mean '
              f'{sum(s["calls"] for s in ch) / n:.1f}, chains with a repeated call {sum(s["repeats"] > 0 for s in ch)}, '
              f'NOT FOUND {sum(s["not_found"] for s in ch)}, answers with pointers '
              f'{sum(s["pointers"] > 0 for s in ch)} (pointers valid {sum(s["pointers_ok"] for s in ch)}/'
              f'{sum(s["pointers"] for s in ch)}), tokens/turn median '
              f'{sorted(t["tokens"] or 0 for s in ch for t in s["turns"])[sum(len(s["turns"]) for s in ch) // 2]}')


def main():
    verbose = '-v' in sys.argv
    runs = [a for a in sys.argv[1:] if a != '-v']
    data = {r: load(r) for r in runs}
    kinds = sorted({s['kind'] for ss in data.values() for s in ss})
    for kind in kinds + ['all']:
        print(f'== {kind}')
        for r, ss in data.items():
            ss = [s for s in ss if kind == 'all' or s['kind'] == kind]
            c = collections.Counter(s['label'] for s in ss)
            bad = sum(c[k] for k in FAIL)
            tools = collections.Counter(s['tool'] for s in ss if s['label'].startswith('other_tool'))
            jq = sum(v for k, v in c.items() if k.endswith('jsonq'))
            toks = sorted(s['tokens'] for s in ss if s.get('tokens'))
            th = sorted(s['think_chars'] for s in ss if s.get('think_chars') is not None)
            med = lambda v: v[len(v) // 2] if v else '-'
            print(f'   {r.split("/")[-1]:22s} tokens/call median {med(toks)} p90 {toks[int(len(toks) * .9)] if toks else "-"}  '
                  f'thinking chars median {med(th)}')
            print(f'   {r.split("/")[-1]:22s} n={len(ss):3d}  malformed={bad:3d} ({100 * bad / max(len(ss), 1):4.1f}%)  '
                  f'json-quotes={jq:3d}  other_tool={sum(tools.values()):3d} {dict(tools)}  {dict(c)}')
            extra(r, kind, ss)
    if verbose:
        for r, ss in data.items():
            print(f'\n#### {r}: malformed samples')
            for s in ss:
                if s['label'] in FAIL or s['label'].endswith('jsonq'):
                    print(f"-- {s['kind']} {s['trace'].split('trace_')[-1][:-5]} step {s['step']} seed {s['seed']}: "
                          f"{s['label']}\n   {s['raw_call'][:400]!r}")


if __name__ == '__main__':
    main()
