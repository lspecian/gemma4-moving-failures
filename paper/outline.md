# Paper outline (draft 1, 2026-10-08)

Track: Kaggle "Gemma 4 Developer Agent – Paper Track". Deadline **Nov 12, 2026, 23:59 UTC**. At most 3,000 words,
non-archival (an arXiv copy is allowed). Judged 0–5 each on novelty, quality/generality, relevance, verifiability
and clarity. Up to two write-ups per team.

> Before anything here goes public: no competition task text, gold patches or test files (our repo stays private).
> Quote task ids and counts only; publish our scripts and logs separately if we decide to.

## Working title

**Where the Minutes Go: Failure Modes, Budget Control and Evaluation Pitfalls for a 31B Local Coding Agent**

Alternative: *Moving Failures Around: What Harness Engineering Can and Cannot Fix in a Small, Time-Bound SWE Agent*

## Thesis (one sentence)

For a small local model fixing real issues under a hard wall-clock budget, most harness and post-training changes
**move tasks between failure modes** (no edit ↔ wrong edit) instead of creating passes. Common evaluation shortcuts
(emulated speed, contaminated task sets, single runs) are large enough to **reverse** the conclusions a developer
draws.

Why it scores on the criteria:
- **Novelty:** the failure-mode view of harness changes, plus three concrete pitfalls with measured effects.
- **Quality/generality:** applies to any budgeted agent on a small model (on-device, offline, CI bots).
- **Relevance:** the competition's own premise (an offline agent on modest hardware).
- **Verifiability:** every number has an experiment id, traces, raw model I/O and a cost ledger (~$150 of GPU time).
- **Clarity:** one setting, one table of failure modes per experiment, five findings.

## Structure (~3,000 words)

| § | Section | Words | Content |
|---|---|---|---|
| — | Abstract | 150 | Setting, thesis, 5 findings in one line each, release. |
| 1 | Introduction | 350 | Small local agents; budget + slow decoding; contributions list. |
| 2 | Setting | 300 | Gemma 4 31B w4a16, 4×L4 TP4, ~35 tok/s, 4 min / 40 calls per task, 32k context with ADK compaction at 14k (visible text only), sequential 12 h; hidden tests; leaderboard on 58 tasks. |
| 3 | System | 400 | Narrow interface (one skill, 21 stdlib scripts, no shell); the flow engine (stateful footers: NEXT, BUDGET, MEMO, refusals); robust edit tool (fuzzy repair, CHECK diagnostics, multi-edit). One footer example figure. |
| 4 | Evaluation method | 350 | Kaggle-faithful local eval: real-speed pacing proxy, compaction + scorer template, per-task environments (gold pass 52 → 115/129), failure-mode accounting (pass / wrong / empty / timeout), paired Fisher tests, cost tracking. |
| 5 | Findings | 1,100 | F1–F5 below, one table or figure each. |
| 6 | Related work | 250 | See list below. |
| 7 | Limitations & conclusion | 200 | One model, 3–4 repositories, public LB of 58 tasks, few repeats. |

## Findings and evidence

**F1. Emulated speed reversed the diagnosis and misdirected a training run.**
- At 12 tok/s (a forum figure): v18d 26/113, 60% empty patches. The dominant failure was "never edits"
  (~70 runs per arm reading until timeout). E057/E058.
- That diagnosis motivated rft3, a LoRA trained to edit earlier.
- The L4 probe measured the real speed at 33–39 tok/s. Re-run at 35 tok/s: ~42% pass, empty 17–29%. v18d ≈ v19b ≈
  v20; "never edits" was mostly an artefact. E059–E061, L4 probe v3.
- Figure: failure-mode stacks, 12 vs 35 tok/s, same agents.

**F2. Budget control trades empty patches for wrong patches.**
- v20b vs v21 vs v21b on eval113 at 35 tok/s (E064/E065/E066): passes **43 / 43 / 42**.
- Wrong patches 55 / 41 / 50; empty 15 / 29 / 21; timeouts 26 / 40 / 34.
- v21 (time-only "edit now", fired ~call 16): fewer wrong patches, more empty ones.
- v21b (call-10 trigger back): every failure count between the two, same passes.
- When the pressure to edit fires decides the failure type, not the solve rate.
- Needs: repeats (2 more runs per arm) for a significance claim.

**F3. Imitation of oracle trajectories teaches the action, not the judgement.**
- rft3 (1,199 oracle + relabelled + self-rationalised sequences, 10.2M tokens).
- Probe on 241 states: P(edit | gold code visible) 0.11 → 0.49. A real-speed A/B: 16 → 14/37.
- Edit calls 105 → 190, failed edits 32 → 65, timeouts 6 → 11.
- Contrast: on-policy RFT (rft1/rft2) gave safe gains in format and speed. rft2: malformed calls 6.7% → 1.9%.

**F4. Local–leaderboard gap and a contamination gradient.**
- Local ~42% vs leaderboard 0.10 (6/58); other teams report 0.23 → 0.12.
- Pass rate by issue date over ~400 real-speed runs (E059, E060, E064, E065): ≤2024 52%, 2025 H2 41%, 2026 32%.
- Same-agent noise: 8/12 vs 2/12 on different nights (p≈0.04). A local "win" (v16 > v14, 40.3% vs 35.8%) lost on
  the leaderboard (0.06 vs 0.10).
- Needs: a per-task model (logistic regression with task effects, or just repo + date), CIs, and ideally a fresh
  2026-only task set.

**F5. What the wrong patches are.**
- 134 wrong-patch runs, hand-classified (note 16): SPEC_GAP 49% (the tests expect behaviour the issue does not
  state), WRONG_LOGIC 19%, WRONG_LOCATION 14%, INCOMPLETE 11%, BROKE_OTHER 4%, EDIT_DAMAGE 1%.
- The tool-level failures that dominated early (edit failures 34–70%) now cause ~1.5% of wrong patches (EDIT_DAMAGE) after interface work
  (toolbench: robust intended edits 67 → 90%).
- So the remaining headroom is model judgement + specification, not tooling.

Optional sixth: **narrow interface vs stock tools** (v1–v7: 70% edit failures, loops of 29/53 identical calls with
thinking off). Needs a stock-tools run at real speed to be a fair comparison.

## Experiments still needed (cheap first)

1. E066 final (running) → F2 table.
2. Re-analysis of existing runs for F4 (no GPU): date/repo regression over all real-speed runs.
3. Repeats for F2: v20b / v21 / v21b × 2 more on eval113 (~$15).
4. Stock-tools baseline at real speed (~$5), for the optional F6.
5. Kaggle LB results for v20b (and v21b if submitted) as the external check.

## Related work to cite (verify every reference before use)

- SWE-bench (Jimenez et al. 2024); SWE-bench Verified; contamination / fresh-task benchmarks (SWE-rebench,
  SWE-bench+).
- SWE-agent, agent-computer interfaces (Yang et al. 2024); OpenHands; Agentless (Xia et al. 2024).
- SWE-Gym, SWE-smith (training data for SWE agents); STaR / rejection-sampling fine-tuning; LoRA (Hu et al. 2021).
- The harness-engineering survey we read (arXiv 2609.00006).
- DAgger (Ross et al. 2011), for relabelled states.

## Figures

1. Failure-mode stacked bars (pass / wrong / empty) per experiment: 12 vs 35 tok/s (F1), v20b / v21 / v21b (F2).
2. Pass rate by issue date (F4).
3. rft3: probe edit rate vs real outcomes (F3).
4. A footer example and the per-task call timeline (system figure).

## Open decisions (user)

- One write-up or two? A second could be a short "evaluation pitfalls" paper (F1 + F4) aimed at the
  "Tasks & Benchmarks" topic.
- Release code/logs publicly (a public notebook or repo with scripts only, no competition data)?
- Kaggle Writeup text vs an arXiv-style PDF.
