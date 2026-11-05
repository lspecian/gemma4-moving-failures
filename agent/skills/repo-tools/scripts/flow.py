"""Flow engine: working memory for the repo-tools scripts and the rules that steer the agent's next call.

The model is good at *content* (which function, what change) and bad at *flow* (when to stop exploring, when to
edit, when to submit). Each script therefore records what it did in a small sqlite file and prints a footer of
at most 3 lines: an optional `BUDGET:` line (the agent's only clock), then `NEXT:` lines with pre-filled,
copy-pasteable calls. Reads and reproduction runs are mostly not refused (v13): the footer steers instead. A few calls
are not run: an identical repeat (its stored output is replayed), checks beyond the hard cap after an edit, pre-edit
snippets past the snippet gate (3c), reads and snippets once the context is nearly full with an edit in place (10),
and whatever a phase hook refuses.

State: $REPO_TOOLS_DIR/state.sqlite (default /workspace/.git/repo_tools, v19b). The state resets itself when the workspace
(path + git HEAD) changes. Stdlib only; every public function catches all errors and degrades to "no footer".

Rules (thresholds are the constants below):
  1. Order: before locate.py has run, look-around scripts (search/show/outline/callers/tests_for/history) run, and
     their footer's first NEXT line is a pre-filled locate.py call (v12 refused them).
  2. Repeat: an identical call (script + canonical args, see canon(): the object form {path: a.py, ...} and
     the positional form ["a.py", ...] are the same call) with no successful edit since then is not rerun; the
     stored output is replayed with "already done at call N". A call whose stored result was a usage error or a
     refusal is not a repeat: it runs (it may succeed now).
  3. Exploration budget (soft, v13): from N_EXPLORE_FIRST look-around calls since locate (before any edit attempt),
     or N_EXPLORE since the last edit attempt (N_EXPLORE_THIN in hypothesis mode), every further look-around call
     still runs, and its footer's first NEXT line is the pre-filled line-range edit of the region most likely to
     need the change (pick_regions()), saying how many reads there were since locate; the second line points at
     that code ("the code most likely to change: <file> <symbol> lines a-b (shown at call N)") and gives the
     old-text form. (E025: v12's refusals with the code were followed by an edit 1 time in 47; the thinking showed
     targeted questions the refusal did not answer.) locate.py calls beyond MAX_LOCATES and run.py snippets that
     only read repository source (snippet_reads()) count as look-around. Before the first edit, other run.py calls
     are soft-capped the same way: from MAX_RUNS_BEFORE_EDIT runs (MAX_RUNS_BEFORE_EDIT_THIN in hypothesis mode) the
     run still runs and its footer's first NEXT is "N runs and no edit yet: edit now ..." with the line-range edit.
  3b. Never refuse twice in a row: when the previous call was refused (any rule, a phase hook, or a rule-2
     replay), the next call runs normally; its footer still pushes the edit. So refusal streaks cannot happen.
  4. Edit cascade: after MAX_FAILED_EDITS consecutive failed edits the footer gives the exact line-number edit
     (path, start, end, new) for the closest region reported by edit.py.
  5. Verify cap: after MAX_CHECKS_PER_EDIT checks since the last successful edit the footer says check_patch then
     submit; check number MAX_CHECKS_HARD + 1 since the last edit is refused.
  6. After a passing check (exit 0) with an edit in place: check_patch.py then submit_patch.
  7. Thin issue: locate reports <= THIN_MAX_IDENTIFIERS identifiers, or LOW CONFIDENCE with
     <= THIN_LOWCONF_MAX_IDENTIFIERS identifiers (or facts thin=True) -> mode "hypothesis": add 2-3 hypotheses
     with hypothesis.py, test each with run.py snippet, confirm/reject, edit where confirmed. v19: off
     (HYPOTHESIS_MODE = False; thin is still recorded); the mode starts only when the model adds a hypothesis itself.
  8. Footer: at most FOOTER_LINES lines, each starting "NEXT:" (after the BUDGET line), with pre-filled calls in
     the quote-free notation of argument schema v3, e.g.   edit.py  target=a/b.py  start=12  end=14  new=<new text>
     (the object args {target, start, end, new}). Never JSON and never arrays: copied JSON quotes and array args
     break Kaggle's vLLM Gemma 4 tool parser (hang, or file_path/skill_name swallowed into args).
     Edit early: after a confident locate that printed its best function in full (or with only its docstring
     folded), and after a show.py of a region in one of locate's top files, the first NEXT is
     the line-range edit of that region (old-text form second). v19: when locate printed no code, or only a window
     of a long function, the first NEXT is the show.py read of that function (locate_lines()), never an edit of code
     the model has not seen; the second is ONE other candidate (another file when locate's confidence is low).
  9. Budget: from call BUDGET_SHOW_FROM_CALL the footer's first line is "BUDGET: call N of ~TARGET_CALLS |
     XmYYs of 4m30s used" (counted in the FOOTER_LINES). Past BUDGET_EDIT_FRAC with no successful edit the NEXT lines
     become "edit your best candidate now"; past BUDGET_FINISH_FRAC they become check_patch then submit_patch.
  v16 (research/notes/11-failures-all-tasks.md, levers L1, L2, L4, L9):
  3c. Probing (L1): run.py snippets before the first successful edit are counted. A snippet "did not reproduce" when
     it exited 0 and its output (without NOTE/exit lines) has no traceback and no ...Error/...Exception word, or when
     its output equals an earlier pre-edit snippet's (nothing new learned). From NOREPRO_NUDGE such snippets, a
     snippet's footer is "N snippets did not reproduce it: sweep the invariant over many inputs in ONE snippet, or edit
     the most suspicious line now: <line-range edit of the best region>". v17 (L5, the snippet gate): from
     MAX_SNIPPETS_BEFORE_EDIT pre-edit snippets on, every pre-edit snippet is refused until the first successful edit
     ("NOT RUN: N snippets and no edit yet; snippets are refused until you edit. The code most likely to change: <file>
     <function> lines a-b." + its line-range edit); rule 3b still applies, so every second one runs, and its footer
     says "N snippets and no edit yet: the next snippet is not run until you edit" with the same edit. (v16 refused
     once per task; the model then sent 12-20 snippets and never edited: rich_3063/3506/3180.) Snippets that only
     read source ('read') and run.py tests are not snippets here.
  4b. Several sites (L4): when edit.py's OLD matches several places (fact sites), the footer names them and gives
     the line-range edit of the first; after a successful edit whose changed line is still elsewhere in the file
     (fact also), the first NEXT is "if <file>:<line> <function> has the same bug, fix it too: <line-range edit>".
  10. Context (v16 L9, recalibrated in v17 L1; v18c: the sawtooth of v18 A1, fit: scripts/fit_ctx_sawtooth.py): the
     characters of every result, footer, refusal and argument since the last presumed compaction (rule 13) are summed
     (meta ctx_chars) and, with the number of calls since then, give the central estimate of the next prompt's tokens
     (ctx_central(): CTX_BASE_TOKENS, or CTX_POST_BASE_TOKENS after a compaction, + CTX_TOKENS_PER_CHAR * chars +
     CTX_TOKENS_PER_CALL * calls); ctx_estimate() adds CTX_MARGIN_TOKENS (conservative). context_tokens() is that
     conservative estimate for other scripts (it drops after a presumed compaction). From CTX_WARN_TOKENS the NEXT
     lines become "context nearly full: make your edit
     now: <line-range edit>" (no edit yet; a failed edit keeps its repair line) or "context nearly full: check_patch.py,
     then submit_patch now" (edit in place). From CTX_STOP_TOKENS every look-around, locate and run.py snippet call
     is refused ("NOT RUN: context nearly full ...", also right after another refusal: rule 3b does not apply), while
     edits, check_patch and run.py tests run; without an edit (v17b) the refusal's NEXT is the edit. A context overflow (CTX_LIMIT_TOKENS) is graded with an EMPTY patch (8 runs of E036/E037 lost their edits).
  8b. Compact footers (v17 L1; footers were 20% of all tool output): from FOOTER_COMPACT_FROM_TOKENS estimated, at
     most COMPACT_NEXT_LINES NEXT lines, secondary lines dropped (SECONDARY: the old-text alternative, explanations),
     each line cut to COMPACT_LINE_CHARS by fit_line() without cutting its call, and a short BUDGET line. v18c: the
     footers keep v17b's trigger exactly: v17b's cumulative estimate (ctx_estimate_v17b(): the V17B_CTX_* fit over all
     characters and calls of the task, never reset by a compaction) >= FOOTER_COMPACT_FROM_TOKENS (15000), so a
     footer is v17b's for the same calls. show.py/search.py read context_tokens() (the sawtooth, rule 10) against
     COMPACT_FROM_TOKENS for their own compact output.
  12. Tests after an edit (v17 L6): after a successful edit with at least POST_EDIT_TEST_LEFT_S seconds left (and
     3 calls), the NEXT order is: re-run the repro snippet of before the edit (State.repro, if one ran), then the
     existing tests that mention the edited function (edited_names(): the functions the edit's diff touched, then
     their class; related_tests(): the test file under the workspace with the most mentions, scanned within
     TEST_SCAN_S, and a k= expression naming its tests that mention it), then check_patch.py and submit_patch. The
     same tests are not suggested again once that file ran after the edit; a passing repro re-run points at them;
     a failing test run is not cut short by rule 5's soft cap (it may show a regression of the edit); in the budget's
     finish phase the tests stay the next call while the time allows. (fastapi_14964/14986, rich_3130 broke existing
     tests with 100+ s left.)
  11. Backslashes (L2): results are JSON-encoded, so the model sees every backslash doubled. When a show/run/search/
     locate result has a backslash in code or program output, BACKSLASH_NOTE is printed above the footer (at most
     BACKSLASH_NOTES times per task).
  13. Compaction (ported from v18 A into v18c; the footer rules above are v17b's, unchanged). Kaggle's scorer compacts
     the conversation: before a model call, once the previous prompt had >= 14,336 tokens, every event but the last 5
     is replaced by an LLM summary of their text parts (tool calls and results are dropped). The sqlite state
     survives; the model's memory of tool results does not. flow models this as a sawtooth (rule 10): when the central
     estimate of the prompt that produced call n was >= COMPACT_AT - COMPACT_MARGIN_TOKENS (REPO_TOOLS_COMPACT_AT
     overrides COMPACT_AT; 0 turns the model off), the request after call n is compacted: compacted_at = n, results
     of calls before n - CTX_VISIBLE_CALLS + 1 are presumed gone (State.visible()), and the sums restart from the last
     CTX_KEEP_CALLS calls. Then:
     - MEMO: a line of <= MEMO_CHARS characters built from sqlite with current line numbers (the code to change, or
       the edited ranges from git diff; the last repro/test verdict; patch READY or not) is printed on its own line
       above the footer of call n (the first result the model sees after the compaction), and once per sawtooth
       cycle when the central estimate first reaches MEMO_AT_TOKENS. status.py prints the same MEMO.
     - Rule 2 replays only results still in context: an identical call whose stored result is gone runs normally,
       and such a recovery call (any look-around call, or a show.py range only printed before the compaction) does
       not count toward the exploration budget (rule 3).
     - check_patch.py prints the full diff (<= 30 lines) when the last edit's own output is gone.
  P5 (v18): edit.py's closest region (fact similarity) below CLOSEST_MIN_SIMILARITY is not suggested, and a failed
     edit without a closest region clears the earlier one (it no longer describes the code being edited).
  ADK separator (v18b): ADK puts a bare '--' before positional_args; flow.main drops it (strip_separator()).

Region choice (pick_regions / best_region, used by the over-budget footer and every "edit now" suggestion): regions the
model viewed in one of locate's top TOP_CANDIDATE_FILES source files, by locate's file rank (within a file: locate's
own candidate functions first, then the latest viewed); else locate's top candidate function; else the latest
viewed source region. A viewed range is widened to the innermost function containing its first line; a function
longer than PAYLOAD_LINES is shown as its head plus a window at the lines the model viewed inside it (else at its
body, after the docstring), and that window is the line range to edit. Tests/docs/examples are never chosen.

Phase hook (for multi-agent variants). A sibling module phase.py, normally absent, may steer the flow:
    phase.before(st, script, argv) -> None or text      called first in before(); text is printed instead of
                                                         running the script and recorded as a refusal ('phase'),
                                                         so rule 3b and rule 2 treat it like any refusal. Flow's
                                                         own rule 3b does not override a phase refusal.
    phase.footer(st, script, status, lines) -> lines     called on the NEXT lines just before they are formatted
                                                         (after budget escalation; refusal footers too, with status
                                                         'refused'); returns the replacement list.
Both calls are wrapped in try/except: a broken phase module never breaks a script (it is then ignored).
State attributes a phase module may rely on (read-only unless noted): n (number of this call in before(); of the
last recorded call in after()), actions [(n, script, key, status, summary)], real (actions not refused/usage),
last_refused (the last recorded action was refused: in before() that is the previous call), located, mode
('normal'|'hypothesis'), candidates [(path, symbol, start, end, score, evidence)], viewed [(path, start, end, n,
label)] newest first, edits [(n, path, ok, message)], has_edit, edited_paths, last_edit_attempt, last_edit_ok,
failed_streak, look_since_edit, explore_budget, checks [(n, kind, target, exit_code)], checks_since_edit,
hypotheses, patch_ready, first_ts, now, compacted_at (0: no compaction presumed yet), visible_from (results of
calls before it are gone; visible(n)), ctx_central/ctx_tokens (rule 10), memo (the MEMO line format_footer prints,
or None), and conn (the open sqlite connection: a phase may keep its own keys with get_meta/set_meta(st.conn,
'phase:...', value)).

Integration (for the lead). Scripts import this module from their own directory (runpy does not add it to
sys.path) and must disable bytecode first, or __pycache__ lands in agent/:

    import os, sys
    sys.dont_write_bytecode = True
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import flow

Simplest hook: replace the `sys.exit(main(sys.argv[1:]))` line with

    if __name__ == '__main__':
        sys.exit(flow.main('show.py', main))

flow.main() calls before(); if it refuses, prints the refusal and returns 0; otherwise it runs main(argv) while
teeing stdout, then calls after() with the exit code, the captured output and the facts, and prints the footer.
Facts are parsed from the output when a script does not supply them (parse_facts below documents the formats it
reads), and a script can supply or override them from inside main() with flow.fact(**kw). Per script:

    locate.py      flow.fact(candidates=[(path, symbol, start, end, score), ...],  # best first
                             identifier_count=len(q['idents']), low_confidence=low,
                             shown=(path, start, stop))                            # the function printed
                   (optional: thin=True/False to override rule 7's heuristic)
    show.py        flow.fact(path=rel, start=start, end=stop)
                   (a long class or a file outline: next_show=(rel, qualname) of the definition to read next;
                    an outline with nothing to suggest: outlined=rel)
    search.py      flow.fact(hits=[(rel, line), ...])                              # optional
                   (nothing matched an identifier but a definition exists: next_show=(rel, qualname))
    outline.py     flow.fact(paths=[rel, ...])                                     # optional
    edit.py        flow.fact(ok=True/False, path=rel, closest=(rel, start, end))   # closest only when NOT APPLIED
                   (parsed from 'EDITED ...' / 'NOT APPLIED' / 'CLOSEST: path start end' if absent)
                   (v18 P5: similarity=<float 0-1> of the closest region; below CLOSEST_MIN_SIMILARITY it is ignored,
                    parsed from '(similarity 0.55)' if absent; missing = unknown, used)
                   (v15b: an edit refused by a guard adds guard='property'|'import', guard_line=line of the
                    assignment (property), closest=(rel, start, end) (import))
                   (v17: guard='definition' with closest=(rel, start, end): the edit would delete a def/class/
                    property the code still uses; closest is the narrower range that keeps it)
    create.py      flow.fact(ok=True/False, path=rel)
    run.py         flow.fact(exit_code=code_or_'TIMEOUT', kind='test'|'snippet'|'read', target=file_or_None)
                   (kind 'read': a snippet that only reads repository source, see snippet_reads(); counted as
                    look-around, not as a check)
    check_patch.py flow.fact(ready=True/False, empty=not changed and not untracked)
    status.py / hypothesis.py use this module directly (they are never refused).

Arguments: every script takes the object form (argv ["--key", "value", ...], keys per script in KEYS) and its
old positional form. The rules only ever see canon(script, argv), so both forms count as the same call.

Drop the scripts' own trailing hints that the footer replaces (locate.py's 'next: ...' line), or they will
contradict it. Every call must go through the hook, or budgets and repeat detection undercount.

Explicit API (what flow.main does):

    refusal = flow.before(script, argv)            # None, or text to print instead of running (then exit 0)
    footer = flow.after(script, argv, status, summary, **facts)   # [MEMO] BUDGET/NEXT lines ('' on any error)
    tokens = flow.context_tokens()                 # rule 10's context estimate; >= flow.COMPACT_FROM_TOKENS: be brief
    text = flow.memo(flow.peek())                  # the MEMO line for the current state (rule 13)

`status` is the script's exit code (0 ok, 2 usage error, other = failed) or one of 'ok'/'usage'/'fail';
`summary` is the script's stdout (stored, capped, and replayed by rule 2).
"""

import contextlib
import io
import json
import os
import re
import sqlite3
import sys
import time
import warnings

# Every script imports flow, so this silences, process-wide, Python's warnings about the REPOSITORY's code that
# parsing it prints to stderr (kb/locate/show/search/outline/names/guards/check_patch: invalid escape sequences such
# as "\d", `is` with a literal, ...; 56% of the rft3 oracle locate results had them). Filters are per process:
# run.py's snippet and test subprocesses still print their own warnings.
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', message='invalid escape sequence', category=DeprecationWarning)  # Python < 3.12

try:  # optional sibling module of multi-agent variants (see "Phase hook" above); normally absent
    import phase
except ImportError:
    phase = None
except Exception:  # a broken phase module must never break a script
    phase = None

# ---------------------------------------------------------------- thresholds (all rules read these)
N_EXPLORE_FIRST = 3            # rule 3: look-around calls allowed after locate, before the first edit attempt
N_EXPLORE = 4                  # rule 3: look-around calls allowed since the last edit attempt
N_EXPLORE_THIN = 6             # rule 3 in hypothesis mode
MAX_LOCATES = 2                # locate calls beyond this count as look-around
MAX_RUNS_BEFORE_EDIT = 2       # rule 3: run.py calls before the first successful edit before the footer pushes it
MAX_RUNS_BEFORE_EDIT_THIN = 4  # ... in hypothesis mode (runs test the hypotheses)
MAX_FAILED_EDITS = 2           # rule 4: consecutive failed edits before the --lines instruction
MAX_CHECKS_PER_EDIT = 2        # rule 5: checks since the last edit before "check_patch then submit"
MAX_CHECKS_HARD = 3            # rule 5: checks since the last edit before further ones are refused
THIN_MAX_IDENTIFIERS = 0       # rule 7: this many identifiers or fewer -> thin issue
THIN_LOWCONF_MAX_IDENTIFIERS = 1  # rule 7: with LOW CONFIDENCE, this many or fewer -> thin issue
HYPOTHESIS_MODE = False        # v19: rule 7 off. E053-E056 (300 runs): thin -> hypothesis mode in 42% of runs, pass
                               # 17.5% vs 27.6%, empty patch 70% vs 49%, first edit call 8 vs 5, hypothesis.py used 19
                               # times in 126 runs; same task in both modes (13 tasks): edited 22% vs 50%. The mode
                               # still starts when the model adds hypotheses on its own.
FOOTER_LINES = 3               # rule 8
LINE_CHARS = 480               # cap per footer line (calls are kept shorter; see VALUE_CHARS)
LINES_SUGGEST_MAX = 40         # line-range edits longer than this get "narrow start/end" advice
SUMMARY_CHARS = 4000           # stored output per call
REPLAY_LINES = 25              # output lines replayed by rule 2
VIEWED_LIST = 4                # viewed regions listed by rule 3
# Region choice (and the line range an over-budget footer suggests editing)
TOP_CANDIDATE_FILES = 3        # viewed regions in locate's top files (by rank) are preferred
CLOSE_SCORE = 0.15             # second file within this fraction of the top file's score -> two regions
PAYLOAD_LINES = 60             # longest line range suggested for one edit (a window of a long function)
PAYLOAD_HEAD = 6               # head lines of a long function before its focus window
# Budget awareness (rule 9). The harness never shows budget warnings on run_skill_script results, so the flow
# engine is the agent's only clock. Progress = max(elapsed / TIME_BUDGET_S, calls / TARGET_CALLS), where elapsed
# runs from the first recorded call and calls counts every script call (refusals included).
TARGET_CALLS = 24              # calls a task should take at most (v22: 24 of the 28-call cap: edit now at call 12,
                               # finish at ~19, as the public 0.18 bundle's prompt: edit by call 12, no exploring after 20)
TIME_BUDGET_S = 460            # v22: eval_config max_time_minutes 8 = 480 s (v20b: 4.0 min, 220), minus ~20 s the harness clock runs before
                               # the first script call (edit-failure analysis, note 15)
BUDGET_EDIT_FRAC = 0.5         # past this with no successful edit: "edit your best candidate now"
BUDGET_FINISH_FRAC = 0.8       # past this: "run check_patch.py and submit_patch now"
BUDGET_SHOW_FROM_CALL = 3      # the BUDGET line appears from this call on (or earlier once escalated)
# v16 (research/notes/11-failures-all-tasks.md)
NOREPRO_NUDGE = 2              # L1: pre-edit snippets that did not reproduce before the footer says sweep-or-edit
MAX_SNIPPETS_BEFORE_EDIT = 5   # L1/v17 L5: from here every pre-edit snippet is refused until an edit (rule 3b alternates)
# v18 A1, ported into v18c (research/notes/13-tool-efficiency.md; scripts/fit_ctx_sawtooth.py): the sawtooth context
# estimate. Least squares over the 5570 model calls of E041 (v17b, eval113 x2, local, never compacted) + E042 (v17b,
# Kaggle-like with the scorer's compaction): prompt_tokens ~ 5200 + 0.2953 * chars + 336 * calls, where chars = raw
# stdout + argument characters of the run_skill_script calls before it and calls = their number (ADK-rejected calls
# and the model's text are in the per-call term). Residual sd 1109, p95 +1948. Right after a compaction the prompt is
# ~3380 + the same two terms over the last 3 calls (sd 550): system prompt, summary and the last 5 events.
CTX_BASE_TOKENS = 5240         # system prompt, tools, harness message with the issue and the directory tree
                               # (fit on v17b traces: 5200; v18c's prompt is v17b's plus ~40 tokens)
CTX_TOKENS_PER_CHAR = 0.2953   # ~1 token per 3.4 raw characters (JSON escaping included)
CTX_TOKENS_PER_CALL = 336      # thinking, call text and the rejected calls flow never sees
CTX_MARGIN_TOKENS = 2000       # ~p95 residual: context_tokens() and rule 10 are conservative
COMPACT_AT = int(os.environ.get('REPO_TOOLS_COMPACT_AT', '14336') or 0)  # the scorer's threshold; 0: no compaction
COMPACT_MARGIN_TOKENS = 0      # a compaction is presumed when the central estimate is within this of COMPACT_AT
CTX_POST_BASE_TOKENS = 3420    # after a compaction: system prompt + tools + the summary (+ the kept calls' terms)
CTX_KEEP_CALLS = 3             # calls whose characters are still counted after a compaction (5 events kept)
CTX_VISIBLE_CALLS = 2          # results presumed still visible after a compaction (the newest ones)
MEMO_AT_TOKENS = 12500         # rule 13: MEMO once per cycle when the central estimate reaches this
MEMO_CHARS = 250               # rule 13: cap of the MEMO line
CLOSEST_MIN_SIMILARITY = 0.6   # P5: edit.py's closest region below this similarity is not suggested
CTX_LIMIT_TOKENS = 24576       # hard limit: 32768 - 8192 output (v22; v20b 1536 -> 31232): vLLM refuses a longer prompt
CTX_WARN_TOKENS = 20500        # rule 10: NEXT = edit now / check_patch then submit now (v22: limit - 4076, was 24000;
                               # compaction at 14,336 keeps the conservative estimate below ~17k in normal runs)
CTX_STOP_TOKENS = 22500        # rule 10: reads and snippets are refused (no 3b) (v22: limit - 2076, was 27500)
COMPACT_FROM_TOKENS = 14000    # show.py/search.py print compact output from this conservative (sawtooth) estimate
                               # (central ~12k: close to the compaction threshold)
# Rule 8b's footer trigger is v17b's, unchanged (v18c: footers are v17b's): v17b's fit over 4028 calls of E037,
# prompt_tokens ~ 5094 + 0.3147 * chars + 284 * calls + a 2800 margin, over every call of the task (no sawtooth).
V17B_CTX_BASE_TOKENS, V17B_CTX_TOKENS_PER_CHAR, V17B_CTX_TOKENS_PER_CALL, V17B_CTX_MARGIN_TOKENS = 5094, 0.3147, 284, 2800
FOOTER_COMPACT_FROM_TOKENS = 15000  # rule 8b: compact footers from this v17b estimate
COMPACT_NEXT_LINES = 2         # rule 8b: NEXT lines in a compact footer
COMPACT_LINE_CHARS = 260       # rule 8b: cap per compact footer line
# v17 L6: tests after an edit
POST_EDIT_TEST_LEFT_S = 45     # suggest the existing tests of the edited function with at least this much time left
TEST_SCAN_S = 0.5              # bounded scan of the test files
TEST_SCAN_FILES = 1500         # test files read at most
TEST_SCAN_BYTES = 400000       # larger test files are skipped
TEST_K_NAMES = 4               # test names joined into one k= expression
TASK_MAX_CALLS = 28            # eval_config max_tool_calls (v22; v20b 40): no test suggestion when fewer than 3 calls are left
SKIP_DIRS = {'.git', '.hg', '.svn', '.tox', '.nox', '.venv', 'venv', 'env', 'node_modules', '__pycache__', 'build',
             'dist', 'site-packages', '.mypy_cache', '.pytest_cache', 'docs', 'docs_src', 'site', '.eggs'}
BACKSLASH_NOTES = 2           # L2: the JSON-escaping note is shown at most this many times per task
BACKSLASH_SCRIPTS = {'show.py', 'run.py', 'search.py', 'locate.py'}
BACKSLASH_NOTE = ('note: results are JSON-encoded, so every backslash of the file is shown doubled here (two shown = '
                  'one in the file); do not spend calls testing how backslashes print')
REPRO_MARK = re.compile(r'Traceback \(most recent call last\)|\b\w*(?:Error|Exception)\b')
META_LINE = re.compile(r'^(?:NOTE:|USAGE ERROR|exit code:|NEXT:|BUDGET:|note:|MEMO:)')

LOOK_AROUND = {'search.py', 'show.py', 'outline.py', 'callers.py', 'tests_for.py', 'history.py'}
EDITS = {'edit.py', 'create.py'}
NEVER_REFUSED = {'status.py', 'hypothesis.py'}
AUX = re.compile(r'(^|/)(tests?|testing|docs|docs_src|examples?|benchmarks)/|(^|/)test_[^/]*\.py$|_test\.py$|'
                 r'(^|/)conftest\.py$')
# v19b/v20: the subprocess sandbox has no /workspace; ADK runs a skill from a temp cwd, the task workspace is on
# PYTHONPATH (and next to $TMPDIR in the sandbox root). L4 probe v4: the cwd fallback indexed scripts/ itself.
ROOT_DEFAULT = '/workspace' if os.path.isdir('/workspace') else next(
    (p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep)
     + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')]
     if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())

SCHEMA = """
CREATE TABLE IF NOT EXISTS actions(n INTEGER PRIMARY KEY, script TEXT, args TEXT, key TEXT, status TEXT,
                                   summary TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS viewed(path TEXT, start INTEGER, "end" INTEGER, n INTEGER, label TEXT);
CREATE TABLE IF NOT EXISTS candidates(rank INTEGER, path TEXT, symbol TEXT, start INTEGER, "end" INTEGER,
                                      score REAL, evidence TEXT);
CREATE TABLE IF NOT EXISTS hypotheses(id INTEGER PRIMARY KEY, text TEXT, status TEXT DEFAULT 'untested',
                                      evidence TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS edits(n INTEGER, path TEXT, ok INTEGER, message TEXT);
CREATE TABLE IF NOT EXISTS checks(n INTEGER, kind TEXT, target TEXT, exit_code TEXT);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""
SCHEMA_VERSION = 3              # v18c: v17b's tables (v18's viewed.focus/kind are not used); a state written by another
                                # version is dropped and recreated
TABLES = ('actions', 'viewed', 'candidates', 'hypotheses', 'edits', 'checks', 'meta')

_FACTS = {}


# ---------------------------------------------------------------- storage

def state_dir():
    """$REPO_TOOLS_DIR, else <workspace>/.git/repo_tools (v19b: per task by construction - the harness rebuilds
    /workspace for every task, and .git is never part of the patch), else /tmp/repo_tools. A shared /tmp let the
    working memory of one task leak into the next under the subprocess sandbox (L4 probe: call 52/~20 at a task's
    first call, every read refused as 'context nearly full')."""
    d = os.environ.get('REPO_TOOLS_DIR')
    if d:
        return d
    git = os.path.join((os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd()))), '.git')
    return os.path.join(git, 'repo_tools') if os.path.isdir(git) else '/tmp/repo_tools'


def root():
    return os.environ.get('WORKSPACE', ROOT_DEFAULT)


def _head(ws):
    """Cheap git HEAD sha (no subprocess)."""
    try:
        head = open(os.path.join(ws, '.git', 'HEAD')).read().strip()
        if not head.startswith('ref:'):
            return head
        ref = head[4:].strip()
        p = os.path.join(ws, '.git', ref)
        if os.path.exists(p):
            return open(p).read().strip()
        for line in open(os.path.join(ws, '.git', 'packed-refs')):
            if line.rstrip().endswith(' ' + ref):
                return line.split()[0]
    except OSError:
        pass
    return ''


def connect():
    d = state_dir()
    os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(os.path.join(d, 'state.sqlite'), timeout=5, isolation_level=None)
    if conn.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION:
        conn.executescript(''.join(f'DROP TABLE IF EXISTS {t};' for t in TABLES) + SCHEMA
                           + f'PRAGMA user_version = {SCHEMA_VERSION};')
    ws = os.path.realpath(root())
    key = ws + '@' + _head(ws)
    row = conn.execute("SELECT value FROM meta WHERE key='workspace'").fetchone()
    if row is None or row[0] != key:
        conn.execute('BEGIN IMMEDIATE')
        for t in TABLES:
            conn.execute(f'DELETE FROM {t}')
        conn.execute("INSERT INTO meta VALUES('workspace', ?)", (key,))
        conn.execute('COMMIT')
    return conn


def get_meta(conn, key, default=None):
    row = conn.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
    return row[0] if row else default


def set_meta(conn, key, value):
    conn.execute('INSERT OR REPLACE INTO meta VALUES(?, ?)', (key, value))


def ctx_estimate(chars, calls, base=None):
    """Rule 10: conservative estimate of the model's prompt tokens when `calls` recorded calls put `chars` characters
    (outputs, footers, refusals, arguments) into the context since `base` (CTX_BASE_TOKENS at the start of the task,
    CTX_POST_BASE_TOKENS after a compaction); fit and margin: the CTX_* constants."""
    return int(ctx_central(chars, calls, base) + CTX_MARGIN_TOKENS)


def ctx_central(chars, calls, base=None):
    """The fitted (central) estimate of the prompt tokens, without the margin (the compaction model uses it)."""
    return (CTX_BASE_TOKENS if base is None else base) + CTX_TOKENS_PER_CHAR * chars + CTX_TOKENS_PER_CALL * calls


def ctx_estimate_v17b(chars, calls):
    """Rule 8b's trigger (v17b's estimate, unchanged): every character and call of the task, never reset."""
    return int(V17B_CTX_BASE_TOKENS + V17B_CTX_TOKENS_PER_CHAR * chars + V17B_CTX_TOKENS_PER_CALL * calls
               + V17B_CTX_MARGIN_TOKENS)


def _ctx_parts(conn):
    """(chars, calls, base) of the context since the last presumed compaction (rules 10 and 13)."""
    chars = int(get_meta(conn, 'ctx_chars', '0') or 0)
    since = int(get_meta(conn, 'ctx_from', '0') or 0)
    calls = conn.execute('SELECT COUNT(*) FROM actions WHERE n > ?', (since,)).fetchone()[0]
    base = CTX_POST_BASE_TOKENS if int(get_meta(conn, 'compacted_at', '0') or 0) else CTX_BASE_TOKENS
    return chars, calls, base


def context_tokens():
    """The conservative context estimate for the current task (rule 10, sawtooth: it drops after a presumed
    compaction), for scripts that shorten their output from COMPACT_FROM_TOKENS on (inside a script's main() it covers
    the calls before this one); 0 on any error."""
    try:
        conn = connect()
        try:
            return ctx_estimate(*_ctx_parts(conn))
        finally:
            conn.close()
    except Exception:
        return 0


def _ctx_step(conn, n, chars, prev):
    """Account call n (its output, refusal and arguments: `chars`; the footer follows with _add_ctx). `prev` is the
    central estimate of the prompt that produced call n: from COMPACT_AT - COMPACT_MARGIN_TOKENS on, the scorer
    compacts before the next request, so compacted_at = n and the sums restart from the last CTX_KEEP_CALLS calls."""
    recent = (_recent(conn) + [int(chars)])[-CTX_KEEP_CALLS:]
    set_meta(conn, 'ctx_recent', json.dumps(recent))
    set_meta(conn, 'ctx_chars_all', str(int(get_meta(conn, 'ctx_chars_all', '0') or 0) + int(chars)))
    if COMPACT_AT and prev + COMPACT_MARGIN_TOKENS >= COMPACT_AT:
        set_meta(conn, 'compacted_at', str(n))
        set_meta(conn, 'compactions', str(int(get_meta(conn, 'compactions', '0') or 0) + 1))
        set_meta(conn, 'ctx_from', str(max(0, n - CTX_KEEP_CALLS)))
        set_meta(conn, 'ctx_chars', str(sum(recent)))
    else:
        set_meta(conn, 'ctx_chars', str(int(get_meta(conn, 'ctx_chars', '0') or 0) + int(chars)))


def _recent(conn):
    """Characters of the last CTX_KEEP_CALLS calls (result + arguments + footer), oldest first."""
    try:
        v = json.loads(get_meta(conn, 'ctx_recent', '[]') or '[]')
        return [int(x) for x in v] if isinstance(v, list) else []
    except (ValueError, TypeError):
        return []


# ---------------------------------------------------------------- normalisation

def script_name(script):
    s = os.path.basename(str(script or '')).strip()
    return s if s.endswith('.py') else s + '.py'


def clean_path(p):
    p = str(p).strip().strip('"\'')
    p = p.lstrip('/')
    return p[len('workspace/'):] if p.startswith('workspace/') else p


def _unquote(s):
    s = str(s).strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in '"\'':
        s = s[1:-1].strip()
    return s


def norm_args(argv):
    """Arguments as the script means them: a JSON list packed into one string is unpacked, and a JSON object
    packed into one string becomes the object form ["--key", "value", ...]."""
    args = [str(a) for a in (argv or [])]
    if len(args) == 1 and args[0].strip()[:1] in ('[', '{'):
        try:
            v = json.loads(args[0])
            if isinstance(v, list):
                args = [str(a) for a in v]
            elif isinstance(v, dict):
                args = [x for k, val in v.items() for x in ('--' + str(k), str(val))]
        except ValueError:
            pass
    return args


# Argument schema v2 (research/notes/6-flow-design.md, "object args"). The model passes `args` as an OBJECT
# (array args can hang Kaggle's vLLM tool parser); run_skill_script turns {"k": "v"} into argv ["--k", "v", ...].
# Every script also keeps its positional form. canon() maps both forms to one dict of canonical keys, so repeat
# detection, budgets, edit/check counting and hypothesis mode see the same call either way, and cmd() renders
# every suggested call in the canonical object form.
KEYS = {
    'locate.py': ('query', 'k', 'show'),
    'show.py': ('file', 'symbol', 'start', 'end'),
    'outline.py': ('files', 'grep'),
    'search.py': ('text', 'regex', 'path'),
    'callers.py': ('name',),
    'tests_for.py': ('target',),
    'history.py': ('file', 'similar'),
    'edit.py': ('file', 'old', 'new', 'start', 'end') + tuple(f'{k}{n}' for n in range(2, 5)
                                                              for k in ('start', 'end', 'new', 'old')),
    'create.py': ('file', 'content'),
    'run.py': ('test', 'k', 'snippet'),
    'check_patch.py': ('clean',),
    'hypothesis.py': ('add', 'add2', 'add3', 'list', 'confirm', 'reject', 'evidence'),
    'status.py': (),
}
OBJ_KEY = re.compile(r'--([A-Za-z_]\w*)$')
ADD_KEY = re.compile(r'add\d*$')
PATH_KEYS = {'file', 'path', 'target', 'symbol', 'name'}   # quotes and /workspace/ stripped
TEXT_KEYS = {'text', 'regex', 'similar', 'grep'}           # quotes stripped, whitespace collapsed
FLAG_DEFAULTS = {'show': 'yes', 'clean': 'no', 'icase': 'no', 'word': 'no'}
YES = {'yes', 'y', 'true', '1', 'on'}
SEARCH_FLAGS = {'-E': 'regex', '--regex': 'regex', '-P': 'regex', '-i': 'icase', '--ignore-case': 'icase',
                '-w': 'word', '-F': None}
SEARCH_IGNORED = {'-r', '-R', '-n', '-rn', '-nr', '-rni', '-I', '-H', '--recursive', '--line-number'}


def object_args(args):
    """The object form as a dict ({"k": "v"} arrives as ["--k", "v"]), or None when argv is not of that shape.
    Pairs are read strictly by position: a value may itself start with '-' or '--' (code, diffs)."""
    if not args or len(args) % 2:
        return None
    out = {}
    for i in range(0, len(args), 2):
        m = OBJ_KEY.match(str(args[i]).strip())
        if not m:
            return None
        out[m.group(1)] = str(args[i + 1])
    return out


FILE_ALIASES = ('target', 'path', 'within', 'file_path', 'filepath', 'filename', 'file_name', 'dir', 'directory', 'in')
KEY_ALIASES = {'search.py': {'query': 'text'}}   # argfix.KEY_ALIASES: other names the model used for a key


def file_key(keys):
    """The internal key that names the repository file(s) of a script: file, path (search.py) or files."""
    return next((k for k in ('file', 'path', 'files') if k in keys), None)


def alias_keys(s, d):
    """argfix.alias_pairs for canon(): model-facing "target" (and aliases) -> internal "file" ("path" for search.py,
    "files" for outline.py); search.py's "query" -> "text"; ADK parameters that landed inside args (skill_name,
    file_path naming the script) dropped."""
    keys = KEYS.get(s, ())
    target = file_key(keys)
    extra = KEY_ALIASES.get(s, {})
    out = {}
    for n, v in d.items():
        if n == 'skill_name' or (n in ('file_path', 'filepath')
                                 and re.fullmatch(r'(?:scripts/)?' + re.escape(s), str(v).strip())):
            continue
        if target and n in FILE_ALIASES + ('file',) and n != target and target not in d:
            n = target
        elif n in extra and n not in keys and extra[n] not in d and 'regex' not in d:
            n = extra[n]
        out[n] = v
    return out


def _looks_path(s):
    s = s.strip()
    return s.endswith(('.py', '/')) or ('/' in s and ' ' not in s)


def _positional(s, args):
    """Canonical dict for the positional form of script s (argument lists as the v8 scripts took them)."""
    generic = {'argv': json.dumps(args, ensure_ascii=False)} if args else {}
    if s == 'locate.py':
        d, rest, i = {}, [], 0
        while i < len(args):
            a = args[i].strip()
            if a in ('--k', '-k') and i + 1 < len(args):
                d['k'], i = args[i + 1], i + 2
                continue
            if a == '--no-show':
                d['show'] = 'no'
            else:
                rest.append(args[i])
            i += 1
        if rest:
            d['query'] = ' '.join(rest)
        return d
    if s == 'show.py':
        toks = [t for t in (clean_path(_unquote(t)) for a in args for t in a.split()) if t]
        d = {'file': toks[0]} if toks else {}
        rest = toks[1:]
        rng = re.fullmatch(r'(\d+)\s*[-:,]\s*(\d+)', rest[0]) if len(rest) == 1 else None
        if len(rest) == 2 and rest[0].isdigit() and rest[1].isdigit():
            d.update(start=rest[0], end=rest[1])
        elif rng:
            d.update(start=rng.group(1), end=rng.group(2))
        elif len(rest) == 1 and rest[0].isdigit():
            d['start'] = rest[0]
        elif rest:
            d['symbol'] = ' '.join(rest)
        return d
    if s == 'outline.py':
        d, files, i = {}, [], 0
        while i < len(args):
            if args[i].strip() in ('--grep', '-g') and i + 1 < len(args):
                d['grep'], i = args[i + 1], i + 2
                continue
            files.extend(args[i].split())
            i += 1
        if files:
            d['files'] = ' '.join(files)
        return d
    if s == 'search.py':
        d, words, mode = {}, [], 'text'
        for a in args:
            t = a.strip()
            if t in SEARCH_FLAGS:
                if SEARCH_FLAGS[t] == 'regex':
                    mode = 'regex'
                elif SEARCH_FLAGS[t]:
                    d[SEARCH_FLAGS[t]] = 'yes'
            elif t not in SEARCH_IGNORED:
                words.append(a)
        if words:
            text, path = words[0], ' '.join(words[1:])
            if path and _looks_path(text) and not _looks_path(path):
                text, path = path, text  # ["rich/", "TEXT"]: the scripts read it as TEXT PATH
            d[mode] = text
            if path:
                d['path'] = path
        return d
    if s == 'callers.py':
        return {'name': ' '.join(args)} if args else {}
    if s == 'tests_for.py':
        return {'target': ' '.join(args)} if args else {}
    if s == 'history.py':
        if args and args[0].strip() in ('--similar', '-s'):
            return {'similar': ' '.join(args[1:])}
        return {'file': ' '.join(args)} if args else {}
    if s == 'edit.py':
        if args and args[0].strip() == '--lines':
            names, rest = ('file', 'start', 'end', 'new'), args[1:]
        else:
            names, rest = ('file', 'old', 'new'), args
        d = dict(zip(names, rest))
        if len(rest) > len(names):
            d['extra'] = json.dumps(rest[len(names):], ensure_ascii=False)
        return d
    if s == 'create.py':
        d = dict(zip(('file', 'content'), args))
        if len(args) > 2:
            d['extra'] = json.dumps(args[2:], ensure_ascii=False)
        return d
    if s == 'run.py':
        head = args[0].strip() if args else ''
        if head == 'test' and len(args) >= 2:
            d, rest, i = {'test': args[1]}, args[2:], 0
            extra = []
            while i < len(rest):
                if rest[i].strip() == '-k' and i + 1 < len(rest):
                    d['k'], i = rest[i + 1], i + 2
                    continue
                extra.append(rest[i])
                i += 1
            if extra:
                d['extra'] = json.dumps(extra, ensure_ascii=False)
            return d
        if head == 'snippet' and len(args) == 2:
            return {'snippet': args[1]}
        return generic
    if s == 'check_patch.py':
        rest = [a.strip() for a in args if a.strip()]
        if all(a == '--clean' for a in rest):
            return {'clean': 'yes'} if rest else {}
        return generic
    if s == 'hypothesis.py':
        if args and ' ' in args[0].strip() and args[0].split()[0] in ('add', 'list', 'confirm', 'reject'):
            head, rest = args[0].strip().split(None, 1)  # ["add H1 text"] packed into one string
            args = [head, rest] + args[1:]
        cmd_, rest = (args[0].strip() if args else ''), [a.strip() for a in args[1:] if a.strip()]
        if cmd_ == 'add':
            return {('add' if i == 0 else f'add{i + 1}'): t for i, t in enumerate(rest)} or {'add': ''}
        if cmd_ == 'list':
            return {'list': 'yes'}
        if cmd_ in ('confirm', 'reject'):
            d = {cmd_: rest[0] if rest else ''}
            if len(rest) > 1:
                d['evidence'] = ' '.join(rest[1:])
            return d
        return generic
    if s == 'status.py':
        return {}
    return generic


def _norm_value(k, v):
    v = str(v)
    if k in PATH_KEYS:
        return clean_path(_unquote(v))
    if k == 'files':
        return ' '.join(t for t in (clean_path(_unquote(t)) for t in v.split()) if t)
    if k == 'query':
        return ' '.join(t for t in (clean_path(_unquote(t)) for t in v.split()) if t)
    if k in TEXT_KEYS:
        return ' '.join(_unquote(v).split())
    if k in ('start', 'end', 'k'):
        v = _unquote(v)
        return str(int(v)) if v.isdigit() else v
    if k in ('confirm', 'reject'):
        t = _unquote(v).lstrip('#').upper().lstrip('H')
        return str(int(t)) if t.isdigit() else v.strip()
    if k in FLAG_DEFAULTS or k == 'list':
        return 'yes' if _unquote(v).lower() in YES else 'no'
    return v  # old/new/content/snippet/evidence/add*: whitespace is meaningful


def canon(script, argv):
    """The call as {canonical key: value}, the same for the object form and the positional form."""
    s = script_name(script)
    args = norm_args(argv)
    obj = object_args(args)
    if obj is not None:
        obj = alias_keys(s, obj)
    known = KEYS.get(s)
    if obj is not None and (known is None or any(k in known or (s == 'hypothesis.py' and ADD_KEY.match(k))
                                                 for k in obj)):
        d = obj
    else:
        d = _positional(s, args)
    out = {}
    for k, v in d.items():
        v = _norm_value(k, v)
        if FLAG_DEFAULTS.get(k) == v:
            continue  # {"show": "yes"} == no flag at all
        out[k] = v
    return out


def action_key(script, argv):
    """Identity for rule 2: script + canonical args. The object form and the positional form of a call are the
    same call; so are ["a.py 1 9"] and ["a.py", "1", "9"], '"x"' and 'x', /workspace/a.py and a.py."""
    s = script_name(script)
    return s + ' ' + json.dumps(sorted(canon(s, argv).items()), ensure_ascii=False)


def _status(status):
    if isinstance(status, bool):
        return 'ok' if status else 'fail'
    if isinstance(status, int):
        return {0: 'ok', 2: 'usage'}.get(status, 'fail')
    s = str(status or '').lower()
    if s in ('ok', 'success', '0', 'none'):
        return 'ok'
    if s in ('usage', '2'):
        return 'usage'
    if s == 'refused':
        return 'refused'
    return 'fail'


# ---------------------------------------------------------------- state view

class State:
    """Everything the rules need, read in a handful of queries."""

    def __init__(self, conn):
        self.conn = conn
        self.actions = conn.execute('SELECT n, script, key, status, summary FROM actions ORDER BY n').fetchall()
        self.n = self.actions[-1][0] if self.actions else 0
        self.last_refused = bool(self.actions) and self.actions[-1][3] == 'refused'
        real = [a for a in self.actions if a[3] not in ('refused', 'usage')]
        self.real = real
        self.located = any(a[1] == 'locate.py' for a in real)
        first_locate = min([a[0] for a in real if a[1] == 'locate.py'], default=0)
        edits = conn.execute('SELECT n, path, ok, message FROM edits ORDER BY n').fetchall()
        self.edits = edits
        self.last_edit_attempt = max([e[0] for e in edits], default=0)
        self.last_edit_ok = max([e[0] for e in edits if e[2]], default=0)
        self.has_edit = self.last_edit_ok > 0
        self.edited_paths = list(dict.fromkeys(e[1] for e in edits if e[2]))
        self.failed_streak = 0
        for e in reversed(edits):
            if e[2]:
                break
            self.failed_streak += 1
        self.last_failed_path = edits[-1][1] if edits and not edits[-1][2] else None
        self.mode = get_meta(conn, 'mode', 'normal')
        locs = [a for a in real if a[1] == 'locate.py']
        extra_locates = [a for a in locs[MAX_LOCATES:] if a[0] > self.last_edit_attempt]
        window = max(self.last_edit_attempt, first_locate)  # look-around before locate is rule 1's business
        # rule 13: recovery reads after a presumed compaction do not count toward the exploration budget
        try:
            self.recovery = set(json.loads(get_meta(conn, 'recovery', '[]') or '[]'))
        except ValueError:
            self.recovery = set()
        self.look_since_edit = sum(1 for a in real if a[1] in LOOK_AROUND and a[0] > window
                                   and a[0] not in self.recovery) + len(extra_locates)
        all_checks = conn.execute('SELECT n, kind, target, exit_code FROM checks ORDER BY n').fetchall()
        self.checks = [c for c in all_checks if c[1] != 'read']   # snippets that only read source are reads
        self.look_since_edit += sum(1 for c in all_checks if c[1] == 'read' and c[0] > window
                                    and c[0] not in self.recovery)
        self.checks_since_edit = [c for c in self.checks if c[0] > self.last_edit_ok]
        self.viewed = conn.execute('SELECT path, start, "end", n, label FROM viewed ORDER BY n DESC, rowid DESC'
                                   ).fetchall()
        self.candidates = conn.execute('SELECT path, symbol, start, "end", score, evidence FROM candidates '
                                       'ORDER BY rank').fetchall()
        self.hypotheses = conn.execute('SELECT id, text, status, evidence FROM hypotheses ORDER BY id').fetchall()
        closest = get_meta(conn, 'closest')
        c = json.loads(closest) if closest else None
        # P5: a low-similarity closest region is not suggested (similarity unknown: used)
        self.closest_similarity = c[3] if c and len(c) > 3 else None
        self.closest = tuple(c[:3]) if c and (self.closest_similarity is None
                                              or self.closest_similarity >= CLOSEST_MIN_SIMILARITY) else None
        self.last = real[-1] if real else None
        self.first_ts = conn.execute('SELECT MIN(ts) FROM actions').fetchone()[0]
        self.now = time.time()
        self.patch_ready = get_meta(conn, 'patch_ready') == str(self.n_real_last('check_patch.py'))
        # L1: run.py snippets before the first successful edit; those that did not reproduce or answer anything
        self.snippets, self.norepro = [], 0
        # v17 L6: the snippet to re-run after an edit: the latest one before the last successful edit that reproduced
        # something (else the latest one before it); None when one already ran since that edit
        self.repro = None
        out = {a[0]: a[4] or '' for a in self.actions}
        seen, before_edit = set(), []
        for n, kind, target, code in all_checks:
            if kind != 'snippet':
                continue
            body = '\n'.join(l for l in out.get(n, '').splitlines() if not META_LINE.match(l)).strip()
            answered = str(code) not in ('0', 'None') or bool(REPRO_MARK.search(body))
            if not self.has_edit:
                self.snippets.append(n)
                # v18 run.py prints "same output as call N ..." for a repeated output: nothing new learned
                self.norepro += int(not answered or body in seen or body.startswith('same output as call'))
                seen.add(body)
            elif n < self.last_edit_ok:
                before_edit.append((n, answered))
        if before_edit and not any(c[1] == 'snippet' and c[0] > self.last_edit_ok for c in all_checks):
            self.repro = max([n for n, a in before_edit if a] or [n for n, a in before_edit])
        # rule 10 (v18c: the sawtooth of v18 A1): the estimate of the next prompt's size (central, and conservative)
        self.ctx_chars, self.ctx_calls, self.ctx_base = _ctx_parts(conn)
        self.ctx_central = ctx_central(self.ctx_chars, self.ctx_calls, self.ctx_base)
        self.ctx_tokens = ctx_estimate(self.ctx_chars, self.ctx_calls, self.ctx_base)
        # rule 8b: v17b's trigger for compact footers (its cumulative estimate), so the footers stay v17b's
        self.ctx_v17b = ctx_estimate_v17b(int(get_meta(conn, 'ctx_chars_all', '0') or 0), len(self.actions))
        self.compact = self.ctx_v17b >= FOOTER_COMPACT_FROM_TOKENS
        # rule 13: the last presumed compaction; results of calls before visible_from are gone from the context
        self.compacted_at = int(get_meta(conn, 'compacted_at', '0') or 0)
        self.compactions = int(get_meta(conn, 'compactions', '0') or 0)
        self.visible_from = max(1, self.compacted_at - CTX_VISIBLE_CALLS + 1) if self.compacted_at else 0
        self.memo = None                 # set by after()/before() when a MEMO line is due (format_footer prints it)
        self._edited_ranges = None

    def n_real_last(self, script):
        return max([a[0] for a in self.real if a[1] == script], default=0)

    def visible(self, n):
        """Is the result of call n still in the model's context (no presumed compaction since, rule 13)?"""
        return bool(n) and n >= self.visible_from

    @property
    def explore_budget(self):
        if self.mode == 'hypothesis':
            return N_EXPLORE_THIN
        return N_EXPLORE if self.last_edit_attempt else N_EXPLORE_FIRST

    @property
    def run_budget(self):
        return MAX_RUNS_BEFORE_EDIT_THIN if self.mode == 'hypothesis' else MAX_RUNS_BEFORE_EDIT


# ---------------------------------------------------------------- suggestions

VALUE_CHARS = 120              # longest value (path, symbol) copied into a suggested call


def _short(v):
    v = ' '.join(str(v).split())
    return v if len(v) <= VALUE_CHARS else v[:VALUE_CHARS - 1] + '…'


SHOWN_KEY = {'file': 'target', 'path': 'within'}   # internal key -> the key the model sees (argfix.SHOWN_KEY)
PROBE_RENAMES = {'file': 'target', 'path': 'within'}  # for scripts/call_probe.py: history arg keys of older agents, as this one names them


def cmd(script, args=None):
    """A suggested call in the notation the model reads (argfix.text): `show.py  path=a.py  symbol=f`.
    Never JSON: copied JSON quotes break Kaggle's vLLM tool parser (LOG 2026-09-30, E023)."""
    pairs = [f'{SHOWN_KEY.get(str(k), str(k))}={_short(v)}' for k, v in (args or {}).items()]
    return '  '.join([script] + pairs)


SUBMIT = 'submit_patch'
CHECK = cmd('check_patch.py', {})
OLD, NEW = '<exact current lines>', '<new lines>'


# ---------------------------------------------------------------- regions (what the model most likely edits)

MAX_PARSE_LINES = 40000        # files longer than this are not parsed (regions fall back to recorded ranges)
_SRC, _DEFS = {}, {}           # per-process caches: file lines, function definitions


def _file_key(path):
    """Cache key of a repository file: its full path and version (an edit in the same process changes it)."""
    full = os.path.join(root(), path)
    try:
        st = os.stat(full)
        return full, st.st_mtime_ns, st.st_size
    except OSError:
        return full, None, None


def _source(path):
    """The lines of a repository file (cached per process and file version); [] when unreadable."""
    key = _file_key(path)
    if key not in _SRC:
        try:
            with open(key[0], encoding='utf-8', errors='replace') as fh:
                _SRC[key] = fh.read().split('\n')
        except (OSError, ValueError):
            _SRC[key] = []
    return _SRC[key]


def _defs(path):
    """[(start, end, qualname, def_line, body_line, head_line)] for every function (not class) of a Python file; start
    includes decorators, body_line is the first statement after the docstring, head_line the first statement (the
    docstring when there is one). Cached per process and file version."""
    key = _file_key(path)
    if key in _DEFS:
        return _DEFS[key]
    out = []
    lines = _source(path) if path.endswith('.py') else []
    if lines and len(lines) <= MAX_PARSE_LINES:
        import ast
        try:
            tree = ast.parse('\n'.join(lines))
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            tree = None
        stack = [(tree, '')] if tree is not None else []
        while stack:
            node, prefix = stack.pop()
            for ch in ast.iter_child_nodes(node):
                if not isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                qn = prefix + ch.name
                if not isinstance(ch, ast.ClassDef):
                    first = head = ch.body[0]
                    if len(ch.body) > 1 and isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                            and isinstance(first.value.value, str):
                        first = ch.body[1]
                    start = min([d.lineno for d in ch.decorator_list] + [ch.lineno])
                    out.append((start, ch.end_lineno, qn, ch.lineno, first.lineno, head.lineno))
                stack.append((ch, qn + '.'))
    _DEFS[key] = out
    return out


def _enclosing(path, line):
    """The innermost function containing line, as a _defs() entry, or None."""
    best = None
    if line:
        for d in _defs(path):
            if d[0] <= line <= d[1] and (best is None or d[0] >= best[0]):
                best = d
    return best


def _region(path, start, end, label):
    """A region dict for a viewed or located range, widened to the innermost function containing its first line:
    path, start, end, label, def (the def line; None for a plain range), body (first statement after the
    docstring)."""
    d = _enclosing(path, start)
    if d:
        return {'path': path, 'start': d[0], 'end': d[1], 'label': d[2], 'def': d[3], 'body': d[4], 'head': d[5]}
    start = start or 1
    end = end if end and end >= start else start + PAYLOAD_LINES - 1
    return {'path': path, 'start': start, 'end': end, 'label': label or '', 'def': None, 'body': None}


def _overlaps(a, b):
    return a['path'] == b['path'] and a['start'] <= b['end'] and b['start'] <= a['end']


def candidate_files(st):
    """[(path, score)] of locate's source (non-test/doc) files, best first."""
    out = []
    for path, symbol, start, end, score, ev in st.candidates:
        if not AUX.search(path) and all(p != path for p, _ in out):
            out.append((path, score or 0.0))
    return out


def pick_regions(st):
    """The 1-2 regions most likely to need the change, best first (module docstring, "Region choice"). Each is a
    _region() dict plus src: 'viewed' or 'locate'. The second is included only when the top two are close: both
    viewed, or locate's top two files score within CLOSE_SCORE."""
    files = candidate_files(st)
    top = [p for p, _ in files[:TOP_CANDIDATE_FILES]]
    cand_fns = [(c[0], c[2], c[3]) for c in st.candidates if c[2] and c[3]]
    picks = []

    def add(reg, src):
        if not any(_overlaps(reg, r) for r in picks):
            picks.append(dict(reg, src=src))

    def viewed_region(v):
        """(region, is a candidate function): the function at the viewed range's first line, else the first of
        locate's candidate functions inside the range (e.g. a class was viewed), else the range itself."""
        reg = _region(v[0], v[1], v[2], v[4])
        if reg['def'] is None and v[1] is not None:
            inside = [(p, s, e) for p, s, e in cand_fns if p == v[0] and s <= (v[2] or v[1]) and v[1] <= e]
            if inside:
                reg = _region(v[0], inside[0][1], inside[0][2], '')
        hit = any(p == reg['path'] and reg['start'] <= e and s <= reg['end'] for p, s, e in cand_fns) \
            and reg['def'] is not None
        return reg, hit

    viewed = [v for v in st.viewed if not AUX.search(v[0])]  # newest first; sorted() below is stable
    ranked = [(v, *viewed_region(v)) for v in viewed if v[0] in top]
    for v, reg, hit in sorted(ranked, key=lambda x: (top.index(x[0][0]), not x[2])):
        add(reg, 'viewed')
    if not picks:
        c = next((c for c in st.candidates if not AUX.search(c[0])), None)
        if c:
            add(_region(c[0], c[2], c[3], c[1]), 'locate')
    if not picks and viewed:
        add(_region(viewed[0][0], viewed[0][1], viewed[0][2], viewed[0][4]), 'viewed')
    if not picks:
        return []
    first, second = picks[0], None
    if len(picks) > 1 and first['src'] == picks[1]['src'] == 'viewed' and picks[1]['def'] is not None:
        second = picks[1]  # both viewed (the second a function, not just a range of lines)
    elif len(files) > 1 and files[0][1] > 0 and files[1][1] >= files[0][1] * (1 - CLOSE_SCORE):
        other = files[1][0] if files[0][0] == first['path'] else files[0][0]
        second = next((r for r in picks[1:] if r['path'] == other), None)
        c = next((c for c in st.candidates if c[0] == other), None)
        if second is None and c:
            reg = _region(c[0], c[2], c[3], c[1])
            second = None if _overlaps(reg, first) else dict(reg, src='locate')
    return [first] + ([second] if second else [])


def render_region(st, r, cap):
    """(numbered source lines, (a, b) = the line range to edit) for a region, at most cap source lines. A function
    longer than cap: its first PAYLOAD_HEAD lines (through the def line), then a window at the newest range the
    model viewed inside it (else at its body, after the docstring); the window is the range to edit. An unreadable
    file gives no lines and the region's own range."""
    src = _source(r['path'])
    if not src or r['start'] > len(src):
        return [], (r['start'], r['end'])
    fs, fe = r['start'], min(r['end'], len(src))

    def num(i):
        return f'{i:5d}  {src[i - 1]}'
    if fe - fs + 1 <= cap:
        return [num(i) for i in range(fs, fe + 1)], (fs, fe)
    if r.get('def') is None:  # a plain range: its first cap lines
        return [num(i) for i in range(fs, fs + cap)], (fs, fs + cap - 1)
    head_end = max(fs + PAYLOAD_HEAD - 1, min(r['def'], fs + cap // 3))
    inner = [v for v in st.viewed if v[0] == r['path'] and v[1] and head_end < v[1] <= fe]
    if inner:
        a = inner[0][1]
    elif r.get('body') and head_end < r['body'] <= fe:
        a = r['body']
    else:
        a = head_end + 1
    b = min(fe, a + cap - (head_end - fs + 1) - 1)
    out = [num(i) for i in range(fs, head_end + 1)]
    if a > head_end + 1:
        out.append(f'      ... (lines {head_end + 1}-{a - 1} not shown)')
    out += [num(i) for i in range(a, b + 1)]
    if b < fe:
        out.append(f'      ... (the function continues to line {fe})')
    return out, (a, b)


def best_region(st):
    """(path, start, end, label) of the code the model most likely means to edit (pick_regions' first), start/end
    being the line range to edit (a window for a long function); None when nothing is known."""
    regs = pick_regions(st)
    if not regs:
        return None
    r = regs[0]
    a, b = render_region(st, r, PAYLOAD_LINES)[1]
    return r['path'], a, b, r['label']


def edit_cmd(region):
    return cmd('edit.py', {'file': region[0] if region else '<file.py>', 'old': OLD, 'new': NEW})


def lines_cmd(path, start, end):
    return cmd('edit.py', {'file': path, 'start': str(start), 'end': str(end),
                           'new': f'<new text for lines {start}-{end}>'})


def range_edit(path, start, end, lead):
    """'<lead>: edit.py  target=..  start=a  end=b  new=<...>', advising to narrow a long range."""
    long = _int(end) is not None and _int(start) is not None and int(end) - int(start) + 1 > LINES_SUGGEST_MAX
    return f'{lead}{" (narrow start/end to the lines you change)" if long else ""}: ' + lines_cmd(path, start, end)


def edit_now(st, lead):
    """NEXT lines that push the edit of the best region: the line-range edit first, the old-text form second."""
    region = best_region(st)
    if region and region[1] and region[2]:
        return [range_edit(region[0], region[1], region[2], lead), 'NEXT: or by text: ' + edit_cmd(region)]
    return [f'{lead}: ' + edit_cmd(region)]


def small(start, end):
    return bool(start and end) and 0 <= int(end) - int(start) < LINES_SUGGEST_MAX


def covered(st, path, start, end):
    for p, s, e, n, label in st.viewed:
        if p == path and s is not None and e is not None and start is not None and s <= start and (end or start) <= e:
            return True
    return False


def read_cmd(c):
    """The show.py call that reads locate candidate c: its symbol, else the first lines of its file."""
    path, symbol, start = c[0], c[1], c[2]
    if symbol:
        return cmd('show.py', {'file': path, 'symbol': symbol})
    return cmd('show.py', {'file': path, 'start': str(start or 1), 'end': str((start or 1) + 60)})


def next_candidate(st, exclude=(), skip=()):
    """First locate candidate (source files only, not in a file of exclude, not a (path, symbol) of skip) whose region
    has not been viewed yet."""
    for c in st.candidates:
        path, symbol, start, end = c[:4]
        if AUX.search(path) or path in exclude or (path, symbol) in skip:
            continue
        if covered(st, path, start, end):
            continue
        return read_cmd(c)
    return None


def guess_test(path):
    """A test file named after the edited module (tests/**/test_<stem>.py), or None. Bounded walk."""
    stem = os.path.basename(path)[:-3].lstrip('_') if path.endswith('.py') else ''
    if not stem or stem == '_init_':
        return None
    want = f'test_{stem}.py'
    base = root()
    found, seen = [], 0
    for top in ('tests', 'test', 'testing'):
        d0 = os.path.join(base, top)
        if not os.path.isdir(d0):
            continue
        for d, dirs, files in os.walk(d0):
            dirs[:] = [x for x in dirs if not x.startswith('.') and x != '__pycache__']
            seen += len(files)
            if want in files:
                found.append(os.path.relpath(os.path.join(d, want), base))
            if seen > 3000:
                break
    return min(found, key=len) if found else None


def verify_cmd(st):
    for p in reversed(st.edited_paths):
        t = guess_test(p)
        if t:
            return cmd('run.py', {'test': t})
    return cmd('run.py', {'snippet': "<the issue's example code, printing the result>"})


# ---------------------------------------------------------------- v17 L6: existing tests of the edited function

RE_HUNK = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', re.M)


def time_left(st):
    """Seconds left on the TIME_BUDGET_S clock (which starts at the first recorded call)."""
    elapsed = max(0.0, st.now - st.first_ts) if st.first_ts else 0.0
    return TIME_BUDGET_S - elapsed


def edited_names(st):
    """Names to look for in the tests after the last successful edit: the short name of each function its diff
    touched (edit.py prints a unified diff with one context line), then its class; dunders and names under 4
    characters are skipped. [] when unknown."""
    if not st.has_edit:
        return []
    path = next((e[1] for e in st.edits if e[0] == st.last_edit_ok), None)
    summary = next((a[4] for a in st.actions if a[0] == st.last_edit_ok), '') or ''
    if not path or not path.endswith('.py'):
        return []
    lines = [int(m.group(1)) + (1 if int(m.group(2) or 1) >= 2 else 0) for m in RE_HUNK.finditer(summary)]
    if not lines:
        region = best_region(st)
        lines = [region[1]] if region and region[0] == path and region[1] else []
    names = []
    for line in lines:
        d = _enclosing(path, line)
        if not d:
            continue
        parts = d[2].split('.')
        for nm in (parts[-1], parts[0] if len(parts) > 1 else None):
            if nm and not nm.startswith('__') and len(nm) >= 4 and nm not in names:
                names.append(nm)
    return names


def _test_paths(deadline):
    """Test files (test_*.py, *_test.py) under the workspace, bounded by TEST_SCAN_FILES and the deadline."""
    base, n = root(), 0
    for d, dirs, files in os.walk(base):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith('.'))
        for f in sorted(files):
            if f.endswith('.py') and (f.startswith('test_') or f.endswith('_test.py')):
                if n >= TEST_SCAN_FILES or time.time() > deadline:
                    return
                n += 1
                yield os.path.join(d, f)


def _test_names(full, pat):
    """Names of the test functions of a file whose code matches pat ([] when unreadable)."""
    import ast
    try:
        text = open(full, encoding='utf-8', errors='replace').read()
        tree = ast.parse(text)
    except (OSError, SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    lines = text.split('\n')
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test') \
                and node.name not in hits and pat.search('\n'.join(lines[node.lineno - 1:node.end_lineno])):
            hits.append(node.name)
    return hits


def related_tests(names):
    """(test file, k expression, name): the test file that mentions the first of `names` most often, with its test
    functions that mention it (joined with ' or ', at most TEST_K_NAMES; the name itself when more tests use it and
    it is part of their names or of the file name); None when no test mentions any. Bounded: TEST_SCAN_S seconds."""
    deadline = time.time() + TEST_SCAN_S
    pats = {nm: re.compile(r'\b' + re.escape(nm) + r'\b') for nm in names}
    counts = {nm: {} for nm in names}
    base = root()
    for full in _test_paths(deadline):
        try:
            if os.path.getsize(full) > TEST_SCAN_BYTES:
                continue
            text = open(full, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        for nm, p in pats.items():
            if nm in text:
                c = len(p.findall(text))
                if c:
                    counts[nm][os.path.relpath(full, base)] = c
        if time.time() > deadline:
            break
    for nm in names:
        for rel in sorted(counts[nm], key=lambda r: (-counts[nm][r], len(r), r))[:3]:
            hits = _test_names(os.path.join(base, rel), pats[nm])
            if not hits:
                continue
            if len(hits) > TEST_K_NAMES and (any(nm.lower() in h.lower() for h in hits)
                                             or nm.lower() in os.path.basename(rel).lower()):
                k = nm  # pytest -k matches test and module names: all of them
            else:
                take = hits[:TEST_K_NAMES]
                while len(take) > 1 and len(' or '.join(take)) > VALUE_CHARS - 10:
                    take.pop()
                k = ' or '.join(take)
            return rel, k, nm
    return None


def post_edit_tests(st):
    """related_tests() for the last successful edit, cached per edit in meta 'l6_tests'."""
    if not st.has_edit:
        return None
    raw = get_meta(st.conn, 'l6_tests')
    try:
        v = json.loads(raw) if raw else None
    except ValueError:
        v = None
    if v and v[0] == st.last_edit_ok:
        return tuple(v[1:]) if v[1] else None
    names = edited_names(st)
    found = related_tests(names) if names else None
    set_meta(st.conn, 'l6_tests', json.dumps([st.last_edit_ok] + (list(found) if found else [None, None, None])))
    return found


def tests_step(st):
    """(why, call) running the existing tests of the edited function, or None: none found, under
    POST_EDIT_TEST_LEFT_S seconds or few calls left, or that test file already ran since the edit."""
    if not st.has_edit or time_left(st) < POST_EDIT_TEST_LEFT_S or st.n + 3 > TASK_MAX_CALLS:
        return None
    found = post_edit_tests(st)
    if not found:
        return None
    rel, k, nm = found
    if any(c[1] == 'test' and c[0] > st.last_edit_ok and clean_path(str(c[2] or '').split('::')[0]) == rel
           for c in st.checks):
        return None
    return f'the existing tests that use {nm}', cmd('run.py', {'test': rel, 'k': k})


def repro_step(st):
    """(why, call) re-running the repro snippet of before the edit (State.repro), or None."""
    if not st.has_edit or not st.repro:
        return None
    return f'your repro of call {st.repro}, unchanged', cmd('run.py', {'snippet': f'<the same code as call {st.repro}>'})


def verify_call(st):
    """The next verification call after an edit: the repro re-run, else the related tests, else verify_cmd()."""
    step = repro_step(st) or tests_step(st)
    return step[1] if step else verify_cmd(st)


def post_edit_lines(st, lead='NEXT: verify once: '):
    """Footer after a successful edit (v17 L6): re-run the repro, then the existing tests of the edited function,
    then check_patch and submit; v14's verify_cmd when neither applies."""
    steps = [s for s in (repro_step(st), tests_step(st)) if s]
    finish = CHECK + ' and ' + SUBMIT
    if len(steps) == 2:
        return [f'{lead}{steps[0][1]}  ({steps[0][0]})',
                f'NEXT: then {steps[1][0]}: {steps[1][1]}  then {finish}']
    if steps:
        return [f'{lead}{steps[0][1]}  ({steps[0][0]})', 'NEXT: then ' + finish]
    return [lead + verify_cmd(st), 'NEXT: then ' + finish]


def _path_in(text):
    m = re.search(r'([\w./-]+\.py)', text or '')
    return clean_path(m.group(1)) if m else None


def verdict_cmds(hid):
    """confirm call + the reject alternative (same script, so only its args are repeated)."""
    return (cmd('hypothesis.py', {'confirm': str(hid), 'evidence': '<what the output showed>'}) + '  or  '
            + cmd('hypothesis.py', {'reject': str(hid), 'evidence': '<what the output showed>'}))


def hypothesis_steps(st, script):
    """Rule 7 footers (hypothesis mode, before the first edit)."""
    hs = st.hypotheses
    untested = [h for h in hs if h[2] == 'untested']
    confirmed = [h for h in hs if h[2] == 'confirmed']
    cands = ', '.join(dict.fromkeys(c[0] for c in st.candidates if not AUX.search(c[0])))
    if confirmed:
        h = confirmed[-1]
        region = best_region(st)
        p = _path_in(h[1]) or (region[0] if region else '<file.py>')
        return [f'NEXT: edit where H{h[0]} is confirmed: ' + cmd('edit.py', {'file': p, 'old': OLD, 'new': NEW})]
    if not hs:
        return ['NEXT: the issue names few identifiers; write down 2-3 concrete guesses first: '
                + cmd('hypothesis.py', {'add': '<file.py:function does X wrong because Y>', 'add2': '<second guess>'}),
                'NEXT: then test each with ' + cmd('run.py', {'snippet': '<code that shows the behaviour>'})
                + '  and mark it with ' + cmd('hypothesis.py', {'confirm': '1', 'evidence': '<evidence>'})
                + '  or  ' + cmd('hypothesis.py', {'reject': '1', 'evidence': '<evidence>'}),
                'NEXT: edit where a hypothesis is confirmed' + (f'; locate candidates: {cands[:200]}' if cands else '')]
    if untested:
        h = untested[0]
        if script == 'run.py' or (st.last and st.last[1] == 'run.py'):
            return [f'NEXT: record what the run showed for H{h[0]}: ' + verdict_cmds(h[0])]
        return [f'NEXT: test H{h[0]} ({h[1][:80]}): '
                + cmd('run.py', {'snippet': f'<code whose output shows whether H{h[0]} holds>'}),
                'NEXT: then ' + verdict_cmds(h[0])]
    region = best_region(st)
    return ['NEXT: all hypotheses rejected; add a new one ' + cmd('hypothesis.py', {'add': '<new guess>'})
            + ' or edit the best candidate: ' + edit_cmd(region)]


def _shown_at(st, path, a, b):
    """The call that printed lines a-b of path (a recorded view covering them), or None."""
    for p, s, e, n, label in st.viewed:  # newest first
        if p == path and s is not None and e is not None and s <= a and b <= e:
            return n
    return None


def over_budget(st, script, runs=False):
    """Rule 3 (soft): NEXT lines of a look-around call at or past the exploration budget (runs=True: a reproduction run
    at or past the run cap). Before an edit: the line-range edit of the best region first, then a pointer to that code
    and the old-text form."""
    k = st.look_since_edit
    since = 'the last edit' if st.last_edit_attempt else 'locate'
    reads = f'{k} read{"s" if k != 1 else ""} since {since}'
    if runs:
        k = len(st.checks_since_edit)
        reads = f'{k} run{"s" if k != 1 else ""} and no edit yet'
    if st.has_edit:
        return [f'NEXT: {reads}: make your next change now (edit.py by line numbers), or verify: ' + verify_call(st),
                'NEXT: then ' + CHECK + ' and ' + SUBMIT]
    regs = pick_regions(st)
    lead = f'NEXT: {reads}: edit now, even if not fully sure (an imperfect change in the right place beats no change)'
    if not regs:
        return [f'{lead}: ' + edit_cmd(None),
                'NEXT: if the old text is not found, edit.py prints the closest region and its line numbers']
    r = regs[0]
    a, b = render_region(st, r, PAYLOAD_LINES)[1]
    out = [range_edit(r['path'], a, b, lead)]
    where = f'{r["path"]}{" " + r["label"] if r["label"] else ""} lines {a}-{b}'
    n = _shown_at(st, r['path'], a, b)
    if n:
        ptr = f'NEXT: the code most likely to change: {where} (shown at call {n}); or edit by text: ' \
            + edit_cmd((r['path'],))
    else:
        ptr = f'NEXT: the code most likely to change: {where}, not shown yet: ' \
            + cmd('show.py', {'file': r['path'], 'start': str(a), 'end': str(b)})
    if st.mode == 'hypothesis' and not st.hypotheses:
        ptr = 'NEXT: or, if you cannot tell where yet, write guesses: ' \
            + cmd('hypothesis.py', {'add': '<guess 1>', 'add2': '<guess 2>'})
    return out + [ptr]


def probe_nudge(st):
    """L1 footer: after NOREPRO_NUDGE snippets that did not reproduce the bug (or at the snippet cap), one sweeping
    snippet or the edit of the best region."""
    if st.norepro >= NOREPRO_NUDGE:
        lead = (f'NEXT: {st.norepro} snippets did not reproduce it: sweep the invariant over many inputs in ONE snippet, '
                'or edit the most suspicious line now')
    else:
        lead = f'NEXT: {len(st.snippets)} snippets and no edit yet: edit the most suspicious line now'
    return edit_now(st, lead)


def snippet_gate_lines(st):
    """v17 L5 footer from MAX_SNIPPETS_BEFORE_EDIT pre-edit snippets on: the next snippet is refused until an edit."""
    return edit_now(st, f'NEXT: {len(st.snippets)} snippets and no edit yet: the next snippet is not run until you '
                        'edit; edit the most suspicious line now')


def snippet_gate_text(st, script):
    """v17 L5 refusal: 'NOT RUN: N snippets and no edit yet', the region to change, and its line-range edit."""
    region = best_region(st)
    where = ''
    if region and region[1] and region[2]:
        where = f' The code most likely to change: {region[0]}{" " + region[3] if region[3] else ""} ' \
                f'lines {region[1]}-{region[2]}.'
    lines = edit_now(st, f'NEXT: {len(st.snippets)} snippets and no edit yet: edit the most suspicious line now')
    return (f'NOT RUN: {len(st.snippets)} snippets and no edit yet; snippets are refused until you edit.{where}\n'
            + format_footer(lines, st, script, 'refused'))


def suggest(st, script, facts):
    """Footer lines for the current state, most specific rule first."""
    status = facts.get('_status', 'ok')
    exit_code = facts.get('exit_code')
    read_run = script == 'run.py' and facts.get('kind') == 'read'
    look = script in LOOK_AROUND or read_run
    # 1: a read before locate ran (it ran; locate is next)
    if look and not st.located:
        if status == 'usage':
            return []
        return [locate_line(script, facts.get('_args') or []),
                'NEXT: locate ranks files and functions for the whole issue and prints the best function, which '
                'makes most reads unnecessary']
    # 6: passing check after an edit (v17 L6: the existing tests of the edited function first, when not run yet)
    if script == 'run.py' and not read_run and st.has_edit and str(exit_code) == '0':
        ts = tests_step(st)
        if ts:
            return [f'NEXT: the check passes; now {ts[0]}: {ts[1]}', 'NEXT: then ' + CHECK + ' and ' + SUBMIT]
        return ['NEXT: the check passes: ' + CHECK + ', then ' + SUBMIT]
    # check_patch results
    if script == 'check_patch.py':
        if facts.get('ready'):
            return [f'NEXT: {SUBMIT} (the patch is ready)']
        if facts.get('empty') or not st.has_edit:
            return ['NEXT: the patch is empty; make the source change: ' + edit_cmd(best_region(st))]
        return ['NEXT: fix the problems listed; stray files or test edits: ' + cmd('check_patch.py', {'clean': 'yes'}),
                'NEXT: then ' + SUBMIT]
    # edit guards (agent-v15b): a refused edit gets a targeted footer, not the text-matching cascade
    if script in EDITS and facts.get('guard') and status != 'usage':
        p = clean_path(facts.get('path') or '') or '<file.py>'
        if facts['guard'] == 'property' and _int(facts.get('guard_line')):
            ln = _int(facts['guard_line'])
            return ['NEXT: read where the value is assigned and change it there: '
                    + cmd('show.py', {'file': p, 'start': str(max(1, ln - 10)), 'end': str(ln + 10)}),
                    'NEXT: then edit those lines: ' + cmd('edit.py', {'file': p, 'start': '<START>', 'end': '<END>',
                                                                      'new': '<new text>'})]
        c = st.closest
        if c and facts['guard'] == 'definition':  # v17 L2: the edit deleted a def/class/property still in use
            return ['NEXT: redo the edit and keep the definition it deleted (change only the lines around it, or copy '
                    'it into NEW unchanged): ' + lines_cmd(*c)]
        if c:
            return ['NEXT: redo the edit so the module still imports (keep what the rest of the file uses): '
                    + lines_cmd(*c)]
    # 4: edit cascade
    if st.failed_streak >= MAX_FAILED_EDITS:
        c = st.closest
        if c:
            return [f'NEXT: {st.failed_streak} edits failed; stop matching text and replace the lines by number '
                    '(write the new text with real indentation): ' + lines_cmd(*c)]
        region = best_region(st)
        if region and small(region[1], region[2]):
            return [f'NEXT: {st.failed_streak} edits failed; replace the lines by number: '
                    + lines_cmd(region[0], region[1], region[2])]
        return [f'NEXT: {st.failed_streak} edits failed; read the exact lines with show.py, then replace them by '
                'number: ' + lines_cmd(st.last_failed_path or '<file.py>', '<START>', '<END>')]
    # 5: verify cap (v17 L6: not for a failing test run: it may show a regression of the edit)
    failing_test = script == 'run.py' and facts.get('kind') == 'test' and str(exit_code) not in ('0', 'None') \
        and str(exit_code).upper() != 'TIMEOUT'
    if st.has_edit and len(st.checks_since_edit) >= MAX_CHECKS_PER_EDIT and not failing_test:
        return [f'NEXT: {len(st.checks_since_edit)} checks since the last edit: stop verifying: ' + CHECK
                + ', then ' + SUBMIT]
    if script == 'edit.py' or script == 'create.py':
        if facts.get('ok'):
            also = facts.get('also') or []
            if also:  # L4: a changed line is still elsewhere in the file (a sibling with the same bug?)
                p, a, b, fn = also[0]
                verify = post_edit_lines(st, '')[0]
                return [f'NEXT: if {p}:{a}{" " + fn if fn else ""} has the same bug, fix it too: ' + lines_cmd(p, a, b),
                        'NEXT: then verify once: ' + verify + '  then ' + CHECK + ' and ' + SUBMIT]
            return post_edit_lines(st)
        if status == 'usage':
            return []
        sites = facts.get('sites') or []
        if len(sites) > 1:  # L4: OLD matched several places: each site with the bug gets its own line-range edit
            where = ', '.join(f'{p}:{a}{" " + fn if fn else ""}' for p, a, b, fn in sites[:3])
            p, a, b, fn = sites[0]
            return [f'NEXT: OLD is in {len(sites)} places ({where}): fix each one that has the bug by line numbers, '
                    'starting with: ' + lines_cmd(p, a, b)]
        c = st.closest
        out = ['NEXT: copy the old text exactly from the region printed above and retry edit.py']
        if c:
            out.append('NEXT: or replace that region by line numbers: ' + lines_cmd(*c))
        return out
    if script == 'run.py' and not read_run:
        if str(exit_code).upper() == 'TIMEOUT':
            return ['NEXT: skip verification: ' + CHECK + ', then ' + SUBMIT] if st.has_edit else \
                ['NEXT: skip reproduction and edit: ' + edit_cmd(best_region(st))]
        if st.has_edit:
            return ['NEXT: if the failure comes from your change, fix it with edit.py; if it is unrelated or '
                    'pre-existing: ' + CHECK + ', then ' + SUBMIT]
    # L1: snippets before the first edit that did not reproduce anything: sweep the invariant, or edit now
    # (v17 L5: from the snippet cap on, the snippet gate: the next snippet is refused until an edit)
    if script == 'run.py' and facts.get('kind') == 'snippet' and not st.has_edit and status != 'usage' \
            and (st.norepro >= NOREPRO_NUDGE or len(st.snippets) >= MAX_SNIPPETS_BEFORE_EDIT):
        lines = snippet_gate_lines(st) if len(st.snippets) >= MAX_SNIPPETS_BEFORE_EDIT else probe_nudge(st)
        if st.mode == 'hypothesis' or st.hypotheses:  # keep the hypothesis bookkeeping as the second line
            return lines[:1] + hypothesis_steps(st, script)[:1]
        return lines
    left = st.explore_budget - st.look_since_edit
    over_locate = script == 'locate.py' and sum(1 for a in st.real if a[1] == 'locate.py') > MAX_LOCATES
    # 3 (soft): at or past the exploration budget every read runs and its footer pushes the edit
    if (look or over_locate) and left <= 0 and status != 'usage':
        return over_budget(st, script)
    # 3 (soft): reproduction runs at or past the cap before the first edit run; the footer pushes the edit
    if script == 'run.py' and not read_run and not st.has_edit and status != 'usage' \
            and len(st.checks_since_edit) >= st.run_budget:
        return over_budget(st, script, runs=True)
    # 7: hypothesis mode before the first edit
    if not st.has_edit and (st.mode == 'hypothesis' or st.hypotheses):
        return hypothesis_steps(st, script)
    if script == 'run.py' and not read_run:
        if status == 'usage':
            return []
        return ['NEXT: make the fix now: ' + edit_cmd(best_region(st))]
    budget = f' ({max(left, 0)} read{"s" if left != 1 else ""} left before you should edit)' if left <= 2 else ''
    if script == 'locate.py':
        return locate_lines(st, facts)
    if look:
        if status == 'usage':
            return []
        if script == 'show.py' and facts.get('next_show'):  # a long class or a file: its outline was printed
            p, sym = facts['next_show']
            read = f'NEXT: read the definition most related to the issue{budget}: ' \
                + cmd('show.py', {'file': p, 'symbol': sym})
            return [read, 'NEXT: or, if you have seen enough, edit: ' + edit_cmd(best_region(st))]
        if script == 'search.py' and facts.get('next_show'):  # nothing matched, but a definition was found
            p, sym = facts['next_show']
            return [f'NEXT: read it{budget}: ' + cmd('show.py', {'file': p, 'symbol': sym}),
                    'NEXT: or, if you have seen enough, edit: ' + edit_cmd(best_region(st))]
        if script == 'show.py' and facts.get('outlined'):
            p = facts['outlined']
            return [f'NEXT: read one definition from the outline{budget}: '
                    + cmd('show.py', {'file': p, 'symbol': '<Class.method>'})]
        top_files = [p for p, _ in candidate_files(st)[:TOP_CANDIDATE_FILES]]
        if script == 'show.py' and facts.get('path') and clean_path(facts['path']) in top_files \
                and _int(facts.get('start')) and _int(facts.get('end')):  # E: edit early (a function shown)
            p = clean_path(facts['path'])
            r = _region(p, _int(facts['start']), _int(facts['end']), facts.get('label') or '')
            if r['def'] is not None:
                a, b = render_region(st, r, PAYLOAD_LINES)[1]
                return [range_edit(p, a, b, f'NEXT: if this is the code to change{budget}, replace its lines'),
                        'NEXT: or by text: ' + cmd('edit.py', {'file': p, 'old': OLD, 'new': NEW})]
        region = best_region(st)
        if script == 'show.py' and facts.get('path'):
            p, s, e = facts['path'], facts.get('start'), facts.get('end')
            out = [f'NEXT: if this is the code to change{budget}: ' + cmd('edit.py', {'file': p, 'old': OLD, 'new': NEW})]
            if small(s, e):
                out.append('NEXT: or replace lines by number: ' + lines_cmd(p, s, e))
            else:
                alt = next_candidate(st, exclude=(p,))
                if alt:
                    out.append('NEXT: otherwise read ONE other candidate: ' + alt)
            return out
        if script == 'search.py' and facts.get('hits'):
            src = [h for h in facts['hits'] if not AUX.search(h[0])] or facts['hits']
            p, line = src[0][0], int(src[0][1])
            return [f'NEXT: read the best hit{budget}: '
                    + cmd('show.py', {'file': p, 'start': str(max(1, line - 15)), 'end': str(line + 25)}),
                    'NEXT: or, if you have seen enough, edit: ' + edit_cmd(region)]
        if script == 'outline.py':
            p = (facts.get('paths') or [None])[0]
            return [f'NEXT: read one definition from the outline{budget}: '
                    + cmd('show.py', {'file': p or '<file.py>', 'symbol': '<Class.method>'})]
        return [f'NEXT: edit when you know the place{budget}: ' + edit_cmd(region)]
    if script in ('status.py', 'hypothesis.py'):
        if not st.located:
            return [locate_line('status.py', [])]
        if st.has_edit:
            return ['NEXT: verify once: ' + verify_call(st) + '  or  ' + CHECK + ', then ' + SUBMIT]
        return ['NEXT: ' + edit_cmd(best_region(st))]
    return []


def _shown_in_full(st, shown):
    """(path, start, end) when locate printed its best function completely (shown = (path, start, last line))."""
    if not shown:
        return None
    p, start, last = clean_path(shown[0]), _int(shown[1]), _int(shown[2])
    c = next((c for c in st.candidates if c[0] == p and c[2] == start and c[3]), None)
    return (p, start, c[3]) if c and last and last >= c[3] and not AUX.search(p) else None


LOCATE_SIG_LINES = 3  # locate.py SIG_LINES: a longer signature is folded in its preview


def _shown_part(st, shown):
    """(path, a, b, candidate) of a preview locate did not print in full: a-b is the last run of printed lines and
    candidate the locate candidate function containing it (None when unknown)."""
    if not shown:
        return None
    p, a, b = clean_path(shown[0]), _int(shown[1]), _int(shown[2])
    if not a or not b or AUX.search(p):
        return None
    c = next((c for c in st.candidates if c[0] == p and c[2] and c[3] and c[2] <= a <= c[3]), None)
    return p, a, b, c


def locate_lines(st, facts):
    """NEXT lines after locate (normal mode). The code shown -> its edit first (v17b's edit early): the function in
    full (confident: by line numbers), or its whole body and signature with only the docstring folded (confident).
    Nothing shown, or only a window of a long function -> the read of the best match first (v19: the old 'edit it'
    line pointed at code never printed in ~78% of tasks), then ONE other candidate: in another file when locate's
    confidence is low (the top files score close), else the next in locate's order. (A window's line-range edit
    first instead of the read: toolbench local, bot edit suggestion on a gold span 17.0% vs 20.5% of tasks.)"""
    top = next((c for c in st.candidates if not AUX.search(c[0])), None)
    low = bool(facts.get('low_confidence'))
    full = _shown_in_full(st, facts.get('shown'))
    part = None if full else _shown_part(st, facts.get('shown'))
    alt_line = 'NEXT: otherwise read ONE other candidate: '
    if part and part[3]:
        p, a, b, c = part
        r = _region(p, c[2], c[3], c[1])
        sig = (r.get('head') or 0) - (r.get('def') or 0)  # signature lines (locate folds more than LOCATE_SIG_LINES)
        if b >= c[3] and r.get('body') and a <= r['body'] and 0 < sig <= LOCATE_SIG_LINES:
            full, part = (p, a, b), None  # only the docstring folded: the whole body and signature were printed
    if full and not low:  # E: edit early
        out = [range_edit(*full, 'NEXT: if the function shown above is the place, replace its lines'),
               'NEXT: or by text: ' + cmd('edit.py', {'file': full[0], 'old': OLD, 'new': NEW})]
        alt = next_candidate(st, exclude=(full[0],)) or next_candidate(st)
        return out + ([alt_line + alt] if alt else [])
    best = None if full else (part[3] if part and part[3] else top)
    if best is None or covered(st, best[0], best[2], best[3]):  # its code was shown (above, or read before)
        out = ['NEXT: if the function shown above is the place, edit it: '
               + cmd('edit.py', {'file': top[0] if top else '<file.py>', 'old': OLD, 'new': NEW})]
        alt = next_candidate(st, exclude=(top[0],) if top else ()) or next_candidate(st)
        return out + ([alt_line + alt] if alt else [])
    lead = 'NEXT: read the best match (only part of it is shown above): ' if part and part[3] else \
        'NEXT: read the top candidate (its code is not shown above): '
    alt = (next_candidate(st, exclude=(best[0],)) if low else None) or next_candidate(st, skip={(best[0], best[1])})
    return [lead + read_cmd(best)] + ([alt_line + alt] if alt else [])


QUERY_TERMS = ('query', 'symbol', 'name', 'target', 'text', 'regex', 'similar', 'grep', 'file', 'files', 'path')


def locate_line(script, args):
    """Pre-filled locate call; the terms of the refused call (search text, symbol, path) are appended."""
    d = canon(script, args)
    terms = [d[k] for k in QUERY_TERMS if d.get(k)]
    if not terms and d.get('argv'):
        terms = [t for t in json.loads(d['argv']) if not t.startswith('-') and not t.isdigit()]
    extra = ' '.join(_unquote(t) for t in terms)[:80]
    text = '<issue title copied verbatim> <every identifier, path and error message in the issue>'
    return 'NEXT: ' + cmd('locate.py', {'query': text + (' ' + extra if extra else '')})


def _mmss(t):
    """270 -> '4m30s', 300 -> '5m'."""
    return f'{t // 60}m' + (f'{t % 60:02d}s' if t % 60 else '')


def budget(st):
    """(phase, BUDGET line or None). phase: 'ok', 'edit' (edit now) or 'finish' (check_patch and submit now)."""
    elapsed = max(0.0, st.now - st.first_ts) if st.first_ts else 0.0
    frac = max(elapsed / TIME_BUDGET_S, st.n / TARGET_CALLS)
    if frac >= BUDGET_FINISH_FRAC:
        phase = 'finish'
    elif frac >= BUDGET_EDIT_FRAC and not st.has_edit:
        phase = 'edit'
    else:
        phase = 'ok'
    if st.n < BUDGET_SHOW_FROM_CALL and phase == 'ok':
        return phase, None
    m, sec = divmod(int(elapsed), 60)
    if getattr(st, 'compact', False):  # rule 8b: the short form
        line = f'BUDGET: call {st.n}/~{TARGET_CALLS} | {m}m{sec:02d}s/{_mmss(TIME_BUDGET_S)}'
        if phase == 'finish':
            line += ' | check_patch.py and submit_patch now' if st.has_edit else ' | edit now, then submit_patch'
        elif phase == 'edit':
            line += ' | edit now'
        return phase, line
    line = f'BUDGET: call {st.n} of ~{TARGET_CALLS} | {m}m{sec:02d}s of {_mmss(TIME_BUDGET_S)} used'
    if phase == 'finish':
        line += ' | run check_patch.py and submit_patch now' if st.has_edit else \
            ' | edit your best candidate now, then check_patch.py and submit_patch'
    elif phase == 'edit':
        line += ' | edit your best candidate now'
    return phase, line


# ---------------------------------------------------------------- rule 13 (v18 A3, ported): the MEMO line

RE_HUNK_FULL = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', re.M)


def _merge(ranges):
    out = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _git_ranges(paths):
    """{path: [(a, b)]} of the lines the working tree adds or changes against HEAD (git diff -U0), None on error."""
    import subprocess
    try:
        r = subprocess.run(['git', '-C', root(), 'diff', '-U0', '--no-color', '--no-ext-diff', 'HEAD', '--', *paths],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    out, cur = {}, None
    for line in r.stdout.splitlines():
        if line.startswith('+++ '):
            cur = clean_path(line[4:].strip()[2:] if line[4:].startswith('b/') else line[4:].strip())
        m = RE_HUNK_FULL.match(line)
        if m and cur:
            c, d = int(m.group(3)), int(m.group(4) if m.group(4) is not None else 1)
            out.setdefault(cur, []).append((c, c + d - 1) if d else (c, c))
    return {p: _merge(v) for p, v in out.items()}


def edited_ranges(st):
    """[(path, a, b)]: the current line numbers of what the successful edits changed (git diff against HEAD; the
    edit outputs' hunks when git is unavailable), files in edit order; [] without an edit. Cached on st."""
    if st is None or not st.has_edit:
        return []
    if getattr(st, '_edited_ranges', None) is not None:
        return st._edited_ranges
    paths = [p for p in st.edited_paths if p]
    found = _git_ranges(paths) if paths else None
    out = []
    for p in paths:
        rngs = (found or {}).get(p)
        if found is None:
            rngs = []
            for n, path, ok, msg in st.edits:
                if ok and path == p:
                    summary = next((a[4] for a in st.actions if a[0] == n), '') or ''
                    rngs += [(int(m.group(3)), int(m.group(3)) + max(1, int(m.group(4) or 1)) - 1)
                             for m in RE_HUNK_FULL.finditer(summary)]
            rngs = _merge(rngs)
        out += [(p, a, b) for a, b in (rngs or [])]
    st._edited_ranges = out
    return out


def _rng(a, b):
    return f'{a}' if a == b else f'{a}-{b}'


def _check_word(kind):
    return {'snippet': 'repro', 'test': 'tests'}.get(kind, kind or 'run')


def memo(st):
    """Rule 13: the MEMO line (<= MEMO_CHARS) built from sqlite with current line numbers: the code to change (the
    region the footers' edit pushes name: pick_regions/render_region) or the edited ranges (git diff), the last
    repro/test verdict since the edit, whether the patch is READY. '' when nothing is known."""
    if st is None:
        return ''
    try:
        if st.has_edit:
            by_path = {}
            for p, a, b in edited_ranges(st):
                by_path.setdefault(p, []).append(f'L{_rng(a, b)}')
            where = '; '.join(f'{p} {", ".join(r[:2])}{" +" + str(len(r) - 2) if len(r) > 2 else ""}'
                              for p, r in by_path.items()) or ', '.join(st.edited_paths)
            parts = [f'MEMO: edited {where} at call {st.last_edit_ok}']
            after_edit = [c for c in st.checks if c[0] > st.last_edit_ok]
            if after_edit:
                n, kind, target, code = after_edit[-1]
                parts.append(f'{_check_word(kind)} (call {n}) exit {code}')
            else:
                parts.append('not run since the edit')
            cp = st.n_real_last('check_patch.py')
            if st.patch_ready and cp > st.last_edit_ok:
                parts.append(f'patch READY (call {cp}): submit_patch')
            elif cp > st.last_edit_ok:
                parts.append(f'check_patch (call {cp}) found problems')
            else:
                parts.append('patch not checked')
            optional = []
        else:
            failed = len(st.edits)
            regs = pick_regions(st)
            if not regs:
                if not st.located:
                    return 'MEMO: no edit yet; locate.py not run yet'
                return 'MEMO: no edit yet; no code read yet'
            r = regs[0]
            a, b = render_region(st, r, PAYLOAD_LINES)[1]
            seen = _shown_at(st, r['path'], a, b)
            label = f' {r["label"]}' if r.get('label') else ''
            parts = [f'MEMO: no edit yet{f" ({failed} failed)" if failed else ""}. Code to change: {r["path"]}{label} '
                     f'L{_rng(a, b)}' + (f' (read at call {seen})' if seen else ' (not read yet)')]
            optional = []
            if len(regs) > 1:
                o = regs[1]
                optional.append(f'other: {o["path"]}{" " + o["label"] if o.get("label") else ""} '
                                f'L{_rng(o["start"], o["end"])}')
        text = '; '.join(parts + optional)
        while len(text) > MEMO_CHARS and optional:
            optional.pop()
            text = '; '.join(parts + optional)
        return text if len(text) <= MEMO_CHARS else text[:MEMO_CHARS - 1] + '…'
    except Exception:
        return ''


def memo_due(st):
    """Rule 13: 'compaction' for the footer of the call right before a presumed compaction (the first result the
    model sees after it), 'threshold' once per sawtooth cycle when the central estimate first reaches
    MEMO_AT_TOKENS, else None."""
    if st.compacted_at and st.compacted_at == st.n:
        return 'compaction'
    if st.ctx_central >= MEMO_AT_TOKENS and get_meta(st.conn, 'memo_cycle') != str(st.compactions):
        return 'threshold'
    return None


def escalate(st, script, lines, phase):
    """Budget overrides of the NEXT lines (rule 9), and the context guard (rule 10)."""
    if st.ctx_tokens >= CTX_WARN_TOKENS:
        if st.has_edit:
            if script == 'check_patch.py' or (lines and SUBMIT in lines[0] and 'check_patch' not in lines[0]):
                return lines
            return ['NEXT: context nearly full: ' + CHECK + ', then ' + SUBMIT + ' now']
        repair = next((str(l) for l in lines if str(l).startswith('NEXT: ') and 'edit.py  target=' in str(l)), None)
        if script in EDITS and st.failed_streak and repair:  # a failed edit: keep its repair instruction
            return ['NEXT: context nearly full: ' + repair[len('NEXT: '):]]
        out = edit_now(st, 'NEXT: context nearly full: make your edit now')
        return out[:1] if st.ctx_tokens >= CTX_STOP_TOKENS else out
    if phase == 'finish' and st.has_edit:
        if script == 'check_patch.py' or (lines and SUBMIT in lines[0] and 'check_patch' not in lines[0]):
            return lines
        if any('existing tests that use' in str(l) for l in lines[:2]):  # L6: worth a call while 45 s are left
            ts = tests_step(st)
            if ts:
                return [f'NEXT: time is nearly up: run {ts[0]} once: {ts[1]}', 'NEXT: then ' + CHECK + ' and ' + SUBMIT]
        return ['NEXT: time is nearly up: ' + CHECK + ', then ' + SUBMIT]
    if phase in ('edit', 'finish') and not st.has_edit:
        if st.failed_streak and script in EDITS:
            return lines  # already editing: keep the edit-repair instructions
        if lines and str(lines[0]).startswith('NEXT: read the definition most related'):
            return lines  # a long class was outlined: one more targeted read
        if lines and re.match(r'NEXT: \d+ (?:reads? since|runs? and no edit|snippets? did not reproduce|snippets? and no '
                              r'edit) ', str(lines[0])) and 'edit.py' in str(lines[0]):
            return lines  # the over-budget push (rule 3) already edits the best region
        return edit_now(st, 'NEXT: edit your best candidate now')
    return lines


def format_footer(lines, st=None, script=None, status='ok'):
    """At most FOOTER_LINES lines: an optional BUDGET line first, then 'NEXT:' lines (budget escalation, then the
    phase hook, applied to the lines first). Rule 13: when st.memo is set, the MEMO line comes first, on its own line,
    above them (not counted in FOOTER_LINES)."""
    head = []
    compact = bool(st is not None and getattr(st, 'compact', False))
    if st is not None:
        try:
            stage, bline = budget(st)
            lines = escalate(st, script, list(lines), stage)
            head = [bline] if bline else []
        except Exception:
            head = []
        lines = _phase_footer(st, script, status, lines)
    room = FOOTER_LINES - len(head)
    if compact:  # rule 8b: no secondary lines, at most COMPACT_NEXT_LINES, each at most COMPACT_LINE_CHARS
        lines = [re.sub(r'; or edit by text: .*$', '', str(l)) for l in lines if not SECONDARY.match(str(l))]
        room = min(room, COMPACT_NEXT_LINES)
    out = list(head)
    for line in lines[:room]:
        line = re.sub(r'\s*\n\s*', ' ', str(line)).strip()  # one line; keep the notation's two-space separators
        if not line.startswith('NEXT:'):
            line = 'NEXT: ' + line
        out.append(fit_line(line, COMPACT_LINE_CHARS) if compact else line[:LINE_CHARS])
    if not (len(out) > len(head) or head):
        out = []
    memo_line = getattr(st, 'memo', None) if st is not None else None
    if memo_line:  # rule 13: the MEMO on its own line above the footer (not counted in FOOTER_LINES)
        out = [memo_line] + out
    return '\n'.join(out)


# rule 8b: secondary footer lines dropped in compact mode (alternatives and explanations, not the next call)
SECONDARY = re.compile(r'NEXT: (?:or by text|if the old text is not found|locate ranks files)')
CALL_AT = re.compile(r'(?<![\w/.-])(?:' + '|'.join(re.escape(s) for s in KEYS) + r')(?![\w/])')


def fit_line(line, cap):
    """A footer line shortened to cap characters without cutting its first call: parentheses of the lead go first,
    then the lead is shortened at a word, then the text after the call is dropped; a call longer than the cap is
    cut last."""
    if len(line) <= cap:
        return line
    m = CALL_AT.search(line, len('NEXT: '))
    if not m:
        return line[:cap - 1] + '…'
    end = m.end()
    if line[end:].startswith('  '):  # the call's key=value parts (two spaces apart)
        for part in line[m.end() + 2:].split('  '):
            if not re.match(r'[a-z_0-9]+=', part):
                break
            end += 2 + len(part)
    call, trail = line[m.start():end], line[end:]
    lead = re.sub(r' \([^()]*\)', '', line[:m.start()])
    for tail in (trail, ''):
        if len(lead) + len(call) + len(tail) <= cap:
            return lead + call + tail
        room = cap - len(call) - len(tail) - len('…: ')
        short = lead[:room].rsplit(' ', 1)[0].rstrip(' :,;') if room > 0 else ''
        if len(short) > len('NEXT: '):
            return short + '…: ' + call + tail
    return ('NEXT: ' + call)[:cap]


def _phase_before(st, script, argv):
    """phase.before(st, script, argv) when a phase module is present: text to print instead of running, or None."""
    fn = getattr(phase, 'before', None) if phase is not None else None
    if fn is None:
        return None
    try:
        msg = fn(st, script, list(argv or []))
        return str(msg) if msg else None
    except Exception:
        return None


def _phase_footer(st, script, status, lines):
    """phase.footer(st, script, status, lines) when a phase module is present: the replacement NEXT lines."""
    fn = getattr(phase, 'footer', None) if phase is not None else None
    if fn is None:
        return list(lines)
    try:
        out = fn(st, script, status, list(lines))
        return list(lines) if out is None else [str(x) for x in out]
    except Exception:
        return list(lines)


# ---------------------------------------------------------------- public API

def _record(conn, script, argv, status, summary):
    cur = conn.execute('INSERT INTO actions(script, args, key, status, summary, ts) VALUES(?,?,?,?,?,?)',
                       (script, json.dumps(norm_args(argv), ensure_ascii=False), action_key(script, argv), status,
                        (summary or '')[:SUMMARY_CHARS], time.time()))
    return cur.lastrowid


def _viewed_summary(st):
    parts = []
    for path, start, end, n, label in st.viewed[:VIEWED_LIST]:
        rng = f' L{start}-{end}' if start else ''
        parts.append(f'{path}{" " + label if label else ""}{rng} (call {n})')
    return ('; '.join(parts) or 'nothing yet')[:300]


def _stored(st, key):
    """(n, summary) of the newest identical call with a real result and no successful edit since, or None."""
    for n, s, k, status, summary in reversed(st.actions):
        if k == key and status not in ('refused', 'usage') and n > st.last_edit_ok:
            return n, summary
    return None


def _refusal(st, script, argv):
    """(kind, text, viewed records) for a call that must not run, or None. Kinds: context (rule 10), repeat (2),
    snippets (3c), checks (5). Other reads and reproduction runs are not refused (v13: rules 1 and 3 only steer the
    footer). Reads state only."""
    # 10 (v17 L1): with an edit in place and the context at CTX_STOP_TOKENS, reads and snippets are refused, even
    # right after another refusal (a refusal is short; an overflow is graded with an empty patch)
    if st.has_edit and st.ctx_tokens >= CTX_STOP_TOKENS:
        kind = _run_kind(canon(script, argv)) if script == 'run.py' else None
        if script in LOOK_AROUND or script == 'locate.py' or (script == 'run.py' and kind != 'test'):
            return 'context', (f'NOT RUN: context nearly full (~{st.ctx_tokens} of {CTX_LIMIT_TOKENS} tokens; at the '
                               'limit the task ends with an EMPTY patch). Your edit is in place: finish now.\n'
                               + format_footer(['NEXT: ' + CHECK + ', then ' + SUBMIT + ' now'], st, script,
                                               'refused')), None
    # 10 (v17b): no edit yet at CTX_STOP_TOKENS: reads and snippets are refused too, the only way on is the edit
    # (E039 fastapi_14262 ignored "make your edit now" for 20 reads and overflowed with an empty patch). A refusal
    # costs ~350 tokens of context, a read ~1100.
    if not st.has_edit and st.ctx_tokens >= CTX_STOP_TOKENS:
        kind = _run_kind(canon(script, argv)) if script == 'run.py' else None
        if script in LOOK_AROUND or script == 'locate.py' or (script == 'run.py' and kind != 'test'):
            return 'context', (f'NOT RUN: context nearly full (~{st.ctx_tokens} of {CTX_LIMIT_TOKENS} tokens; at the '
                               'limit the task ends with an EMPTY patch) and there is no edit yet: edit now with what '
                               'you know.\n' + format_footer(edit_now(st, 'NEXT: edit now')[:1], st, script,
                                                             'refused')), None
    if st.last_refused:
        return None  # 3b: never refuse twice in a row
    # 2: identical call with no successful edit since then (usage errors and refusals are not results) whose result
    # is still in the model's context (rule 13: one a presumed compaction removed runs again)
    hit = _stored(st, action_key(script, argv))
    if hit and st.visible(hit[0]):
        n, summary = hit
        lines = (summary or '').rstrip().splitlines()
        head = f'ALREADY DONE at call {n}: identical {script} call and nothing was edited since, so it was ' \
               'not rerun. Its output was:'
        body = lines[:REPLAY_LINES] + ([f'... ({len(lines) - REPLAY_LINES} more lines)']
                                        if len(lines) > REPLAY_LINES else [])
        foot = suggest(st, script if script in ('check_patch.py',) else 'status.py',
                       {'ready': st.patch_ready} if script == 'check_patch.py' else {})
        foot = format_footer(foot, st, script, 'refused')
        return 'repeat', '\n'.join([head] + body + ([foot] if foot else [])), None
    # 5 (hard check cap after an edit); a snippet that only reads source is a read. Reproduction runs before the
    # first edit are never refused (v13: rule 3's run cap only steers the footer)
    kind = _run_kind(canon(script, argv)) if script == 'run.py' else None
    # L1/v17 L5: from MAX_SNIPPETS_BEFORE_EDIT snippets with no edit, pre-edit snippets are refused until an edit
    # (rule 3b above lets every second one run)
    if kind == 'snippet' and not st.has_edit and len(st.snippets) >= MAX_SNIPPETS_BEFORE_EDIT:
        return 'snippets', snippet_gate_text(st, script), None
    if script == 'run.py' and kind != 'read':
        runs = len(st.checks_since_edit)
        if st.has_edit and runs >= MAX_CHECKS_HARD:
            return 'checks', (f'NOT RUN: {runs} checks since the last edit. Finish now.\n'
                              + format_footer([CHECK + ', then ' + SUBMIT], st, script, 'refused')), None
    return None


def refusal(st, script, argv):
    """The reason a call must not run, as printable text, or None. Pure: reads state only."""
    r = _refusal(st, script, argv)
    return r[1] if r else None


def _will_compact(st):
    """Rule 13: is a compaction presumed right after the call being recorded (st: the state before it)?"""
    return bool(COMPACT_AT) and st.ctx_central + COMPACT_MARGIN_TOKENS >= COMPACT_AT


def _recovery(st, script, argv, facts):
    """Rule 13: a look-around call that re-reads what a presumed compaction removed (the identical call's result is
    gone, or a show.py range only printed before the compaction). It does not count toward the exploration budget."""
    if not st.compacted_at or not (script in LOOK_AROUND or script == 'locate.py' or facts.get('kind') == 'read'):
        return False
    hit = _stored(st, action_key(script, argv))
    if hit:
        return not st.visible(hit[0])
    if script == 'show.py' and facts.get('path') and _int(facts.get('start')) and _int(facts.get('end')):
        p, a, b = clean_path(facts['path']), _int(facts['start']), _int(facts['end'])
        old = [v for v in st.viewed if v[0] == p and v[1] is not None and v[2] is not None and v[1] <= b and a <= v[2]]
        return bool(old) and not any(st.visible(v[3]) for v in old)
    return False


def before(script, argv):
    """None to go ahead, or a message to print instead of running the script (then exit 0)."""
    try:
        script = script_name(script)
        if script in NEVER_REFUSED and phase is None:
            return None
        conn = connect()
        try:
            st = State(conn)
            prev = st.ctx_central
            st.n += 1  # the budget line counts this (refused) call
            if _will_compact(st):  # rule 13: this refusal is the first result the model sees after the compaction
                st.compacted_at, st.compactions = st.n, st.compactions + 1
                st.visible_from = max(1, st.n - CTX_VISIBLE_CALLS + 1)
                st.memo = memo(st)
            msg = _phase_before(st, script, argv)
            r = ('phase', msg, None) if msg else (None if script in NEVER_REFUSED else _refusal(st, script, argv))
            if not r:
                return None
            kind, msg, views = r
            conn.execute('BEGIN IMMEDIATE')
            n = _record(conn, script, argv, 'refused', kind)
            for path, a, b, label in views or ():
                conn.execute('INSERT INTO viewed VALUES(?,?,?,?,?)', (path, a, b, n, label or ''))
            conn.execute('COMMIT')
            _ctx_step(conn, n, len(msg) + _args_chars(argv), prev)
            return msg
        finally:
            conn.close()
    except Exception:
        return None


def after(script, argv, status, summary, **facts):
    """Record the call and return its footer: up to FOOTER_LINES lines ('BUDGET:' first when shown, then
    'NEXT:' lines); '' on any error."""
    try:
        script = script_name(script)
        facts = dict(parse_facts(script, argv, summary or ''), **{k: v for k, v in facts.items() if v is not None})
        st_code = _status(status)
        conn = connect()
        try:
            st0 = State(conn)
            prev = st0.ctx_central
            recovery = st_code != 'usage' and _recovery(st0, script, argv, facts)
            conn.execute('BEGIN IMMEDIATE')
            n = _record(conn, script, argv, st_code, summary)
            if st_code != 'usage':
                _apply_facts(conn, n, script, argv, st_code, facts)
            if recovery:
                set_meta(conn, 'recovery', json.dumps(sorted(st0.recovery | {n})))
            conn.execute('COMMIT')
            facts['_status'] = st_code
            facts['_args'] = list(argv or [])
            _ctx_step(conn, n, len(summary or '') + _args_chars(argv), prev)  # before State: this output is in context
            st = State(conn)
            due = memo_due(st) if script != 'status.py' else None
            if due:
                st.memo = memo(st)
            footer = format_footer(suggest(st, script, facts), st, script, st_code)
            note = BACKSLASH_NOTE if script in BACKSLASH_SCRIPTS and st_code != 'usage' \
                and _code_backslash(summary) and int(get_meta(conn, 'backslash_notes', '0')) < BACKSLASH_NOTES else ''
            if note:
                set_meta(conn, 'backslash_notes', str(int(get_meta(conn, 'backslash_notes', '0')) + 1))
            footer = '\n'.join(x for x in (note, footer) if x)
            _add_ctx(conn, len(footer))
            if due == 'threshold' and st.memo:
                set_meta(conn, 'memo_cycle', str(st.compactions))
            return footer
        finally:
            conn.close()
    except Exception:
        return ''


def _add_ctx(conn, chars):
    """Rule 10: the footer of the call just accounted by _ctx_step (it is part of that call's result)."""
    set_meta(conn, 'ctx_chars', str(int(get_meta(conn, 'ctx_chars', '0') or 0) + int(chars)))
    set_meta(conn, 'ctx_chars_all', str(int(get_meta(conn, 'ctx_chars_all', '0') or 0) + int(chars)))
    recent = _recent(conn)
    if recent:
        recent[-1] += int(chars)
        set_meta(conn, 'ctx_recent', json.dumps(recent))


def _args_chars(argv):
    return sum(len(str(a)) for a in (argv or []))


def _code_backslash(out):
    """Rule 11: a backslash in a line of code or program output (not in the scripts' own NOTE/usage lines)."""
    return any('\\' in line for line in str(out or '').splitlines() if not META_LINE.match(line.lstrip()))


def _apply_facts(conn, n, script, argv, status, facts):
    d = canon(script, argv)
    if script == 'locate.py':
        if d.get('query'):
            set_meta(conn, 'query', d['query'][:2000])
        cands = facts.get('candidates') or []
        if cands:
            conn.execute('DELETE FROM candidates')
            for rank, c in enumerate(cands):
                c = _cand(c)
                conn.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,?)', (rank, *c))
        shown = facts.get('shown')
        if shown:
            p, s, e = shown[:3]
            label = next((c[1] for c in map(_cand, cands) if c[0] == p and c[2] == s), '') if cands else ''
            conn.execute('INSERT INTO viewed VALUES(?,?,?,?,?)', (clean_path(p), _int(s), _int(e), n, label or ''))
        ids = facts.get('identifier_count')
        thin = facts.get('thin')
        if thin is None and ids is not None:
            ids = int(ids)
            thin = ids <= THIN_MAX_IDENTIFIERS or (bool(facts.get('low_confidence'))
                                                   and ids <= THIN_LOWCONF_MAX_IDENTIFIERS)
        if thin is not None:
            set_meta(conn, 'thin', '1' if thin else '0')
            has_h = conn.execute('SELECT 1 FROM hypotheses LIMIT 1').fetchone()
            set_meta(conn, 'mode', 'hypothesis' if (thin and HYPOTHESIS_MODE) or has_h else 'normal')
    elif script == 'show.py' and facts.get('path') and status == 'ok':
        conn.execute('INSERT INTO viewed VALUES(?,?,?,?,?)', (clean_path(facts['path']), _int(facts.get('start')),
                                                              _int(facts.get('end')), n, facts.get('label') or ''))
    elif script in EDITS:
        ok = bool(facts.get('ok', status == 'ok'))
        path = clean_path(facts.get('path') or d.get('file') or '')
        conn.execute('INSERT INTO edits VALUES(?,?,?,?)', (n, path, int(ok), str(facts.get('message') or '')[:300]))
        c = facts.get('closest')
        if ok:
            conn.execute("DELETE FROM meta WHERE key='closest'")
        elif c:
            sim = facts.get('similarity')
            try:
                sim = None if sim is None else float(sim)
            except (TypeError, ValueError):
                sim = None
            set_meta(conn, 'closest', json.dumps([clean_path(c[0]), _int(c[1]), _int(c[2]), sim]))
        else:  # P5: a failed edit without a closest region: an earlier one's region is stale
            conn.execute("DELETE FROM meta WHERE key='closest'")
    elif script == 'run.py':
        kind = facts.get('kind') or _run_kind(d)
        target = facts.get('target') or (d['test'][:200] if d.get('test') else None)
        conn.execute('INSERT INTO checks VALUES(?,?,?,?)', (n, kind, target, str(facts.get('exit_code'))))
    elif script == 'check_patch.py':
        if facts.get('ready'):
            set_meta(conn, 'patch_ready', str(n))


def _cand(c):
    if isinstance(c, dict):
        return (clean_path(c.get('path', '')), c.get('symbol') or '', _int(c.get('start')), _int(c.get('end')),
                float(c.get('score') or 0), str(c.get('evidence') or ''))
    c = list(c) + [None] * 6
    return clean_path(c[0]), c[1] or '', _int(c[2]), _int(c[3]), float(c[4] or 0), str(c[5] or '')


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _run_kind(d):
    """'test', 'snippet', 'read' (a snippet that only reads repository source, see snippet_reads()), or the first
    word of an unrecognised run.py call."""
    if 'snippet' in d and snippet_reads(d['snippet']):
        return 'read'
    for k in ('test', 'snippet'):
        if k in d:
            return k
    try:
        first = (json.loads(d.get('argv') or '[]') or [''])[0]
    except ValueError:
        first = ''
    return first.split()[0] if first.strip() else ''


# ---------------------------------------------------------------- run.py snippets that only read source

READ_FUNCS = {'open', 'read', 'read_text', 'readlines', 'readline', 'getsource', 'getsourcelines', 'getsourcefile',
              'getfile', 'getline', 'getlines'}
SAFE_FUNCS = READ_FUNCS | {
    'print', 'range', 'enumerate', 'len', 'str', 'int', 'list', 'tuple', 'set', 'dict', 'sorted', 'reversed', 'zip',
    'min', 'max', 'abs', 'any', 'all', 'map', 'filter', 'repr', 'Path', 'splitlines', 'split', 'rsplit', 'strip',
    'rstrip', 'lstrip', 'join', 'format', 'startswith', 'endswith', 'find', 'rfind', 'index', 'replace', 'lower',
    'upper', 'count', 'items', 'keys', 'values', 'get', 'append', 'extend', 'dedent', 'indent', 'exists', 'isfile',
    'search', 'findall', 'finditer', 'match', 'fullmatch', 'compile', 'group', 'groups', 'start', 'end', 'partition'}
SOURCE_FUNCS = {'getsource', 'getsourcelines', 'getsourcefile', 'getfile'}


def _decode_escapes(code):
    return re.sub(r'\\(.)', lambda m: {'n': '\n', 't': '\t', '"': '"', "'": "'", '\\': '\\'}.get(m.group(1), m.group(0)),
                  code)


def snippet_reads(code):
    """What a run.py snippet reads when all it does is read repository source (open()/Path.read_text of a repository
    file, linecache, inspect.getsource) and print it: [{'file': rel, 'start': a, 'end': b} | {'symbol': dotted,
    'module': dotted module or None}]; None for any other snippet (it calls anything else, writes, has no read)."""
    import ast
    tree = None
    for c in (code, _decode_escapes(code)):
        try:
            tree = ast.parse(c)
            break
        except (SyntaxError, ValueError):
            continue
    if tree is None:
        return None
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                imports[a.asname or a.name] = f'{node.module}.{a.name}'
        elif isinstance(node, ast.Import):
            for a in node.names:
                imports[a.asname or a.name.split('.')[0]] = a.asname and a.name or a.name.split('.')[0]
    reads, rng = [], None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)
            if name not in SAFE_FUNCS:
                return None
            args = node.args
            if name in ('open', 'Path', 'getline', 'getlines') and args and isinstance(args[0], ast.Constant) \
                    and isinstance(args[0].value, str):
                modes = [a.value for a in args[1:2] if isinstance(a, ast.Constant)] + \
                    [k.value.value for k in node.keywords if k.arg == 'mode' and isinstance(k.value, ast.Constant)]
                if any(isinstance(m, str) and set(m) & set('wax+') for m in modes):
                    return None
                rel = clean_path(args[0].value)
                if rel and os.path.isfile(os.path.join(root(), rel)):
                    reads.append({'file': rel})
            elif name in SOURCE_FUNCS and args:
                try:
                    dotted = ast.unparse(args[0])
                except Exception:
                    continue
                head = dotted.split('.')[0]
                full = imports.get(head)
                reads.append({'symbol': dotted, 'module': full})
            elif name == 'range' and len(args) >= 2 and all(isinstance(a, ast.Constant) and isinstance(a.value, int)
                                                          for a in args[:2]):
                rng = (args[0].value + 1, args[1].value)
        elif isinstance(node, ast.Slice) and isinstance(node.lower, ast.Constant) and isinstance(node.upper, ast.Constant) \
                and isinstance(node.lower.value, int) and isinstance(node.upper.value, int):
            rng = (node.lower.value + 1, node.upper.value)
        elif isinstance(node, (ast.Delete, ast.Global, ast.Nonlocal)):
            return None
    if not reads:
        return None
    if rng and rng[0] <= rng[1]:
        for r in reads:
            if 'file' in r:
                r['start'], r['end'] = rng
    return reads


# ---------------------------------------------------------------- fact parsing (fallback when scripts pass none)

RE_SCORE = re.compile(r'^\s*(-?\d+(?:\.\d+)?)\s+(\S+\.py)\b(?:\s+\[(.*)\])?\s*$')
RE_FUNC = re.compile(r'^\s{4,}(\S+)\s+L(\d+)-(\d+)\s*$')
RE_BEST = re.compile(r'^--- best match: (\S+) (\S+) L(\d+)-(\d+)')
RE_SHOW = re.compile(r'^(\S+?)(?: (\S+))? L(\d+)-(\d+)(?: of \d+)?\s*$')
RE_HIT = re.compile(r'^(\S+?\.\w+):(\d+)(?:: | [\w.<>]+:)')  # file:line: text  /  file:line Encl.def: text
RE_CLOSEST = re.compile(r'^CLOSEST: (\S+) (\d+) (\d+)\s*$', re.M)
RE_EXIT = re.compile(r'^exit code: (\S+)\s*$', re.M)
RE_SIMILARITY = re.compile(r'\bsimilarity[ =:]+([01](?:\.\d+)?)\b')


def parse_facts(script, argv, out):
    """Facts read from a script's printed output (formats of the v8 scripts, plus C's CLOSEST line)."""
    f = {}
    try:
        d = canon(script, argv)
        lines = out.splitlines()
        if script == 'locate.py':
            cands, cur, score, ev = [], None, 0.0, ''
            for line in lines:
                m = RE_SCORE.match(line)
                if m:
                    if cur and not any(c[0] == cur for c in cands):
                        cands.append((cur, '', None, None, score, ev))
                    cur, score, ev = m.group(2), float(m.group(1)), m.group(3) or ''
                    continue
                m = RE_FUNC.match(line)
                if m and cur:
                    cands.append((cur, m.group(1), int(m.group(2)), int(m.group(3)), score, ev))
                    continue
                m = RE_BEST.match(line)
                if m:
                    shown = [ln for ln in lines if re.match(r'^\s*\d+  ', ln)]
                    last = int(shown[-1].split()[0]) if shown else int(m.group(4))
                    f['shown'] = (m.group(1), int(m.group(3)), last)
                if line.startswith('identifiers:'):
                    rest = line.split(':', 1)[1].strip()
                    f['identifier_count'] = 0 if rest in ('', '(none)') else len([x for x in rest.split(',') if x.strip()])
            if cur and not any(c[0] == cur for c in cands):
                cands.append((cur, '', None, None, score, ev))
            if cands:
                f['candidates'] = cands
            if 'LOW CONFIDENCE' in out or re.search(r'^low confidence:', out, re.M):
                f['low_confidence'] = True
            # locate's own verdict (query-text poverty) must agree with this module's heuristic (few identifiers
            # and low confidence) before hypothesis mode starts: the query is what the model passed, not the issue,
            # so a lazy one-keyword query alone must not flip a rich issue into hypothesis mode.
            m = re.search(r'^signal: .*\bthin=(yes|no)\b', out, re.M)
            if m:
                ids = f.get('identifier_count', 0)
                f['thin'] = m.group(1) == 'yes' and (ids == 0 or (f.get('low_confidence', False) and ids <= 1))
        elif script == 'show.py':
            for line in lines[:3]:
                m = RE_SHOW.match(line)
                if m:
                    shown = [ln for ln in lines if re.match(r'^\s*\d+  ', ln)]
                    last = int(shown[-1].split()[0]) if shown else int(m.group(4))
                    f.update(path=m.group(1), label=m.group(2) or '', start=int(m.group(3)), end=last)
                    break
        elif script == 'search.py':
            hits = [(m.group(1), int(m.group(2))) for m in map(RE_HIT.match, lines) if m]
            if hits:
                f['hits'] = hits
        elif script == 'outline.py':
            paths = [a for a in d.get('files', '').split() if a.endswith('.py')]
            if paths:
                f['paths'] = paths
        elif script in EDITS:
            if re.search(r'^(EDITED|CREATED)\b', out, re.M):
                f['ok'] = True
            elif re.search(r'NOT APPLIED|REFUSED|not found', out):
                f['ok'] = False
            m = RE_CLOSEST.search(out)
            if m:
                f['closest'] = (m.group(1), int(m.group(2)), int(m.group(3)))
            m = RE_SIMILARITY.search(out)
            if m:
                f['similarity'] = float(m.group(1))
            if d.get('file'):
                f['path'] = d['file']
            first = lines[0] if lines else ''
            f['message'] = first
        elif script == 'run.py':
            m = RE_EXIT.findall(out)
            if m:
                f['exit_code'] = m[-1]
            if d:
                f['kind'] = _run_kind(d)
        elif script == 'check_patch.py':
            f['ready'] = bool(re.search(r'^READY\b', out, re.M))
            f['empty'] = 'NO CHANGES' in out
    except Exception:
        pass
    return f


# ---------------------------------------------------------------- one-line integration

def peek():
    """The working memory as a State, for scripts that consult it (show.py, outline.py); None when unavailable."""
    try:
        return State(connect())
    except Exception:
        return None


def current_file(st=None):
    """The file the model is working on: the latest viewed file, else locate's top source file; None if unknown."""
    st = st or peek()
    if st is None:
        return None
    if st.viewed:
        return st.viewed[0][0]
    c = next((c for c in st.candidates if not AUX.search(c[0])), None)
    return c[0] if c else None


def _words(text):
    """Lower-case words of identifiers and text (snake_case and CamelCase split), 3+ letters."""
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', str(text))
    return {w for w in re.findall(r'[a-z0-9]+', text.lower()) if len(w) >= 3}


def related_symbol(st, path, names):
    """Of the qualnames `names` (definitions in path), the one most related to the task: one of locate's
    candidates (by rank), else the one sharing most words with the locate query; None when nothing relates."""
    names = list(names)
    if st is None or not names:
        return None
    for c in st.candidates:
        if c[0] == path and c[1] in names:
            return c[1]
    q = _words(get_meta(st.conn, 'query', '') or '') - {'self', 'init', 'def', 'class', 'the'}
    if not q:
        return None
    best = max(names, key=lambda n: len(_words(n.split('.')[-1]) & q))
    return best if _words(best.split('.')[-1]) & q else None


def fact(**kw):
    """Called from inside a script's main() to hand structured facts to after()."""
    _FACTS.update(kw)


class _Tee(io.TextIOBase):
    def __init__(self, real):
        self.real, self.buf = real, io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self.real.write(s)

    def flush(self):
        self.real.flush()


def strip_separator(argv):
    """ADK puts a bare '--' before positional_args (skill_toolset: argv = --key value ... -- pos ...). v18b: drop it,
    or the scripts read '--' as a key or a path (E044 fastapi_14356: create.py positional_args ["repro.py"] created a
    file named "--", and flow then saw an edit). Positional values alone are the scripts' positional form; after
    key/value pairs they are appended (the scripts' parsers decide)."""
    argv = list(argv)
    for i in range(0, len(argv), 2):            # a '--' at a key position, never a value
        if argv[i] == '--':
            return argv[:i] + argv[i + 1:]
    return argv


def main(script, main_fn, argv=None):
    """Run a script's main(argv) under the flow rules: refusal, output capture, footer. Returns the exit code."""
    argv = strip_separator(sys.argv[1:] if argv is None else argv)
    if os.environ.get('REPO_TOOLS_FLOW', 'on') == 'off':  # unit tests of a single script's own behaviour
        return main_fn(argv)
    msg = before(script, argv)
    if msg:
        print(msg)
        return 0
    _FACTS.clear()
    tee = _Tee(sys.stdout)
    code, err = 1, None
    try:
        with contextlib.redirect_stdout(tee):
            code = main_fn(argv)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException as e:  # record, print the footer, then re-raise
        err = e
    footer = after(script, argv, code if err is None else 'fail', tee.buf.getvalue(), **_FACTS)
    if footer:
        sys.stdout.flush()
        print(footer)
    if err is not None:
        raise err
    return code
