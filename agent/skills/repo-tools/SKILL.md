---
name: repo-tools
description: Deterministic helpers for the repository in /workspace - locate code for an issue, show and search code, edit and create files safely, run quick checks, and check the patch before submitting. Every result ends with the recommended next call.
---
Call formats are in the system prompt. Notes: edit.py with start and no end replaces that one line (several new
lines need end); OLD found in several places lists each with a ready line-range edit; start2 end2 new2 (up to 4
hunks) change several places of one file in one call. run.py: tests 90 s, snippets
15 s (a timeout shows where the code was). check_patch.py clean=yes reverts stray files and test edits.
