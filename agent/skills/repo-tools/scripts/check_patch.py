"""Check the working tree before submit_patch.

usage: check_patch.py   (no args)   or   check_patch.py  clean=yes
       (positional form, still accepted: check_patch.py [--clean])

Lists changed files and flags ones that will hurt the submission: test files and runner config (the grader
resets them), scratch files (repro*, scratch*, tmp*, debug*, *.log, new test files, new .py files at the root)
and other new non-Python files. New files made with create.py are part of the fix and listed as "new file"
(so is a new .py file inside a package that was written some other way). Compiles every changed or new .py file
and tries to import changed modules; warns about `while True:` loops the patch adds without break/return/raise, and
(v17) about module/class-level definitions that exist at HEAD in an edited file, are gone now and are still used.
With "clean": "yes" (--clean), reverts flagged edits and deletes flagged
untracked files; files made with create.py are never deleted. Ends with READY or a list of problems.
v18: when a context compaction removed the last edit's own output, the full diff is printed too (if <= 30 lines).
"""

import os
import re
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow  # noqa: E402  (flow engine: refusals + NEXT/BUDGET footers)
import argfix  # noqa: E402
import guards  # noqa: E402  (v16: infinite loops added by the patch; v17: deleted definitions)

ROOT = (os.environ.get('WORKSPACE') or ('/workspace' if os.path.isdir('/workspace') else next((p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) + [os.path.join(os.path.dirname(os.environ.get('TMPDIR', '/nonexistent').rstrip('/')), 'workspace')] if p and os.path.isdir(os.path.join(p, '.git'))), os.getcwd())))  # overridable for local testing
STATE = (os.environ.get('REPO_TOOLS_DIR')  # v19b: per task, as flow.state_dir()
         or (os.path.join(ROOT, '.git', 'repo_tools') if os.path.isdir(os.path.join(ROOT, '.git')) else '/tmp/repo_tools'))
CREATED = os.path.join(STATE, 'created.txt')
TEST_OR_CONFIG = re.compile(
    r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)(conftest\.py|pytest\.ini|pyproject\.toml|'
    r'setup\.cfg|tox\.ini|\.pytest\.ini|sitecustomize\.py)$')
SCRATCH = re.compile(r'(^|/)(repro|reproduce|scratch|tmp|temp|debug)([_\-.\d][^/]*)?$|'
                     r'\.(log|orig|rej|bak|swp|tmp|diff|patch|out)$|(^|/)nohup\.out$', re.I)
NO_IMPORT = re.compile(r'(^|/)(scripts|benchmarks|tools)/')  # standalone scripts: compile only


def git(*args):
    return subprocess.run(['git', '-C', ROOT, *args], capture_output=True, text=True, timeout=60)


def module_name(rel):
    parts = rel[:-3].split('/')
    if parts and parts[0] == 'src':
        parts = parts[1:]
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts) if parts and all(p.isidentifier() for p in parts) else None


def created_files():
    try:
        return {l.strip() for l in open(CREATED, encoding='utf-8') if l.strip()}
    except OSError:
        return set()


def untracked_problem(rel):
    """Why an untracked file not made with create.py should not be in the patch, or None if it looks intended."""
    if TEST_OR_CONFIG.search(rel):
        return 'new test/config file: the grader discards it'
    if SCRATCH.search(rel):
        return 'scratch file: it would be added to the patch'
    if '/' not in rel and rel.endswith('.py'):
        return 'new .py file at the repository root (looks like a scratch script): it would be added to the patch'
    if not rel.endswith('.py'):
        return ('new non-Python file not made with create.py: it would be added to the patch (if it is part of '
                'the fix, write it again with create.py, which records it)')
    return None


def added_lines(rel):
    """Line numbers (1-based, in the working tree) that the patch adds or changes in a tracked file."""
    out = set()
    for m in re.finditer(r'^@@ -\S+ \+(\d+)(?:,(\d+))? @@', git('diff', '-U0', 'HEAD', '--', rel).stdout, re.M):
        a, n = int(m.group(1)), int(m.group(2) or 1)
        out.update(range(a, a + n))
    return out


def loop_warnings(rel, new_file=False):
    """WARNING lines for `while True:` loops the patch adds without a way out (guards.infinite_loops)."""
    try:
        text = open(os.path.join(ROOT, rel), encoding='utf-8').read()
    except (OSError, ValueError):
        return []
    return [f'WARNING: {rel}:{n} `while True:` has no break, return or raise inside: if it never ends, the hidden '
            'tests hang and fail' for n in guards.infinite_loops(text, None if new_file else added_lines(rel))]


def lost_warnings(rel):
    """WARNING lines for definitions in REL at HEAD that the working tree lost while their name is still used
    (guards.lost_definitions; v17, note 12 L2: an earlier range edit deleted computed_scope / _parse)."""
    head = git('show', f'HEAD:{rel}')
    if head.returncode != 0:
        return []
    try:
        now = open(os.path.join(ROOT, rel), encoding='utf-8').read()
    except (OSError, ValueError):
        return []
    return [f'WARNING: {rel}: {guards.describe(d)} existed at HEAD but the patch deleted it, and it is still used at '
            f'{guards.uses(d["refs"])}: an edit removed it by mistake? Put the definition back with edit.py.'
            for d in guards.lost_definitions(head.stdout, now, rel) if d['refs']]


def check_python(rel, problems, import_it=True):
    try:
        compile(open(os.path.join(ROOT, rel), encoding='utf-8').read(), rel, 'exec')
    except SyntaxError as e:
        problems.append(f'{rel}: does not compile: line {e.lineno}: {e.msg}')
        return
    mod = module_name(rel)
    if mod and import_it:
        r = subprocess.run([sys.executable, '-c', f'import {mod}'], cwd=ROOT, capture_output=True,
                           text=True, timeout=60, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        if r.returncode != 0:
            last = (r.stderr.strip().splitlines() or ['?'])[-1]
            problems.append(f'{rel}: import {mod} fails: {last[:200]}')


DIFF_MAX_LINES = 30  # v18 A4: after a compaction the full diff is shown when it is at most this long


def diff_after_compaction():
    """The working-tree diff (git diff HEAD) as lines when a context compaction presumably removed the output of the
    last successful edit (flow rule 13) and the diff has at most DIFF_MAX_LINES lines; [] otherwise."""
    try:
        st = flow.peek()
        if st is None:
            return []
        try:
            if not st.has_edit or st.visible(st.last_edit_ok):
                return []
        finally:
            st.conn.close()
    except Exception:
        return []
    lines = git('diff', '--no-color', '--no-ext-diff', 'HEAD').stdout.rstrip('\n').splitlines()
    return lines if 0 < len(lines) <= DIFF_MAX_LINES else []


REVIEW_DIFF_LINES = 40  # v24: the diff shown with the review (longer diffs are cut)
REVIEW = ('REVIEW before submit_patch (one sentence each in your thinking; fix what fails, else submit):',
          '  1. The issue\'s own example, through the public entry point it uses: did it run after your edit, and does '
          'its output now match what the issue asks for?',
          '  2. Is every class, function, parameter and call site the issue names handled (all of them, not only the '
          'first)?',
          '  3. Exception types and messages, names and output format: the issue\'s literal text, or else the style '
          'of the nearest existing code in the same module?',
          '  4. Is there any change the issue did not ask for? Remove it.')


def review_lines(shown_diff):
    """v24 (research/notes/19-fix-quality.md, item 1): with READY, the diff (unless already shown), what ran since the
    last edit (flow's records: grounded review, not free self-critique) and four questions."""
    out = []
    if not shown_diff:
        lines = git('diff', '--no-color', '--no-ext-diff', 'HEAD').stdout.rstrip('\n').splitlines()
        if lines:
            out.append('your patch (git diff):')
            out += lines[:REVIEW_DIFF_LINES]
            if len(lines) > REVIEW_DIFF_LINES:
                out.append(f'... ({len(lines) - REVIEW_DIFF_LINES} more lines)')
    try:
        st = flow.peek()
        if st is not None:
            try:
                ran = [f'call {n} {kind}{" " + str(target) if target else ""}: exit {code}'
                       for n, kind, target, code in st.checks_since_edit[-3:]] if st.has_edit else []
            finally:
                st.conn.close()
            out.append('ran since your last edit: ' + ('; '.join(ran) if ran else 'nothing'))
    except Exception:
        pass
    return out + list(REVIEW)


USAGE = '\n'.join(__doc__.strip().splitlines()[2:4])
CLEAN = argfix.obj('check_patch.py', {'clean': 'yes'})


def main(argv):
    try:
        d, note = argfix.object_args(argv, 'check_patch.py')
        clean = argfix.flag(d, 'clean', 'check_patch.py') if d is not None else '--clean' in argv
    except argfix.Misuse as e:
        print(f'USAGE ERROR: {e}')
        print(USAGE)
        return 2
    if note:
        print(f'NOTE: {note}')
    status = git('status', '--porcelain', '--untracked-files=all').stdout.splitlines()
    created = created_files()
    changed, untracked, flagged = [], [], []
    for line in status:
        code, rel = line[:2], line[3:].strip().strip('"')
        if ' -> ' in rel:
            rel = rel.split(' -> ')[1]
        if code == '??':
            untracked.append(rel)
        else:
            changed.append(rel)
    if not changed and not untracked:
        print('NO CHANGES: the patch is empty. Make the source edit before submitting.')
        return 1
    new_files, stray = [], []
    for rel in changed:
        if TEST_OR_CONFIG.search(rel):
            flagged.append((rel, 'test/config file: the grader resets it'))
    for rel in untracked:
        if rel in created:
            new_files.append(rel)
            continue
        why = untracked_problem(rel)
        if why:
            flagged.append((rel, why))
        else:
            new_files.append(rel)
            stray.append(rel)

    problems = []
    for rel in changed:
        if rel.endswith('.py') and os.path.exists(os.path.join(ROOT, rel)) and not TEST_OR_CONFIG.search(rel):
            check_python(rel, problems)
    for rel in new_files:
        if rel.endswith('.py') and os.path.exists(os.path.join(ROOT, rel)):
            check_python(rel, problems, import_it=not NO_IMPORT.search(rel))

    stat = git('diff', '--stat', 'HEAD').stdout.strip().splitlines()
    print('changed files:')
    for line in stat[:15]:
        print('  ' + line)
    for rel in new_files[:15]:
        n = sum(1 for _ in open(os.path.join(ROOT, rel), 'rb')) if os.path.isfile(os.path.join(ROOT, rel)) else 0
        print(f'  {rel} (new file, {n} lines{"; not made with create.py" if rel in stray else ""})')
    if len(new_files) > 15:
        print(f'  ... {len(new_files) - 15} more new files')
    for rel in [r for r in untracked if r not in new_files][:10]:
        print(f'  {rel} (untracked)')
    diff = diff_after_compaction()
    if diff:
        print('your edits (git diff; earlier results left the context):')
        for line in diff:
            print(line)

    warnings, lost = [], []
    for rel in changed:
        if rel.endswith('.py') and os.path.exists(os.path.join(ROOT, rel)) and not TEST_OR_CONFIG.search(rel):
            lost += lost_warnings(rel)
            warnings += loop_warnings(rel)
    for w in lost[:3]:
        print(w)
    for rel in new_files:
        if rel.endswith('.py') and os.path.exists(os.path.join(ROOT, rel)):
            warnings += loop_warnings(rel, new_file=True)
    for w in warnings[:3]:
        print(w)

    if clean and flagged:
        for rel, _ in flagged:
            if rel in untracked:
                if rel in created:  # never delete files made on purpose with create.py
                    continue
                try:
                    os.remove(os.path.join(ROOT, rel))
                except OSError:
                    pass
            else:
                git('checkout', 'HEAD', '--', rel)
        print('cleaned: ' + ', '.join(r for r, _ in flagged))
        flagged = []

    for rel, why in flagged:
        problems.append(f'{rel}: {why} (to remove it: {CLEAN})' if rel in untracked else
                        f'{rel}: {why} (to revert it: {CLEAN})')
    if problems:
        print('PROBLEMS:')
        for p in problems:
            print('  - ' + p)
        return 1
    for line in review_lines(bool(diff)):
        print(line)
    print('READY: call submit_patch now, unless the review found something to fix.')
    return 0


if __name__ == '__main__':
    sys.exit(flow.main('check_patch.py', main))
