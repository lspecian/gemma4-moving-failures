"""rft3 part 1: grounded oracle trajectories from SWE-bench train in a frozen agent format (agent-v19, agent-v20; CPU
only, no model).

    .venv/bin/python scripts/rft3_oracle.py select [--n 3000] [--per-repo 120]
    .venv/bin/python scripts/rft3_oracle.py [--version v20] build [--workers 10] [--only ID,ID] [--limit N]  # ~75 min
    .venv/bin/python scripts/rft3_oracle.py [--version v20] collect                           # states + report
    .venv/bin/python scripts/rft3_oracle.py --version v20 compare [--base v19]                # per-task outcomes

--version (rft3_states.VERSIONS): the agent whose scripts and prompt are used, and where the data goes: v19 (default,
the step A build) data/rft3/{oracle_raw,states,oracle_report.json}; v20 data/rft3/{oracle_raw_v20,states_v20,
oracle_report_v20.json} (workspaces in ws_v20, tmp_v20). The task selection is shared (experiments/rft3_oracle_tasks.txt).

Format, schema and the message rebuild: scripts/rft3_states.py. Plan: research/notes/14-oracle-trajectories.md.

select: princeton-nlp/SWE-bench train (data/toolbench/swebench, scripts/toolbench_data.py), toolbench_data.eligible()
plus: <= MAX_SPANS gold change spans in non-test .py files, <= MAX_LINES changed lines, problem statement <=
MAX_STATEMENT characters (it is in the prompt twice); NOT the toolbench benchmark instances
(experiments/toolbench_external.txt, a clean benchmark) and not fastapi/rich/requests/httpx. Per repository the
instances are shuffled (seed 0) and at most --per-repo taken, round-robin over repositories until --n. Writes
experiments/rft3_oracle_tasks.txt and data/rft3/oracle_tasks.jsonl.

build (per repository one worker, instances sequentially): workspace data/rft3/ws/<id> = the toolbench clone
(data/toolbench/repos, blob-less) checked out at base_commit, copied copy-on-write, own .git sharing the objects,
then the harness's container setup (git exclude, pytest.ini / conftest.py of setup_workspace_test_config, baseline
commit + tag); deleted after the trajectory is built. The user message gets the harness's tree command on it.
Scripts of agent-v19 run as ADK runs them (tests of toolbench.Session: argv --key value, runpy from a temp cwd, flow
state on, PYTHONHASHSEED 0) under python3.13 on a simulated clock (call n at T0 + (n-1) x 20 s, so BUDGET footers show
Kaggle-like elapsed time). Results are wrapped as ADK returns them (rft3_states.adk_result, incl. the executor's
"Command failed with status=error, exit_code=N" stderr). Repository dependencies are NOT installed (and old code under
python3.13 often fails to import anyway), so the import checks of check_patch.py and of edit.py's import guard
(`python -c "import <module>"` of a workspace module) are EMULATED as succeeding: data/rft3/stubenv/sitecustomize.py
(on PYTHONPATH, active only in such `-c "import ..."` processes) registers the module as imported without running it.
Kaggle's provisioned sandbox imports the gold code; printing ModuleNotFoundError lines here would teach "submit
despite failing imports". Everything else check_patch.py checks (compile, lost definitions, stray files) is real.
(A first version stubbed the missing third-party modules instead: 50% of the gold modules still failed to import.)
Fallback: when a gold module still fails the emulated check at baseline, check_patch.py is skipped (edits -> submit).

Trajectory (source by what was needed):
  locate.py query=build_query(statement) (toolbench's query; the model's real queries are similar)
  -> reads: per gold span not yet printed as numbered lines, show.py symbol=<enclosing def> (once per def, <= 3) or a
     range read (folded signature/docstring, window of a long def, module-level code)
  -> edit.py line-range per gold span, bottom-up (base numbers stay valid; a refused edit is retried merged with the
     span above, as the toolbench oracle), NEW = the gold text (pure insertions/deletions anchored on a neighbour).
     v20 (agent with multi-edit): a file with 2-4 gold spans, all printed before the edit turn (always: the reads come
     first), gets ONE edit.py call with every span as a hunk (start/end/new, start2/end2/new2, ...; top-down, the
     numbers as printed; spans whose anchored ranges overlap are merged into one hunk), run on a copy of the flow state
     and kept only when the result is exactly the gold (else single edits). A seeded SINGLE_SHARE (25%) of such files
     keeps single edits; their order puts a hunk that defines or imports a name another hunk uses first when the line
     numbers allow it (Builder.single_order: the spans above an edit applied before it keep the line count), so no
     CHECK line reports a transient `undefined name`; when no order avoids that, the file gets the multi-edit
     (edit_forms *_dependent). Repair trajectories keep v19's single edits (with v20's outputs).
  -> check_patch.py -> submit_patch (its {"result": ...} as swegemma's submit_patch returns it).
  Grounding rule (every check below uses it; Builder.symbol_grounded / grounds / refs_by_file / sym_in):
  - a show turn only if its file and its def were printed in an earlier output (locate list/preview, NEXT line,
    names: block, search hit, outline) or the def is named in the issue. Names count per file: only lines attributed
    to that file (the last file path named on or above them) are searched. A def is named by its qualname
    (Class.method) or by its last part in a code-like position (`def x`, `.x`, `x(`, `x  L12-30`, backticks); for a
    method the class name must appear in the same text too, and a dunder (`__init__`) is never named by itself;
  - a range read only inside a def whose L-range or fold notice was printed for that file, near a line reference of
    that file (names: L12, path:12, within 12 lines), or the file head for a module-level change near the top;
  - an edit only if every line it replaces was printed numbered earlier.
  Each turn's `grounding` says where.
  oracle_clean: every read grounded by locate's output, the issue or the gold reads themselves (names: blocks).
  oracle_recovery: the gold file/def is not grounded that way (locate missed it), or the gold file is far down
     (rank > 3): at most MAX_EXTRA extra real calls that make the next gold read grounded, tried in this order:
     every NEXT read of the last output, outline.py of a grounded gold file whose def was not named, search.py
     text=<identifier> (visible in the issue or an earlier output AND an identifier of the gold def's real code --
     tokenize NAMEs, not docstrings, comments or strings -- and not containing the project's own name, e.g. no
     PyTorchLightning in Lightning-AI/lightning; rarest first, <= SEARCH_TRIES) whose output shows the gold file
     (+ the def via search's "in <function>" column), history.py
     similar=<issue title>; two-step paths: such an output names the file -> outline.py names the def, or its NEXT
     read leads there. Module-level spans: outline.py whose line references (names: L12) are near the change.
     Candidates run on copies of the flow state (MAX_TRIALS); only the chosen calls enter the trajectory. No path ->
     discarded (rank > 3 with the def listed: the direct read stays, noted in meta.notes). A range read whose output
     folds the span (docstring) is started inside the string, or repeated once from the fold notice's first line.
  oracle_repair: for ~REPAIR_SHARE of the tasks (seeded by id), one edit is first sent with a typical mistake
     (toolbench robustness generators: start_only, json_old, json_both, line_prefix, indent_plus4) and kept only if
     edit.py REFUSES it (file unchanged); the turn is label "context" (no loss), followed by the corrected call
     (the ready call the refusal printed when it gives the intended text, else the clean line-range edit; "train").
  No run.py: train repositories have no environment here (outputs would be import errors); the real episodes in
  the mix keep run.py. No final text-only turn after submit_patch (step D adds it if wanted). thinking/visible are
  null on every synthetic turn (step B samples them). Discards: no gold span in an existing .py, Python-2 gold files,
  setup failure, ungrounded, edit not applied / final != gold, check_patch problems, estimated prompt >= 14,336
  (collect: June-template render of the last request + rft3_states.THINK_EST tokens per null-thinking turn).
  Tests: tests/test_rft3_oracle.py (grounding rules; v20 multi-edit path with the agent's real edit.py).
  collect also reports the edit forms per file, hunks per edit call, the CHECK line of every applied trained edit
  (clean / undefined / timeout / none) and the NOTE lines edit.py printed (edit_report).

Outputs: data/rft3/oracle_raw[_v20]/<repo>.jsonl (one line per task: record or discard reason; resume skips done ids),
collect -> data/rft3/states[_v20]/oracle_{clean,recovery,repair}.jsonl, data/rft3/oracle_report[_v20].json.
"""

import argparse
import collections
import concurrent.futures
import fcntl
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rft3_states as S  # noqa: E402
import toolbench as tb  # noqa: E402
import toolbench_data as td  # noqa: E402

OUT = os.path.join(ROOT, 'data', 'rft3')
STUB_DIR = os.path.join(OUT, 'stubenv')


def configure(version=None):
    """Paths of agent VERSION (rft3_states.VERSIONS; default: the process's, RFT3_VERSION): v19 writes
    data/rft3/{oracle_raw,states,ws,tmp,oracle_report.json} (the step A build), v20 the same names with _v20."""
    global WS_ROOT, TMP_ROOT, RAW_DIR, STATES, REPORT
    if version:
        S.use_version(version)
    WS_ROOT = S.suffixed(os.path.join(OUT, 'ws'))
    TMP_ROOT = S.suffixed(os.path.join(OUT, 'tmp'))
    RAW_DIR = S.suffixed(os.path.join(OUT, 'oracle_raw'))
    STATES = S.STATES_DIR
    REPORT = S.suffixed(os.path.join(OUT, 'oracle_report.json'))


configure()
TASKS_JSONL = os.path.join(OUT, 'oracle_tasks.jsonl')
TASKS_TXT = os.path.join(ROOT, 'experiments', 'rft3_oracle_tasks.txt')
TOOLBENCH_IDS = os.path.join(ROOT, 'experiments', 'toolbench_external.txt')
EXCLUDE_REPOS = td.EXCLUDE_REPOS | {'Textualize/rich', 'psf/requests', 'encode/httpx'}
MAX_SPANS, MAX_LINES, MAX_STATEMENT = 3, 40, 6000
SEED = 0
CALL_SECONDS = 20.0
T0 = 1_800_000_000.0
MAX_EXTRA = 2
MAX_SHOW_DEFS = 3
REPAIR_SHARE = 0.12
SINGLE_SHARE = 0.25    # v20: multi-span files kept as separate single edits (seeded per task and file)
MAX_HUNKS = 4          # v20 edit.py: hunks per call
REPAIR_MISTAKES = ('start_only', 'json_old', 'json_both', 'line_prefix', 'indent_plus4')
SEARCH_TRIES = 10
MAX_TRIALS = 16
CALL_TIMEOUT = 180

STUB = r'''# rft3 import-check emulation (scripts/rft3_oracle.py), active only in `python -c "import <module>"`
# processes
# (the import checks of check_patch.py and of edit.py's guard) when <module> is a file of the workspace ($WORKSPACE):
# the module is registered as imported without running it. Kaggle's sandbox has the repository's dependencies
# installed (and the right Python), where the gold code imports; here they are missing.
import os
import sys


def _install():
    argv = list(getattr(sys, 'orig_argv', []) or [])
    if '-c' not in argv:
        return
    i = argv.index('-c')
    code = (argv[i + 1] if i + 1 < len(argv) else '').strip()
    ws = os.environ.get('WORKSPACE')
    if not ws or not code.startswith('import ') or ';' in code or '\n' in code:
        return
    mod = code[len('import '):].strip()
    parts = mod.split('.')
    if not parts or not all(p.isidentifier() for p in parts):
        return
    rel = os.path.join(*parts)
    if not any(os.path.isfile(os.path.join(ws, root, rel + suffix)) for root in ('', 'src')
               for suffix in ('.py', os.sep + '__init__.py')):
        return
    import types
    for k in range(1, len(parts) + 1):
        name = '.'.join(parts[:k])
        if name not in sys.modules:
            m = types.ModuleType(name)
            if k < len(parts):
                m.__path__ = []
            sys.modules[name] = m


_install()
'''


# ------------------------------------------------------------------ select

def gold_span_count(patch):
    n = 0
    for p, f in tb.parse_patch(patch).items():
        if f['kind'] == 'modified' and p.endswith('.py') and not tb.TEST_PATH.search(p):
            n += len(tb.spans_of(p, f['hunks']))
    return n


def extra_reason(r):
    """None when the instance also passes the rft3 filters, else the reason."""
    why = td.eligible(r)
    if why:
        return why
    if r['repo'] in EXCLUDE_REPOS:
        return 'repo excluded'
    n = gold_span_count(r['patch'])
    if not 1 <= n <= MAX_SPANS:
        return f'{n} gold spans'
    py = {p for p, k in td.patch_files(r['patch']) if p.endswith('.py') and not td.TEST_PATH.search(p)}
    if td.changed_lines(r['patch'], py) > MAX_LINES:
        return 'too many changed lines'
    if len(r['problem_statement'] or '') > MAX_STATEMENT:
        return 'long statement'
    return None


def toolbench_ids():
    return {l.split()[0] for l in open(TOOLBENCH_IDS) if l.strip() and not l.startswith('#')}


def select(n, per_repo):
    import pyarrow.parquet as pq
    rows = pq.read_table(td.PARQUET, columns=['repo', 'instance_id', 'base_commit', 'patch', 'problem_statement',
                                              'created_at']).to_pylist()
    bench = toolbench_ids()
    reasons = collections.Counter()
    by_repo = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r['instance_id']):
        why = extra_reason(r)
        if why is None and r['instance_id'] in bench:
            why = 'toolbench benchmark instance'
        reasons[why or 'ok'] += 1
        if why is None:
            by_repo[r['repo']].append(r)
    rng = random.Random(SEED)
    queues = {}
    for repo in sorted(by_repo):
        rs = by_repo[repo][:]
        rng.shuffle(rs)
        queues[repo] = rs[:per_repo]
    chosen = []
    while len(chosen) < n and any(queues.values()):
        for repo in sorted(queues):
            if queues[repo] and len(chosen) < n:
                chosen.append(queues[repo].pop(0))
    chosen.sort(key=lambda r: r['instance_id'])
    os.makedirs(OUT, exist_ok=True)
    with open(TASKS_JSONL, 'w') as f:
        for r in chosen:
            f.write(json.dumps({k: r[k] for k in ('instance_id', 'repo', 'base_commit', 'patch', 'problem_statement',
                                                  'created_at')}) + '\n')
    counts = collections.Counter(r['repo'] for r in chosen)
    assert not {r['instance_id'] for r in chosen} & bench
    with open(TASKS_TXT, 'w') as f:
        f.write(f'# rft3 oracle tasks: SWE-bench train, seed {SEED}, {len(chosen)} instances from {len(counts)} '
                f'repositories (<= {per_repo} each, round-robin); scripts/rft3_oracle.py select\n'
                f'# filters: toolbench_data.eligible + <= {MAX_SPANS} gold spans, <= {MAX_LINES} changed lines, '
                f'statement <= {MAX_STATEMENT} chars; excluded: the {len(bench)} toolbench benchmark instances, '
                f'{sorted(EXCLUDE_REPOS)}\n')
        f.writelines(r['instance_id'] + '\n' for r in chosen)
    print(f'{len(rows)} train instances; ' + ', '.join(f'{k}: {v}' for k, v in reasons.most_common()))
    print(f'selected {len(chosen)} from {len(counts)} repos: ' + ', '.join(f'{k} {v}' for k, v in
                                                                           sorted(counts.items(), key=lambda kv:
                                                                                  -kv[1])))


def load_tasks(only=None):
    keep = [l.strip() for l in open(TASKS_TXT) if l.strip() and not l.startswith('#')]
    recs = {}
    for l in open(TASKS_JSONL):
        d = json.loads(l)
        recs[d['instance_id']] = d
    out = [recs[i] for i in keep if i in recs]
    return [r for r in out if not only or r['instance_id'] in only]


# ------------------------------------------------------------------ workspace

HARNESS_INI = """[pytest]
addopts = -p no:anyio
norecursedirs = .* build dist venv
python_classes = Test* *Test
python_files = test_*.py *_test.py
filterwarnings =
    ignore::DeprecationWarning
    ignore::UserWarning
"""
HOOK_HEADER = '# Hermetic test discovery hook for SWE-gemma\n'


def harness_test_config(ws):
    """swegemma container_setup.setup_workspace_test_config (pytest.ini, conftest.py), byte for byte."""
    ini = os.path.join(ws, 'pytest.ini')
    existing = open(ini, errors='replace').read() if os.path.exists(ini) else ''
    if '# Hermetic test discovery' not in existing and '[pytest]' not in existing:
        open(ini, 'w').write(HARNESS_INI)
    conf = os.path.join(ws, 'conftest.py')
    existing = open(conf, errors='replace').read() if os.path.exists(conf) else ''
    if '# Hermetic test discovery hook for SWE-gemma' not in existing and \
            '# Standard test discovery hook for SWE-gemma' not in existing:
        clean = existing.replace(HOOK_HEADER, '').strip()
        open(conf, 'w').write(HOOK_HEADER + ('\n' + clean if clean else '') + '\n')


def make_ws(rec, mirror):
    """data/rft3/ws/<id> at base_commit (toolbench_data.make_ws + the harness's test config before the baseline)."""
    iid, sha = rec['instance_id'], rec['base_commit']
    ws = os.path.join(WS_ROOT, iid)
    if td.git(mirror, 'cat-file', '-t', sha, check=False).strip() != 'commit':
        td.git(mirror, 'fetch', '-q', 'origin', sha, check=False)
        if td.git(mirror, 'cat-file', '-t', sha, check=False).strip() != 'commit':
            return None, 'base_commit not in the repository'
    try:
        td.git(mirror, 'checkout', '-q', '-f', '--detach', sha)
    except RuntimeError as e:
        return None, f'checkout failed: {str(e)[:200]}'
    shutil.rmtree(ws, ignore_errors=True)
    os.makedirs(ws)
    for name in os.listdir(mirror):
        if name == '.git':
            continue
        subprocess.run(['cp', '-cRP', os.path.join(mirror, name), ws], check=True, capture_output=True)
    td.git(ws, 'init', '-q')
    with open(os.path.join(ws, '.git', 'objects', 'info', 'alternates'), 'w') as f:
        f.write(os.path.join(mirror, '.git', 'objects') + '\n')
    td.git(ws, 'update-ref', 'refs/heads/main', sha)
    td.git(ws, 'symbolic-ref', 'HEAD', 'refs/heads/main')
    td.git(ws, 'read-tree', 'HEAD')
    td.git(ws, 'update-index', '-q', '--refresh', check=False)
    os.makedirs(os.path.join(ws, '.git', 'info'), exist_ok=True)
    with open(os.path.join(ws, '.git', 'info', 'exclude'), 'a') as f:
        f.write(td.GIT_EXCLUDE)
    harness_test_config(ws)
    td.git(ws, 'add', '-A')
    td.git(ws, 'commit', '-q', '-m', 'baseline', '--allow-empty', '--no-verify')
    td.git(ws, 'tag', '-f', td.BASELINE)
    return ws, None


# ------------------------------------------------------------------ script runner

class Runner:
    """One flow session on one workspace: runs agent-v19 scripts as ADK does (toolbench.Session) on the simulated
    clock, with the stub environment for import checks; returns stdout/stderr/exit code and the ADK result JSON."""

    def __init__(self, agent, ws, state, python):
        self.agent, self.ws, self.state, self.python = agent, ws, state, python
        self.i = 0

    def run(self, script, args, index=None, state=None):
        i = self.i + 1 if index is None else index
        now = T0 + CALL_SECONDS * (i - 1)
        argv = [f'scripts/{script}'] + [x for k, v in (args or {}).items() for x in (f'--{k}', str(v))]
        env = dict(os.environ, WORKSPACE=self.ws, REPO_TOOLS_DIR=state or self.state, PYTHONDONTWRITEBYTECODE='1',
                   REPLAY_NOW=repr(now), REPLAY_END=repr(now + 1.0), GIT_TERMINAL_PROMPT='0', PYTHONHASHSEED='0',
                   PYTHONPATH=STUB_DIR)
        for k in ('REPO_TOOLS_FLOW', 'PYTHONHOME', 'VIRTUAL_ENV'):
            env.pop(k, None)
        with tempfile.TemporaryDirectory(prefix='adk_') as tmp:
            os.symlink(self.agent.scripts, os.path.join(tmp, 'scripts'))
            try:
                r = subprocess.run([self.python, '-c', tb.WRAPPER % (argv,)], cwd=tmp, env=env, capture_output=True,
                                   text=True, timeout=CALL_TIMEOUT, errors='replace')
                out, err, code = r.stdout, r.stderr, r.returncode
            except subprocess.TimeoutExpired:
                out, err, code = '', '', 'timeout'
        ws = self.ws.rstrip('/')
        fix = lambda s: s.replace(ws + '/', '/workspace/').replace(ws, '/workspace')  # noqa: E731
        out, err = fix(out), fix(err)
        res = S.adk_result(script, out, err, code, timeout=CALL_TIMEOUT)
        return {'script': script, 'args': dict(args or {}), 'out': out, 'err': err, 'code': code, 'result': res,
                'crash': 'Traceback (most recent call last)' in err}

    def call(self, script, args):
        self.i += 1
        return self.run(script, args, index=self.i)

    def trial(self, script, args, base_state=None, index=None):
        """Run the next call on a copy of the flow state (the workspace must not change: read-only scripts);
        returns (rec, state copy) -> commit() makes it the session's state."""
        cp = tempfile.mkdtemp(prefix='trial_', dir=os.path.dirname(self.state))
        shutil.rmtree(cp)
        tb.cp_tree(base_state or self.state, cp)
        return self.run(script, args, index=index or self.i + 1, state=cp), cp

    def commit(self, cp):
        shutil.rmtree(self.state, ignore_errors=True)
        os.rename(cp, self.state)
        self.i += 1


# ------------------------------------------------------------------ trajectory builder

WORD = r'(?<![\w.]){}(?![\w])'


def mentions(text, name):
    return bool(name) and re.search(WORD.format(re.escape(name)), text or '') is not None


def sym_in(text, q):
    """Is def q named in text? Its qualname as a word (Class.method), or its last part in a code-like position
    (`def x`, `.x`, `x(`, locate/outline's `x  L12-30`, backticks) -- for a method only when its class name also
    appears as a word in the same text, and never for a dunder (a bare `__init__` names nothing)."""
    if not q:
        return False
    parts = q.split('.')
    if len(parts) > 1 and mentions(text, q):
        return True
    last = parts[-1]
    if last.startswith('__') and last.endswith('__'):
        return False
    if len(parts) > 1 and not mentions(text, parts[-2]):
        return False
    last = re.escape(last)
    return re.search(rf'(?:\bdef |\.|`){last}\b|(?<![\w.]){last}(?:\(|`|  L\d)', text or '') is not None


def code_names(code):
    """Identifiers of real code: NAME tokens outside strings, docstrings and comments (tokenize; a regex over the code
    with strings and comments removed when tokenize fails on a fragment)."""
    import io
    import tokenize
    try:
        return {t.string for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type == tokenize.NAME}
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        stripped = re.sub(r'(?s)""".*?"""|\'\'\'.*?\'\'\'', '', code)
        stripped = re.sub(r'#[^\n]*|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', '', stripped)
        return set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', stripped))


def project_words(repo):
    """Lower-case parts (> 3 chars) of the repository's owner/name: identifiers containing them name the project
    (PyTorchLightning in Lightning-AI/lightning), not the code to change."""
    return {x for x in re.split(r'[-_/.]', repo.lower()) if len(x) > 3}


class Builder:
    def __init__(self, task, agent, ws, python, workdir):
        self.task, self.agent, self.ws = task, agent, ws
        self.state = os.path.join(workdir, 'state')
        os.makedirs(self.state)
        self.R = Runner(agent, ws, self.state, python)
        self.turns = []          # trajectory turns (rft3 schema, without ids)
        self.outs = []           # stdout per turn
        self.shown = collections.defaultdict(set)
        self.extra = 0           # recovery calls
        self.notes = []
        self.ps = task['problem_statement']
        self.gold = tb.gold_of(task['patch'], self.read)
        self.base = {f: self.read(f) for f in self.gold['files']}
        self._strings = {}

    def read(self, rel):
        return tb.read_file(self.ws, rel)

    def is_file(self, rel):
        return os.path.isfile(os.path.join(self.ws, rel))

    # ---- recording
    def add(self, rec, kind, grounding, label='train', meta=None):
        k = len(self.turns)
        call = S.script_call(rec['script'], rec['args']) if rec['script'] != 'submit_patch' else S.submit_call()
        self.turns.append({'call': call, 'result': rec['result'], 'label': label, 'thinking': None, 'visible': None,
                           'grounding': grounding, 'kind': kind, **({'meta': meta} if meta else {})})
        self.outs.append(rec['out'])
        for f, ls in tb.shown_lines(rec['out'], self.is_file).items():
            self.shown[f] |= ls
        return k

    def seen_text(self):
        return '\n'.join(self.outs)

    def where(self, pred):
        """The first turn whose output satisfies pred(text), else None."""
        return next((k for k, o in enumerate(self.outs) if pred(o)), None)

    def file_grounded(self, f):
        k = self.where(lambda o: f in o)
        if k is not None:
            return f'{f} printed at turn {k} ({self.kind(k)})'
        if f in self.ps:
            return f'{f} named in the issue'
        return None

    def symbol_grounded(self, f, q):
        """Where file f and def q (qualname or its last part) were printed together (same output), else None."""
        for k, o in enumerate(self.outs):
            if f in o and sym_in(self.section_of(o, f), q):
                return f'{f} and {q} printed at turn {k} ({self.kind(k)})'
        fk = self.where(lambda o: f in o)
        if fk is not None and sym_in(self.ps, q):
            return f'{f} printed at turn {fk} ({self.kind(fk)}), {q.split(".")[-1]} named in the issue'
        return None

    def kind(self, k):
        t = self.turns[k]
        return t.get('kind') or S.script_of(t['call'])

    def lines_printed(self, f, a, b):
        return all(x in self.shown[f] for x in range(a, b + 1))

    def printed_at(self, f, a, b):
        """The earliest turn after which lines a..b of f were all printed (numbered)."""
        acc = set()
        for k, o in enumerate(self.outs):
            acc |= tb.shown_lines(o, self.is_file).get(f, set())
            if all(x in acc for x in range(a, b + 1)):
                return k
        return None

    # ---- steps
    def locate(self):
        q = tb.build_query(self.ps)
        rec = self.R.call('locate.py', {'query': q})
        self.add(rec, 'locate', 'first call (system prompt): the issue title + identifiers, paths and errors of the '
                                'statement')
        self.loc = tb.parse_locate(rec['out'])
        rank = next((k + 1 for k, f in enumerate(self.loc['files']) if f in self.gold['files']), None)
        self.loc_rank = rank
        return rec

    def needed_reads(self):
        """Reads still needed: [('def', file, qual, [spans]) | ('range', file, lo, hi, span)] for spans whose anchored
        edit range is not yet printed."""
        out, by_def = [], collections.OrderedDict()
        for s in self.gold['spans']:
            a, b, _ = tb.anchored(s, self.base[s['file']].split('\n'))
            if self.lines_printed(s['file'], a, b):
                continue
            if s.get('def'):
                by_def.setdefault((s['file'], s['def'][2]), []).append(s)
            else:
                out.append(('range', s['file'], s))
        return [('def', f, q, ss) for (f, q), ss in by_def.items()] + out

    def first_steps(self, f, q):
        """Candidate recovery calls from the current state: [(script, args, kind, grounding)]."""
        out = []
        last = len(self.outs) - 1
        for script, args, _, _ in self.next_reads(self.outs[-1]):
            out.append((script, args, 'recovery_next', f'NEXT line of turn {last} ({self.kind(last)})'))
        if q and self.file_grounded(f):
            out.append(('outline.py', {'files': f}, 'recovery_outline', self.file_grounded(f) + '; its defs not '
                                                                                                'named yet'))
        for ident, src in self.search_candidates(f, q):
            out.append(('search.py', {'text': ident}, 'recovery_search', f'{ident!r} {src}'))
        title = tb.build_query(self.ps).split(' ')[:12]
        out.append(('history.py', {'similar': ' '.join(title)}, 'recovery_history', 'the issue title (past commits '
                                                                                     'with a similar message)'))
        return out

    @staticmethod
    def next_reads(out):
        """[(script, args, k, line)] of every NEXT line's first read call with its values filled in."""
        res = []
        for k, line in enumerate(tb.next_lines(out)):
            for script, args in tb.calls_in(line[5:]):
                if script in ('show.py', 'search.py', 'outline.py', 'callers.py') and args and \
                        not any(tb.TEMPLATE.search(v) for v in args.values()):
                    res.append((script, args, k, line))
                break
        return res

    def try_ground(self, f, q):
        """Recovery: make file f (and def q) grounded with <= MAX_EXTRA - self.extra real calls, tried on copies of the
        flow state (read-only scripts); only the chosen calls enter the trajectory. True when done."""
        budget = MAX_EXTRA - self.extra
        if budget <= 0:
            return False
        firsts = self.first_steps(f, q)
        tried, partial = 0, []
        for script, args, kind, why in firsts:
            if tried >= MAX_TRIALS:
                break
            tried += 1
            rec, cp = self.R.trial(script, args)
            if self.grounds(rec['out'], f, q):
                self.R.commit(cp)
                self.extra += 1
                self.add(rec, kind, why, meta={'recovery': kind.split('_')[1], **({'ident': args['text']}
                                                                                 if script == 'search.py' else {})})
                return True
            if budget >= 2 and (f in rec['out'] or self.next_reads(rec['out'])):
                partial.append((script, args, kind, why, rec, cp))
            else:
                shutil.rmtree(cp, ignore_errors=True)
        # two steps: the first output names the file (outline names the def) or its NEXT read leads there
        done = False
        for script, args, kind, why, rec, cp in partial:
            if done:
                shutil.rmtree(cp, ignore_errors=True)
                continue
            seconds = []
            if q and f in rec['out']:
                seconds.append(('outline.py', {'files': f}, 'recovery_outline', None))
            seconds += [(s2, a2, 'recovery_next', None) for s2, a2, _, _ in self.next_reads(rec['out'])]
            for s2, a2, k2, _ in seconds[:3]:
                if tried >= MAX_TRIALS:
                    break
                tried += 1
                rec2, cp2 = self.R.trial(s2, a2, base_state=cp, index=self.R.i + 2)
                if self.grounds(rec2['out'], f, q):
                    self.R.commit(cp)
                    self.extra += 1
                    self.add(rec, kind, why, meta={'recovery': kind.split('_')[1], **({'ident': args['text']}
                                                                                     if script == 'search.py' else {})})
                    self.R.commit(cp2)
                    self.extra += 1
                    g2 = (f'{f} printed at turn {len(self.turns) - 1} ({kind})' if k2 == 'recovery_outline' else
                          f'NEXT line of turn {len(self.turns) - 1} ({kind})')
                    self.add(rec2, k2, g2, meta={'recovery': k2.split('_')[1]})
                    done = True
                    break
                shutil.rmtree(cp2, ignore_errors=True)
            if not done:
                shutil.rmtree(cp, ignore_errors=True)
        return done

    def grounds(self, out, f, q):
        if f not in out:
            return False
        return not q or sym_in(self.section_of(out, f), q)

    def search_candidates(self, f, q):
        """[(identifier, why visible)] occurring in the gold def's code (or the span's region), visible in the issue or
        an earlier output, most specific first; at most SEARCH_TRIES."""
        base = self.base.get(f) or ''
        lines = base.split('\n')
        defs = [d for d in tb.def_ranges(base) if d[2] == q] if q else []
        if defs:
            code = '\n'.join(lines[defs[0][0] - 1:defs[0][1]])
        else:
            ss = [s for s in self.gold['spans'] if s['file'] == f]
            lo = max(1, min(s['a'] for s in ss) - 10) if ss else 1
            hi = max(s['b'] for s in ss) + 10 if ss else len(lines)
            code = '\n'.join(lines[lo - 1:hi])
        # identifiers of the code itself (not its docstrings or comments), never the project's own name
        import textwrap
        proj = project_words(self.task['repo'])
        code_ids = {t for t in code_names(textwrap.dedent(code))
                    if len(t) > 3 and not any(p in t.lower() for p in proj)}
        out, seen = [], set()
        sources = [(self.ps, 'named in the issue')] + [(o, f'printed at turn {k} ({self.kind(k)})')
                                                       for k, o in enumerate(self.outs)]
        for text, why in sources:
            for m in re.finditer(r'(?<![\w.])([A-Za-z_][A-Za-z0-9_]{3,})(?![\w])', text or ''):
                t = m.group(1)
                if t in seen or t not in code_ids or t in tb.STOP or not tb.is_identifier_like(t, False):
                    continue
                seen.add(t)
                out.append((t, why))
        # rarer identifiers first (more specific search output)
        out.sort(key=lambda x: (code.count(x[0]) == 0, len(re.findall(WORD.format(re.escape(x[0])), base)) > 20,
                                -len(x[0])))
        return out[:SEARCH_TRIES]

    def show_def(self, f, q):
        g = self.symbol_grounded(f, q)
        if g is None:
            return False
        rec = self.R.call('show.py', {'target': f, 'symbol': q})
        if rec['code'] not in (0, None) and not rec['out'].strip():
            return False
        self.add(rec, 'show', g)
        return True

    def range_read(self, f, lo, hi, why):
        rec = self.R.call('show.py', {'target': f, 'start': str(lo), 'end': str(hi)})
        self.add(rec, 'show_range', why)

    def def_range_grounding(self, f, a, b):
        """Why a range read of f lines a..b is grounded: a printed def L-range / fold notice containing it."""
        for k, o in enumerate(self.outs):
            for lo, hi in self.refs_by_file(o).get(f, ()):
                if lo < hi and lo <= a and b <= hi:
                    return f'range L{lo}-{hi} of {f} printed at turn {k} ({self.kind(k)})'
        return None

    def section_of(self, out, f):
        """The lines of OUT attributed to file f (see refs_by_file), plus every line naming f itself."""
        keep, cur = [], None
        for line in (out or '').splitlines():
            toks = [t.strip('[](),:;"\'`') for t in line.split()[:3]]
            path = next((t.split(':')[0] for t in toks if t.split(':')[0].endswith('.py') and
                         self.is_file(t.split(':')[0])), None)
            if path:
                cur = path
            if cur == f or f in line:
                keep.append(line)
        return '\n'.join(keep)

    def refs_by_file(self, out):
        """{file: [(lo, hi)]}: line references printed for each file (`L12`, `L12-30`, `lines 12-30`, `path:12`),
        attributed to the last workspace file named on or above the line (locate/outline/show/search layouts)."""
        refs, cur = collections.defaultdict(list), None
        for line in (out or '').splitlines():
            for m in re.finditer(r'([\w./-]+\.py):(\d+)\b', line):
                if self.is_file(m.group(1)):
                    refs[m.group(1)].append((int(m.group(2)), int(m.group(2))))
            toks = [t.strip('[](),:;"\'`') for t in line.split()[:3]]
            path = next((t.split(':')[0] for t in toks if t.split(':')[0].endswith('.py') and
                         self.is_file(t.split(':')[0])), None)
            if path:
                cur = path
            if cur is None:
                continue
            for m in re.finditer(r'\bL(\d+)(?:-(\d+))?\b|\blines (\d+)-(\d+)', line):
                lo = int(m.group(1) or m.group(3))
                hi = int(m.group(2) or m.group(4) or lo)
                refs[cur].append((lo, hi))
        return refs

    def near_name_grounding(self, f, a, b, only=None, near=12):
        """A line reference printed for file f (outline names:, locate/search lists) within NEAR lines of a..b: the
        place of the change is known."""
        for k, o in enumerate(self.outs):
            if only is not None and k != only:
                continue
            for lo, hi in self.refs_by_file(o).get(f, ()):
                if lo - near <= b and a <= hi + near:
                    return f'{f} line reference L{lo}{"-" + str(hi) if hi != lo else ""} printed at turn {k}'
        return None

    def range_bounds(self, f, a, b):
        """(lo, hi) of a range read around lines a..b: SHOW_PAD lines of context, but starting inside a multi-line
        string that contains the span (show.py folds a docstring whose opening line is in the range)."""
        lo, hi = max(1, a - tb.SHOW_PAD), b + tb.SHOW_PAD
        for s0, e0 in self.string_ranges(f):
            if lo <= s0 < a <= e0:
                lo = s0 + 1
        return lo, hi

    def string_ranges(self, f):
        if f not in self._strings:
            import ast
            out = []
            try:
                for n in ast.walk(ast.parse(self.base[f])):
                    if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.end_lineno and \
                            n.end_lineno > n.lineno:
                        out.append((n.lineno, n.end_lineno))
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                pass
            self._strings[f] = out
        return self._strings[f]

    def read_until_printed(self, f, a, b, lo, hi, why):
        """A grounded range read; once more from the fold start when the output folded the span."""
        self.range_read(f, lo, hi, why)
        if self.lines_printed(f, a, b):
            return True
        for m in re.finditer(r'lines (\d+)-(\d+)', self.outs[-1]):
            x, y = int(m.group(1)), int(m.group(2))
            if x <= b and a <= y and x > lo:
                self.range_read(f, x, max(hi, b + tb.SHOW_PAD), f'fold notice "lines {x}-{y}" at turn '
                                                                f'{len(self.turns) - 1}')
                return self.lines_printed(f, a, b)
        return False

    def ensure_printed(self, f, s):
        """Make the lines of span s's edit printed (numbered) with grounded calls. False when impossible."""
        a, b, _ = tb.anchored(s, self.base[f].split('\n'))
        if self.lines_printed(f, a, b):
            return True
        lo, hi = self.range_bounds(f, a, b)
        # a NEXT line of the last output that reads these lines (show.py of the file by range)
        for script, args, _, _ in self.next_reads(self.outs[-1]):
            if script == 'show.py' and args.get('target') == f and 'start' in args:
                try:
                    x, y = int(args['start']), int(args.get('end') or args['start'])
                except ValueError:
                    continue
                if x <= a and b <= y:
                    rec, cp = self.R.trial(script, args)
                    if all(v in tb.shown_lines(rec['out'], self.is_file).get(f, set()) for v in range(a, b + 1)):
                        self.R.commit(cp)
                        last = len(self.turns) - 1
                        self.add(rec, 'show_range', f'NEXT line of turn {last} ({self.kind(last)})')
                        return True
                    shutil.rmtree(cp, ignore_errors=True)
        g = self.def_range_grounding(f, a, b)
        if g is None and s.get('def') is None and a <= 60 and self.file_grounded(f):
            lo, g = 1, f'module-level change near the top of {f} ({self.file_grounded(f)})'
            hi = max(hi, 40)
            for s0, e0 in self.string_ranges(f):
                if lo <= s0 < a <= e0:
                    lo = s0 + 1
        if g is None:
            g = self.near_name_grounding(f, a, b)
        if g is None and s.get('def') is None and self.extra < MAX_EXTRA and self.file_grounded(f):
            # recovery: outline.py of the file, or a search for an identifier of the changed region, whose output
            # names a line near the change
            cands = [('outline.py', {'files': f}, 'recovery_outline', self.file_grounded(f) + '; where its '
                                                                                            'module-level code is')]
            cands += [('search.py', {'text': ident}, 'recovery_search', f'{ident!r} {src}')
                      for ident, src in self.search_candidates(f, None)]
            for script, args, kind, why in cands[:MAX_TRIALS]:
                rec, cp = self.R.trial(script, args)
                self.outs.append(rec['out'])
                g = self.near_name_grounding(f, a, b, only=len(self.outs) - 1)
                self.outs.pop()
                if g is None:
                    shutil.rmtree(cp, ignore_errors=True)
                    continue
                self.R.commit(cp)
                self.extra += 1
                self.add(rec, kind, why, meta={'recovery': kind.split('_')[1], **({'ident': args['text']}
                                                                                 if script == 'search.py' else {})})
                g = self.near_name_grounding(f, a, b)
                break
        if g is None:
            return False
        return self.read_until_printed(f, a, b, lo, hi, g)

    def reads(self):
        """Clean / recovery reads until every span's edit range is printed. Returns None or the discard reason."""
        defs_shown = 0
        for item in self.needed_reads():
            if item[0] == 'def':
                _, f, q, ss = item
                if all(self.lines_printed(s['file'], *tb.anchored(s, self.base[f].split('\n'))[:2]) for s in ss):
                    continue
                far = self.loc_rank is not None and self.loc_rank > 3 and self.extra == 0 and f not in self.ps and \
                    defs_shown == 0
                if self.symbol_grounded(f, q) is None:
                    rank_ok = self.loc_rank is not None and self.loc_rank <= 3
                    if not self.try_ground(f, q):
                        return 'ungrounded: no recovery path to ' + ('the def' if rank_ok or self.file_grounded(f)
                                                                     else 'the file')
                elif far and not self.try_ground(f, q):
                    # gold far down in locate's list (rank > 3) but listed with its def: no independent pointer
                    # found within MAX_EXTRA calls; the direct read stays (grounded by the list)
                    self.notes.append(f'gold at locate rank {self.loc_rank}: read directly from the list')
                if defs_shown >= MAX_SHOW_DEFS:
                    return 'too many defs to show'
                if not self.show_def(f, q):
                    return 'show failed'
                defs_shown += 1
                for s in ss:
                    if not self.ensure_printed(f, s):
                        return 'span not printed by the def read (no grounded range read)'
            else:
                _, f, s = item
                if self.file_grounded(f) is None and not self.try_ground(f, None):
                    return 'ungrounded: module-level span in a file never printed'
                if not self.ensure_printed(f, s):
                    return 'ungrounded: module-level span'
        if self.loc_rank is None or self.loc_rank > 3:
            if self.extra == 0:
                # gold file not in locate's top 3 but every read grounded otherwise (named in the issue or by a gold
                # read's names: block): a recovery-free path that does not start from locate's top candidates
                self.notes.append('gold file not in the top 3 of locate')
        return None

    def edits(self, repair=None):
        """The gold edits, file by file. v19 (and every repair trajectory, and agents without multi-edit): bottom-up
        line-range edits per span (single_edits). v20: a file with 2..MAX_HUNKS spans, all printed before the edits
        (always so here: the reads come first), gets ONE multi-edit call (multi_edit; numbers as read, edit.py applies
        bottom-up), except a seeded SINGLE_SHARE of such files, which keep single edits ordered so that a hunk
        defining a name another hunk uses comes first when the line numbers allow it (single_order; when no order
        avoids a transient `undefined name` CHECK line, the file gets the multi-edit after all: form *_dependent).
        repair: (rng) -> the first edit (the bottom span of the first file) is first sent with a mistake.
        Returns ({statuses, merges, repair, forms}, None) or (None, discard reason)."""
        st = {'statuses': [], 'merges': 0, 'repair': None, 'forms': []}
        files = self.gold['files']
        if repair is not None:
            f = files[0]
            base_lines = self.base[f].split('\n')
            span = sorted([s for s in self.gold['spans'] if s['file'] == f], key=lambda s: s['a'])[-1]
            a, b, _ = tb.anchored(span, base_lines)
            pk = self.printed_at(f, a, b)
            if pk is None:
                return None, f'edit lines {a}-{b} of {f} never printed'
            st['repair'] = self.try_repair(f, span, base_lines, repair, pk)
            if st['repair'] == 'failed':
                return None, 'repair: corrected edit not applied'
            if st['repair'] == 'done':
                st['statuses'].append('repaired')
        v19 = st['repair'] == 'done' or not self.agent.multi_edit
        for fi, f in enumerate(files):
            base_lines = self.base[f].split('\n')
            fs = sorted([s for s in self.gold['spans'] if s['file'] == f], key=lambda s: s['a'])
            if fi == 0 and st['repair'] == 'done':
                fs = fs[:-1]
            if not fs:
                continue
            if v19 or len(fs) == 1:
                why = self.single_edits(f, fs, base_lines, st)
                st['forms'].append('single' if len(fs) == 1 else 'single_v19')
            elif len(fs) > MAX_HUNKS:
                why = self.single_edits(f, fs, base_lines, st)
                st['forms'].append('single_many')
            elif single_selected(self.task['instance_id'], f) and not (order := self.single_order(f, fs,
                                                                                                    base_lines))[1]:
                why = self.single_edits(f, fs, base_lines, st, order=order[0])
                st['forms'].append('single_selected' + ('_reordered' if order[0] else ''))
            else:
                # (a selected file whose single edits would show a transient undefined name in a CHECK line in every
                # order the line numbers allow - a use applied before its import - gets the multi-edit too)
                form = self.multi_edit(f, fs, base_lines, st)
                if form and single_selected(self.task['instance_id'], f):
                    form += '_dependent'
                why = None
                if form is None:  # refused or not the gold result: single edits as v19
                    why = self.single_edits(f, fs, base_lines, st)
                    form = 'single_multi_refused'
                st['forms'].append(form)
            if why:
                return None, why
        return st, None

    def single_edits(self, f, fs, base_lines, st, order=None):
        """Line-range edits of spans FS of file f (ascending), one call each. Default: bottom-up (base numbers stay
        valid; a refused edit is retried merged with the span above, as the toolbench oracle). ORDER (single_order):
        the spans in that order, none may be refused. Returns None or the discard reason."""
        if order:
            for span in order:
                a, b, new = tb.anchored(span, base_lines)
                pk = self.printed_at(f, a, b)
                if pk is None:
                    return f'edit lines {a}-{b} of {f} never printed'
                rec = self.R.call('edit.py', {'target': f, 'start': str(a), 'end': str(b), 'new': '\n'.join(new)})
                stt = tb.edit_status(rec['out'])
                self.add(rec, 'edit', f'lines {a}-{b} of {f} printed numbered at turn {pk} ({self.kind(pk)})',
                         meta={'order': 'definitions first'})
                st['statuses'].append(stt)
                if stt != 'applied':
                    return 'reordered single edit not applied'
            return None
        i = len(fs) - 1
        while i >= 0:
            j = i
            while True:
                span = fs[i] if j == i else tb.merge_spans(fs[j:i + 1], base_lines)
                a, b, new = tb.anchored(span, base_lines)
                pk = self.printed_at(f, a, b)
                if pk is None:
                    return f'edit lines {a}-{b} of {f} never printed'
                rec = self.R.call('edit.py', {'target': f, 'start': str(a), 'end': str(b), 'new': '\n'.join(new)})
                stt = tb.edit_status(rec['out'])
                self.add(rec, 'edit', f'lines {a}-{b} of {f} printed numbered at turn {pk} ({self.kind(pk)})')
                st['statuses'].append(stt)
                if stt == 'applied' or j == 0:
                    break
                j -= 1
                st['merges'] += 1
            i = j - 1
        return None

    @staticmethod
    def hunk_groups(fs, base_lines):
        """Spans FS (ascending) grouped so that no two anchored line ranges overlap (edit.py refuses overlapping hunks
        in one call): [(a, b, new lines, [spans])], ascending."""
        groups = []
        for s in fs:
            a, b, new = tb.anchored(s, base_lines)
            if groups and a <= groups[-1][1]:
                ss = groups[-1][3] + [s]
                a, b, new = tb.anchored(tb.merge_spans(ss, base_lines), base_lines)
                groups[-1] = (a, b, new, ss)
                continue
            groups.append((a, b, new, [s]))
        return groups

    @staticmethod
    def multi_args(f, groups):
        """edit.py args of one multi-edit call: hunk 1 start/end/new, hunk n start<n>/end<n>/new<n> (top-down, base
        numbers)."""
        args = {'target': f}
        for n, (a, b, new, _) in enumerate(groups, 1):
            sfx = '' if n == 1 else str(n)
            args.update({f'start{sfx}': str(a), f'end{sfx}': str(b), f'new{sfx}': '\n'.join(new)})
        return args

    def multi_edit(self, f, fs, base_lines, st):
        """One edit.py call with every span of file f as a hunk (multi_args). Run on a copy of the flow state: kept
        only when applied with exactly the gold result, else the file is restored and None returned (the caller falls
        back to single edits)."""
        groups = self.hunk_groups(fs, base_lines)
        if len(groups) < 2:
            return None
        where = []
        for a, b, _, _ in groups:
            pk = self.printed_at(f, a, b)
            if pk is None:
                return None
            where.append(f'lines {a}-{b} printed numbered at turn {pk} ({self.kind(pk)})')
        before = self.read(f)
        want = tb.apply_spans(self.base[f], fs)
        rec, cp = self.R.trial('edit.py', self.multi_args(f, groups))
        got = self.read(f) or ''
        ok = tb.edit_status(rec['out']) == 'applied' and (got == want or [x.rstrip() for x in got.split('\n')] ==
                                                           [x.rstrip() for x in want.split('\n')])
        if not ok:
            if got != before:
                open(os.path.join(self.ws, f), 'w', newline='').write(before)
            shutil.rmtree(cp, ignore_errors=True)
            self.notes.append(f'multi-edit of {f} not applied as gold ({tb.edit_status(rec["out"])}): single edits')
            return None
        self.R.commit(cp)
        self.add(rec, 'edit', f'{f}: ' + '; '.join(where), meta={'hunks': len(groups)})
        st['statuses'].append('applied')
        return 'multi' if len(groups) == len(fs) else 'multi_merged'

    def single_order(self, f, fs, base_lines):
        """(order, transient) for single edits of FS: order None = bottom-up (the default; base numbers stay valid),
        else the spans in an order where a hunk that defines or imports a name another hunk uses comes first
        (edit.py's CHECK line reports a new `undefined name` after each call: bottom-up applies the use before the
        import). An order is allowed only when every span keeps its base numbers (the spans above it applied before
        it keep the line count) and every intermediate file parses. transient: the order used (bottom-up) still
        leaves a new undefined name after some call."""
        import ast
        import itertools
        g = guards(self.agent.scripts)
        spans = [(s, *tb.anchored(s, base_lines)) for s in fs]

        def transients(order):
            lines = list(base_lines)
            done, n = [], 0
            for s, a, b, new in order:
                for _, a2, b2, new2 in done:
                    if a2 < a and (b2 >= a or len(new2) != b2 - a2 + 1):
                        return None  # a span above, applied earlier, moved (or overlaps) these lines
                before = '\n'.join(lines)
                lines[a - 1:b] = new
                after = '\n'.join(lines)
                try:
                    ast.parse(after)
                except (SyntaxError, ValueError, RecursionError, MemoryError):
                    return None
                if g.new_undefined_names(before, after):
                    n += 1
                done.append((s, a, b, new))
            return n
        bottom = sorted(spans, key=lambda x: -x[1])
        if not transients(bottom):
            return None, False
        best = None
        for perm in itertools.permutations(spans):
            if [x[1] for x in perm] == [x[1] for x in bottom]:
                continue
            t = transients(perm)
            if t is not None and (best is None or t < best[0]):
                best = (t, perm)
        if best is not None and best[0] == 0:
            return [x[0] for x in best[1]], False
        return None, True

    def try_repair(self, f, span, base_lines, rng, pk):
        """One mistaken edit that edit.py refuses (context turn), then the corrected call (train): 'done', 'none' (no
        mistake was refused: nothing recorded) or 'failed' (recorded, but the correction did not apply)."""
        names = list(REPAIR_MISTAKES)
        rng.shuffle(names)
        cands = {n: (e, goal) for n, e, goal in tb.mistakes(span, base_lines, tb.string_lines(self.base[f]))
                 if n in names and len(e) == 1}
        a, b, new = tb.anchored(span, base_lines)
        clean = {'target': f, 'start': str(a), 'end': str(b), 'new': '\n'.join(new)}
        for name in names:
            if name not in cands:
                continue
            edits, goal = cands[name]
            before = self.read(f)
            want = goal(self.base[f]) if before == self.base[f] else None
            rec, cp = self.R.trial('edit.py', edits[0])
            after = self.read(f)
            if after != before:  # applied (intended, wrong or cosmetic): not a refusal; undo
                open(os.path.join(self.ws, f), 'w', newline='').write(before)
                shutil.rmtree(cp, ignore_errors=True)
                continue
            if tb.edit_status(rec['out']) not in ('refused', 'usage'):
                shutil.rmtree(cp, ignore_errors=True)
                continue
            self.R.commit(cp)
            self.add(rec, 'edit_mistake', f'mistake {name}: lines {a}-{b} of {f} printed at turn {pk}',
                     label='context', meta={'mistake': name})
            mk = len(self.turns) - 1
            fixed = None
            rep = tb.repair_call(rec['out'])
            if rep is not None:
                args = tb.fill_repair(rep, edits[0], f)
                r2, cp2 = self.R.trial('edit.py', args)
                got = self.read(f)
                if want is not None and got == want:
                    self.R.commit(cp2)
                    self.add(r2, 'edit_repair', f'the ready edit.py call printed by the refusal at turn {mk}',
                             meta={'mistake': name, 'via': 'printed repair call'})
                    fixed = True
                else:
                    open(os.path.join(self.ws, f), 'w', newline='').write(before)
                    shutil.rmtree(cp2, ignore_errors=True)
            if not fixed:
                r3 = self.R.call('edit.py', clean)
                self.add(r3, 'edit_repair', f'refusal at turn {mk}; lines {a}-{b} of {f} printed numbered at turn '
                                            f'{pk} ({self.kind(pk)})', meta={'mistake': name, 'via': 'clean edit'})
                if tb.edit_status(r3['out']) != 'applied':
                    return 'failed'
            return 'done'
        return 'none'

    def base_import_failures(self):
        """Gold modules whose import fails on the untouched workspace with the stub environment (check_patch.py's
        own check: `python -c "import <module>"`, cwd = workspace): the environment, not the patch."""
        bad = []
        for f in self.gold['files']:
            parts = f[:-3].split('/')
            if parts and parts[0] == 'src':
                parts = parts[1:]
            if parts and parts[-1] == '__init__':
                parts = parts[:-1]
            if not parts or not all(p.isidentifier() for p in parts):
                continue
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=STUB_DIR, PYTHONHASHSEED='0',
                       WORKSPACE=self.ws)
            env.pop('VIRTUAL_ENV', None)
            try:
                r = subprocess.run([self.R.python, '-c', f'import {".".join(parts)}'], cwd=self.ws, env=env,
                                   capture_output=True, text=True, timeout=60)
                ok = r.returncode == 0
                last = (r.stderr.strip().splitlines() or ['?'])[-1][:160]
            except subprocess.TimeoutExpired:
                ok, last = False, 'timeout'
            if not ok:
                bad.append(f'{f}: {last}')
        return bad

    def check_and_submit(self, skip_check=None):
        """check_patch.py (unless skip_check: a gold module does not import here even before the edits, so its import
        check would report the environment, not the patch), then submit_patch."""
        ready, out = None, ''
        if skip_check:
            why = f'edits done (check_patch.py skipped here: {skip_check})'
        else:
            rec = self.R.call('check_patch.py', {})
            self.add(rec, 'check', 'after the edits (system prompt: check_patch.py, then submit_patch)')
            out = rec['out']
            ready = bool(re.search(r'^READY\b', out, re.M))
            why = f'check_patch at turn {len(self.turns) - 1}: ' + ('READY: call submit_patch now' if ready else
                                                                    'printed problems')
        td.git(self.ws, 'add', '-N', '.', check=False)
        diff = subprocess.run(['git', '-C', self.ws, 'diff', '--binary', td.BASELINE], capture_output=True,
                              text=True, errors='replace').stdout
        self.R.i += 1
        sub = {'script': 'submit_patch', 'args': {}, 'out': '', 'result': S.submit_result(diff)}
        self.add(sub, 'submit', why)
        return ready, out, diff

    def final_matches(self):
        exact = ws_eq = True
        for f in self.gold['files']:
            want = tb.apply_spans(self.base[f], [s for s in self.gold['spans'] if s['file'] == f])
            got = self.read(f) or ''
            exact &= got == want
            ws_eq &= [l.rstrip() for l in got.split('\n')] == [l.rstrip() for l in want.split('\n')]
        return exact, ws_eq


def single_selected(iid, f):
    """v20: is this multi-span file of task IID kept as single edits (SINGLE_SHARE, seeded)?"""
    return int(hashlib.sha1(f'single|{iid}|{f}'.encode()).hexdigest()[:8], 16) / 0xffffffff < SINGLE_SHARE


_GUARDS = {}


def guards(scripts):
    """The agent's own guards module in SCRIPTS (new_undefined_names: what edit.py's CHECK line reports)."""
    if scripts not in _GUARDS:
        import importlib.util
        spec = importlib.util.spec_from_file_location(f'rft3_agent_guards_{len(_GUARDS)}',
                                                      os.path.join(scripts, 'guards.py'))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _GUARDS[scripts] = m
    return _GUARDS[scripts]


def repair_selected(iid):
    return int(hashlib.sha1(f'repair|{iid}'.encode()).hexdigest()[:8], 16) / 0xffffffff < REPAIR_SHARE


def build_one(task, agent, python, mirror):
    """(record or None, discard reason or None, info)."""
    iid = task['instance_id']
    workdir = tempfile.mkdtemp(prefix=f'rft3_{iid[:40]}_', dir=TMP_ROOT)
    ws = None
    t0 = time.time()
    try:
        ws, why = make_ws(task, mirror)
        if why:
            return None, f'setup: {why}', {}
        b = Builder(task, agent, ws, python, workdir)
        g = b.gold
        if not g['spans']:
            return None, 'no modified .py gold span', {}
        if g['unparsable']:
            return None, 'gold file does not parse (Python 2?)', {}
        tree = S.workspace_tree(ws)
        env_bad = b.base_import_failures()
        b.locate()
        why = b.reads()
        info = {'locate_rank': b.loc_rank, 'extra': b.extra}
        if why:
            return None, why, info
        rng = random.Random(f'rft3|{iid}')
        res, why = b.edits(repair=rng if repair_selected(iid) else None)
        if why:
            return None, why, info
        exact, ws_eq = b.final_matches()
        if not (exact or ws_eq):
            return None, 'final != gold', info
        if 'refused' in res['statuses'] or 'usage' in res['statuses']:
            info['refused_then_merged'] = True
        ready, cp_out, diff = b.check_and_submit(skip_check='a gold module does not import at baseline in this '
                                                            'environment' if env_bad else None)
        problems = [l.strip()[2:] for l in cp_out.splitlines() if l.startswith('  - ')]
        if env_bad:
            info['env_import_failures'] = env_bad[:2]
        if ready is False and problems:
            info['check_problems'] = problems[:3]
            return None, 'check_patch problems: ' + re.sub(r'\S+\.py', '<f>', problems[0])[:60], info
        if any(t.get('kind') == 'edit_mistake' for t in b.turns):
            source = 'oracle_repair'
        elif b.extra:
            source = 'oracle_recovery'
        else:
            source = 'oracle_clean'
        turns = []
        for k, t in enumerate(b.turns):
            turns.append({'id': S.call_id('rft3', iid, k), **t})
        rec = {'id': f'{source}_{iid}', 'source': source, 'task': iid, 'repo': task['repo'],
               'request': S.build_request(agent, task['repo'], task['problem_statement'], tree), 'turns': turns,
               'gold': {'files': g['files'], 'functions': g['functions'],
                        'spans': [{'file': s['file'], 'a': s['a'], 'b': s['b']} for s in g['spans']]},
               'meta': {'locate_rank': b.loc_rank, 'locate_query': tb.build_query(task['problem_statement']),
                        'extra_calls': b.extra, 'edit_statuses': res['statuses'], 'merged_retries': res['merges'],
                        'edit_forms': res['forms'], 'agent': S.AGENT_TAG,
                        'final_exact': exact, 'check_ready': ready, 'check_skipped_env': env_bad[:2] or None,
                        'notes': b.notes, 'base_commit': task['base_commit'],
                        'created_at': task.get('created_at'), 'call_seconds': CALL_SECONDS,
                        'crashes': sum(1 for o in b.turns if 'Traceback (most recent call last)' in
                                       (json.loads(o['result']).get('stderr') or '')),
                        'build_seconds': round(time.time() - t0, 1)}}
        return rec, None, info
    except Exception:  # noqa: BLE001 - one broken task must not stop the batch
        return None, 'error: ' + traceback.format_exc()[-300:].replace('\n', ' | '), {}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if ws:
            shutil.rmtree(ws, ignore_errors=True)


def build_repo(job):
    repo, tasks, python = job
    agent = S.Agent()
    path = os.path.join(RAW_DIR, repo.replace('/', '__') + '.jsonl')
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)['task'] for l in open(path) if l.strip()}
    todo = [t for t in tasks if t['instance_id'] not in done]
    if not todo:
        return repo, 0, 0
    try:
        mirror = td.ensure_mirror(repo)
    except Exception as e:  # noqa: BLE001
        with open(path, 'a') as f:
            for t in todo:
                f.write(json.dumps({'task': t['instance_id'], 'repo': repo, 'discard': f'setup: clone: {e}'[:300]})
                        + '\n')
        return repo, len(todo), 0
    lock = open(mirror + '.rft3.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    n_ok = 0
    try:
        for t in todo:
            rec, why, info = build_one(t, agent, python, mirror)
            line = {'task': t['instance_id'], 'repo': repo, 'info': info}
            if rec:
                line['record'] = rec
                n_ok += 1
            else:
                line['discard'] = why
            with open(path, 'a') as f:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return repo, len(todo), n_ok


def build(workers, only=None, limit=0, python=None):
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(TMP_ROOT, exist_ok=True)
    os.makedirs(STUB_DIR, exist_ok=True)
    open(os.path.join(STUB_DIR, 'sitecustomize.py'), 'w').write(STUB)
    S.Agent()  # materialize once
    tasks = load_tasks(only)
    by = collections.defaultdict(list)
    for t in tasks:
        by[t['repo']].append(t)
    if limit:
        by = {r: ts[:limit] for r, ts in by.items()}
    python = python or tb.default_python()
    jobs = sorted(((r, ts, python) for r, ts in by.items()), key=lambda j: -len(j[1]))
    t0 = time.time()
    print(f'{sum(len(j[1]) for j in jobs)} tasks, {len(jobs)} repos, workers {workers}, python {python}', flush=True)
    with concurrent.futures.ProcessPoolExecutor(workers) as ex:
        for repo, n, ok in ex.map(build_repo, jobs):
            print(f'{repo}: {ok}/{n} built ({time.time() - t0:.0f}s)', flush=True)


# ------------------------------------------------------------------ collect

def collect(max_prompt=S.COMPACT_AT):
    lines = []
    for fn in sorted(os.listdir(RAW_DIR)):
        if fn.endswith('.jsonl'):
            lines += [json.loads(l) for l in open(os.path.join(RAW_DIR, fn)) if l.strip()]
    order = {t: k for k, t in enumerate(l.strip() for l in open(TASKS_TXT) if l.strip() and not l.startswith('#'))}
    lines = [l for l in lines if l['task'] in order]
    lines.sort(key=lambda l: order[l['task']])
    discards = collections.Counter()
    out = collections.defaultdict(list)
    tokens = []
    for l in lines:
        if 'discard' in l:
            discards[re.sub(r'\d+', 'N', l['discard'].split(':')[0] if l['discard'].startswith(('error', 'setup'))
                            else l['discard'])] += 1
            continue
        rec = l['record']
        bad = check_not_clean(rec)
        if bad:  # v20: the gold leaves a CHECK warning (a checker false alarm): would teach "submit despite CHECK"
            discards['CHECK line not clean after the last gold edit of a file'] += 1
            continue
        n, est = S.prompt_tokens(rec)
        rec['meta']['rendered_prompt_tokens'] = n
        rec['meta']['est_prompt_tokens'] = est
        tokens.append(est)
        if est >= max_prompt:
            discards['estimated prompt >= 14,336 (compaction)'] += 1
            continue
        out[rec['source']].append(rec)
    os.makedirs(STATES, exist_ok=True)
    for src in ('oracle_clean', 'oracle_recovery', 'oracle_repair'):
        with open(os.path.join(STATES, f'{src}.jsonl'), 'w') as f:
            for r in out[src]:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    kept = [r for rs in out.values() for r in rs]
    rep = {'tasks': len(lines), 'kept': len(kept), 'by_source': {k: len(v) for k, v in out.items()},
           'discards': dict(discards.most_common()),
           'by_repo': dict(collections.Counter(r['repo'] for r in kept).most_common()),
           'calls': dict(collections.Counter(len(r['turns']) for r in kept)),
           'locate_rank': dict(collections.Counter(str(r['meta']['locate_rank']) for r in kept)),
           'check_ready': sum(bool(r['meta']['check_ready']) for r in kept),
           'check_skipped_env': sum(bool(r['meta'].get('check_skipped_env')) for r in kept),
           'recovery_kinds': dict(collections.Counter(t['meta']['recovery'] for r in kept for t in r['turns']
                                                      if t.get('meta', {}).get('recovery'))),
           'repair_mistakes': dict(collections.Counter(t['meta']['mistake'] for r in kept for t in r['turns']
                                                       if t.get('kind') == 'edit_mistake')),
           'repair_via': dict(collections.Counter(t['meta']['via'] for r in kept for t in r['turns']
                                                  if t.get('kind') == 'edit_repair'))}
    rep.update(edit_report(kept))
    json.dump(rep, open(REPORT, 'w'), indent=1)
    print(json.dumps(rep, indent=1))
    print(S.stats([os.path.join(STATES, f'{s}.jsonl') for s in ('oracle_clean', 'oracle_recovery', 'oracle_repair')
                   if out[s]]))


CHECK_RE = re.compile(r'^CHECK: (.*)$', re.M)


def edit_turns(rec):
    """[(k, turn, stdout)] of the record's applied-or-not edit.py calls with label train (edits and repairs)."""
    out = []
    for k, t in enumerate(rec['turns']):
        if S.script_of(t['call']) == 'edit.py' and t['label'] == 'train':
            try:
                stdout = json.loads(t['result']).get('stdout') or ''
            except (ValueError, AttributeError):
                stdout = ''
            out.append((k, t, stdout))
    return out


def check_status(stdout):
    """clean | undefined | timeout | other | none: edit.py's v20 CHECK line of one result."""
    m = CHECK_RE.search(stdout or '')
    if not m:
        return 'none'
    c = m.group(1)
    if re.search(r"undefined name '", c):
        return 'undefined'
    if 'timed out' in c:
        return 'timeout'
    return 'clean' if c.startswith('compiles') else 'other'


def check_not_clean(rec):
    """Files whose LAST applied edit in the record has a CHECK line other than clean (a transient warning that a later
    edit of the same file clears is not counted); [] for agents without CHECK lines (v19)."""
    last = {}
    for k, t, out in edit_turns(rec):
        if tb.edit_status(out) == 'applied':
            last[(t['call']['arguments'].get('args') or {}).get('target')] = check_status(out)
    if all(st == 'none' for st in last.values()):
        return []
    return [f for f, st in last.items() if st != 'clean']


def edit_report(recs):
    """Edit forms (multi / single), CHECK-line cleanliness and NOTE lines of the trained edit calls."""
    forms = collections.Counter()
    turns = collections.Counter()
    hunks = collections.Counter()
    checks = collections.Counter()
    notes = collections.Counter()
    transient = collections.Counter()
    eligible = with_multi = 0
    for r in recs:
        fs = r['meta'].get('edit_forms') or []
        forms.update(fs)
        multi_files = [x for x in fs if x != 'single']
        eligible += bool(multi_files)
        has_multi = False
        bad = False
        for k, t, out in edit_turns(r):
            h = (t.get('meta') or {}).get('hunks', 1)
            has_multi |= h > 1
            hunks[h] += 1
            form = 'multi' if h > 1 else ('repair' if t.get('kind') == 'edit_repair' else 'single')
            turns[form] += 1
            if tb.edit_status(out) == 'applied':
                cs = check_status(out)
                checks[f'{form}: {cs}'] += 1
                bad |= cs != 'clean'
            for line in out.splitlines():
                if line.startswith('NOTE:'):
                    notes[re.sub(r'\d+', 'N', line[5:].strip())[:70]] += 1
        with_multi += has_multi
        if bad:
            transient[r['source']] += 1
    return {'edit_forms_per_file': dict(forms.most_common()), 'edit_turns': dict(turns),
            'hunks_per_edit_call': dict(sorted(hunks.items())),
            'trajectories_with_multispan_file': eligible, 'trajectories_with_multi_edit': with_multi,
            'check_lines_of_applied_edits': dict(sorted(checks.items())),
            'trajectories_with_a_not_clean_check': dict(transient),
            'note_lines_in_edit_results': dict(notes.most_common(12))}


def raw_lines(raw_dir):
    out = {}
    for fn in sorted(os.listdir(raw_dir)):
        if fn.endswith('.jsonl'):
            for line in open(os.path.join(raw_dir, fn)):
                if line.strip():
                    d = json.loads(line)
                    out[d['task']] = d
    return out


def outcome(d, kept_ids):
    if d is None:
        return 'missing'
    if 'discard' in d:
        return 'discard: ' + re.sub(r'\d+', 'N', d['discard'].split(':')[0] if d['discard'].startswith(
            ('error', 'setup', 'edit lines')) else d['discard'])[:60]
    r = d['record']
    if r['id'] in kept_ids:
        return r['source']
    return 'discard: CHECK line not clean after the last gold edit of a file' if check_not_clean(r) else \
        'discard: estimated prompt >= 14,336 (compaction)'


def compare(base='v19'):
    """Per task: the outcome (source or discard reason) in BASE's build vs this version's (raw lines + the kept states
    of each)."""
    cur_raw, cur_states = RAW_DIR, STATES
    old_raw = os.path.join(OUT, 'oracle_raw') if base == 'v19' else os.path.join(OUT, f'oracle_raw_{base}')
    old_states = os.path.join(OUT, S.VERSIONS[base]['states'])
    tasks = [l.strip() for l in open(TASKS_TXT) if l.strip() and not l.startswith('#')]

    def kept(d):
        ids = set()
        for src in ('oracle_clean', 'oracle_recovery', 'oracle_repair'):
            p = os.path.join(d, f'{src}.jsonl')
            if os.path.exists(p):
                ids |= {json.loads(l)['id'] for l in open(p) if l.strip()}
        return ids
    a, b = raw_lines(old_raw), raw_lines(cur_raw)
    ka, kb = kept(old_states), kept(cur_states)
    pairs = collections.Counter((outcome(a.get(t), ka), outcome(b.get(t), kb)) for t in tasks)
    tot_a = collections.Counter(x for x, _ in pairs.elements())
    tot_b = collections.Counter(y for _, y in pairs.elements())
    print(f'{base} -> {S.VERSION}: {len(tasks)} tasks')
    for k in sorted(set(tot_a) | set(tot_b), key=lambda k: -max(tot_a[k], tot_b[k])):
        print(f'  {k:70s} {tot_a[k]:5d} -> {tot_b[k]:5d}')
    print('changed outcomes:')
    for (x, y), n in pairs.most_common():
        if x != y:
            print(f'  {n:4d}  {x}  ->  {y}')
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--version', default=None, help='agent version (rft3_states.VERSIONS: v19 default, v20)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select')
    s.add_argument('--n', type=int, default=3000)
    s.add_argument('--per-repo', type=int, default=120)
    b = sub.add_parser('build')
    b.add_argument('--workers', type=int, default=10)
    b.add_argument('--only', default=None)
    b.add_argument('--limit', type=int, default=0, help='first N tasks per repo')
    b.add_argument('--python', default=None)
    sub.add_parser('collect')
    c = sub.add_parser('compare', help='per-task outcome of this version vs another build (default v19)')
    c.add_argument('--base', default='v19')
    a = ap.parse_args()
    if a.version:
        configure(a.version)
    if a.cmd == 'select':
        select(a.n, a.per_repo)
    elif a.cmd == 'build':
        build(a.workers, set(a.only.split(',')) if a.only else None, a.limit, a.python)
    elif a.cmd == 'compare':
        compare(a.base)
    else:
        collect()


if __name__ == '__main__':
    main()
