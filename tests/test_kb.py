"""Unit tests for the code knowledge base (kb.py) and the scripts built on it (locate, callers, tests_for, history).

Each test builds a small git repository with a few commits in a temp dir. Scripts run as subprocesses with
WORKSPACE / REPO_TOOLS_DIR pointing at temp dirs (the way run_skill_script runs them); the kb API is also tested
in-process. Run inside the sandbox image with `scripts/test_skills.sh` (Python 3.13), or with any local pytest.
"""

import collections
import math
import os
import pathlib
import sqlite3
import subprocess
import sys
import textwrap
import time

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'agent/skills/repo-tools/scripts'
sys.dont_write_bytecode = True  # never leave .pyc files in agent/ (the harness rejects them)
sys.path.insert(0, str(SCRIPTS))
import kb  # noqa: E402

INVOICE_V1 = textwrap.dedent('''\
    """Payment helpers."""
    import typing

    from .http import parse_header

    if typing.TYPE_CHECKING:
        def only_for_types(x):
            return x


    class Invoice:
        """An invoice with an amount."""

        currency = "EUR"

        def __init__(self, amount):
            self.amount = amount

        def total(self, tax=0.1):
            """Total with tax."""
            if self.amount == None:
                raise ValueError("amount must not be missing for totals")
            return self.amount * (1 + tax)


    async def fetch_invoice(invoice_id):
        def inner():
            return invoice_id
        return Invoice(inner())


    def header_total(raw):
        return sum(float(v) for v in parse_header(raw))


    if __name__ == "__main__":
        def demo():
            print(Invoice(1).total())
    ''')

HTTP_V1 = 'def parse_header(value):\n    return value.split(",")\n\n\ndef send(req):\n    return req\n'
API = textwrap.dedent('''\
    from . import http
    from .invoice import Invoice, fetch_invoice


    def handle(raw):
        items = http.parse_header(raw)
        return [Invoice(float(i)).total() for i in items]
    ''')
TEST_INVOICE = textwrap.dedent('''\
    from pay.invoice import Invoice


    def test_total():
        assert Invoice(10).total() == 11


    class TestInvoice:
        def test_currency(self):
            assert Invoice(1).currency == "EUR"
    ''')
TEST_HTTP = 'from pay.http import parse_header\n\n\ndef test_parse_header():\n    assert parse_header("a,b") == ["a", "b"]\n'


def git(ws, *args, when=None):
    env = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t',
               GIT_COMMITTER_EMAIL='t@t')
    if when:
        env['GIT_AUTHOR_DATE'] = env['GIT_COMMITTER_DATE'] = f'{when} +0000'
    subprocess.run(['git', '-C', str(ws), *args], check=True, capture_output=True, env=env)


def commit(ws, msg, when):
    git(ws, 'add', '-A')
    git(ws, 'commit', '-qm', msg, when=when)


def write(ws, rel, text):
    p = ws / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    ws = tmp_path / 'ws'
    ws.mkdir()
    git(ws, 'init', '-q')
    write(ws, 'pay/__init__.py', '')
    write(ws, 'pay/invoice.py', INVOICE_V1)
    write(ws, 'pay/http.py', HTTP_V1)
    write(ws, 'tests/test_invoice.py', TEST_INVOICE)
    write(ws, 'README.md', '# pay\n')
    commit(ws, 'Add invoice module', '2024-01-01T10:00:00')
    write(ws, 'pay/http.py', HTTP_V1.replace('value.split(",")', '[v.strip() for v in value.split(",")]'))
    write(ws, 'pay/invoice.py', INVOICE_V1.replace('"EUR"', '"USD"'))
    commit(ws, 'Fix header parsing with spaces in invoice totals', '2024-02-01T10:00:00')
    write(ws, 'pay/api.py', API)
    write(ws, 'tests/test_http.py', TEST_HTTP)
    commit(ws, 'Add request handler API', '2024-03-01T10:00:00')
    write(ws, 'pay/invoice.py', INVOICE_V1)
    commit(ws, 'Handle missing amount in Invoice.total', '2024-04-01T10:00:00')
    state = tmp_path / 'state'
    monkeypatch.setenv('WORKSPACE', str(ws))
    monkeypatch.setenv('REPO_TOOLS_DIR', str(state))
    kb._OPEN.clear()
    yield ws
    for conn in kb._OPEN.values():
        conn.close()
    kb._OPEN.clear()


def run(script, *args, ws):
    env = dict(os.environ, WORKSPACE=str(ws), PYTHONDONTWRITEBYTECODE='1', REPO_TOOLS_FLOW='off')
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True, env=env,
                       timeout=120)
    assert not r.stderr.strip(), r.stderr
    return r.returncode, r.stdout


def meta(conn):
    return dict(conn.execute('SELECT key, value FROM meta'))


def reopen(ws):
    for conn in kb._OPEN.values():
        conn.close()
    kb._OPEN.clear()
    return kb.open(str(ws))


# ---------------- build + schema ----------------

def test_build_creates_kb_in_repo_tools_dir(repo):
    conn = kb.open(str(repo))
    path = pathlib.Path(os.environ['REPO_TOOLS_DIR']) / 'kb.sqlite'
    assert path.exists() and kb.db_path() == str(path)
    m = meta(conn)
    assert m['schema'] == kb.SCHEMA and m['root'] == os.path.abspath(str(repo))
    assert m['head'] == subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True,
                                       text=True).stdout.strip()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert {'files', 'symbols', 'refs', 'imports', 'tests', 'fts', 'fts_files', 'commits', 'commit_files',
            'cochange', 'fts_commits', 'meta'} <= tables
    assert not list(pathlib.Path(os.environ['REPO_TOOLS_DIR']).glob('*.tmp*'))  # temp files cleaned up
    assert not (repo / 'kb.sqlite').exists() and 'kb.sqlite' not in subprocess.run(
        ['git', '-C', str(repo), 'status', '--porcelain'], capture_output=True, text=True).stdout


def test_files_table_langs_and_flags(repo):
    conn = kb.open(str(repo))
    rows = {p: (lang, lines, aux, test) for p, lang, lines, aux, test in
            conn.execute('SELECT path, lang, lines, is_aux, is_test FROM files')}
    assert rows['pay/invoice.py'][0] == 'python' and rows['pay/invoice.py'][1] == INVOICE_V1.count('\n')
    assert rows['README.md'][0] == 'markdown'
    assert rows['tests/test_invoice.py'][2:] == (1, 1)
    assert rows['pay/http.py'][2:] == (0, 0)
    assert conn.execute("SELECT module FROM files WHERE path = 'pay/__init__.py'").fetchone()[0] == 'pay'


def test_symbols_kinds_ranges_signatures(repo):
    conn = kb.open(str(repo))
    syms = {qn: (kind, start, end, sig, doc) for qn, kind, start, end, sig, doc in conn.execute(
        'SELECT qualname, kind, start, "end", signature, doc FROM symbols WHERE path = ?', ('pay/invoice.py',))}
    lines = INVOICE_V1.splitlines()
    assert syms['Invoice'][0] == 'class' and syms['Invoice'][4] == 'An invoice with an amount.'
    kind, start, end, sig, doc = syms['Invoice.total']
    assert kind == 'def' and lines[start - 1].strip() == 'def total(self, tax=0.1):' and sig == lines[start - 1].strip()
    assert lines[end - 1].strip() == 'return self.amount * (1 + tax)' and doc == 'Total with tax.'
    assert syms['fetch_invoice'][0] == 'async def'
    assert 'fetch_invoice.inner' in syms            # nested functions
    assert 'only_for_types' in syms                  # defs inside if/try blocks
    assert 'demo' not in syms                        # demo code under __main__ is skipped
    parent = conn.execute("SELECT parent FROM symbols WHERE qualname = 'Invoice.total'").fetchone()[0]
    assert conn.execute('SELECT qualname FROM symbols WHERE id = ?', (parent,)).fetchone()[0] == 'Invoice'


def test_refs_attribute_calls_and_owner(repo):
    conn = kb.open(str(repo))
    rows = conn.execute("""SELECT r.path, coalesce(s.qualname, ''), r.line, r.call FROM refs r
                           LEFT JOIN symbols s ON s.id = r.src_symbol WHERE r.name = 'parse_header'""").fetchall()
    assert ('pay/api.py', 'handle', 6, 1) in rows                         # attribute call http.parse_header(...)
    assert any(r[0] == 'pay/invoice.py' and r[1] == 'header_total' and r[3] == 1 for r in rows)
    assert any(r[0] == 'tests/test_http.py' and r[1] == 'test_parse_header' for r in rows)
    builtins_refs = conn.execute("SELECT count(*) FROM refs WHERE name IN ('print', 'float', 'self')").fetchone()[0]
    assert builtins_refs == 0


def test_imports_resolve_relative_modules(repo):
    conn = kb.open(str(repo))
    rows = set(conn.execute("SELECT module, name FROM imports WHERE path = 'pay/api.py'"))
    assert rows == {('pay', 'http'), ('pay.invoice', 'Invoice'), ('pay.invoice', 'fetch_invoice')}
    assert ('typing', '') in set(conn.execute("SELECT module, name FROM imports WHERE path = 'pay/invoice.py'"))


def test_tests_table_maps_test_functions_to_source_names(repo):
    conn = kb.open(str(repo))
    rows = set(conn.execute("""SELECT s.qualname, t.test_path, t.target_name FROM tests t
                               JOIN symbols s ON s.id = t.test_symbol"""))
    assert ('test_total', 'tests/test_invoice.py', 'Invoice') in rows
    assert ('test_total', 'tests/test_invoice.py', 'total') in rows
    assert ('TestInvoice.test_currency', 'tests/test_invoice.py', 'Invoice') in rows
    assert ('test_parse_header', 'tests/test_http.py', 'parse_header') in rows


def test_git_history_commits_and_cochange(repo):
    conn = kb.open(str(repo))
    assert conn.execute('SELECT count(*) FROM commits').fetchone()[0] == 4
    subjects = [s for _, s, _ in kb.commits_for('pay/invoice.py', conn=conn)]
    assert subjects[0] == 'Handle missing amount in Invoice.total' and subjects[-1] == 'Add invoice module'
    co = dict(kb.cochanged('/workspace/pay/invoice.py', conn=conn))
    assert co['pay/http.py'] == 2 and co['tests/test_invoice.py'] == 1
    assert dict(kb.cochanged('pay/http.py', conn=conn))['pay/invoice.py'] == 2


def test_similar_commits(repo):
    conn = kb.open(str(repo))
    rows = kb.similar_commits('header parsing breaks with spaces', conn=conn)
    sha, subject, files = rows[0]
    assert subject == 'Fix header parsing with spaces in invoice totals' and len(sha) == 40
    assert set(files) == {'pay/http.py', 'pay/invoice.py'}
    assert kb.similar_commits('!!! ...', conn=conn) == []


def test_find_symbol_forms(repo):
    conn = kb.open(str(repo))
    assert kb.find_symbol('Invoice.total', conn=conn)[0][:3] == ('pay/invoice.py', 'Invoice.total', 'def')
    assert kb.find_symbol('total()', conn=conn)[0][1] == 'Invoice.total'
    assert kb.find_symbol('pay.invoice.Invoice', conn=conn)[0][1] == 'Invoice'
    assert kb.find_symbol('pay/invoice.py:fetch_invoice', conn=conn)[0][2] == 'async def'
    defs = kb.find_symbol('parse_header', conn=conn)
    assert [d[0] for d in defs] == ['pay/http.py']
    assert kb.find_symbol('nope_nothing', conn=conn) == []


def test_callers_api_orders_source_before_tests(repo):
    sites = kb.callers('parse_header', conn=kb.open(str(repo)))
    paths = [s[0] for s in sites]
    assert paths.index('pay/api.py') < paths.index('tests/test_http.py')
    assert all(s[2] > 0 for s in sites)


def test_tests_for_file_and_symbol(repo):
    conn = kb.open(str(repo))
    t = kb.tests_for('pay/invoice.py', conn=conn)
    assert t[0][0] == 'tests/test_invoice.py' and t[0][1] in ('test_total', 'test_currency')
    assert kb.tests_for('Invoice.total', conn=conn)[0] == ('tests/test_invoice.py', 'test_total')
    assert kb.tests_for('pay/http.py', conn=conn)[0] == ('tests/test_http.py', 'test_parse_header')
    assert kb.tests_for('pay/api.py', conn=conn) == [] or all(p.startswith('tests/') for p, _ in
                                                              kb.tests_for('pay/api.py', conn=conn))


def test_search_text(repo):
    rows = kb.search_text('missing amount totals', k=5, conn=kb.open(str(repo)))
    assert rows and rows[0][:2] == ('pay/invoice.py', 'Invoice.total')
    assert kb.search_text('', conn=kb.open(str(repo))) == []


def reference_bm25(docs, qtoks, k1=1.2, b=0.75):
    """The BM25 of the original locate.py, as the reference for kb.bm25."""
    tf = [collections.Counter(d) for d in docs]
    avg = sum(len(d) for d in docs) / len(docs)
    df = collections.Counter()
    for c in tf:
        df.update(c.keys())
    idf = {t: math.log(1 + (len(docs) - v + 0.5) / (v + 0.5)) for t, v in df.items()}
    out = []
    for c, d in zip(tf, docs):
        s = 0.0
        for t, w in collections.Counter(qtoks).items():
            f = c.get(t)
            if f:
                s += idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avg)) * (1 + math.log(w))
        out.append(s)
    return out


def test_bm25_matches_reference_formula(repo):
    conn = kb.open(str(repo))
    q = kb.tokenize('Invoice total amount header parse_header tax tax')
    ids, docs = [], []
    for fid, path in conn.execute('SELECT id, path FROM files WHERE indexed = 1 ORDER BY id'):
        src = (repo / path).read_text()
        ids.append(fid)
        docs.append(kb.tokenize(src) + kb.tokenize(path.replace('/', ' ')) * 3)
    got, _ = kb.bm25(conn, 'files', q)
    for fid, want in zip(ids, reference_bm25(docs, q)):
        assert abs(got.get(fid, 0.0) - want) < 1e-9
    lines = {}
    ids, docs = [], []
    for sid, path, name, start, end in conn.execute(
            "SELECT id, path, name, start, \"end\" FROM symbols WHERE kind != 'class' ORDER BY id"):
        src = lines.setdefault(path, (repo / path).read_text().splitlines())
        ids.append(sid)
        docs.append(kb.tokenize('\n'.join(src[start - 1:end])) + kb.subtoks(name) * 3)
    got, _ = kb.bm25(conn, 'defs', q)
    for sid, want in zip(ids, reference_bm25(docs, q)):
        assert abs(got.get(sid, 0.0) - want) < 1e-9


# ---------------- freshness: reuse, incremental refresh, rebuilds ----------------

def test_reopen_reuses_kb(repo):
    conn = kb.open(str(repo))
    built = meta(conn)['built_at'], os.path.getmtime(kb.db_path())
    time.sleep(0.05)
    conn = reopen(repo)
    assert (meta(conn)['built_at'], os.path.getmtime(kb.db_path())) == built


def test_incremental_refresh_of_edited_new_and_deleted_files(repo):
    conn = kb.open(str(repo))
    built = meta(conn)['built_at']
    time.sleep(0.01)
    write(repo, 'pay/http.py', HTTP_V1 + '\n\ndef parse_cookie(value):\n    return parse_header(value)\n')
    write(repo, 'pay/refunds.py', 'def refund_invoice(inv):\n    return -inv.total()\n')  # untracked
    (repo / 'pay/api.py').unlink()
    conn = reopen(repo)
    assert meta(conn)['built_at'] == built  # refreshed in place, not rebuilt
    assert kb.find_symbol('parse_cookie', conn=conn)[0][0] == 'pay/http.py'
    assert kb.find_symbol('refund_invoice', conn=conn)[0][0] == 'pay/refunds.py'
    assert kb.find_symbol('handle', conn=conn) == []
    assert not conn.execute("SELECT 1 FROM refs WHERE path = 'pay/api.py'").fetchone()
    assert ('pay/http.py', 'parse_cookie', 10, 1) in kb.callers('parse_header', conn=conn)
    assert kb.search_text('refund', conn=conn)[0][1] == 'refund_invoice'
    got, _ = kb.bm25(conn, 'defs', ['cookie'])
    assert len(got) == 1  # the fts rows of the old file version are gone; the new one is indexed once
    ids = [i for (i,) in conn.execute('SELECT id FROM symbols')]
    assert len(ids) == len(set(ids)) == conn.execute('SELECT count(*) FROM fts').fetchone()[0]


def test_new_commit_triggers_rebuild(repo):
    conn = kb.open(str(repo))
    head = meta(conn)['head']
    write(repo, 'pay/tax.py', 'RATE = 0.2\n')
    commit(repo, 'Add tax rate', '2024-05-01T10:00:00')
    conn = reopen(repo)
    assert meta(conn)['head'] != head and conn.execute('SELECT count(*) FROM commits').fetchone()[0] == 5


def test_other_root_triggers_rebuild(repo, tmp_path):
    kb.open(str(repo))
    other = tmp_path / 'other'
    other.mkdir()
    write(other, 'lib/thing.py', 'def only_here():\n    pass\n')
    conn = kb.open(str(other))
    assert meta(conn)['root'] == str(other) and kb.find_symbol('only_here', conn=conn)
    assert kb.find_symbol('Invoice', conn=conn) == []


def test_many_changes_trigger_full_rebuild(repo, monkeypatch):
    conn = kb.open(str(repo))
    built = meta(conn)['built_at']
    monkeypatch.setattr(kb, 'REINDEX_MAX', 1)
    for i in range(6):
        write(repo, f'pay/gen{i}.py', f'def gen{i}():\n    pass\n')
    time.sleep(1.1)  # built_at has 1 s resolution
    conn = reopen(repo)
    assert meta(conn)['built_at'] != built and kb.find_symbol('gen5', conn=conn)


def test_corrupt_kb_is_rebuilt(repo):
    kb.open(str(repo))
    reopen(repo).close()
    kb._OPEN.clear()
    pathlib.Path(kb.db_path()).write_bytes(b'not a database' * 100)
    conn = kb.open(str(repo))
    assert kb.find_symbol('Invoice', conn=conn)


def test_non_git_directory_falls_back_to_walk(tmp_path, monkeypatch):
    ws = tmp_path / 'plain'
    write(ws, 'mod/a.py', 'def alpha():\n    return beta()\n')
    write(ws, 'mod/b.py', 'def beta():\n    return 1\n')
    write(ws, '.hidden/c.py', 'def hidden():\n    pass\n')
    monkeypatch.setenv('REPO_TOOLS_DIR', str(tmp_path / 'st'))
    kb._OPEN.clear()
    conn = kb.open(str(ws))
    assert meta(conn)['head'] == '' and conn.execute('SELECT count(*) FROM commits').fetchone()[0] == 0
    assert kb.find_symbol('beta', conn=conn) and not kb.find_symbol('hidden', conn=conn)
    assert kb.callers('beta', conn=conn)[0][:2] == ('mod/a.py', 'alpha')
    assert kb.similar_commits('alpha', conn=conn) == [] and kb.cochanged('mod/a.py', conn=conn) == []
    conn.close()
    kb._OPEN.clear()


def test_syntax_errors_big_files_and_odd_encodings(repo):
    write(repo, 'pay/broken.py', 'def broken(:\n    unique_broken_token = 1\n')
    write(repo, 'pay/big.py', '# big\n' + 'x = 1\n' * 80_000)
    (repo / 'pay/latin.py').write_bytes(b'def caf\xe9():\n    pass\n\ndef ok_after_bad_bytes():\n    pass\n')
    conn = reopen(repo)
    rows = dict(conn.execute("SELECT path, indexed FROM files WHERE path LIKE 'pay/%'"))
    assert rows['pay/broken.py'] == 1 and rows['pay/big.py'] == 0
    got, _ = kb.bm25(conn, 'files', ['unique_broken_token'])
    assert list(got) == [conn.execute("SELECT id FROM files WHERE path = 'pay/broken.py'").fetchone()[0]]
    assert kb.find_symbol('broken', conn=conn) == []


def test_huge_repo_cap_indexes_source_first_and_stays_stable(repo, monkeypatch):
    monkeypatch.setattr(kb, 'MAX_PY_FILES', 3)
    conn = kb.open(str(repo))
    rows = dict(conn.execute("SELECT path, indexed FROM files WHERE lang = 'python'"))
    assert rows['tests/test_invoice.py'] == 0 and rows['tests/test_http.py'] == 0
    assert len(rows) == 6 and sum(rows.values()) == 3  # 4 source files compete for 3 slots; tests come last
    built = meta(conn)['built_at']
    conn = reopen(repo)
    assert meta(conn)['built_at'] == built  # listed-but-unindexed files do not trigger rebuilds


def test_parallel_parse_matches_sequential(repo, monkeypatch):
    for i in range(12):
        write(repo, f'pkg/m{i}.py', f'def f{i}(x):\n    return g{i}(x)\n\n\nclass C{i}:\n    def m(self):\n'
                                    f'        return f{i}(1)\n')
    rels = sorted(p for p in kb.list_files(str(repo)) if p.endswith('.py'))
    seq = kb._parse_list(str(repo), rels)
    monkeypatch.setattr(kb, 'PARALLEL_MIN_FILES', 1)
    monkeypatch.setattr(os, 'cpu_count', lambda: 2)
    os.makedirs(kb.state_dir(), exist_ok=True)
    par = kb.parse_many(str(repo), rels)
    assert sorted(par, key=lambda r: r['path']) == sorted(seq, key=lambda r: r['path'])
    assert not list(pathlib.Path(kb.state_dir()).glob('parse.*'))


def test_build_time_is_reasonable(tmp_path, monkeypatch):
    ws = tmp_path / 'big'
    body = ''.join(f'def func_{j}(a, b):\n    """Doc {j}."""\n    return helper_{j}(a) + b.attr_{j}\n\n\n'
                   for j in range(20))
    for i in range(300):
        write(ws, f'pkg/sub{i % 10}/mod_{i}.py', f'import os\nfrom . import x\n\n\nclass K{i}:\n    pass\n\n\n'
                                                 + body)
    git(ws, 'init', '-q')
    commit(ws, 'initial', '2024-01-01T00:00:00')
    monkeypatch.setenv('REPO_TOOLS_DIR', str(tmp_path / 'st'))
    kb._OPEN.clear()
    t0 = time.time()
    conn = kb.open(str(ws))
    assert conn.execute('SELECT count(*) FROM symbols').fetchone()[0] == 300 * 21
    assert time.time() - t0 < 15  # ~0.3 s locally; generous for slow CI
    t0 = time.time()
    reopen(ws)
    assert time.time() - t0 < 2
    for c in kb._OPEN.values():
        c.close()
    kb._OPEN.clear()


# ---------------- scripts ----------------

def lines_of(out):
    return out.strip().splitlines()


def test_locate_on_kb_ranks_and_signals(repo):
    code, out = run('locate.py', '--no-show', 'Invoice.total raises ValueError when amount is None', ws=repo)
    assert code == 0
    files = [ln.split()[1] for ln in lines_of(out) if ln[:1].isdigit()]
    assert files[0] == 'pay/invoice.py' and 'Invoice.total' in out
    # v18: the flow facts (identifier count, confidence, thin) go through flow.fact, not a printed signal line
    assert 'signal:' not in out and 'identifiers:' not in out
    assert 'tests: run.py  test=tests/test_invoice.py\n' in out
    assert len(lines_of(out)) <= 40
    assert (pathlib.Path(os.environ['REPO_TOOLS_DIR']) / 'kb.sqlite').exists()


def test_locate_thin_issue_signal(repo):
    # v18: thin/identifier_count reach flow via flow.fact (tests/test_v18_locate.py checks the facts)
    code, out = run('locate.py', '--no-show', 'fix padding', ws=repo)
    assert code == 0 and 'thin=' not in out and 'pay/' in out


def test_locate_history_signal_and_hint(repo):
    code, out = run('locate.py', '--no-show', 'header parsing breaks with spaces', ws=repo)
    assert code == 0 and 'past commit "Fix header parsing with spaces' in out  # v18: on the one meta line
    first = [ln for ln in lines_of(out) if ln[:1].isdigit()][0]
    assert 'pay/http.py' in first or 'pay/invoice.py' in first


def test_locate_error_text_boost(repo):
    code, out = run('locate.py', '--no-show', 'ValueError: amount must not be missing for totals', ws=repo)
    first = [ln for ln in lines_of(out) if ln[:1].isdigit()][0]
    assert 'pay/invoice.py' in first and 'contains error text' in first


def test_locate_k_and_show_flags(repo):
    code, out = run('locate.py', '--k', '2', '--show', 'Invoice total', ws=repo)
    assert code == 0 and len([ln for ln in lines_of(out) if ln[:1].isdigit()]) == 2 and '--- best match' in out
    code, out = run('locate.py', '--k', 'x', 'Invoice', ws=repo)
    assert code == 2 and 'usage' in out


def test_locate_sees_uncommitted_edits(repo):
    run('locate.py', '--no-show', 'Invoice', ws=repo)
    write(repo, 'pay/shipping.py', 'def compute_shipping_cost(weight):\n    return weight * 2\n')
    code, out = run('locate.py', '--no-show', '`compute_shipping_cost` is wrong', ws=repo)
    assert [ln for ln in lines_of(out) if ln[:1].isdigit()][0].split()[1] == 'pay/shipping.py'


def test_callers_script(repo):
    code, out = run('callers.py', 'parse_header', ws=repo)
    assert code == 0 and lines_of(out)[0].startswith('parse_header: 1 definition(s)')
    assert 'def pay/http.py:1-2 parse_header' in out
    assert 'pay/api.py:6 in handle (call)  | items = http.parse_header(raw)' in out
    code, out = run('callers.py', 'parse_header\\n', ws=repo)  # literal escape from the model
    assert code == 0 and 'pay/api.py:6' in out
    code, out = run('callers.py', 'nothing_like_this', ws=repo)
    assert code == 1 and 'no definition or reference' in out
    assert run('callers.py', ws=repo)[0] == 2


def test_callers_output_is_capped(repo):
    write(repo, 'pay/many.py', 'from .http import send\n\n\ndef spam():\n' + ''.join('    send(1)\n' for _ in range(80)))
    code, out = run('callers.py', 'send', ws=repo)
    assert code == 0 and len(lines_of(out)) <= 40 and 'more references' in out


def test_tests_for_script(repo):
    code, out = run('tests_for.py', '/workspace/pay/invoice.py', ws=repo)
    assert code == 0 and 'tests/test_invoice.py' in out
    assert '\n  run.py  test=tests/test_invoice.py\n' in out  # a file target runs whole test files
    assert '{"' not in out and 'args' not in out
    code, out = run('tests_for.py', 'Invoice.total', ws=repo)
    assert '\n  run.py  test=tests/test_invoice.py  k=test_total\n' in out
    assert '{"' not in out
    code, out = run('tests_for.py', 'pay/nothing.py', ws=repo)
    assert code == 1 and 'no tests found' in out
    assert run('tests_for.py', ws=repo)[0] == 2
    assert len(lines_of(out)) <= 40


def test_history_script(repo):
    code, out = run('history.py', 'pay/invoice.py', ws=repo)
    assert code == 0 and 'pay/invoice.py: 3 commit(s)' in out
    assert lines_of(out)[1].split()[2] == 'Handle' and '2024-04-01' in lines_of(out)[1]
    assert 'usually changed together' in out and 'pay/http.py (2)' in out
    code, out = run('history.py', '--similar', 'Header parsing fails when values have spaces', ws=repo)
    assert code == 0 and 'Fix header parsing with spaces' in lines_of(out)[1]
    assert 'files: pay/http.py, pay/invoice.py' in out
    code, out = run('history.py', 'pay/nope.py', ws=repo)
    assert code == 1 and 'not a file' in out
    assert run('history.py', ws=repo)[0] == 2 and run('history.py', '--similar', ws=repo)[0] == 2
    assert len(lines_of(out)) <= 40


def test_kb_script_stats(repo):
    code, out = run('kb.py', '--stats', ws=repo)
    assert code == 0 and 'symbols:' in out and 'commits: 4' in out


def test_scripts_leave_workspace_clean(repo):
    for script, args in (('locate.py', ['Invoice']), ('callers.py', ['Invoice']), ('tests_for.py', ['Invoice']),
                         ('history.py', ['pay/http.py'])):
        run(script, *args, ws=repo)
    assert subprocess.run(['git', '-C', str(repo), 'status', '--porcelain'], capture_output=True,
                          text=True).stdout == ''


def test_kb_file_is_valid_sqlite_with_fts5():
    conn = sqlite3.connect(':memory:')
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize=\"unicode61 tokenchars '_'\")")
    conn.execute("INSERT INTO t VALUES ('parse_header value')")
    assert conn.execute("SELECT count(*) FROM t WHERE t MATCH '\"parse_header\"'").fetchone()[0] == 1
