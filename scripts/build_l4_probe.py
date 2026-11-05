"""Build a Kaggle notebook that measures the scorer's real conditions on Kaggle's 4×L4 (machine_shape NvidiaL4).

1. Installs the competition wheelhouse and starts vLLM exactly like the hosts' Getting Started notebook (VllmServer,
   TP4, gemma4 parsers; adk_submission adds --reasoning-config).
2. Speed: single-stream chat requests at ~2k/8k/14k/24k prompt tokens (time to first token = prefill, then decode
   tok/s), and one ADK-style compaction summary request (LlmEventSummarizer's prompt, no max_tokens, server-default
   thinking) on ~10k tokens of conversation text.
3. End to end: the agent in <agent_dir> on TASKS with the scorer's EvalConfig (HARNESS_README 7.2: events compaction
   interval 5 / overlap 2 / threshold 14336 / retention 5, context cache 2048/1800/10; subprocess sandbox), traces
   kept, so we see what compaction and L4 speed do to our runs.
Results land in /kaggle/working/probe/ (bench.json, results/...) and probe.zip.

    python3 scripts/build_l4_probe.py results/E041_v17b_eval113/_agent lspecian/gemma4-l4-probe kaggle/gemma4-l4-probe
    kaggle kernels push -p kaggle/gemma4-l4-probe
"""

import argparse
import json
import pathlib

TASKS = ['rich_3518', 'rich_3882', 'rich_2943', 'rich_3894', 'fastapi_14372', 'requests_7309', 'rich_4077',
         'rich_3777']
HOST = pathlib.Path(__file__).resolve().parents[1] / ('research/kaggle-pages/2026-10-01/host_notebook/'
                                                      'getting-started-gemma-4-developer-agent.ipynb')

ap = argparse.ArgumentParser()
ap.add_argument('agent')
ap.add_argument('kernel_id')
ap.add_argument('out')
ap.add_argument('--tasks', default=','.join(TASKS))
ap.add_argument('--adapter-dataset', action='append', default=[],
                help='name=owner/slug: attach that private dataset and copy it into agent/adapters/<name>/ (the agent.yaml'
                     ' in <agent> decides whether the e2e agent uses it; the bench measures every served model)')
ap.add_argument('--no-e2e', action='store_true', help='speed bench only')
ap.add_argument('--model', default='gemma-4-31b-it-qat-w4a16-ct',
                help='served model (Kaggle model google/gemma-4/other/<model>/2), e.g. gemma-4-e4b-it-qat-w4a16-ct')
ap.add_argument('--canary', action='store_true',
                help='LoRA canary only (note 20 C-critique G3): build zero / mild / loud adapters in the notebook '
                     '(scripts/lora_canary.py make), serve them with any --adapter-dataset adapters through the scorer\'s '
                     'VllmServer (TP4), and compare greedy outputs and logprobs with the base model; no bench, no e2e')
a = ap.parse_args()
adapters_ds = dict(x.split('=', 1) for x in a.adapter_dataset)
agent = pathlib.Path(a.agent)
files = {p.relative_to(agent).as_posix(): p.read_text() for p in sorted(agent.rglob('*'))
         if p.is_file() and p.suffix in {'.yaml', '.yml', '.md', '.txt', '.py'}}
host = json.load(open(HOST))
install = ''.join(host['cells'][2]['source'])  # wheelhouse install + env vars (verbatim)

setup = f'''import json, pathlib, time
from swegemma.models import load_tasks
DATA_DIR = pathlib.Path('/kaggle/input/competitions/gemma-4-developer-agent')
WORK = pathlib.Path('/kaggle/working'); PROBE = WORK / 'probe'; PROBE.mkdir(parents=True, exist_ok=True)
AGENT_DIR = WORK / 'agent'
FILES = {json.dumps(files)}
for rel, text in FILES.items():
    p = AGENT_DIR / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)
tasks = load_tasks(DATA_DIR / 'tasks.jsonl')
TASK_IDS = {json.dumps(a.tasks.split(','))}
ADAPTERS = {json.dumps(adapters_ds)}
import shutil
for name, slug in ADAPTERS.items():  # the mount path differs between Kaggle versions: find adapter_config.json
    cands = [c for c in pathlib.Path('/kaggle/input').rglob('adapter_config.json') if slug.split('/')[1] in str(c)]
    assert cands, ('adapter dataset not mounted', slug)
    dst = AGENT_DIR / 'adapters' / name; dst.mkdir(parents=True, exist_ok=True)
    for f in cands[0].parent.iterdir():
        shutil.copy(f, dst / f.name)
    print('adapter', name, '<-', cands[0].parent, sorted(p.name for p in dst.iterdir()))
print(len(FILES), 'agent files;', len(tasks), 'tasks')
'''

server = '''import litellm, torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
try:  # swegemma 0.2.7; 0.2.11 (Kaggle since 10-08) removed it (same fallback as scripts/check_submission.py)
    from swegemma.models.discovery import validate_single_declared_model
except ImportError:
    from adk_submission.discovery import discover_declared_models
    def validate_single_declared_model(sub):
        models = sorted({m.strip() for m in discover_declared_models(sub)})
        assert len(models) == 1, models
        return models[0]
litellm.drop_params = True
TARGET = 'gemma-4-31b-it-qat-w4a16-ct'
MODEL_PATH = pathlib.Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
declared = validate_single_declared_model(AGENT_DIR)
adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
n = torch.cuda.device_count(); tp = 4 if n >= 4 else (2 if n >= 2 else 1)
print('GPUs', n, [torch.cuda.get_device_name(i) for i in range(n)])
cfg = VllmConfig(model=str(MODEL_PATH), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
                 max_model_len=32768, dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'auto',
                 gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=True, max_loras=8,
                 max_lora_rank=128, tensor_parallel_size=tp, startup_timeout=60 * 20)
t0 = time.time()
server = VllmServer(cfg, adapter_manifest=adapters); server.start()
print('server up in', round(time.time() - t0), 's; cmd:', ' '.join(map(str, server.build_cmd())))
models = server.create_model_registry(aliases=[declared, TARGET], model_prefix='openai/', api_key='EMPTY')
'''

bench = '''import urllib.request
URL = 'http://127.0.0.1:8000/v1/chat/completions'
# the scorer's VllmServer registers the model under its path (no --served-model-name): ask the server
SERVED = [m['id'] for m in json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/models'))['data']]
TARGET = SERVED[0]
print('served model ids:', SERVED)
CODE = open(json.__file__).read() + open(json.decoder.__file__).read()
def stream(messages, max_tokens=None, thinking=None):
    body = {'model': TARGET, 'messages': messages, 'stream': True, 'temperature': 0.2,
            'stream_options': {'include_usage': True}}
    if max_tokens: body['max_tokens'] = max_tokens
    if thinking is not None: body['chat_template_kwargs'] = {'enable_thinking': thinking}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    t0 = time.time(); first = None; usage = None; chunks = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]': continue
            d = json.loads(line[6:])
            if d.get('usage'): usage = d['usage']
            if d.get('choices') and first is None:
                delta = d['choices'][0].get('delta') or {}
                if delta.get('content') or delta.get('reasoning') or delta.get('reasoning_content'): first = time.time()
    total = time.time() - t0
    out = (usage or {}).get('completion_tokens') or 0
    ttft = (first or time.time()) - t0
    return {'prompt_tokens': (usage or {}).get('prompt_tokens'), 'completion_tokens': out, 'ttft_s': round(ttft, 2),
            'total_s': round(total, 2), 'decode_tok_s': round(out / max(total - ttft, 1e-6), 2)}
res = {'speed': [], 'summary': None, 'served': SERVED}
for model in SERVED:  # base model, then every adapter (LoRA serving speed on 4xL4)
    TARGET = model
    for chars in (6000, 26000, 46000, 76000):
        text = (CODE * 20)[:chars]
        msgs = [{'role': 'user', 'content': 'Here is some code:\\n' + text + '\\n\\nDescribe in 3 sentences what it does.'}]
        for think in (False, True):
            r = stream(msgs, max_tokens=300, thinking=think); r['thinking'] = think; r['model'] = model
            res['speed'].append(r); print(r)
TARGET = SERVED[0]
conv = '\\n'.join(f'swe_agent: I will read {i}. ' + CODE[i * 300:(i + 1) * 300] for i in range(100))[:34000]
prompt = ('The following is a conversation history between a user and an AI agent. Please summarize the conversation, '
          'focusing on key information and decisions made, as well as any unresolved questions or tasks. The summary '
          'should be concise and capture the essence of the interaction.\\n\\n' + conv)
res['summary'] = stream([{'role': 'user', 'content': prompt}]); print('summary call', res['summary'])
json.dump(res, open(PROBE / 'bench.json', 'w'), indent=1)
'''

e2e = '''import asyncio, concurrent.futures, yaml, shutil
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
def run_sync(fn, *args, **kw):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(fn(*args, **kw))).result()
ev = yaml.safe_load((AGENT_DIR / 'eval_config.yaml').read_text())['evaluation']
limits, gen = build_submission_limits()
cfg = EvalConfig(tasks_path=DATA_DIR / 'tasks.jsonl', snapshots_dir=DATA_DIR / 'snapshots', results_dir=PROBE / 'results',
                 submission_dir=AGENT_DIR, models=models, sandbox='subprocess',
                 timeout_seconds=int(ev['timeout_seconds']), max_time_minutes=float(ev['max_time_minutes']),
                 max_tool_calls=int(ev['max_tool_calls']), max_turns=int(ev['max_turns']), limits=limits,
                 generation_constraints=gen, adapter_manifest=adapters,
                 context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
                 events_compaction_config=EventsCompactionConfig(compaction_interval=5, overlap_size=2,
                                                                 token_threshold=14336, event_retention_size=5),
                 graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'),
                 wheels_dir=DATA_DIR / 'wheels', verbose=False)
evaluator = Evaluator(cfg)
rows = []
for i, t in enumerate([t for t in tasks if t.instance_id in TASK_IDS], 1):
    t0 = time.time()
    r = run_sync(evaluator.evaluate_task, task=t, task_index=i, total_tasks=len(TASK_IDS))
    row = {'id': t.instance_id, 'resolved': r.resolved, 'tool_calls': r.tool_calls, 'duration': round(r.duration_seconds),
           'wall': round(time.time() - t0), 'patch_chars': len(r.agent_patch or ''), 'error': getattr(r, 'error', None)}
    rows.append(row); print(row, flush=True)
    json.dump(rows, open(PROBE / 'e2e.json', 'w'), indent=1)
shutil.make_archive(str(WORK / 'probe'), 'zip', PROBE)
print(sum(r['resolved'] for r in rows), '/', len(rows))
'''

CANARY_SRC = (pathlib.Path(__file__).resolve().parent / 'lora_canary.py').read_text()
canary_make = f'''import subprocess, sys
# synthetic adapters (scripts/lora_canary.py make): zero = exact no-op (lora_B = 0), mild = small effect, loud = breaks the
# model at TP1 (LOG 2026-10-01 on the L40S: base = zero; mild mean logprob -0.029 -> -0.026; loud -> -3.0)
(WORK / 'lora_canary.py').write_text({json.dumps(CANARY_SRC)})
MODEL_CFG = pathlib.Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2') / 'config.json'
for name, scale in (('zero', 0), ('mild', 0.01), ('loud', 0.2)):
    r = subprocess.run([sys.executable, str(WORK / 'lora_canary.py'), 'make', '--config', str(MODEL_CFG),
                        '--out', str(AGENT_DIR / 'adapters' / name), '--scale', str(scale)], capture_output=True, text=True)
    print(name, r.returncode, r.stdout.strip(), r.stderr.strip()[-500:])
    assert r.returncode == 0
'''

canary_check = '''import urllib.request
URL = 'http://127.0.0.1:8000/v1/chat/completions'
IDS = [m['id'] for m in json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/models'))['data']]
NAMES = {p.name for p in (AGENT_DIR / 'adapters').iterdir()}
BASE = next(m for m in IDS if m not in NAMES)
SERVED = [BASE] + [m for m in IDS if m in NAMES]  # base first, then the adapters
print('served model ids:', IDS, 'base:', BASE)
PROMPTS = [
    'Write a Python function that returns the n-th Fibonacci number, then explain its complexity in two sentences.',
    'In one paragraph, explain what a race condition is.',
    'Fix the bug: def mean(xs): return sum(xs) / len(xs) + 1',
    'List three differences between TCP and UDP.',
    'Translate to French: The tests pass on the final working tree.',
]
def call(model, prompt):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 64, 'temperature': 0,
            'seed': 1, 'logprobs': True, 'top_logprobs': 1, 'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=600)); dt = time.time() - t0
    c = d['choices'][0]
    toks = [(x['token'], x['logprob']) for x in (c.get('logprobs') or {}).get('content') or []]
    return {'text': c['message'].get('content') or '', 'toks': toks, 'tok_s': d['usage']['completion_tokens'] / dt}
out = {m: [call(m, p) for p in PROMPTS] for m in SERVED}
out['_base_repeat'] = [call(BASE, p) for p in PROMPTS]
def compare(a, b):
    same_text = a['text'] == b['text']
    n = 0; diffs = []
    for (ta, la), (tb, lb) in zip(a['toks'], b['toks']):
        if ta != tb: break
        n += 1; diffs.append(abs(la - lb))
    return {'same_text': same_text, 'common_prefix_tokens': n, 'max_abs_dlogprob_on_prefix': round(max(diffs or [0]), 6),
            'mean_logprob': round(sum(l for _, l in b['toks']) / max(len(b['toks']), 1), 4)}
verdict = {}
for m in SERVED[1:] + ['_base_repeat']:
    rows = [compare(out[BASE][i], out[m][i]) for i in range(len(PROMPTS))]
    verdict[m] = {'identical_texts': sum(r['same_text'] for r in rows), 'of': len(rows),
                  'max_dlogprob': max(r['max_abs_dlogprob_on_prefix'] for r in rows),
                  'mean_logprob': round(sum(r['mean_logprob'] for r in rows) / len(rows), 4),
                  'tok_s': round(sum(x['tok_s'] for x in out[m]) / len(out[m]), 1), 'rows': rows}
    print(m, {k: v for k, v in verdict[m].items() if k != 'rows'})
print('base mean logprob', round(sum(sum(l for _, l in x['toks']) / max(len(x['toks']), 1) for x in out[BASE]) / len(PROMPTS), 4))
for m in SERVED[1:]:
    print(m, 'first reply:', repr(out[m][0]['text'][:120]))
json.dump({'served': SERVED, 'verdict': verdict, 'outputs': out}, open(PROBE / 'canary.json', 'w'), indent=1)
alive = [m for m in SERVED[1:] if m != 'zero' and verdict[m]['identical_texts'] < verdict[m]['of']]
print('LORA CANARY:', 'ADAPTERS CHANGE OUTPUTS on this stack:' if alive else 'NO adapter changed any output (inert?)', alive)
'''

cells = (install, setup, canary_make, server, canary_check) if a.canary else \
    (install, setup, server, bench) + (() if a.no_e2e else (e2e,))
nb = {'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
      'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': [
          '# L4 probe: LoRA canary (base vs adapters, TP4)' if a.canary else '# L4 probe: speed, compaction, end to end']}]
      + [{'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [c]} for c in cells]}
slug = a.kernel_id.split('/')[1]
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / f'{slug}.ipynb').write_text(json.dumps(nb, indent=1).replace('gemma-4-31b-it-qat-w4a16-ct', a.model))
(out / 'kernel-metadata.json').write_text(json.dumps({
    'id': a.kernel_id, 'title': slug, 'code_file': f'{slug}.ipynb', 'language': 'python', 'kernel_type': 'notebook',
    'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
    'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse'] + sorted(adapters_ds.values()), 'kernel_sources': [],
    'competition_sources': ['gemma-4-developer-agent'],
    'model_sources': [f'google/gemma-4/Other/{a.model}/2'], 'machine_shape': 'NvidiaL4',
    # the hosts' notebook image (Python 3.12): the latest Kaggle image is 3.13 and the cp312 wheelhouse fails
    'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'}, indent=2))
print(f'wrote {out}/ ({len(files)} agent files, tasks {a.tasks})')
