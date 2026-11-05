"""Build a Kaggle notebook that writes a submission directory as /kaggle/working/submission.zip.

Text files are embedded in the notebook. LoRA adapters (binary, ~200 MB) cannot be embedded: upload each one as a
private Kaggle dataset and pass it with --adapter-dataset; the notebook attaches the dataset and copies its files
into adapters/<name>/ of the submission (the adapter files must be at the dataset's top level).

    python3 scripts/build_kaggle_notebook.py submissions/v3 lspecian/gemma4-agent-v3 kaggle/v3
    python3 scripts/build_kaggle_notebook.py results/kaggle_v14rft1 lspecian/gemma4-agent-v14-rft1 \
        kaggle/gemma4-agent-v14-rft1 --adapter-dataset rft1=lspecian/gemma4-lora-rft1
    kaggle kernels push -p kaggle/v3
"""

import argparse
import json
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument('sub')
ap.add_argument('kernel_id')
ap.add_argument('out')
ap.add_argument('--adapter-dataset', action='append', default=[],
                help='name=owner/dataset-slug: copy that dataset into adapters/<name>/ (repeatable)')
a = ap.parse_args()
sub, kernel_id, out = pathlib.Path(a.sub), a.kernel_id, pathlib.Path(a.out)
adapters = dict(x.split('=', 1) for x in a.adapter_dataset)
files = {p.relative_to(sub).as_posix(): p.read_text() for p in sorted(sub.rglob('*'))
         if p.is_file() and p.suffix in {'.yaml', '.yml', '.md', '.txt', '.py'}
         and not p.relative_to(sub).as_posix().startswith('adapters/')}

write_cell = f'''import pathlib, shutil, zipfile

FILES = {json.dumps(files, indent=1)}
ADAPTERS = {json.dumps(adapters)}  # adapter name -> attached dataset

root = pathlib.Path("/kaggle/working/submission")
for rel, text in FILES.items():
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)

for name, ds in ADAPTERS.items():
    slug = ds.split("/")[-1]
    # the mount path differs between Kaggle versions: find the dataset's adapter_config.json
    cands = [c for c in pathlib.Path("/kaggle/input").rglob("adapter_config.json") if slug in str(c)]
    assert len(cands) == 1, (name, ds, cands)
    dst = root / "adapters" / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in cands[0].parent.iterdir():
        if f.suffix in (".json", ".safetensors"):
            shutil.copy(f, dst / f.name)

out = pathlib.Path("/kaggle/working/submission.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            zf.write(p, p.relative_to(root).as_posix())

with zipfile.ZipFile(out) as zf:
    names = zf.namelist()
assert "agent.yaml" in names, names
for name in ADAPTERS:
    assert f"adapters/{{name}}/adapter_model.safetensors" in names, names
print(out, out.stat().st_size, "bytes")
print("\\n".join(sorted(names)))
'''

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
    'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [f'# {sub.name}\n\nWrites `{sub}` as `/kaggle/working/submission.zip`.']},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [write_cell]},
    ],
}
slug = kernel_id.split('/')[1]
out.mkdir(parents=True, exist_ok=True)
(out / f'{slug}.ipynb').write_text(json.dumps(nb, indent=1))
(out / 'kernel-metadata.json').write_text(json.dumps({
    'id': kernel_id,
    'title': slug,
    'code_file': f'{slug}.ipynb',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': True,
    'enable_gpu': True,
    'enable_tpu': False,
    'enable_internet': False,
    'dataset_sources': sorted(set(adapters.values())),
    'kernel_sources': [],
    'competition_sources': ['gemma-4-developer-agent'],
    'model_sources': [],
    'machine_shape': 'NvidiaTeslaT4',
}, indent=2))
print(f'wrote {out}/ with {len(files)} files: {sorted(files)}' + (f'; adapters from datasets: {adapters}' if adapters else ''))
