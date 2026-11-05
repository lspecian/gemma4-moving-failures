"""Validate and compile a submission directory with the harness's own code, then zip it.

Runs the same checks Kaggle's scorer does before launching the model (directory limits,
single-model rule, YAML compilation), using stub tools so no sandbox or model is needed.

    .venv/bin/python scripts/check_submission.py submissions/v1 [--zip]
"""

import argparse
import pathlib
import sys
import zipfile

from adk_submission import compile_submission, validate_directory
from swegemma.config import build_submission_limits

try:  # swegemma 0.2.7 (local .venv)
    from swegemma.models import setup_gemma_model_registry, validate_single_declared_model
except ImportError:  # swegemma 0.2.11 / adk_submission 0.2.13 (Kaggle's wheelhouse since 2026-10-08; .venv-0211)
    from adk_submission.discovery import discover_declared_models
    from swegemma.cli import _build_cli_model_registry

    setup_gemma_model_registry = None

    def validate_single_declared_model(sub):
        models = sorted({m.strip() for m in discover_declared_models(sub)})
        if len(models) != 1:
            raise SystemExit(f'FAIL: expected exactly one declared model, found {models}')
        return models[0]

# 2026-10-11 (Ryan Holbrook, thread 748287): one variant per submission, any of these
ALLOWED_MODELS = ('gemma-4-31b-it-qat-w4a16-ct', 'gemma-4-12b-it-qat-w4a16-ct', 'gemma-4-e4b-it-qat-w4a16-ct',
                  'gemma-4-e2b-it-qat-w4a16-ct')
HARNESS_TOOLS = [
    'run_command', 'submit_patch', 'get_status', 'read_file', 'edit_file', 'write_file',
    'get_code_neighbors', 'search_similar_code', 'get_code_subgraph',
]


def _stub(name):
    def tool(**kwargs) -> str:
        return '{"status": "ok"}'

    tool.__name__ = name
    tool.__doc__ = f'Stub for {name}.'
    return tool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('submission_dir', type=pathlib.Path)
    parser.add_argument('--zip', action='store_true', help='write <dir>.zip with agent.yaml at the root')
    args = parser.parse_args()
    sub = args.submission_dir.resolve()

    limits, gen_constraints = build_submission_limits()
    validate_directory(sub, limits=limits)
    model = validate_single_declared_model(sub)
    if model not in ALLOWED_MODELS:
        print(f'FAIL: declared model {model!r}, Kaggle allows {ALLOWED_MODELS}')
        return 1

    agent = compile_submission(
        submission_dir=sub,
        tool_registry={name: _stub(name) for name in HARNESS_TOOLS},
        model_registry=setup_gemma_model_registry() if setup_gemma_model_registry else _build_cli_model_registry(sub),
        limits=limits,
        generation_constraints=gen_constraints,
    )
    tools = [getattr(t, 'name', getattr(t, '__name__', str(t))) for t in getattr(agent, 'tools', [])]
    print(f'OK: {agent.name} | model {model} | tools {tools}')
    print(f'    instruction {len(getattr(agent, "instruction", "") or "")} chars')

    if args.zip:
        out = sub.with_suffix('.zip')
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(sub.rglob('*')):
                if p.is_file() and p.name != '.DS_Store':
                    zf.write(p, p.relative_to(sub).as_posix())
        print(f'    wrote {out} ({out.stat().st_size:,} bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
