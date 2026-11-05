"""agent-v17 prompt rules (research/notes/12-failures-eval113.md, levers L3, L4, L7)."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = (ROOT / 'agent/prompts/system.md').read_text()


def test_l3_rejected_call_rule_names_the_shapes_without_writing_brackets():
    for text in ("Skill '...' not found", 'brackets or quote marks', 'written inside args',
                 'Write skill_name as exactly', 'Never send a rejected call unchanged'):
        assert text in PROMPT, text
    # the prompt must not show the bracketed form the model drifts into (it would prime it)
    assert '「' not in PROMPT and '」' not in PROMPT


def test_l4_repro_through_the_public_entry_point():
    for text in ("Reproduce the issue's own example through the public entry point", 'test client',
                 'print repr() of it', 'not by calling the helper you suspect'):
        assert text in PROMPT, text


def test_l7_scope_and_conventions():
    assert 'Change only what the issue asks for' in PROMPT
    assert "Reuse the\n  repo's own helpers, warning classes and patterns" in PROMPT


def test_prompt_stays_small_and_plain():
    assert '\\' not in PROMPT
    assert len(PROMPT) < 7000, len(PROMPT)
    assert 'name: swe_agent_v21' in (ROOT / 'agent/agent.yaml').read_text()
