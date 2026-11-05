"""agent-v18c prompt: agent-v17b's prompt (E045: the halved v18 prompt cost ~6 of 113 tasks at Kaggle speed) plus one
sentence, the MEMO rule (the MEMO line of flow rule 13), right after the paragraph about the NEXT line. v18b's footer
wording ("[call/limit time | edit state]") is not used: v18c's footers are v17b's (BUDGET and NEXT lines)."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT = (ROOT / 'agent/prompts/system.md').read_text()
FLAT = ' '.join(PROMPT.split())
MEMO_RULE = 'A line starting with MEMO: means earlier results may be gone from your context: trust its files, lines and state.'


def test_v17b_rules_are_there():
    for text in ('Once you have read the code that needs to change, the next call is edit.py',
                 'Do not keep reading to be sure: edit, then check',
                 'an imperfect change in the right place beats no change',
                 "Reproduce the issue's own example through the public entry point",
                 'Change only what the issue asks for', 'Never send a rejected call unchanged',
                 'Each result ends with a NEXT: line giving the recommended next call'):
        assert text in FLAT, text


def test_memo_rule_follows_the_next_line_paragraph():
    paras = [' '.join(p.split()) for p in PROMPT.split('\n\n')]
    i = next(i for i, p in enumerate(paras) if p.startswith('The scripts remember what you did. Each result ends with a '
                                                           'NEXT: line'))
    assert paras[i + 1] == MEMO_RULE, paras[i + 1]
    assert FLAT.count('MEMO') == 1


def test_no_v18_footer_wording():
    assert '[call/limit' not in PROMPT and 'edit state]' not in PROMPT


def test_plain():
    assert '\\' not in PROMPT and '「' not in PROMPT
    assert 'name: swe_agent_v21' in (ROOT / 'agent/agent.yaml').read_text()
