"""Alias for an invented script name (v18, research/notes/13 audit #7): the code passed as snippet, code or content
runs like run.py snippet, and the call is recorded and budgeted as run.py. Not listed in SKILL.md.
"""

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run  # noqa: E402

if __name__ == '__main__':
    sys.exit(run.alias_main('reproduce_issue.py'))
