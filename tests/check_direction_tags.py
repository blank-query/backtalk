# Run: .venv/bin/python tests/check_direction_tags.py
# A long <<session>> brief must still parse as one tag; other tags stay bounded.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backtalk.brain import _DIRECTION_TAG as R
brief = "x" * 1100
m = R.findall(f'Over to you. <<session {{"purpose": "cook with me", "brief": "{brief}"}}>> Enjoy.')
assert len(m) == 1 and m[0].startswith("session ") and len(m[0]) > 1100, m
assert R.findall("<<quiet>> hi <<timers []>>") == ["quiet", "timers []"]
assert R.findall("<<" + "y" * 700 + ">>") == []
print("OK")
