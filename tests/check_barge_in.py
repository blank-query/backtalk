# Run: .venv/bin/python tests/check_barge_in.py
# Self-check: hands-free speech over a reply stops it only for a stop
# command (anyone) or exactly one enrolled voice.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backtalk.main import barges_in

assert barges_in("Jarvis, what about the oil filter?", ["sir"])        # single match
assert not barges_in("Jarvis, what about the oil filter?", ["sir", "maam"])  # two voices
assert not barges_in("what time is it", [])                             # unmatched
for s in ("Jarvis, stop.", "Jarvis stop talking", "Javi, enough!",
          "Jarvis, shut up", "Jarvis, hold on"):
    assert barges_in(s, []) and barges_in(s, ["sir", "maam"]), s       # stop command, anyone
assert not barges_in("stop", [])                                        # needs the name
assert not barges_in("Jarvis, stop listening", [])                      # that's mute, not stop
print("barge-in self-check ok")
