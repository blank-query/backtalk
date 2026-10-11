# Run: .venv/bin/python tests/check_tap_endpoint.py
# Self-check: a tapped press ends on silence after speech, with the open
# mic's own rule and delay, and a tap nobody speaks into is dropped.
import os, sys, wave
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from backtalk.config import CFG
from backtalk.ears import Endpoint, FRAME_MS, TAP_NO_SPEECH_S

with wave.open(os.path.join(os.path.dirname(__file__), "tap_clip.wav")) as w:
    speech = np.frombuffer(w.readframes(w.getnframes()), np.int16)   # 16 kHz, 4.8 s


def run(pcm, chunk=1600):
    """Feed in 100 ms chunks (like the app); seconds in when it answered."""
    ep = Endpoint()
    for i in range(0, len(pcm), chunk):
        end = ep.feed(pcm[i:i + chunk])
        if end:
            return end, (i + chunk) / 16000
    return None, len(pcm) / 16000

quiet = np.zeros(16000 * 8, np.int16)
end, t = run(quiet)
assert end == "none" and abs(t - TAP_NO_SPEECH_S) < 0.2, (end, t)

wait = int(CFG.get("open_mic_silence_ms") or 480) / 1000
lead = np.zeros(16000, np.int16)
end, t = run(np.concatenate([lead, speech, quiet]))
said = (len(lead) + len(speech)) / 16000
assert end == "end" and said - 1 < t < said + wait + 0.3, (end, t, said, wait)
print(f"tap endpoint ok: no speech dropped at {TAP_NO_SPEECH_S:.0f} s; "
      f"speech ended {t - said:+.2f} s after the clip (silence wait {wait:.1f} s)")

# speech started after a slow start still counts (not dropped at 5 s)
end, t = run(np.concatenate([np.zeros(16000 * 3, np.int16), speech, quiet]))
assert end == "end", end
