# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""VOICEPRINTS: who is talking, on a device several people share.

A speaker-embedding model (sherpa-onnx, onnxruntime only, no PyTorch;
NVIDIA's TitaNet small separated real voices best of three tried) turns
an utterance into a fingerprint, compared against each enrolled person's
prints. Prints are kept per device as well as per person, because a
different microphone is what trips speaker ID most: a device with three
or more prints of someone is matched against those, others against all
of that person's prints.

Not security: a recording can fool it. It only decides who the agent
thinks it's talking to.

Off unless voiceprint_model points at the .onnx. Prints live in
<agent_dir>/.backtalk/voiceprints.json: {person: {device_id: [[...], ...]}}.
"""
import json
import os
import threading

import numpy as np

from backtalk.config import CFG
from backtalk.vlog import log

RATE = 16000
MIN_S = 1.0          # shorter clips print too unreliably to judge
KEEP = 30            # newest prints kept per person per device
MARGIN = 0.05        # the best match must beat the runner-up by this

_ex = None
_lock = threading.Lock()
PATH = os.path.join(CFG["agent_dir"], ".backtalk", "voiceprints.json")


def enabled() -> bool:
    return bool(CFG.get("voiceprint_model"))


def embed(pcm: np.ndarray) -> np.ndarray | None:
    """A unit-length print of 16 kHz int16 audio, or None (off, too short)."""
    global _ex
    if not enabled() or pcm is None or len(pcm) < RATE * MIN_S:
        return None
    import sherpa_onnx
    with _lock:
        if _ex is None:
            _ex = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=os.path.expanduser(CFG["voiceprint_model"]), num_threads=2))
            log("[voice] voiceprint model ready")
        s = _ex.create_stream()
        s.accept_waveform(RATE, pcm.astype(np.float32) / 32768.0)
        s.input_finished()
        e = np.array(_ex.compute(s), dtype=np.float32)
    return e / np.linalg.norm(e)


def _load() -> dict:
    try:
        with open(PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def add(person: str, device: str, e: np.ndarray):
    """Keep a print of this person, heard on this device."""
    prints = _load()
    lst = prints.setdefault(person, {}).setdefault(device or "?", [])
    lst.append([round(float(x), 5) for x in e])
    del lst[:-KEEP]
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH + ".tmp", "w") as f:
        json.dump(prints, f)
    os.replace(PATH + ".tmp", PATH)


def _scores(e: np.ndarray, device: str) -> list[tuple[float, str]]:
    """Every enrolled voice's score for this print, best first."""
    scores = []
    for person, by_dev in _load().items():
        here = by_dev.get(device) or []
        pool = here if len(here) >= 3 else [p for lst in by_dev.values() for p in lst]
        if pool:
            c = np.mean(np.array(pool, dtype=np.float32), axis=0)
            scores.append((float(e @ (c / np.linalg.norm(c))), person))
    return sorted(scores, reverse=True)


def identify(e: np.ndarray, device: str) -> tuple[str | None, float]:
    """The enrolled person this print matches, with the score, or
    (None, best score) when nobody clears voiceprint_threshold by MARGIN."""
    scores = _scores(e, device)
    if not scores:
        return None, 0.0
    best, who = scores[0]
    ok = best >= float(CFG.get("voiceprint_threshold", 0.25)) and \
        (len(scores) == 1 or best - scores[1][0] >= MARGIN)
    return (who if ok else None), best


def ignored(e: np.ndarray, device: str) -> tuple[str, float] | None:
    """(voice, score) when this print is best matched by a voice to
    ignore (ignore_voices: another agent's speech reaching a mic), else
    None. No runner-up margin: any closer match to it than to a person
    is enough, since answering another agent is the worse mistake."""
    scores = _scores(e, device)
    if scores and scores[0][1] in IGNORE \
            and scores[0][0] >= float(CFG.get("voiceprint_threshold", 0.25)):
        return scores[0][1], scores[0][0]
    return None


IGNORE = set(CFG.get("ignore_voices") or [])


if __name__ == "__main__":
    # Self-check on synthetic prints (no model needed): the right person
    # wins on a clear match, nobody wins below the threshold or on a tie.
    PATH = "/tmp/voiceprint-selftest.json"
    if os.path.exists(PATH):
        os.remove(PATH)
    rng = np.random.default_rng(0)
    a, b = (v / np.linalg.norm(v) for v in rng.normal(size=(2, 192)))
    for _ in range(3):
        add("sir", "phone", a)
        add("maam", "echo", b)
    assert identify(a, "echo")[0] == "sir"          # falls back to all of sir's prints
    assert identify(b, "echo")[0] == "maam"
    c = rng.normal(size=192); c /= np.linalg.norm(c)
    assert identify(c, "echo")[0] is None           # a stranger
    m = (a + b) / np.linalg.norm(a + b)
    assert identify(m, "echo")[0] is None           # too close to call
    j = rng.normal(size=192); j /= np.linalg.norm(j)
    for _ in range(3):
        add("desktop jarvis", "phone", j)
    assert ignored(j, "phone")[0] == "desktop jarvis"
    assert ignored(a, "phone") is None              # sir is never dropped
    assert ignored(c, "phone") is None              # nor a stranger
    print("voiceprint self-check ok")
