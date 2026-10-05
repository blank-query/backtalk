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
"""The signal bus — tiny files any other program can watch.

The voice line leaves notes; faces read the notes. That one dumb trick
is the whole integration surface:

  .voice_state        idle | listening | thinking | speaking
  .voice_waveform     JSON {ts, samples: [64 floats]} while audio plays
  .voice_loading_pid  exists while the thinking sound is playing
  .voice_rate_limits  JSON {window: {utilization, resets_at}} — only
                      written when show_usage is on
  .voice_mic_mode     ptt | open | paused (paused wins over open, open
                      over ptt, across the local mic and every tab)
  .voice_capturing    1 while an open mic is mid-utterance, else 0

Written to signals_dir (default: the repo root). Visualizers built on
this contract just work.

THE BAREHANDS SEAM: set barehands_state_dir in backtalk.json to a
barehands checkout's state/ folder and the same signals are mirrored in
its format (state/state as a bare word, state/wave.json normalized
0..1) — the on-screen ring becomes your agent's face with zero glue.

Every write is wrapped: the bus must never crash the voice line.
"""
import json
import os
import threading
import subprocess
import sys
import time

import numpy as np

from backtalk.config import CFG

_DIR = CFG["signals_dir"]
_STATE_FILE = os.path.join(_DIR, ".voice_state")
_WAVEFORM_FILE = os.path.join(_DIR, ".voice_waveform")
_LOADING_PID_FILE = os.path.join(_DIR, ".voice_loading_pid")
_DIRECTION_FILE = os.path.join(_DIR, ".voice_direction")
_REPLY_DONE_FILE = os.path.join(_DIR, ".voice_reply_done")
_RATE_LIMIT_FILE = os.path.join(_DIR, ".voice_rate_limits")
_TASKS_FILE = os.path.join(_DIR, ".voice_tasks")
_ACTIVE_CONN_FILE = os.path.join(_DIR, ".voice_active_conn")
_MIC_MODE_FILE = os.path.join(_DIR, ".voice_mic_mode")
_CAPTURING_FILE = os.path.join(_DIR, ".voice_capturing")

_BH = CFG.get("barehands_state_dir") or ""
_BH_STATE = os.path.join(_BH, "state") if _BH else ""
_BH_WAVE = os.path.join(_BH, "wave.json") if _BH else ""

_THINKING_SOUND = CFG.get("thinking_sound") or ""

_WAVEFORM_MIN_INTERVAL = 1.0 / 15   # ~15 writes/sec is plenty for 60fps reads
_last_waveform_write = 0.0
_static_proc: subprocess.Popen | None = None


def set_state(name: str):
    """Write the state. Never raises — the show must go on."""
    try:
        with open(_STATE_FILE, "w") as f:
            f.write(name)
    except OSError:
        pass
    if _BH_STATE:
        try:
            with open(_BH_STATE, "w") as f:
                f.write(name)
        except OSError:
            pass


def set_capturing(on: bool):
    """1 while any open mic (local or a browser's hands-free) is mid-
    utterance, else 0: for a restart to wait on, so it never cuts
    someone off mid-sentence. Never raises."""
    try:
        with open(_CAPTURING_FILE, "w") as f:
            f.write("1" if on else "0")
    except OSError:
        pass


def set_mic_mode(mode: str):
    """ptt | open | paused, for a glance-at-it display. Never raises."""
    try:
        with open(_MIC_MODE_FILE, "w") as f:
            f.write(mode)
    except OSError:
        pass


def set_tasks(n: int):
    """How many background tasks are running right now (the faces draw
    one satellite per task). Temp-file-then-rename, unlike set_state's
    direct write: this one's read on every poll tick by a server that
    might catch it mid-write, and a half-written int fails to parse
    where a half-written state name would just look like a typo.
    Never raises."""
    try:
        tmp = f"{_TASKS_FILE}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(str(n))
        os.replace(tmp, _TASKS_FILE)
    except OSError:
        pass


# The face animates only the tab a turn belongs to. Two facts feed that:
#   - the tab whose QUESTION is in flight (stamped at dispatch, cleared at
#     every turn end, so an unprompted turn means "everyone"), and
#   - the tab whose AUDIO is playing right now (stamped as each chunk
#     starts, cleared when playback goes quiet).
# Audio wins while it plays (replies lag their text by seconds, so a newer
# question can be in flight while an older reply still speaks); otherwise
# the in-flight question decides. Mouth runs in its own thread; a lock
# keeps the pair consistent.
_conn_lock = threading.Lock()
_turn_conn: str = ""
_playing: bool = False
_playing_conn: str = ""


def _write_active_conn():
    with _conn_lock:
        val = _playing_conn if _playing else _turn_conn
    try:
        tmp = f"{_ACTIVE_CONN_FILE}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(val)
        os.replace(tmp, _ACTIVE_CONN_FILE)
    except OSError:
        pass


def set_active_conn(conn_id: str | None):
    """The tab whose question is in flight ("" / None = nobody specific:
    every tab). Stamped by brain at dispatch, cleared at each turn end.
    Never raises."""
    global _turn_conn
    with _conn_lock:
        _turn_conn = str(conn_id) if conn_id is not None else ""
    _write_active_conn()


# After playback stops, the face server still reports "speaking" while the
# last waveform is fresh (ai-visualizer's WAVEFORM_STALE_S, 0.6 s) plus a
# poll tick. Releasing the playing tab at once let that tail show as
# "speaking, for everyone", flashing the Interrupt button on other tabs.
# So the release is held a moment past that window.
_PLAYING_RELEASE_S = 1.0
_release_timer: "threading.Timer | None" = None


def set_playing_conn(conn_id: str | None):
    """Audio for this tab (None = broadcast: every tab) just started
    playing. Wins over the in-flight question until clear_playing_conn()."""
    global _playing, _playing_conn, _release_timer
    with _conn_lock:
        if _release_timer is not None:
            _release_timer.cancel()
            _release_timer = None
        _playing = True
        _playing_conn = str(conn_id) if conn_id is not None else ""
    _write_active_conn()


def clear_playing_conn():
    """Playback went quiet: hand the face back to the in-flight question,
    after _PLAYING_RELEASE_S (see above). New audio starting first cancels
    the release."""
    global _release_timer

    def _release():
        global _playing, _playing_conn, _release_timer
        with _conn_lock:
            _playing = False
            _playing_conn = ""
            _release_timer = None
        _write_active_conn()

    with _conn_lock:
        if _release_timer is not None:
            _release_timer.cancel()
        _release_timer = threading.Timer(_PLAYING_RELEASE_S, _release)
        _release_timer.daemon = True
        _release_timer.start()


def feed_waveform(pcm: np.ndarray):
    """Feed one PCM block (int16) — throttled, downsampled to 64 points.

    Also re-asserts state="speaking" on the same throttle: this only runs
    while the mouth is audibly playing, so the bus self-heals within
    ~70ms if a stray writer stomps the state mid-speech. (That self-heal
    rule once closed a bug that took a whole evening to find.)"""
    global _last_waveform_write
    if pcm.size == 0:
        return
    now = time.time()
    if now - _last_waveform_write < _WAVEFORM_MIN_INTERVAL:
        return
    _last_waveform_write = now
    try:
        idx = np.linspace(0, pcm.size - 1, 64).astype(int)
        raw = pcm[idx].astype(float)
        with open(_WAVEFORM_FILE, "w") as f:
            f.write(json.dumps({"ts": now, "samples": raw.tolist()}))
        if _BH_WAVE:
            norm = np.clip(np.abs(raw) / 32768.0, 0.0, 1.0)
            with open(_BH_WAVE, "w") as f:
                f.write(json.dumps({"ts": now, "samples": norm.tolist()}))
    except (OSError, ValueError):
        pass
    set_state("speaking")


def direction(items):
    """Stage directions the agent wrote into its reply, published at the
    moment the audio carrying them starts playing.

    Your agent can emit `<<anything>>` inline and backtalk will never speak
    it. What the tag MEANS is deliberately not backtalk's business: it
    publishes the raw strings and something else decides. That is the whole
    reason this is a file and not a plugin API.

    The timing is the point, and it is the one part a watcher cannot do for
    itself: these fire when the sentence becomes AUDIBLE, not when the model
    generated it. A screen cue lands on the spoken word instead of seconds
    early. Never raises."""
    if not items:
        return
    try:
        with open(_DIRECTION_FILE, "w") as f:
            f.write(json.dumps({"ts": time.time(), "directions": list(items)}))
    except OSError:
        pass


def reply_done():
    """One reply has finished speaking and its audio has fully drained.

    Distinct from the state going idle, which also happens in the gaps
    BETWEEN sentences of the same reply. Anything waiting for the agent to
    genuinely stop talking wants this rather than a state flicker. Never
    raises."""
    try:
        with open(_REPLY_DONE_FILE, "w") as f:
            f.write(json.dumps({"ts": time.time()}))
    except OSError:
        pass


_rate_limits: dict = {}


def set_rate_limit(window: str, utilization, resets_at):
    """One usage window's reading — how much of the plan is spent.

    Merged rather than replaced, because the reading arrives one window
    at a time and a face wants to draw both at once. `utilization` is a
    0..1 fraction (or None when the window has not reported a number
    yet, which is a real state and not an error); `resets_at` is a unix
    epoch.

    NOTHING CALLS THIS UNLESS show_usage IS ON. That is a privacy
    default, not a performance one: this is the account holder's own
    spend, and it renders on a face that may well be pointed at a
    camera. It never appears without being asked for. (Community fix,
    ai-visualizer issue #1.)

    Never raises."""
    if not window:
        return
    _rate_limits[window] = {"utilization": utilization,
                            "resets_at": resets_at}
    try:
        with open(_RATE_LIMIT_FILE, "w") as f:
            f.write(json.dumps(_rate_limits))
    except OSError:
        pass


def _player_cmd(path: str) -> list[str] | None:
    if sys.platform == "darwin":
        return ["afplay", "-v", "0.35", path]
    for cand in ("ffplay", "aplay", "paplay"):
        from shutil import which
        if which(cand):
            if cand == "ffplay":
                return ["ffplay", "-nodisp", "-autoexit", "-loglevel",
                        "quiet", "-volume", "35", path]
            return [cand, path]
    return None


def static_start():
    """Optional thinking sound — plays while the brain works."""
    global _static_proc
    if not _THINKING_SOUND or not os.path.exists(_THINKING_SOUND):
        return
    static_stop()
    cmd = _player_cmd(_THINKING_SOUND)
    if not cmd:
        return
    try:
        _static_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(_LOADING_PID_FILE, "w") as f:
            f.write(str(_static_proc.pid))
    except OSError:
        _static_proc = None


def static_stop():
    global _static_proc
    if _static_proc is not None:
        try:
            _static_proc.terminate()
        except OSError:
            pass
        _static_proc = None
    try:
        os.remove(_LOADING_PID_FILE)
    except OSError:
        pass
