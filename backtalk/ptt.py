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
"""Hold-to-talk — a global key listener.

HOLD the key -> mic opens. RELEASE -> mic closes and the utterance is
processed. Or TAP it (a press shorter than TAP_S): the mic stays open
until the next press, like the desk unit (see talking()). The button IS the voice-activity detector, which is why this
mode is speaker-safe with no headphones: the mic simply isn't open while
the assistant talks, unless you press the key — and pressing while it
talks interrupts it.

THE KEY-REPEAT TRAP (the bug that kills every naive build): the OS fires
on_press events CONTINUOUSLY while a key is held. Without the held-state
filter below, every repeat reads as a fresh press and keeps cancelling
the reply before it can speak.

AND THE HALF THAT TRAP HIDES: some keyboards send auto-repeat as full
DOWN/UP PAIRS rather than the repeated DOWN-only stream. Filtering the
presses and trusting every release then breaks the OTHER way -- a single
hold is chopped into dozens of ~50ms recordings, each too short to
transcribe, and the whole thing is SILENT. No exception, no log line,
nothing to search for; it simply reads as "the microphone does not work".
Measured in the field on a Logitech MX Mechanical through a Bolt
receiver: one 2.6-second hold produced 186 key events and about fifty
recordings. So a release is never trusted on sight -- see is_held().

macOS needs Input Monitoring permission for the hosting terminal
(System Settings -> Privacy & Security -> Input Monitoring). Windows
works out of the box; some Linux desktops need the user in the `input`
group or an X11 session.
"""
import threading
import time

from pynput import keyboard

TAP_S = 0.35     # a press released sooner than this latches the mic on


def resolve_key(name: str):
    """'home' / 'f13' / 'right_alt' / any single character -> pynput key."""
    name = (name or "home").strip().lower()
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    # Friendly names -> pynput's names. pynput calls the right option key
    # alt_r, not right_alt; the docs speak human, this map translates.
    # (Field-caught: right_alt silently fell back to home, which Mac
    # laptops cannot press, so the voice looked healthy and never fired.)
    aliases = {
        "right_alt": "alt_r", "left_alt": "alt_l",
        "right_option": "alt_r", "left_option": "alt_l",
        "right_ctrl": "ctrl_r", "left_ctrl": "ctrl_l",
        "right_cmd": "cmd_r", "left_cmd": "cmd_l",
        "right_shift": "shift_r", "left_shift": "shift_l",
    }
    name = aliases.get(name, name)
    try:
        return getattr(keyboard.Key, name)
    except AttributeError:
        print(f"[ptt] unknown key {name!r} — falling back to 'home'",
              flush=True)
        return keyboard.Key.home


class PTTListener:
    # How long a release must stand unchallenged before it is believed.
    # Comfortably longer than any keyboard's auto-repeat period (measured
    # at ~50ms on the hardware that exposed this; Windows' fastest setting
    # is ~30ms) and short enough that letting go still feels instant.
    RELEASE_GRACE = 0.12

    def __init__(self, key="home"):
        self._key = resolve_key(key) if isinstance(key, str) else key
        self._held = False
        self._release_t = None          # a release awaiting confirmation
        self._press_t = self._last_release = 0.0
        self._press_evt = threading.Event()
        self._listener = keyboard.Listener(on_press=self._on_press,
                                           on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, k):
        if k != self._key:
            return
        # A press cancels any pending release: that release was auto-repeat,
        # not a human letting go.
        self._release_t = None
        if not self._held:                      # filter key-repeat
            self._held = True
            self._press_t = time.monotonic()
            self._press_evt.set()

    def _on_release(self, k):
        if k == self._key:
            # PROVISIONAL. Believed only if no press follows; see _settle().
            self._release_t = time.monotonic()

    def _settle(self):
        """Commit a release that has stood unchallenged for the grace window."""
        r = self._release_t
        if self._held and r is not None and \
                time.monotonic() - r >= self.RELEASE_GRACE:
            self._held = False
            self._last_release = r
            self._release_t = None

    def wait_press(self):
        """Block until the key goes DOWN (one event per physical press)."""
        # Settled on a loop, not once. A release landing after the last
        # is_held() poll leaves _held provisionally True, and a single
        # settle-then-wait would then block forever: the next press is
        # filtered as key-repeat, so nothing ever sets the event again.
        while True:
            self._settle()
            if self._press_evt.wait(timeout=self.RELEASE_GRACE):
                self._press_evt.clear()
                return

    def is_held(self) -> bool:
        self._settle()
        return self._held

    def talking(self, latch=True, on_latch=None):
        """The record-while predicate for one press (call right after
        wait_press): True while the key is held, as before. A press
        released within TAP_S instead latches it on until the next
        press, which ends it (tap to start, tap again to send) and is
        consumed so it doesn't start a recording of its own. The
        caller's max_s still caps a forgotten latch. on_latch() runs
        once, when a tap latches (the start beep)."""
        latched = False

        def on():
            nonlocal latched
            if latched:
                if self._press_evt.is_set():
                    self._press_evt.clear()
                    return False
                return True
            if self.is_held():
                return True
            latched = latch and self._last_release - self._press_t < TAP_S
            if latched and on_latch:
                on_latch()
            return latched
        return on


if __name__ == "__main__":
    # Self-check of the tap/hold logic (no keyboard needed): fake key
    # events with a zero release grace.
    p = object.__new__(PTTListener)
    p._key, p._held, p._release_t = "k", False, None
    p._press_t = p._last_release = 0.0
    p._press_evt = threading.Event()
    p.RELEASE_GRACE = 0
    p._on_press("k"); p.wait_press(); on = p.talking()
    assert on()                              # held
    time.sleep(TAP_S + 0.05); p._on_release("k")
    assert not on()                          # a long hold ends on release
    p._on_press("k"); p.wait_press(); on = p.talking()
    p._on_release("k")
    hits = []
    on = p.talking(on_latch=lambda: hits.append(1))
    assert on() and on() and hits == [1]     # a tap latches, beeps once
    p._on_press("k")
    assert not on()                          # the next press sends
    assert not p._press_evt.is_set()         # ...and is consumed
    p._on_release("k"); p._settle()
    p._on_press("k"); p.wait_press(); on = p.talking(latch=False)
    p._on_release("k")
    assert not on()                          # latch=False: a tap is a tap
    print("[ptt] tap/hold check passed")
