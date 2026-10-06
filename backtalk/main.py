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
"""backtalk — talk to your Claude Code agent out loud.

Flow: hold the key and speak -> local transcription -> your agent's warm
Claude session streams the reply -> sentences go to the mouth the moment
they complete (~1-2s to first audio on warm turns). The greeting plays
over a hidden warmup query so the first real turn is already hot.

Typing in this terminal is a first-class turn too: same conversation,
spoken reply, and typing while it talks interrupts it.

THE VOICE CONSOLE: exact phrases, spoken (or typed) alone and led by
the agent's name ("Jarvis, clear the session"), control the session itself so you never go back to the keyboard: "clear the
session" / "compact the session" / "switch to the deep model" / "back
to the fast model" / "set effort to low" (or medium, high, max) /
"usage report" / "go hands free" and "push to talk mode" (the MIC) /
"stop listening" (hands-free pauses until the next talk-key press) /
"stop asking for permission" and "start asking again" (permissions,
called auto-approve, a different axis than the microphone on purpose).
And with permission_mode "ask" (the default), gated tool calls ASK OUT
LOUD and your spoken yes or no decides them; any other answer is
passed back to the agent as the reason.

Flags:
  --open-mic   start in hands-free listening for this session (the
               config key mic_mode makes it the standing default, and
               the voice can switch live either way: "go hands free" /
               "push to talk mode"). Know the tradeoff: room audio (a
               video, music, another voice assistant) can trigger
               replies to speech never meant for the agent. The talk
               key keeps working: it interrupts, and holding it always
               gets you heard.
  --barge-in   with --open-mic: keep listening WHILE speaking.
               HEADPHONES REQUIRED — with open speakers the mic hears
               the reply and the agent interrupts itself.
  --model X    override the model for this session (full id).

Say "goodbye <name>" / "end voice mode" to hang up. Ctrl-C works.
"""
import asyncio
import glob
import json
import os
import queue
import re
import socket
import sys
import threading
from difflib import SequenceMatcher
import time

import numpy as np

from backtalk import signals, voiceprint
from backtalk.brain import WarmBrain
from backtalk.config import CFG
from backtalk.ears import (Ears, Session, explain_audio_failure, record_held,
                           warm as warm_ears)
from backtalk.mouth import DIRECTION_HOOKS, Mouth, synth_stream
from backtalk.ptt import PTTListener
from backtalk.vlog import log
from backtalk.web import BrowserBridge, ListenStream, chime

NAME = CFG["name"]
# When the last spoken (not typed) question came in, any device; see handle().
_LAST_SPOKEN = [0.0]
QUIT_PHRASES = CFG["quit_phrases"]
ENROLL_S = 45      # seconds of speech an <<enroll>> collects before it says done
ANNOUNCE_LULL_S = 1.5   # quiet needed on a device before an announcement plays


def rewind_point(pcm, pos: int, rate: int, back_s: float = 3.0) -> int:
    """Where a paused announcement resumes: about back_s before `pos`, at
    the quietest 30 ms within a second of that mark (a gap between words),
    so it never restarts mid-syllable and the listener gets a lead-in."""
    target = max(0, pos - int(back_s * rate))
    lo, hi, f = max(0, target - rate), min(pos, target + rate), int(rate * 0.03)
    if hi - lo < f:
        return target
    frames = range(lo, hi - f, f // 2)
    return min(frames, key=lambda i: float(np.mean(np.abs(pcm[i:i + f].astype(np.float32)))))
# What a dedicated session is told on top of the usual (see _session).
_SESSION_PROMPT = (
    "\n\nTHIS IS A DEDICATED SESSION, not the main one. Its purpose: {purpose}. "
    "It owns one device, {device}: everything said there comes to you, and your "
    "replies and your background tasks' reports go there only. The main session "
    "carries on separately and handles everything else. Your first message is its "
    "brief. When the purpose is done (the person says so), end that reply with "
    "<<session end>> to hand the device back.")

# ---- THE SPOKEN PERMISSION GATE (permission_mode "ask", the default).
# When the agent wants a gated tool, the SDK routes the decision here:
# the ask is spoken, the turn pauses (the SDK waits indefinitely; the
# timeout below is ours), and the NEXT utterance or typed line is the
# answer. "yes" approves; anything else denies, with the user's own
# words passed back as the reason. Silence means no.
PERM_TIMEOUT_S = 75
_PERM = {"fut": None, "asked_at": 0.0,   # pending ask + when it was posed
         "hinted": False}                # escape-hatch hint said yet?
_CONFIRM = {"verb": None, "at": 0.0}     # pending "say confirm" + when
_INTERRUPT_ANSWER = "\x00interrupt"      # sentinel: turn is being killed
# Live AUTO-APPROVE is OUR flag, not an SDK mode flip: the CLI refuses
# a live switch INTO bypassPermissions unless it was launched with the
# danger flag, so instead the gate below auto-approves silently while
# this is on. Same behavior, no reconnect, conversation intact. A
# session that BOOTS in bypassPermissions never consults the gate at
# all; saying "start asking again" flips the SDK side live (that
# direction is allowed) and turns this off. ONLY the explicit
# bypassPermissions value arms this: any other mode (acceptEdits, plan)
# passes through to the SDK and keeps the spoken gate for whatever the
# SDK routes here. (Auto-approve is about PERMISSIONS; hands-free
# LISTENING is about the microphone: see _MIC below. Two different
# axes, deliberately never sharing a name.)
_AUTOAPPROVE = {"on": False}
# The microphone mode, switchable live by voice. "ptt" = mic closed
# except while the key is held. "open" = hands-free listening (VAD).
# The key keeps working in open mode: it interrupts, and holding it
# always gets you heard. gen bumps on every switch so an in-flight
# open-mic capture from before the switch gets discarded, never
# processed.
_MIC = {"mode": "ptt", "gen": 0, "btn": False, "muted": False}


def _unmute():
    """After "stop listening", the next talk-key press (held or tapped)
    brings hands-free back once its own capture is done."""
    if _MIC["muted"]:
        _MIC.update(muted=False, mode="open", gen=_MIC["gen"] + 1)
        log("[console] talk key: hands-free listening back on")

# Approvals are EXACT matches after normalization, never prefixes:
# "yesterday", "yes or no", and "yes, but do not overwrite" must all
# fail. Anything that is not an exact yes DENIES, with the words passed
# back to the agent as the reason. Deny is always the default.
# Exact matches only, and the reason is in the comment on _norm_speech:
# prefix matching turns "yesterday" and "yes or no" into consent. So the
# set has to actually CONTAIN what people say -- and the phrase somebody
# reaches for is the one the prompt just put in their head. Asking for
# PERMISSION and then denying "permission granted" is the system tripping
# a user with its own vocabulary, and it quotes their words back as the
# reason for the refusal.
_YES = {"yes", "yeah", "yep", "yup", "sure", "approve", "approved",
        "go ahead", "do it", "yes please", "yes sir", "yes boss",
        "yes go ahead", "go for it", "green light", "okay", "ok", "y",
        "permission granted", "granted", "you have permission",
        "you may", "allowed", "allow it", "confirmed", "affirmative"}
_CHAIN_MARKS = ("&&", "||", ";", "|", "$(", "`", "\n")


def _norm_speech(text):
    """Lowercase, every non-letter to space, collapse. Whisper loves
    interior commas ("yes, confirm"); end-stripping alone misses them."""
    out = []
    for ch in text.lower():
        out.append(ch if "a" <= ch <= "z" else " ")
    return " ".join("".join(out).split())


def _deny_pending(reason=_INTERRUPT_ANSWER):
    """Resolve a pending spoken ask as a deny. Called whenever the turn
    that posed it is being interrupted, so the ask can never outlive its
    turn and hijack a later utterance (or stall the pipe drain)."""
    f = _PERM["fut"]
    if f is not None and not f.done():
        f.set_result(reason)


def _human_what(tool, tool_input, ctx):
    """The SHORT spoken form, built for a person who has never seen a
    terminal: plain words, no paths, no syntax. Built by code, never by
    the model, so it cannot understate; and every ask offers "details",
    which reads the full literal form below. (Field case: the gate read
    whole file paths and command syntax at a brand-new user.)"""
    d = tool_input or {}
    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = str(d.get("file_path") or d.get("notebook_path")
                   or "a file").replace("\\", "/")
        name = path.rsplit("/", 1)[-1]
        import os as _os
        homes = [CFG.get("agent_dir", "")] + list(CFG.get("extra_dirs")
                                                  or [])
        in_vault = any(h and path.startswith(str(h).rstrip("/") + "/")
                       for h in (CFG.get("extra_dirs") or []))
        verb = "edit" if "Edit" in tool else "create or change"
        if in_vault and name.endswith(".md"):
            return f"{verb} a note in your vault called {name[:-3]}"
        return f"{verb} a file called {name}"
    if tool == "Bash":
        cmd = " ".join(str(d.get("command", "")).split())
        first = (cmd.split() or ["a"])[0].rsplit("/", 1)[-1]
        chained = any(m in cmd for m in _CHAIN_MARKS)
        return (f"run a {first} command in the terminal"
                + (", with several chained parts" if chained else ""))
    if tool == "WebFetch":
        url = str(d.get("url", ""))
        host = url.split("//", 1)[-1].split("/", 1)[0] or "a site"
        return f"read a web page at {host}"
    name = getattr(ctx, "display_name", None) or tool
    return f"use the {name} tool"


_DETAILS = {"details", "the details", "give me details",
            "give me the details", "what command", "what is it",
            "say more", "more", "what exactly", "the exact command"}


def _full_detail(tool, tool_input, ctx):
    """The full literal form, spoken only when the person asks for
    "details". Never lets a long command hide its tail: truncation is
    DISCLOSED and shell chaining is called out (the agent composes
    tool_input itself, so this line must not be steerable into
    understatement)."""
    d = tool_input or {}
    if tool == "Bash":
        cmd = " ".join(str(d.get("command", "")).split())
        chained = any(m in cmd for m in _CHAIN_MARKS)
        line = ("a chained command: " if chained else
                "run a command: ") + cmd[:90]
        if len(cmd) > 90:
            line += (f", and {len(cmd) - 90} more characters. "
                     "Check the log before approving")
        return line
    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = str(d.get("file_path") or d.get("notebook_path")
                   or "a file").replace("\\", "/")
        bits = path.rsplit("/", 2)
        name = "/".join(bits[-2:]) if len(bits) >= 2 else path
        return f"{'edit' if 'Edit' in tool else 'write'} the file {name}"
    if tool == "WebFetch":
        return f"fetch a web page: {str(d.get('url', ''))[:70]}"
    desc = (getattr(ctx, "description", None) or "").strip()
    name = getattr(ctx, "display_name", None) or tool
    return f"use {name}" + (f", {desc[:70]}" if desc else "")


def make_permission_gate(mouth, brain):
    from claude_agent_sdk import (PermissionResultAllow,
                                  PermissionResultDeny)

    async def gate(tool, tool_input, ctx):
        if _AUTOAPPROVE["on"]:
            return PermissionResultAllow(behavior="allow")
        what = _human_what(tool, tool_input, ctx)
        detail = _full_detail(tool, tool_input, ctx)
        loop = asyncio.get_running_loop()
        signals.static_stop()
        log(f"[perm]   asking: {what}")
        log(f"[perm]   detail: {detail}")
        if tool == "Bash":   # the FULL command always reaches the log
            log(f"[perm]   full command: {str((tool_input or {}).get('command', ''))[:2000]}")
        ask = f"Permission check. I want to {what}. Yes, no, or details?"
        if not _PERM["hinted"]:
            # the escape hatch announces itself exactly once, at the
            # moment it becomes relevant (a field case: a new user
            # couldn't find the phrase to turn the checks off)
            _PERM["hinted"] = True
            ask += (" And any time you're done with these checks, say "
                    "stop asking for permission.")
        mouth.say(ask, remote_sink=brain.remote_sink)
        answer = None
        try:
            deadline = loop.time() + PERM_TIMEOUT_S
            while answer is None:
                fut = loop.create_future()
                _PERM["fut"] = fut
                _PERM["asked_at"] = time.monotonic()
                while True:
                    try:
                        got = await asyncio.wait_for(
                            asyncio.shield(fut), 1.0)
                        break
                    except asyncio.TimeoutError:
                        if loop.time() >= deadline:
                            fut.cancel()
                            mouth.say("No answer, so I didn't do it.",
                                     remote_sink=brain.remote_sink)
                            log("[perm]   timed out, denied")
                            return PermissionResultDeny(
                                behavior="deny",
                                message="No spoken answer within the "
                                        "timeout; the action was not "
                                        "approved.",
                                interrupt=False)
                        # keep the ring honest while we wait
                        if not mouth.speaking:
                            signals.set_state("listening")
                if (got != _INTERRUPT_ANSWER
                        and _norm_speech(got) in _DETAILS):
                    # read the full literal form, then ask again with a
                    # fresh clock: asking for details is engagement,
                    # not silence
                    log("[perm]   details requested")
                    mouth.say(f"The details: I want to {detail}. "
                              "Yes or no?", remote_sink=brain.remote_sink)
                    deadline = loop.time() + PERM_TIMEOUT_S
                    continue
                answer = got
        finally:
            _PERM["fut"] = None
        if answer == _INTERRUPT_ANSWER:
            log("[perm]   turn interrupted, denied silently")
            return PermissionResultDeny(
                behavior="deny",
                message="Interrupted by the user; the turn is being "
                        "cancelled.",
                interrupt=False)
        approved = _norm_speech(answer) in _YES
        # the model keeps working either way: restore the working state
        signals.set_state("thinking")
        signals.static_start()
        if approved:
            log("[perm]   approved by voice")
            return PermissionResultAllow(behavior="allow")
        log(f"[perm]   denied: {answer!r}")
        return PermissionResultDeny(
            behavior="deny",
            message=f'Denied by voice. The user said: "{answer[:500]}"',
            interrupt=False)
    return gate


# ---- THE VOICE CONSOLE: session verbs, spoken. Exact phrases only,
# spoken alone, so ordinary sentences can never trigger them. (Grown
# from a community member's own build shared in the Discord.)
CONSOLE_VERBS = {
    "clear":     ("clear the session", "clear the context",
                  "clear context", "fresh slate", "slash clear"),
    "compact":   ("compact the session", "compact the context",
                  "compact context", "slash compact"),
    "deep":      ("switch to the deep model", "use the deep model",
                  "slash model deep"),
    "fast":      ("switch to the fast model", "use the fast model",
                  "back to the fast model", "slash model fast"),
    "usage":     ("usage report", "slash usage"),
    "micopen":   ("go hands free", "hands free mode", "start listening",
                  "wake up", "week up", "weak up",  # how whisper hears it
                  "listen up", "resume listening", "i m back",
                  "unmute",
                  "hands free listening", "open mic", "open the mic"),
    "micptt":    ("push to talk", "push to talk mode",
                  "back to push to talk", "back to the button",
                  "go push to talk", "go to push to talk",
                  "pushed to talk", "pushed to talk mode"),  # how it's heard
    "micmute":   ("stop listening", "mute yourself", "mute the mic", "mute",
                  "pause", "paused", "stand by", "standby", "go to sleep"),
    "noask":     ("stop asking for permission",
                  "stop asking permission",
                  "stop asking me for permission",
                  "turn off the permission prompt",
                  "turn off the permission prompts",
                  "turn off the permissions prompt",
                  "turn off the permissions prompts",
                  "turn off permissions", "turn off permission checks",
                  "disable the permission checks",
                  "disable permission checks", "auto approve",
                  "auto approve mode"),
    "ask":       ("start asking again", "ask before acting",
                  "ask for permission again"),
}
_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def console_match(text):
    """A console phrase must open with the agent's name ("Jarvis, stop
    listening"), so everyday speech near an open mic can't flip a
    setting. Close mishearings of the name count ("Javi")."""
    words, name = _norm_speech(text).split(), NAME.lower().split()
    head = " ".join(words[:len(name)])
    if not words or SequenceMatcher(None, head, " ".join(name)).ratio() < 0.75:
        return None
    norm = " ".join(words[len(name):])
    for verb, phrases in CONSOLE_VERBS.items():
        if norm in phrases:
            return verb
    for lvl in _EFFORTS:
        if norm in (f"set effort to {lvl}", f"effort {lvl}",
                    f"slash effort {lvl}"):
            return f"effort:{lvl}"
    return None


def trailing_mic_verb(text):
    """A listening command may also END a longer utterance, as its own
    sentence: "That worked. Jarvis, pause." -> ("micmute", "That
    worked."). Mic verbs only; the heavier ones (clear, compact...)
    still have to be said alone. Whisper's sentence break is what
    keeps "add Jarvis mute and Jarvis unmute" from firing: no full
    stop before the name, no command."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    verb = console_match(parts[-1]) if len(parts) > 1 else None
    if verb in ("micopen", "micptt", "micmute"):
        return verb, " ".join(parts[:-1])
    return None, text


def _write_config_key(key, value):
    """The agent rewrites the config; the person never hand-edits it.
    Returns True on a persisted write. A file that fails to PARSE is
    left untouched (rewriting from {} would wipe every other setting);
    the in-memory CFG updates either way so the session behaves."""
    from backtalk.config import CONFIG_PATH
    CFG[key] = value
    try:
        data = json.loads(CONFIG_PATH.read_text())
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as e:
        log(f"[console] config not writable/parsable, session-only: {e}")
        return False
    data[key] = value
    try:
        CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n")
    except OSError as e:
        log(f"[console] config write failed, session-only: {e}")
        return False
    return True


def _fmt_tokens(n):
    if n >= 1_000_000:
        return f"about {round(n / 1_000_000, 1):g} million tokens"
    if n >= 1000:
        return f"about {round(n / 1000)} thousand tokens"
    return f"{n} tokens"


def _spoken_usage(sess, ctx_usage):
    """A short CFO brief of the session, written for the ear: plain
    numerals only (the TTS reads "40" fine; symbols come out garbled)."""
    turns = sess["turns"]
    parts = [f"{turns} turn{'s' if turns != 1 else ''} this session",
             _fmt_tokens(sess["out_tokens"]) + " spoken out"]
    cents = round(sess["cost"] * 100)
    if cents >= 1:
        parts.append(f"roughly {cents} cents" if cents < 100
                     else f"roughly {round(cents / 100)} dollars")
    try:
        cats = (getattr(ctx_usage, "categories", None)
                or (ctx_usage or {}).get("categories") or [])
        # the breakdown includes "Free space" and the autocompact
        # buffer; only OCCUPIED categories belong in the spoken number
        total = sum(int(c.get("tokens") or 0) for c in cats
                    if isinstance(c, dict)
                    and "free" not in str(c.get("name", "")).lower()
                    and "buffer" not in str(c.get("name", "")).lower())
        if total:
            parts.append(_fmt_tokens(total)
                         + " sitting in the context window")
    except Exception:
        pass
    return ". ".join(parts) + "."

_PASTE_ON = "\x1b[200~"    # bracketed-paste markers (we enable the mode below)
_PASTE_OFF = "\x1b[201~"


def _clean_typed(line: str) -> str:
    """Scrub terminal-copy artifacts: blockquote gutter glyphs and stray
    whitespace (copying from a CLI chat render drags bars along)."""
    line = line.strip()
    while line[:1] in ("▎", "│", ">"):
        line = line[1:].lstrip()
    return line


def _join_paste(body: str) -> str:
    """Pasted blob -> one clean message (gutters scrubbed, lines joined)."""
    parts = [_clean_typed(l) for l in body.split("\n")]
    return " ".join(" ".join(p for p in parts if p).split())


def _typed_reader_pipe(q: "queue.Queue[str]", fd: int):
    """Non-tty stdin (pipes/tests): line assembly with paste markers."""
    import os
    pend = ""
    while True:
        try:
            b = os.read(fd, 65536)
        except OSError:
            return
        if not b:
            return
        pend += b.decode("utf-8", "replace")
        while True:
            if _PASTE_ON in pend:
                if _PASTE_OFF not in pend:
                    break
                head, rest = pend.split(_PASTE_ON, 1)
                body, pend = rest.split(_PASTE_OFF, 1)
                *hlines, hpart = head.split("\n")
                for l in hlines:
                    l = _clean_typed(l)
                    if l:
                        q.put(l)
                text = _join_paste(hpart + body)
                if text:
                    q.put(text)
                continue
            if "\n" in pend:
                line, pend = pend.split("\n", 1)
                line = _clean_typed(line)
                if line:
                    q.put(line)
                continue
            break


def _typed_reader_simple(q: "queue.Queue[str]"):
    """Windows (no termios): plain line input on a thread. Pastes work;
    they just echo normally instead of collapsing to a count."""
    while True:
        try:
            line = _clean_typed(input())
        except (EOFError, OSError):
            return
        if line:
            q.put(line)


def _typed_reader(q: "queue.Queue[str]"):
    """Terminal stdin -> typed messages (daemon thread). Typed lines are
    first-class turns: same pipeline as a spoken utterance, spoken reply.

    On a POSIX tty we OWN the input line (cbreak: no kernel echo, no
    canonical buffering — the little line editor below echoes keys,
    handles backspace, and assembles bracketed pastes invisibly). The
    kernel's canonical mode is unfixable for pastes: it echoes the
    markers as visible junk and holds unfinished marker lines hostage.
    Pastes show as `[pasted N chars]`; Enter sends everything as ONE
    message. Ctrl-C still works (ISIG stays on); termios restored at
    exit."""
    import atexit
    import os
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        _typed_reader_pipe(q, fd)
        return
    try:
        import termios
        import tty as _tty
    except ImportError:            # Windows: no termios — simple reader
        _typed_reader_simple(q)
        return
    old = termios.tcgetattr(fd)
    _tty.setcbreak(fd)                      # ECHO+ICANON off, ISIG kept
    sys.stdout.write("\x1b[?2004h")         # bracket pastes, please
    sys.stdout.flush()

    def _restore():
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()
    atexit.register(_restore)

    MARKS = (_PASTE_ON, _PASTE_OFF)

    def _partial_tail(s: str) -> int:
        """Length of a trailing partial paste-marker (hold it for the
        next read)."""
        for m in MARKS:
            for k in range(min(len(s), len(m) - 1), 0, -1):
                if m.startswith(s[-k:]):
                    return k
        return 0

    buf = ""          # the input line being composed
    paste = None      # accumulating paste body, or None
    pend = ""
    while True:
        try:
            b = os.read(fd, 4096)
        except OSError:
            _restore()
            return
        if not b:
            _restore()
            return
        pend += b.decode("utf-8", "replace")
        keep = _partial_tail(pend)
        proc = pend[:len(pend) - keep] if keep else pend
        pend = pend[len(pend) - keep:] if keep else ""
        i = 0
        while i < len(proc):
            if paste is not None:
                j = proc.find(_PASTE_OFF, i)
                if j < 0:
                    paste += proc[i:]
                    break
                paste += proc[i:j]
                i = j + len(_PASTE_OFF)
                text = _join_paste(paste)
                paste = None
                if text:
                    if buf and not buf.endswith(" "):
                        buf += " "
                    buf += text
                    sys.stdout.write(text if len(text) <= 60
                                     else f"[pasted {len(text)} chars]")
                    sys.stdout.flush()
                continue
            if proc.startswith(_PASTE_ON, i):
                paste = ""
                i += len(_PASTE_ON)
                continue
            ch = proc[i]
            i += 1
            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                line = buf.strip()
                buf = ""
                if line:
                    q.put(line)
            elif ch in ("\x7f", "\x08"):     # backspace
                if buf:
                    buf = buf[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif ch >= " " or ch == "\t":    # printable: echo + collect
                buf += ch
                sys.stdout.write(ch)
                sys.stdout.flush()


async def amain():
    open_mic = "--open-mic" in sys.argv
    barge_in = "--barge-in" in sys.argv
    model = None
    if "--model" in sys.argv:
        try:
            model = sys.argv[sys.argv.index("--model") + 1]
        except IndexError:
            pass

    CFG_BOOT_MODE = CFG["permission_mode"]
    _AUTOAPPROVE["on"] = CFG_BOOT_MODE == "bypassPermissions"
    _MIC["active"] = time.monotonic()    # the hands-free idle clock
    _MIC["mode"] = "open" if (open_mic
                              or CFG.get("mic_mode") == "open") else "ptt"
    # resume_last_session: reattach to the saved conversation, if any
    resume_id = None
    if CFG.get("resume_last_session"):
        try:
            from backtalk.brain import SESSION_FILE
            with open(SESSION_FILE) as f:
                resume_id = f.read().strip() or None
        except OSError:
            resume_id = None

    mouth = Mouth()
    ears = Ears(aggressiveness=int(CFG.get("open_mic_vad_level", 2)),
                silence_ms=int(CFG.get("open_mic_silence_ms") or 480))
    # can_use_tool needs brain.remote_sink, but brain needs can_use_tool
    # at construction — broken by constructing without it, then setting
    # it once the gate has a real brain to close over.
    brain = WarmBrain(model=model, resume_id=resume_id, mouth=mouth)
    brain._can_use_tool = make_permission_gate(mouth, brain)

    mode = ("hands-free listening (the talk key still works)"
            if _MIC["mode"] == "open"
            else f"push-to-talk ({CFG['ptt_key']})")
    log(f"[backtalk] up — agent={NAME} dir={CFG['agent_dir']} "
        f"model={brain.model} mic={mode} "
        f"(say 'goodbye {NAME.lower()}' to hang up)")
    mouth.say(CFG["greeting"])

    loop = asyncio.get_event_loop()
    # Warm the engines while the greeting plays: the STT model load and
    # the brain's prompt-cache toll both hide behind the spoken line.
    loop.run_in_executor(None, warm_ears)
    # THE BRAIN CONNECT, guarded. This is the one startup step that
    # needs a signed-in Claude Code, internet, and available usage.
    # When it fails or hangs, the mouth still works, so SAY SO instead
    # of dying silently with the face stuck on idle (a real field
    # case: the greeting played, then nothing, and on Windows the
    # window closed before anyone could read the error).
    log("[backtalk] connecting the brain...")
    try:
        await asyncio.wait_for(brain.start(), 120)
        await asyncio.wait_for(brain.capture(
            "Warmup ping - reply with the single word: ready"), 180)
    except (Exception, asyncio.TimeoutError) as e:
        kind = ("timed out" if isinstance(e, asyncio.TimeoutError)
                else f"failed: {e!r}"[:220])
        log(f"[backtalk] BRAIN CONNECT {kind}")
        mouth.say("Bad news. The voice and the face are fine, but I "
                  "couldn't reach my brain, the Claude Code session. "
                  "Check this window for the error. The usual causes: "
                  "Claude Code isn't signed in, the internet is down, "
                  "or the plan is out of usage.")
        mouth.wait_done(timeout=30)
        raise SystemExit(1)
    log("[backtalk] brain warm")
    # the hidden warmup ping is plumbing, not conversation
    brain.session.update(turns=0, out_tokens=0, in_tokens=0, cost=0.0)
    # a configured effort level applies at launch (saved by the spoken
    # "set effort to X", or written by the person's agent on request)
    boot_effort = str(CFG.get("effort") or "").strip().lower()
    if boot_effort in _EFFORTS:
        await brain.command(f"/effort {boot_effort}")
        log(f"[backtalk] effort set to {boot_effort} (from config)")
    elif boot_effort:
        log(f"[backtalk] ignoring unknown effort {boot_effort!r} in config")

    typed_q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_typed_reader, args=(typed_q,), daemon=True).start()
    typed_fut: asyncio.Future | None = None

    async def run_console(verb, sink=None):
        """One voice-console verb. The current reply was already
        cancelled and awaited by handle(); the pipe gets drained here
        before the command goes out. A verb that blows up must never
        take the whole voice session down with it. `sink` is the
        asking browser's, when a browser asked: the spoken answer goes
        back there, and the mic verbs act on that browser alone."""
        try:
            await _run_console_inner(verb, sink)
        except Exception as e:
            log(f"[console] {verb} failed: {e}")
            mouth.say("That command hit an error. Check the log.",
                      remote_sink=sink)
            signals.set_state("idle")

    async def _run_console_inner(verb, sink):
        _deny_pending()
        conn = getattr(sink, "conn", None)


        def say(text):   # every line below answers whoever asked
            mouth.say(text, remote_sink=sink)
        say_after = None
        if verb == "clear":
            resp = await brain.command("/clear")
            brain.clear_tasks()
            say_after = "Cleared. Fresh slate."
        elif verb == "compact":
            say("Compacting. One moment.")
            resp = await brain.command("/compact")
            say_after = "Compacted. Same conversation, smaller footprint."
        elif verb == "deep":
            say("Switching to the deep model. Heads up, replies "
                f"get slower. Say {NAME}, back to the fast model when "
                "you're done.")
            resp = await brain.command(f"/model {CFG['deep_model']}")
            say_after = "Deep model online, for this session only."
        elif verb == "fast":
            resp = await brain.command(f"/model {CFG['model']}")
            say_after = "Back on the fast model."
        elif verb.startswith("effort:"):
            lvl = verb.split(":", 1)[1]
            resp = await brain.command(f"/effort {lvl}")
            saved = _write_config_key("effort", lvl)
            say_after = (f"Effort set to {lvl}, and saved as your "
                         "default." if saved else
                         f"Effort set to {lvl} for this session. The "
                         "config file couldn't be written, so it won't "
                         "stick past a restart.")
        elif verb == "usage":
            resp = ""
            say(_spoken_usage(brain.session,
                              await brain.context_usage()))
        elif verb in ("micopen", "micptt", "micmute") and conn is not None:
            resp = ""
            if verb == "micopen" and conn.listening and not conn.listen_muted:
                say("Already listening on this device.")
            elif verb == "micopen":
                bridge.set_listening(conn, True)
                say(f"Hands-free on for this device. Say {NAME}, stop "
                    f"listening to pause me, or {NAME}, push to talk "
                    "mode to turn it off.")
            elif not conn.listening:
                say("This device is already on push to talk.")
            elif verb == "micmute" and conn.listen_muted:
                say("Already paused.")
            elif verb == "micmute":
                bridge.set_listening(conn, True, muted=True)
                say(f"Paused. Say {NAME}, start listening, or tap me, "
                    "to bring me back.")
            else:
                bridge.set_listening(conn, False)
                say("Push to talk. Tap and hold me to talk.")
        elif verb == "micopen":
            resp = ""
            if _MIC["mode"] == "open":
                say("Already in hands-free listening.")
            else:
                _MIC["mode"] = "open"
                _MIC["muted"] = False
                _MIC["active"] = time.monotonic()
                _MIC["gen"] += 1
                _write_config_key("mic_mode", "open")
                log("[console] mic_mode -> open (hands-free listening)")
                say("Hands-free listening on. I'm always "
                    "listening now, so anything said in the room "
                    "can reach me. The talk key still works, and "
                    f"holding it always gets you heard. Say {NAME}, "
                    "push to talk mode to bring the button back.")
        elif verb == "micptt":
            resp = ""
            if _MIC["mode"] == "ptt" and not _MIC["muted"]:
                say("Already on push to talk.")
            else:
                _MIC["mode"] = "ptt"
                _MIC["muted"] = False
                _MIC["gen"] += 1
                _write_config_key("mic_mode", "ptt")
                log("[console] mic_mode -> ptt")
                key = str(CFG.get("ptt_key", "home")).replace("_", " ")
                say(f"Push to talk. Hold the {key} key and "
                    "talk; the mic stays closed otherwise.")
        elif verb == "micmute":
            resp = ""
            if _MIC["mode"] != "open":
                say("The open mic is already off; I only hear "
                    "the talk key.")
            else:
                # Not persisted: a restart comes back hands-free.
                _MIC.update(muted=True, mode="ptt", gen=_MIC["gen"] + 1)
                log("[console] open mic muted until the next talk-key press")
                say("Not listening. Press the talk key when you "
                    "want me back.")
        elif verb == "noask":
            resp = ""
            _CONFIRM["verb"] = "noask"
            _CONFIRM["at"] = time.monotonic()
            say("Auto-approve means I act without asking "
                "permission, and it becomes your saved default. "
                "Say confirm to switch.")
        elif verb == "noask:confirmed":
            resp = ""
            saved = _write_config_key("permission_mode",
                                      "bypassPermissions")
            _AUTOAPPROVE["on"] = True
            log("[console] permission_mode -> bypassPermissions"
                + (" (saved)" if saved else " (session only)"))
            say(("Auto-approve on, and saved as your default. "
                 if saved else
                 "Auto-approve on for this session. The config "
                 "file couldn't be written, so it won't stick "
                 "past a restart. ")
                + f"Say {NAME}, start asking again any time to flip it "
                  "back.")
        elif verb == "ask":
            resp = ""
            saved = _write_config_key("permission_mode", "ask")
            _AUTOAPPROVE["on"] = False
            flipped = True
            if CFG_BOOT_MODE == "bypassPermissions":
                # a bypass-booted session never consults the gate, so
                # the SDK itself must flip (the safe direction is
                # allowed live). If that fails, saying "done" would be
                # a lie: the agent would keep acting silently.
                try:
                    await brain.set_permission_mode("ask")
                except Exception as e:
                    flipped = False
                    log(f"[console] live flip to ask FAILED: {e}")
            log("[console] permission_mode -> ask"
                + (" (saved)" if saved else " (session only)"))
            if flipped:
                say("Done. I'll ask out loud before real "
                    "actions"
                    + (", and that's saved as your default."
                       if saved else
                       ". The config file couldn't be written, "
                       "so tell me again after a restart."))
            else:
                say("I saved asking as your default, but this "
                    "session couldn't switch over. Restart the "
                    "voice line to get asking back.")
        else:
            resp = ""
        if say_after:
            # the CLI answers slash commands with its own text
            # (confirmations, API errors); an error outranks our line
            low = (resp or "").lower()
            if resp and ("error" in low or "invalid" in low):
                say(resp[:160])
                log(f"[console] {verb} answered: {resp[:120]}")
            else:
                say(say_after)
        signals.set_state("idle")

    # DEDICATED SESSIONS: a second Claude Code session that owns one
    # device while it runs (cook-with-me in the kitchen), so it's never
    # stuck behind the main session's work. Opt-in and per purpose, NOT
    # per connection (that was tried and reverted, see below): the main
    # session starts one with <<session {"purpose", "brief"}>>, the
    # device's speech goes to it until it ends itself with <<session end>>.
    # Its face signals go to that device's own channel (signals.DeviceBus).
    sessions: dict = {}           # device id -> WarmBrain
    handed_back: dict = {}        # device id -> a note for the main session's next message from it
    hf_locked: dict = {}          # device id -> was it hands-free before a session locked it on

    def brain_for(x):
        """The session that owns this device (a conn, a sink, or an id),
        else the main one."""
        cid = x if isinstance(x, str) else (getattr(x, "conn_id", None)
                                            or getattr(x, "id", None))
        return sessions.get(cid, brain)

    async def handle(text: str, spoke_from: float | None = None,
                     interrupt: bool = True, remote_sink=None,
                     typed: bool = False, who: str | None = None) -> bool:
        """Process one utterance; returns False on quit. spoke_from is
        when the utterance STARTED (the PTT press), so an answer can be
        told apart from speech that began before the ask even existed.
        interrupt=False is a QUEUED browser tap (Part 2): touch nothing
        already playing, just line this one up behind it. `remote_sink`,
        when given (a browser press), tags THIS utterance with the
        asking connection, so its reply routes back there instead of
        broadcasting; see brain.py's ask()."""
        log(f"[you]    {text}")
        if hasattr(remote_sink, "send"):
            remote_sink.send({"type": "line", "who": "you", "text": text})
        # A pending spoken permission ask owns the next utterance IF
        # that utterance started after the ask was posed. Speech that
        # began earlier is the user interrupting the turn, not
        # answering a question they never heard: the ask resolves as a
        # silent deny and the utterance falls through as a normal
        # interrupt. Quit wins either way, but only as an EXACT phrase
        # here ("No! Don't hang up, skip it" must stay a deny reason,
        # not kill the session).
        if _PERM["fut"] is not None and not _PERM["fut"].done():
            started_after = (spoke_from is None
                             or spoke_from >= _PERM["asked_at"])
            if _norm_speech(text) in {_norm_speech(q)
                                      for q in QUIT_PHRASES}:
                _PERM["fut"].set_result("no")
                # falls through to the quit body below
            elif started_after:
                _PERM["fut"].set_result(text)
                return True
            else:
                _deny_pending()
        # A pending auto-approve confirm owns it too, for two minutes;
        # after that it expires and speech flows normally again.
        verb = None
        if _CONFIRM["verb"]:
            pend, _CONFIRM["verb"] = _CONFIRM["verb"], None
            expired = time.monotonic() - _CONFIRM["at"] > 120
            if not expired and _norm_speech(text) in (
                    "confirm", "confirmed", "yes confirm",
                    "yes confirmed"):
                verb = pend + ":confirmed"
            elif not expired and not any(q in text.lower()
                                         for q in QUIT_PHRASES):
                mouth.say("Staying as we are.")
                return True
        if any(q in text.lower() for q in QUIT_PHRASES):
            await brain.interrupt()
            mouth.shut_up()
            if brain.remote_sink is not None:
                brain.remote_sink.stop()
            mouth.say(CFG["signoff"])
            mouth.wait_done(timeout=15)
            return False
        if interrupt:
            # mouth/remote_sink get stopped regardless of whether the
            # SDK turn itself is still active: with real-time pacing,
            # mouth can still be sending audio for many seconds after
            # its ResultMessage already landed (see the Backtalk Turn
            # Stream Redesign note's interrupt-gap fix), so gating this
            # on brain.turn_active left that whole window unstoppable.
            b = brain_for(remote_sink)
            if b.turn_active:
                # The reader (brain.py) owns speaking now — interrupt()
                # is a clean async call, nothing here to cancel-and-await.
                log("[turn] interrupted mid-reply by new input")
                _deny_pending()      # an ask never outlives its turn
                await b.interrupt()
            b.mouth.shut_up()
            if b.remote_sink is not None:
                b.remote_sink.stop()
        verb = verb or console_match(text)
        if verb:
            await run_console(verb, remote_sink)
            return True
        tail_verb, text = trailing_mic_verb(text)
        if tail_verb:
            await run_console(tail_verb, remote_sink)
        # Which device asked, by name, so the agent knows the room; and
        # whether it was typed, with how long since anyone last spoke, so
        # the agent can judge whether to answer out loud (see <<quiet>>).
        where = bridge.name_of(getattr(remote_sink, "conn_id", None)) if bridge else None
        tags = [f"from {where}"] if where else []
        if who:
            tags.append(who)
        if remote_sink is not None and getattr(remote_sink, "conn_id", None) in handed_back \
                and brain_for(remote_sink) is brain:
            tags.append(handed_back.pop(remote_sink.conn_id))
        if typed:
            ago = time.monotonic() - _LAST_SPOKEN[0] if _LAST_SPOKEN[0] else None
            tags.append("typed; " + (f"last spoken exchange {ago / 60:.0f} min ago"
                                     if ago is not None else "nothing spoken this session"))
        else:
            _LAST_SPOKEN[0] = time.monotonic()
        if tags:
            text = f"[{', '.join(tags)}] {text}"
        shared = images.pop(getattr(remote_sink, "conn_id", None), None)
        if shared:
            text = ("[The user shared an image with this message; view it "
                    f"with the Read tool: {', '.join(shared)}] {text}")
            remote_sink.send({"type": "image_used"})
        b = brain_for(remote_sink)     # its own face channel if a session owns the device
        b.bus.set_state("thinking")
        if not typed:          # typing usually means keep it quiet
            b.bus.static_start()
        _deny_pending()
        b.ask(text, remote_sink=remote_sink)
        return True

    try:
        # ONE loop, two mic modes, switchable live (_MIC). The talk key
        # is constructed and honored in BOTH modes: in hands-free
        # listening it is the interrupt and the guaranteed way to be
        # heard over room noise. The open mic joins the wait-set only
        # in "open" mode; a mode switch bumps _MIC["gen"], the abort
        # callable closes the in-flight open mic promptly, and any
        # capture born under an old gen is discarded unprocessed.
        # THE BROWSER BRIDGE: a click/tap on the face in ai-visualizer is
        # another press, feeding this SAME loop and the SAME handle()
        # path as the key — see web.py's docstring for why.
        #
        # ONE SHARED CLAUDE CODE SESSION, every device. A separate
        # WarmBrain per connection was tried and reverted: it meant a
        # brand-new, memory-less session every time a tab reconnected,
        # and it broke the voice-console commands (clear/compact/...),
        # which only ever acted on one fixed brain regardless of which
        # tab said them. Routing per device is handled below instead,
        # by tagging each ask() with the asking connection's own sink
        # (see brain.py's _current_asker), no separate session needed
        # for that. brain.remote_sink stays the broadcast one, for
        # turns nobody specific asked for (background reports).
        bridge: BrowserBridge | None = None

        # Pictures shared from a device (the app's share target), by
        # device id: saved where the agent can read them, and attached to
        # that device's next question.
        images: dict[str, list[str]] = {}

        def _on_image(conn, data: bytes):
            d = os.path.join(CFG["agent_dir"], ".backtalk", "images")
            os.makedirs(d, exist_ok=True)
            for old in sorted(glob.glob(os.path.join(d, "*.jpg")))[:-30]:
                os.remove(old)   # keep the last 30
            p = os.path.join(d, time.strftime("%Y%m%d-%H%M%S-")
                             + f"{time.monotonic_ns() % 10**6}.jpg")
            with open(p, "wb") as f:
                f.write(data)
            images.setdefault(conn.id, []).append(p)
            log(f"[web] image from {str(conn.id)[:8]}: {p} ({len(data) // 1024} KB)")

        def _announce(directions, asker):
            """<<announce {"to": name, "text": ...}>>: a chime, then the
            text in this voice, on that device only. Not delivered (no
            such device, or it isn't connected): the agent is told."""
            for d in directions:
                if d.startswith("call "):
                    _call(d[5:], asker)
                if d.startswith("enroll "):
                    _enroll(d[7:], asker)
                if d.startswith("session "):
                    _session(d[8:].strip(), asker)
                if d.startswith("timers "):
                    _timers(d[7:], asker)
                if d.startswith("show "):
                    _show(d[5:].strip().strip('"'), asker)
                if not d.startswith("announce "):
                    continue
                try:
                    a = json.loads(d[9:])
                    conn = bridge.find(str(a["to"]))
                    if conn is None:
                        names = ", ".join(v.get("name", "?") for v in bridge.devices().values())
                        raise LookupError(f"{a['to']!r} isn't connected (devices: {names})")
                except Exception as e:
                    log(f"[announce] not delivered: {e}")
                    loop.call_soon_threadsafe(
                        brain_for(asker).ask, f"[Announcement not delivered: {e}]", asker)
                    continue
                threading.Thread(target=_play_announcement, daemon=True,
                                 args=(conn, str(a.get("text") or ""))).start()

        def _who(conn, pcm) -> str | None:
            # A voiceprint failure must never cost an utterance: untagged.
            try:
                return _who_(conn, pcm)
            except Exception as e:
                log(f"[voice] voiceprint skipped: {e!r}")
                return None

        def _who_(conn, pcm) -> str | None:
            """Who said this, for the agent's tag: the device's owner on a
            personal device (its prints are kept too, so they're ready for
            shared ones), else the voiceprint, else "voice unknown". A clip
            too short to print goes to whoever spoke last here, within two
            minutes. While enrolling, every print goes to that person, and
            the agent hears when enough speech is in. Runs off the loop."""
            owner = bridge.devices().get(conn.id, {}).get("owner")
            e = voiceprint.embed(pcm)
            enroll = getattr(conn, "enroll", None)
            if enroll and e is not None:
                voiceprint.add(enroll["name"], conn.id, e)
                enroll["s"] += len(pcm) / voiceprint.RATE
                if enroll["s"] >= ENROLL_S:
                    conn.enroll = None
                    log(f"[voice] enrolled {enroll['name']} on {bridge.name_of(conn.id)}")
                    loop.call_soon_threadsafe(
                        brain_for(conn).ask, f"[Voice enrollment done: {enroll['name']}, "
                        f"{enroll['s']:.0f} s of speech on this device]", bridge.make_sink(conn))
                return f"{enroll['name']} (enrolling)"
            if owner:
                # Learn only a voice that already matches the owner (from an
                # enrollment, or earlier matches): phones get handed around,
                # and someone else's voice must never become the owner's.
                if e is not None and voiceprint.identify(e, conn.id)[0] == owner:
                    voiceprint.add(owner, conn.id, e)
                return owner
            if not voiceprint.enabled():
                return None
            last = getattr(conn, "last_who", None)
            if e is None:
                return last[0] if last and time.monotonic() - last[1] < 120 else None
            who, score = voiceprint.identify(e, conn.id)
            log(f"[voice] {who or 'unknown'} ({score:.2f}) on {bridge.name_of(conn.id)}")
            if who:
                conn.last_who = (who, time.monotonic())
            return who or "voice unknown"

        def _enroll(arg: str, asker):
            """<<enroll {"name": ...}>>: this device's next ENROLL_S
            seconds of speech are that person's voiceprints."""
            try:
                name = str(json.loads(arg)["name"]).strip()
                conn = next(c for c in list(bridge._conns) if c.id is not None
                            and c.id == getattr(asker, "conn_id", None) and not c.disconnected)
                if not voiceprint.enabled():
                    raise RuntimeError("voiceprints are off (no voiceprint_model)")
                conn.enroll = {"name": name, "s": 0.0}
                log(f"[voice] enrolling {name} on {bridge.name_of(conn.id)}")
            except Exception as e:
                log(f"[voice] enrollment not started: {e!r}")
                loop.call_soon_threadsafe(brain_for(asker).ask, f"[Enrollment not started: {e!r}]", asker)

        def _show(arg: str, asker):
            """<<show path/to/note.md>>: that note, whole, as a full-screen
            card on the asking device (a recipe, a list); <<show close>>
            closes it. Only files inside the agent's folders."""
            send = getattr(asker, "send", None)
            if send is None:
                return
            if arg == "close":
                send({"type": "show", "close": True})
                return
            try:
                roots = [os.path.realpath(r) for r in [CFG["agent_dir"], *CFG["extra_dirs"]]]
                path = os.path.realpath(arg if os.path.isabs(arg) else os.path.join(CFG["agent_dir"], arg))
                if not any(path == r or path.startswith(r + os.sep) for r in roots):
                    raise PermissionError("outside the agent's folders")
                with open(path, encoding="utf-8") as f:
                    text = f.read(200_000)
                if text.startswith("---"):           # drop the note's frontmatter
                    end = text.find("\n---", 3)
                    text = text[end + 4:].lstrip() if end > 0 else text
                send({"type": "show", "title": os.path.splitext(os.path.basename(path))[0],
                      "markdown": text})
                log(f"[show] {os.path.basename(path)} to {bridge.name_of(asker.conn_id)}")
            except Exception as e:
                log(f"[show] not shown: {e!r}")
                loop.call_soon_threadsafe(brain_for(asker).ask, f"[Not shown: {e!r}]", asker)

        def _timers(arg: str, asker):
            """<<timers [{"label": ..., "at": <epoch s>, "clock": bool?}, ...]>>:
            the countdowns the asking device's face shows ([] clears)."""
            try:
                items = [{"label": str(t["label"])[:40], "at": float(t["at"]),
                          "clock": bool(t.get("clock"))} for t in json.loads(arg)][:8]
                signals.set_device_timers(asker.conn_id, items)
            except Exception as e:
                log(f"[timers] ignored: {e!r}")

        def _session(arg: str, asker):
            """<<session {"purpose", "brief", "hands_free"?}>> from the main
            session: a dedicated session takes over the asking device.
            <<session end>> from that dedicated session: it hands the
            device back. Failures go back to whoever asked."""
            cid = getattr(asker, "conn_id", None)

            async def run():
                try:
                    if arg == "end":
                        b = sessions.pop(cid, None)
                        if b is None:
                            return
                        log(f"[session] ended on {bridge.name_of(cid)}")
                        handed_back[cid] = "the dedicated session for this device has ended; it's yours again"
                        await loop.run_in_executor(None, lambda: b.mouth.wait_done(timeout=30))
                        await b.stop()
                        b.mouth.shutdown()
                        b.bus.close()
                        was = hf_locked.pop(cid, None)
                        conn = next((c for c in list(bridge._conns) if c.id == cid
                                     and not c.disconnected), None)
                        if was is not None and conn is not None and not was:
                            bridge.set_listening(conn, False)
                        return
                    a = json.loads(arg)
                    conn = next((c for c in list(bridge._conns) if c.id == cid
                                 and not c.disconnected), None)
                    if conn is None:
                        raise LookupError("a dedicated session needs a device to run on")
                    if cid in sessions:
                        raise RuntimeError(f"{bridge.name_of(cid)} already has a dedicated session")
                    name = bridge.name_of(cid)
                    bus = signals.DeviceBus(cid)
                    m = Mouth(bus=bus, local=False)
                    b = WarmBrain(model=brain.model, mouth=m, bus=bus, persist=False,
                                  label=f"Jarvis@{name}",
                                  append=_SESSION_PROMPT.format(purpose=a.get("purpose", "?"),
                                                                device=name))
                    b._can_use_tool = make_permission_gate(m, b)
                    b.remote_sink = bridge.make_sink(conn)
                    await b.start()
                    sessions[cid] = b
                    if a.get("hands_free"):
                        # hands busy (cooking): open mic, no idle timeout,
                        # until the session ends and the old mode returns
                        hf_locked[cid] = bool(conn.listening and not conn.listen_muted)
                        bridge.set_listening(conn, True)
                    log(f"[session] {a.get('purpose')!r} started on {name}"
                        + (" (hands-free locked on)" if cid in hf_locked else ""))
                    b.ask(f"[Dedicated session started: {a.get('purpose')}. Brief from the "
                          f"main session: {a.get('brief', '')}]", bridge.make_sink(conn))
                except Exception as e:
                    log(f"[session] not started: {e!r}")
                    brain_for(asker).ask(f"[Dedicated session not started: {e!r}]", asker)
            asyncio.run_coroutine_threadsafe(run(), loop)

        def _call(arg: str, asker):
            """<<call {"to": name}>>: an intercom call from the asking
            device to that one (web.Call). Not placed: the agent is told."""
            def place():
                try:
                    to = str(json.loads(arg)["to"])
                    callee = bridge.find(to)
                    caller = next((c for c in list(bridge._conns) if c.id is not None
                                   and c.id == getattr(asker, "conn_id", None)
                                   and not c.disconnected), None)
                    if caller is None:
                        raise LookupError("a call has to be asked for from a device")
                    if callee is None:
                        names = ", ".join(v.get("name", "?") for v in bridge.devices().values())
                        raise LookupError(f"{to!r} isn't connected (devices: {names})")
                    bridge.start_call(caller, callee)
                except Exception as e:
                    log(f"[call] not placed: {e}")
                    brain_for(asker).ask(f"[Call not placed: {e}]", asker)
            loop.call_soon_threadsafe(place)

        def _busy(cid) -> bool:
            """Is that device's speaker in use by anything but an
            announcement, or about to be (the session that owns it is
            mid-turn; the main session's turns are for every device, so
            they don't count)?"""
            return (time.monotonic() < bridge.voice_until.get(cid, 0) + ANNOUNCE_LULL_S
                    or (cid in sessions and (sessions[cid].turn_active
                                             or sessions[cid].mouth.speaking)))

        def _play_announcement(conn, text: str):
            """An announcement yields: it waits for a lull on that device,
            and if the device gets busy mid-way (a cooking step, a reply)
            it pauses, then resumes about 3 s back, from a quiet gap
            between words. Paced in real time so the pause lands on time."""
            sink = bridge.make_sink(conn, low=True)
            chunks = list(synth_stream(text))
            rate = chunks[0][0]
            pcm = np.concatenate([chime(rate)] + [p for _, p in chunks])
            deadline = time.monotonic() + 15 * 60     # never wait forever
            block, pos = rate // 10, 0
            while pos < len(pcm):
                if _busy(conn.id) and time.monotonic() < deadline:
                    if pos:
                        log(f"[announce] paused on {bridge.name_of(conn.id)}")
                    while _busy(conn.id) and time.monotonic() < deadline:
                        time.sleep(0.2)
                    pos = rewind_point(pcm, pos, rate) if pos else 0
                    continue
                sink(rate, pcm[pos:pos + block])
                time.sleep(block / rate)
                pos += block
            sink.reply_done()
            log(f"[announce] to {bridge.name_of(conn.id)}: {text[:120]}")

        def _on_phone_result(conn, text: str):
            """A phone command's outcome the agent needs (a failure, a
            contact lookup): asked as that device's turn, so the answer
            goes back there."""
            log(f"[phone] result from {str(conn.id)[:8]}: {text[:200]}")
            brain_for(conn).bus.set_state("thinking")
            brain_for(conn).ask(f"[The phone reports back on your last command: {text}]",
                      remote_sink=bridge.make_sink(conn))

        # Browser hands-free: one listener thread per listening tab,
        # the same Ears endpointing and filters as the local open mic,
        # fed from that tab's stream. Utterances land in hf_q.
        hf_q: "asyncio.Queue" = asyncio.Queue()
        hf_fut: asyncio.Future | None = None

        def _capturing(conn, on):
            """An open mic started or stopped capturing an utterance
            (listener thread). The tab hears about it, so its face shows
            the listening rings in hands-free too, and the bus gets one
            flag across every open mic, for restarts to wait on."""
            if conn is None:
                _MIC["capturing"] = on
            else:
                conn.capturing = on
                asyncio.run_coroutine_threadsafe(conn.ws.send(json.dumps(
                    {"type": "capturing", "on": on})), loop)
            conns = list(bridge._conns) if bridge is not None else []
            signals.set_capturing(bool(_MIC.get("capturing"))
                                  or any(getattr(c, "capturing", False) for c in conns))

        def _hf_listen(conn):
            t = getattr(conn, "hf_thread", None)
            if t is not None and t.is_alive():
                return
            stream, hf_ears = ListenStream(conn), Ears(
                aggressiveness=int(CFG.get("open_mic_vad_level", 2)),
                silence_ms=int(CFG.get("open_mic_silence_ms") or 480))
            # Closed only during this tab's own press. The tab itself
            # goes deaf while it plays the reply or the thinking sound
            # (core.js), which is what actually reaches its mic; gating
            # on the turn here also deafened every OTHER device while
            # one was being answered. Speech mid-turn queues (below).
            gate = lambda: conn.recording
            stop = lambda: not conn.listening or conn.disconnected

            def work():
                # ONE listener per tab for its whole connection, idling
                # while hands-free is off. One that exited on "off" could
                # still be shutting down when an "on" came straight after
                # (a triple click, 2026-10-04), which then saw it alive,
                # started nothing, and left the tab unheard.
                while not conn.disconnected:
                    if not conn.listening:
                        time.sleep(0.1)
                        continue
                    try:
                        text = hf_ears.listen_once(stream=stream, gate=gate,
                                                   busy=lambda b: _capturing(conn, b),
                                                   abort=stop)
                    except Exception as e:
                        log(f"[web] hands-free listener failed: {e!r}")
                        time.sleep(1)
                        continue
                    if text and not stop():
                        who = _who(conn, hf_ears.last_pcm)
                        loop.call_soon_threadsafe(hf_q.put_nowait,
                                                  (conn, text, False, who))
            conn.hf_thread = threading.Thread(target=work, daemon=True)
            conn.hf_thread.start()

        def _hands_free_timeout(conns):
            """Hands-free goes back to push-to-talk after
            hands_free_timeout_s with no conversation, so it's never
            left listening forever. A reply in progress counts as
            conversation."""
            limit = float(CFG.get("hands_free_timeout_s") or 0)
            if not limit:
                return
            now = time.monotonic()
            if any(b.turn_active or b.mouth.speaking
                   for b in [brain, *sessions.values()]):
                _MIC["active"] = now
                for c in conns:
                    c.active = now
                return
            # Mid-capture, wait: it may be you talking through noise. Only
            # transcribed words reset the clock, so pure noise (which ends
            # as an empty capture, 30 s at most) can't keep it on forever.
            for c in conns:
                if c.listening and not getattr(c, "capturing", False) \
                        and c.id not in hf_locked \
                        and now - getattr(c, "active", now) > limit:
                    bridge.set_listening(c, False)
                    log(f"[web] hands-free timed out for {str(c.id)[:8]}")
                    mouth.say("Hands-free off.", remote_sink=bridge.make_sink(c))
            if _MIC["mode"] == "open" and not _MIC.get("capturing") \
                    and now - _MIC["active"] > limit:
                _MIC.update(mode="ptt", muted=False, gen=_MIC["gen"] + 1)
                _write_config_key("mic_mode", "ptt")
                log("[console] hands-free timed out -> ptt")
                mouth.say("Hands-free off.")

        async def _publish_mic_mode():
            """One word for a glance-at-it display (signals.set_mic_mode):
            paused beats hands-free beats push-to-talk, across the local
            mic and every connected tab. Polled: modes change rarely and
            from several places."""
            last = None
            while True:
                conns = list(bridge._conns) if bridge is not None else []
                _hands_free_timeout(conns)
                mode = ("paused" if _MIC["muted"] or any(c.listen_muted for c in conns)
                        else "open" if _MIC["mode"] == "open" or any(c.listening for c in conns)
                        else "ptt")
                if mode != last:
                    signals.set_mic_mode(mode)
                    last = mode
                await asyncio.sleep(0.5)

        asyncio.create_task(_publish_mic_mode())
        if CFG.get("web", {}).get("enabled"):
            bridge = BrowserBridge(CFG["web"])
            bridge.on_listen = _hf_listen
            bridge.on_image = _on_image
            bridge.on_phone_result = _on_phone_result
            bridge.on_text = lambda conn, t: hf_q.put_nowait(
                (conn, t, True, bridge.devices().get(conn.id, {}).get("owner")))
            bridge.devices_file = os.path.join(CFG["agent_dir"], ".backtalk", "devices.json")
            bridge.update_dir = os.path.join(CFG["agent_dir"], ".backtalk", "app")
            os.makedirs(os.path.dirname(bridge.devices_file), exist_ok=True)
            DIRECTION_HOOKS.append(_announce)
            bridge.owned = sessions
            brain.remote_sink = bridge.make_broadcast_sink()
            asyncio.create_task(bridge.serve())
        ptt = PTTListener(CFG["ptt_key"])
        press_fut: asyncio.Future | None = None
        mic_fut: asyncio.Future | None = None
        ws_press_fut: asyncio.Future | None = None
        mic_gen_seen = _MIC["gen"]

        async def _begin_capture(interrupt: bool = True):
            """Shared by the local key and a browser press. The local
            key always interrupts (dev tool, unchanged). A browser
            press interrupts only for the Interrupt button; a plain
            tap passes interrupt=False and this becomes a no-op on
            anything already playing — the recording still happens,
            it just queues instead (Part 2). Either way: duck, and
            mark the mic busy so the open mic (if any) yields."""
            if interrupt:
                perm_wait = (_PERM["fut"] is not None
                             and not _PERM["fut"].done())
                if brain.turn_active and not perm_wait:
                    log("[turn] interrupted mid-reply — new capture started")
                    await brain.interrupt()
                # Unconditional, same reasoning as handle(): mouth can
                # still be sending audio long after turn_active goes
                # False (real-time pacing), so stopping it can't be
                # gated on the SDK turn still being live.
                mouth.shut_up()
                signals.static_stop()
                if brain.remote_sink is not None:
                    brain.remote_sink.stop()
            signals.set_state("listening")
            mouth.ducker.speech_start()
            _MIC["btn"] = True
        # The open mic yields while the BUTTON records (or the double
        # capture would turn one held utterance into two turns), and,
        # without barge-in, for the whole turn: while the brain works
        # as well as while the mouth speaks. The face's thinking sound
        # plays through the same speakers during the working phase,
        # and an open mic listening then transcribed it as nonsense
        # ("9, 9, 9...") and interrupted the reply.
        mic_gate = (lambda: _MIC["btn"]
                    or (not barge_in and (mouth.speaking or brain.turn_active)))
        mic_fails = 0
        while True:
            if _MIC["gen"] != mic_gen_seen:
                mic_gen_seen = _MIC["gen"]
                # consume futures that completed under the old mode so
                # a stale press or capture can't fire after a switch
                if press_fut is not None and press_fut.done():
                    press_fut.result(); press_fut = None
                if mic_fut is not None and mic_fut.done():
                    mic_fut.result(); mic_fut = None
            if typed_fut is None:
                typed_fut = loop.run_in_executor(None, typed_q.get)
            if press_fut is None:
                press_fut = loop.run_in_executor(None, ptt.wait_press)
            waiters = {press_fut, typed_fut}
            if _MIC["mode"] == "open":
                if mic_fut is None:
                    g = _MIC["gen"]
                    mic_fut = loop.run_in_executor(
                        None, lambda g=g: (g, ears.listen_once(
                            gate=mic_gate,
                            busy=lambda b: _capturing(None, b),
                            abort=lambda: _MIC["gen"] != g)))
                waiters.add(mic_fut)
            if bridge is not None:
                if ws_press_fut is None:
                    ws_press_fut = asyncio.ensure_future(bridge.wait_press())
                if hf_fut is None:
                    hf_fut = asyncio.ensure_future(hf_q.get())
                waiters |= {ws_press_fut, hf_fut}
            done, _ = await asyncio.wait(
                waiters, return_when=asyncio.FIRST_COMPLETED)
            if hf_fut is not None and hf_fut in done:
                conn, text, typed, who = hf_fut.result(); hf_fut = None
                if typed:
                    # Typed in the face's terminal: no listening checks;
                    # queued behind a reply like a tap. A quit phrase is
                    # ignored, as from hands-free (it would hang up every
                    # device).
                    if any(q in text.lower() for q in QUIT_PHRASES):
                        log("[web] quit phrase typed, ignored")
                        continue
                    await handle(text, spoke_from=time.monotonic(),
                                 interrupt=False,
                                 remote_sink=bridge.make_sink(conn), typed=True, who=who)
                    continue
                if conn.disconnected or not conn.listening:
                    continue
                if conn.listen_muted:
                    # Paused: only "(name,) start listening" or "go
                    # hands free" gets through; the room is heard, never
                    # acted on.
                    if "micopen" in (console_match(text),
                                     trailing_mic_verb(text)[0]):
                        bridge.set_listening(conn, True)
                        mouth.say("Listening.",
                                  remote_sink=bridge.make_sink(conn))
                    else:
                        log(f"[web] paused, ignored: {text[:60]!r}")
                    continue
                if any(q in text.lower() for q in QUIT_PHRASES):
                    log("[web] quit phrase heard hands-free, ignored")
                    continue
                # interrupt=False: an open mic queues behind a reply in
                # progress rather than cutting it off; the Interrupt
                # button is the way to stop one.
                conn.active = time.monotonic()
                await handle(text, spoke_from=time.monotonic(),
                             interrupt=False,
                             remote_sink=bridge.make_sink(conn), who=who)
                continue
            if typed_fut in done:
                text = typed_fut.result(); typed_fut = None
                if text and not await handle(text):
                    return
                continue
            if mic_fut is not None and mic_fut in done:
                try:
                    g, text = mic_fut.result()
                except Exception as e:
                    mic_fut = None
                    mic_fails += 1
                    if not explain_audio_failure(e):
                        log(f"[ears] open mic failed ({mic_fails}): {e!r}")
                    if mic_fails >= 3:
                        _MIC["mode"] = "ptt"
                        _MIC["gen"] += 1
                        mic_fails = 0
                        mouth.say("The open microphone keeps failing, "
                                  "so I'm switching to push to talk. "
                                  "Hold the key to reach me, and "
                                  "check this window for the error.")
                    continue
                mic_fut = None
                if g != _MIC["gen"]:
                    continue             # captured before a switch
                if text:
                    _MIC["active"] = time.monotonic()
                if text and not await handle(text):
                    return
                continue
            if press_fut in done:
                press_fut.result(); press_fut = None
                press_t = _MIC["active"] = time.monotonic()
                await _begin_capture()
                print("[ptt] recording (release to send)...", flush=True)
                try:
                    text = await loop.run_in_executor(
                        None, lambda: record_held(ptt.is_held))
                except Exception as e:
                    # A device-level failure gets plain words instead of a
                    # raw exception. The pre-flight at startup cannot catch
                    # a microphone unplugged mid-session, and that is the
                    # case where the old message was worst: jargon, on
                    # every press, with the key hook still working so it
                    # looked like it was listening.
                    if explain_audio_failure(e):
                        mouth.say("I can't hear you. There's no working "
                                  "microphone I can use.")
                    else:
                        log(f"[ears] record/transcribe failed: {e!r}")
                        mouth.say("My ears hit an error. Check this "
                                  "window for the details.")
                    text = None
                finally:
                    _MIC["btn"] = False
                    _unmute()
                mouth.ducker.speech_end(0.2)     # snap back fast on release
                if not text:
                    log("[ptt] (tap or empty — ignored)")
                    signals.set_state("idle")
                    continue
                if not await handle(text, spoke_from=press_t):
                    return
                continue
            if ws_press_fut is not None and ws_press_fut in done:
                conn, is_interrupt = ws_press_fut.result(); ws_press_fut = None
                press_t = conn.active = time.monotonic()
                await _begin_capture(interrupt=is_interrupt)
                log("[web] recording (release to send)..."
                    + ("" if is_interrupt else " (queued)"))
                g = _MIC["gen"]
                session = Session()
                try:
                    pcm = await bridge.record_until_release(
                        conn, abort=lambda: _MIC["gen"] != g,
                        on_audio=session.add)
                    if pcm is None:
                        session.cancel()
                    text = (await loop.run_in_executor(None, session.finish)
                            if pcm is not None else None)
                    who = (await loop.run_in_executor(None, _who, conn, pcm)
                           if text else None)
                except Exception as e:
                    session.cancel()
                    if explain_audio_failure(e):
                        mouth.say("I can't hear you. There's no working "
                                  "microphone I can use.")
                    else:
                        log(f"[web] record/transcribe failed: {e!r}")
                        mouth.say("My ears hit an error. Check this "
                                  "window for the details.")
                    text = None
                finally:
                    _MIC["btn"] = False
                    _unmute()
                mouth.ducker.speech_end(0.2)
                if not text:
                    log("[web] (tap or empty — ignored)")
                    if is_interrupt:
                        signals.set_state("idle")
                    # else: a queued tap that came up empty shouldn't
                    # stomp on whatever Jarvis is legitimately doing
                    continue
                if any(q in text.lower() for q in QUIT_PHRASES):
                    # A quit phrase from a tab closes only that tab:
                    # the brain is shared now, so running the normal
                    # quit body (interrupt/signoff/stop) would hang up
                    # every OTHER device's turn and speak the signoff
                    # to every tab, not just this one.
                    log("[web] quit phrase, closing this tab only")
                    try:
                        await conn.ws.close()
                    except Exception:
                        pass
                    continue
                await handle(text, spoke_from=press_t,
                            interrupt=is_interrupt,
                            remote_sink=bridge.make_sink(conn), who=who)
    except KeyboardInterrupt:
        pass
    finally:
        _MIC["gen"] += 1     # abort any live open-mic capture promptly
        mouth.shutdown()  # restores the music on Ctrl-C / crash paths too
        signals.static_stop()
        if bridge is not None:
            await bridge.stop()
        signals.set_state("idle")
        await brain.stop()
        log("[backtalk] hung up")


# Loopback port used purely as a mutex. Nothing is ever served on it.
_INSTANCE_PORT = 8791
_instance_lock = None


def _claim_single_instance() -> bool:
    """Refuse to be the second voice line on this machine, out loud.

    Two instances both hold the keyboard hook and both open the
    microphone, and the result looks EXACTLY like a broken talk key:
    presses register, the audio goes to whichever process won the
    device, and the loser reports an ignored tap. Nothing warned about
    it, so a user who double-clicks the Talk icon twice concludes the
    product is broken. The tell, when it was finally caught, was the
    same sentence transcribed twice at an identical timestamp.

    A bound socket is the mutex rather than a pid file, because the
    operating system releases it when this process dies HOWEVER it dies.
    A pid file outlives a crash or a force-kill and then lies about a
    process that is long gone, which is the failure it would exist to
    prevent.
    """
    global _instance_lock
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # No SO_REUSEADDR here on purpose: reuse is exactly what would let a
    # second instance bind alongside the first and defeat the whole point.
    try:
        s.bind(("127.0.0.1", _INSTANCE_PORT))
        s.listen(1)
    except OSError:
        s.close()
        return False
    _instance_lock = s
    return True


def main():
    if not _claim_single_instance():
        print("[backtalk] ANOTHER VOICE LINE IS ALREADY RUNNING on this "
              "machine, so this one is stopping.", flush=True)
        print("[backtalk] Two of them fight over the microphone and the "
              "talk key, which looks exactly like the talk key being "
              "broken. Use the window that is already open, or close it "
              "and start again.", flush=True)
        sys.exit(1)
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n[backtalk] interrupted — hanging up", flush=True)


if __name__ == "__main__":
    main()
