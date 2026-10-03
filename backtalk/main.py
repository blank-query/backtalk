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

THE VOICE CONSOLE: exact phrases, spoken (or typed) alone, control the
session itself so you never go back to the keyboard: "clear the
session" / "compact the session" / "switch to the deep model" / "back
to the fast model" / "set effort to low" (or medium, high, max) /
"usage report" / "go hands free" and "push to talk mode" (the MIC) /
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
import json
import queue
import re
import socket
import sys
import threading
import time

from backtalk import signals
from backtalk.brain import WarmBrain
from backtalk.config import CFG
from backtalk.ears import (Ears, explain_audio_failure, record_held,
                           transcribe, warm as warm_ears)
from backtalk.mouth import Mouth
from backtalk.ptt import PTTListener
from backtalk.vlog import log
from backtalk.web import BrowserBridge

NAME = CFG["name"]
QUIT_PHRASES = CFG["quit_phrases"]

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
_MIC = {"mode": "ptt", "gen": 0, "btn": False}

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


def make_permission_gate(mouth):
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
        mouth.say(ask)
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
                            mouth.say("No answer, so I didn't do it.")
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
                              "Yes or no?")
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
    "micopen":   ("go hands free", "hands free mode",
                  "hands free listening", "open mic", "open the mic"),
    "micptt":    ("push to talk", "push to talk mode",
                  "back to push to talk", "back to the button"),
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
    norm = " ".join(text.lower().replace("-", " ").split()).strip(" .,!?")
    for verb, phrases in CONSOLE_VERBS.items():
        if norm in phrases:
            return verb
    for lvl in _EFFORTS:
        if norm in (f"set effort to {lvl}", f"effort {lvl}",
                    f"slash effort {lvl}"):
            return f"effort:{lvl}"
    return None


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
    ears = Ears()
    brain = WarmBrain(model=model,
                      can_use_tool=make_permission_gate(mouth),
                      resume_id=resume_id, mouth=mouth)

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

    async def run_console(verb):
        """One voice-console verb. The current reply was already
        cancelled and awaited by handle(); the pipe gets drained here
        before the command goes out. A verb that blows up must never
        take the whole voice session down with it."""
        try:
            await _run_console_inner(verb)
        except Exception as e:
            log(f"[console] {verb} failed: {e}")
            mouth.say("That command hit an error. Check the log.")
            signals.set_state("idle")

    async def _run_console_inner(verb):
        _deny_pending()
        say_after = None
        if verb == "clear":
            resp = await brain.command("/clear")
            say_after = "Cleared. Fresh slate."
        elif verb == "compact":
            mouth.say("Compacting. One moment.")
            resp = await brain.command("/compact")
            say_after = "Compacted. Same conversation, smaller footprint."
        elif verb == "deep":
            mouth.say("Switching to the deep model. Heads up, replies "
                      "get slower. Say back to the fast model when "
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
            mouth.say(_spoken_usage(brain.session,
                                    await brain.context_usage()))
        elif verb == "micopen":
            resp = ""
            if _MIC["mode"] == "open":
                mouth.say("Already in hands-free listening.")
            else:
                _MIC["mode"] = "open"
                _MIC["gen"] += 1
                _write_config_key("mic_mode", "open")
                log("[console] mic_mode -> open (hands-free listening)")
                mouth.say("Hands-free listening on. I'm always "
                          "listening now, so anything said in the room "
                          "can reach me. The talk key still works, and "
                          "holding it always gets you heard. Say push "
                          "to talk mode to bring the button back.")
        elif verb == "micptt":
            resp = ""
            if _MIC["mode"] == "ptt":
                mouth.say("Already on push to talk.")
            else:
                _MIC["mode"] = "ptt"
                _MIC["gen"] += 1
                _write_config_key("mic_mode", "ptt")
                log("[console] mic_mode -> ptt")
                key = str(CFG.get("ptt_key", "home")).replace("_", " ")
                mouth.say(f"Push to talk. Hold the {key} key and "
                          "talk; the mic stays closed otherwise.")
        elif verb == "noask":
            resp = ""
            _CONFIRM["verb"] = "noask"
            _CONFIRM["at"] = time.monotonic()
            mouth.say("Auto-approve means I act without asking "
                      "permission, and it becomes your saved default. "
                      "Say confirm to switch.")
        elif verb == "noask:confirmed":
            resp = ""
            saved = _write_config_key("permission_mode",
                                      "bypassPermissions")
            _AUTOAPPROVE["on"] = True
            log("[console] permission_mode -> bypassPermissions"
                + (" (saved)" if saved else " (session only)"))
            mouth.say(("Auto-approve on, and saved as your default. "
                       if saved else
                       "Auto-approve on for this session. The config "
                       "file couldn't be written, so it won't stick "
                       "past a restart. ")
                      + "Say start asking again any time to flip it "
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
                mouth.say("Done. I'll ask out loud before real "
                          "actions"
                          + (", and that's saved as your default."
                             if saved else
                             ". The config file couldn't be written, "
                             "so tell me again after a restart."))
            else:
                mouth.say("I saved asking as your default, but this "
                          "session couldn't switch over. Restart the "
                          "voice line to get asking back.")
        else:
            resp = ""
        if say_after:
            # the CLI answers slash commands with its own text
            # (confirmations, API errors); an error outranks our line
            low = (resp or "").lower()
            if resp and ("error" in low or "invalid" in low):
                mouth.say(resp[:160])
                log(f"[console] {verb} answered: {resp[:120]}")
            else:
                mouth.say(say_after)
        signals.set_state("idle")

    async def handle(text: str, spoke_from: float | None = None,
                     interrupt: bool = True) -> bool:
        """Process one utterance; returns False on quit. spoke_from is
        when the utterance STARTED (the PTT press), so an answer can be
        told apart from speech that began before the ask even existed.
        interrupt=False is a QUEUED browser tap (Part 2): touch nothing
        already playing, just line this one up behind it."""
        log(f"[you]    {text}")
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
            if brain.turn_active:
                # The reader (brain.py) owns speaking now — interrupt()
                # is a clean async call, nothing here to cancel-and-await.
                log("[turn] interrupted mid-reply by new input")
                _deny_pending()      # an ask never outlives its turn
                await brain.interrupt()
            mouth.shut_up()
            if brain.remote_sink is not None:
                brain.remote_sink.stop()
        verb = verb or console_match(text)
        if verb:
            await run_console(verb)
            return True
        signals.set_state("thinking")
        signals.static_start()
        _deny_pending()
        brain.ask(text)
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
        # brain.remote_sink follows the CONNECTION, not the turn that
        # was asked — once a browser tab is connected, every turn's
        # audio goes to it too, whoever's turn it is (see the Backtalk
        # Turn Stream Redesign note: with turns no longer paired to the
        # query that may have started them, trying to route each one
        # back to its original asker is exactly the kind of per-turn
        # bookkeeping that caused the bug this redesign fixes).
        bridge: BrowserBridge | None = None
        _ws_conn = {"current": None}

        def _ws_connected(conn):
            _ws_conn["current"] = conn
            brain.remote_sink = bridge.make_sink(conn)

        def _ws_disconnected(conn):
            if _ws_conn["current"] is conn:
                _ws_conn["current"] = None
                brain.remote_sink = None

        if CFG.get("web", {}).get("enabled"):
            bridge = BrowserBridge(CFG["web"], on_connect=_ws_connected,
                                   on_disconnect=_ws_disconnected)
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
        # without barge-in, while the mouth speaks.
        mic_gate = (lambda: _MIC["btn"]
                    or (not barge_in and mouth.speaking))
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
                            abort=lambda: _MIC["gen"] != g)))
                waiters.add(mic_fut)
            if bridge is not None:
                if ws_press_fut is None:
                    ws_press_fut = asyncio.ensure_future(bridge.wait_press())
                waiters.add(ws_press_fut)
            done, _ = await asyncio.wait(
                waiters, return_when=asyncio.FIRST_COMPLETED)
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
                if text and not await handle(text):
                    return
                continue
            if press_fut in done:
                press_fut.result(); press_fut = None
                press_t = time.monotonic()
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
                press_t = time.monotonic()
                await _begin_capture(interrupt=is_interrupt)
                log("[web] recording (release to send)..."
                    + ("" if is_interrupt else " (queued)"))
                g = _MIC["gen"]
                try:
                    pcm = await bridge.record_until_release(
                        conn, abort=lambda: _MIC["gen"] != g)
                    text = (await loop.run_in_executor(None, transcribe, pcm)
                            if pcm is not None else None)
                except Exception as e:
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
                mouth.ducker.speech_end(0.2)
                if not text:
                    log("[web] (tap or empty — ignored)")
                    if is_interrupt:
                        signals.set_state("idle")
                    # else: a queued tap that came up empty shouldn't
                    # stomp on whatever Jarvis is legitimately doing
                    continue
                if not await handle(text, spoke_from=press_t,
                                    interrupt=is_interrupt):
                    return
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
