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
"""The warm brain — a persistent Claude session via the Agent SDK,
streaming.

One ClaudeSDKClient lives for the whole voice session: no per-turn
process spawn, no per-turn context reload. Partial-message streaming
means sentences are yielded the moment they're complete, so the mouth
starts speaking while the rest of the thought is still forming.

THE TURN STREAM: a Claude Code session is not request/response. Some
turns start without anyone here asking — a background task finishing,
a subagent reporting — and each has its own ResultMessage on the SAME
shared stream. The old design (`ask_stream`, per-query
`receive_response()`) assumed the next ResultMessage after a query was
that query's own answer; it wasn't, for any turn that wasn't started
by a direct ask. One background report and the whole session's answers
ran one turn behind, permanently (see the Backtalk Turn Stream
Redesign note). So instead: ONE reader task owns `receive_messages()`
for the session's lifetime and speaks every turn as it arrives,
whoever started it. Sending is just sending — `ask()` calls `query()`
and returns; it never assumes the next reply is its own.

The session's cwd is YOUR agent's folder (agent_dir in backtalk.json) —
whatever CLAUDE.md lives there defines who is speaking. backtalk adds
only the spoken-delivery discipline (config.DISCIPLINE): the medium,
never the character.
"""
import asyncio
import os
import re
import time
import warnings
from collections import deque
from datetime import datetime

from claude_agent_sdk import (
    ClaudeAgentOptions, ClaudeSDKClient, TERMINAL_TASK_STATUSES,
    TaskNotificationMessage, TaskStartedMessage, TaskUpdatedMessage,
)

try:
    from claude_agent_sdk import CanUseToolShadowedWarning
except ImportError:                       # older SDKs: nothing to silence
    CanUseToolShadowedWarning = None

from backtalk import signals
from backtalk.config import CFG, DISCIPLINE
from backtalk.vlog import log

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def _sentence_end(buf: str):
    """The first sentence break that isn't inside an unfinished <<tag>>: an
    announcement's text has full stops of its own, and splitting there
    left the tag in pieces, spoken aloud instead of acted on."""
    for m in _SENTENCE_END.finditer(buf):
        head = buf[:m.start()]
        if head.count("<<") <= head.count(">>"):
            return m
    return None
# <<anything>> is a stage direction: lifted out, never spoken, published on
# the bus when the audio carrying it starts. Bounded so a runaway model
# cannot swallow a paragraph into one "tag" (long enough for a phone
# command's text message; see mouth.py's "phone" directions).
_DIRECTION_TAG = re.compile(r"<<([^<>]{1,600})>>")
def _line(sink, text: str):
    """The reply as text for the asking device's terminal (see web.py)."""
    send = getattr(sink, "send", None)
    if send is not None:
        send({"type": "line", "who": "jarvis", "text": text})


FLUSH_AFTER = 0.75   # seconds of silence before a lone sentence is spoken,
                     # see the Backtalk Orphaned Sentence Bug note


SESSION_FILE = os.path.join(CFG["signals_dir"], ".backtalk_session")


class WarmBrain:
    def __init__(self, model: str | None = None, can_use_tool=None,
                 resume_id: str | None = None, mouth=None, bus=None,
                 append: str = "", persist: bool = True, label: str = "Jarvis"):
        # Full model id ON PURPOSE — never a bare alias. The SDK
        # resolves aliases through its own bundled CLI and can silently
        # land on an older model.
        self.model = model or CFG["model"]
        # Where face signals go (see Mouth), and extra system-prompt text
        # for a dedicated session ("you are the kitchen's cooking session").
        self.bus = bus or signals
        self._append = append
        # Only the main session's id is saved for resume: a dedicated
        # session saving its own would hijack the next launch.
        self._persist = persist
        self.label = label           # the log's speaker tag: "Jarvis", "Jarvis@Kitchen"
        # The spoken permission gate (main.py builds it). Wired at
        # connect in EVERY mode, so a live mode flip needs no reconnect;
        # bypass simply never consults it.
        self._can_use_tool = can_use_tool
        # The mouth the reader speaks every turn through. Set once at
        # construction; main.py owns the Mouth instance's lifetime.
        self.mouth = mouth
        # Fallback sink for a turn nobody specific asked for (a
        # background report); main.py sets this once to a broadcast
        # sink reaching every connected browser. A turn that WAS asked
        # by a specific connection routes to that connection instead,
        # via _current_asker below.
        self.remote_sink = None
        # Utterances not yet sent to the SDK: (utterance, remote_sink)
        # pairs, queued because a turn was already in flight when
        # ask() was called. The session only ever has ONE query
        # outstanding at a time (see _dispatch_next), so there is
        # never more than one live asker to route a turn to: the
        # asker is just whoever's query is actually running right
        # now (_current_asker below), a direct reference, never a
        # position guessed out of a queue.
        self._ask_queue: deque = deque()
        self._current_asker = None   # remote_sink owed the in-flight turn
        self._ask_t0 = None          # when that turn's ask went out, for the log
        # Session usage, spoken on request ("usage report").
        self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0,
                        "cost": 0.0}
        self._client: ClaudeSDKClient | None = None
        # The session to reattach to at the FIRST start only (config key
        # resume_last_session). Consumed on use.
        self._resume_id = resume_id
        self._reader_task: asyncio.Task | None = None
        # Set by interrupt() when a turn is actually in flight; tells
        # the reader to discard that turn's trailing content (already
        # cut off locally by mouth.shut_up()) through its ResultMessage,
        # rather than speaking a dead turn's leftovers after the fact.
        self._discard_until_result = False
        # True from the first content of a turn until its ResultMessage.
        # interrupt() is a no-op when this is False: nothing in flight,
        # nothing to discard.
        self._turn_active = False
        # When set, the current turn's text is being collected for the
        # caller (console commands, the warmup ping) instead of being
        # spoken — see capture().
        self._capture: asyncio.Future | None = None
        self._capture_count_turn = True
        self._capture_buf: list[str] = []
        # Background tasks currently running, keyed by task_id (a set,
        # not a bare counter: duplicate or missed events must not drift
        # the count). Published on every change so faces can draw one
        # satellite per task.
        self._active_tasks: set[str] = set()
        # Which device's turn started each background task. Its finish
        # makes the agent speak up in a turn nobody asked for just then,
        # but it WAS asked for, in advance (a cooking timer, a long
        # build): that turn goes back to the device that asked.
        self._task_owner: dict = {}

    async def start(self):
        mode = CFG["permission_mode"]
        if mode == "default":
            mode = "ask"     # legacy alias, see config.py
        # backtalk's "ask" = the SDK's "default" mode with gated calls
        # routed to the spoken can_use_tool gate.
        sdk_mode = "default" if mode == "ask" else mode
        if sdk_mode == "bypassPermissions" and self._can_use_tool \
                and CanUseToolShadowedWarning:
            # Deliberate auto-approve: the SDK warns that the callback is
            # shadowed. That IS the chosen behavior, so boot quietly.
            warnings.filterwarnings("ignore",
                                    category=CanUseToolShadowedWarning)
        resume, self._resume_id = self._resume_id, None   # consume once

        def _opts(rid):
            return ClaudeAgentOptions(
                cwd=CFG["agent_dir"],
                model=self.model,
                system_prompt={"type": "preset", "preset": "claude_code",
                               "append": DISCIPLINE + self._append},
                include_partial_messages=True,
                permission_mode=sdk_mode,
                can_use_tool=self._can_use_tool,
                add_dirs=CFG["extra_dirs"],
                mcp_servers=CFG["mcp_servers"],
                skills=CFG["visible_skills"],
                resume=rid,
                # SDK default is 1 MB; one screenshot or gif frame blew it
                # and wiped the session. 50 MB fits big PDFs and images.
                max_buffer_size=50 * 1024 * 1024,
            )
        if resume:
            try:
                self._client = ClaudeSDKClient(options=_opts(resume))
                await self._client.connect()
                log(f"[brain] resumed session {resume[:8]}")
                self._start_reader()
                self.clear_tasks()
                return
            except Exception as e:
                # a stale or invalid saved session must never brick the
                # launch. Fall back to a fresh conversation and say so.
                log(f"[brain] resume failed ({str(e)[:80]}), "
                    f"starting fresh")
                try:
                    await self._client.disconnect()
                except Exception:
                    pass
        self._client = ClaudeSDKClient(options=_opts(None))
        await self._client.connect()
        self._start_reader()
        self.clear_tasks()

    def _dispatch_next(self):
        """Send the next queued utterance, if the session is free.
        Whoever's query this is becomes _current_asker directly, no
        popping-and-hoping at the other end when the turn's content
        shows up later. The asker's tab is stamped HERE, not at first
        content: the model can think for seconds before any text
        streams, and a face polling during that gap would otherwise
        show the working ring on whichever tab asked last. While a
        reply's audio is actually playing, mouth.py's playing-tab
        marker overrides this one (see signals.set_playing_conn)."""
        if self._turn_active or not self._ask_queue:
            return
        utterance, remote_sink, self._ask_t0 = self._ask_queue.popleft()
        self._turn_active = True
        self._current_asker = remote_sink
        self.bus.set_active_conn(getattr(remote_sink, "conn_id", None))
        asyncio.ensure_future(self._client.query(utterance))

    def clear_tasks(self):
        """Reset the active-task set to empty (fresh launch, or the
        session itself was cleared/reset underneath it)."""
        self._active_tasks.clear()
        self._task_owner.clear()
        self.bus.set_tasks(0)
        self.bus.set_active_conn(None)

    def _start_reader(self):
        self._reader_task = asyncio.ensure_future(self._read_forever())

    @property
    def turn_active(self) -> bool:
        """Whether a turn is currently in flight (content seen since
        the last ResultMessage). main.py uses this to decide whether
        there's anything for interrupt() to actually interrupt, and
        whether to log it as one."""
        return self._turn_active

    async def set_permission_mode(self, backtalk_mode: str):
        """Live flip, no reconnect, conversation intact ("ask" maps to
        the SDK's "default", whose gated calls hit the spoken gate)."""
        if self._client:
            sdk_mode = "default" if backtalk_mode == "ask" \
                else backtalk_mode
            await self._client.set_permission_mode(sdk_mode)

    async def context_usage(self):
        """The CLI's own context-window breakdown, or None."""
        try:
            return await self._client.get_context_usage()
        except Exception:
            return None

    def _remember_session(self, rm):
        """Persist the session id after a completed turn, so the next
        launch can reattach (config: resume_last_session). Must never
        break a turn; silence on any failure."""
        if not CFG.get("resume_last_session") or not self._persist:
            return
        sid = getattr(rm, "session_id", None)
        if not sid:
            return
        try:
            with open(SESSION_FILE, "w") as f:
                f.write(sid)
        except OSError:
            pass

    def _tally(self, rm, count_turn=True):
        """Session usage bookkeeping. Must never break a turn."""
        try:
            u = getattr(rm, "usage", None) or {}
            s = self.session
            if count_turn:
                s["turns"] += 1
            s["out_tokens"] += int(u.get("output_tokens") or 0)
            s["in_tokens"] += (int(u.get("input_tokens") or 0)
                               + int(u.get("cache_read_input_tokens")
                                     or 0))
            c = getattr(rm, "total_cost_usd", None)
            if c:
                s["cost"] += float(c)
        except Exception:
            pass

    async def _pull_rate_limits(self):
        """Ask the CLI outright how much of the plan is spent.

        A DIRECT QUERY, not the RateLimitEvent stream. The event fires
        rarely and usually arrives carrying resets_at with no utilization
        at all, so a listener built on it reports nothing most of the
        time -- which is exactly how this feature looked broken for its
        whole life. (Community fix, ai-visualizer issue #1.)

        THIS REACHES PAST THE SDK'S PUBLIC SURFACE ON PURPOSE, and a
        reader should know it rather than discover it. `get_usage` is a
        control request the bundled CLI answers but the SDK never wraps,
        so there is no supported call to make. The supported-looking
        alternative is a dead end and was tested as one: the terminal
        status line never fires in a headless session, so its numbers
        are unreachable from here.

        Which means this can stop working without anyone doing anything
        wrong, and the containment is the point. Every failure is
        swallowed and the readout simply goes quiet. It must never cost
        a turn, so it is also bounded -- an unanswered control request
        would otherwise hang the voice line mid-conversation."""
        if not CFG.get("show_usage"):
            return
        try:
            usage = await asyncio.wait_for(
                self._client._query._send_control_request(
                    {"subtype": "get_usage"}), 5)
            for window in ("five_hour", "seven_day"):
                w = (usage.get("rate_limits") or {}).get(window)
                if not w:
                    continue
                # Two spellings accepted deliberately: this shape is not
                # documented anywhere, so the cheap tolerance is worth
                # more than the tidiness. Both are percentages, and the
                # rest of the pipeline wants a 0..1 fraction.
                pct = w.get("utilization")
                if pct is None:
                    pct = w.get("used_percentage")
                pct = pct / 100 if pct is not None else None
                resets = w.get("resets_at")
                if isinstance(resets, str):
                    resets = int(datetime.fromisoformat(resets).timestamp())
                self.bus.set_rate_limit(window, pct, resets)
        except Exception:
            pass

    def ask(self, utterance: str, remote_sink=None):
        """Queue an utterance. SENDING IS JUST SENDING: this does not
        wait for or return the reply — the reader speaks it, whoever's
        turn it turns out to be. Equivalent for the caller's purposes
        to firing a query and walking away.

        `remote_sink`, when given, names the SPECIFIC connection that
        asked. Omit it (the local key's case) and the turn falls back
        to whatever self.remote_sink is set to.

        If nothing else is in flight, this dispatches immediately and
        turn_active flips True synchronously, before returning: a
        press landing in the gap between sending and the model's
        first token must still find a turn to interrupt. If a turn IS
        already running, this just queues: the session never has two
        queries outstanding at once, so there's never ambiguity about
        which asker a turn belongs to later."""
        self._ask_queue.append((utterance, remote_sink, time.time()))
        self._dispatch_next()

    async def capture(self, text: str, count_turn: bool = True) -> str:
        """Send `text` and wait for the FULL text reply, un-spoken —
        for console slash commands and the startup warmup ping, the two
        callers that want an answer back as a return value rather than
        audio. Bounded: this stream is not trusted to always deliver,
        and an unbounded await here would deafen the whole voice loop.
        Only one capture may be in flight at a time (both callers are
        already serialized by the caller)."""
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._capture = fut
        self._capture_count_turn = count_turn
        self._capture_buf = []
        await self._client.query(text)
        try:
            return await asyncio.wait_for(fut, 90)
        except asyncio.TimeoutError:
            log(f"[brain] capture timed out: {text!r}")
            return "error: the command timed out"
        finally:
            if self._capture is fut:
                self._capture = None

    async def command(self, cmd: str) -> str:
        """Run a console slash command (/clear, /compact, /model,
        /effort) and return whatever text the CLI answered with
        (confirmations, errors)."""
        return await self.capture(cmd, count_turn=False)

    async def interrupt(self):
        """Stop whatever turn is live. A no-op when nothing is in
        flight — there would be nothing for the reader to discard, and
        discarding with nothing to discard-UNTIL would eat the next
        turn's real content instead."""
        if not self._client or not self._turn_active:
            return
        self._discard_until_result = True
        try:
            await asyncio.wait_for(self._client.interrupt(), 5)
        except Exception:
            pass  # the turn may already be over; the discard flag is the point

    async def stop(self):
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None
        if self._client:
            await self._client.disconnect()
            self._client = None

    async def _read_forever(self):
        """THE reader: the ONLY consumer of the session's message
        stream, for its whole lifetime. Never stops, never pairs a
        turn to the query that may or may not have started it — it
        just speaks content as it arrives and resets at every
        ResultMessage. A turn that was discarded (interrupt()) or
        captured (capture()) takes the same boundary, just routed
        differently instead of spoken.

        Per-turn speaking state (first/batch/pending) lives in this
        loop's locals, not on self: there is exactly one turn's worth
        of it alive at a time, same invariant the old speak_reply()
        held, just no longer scoped to a single query's generator."""
        first = True
        batch: list[str] = []
        pending: list[str] = []
        buf = ""
        turn_sink = None   # resolved at this turn's first content, below
        quiet = False      # <<quiet>>: this reply is text only (a typed question)
        # The turn's content blocks in order, [type, chars of thinking].
        # Logged once per turn: an answer written only as reasoning (a
        # thinking block, then a tool call, no text) is never spoken, and
        # this line is how that gets caught (seen once, 2026-10-04).
        shape: list = []

        def log_shape():
            nonlocal shape
            if shape:
                desc = " ".join(f"{b}({n})" if b == "thinking" and n else b for b, n in shape)
                first_tool = next((i for i, (b, _) in enumerate(shape) if b == "tool_use"), len(shape))
                head = shape[:first_tool]
                if first_tool < len(shape) and not any(b == "text" for b, _ in head) \
                        and any(b == "thinking" and n for b, n in head):
                    log(f"[turn] WARNING: reasoning but no spoken text before the first tool call: {desc}")
                else:
                    log(f"[turn] blocks: {desc}")
            shape = []

        def flush():
            nonlocal batch, pending
            if batch and self.mouth:
                self.mouth.say_chunk(" ".join(batch), pending, turn_sink)
                pending = []
                batch = []

        def emit(raw: str):
            nonlocal first, batch, pending, turn_sink, quiet
            found = _DIRECTION_TAG.findall(raw)
            if found:
                pending += [d.strip() for d in found if d.strip()]
            if "quiet" in pending:
                quiet = True
                pending = [d for d in pending if d != "quiet"]
            raw = _DIRECTION_TAG.sub(" ", raw)
            s = " ".join(raw.replace("`", "").split()).strip()
            if not s or not self.mouth:
                return
            if quiet:
                # Text only: the asking device's terminal gets it, nothing
                # is spoken; its tags still fire (see mouth._run).
                if first:
                    turn_sink = self._current_asker or self.remote_sink
                    self._ask_t0 = None
                    self.bus.static_stop()
                log(f"[{self.label}] (quiet) {s}"
                    + (f"  <directions: {pending}>" if pending else ""))
                _line(turn_sink, s)
                if pending:
                    self.mouth.say_chunk("", pending, turn_sink)
                pending = []
                first = False
                return
            if first:
                # _current_asker is the exact remote_sink whose ask()
                # caused the query now running: a direct reference,
                # set at dispatch, not inferred from queue order. Only
                # check liveness here: the tab may have closed in the
                # time between being queued and this turn starting.
                turn_sink = self._current_asker
                if turn_sink is not None and hasattr(turn_sink, "is_live") \
                        and not turn_sink.is_live():
                    turn_sink = None
                turn_sink = turn_sink or self.remote_sink
                # time from the ask going out to the first spoken sentence
                # (a turn nobody asked for, a background report, has none)
                lag = (f"({time.time() - self._ask_t0:.1f}s to first) "
                       if self._ask_t0 else "")
                self._ask_t0 = None
                log(f"[{self.label}] {lag}{s}"
                    + (f"  <directions: {pending}>" if pending else ""))
                _line(turn_sink, s)
                self.mouth.say_chunk(s, pending, turn_sink)
                pending = []
                first = False
            else:
                log(f"[{self.label}] {s}"
                    + (f"  <directions: {pending}>" if pending else ""))
                _line(turn_sink, s)
                batch.append(s)
                if len(batch) >= 2:
                    flush()

        def end_turn():
            nonlocal first, batch, pending, buf, quiet
            tail = buf.strip()
            buf = ""
            if tail:
                emit(tail)
            flush()
            if pending and self.mouth:
                # A tag after the last sentence: nothing left to carry it,
                # so it goes alone and fires when the speech before it ends.
                self.mouth.say_chunk("", pending, self._current_asker if first else turn_sink)
            if first or quiet:
                # Zero sentences yielded (brain error / empty turn), or a
                # text-only reply: park the bus rather than leave it on
                # "thinking" forever.
                self.bus.static_stop()
                self.bus.set_state("idle")
            # The turn is over, spoken or not: a quiet reply has no
            # reply_done, and a peer agent waits on this (see say.py).
            send = getattr(turn_sink if not first else (self._current_asker or self.remote_sink),
                           "send", None)
            if send is not None:
                send({"type": "turn_done"})
            first, batch, pending, quiet = True, [], [], False

        stream = self._client.receive_messages().__aiter__()
        nxt = asyncio.ensure_future(stream.__anext__())
        try:
            while True:
                done, _ = await asyncio.wait({nxt}, timeout=FLUSH_AFTER)
                if not done:
                    if not self._discard_until_result:
                        flush()      # a pause (tool work): speak what's waiting
                    await asyncio.wait({nxt})
                try:
                    msg = nxt.result()
                except StopAsyncIteration:
                    break
                except Exception as e:
                    log(f"[brain] reader stream error: {e!r} — rebuilding")
                    await self._rebuild()
                    stream = self._client.receive_messages().__aiter__()
                    nxt = asyncio.ensure_future(stream.__anext__())
                    continue
                nxt = asyncio.ensure_future(stream.__anext__())
                t = type(msg).__name__

                if isinstance(msg, TaskStartedMessage):
                    self._active_tasks.add(msg.task_id)
                    self.bus.set_tasks(len(self._active_tasks))
                    if self._current_asker is not None:
                        self._task_owner[msg.task_id] = self._current_asker
                elif isinstance(msg, (TaskNotificationMessage,
                                      TaskUpdatedMessage)):
                    # A terminal status can arrive on EITHER message type,
                    # and for TaskUpdatedMessage only in `patch` on some
                    # SDK versions — check both spots, see the class docs.
                    status = getattr(msg, "status", None) \
                        or (getattr(msg, "patch", None) or {}).get("status")
                    if status in TERMINAL_TASK_STATUSES:
                        self._active_tasks.discard(msg.task_id)
                        self.bus.set_tasks(len(self._active_tasks))
                        # The turn this finish triggers comes next: owe it
                        # to the device that started the task, unless a
                        # turn is already running (it keeps its own asker).
                        owner = self._task_owner.pop(msg.task_id, None)
                        if owner is not None and not self._turn_active \
                                and self._current_asker is None:
                            self._current_asker = owner
                            self.bus.set_active_conn(getattr(owner, "conn_id", None))

                if self._capture is not None:
                    if t == "AssistantMessage":
                        for b in getattr(msg, "content", []) or []:
                            txt = getattr(b, "text", None)
                            if txt:
                                self._capture_buf.append(txt)
                    elif t == "ResultMessage":
                        self._turn_active = False
                        self._tally(msg, count_turn=self._capture_count_turn)
                        self._remember_session(msg)
                        fut, self._capture = self._capture, None
                        if fut and not fut.done():
                            fut.set_result(" ".join(self._capture_buf).strip())
                    continue

                if t == "StreamEvent":
                    self._turn_active = True
                    if self._discard_until_result:
                        continue
                    ev = getattr(msg, "event", {}) or {}
                    kind = ev.get("type")
                    if kind == "content_block_start":
                        shape.append([(ev.get("content_block") or {}).get("type", "?"), 0])
                    elif kind == "content_block_delta":
                        delta = ev.get("delta", {}) or {}
                        if delta.get("type") == "thinking_delta" and shape:
                            shape[-1][1] += len(delta.get("thinking", ""))
                        if delta.get("type") == "text_delta":
                            buf += delta.get("text", "")
                            while True:
                                m = _sentence_end(buf)
                                if not m:
                                    break
                                sentence, buf = (buf[:m.end()].strip(),
                                                 buf[m.end():])
                                if sentence:
                                    emit(sentence)
                    elif kind == "content_block_stop":
                        # End of a speech block (e.g. right before a tool
                        # call): flush NOW, or pre-tool filler sits
                        # silent through the whole tool run then plays
                        # glued to the answer.
                        tail = buf.strip()
                        buf = ""
                        if tail:
                            emit(tail)
                elif t == "ResultMessage":
                    was_discarding = self._discard_until_result
                    self._discard_until_result = False
                    self._turn_active = False
                    log_shape()
                    # This turn is over: nobody's question is in flight
                    # until the next dispatch stamps one. Clearing here
                    # (before any next dispatch) means an UNPROMPTED turn
                    # (background report) animates every tab, matching
                    # its broadcast audio. A reply still playing keeps its
                    # tab via mouth's playing stamp, which wins.
                    self.bus.set_active_conn(None)
                    if was_discarding:
                        buf = ""
                        first, batch, pending, quiet = True, [], [], False
                        # tally/remember still happen: the turn really
                        # did run and spend usage, it just wasn't spoken.
                        self._tally(msg)
                        self._remember_session(msg)
                        self._current_asker = None
                        self._dispatch_next()
                        continue
                    self._tally(msg)
                    self._remember_session(msg)
                    await self._pull_rate_limits()
                    end_turn()
                    self._current_asker = None
                    self._dispatch_next()
        finally:
            if not nxt.done():
                nxt.cancel()

    async def _rebuild(self):
        """The stream itself broke (not a turn being interrupted — the
        underlying receive_messages() iterator raised). Reconnect fresh
        rather than run with a dead stream for the rest of the session.
        Loses this voice session's conversation memory; better than a
        silent voice line."""
        try:
            await self._client.disconnect()
        except Exception:
            pass
        self._client = None
        resume, self._resume_id = None, None   # a desync rebuild never gambles on resume
        self._client = ClaudeSDKClient(options=ClaudeAgentOptions(
            cwd=CFG["agent_dir"], model=self.model,
            system_prompt={"type": "preset", "preset": "claude_code",
                           "append": DISCIPLINE + self._append},
            include_partial_messages=True,
            permission_mode=("default" if CFG["permission_mode"] == "ask"
                             else CFG["permission_mode"]),
            can_use_tool=self._can_use_tool, add_dirs=CFG["extra_dirs"],
            mcp_servers=CFG["mcp_servers"],
            skills=CFG["visible_skills"], resume=None,
            max_buffer_size=50 * 1024 * 1024))
        await self._client.connect()
        # The turn that broke is gone; whatever was "in flight" no
        # longer is. Without this, a dead _turn_active=True would
        # wedge _dispatch_next forever, and nothing queued after a
        # rebuild would ever get sent.
        self._turn_active = False
        self._current_asker = None
        self._dispatch_next()
        log("[brain] reader rebuilt the session after a stream error "
            "(conversation memory for this session resets)")


if __name__ == "__main__":
    import time

    async def demo():
        from backtalk.mouth import Mouth
        m = Mouth()
        b = WarmBrain(mouth=m)
        await b.start()
        t0 = time.time()
        for prompt in ("Voice check: greet me in one sentence.",
                       "And what's two plus two, spoken like yourself?"):
            b.ask(prompt)
            await asyncio.sleep(8)
        print(f"done in {time.time()-t0:.1f}s, check speakers")
        await b.stop()

    asyncio.run(demo())
