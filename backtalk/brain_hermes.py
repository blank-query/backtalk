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
"""The Hermes brain: WarmBrain with Hermes Agent behind it instead of the
Claude Agent SDK ("brain": "hermes" in backtalk.json).

Hermes runs as its TUI gateway (`hermes --run-module tui_gateway.entry`),
newline-delimited JSON-RPC over stdio, the same protocol its own TUI and
desktop app speak. HermesClient below translates that protocol into the
handful of SDK message shapes WarmBrain's reader already understands
(StreamEvent, AssistantMessage, ResultMessage, the task messages), so the
speaking, tags, queueing, capture, interrupt, and face logic are WarmBrain's
own, unchanged: one reader, two brains, nothing to drift.

Event mapping (tui_gateway/contracts/events.py):
  message.start                 -> a turn begins (face: thinking)
  reasoning.delta/thinking.delta -> a thinking block (logged shape only)
  message.delta                 -> text_delta (spoken sentence by sentence)
  tool.generating / tool.start  -> the text block ends (pre-tool speech plays now)
  subagent.start / .complete    -> TaskStarted / TaskNotification (satellites)
  message.complete              -> ResultMessage (turn over; usage; resume id)

Config ("hermes" in backtalk.json):
  home     the Hermes profile dir (HERMES_HOME), e.g. ~/.hermes/profiles/friday
  cwd      the session's working directory (default: agent_dir)
  command  the gateway command (default: hermes --run-module tui_gateway.entry)
  env      extra environment for the gateway, e.g. HERMES_TUI_TOOLSETS
The spoken-delivery discipline reaches the model through the environment:
the profile's config.yaml sets agent.system_prompt: "${BACKTALK_SYSTEM_PROMPT}".
"""
import asyncio
import itertools
import json
import os

from claude_agent_sdk import TaskNotificationMessage, TaskStartedMessage

from backtalk.brain import WarmBrain
from backtalk.config import CFG, DISCIPLINE
from backtalk.vlog import log

SESSION_FILE = os.path.join(CFG["signals_dir"], ".backtalk_session_hermes")
_TERMINAL = {"completed": "completed", "failed": "failed", "error": "failed",
             "timeout": "failed", "interrupted": "stopped"}
# Server-to-client requests answered: approval (the spoken gate). Anything
# else (clarify, sudo, secret...) gets -32601 so the agent fails fast
# instead of waiting out its timeout.


# The SDK message shapes WarmBrain's reader dispatches on (by class name).
class StreamEvent:
    def __init__(self, event):
        self.event = event


class _Text:
    def __init__(self, text):
        self.text = text


class AssistantMessage:
    def __init__(self, text):
        self.content = [_Text(text)]


class ResultMessage:
    def __init__(self, usage, session_id, cost):
        self.usage, self.session_id, self.total_cost_usd = usage, session_id, cost


class HermesClient:
    """ClaudeSDKClient's surface (connect, query, receive_messages,
    interrupt, disconnect, get_context_usage, set_permission_mode) over a
    Hermes TUI gateway process."""

    def __init__(self, system_prompt: str, can_use_tool=None):
        h = CFG.get("hermes") or {}
        self.home = os.path.expanduser(h.get("home") or "~/.hermes")
        self.cwd = os.path.expanduser(h.get("cwd") or CFG["agent_dir"])
        self.cmd = h.get("command") or ["hermes", "--run-module", "tui_gateway.entry"]
        self.env = {k: str(v) for k, v in (h.get("env") or {}).items()}
        self.system_prompt = system_prompt
        self.can_use_tool = can_use_tool
        self.sid = None              # the live (runtime) session id
        self.stored_id = None        # the durable one: what resume takes
        self._proc = None
        self._ids = itertools.count(1)
        self._calls: dict = {}
        self._q: asyncio.Queue = asyncio.Queue()
        self._pump_task = None
        self._ready = None
        self._block = None           # the open content block: "text" / "thinking"
        self._last_usage = {}        # message.complete usage is cumulative

    # -- process and RPC -------------------------------------------------
    async def connect(self, resume: str | None = None):
        env = dict(os.environ, HERMES_HOME=self.home, PYTHONUNBUFFERED="1",
                   BACKTALK_SYSTEM_PROMPT=self.system_prompt, **self.env)
        log_dir = os.path.dirname(os.environ.get("BACKTALK_LOG") or "") or CFG["signals_dir"]
        err = open(os.path.join(log_dir, "hermes-gateway.log"), "ab")
        self._ready = asyncio.get_running_loop().create_future()
        self._proc = await asyncio.create_subprocess_exec(
            *self.cmd, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=err, cwd=self.cwd,
            env=env, limit=64 * 1024 * 1024)
        err.close()
        self._pump_task = asyncio.ensure_future(self._pump())
        await asyncio.wait_for(self._ready, 120)
        await self.call("client.capabilities", {"server_requests": True})
        if resume:
            try:
                r = await self.call("session.resume", {"session_id": resume, "omit_messages": True})
                self.sid = r["session_id"]
                self.stored_id = r.get("stored_session_id") or resume
                log(f"[brain] hermes resumed session {self.stored_id}")
                return
            except Exception as e:
                log(f"[brain] hermes resume failed ({str(e)[:80]}), starting fresh")
        await self._create()

    async def _create(self):
        r = await self.call("session.create", {"cwd": self.cwd, "cwd_explicit": True})
        self.sid = r["session_id"]
        self.stored_id = r.get("stored_session_id") or (r.get("info") or {}).get("stored_session_id")
        log(f"[brain] hermes session {self.stored_id} in {self.cwd}")

    async def call(self, method: str, params: dict, timeout: float = 60):
        rid = str(next(self._ids))
        fut = asyncio.get_running_loop().create_future()
        self._calls[rid] = fut
        self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._calls.pop(rid, None)

    def _write(self, frame: dict):
        if self._proc and self._proc.stdin and not self._proc.stdin.is_closing():
            self._proc.stdin.write((json.dumps(frame) + "\n").encode())

    async def _pump(self):
        """The gateway's stdout: responses resolve calls, events become
        SDK-shaped messages on the queue, server requests get answered."""
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get("method") == "event":
                    p = msg.get("params") or {}
                    if p.get("type") == "gateway.ready" and not self._ready.done():
                        self._ready.set_result(True)
                    elif p.get("session_id") in (None, "", self.sid):
                        self._event(p.get("type"), p.get("payload") or {})
                elif "method" in msg:
                    asyncio.ensure_future(self._server_request(msg))
                else:
                    fut = self._calls.get(str(msg.get("id")))
                    if fut and not fut.done():
                        if "error" in msg:
                            fut.set_exception(RuntimeError((msg["error"] or {}).get("message", "error")))
                        else:
                            fut.set_result(msg.get("result") or {})
        except Exception as e:
            log(f"[brain] hermes pump failed: {e!r}")
        finally:
            for fut in self._calls.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("hermes gateway exited"))
            if self._ready and not self._ready.done():
                self._ready.set_exception(RuntimeError("hermes gateway exited before ready"))
            self._q.put_nowait(RuntimeError("hermes gateway exited"))

    async def _server_request(self, msg):
        rid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if method != "approval":
            self._write({"jsonrpc": "2.0", "id": rid,
                         "error": {"code": -32601, "message": f"{method} not supported by backtalk"}})
            return
        choice = "deny"
        try:
            res = await self.can_use_tool("Bash", {"command": params.get("command", ""),
                                                   "description": params.get("description", "")}, None)
            choice = "once" if getattr(res, "behavior", "") == "allow" else "deny"
        except Exception as e:
            log(f"[brain] hermes approval gate failed: {e!r}")
        log(f"[perm]   hermes approval: {choice}")
        self._write({"jsonrpc": "2.0", "id": rid, "result": {"choice": choice}})

    # -- events -> SDK shapes --------------------------------------------
    def _put(self, ev: dict):
        self._q.put_nowait(StreamEvent(ev))

    def _open(self, kind: str):
        if self._block != kind:
            self._close()
            self._put({"type": "content_block_start", "content_block": {"type": kind}})
            self._block = kind

    def _close(self):
        if self._block:
            self._put({"type": "content_block_stop"})
            self._block = None

    def _event(self, kind: str, p: dict):
        if kind == "message.start":
            self._block = None
            self._put({"type": "message_start"})
        elif kind in ("reasoning.delta", "thinking.delta"):   # not .available: it repeats the deltas
            if not p.get("text"):
                return
            self._open("thinking")
            self._put({"type": "content_block_delta",
                       "delta": {"type": "thinking_delta", "thinking": p.get("text", "")}})
        elif kind == "message.delta" or (kind == "message.interim" and not p.get("already_streamed")):
            self._open("text")
            self._put({"type": "content_block_delta",
                       "delta": {"type": "text_delta", "text": p.get("text", "")}})
        elif kind == "tool.generating":
            self._close()
        elif kind == "tool.start":
            self._close()
            self._put({"type": "content_block_start", "content_block": {"type": "tool_use"}})
            self._put({"type": "content_block_stop"})
        elif kind == "subagent.start" and p.get("subagent_id"):
            self._q.put_nowait(TaskStartedMessage(
                subtype="task_started", data=p, task_id=p["subagent_id"],
                description=p.get("goal", ""), uuid="", session_id=self.stored_id or ""))
        elif kind == "subagent.complete" and p.get("subagent_id"):
            self._q.put_nowait(TaskNotificationMessage(
                subtype="task_notification", data=p, task_id=p["subagent_id"],
                status=_TERMINAL.get(str(p.get("status") or "completed"), "completed"),
                output_file="", summary=p.get("summary") or "", uuid="",
                session_id=self.stored_id or ""))
        elif kind == "session.info" and p.get("stored_session_id"):
            self.stored_id = p["stored_session_id"]
        elif kind == "message.complete":
            self._close()
            text = p.get("text")
            if isinstance(text, str) and text:
                self._q.put_nowait(AssistantMessage(text))
            if p.get("status") == "error" or p.get("error"):
                log(f"[brain] hermes turn error: {str(p.get('error') or p.get('failure_reason'))[:200]}")
            u = p.get("usage") or {}
            delta = {k: max(0, int(u.get(k) or 0) - int(self._last_usage.get(k) or 0))
                     for k in ("input", "output")}
            if u:
                self._last_usage = u
            self._q.put_nowait(ResultMessage(
                {"input_tokens": delta["input"], "output_tokens": delta["output"]},
                self.stored_id, None))
        elif kind == "error":
            log(f"[brain] hermes error: {p.get('message', '')[:200]}")
        elif kind == "status.update" and p.get("kind") in ("compacting", "lifecycle"):
            log(f"[brain] hermes {p.get('kind')}: {p.get('text', '')[:120]}")

    # -- the SDK client surface --------------------------------------------
    async def query(self, text: str):
        try:
            await self.call("prompt.submit", {"session_id": self.sid, "text": text})
        except Exception as e:
            # no turn will come back for a refused submit: end one here, said out loud
            log(f"[brain] hermes refused the prompt: {e!r}")
            self._put({"type": "message_start"})
            self._put({"type": "content_block_delta", "delta": {
                "type": "text_delta", "text": "My brain refused that message. Check the log."}})
            self._q.put_nowait(ResultMessage({}, self.stored_id, None))

    async def receive_messages(self):
        while True:
            m = await self._q.get()
            if isinstance(m, Exception):
                raise m
            yield m

    async def interrupt(self):
        await self.call("session.interrupt", {"session_id": self.sid}, timeout=5)

    async def get_context_usage(self):
        u = await self.call("session.usage", {"session_id": self.sid}, timeout=10)
        return {"categories": [{"name": "context used", "tokens": u.get("context_used") or 0}]}

    async def set_permission_mode(self, mode: str):
        pass   # approvals reach the spoken gate, which reads backtalk's own mode

    async def disconnect(self):
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.stdin.close()
                await asyncio.wait_for(self._proc.wait(), 10)
            except Exception:
                self._proc.kill()
        if self._pump_task:
            self._pump_task.cancel()


class HermesBrain(WarmBrain):
    """WarmBrain with a HermesClient: everything that speaks is inherited."""

    def _new_client(self):
        return HermesClient(DISCIPLINE + self._append, self._can_use_tool)

    async def start(self):
        resume, self._resume_id = self._resume_id, None
        self._client = self._new_client()
        await self._client.connect(resume)
        self._start_reader()
        self.clear_tasks()

    async def _rebuild(self):
        """The gateway died under the reader: start a new one on the same
        stored session (Hermes keeps it in its state.db), so a crash costs
        the turn in flight, not the conversation."""
        old = self._client
        try:
            await old.disconnect()
        except Exception:
            pass
        self._client = self._new_client()
        await self._client.connect(old.stored_id)
        self._turn_active = False
        self._current_asker = None
        self._dispatched = False
        self._dispatch_next()
        log("[brain] reader rebuilt the hermes gateway after a stream error")

    async def _pull_rate_limits(self):
        pass   # no plan limits behind a local model

    def _remember_session(self, rm):
        if not CFG.get("resume_last_session") or not self._persist or not rm.session_id:
            return
        try:
            with open(SESSION_FILE, "w") as f:
                f.write(rm.session_id)
        except OSError:
            pass

    async def command(self, cmd: str) -> str:
        """The console verbs, as gateway RPCs instead of typed slash text."""
        verb, _, arg = cmd.strip().partition(" ")
        c = self._client
        try:
            if verb == "/clear":
                await c.call("session.close", {"session_id": c.sid})
                await c._create()
                c._last_usage = {}
                return "cleared"
            if verb == "/effort":
                r = await c.call("config.set", {"session_id": c.sid, "key": "reasoning", "value": arg})
                return f"reasoning {r.get('value', arg)}"
            if verb == "/model":
                r = await c.call("command.dispatch", {"name": "model", "arg": arg, "session_id": c.sid})
                return r.get("output") or r.get("notice") or ""
            if verb == "/compact":
                # a turn of its own: asks queue behind it, the face shows thinking
                self._turn_active = True
                self.bus.set_state("thinking")
                try:
                    r = await c.call("session.compress", {"session_id": c.sid},
                                     timeout=float(CFG.get("capture_timeout_s") or 90) * 4)
                    return str((r.get("summary") or {}).get("headline") or r.get("status") or "")
                finally:
                    self._turn_active = False
                    self._dispatch_next()
                    if not self._turn_active and not (self.mouth and self.mouth.speaking):
                        self.bus.set_state("idle")
            r = await c.call("slash.exec", {"session_id": c.sid, "command": cmd.strip()})
            return r.get("output") or ""
        except Exception as e:
            log(f"[brain] hermes {verb} failed: {e!r}")
            return f"error: {e}"

