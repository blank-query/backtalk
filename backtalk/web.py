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
"""THE BROWSER BRIDGE: a second push-to-talk, a click/tap on the face in
ai-visualizer instead of (or alongside) the hold-to-talk key.

This module ONLY moves bytes and press/release events over a WebSocket.
It deliberately knows nothing about ears.transcribe(), handle(), or
_MIC — main.py owns all of that, exactly as it already does for the
local key and the open mic, so there is exactly one place that
interrupts a reply and transitions session state. Treat a browser
press as another input source feeding the SAME turn-handling path, not
a parallel reimplementation of it.

Wire protocol, deliberately tiny:
  text frame  {"type": "press"}             browser -> server
  text frame  {"type": "release"}           browser -> server
  binary      <uint32 LE rate><int16 LE PCM...>   either direction
  text frame  {"type": "reply_done"}        server -> browser

Every binary frame carries its own sample rate rather than assuming
16000, because a browser's actual capture rate is a cross-browser
coin flip (see ears.RATE and the resampling in record_until_release).

ONE reader loop per connection (Conn.reader): a WebSocket's incoming
message stream can only have a single consumer, so control frames
(press/release) and binary audio frames are both dispatched from that
one loop — never iterate `conn.ws` a second time anywhere else.
"""
import asyncio
import audioop
import json
import struct

import numpy as np
import websockets

from backtalk.vlog import log

RATE = 16000   # must match ears.RATE — the fixed transcribe() contract


class Conn:
    """One browser tab's live connection."""

    def __init__(self, ws, bridge: "BrowserBridge"):
        self.ws = ws
        self.bridge = bridge
        self.disconnected = False
        self._frames: "asyncio.Queue" = asyncio.Queue()
        self._released = asyncio.Event()

    async def reader(self):
        """The only loop that ever reads this connection's socket."""
        try:
            async for msg in self.ws:
                if isinstance(msg, (bytes, bytearray)):
                    await self._frames.put(bytes(msg))
                    continue
                try:
                    data = json.loads(msg)
                except ValueError:
                    continue
                kind = data.get("type")
                if kind == "press":
                    self.bridge._on_press(self)
                elif kind == "release":
                    self._released.set()
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.disconnected = True
            self._released.set()         # unblock a capture awaiting release
            await self._frames.put(None)  # unblock a capture awaiting a frame


class BrowserBridge:
    """websockets.serve() wrapper. Exactly one connection may be
    "active" (mid-press) at a time — a second tab's press while one is
    in flight is logged and ignored, rather than tracked with a bare
    boolean, so growing this into a small per-connection table later
    (real multi-client support) is a small diff, not a rewrite. That
    is explicitly NOT being built now."""

    def __init__(self, cfg: dict, on_connect=None, on_disconnect=None):
        self._cfg = cfg
        self._server = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._press_q: "asyncio.Queue[Conn]" = asyncio.Queue()
        self._active: Conn | None = None
        # Connection-lifecycle hooks, not press-lifecycle: main.py uses
        # these to route EVERY turn's audio to a connected browser
        # (see main.py's amain), not just turns that tab itself asked
        # for — a tab can be listening without ever pressing anything.
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect

    async def serve(self):
        self._loop = asyncio.get_running_loop()
        host = self._cfg.get("host", "127.0.0.1")
        port = self._cfg.get("port", 8792)

        async def handler(ws):
            conn = Conn(ws, self)
            if self._on_connect:
                self._on_connect(conn)
            try:
                await conn.reader()   # runs until this connection closes
            finally:
                if self._on_disconnect:
                    self._on_disconnect(conn)

        self._server = await websockets.serve(handler, host, port)
        log(f"[web] browser bridge listening on ws://{host}:{port}")
        await self._server.wait_closed()

    def _on_press(self, conn: Conn):
        """Called synchronously from within conn.reader(), already on
        the event loop — no thread hop needed here."""
        if self._active is not None and not self._active.disconnected:
            log("[web] press ignored — another browser turn is live")
            return
        self._active = conn
        conn._released.clear()
        while not conn._frames.empty():
            conn._frames.get_nowait()   # drop anything stale from before this press
        self._press_q.put_nowait(conn)

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def wait_press(self) -> Conn:
        return await self._press_q.get()

    async def record_until_release(self, conn: Conn, abort=None) -> np.ndarray | None:
        """Collect PCM frames until release, disconnect, or abort().
        Resamples any frame whose declared rate isn't already 16000
        (defense against browser/engine quirks) via stdlib audioop —
        no new dependency. Returns None on no audio at all."""
        frames: list[np.ndarray] = []
        try:
            while True:
                if abort and abort():
                    return None
                if conn._released.is_set() and conn._frames.empty():
                    break
                try:
                    chunk = await asyncio.wait_for(conn._frames.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if chunk is None:   # disconnect sentinel
                    break
                if len(chunk) < 4:
                    continue
                rate = struct.unpack_from("<I", chunk, 0)[0]
                pcm_bytes = chunk[4:]
                if rate != RATE and rate > 0:
                    pcm_bytes, _ = audioop.ratecv(
                        pcm_bytes, 2, 1, rate, RATE, None)
                frames.append(np.frombuffer(pcm_bytes, dtype=np.int16))
        finally:
            if self._active is conn:
                self._active = None
        if not frames:
            return None
        return np.concatenate(frames)

    def make_sink(self, conn: Conn):
        """Returns a (rate, pcm) -> None closure for mouth.py's _write
        to call per audio block. Must never let a dead browser socket
        affect local playback — every failure is swallowed here, not
        left for the caller to handle."""
        loop = self._loop

        def _sink(rate: int, pcm: np.ndarray):
            if conn.disconnected or loop is None:
                return
            try:
                frame = struct.pack("<I", rate) + pcm.tobytes()
                asyncio.run_coroutine_threadsafe(conn.ws.send(frame), loop)
            except Exception:
                pass

        def _reply_done():
            if conn.disconnected or loop is None:
                return
            try:
                asyncio.run_coroutine_threadsafe(
                    conn.ws.send(json.dumps({"type": "reply_done"})), loop)
            except Exception:
                pass

        _sink.reply_done = _reply_done
        return _sink
