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
  text frame  {"type": "hello", "device_id": ...}  browser -> server,
                                             first message, sent right
                                             after connecting: the
                                             browser's own persistent
                                             token (see core.js's
                                             DEVICE_ID), used for
                                             Conn.id and per-device
                                             routing.
                                             An optional "model" (the
                                             app's phone model) names a
                                             new device; see saw().
  text frame  {"type": "press"}             browser -> server (queue)
  text frame  {"type": "interrupt_press"}   browser -> server (the
                                             Interrupt button: stop the
                                             current turn first, THEN
                                             record)
  text frame  {"type": "release"}           browser -> server
  binary      <uint32 LE rate><int16 LE PCM...>   either direction
  text frame  {"type": "reply_done"}        server -> browser
  text frame  {"type": "stop"}              server -> browser (an
                                             interrupt: discard every
                                             scheduled-but-unplayed
                                             chunk already sent)
  text frame  {"type": "unmute"}            browser -> server: a click
                                             while paused ("stop
                                             listening"); hands-free
                                             comes back
  text frame  {"type": "hands_free",        browser -> server: switch
               "on": b, "muted": b}          hands-free on, paused, or off
                                             for this tab, silently
  text frame  {"type": "image", "data": b64}  client -> server: a shared
                                             JPEG (the app's share
                                             target); acked with
                                             {"type": "image_ok"}, and
                                             {"type": "image_used"} once
                                             it rides along with a
                                             question
  text frame  {"type": "phone", "do": ...}  server -> client: a command for
                                             the device that asked (the
                                             agent's <<phone {...}>> tag;
                                             see mouth.py)
  text frame  {"type": "phone_result",      client -> server: what came of
               "text": ...}                  it, when the agent needs to
                                             know (a failure, a lookup);
                                             asked as that device's turn
  text frame  {"type": "listen", "on": b,   server -> browser: hands-
               "muted": b}
                                             free on or off for that
                                             tab. While on, the tab
                                             streams its mic (echo
                                             cancellation ON)
                                             continuously; frames
                                             outside a press feed
                                             ListenStream, and the voice
                                             line does the endpointing.

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
import base64
import json
import os
import queue
import struct
import time

import numpy as np
import websockets

from backtalk.vlog import log

RATE = 16000   # must match ears.RATE — the fixed transcribe() contract


class Conn:
    """One browser tab's live connection."""

    def __init__(self, ws, bridge: "BrowserBridge"):
        self.ws = ws
        self.bridge = bridge
        # Set from the client's own "hello" (a random token the browser
        # generates once and keeps in sessionStorage, see core.js's
        # DEVICE_ID), not assigned here. A server-assigned counter
        # resets on every reconnect, silently reassigning the tab
        # mid-session; the client's token survives reloads and drops.
        self.id = None
        self.disconnected = False
        self.recording = False   # this tab's own slot, not bridge-wide
        self.listening = False   # hands-free: frames outside a press
        self.listen_muted = False   # paused: hearing only "start listening"
        # Read from a worker thread (ListenStream), hence queue.Queue.
        # Bounded: a stalled listener drops audio, never memory.
        self._listen_q: "queue.Queue[bytes]" = queue.Queue(maxsize=500)
        self._frames: "asyncio.Queue" = asyncio.Queue()
        self._released = asyncio.Event()

    async def reader(self):
        """The only loop that ever reads this connection's socket."""
        try:
            async for msg in self.ws:
                if isinstance(msg, (bytes, bytearray)):
                    if self.recording:
                        await self._frames.put(bytes(msg))
                    elif self.listening:
                        try:
                            self._listen_q.put_nowait(bytes(msg))
                        except queue.Full:
                            pass
                    continue
                try:
                    data = json.loads(msg)
                except ValueError:
                    continue
                kind = data.get("type")
                if kind == "hello":
                    self.id = data.get("device_id") or self.id
                    self.bridge.saw(self, data.get("model"))
                    if self.id in self.bridge._hf_ids:
                        self.bridge.set_listening(self, True)
                elif kind == "press":
                    self.bridge._on_press(self, interrupt=False)
                elif kind == "interrupt_press":
                    self.bridge._on_press(self, interrupt=True)
                elif kind == "release":
                    self._released.set()
                elif kind == "unmute" and self.listen_muted:
                    self.bridge.set_listening(self, True)
                elif kind == "hands_free":
                    # a silent switch from the client's own controls (the
                    # app's taps, its assistant gesture): no spoken line;
                    # on + muted = paused
                    self.bridge.set_listening(self, bool(data.get("on")),
                                              muted=bool(data.get("muted")))
                elif kind == "phone_result" and self.bridge.on_phone_result is not None:
                    self.bridge.on_phone_result(self, str(data.get("text") or ""))
                elif kind == "image" and self.bridge.on_image is not None:
                    try:
                        self.bridge.on_image(self, base64.b64decode(data.get("data") or ""))
                        await self.ws.send(json.dumps({"type": "image_ok"}))
                    except Exception as e:
                        log(f"[web] shared image dropped: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.disconnected = True
            self._released.set()         # unblock a capture awaiting release
            await self._frames.put(None)  # unblock a capture awaiting a frame


class BrowserBridge:
    """websockets.serve() wrapper. Every connected tab gets its own
    capture slot (Conn.recording), so N tabs can each record
    independently: the guard in _on_press is per-connection, not
    bridge-wide. _conns tracks every live connection so a sink can
    broadcast to all of them (see make_broadcast_sink)."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._server = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._press_q: "asyncio.Queue[tuple[Conn, bool]]" = asyncio.Queue()
        self._conns: set[Conn] = set()
        # Devices in hands-free, by the browser's own token, so a reload
        # or reconnect comes back listening. Memory only: a voice-line
        # restart returns every browser to push-to-talk.
        self._hf_ids: set[str] = set()
        self.on_listen = None   # main.py: start a listener for a Conn
        self.on_image = None    # main.py: (Conn, jpeg bytes), a shared picture
        self.on_phone_result = None   # main.py: (Conn, text) from a phone command
        self.devices_file = None      # main.py: the device names (see saw)

    async def serve(self):
        self._loop = asyncio.get_running_loop()
        host = self._cfg.get("host", "127.0.0.1")
        port = self._cfg.get("port", 8792)

        async def handler(ws):
            conn = Conn(ws, self)
            self._conns.add(conn)
            try:
                await conn.reader()   # runs until this connection closes
            finally:
                self._conns.discard(conn)

        self._server = await websockets.serve(handler, host, port,
                                              max_size=16 * 2**20)   # shared pictures
        log(f"[web] browser bridge listening on ws://{host}:{port}")
        await self._server.wait_closed()

    def _on_press(self, conn: Conn, interrupt: bool = False):
        """Called synchronously from within conn.reader(), already on
        the event loop — no thread hop needed here. `interrupt` is the
        Interrupt button (stop the current turn, then record) versus a
        plain tap (queue behind it, touch nothing). This guard is about
        overlapping RECORDINGS on ONE connection, not overlapping
        turns or other tabs; a press queues fine the instant THIS
        tab's previous recording ends, and a different tab's press is
        never affected by it at all."""
        if conn.recording:
            log("[web] press ignored, this tab is already recording")
            return
        conn.recording = True
        conn._released.clear()
        while not conn._frames.empty():
            conn._frames.get_nowait()   # drop anything stale from before this press
        self._press_q.put_nowait((conn, interrupt))

    def set_listening(self, conn: Conn, on: bool, muted: bool = False):
        """Hands-free on or off for one tab. Call on the event loop.
        on + muted is paused ("stop listening"): the tab keeps
        streaming, but main.py acts on nothing except "start
        listening", and a click on the face also resumes."""
        conn.listening, conn.listen_muted = on, muted and on
        conn.active = time.monotonic()      # the hands-free idle clock
        if conn.id:
            if on:
                self._hf_ids.add(conn.id)
            else:
                self._hf_ids.discard(conn.id)
        if not on:
            while not conn._listen_q.empty():
                conn._listen_q.get_nowait()
        asyncio.ensure_future(
            conn.ws.send(json.dumps({"type": "listen", "on": on,
                                     "muted": conn.listen_muted})))
        log(f"[web] hands-free {'off' if not on else 'paused' if muted else 'on'}"
            f" for {str(conn.id)[:8]}")
        if on and self.on_listen is not None:
            self.on_listen(conn)

    # ---- device names: "the kitchen" is the Echo Show. A JSON file of
    # {device_id: {"name", "model", "seen"}}; a new device is named after
    # its model (or "browser xxxx"), and the agent renames one by editing
    # the file. Read fresh each time, so its edits apply at once.

    def devices(self) -> dict:
        try:
            with open(self.devices_file) as f:
                return json.load(f)
        except (OSError, TypeError, ValueError):
            return {}

    def saw(self, conn: Conn, model: str | None):
        if not (self.devices_file and conn.id):
            return
        devs = self.devices()
        d = devs.setdefault(conn.id, {"name": model or f"browser {conn.id[:4]}"})
        if model:
            d["model"] = model
        d["seen"] = time.strftime("%Y-%m-%d %H:%M")
        tmp = self.devices_file + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(devs, f, indent=1)
            os.replace(tmp, self.devices_file)
        except OSError as e:
            log(f"[web] device list not saved: {e}")

    def name_of(self, conn_id) -> str | None:
        return self.devices().get(conn_id, {}).get("name") if conn_id else None

    def find(self, name: str) -> Conn | None:
        """The live connection of the device with this name (any case)."""
        ids = {i for i, d in self.devices().items()
               if str(d.get("name", "")).casefold() == name.strip().casefold()}
        return next((c for c in list(self._conns)
                     if c.id in ids and not c.disconnected), None)

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def wait_press(self) -> tuple[Conn, bool]:
        return await self._press_q.get()

    async def record_until_release(self, conn: Conn, abort=None,
                                   on_audio=None) -> np.ndarray | None:
        """Collect PCM frames until release, disconnect, or abort().
        `on_audio`, if given, also gets each 16 kHz chunk as it arrives
        (a streaming transcriber; see ears.Session).
        Resamples any frame whose declared rate isn't already 16000
        (defense against browser/engine quirks) via stdlib audioop —
        no new dependency. Returns None on no audio at all."""
        frames: list[np.ndarray] = []
        # audioop.ratecv's filter memory, carried across chunks. Passing
        # None on every chunk instead of this would reset the resample
        # filter at each chunk boundary, glitching audio once per chunk
        # whenever the browser's actual mic rate isn't exactly 16000
        # (common: many browsers ignore the AudioContext sampleRate
        # request).
        resample_state = None
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
                    pcm_bytes, resample_state = audioop.ratecv(
                        pcm_bytes, 2, 1, rate, RATE, resample_state)
                frames.append(np.frombuffer(pcm_bytes, dtype=np.int16))
                if on_audio is not None:
                    on_audio(frames[-1])
        finally:
            conn.recording = False
        if not frames:
            return None
        return np.concatenate(frames)

    def make_sink(self, conn: Conn):
        """Returns a (rate, pcm) -> None closure for mouth.py's _write
        to call per audio block. Must never let a dead browser socket
        affect local playback — every failure is swallowed here, not
        left for the caller to handle."""
        loop = self._loop

        def live():
            """The connection to send to NOW: this one, or, if its tab was
            reloaded, the new connection with the same device id, so a
            reply carries on mid-sentence instead of going to a dead one."""
            if not conn.disconnected:
                return conn
            for c in list(self._conns):
                if c.id is not None and c.id == conn.id and not c.disconnected:
                    return c
            return None

        def _sink(rate: int, pcm: np.ndarray):
            c = live()
            if c is None or loop is None:
                return
            try:
                frame = struct.pack("<I", rate) + pcm.tobytes()
                asyncio.run_coroutine_threadsafe(c.ws.send(frame), loop)
            except Exception:
                pass

        def _send(obj: dict):
            c = live()
            if c is None or loop is None:
                return
            try:
                asyncio.run_coroutine_threadsafe(c.ws.send(json.dumps(obj)), loop)
            except Exception:
                pass

        def _reply_done():
            _send({"type": "reply_done"})

        def _stop():
            """An interrupt: tell the browser to discard every chunk
            already sent but not yet played — mirrors mouth.shut_up()
            on this end."""
            _send({"type": "stop"})

        _sink.reply_done = _reply_done
        _sink.stop = _stop
        _sink.send = _send
        _sink.conn_id = conn.id
        _sink.conn = conn
        # Live lookup, not a snapshot: a sink can sit queued in
        # brain.py's _ask_queue for a while before its turn starts,
        # and the tab it belongs to may have long since closed by
        # then; see is_live's use there.
        _sink.is_live = lambda: live() is not None
        return _sink

    def make_broadcast_sink(self):
        """A (rate, pcm) -> None closure for turns nobody specific
        asked for (background reports) or where there's no one
        connection to prefer: sends to every tab connected AT SEND
        TIME, built fresh from self._conns each call so a tab
        connecting or disconnecting mid-reply needs no bookkeeping
        here."""
        def _sink(rate: int, pcm: np.ndarray):
            for conn in list(self._conns):
                self.make_sink(conn)(rate, pcm)

        def _reply_done():
            for conn in list(self._conns):
                self.make_sink(conn).reply_done()

        def _stop():
            for conn in list(self._conns):
                self.make_sink(conn).stop()

        _sink.reply_done = _reply_done
        _sink.stop = _stop
        return _sink


class ListenStream:
    """One tab's hands-free audio as a mic-shaped stream for
    ears.Ears.listen_once(stream=...): read(n) blocks for n samples at
    RATE, resampling whatever rate the browser actually captures at.
    A gap in the audio (network stall) reads as silence after 100ms,
    so the caller's abort check keeps getting a turn."""

    def __init__(self, conn: Conn):
        self.conn = conn
        self._buf = np.zeros(0, dtype=np.int16)
        self._state = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n: int):
        while len(self._buf) < n:
            try:
                chunk = self.conn._listen_q.get(timeout=0.1)
            except queue.Empty:
                # A gap: hand back what's here padded with silence. Waiting
                # for the rest froze the listener mid-utterance when the tab
                # stopped streaming (switched to push to talk), so its
                # capturing flag (the listening rings, the quiet-restart
                # check) never cleared, and abort was never checked.
                out = np.zeros(n, dtype=np.int16)
                out[:len(self._buf)] = self._buf
                self._buf = self._buf[:0]
                return out.reshape(-1, 1), False
            if len(chunk) < 4:
                continue
            rate = struct.unpack_from("<I", chunk, 0)[0]
            pcm = chunk[4:]
            if rate != RATE and rate > 0:
                pcm, self._state = audioop.ratecv(pcm, 2, 1, rate, RATE,
                                                  self._state)
            self._buf = np.concatenate(
                [self._buf, np.frombuffer(pcm, dtype=np.int16)])
        out, self._buf = self._buf[:n], self._buf[n:]
        return out.reshape(-1, 1), False
