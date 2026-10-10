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
  text frame  {"type": "release"}           browser -> server (with
                                             "text": the phone's own
                                             transcript of the press, used
                                             instead of transcribing the
                                             audio, which still streams as
                                             the fallback and voiceprint)
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
  text frame  {"type": "file", "name": ...,   client -> server: any other
               "mime": ..., "data": b64}       shared file (share sheet or
                                             the terminal's +); acked with
                                             {"type": "file_ok", "name"},
                                             then rides along with the
                                             next question like an image
  text frame  {"type": "phone", "do": ...}  server -> client: a command for
                                             the device that asked (the
                                             agent's <<phone {...}>> tag;
                                             see mouth.py)
  text frame  {"type": "phone_result",      client -> server: what came of
               "text": ...}                  it, when the agent needs to
                                             know (a failure, a lookup);
                                             asked as that device's turn
  text frame  {"type": "text", "text": ...}  client -> server: a typed
                                             question (the face's
                                             terminal), queued like a tap;
                                             with "spoken": true, words the
                                             phone transcribed itself (on-
                                             device push to talk): treated
                                             as speech, not typing
  text frame  {"type": "line", "who":       server -> client: the
               "you"|"jarvis", "text": ...}  conversation as text, this
                                             device's turns only;
                                             {"type": "lines", "lines":
                                             [...]} replays the last 60
                                             on hello. A line with
                                             "show": {title, path} is a
                                             card's reopen button; the
                                             client sends {"type":
                                             "reopen", "path": ...} and
                                             gets the file's current
                                             contents as a "show"
  text frame  {"type": "call", "on": b,     server -> client: an intercom
               "with": name}                 call started or ended. While
                                             on, the client streams its mic
                                             (call-mode echo cancellation)
                                             and plays what arrives; the
                                             bridge relays the frames to
                                             the other end, untranscribed
  text frame  {"type": "app_update",        server -> client: a newer app
               "code": N}                    waits (the hello's "version" was
                                             older); {"type": "app_get"} back
                                             asks for it, and it comes as
                                             {"type": "app_chunk", "data":
                                             base64, "last": b} frames
  text frame  {"type": "hangup"}            client -> server: end the call
                                             (so does either end
                                             disconnecting, or a press on a
                                             device no session owns; on an
                                             owned one a press talks to its
                                             agent, muted to the far end)
  text frame  {"type": "turn_done"}         server -> client: the agent's
                                             turn for this device is over
                                             (sent even for a text-only
                                             reply, which has no reply_done)
  PEERS (another agent; see say.py): a hello with "token" (the shared
  peer secret) and "name" makes the connection a peer, answered with
  {"type": "peer_ok"}; a wrong token closes it, and so does a missing
  one from off this machine when web.require_token is on. A peer hello
  with "link": name is that machine's outbound link (config peer_link);
  one with "to": name is relayed down that link whole, every JSON frame
  each way wrapped as {"type": "relay", "rid": ..., "frame": {...}}
  ({"close": true} when either end goes).
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
  text frame  {"type": "stt", "on": b}      client -> server: the phone's
                                             own recognizer is (or isn't)
                                             hearing its hands-free stream
  text frame  {"type": "heard", "text": ...}  client -> server: one
                                             hands-free utterance as the
                                             phone transcribed it (its
                                             recognizer endpoints); the
                                             server's own transcript of the
                                             same speech is then dropped,
                                             and used only when the phone
                                             had nothing

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
import collections
import hmac
import ipaddress
import json
import os
import queue
import struct
import time

import numpy as np
import websockets

from backtalk.vlog import log

RATE = 16000   # must match ears.RATE — the fixed transcribe() contract
CALL_VOICE_RMS = 500   # a caller frame this loud opens the call at the far end
CALL_RING_S = 20       # the caller has this long to start talking


def chime(rate: int) -> np.ndarray:
    """Two soft rising tones and a beat of silence: an announcement or a
    call is coming."""
    t = np.arange(int(rate * 0.14)) / rate
    env = np.sin(np.pi * t / t[-1])
    c = np.concatenate([np.sin(2 * np.pi * f * t) * env for f in (880, 1320)]
                       + [np.zeros(int(rate * 0.2))])
    return (c * 9000).astype(np.int16)


def beep(rate: int) -> np.ndarray:
    """One short soft tone: a tapped push to talk is listening."""
    t = np.arange(int(rate * 0.12)) / rate
    return (np.sin(2 * np.pi * 660 * t) * np.sin(np.pi * t / t[-1]) * 7000).astype(np.int16)


class Call:
    """An intercom call. The caller speaks first: their frames are held
    until one is loud enough, then the far end gets a chime, the call
    frame (its mic opens), and the held audio, and after that every frame
    goes straight across both ways."""

    def __init__(self, caller: "Conn", callee: "Conn"):
        self.caller, self.callee = caller, callee
        self.open = False
        self.held: collections.deque = collections.deque(maxlen=12)

    def peer(self, conn: "Conn") -> "Conn":
        return self.callee if conn is self.caller else self.caller


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
        self.peer = None      # a peer agent's name, once its hello's token checks out
        self.via = None       # relayed whole down this peer link (a hello's "to")
        self.rid = None       # ...under this relay id
        self.peer_link = None # the name this peer's outbound link serves (a hello's "link")
        self.authed = True    # False: refused unless the first frame is a peer hello
        self.disconnected = False
        self.recording = False   # this tab's own slot, not bridge-wide
        self.listening = False   # hands-free: frames outside a press
        self.listen_muted = False   # paused: hearing only "start listening"
        self.call: Call | None = None   # an intercom call: audio relays, untranscribed
        # Agent speech for this device while it's on a call, at RATE, waiting
        # to be mixed into the far end's audio (Bridge._mix); see make_sink.
        self.overlay: "collections.deque[np.ndarray]" = collections.deque()
        self.relayed_at = 0.0   # when call audio last went to this device
        # Read from a worker thread (ListenStream), hence queue.Queue.
        # Bounded: a stalled listener drops audio, never memory.
        self._listen_q: "queue.Queue[tuple[float, bytes]]" = queue.Queue(maxsize=500)
        self.phone_stt = False   # the phone's recognizer hears the hands-free stream
        self.heard_at = 0.0      # when its last words ("heard") arrived
        # The last few seconds of hands-free audio as it arrived, (time,
        # frame): the voiceprint for the phone's words.
        self.recent: "collections.deque[tuple[float, bytes]]" = collections.deque(maxlen=100)
        self._frames: "asyncio.Queue" = asyncio.Queue()
        self._released = asyncio.Event()

    def heard_pcm(self, s: float = 6.0):
        """The audio before the phone's last words, for their voiceprint.
        ponytail: a fixed window (the phone gives no timings), so it can
        hold silence or the end of earlier speech; fine for spotting a voice."""
        pcm = [f[4:] for t, f in list(self.recent)
               if self.heard_at - s <= t <= self.heard_at
               and len(f) > 4 and struct.unpack_from("<I", f, 0)[0] == 16000]
        return np.frombuffer(b"".join(pcm), dtype=np.int16) if pcm else None

    async def reader(self):
        """The only loop that ever reads this connection's socket."""
        try:
            async for msg in self.ws:
                if isinstance(msg, (bytes, bytearray)) and not self.authed:
                    await self.ws.close(4401, "token required")
                    return
                if isinstance(msg, (bytes, bytearray)):
                    # A held press wins over a call (talking to the agent
                    # on a device a session owns; muted to the far end).
                    if self.call is not None and not self.recording:
                        self.bridge._call_audio(self, bytes(msg))
                    elif self.recording:
                        await self._frames.put(bytes(msg))
                    elif self.listening:
                        item = (time.monotonic(), bytes(msg))
                        if self.phone_stt:
                            self.recent.append(item)
                        try:
                            self._listen_q.put_nowait(item)
                        except queue.Full:
                            pass
                    continue
                try:
                    data = json.loads(msg)
                except ValueError:
                    continue
                kind = data.get("type")
                if not self.authed and not (kind == "hello" and data.get("token")):
                    log(f"[peer] refused {self.ws.remote_address[0]}: no token")
                    await self.ws.close(4401, "token required")
                    return
                if self.via:
                    self.bridge.relay_up(self, data)
                    continue
                if kind == "relay" and self.bridge.links.get(self.peer_link) is self:
                    self.bridge.relay_down(data)
                    continue
                if kind == "hello" and data.get("token") is not None:
                    if not self.bridge.check_token(data.get("token")):
                        log(f"[peer] refused {self.ws.remote_address[0]}: wrong token")
                        await self.ws.close(4401, "wrong token")
                        return
                    self.authed = True
                    self.peer = str(data.get("name") or "peer")[:40]
                    if data.get("to"):
                        if not self.bridge.relay_open(self, str(data["to"]), data):
                            await self.ws.close(4404, "no link")
                            return
                        continue
                    if data.get("link"):
                        self.peer_link = str(data["link"])
                        self.bridge.link_up(self)
                    log(f"[peer] {self.peer} connected"
                        + (f" (link for {self.peer_link})" if self.peer_link else ""))
                    await self.ws.send(json.dumps({"type": "peer_ok"}))
                    if self.peer_link:
                        continue
                if kind == "hello":
                    self.id = data.get("device_id") or self.id
                    self.bridge.saw(self, self.peer or data.get("model"))
                    self.bridge.offer_update(self, data.get("version"))
                    if self.bridge._lines.get(self.id):
                        await self.ws.send(json.dumps(
                            {"type": "lines", "lines": list(self.bridge._lines[self.id])}))
                    # Hands-free survives a reconnect; anything else is told
                    # it's off, or a client still in hands-free from before a
                    # restart streams to a server that drops it all.
                    if self.id in self.bridge._hf_ids:
                        self.bridge.set_listening(self, True, muted=self.bridge._hf_ids[self.id])
                    else:
                        await self.ws.send(json.dumps({"type": "listen", "on": False, "muted": False}))
                elif kind == "text" and self.bridge.on_text is not None:
                    t = str(data.get("text") or "").strip()[:4000]
                    if t:
                        # "spoken": the phone transcribed it itself (on-device
                        # push to talk), so it's speech, not typing
                        self.bridge.on_text(self, t, bool(data.get("spoken")))
                    if self.id in self.bridge._hf_ids:
                        self.bridge.set_listening(self, True)
                elif kind == "stt":
                    self.phone_stt = bool(data.get("on"))
                    log(f"[web] phone recognizer {'on' if self.phone_stt else 'off'}"
                        f" for {str(self.id)[:8]}")
                elif kind == "heard" and self.bridge.on_heard is not None:
                    t = str(data.get("text") or "").strip()[:4000]
                    if t:
                        self.heard_at = time.monotonic()
                        self.bridge.on_heard(self, t)
                elif kind == "reopen" and self.bridge.on_reopen is not None:
                    self.bridge.on_reopen(self, str(data.get("path") or ""))
                elif kind == "app_status":
                    # Android's installer on a self-update (0 installed, -1
                    # waiting for the person, 3 cancelled, others failed)
                    log(f"[app] {self.bridge.name_of(self.id)} install status "
                        f"{data.get('code')} {str(data.get('message') or '')[:200]}")
                elif kind == "app_get":
                    asyncio.ensure_future(self.bridge.send_update(self))
                elif kind == "hangup" and self.call:
                    self.bridge.end_call(self.call, "hung up")
                elif kind in ("press", "interrupt_press") and self.call \
                        and self.id not in self.bridge.owned:
                    # no session owns this device: a tap on the orb hangs up
                    self.bridge.end_call(self.call, "hung up")
                elif kind == "press":
                    self.bridge._on_press(self, interrupt=False)
                elif kind == "interrupt_press":
                    self.bridge._on_press(self, interrupt=True)
                elif kind == "release":
                    # the phone's own transcript of this press, if it made
                    # one (on-device recognition fed the same audio)
                    self.release_text = str(data.get("text") or "").strip() or None
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
                elif kind == "file" and self.bridge.on_file is not None:
                    try:
                        name = str(data.get("name") or "file")
                        self.bridge.on_file(self, name, str(data.get("mime") or ""),
                                            base64.b64decode(data.get("data") or ""))
                        await self.ws.send(json.dumps({"type": "file_ok", "name": name}))
                    except Exception as e:
                        log(f"[web] shared file dropped: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.disconnected = True
            self.bridge.peer_gone(self)
            if self.call is not None:
                self.bridge.end_call(self.call, "disconnected")
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
        self._hf_ids: dict[str, bool] = {}     # ...and whether paused
        self.on_listen = None   # main.py: start a listener for a Conn
        self.on_image = None    # main.py: (Conn, jpeg bytes), a shared picture
        self.on_file = None     # main.py: (Conn, name, mime, bytes), any other shared file
        self.on_phone_result = None   # main.py: (Conn, text) from a phone command
        self.devices_file = None      # main.py: the device names (see saw)
        self.update_dir = None        # main.py: where a newer app waits (see offer_update)
        self.on_text = None           # main.py: (Conn, text) typed in the terminal
        self.on_heard = None          # main.py: (Conn, text) hands-free words from the phone
        self.on_reopen = None         # main.py: (Conn, path) a card's reopen button
        # Devices a dedicated session owns (main.py's sessions, by id): the
        # main session's broadcasts skip them, so its replies and stops
        # never land on, say, the kitchen mid-recipe.
        self.owned: dict = {}
        # Per device id: until when its speaker is busy with something
        # that isn't an announcement (make_sink). Announcements wait for a
        # lull and pause when it gets busy again (main.py).
        self.voice_until: dict[str, float] = {}
        # Each device's recent conversation lines, replayed on hello so a
        # reload keeps its terminal. Memory only.
        self._lines: dict[str, collections.deque] = {}
        self._token = None            # the peer secret, read once (see check_token)
        self.links: dict[str, Conn] = {}    # peer links, by the name they serve
        self._relays: dict[str, Conn] = {}  # relayed connections, by rid

    # ---- peers (see the docstring's PEERS)

    def check_token(self, tok) -> bool:
        if self._token is None:
            from backtalk.config import peer_token
            self._token = peer_token()
        return bool(self._token) and hmac.compare_digest(str(tok), self._token)

    def link_up(self, conn: Conn):
        old = self.links.get(conn.peer_link)
        self.links[conn.peer_link] = conn
        if old is not None and old is not conn:
            asyncio.ensure_future(old.ws.close())   # a stale link from before a sleep

    def relay_open(self, conn: Conn, to: str, hello: dict) -> bool:
        if to not in self.links:
            log(f"[peer] {conn.peer} asked for {to}, which has no link up")
            return False
        conn.via, conn.rid = to, f"{id(conn):x}"
        self._relays[conn.rid] = conn
        log(f"[peer] {conn.peer} relayed to {to}")
        self.relay_up(conn, {k: v for k, v in hello.items() if k != "to"})
        return True

    def relay_up(self, conn: Conn, frame: dict | None):
        link = self.links.get(conn.via)
        if link is None:
            asyncio.ensure_future(conn.ws.close(4404, "link gone"))
            return
        self._send(link, {"type": "relay", "rid": conn.rid}
                   | ({"frame": frame} if frame is not None else {"close": True}))

    def relay_down(self, data: dict):
        conn = self._relays.get(str(data.get("rid")))
        if conn is None:
            return
        if data.get("close"):
            asyncio.ensure_future(conn.ws.close())
        elif isinstance(data.get("frame"), dict):
            self._send(conn, data["frame"])

    def peer_gone(self, conn: Conn):
        if conn.via and self._relays.pop(conn.rid, None):
            self.relay_up(conn, None)
        if conn.peer_link and self.links.get(conn.peer_link) is conn:
            del self.links[conn.peer_link]
            log(f"[peer] link for {conn.peer_link} down")
            for c in [c for c in self._relays.values() if c.via == conn.peer_link]:
                asyncio.ensure_future(c.ws.close(4404, "link gone"))

    async def serve(self):
        self._loop = asyncio.get_running_loop()
        host = self._cfg.get("host", "127.0.0.1")
        port = self._cfg.get("port", 8792)

        async def handler(ws):
            conn = Conn(ws, self)
            # off this machine, with require_token on: a peer, or nothing
            addr = ipaddress.ip_address(ws.remote_address[0])
            local = addr.is_loopback or bool(getattr(addr, "ipv4_mapped", None)
                                             and addr.ipv4_mapped.is_loopback)
            conn.authed = local or not self._cfg.get("require_token")
            self._conns.add(conn)
            try:
                await conn.reader()   # runs until this connection closes
            finally:
                self._conns.discard(conn)

        self._server = await websockets.serve(handler, host, port,
                                              max_size=48 * 2**20)   # shared pictures and files (~35 MB after base64)
        log(f"[web] browser bridge listening on ws://{host}:{port}")
        asyncio.ensure_future(self._update_watch())
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
        conn.release_text = None
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
                self._hf_ids[conn.id] = conn.listen_muted
            else:
                self._hf_ids.pop(conn.id, None)
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
            if d.get("name") == f"browser {conn.id[:4]}":
                d["name"] = model     # first seen on an app too old to say
        d["seen"] = time.strftime("%Y-%m-%d %H:%M")
        tmp = self.devices_file + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(devs, f, indent=1)
            os.replace(tmp, self.devices_file)
        except OSError as e:
            log(f"[web] device list not saved: {e}")

    # ---- app updates: the newest Android app waits in update_dir as
    # app.apk plus version.json {"code": N}. A client whose hello says an
    # older "version" is offered it; on "app_get" it's sent down the same
    # socket in base64 chunks ({"type": "app_chunk", "data", "last"}) and
    # the app installs it. No adb, no cable, no store.

    def _latest(self) -> int:
        try:
            with open(os.path.join(self.update_dir, "version.json")) as f:
                return int(json.load(f)["code"])
        except (OSError, TypeError, ValueError, KeyError):
            return 0

    def offer_update(self, conn: Conn, version):
        """Offer the newer build, once per build per connection (declining
        isn't nagged; a reconnect asks again)."""
        if version is not None:
            conn.version = int(version)
        if not self.update_dir or getattr(conn, "version", None) is None:
            return
        latest = self._latest()
        if latest > conn.version and getattr(conn, "offered", 0) < latest:
            conn.offered = latest
            self._send(conn, {"type": "app_update", "code": latest})
            log(f"[app] {self.name_of(conn.id)} has version {conn.version}, offered {latest}")

    async def _update_watch(self):
        """A device that stays connected for days (the kitchen) hears about
        a newly parked build too, within ten minutes."""
        while True:
            await asyncio.sleep(600)
            for c in list(self._conns):
                if not c.disconnected:
                    self.offer_update(c, None)

    async def send_update(self, conn: Conn, chunk: int = 256 * 1024):
        try:
            with open(os.path.join(self.update_dir, "app.apk"), "rb") as f:
                apk = f.read()
        except (OSError, TypeError) as e:
            log(f"[app] no update to send: {e}")
            return
        for i in range(0, len(apk), chunk):
            last = i + chunk >= len(apk)
            await conn.ws.send(json.dumps({"type": "app_chunk", "last": last,
                                           "data": base64.b64encode(apk[i:i + chunk]).decode()}))
        log(f"[app] sent {len(apk) // 1024} KB to {self.name_of(conn.id)}")

    def name_of(self, conn_id) -> str | None:
        return self.devices().get(conn_id, {}).get("name") if conn_id else None

    def find(self, name: str) -> Conn | None:
        """The live connection of the device with this name (any case)."""
        ids = {i for i, d in self.devices().items()
               if str(d.get("name", "")).casefold() == name.strip().casefold()}
        return next((c for c in list(self._conns)
                     if c.id in ids and not c.disconnected), None)

    # ---- intercom (see Call). start_call and end_call run on the loop.

    def start_call(self, caller: Conn, callee: Conn):
        if caller is callee:
            raise ValueError("that's the device asking")
        busy = [c for c in (caller, callee) if c.call is not None]
        if busy:
            raise RuntimeError(f"{self.name_of(busy[0].id)} is already on a call")
        call = Call(caller, callee)
        caller.call = callee.call = call
        self._send(caller, {"type": "call", "on": True, "with": self.name_of(callee.id)})
        self._loop.call_later(CALL_RING_S, lambda: call.open or self.end_call(call, "nobody spoke"))
        log(f"[call] {self.name_of(caller.id)} -> {self.name_of(callee.id)}: waiting for the caller to speak")

    def end_call(self, call: Call, why: str):
        if call.caller.call is not call:
            return                          # already over
        for c in (call.caller, call.callee):
            c.call = None
            if c is call.caller or call.open:
                self._send(c, {"type": "call", "on": False})
        log(f"[call] {self.name_of(call.caller.id)} -> {self.name_of(call.callee.id)} ended: {why}")

    def _call_audio(self, conn: Conn, frame: bytes):
        call = conn.call
        if call.open:
            peer = call.peer(conn)
            peer.relayed_at = time.monotonic()
            self._send(peer, self._mix(peer, frame))
            return
        if conn is not call.caller or len(frame) < 6:
            return                          # the far end isn't on yet
        call.held.append(frame)
        pcm = np.frombuffer(frame[4:len(frame) - (len(frame) - 4) % 2], dtype=np.int16)
        if np.sqrt(np.mean(pcm.astype(np.float32) ** 2)) < CALL_VOICE_RMS:
            return
        call.open = True
        rate = struct.unpack_from("<I", frame, 0)[0]
        self._send(call.callee, struct.pack("<I", rate) + chime(rate).tobytes())
        self._send(call.callee, {"type": "call", "on": True, "with": self.name_of(conn.id)})
        call.callee.relayed_at = time.monotonic()
        while call.held:
            self._send(call.callee, call.held.popleft())
        log(f"[call] {self.name_of(conn.id)} -> {self.name_of(call.callee.id)}: connected")

    # ---- the agent talking over a call: its speech for a device on a
    # call is queued (Conn.overlay) and mixed into the far end's frames,
    # the call ducked under it, so a cooking step plays without dropping
    # the call. When the far end sends nothing (a browser taking turns),
    # _drain plays the speech on its own.

    def _overlay_take(self, conn: Conn, n: int) -> np.ndarray:
        out, got = [], 0
        while conn.overlay and got < n:
            a = conn.overlay.popleft()
            if got + len(a) > n:
                conn.overlay.appendleft(a[n - got:])
                a = a[:n - got]
            out.append(a)
            got += len(a)
        return np.concatenate(out) if out else np.zeros(0, np.int16)

    def _mix(self, conn: Conn, frame: bytes) -> bytes:
        if not conn.overlay or len(frame) < 6:
            return frame
        rate = struct.unpack_from("<I", frame, 0)[0] or RATE
        call = np.frombuffer(frame[4:len(frame) - (len(frame) - 4) % 2], dtype=np.int16)
        take = self._overlay_take(conn, max(1, round(len(call) * RATE / rate)))
        if len(take) != len(call):      # to the call's rate and length
            take = np.interp(np.linspace(0, len(take) - 1, len(call)),
                             np.arange(len(take)), take)
        mixed = np.clip(call * 0.35 + take, -32768, 32767).astype(np.int16)
        return struct.pack("<I", rate) + mixed.tobytes()

    def overlay_add(self, conn: Conn, rate: int, pcm: np.ndarray):
        """Speech for a device on a call (from any thread)."""
        if rate != RATE:
            pcm = np.interp(np.linspace(0, len(pcm) - 1, round(len(pcm) * RATE / rate)),
                            np.arange(len(pcm)), pcm).astype(np.int16)
        conn.overlay.append(pcm)
        if not getattr(conn, "draining", False):
            conn.draining = True
            asyncio.run_coroutine_threadsafe(self._drain(conn), self._loop)

    async def _drain(self, conn: Conn):
        block = RATE // 10
        try:
            while conn.overlay and not conn.disconnected:
                if time.monotonic() - conn.relayed_at > 0.25:
                    pcm = self._overlay_take(conn, block)
                    await conn.ws.send(struct.pack("<I", RATE) + pcm.tobytes())
                await asyncio.sleep(0.1)
        except Exception:
            conn.overlay.clear()
        finally:
            conn.draining = False

    def _send(self, conn: Conn, obj):
        """One frame (bytes) or message (dict) to a connection, from the loop."""
        data = obj if isinstance(obj, bytes) else json.dumps(obj)

        async def send():
            try:
                await conn.ws.send(data)
            except Exception:
                pass                        # gone; its reader ends the call
        asyncio.ensure_future(send())

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

    def make_sink(self, conn: Conn, low: bool = False):
        """Returns a (rate, pcm) -> None closure for mouth.py's _write
        to call per audio block. `low`: an announcement's audio, which
        yields to everything else; any other audio marks the device as
        busy talking (voice_until) so announcements wait for a lull. Must never let a dead browser socket
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
            if not low and conn.id:
                self.voice_until[conn.id] = time.monotonic() + len(pcm) / rate + 0.3
            if c.call is not None and c.call.open:
                self.overlay_add(c, rate, pcm)     # mixed into the call
                return
            try:
                frame = struct.pack("<I", rate) + pcm.tobytes()
                asyncio.run_coroutine_threadsafe(c.ws.send(frame), loop)
            except Exception:
                pass

        def _send(obj: dict):
            if obj.get("type") == "line" and conn.id:
                self._lines.setdefault(conn.id, collections.deque(maxlen=60)).append(obj)
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
            c = live()
            if c is not None and c.call is not None:
                c.overlay.clear()       # never "stop" a call's own audio
                return
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
        def conns():
            return [c for c in list(self._conns) if c.id not in self.owned and c.peer is None]

        def _sink(rate: int, pcm: np.ndarray):
            for conn in conns():
                self.make_sink(conn)(rate, pcm)

        def _reply_done():
            for conn in conns():
                self.make_sink(conn).reply_done()

        def _stop():
            for conn in conns():
                self.make_sink(conn).stop()

        def _send(obj: dict):
            for conn in conns():
                self.make_sink(conn).send(obj)

        _sink.reply_done = _reply_done
        _sink.stop = _stop
        _sink.send = _send
        return _sink


class ListenStream:
    """One tab's hands-free audio as a mic-shaped stream for
    ears.Ears.listen_once(stream=...): read(n) blocks for n samples at
    RATE, resampling whatever rate the browser actually captures at.
    A gap in the audio (network stall) reads as silence after 100ms,
    so the caller's abort check keeps getting a turn. `t` is when the
    newest audio read so far arrived (the listener can run behind)."""

    def __init__(self, conn: Conn):
        self.conn = conn
        self._buf = np.zeros(0, dtype=np.int16)
        self._state = None
        self.t = time.monotonic()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n: int):
        while len(self._buf) < n:
            try:
                self.t, chunk = self.conn._listen_q.get(timeout=0.1)
            except queue.Empty:
                self.t = time.monotonic()
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
