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
"""MESSAGE A PEER: another backtalk agent, by its name in the config's
"peers" (see config.py).

  python -m backtalk.say <peer> "text"     wake it (if it has a "mac"),
                                           connect (retrying up to 90 s),
                                           send, print the reply's lines
                                           until its turn ends; nonzero
                                           exit on failure
  python -m backtalk.say --self-check      a peer, a relay, and a refusal,
                                           against bridges on this machine

Also the peer link (config peer_link): main.py runs link() so a peer that
can't dial this machine reaches it through its own bridge (web.py's PEERS).
"""
import asyncio
import json
import re
import socket
import sys
import time

import websockets

from backtalk.vlog import log

CONNECT_S = 90      # a sleeping machine needs this long to wake and link up
REPLY_S = 15 * 60   # a long task's reply
MAX = 16 * 2**20


def wake(mac: str, broadcast: str = "<broadcast>"):
    """A Wake-on-LAN magic packet."""
    pkt = b"\xff" * 6 + bytes.fromhex(re.sub(r"[^0-9a-fA-F]", "", mac)) * 16
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.sendto(pkt, (broadcast, 9))


def hello(me: str, token: str, **extra) -> str:
    return json.dumps({"type": "hello", "device_id": "peer-" + re.sub(r"\W+", "-", me.lower()),
                       "name": me, "token": token, **extra})


async def talk(url: str, text: str, token: str, me: str, via: str | None = None,
               connect_s: float = CONNECT_S, out=print) -> None:
    """Connect (retrying until connect_s), send text, out() each reply line
    until the turn ends."""
    deadline = time.monotonic() + connect_s
    while True:
        ws = None
        try:
            ws = await websockets.connect(url, max_size=MAX, open_timeout=5)
            await ws.send(hello(me, token, **({"to": via} if via else {})))
            while True:
                m = await asyncio.wait_for(ws.recv(), 10)
                if isinstance(m, str) and json.loads(m).get("type") == "peer_ok":
                    break
            break
        except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as e:
            if ws is not None:
                await ws.close()
            if getattr(getattr(e, "rcvd", None), "code", None) == 4401:
                raise PermissionError("refused: wrong peer token") from e
            if time.monotonic() > deadline:
                raise ConnectionError(f"no answer from {url} in {connect_s:.0f} s: {e!r}") from e
            await asyncio.sleep(3)
    try:
        await ws.send(json.dumps({"type": "text", "text": text}))
        end = time.monotonic() + REPLY_S
        while True:
            m = await asyncio.wait_for(ws.recv(), max(1, end - time.monotonic()))
            if isinstance(m, bytes):
                continue                       # the reply's audio, if it spoke
            d = json.loads(m)
            if d.get("type") == "line" and d.get("who") == "jarvis":
                out(d.get("text", ""))
            elif d.get("type") == "turn_done":
                return
    finally:
        await ws.close()


async def link(url: str, name: str, local_url: str, token: str, me: str):
    """Keep an outbound link to a peer's bridge open, forever: each relayed
    connection it carries gets its own connection to this machine's bridge
    (local_url), every JSON frame passed through both ways. Short pings, so
    a link killed by sleep is noticed and re-dialed within seconds of
    waking."""
    while True:
        local: dict = {}
        wait = 3
        try:
            async with websockets.connect(url, max_size=MAX, open_timeout=10,
                                          ping_interval=5, ping_timeout=5) as up:

                async def pump(rid, ws):
                    try:
                        async for m in ws:
                            if isinstance(m, str):
                                await up.send(json.dumps({"type": "relay", "rid": rid,
                                                          "frame": json.loads(m)}))
                    except websockets.exceptions.ConnectionClosed:
                        pass
                    finally:
                        if local.pop(rid, None) is not None:
                            try:
                                await up.send(json.dumps({"type": "relay", "rid": rid, "close": True}))
                            except websockets.exceptions.ConnectionClosed:
                                pass

                await up.send(hello(me, token, link=name))
                async for m in up:
                    d = json.loads(m) if isinstance(m, str) else {}
                    if d.get("type") == "peer_ok":
                        log(f"[peer] link up to {url} as {name}")
                    if d.get("type") != "relay":
                        continue
                    rid, ws = str(d.get("rid")), local.get(str(d.get("rid")))
                    if d.get("close"):
                        local.pop(rid, None)
                        if ws is not None:
                            await ws.close()
                    elif isinstance(d.get("frame"), dict):
                        if ws is None:
                            ws = local[rid] = await websockets.connect(local_url, max_size=MAX)
                            asyncio.ensure_future(pump(rid, ws))
                        await ws.send(json.dumps(d["frame"]))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if getattr(getattr(e, "rcvd", None), "code", None) == 4401:
                wait = 60                      # wrong token: don't hammer
            log(f"[peer] link to {url} down ({e!r}), retrying in {wait} s")
        finally:
            for ws in list(local.values()):
                await ws.close()
        await asyncio.sleep(wait)


def main():
    if sys.argv[1:] == ["--self-check"]:
        asyncio.run(_self_check())
        print("self-check passed")
        return
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    from backtalk.config import CFG, peer_token
    peer, text = sys.argv[1], sys.argv[2]
    p = (CFG.get("peers") or {}).get(peer)
    if not p:
        sys.exit(f"no peer {peer!r} in the config (peers: {', '.join(CFG.get('peers') or {}) or 'none'})")
    token = peer_token()
    if not token:
        sys.exit("no peer token (env JARVIS_PEER_TOKEN, or the keyring item service jarvis-peer user token)")
    if p.get("mac"):
        wake(p["mac"], p.get("broadcast", "<broadcast>"))
        print(f"[sent a wake packet to {peer}]", file=sys.stderr, flush=True)
    try:
        asyncio.run(talk(p["url"], text, token, CFG.get("peer_name") or CFG["name"],
                         via=p.get("via"), out=lambda t: print(t, flush=True)))
    except Exception as e:
        sys.exit(f"[{peer} not reached: {e}]")


async def _self_check():
    """Two bridges on loopback: A is "the Pi", B "the desktop" that links
    out to A. A direct peer turn on B, the same turn relayed through A,
    and a wrong token refused."""
    from backtalk.web import BrowserBridge

    def bridge(port):
        b = BrowserBridge({"host": "127.0.0.1", "port": port})
        b._token = "secret"

        def on_text(conn, t, spoken=False):
            s = b.make_sink(conn)
            s.send({"type": "line", "who": "jarvis", "text": f"{conn.peer} said {t}"})
            s.send({"type": "turn_done"})
        b.on_text = on_text
        asyncio.ensure_future(b.serve())
        return b

    pa, pb = 18792, 18793
    a, _ = bridge(pa), bridge(pb)
    await asyncio.sleep(0.5)
    got = []
    await talk(f"ws://127.0.0.1:{pb}", "hi", "secret", "Pi Jarvis", out=got.append)
    assert got == ["Pi Jarvis said hi"], got
    try:
        await talk(f"ws://127.0.0.1:{pb}", "hi", "wrong", "Pi Jarvis", out=got.append)
        raise AssertionError("a wrong token was let in")
    except PermissionError:
        pass
    lk = asyncio.ensure_future(link(f"ws://127.0.0.1:{pa}", "desktop", f"ws://127.0.0.1:{pb}",
                                    "secret", "Desktop Jarvis"))
    got = []
    await talk(f"ws://127.0.0.1:{pa}", "relayed", "secret", "Pi Jarvis", via="desktop",
               connect_s=20, out=got.append)
    assert got == ["Pi Jarvis said relayed"], got
    assert "desktop" in a.links
    lk.cancel()


if __name__ == "__main__":
    main()
