# Run: .venv/bin/python tests/check_capture_timeout.py
# Self-check: a capture (warmup ping, console command) on a slow backend that
# only thinks and never finishes is interrupted at capture_timeout_s, its
# leftovers are never spoken, an ask made meanwhile waits for it and then gets
# its own reply, the face shows thinking while it runs, and interrupt() reaches
# a capture that has streamed nothing but thinking.
import asyncio, sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
from claude_agent_sdk import ResultMessage
from claude_agent_sdk.types import StreamEvent
from backtalk import brain as B

class Sink:
    def __init__(s): s.got, s.conn_id = [], "sir"
    def send(s, obj):
        if obj.get("type") == "line": s.got.append(obj["text"])
    def is_live(s): return True
class Bus:
    def __init__(s): s.states = []
    def set_state(s, n): s.states.append(n)
    def __getattr__(s, n): return lambda *a, **k: None
class Mouth:
    speaking = False
    def say_chunk(s, text, pending, sink): pass
    def __getattr__(s, n): return lambda *a, **k: None

def ev(delta): return StreamEvent(uuid="u", session_id="s", event={"type": "content_block_delta", "delta": delta})
def think(): return ev({"type": "thinking_delta", "thinking": "hmm "})
def text(t): return ev({"type": "text_delta", "text": t})
def res(): return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1, session_id="s")

class Client:
    """A slow backend: each query thinks forever, a few tokens a tick; only
    an interrupt ends it (with a ResultMessage, as the CLI does)."""
    def __init__(s): s.q = asyncio.Queue(); s.queries = []; s.interrupts = 0; s.busy = False
    async def query(s, u):
        s.queries.append(u)
        if not s.busy:
            s.busy = True
            asyncio.ensure_future(s._slow())
    async def _slow(s):
        while s.busy:
            s.q.put_nowait(think()); await asyncio.sleep(0.1)
    async def interrupt(s):
        s.interrupts += 1
        if s.busy:
            s.busy = False
            s.q.put_nowait(text("Leftover from the dead turn. ")); s.q.put_nowait(res())
    def receive_messages(s):
        async def gen():
            while True: yield await s.q.get()
        return gen()

async def main():
    B.CFG["capture_timeout_s"] = 1
    b = B.WarmBrain(mouth=Mouth(), bus=Bus())
    b._tally = lambda *a, **k: None; b._remember_session = lambda *a, **k: None
    async def nop(): pass
    b._pull_rate_limits = nop
    b._client = c = Client(); b._start_reader()
    sir = Sink(); b.remote_sink = sir

    # 1. timeout: the capture is interrupted, the typed ask waits behind it
    cap = asyncio.ensure_future(b.capture("Warmup ping"))
    await asyncio.sleep(0.3)
    assert "thinking" in b.bus.states, "face not busy during the capture"
    b.ask("typed message", sir)
    assert c.queries == ["Warmup ping"], "ask jumped the running capture"
    assert await asyncio.wait_for(cap, 10) == "error: the command timed out"
    assert c.interrupts == 1, "timed-out capture was not interrupted"
    assert c.queries == ["Warmup ping", "typed message"], "queued ask never sent"
    assert not b._discard_until_result
    # the typed turn's own reply is spoken, the dead turn's leftovers never
    c.busy = False
    c.q.put_nowait(text("Hello sir. ")); c.q.put_nowait(res())
    await asyncio.sleep(0.3)
    assert sir.got == ["Hello sir."], sir.got

    # 2. interrupt() reaches a capture that has only thought so far
    cap = asyncio.ensure_future(b.capture("/effort high", count_turn=False))
    await asyncio.sleep(0.3)
    await b.interrupt()
    assert c.interrupts == 2, "interrupt was a no-op on a thinking capture"
    assert not b._discard_until_result, "discard armed for a capture turn"
    await asyncio.wait_for(cap, 2)
    assert not b.turn_active
    c.busy = False
    b.ask("next", sir); c.q.put_nowait(text("Next reply. ")); c.q.put_nowait(res())
    await asyncio.sleep(0.3)
    assert sir.got[-1] == "Next reply.", sir.got
    print("OK")
asyncio.run(main())
