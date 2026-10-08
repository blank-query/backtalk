# Run: .venv/bin/python tests/check_followup_routing.py
# Self-check: a background task's follow-up goes to the device that started
# the task even when another device's turn was running when it finished.
import asyncio, sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
from claude_agent_sdk import TaskStartedMessage, TaskNotificationMessage, ResultMessage
from claude_agent_sdk.types import StreamEvent
from backtalk import brain as B

class Sink:
    def __init__(s, name): s.name, s.got, s.conn_id = name, [], name
    def send(s, obj):
        if obj.get("type") == "line": s.got.append(obj["text"])
    def is_live(s): return True
class Bus:
    def __getattr__(s, n): return lambda *a, **k: None
class Mouth:
    def say_chunk(s, text, pending, sink): pass
    def __getattr__(s, n): return lambda *a, **k: None

def ev(text): return StreamEvent(uuid="u", session_id="s", event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}})
def res(): return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1, session_id="s")
def started(t): return TaskStartedMessage(subtype="task_started", data={}, task_id=t, description="d", uuid="u", session_id="s")
def done(t): return TaskNotificationMessage(subtype="task_notification", data={}, task_id=t, status="completed", output_file="", summary="", uuid="u", session_id="s")

class Client:
    def __init__(s): s.q = asyncio.Queue(); s.queries = []
    async def query(s, u): s.queries.append(u)
    def receive_messages(s):
        async def gen():
            while True: yield await s.q.get()
        return gen()

async def main():
    b = B.WarmBrain(mouth=Mouth(), bus=Bus())
    b._tally = lambda *a, **k: None; b._remember_session = lambda *a, **k: None
    async def nop(): pass
    b._pull_rate_limits = nop
    b._client = c = Client(); b._start_reader()
    sir, maam = Sink("sir"), Sink("maam")
    b.remote_sink = Sink("everyone")
    feed = lambda *ms: [c.q.put_nowait(m) for m in ms]
    b.ask("sir asks", sir); await asyncio.sleep(0.05)
    feed(started("t1"), ev("On it, sir. "), res()); await asyncio.sleep(1.2)
    b.ask("maam asks", maam); await asyncio.sleep(0.05)
    feed(ev("Fridge answer, ma'am. "), done("t1"), ev("More for ma'am. "), res()); await asyncio.sleep(1.2)
    feed(ev("Your peer result, sir. "), res()); await asyncio.sleep(1.2)
    print("sir:", sir.got); print("maam:", maam.got); print("everyone:", b.remote_sink.got)
    assert any("peer result" in t for t in sir.got), "follow-up did not reach sir"
    assert not any("peer result" in t for t in maam.got + b.remote_sink.got)
    print("OK")
asyncio.run(main())
