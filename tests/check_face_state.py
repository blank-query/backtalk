# Run: .venv/bin/python tests/check_face_state.py
# Self-check: the face follows the model, not just backtalk's own turns. A
# reply that finishes playing while a tool still runs goes back to thinking;
# a background task's report that wakes the session after its turn ended
# shows thinking from the CLI's init until its ResultMessage, then idle; the
# task count tracks start and finish; a press that came to nothing mid-turn
# settles back to thinking (main.py's _settle_state, via mouth.busy).
import asyncio, sys, threading
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
from claude_agent_sdk import (ResultMessage, SystemMessage, TaskStartedMessage,
                              TaskNotificationMessage)
from claude_agent_sdk.types import StreamEvent
from backtalk import brain as B

class Bus:
    def __init__(s): s.state, s.tasks = "idle", 0
    def set_state(s, n): s.state = n
    def set_tasks(s, n): s.tasks = n
    def __getattr__(s, n): return lambda *a, **k: None
class Mouth:
    """Plays each chunk for 0.2 s, then settles the state the way
    mouth._reply_finished does."""
    def __init__(s, bus): s.bus, s._speaking, s.busy = bus, False, lambda: False
    @property
    def speaking(s): return s._speaking
    def say_chunk(s, text, pending, sink):
        if not text: return
        s._speaking = True; s.bus.set_state("speaking")
        def done():
            s._speaking = False
            s.bus.set_state("thinking" if s.busy() else "idle")
        threading.Timer(0.2, done).start()

def ev(e): return StreamEvent(uuid="u", session_id="s", event=e)
def think(): return ev({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm "}})
def text(t): return ev({"type": "content_block_delta", "delta": {"type": "text_delta", "text": t}})
def stop(): return ev({"type": "content_block_stop"})
def init(): return SystemMessage(subtype="init", data={})
def res(): return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1, session_id="s")
def started(t): return TaskStartedMessage(subtype="task_started", data={}, task_id=t, description="d", uuid="u", session_id="s")
def done(t): return TaskNotificationMessage(subtype="task_notification", data={}, task_id=t, status="completed", output_file="", summary="", uuid="u", session_id="s")

class Client:
    def __init__(s): s.q = asyncio.Queue()
    async def query(s, u): pass
    def receive_messages(s):
        async def gen():
            while True: yield await s.q.get()
        return gen()

async def main():
    bus = Bus()
    b = B.WarmBrain(mouth=Mouth(bus), bus=bus)
    b._tally = lambda *a, **k: None; b._remember_session = lambda *a, **k: None
    async def nop(): pass
    b._pull_rate_limits = nop
    b._client = c = Client(); b._start_reader()
    feed = lambda *ms: [c.q.put_nowait(m) for m in ms]
    tick = lambda s=0.5: asyncio.sleep(s)

    # an asked turn: speaks, starts a background task, keeps working
    bus.set_state("thinking"); b.ask("start the build")
    feed(init(), text("On it, sir. "), stop(), started("t1")); await tick()
    assert bus.tasks == 1, bus.tasks
    assert bus.state == "thinking", f"reply ended mid-turn but face says {bus.state}"
    feed(text("Started. "), stop(), res()); await tick()
    assert bus.state == "idle", bus.state

    # the task finishes after the turn ended: the session wakes on its own
    feed(done("t1"), init()); await tick(0.1)
    assert bus.tasks == 0, bus.tasks
    assert bus.state == "thinking", f"unsolicited turn shows {bus.state}"
    feed(think(), think()); await tick(0.1)
    assert bus.state == "thinking", bus.state
    feed(text("The build is done, sir. "), stop()); await tick(0.05)
    assert bus.state == "speaking", bus.state
    await tick(); assert bus.state == "thinking", bus.state   # spoken, turn still open
    feed(res()); await tick()
    assert bus.state == "idle", bus.state

    # a silent unsolicited turn (thinking only, no text) also ends idle
    feed(init(), think()); await tick(0.1)
    assert bus.state == "thinking", bus.state
    feed(res()); await tick(0.1)
    assert bus.state == "idle", bus.state
    # what main.py's _settle_state reads after a press that came to nothing
    # (a queued tap during a turn must hand back thinking, not "listening")
    feed(init(), think()); await tick(0.1)
    assert b.mouth.busy(), "turn running but busy() is False"
    b._discard_until_result = True        # an interrupted turn doesn't count
    assert not b.mouth.busy()
    feed(res()); await tick(0.1)
    assert not b.mouth.busy() and not b.turn_active
    print("OK")
asyncio.run(main())
