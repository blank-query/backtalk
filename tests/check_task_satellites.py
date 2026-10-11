# Run: .venv/bin/python tests/check_task_satellites.py
# Self-check: satellites never ghost. A task that ends without a recognised
# bookend (TaskStop's "killed" on task_updated, an unknown status word, a
# shell that exits with no notification at all) leaves the count, the CLI's
# background_tasks_changed level replaces the set outright (ambient tasks
# hidden), and a rebuilt session starts at zero.
import asyncio, sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
from claude_agent_sdk import SystemMessage, TaskStartedMessage, TaskUpdatedMessage
from backtalk import brain as B

class Bus:
    tasks = 0
    def set_tasks(s, n): s.tasks = n
    def __getattr__(s, n): return lambda *a, **k: None
class Mouth:
    speaking, busy = False, (lambda s: False)
    def say_chunk(s, *a): pass
class Client:
    def __init__(s): s.q = asyncio.Queue()
    async def query(s, u): pass
    async def connect(s): pass
    async def disconnect(s): pass
    def receive_messages(s):
        async def gen():
            while True: yield await s.q.get()
        return gen()

def started(t): return TaskStartedMessage(subtype="task_started", data={}, task_id=t, description="d", uuid="u", session_id="s")
def updated(t, st): return TaskUpdatedMessage(subtype="task_updated", data={}, task_id=t, patch={"status": st})
def level(*ts, ambient=()):
    return SystemMessage(subtype="background_tasks_changed", data={"tasks":
        [{"task_id": t} for t in ts] + [{"task_id": t, "ambient": True} for t in ambient]})

async def main():
    bus = Bus()
    b = B.WarmBrain(mouth=Mouth(), bus=bus)
    b._tally = lambda *a, **k: None; b._remember_session = lambda *a, **k: None
    async def nop(): pass
    b._pull_rate_limits = nop
    b._client = c = Client(); b._start_reader()
    async def feed(*ms):
        for m in ms: c.q.put_nowait(m)
        await asyncio.sleep(0.1)

    await feed(*(started(t) for t in "abcd"))
    assert bus.tasks == 4, bus.tasks
    await feed(updated("a", "killed"), updated("b", "cancelled"), updated("c", "running"))
    assert bus.tasks == 2, f"killed/cancelled must clear: {bus.tasks}"
    await feed(level("c", ambient=("x",)))     # d's shell exited with no bookend
    assert bus.tasks == 1 and b._active_tasks == {"c"}, b._active_tasks
    await feed(started("e"), level("c", "e"))
    assert bus.tasks == 2, bus.tasks
    await feed(level())
    assert bus.tasks == 0, bus.tasks

    await feed(started("f")); assert bus.tasks == 1
    B.ClaudeSDKClient = lambda **k: Client()
    await b._rebuild()
    assert bus.tasks == 0 and not b._active_tasks, "rebuild kept stale tasks"
    print("OK")
asyncio.run(main())
