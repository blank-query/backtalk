"""brain_hermes's event translation: Hermes gateway events in, the SDK message
shapes WarmBrain's reader speaks from out. No Hermes needed.
Run: .venv/bin/python tests/check_hermes_events.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtalk.brain_hermes import HermesClient

c = HermesClient("")
c.sid, c.stored_id = "live", "stored-1"
for kind, p in [("message.start", {}), ("reasoning.delta", {"text": "hmm"}), ("reasoning.delta", {"text": ""}),
                ("message.delta", {"text": "Let me check. "}), ("tool.generating", {"name": "terminal"}),
                ("tool.start", {"tool_id": "t1", "name": "terminal"}), ("message.delta", {"text": "Done."}),
                ("subagent.start", {"subagent_id": "s1", "goal": "g", "task_count": 1, "task_index": 0}),
                ("subagent.complete", {"subagent_id": "s1", "goal": "g", "task_count": 1, "task_index": 0, "status": "error"}),
                ("message.complete", {"text": "Let me check. Done.", "usage": {"input": 120, "output": 30}}),
                ("message.complete", {"text": "x", "usage": {"input": 200, "output": 35}})]:
    c._event(kind, p)
out = []
while not c._q.empty():
    m = c._q.get_nowait()
    out.append(m.event.get("type") + ":" + str((m.event.get("content_block") or m.event.get("delta") or {}).get("type", ""))
               if type(m).__name__ == "StreamEvent" else type(m).__name__)
assert out == ["message_start:", "content_block_start:thinking", "content_block_delta:thinking_delta",
               "content_block_stop:", "content_block_start:text", "content_block_delta:text_delta",
               "content_block_stop:", "content_block_start:tool_use", "content_block_stop:",
               "content_block_start:text", "content_block_delta:text_delta",
               "TaskStartedMessage", "TaskNotificationMessage",
               "content_block_stop:", "AssistantMessage", "ResultMessage", "AssistantMessage", "ResultMessage"], out
print("OK")
