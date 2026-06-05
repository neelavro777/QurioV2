"""
logger.py
---------
Observability layer for the Qurio LangGraph agent.

Provides:
  @log_node(node_name)
      Decorator for any LangGraph node function. On every call:
        - Prints colour-coded entry/exit headers to the terminal (unchanged)
        - Records a structured event (incoming + outgoing state) to the
          in-memory _session_trace list, which is flushed to the log file
          on save_conversation_logs().

  reset_session_trace()
      Clears _session_trace. Call at the start of each new REPL session
      so stale events from a previous run do not leak into the new file.

  save_conversation_logs(session_id, messages, session_start)
      Writes logs/<YYYY-MM-DD>_<HH-MM-SS>_<8char-id>.json containing:
        session_id     — full UUID string
        session_start  — ISO-format local datetime string
        node_trace     — list of per-node event dicts (see @log_node)
        conversation   — simple [{role, content}] list for quick reading

Log filename examples:
  2026-06-02_11-24-32_96a4f78c.json   ← easy to sort, immediately readable
  2026-06-02_14-05-11_3bd9f21a.json
"""

import os
import json
import logging
from datetime import datetime
from functools import wraps
from langchain_core.messages import message_to_dict, BaseMessage, AIMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Absolute path for the logs directory (workspace root)
# ---------------------------------------------------------------------------
from pathlib import Path
LOGS_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "logs" / "conversation")
os.makedirs(LOGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# In-memory session trace — accumulated by @log_node, flushed on save
# ---------------------------------------------------------------------------
_session_trace: list[dict] = []
_turn_counter: list[int] = [0]   # mutable int via list so inner functions can mutate


def reset_session_trace() -> None:
    """Clear the in-memory trace and turn counter. Call at the start of each session."""
    _session_trace.clear()
    _turn_counter[0] = 0


# ---------------------------------------------------------------------------
# @log_node decorator
# ---------------------------------------------------------------------------

def log_node(node_name: str):
    """
    Decorator that wraps a LangGraph node function to:
      1. Print colour-coded terminal output (entry state, exit updates).
      2. Append a structured event dict to _session_trace for file logging.

    The event dict captures ALL state fields on entry and ALL returned keys
    on exit, including tool_calls extracted from any AIMessage in the result.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            entry_time = datetime.now()
            _turn_counter[0] += 1
            turn = _turn_counter[0]

            # ── Terminal: entry header ────────────────────────────────────────
            print("\n" + "="*80)
            print(f"\033[93m[NODE ENTER] >>> Entering Node: '{node_name}'\033[0m")
            print("="*80)

            incoming_messages = state.get("messages", [])
            print(f"\033[36mIncoming State:\033[0m")
            print(f"  - Message Count   : {len(incoming_messages)}")
            if incoming_messages:
                if len(incoming_messages) > 1:
                    prev_msg = incoming_messages[-2]
                    role = type(prev_msg).__name__.replace("Message", "").lower()
                    print(f"  - Prev Message    : \033[32m[{role}]\033[0m {prev_msg.content[:200]!r}")
                last_msg = incoming_messages[-1]
                role = type(last_msg).__name__.replace("Message", "").lower()
                print(f"  - Current Message : \033[32m[{role}]\033[0m {last_msg.content[:200]!r}")
            print(f"  - message_intent  : \033[35m{state.get('message_intent')}\033[0m")
            print("-" * 80)

            # ── Build incoming snapshot for log file ─────────────────────────
            incoming_snap: dict = {
                "message_count": len(incoming_messages),
                "message_intent": state.get("message_intent"),
            }
            if incoming_messages:
                if len(incoming_messages) > 1:
                    prev = incoming_messages[-2]
                    incoming_snap["previous_message_role"] = (
                        type(prev).__name__.replace("Message", "").lower()
                    )
                    incoming_snap["previous_message_content"] = (
                        prev.content[:500] if prev.content else ""
                    )
                last = incoming_messages[-1]
                incoming_snap["current_message_role"] = (
                    type(last).__name__.replace("Message", "").lower()
                )
                incoming_snap["current_message_content"] = (
                    last.content[:500] if last.content else ""
                )

            # ── Call the node ─────────────────────────────────────────────────
            result = func(state, *args, **kwargs)
            exit_time = datetime.now()

            # ── Terminal: exit header ─────────────────────────────────────────
            print("="*80)
            print(f"\033[92m[NODE EXIT] <<< Exited Node: '{node_name}'\033[0m")
            print("="*80)
            print(f"\033[36mOutgoing State Updates:\033[0m")
            for key, val in result.items():
                if key == "messages":
                    print(f"  - New Messages Appended: {len(val)}")
                    for msg in val:
                        if isinstance(msg, BaseMessage):
                            role = type(msg).__name__.replace("Message", "").lower()
                            print(f"    * \033[32m[{role}]\033[0m {msg.content!r}")
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print(f"      \033[33m→ tool_call:\033[0m {tc.get('name')}({tc.get('args')})")
                        else:
                            print(f"    * {msg}")
                else:
                    print(f"  - \033[35m{key}\033[0m: {val}")
            print("="*80 + "\n")

            # ── Build outgoing snapshot for log file ─────────────────────────
            outgoing_snap: dict = {}
            for key, val in result.items():
                if key == "messages":
                    appended = []
                    for msg in val:
                        if isinstance(msg, BaseMessage):
                            role = type(msg).__name__.replace("Message", "").lower()
                            entry: dict = {
                                "role": role,
                                "content": msg.content[:500] if msg.content else "",
                            }
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                entry["tool_calls"] = [
                                    {
                                        "tool": tc.get("name"),
                                        "args": tc.get("args"),
                                    }
                                    for tc in msg.tool_calls
                                ]
                            appended.append(entry)
                    outgoing_snap["messages_appended"] = len(appended)
                    outgoing_snap["appended"] = appended
                else:
                    # Scalars and simple types — safe to include directly
                    outgoing_snap[key] = val

            # ── Append to in-memory trace ─────────────────────────────────────
            _session_trace.append({
                "node":           node_name,
                "turn":           turn,
                "entry_time":     entry_time.isoformat(timespec="milliseconds"),
                "exit_time":      exit_time.isoformat(timespec="milliseconds"),
                "duration_ms":    round((exit_time - entry_time).total_seconds() * 1000, 1),
                "incoming_state": incoming_snap,
                "outgoing_state": outgoing_snap,
            })

            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# save_conversation_logs
# ---------------------------------------------------------------------------

def save_conversation_logs(
    session_id: str,
    messages: list,
    session_start: datetime,
) -> None:
    """
    Flush _session_trace and conversation history to a JSON log file.

    Filename format: YYYY-MM-DD_HH-MM-SS_<first8chars>.json
    Example:         2026-06-02_11-24-32_96a4f78c.json

    File structure:
      session_id    — full UUID string
      session_start — ISO-format local datetime
      node_trace    — list of per-node event dicts from @log_node
      conversation  — simple [{role, content}] list for quick reading
    """
    # Build filename from session_start (local time) and first 8 chars of UUID
    short_id  = session_id.replace("-", "")[:8]
    timestamp = session_start.strftime("%Y-%m-%d_%H-%M-%S")
    filename  = f"{timestamp}_{short_id}.json"
    log_path  = os.path.join(LOGS_DIR, filename)

    # Build simple conversation list
    conversation: list[dict] = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            role = msg.type if hasattr(msg, "type") else (
                type(msg).__name__.replace("Message", "").lower()
            )
            if role == "ai":
                role = "assistant"
            entry: dict = {"role": role, "content": msg.content}
            if isinstance(msg, AIMessage) and msg.tool_calls:
                entry["tool_calls"] = [
                    {"tool": tc.get("name"), "args": tc.get("args")}
                    for tc in msg.tool_calls
                ]
            conversation.append(entry)
        elif isinstance(msg, dict):
            conversation.append(msg)

    payload = {
        "session_id":    session_id,
        "session_start": session_start.isoformat(timespec="seconds"),
        "node_trace":    list(_session_trace),   # snapshot — don't clear yet
        "conversation":  conversation,
    }

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.debug(f"[Logger] Session log written to: {filename}")
    except Exception as e:
        print(f"\033[91m[Error writing session log: {e}]\033[0m")
