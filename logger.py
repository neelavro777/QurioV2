import os
import json
from datetime import datetime
from functools import wraps
from langchain_core.messages import message_to_dict, BaseMessage

# Define the absolute path for the logs directory in the project
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Ensure the logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

def log_node(node_name: str):
    """
    A decorator to log incoming and outgoing states of a LangGraph node in the terminal.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            # Print a neat entering header
            print("\n" + "="*80)
            print(f"\033[93m[NODE ENTER] >>> Entering Node: '{node_name}'\033[0m")
            print("="*80)
            
            # Extract and display incoming state
            incoming_messages = state.get("messages", [])
            print(f"\033[36mIncoming State:\033[0m")
            print(f"  - Message Count: {len(incoming_messages)}")
            if incoming_messages:
                last_msg = incoming_messages[-1]
                role = type(last_msg).__name__.replace("Message", "").lower()
                print(f"  - Last Message: \033[32m[{role}]\033[0m {last_msg.content!r}")
            print(f"  - Message Intent: \033[35m{state.get('message_intent')}\033[0m")
            print("-" * 80)

            # Invoke the wrapped node function
            result = func(state, *args, **kwargs)

            # Print a neat exiting header
            print("="*80)
            print(f"\033[92m[NODE EXIT] <<< Exited Node: '{node_name}'\033[0m")
            print("="*80)
            
            # Extract and display outgoing updates
            print(f"\033[36mOutgoing State Updates:\033[0m")
            for key, val in result.items():
                if key == "messages":
                    print(f"  - New Messages Appended: {len(val)}")
                    for msg in val:
                        if isinstance(msg, BaseMessage):
                            role = type(msg).__name__.replace("Message", "").lower()
                            print(f"    * \033[32m[{role}]\033[0m {msg.content!r}")
                        else:
                            print(f"    * {msg}")
                else:
                    print(f"  - \033[35m{key}\033[0m: {val}")
            print("="*80 + "\n")

            return result
        return wrapper
    return decorator


def save_conversation_logs(session_id: str, messages: list):
    """
    Serializes LangChain messages into raw and simplified dictionaries, 
    then saves the full conversation log history into the logs/ directory.
    """
    raw_history = []
    simple_history = []
    
    for msg in messages:
        if isinstance(msg, BaseMessage):
            # Full raw LangChain dictionary serialization
            raw_dict = message_to_dict(msg)
            raw_history.append(raw_dict)
            
            # Simple flat role/content serialization
            role = msg.type if hasattr(msg, "type") else type(msg).__name__.replace("Message", "").lower()
            if role == "ai":
                role = "assistant"
            simple_history.append({
                "role": role,
                "content": msg.content
            })
        elif isinstance(msg, dict):
            raw_history.append(msg)
            simple_history.append(msg)
            
    log_file = os.path.join(LOGS_DIR, f"session_{session_id}.json")
    
    log_payload = {
        "session_id": session_id,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "message_count": len(messages),
        "history_simple": simple_history,
        "history_raw_dict": raw_history
    }
    
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"\033[91m[Error writing conversation logs: {e}]\033[0m")
