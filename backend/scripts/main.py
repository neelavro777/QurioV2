"""
main.py
-------
Qurio application entry point.

Responsibilities (this file only):
  1. Configure logging for the entire application.
  2. Call database.setup_database() once at startup.
  3. Run the REPL loop — read user input, invoke the graph, print response,
     save conversation logs.

What this file does NOT do:
  - Define any LangGraph nodes (→ nodes.py)
  - Assemble or wire the graph (→ graph.py)
  - Implement tools (→ tools/)
  - Define database schema or seed data (→ database/setup.py)

Run with:
  uv run python main.py
"""

import sys
import uuid
import logging
from datetime import datetime
from pathlib import Path

# Add backend/ to sys.path so 'app.' imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.logger import save_conversation_logs, reset_session_trace
from app.database.setup import setup_database
from app.knowledge_base.sync import run_sync
from app.agent.graph import graph, default_config


# ---------------------------------------------------------------------------
# Logging configuration
# Adjust level to DEBUG to see node entry/exit details and SQL validation logs.
# Set to WARNING in production to suppress verbose node tracing.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(name)s | %(levelname)s | %(message)s",
)

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langgraph").setLevel(logging.WARNING)


def main():
    print("=" * 60)
    print("  Qurio — Local AI Finance Assistant")
    print("  Model: Qwen-2.5-1.5B-Instruct (vLLM)")
    print("  Type 'exit' or 'quit' to end the session.")
    print("=" * 60)

    # Initialize databases on startup — idempotent, safe to run every time
    print("\n[Startup]")
    setup_database()
    try:
        run_sync()
    except Exception as e:
        print(f"\n[Error] Knowledge base sync failed. Ensure vLLM is running at http://localhost:8001.\nDetails: {e}")
        sys.exit(1)
    print()

    # Each REPL session gets its own UUID thread_id so InMemorySaver
    # maintains a separate conversation history per session.
    session_id    = str(uuid.uuid4())
    session_start = datetime.now()   # local time — used for log filename
    config = {**default_config, "configurable": {"thread_id": session_id}}

    # Clear any stale trace from a previous run in the same process
    reset_session_trace()

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            if not user_input:
                continue

            print("Thinking...", end="\r")

            response = graph.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )

            # The last message is always the final AIMessage response
            print(f"Qurio: {response['messages'][-1].content}")

            # Persist conversation and node trace to logs/<datetime>_<id>.json
            save_conversation_logs(session_id, response["messages"], session_start)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            err_str = str(e)
            if ("auto" in err_str or "required" in err_str) and "tool choice" in err_str:
                print("\n[Error] vLLM is missing tool-calling flags.")
                print("  Restart vLLM with these two extra flags:")
                print("    --enable-auto-tool-choice --tool-call-parser qwen3_coder --language-model-only")
                print()
                print("  Full Docker command:")
                print("    sudo docker rm -f qwen-server && sudo docker run --name qwen-server \\")
                print("      --runtime nvidia --gpus all \\")
                print("      -v /home/neelavro/models/huggingface:/models \\")
                print("      -p 8000:8000 --ipc=host vllm/vllm-openai:latest \\")
                print("      --model /models/Qwen3.5-2B-AWQ-4bit \\")
                print("      --gpu-memory-utilization 0.50 \\")
                print("      --max-model-len 4096 --enforce-eager \\")
                print("      --enable-auto-tool-choice --tool-call-parser qwen3_coder --language-model-only")
            else:
                print(f"\n[Error] {e}")
                print("Ensure vLLM is running at http://localhost:8000.")


if __name__ == "__main__":
    main()
