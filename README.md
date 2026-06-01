# Qurio - Local AI Personal Finance Assistant

Qurio is a local, terminal-based AI personal finance assistant built using LangGraph, LangChain, and a local vLLM server running Qwen-2.5-1.5B-Instruct. It acts as an autonomous database agent that categorizes queries and safely executes SQL queries against a local SQLite database.

---

## 1. System Architecture

Qurio processes conversations via a stateful, directed acyclic graph orchestrated by LangGraph. 

### A. Graph Flow

```
  START
    |
    v
classify_intent (Structured Intent Classifier)
    |
    +---> "chat"                  ==> prompt_llm_chat ==> END
    +---> "search_knowledge_base" ==> prompt_llm_search_knowledge_base ==> END
    +---> "query_user_information"==> agent_node (ReAct Loop) <------+
                                         |                           |
                                  [Has Tool Calls?]                  |
                                     /       \                       |
                                   YES        NO                     |
                                    v          v                     |
                                tool_node     END                    |
                                    |                                |
                                    +--------------------------------+
```

### B. Core Components

*   **classify_intent Node**: Reads the last message and passes it through the LLM with structured output (`IntentClassifier` Pydantic schema). It uses Chain-of-Thought (reasoning first) to classify the user query into chat, query_user_information, or search_knowledge_base.
*   **agent_node (ReAct Agent)**: The execution node for database queries. Binds the LLM to database tools and manages three-state tool choice requirements based on execution history.
*   **tool_node**: Built-in LangGraph execution layer that triggers SQLite queries against the database.
*   **prompt_llm_chat Node**: Handles conversational queries, greetings, and generic factual questions.
*   **prompt_llm_search_knowledge_base Node**: A placeholder node for operational documentation and policy FAQs.

---

## 2. Security, Guardrails & Protections

To guarantee safe and predictable local execution on a small 1.5B parameter model, the codebase incorporates a multi-layer defense strategy.

### A. ReAct Loop Guard (Infinite Loop Prevention)
Small local models can get stuck in repeating cycles of generating identical tool queries. The `agent_node` implemented in `nodes.py` actively monitors a rolling history window of the last 6 messages. 
If the exact same tool name and arguments are generated twice within this window, the guard triggers:
1. It injects a strict instruction forcing the model to stop calling tools.
2. It strips tool-calling capabilities (`tool_choice=None`) on the subsequent invocation, forcing the model to answer using only the data it has already retrieved.

### B. Two-Layer SQL Validation
To prevent destructive database actions or parsing errors:
*   **Schema & Docstring Context (Layer 1)**: The exact table schema, column restrictions, and few-shot query examples are embedded directly in the `query_database` tool docstring. The LLM reads this at call-time.
*   **Static AST Validator (Layer 2)**: The query string is passed through `validators.validate_query` before hitting the database. It enforces that only SELECT statements are executed, blocks destructive operations (DROP, INSERT, DELETE, UPDATE), and ensures proper filtering constraints.

---

## 3. Directory Map

*   `main.py`: Application entry point, REPL execution loop, database setup, and conversation history saving.
*   `graph.py`: LangGraph state machine assembly, intent routing edges, ReAct loops, and compilation with InMemorySaver checkpoints.
*   `nodes.py`: Node functions, Pydantic classification schema, ReAct tool bindings, and loop guard logic.
*   `logger.py`: Interceptor decorator (`@log_node`) providing color-coded, real-time terminal tracing of state transitions, plus JSON log writers.
*   `prompts.py` & `prompts.yaml`: Centralized system message registry that decouples prompts from code execution.
*   `tools/query_database.py`: Executable SQL SELECT tool with SQLite connectivity.
*   `tools/validators.py`: Static semantic validation logic for generated SQL strings.
*   `database/setup.py`: Database bootstrap and idempotent transactional demo seeding.
*   `logs/`: Folder containing local conversation logs stored as raw and simplified JSON dictionaries.

---

## 4. Local Environment Setup

This project uses the modern Python package manager `uv` to manage environments and dependencies.

### A. Prerequisites
Ensure you have Node (for npx), Python, and Git installed.

### B. Initialize Environment
From the project root directory, synchronize the virtual environment and install all pinned dependencies:

```bash
uv sync
```

This will automatically create a local `.venv` environment and install LangChain, LangGraph, Pydantic, and SQLite adapters.

---

## 5. Startup & Running the Application

### A. Local vLLM Server Startup
For the database agent to execute tools, the vLLM server must be started with auto-tool choice and Hermes function-calling parsers active. Use the following Docker run command to start your Qwen server:

```bash
sudo docker run --name qwen-server --runtime nvidia --gpus all \
  -v ~/models/huggingface:/models \
  -p 8000:8000 --ipc=host vllm/vllm-openai:latest \
  --model /models/Qwen2.5-1.5B-Instruct-AWQ \
  --quantization awq --gpu-memory-utilization 0.4 \
  --max-model-len 4096 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes
```

### B. Running the Interactive Client
To start the interactive Qurio REPL shell, run:

```bash
uv run python main.py
```

This initializes the local `qurio.db` file, seeds the initial mock transactions, and launches a stateful, interactive chat session in the terminal.

