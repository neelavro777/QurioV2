# Qurio - Local AI Personal Finance Assistant

Qurio is a local, terminal-based AI personal finance assistant built using LangGraph, LangChain, and a local vLLM server running Qwen-3.5-2B-AWQ-4bit. It acts as an autonomous database agent that categorizes queries via ultra-fast semantic routing, safely executes SQL queries against a local SQLite database, and retrieves FAQ documentation via a local ChromaDB RAG pipeline.

---

## 1. System Architecture

Qurio processes conversations via a stateful, directed acyclic graph orchestrated by LangGraph. 

### A. Graph Flow

```
  START
    |
    v
classify_intent (Aurelio Semantic Router)
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

*   **classify_intent Node**: Uses `semantic-router` to embed the user query (via BAAI/bge-base-en-v1.5) and calculate cosine similarity against a YAML library of ~190 utterances. This deterministic, embedding-based routing replaces unreliable LLM-based intent classification.
*   **agent_node (ReAct Agent)**: The execution node for database queries. Binds the Qwen 3.5 LLM to SQLite database tools, enforcing history trimming to prevent stale-history hallucinations.
*   **prompt_llm_search_knowledge_base Node**: A full RAG pipeline node that queries a local offline ChromaDB vector database (using the bge-base embedding model) to answer user FAQs and policy questions.
*   **prompt_llm_chat Node**: Handles conversational queries and greetings.

---

## 2. Security, Guardrails & Protections

To guarantee safe and predictable local execution on a small 2B parameter model, the codebase incorporates a multi-layer defense strategy.

### A. ReAct Loop Guard (Infinite Loop Prevention)
The `agent_node` actively monitors a rolling history window of the last 6 messages. If the exact same tool name and arguments are generated twice, the guard injects a strict instruction forcing the model to stop calling tools.

### B. Two-Layer SQL Validation
*   **Schema & Docstring Context**: The exact table schema, column restrictions, and few-shot query examples are embedded directly in the `query_database` tool docstring.
*   **Static AST Validator**: The query string is passed through `validators.validate_query` before hitting the database. It enforces that only SELECT statements are executed, blocks destructive operations (DROP, INSERT, DELETE, UPDATE), and ensures proper filtering constraints.

### C. History Trimming
To prevent "stale-context hallucination", only a sliding window of the last 12 messages is passed to the generation model. This guarantees the model uses fresh database queries rather than hallucinating aggregations from past turns.

---

## 3. Directory Map

*   `backend/scripts/main.py`: Application entry point and interactive REPL shell.
*   `backend/app/agent/graph.py`: LangGraph state machine assembly.
*   `backend/app/agent/nodes.py`: Node execution logic.
*   `backend/app/agent/semantic_router.py`: Embeddings-based intent classification layer.
*   `backend/app/knowledge_base/`: ChromaDB RAG ingestion pipeline, hash-based synchronization (`sync.py`), and retrieval logic.
*   `backend/app/utils/logger.py`: Unified `@log_node` decorator providing state tracing to NDJSON and JSON logs.
*   `backend/app/tools/query_database.py`: Executable SQL SELECT tool with SQLite connectivity.

---

## 4. Local Environment Setup

This project uses the modern Python package manager `uv` to manage environments and dependencies.

### A. Prerequisites
Ensure you have Node, Python `3.11+`, and Git installed.

### B. Initialize Environment
From the project root directory, synchronize the virtual environment and install dependencies:

```bash
uv sync
```

---

## 5. Startup & Running the Application

Qurio's offline architecture requires **two** separate vLLM instances running simultaneously.

### A. Start the Embeddings Server (Port 8001)
Used by the Semantic Router and the RAG pipeline.

```bash
sudo docker run --name bge-embed --runtime nvidia --gpus all \
  -v ~/models/huggingface:/models \
  -p 8001:8000 --ipc=host vllm/vllm-openai:latest \
  /models/bge-base-en-v1.5 \
  --served-model-name BAAI/bge-base-en-v1.5 \
  --gpu-memory-utilization 0.20
```

### B. Start the Generation Server (Port 8000)
Used by the ReAct agent and chat nodes.

```bash
sudo docker run --name qwen35-server --runtime nvidia --gpus all \
  -v ~/models/huggingface:/models \
  -p 8000:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model /models/Qwen3.5-2B-AWQ-4bit \
  --gpu-memory-utilization 0.50 \
  --max-model-len 4096 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --language-model-only
```

### C. Run the Interactive Client
Open a third terminal and start the Qurio REPL shell:

```bash
uv run --project backend python backend/scripts/main.py
```
