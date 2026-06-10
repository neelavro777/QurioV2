"""
api/server.py
-------------
FastAPI server for the Qurio observability frontend.

Endpoints:
  GET  /api/health              — Health check
  GET  /api/database/tables     — List all SQLite tables with row counts
  GET  /api/database/table/{name} — Get all rows for a specific table
  GET  /api/knowledge-base      — Get KB structure from headings_map.json
  GET  /api/sessions            — List all conversation log session files
  GET  /api/sessions/{id}       — Get full session log (node trace + conversation)
  GET  /api/retrieval-traces    — Get all retrieval trace records from NDJSON
  GET  /api/routing-traces      — Get all routing trace records from NDJSON
  POST /api/chat                — Send a message and get agent response + trace

CORS is open for localhost development.
"""

import json
import logging
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup — ensure backend/ is importable as 'app.*'
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.database.setup import DATABASE_PATH, setup_database
from app.utils.logger import save_conversation_logs, reset_session_trace

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(name)s | %(levelname)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langgraph").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / "logs"
CONVERSATION_LOGS_DIR = LOGS_DIR / "conversation"
RETRIEVAL_TRACE_PATH = LOGS_DIR / "retrieval_trace.ndjson"
ROUTING_TRACE_PATH = LOGS_DIR / "routing_trace.ndjson"
HEADINGS_MAP_PATH = PROJECT_ROOT / "knowledge_database" / "headings_map.json"

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Qurio Observability API",
    description="Backend API for the Qurio agent traceability frontend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-process agent session state
# Per request we create a fresh session — the graph has InMemorySaver
# so each session_id gets its own conversation thread.
# ---------------------------------------------------------------------------
_current_session: dict = {}


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    ai_response: str
    node_trace: list[dict]
    intent: Optional[str] = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize the SQLite database on startup."""
    try:
        setup_database()
        logger.info("[API] Database initialized.")
    except Exception as e:
        logger.error("[API] Database init failed: %s", e)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "Qurio API", "timestamp": datetime.now().isoformat()}


# ---------------------------------------------------------------------------
# Database endpoints
# ---------------------------------------------------------------------------

def _get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/api/database/tables")
async def list_tables():
    """List all tables in the SQLite database with their row counts and schema."""
    try:
        with _get_db_connection() as conn:
            # Get all table names
            tables_cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            )
            table_names = [row[0] for row in tables_cursor.fetchall()]

            tables = []
            for name in table_names:
                # Row count
                count_row = conn.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()
                row_count = count_row[0] if count_row else 0

                # Schema (column info)
                pragma_cursor = conn.execute(f"PRAGMA table_info([{name}]);")
                columns = [
                    {
                        "cid": r["cid"],
                        "name": r["name"],
                        "type": r["type"],
                        "notnull": bool(r["notnull"]),
                        "pk": bool(r["pk"]),
                    }
                    for r in pragma_cursor.fetchall()
                ]

                tables.append({
                    "name": name,
                    "row_count": row_count,
                    "columns": columns,
                })

            return {"tables": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/database/table/{table_name}")
async def get_table_rows(table_name: str, limit: int = 100, offset: int = 0):
    """Get rows from a specific table with pagination."""
    # Whitelist table names to prevent SQL injection
    allowed = {"transactions"}
    if table_name not in allowed:
        # Also allow any table that exists in the DB
        try:
            with _get_db_connection() as conn:
                result = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                    (table_name,)
                ).fetchone()
                if not result:
                    raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    try:
        with _get_db_connection() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM [{table_name}]").fetchone()[0]
            cursor = conn.execute(f"SELECT * FROM [{table_name}] LIMIT ? OFFSET ?", (limit, offset))
            rows = [dict(r) for r in cursor.fetchall()]
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            return {
                "table": table_name,
                "total_rows": total,
                "returned": len(rows),
                "offset": offset,
                "limit": limit,
                "columns": columns,
                "rows": rows,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Knowledge base endpoints
# ---------------------------------------------------------------------------

@app.get("/api/knowledge-base")
async def get_knowledge_base():
    """Return the knowledge base structure from headings_map.json."""
    try:
        with open(HEADINGS_MAP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"knowledge_base": data}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="headings_map.json not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge-base/item")
async def get_kb_item(type: str, id: str):
    """Get a specific chunk or parent from chunks.json."""
    chunks_path = PROJECT_ROOT / "backend" / "app" / "knowledge_base" / "chunks.json"
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if type == "parent":
            item = data.get("parents", {}).get(id)
            if item:
                return {"item": item}
            raise HTTPException(status_code=404, detail="Parent not found")
            
        elif type == "chunk":
            for chunk in data.get("chunks", []):
                if chunk.get("chunk_id") == id:
                    return {"item": chunk}
            raise HTTPException(status_code=404, detail="Chunk not found")
            
        else:
            raise HTTPException(status_code=400, detail="Invalid type")
            
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="chunks.json not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Session log endpoints
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def list_sessions():
    """List all available conversation session log files."""
    try:
        if not CONVERSATION_LOGS_DIR.exists():
            return {"sessions": []}

        sessions = []
        for f in sorted(CONVERSATION_LOGS_DIR.iterdir(), reverse=True):
            if f.suffix == ".json" and f.is_file():
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    
                    node_count = len(data.get("node_trace", []))
                    msg_count = len(data.get("conversation", []))
                    session_start = data.get("session_start", "")
                    
                    # Determine intents used
                    intents = list(set(
                        n.get("outgoing_state", {}).get("message_intent", "")
                        for n in data.get("node_trace", [])
                        if n.get("node") == "classify_intent"
                        and n.get("outgoing_state", {}).get("message_intent")
                    ))

                    sessions.append({
                        "id": data.get("session_id", f.stem),
                        "filename": f.name,
                        "session_start": session_start,
                        "node_count": node_count,
                        "message_count": msg_count,
                        "intents": intents,
                        "file_size_kb": round(f.stat().st_size / 1024, 1),
                    })
                except Exception:
                    sessions.append({
                        "id": f.stem,
                        "filename": f.name,
                        "session_start": "",
                        "node_count": 0,
                        "message_count": 0,
                        "intents": [],
                        "file_size_kb": round(f.stat().st_size / 1024, 1),
                        "parse_error": True,
                    })

        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get the full log for a specific session by ID or filename."""
    try:
        if not CONVERSATION_LOGS_DIR.exists():
            raise HTTPException(status_code=404, detail="Logs directory not found")

        # Search by session_id field or filename stem
        for f in CONVERSATION_LOGS_DIR.iterdir():
            if f.suffix == ".json" and f.is_file():
                if session_id in f.stem or session_id in f.name:
                    with open(f, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    return data

        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Trace log endpoints
# ---------------------------------------------------------------------------

@app.get("/api/retrieval-traces")
async def get_retrieval_traces(limit: int = 50, offset: int = 0):
    """Get retrieval trace records from the NDJSON log."""
    try:
        if not RETRIEVAL_TRACE_PATH.exists():
            return {"traces": [], "total": 0}

        all_lines = []
        with open(RETRIEVAL_TRACE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        # Reverse so most recent first
        all_lines.reverse()
        total = len(all_lines)
        sliced = all_lines[offset: offset + limit]

        return {"traces": sliced, "total": total, "offset": offset, "limit": limit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/routing-traces")
async def get_routing_traces(limit: int = 50, offset: int = 0):
    """Get semantic router trace records from the NDJSON log."""
    try:
        if not ROUTING_TRACE_PATH.exists():
            return {"traces": [], "total": 0}

        all_lines = []
        with open(ROUTING_TRACE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        all_lines.reverse()
        total = len(all_lines)
        sliced = all_lines[offset: offset + limit]

        return {"traces": sliced, "total": total, "offset": offset, "limit": limit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Send a user message to the Qurio agent and return the response + node trace.
    
    Lazy-imports the graph to avoid loading LangGraph at startup when it's
    not needed (e.g., when only browsing logs).
    """
    from app.agent.graph import graph, default_config
    from app.utils.logger import _session_trace

    session_id = request.session_id or str(uuid.uuid4())
    session_start = datetime.now()

    # Clear in-memory trace for this turn
    reset_session_trace()

    try:
        config = {
            **default_config,
            "configurable": {"thread_id": session_id}
        }

        response = graph.invoke(
            {"messages": [{"role": "user", "content": request.message}]},
            config=config,
        )

        ai_response = response["messages"][-1].content

        # Capture the trace that was accumulated during this invoke
        node_trace = list(_session_trace)

        # Extract intent from trace
        intent = None
        for node in node_trace:
            if node.get("node") == "classify_intent":
                intent = node.get("outgoing_state", {}).get("message_intent")
                break

        # Save logs
        try:
            save_conversation_logs(session_id, response["messages"], session_start)
        except Exception as log_err:
            logger.warning("[API] Failed to save conversation logs: %s", log_err)

        return ChatResponse(
            session_id=session_id,
            user_message=request.message,
            ai_response=ai_response,
            node_trace=node_trace,
            intent=intent,
        )

    except Exception as e:
        logger.error("[API] Chat error: %s", e)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


# ---------------------------------------------------------------------------
# Mount Frontend Static Files
# ---------------------------------------------------------------------------
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "qurio"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning("[API] Frontend directory not found at %s. UI will not be served.", FRONTEND_DIR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="warning",
    )
