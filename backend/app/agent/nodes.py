"""
nodes.py
--------
All LangGraph node functions for the Qurio agent graph.

Nodes defined here:
  classify_intent    — Intent router. Uses the SemanticRouter (embedding-based,
                       zero-LLM) to classify each incoming message into one of
                       three routes: 'chat', 'query_user_information',
                       'search_knowledge_base'. Classification is done via
                       cosine similarity against curated utterance vectors.
                       See semantic_router.py for full algorithm documentation.

  agent_execute      — The ReAct execution node. Binds tools with
                       tool_choice="auto" on every entry. The model decides
                       whether to call a tool or answer directly — no external
                       gate or phase classifier overrides it.

                       After a ToolMessage lands in context the node is called
                       again (still tool_choice="auto") so the model can answer
                       or issue a follow-up query. A loop guard prevents
                       infinite identical-call cycles.

  prompt_llm_chat    — Standard conversational handler. No tools.
  prompt_llm_search_knowledge_base — Full RAG pipeline node.

Tool Execution:
  tool_node is NOT defined here. It is LangGraph's built-in ToolNode,
  assembled in graph.py with: ToolNode(TOOLS).

Loop Guard (in agent_execute):
  Scans the last LOOP_GUARD_WINDOW messages. If the exact same
  (tool_name, tool_args) combination appears twice, the guard fires and
  injects a stop HumanMessage. Window size = 6 messages (3 tool
  call/result pairs).

Architecture note — what was tried and removed:
  LLM-based classify_intent (IntentClassifier Pydantic model, llm_cold at
  temp=0.1) was removed in favour of SemanticRouter. The 1.5B model was
  unreliable at structured-output classification (per 'Let Me Speak Freely?'
  ACL 2024); misclassifications cascaded into bad downstream behaviour.

  Two-phase decoding (agent_tool_decision → agent_execute) was implemented
  earlier and removed — see ARCHITECTURE.md §6 for full failure analysis.

  Regex fast-path in classify_intent was similarly removed — it caused
  mis-routing when informal words appeared alongside genuine queries
  (e.g. "yo, list my subscriptions").
"""

import logging
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages

from app.agent.prompts import prompts
from app.agent.semantic_router import get_router
from app.tools import TOOLS
from app.utils.logger import log_node
from app.knowledge_base.retriever import retrieve, RetrievalResult
from app.knowledge_base.populate_chroma import NO_CONTEXT_MESSAGE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loop guard window — how many recent messages to scan for duplicate calls.
# 6 = 3 (tool_call + tool_result) pairs. Enough to catch tight loops without
# penalising legitimate multi-step queries.
# ---------------------------------------------------------------------------
LOOP_GUARD_WINDOW = 6

# How many recent messages to pass to the LLM in agent_execute.
# Keeping a bounded window prevents the model from hallucinating based on
# transaction data seen many turns ago. The first HumanMessage is always
# preserved (it anchors the conversation context).
# 12 = ~3 full ReAct cycles (HumanMsg + AIMsg + ToolMsg + AIMsg answer)
HISTORY_WINDOW = 12


# ---------------------------------------------------------------------------
# Shared LLM client — single temperature for generation and tool execution.
# llm_cold (temp=0.1) has been REMOVED — intent classification no longer
# uses the LLM. The SemanticRouter handles it via embeddings.
# ---------------------------------------------------------------------------

def _make_llm(temperature: float) -> ChatOpenAI:
    """
    Creates a ChatOpenAI client pointed at the local vLLM server.

    vLLM requirements for tool calling:
      --enable-auto-tool-choice --tool-call-parser qwen3_coder --language-model-only

    Full Docker command:
      docker run --name qwen-server --runtime nvidia --gpus all \
        -v /home/neelavro/models/huggingface:/models \
        -p 8000:8000 --ipc=host vllm/vllm-openai:latest \
        --model /models/Qwen3.5-2B-AWQ-4bit \
        --gpu-memory-utilization 0.50 \
        --max-model-len 4096 --enforce-eager \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder --language-model-only
    """
    return ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
        model="/models/Qwen3.5-2B-AWQ-4bit",
        temperature=temperature,
    )


llm_hot = _make_llm(temperature=0.7)   # generation / tool execution
llm     = llm_hot                       # alias for backward compatibility


# ---------------------------------------------------------------------------
# State schema — shared between nodes.py and graph.py
# ---------------------------------------------------------------------------

class State(TypedDict):
    messages:       Annotated[list, add_messages]
    message_intent: str | None
    # Tracks the previous turn's intent for follow-up context inheritance.
    # Updated by classify_intent on every turn. Not used downstream by agents.
    prev_intent:    str | None


# ---------------------------------------------------------------------------
# IntentClassifier Pydantic model — REMOVED
# The LLM-based structured-output classifier has been replaced by
# SemanticRouter (semantic_router.py). No Pydantic model is needed for
# intent classification anymore.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_history(messages: list) -> list:
    """
    Trim the conversation history before passing to agent_execute.

    Strategy:
      - Always keep the FIRST HumanMessage (establishes context).
      - Always keep the last HISTORY_WINDOW messages (recency window).
      - Deduplicate: if the first message is already in the last window,
        don't include it twice.

    Why this matters for 1.5B models:
      A long conversation accumulates many tool results — rows of transactions
      from 5+ turns ago. When the model sees those in context it sometimes
      "answers" aggregation questions from them instead of querying the DB.
      Trimming the history reduces the amount of stale data visible.

    This does NOT affect the LangGraph state — messages is never mutated here.
    We only control what the LLM sees, not what is persisted.
    """
    if len(messages) <= HISTORY_WINDOW:
        return list(messages)

    # Find the first HumanMessage to always anchor with
    first_human_idx = next(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
        None,
    )

    tail = messages[-HISTORY_WINDOW:]

    if first_human_idx is None or first_human_idx >= len(messages) - HISTORY_WINDOW:
        # First human message is already in the tail — no need to prepend
        return list(tail)

    # Prepend first human message as the conversation anchor
    return [messages[first_human_idx]] + list(tail)


@log_node("classify_intent")
def classify_intent(state: State) -> dict:
    """
    Intent routing node — EMBEDDING-BASED (no LLM call).

    Uses SemanticRouter to classify the incoming message via cosine similarity
    against curated utterance vectors from routes.yaml. Classification is
    deterministic, <15ms, and does not call Qwen2.5 at any point.

    Algorithm (see semantic_router.py for full details):
      1. Embed the user query via bge-base-en-v1.5 on localhost:8001.
      2. Compute top-K mean cosine similarity against each route's utterance
         matrix (pre-computed at startup).
      3. Win: highest-scoring route, if score >= HIGH_CONFIDENCE_THRESHOLD (0.50).
      4. Inherit: if ambiguous (0.38-0.50) AND previous turn was non-chat AND
         query is short — inherit the previous intent (follow-up handling).
      5. Default: if below all thresholds → 'chat' (safe fallback).

    Falls back to 'chat' on any exception to avoid blocking the user.
    """
    # Extract the latest user message
    messages = state["messages"]
    last_msg = messages[-1] if messages else None
    user_query = (
        last_msg.content
        if last_msg and hasattr(last_msg, "content")
        else ""
    )

    # Retrieve previous intent for follow-up inheritance
    prev_intent: str | None = state.get("prev_intent")

    try:
        router = get_router()
        result = router.classify(user_query, prev_intent=prev_intent)

        logger.debug(
            "[classify_intent] intent=%r | confidence=%.3f | method=%s",
            result.intent, result.confidence, result.method,
        )

        return {
            "message_intent": result.intent,
            "prev_intent":    result.intent,   # update for next turn's inheritance
        }

    except Exception as e:
        logger.error(
            "[classify_intent] SemanticRouter failed, defaulting to 'chat'. Error: %s", e
        )
        return {
            "message_intent": "chat",
            "prev_intent":    prev_intent,     # preserve previous on failure
        }


@log_node("agent_execute")
def agent_execute(state: State) -> dict:
    """
    The ReAct execution node — hardened for Qwen2.5-1.5B-Instruct-AWQ.

    Architecture:
      - tool_choice="auto": The model decides whether to call a tool or answer.
        Never forced to "required" (that was the failed two-phase approach).
      - Loop guard: Scans last LOOP_GUARD_WINDOW messages for duplicate
        (tool_name, args) pairs. If found, injects a stop message and strips
        tools so the model MUST answer from existing results.
      - SQL leak guard: If the model writes SQL in its text response instead
        of making a tool call, we log a warning. The model violated RULE 2.
        We cannot force a tool call after the fact but we log it for debugging.
      - History trimming: Only the last HISTORY_WINDOW messages are passed to
        the LLM to prevent context bloat from causing stale-history hallucination.
        The system message and very first human message are always included.

    Known failure modes for 1.5B models addressed here:
      1. "Shows SQL instead of calling tool" — RULE 2 in prompt, logged in guard.
      2. "Hallucinates from conversation history" — RULE 1 + RULE 5 in prompt,
         history trimming reduces the amount of stale context visible.
      3. "Relies on previously-seen transactions for aggregations" — RULE 5 forces
         fresh DB query for any SUM/MAX/MIN/COUNT question.
      4. "Infinite tool call loop" — loop guard fires on exact duplicate calls.
    """
    system_msg = prompts.get_system_message("agent_node")
    messages   = state["messages"]

    # ── History trimming — reduce stale context the model can hallucinate from ──
    # Keep: [first human message] + last HISTORY_WINDOW messages.
    # This prevents the model from seeing transactions from many turns ago and
    # "answering" based on them instead of making a fresh tool call.
    trimmed_messages = _trim_history(messages)

    # ── Loop Guard ──────────────────────────────────────────────────────────────
    recent_msgs = (
        trimmed_messages[-LOOP_GUARD_WINDOW:]
        if len(trimmed_messages) > LOOP_GUARD_WINDOW
        else trimmed_messages
    )
    seen_calls: dict[tuple, int] = {}
    has_prior_tool_results = False

    for msg in recent_msgs:
        if isinstance(msg, ToolMessage):
            has_prior_tool_results = True
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                args_key = tuple(sorted(call.get("args", {}).items()))
                call_key = (call.get("name"), args_key)
                seen_calls[call_key] = seen_calls.get(call_key, 0) + 1

    duplicate_detected = any(count >= 2 for count in seen_calls.values())

    if duplicate_detected:
        logger.warning(
            "[agent_execute] Loop guard triggered — duplicate tool call detected. "
            "Injecting stop message and forcing final answer."
        )
        stop_msg = HumanMessage(
            content=(
                "STOP. You have already called that tool with the same SQL. "
                "Do NOT call it again. "
                "Use the tool results already in this conversation to answer the user's question now."
            )
        )
        invoke_messages = list(trimmed_messages) + [stop_msg]
        # No tools — model must answer from existing data
        invoke_llm = llm_hot
        logger.debug("[agent_execute] Loop guard active — no tools, forced answer")
    else:
        invoke_messages = list(trimmed_messages)
        invoke_llm = llm_hot.bind_tools(TOOLS, tool_choice="auto")
        logger.debug(
            "[agent_execute] tool_choice='auto' | "
            "has_prior_tool_results=%s | trimmed_history=%d msgs",
            has_prior_tool_results, len(trimmed_messages),
        )

    result = invoke_llm.invoke([system_msg] + invoke_messages)

    # ── SQL Leak Guard ───────────────────────────────────────────────────────────
    # Detects when the model wrote SQL in its text response instead of calling
    # the tool. We cannot auto-fix this (we'd need to re-invoke), but we log it
    # prominently so it shows up in conversation logs for debugging.
    if result.content and not result.tool_calls:
        content_upper = result.content.upper()
        sql_keywords  = ("SELECT ", "FROM ", "WHERE ", "ORDER BY", "LIMIT ")
        if any(kw in content_upper for kw in sql_keywords):
            logger.warning(
                "[agent_execute] SQL LEAK DETECTED — model wrote SQL in text response "
                "instead of calling the tool. RULE 2 violation. Response: %r",
                result.content[:300],
            )
            print(
                f"\n\033[93m[WARN] SQL leak: model showed SQL instead of calling tool. "
                f"Check logs/conversation/ for full trace.\033[0m"
            )

    logger.debug(
        "[agent_execute] response has tool_calls: %s | content_len: %d",
        bool(result.tool_calls),
        len(result.content) if result.content else 0,
    )
    return {"messages": [result]}


# ---------------------------------------------------------------------------
# Non-agentic nodes
# ---------------------------------------------------------------------------

@log_node("prompt_llm_chat")
def prompt_llm_chat(state: State) -> dict:
    """
    Standard conversational node for 'chat' intent.
    No tools. Responds directly to the user's message.
    """
    system_msg = prompts.get_system_message("prompt_llm_chat")
    result = llm_hot.invoke([system_msg] + state["messages"])
    return {"messages": [result]}


@log_node("prompt_llm_search_knowledge_base")
def prompt_llm_search_knowledge_base(state: State, config: RunnableConfig) -> dict:
    """
    RAG node for knowledge-base queries.

    Pipeline:
      1. Extract the user's latest message.
      2. Call retrieve(query) → RetrievalResult.
         - If no_context=True: return NO_CONTEXT_MESSAGE immediately.
           No LLM call is made (zero hallucination risk, zero latency cost).
         - If context found: build a labelled context prompt and call the LLM.
      3. Inject the formatted context string into the user message.
         The LLM receives a structured message:
           RETRIEVED CONTEXT:\n{context_string}\n\nUSER QUESTION: {query}
      4. Return the LLM response as a new message.

    Confidence scores, chunk IDs, and parent sections are logged to:
      - Terminal (colour-coded, via retriever.py)
      - logs/retrieval_trace.ndjson (one JSON line per call)
      - The @log_node decorator logs full incoming/outgoing state.

    NO_CONTEXT fallback:
      Returns a deterministic constant string. Does NOT call the LLM.
      This guarantees: no hallucination, no latency, no API cost on failure path.
    """
    messages   = state["messages"]
    last_msg   = messages[-1]
    user_query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # ── Step 1: Retrieve relevant context ─────────────────────────────────────
    logger.info(
        "[prompt_llm_search_knowledge_base] Retrieving context for query: %r",
        user_query[:120],
    )

    session_id = config.get("configurable", {}).get("thread_id")

    try:
        retrieval: RetrievalResult = retrieve(user_query, session_id=session_id)
    except RuntimeError as e:
        # ChromaDB not populated yet — hard fallback
        logger.error(
            "[prompt_llm_search_knowledge_base] ChromaDB unavailable: %s. "
            "Returning NO_CONTEXT_MESSAGE.", e
        )
        from langchain_core.messages import AIMessage as _AI
        return {"messages": [_AI(content=NO_CONTEXT_MESSAGE)]}

    # ── Step 2: NO_CONTEXT path (no LLM call) ─────────────────────────────────
    if retrieval.no_context:
        logger.info(
            "[prompt_llm_search_knowledge_base] no_context=True → returning "
            "graceful deflection (no LLM call)"
        )
        return {"messages": [AIMessage(content=NO_CONTEXT_MESSAGE)]}

    # ── Step 3: Build context-injected prompt ─────────────────────────────────
    logger.info(
        "[prompt_llm_search_knowledge_base] context_blocks=%d | "
        "product_areas=%s | duration_ms=%.1f",
        len(retrieval.context_blocks),
        [cb.product_area for cb in retrieval.context_blocks],
        retrieval.duration_ms,
    )

    augmented_content = (
        f"RETRIEVED CONTEXT:\n"
        f"{retrieval.context_string}\n\n"
        f"---\n\n"
        f"USER QUESTION: {user_query}"
    )

    system_msg = prompts.get_system_message("prompt_llm_search_knowledge_base")

    # Replace the last user message with the context-augmented version.
    # All prior messages are passed through unchanged (preserves conversation history).
    messages_with_context = list(messages[:-1]) + [HumanMessage(content=augmented_content)]

    # ── Step 4: Call LLM with context ─────────────────────────────────────────
    result = llm_hot.invoke([system_msg] + messages_with_context)

    logger.debug(
        "[prompt_llm_search_knowledge_base] LLM response: %r",
        result.content[:200] if result.content else "(empty)",
    )

    return {"messages": [result]}
