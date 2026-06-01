"""
graph.py
--------
Assembles and compiles the Qurio LangGraph state machine.

This module is the single place where nodes are wired together into a graph.
Import `graph` and `default_config` in main.py — do not import nodes or
graph_builder directly from outside this module.

Graph Architecture:

  START
    └─► classify_intent
          ├── "chat"                  → prompt_llm_chat → END
          ├── "search_knowledge_base" → prompt_llm_search_knowledge_base → END
          └── "query_user_information"
                └─► agent_node ◄──────────────────────────────┐
                        │                                      │
                   [has tool calls?]                           │
                    YES │           NO                         │
                        ▼           ▼                          │
                    tool_node     END                          │
                        │                                      │
                        └──────────────────────────────────────┘

Key Design Decisions:
  - tools_condition is LangGraph's built-in conditional edge. It reads the
    last AIMessage: if it has .tool_calls, route to tool_node; else go to END.
    This replaces any custom loop termination logic at the edge level.
  - recursion_limit=12 is set in RunnableConfig. This is a hard ceiling —
    the agent will raise GraphRecursionError before exceeding 12 node visits.
    The soft ceiling is the loop guard in agent_node (LOOP_GUARD_WINDOW=6).
  - checkpointer=InMemorySaver() preserves full conversation history per
    thread_id across multiple graph.invoke() calls in main.py's REPL loop.

Exports:
  graph          — Compiled CompiledStateGraph ready for .invoke()
  default_config — Base config dict. In main.py, merge with thread_id:
                   config = {**default_config, "configurable": {"thread_id": <id>}}
"""

import logging

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from nodes import (
    State,
    classify_intent,
    agent_node,
    prompt_llm_chat,
    prompt_llm_search_knowledge_base,
)
from tools import TOOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

graph_builder = StateGraph(State)

# Register all nodes
graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("agent_node", agent_node)
graph_builder.add_node("tool_node", ToolNode(TOOLS))
graph_builder.add_node("prompt_llm_chat", prompt_llm_chat)
graph_builder.add_node("prompt_llm_search_knowledge_base", prompt_llm_search_knowledge_base)


# ---------------------------------------------------------------------------
# Routing edge for intent classification
# ---------------------------------------------------------------------------

def route_intent(state: State) -> str:
    """
    Reads state['message_intent'] and returns the target node name.
    Defaults to 'prompt_llm_chat' if the intent is missing or unrecognised.
    """
    intent = state.get("message_intent")
    valid = {"chat", "query_user_information", "search_knowledge_base"}
    if intent not in valid:
        logger.warning(f"[route_intent] Unknown intent '{intent}', defaulting to chat.")
        return "prompt_llm_chat"
    # Map intent → node name
    return {
        "chat": "prompt_llm_chat",
        "query_user_information": "agent_node",
        "search_knowledge_base": "prompt_llm_search_knowledge_base",
    }[intent]


# ---------------------------------------------------------------------------
# Wire edges
# ---------------------------------------------------------------------------

# Entry point
graph_builder.add_edge(START, "classify_intent")

# Intent router
graph_builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "prompt_llm_chat": "prompt_llm_chat",
        "agent_node": "agent_node",
        "prompt_llm_search_knowledge_base": "prompt_llm_search_knowledge_base",
    }
)

# ReAct loop: agent_node ↔ tool_node
# tools_condition: if last AIMessage has .tool_calls → "tool_node", else → END
graph_builder.add_conditional_edges(
    "agent_node",
    tools_condition,
    {"tools": "tool_node", END: END},
)
graph_builder.add_edge("tool_node", "agent_node")

# Terminal edges for non-agentic paths
graph_builder.add_edge("prompt_llm_chat", END)
graph_builder.add_edge("prompt_llm_search_knowledge_base", END)


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# Base RunnableConfig — recursion_limit is the hard ceiling for node visits.
# Callers merge this with {"configurable": {"thread_id": <session_id>}}.
default_config = {
    "recursion_limit": 12,
}

logger.info("Qurio graph compiled successfully.")
