"""
graph.py
--------
Assembles and compiles the Qurio LangGraph state machine.

This module is the single place where nodes are wired together into a graph.
Import `graph` and `default_config` in main.py — do not import nodes or
graph_builder directly from outside this module.

Graph Architecture (single-node ReAct):

  START
    └─► classify_intent
          │
          ├── "chat"                  → prompt_llm_chat → END
          ├── "search_knowledge_base" → prompt_llm_search_knowledge_base → END
          └── "query_user_information"
                │
                └─► agent_execute (tool_choice="auto" — true ReAct node)
                        │
                   [has tool calls?]
                    YES │       NO
                        ▼       ▼
                    tool_node  END
                        │
                        └──► agent_execute ◄── (ReAct loop back)
                                  │
                             [has tool calls?]
                              YES │       NO
                                  ▼       ▼
                              tool_node  END
                               (repeats up to recursion_limit)

Key Design Decisions:
  - agent_execute uses tool_choice="auto" on every entry. The model's own
    reasoning decides whether a tool call is needed. There is no Phase-1
    gate node — that approach (agent_tool_decision) was removed because it
    blocked tool calls by frequently returning needs_tool=False even when
    the user explicitly requested database access.

  - tools_condition reads the last AIMessage: if it has .tool_calls →
    tool_node → agent_execute again. This is the standard LangGraph ReAct
    pattern.

  - recursion_limit=20 — sufficient for multiple tool-call round trips.
    Each full round trip is: agent_execute(1) + tool_node(1) + agent_execute(1)
    = 3 steps. 20 steps ≈ 6 full round trips plus final answer.

  - checkpointer=InMemorySaver() preserves full conversation history per
    thread_id across multiple graph.invoke() calls.

Exports:
  graph          — Compiled CompiledStateGraph ready for .invoke()
  default_config — Base config dict. In main.py, merge with thread_id:
                   config = {**default_config, "configurable": {"thread_id": <id>}}
"""

import logging

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.nodes import (
    State,
    classify_intent,
    agent_execute,
    prompt_llm_chat,
    prompt_llm_search_knowledge_base,
)
from app.tools import TOOLS
from app.utils.logger import log_node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

graph_builder = StateGraph(State)

# Register all nodes
graph_builder.add_node("classify_intent",                  classify_intent)
graph_builder.add_node("agent_execute",                    agent_execute)

# Wrap ToolNode to capture execution in the terminal/log trace
_tool_node_instance = ToolNode(TOOLS)
@log_node("tool_node")
def wrapped_tool_node(state: State):
    return _tool_node_instance.invoke(state)

graph_builder.add_node("tool_node",                        wrapped_tool_node)
graph_builder.add_node("prompt_llm_chat",                  prompt_llm_chat)
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
    valid  = {"chat", "query_user_information", "search_knowledge_base"}
    if intent not in valid:
        logger.warning(f"[route_intent] Unknown intent '{intent}', defaulting to chat.")
        return "prompt_llm_chat"
    return {
        "chat":                   "prompt_llm_chat",
        "query_user_information": "agent_execute",       # direct → ReAct node
        "search_knowledge_base":  "prompt_llm_search_knowledge_base",
    }[intent]


# ---------------------------------------------------------------------------
# Wire edges
# ---------------------------------------------------------------------------

# Entry point
graph_builder.add_edge(START, "classify_intent")

# Intent router: classify_intent → one of three paths
graph_builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "prompt_llm_chat":                  "prompt_llm_chat",
        "agent_execute":                    "agent_execute",
        "prompt_llm_search_knowledge_base": "prompt_llm_search_knowledge_base",
    }
)

# ReAct loop: agent_execute ↔ tool_node
# tools_condition: if last AIMessage has .tool_calls → "tool_node", else → END
graph_builder.add_conditional_edges(
    "agent_execute",
    tools_condition,
    {"tools": "tool_node", END: END},
)
graph_builder.add_edge("tool_node", "agent_execute")

# Terminal edges for non-agentic paths
graph_builder.add_edge("prompt_llm_chat",                  END)
graph_builder.add_edge("prompt_llm_search_knowledge_base", END)


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# Base RunnableConfig — recursion_limit is the hard ceiling for node visits.
default_config = {
    "recursion_limit": 20,
}

logger.info("Qurio graph compiled successfully (single-node ReAct architecture).")
