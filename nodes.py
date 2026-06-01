"""
nodes.py
--------
All LangGraph node functions for the Qurio agent graph.

Nodes defined here:
  classify_intent               — Intent router. Reads last HumanMessage and
                                  calls LLM with structured output (IntentClassifier)
                                  to set state['message_intent'].
  agent_node                    — ReAct agent for query_user_information path.
                                  Binds the LLM to TOOLS, includes a loop guard
                                  to prevent infinite tool call cycles.
  prompt_llm_chat               — Standard conversational handler.
  prompt_llm_search_knowledge_base — Placeholder for future RAG pipeline.

Tool Execution:
  tool_node is NOT defined here. It is LangGraph's built-in ToolNode, assembled
  in graph.py with: ToolNode(TOOLS). It handles all @tool invocations automatically.

Loop Guard (in agent_node):
  Scans the last LOOP_GUARD_WINDOW messages. If the exact same tool_name +
  tool_args combination appears twice within that window, the guard fires and
  injects a HumanMessage forcing the agent to answer with what it has instead
  of calling the tool again. Window size = 6 messages (3 tool call/result pairs).
"""

import logging
from typing import Annotated
from pydantic import BaseModel, Field
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages

from prompts import prompts
from tools import TOOLS
from logger import log_node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# How many recent messages to scan when checking for duplicate tool calls.
# 6 messages = 3 (tool_call, tool_result) pairs — enough to catch tight loops
# without penalizing legitimate multi-step queries.
# ---------------------------------------------------------------------------
LOOP_GUARD_WINDOW = 6


# ---------------------------------------------------------------------------
# Shared LLM client — initialized once, imported by graph.py
# ---------------------------------------------------------------------------

def create_llm() -> ChatOpenAI:
    """
    Creates the shared ChatOpenAI client pointed at the local vLLM server.

    vLLM Tool-Calling Requirement:
      For agent_node to use tools, vLLM must be started with:
        --enable-auto-tool-choice --tool-call-parser hermes
      Without these flags, any request that includes tools will return a 400 error.
      The Qwen2.5 family uses the Hermes function-calling format.

      Full Docker command (update your startup script):
        docker run --name qwen-server --runtime nvidia --gpus all \\
          -v /home/neelavro/models/huggingface:/models \\
          -p 8000:8000 --ipc=host vllm/vllm-openai:latest \\
          --model /models/Qwen2.5-1.5B-Instruct-AWQ \\
          --quantization awq --gpu-memory-utilization 0.4 \\
          --max-model-len 4096 --enforce-eager \\
          --enable-auto-tool-choice --tool-call-parser hermes
    """
    return ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
        model="/models/Qwen2.5-1.5B-Instruct-AWQ",
        temperature=0.7,
    )

llm = create_llm()


# ---------------------------------------------------------------------------
# State schema — defined here so nodes.py and graph.py share one definition
# ---------------------------------------------------------------------------

from typing import TypedDict

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None


# ---------------------------------------------------------------------------
# Intent Classifier Pydantic schema
# ---------------------------------------------------------------------------

class IntentClassifier(BaseModel):
    """
    Structured output schema for the classify_intent node.
    The LLM must fill both fields.
    """
    reasoning: str = Field(
        ...,
        description=(
            "Explain your step-by-step reasoning on whether this query refers to "
            "personal user information/databases, business policies/knowledge bases, "
            "or general conversation."
        )
    )
    message_intent: Literal["chat", "query_user_information", "search_knowledge_base"] = Field(
        ...,
        description=(
            "Classify whether the user query requires querying the user's personal "
            "information, searching the knowledge base, or just chatting."
        )
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@log_node("classify_intent")
def classify_intent(state: State) -> dict:
    """
    Intent routing node.

    Reads the last HumanMessage from state['messages'], passes it through
    the LLM with IntentClassifier structured output, and returns
    {'message_intent': <'chat'|'query_user_information'|'search_knowledge_base'>}.

    Falls back to 'chat' on any exception to avoid hard crashes.
    """
    try:
        structured_llm = llm.with_structured_output(IntentClassifier)
        user_query = state["messages"][-1].content
        system_msg = prompts.get_system_message("classify_intent")

        result = structured_llm.invoke([
            system_msg,
            HumanMessage(content=f"User Query: {user_query}")
        ])
        logger.debug(f"[classify_intent] Reasoning: {result.reasoning}")
        return {"message_intent": result.message_intent}

    except Exception as e:
        logger.error(f"[classify_intent] Structured output failed, defaulting to 'chat'. Error: {e}")
        return {"message_intent": "chat"}


@log_node("agent_node")
def agent_node(state: State) -> dict:
    """
    ReAct agent node for the query_user_information path.

    Responsibilities:
      1. Load the agent system prompt from prompts.yaml (key: 'agent_node').
      2. Run the loop guard — scan the last LOOP_GUARD_WINDOW messages for
         duplicate (tool_name, tool_args) pairs. If a duplicate is found,
         inject a HumanMessage forcing the agent to stop looping and answer.
      3. Invoke the LLM with smart tool_choice:
           - No prior ToolMessage in window → tool_choice='required'
             Forces the model to call query_database. Prevents hallucination.
           - Prior ToolMessage(s) exist → tool_choice=None
             Model has data and can answer directly. No need to force.
      4. Return {'messages': [ai_response]}.

    Why tool_choice='required' is used (not 'auto'):
      tool_choice='auto' sends 'auto' to the API, which vLLM rejects unless
      started with --enable-auto-tool-choice --tool-call-parser hermes.
      tool_choice='required' sends 'required' — also needs those flags, but
      unlike 'auto', it guarantees the model actually calls a tool on the first
      pass. Without 'required', a 1.5B model will skip tool calls entirely and
      hallucinate from training data, even with explicit system prompt instructions.

    Server requirement:
      vLLM must be started with:
        --enable-auto-tool-choice --tool-call-parser hermes
    """
    system_msg = prompts.get_system_message("agent_node")
    messages = state["messages"]

    # --- Loop Guard ---
    # Scan the last LOOP_GUARD_WINDOW messages for ToolMessage objects.
    # Build a set of (tool_name, frozenset_of_args) for dedup detection.
    recent_messages = messages[-LOOP_GUARD_WINDOW:] if len(messages) > LOOP_GUARD_WINDOW else messages
    seen_calls: dict[tuple, int] = {}
    has_prior_tool_results = False

    for msg in recent_messages:
        if isinstance(msg, ToolMessage):
            has_prior_tool_results = True
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                # Normalize args to a hashable key
                args_key = tuple(sorted(call.get("args", {}).items()))
                call_key = (call.get("name"), args_key)
                seen_calls[call_key] = seen_calls.get(call_key, 0) + 1

    duplicate_detected = any(count >= 2 for count in seen_calls.values())

    if duplicate_detected:
        logger.warning(
            "[agent_node] Loop guard triggered — duplicate tool call detected. "
            "Injecting stop message."
        )
        stop_injection = HumanMessage(
            content=(
                "You have already called that tool with the same arguments. "
                "Do NOT call it again. Answer the user's question using the "
                "information you have already received from the tool results."
            )
        )
        messages = list(messages) + [stop_injection]

    # --- Three-state tool_choice strategy ---
    #
    # State 1 — No ToolMessages in window (first visit):
    #   tool_choice="required" — model MUST call a tool. Prevents hallucination.
    #   A 1.5B model will skip tool calls without this constraint.
    #
    # State 2 — Has ToolMessages in window (subsequent visits):
    #   tool_choice="auto" — model decides whether to call the tool again or answer.
    #   This is the correct ReAct state. It handles:
    #     - Error recovery: model sees QueryResult(error=...) and retries with fixed SQL
    #     - Multi-step queries: model calls tool again for more data before answering
    #     - Completion: model has enough data and answers directly (no tool call → END)
    #
    # State 3 — Loop guard fired:
    #   tool_choice=None — hard stop. Model cannot call tool, must answer from what it has.
    #   Only reached when the same (tool_name, args) pair appears twice in the window.
    #
    # IMPORTANT: 'required' and 'auto' both require vLLM started with:
    #   --enable-auto-tool-choice --tool-call-parser hermes
    if duplicate_detected:
        # Loop guard already fired and injected the stop message above.
        # Force the model to answer — it cannot call the tool again.
        tool_choice = None
        logger.debug("[agent_node] Loop guard active — tool_choice=None (forced answer)")
    elif has_prior_tool_results:
        # Has data. Let the model decide: answer, retry on error, or query more.
        tool_choice = "auto"
        logger.debug("[agent_node] Has tool results — tool_choice='auto' (model decides)")
    else:
        # No data yet. Force at least one tool call to prevent hallucination.
        tool_choice = "required"
        logger.debug("[agent_node] No tool results — tool_choice='required' (forced call)")

    llm_with_tools = llm.bind_tools(TOOLS, tool_choice=tool_choice)
    result = llm_with_tools.invoke([system_msg] + list(messages))

    logger.debug(
        f"[agent_node] tool_choice={tool_choice!r} | "
        f"response has tool_calls: {bool(result.tool_calls)}"
    )
    return {"messages": [result]}



@log_node("prompt_llm_chat")
def prompt_llm_chat(state: State) -> dict:
    """
    Standard conversational node for 'chat' intent.
    No tools. Responds directly to the user's message.
    """
    system_msg = prompts.get_system_message("prompt_llm_chat")
    result = llm.invoke([system_msg] + state["messages"])
    return {"messages": [result]}


@log_node("prompt_llm_search_knowledge_base")
def prompt_llm_search_knowledge_base(state: State) -> dict:
    """
    Placeholder node for the future RAG/knowledge-base pipeline.
    Currently returns a stub message. Will be expanded in Phase 3.
    """
    system_msg = prompts.get_system_message("prompt_llm_search_knowledge_base")
    result = llm.invoke([system_msg] + state["messages"])
    return {"messages": [result]}
