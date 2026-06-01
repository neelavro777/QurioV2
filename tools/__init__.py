"""
tools/
------
LangChain-compatible tool definitions for the Qurio ReAct agent.

Exports:
  TOOLS  — A list of all registered @tool functions that should be bound
           to the LLM in agent_node (nodes.py). Add new tools here.

  query_database  — Executes a SQL SELECT against qurio.db and returns a
                    QueryResult dict (rows, row_count, error).
"""

from .query_database import query_database

# This list is the single source of truth for which tools the agent has access to.
# Import new tools above and append them here.
TOOLS = [query_database]

__all__ = ["TOOLS", "query_database"]
