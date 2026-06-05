"""
tools/validators.py
-------------------
Layer 2 of the three-layer SQL defense architecture.

Purpose:
  Intercepts generated SQL *before* it is executed against qurio.db.
  Catches semantic errors that a syntactically valid query can still contain.
  Returns a human-readable error string that the agent receives as a
  ToolMessage observation, allowing it to self-correct and retry.

Design Philosophy (from architecture spec):
  This is Option A — deterministic rules — not an LLM pre-check.
  For a 1.5B model, deterministic guardrails are preferred over asking
  a small model to check its own work. Rules are fast, predictable,
  and always fire the same way.

Function:
  validate_query(sql, original_question) -> str | None
    Returns None if the query passes all checks.
    Returns an error string describing the violation if any check fails.
    The error string is designed to be injected directly into a ToolMessage
    so the agent can read it and generate a corrected query.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Keywords that strongly imply the user is asking about money spent (outflows only)
_SPENDING_KEYWORDS = re.compile(
    r"\b(spend|spent|spending|expense|expenses|cost|costs|paid|pay|purchase|bought|debit)\b",
    re.IGNORECASE,
)

# Keywords that imply the user wants the single most recent / top-ranked item
_LIMIT_KEYWORDS = re.compile(
    r"\b(last|latest|most recent|recent|first|top|highest|lowest|cheapest|most expensive)\b",
    re.IGNORECASE,
)


def validate_query(sql: str, original_question: str = "") -> str | None:
    """
    Deterministic SQL validation for Qurio's transactions database.

    Runs a series of rule-based checks on the generated SQL string.
    Returns None if all checks pass (query is safe to execute).
    Returns a descriptive error string if any check fails — this string
    is returned to the agent as a tool observation for self-correction.

    Rules applied (in order):
      1. Reject empty SQL
      2. Reject non-SELECT statements (write protection)
      3. Reject missing LIMIT on queries where original_question implies
         a single result (last, most expensive, etc.)
      4. Warn when a spending question omits WHERE direction = 'debit'
    """
    sql_stripped = sql.strip()

    # Rule 1: Empty SQL
    if not sql_stripped:
        logger.warning("[Validator] Rejected: empty SQL string.")
        return "SQL query is empty. Generate a valid SELECT statement."

    # Rule 2: Write protection — only SELECT is allowed
    first_token = sql_stripped.split()[0].upper()
    if first_token not in ("SELECT", "WITH", "EXPLAIN"):
        logger.warning(f"[Validator] Rejected non-SELECT statement: {first_token}")
        return (
            f"Only SELECT statements are permitted. "
            f"Received a '{first_token}' statement. Rewrite as a SELECT query."
        )

    sql_upper = sql_stripped.upper()

    # Rule 3: LIMIT required for "last / most expensive / top" queries
    if _LIMIT_KEYWORDS.search(original_question) and "LIMIT" not in sql_upper:
        logger.warning("[Validator] Rejected: LIMIT missing on a ranked/recent query.")
        return (
            "The user asked for a specific single result (e.g. 'last', 'most expensive', 'latest'), "
            "but the query has no LIMIT clause. Add 'LIMIT 1' (or appropriate value) to the query."
        )

    # Rule 4: Spending question without direction filter
    # Only flag this if 'direction' is not already referenced in the SQL at all
    if _SPENDING_KEYWORDS.search(original_question) and "DIRECTION" not in sql_upper:
        logger.warning("[Validator] Rejected: spending query missing direction filter.")
        return (
            "The user asked about spending or expenses, but the query does not filter by "
            "direction = 'debit'. Credits (income, refunds) would be incorrectly included. "
            "Add WHERE direction = 'debit' (or AND direction = 'debit') to your query."
        )

    logger.debug("[Validator] Query passed all validation rules.")
    return None
