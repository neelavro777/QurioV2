"""
tools/query_database.py
-----------------------
The primary tool available to the Qurio ReAct agent.

Architecture (Three-Layer Defense):
  Layer 1 — Schema + few-shot examples in this docstring (you are reading it now).
             The LLM reads the tool docstring at call time, so the schema lives
             here — not in the system prompt — where it is most useful.
  Layer 2 — validators.validate_query() runs before SQL execution.
             Catches semantic errors deterministically (missing LIMIT, missing
             direction filter, non-SELECT statements).
  Layer 3 — QueryResult always includes row_count. The agent's system prompt
             instructs it to treat 0-row results as "no data found" rather than
             hallucinating an answer.

Components:
  QueryResult    — Pydantic model for structured tool return values.
  query_database — The @tool function bound to the LLM in agent_node.

Implementation Status:
  Phase 2 (current): Executes real SQL against database/qurio.db via
                     database.get_connection(). Validated by validators.py
                     before execution. Returns QueryResult as a dict.
"""

import logging
from typing import Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type — always returned as .model_dump() dict so LangGraph can
# serialize it cleanly into a ToolMessage
# ---------------------------------------------------------------------------

class QueryResult(BaseModel):
    """
    Structured return type for query_database.

    Fields:
      rows       — List of result rows, each row as {column_name: value}.
                   Empty list if no rows matched or an error occurred.
      row_count  — Number of rows returned. Used by the agent as a trust signal:
                   0 rows = no data found (do not hallucinate an answer).
                   >20 rows without LIMIT = flag to user before summarizing.
      error      — Human-readable error string if something went wrong
                   (validator rejection, bad SQL, DB error). None on success.
                   The agent observes this and retries with a corrected query.
    """
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = Field(default=0)
    error: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# The Tool
# ---------------------------------------------------------------------------

@tool
def query_database(sql: str) -> dict:
    """
    Execute a SQL SELECT query against the user's personal finance database
    and return the results.

    SCHEMA:
      transactions(id, merchant, category, amount, currency, date, direction, note)

      Column details:
        id        INTEGER  — Auto-incrementing primary key
        merchant  TEXT     — Name of the merchant (e.g. 'Apple Store', 'Uber Eats')
        category  TEXT     — One of: 'electronics' | 'food' | 'clothing' |
                             'travel' | 'subscription' | 'income'
        amount    REAL     — Transaction amount in the specified currency
        currency  TEXT     — Currency code (e.g. 'USD')
        date      TEXT     — ISO 8601 date string: 'YYYY-MM-DD' (e.g. '2025-05-14')
        direction TEXT     — 'debit' (money OUT) | 'credit' (money IN, e.g. salary/refund)
        note      TEXT     — Free-text description of the transaction

    CRITICAL RULES:
      - Always filter direction = 'debit' when the user asks about spending,
        expenses, or purchases. Credits (salary, refunds) must not be included.
      - Only SELECT statements are allowed. Never generate INSERT/UPDATE/DELETE.
      - date column is a text string — use LIKE '2025-05%' for month filtering,
        or comparison operators like date >= '2025-05-01' AND date <= '2025-05-31'.

    FEW-SHOT EXAMPLES:
      # Last transaction (most recent by date):
      SELECT * FROM transactions ORDER BY date DESC LIMIT 1

      # Total spending in May 2025 (debits only):
      SELECT SUM(amount) as total_spent FROM transactions
      WHERE date LIKE '2025-05%' AND direction = 'debit'

      # Most expensive single purchase:
      SELECT * FROM transactions WHERE direction = 'debit'
      ORDER BY amount DESC LIMIT 1

      # All electronics purchases:
      SELECT * FROM transactions
      WHERE category = 'electronics' AND direction = 'debit'
      ORDER BY date DESC

      # Spending by category:
      SELECT category, SUM(amount) as total FROM transactions
      WHERE direction = 'debit'
      GROUP BY category ORDER BY total DESC

      # Find transactions by merchant (partial match):
      SELECT * FROM transactions WHERE merchant LIKE '%Apple%'

    RETURN VALUE:
      A dict with keys: rows (list of dicts), row_count (int), error (str|None).
      If error is not None, do NOT answer from assumptions — read the error,
      fix the SQL, and call this tool again with the corrected query.
      If row_count is 0, tell the user no matching transactions were found.
    """
    logger.info(f"[Tool] query_database called with SQL: {sql!r}")

    from app.tools.validators import validate_query
    from app.database import get_connection

    # Layer 2: Validate before execution
    # Pass empty string for original_question — the validator still catches
    # write statements, empty SQL, and structural issues.
    validation_error = validate_query(sql, original_question="")
    if validation_error:
        logger.warning(f"[Tool] Validation failed: {validation_error}")
        return QueryResult(error=validation_error).model_dump()

    # Execute against the real database
    try:
        with get_connection() as conn:
            cursor = conn.execute(sql)
            raw_rows = cursor.fetchall()
            # sqlite3.Row supports keys() — convert to plain dicts for JSON serialization
            rows = [dict(row) for row in raw_rows]
            row_count = len(rows)
            logger.info(f"[Tool] Query returned {row_count} rows.")

            # Layer 3 early-warning: log if result set is suspiciously large
            if row_count > 20 and "LIMIT" not in sql.upper():
                logger.warning("[Tool] Large result set (>20 rows) returned without LIMIT clause.")

            return QueryResult(rows=rows, row_count=row_count).model_dump()

    except Exception as e:
        error_msg = f"SQL execution error: {e}. Check your syntax and retry."
        logger.error(f"[Tool] {error_msg}")
        return QueryResult(error=error_msg).model_dump()
