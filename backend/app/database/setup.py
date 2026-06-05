"""
database/setup.py
-----------------
Handles all SQLite initialization for Qurio.

Functions:
  setup_database()  — Idempotent. Creates qurio.db and the 'transactions'
                      table if they don't exist, then seeds demo rows.
                      Safe to call on every startup.
  get_connection()  — Returns a configured sqlite3.Connection.
                      row_factory is set to sqlite3.Row so columns are
                      accessible by name (row['merchant']) in addition to
                      index (row[1]).

Database File Location:
  qurio.db is created in the same directory as this file (Qurio/database/).
  The absolute path is DATABASE_PATH, exported for reference.
"""

import os
import sqlite3
import logging

logger = logging.getLogger(__name__)

from pathlib import Path

# Absolute path to the database file — lives in backend/data/
DATABASE_PATH = str(Path(__file__).parent.parent.parent / "data" / "qurio.db")

# ---------------------------------------------------------------------------
# Seed data — 15 rows deliberately designed to test specific SQL edge cases:
#   - Rows 1, 13, 15: same merchant, different categories (filter by category)
#   - Rows 3, 11:     credits mixed into debits (WHERE direction = 'debit')
#   - Rows 4 & 11:    purchase + refund pair (SUM vs net)
#   - Rows 1,7,10,15: multiple electronics (ORDER BY amount DESC)
#   - Rows 6, 12:     same merchant, different dates (date range filtering)
# ---------------------------------------------------------------------------
SEED_TRANSACTIONS = [
    (1,  "Apple Store",     "electronics",  1299.00, "USD", "2025-05-03", "debit",  "MacBook Air M3"),
    (2,  "Steam",           "subscription",   59.99, "USD", "2025-05-05", "debit",  "Elden Ring DLC"),
    (3,  "Salary",          "income",        3200.00, "USD", "2025-05-07", "credit", "May paycheck"),
    (4,  "Nike",            "clothing",       184.95, "USD", "2025-05-09", "debit",  "Air Max 95"),
    (5,  "Zara",            "clothing",        94.00, "USD", "2025-05-11", "debit",  "Summer jacket"),
    (6,  "Uber Eats",       "food",            34.50, "USD", "2025-05-13", "debit",  "Dinner"),
    (7,  "Sony",            "electronics",    349.00, "USD", "2025-05-14", "debit",  "WH-1000XM5 headphones"),
    (8,  "Spotify",         "subscription",     9.99, "USD", "2025-05-15", "debit",  "Monthly plan"),
    (9,  "Emirates",        "travel",          870.00, "USD", "2025-05-18", "debit",  "Dubai flight"),
    (10, "Amazon",          "electronics",    129.00, "USD", "2025-05-20", "debit",  "Kindle Paperwhite"),
    (11, "Refund - Zara",   "clothing",        94.00, "USD", "2025-05-22", "credit", "Returned jacket"),
    (12, "Uber Eats",       "food",            22.75, "USD", "2025-05-24", "debit",  "Lunch"),
    (13, "Apple Store",     "subscription",     2.99, "USD", "2025-05-25", "debit",  "iCloud 50GB"),
    (14, "Booking.com",     "travel",          430.00, "USD", "2025-05-27", "debit",  "Hotel Dubai"),
    (15, "Apple Store",     "electronics",    429.00, "USD", "2025-05-30", "debit",  "AirPods Pro"),
]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
    id        INTEGER PRIMARY KEY,
    merchant  TEXT    NOT NULL,
    category  TEXT    NOT NULL,   -- 'electronics' | 'food' | 'clothing' | 'travel' | 'subscription' | 'income'
    amount    REAL    NOT NULL,
    currency  TEXT    NOT NULL DEFAULT 'USD',
    date      TEXT    NOT NULL,   -- ISO 8601: 'YYYY-MM-DD'
    direction TEXT    NOT NULL,   -- 'debit' (money out) | 'credit' (money in)
    note      TEXT
);
"""

INSERT_SQL = """
INSERT OR IGNORE INTO transactions
    (id, merchant, category, amount, currency, date, direction, note)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);
"""


def get_connection() -> sqlite3.Connection:
    """
    Returns a configured sqlite3.Connection to qurio.db.

    The row_factory is set to sqlite3.Row so that result rows can be
    accessed both by column index AND by name:
        row['merchant']   # works
        row[1]            # also works

    The caller is responsible for closing the connection, or use it
    as a context manager:
        with get_connection() as conn:
            ...  # auto-commits on exit, rolls back on exception
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def setup_database() -> None:
    """
    Idempotent database initialization.

    - Creates qurio.db at DATABASE_PATH if it doesn't exist.
    - Creates the 'transactions' table if it doesn't exist (CREATE TABLE IF NOT EXISTS).
    - Seeds 15 demo rows using INSERT OR IGNORE so re-runs are safe.
    - Logs the final row count at INFO level.

    Call this once at application startup (main.py).
    """
    logger.info(f"Initializing database at: {DATABASE_PATH}")

    with get_connection() as conn:
        # Create table
        conn.execute(CREATE_TABLE_SQL)
        logger.debug("transactions table created or already exists.")

        # Seed rows — INSERT OR IGNORE means this is idempotent
        conn.executemany(INSERT_SQL, SEED_TRANSACTIONS)
        logger.debug(f"Attempted to seed {len(SEED_TRANSACTIONS)} rows.")

        # Verify
        row_count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        logger.info(f"Database ready. transactions table has {row_count} rows.")

    print(f"  [DB] qurio.db ready — {row_count} transactions loaded.")
