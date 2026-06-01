"""
database/
---------
Manages the SQLite persistence layer for Qurio.

Exports:
  setup_database()  — Creates the 'transactions' table and seeds it with
                      demo data if it doesn't already exist. Call once at
                      application startup (main.py).
  get_connection()  — Returns a live sqlite3.Connection to qurio.db.
                      The caller is responsible for closing it (or use as
                      a context manager).
"""

from .setup import setup_database, get_connection

__all__ = ["setup_database", "get_connection"]
