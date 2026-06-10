import sys
import unittest
from pathlib import Path

# Add backend to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools.validators import validate_query

class TestSQLValidators(unittest.TestCase):
    
    def test_valid_select_queries(self):
        """Test that normal, safe SELECT queries pass validation."""
        valid_queries = [
            ("SELECT * FROM transactions;", ""),
            ("SELECT merchant, amount FROM transactions WHERE direction = 'debit';", ""),
            ("SELECT sum(amount) FROM transactions;", ""),
            ("SELECT * FROM transactions ORDER BY amount DESC LIMIT 5;", "last 5"),
            ("SELECT date, amount FROM transactions WHERE date > '2023-01-01';", "")
        ]
        
        for q, orig in valid_queries:
            with self.subTest(query=q):
                result = validate_query(q, orig)
                self.assertIsNone(result, f"Query '{q}' should have been returned None (safe). Error: {result}")

    def test_destructive_operations_blocked(self):
        """Test that DROP, DELETE, INSERT, UPDATE are blocked."""
        destructive_queries = [
            "DROP TABLE transactions;",
            "DELETE FROM transactions WHERE id = 1;",
            "INSERT INTO transactions (merchant, amount) VALUES ('Test', 100);",
            "UPDATE transactions SET amount = 0;",
            "TRUNCATE TABLE transactions;",
            "ALTER TABLE transactions ADD COLUMN dummy TEXT;"
        ]
        
        for q in destructive_queries:
            with self.subTest(query=q):
                result = validate_query(q)
                self.assertIsNotNone(result, f"Destructive query '{q}' was NOT blocked.")
                self.assertIn("Only SELECT statements are permitted", result)

    def test_limit_required_for_ranked_queries(self):
        """Test that LIMIT is required for queries asking for last/top results."""
        q = "SELECT * FROM transactions ORDER BY amount DESC"
        orig = "show me the most expensive transaction"
        result = validate_query(q, orig)
        self.assertIsNotNone(result)
        self.assertIn("add 'limit 1'", result.lower())

    def test_direction_filter_required_for_spending(self):
        """Test that spending queries require direction = 'debit'."""
        q = "SELECT sum(amount) FROM transactions"
        orig = "how much did I spend?"
        result = validate_query(q, orig)
        self.assertIsNotNone(result)
        self.assertIn("direction = 'debit'", result)

if __name__ == '__main__':
    unittest.main()
