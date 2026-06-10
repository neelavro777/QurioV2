import sys
import unittest
from pathlib import Path

# Add backend to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.semantic_router import SemanticRouter

class TestSemanticRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize the router once for all tests."""
        cls.router = SemanticRouter()
    
    def test_chat_intent_edge_cases(self):
        """Test edge cases that should be classified as 'chat'."""
        queries = [
            "yo what is good",
            "I spent a lot this month", # Statement, not a query
            "money is tight right now",
            "I am stressed about my finances",
            "what is inflation?",
            "asdfghjkl", # Gibberish should fallback to chat
            "tell me a joke about banking",
            "I hate paying fees"
        ]
        
        for q in queries:
            with self.subTest(query=q):
                result = self.router.classify(q)
                self.assertEqual(result.intent, "chat", f"Expected 'chat', but got '{result.intent}' for query: '{q}'")

    def test_query_user_information_edge_cases(self):
        """Test edge cases for user database queries."""
        queries = [
            "how much did I spend this month?",
            "show me my transactions",
            "did I get paid?",
            "what is my balance?",
            "how much did I spend on food?",
            "show my top 3 expenses",
            "what is my total income?"
        ]
        
        for q in queries:
            with self.subTest(query=q):
                result = self.router.classify(q)
                self.assertEqual(result.intent, "query_user_information", f"Expected 'query_user_information', but got '{result.intent}' for query: '{q}'")

    def test_search_knowledge_base_edge_cases(self):
        """Test edge cases for knowledge base FAQ searches."""
        queries = [
            "how do I activate my debit card?",
            "what are the fees for RTGS transfer?",
            "what is the ATM withdrawal limit?",
            "how do I block my card?",
            "what is the interest rate for FD?",
            "can I open a goal based dps?",
            "is there a fee for npsb?",
            "how to contact customer support?"
        ]
        
        for q in queries:
            with self.subTest(query=q):
                result = self.router.classify(q)
                self.assertEqual(result.intent, "search_knowledge_base", f"Expected 'search_knowledge_base', but got '{result.intent}' for query: '{q}'")

if __name__ == '__main__':
    unittest.main()
