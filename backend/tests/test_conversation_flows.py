import sys
import unittest
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.graph import graph
from app.agent.nodes import classify_intent

class TestConversationFlows(unittest.TestCase):
    
    def test_classify_intent_node_direct(self):
        """Test the graph routing node directly to ensure it updates message_intent and prev_intent."""
        state = {
            "messages": [HumanMessage(content="how do I block my card?")],
            "message_intent": None,
            "prev_intent": None
        }
        
        result = classify_intent(state)
        self.assertEqual(result["message_intent"], "search_knowledge_base")
        self.assertEqual(result["prev_intent"], "search_knowledge_base")

    def test_follow_up_intent_inheritance(self):
        """Test that short ambiguous follow-ups inherit the previous intent."""
        state = {
            "messages": [
                HumanMessage(content="how much did I spend this month?"),
                AIMessage(content="You spent BDT 5000."),
                HumanMessage(content="and last month?") # Ambiguous follow-up
            ],
            "message_intent": None,
            "prev_intent": "query_user_information" # Inherited from previous turn
        }
        
        # The SemanticRouter's classify method handles the inheritance if prev_intent is passed.
        # classify_intent node logic:
        # intent = router.classify(user_query, prev_intent=state.get("prev_intent"))
        result = classify_intent(state)
        
        self.assertEqual(result["message_intent"], "query_user_information")

if __name__ == '__main__':
    unittest.main()
