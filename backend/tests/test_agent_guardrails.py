import sys
import unittest
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.nodes import _trim_history, HISTORY_WINDOW

class TestAgentGuardrails(unittest.TestCase):
    
    def test_trim_history_preserves_first_message(self):
        """Test that _trim_history always keeps the first HumanMessage."""
        messages = [HumanMessage(content="First message ever")]
        
        # Add a bunch of filler messages to push length past HISTORY_WINDOW
        for i in range(HISTORY_WINDOW + 5):
            messages.append(AIMessage(content=f"Filler {i}"))
            
        trimmed = _trim_history(messages)
        
        self.assertEqual(len(trimmed), HISTORY_WINDOW + 1)
        self.assertEqual(trimmed[0].content, "First message ever")
        self.assertEqual(trimmed[-1].content, f"Filler {HISTORY_WINDOW + 4}")

    def test_trim_history_short_conversation(self):
        """Test that short conversations are returned unmodified."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there")
        ]
        trimmed = _trim_history(messages)
        self.assertEqual(len(trimmed), 2)
        self.assertEqual(trimmed, messages)

    # Note: We do not directly test `agent_execute` here because it requires a running
    # vLLM instance to call the ChatOpenAI client. Instead, we test the logic components
    # (like trim_history). For full agent loop simulation, integration tests with a 
    # mocked LLM are required, but the logic inside _trim_history is the core deterministic guard.

if __name__ == '__main__':
    unittest.main()
