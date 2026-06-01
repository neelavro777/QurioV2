import os
import yaml
from langchain_core.messages import SystemMessage

class PromptRegistry:
    def __init__(self, yaml_filename="prompts.yaml"):
        # Resolve path relative to this Python module
        self.yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_filename)
        self.load_prompts()
        
    def load_prompts(self):
        """Loads or reloads the YAML prompt file from disk."""
        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                self._prompts = yaml.safe_load(f)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load prompts file at {self.yaml_path}. Error: {e}")
            
    def get_system_message(self, agent_name: str, **kwargs) -> SystemMessage:
        """
        Retrieves a system message for a given node. 
        Supports dynamic string formatting if variables are provided in the future.
        """
        raw_prompt = self._prompts.get(agent_name, {}).get("system", "")
        if not raw_prompt:
            raise KeyError(f"Prompt for agent '{agent_name}' was not found in prompts.yaml.")
            
        # Strip trailing newlines and format variables if they exist
        formatted_prompt = raw_prompt.strip().format(**kwargs)
        return SystemMessage(content=formatted_prompt)

# Instantiate a single shared instance of the registry
prompts = PromptRegistry()
