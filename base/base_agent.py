from abc import abstractmethod
from typing import List, Optional, Any, Dict
import json

from pydantic import Field, BaseModel

from base.tool.base_tool import BaseTool
from base.tool.page_cache_manager import PageCacheManager
from base.engine.async_llm import AsyncLLM

class BaseAgent(BaseTool, BaseModel):
    
    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Prompts
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Dependencies
    # Make LLM optional; concrete agents may initialize it from config.
    llm: Optional[AsyncLLM] = Field(default=None, description="Language model instance")

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")

    # Agent-As-An-Action
    parameters: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        
    @abstractmethod
    async def step(self) -> str:
        """Execute a single step in the agent's workflow.

        Must be implemented by subclasses to define specific behavior.
        """

    @abstractmethod
    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.
        
        Args:
            request: Optional initial user request to process.
        """

    async def __call__(self, **kwargs) -> Any:
        """Execute the agent with given parameters."""
        return await self.run(**kwargs)
    
    def to_param(self) -> Dict[str, Any]:
        return {
            "type": "agent-as-function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def system_prompt_formatter(self, system_prompt: str, tools: Dict[str, Any]) -> str:
        """
        Format the system prompt with tool specifications.
        """
        tool_specs = "\n".join([json.dumps(tool.to_param()) for tool in tools.values()])
        tool_usage = "\n".join([tool.tool_usage for tool in tools.values()])
        tool_usage_workflow = "\n".join([tool.tool_usage_workflow for tool in tools.values()])
        
        return system_prompt.format(
            tool_specs=tool_specs, 
            tool_usage=tool_usage, 
            tool_usage_workflow=tool_usage_workflow
        )

    def convert_history_to_string(self, history: List[Dict[str, str]]) -> str:
        history_str = ""
        for dic in history:
            history_str += dic["content"] + "\n"
        return history_str

    def get_cache_info(self, cache_manager: PageCacheManager) -> Dict[str, Any]:
        """
        Get current cache information
        Returns:
            Dictionary containing cache size and all cursor IDs
        """
        return {
            "cache_size": cache_manager.get_cache_size(),
            "cursors": cache_manager.get_all_cursors()
        }
    
    def clear_cache(self, cache_manager: PageCacheManager):
        """
        Clear the page cache for the current task
        Used to manually clear the cache, usually called automatically in the run() method
        """
        cache_manager.clear_cache()
        print(f"[{self.name}] Cache cleared")