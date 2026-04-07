import json
import os
import http.client
import time
import random
from typing import List, Union, Optional, Dict, Any
import requests
from urllib.parse import quote
from .page_cache_manager import get_cache_manager, SearchResult
from .base_tool import BaseTool


class GetFinalAnswerAnswerer(BaseTool):
    name: str = "get_final_answer"
    tool_usage: str = "**get_final_answer**: Returns the final answer to the question"
    tool_usage_workflow: str = """- get_final_answer({"final_answer": "The final answer to the question"}) → Returns the final answer to the question"""
    description: str = "Tool to get the final answer to the question"
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "final_answer": {
                "type": "string",
                "description": "The final answer to the question"
            },
        },
        "required": ["final_answer"],
    }


    def __init__(self, 
                 custom_tool_usage: Optional[str] = None,
                 custom_tool_usage_workflow: Optional[str] = None,
                 custom_description: Optional[str] = None,
                 custom_final_answer_description: Optional[str] = None,
                 **data):
        super().__init__(**data)
        
        # Allow customization of tool descriptions for different agent types
        if custom_tool_usage:
            self.tool_usage = custom_tool_usage
        if custom_tool_usage_workflow:
            self.tool_usage_workflow = custom_tool_usage_workflow
        if custom_description:
            self.description = custom_description
        if custom_final_answer_description:
            self.parameters["properties"]["final_answer"]["description"] = custom_final_answer_description

    async def __call__(self, **kwargs) -> str:
        params = kwargs
        try:
            final_answer = params["final_answer"]
        except:
            return "[GetFinalAnswer] Invalid request format: Input must be a JSON object containing 'final_answer' field"
        
        try:
            return final_answer
        except Exception as e:
            return f"[GetFinalAnswer] Error: {str(e)}"