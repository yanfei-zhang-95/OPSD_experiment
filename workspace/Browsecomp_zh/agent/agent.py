import asyncio
from typing import Optional, Dict, Any, List
import workspace.Browsecomp_zh.template.operator as operator
from base.engine.async_llm import create_llm_instance

import workspace.Browsecomp_zh.agent.prompt_agent as prompt_agent

import traceback
import json

import traceback
from base.utils.utils import _parse_json_strict, _dispatch_tool, extract_xml
from base.base_agent import BaseAgent
from base.tool import WebExplorerSearch, WebExplorerOpen, WebExplorerFind, GetFinalAnswerAnswerer
from base.tool.page_cache_manager import PageCacheManager, get_cache_manager
from base.engine.async_llm import AsyncLLM
import openai
import tiktoken
import traceback

class Workflow:
    def __init__(self, 
        llm_config: Optional[Dict[str, Any]] = None, 
        **kwargs):
        self.llm_config = llm_config
        
        # Create LLM instance
        llm = create_llm_instance(llm_config) if llm_config else None
        self.llm = llm
        
        # Initialize operators
        self.agentgenerate = operator.AgentGenerate(self.llm)
        self.custom = operator.Custom(self.llm)
        self.answergenerate = operator.AnswerGenerate(self.llm)
        self.scensemble = operator.ScEnsemble(self.llm)

        self.kwargs = kwargs

    async def run(self, request: str) -> Dict[str, Any]:
        # Workflow run delegates to the agent
        # In the future, this can coordinate multiple agents
        max_steps = self.kwargs.get("max_steps", 40)
        # 移除 kwargs 中的 max_steps，避免重复传递
        agent_kwargs = {k: v for k, v in self.kwargs.items() if k != "max_steps"}
        agent = self.ReActAgent(
            system_prompt=prompt_agent.INITIAL_REACT_AGENT_PROMPT, 
            max_steps=max_steps,
            llm=self.llm, 
            **agent_kwargs
        )
        return await agent.run(request)

    async def __call__(self, request: str, **kwargs) -> Dict[str, Any]:
        """Make Workflow callable for compatibility with evaluation code."""
        return await self.run(request)

    class ReActAgent(BaseAgent):
        """
        React: Reasoning + Acting
        """

        class Config:
            arbitrary_types_allowed = True
            extra = "allow"
        
        def __init__(
            self,
            name: str = "ReactAgent",
            description: str = "A simple React agent that thinks and acts iteratively to answer questions",
            llm: Optional[AsyncLLM] = None,
            max_steps: int = 50,
            system_prompt: str = prompt_agent.INITIAL_REACT_AGENT_PROMPT,
            cache_manager: Optional[PageCacheManager] = None,
            **kwargs
        ):

            # Initialize base class
            super().__init__(
                name=name,
                description=description,
                system_prompt=system_prompt,
                llm=llm,
                max_steps=max_steps,
                **kwargs
            )
            
            # Initialize operators
            self.agentgenerate = operator.AgentGenerate(self.llm)
            self.custom = operator.Custom(self.llm)
            self.answergenerate = operator.AnswerGenerate(self.llm)
            self.scensemble = operator.ScEnsemble(self.llm)

            # Initialize context window
            self.context_window = kwargs.get("context_window") or (self.llm.config.context_window if self.llm else None)
            self.max_token_len = (self.llm.config.max_tokens if self.llm else 8192)
            
            # Initialize cache manager
            # Use provided manager or create a NEW isolated instance (not global singleton)
            self.cache_manager = cache_manager or PageCacheManager()
            
            # Get tool configuration from kwargs
            tool_cfg = kwargs.get("tool_cfg", {})
            self._tool_cfg = tool_cfg  # Store for use in run() method
            
            # Initialize tools with the specific cache manager and configuration
            self.tools = {
                "search": WebExplorerSearch(cache_manager=self.cache_manager, cfg=tool_cfg.get("search", {})),
                "open": WebExplorerOpen(cache_manager=self.cache_manager, cfg=tool_cfg.get("open", {})),
                "find": WebExplorerFind(cache_manager=self.cache_manager, cfg=tool_cfg.get("find", {})),
                "get_final_answer": GetFinalAnswerAnswerer()
            }
            
            # Generate tool specifications
            self.tool_specs = "\n".join([json.dumps(tool.to_param()) for tool in self.tools.values()])
            self.tool_usage = "\n".join([tool.tool_usage for tool in self.tools.values()])
            self.tool_usage_workflow = "\n".join([tool.tool_usage_workflow for tool in self.tools.values()])

        def system_prompt_formatter(self, system_prompt: str, tools: Dict[str, Any]) -> str:
            """
            Format the system prompt with tool specifications.
            Overridden to include thought_tag handling.
            """
            tool_specs = "\n".join([json.dumps(tool.to_param()) for tool in tools.values()])
            tool_usage = "\n".join([tool.tool_usage for tool in tools.values()])
            tool_usage_workflow = "\n".join([tool.tool_usage_workflow for tool in tools.values()])
            
            # Determine thought tag based on enable_thinking config
            enable_thinking = False
            if self.llm and hasattr(self.llm, "config"):
                if hasattr(self.llm.config, "enable_thinking"):
                    enable_thinking = self.llm.config.enable_thinking
                elif isinstance(self.llm.config, dict):
                    enable_thinking = self.llm.config.get("enable_thinking", False)
            
            thought_tag = "think" if enable_thinking else "thought"
            
            return system_prompt.format(
                tool_specs=tool_specs, 
                tool_usage=tool_usage, 
                tool_usage_workflow=tool_usage_workflow,
                thought_tag=thought_tag
            )

        async def step(self, history: List[Dict[str, str]], tools: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """
            Execute a step: think -> tool call -> result
            
            Args:
                history: Current conversation history
                tools: Optional dictionary of tools to use for this step. 
                       If None, uses self.tools (which might not be thread-safe/isolated).
            
            Returns:
                A dictionary containing thought, tool_call, and results
            """
            
            # Use provided tools or fall back to default (though default is discouraged in this new architecture)
            current_tools = tools or self.tools
            
            enable_thinking = False
            if self.llm and hasattr(self.llm, "config"):
                if hasattr(self.llm.config, "enable_thinking"):
                    enable_thinking = self.llm.config.enable_thinking
                elif isinstance(self.llm.config, dict):
                    enable_thinking = self.llm.config.get("enable_thinking", False)

            # Special handling for qwen3 models: force append <think></think> to prevent unwanted output if needed
            # Only do this if thinking is disabled (shielding mode)
            if not enable_thinking:
                if self.llm and self.llm.config.model and "qwen3" in self.llm.config.model.lower():
                    # Check if the last message is from user and doesn't already have the tag
                    history[-1]["content"] += "\n<think>\n\n</think>\n\n"

            # Convert history to string format
            history_str = self.convert_history_to_string(history)
            
            # Use AgentGenerate operator to generate response
            try:
                solution = await self.agentgenerate(history_str)
            except openai.APIConnectionError as e:
                print(f"Connection error in agent step: {e}")
                print(traceback.format_exc())
                # Return None so the run loop can decide whether to continue or retry
                return None
            except openai.BadRequestError as e:
                status_code = getattr(e, "status_code", None)
                if getattr(e, "code", None) == "context_length_exceeded" or "context length" in str(e).lower() or status_code == 400:
                    return {"error": "context_length_exceeded", "message": str(e)}
                print(f"BadRequestError in agent step: {e}")
                return None
            except Exception as e:
                # Catch-all for other exceptions, including litellm specific errors or other library errors
                error_msg = str(e).lower()
                if "context_length_exceeded" in error_msg or "context length" in error_msg or "contextwindowexceedederror" in error_msg or "maximum context length" in error_msg:
                    return {"error": "context_length_exceeded", "message": str(e)}
                
                print(f"Error in agent step: {e}")
                return None
            
            # Get token usage information from the last LLM call
            last_usage = self.agentgenerate.llm.get_last_usage()
            input_tokens = last_usage.get("input_tokens", 0) if last_usage else 0
            output_tokens = last_usage.get("output_tokens", 0) if last_usage else 0
            
            # Extract thought and tool_calls based on mode
            if enable_thinking:
                thought = solution.get("think", "")
            else:
                thought = solution.get("thought", "")
                
            tool_call_str = solution.get("tool_call", "")
            
            # Parse tool calls
            try:
                tool_call = _parse_json_strict(tool_call_str)
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
            except Exception as e:
                print(f"Failed to parse tool calls: {e}")
                return
            
            # Call tools
            try:
                results = await _dispatch_tool(tool_name, tool_args, current_tools)
            except Exception as e:
                results = f"Error executing tool: {str(e)}"
            
            # Update history with correct tag
            thought_tag = "think" if enable_thinking else "thought"
            content = f"<{thought_tag}>{thought}</{thought_tag}>\n<tool_call>{tool_call_str}</tool_call>\n<tool_response>{results}</tool_response>"
            history.append({"role": "assistant", "content": content})
            
            return {
                "thought": thought,
                "tool_call": tool_call,
                "results": results,
                "history": history,
                "total_tokens": output_tokens + input_tokens
            }
        
        async def run(self, request: str) -> Dict[str, Any]:
            """
            Execute the core reasoning loop as an independent agent session.
            Each call creates a fresh environment (cache, history) for the task.
            
            Args:
                request: The user query to solve
                system_prompt: Optional override for the system prompt. If None, uses self.system_prompt
            """
            # Determine which system prompt to use
            current_system_prompt = self.system_prompt

            # Create a fresh cache manager for this specific agent execution session
            # This ensures complete isolation for this specific request
            session_cache_manager = PageCacheManager()
            
            # Get tool configuration (reuse from self.tools config)
            tool_cfg = getattr(self, '_tool_cfg', {})
            
            # Initialize tools with this session's cache manager and configuration
            # We need to recreate tools here to bind them to the new cache manager
            session_tools = {
                "search": WebExplorerSearch(cache_manager=session_cache_manager, cfg=tool_cfg.get("search", {})),
                "open": WebExplorerOpen(cache_manager=session_cache_manager, cfg=tool_cfg.get("open", {})),
                "find": WebExplorerFind(cache_manager=session_cache_manager, cfg=tool_cfg.get("find", {})),
                "get_final_answer": self.tools["get_final_answer"]
            }
            
            # Initialize history
            histories = []
            history = []
            current_step = 0
            
            # Format the system prompt using the session's specific tools and the chosen prompt template
            # Uses the overridden formatter
            formatted_system_prompt = self.system_prompt_formatter(current_system_prompt, session_tools)
            
            # Add initial user question formatted by system prompt
            history.append({"role": "system", "content": formatted_system_prompt})
            history.append({"role": "user", "content": request})
            
            answer = "No answer found"
            consecutive_failures = 0
            
            print(f"[ReActAgent] Starting task, max_steps={self.max_steps}")
            
            # Main loop
            while current_step < self.max_steps:
                current_step += 1
                
                # Execute one step with session-specific tools
                response = await self.step(history, tools=session_tools)

                if response is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        answer = "Error: LLM backend failed repeatedly"
                        break
                    await asyncio.sleep(min(consecutive_failures, 3))
                    continue
                
                consecutive_failures = 0

                # Update history
                history = response.get("history")

                # Check if completed
                tool_call = response.get("tool_call")
                if tool_call is None:
                    print(f"[Step {current_step}] tool_call is None, continuing...")
                    continue
                
                if tool_call.get("name") == "get_final_answer":
                    answer = tool_call.get("args", {}).get("final_answer", "")
                    if answer == "":
                        print(f"[Step {current_step}] get_final_answer returned empty answer, continuing...")
                        continue # continue the loop because the answer returned was not correct
                    else:
                        print(f"[Step {current_step}] Final answer received, breaking loop.")
                        break # break the loop to return the answer

            if current_step >= self.max_steps:
                answer = "Max step reached, no answer found."
            
            histories.append(history)
            compressed_history = {"initial_agent": histories}
            
            print(f"[ReActAgent] Task completed. Total steps: {current_step}, answer: {answer[:100]}...")
                
            return {
                "answer": answer,
                "history": compressed_history,
            }
