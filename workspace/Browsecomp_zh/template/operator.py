import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union

from tenacity import retry, stop_after_attempt, wait_fixed

from base.engine.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.Browsecomp_zh.template.operator_an import *
from workspace.Browsecomp_zh.template.op_prompt import *
from base.engine.async_llm import AsyncLLM
from base.engine.logs import logger
from base.tool.page_cache_manager import get_cache_manager, PageCacheManager
import re
import openai


from base.operator.operators import Operator


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response
    
class AnswerGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
        super().__init__(llm, name)

    async def __call__(self, input: str, mode: str = None) -> Tuple[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
        return response
    
class AgentGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "AgentGenerate", stop: Optional[List[str]] = None):
        super().__init__(llm, name)
        self.stop = stop
    
    async def __call__(self, input: Union[str, list]) -> Tuple[str, str]:
        if self.stop is not None: 
            response = await self._fill_node(AgentGenerateOp, input, mode="xml_fill", stop=self.stop)
        else:
            response = await self._fill_node(AgentGenerateOp, input, mode="xml_fill")
        return response

class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str]):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}
    

class ReSum(Operator):
    """
    Paper: ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization
    Link: https://arxiv.org/abs/2509.13313
    """

    def __init__(self, llm: AsyncLLM, name: str = "ReSum"):
        super().__init__(llm, name)

    async def __call__(self, question: str, system_prompt: str, history: List[str], cache_manager: Optional[PageCacheManager] = None) -> Tuple[List[str], int]:

        try:
            prompt = RESUM_PROMPT.format(question=question, history=history)
            
            # Pre-emptive check for context length to avoid API error logs
            # Estimate token count (rough approximation: 4 chars per token)
            # If prompt is excessively long (e.g. > 100k tokens ~ 400k chars), truncate immediately
            if len(prompt) > 400000:
                logger.warning(f"ReSum input is too long ({len(prompt)} chars), truncating before API call.")
                history_str = str(history)
                # Keep start and end to preserve context structure
                truncated_history = history_str[:100000] + "\n...[History Truncated due to Length]...\n" + history_str[-100000:]
                prompt = RESUM_PROMPT.format(question=question, history=truncated_history)
            
            response = await self._fill_node(ResumOp, prompt, mode="xml_fill")
        except openai.BadRequestError as e:
            if getattr(e, "code", None) == "context_length_exceeded" or "context length" in str(e).lower() or e.status_code == 400:
                logger.error(f"Context length exceeded in ReSum: {e}")
                history_str = str(history)
                if len(history_str) > 100000:
                    history_str = history_str[:50000] + "\n...[History Truncated due to Length]...\n" + history_str[-50000:]
                prompt = RESUM_PROMPT.format(question=question, history=history_str)
                response = await self._fill_node(ResumOp, prompt, mode="xml_fill")
        except Exception as e:
            logger.error(f"Error in ReSum: {e}")
            logger.error(traceback.format_exc())
            return history, 0
        summary = response.get("Summary", "")
        summary_conditioned_reasoning_prompt = f"""{question}
Below is a summary of the previous conversation. This summary condenses key information from earlier steps, so please consider it carefully. Assess whether the summary provides enough information to answer the question and use it as the basis for further reasoning and information gathering to answer the question.
Summary: {summary}
"""
        # Clear cache when summarizing history, as the context is being reset
        if cache_manager is None:
            cache_manager = get_cache_manager()
            
        task_id = cache_manager._get_task_id()
        cache_size_before = cache_manager.get_cache_size(task_id)
        cache_manager.clear_cache()
        task_type = "asyncio task" if task_id.startswith("asyncio_task_") else "thread"
        logger.info(f"[{self.name}] Cache cleared after summarization ({task_type}={task_id}, cleared {cache_size_before} pages)")
        
        history = [] # the history would totally be reset after the summarization
        history.append({"role": "system", "content": system_prompt})
        history.append({"role": "user", "content": summary_conditioned_reasoning_prompt})

        total_input_tokens = 0
        return history, total_input_tokens


class AgentFold(Operator):
    """
    Paper: AgentFold: Long-Horizon Web Agents with Proactive Context Management
    Link: https://arxiv.org/abs/2510.24699
    
    Proactive context management operator for long-horizon web agents.
    Implements multi-level context folding mechanism with:
    - Fine-grained condensation: Preserves key details while removing redundancy
    - Deep integration: Abstracts multi-step sub-tasks into high-level summaries
    - Automatic decision making: Determines when and how to fold context
    - Manual control: Allows caller to specify fold level and timing
    
    Input:
        - question (str): Current question/task
        - system_prompt (str): System prompt for the agent
        - history (List[str]): Conversation history (list of message dicts or strings)
        - fold_level (Optional[str]): Folding level - "fine-grained" (preserve details, remove redundancy), 
          "deep" (abstract to high-level summaries), "auto" (automatic decision), or None (full auto decision)
        - auto_threshold (Optional[int]): Threshold for automatic folding decision (number of history items, default: 20)
    
    Output:
        - List[dict]: Processed history list compatible with ReSum interface
          - If folding is performed: Returns new history with system prompt and folded context user message
          - If no folding needed: Returns original history unchanged
    
    Usage Examples:
        # Automatic mode (decides when and how to fold)
        history = await agentfold(question, system_prompt, history)
        
        # Manual mode - specify fold level
        history = await agentfold(question, system_prompt, history, fold_level="fine-grained")
        history = await agentfold(question, system_prompt, history, fold_level="deep")
        
        # Custom threshold
        history = await agentfold(question, system_prompt, history, auto_threshold=30)
    """

    def __init__(self, llm: AsyncLLM, name: str = "AgentFold", default_auto_threshold: int = 20):
        super().__init__(llm, name)
        self.default_auto_threshold = default_auto_threshold

    async def _should_fold_auto(self, question: str, history: List[str]) -> Tuple[bool, str]:
        """
        Automatically decide whether to fold and what level to use.
        Returns: (should_fold: bool, fold_level: str)
        """
        history_length = len(history) if isinstance(history, list) else 0
        
        # Format history for prompt
        history_str = str(history) if history else "[]"
        
        prompt = AGENTFOLD_DECISION_PROMPT.format(
            question=question,
            history_length=history_length,
            history=history_str
        )
        
        try:
            response = await self._fill_node(AgentFoldDecisionOp, prompt, mode="xml_fill")
            should_fold = response.get("should_fold", False)
            fold_level = response.get("fold_level", "auto")
            reasoning = response.get("reasoning", "")
            
            logger.info(f"AgentFold auto decision: should_fold={should_fold}, fold_level={fold_level}, reasoning={reasoning}")
            
            return should_fold, fold_level
        except Exception as e:
            logger.warning(f"Error in auto decision, defaulting to threshold-based: {str(e)}")
            # Fallback to threshold-based decision
            if history_length >= self.default_auto_threshold:
                return True, "fine-grained"
            return False, "auto"

    async def _decide_fold_level(self, question: str, history: List[str]) -> str:
        """
        Automatically decide the fold level when fold_level is "auto".
        Returns: "fine-grained" or "deep"
        """
        history_length = len(history) if isinstance(history, list) else 0
        history_str = str(history) if history else "[]"
        
        prompt = AGENTFOLD_DECISION_PROMPT.format(
            question=question,
            history_length=history_length,
            history=history_str
        )
        
        try:
            response = await self._fill_node(AgentFoldDecisionOp, prompt, mode="xml_fill")
            fold_level = response.get("fold_level", "fine-grained")
            
            # Normalize fold_level
            if fold_level.lower() == "auto":
                # If still auto, decide based on history length
                if history_length > 30:
                    return "deep"
                else:
                    return "fine-grained"
            
            return fold_level.lower()
        except Exception as e:
            logger.warning(f"Error deciding fold level, defaulting to fine-grained: {str(e)}")
            return "fine-grained"

    async def _fold_fine_grained(self, question: str, history: List[str]) -> str:
        """
        Perform fine-grained condensation: preserve key details while removing redundancy.
        Returns: folded context string
        """
        history_str = str(history) if history else "[]"
        
        prompt = AGENTFOLD_FINE_GRAINED_PROMPT.format(
            question=question,
            history=history_str
        )
        
        response = await self._fill_node(AgentFoldOp, prompt, mode="xml_fill")
        folded_context = response.get("folded_context", "")
        reasoning = response.get("reasoning", "")
        
        logger.info(f"AgentFold fine-grained folding completed. Reasoning: {reasoning}")
        
        return folded_context

    async def _fold_deep_integration(self, question: str, history: List[str]) -> str:
        """
        Perform deep integration: abstract multi-step sub-tasks into high-level summaries.
        Returns: folded context string
        """
        history_str = str(history) if history else "[]"
        
        prompt = AGENTFOLD_DEEP_INTEGRATION_PROMPT.format(
            question=question,
            history=history_str
        )
        
        response = await self._fill_node(AgentFoldOp, prompt, mode="xml_fill")
        folded_context = response.get("folded_context", "")
        reasoning = response.get("reasoning", "")
        
        logger.info(f"AgentFold deep integration completed. Reasoning: {reasoning}")
        
        return folded_context

    async def __call__(
        self, 
        question: str, 
        system_prompt: str, 
        history: List[str], 
        fold_level: Optional[str] = None,
        auto_threshold: Optional[int] = None
    ) -> List[dict]:
        """
        Execute context folding operation.
        
        Args:
            question: Current question/task
            system_prompt: System prompt for the agent
            history: Conversation history (list of message dicts or strings)
            fold_level: Folding level - "fine-grained", "deep", "auto", or None (auto decision)
            auto_threshold: Threshold for automatic folding decision (number of history items)
        
        Returns:
            Processed history list compatible with ReSum interface
        """
        # Use provided threshold or default
        threshold = auto_threshold if auto_threshold is not None else self.default_auto_threshold
        
        # Determine if we should fold
        should_fold = False
        actual_fold_level = None
        
        if fold_level is None:
            # Automatic decision mode
            should_fold, actual_fold_level = await self._should_fold_auto(question, history)
            
            # Also check threshold as a fallback
            history_length = len(history) if isinstance(history, list) else 0
            if history_length >= threshold:
                should_fold = True
                if actual_fold_level == "auto" or actual_fold_level is None:
                    actual_fold_level = await self._decide_fold_level(question, history)
        else:
            # Manual mode: caller decides when to fold
            should_fold = True
            actual_fold_level = fold_level.lower()
            
            # If fold_level is "auto", decide the actual level
            if actual_fold_level == "auto":
                actual_fold_level = await self._decide_fold_level(question, history)
        
        # Perform folding if needed
        if should_fold:
            if actual_fold_level == "fine-grained":
                folded_context = await self._fold_fine_grained(question, history)
            elif actual_fold_level == "deep":
                folded_context = await self._fold_deep_integration(question, history)
            else:
                # Default to fine-grained if level is unclear
                logger.warning(f"Unknown fold_level '{actual_fold_level}', defaulting to fine-grained")
                folded_context = await self._fold_fine_grained(question, history)
            
            # Create folded context prompt similar to ReSum
            folded_reasoning_prompt = f"""{question}
Below is a folded context from the previous conversation. This context has been proactively managed through {actual_fold_level} folding to preserve essential information while managing context size. Please consider it carefully and use it as the basis for further reasoning and information gathering to answer the question.

Folded Context: {folded_context}
"""
            
            # Reset history and add folded context (similar to ReSum behavior)
            new_history = []
            new_history.append({"role": "system", "content": system_prompt})
            new_history.append({"role": "user", "content": folded_reasoning_prompt})
            
            return new_history
        else:
            # No folding needed, return original history
            logger.info("AgentFold: No folding needed, returning original history")
            return history if isinstance(history, list) else []