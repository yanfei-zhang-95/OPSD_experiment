# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : Zhaoyang
# @Desc    : 

import asyncio
import openai
from openai import AsyncOpenAI
from base.engine.formatter import BaseFormatter, FormatError

import yaml
from pathlib import Path
from typing import Dict, Optional, Any

class LLMConfig:
    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 1)
        self.key = config.get("key", None)
        self.base_url = config.get("base_url", "https://oneapi.deepwisdom.ai/v1")
        self.top_p = config.get("top_p", 1)
        self.top_k = config.get("top_k", None)
        self.context_window = config.get("context_window", None)
        self.max_tokens = config.get("max_tokens", 8192)
        self.frequency_penalty = config.get("frequency_penalty", 0.0)
        self.presence_penalty = config.get("presence_penalty", 0.0)
        self.repetition_penalty = config.get("repetition_penalty", 1.0)
        self.enable_thinking = config.get("enable_thinking", False)
        self.extra_body = config.get("extra_body", None)
class LLMsConfig:
    """Configuration manager for multiple LLM configurations"""
    
    _instance = None  # For singleton pattern if needed
    _default_config = None
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with an optional configuration dictionary"""
        self.configs = config_dict or {}
    
    @classmethod
    def default(cls):
        """Get or create a default configuration from YAML file"""
        if cls._default_config is None:
            # Look for the config file in common locations
            # Prioritize the project root config based on file location
            project_root = Path(__file__).resolve().parent.parent.parent
            
            config_paths = [
                project_root / "config" / "config.yaml",
                Path("config/config.yaml"),
                Path("config.yaml"),
                Path("./config/config.yaml")
            ]
            
            config_file = None
            for path in config_paths:
                if path.exists():
                    config_file = path
                    break
            
            if config_file is None:
                raise FileNotFoundError("No default configuration file found in the expected locations")
            
            # Load the YAML file
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Your YAML has a 'models' top-level key that contains the model configs
            if 'models' in config_data:
                config_data = config_data['models']
                
            cls._default_config = cls(config_data)
        
        return cls._default_config
    
    def get(self, llm_name: str) -> LLMConfig:
        """Get the configuration for a specific LLM by name"""
        if llm_name not in self.configs:
            raise ValueError(f"Configuration for {llm_name} not found")
        
        config = self.configs[llm_name]
        
        # Create a config dictionary with the expected keys for LLMConfig
        llm_config = {
            "model": llm_name,  # Use the key as the model name
            "temperature": config.get("temperature", 1),
            "key": config.get("key"),  # Map api_key to key
            "base_url": config.get("base_url", "https://oneapi.deepwisdom.ai/v1"),
            "top_p": config.get("top_p", 1),  # Add top_p parameter
            "top_k": config.get("top_k", None),
            "context_window": config.get("context_window", None),  # context_window: 模型的最大输出token数量
            "max_tokens": config.get("max_tokens", 8192),
            "frequency_penalty": config.get("frequency_penalty", 0.0),
            "presence_penalty": config.get("presence_penalty", 0.0),
            "repetition_penalty": config.get("repetition_penalty", 1.1),
            "enable_thinking": config.get("enable_thinking", False),
            "extra_body": config.get("extra_body", None)
        }
        
        # Create and return an LLMConfig instance with the specified configuration
        return LLMConfig(llm_config)
    
    def add_config(self, name: str, config: Dict[str, Any]) -> None:
        """Add or update a configuration"""
        self.configs[name] = config
    
    def get_all_names(self) -> list:
        """Get names of all available LLM configurations"""
        return list(self.configs.keys())
    
class ModelPricing:
    """Pricing information for different models in USD per 1K tokens"""
    PRICES = {
        # GPT-4o models
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-sonnet-4": {"input": 0.0024, "output": 0.012},
        "claude-4-sonnet": {"input": 0.0024, "output": 0.012},
        "o3":{"input":0.003, "output":0.015},
        "o3-mini": {"input": 0.0011, "output": 0.0025},
    }
    
    @classmethod
    def get_price(cls, model_name, token_type):
        """Get the price per 1K tokens for a specific model and token type (input/output)"""
        # Try to find exact match first
        if model_name in cls.PRICES:
            return cls.PRICES[model_name][token_type]
        
        # Try to find a partial match (e.g., if model name contains version numbers)
        for key in cls.PRICES:
            if key in model_name:
                return cls.PRICES[key][token_type]
        
        # Return default pricing if no match found
        return 0

class TokenUsageTracker:
    """Tracks token usage and calculates costs"""
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.usage_history = []
    
    def add_usage(self, model, input_tokens, output_tokens):
        """Add token usage for a specific API call"""
        input_cost = (input_tokens / 1000) * ModelPricing.get_price(model, "input")
        output_cost = (output_tokens / 1000) * ModelPricing.get_price(model, "output")
        total_cost = input_cost + output_cost
        
        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "prices": {
                "input_price": ModelPricing.get_price(model, "input"),
                "output_price": ModelPricing.get_price(model, "output")
            }
        }
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        self.usage_history.append(usage_record)
        
        return usage_record
    
    def get_summary(self):
        """Get a summary of token usage and costs"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "call_count": len(self.usage_history),
            "history": self.usage_history
        }
    
    def get_last_usage(self):
        """Get the token usage information for the last API call"""
        if not self.usage_history:
            return None
        return self.usage_history[-1]

class AsyncLLM:
    def __init__(self, config, system_msg:str = None):
        """
        Initialize the AsyncLLM with a configuration
        
        Args:
            config: Either an LLMConfig instance or a string representing the LLM name
                   If a string is provided, it will be looked up in the default configuration
            system_msg: Optional system message to include in all prompts
        """
        # Handle the case where config is a string (LLM name)
        if isinstance(config, str):
            llm_name = config
            config = LLMsConfig.default().get(llm_name)
        
        # At this point, config should be an LLMConfig instance
        self.config = config
        self.aclient = AsyncOpenAI(api_key=self.config.key, base_url=self.config.base_url)
        self.sys_msg = system_msg
        self.usage_tracker = TokenUsageTracker()
        self._closed = False
        self._recovery_lock = asyncio.Lock()
        
    async def __call__(self, prompt, **kwargs):
        message = []
        if self.sys_msg is not None:
            message.append({
                "content": self.sys_msg,
                "role": "system"
            })

        message.append({"role": "user", "content": prompt})

        # Prepare API call parameters
        api_params = {
            "model": self.config.model,
            "messages": message,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
        }
        
        # Merge kwargs into api_params, allowing overrides
        api_params.update(kwargs)
        
        # Handle parameters that need to go into extra_body for vLLM/SGLang
        # Copy to avoid mutating the config reference
        extra_body = dict(getattr(self.config, "extra_body", {}) or {})
        
        # Add top_k to extra_body if configured (standard OpenAI API doesn't support top_k)
        if hasattr(self.config, "top_k") and self.config.top_k is not None and self.config.top_k > 0:
            extra_body["top_k"] = self.config.top_k

        # Add repetition_penalty to extra_body if configured (standard OpenAI API doesn't support repetition_penalty)
        if hasattr(self.config, "repetition_penalty") and self.config.repetition_penalty is not None:
            extra_body["repetition_penalty"] = self.config.repetition_penalty
        
        # Push enable_thinking into chat_template_kwargs to avoid unsupported top-level args
        # We must pass it even if False, to explicitly disable thinking if the model defaults to True
        if hasattr(self.config, "enable_thinking") and self.config.enable_thinking is not None:
            chat_template_kwargs = extra_body.get("chat_template_kwargs", {})
            if isinstance(chat_template_kwargs, dict):
                chat_template_kwargs["enable_thinking"] = self.config.enable_thinking
                extra_body["chat_template_kwargs"] = chat_template_kwargs
            
        if extra_body:
            api_params["extra_body"] = extra_body
               
        response = None
        model_candidates = self._build_model_candidates(api_params.get("model"))
        max_retry = 4
        for model in model_candidates:
            api_params["model"] = model
            for retry_idx in range(max_retry + 1):
                try:
                    client_to_use = self.aclient
                    response = await client_to_use.chat.completions.create(**api_params)
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    is_rate_limit = isinstance(e, openai.RateLimitError) or "429" in error_msg
                    is_connection_error = (
                        isinstance(e, openai.APIConnectionError)
                        or isinstance(e, openai.APITimeoutError)
                        or "connection error" in error_msg
                        or "connection closed" in error_msg
                    )
                    is_internal_error = (
                        isinstance(e, openai.InternalServerError)
                        or "internal server error" in error_msg
                        or "hosted_vllmexception" in error_msg
                        or "service unavailable" in error_msg
                    )
                    if isinstance(e, openai.BadRequestError) and "content exists risk" in error_msg:
                        print("⚠️  Content Safety Policy triggered (Content Exists Risk). Returning empty response.")
                        return ""
                    if is_rate_limit or is_connection_error:
                        async with self._recovery_lock:
                            if self.aclient != client_to_use:
                                await asyncio.sleep(1)
                                continue
                            await self.aclose()
                            await asyncio.sleep(15)
                            self.aclient = AsyncOpenAI(api_key=self.config.key, base_url=self.config.base_url)
                            self._closed = False
                        continue
                    if is_internal_error and retry_idx < max_retry:
                        await asyncio.sleep(min(2 ** retry_idx, 10))
                        continue
                    is_route_or_model_error = (
                        "received model group" in error_msg
                        or "model group" in error_msg
                        or "model not found" in error_msg
                        or "unknown model" in error_msg
                    )
                    if is_route_or_model_error:
                        break
                    raise e
            if response is not None:
                break
        if response is None:
            raise RuntimeError(f"LLM request failed for all model candidates: {model_candidates}")

        # Extract token usage from response
        usage = getattr(response, "usage", None)
        input_tokens = 0
        output_tokens = 0
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
        
        # Track token usage and calculate cost
        usage_record = self.usage_tracker.add_usage(
            self.config.model,
            input_tokens,
            output_tokens
        )
        
        ret = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        
        # Check if response was truncated due to max_tokens limit
        if finish_reason == "length":
            print(f"⚠️  WARNING: Response was truncated due to max_tokens limit ({self.config.max_tokens}).")
            print(f"   Consider setting max_tokens in config.yaml to a larger value (e.g., 4096, 8192)")
        
        print(ret)
        print(f"Repetition penalty: {self.config.repetition_penalty}")
        print(f"Frequency penalty: {self.config.frequency_penalty}")
        print(f"Presence penalty: {self.config.presence_penalty}")
        
        # # You can optionally print token usage information
        # print(f"Token usage: {input_tokens} input + {output_tokens} output = {input_tokens + output_tokens} total")
        # print(f"Cost: ${usage_record['total_cost']:.6f} (${usage_record['input_cost']:.6f} for input, ${usage_record['output_cost']:.6f} for output)")
        
        return ret

    def _build_model_candidates(self, model: Any):
        if not isinstance(model, str):
            return [model]
        candidates = []
        def _append(val):
            if val and val not in candidates:
                candidates.append(val)
        _append(model)
        normalized = model.lstrip("/")
        _append(normalized)
        if normalized.startswith("data/"):
            _append(normalized[len("data/"):])
        _append(Path(normalized).name)
        return candidates
    
    async def call_with_format(self, prompt: str, formatter: BaseFormatter, **kwargs):
        """
        Call the LLM with a prompt and format the response using the provided formatter
        
        Args:
            prompt: The prompt to send to the LLM
            formatter: An instance of a BaseFormatter to validate and parse the response
            **kwargs: Additional arguments to pass to the LLM call (e.g., stop tokens)
            
        Returns:
            The formatted response data
            
        Raises:
            FormatError: If the response doesn't match the expected format
        """
        # Prepare the prompt with formatting instructions
        formatted_prompt = formatter.prepare_prompt(prompt)
        # Call the LLM
        response = await self.__call__(formatted_prompt, **kwargs)
        
        # Validate and parse the response
        is_valid, parsed_data = formatter.validate_response(response)
        
        if not is_valid:
            error_message = formatter.format_error_message()
            raise FormatError(f"{error_message}. Raw response: {response}")
        
        return parsed_data
    
    def get_usage_summary(self):
        """Get a summary of token usage and costs"""
        return self.usage_tracker.get_summary()
    
    def get_last_usage(self):
        """Get the token usage information for the last API call"""
        return self.usage_tracker.get_last_usage()

    async def aclose(self) -> None:
        """
        Close underlying network resources.

        Without an explicit close, the AsyncOpenAI/httpx client may be finalized after
        the event loop is already closed, producing noisy but mostly harmless warnings:
        "Task exception was never retrieved ... RuntimeError('Event loop is closed')".
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        aclient = getattr(self, "aclient", None)
        if aclient is None:
            return
        try:
            # openai.AsyncOpenAI.close is async in openai>=1.x
            await aclient.close()
        except Exception:
            # Best-effort close; avoid raising during shutdown/cleanup.
            pass

    async def __aenter__(self) -> "AsyncLLM":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
    

def create_llm_instance(llm_config):
    """
    Create an AsyncLLM instance using the provided configuration
    
    Args:
        llm_config: Either an LLMConfig instance, a dictionary of configuration values,
                            or a string representing the LLM name to look up in default config
    
    Returns:
        An instance of AsyncLLM configured according to the provided parameters
    """
    # Case 1: llm_config is already an LLMConfig instance
    if isinstance(llm_config, LLMConfig):
        return AsyncLLM(llm_config)
    
    # Case 2: llm_config is a string (LLM name)
    elif isinstance(llm_config, str):
        return AsyncLLM(llm_config)  # AsyncLLM constructor handles lookup
    
    # Case 3: llm_config is a dictionary
    elif isinstance(llm_config, dict):
        # Create an LLMConfig instance from the dictionary
        llm_config = LLMConfig(llm_config)
        return AsyncLLM(llm_config)
    
    else:
        raise TypeError("llm_config must be an LLMConfig instance, a string, or a dictionary")
