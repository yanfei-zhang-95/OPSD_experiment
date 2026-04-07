# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : Zhaoyang
# @Desc    : 

from typing import Dict, List, Tuple, Type, Optional, Union, Any

from pydantic import BaseModel, Field, create_model
import re

from abc import ABC, abstractmethod

from base.engine.sanitize import sanitize

class FormatError(Exception):
    """Exception raised when response format validation fails"""
    pass

class BaseFormatter(BaseModel):
    """Base class for all formatters"""
    
    @abstractmethod
    def prepare_prompt(self, prompt: str) -> str:
        """Prepare the prompt to instruct the LLM to return in the required format"""
        pass
    
    @abstractmethod
    def validate_response(self, response: str) -> Tuple[bool, Any]:
        """Validate if the response matches the expected format"""
        pass

    def format_error_message(self) -> str:
        """Return an error message for invalid format"""
        return f"Response did not match the expected {self.__class__.__name__} format"

class XmlFormatter(BaseFormatter):
    """Formatter for XML responses"""
    model: Optional[Type[BaseModel]] = None
    fields: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, fields_dict: Dict[str, str]) -> "XmlFormatter":
        """
        Create formatter from a dictionary of field names and descriptions
        
        Args:
            fields_dict: Dictionary where keys are field names and values are field descriptions
            
        Returns:
            An XmlFormatter instance configured with the specified fields
        """
        model_fields = {}
        for name, desc in fields_dict.items():
            model_fields[name] = (str, Field(default="", description=desc))
        
        model_class = create_model("XmlResponseModel", **model_fields)
        
        return cls(model=model_class)
    
    @classmethod
    def from_model(cls, model_class: Type[BaseModel]) -> "XmlFormatter":
        """
        Create formatter from an existing Pydantic model class
        
        Args:
            model_class: A Pydantic model class
            
        Returns:
            An XmlFormatter instance configured with the model's fields
        """
        return cls(model=model_class)
    
    def _get_field_names(self) -> List[str]:
        """Get field names from the model"""
        if self.model:
            return list(self.model.model_fields.keys())
        return []
    
    def _get_field_description(self, field_name: str) -> str:
        """Get field description from the model"""
        if self.model and field_name in self.model.model_fields:
            return self.model.model_fields[field_name].description
        return ""    
    
    def prepare_prompt(self, prompt: str) -> str:
        examples = []
        for field_name in self._get_field_names():
            description = self._get_field_description(field_name)
            # 使用占位符格式，明确告诉模型这是示例格式，需要替换为实际内容
            # 避免模型直接复制描述文本
            example_content = f"[Fill in: {description}]"
            examples.append(f"<{field_name}>{example_content}</{field_name}>")

        example_str = "\n".join(examples)
        
        # 添加更明确的格式说明，强调必须闭合所有标签
        format_instructions = (
            "\n# Response format (must be strictly followed) (do not include any other formats except for the given XML format):\n"
            f"{example_str}\n"
            "\nIMPORTANT: "
            "1. You MUST close ALL XML tags properly. Every opening tag <tag> must have a corresponding closing tag </tag>. "
            "2. Replace '[Fill in: ...]' placeholders with your ACTUAL content. Do NOT copy the placeholder text literally. "
            "3. Generate your own content based on the field description provided in the placeholder. "
            "4. Do not leave any tags unclosed. Make sure your response is valid XML with all tags properly closed."
        )
        
        instructions = prompt + format_instructions
        return instructions
    
    def validate_response(self, response: str) -> Tuple[bool, dict]:
        """Validate if the response contains all required fields in XML format"""
        try:
            # 自动修复未闭合的标签：检测所有已打开但未闭合的标签
            response = self._fix_unclosed_tags(response)
            
            # 尝试自动补齐缺失的 XML 标签
            # response = self._fix_missing_tags(response)
            
            pattern = r"<(\w+)>(.*?)</\1>"
            matches = re.findall(pattern, response, re.DOTALL)
            
            found_fields = {match[0]: match[1].strip() for match in matches}
            
            for field_name in self._get_field_names():
                field = self.model.model_fields[field_name]
                is_required = field.default is None and field.default_factory is None
                
                if is_required and (field_name not in found_fields or not found_fields[field_name]):
                    raise FormatError(f"Field '{field_name}' is missing or empty.")

            return True, found_fields
        except Exception:
            return False, None
    
    def _fix_missing_tags(self, response: str) -> str:
        """
        自动补齐缺失的 XML 标签
        检测响应中是否有字段内容但没有对应的 XML 标签，自动包裹
        
        策略：
        1. 找到所有已有的 XML 标签
        2. 对于缺失的字段，在最后一个标签之后查找可能的内容
        3. 如果内容是 JSON 格式且字段名是 tool_call，自动包裹
        4. 如果内容是文本且字段名是 thought，自动包裹
        """
        pattern = r"<(\w+)>(.*?)</\1>"
        matches = list(re.finditer(pattern, response, re.DOTALL))
        found_tags = {match.group(1) for match in matches}
        
        # 获取所有需要的字段名
        required_fields = set(self._get_field_names())
        missing_fields = required_fields - found_tags
        
        if not missing_fields:
            return response
        
        # 找到最后一个标签的结束位置
        last_tag_end = matches[-1].end() if matches else 0
        
        # 获取最后一个标签之后的内容
        remaining_content = response[last_tag_end:].strip()
        
        if not remaining_content:
            return response
        
        # 对于每个缺失的字段，尝试匹配并包裹内容
        # 优先处理 tool_call（通常是 JSON）
        for field_name in sorted(missing_fields, key=lambda x: (x != "tool_call", x)):
            if field_name == "tool_call" and self._looks_like_json(remaining_content):
                # tool_call 字段且内容是 JSON，自动包裹
                wrapped_content = f"<{field_name}>{remaining_content}</{field_name}>"
                response = response[:last_tag_end] + "\n" + wrapped_content
                return response  # 处理完一个字段后返回，避免重复处理
            elif field_name == "thought" and remaining_content and not self._looks_like_json(remaining_content):
                # thought 字段且内容不是 JSON，自动包裹
                wrapped_content = f"<{field_name}>{remaining_content}</{field_name}>"
                response = response[:last_tag_end] + "\n" + wrapped_content
                return response
        
        return response
    
    def _looks_like_json(self, text: str) -> bool:
        """检查文本是否看起来像 JSON 格式"""
        text = text.strip()
        # 检查是否以 { 开头和 } 结尾（简单 JSON 对象）
        if text.startswith("{") and text.endswith("}"):
            try:
                import json
                json.loads(text)
                return True
            except:
                pass
        # 检查是否以 [ 开头和 ] 结尾（JSON 数组）
        if text.startswith("[") and text.endswith("]"):
            try:
                import json
                json.loads(text)
                return True
            except:
                pass
        return False
    
    def _fix_unclosed_tags(self, response: str) -> str:
        """
        自动修复未闭合的XML标签
        使用栈来跟踪标签的嵌套关系，确保正确闭合所有标签
        """
        # 使用栈来跟踪打开的标签
        tag_stack = []
        
        # 匹配所有标签（包括自闭合标签）
        # 匹配格式：<tag> 或 <tag attr="value"> 或 </tag> 或 <tag/>
        tag_pattern = r"<(/?)(\w+)(?:\s[^>]*)?(/?)>"
        
        # 找到所有标签
        matches = list(re.finditer(tag_pattern, response))
        
        # 遍历所有标签，构建标签栈
        for match in matches:
            is_close = match.group(1) == "/"
            tag_name = match.group(2)
            is_self_closing = match.group(3) == "/"
            
            if is_self_closing:
                # 自闭合标签，不需要处理
                continue
            elif is_close:
                # 闭合标签：从栈中移除匹配的打开标签
                if tag_stack and tag_stack[-1] == tag_name:
                    tag_stack.pop()
            else:
                # 打开标签：压入栈
                tag_stack.append(tag_name)
        
        # 在响应末尾添加所有未闭合的标签（按相反顺序）
        if tag_stack:
            missing_closes = "".join([f"</{tag}>" for tag in reversed(tag_stack)])
            response = response + missing_closes
        
        return response

class CodeFormatter(BaseFormatter):
    """
    Formatter for extracting and sanitizing code from LLM responses.
    Handles both markdown code blocks and raw code responses.
    """
    
    function_name: Optional[str] = None
    
    def prepare_prompt(self, prompt: str) -> str:
        """
        Prepare the prompt to instruct the LLM to return code in a proper format.
        
        Args:
            prompt: The original prompt
            
        Returns:
            The prompt with instructions to return code in markdown format
        """
        # Instructions to return code in appropriate format
        code_instructions = (
            "\n\n"
            "Please write your code solution in Python. "
            "Return ONLY the complete, runnable code without explanations. "
            "Use proper Python syntax and formatting. "
        )

        # Add function-specific instructions if function_name is provided
        if self.function_name:
            code_instructions += (
                f"\nMake sure to include a function named '{self.function_name}' in your solution. "
                f"This function will be the entry point for the program."
            )
        
        return prompt + code_instructions
    
    def validate_response(self, response: str) -> Tuple[bool, Union[Dict[str, str], str, None]]:
        """
        Extract code from response and validate it.
        
        Args:
            response: The LLM response
            
        Returns:
            A tuple with (is_valid, extracted_code)
        """
        try:
            # First try to extract code from markdown code blocks
            code = self._extract_code_from_markdown(response)
    
            # If no code blocks found, treat the entire response as code
            if not code:
                code = response
            
            # Use the sanitize function to extract valid code and handle dependencies
            sanitized_code = sanitize(code=code, entrypoint=self.function_name)
            
            # If sanitize returned empty string, the code is invalid
            if not sanitized_code.strip():
                return False, None
            
            # Return the sanitized code
            result = {"response": sanitized_code}
            return True, result
            
        except Exception as e:
            # Return the error information
            return False, {"error": str(e)}
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """
        Extract code from markdown code blocks in the response.
        
        Args:
            text: The text containing possible markdown code blocks
            
        Returns:
            The extracted code as a string, or empty string if no code blocks found
        """
        # Look for Python code blocks (```python ... ```)
        python_pattern = r"```python\s*([\s\S]*?)\s*```"
        python_matches = re.findall(python_pattern, text)
        
        if python_matches:
            # Join all Python code blocks
            return "\n\n".join(python_matches)
        
        # If no Python blocks found, look for generic code blocks (``` ... ```)
        generic_pattern = r"```\s*([\s\S]*?)\s*```"
        generic_matches = re.findall(generic_pattern, text)
        
        if generic_matches:
            # Join all generic code blocks
            return "\n\n".join(generic_matches)
        
        # No code blocks found
        return ""
    
    def format_error_message(self) -> str:
        """Return a helpful error message if code validation fails"""
        base_message = "Could not extract valid Python code from the response."
        if self.function_name:
            return f"{base_message} Make sure the code includes a function named '{self.function_name}'."
        return base_message

    @classmethod
    def create(cls, function_name: Optional[str] = None) -> "CodeFormatter":
        """
        Factory method to create a CodeFormatter instance
        
        Args:
            function_name: Optional name of the function to extract
            
        Returns:
            A configured CodeFormatter instance
        """
        return cls(function_name=function_name)        
        
class TextFormatter(BaseFormatter):    
    def prepare_prompt(self, prompt: str) -> str:
        return prompt
    
    def validate_response(self, response: str) -> Tuple[bool, Union[str, None]]:
        """
        For plain text formatter, we simply return the response as is without validation
        since there are no format restrictions
        """
        return True, response
    