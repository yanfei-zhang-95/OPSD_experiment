import json, ast ,re
from pathlib import Path
from typing import Any
from pydantic_core import to_jsonable_python

def _parse_json_strict(text: str):
    text = (text or "").strip()
    
    def unwrap_array_if_needed(parsed_data):
        """
        如果解析结果是数组且只有一个元素，返回第一个元素。
        这处理了模型有时会生成数组而不是单个对象的情况。
        例如: [{"name": "get_final_answer", "args": {...}}] -> {"name": "get_final_answer", "args": {...}}
        """
        if isinstance(parsed_data, list):
            if len(parsed_data) == 1:
                # 数组只有一个元素，提取第一个元素
                return parsed_data[0]
            elif len(parsed_data) == 0:
                # 空数组，返回空对象
                return {}
            # 如果数组有多个元素，也返回第一个元素（假设第一个是工具调用）
            return parsed_data[0]
        return parsed_data

    try:

        # 1) 去掉代码围栏 ```json ... ```
        if text.startswith("```"):
            first = text.find("\n")
            if first != -1:
                text = text[first + 1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        # 2) 直接尝试纯 JSON
        try:
            parsed_data = json.loads(text)
            return unwrap_array_if_needed(parsed_data)
        except Exception:
            pass

        # 3) 如果是外层被引号包裹的“JSON 字符串”，先解包为真正的 JSON 文本（仍是字符串）
        unwrapped = text
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
            try:
                # e.g. '{\\n"..."}' -> '{\n"..."}'
                unwrapped = ast.literal_eval(text)
            except Exception:
                unwrapped = text[1:-1]

        # 4) 只在 JSON 字符串字面量内部把原始换行替换为 \n
        def escape_newlines_inside_strings(s: str) -> str:
            out = []
            in_str = False
            esc = False
            for ch in s:
                if in_str:
                    if esc:
                        out.append(ch)
                        esc = False
                    elif ch == '\\':
                        out.append(ch)
                        esc = True
                    elif ch == '"':
                        out.append(ch)
                        in_str = False
                    elif ch == '\n':
                        out.append('\\n')
                    elif ch == '\r':
                        # 丢弃或也可转成 \\n
                        continue
                    else:
                        out.append(ch)
                else:
                    if ch == '"':
                        in_str = True
                    out.append(ch)
            return "".join(out)

        # 4.5) 修复JSON字符串值内部的无效转义序列 \'
        def fix_invalid_escapes_in_strings(s: str) -> str:
            """
            修复JSON字符串值内部的无效转义序列，如 \' 应该被替换为 '
            JSON标准中，单引号不需要转义，只有双引号需要转义为 \"
            """
            out = []
            in_str = False
            esc = False
            i = 0
            while i < len(s):
                ch = s[i]
                if in_str:
                    if esc:
                        # 当前字符是转义序列的一部分
                        if ch == "'":
                            # \' 是无效的JSON转义序列，应该替换为 '
                            out.append("'")
                            esc = False
                        elif ch == "\\" and i + 1 < len(s) and s[i + 1] == "'":
                            # \\' 的情况：第一个 \ 转义第二个 \，得到 \，然后 ' 不需要转义
                            # 但这样会变成 \'，还是无效的。应该变成 '
                            out.append("'")
                            i += 1  # 跳过下一个 '
                            esc = False
                        else:
                            # 其他有效的转义序列，保留
                            out.append("\\")
                            out.append(ch)
                            esc = False
                    elif ch == '\\':
                        esc = True
                    elif ch == '"':
                        out.append(ch)
                        in_str = False
                    else:
                        out.append(ch)
                else:
                    if ch == '"':
                        in_str = True
                    out.append(ch)
                i += 1
            # 如果最后还有未完成的转义序列
            if esc:
                out.append("\\")
            return "".join(out)

        candidate = unwrapped

        # 5) 去 BOM、统一引号
        candidate = candidate.lstrip("\ufeff")
        candidate = candidate.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

        # 6) 裁剪到最外层 {...} 或 [...]
        # 检测是先遇到 { 还是 [，来决定裁剪的边界
        start_brace = candidate.find("{")
        start_bracket = candidate.find("[")
        
        # 判断哪个先出现（-1 表示不存在）
        if start_bracket != -1 and (start_brace == -1 or start_bracket < start_brace):
            # 是数组，裁剪到 [...]
            start = start_bracket
            end = candidate.rfind("]")
            if start != -1 and end != -1 and end > start:
                candidate = candidate[start:end + 1]
        elif start_brace != -1:
            # 是对象，裁剪到 {...}
            start = start_brace
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = candidate[start:end + 1]

        # 6.3) 删除末尾多余的 } 或 ]
        # 通过统计括号的平衡来删除多余的右括号
        # 处理对象的大括号
        open_count = candidate.count("{")
        close_count = candidate.count("}")
        if close_count > open_count:
            # 末尾有多余的 }，删除多余的
            excess = close_count - open_count
            # 从末尾开始删除多余的 }
            while excess > 0 and candidate.endswith("}"):
                candidate = candidate[:-1]
                excess -= 1
        
        # 处理数组的方括号
        open_bracket_count = candidate.count("[")
        close_bracket_count = candidate.count("]")
        if close_bracket_count > open_bracket_count:
            # 末尾有多余的 ]，删除多余的
            excess = close_bracket_count - open_bracket_count
            # 从末尾开始删除多余的 ]
            while excess > 0 and candidate.endswith("]"):
                candidate = candidate[:-1]
                excess -= 1

        # 6.5) 处理双大括号或双方括号的情况（模型可能错误地生成了 {{...}} 或 [[...]]）
        # 如果开头是 {{ 且结尾是 }}，尝试去掉一层大括号
        if candidate.startswith("{{") and candidate.endswith("}}"):
            # 尝试去掉一层大括号
            inner_candidate = candidate[1:-1]
            # 先处理行尾逗号和转义换行，然后验证是否是有效的JSON
            try:
                test_candidate = re.sub(r",\s*([}\]])", r"\1", inner_candidate)
                test_candidate = escape_newlines_inside_strings(test_candidate)
                json.loads(test_candidate)
                # 如果验证成功，使用去掉一层大括号的版本
                candidate = inner_candidate
            except Exception:
                # 如果去掉一层大括号后不是有效JSON，保持原样
                pass
        
        # 如果开头是 [[ 且结尾是 ]]，尝试去掉一层方括号
        if candidate.startswith("[[") and candidate.endswith("]]"):
            # 尝试去掉一层方括号
            inner_candidate = candidate[1:-1]
            # 先处理行尾逗号和转义换行，然后验证是否是有效的JSON
            try:
                test_candidate = re.sub(r",\s*([}\]])", r"\1", inner_candidate)
                test_candidate = escape_newlines_inside_strings(test_candidate)
                json.loads(test_candidate)
                # 如果验证成功，使用去掉一层方括号的版本
                candidate = inner_candidate
            except Exception:
                # 如果去掉一层方括号后不是有效JSON，保持原样
                pass

        # 7) 处理行尾逗号（可选，模型偶尔会多逗号）
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

        # 7.4) 修复键缺少闭合引号的情况，例如 "args: { -> "args": {
        # 这种情况在模型生成工具调用时偶尔发生，且必须在 escape_quotes_inside_strings 之前修复，
        # 否则后续的字符串处理会因为结构错误而产生错误的转义
        candidate = re.sub(r'([,{])(\s*)"(\w+):\s*([{\[])', r'\1\2"\3": \4', candidate)

        # 7.5) 修复JSON字符串值内部的无效转义序列（如 \'）
        candidate = fix_invalid_escapes_in_strings(candidate)

        # 7.5.5) 修复JSON字符串值内部未转义的双引号
        # 例如："Benjamin "Bugsy" Siegel" -> "Benjamin \"Bugsy\" Siegel"
        def escape_quotes_inside_strings(s: str) -> str:
            """在JSON字符串值内部，将未转义的双引号转义为 \" """
            out = []
            in_str = False
            esc = False
            i = 0
            while i < len(s):
                ch = s[i]
                if in_str:
                    if esc:
                        # 当前字符是转义序列的一部分，直接添加
                        out.append(ch)
                        esc = False
                    elif ch == '\\':
                        # 转义字符
                        out.append(ch)
                        esc = True
                    elif ch == '"':
                        # 在字符串内部遇到双引号
                        # 检查下一个字符，如果是明确的结构字符（}, ], ,, :），则这是字符串结束
                        # 否则这是字符串内部的引号，需要转义
                        if i + 1 < len(s):
                            next_ch = s[i + 1]
                            # 只有当下一个字符是明确的结构字符时，才认为是字符串结束
                            # 注意：空格、换行等不算，因为它们可能在字符串值内部
                            # 但是 : 也算，因为 "key": value 中的 " 后面是 :
                            if next_ch in ['}', ']', ',', ':']:
                                out.append(ch)
                                in_str = False
                            else:
                                # 这是字符串内部的引号，需要转义
                                out.append('\\')
                                out.append('"')
                        else:
                            # 字符串末尾，这是字符串结束
                            out.append(ch)
                            in_str = False
                    else:
                        out.append(ch)
                else:
                    if ch == '"':
                        in_str = True
                    out.append(ch)
                i += 1
            return "".join(out)
        
        candidate = escape_quotes_inside_strings(candidate)

        # 7.6) 删除JSON结构部分（不在字符串值内部）的无效转义序列
        # 例如：]}\'} 应该变成 ]}
        # 注意：此时字符串值内部的 \' 已经被修复成 ' 了，所以剩下的 \' 都是无效的
        def remove_invalid_escapes_outside_strings(s: str) -> str:
            """删除JSON结构部分的无效转义序列，如 \' 在 } 或 ] 之后"""
            # 匹配 }\'} 或 ]\'} 这样的模式，删除 \'
            # 模式1: ]}\'} -> ]}
            s = re.sub(r"([}\]])'([}\]])", r"\1\2", s)  # ]}' -> ]}
            # 模式2: ]}\'} -> ]} (如果 \' 在结构符号之后)
            s = re.sub(r"([}\]])'", r"\1", s)  # ]' 或 }' -> ] 或 }
            # 模式3: \'} 或 \'] -> } 或 ]
            s = re.sub(r"\\'([}\]])", r"\1", s)  # \'} 或 \'] -> } 或 ]
            return s
        
        candidate = remove_invalid_escapes_outside_strings(candidate)

        # 7.7) 删除JSON结构部分的多余引号
        # 例如：]"} 应该变成 ]}（在 ] 或 } 之后有多余的 "）
        def remove_extra_quotes_outside_strings(s: str) -> str:
            """删除JSON结构部分的多余引号，如 ]" 或 }" """
            # 匹配 ]" 或 }" 这样的模式（不在字符串值内部）
            # 使用简单的正则：在 ] 或 } 之后如果跟着 "，且这个 " 后面是 } 或 ]，则删除这个 "
            s = re.sub(r'([}\]])"([}\]])', r'\1\2', s)  # ]"} -> ]} 或 }"} -> }}
            # 如果末尾是 ]"} 或 }"}，也删除多余的 "
            s = re.sub(r'([}\]])"$', r'\1', s)  # ]" 或 }" 在末尾 -> ] 或 }
            return s
        
        candidate = remove_extra_quotes_outside_strings(candidate)

        # 8) 转义字符串内部的原始换行
        candidate = escape_newlines_inside_strings(candidate)

        # 9) 最终解析
        parsed_data = json.loads(candidate)
        return unwrap_array_if_needed(parsed_data)
    except Exception as e:
        # 10) 最后的尝试：在字符串中查找所有有效的JSON对象，并返回最后一个
        # 这处理了模型输出多个JSON块（例如先输出工具定义，再输出工具调用）的情况
        try:
            valid_jsons = []
            depth = 0
            start = -1
            # 使用原始文本进行查找，因为candidate可能已经被裁剪坏了
            search_text = text if 'text' in locals() else candidate
            
            for i, char in enumerate(search_text):
                if char == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif char == '}':
                    if depth > 0:
                        depth -= 1
                        if depth == 0:
                            chunk = search_text[start:i+1]
                            try:
                                # 尝试解析这个块
                                # 先做一些基础清理
                                chunk = chunk.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
                                chunk = escape_newlines_inside_strings(chunk)
                                obj = json.loads(chunk)
                                valid_jsons.append(obj)
                            except:
                                pass
            
            if valid_jsons:
                # 返回最后一个有效的JSON对象
                # 通常模型会在修正后输出正确的调用，或者最后的调用是最相关的
                return unwrap_array_if_needed(valid_jsons[-1])
        except:
            pass
            
        raise ValueError("Model did not return valid JSON.")


async def _dispatch_tool(tool_name: str, tool_args: dict, tool_map: dict):
    """
    调度工具调用 - 适配新的异步工具系统
    
    Args:
        tool_name: 工具名称
        tool_args: 工具参数字典
        tool_map: 工具映射字典 {tool_name: tool_instance}
    
    Returns:
        工具执行结果（字符串）
    """
    if tool_name not in tool_map:
        return f"Error: Tool '{tool_name}' not found. Available tools: {list(tool_map.keys())}"
    
    tool = tool_map[tool_name]
    tool_args = tool_args or {}
    
    # 验证必需参数
    if hasattr(tool, 'parameters') and tool.parameters:
        params = tool.parameters
        required = params.get("required", [])
        properties = params.get("properties", {})
        
        # 检查必需参数
        for param in required:
            if param not in tool_args:
                return f"Error: Missing required parameter '{param}' for tool '{tool_name}'"
        
        # 检查未知参数（可选）
        for arg in tool_args.keys():
            if arg not in properties and arg != "params" and arg != "task_id":
                print(f"Warning: Unexpected parameter '{arg}' for tool '{tool_name}'")
    
    try:
        # 调用工具（异步）
        result = await tool(**tool_args)
        return str(result)
    except Exception as e:
        return f"Error calling tool '{tool_name}': {str(e)}"

def extract_xml(xml_response):
    # 使用与XmlFormatter相同的正则表达式
    pattern = r"<(\w+)>(.*?)</\1>"
    matches = re.findall(pattern, xml_response, re.DOTALL)
    
    # 转换为字典
    found_fields = {match[0]: match[1].strip() for match in matches}

    return found_fields

def read_json_file(json_file: str, encoding="utf-8") -> list[Any]:
    if not Path(json_file).exists():
        raise FileNotFoundError(f"json_file: {json_file} not exist, return []")

    with open(json_file, "r", encoding=encoding) as fin:
        try:
            data = json.load(fin)
        except Exception:
            raise ValueError(f"read json file: {json_file} failed")
    return data


def write_json_file(json_file: str, data: list, encoding: str = None, indent: int = 4):
    folder_path = Path(json_file).parent
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    with open(json_file, "w", encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=indent, default=to_jsonable_python)