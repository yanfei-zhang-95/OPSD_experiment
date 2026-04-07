import os
import time
import random
import requests
from typing import Union, Optional, Dict, Any
from openai import OpenAI
import tiktoken
from .page_cache_manager import get_cache_manager
from .base_tool import BaseTool

# 添加这两行来加载 .env 文件
from dotenv import load_dotenv
load_dotenv()  # 这会自动查找并加载 .env 文件

JINA_API_KEY = os.getenv("JINA_API_KEYS", "")

def get_geminiflash_response(query: str, temperature: float = 0.0, max_retry: int = 5) -> str:
    """Get response from Gemini Flash model using standard OpenAI-compatible API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    api_base = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
    
    if not api_key:
        print("Warning: GEMINI_API_KEY not set, skipping Gemini response", flush=True)
        return None
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        for retry_cnt in range(max_retry):
            try:
                response = client.chat.completions.create(
                    model="gemini-2.0-flash-exp",
                    messages=[{"role": "user", "content": query}],
                    temperature=temperature,
                    max_tokens=32768
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                print(f"get_geminiflash_response {retry_cnt} error: {e}", flush=True)
                if retry_cnt == max_retry - 1:
                    return None
                time.sleep(random.uniform(4, 32))
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}", flush=True)
    
    return None


def get_deepseekchat_response(query: str, temperature: float = 0.0, max_retry: int = 3) -> str:
    """Get response from DeepSeek Chat model using standard OpenAI-compatible API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    
    if not api_key:
        print("Warning: DEEPSEEK_API_KEY not set, skipping DeepSeek response", flush=True)
        return None
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        for retry_cnt in range(max_retry):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": query}],
                    temperature=temperature,
                    max_tokens=8192
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                print(f"get_deepseekchat_response {retry_cnt} error: {e}", flush=True)
                if retry_cnt == max_retry - 1:
                    return None
                time.sleep(random.uniform(4, 32))
    except Exception as e:
        print(f"Failed to initialize DeepSeek client: {e}", flush=True)
    
    return None


def get_openai_response(query: str, temperature: float = 0.0, max_retry: int = 3) -> str:
    """Get response from OpenAI API."""
    api_key = os.environ.get("API_KEY")
    url_llm = os.environ.get("API_BASE")
    model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
    
    if not api_key or not url_llm:
        return None
        
    client = OpenAI(
        api_key=api_key,
        base_url=url_llm,
    )
    
    for attempt in range(max_retry):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": query}],
                temperature=temperature
            )
            content = chat_response.choices[0].message.content
            return content
        except Exception as e:
            print(f"get_openai_response {attempt} error: {e}", flush=True)
            if attempt == max_retry - 1:
                return None
            time.sleep(random.uniform(1, 4))
    return None


def jina_readpage(url: str, max_retry: int = 3) -> str:
    """Read webpage content using Jina service."""
    if not JINA_API_KEY:
        return "[browse] JINA_API_KEYS environment variable is not set."
    
    for attempt in range(max_retry):
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
        }
        try:
            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=50
            )
            if response.status_code == 200:
                webpage_content = response.text
                return webpage_content
            else:
                print(f"Jina API error: {response.text}")
                raise ValueError("jina readpage error")
        except Exception as e:
            print(f"jina_readpage {attempt} error: {e}", flush=True)
            time.sleep(0.5)
            if attempt == max_retry - 1:
                return "[browse] Failed to read page."
                
    return "[browse] Failed to read page."


def get_browse_results(url: str, browse_query: str, read_engine: str = "jina", generate_engine: str = "deepseekchat", max_retry: int = 3) -> str:
    """Get browse results by reading webpage and extracting relevant information."""
    time.sleep(random.uniform(0, 16))
    
    # Read webpage content
    source_text = ""
    for retry_cnt in range(max_retry):
        try:
            if read_engine == "jina":
                source_text = jina_readpage(url, max_retry=1)
            else:
                raise ValueError(f"Unsupported read engine: {read_engine}")
            break
        except Exception as e:
            print(f"Read {read_engine} {retry_cnt} error: {e}, url: {url}", flush=True)
            if any(word in str(e) for word in ["Client Error"]):
                return "Access to this URL is denied. Please try again."
            time.sleep(random.uniform(16, 64))
    
    if source_text.strip() == "" or source_text.startswith("[browse] Failed to read page."):
        print(f"Browse error with empty source_text.", flush=True)
        return "Browse error. Please try again."
    

    query = f"Please read the source content and answer a following question:\n---begin of source content---\n{source_text}\n---end of source content---\n\nIf there is no relevant information, please clearly refuse to answer. Now answer the question based on the above content:\n{browse_query}"
    
    # 处理长内容分块（仿照deep_research_utils.py的逻辑）
    encoding = tiktoken.get_encoding("cl100k_base")
    tokenized_source_text = encoding.encode(source_text)
    
    if len(tokenized_source_text) > 95000:  # 使用与原代码相同的token限制
        output = "Since the content is too long, the result is split and answered separately. Please combine the results to get the complete answer.\n"
        num_split = max(2, len(tokenized_source_text) // 95000 + 1)
        chunk_len = len(tokenized_source_text) // num_split
        print(f"Browse too long with length {len(tokenized_source_text)}, split into {num_split} parts, with each part length {chunk_len}", flush=True)
        
        outputs = []
        for i in range(num_split):
            start_idx = i * chunk_len
            end_idx = min(start_idx + chunk_len + 1024, len(tokenized_source_text))
            source_text_i = encoding.decode(tokenized_source_text[start_idx:end_idx])
            query_i = f"Please read the source content and answer a following question:\n--- begin of source content ---\n{source_text_i}\n--- end of source content ---\n\nIf there is no relevant information, please clearly refuse to answer. Now answer the question based on the above content:\n{browse_query}"
            
            if generate_engine == "geminiflash":
                output_i = get_geminiflash_response(query_i, temperature=0.0, max_retry=1)
            elif generate_engine == "deepseekchat":
                output_i = get_deepseekchat_response(query_i, temperature=0.0, max_retry=1)
            elif generate_engine == "openai":
                output_i = get_openai_response(query_i, temperature=0.0, max_retry=1)
            else:
                raise ValueError(f"Unsupported generate engine: {generate_engine}")
            
            outputs.append(output_i or "")
        
        for i in range(num_split):
            output += f"--- begin of result part {i+1} ---\n{outputs[i]}\n--- end of result part {i+1} ---\n\n"
    else:
        if generate_engine == "geminiflash":
            output = get_geminiflash_response(query, temperature=0.0, max_retry=1)
        elif generate_engine == "deepseekchat":
            output = get_deepseekchat_response(query, temperature=0.0, max_retry=1)
        elif generate_engine == "openai":
            output = get_openai_response(query, temperature=0.0, max_retry=1)
        else:
            raise ValueError(f"Unsupported generate engine: {generate_engine}")
    
    if output is None or output.strip() == "":
        print(f"Browse error with empty output.", flush=True)
        return "Browse error. Please try again."
    
    return output


class WebExplorerFind(BaseTool):
    name: str = "find"
    tool_usage: str = """**find**: Use cursor from 'open' to search within that page's full cached content (e.g., {"cursor": 14, "pattern": "keyword"}), or provide url directly. Searches all lines, not just the 60 shown by open."""
    tool_usage_workflow: str = """- find({"cursor": 14, "pattern": "risk"}) → Searches all 500 lines in cache, returns matches with context"""
    description: str = "Find exact matches of a pattern in the current page or a page specified by cursor. Use cursor from a previous 'open' result to search within that cached page (searches full content, not just displayed lines), or provide a url to read and search a new page. Returns all matching locations with context."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The exact pattern to search for in the page content."
            },
            "cursor": {
                "type": "integer",
                "description": "The page cursor from a previous 'open' or 'search' result to search in. If provided, searches in the cached full content. Defaults to -1."
            },
            "url": {
                "type": "string",
                "description": "URL to search in. If cursor is not provided or not found, will read this URL and search within it."
            }
        },
        "required": ["pattern"]
    }
    
    read_engine: str = "jina"
    max_retry: int = 3
    cache_manager: Optional[Any] = None

    def __init__(self, cache_manager=None, **data):
        super().__init__(**data)
        self.cache_manager = cache_manager
        # 允许通过配置覆盖默认值
        if "cfg" in data:
            cfg = data["cfg"]
            if cfg:
                self.read_engine = cfg.get("read_engine", self.read_engine)
                self.max_retry = cfg.get("max_retry", self.max_retry)

    async def __call__(self, **kwargs) -> str:
        params = kwargs
        try:
            pattern = params["pattern"]
            cursor_param = params.get("cursor", -1)
            url = params.get("url", None)
            task_id = params.get("task_id", None)
        except:
            return "[Find] Invalid request format: Input must be a JSON object containing 'pattern' field"

        if not pattern or not isinstance(pattern, str):
            return "[Find] Error: 'pattern' is missing, empty, or not a string"
        
        cache_mgr = self.cache_manager or get_cache_manager()
        source_text = ""
        page_url = ""
        new_cursor = -1
        
        # 优先使用 cursor 从缓存获取
        if cursor_param > 0:
            cached_page = cache_mgr.get_page(cursor_param, task_id=task_id)
            if cached_page:
                source_text = cached_page.content
                page_url = cached_page.url
                new_cursor = cursor_param  # 使用同一个 cursor
            else:
                return f"[Find] Error: Page with cursor {cursor_param} not found in cache"
        elif url:
            # 如果没有 cursor，则通过 url 读取
            try:
                for retry_cnt in range(self.max_retry):
                    try:
                        if self.read_engine == "jina":
                            source_text = jina_readpage(url, max_retry=1)
                            page_url = url
                        else:
                            raise ValueError(f"Unsupported read engine: {self.read_engine}")
                        break
                    except Exception as e:
                        print(f"Read {self.read_engine} {retry_cnt} error: {e}, url: {url}", flush=True)
                        if any(word in str(e) for word in ["Client Error"]):
                            return "Access to this URL is denied. Please try again."
                        time.sleep(random.uniform(4, 16))
                
                if source_text.strip() == "" or source_text.startswith("[browse] Failed to read page."):
                    print(f"Find error with empty source_text.", flush=True)
                    return "Find error. Please try again."
                
                # 缓存读取的页面
                new_cursor = cache_mgr.add_page(
                    url=url,
                    content=source_text,
                    page_type="open",
                    metadata={},
                    task_id=task_id
                )
            except Exception as e:
                return f"[Find] Error reading URL: {str(e)}"
        else:
            return "[Find] Error: Either 'cursor' or 'url' must be provided"
        
        try:
            # Find all occurrences of the pattern
            lines = source_text.split('\n')
            matches = []
            
            for line_num, line in enumerate(lines):
                if pattern in line:
                    matches.append(line_num)
            
            # Format output
            if not matches:
                output = f"Pattern '{pattern}' not found in the page."
            else:
                # 构建 DeepSeek 风格的输出
                output_lines = []
                output_lines.append(f"[{new_cursor + 1}] Find results for text: `{pattern}` in `{page_url}` ({page_url}/find?pattern={pattern})")
                
                # 计算需要显示的总行数
                total_display_lines = 0
                match_blocks = []
                
                for idx, line_num in enumerate(matches[:10]):  # 限制显示前10个匹配
                    block_lines = []
                    block_lines.append(f"# 【{idx}†match at L{line_num}】")
                    
                    # 显示匹配行及其前后各2行的上下文
                    start = max(0, line_num - 2)
                    end = min(len(lines), line_num + 3)
                    
                    for i in range(start, end):
                        line_content = lines[i]
                        if len(line_content) > 2000:
                            line_content = line_content[:2000] + "...[Line Truncated]"
                        block_lines.append(line_content)
                    
                    if idx < len(matches) - 1:
                        block_lines.append("")  # 空行分隔
                    
                    match_blocks.append('\n'.join(block_lines))
                    total_display_lines += len(block_lines)
                
                output_lines.append(f"**viewing lines [0 - {min(total_display_lines, 50)}] of {total_display_lines}**")
                output_lines.append("")
                
                # 添加带行号的匹配块
                line_counter = 0
                for block in match_blocks:
                    for line in block.split('\n'):
                        output_lines.append(f"L{line_counter}: {line}")
                        line_counter += 1
                        if line_counter > 50:  # 限制总行数
                            break
                    if line_counter > 50:
                        break
                
                if len(matches) > 10:
                    output_lines.append(f"... and {len(matches) - 10} more matches")
                
                output = '\n'.join(output_lines)
            
            print(f'Find result: {len(matches)} matches found')
            return output.strip()
            
        except Exception as e:
            return f"[Find] Error: {str(e)}"

if __name__ == "__main__":
    import asyncio
    async def test():
        tool = WebExplorerFind()
        result = await tool(pattern="百度", url="https://www.baidu.com")
        print(result)
    asyncio.run(test())