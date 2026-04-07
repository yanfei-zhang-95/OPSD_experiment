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
import os

# 动态获取 .env 路径并加载
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(env_path)


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


class WebExplorerOpen(BaseTool):
    name: str = "open"
    tool_usage: str = """**open**: Use cursor and link id to open a specific result from search (e.g., {"cursor": 13, "id": 0}). Returns [cursor] with first 60 lines and prompt to use 'find' if more content exists."""
    tool_usage_workflow: str = """- open({"cursor": 13, "id": 1}) → Returns [14] showing lines 0-59 of 500 total, with "... and 440 more lines" prompt"""
    description: str = "Open a link specified by id from a previous search result. The id is a link number from search results (displayed as 【id†title†domain】). You must provide both cursor (from the search result) and id (the link number) to open a link. Returns format: [cursor] title (url) followed by first 60 lines by default. If page has more lines, shows '... and N more lines (use find tool to search in full content)'. Full content is cached and searchable via find tool."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": "Link ID (integer from search results, starting at 0). Must be used together with cursor from the search result."
            },
            "cursor": {
                "type": "integer",
                "description": "Page cursor indicator from a previous search result. Required when using id. Must be provided together with id."
            },
            "loc": {
                "type": "integer",
                "description": "Starting line number for viewing (0-based). Defaults to -1 (start of document)."
            },
            "num_lines": {
                "type": "integer",
                "description": "Number of lines to display. Defaults to 60 lines. Set to higher value if you need to view more content at once."
            },
            "view_source": {
                "type": "boolean",
                "description": "Whether to view source code. Defaults to false."
            },
            "source": {
                "type": "string",
                "description": "Source associated with the URL. Defaults to 'web'."
            }
        },
        "required": ["id", "cursor"]
    }
    
    read_engine: str = "jina"
    generate_engine: str = "deepseekchat"
    max_retry: int = 3
    cache_manager: Optional[Any] = None
    environment_mode: str = "real"  # "real" or "local_simulated"

    def __init__(self, cache_manager=None, **data):
        super().__init__(**data)
        self.cache_manager = cache_manager
        # 允许通过配置覆盖默认值
        if "cfg" in data:
            cfg = data["cfg"]
            if cfg:
                self.read_engine = cfg.get("read_engine", self.read_engine)
                self.generate_engine = cfg.get("generate_engine", self.generate_engine)
                self.max_retry = cfg.get("max_retry", self.max_retry)
                self.environment_mode = cfg.get("environment_mode", self.environment_mode)

    async def __call__(self, **kwargs) -> str:
        params = kwargs
        try:
            url_or_id = params.get("id", -1)
            cursor_param = params.get("cursor", -1)
            loc = params.get("loc", -1)
            num_lines = params.get("num_lines", -1)
            view_source = params.get("view_source", False)
            source = params.get("source", "web")
            task_id = params.get("task_id", None)
        except:
            return "[Open] Invalid request format: Input must be a JSON object containing 'id' field"

        cache_mgr = self.cache_manager or get_cache_manager()
        url = None
        title = None
        doc_data = None  # 本地wiki文档数据
        
        # 解析 id 参数 - 现在只支持通过 cursor 和 id 打开
        if not isinstance(url_or_id, int) or url_or_id < 0:
            return "[Open] Error: 'id' must be a valid link ID (integer >= 0) from search results"
        
        if cursor_param <= 0:
            return "[Open] Error: 'cursor' must be provided and must be a positive integer from a previous search result"
        
        # 通过 cursor 和 id 从搜索结果中获取 URL
        url = cache_mgr.get_search_result_url(cursor_param, url_or_id, task_id=task_id)
        if not url:
            return f"[Open] Error: Link ID {url_or_id} not found in search results (cursor {cursor_param})"
        
        # 获取标题和文档数据
        cached_page = cache_mgr.get_page(cursor_param, task_id=task_id)
        if cached_page and cached_page.search_results:
            for result in cached_page.search_results:
                if result.id == url_or_id:
                    title = result.title
                    doc_data = result.doc_data  # 获取本地wiki文档数据
                    break

        if not url or not isinstance(url, str):
            return "[Open] Error: Invalid URL retrieved from search results"

        try:
            # Read webpage content
            source_text = ""
            
            # 检查是否是本地wiki模式且URL是local_wiki://格式
            is_local_wiki_url = url.startswith("local_wiki://")
            is_local_mode = (self.environment_mode == "local_simulated")
            
            if is_local_mode and (is_local_wiki_url or doc_data is not None):
                # 本地模拟环境：直接从doc_data获取内容，绕过jina
                if doc_data is not None:
                    # 优先使用 _full_content (如果存在)，这是search工具为了模拟摘要而保存的全文
                    source_text = doc_data.get('_full_content', '')
                    
                    if not source_text:
                        # 优先使用text字段，如果没有则使用contents
                        source_text = doc_data.get('text', '')
                        if not source_text:
                            source_text = doc_data.get('contents', '')
                    
                    # 如果没有title，从doc_data获取
                    if not title:
                        title = doc_data.get('title', 'Untitled')
                    
                    if not source_text:
                        return "[Open] Error: No content found in local wiki document"
                else:
                    return f"[Open] Error: Local wiki document data not found for {url}"
            else:
                # 真实环境：使用jina读取网页
                for retry_cnt in range(self.max_retry):
                    try:
                        if self.read_engine == "jina":
                            source_text = jina_readpage(url, max_retry=1)
                        else:
                            raise ValueError(f"Unsupported read engine: {self.read_engine}")
                        break
                    except Exception as e:
                        print(f"Read {self.read_engine} {retry_cnt} error: {e}, url: {url}", flush=True)
                        if any(word in str(e) for word in ["Client Error"]):
                            return "Access to this URL is denied. Please try again."
                        time.sleep(random.uniform(4, 16))
            
            if source_text.strip() == "" or source_text.startswith("[browse] Failed to read page."):
                print(f"Open error with empty source_text.", flush=True)
                return "Open error. Please try again."
            
            # 缓存打开的页面
            new_cursor = cache_mgr.add_page(
                url=url,
                content=source_text,
                page_type="open",
                metadata={"title": title or url},
                task_id=task_id
            )
            
            # Handle line-based viewing
            lines = source_text.split('\n')
            total_lines = len(lines)
            
            # 默认最多显示 60 行，避免上下文爆炸
            DEFAULT_MAX_LINES = 60
            
            if loc > 0 and loc <= total_lines:
                start_line = loc - 1
            else:
                start_line = 0
            
            if num_lines > 0:
                # 用户指定了行数
                end_line = min(start_line + num_lines, total_lines)
            else:
                # 默认只显示前 DEFAULT_MAX_LINES 行
                end_line = min(start_line + DEFAULT_MAX_LINES, total_lines)
            
            # 构建输出
            output_lines = []
            
            # 第一行：[cursor] title (url)
            display_title = title if title else url
            output_lines.append(f"[{new_cursor}] {display_title} ({url})")
            
            # 第二行：viewing lines info
            output_lines.append(f"**viewing lines [{start_line} - {end_line - 1}] of {total_lines}**")
            output_lines.append("")
            
            # 带行号的内容
            for i in range(start_line, end_line):
                line_content = lines[i]
                if len(line_content) > 2000:
                    line_content = line_content[:2000] + "...[Line Truncated]"
                output_lines.append(f"L{i}: {line_content}")
            
            # 如果还有更多内容，添加提示
            if end_line < total_lines:
                output_lines.append("")
                output_lines.append(f"... and {total_lines - end_line} more lines (use 'find' tool to search in full content)")
            
            result = '\n'.join(output_lines)
            # print(f'Open result length: {len(result)} chars, showing lines {start_line}-{end_line-1} of {total_lines} total')
            return result.strip()
            
        except Exception as e:
            return f"[Open] Error: {str(e)}"

if __name__ == "__main__":
    import asyncio
    async def test():
        tool = WebExplorerOpen()
        result = await tool(id="https://www.baidu.com")
        print(result)
    asyncio.run(test())