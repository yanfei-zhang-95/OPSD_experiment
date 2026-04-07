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

# 添加这两行来加载 .env 文件
from dotenv import load_dotenv
import os

# 动态获取 .env 路径并加载
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(env_path)

SERPER_API_KEY = os.environ.get('SERPER_KEY_ID', '')
RAPID_API_KEY = os.environ.get('RAPID_API_KEY', '')

# Local wiki server configuration
LOCAL_WIKI_HOST = os.environ.get('LOCAL_WIKI_HOST', '0.0.0.0')
LOCAL_WIKI_PORT = int(os.environ.get('LOCAL_WIKI_PORT', '8085'))

def format_search_results_with_cursor(query: str, raw_results: str, topk: int, cache_manager=None, is_local_wiki: bool = False, task_id: Optional[str] = None) -> tuple:
    """
    格式化搜索结果并缓存，返回带 cursor 的结果
    
    Returns:
        (formatted_output, cursor_id)
    """
    # 解析原始结果，提取搜索结果项
    search_results = []
    
    if is_local_wiki:
        # 本地wiki格式：直接是文档列表
        if isinstance(raw_results, list):
            for idx, doc in enumerate(raw_results):
                try:
                    # 处理本地wiki文档格式
                    title = doc.get('title', 'Untitled')
                    # 本地wiki使用文档索引作为URL标识
                    url = f"local_wiki://doc_{idx}"
                    # 使用text字段作为snippet，如果没有则使用contents的前200字符
                    text = doc.get('text', '')
                    contents = doc.get('contents', '')
                    snippet = text[:200] if text else (contents[:200] if contents else '')
                    
                    search_results.append(SearchResult(
                        id=idx,
                        title=title,
                        url=url,
                        snippet=snippet,
                        publish_date=None,
                        doc_data=doc  # 保存完整文档数据供open使用
                    ))
                except Exception as e:
                    print(f"Error parsing local wiki doc {idx}: {e}", flush=True)
                    continue
    else:
        # 原有格式：XML格式的搜索结果
        result_items = raw_results.split('\n\n')
        
        for idx, item in enumerate(result_items):
            if '<title>' in item and '<url>' in item:
                try:
                    title = item.split('<title>')[1].split('</title>')[0].strip()
                    url = item.split('<url>')[1].split('</url>')[0].strip()
                    snippet = ""
                    if '<snippet>' in item:
                        snippet = item.split('<snippet>')[1].split('</snippet>')[0].strip()
                    
                    # 提取发布日期（如果有）
                    publish_date = None
                    if 'Date published:' in snippet:
                        date_line = [line for line in snippet.split('\n') if 'Date published:' in line]
                        if date_line:
                            publish_date = date_line[0].replace('Date published:', '').strip()
                    
                    # 提取域名
                    domain = url.split('/')[2] if len(url.split('/')) > 2 else url
                    
                    search_results.append(SearchResult(
                        id=idx,
                        title=title,
                        url=url,
                        snippet=snippet,
                        publish_date=publish_date
                    ))
                except:
                    continue
    
    # 构建格式化的输出内容
    lines = []
    lines.append("")  # L0: 空行
    lines.append(f"URL: Search_Results/{query}")  # L1
    lines.append("# Search Results")  # L2
    lines.append("")  # L3
    
    for result in search_results:
        # 格式：【id†title; publish_date: date†domain】snippet
        date_str = f"; publish_date: {result.publish_date}" if result.publish_date else ""
        if is_local_wiki:
            domain = "local_wiki"
        else:
            domain = result.url.split('/')[2] if len(result.url.split('/')) > 2 else result.url
        
        # 清理 snippet
        snippet = result.snippet.replace('Date published: ' + (result.publish_date or ''), '').strip()
        snippet = snippet.replace('Source:', '').strip()
        
        line = f"  * 【{result.id}†{result.title}{date_str}†{domain}】 {snippet}"
        lines.append(line)
    
    content = '\n'.join(lines)
    
    # 缓存搜索结果
    cache_mgr = cache_manager or get_cache_manager()
    cursor = cache_mgr.add_page(
        url=f"Search_Results/{query}",
        content=content,
        page_type="search",
        metadata={"query": query, "topk": topk},
        search_results=search_results,
        task_id=task_id
    )
    
    # 构建输出格式
    total_lines = len(lines)
    display_lines = min(30, total_lines)  # 默认显示前30行
    
    output_lines = [
        f"[{cursor}] {query} (Search_Results/{query})",
        f"**viewing lines [0 - {display_lines}] of {total_lines}**",
        ""
    ]
    
    # 添加带行号的内容
    for i, line in enumerate(lines[:display_lines + 1]):
        output_lines.append(f"L{i}: {line}")
    
    return '\n'.join(output_lines), cursor


def get_searches_results(queries: List[str], topk: int = 5, engine: str = "serper", max_retry: int = 3, cache_manager=None, is_local_wiki: bool = False, task_id: Optional[str] = None) -> str:
    """Get search results for multiple queries using specified search engine."""
    results = []
    for i, query in enumerate(queries):
        raw_result = get_search_results(query, topk=topk, engine=engine, max_retry=max_retry, is_local_wiki=is_local_wiki)
        formatted_result, cursor = format_search_results_with_cursor(query, raw_result, topk, cache_manager=cache_manager, is_local_wiki=is_local_wiki, task_id=task_id)
        results.append(formatted_result)
    return "\n\n".join(results)


def search_local_wiki(query: str, topk: int = 5, max_retry: int = 3, host: str = None, port: int = None) -> List[Dict]:
    """Search using local wiki server."""
    if host is None:
        host = LOCAL_WIKI_HOST
    if port is None:
        port = LOCAL_WIKI_PORT
    
    url = f"http://{host}:{port}/retrieve"
    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": False
    }
    
    for retry_cnt in range(max_retry):
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if "result" not in result or len(result["result"]) == 0:
                return []
            
            # 返回第一个查询的结果列表
            docs = result["result"][0]
            
            # 模拟搜索引擎只返回摘要（如前100字符），同时保留全文供open使用
            for doc in docs:
                # 保存全文到 _full_content
                full_content = doc.get('text', '') or doc.get('contents', '')
                doc['_full_content'] = full_content
                
                # 截断 text 和 contents 以模拟摘要
                if 'text' in doc and doc['text']:
                    doc['text'] = doc['text'][:100] + "..."
                if 'contents' in doc and doc['contents']:
                    doc['contents'] = doc['contents'][:100] + "..."
            
            return docs
            
        except Exception as e:
            print(f"search_local_wiki {retry_cnt} error: {e}", flush=True)
            if retry_cnt == max_retry - 1:
                return []
            time.sleep(random.uniform(1, 3))
    
    return []


def search_local_wiki_batch(queries: List[str], topk: int = 5, max_retry: int = 3, host: str = None, port: int = None) -> List[List[Dict]]:
    """Search multiple queries using local wiki server in one batch request."""
    if host is None:
        host = LOCAL_WIKI_HOST
    if port is None:
        port = LOCAL_WIKI_PORT
    
    url = f"http://{host}:{port}/retrieve"
    payload = {
        "queries": queries,
        "topk": topk,
        "return_scores": False
    }
    
    for retry_cnt in range(max_retry):
        try:
            response = requests.post(url, json=payload, timeout=60) # Increased timeout for batch
            response.raise_for_status()
            result = response.json()
            
            if "result" not in result:
                return [[] for _ in queries]
            
            batch_docs = result["result"]
            
            # Process each query's docs
            processed_batch = []
            for docs in batch_docs:
                processed_docs = []
                for doc in docs:
                    # Save full content
                    full_content = doc.get('text', '') or doc.get('contents', '')
                    doc['_full_content'] = full_content
                    
                    # Truncate for snippet
                    if 'text' in doc and doc['text']:
                        doc['text'] = doc['text'][:100] + "..."
                    if 'contents' in doc and doc['contents']:
                        doc['contents'] = doc['contents'][:100] + "..."
                    processed_docs.append(doc)
                processed_batch.append(processed_docs)
            
            return processed_batch
            
        except Exception as e:
            print(f"search_local_wiki_batch {retry_cnt} error: {e}", flush=True)
            if retry_cnt == max_retry - 1:
                return [[] for _ in queries]
            time.sleep(random.uniform(1, 3))
    
    return [[] for _ in queries]



def get_search_results(query: str, topk: int = 5, engine: str = "serper", max_retry: int = 3, is_local_wiki: bool = False) -> Union[str, List[Dict]]:
    """Get search results for a single query using specified search engine."""
    if is_local_wiki or engine == "local_wiki":
        return search_local_wiki(query, topk=topk, max_retry=max_retry)
    elif engine == "serper":
        return google_search_with_serp(query, topk=topk, max_retry=max_retry)
    elif engine == "rapid":
        return search_with_rapid(query, topk=topk, max_retry=max_retry)
    else:
        raise ValueError(f"Unsupported search engine: {engine}")


def contains_chinese_basic(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return any('\u4E00' <= char <= '\u9FFF' for char in text)


def google_search_with_serp(query: str, topk: int = 5, max_retry: int = 3) -> str:
    """Perform Google search using Serper API."""
    if not SERPER_API_KEY:
        raise ValueError("SERPER_KEY_ID environment variable is not set")
    
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": topk
    }
    for retry_cnt in range(max_retry):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            results = response.json()
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = []
            
            for page in results["organic"][:topk]:
                # 构建snippet内容
                snippet = ""
                if "snippet" in page:
                    snippet = page["snippet"]
                
                # 添加日期信息到snippet中（如果有的话）
                if "date" in page:
                    snippet = f"Date published: {page['date']}\n{snippet}"
                
                # 添加来源信息到snippet中（如果有的话）
                if "source" in page:
                    snippet = f"Source: {page['source']}\n{snippet}"
                
                # 清理内容
                snippet = snippet.replace("Your browser can't play this video.", "")
                
                # 使用XML格式构建结果
                redacted_version = f"<title>{page['title']}</title>\n<url>{page['link']}</url>\n<snippet>\n{snippet}\n</snippet>"
                web_snippets.append(redacted_version)

            content = "\n\n".join(web_snippets)
            return content
            
        except Exception as e:
            print(f"google_search_with_serp {retry_cnt} error: {e}", flush=True)
            if retry_cnt == max_retry - 1:
                return f"No results found for '{query}'. Try with a more general query. Error: {str(e)}"
            time.sleep(random.uniform(1, 4))
    
    return f"Search failed after {max_retry} retries for query: '{query}'"


def search_with_rapid(query: str, topk: int = 5, max_retry: int = 3) -> str:
    """Perform Google search using Rapid API."""
    if not RAPID_API_KEY:
        raise ValueError("RAPID_API_KEY environment variable is not set")
    url = "real-time-web-search.p.rapidapi.com"
    conn = http.client.HTTPSConnection(url, timeout=15)
    headers = {
        "x-rapidapi-key": RAPID_API_KEY, 
        "x-rapidapi-host": url,
    }
    for retry_cnt in range(max_retry):
        try:
            conn.request("GET", f"/search?q={quote(query)}&limit={topk}", headers=headers)
            res = conn.getresponse()
            data = res.read()
            results = json.loads(data.decode("utf-8"))
            
            if results.get("status") != "OK":
                raise Exception(f"RapidAPI search failed: {results.get('status')}")

            web_snippets = []

            for page in results.get("data", []):

                # 构建snippet内容
                snippet = ""
                if "snippet" in page:
                    snippet = page["snippet"]
                
                # 添加日期信息到snippet中（如果有的话）
                if "date" in page:
                    snippet = f"Date published: {page['date']}\n{snippet}"
                
                # 添加来源信息到snippet中（如果有的话）
                if "source" in page:
                    snippet = f"Source: {page['source']}\n{snippet}"
                
                # 清理内容
                snippet = snippet.replace("Your browser can't play this video.", "")
                
                # 使用XML格式构建结果
                redacted_version = f"<title>{page['title']}</title>\n<url>{page['url']}</url>\n<snippet>\n{snippet}\n</snippet>"
                web_snippets.append(redacted_version)

            content = "\n\n".join(web_snippets)
            return content
            
        except Exception as e:
            print(f"search_with_rapid {retry_cnt} error: {e}", flush=True)
            if retry_cnt == max_retry - 1:
                return f"No results found for '{query}'. Error: {str(e)}"
            time.sleep(random.uniform(1, 4))
    
    return f"Search failed after {max_retry} retries for query: '{query}'. Error: {str(e)}"



class WebExplorerSearch(BaseTool):
    name: str = "search"
    tool_usage: str = "**search**: Returns results with a cursor ID (e.g., [13]) and link IDs (e.g., 【0†title†domain】, 【1†title†domain】)"
    tool_usage_workflow: str = """- search({"queries": ["AI finance"]}) → Returns [13] with 【0†...】, 【1†...】, etc."""
    description: str = "Web search tool that performs batched web searches. Supply an array 'queries'; the tool retrieves search results for each query. Results are returned with cursor IDs and link IDs that can be used with the 'open' tool."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. The queries will be sent to search engine. You will get the brief search results with (title, url, snippet)s for each query. Each result is marked with 【id†title†domain】 where id can be used to open the link."
            },
        },
        "required": ["queries"],
    }
    
    search_engine: str = "rapid"
    topk: int = 5
    max_retry: int = 3
    cache_manager: Optional[Any] = None
    environment_mode: str = "real"  # "real" or "local_simulated"
    local_wiki_host: str = LOCAL_WIKI_HOST
    local_wiki_port: int = LOCAL_WIKI_PORT

    def __init__(self, cache_manager=None, **data):
        super().__init__(**data)
        self.cache_manager = cache_manager
        # 允许通过配置覆盖默认值
        if "cfg" in data:
            cfg = data["cfg"]
            if cfg:
                self.search_engine = cfg.get("search_engine", self.search_engine)
                self.topk = cfg.get("topk", self.topk)
                self.max_retry = cfg.get("max_retry", self.max_retry)
                self.environment_mode = cfg.get("environment_mode", self.environment_mode)
                self.local_wiki_host = cfg.get("local_wiki_host", self.local_wiki_host)
                self.local_wiki_port = cfg.get("local_wiki_port", self.local_wiki_port)

    async def __call__(self, **kwargs) -> str:
        params = kwargs
        try:
            queries = params["queries"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'queries' field"
        
        task_id = params.get("task_id", None)
        
        if isinstance(queries, str):
            # Single query (backward compatibility)
            queries = [queries]
        
        if not isinstance(queries, list):
            return "[Search] Error: 'queries' must be a list of strings"
        
        try:
            # 根据环境模式决定是否使用本地wiki
            is_local_wiki = (self.environment_mode == "local_simulated")
            engine = "local_wiki" if is_local_wiki else self.search_engine
            
            result = get_searches_results(
                queries=queries,
                topk=self.topk,
                engine=engine,
                max_retry=self.max_retry,
                cache_manager=self.cache_manager,
                is_local_wiki=is_local_wiki,
                task_id=task_id
            )
            return result
        except Exception as e:
            return f"[Search] Error: {str(e)}"

    def process_batch_result(self, query: str, raw_results: List[Dict], task_id: Optional[str] = None) -> str:
        """Process results from a batch search, formatting and caching them."""
        # Using is_local_wiki=True because batch search is currently only for local wiki
        fmt_res, _ = format_search_results_with_cursor(
            query, raw_results, 
            topk=self.topk,
            cache_manager=self.cache_manager,
            is_local_wiki=True,
            task_id=task_id
        )
        return fmt_res

if __name__ == "__main__":
    import asyncio
    async def test():
        tool = WebExplorerSearch()
        result = await tool(queries=["What is the capital of July?", "What is the capital of China?"])
        print(result)
    asyncio.run(test())