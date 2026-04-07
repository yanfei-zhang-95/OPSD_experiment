"""全局页面缓存管理器 - 用于 open、find、search 工具之间共享状态"""

import threading
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """搜索结果项"""
    id: int  # 在当前搜索结果中的编号（0, 1, 2...）
    title: str
    url: str
    snippet: str
    publish_date: Optional[str] = None
    doc_data: Optional[Dict[str, Any]] = None  # 本地wiki文档的完整数据（用于local_simulated模式）


@dataclass
class CachedPage:
    """缓存的页面信息"""
    cursor: int  # 全局唯一的 cursor ID
    url: str
    content: str
    lines: List[str]
    page_type: str  # "search" 或 "open"
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外的元数据
    search_results: List[SearchResult] = field(default_factory=list)  # 如果是搜索页面，存储搜索结果


class PageCacheManager:
    """全局页面缓存管理器 - 支持任务级别隔离"""
    
    def __init__(self):
        # 使用字典存储每个任务的缓存和 cursor
        # key: task_id (线程 ID 或任务标识符)
        # value: {"cache": {}, "current_cursor": 0}
        self.task_caches: Dict[str, Dict] = {}
        self.access_lock = threading.Lock()
        
        # 兼容性：默认任务 ID（用于单线程或测试）
        self._default_task_id = "default"
    
    def _get_task_id(self) -> str:
        """
        获取当前任务的 ID
        
        优先级：
        1. 如果当前在异步上下文中，使用 asyncio task ID（更精确）
        2. 否则使用线程 ID（用于同步或线程隔离的场景）
        """
        try:
            # 尝试获取当前运行的 asyncio task
            current_task = asyncio.current_task()
            if current_task is not None:
                # 使用 asyncio task 的 ID（更精确，能区分同一线程中的不同协程）
                return f"asyncio_task_{id(current_task)}"
        except RuntimeError:
            # 不在异步上下文中，使用线程 ID
            pass
        
        # 回退到线程 ID（用于同步代码或线程隔离的场景）
        thread_id = threading.current_thread().ident
        return f"thread_{thread_id}"
    
    def _get_task_cache(self, task_id: Optional[str] = None) -> Dict:
        """获取指定任务的缓存字典"""
        if task_id is None:
            task_id = self._get_task_id()
        
        if task_id not in self.task_caches:
            self.task_caches[task_id] = {
                "cache": {},
                "current_cursor": 0
            }
        
        return self.task_caches[task_id]
    
    def add_page(self, 
                 url: str, 
                 content: str, 
                 page_type: str = "open",
                 metadata: Optional[Dict[str, Any]] = None,
                 search_results: Optional[List[SearchResult]] = None,
                 task_id: Optional[str] = None) -> int:
        """
        添加页面到缓存
        
        Args:
            url: 页面URL或标识符
            content: 页面内容
            page_type: 页面类型（"search" 或 "open"）
            metadata: 额外的元数据
            search_results: 如果是搜索页面，包含的搜索结果列表
            task_id: 任务 ID（可选，默认使用当前线程 ID）
        
        Returns:
            cursor ID
        """
        with self.access_lock:
            task_cache = self._get_task_cache(task_id)
            task_cache["current_cursor"] += 1
            cursor = task_cache["current_cursor"]
            
            lines = content.split('\n')
            
            page = CachedPage(
                cursor=cursor,
                url=url,
                content=content,
                lines=lines,
                page_type=page_type,
                metadata=metadata or {},
                search_results=search_results or []
            )
            
            task_cache["cache"][cursor] = page
            
            return cursor
    
    def get_page(self, cursor: int, task_id: Optional[str] = None) -> Optional[CachedPage]:
        """
        通过 cursor 获取页面
        
        Args:
            cursor: cursor ID
            task_id: 任务 ID（可选，默认使用当前线程 ID）
        """
        with self.access_lock:
            task_cache = self._get_task_cache(task_id)
            return task_cache["cache"].get(cursor)
    
    def get_search_result_url(self, cursor: int, result_id: int, task_id: Optional[str] = None) -> Optional[str]:
        """
        从搜索结果中获取指定 ID 的 URL
        
        Args:
            cursor: 搜索页面的 cursor
            result_id: 搜索结果的 ID（从列表中的索引）
            task_id: 任务 ID（可选，默认使用当前线程 ID）
        
        Returns:
            URL 字符串，如果未找到则返回 None
        """
        page = self.get_page(cursor, task_id)
        if not page or page.page_type != "search":
            return None
        
        if 0 <= result_id < len(page.search_results):
            return page.search_results[result_id].url
        
        return None
    
    def clear_cache(self, task_id: Optional[str] = None):
        """
        清空缓存
        
        Args:
            task_id: 任务 ID（可选）。如果提供，只清空该任务的缓存；否则清空当前线程的缓存
        """
        with self.access_lock:
            if task_id is None:
                task_id = self._get_task_id()
            
            if task_id in self.task_caches:
                self.task_caches[task_id]["cache"].clear()
                self.task_caches[task_id]["current_cursor"] = 0
    
    def clear_all_caches(self):
        """清空所有任务的缓存（通常用于测试或重置）"""
        with self.access_lock:
            self.task_caches.clear()
    
    def get_cache_size(self, task_id: Optional[str] = None) -> int:
        """
        获取缓存的页面数量
        
        Args:
            task_id: 任务 ID（可选，默认使用当前线程 ID）
        """
        with self.access_lock:
            task_cache = self._get_task_cache(task_id)
            return len(task_cache["cache"])
    
    def get_all_cursors(self, task_id: Optional[str] = None) -> List[int]:
        """
        获取所有缓存的 cursor ID
        
        Args:
            task_id: 任务 ID（可选，默认使用当前线程 ID）
        """
        with self.access_lock:
            task_cache = self._get_task_cache(task_id)
            return list(task_cache["cache"].keys())
    
    def get_all_task_ids(self) -> List[str]:
        """获取所有任务 ID"""
        with self.access_lock:
            return list(self.task_caches.keys())


# 全局单例实例
_global_cache = PageCacheManager()


def get_cache_manager() -> PageCacheManager:
    """获取全局缓存管理器实例"""
    return _global_cache

