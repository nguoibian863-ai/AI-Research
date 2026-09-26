import logging
import warnings
from typing import List, Dict, Any, Optional

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        from ddgs import DDGS
except ImportError:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        from duckduckgo_search import DDGS

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SearchResultItem(BaseModel):
    title: str
    url: str
    snippet: str
    query: str = ""
    rank: int = 1



class WebSearchTool:
    def __init__(self, max_results: int = 8, timeout: int = 15):
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResultItem]:
        limit = max_results or self.max_results
        results: List[SearchResultItem] = []
        logger.info(f"[WebSearchTool] Searching: '{query}' (limit={limit})")

        try:
            with DDGS(timeout=self.timeout) as ddgs:
                ddg_gen = ddgs.text(query, max_results=limit)
                if ddg_gen:
                    for i, r in enumerate(ddg_gen, start=1):
                        results.append(SearchResultItem(
                            title=r.get("title", ""),
                            url=r.get("href", ""),
                            snippet=r.get("body", ""),
                            query=query,
                            rank=i
                        ))
        except Exception as e:
            logger.error(f"[WebSearchTool] Search error for query '{query}': {e}")

        logger.info(f"[WebSearchTool] Returned {len(results)} results for query: '{query}'")
        return results
