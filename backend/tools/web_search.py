import logging
from typing import List, Dict, Any, Optional

from backend.tools.search_providers import (
    SearchResultItem,
    BaseSearchProvider,
    DuckDuckGoSearchProvider,
    ArxivSearchProvider,
    OpenAlexSearchProvider,
    CompositeSearchProvider,
)

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Unified multi-provider search tool for Local Deep Research Agent (Plan 12.5).
    Searches academic repositories (arXiv, OpenAlex) and general web (DuckDuckGo),
    prioritizing primary sources and deduplicating by canonical key.
    """
    def __init__(
        self,
        max_results: int = 8,
        timeout: int = 15,
        providers: Optional[List[BaseSearchProvider]] = None
    ):
        self.max_results = max_results
        self.timeout = timeout
        if providers is not None:
            self.composite = CompositeSearchProvider(providers=providers, max_results=max_results)
        else:
            self.composite = CompositeSearchProvider(
                providers=[
                    ArxivSearchProvider(timeout=min(10, timeout)),
                    OpenAlexSearchProvider(timeout=min(10, timeout)),
                    DuckDuckGoSearchProvider(timeout=timeout),
                ],
                max_results=max_results
            )

    def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResultItem]:
        limit = max_results or self.max_results
        return self.composite.search(query=query, max_results=limit)
