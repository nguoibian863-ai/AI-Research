import concurrent.futures
import logging
import re
import urllib.parse
import warnings
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import httpx
from pydantic import BaseModel, Field

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        from ddgs import DDGS
except ImportError:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

from backend.sources.dedup import compute_canonical_key

logger = logging.getLogger(__name__)


class SearchResultItem(BaseModel):
    title: str
    url: str
    snippet: str
    query: str = ""
    rank: int = 1
    source_type: str = "web"  # "web" | "paper" | "academic"
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    venue: Optional[str] = None
    published_at: Optional[str] = None
    citation_count: Optional[int] = None


class BaseSearchProvider(ABC):
    name: str = "base"

    @abstractmethod
    def search(self, query: str, max_results: int = 8) -> List[SearchResultItem]:
        """Executes a search query and returns structured result items."""
        pass


class DuckDuckGoSearchProvider(BaseSearchProvider):
    name: str = "duckduckgo"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def search(self, query: str, max_results: int = 8) -> List[SearchResultItem]:
        results: List[SearchResultItem] = []
        if DDGS is None:
            logger.warning("[DuckDuckGoSearchProvider] DDGS library not installed. Skipping web search.")
            return results

        try:
            with DDGS(timeout=self.timeout) as ddgs:
                ddg_gen = ddgs.text(query, max_results=max_results)
                if ddg_gen:
                    for i, r in enumerate(ddg_gen, start=1):
                        results.append(
                            SearchResultItem(
                                title=r.get("title", ""),
                                url=r.get("href", ""),
                                snippet=r.get("body", ""),
                                query=query,
                                rank=i,
                                source_type="web"
                            )
                        )
        except Exception as e:
            logger.error(f"[DuckDuckGoSearchProvider] Search error for query '{query}': {e}")

        return results


class ArxivSearchProvider(BaseSearchProvider):
    """
    Search provider querying arXiv API (Plan 12.5).
    Provides authentic academic papers with DOI, arXiv ID, and abstracts.
    """
    name: str = "arxiv"
    API_URL = "https://export.arxiv.org/api/query"

    def __init__(self, timeout: int = 10, client: Optional[httpx.Client] = None):
        self.timeout = timeout
        self._client = client

    def _clean_query(self, query: str) -> str:
        # arXiv query syntax: strip boolean operators and special chars
        cleaned = re.sub(r'[^\w\s-]', ' ', query)
        terms = [t for t in cleaned.split() if t]
        return " AND ".join(terms[:8]) if terms else query

    def search(self, query: str, max_results: int = 8) -> List[SearchResultItem]:
        results: List[SearchResultItem] = []
        cleaned_q = self._clean_query(query)
        if not cleaned_q:
            return results

        params = {
            "search_query": f"all:{cleaned_q}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            if self._client:
                resp = self._client.get(self.API_URL, params=params, timeout=self.timeout)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(self.API_URL, params=params)

            if resp.status_code != 200:
                logger.warning(f"[ArxivSearchProvider] HTTP {resp.status_code} for query '{query}'.")
                return results

            root = ET.fromstring(resp.text)
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom"
            }

            entries = root.findall("atom:entry", ns)
            for i, entry in enumerate(entries, start=1):
                title_elem = entry.find("atom:title", ns)
                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else ""
                # Strip excessive whitespace
                title = re.sub(r"\s+", " ", title)

                id_elem = entry.find("atom:id", ns)
                raw_id_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""

                summary_elem = entry.find("atom:summary", ns)
                summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""
                summary = re.sub(r"\s+", " ", summary)

                # Extract arXiv ID (e.g. 2304.08069 or 2304.08069v1)
                arxiv_m = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", raw_id_url)
                arxiv_id = arxiv_m.group(1) if arxiv_m else None

                # Full-text HTML URL preferred over /abs/ for rich benchmark and experiment extraction
                url = f"https://arxiv.org/html/{arxiv_id}" if arxiv_id else raw_id_url

                # Published timestamp
                published_elem = entry.find("atom:published", ns)
                published_at = published_elem.text.strip() if published_elem is not None and published_elem.text else None

                # Optional DOI
                doi_elem = entry.find("arxiv:doi", ns)
                doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None

                if url and title:
                    results.append(
                        SearchResultItem(
                            title=title,
                            url=url,
                            snippet=summary,
                            query=query,
                            rank=i,
                            source_type="paper",
                            arxiv_id=arxiv_id,
                            doi=doi,
                            published_at=published_at,
                        )
                    )
        except Exception as e:
            logger.warning(f"[ArxivSearchProvider] Search failed for query '{query}': {e}")

        logger.info(f"[ArxivSearchProvider] Found {len(results)} results for query '{query}'.")
        return results


def reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Reconstructs linear text from OpenAlex abstract_inverted_index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    pos_words: List[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                pos_words.append((pos, word))
    pos_words.sort(key=lambda x: x[0])
    return " ".join(w for _, w in pos_words)


class OpenAlexSearchProvider(BaseSearchProvider):
    """
    Search provider querying OpenAlex API (Plan 12.5).
    Provides academic papers with venue, citation count, DOI, and year.
    Gracefully handles rate limits (HTTP 429) without crashing pipeline.
    """
    name: str = "openalex"
    API_URL = "https://api.openalex.org/works"

    def __init__(self, timeout: int = 10, email: str = "research_agent@local.internal", client: Optional[httpx.Client] = None):
        self.timeout = timeout
        self.email = email
        self._client = client

    def search(self, query: str, max_results: int = 8) -> List[SearchResultItem]:
        results: List[SearchResultItem] = []
        cleaned_q = query.strip()
        if not cleaned_q:
            return results

        params = {
            "search": cleaned_q,
            "per_page": min(max_results, 10),
            "mailto": self.email,
        }
        headers = {
            "User-Agent": f"AI-Research-Agent/0.1 (mailto:{self.email})"
        }

        try:
            if self._client:
                resp = self._client.get(self.API_URL, params=params, headers=headers, timeout=self.timeout)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.get(self.API_URL, params=params, headers=headers)

            if resp.status_code == 429:
                logger.warning(f"[OpenAlexSearchProvider] Rate-limited (HTTP 429) for query '{query}'. Skipping OpenAlex.")
                return results

            if resp.status_code != 200:
                logger.warning(f"[OpenAlexSearchProvider] HTTP {resp.status_code} for query '{query}'.")
                return results

            data = resp.json()
            work_items = data.get("results", [])

            for i, work in enumerate(work_items, start=1):
                title = work.get("title") or ""
                if not title:
                    continue

                doi = work.get("doi")
                pub_year = work.get("publication_year")
                published_at = str(pub_year) if pub_year else work.get("publication_date")
                cited_by = work.get("cited_by_count")

                primary_loc = work.get("primary_location") or {}
                source_info = primary_loc.get("source") or {}
                venue = source_info.get("display_name")

                landing_url = primary_loc.get("landing_page_url")
                pdf_url = primary_loc.get("pdf_url")
                open_access = work.get("open_access") or {}
                oa_url = open_access.get("oa_url")
                work_id = work.get("id")

                # Choose best URL: prioritize open-access / PDF over paywalled publisher landing page
                url = oa_url or pdf_url or landing_url or doi or work_id or ""
                if not url:
                    continue

                # Abstract / snippet
                abstract_inv = work.get("abstract_inverted_index")
                snippet = reconstruct_openalex_abstract(abstract_inv)
                if not snippet:
                    desc_parts = []
                    if venue:
                        desc_parts.append(f"Published in {venue}")
                    if pub_year:
                        desc_parts.append(f"Year {pub_year}")
                    if cited_by is not None:
                        desc_parts.append(f"Citations: {cited_by}")
                    snippet = ". ".join(desc_parts) + "." if desc_parts else title

                results.append(
                    SearchResultItem(
                        title=title,
                        url=url,
                        snippet=snippet,
                        query=query,
                        rank=i,
                        source_type="academic",
                        doi=doi,
                        venue=venue,
                        published_at=published_at,
                        citation_count=cited_by,
                    )
                )
        except Exception as e:
            logger.warning(f"[OpenAlexSearchProvider] Search failed for query '{query}': {e}")

        logger.info(f"[OpenAlexSearchProvider] Found {len(results)} results for query '{query}'.")
        return results


class CompositeSearchProvider(BaseSearchProvider):
    """
    Composite search provider querying academic sources (arXiv, OpenAlex)
    alongside general web search (DuckDuckGo) concurrently with quota sharing
    and round-robin interleaving to guarantee diverse high-credibility results.
    """
    name: str = "composite"

    def __init__(self, providers: Optional[List[BaseSearchProvider]] = None, max_results: int = 8):
        self.providers = providers or [
            ArxivSearchProvider(timeout=8),
            OpenAlexSearchProvider(timeout=8),
            DuckDuckGoSearchProvider(timeout=12),
        ]
        self.max_results = max_results

    def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResultItem]:
        limit = max_results or self.max_results
        if not self.providers:
            return []

        # Determine quota per provider (ensure each provider can contribute)
        per_provider = max(2, limit // len(self.providers) + 1)

        # Call providers concurrently to minimize search latency
        provider_results: Dict[str, List[SearchResultItem]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(self.providers))) as executor:
            future_to_provider = {
                executor.submit(p.search, query, per_provider): p.name
                for p in self.providers
            }
            for future in concurrent.futures.as_completed(future_to_provider):
                p_name = future_to_provider[future]
                try:
                    provider_results[p_name] = future.result()
                except Exception as e:
                    logger.error(f"[CompositeSearchProvider] Provider {p_name} failed: {e}")
                    provider_results[p_name] = []

        # Round-robin interleaving across providers preserving configured priority order
        lists = [provider_results.get(p.name, []) for p in self.providers]
        max_len = max((len(l) for l in lists), default=0)

        combined: List[SearchResultItem] = []
        seen_keys: set = set()
        seen_urls: set = set()

        for idx in range(max_len):
            for p_list in lists:
                if len(combined) >= limit:
                    break
                if idx < len(p_list):
                    item = p_list[idx]
                    url_clean = (item.url or "").strip().lower()
                    if not url_clean or url_clean in seen_urls:
                        continue

                    canonical_k = compute_canonical_key(item.url, item.title)
                    if canonical_k in seen_keys:
                        continue

                    seen_keys.add(canonical_k)
                    seen_urls.add(url_clean)
                    item.rank = len(combined) + 1
                    combined.append(item)
            if len(combined) >= limit:
                break

        logger.info(f"[CompositeSearchProvider] Returning {len(combined)} interleaved results for '{query}'.")
        return combined

