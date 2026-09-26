import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import httpx
import trafilatura
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/web_cache")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"


class FetchedWebContent(BaseModel):
    url: str
    title: Optional[str] = None
    text: str
    author: Optional[str] = None
    date: Optional[str] = None
    content_hash: str
    is_cached: bool = False


class WebFetchTool:
    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR, timeout: float = 20.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def _url_to_hash(self, url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def fetch(self, url: str, force_refresh: bool = False) -> Optional[FetchedWebContent]:
        url_hash = self._url_to_hash(url)
        cache_html_file = self.cache_dir / f"{url_hash}.html"
        cache_meta_file = self.cache_dir / f"{url_hash}.meta"

        html_content: Optional[str] = None
        is_cached = False

        if not force_refresh and cache_html_file.exists():
            try:
                html_content = cache_html_file.read_text(encoding="utf-8")
                is_cached = True
                logger.info(f"[WebFetchTool] Cache HIT for {url}")
            except Exception as e:
                logger.warning(f"[WebFetchTool] Failed to read cache for {url}: {e}")

        if html_content is None:
            logger.info(f"[WebFetchTool] Fetching live: {url}")
            try:
                headers = {"User-Agent": USER_AGENT}
                with httpx.Client(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    html_content = response.text

                # Save raw HTML to cache
                cache_html_file.write_text(html_content, encoding="utf-8")
            except Exception as e:
                logger.error(f"[WebFetchTool] Failed to fetch {url}: {e}")
                return None

        # Clean HTML using trafilatura
        extracted_text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            no_fallback=False
        ) or ""

        metadata = trafilatura.extract_metadata(html_content)
        title = metadata.title if metadata else None
        author = metadata.author if metadata else None
        date = metadata.date if metadata else None

        content_hash = hashlib.sha256(extracted_text.encode("utf-8")).hexdigest()

        return FetchedWebContent(
            url=url,
            title=title,
            text=extracted_text,
            author=author,
            date=date,
            content_hash=content_hash,
            is_cached=is_cached
        )
