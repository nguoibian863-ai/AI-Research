import hashlib
import logging
from pathlib import Path
from typing import Optional, Union
import httpx
from pydantic import BaseModel, Field

from backend.parsing.pdf_parser import PDFParser
from backend.parsing.html_cleaner import ParsedSection, CleanedDocument

logger = logging.getLogger(__name__)

DEFAULT_PDF_CACHE_DIR = Path("data/pdf")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"


class FetchedPDFContent(BaseModel):
    url: str
    file_path: str
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    page_count: int = 1
    text: str
    sections: list[ParsedSection] = Field(default_factory=list)
    content_hash: str
    is_cached: bool = False


class PDFFetchTool:
    """
    Downloads and caches PDF files to local disk and delegates parsing to PDFParser.
    """

    def __init__(self, cache_dir: Path | str = DEFAULT_PDF_CACHE_DIR, timeout: float = 30.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def _url_to_hash(self, url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def fetch(self, url_or_path: str, force_refresh: bool = False) -> Optional[FetchedPDFContent]:
        """Fetches from URL or reads local PDF path, caching to data/pdf/."""
        # Check if it's already a local file
        local_path = Path(url_or_path)
        if local_path.exists() and local_path.is_file():
            cleaned = PDFParser.parse_file(local_path)
            content_bytes = local_path.read_bytes()
            content_hash = hashlib.sha256(content_bytes).hexdigest()
            return FetchedPDFContent(
                url=str(local_path.resolve()),
                file_path=str(local_path.resolve()),
                title=cleaned.title,
                author=cleaned.author,
                date=cleaned.date,
                page_count=cleaned.metadata.get("page_count", 1),
                text=cleaned.text,
                sections=cleaned.sections,
                content_hash=content_hash,
                is_cached=True
            )

        # Download from URL
        url_hash = self._url_to_hash(url_or_path)
        cache_pdf_file = self.cache_dir / f"{url_hash}.pdf"
        is_cached = False

        if not force_refresh and cache_pdf_file.exists():
            try:
                pdf_bytes = cache_pdf_file.read_bytes()
                is_cached = True
                logger.info(f"[PDFFetchTool] Cache HIT for {url_or_path}")
            except Exception as e:
                logger.warning(f"[PDFFetchTool] Failed to read cache: {e}")
                pdf_bytes = None
        else:
            pdf_bytes = None

        if pdf_bytes is None:
            logger.info(f"[PDFFetchTool] Downloading PDF: {url_or_path}")
            try:
                headers = {"User-Agent": USER_AGENT}
                with httpx.Client(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
                    res = client.get(url_or_path)
                    res.raise_for_status()
                    pdf_bytes = res.content

                cache_pdf_file.write_bytes(pdf_bytes)
            except Exception as e:
                logger.error(f"[PDFFetchTool] Failed to download {url_or_path}: {e}")
                return None

        content_hash = hashlib.sha256(pdf_bytes).hexdigest()
        cleaned = PDFParser.parse_bytes(pdf_bytes, source_name=cache_pdf_file.name)

        return FetchedPDFContent(
            url=url_or_path,
            file_path=str(cache_pdf_file.resolve()),
            title=cleaned.title,
            author=cleaned.author,
            date=cleaned.date,
            page_count=cleaned.metadata.get("page_count", 1),
            text=cleaned.text,
            sections=cleaned.sections,
            content_hash=content_hash,
            is_cached=is_cached
        )
