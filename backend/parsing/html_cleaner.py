import re
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import trafilatura

logger = logging.getLogger(__name__)


class ParsedSection(BaseModel):
    title: str = "General"
    content: str
    page: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    char_start: int = 0
    char_end: int = 0
    blocks: List[Dict[str, Any]] = Field(default_factory=list)


class CleanedDocument(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    text: str
    sections: List[ParsedSection] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HTMLCleaner:
    """
    Cleans raw HTML into normalized, article-focused text with metadata and section boundaries.
    Uses trafilatura as primary extraction engine with regex fallback.
    """

    @staticmethod
    def _fallback_clean(html: str) -> str:
        """Fallback cleaner for non-standard HTML or when trafilatura returns empty."""
        # Strip script and style tags
        cleaned = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        # Strip all other HTML tags
        cleaned = re.sub(r"<[^>]+>", "\n", cleaned)
        # Normalize whitespace
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        return "\n\n".join(lines)

    @classmethod
    def clean(cls, html_content: str, url: Optional[str] = None) -> CleanedDocument:
        if not html_content or not html_content.strip():
            return CleanedDocument(text="", sections=[])

        # Preprocess headings to ensure trafilatura retains them as paragraphs
        preprocessed_html = re.sub(
            r"<h([1-6])[^>]*>(.*?)</h\1>",
            r"<p><b>\2</b></p>",
            html_content,
            flags=re.DOTALL | re.IGNORECASE
        )

        # 1. Trafilatura main text extraction
        extracted_text = trafilatura.extract(
            preprocessed_html,
            include_comments=False,
            include_tables=True,
            include_formatting=True,
            no_fallback=False,
            url=url
        )

        if not extracted_text or not extracted_text.strip():
            logger.warning("[HTMLCleaner] Trafilatura returned empty; using fallback regex parser.")
            extracted_text = cls._fallback_clean(html_content)

        # 2. Extract metadata
        meta = trafilatura.extract_metadata(html_content)
        title = meta.title if meta and meta.title else None
        author = meta.author if meta and meta.author else None
        date = meta.date if meta and meta.date else None
        sitename = meta.sitename if meta and meta.sitename else None

        # 3. Detect sections from extracted text
        sections: List[ParsedSection] = []
        raw_paragraphs = extracted_text.split("\n\n")
        current_section_title = title or "Introduction"
        current_section_lines: List[str] = []

        # Regex heuristic for headings: short lines (<80 chars), capital start, no terminal period
        heading_pattern = re.compile(r"^(#{1,4}\s+|[A-Z0-9][\w\s\-\:\.]{2,70}(?<!\.))$")

        search_cursor = 0

        def build_section(sec_title: str, lines: List[str], cursor: int):
            if not lines:
                return None, cursor
            sec_content = "\n\n".join(lines)
            sec_start = extracted_text.find(sec_content, cursor)
            if sec_start != -1:
                sec_end = sec_start + len(sec_content)
                new_cursor = sec_end
            else:
                sec_start = cursor
                sec_end = sec_start + len(sec_content)
                new_cursor = sec_end + 2

            b_list = []
            b_curr = sec_start
            for l in lines:
                b_s = extracted_text.find(l, b_curr)
                if b_s != -1:
                    b_e = b_s + len(l)
                    b_curr = b_e
                else:
                    b_s = b_curr
                    b_e = b_curr + len(l)
                    b_curr = b_e + 2
                b_list.append({"text": l, "page": 1, "char_start": b_s, "char_end": b_e})

            sec = ParsedSection(
                title=sec_title,
                content=sec_content,
                page=1,
                page_start=1,
                page_end=1,
                char_start=sec_start,
                char_end=sec_end,
                blocks=b_list
            )
            return sec, new_cursor

        for para in raw_paragraphs:
            para_clean = para.strip()
            if not para_clean:
                continue

            # Check if this paragraph is a heading
            header_candidate = para_clean.strip(" *#_")
            is_heading = bool(heading_pattern.match(header_candidate)) and len(header_candidate.split()) <= 10

            if is_heading and current_section_lines:
                sec, search_cursor = build_section(current_section_title, current_section_lines, search_cursor)
                if sec:
                    sections.append(sec)
                current_section_title = header_candidate
                current_section_lines = []
            elif is_heading and not current_section_lines:
                current_section_title = header_candidate
            else:
                current_section_lines.append(para_clean)

        if current_section_lines:
            sec, search_cursor = build_section(current_section_title, current_section_lines, search_cursor)
            if sec:
                sections.append(sec)

        # If no sections could be broken down, wrap entire text
        if not sections and extracted_text:
            sections.append(ParsedSection(
                title=title or "Body",
                content=extracted_text,
                page=1,
                page_start=1,
                page_end=1,
                char_start=0,
                char_end=len(extracted_text),
                blocks=[{"text": extracted_text, "page": 1, "char_start": 0, "char_end": len(extracted_text)}]
            ))

        return CleanedDocument(
            title=title,
            author=author,
            date=date,
            text=extracted_text,
            sections=sections,
            metadata={"sitename": sitename, "url": url}
        )
