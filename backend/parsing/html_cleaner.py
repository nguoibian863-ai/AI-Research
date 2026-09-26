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
    char_start: int = 0
    char_end: int = 0


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
        curr_offset = 0

        # Regex heuristic for headings: short lines (<80 chars), capital start, no terminal period
        heading_pattern = re.compile(r"^(#{1,4}\s+|[A-Z0-9][\w\s\-\:\.]{2,70}(?<!\.))$")

        running_text_offset = 0
        for para in raw_paragraphs:
            para_clean = para.strip()
            if not para_clean:
                continue

            # Check if this paragraph is a heading
            header_candidate = para_clean.strip(" *#_")
            is_heading = bool(heading_pattern.match(header_candidate)) and len(header_candidate.split()) <= 10

            if is_heading and current_section_lines:
                # Flush previous section
                sec_content = "\n\n".join(current_section_lines)
                sec_end = running_text_offset + len(sec_content)
                sections.append(ParsedSection(
                    title=current_section_title,
                    content=sec_content,
                    page=1,
                    char_start=running_text_offset,
                    char_end=sec_end
                ))
                running_text_offset = sec_end + 2
                current_section_title = header_candidate
                current_section_lines = []
            elif is_heading and not current_section_lines:
                current_section_title = header_candidate
            else:
                current_section_lines.append(para_clean)

        if current_section_lines:
            sec_content = "\n\n".join(current_section_lines)
            sec_end = running_text_offset + len(sec_content)
            sections.append(ParsedSection(
                title=current_section_title,
                content=sec_content,
                page=1,
                char_start=running_text_offset,
                char_end=sec_end
            ))

        # If no sections could be broken down, wrap entire text
        if not sections and extracted_text:
            sections.append(ParsedSection(
                title=title or "Body",
                content=extracted_text,
                page=1,
                char_start=0,
                char_end=len(extracted_text)
            ))

        return CleanedDocument(
            title=title,
            author=author,
            date=date,
            text=extracted_text,
            sections=sections,
            metadata={"sitename": sitename, "url": url}
        )
