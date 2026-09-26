import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import pymupdf
from backend.parsing.html_cleaner import ParsedSection, CleanedDocument

logger = logging.getLogger(__name__)

# Common academic & technical paper section patterns
SECTION_HEADER_REGEX = re.compile(
    r"^(?:(?:[0-9]+(?:\.[0-9]+)*\.?|[0-9]+\.)\s+)?(?:Abstract|Introduction|Related Work|Background|Methodology|Method|Architecture|Implementation|Experiments|Results|Evaluation|Benchmark|Discussion|Ablation|Conclusion|References|Appendix)(?:\s*:.*)?$",
    re.IGNORECASE
)


class PDFParser:
    """
    Page-aware and section-aware PDF parser using PyMuPDF (pymupdf).
    Extracts structured text with page attribution, character offsets, and metadata.
    """

    @classmethod
    def parse_file(cls, file_path: Union[str, Path]) -> CleanedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        with pymupdf.open(str(path)) as doc:
            return cls._extract_from_doc(doc, source_name=path.name)

    @classmethod
    def parse_bytes(cls, pdf_bytes: bytes, source_name: Optional[str] = None) -> CleanedDocument:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            return cls._extract_from_doc(doc, source_name=source_name or "stream.pdf")

    @classmethod
    def _extract_from_doc(cls, doc: pymupdf.Document, source_name: str) -> CleanedDocument:
        meta = doc.metadata or {}
        title = meta.get("title") or source_name
        author = meta.get("author") or None
        date = meta.get("creationDate") or None

        sections: List[ParsedSection] = []
        full_text_parts: List[str] = []
        running_char_offset = 0

        current_section_title = "Abstract" if len(doc) > 0 else "Introduction"
        current_section_text: List[str] = []
        current_section_page = 1
        current_section_start = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1  # 1-indexed

            # Get blocks: (x0, y0, x1, y1, text, block_no, block_type)
            blocks = page.get_text("blocks")
            for b in blocks:
                # block_type 0 is text
                if len(b) >= 7 and b[6] != 0:
                    continue

                raw_text = b[4].strip()
                if not raw_text:
                    continue

                # Clean headers/footers heuristic (page numbers like "1", "12 of 30")
                if len(raw_text) <= 5 and raw_text.isdigit():
                    continue

                # Check if this block looks like a section header
                lines = raw_text.splitlines()
                first_line = lines[0].strip() if lines else ""
                is_section_header = (
                    bool(SECTION_HEADER_REGEX.match(first_line)) or
                    (len(lines) == 1 and len(first_line) < 60 and first_line.isupper() and len(first_line.split()) <= 6)
                )

                if is_section_header:
                    # Flush previous section if it has content
                    if current_section_text:
                        sec_str = "\n\n".join(current_section_text)
                        sec_len = len(sec_str)
                        sections.append(ParsedSection(
                            title=current_section_title,
                            content=sec_str,
                            page=current_section_page,
                            char_start=current_section_start,
                            char_end=current_section_start + sec_len
                        ))
                        full_text_parts.append(sec_str)
                        running_char_offset += sec_len + 2  # account for \n\n separator

                    current_section_title = first_line
                    current_section_page = page_num
                    current_section_start = running_char_offset
                    # If block had more than 1 line, rest is content
                    remainder = "\n".join(lines[1:]).strip()
                    current_section_text = [remainder] if remainder else []
                else:
                    current_section_text.append(raw_text)

        # Flush final section
        if current_section_text:
            sec_str = "\n\n".join(current_section_text)
            sec_len = len(sec_str)
            sections.append(ParsedSection(
                title=current_section_title,
                content=sec_str,
                page=current_section_page,
                char_start=current_section_start,
                char_end=current_section_start + sec_len
            ))
            full_text_parts.append(sec_str)

        full_document_text = "\n\n".join(full_text_parts)

        # Fallback if no sections extracted
        if not sections and full_document_text:
            sections.append(ParsedSection(
                title=title,
                content=full_document_text,
                page=1,
                char_start=0,
                char_end=len(full_document_text)
            ))

        return CleanedDocument(
            title=title,
            author=author,
            date=date,
            text=full_document_text,
            sections=sections,
            metadata={
                "page_count": len(doc),
                "source": source_name,
                "format": "pdf"
            }
        )
