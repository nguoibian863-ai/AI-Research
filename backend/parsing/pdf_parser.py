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
        current_section_blocks: List[Dict[str, Any]] = []

        def flush_section(sec_title: str, blocks: List[Dict[str, Any]], offset: int):
            if not blocks:
                return None, offset
            curr = offset
            for blk in blocks:
                blk["char_start"] = curr
                blk["char_end"] = curr + len(blk["text"])
                curr = blk["char_end"] + 2  # account for \n\n separator

            sec_content = "\n\n".join(b["text"] for b in blocks)
            sec = ParsedSection(
                title=sec_title,
                content=sec_content,
                page=blocks[0]["page"],
                page_start=blocks[0]["page"],
                page_end=blocks[-1]["page"],
                char_start=offset,
                char_end=offset + len(sec_content),
                blocks=blocks
            )
            return sec, offset + len(sec_content) + 2

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
                    if current_section_blocks:
                        sec, next_offset = flush_section(current_section_title, current_section_blocks, running_char_offset)
                        if sec:
                            sections.append(sec)
                            full_text_parts.append(sec.content)
                            running_char_offset = next_offset

                    current_section_title = first_line
                    # If block had more than 1 line, rest is content
                    remainder = "\n".join(lines[1:]).strip()
                    current_section_blocks = [{"text": remainder, "page": page_num}] if remainder else []
                else:
                    current_section_blocks.append({"text": raw_text, "page": page_num})

        # Flush final section
        if current_section_blocks:
            sec, next_offset = flush_section(current_section_title, current_section_blocks, running_char_offset)
            if sec:
                sections.append(sec)
                full_text_parts.append(sec.content)

        full_document_text = "\n\n".join(full_text_parts)

        # Fallback if no sections extracted
        if not sections and full_document_text:
            sections.append(ParsedSection(
                title=title,
                content=full_document_text,
                page=1,
                page_start=1,
                page_end=len(doc),
                char_start=0,
                char_end=len(full_document_text),
                blocks=[{"text": full_document_text, "page": 1, "char_start": 0, "char_end": len(full_document_text)}]
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

