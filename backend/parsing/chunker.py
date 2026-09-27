import re
import uuid
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from backend.parsing.html_cleaner import CleanedDocument, ParsedSection

logger = logging.getLogger(__name__)


class ParsedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    page: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0



class SectionAwareChunker:
    """
    Contextual section-based chunking with token overlap.
    Hierarchy: Section -> Heading -> Paragraph -> Sentence -> Token Limit.
    """

    def __init__(self, max_chunk_tokens: int = 350, chunk_overlap_tokens: int = 80, max_chunk_chars: int = 1800):
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.max_chunk_chars = max_chunk_chars

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimates token count (~1.3 tokens per word or whitespace split)."""
        words = text.split()
        return max(1, int(len(words) * 1.3))

    def chunk_document(self, doc_id: str, document: CleanedDocument) -> List[ParsedChunk]:
        """Chunks a cleaned document into section-aware chunks respecting max tokens & overlap."""
        chunks: List[ParsedChunk] = []
        chunk_idx = 0
        full_text = document.text or ""

        # If document has no pre-detected sections, treat the entire text as one section
        sections = document.sections
        if not sections and full_text.strip():
            sections = [
                ParsedSection(
                    title=document.title or "General",
                    content=full_text,
                    page=1,
                    page_start=1,
                    page_end=1,
                    char_start=0,
                    char_end=len(full_text),
                    blocks=[{"text": full_text, "page": 1, "char_start": 0, "char_end": len(full_text)}]
                )
            ]

        for sec in sections:
            sec_chunks = self._chunk_section(doc_id=doc_id, section=sec, full_text=full_text, start_idx=chunk_idx)
            chunks.extend(sec_chunks)
            chunk_idx += len(sec_chunks)

        logger.info(f"[SectionAwareChunker] Generated {len(chunks)} chunks for doc {doc_id}.")
        return chunks

    def _chunk_section(self, doc_id: str, section: ParsedSection, full_text: str, start_idx: int) -> List[ParsedChunk]:
        """Splits a single section into chunks while keeping section context and page attribution."""
        content = section.content.strip()
        if not content:
            return []

        # Ensure we have blocks with page and char offset info
        blocks = section.blocks
        if not blocks:
            # Reconstruct blocks from section content
            blocks = []
            sec_start = section.char_start
            paras = [p.strip() for p in re.split(r"\n\s*\n", section.content) if p.strip()]
            if not paras:
                paras = [content]
            cursor = sec_start
            for p in paras:
                pos = full_text.find(p, cursor)
                if pos == -1:
                    pos = cursor
                blocks.append({
                    "text": p,
                    "page": section.page or 1,
                    "char_start": pos,
                    "char_end": pos + len(p)
                })
                cursor = pos + len(p)

        section_chunks: List[ParsedChunk] = []
        curr_blocks: List[Dict[str, Any]] = []
        curr_tokens = 0
        curr_chars = 0

        def emit_chunk(b_list: List[Dict[str, Any]], chunk_num: int) -> Optional[ParsedChunk]:
            if not b_list:
                return None
            c_start = b_list[0]["char_start"]
            c_end = b_list[-1]["char_end"]
            chunk_text = full_text[c_start:c_end]
            if not chunk_text.strip():
                return None
            t_count = self.estimate_tokens(chunk_text)
            return ParsedChunk(
                chunk_id=f"chk_{doc_id}_{chunk_num:03d}",
                doc_id=doc_id,
                text=chunk_text,
                page=b_list[0].get("page", section.page or 1),
                page_end=b_list[-1].get("page", section.page or 1),
                section=section.title,
                char_start=c_start,
                char_end=c_end,
                token_count=t_count
            )

        for b in blocks:
            b_text = b["text"]
            b_tokens = self.estimate_tokens(b_text)
            b_chars = len(b_text)

            # If a single block exceeds max_chunk_tokens or max_chunk_chars, split it by sentence
            if b_tokens > self.max_chunk_tokens or b_chars > self.max_chunk_chars:
                if curr_blocks:
                    c = emit_chunk(curr_blocks, start_idx + len(section_chunks))
                    if c:
                        section_chunks.append(c)
                    curr_blocks = []
                    curr_tokens = 0
                    curr_chars = 0

                sub_chunks = self._chunk_large_block(doc_id, section, b, full_text, start_idx + len(section_chunks))
                section_chunks.extend(sub_chunks)
                continue

            if (curr_tokens + b_tokens > self.max_chunk_tokens or curr_chars + b_chars > self.max_chunk_chars) and curr_blocks:
                c = emit_chunk(curr_blocks, start_idx + len(section_chunks))
                if c:
                    section_chunks.append(c)

                # Check if last block can be kept as overlap within character limit
                last_b = curr_blocks[-1]
                last_b_tokens = self.estimate_tokens(last_b["text"])
                last_b_chars = len(last_b["text"])
                if last_b_tokens <= self.chunk_overlap_tokens and last_b_chars + b_chars <= self.max_chunk_chars:
                    curr_blocks = [last_b, b]
                    curr_tokens = last_b_tokens + b_tokens
                    curr_chars = last_b_chars + b_chars
                else:
                    curr_blocks = [b]
                    curr_tokens = b_tokens
                    curr_chars = b_chars
            else:
                curr_blocks.append(b)
                curr_tokens += b_tokens
                curr_chars += b_chars

        if curr_blocks:
            c = emit_chunk(curr_blocks, start_idx + len(section_chunks))
            if c:
                section_chunks.append(c)

        return section_chunks

    def _chunk_large_block(self, doc_id: str, section: ParsedSection, block: Dict[str, Any], full_text: str, start_num: int) -> List[ParsedChunk]:
        """Splits an oversized block into sentence-level chunks with exact char offsets."""
        b_text = block["text"]
        b_start = block["char_start"]
        b_page = block.get("page", section.page or 1)

        matches = list(re.finditer(r"(?<=[.!?])\s+", b_text))
        sentence_spans = []
        last_idx = 0
        for m in matches:
            sentence_spans.append((last_idx, m.start()))
            last_idx = m.end()
        if last_idx < len(b_text):
            sentence_spans.append((last_idx, len(b_text)))

        if not sentence_spans:
            sentence_spans = [(0, len(b_text))]

        chunks: List[ParsedChunk] = []
        curr_spans = []
        curr_tokens = 0
        curr_chars = 0

        def emit_sentence_chunk(spans: List[tuple], chunk_idx: int) -> Optional[ParsedChunk]:
            if not spans:
                return None
            start_in_b = spans[0][0]
            end_in_b = spans[-1][1]
            abs_start = b_start + start_in_b
            abs_end = b_start + end_in_b
            chunk_text = full_text[abs_start:abs_end]
            if not chunk_text.strip():
                return None
            return ParsedChunk(
                chunk_id=f"chk_{doc_id}_{chunk_idx:03d}",
                doc_id=doc_id,
                text=chunk_text,
                page=b_page,
                page_end=b_page,
                section=section.title,
                char_start=abs_start,
                char_end=abs_end,
                token_count=self.estimate_tokens(chunk_text)
            )

        for span in sentence_spans:
            s_text = b_text[span[0]:span[1]]
            s_tokens = self.estimate_tokens(s_text)
            s_chars = len(s_text)

            # If a single sentence exceeds max_chunk_chars, split it cleanly
            if s_chars > self.max_chunk_chars:
                if curr_spans:
                    c = emit_sentence_chunk(curr_spans, start_num + len(chunks))
                    if c:
                        chunks.append(c)
                    curr_spans = []
                    curr_tokens = 0
                    curr_chars = 0
                sub_start = span[0]
                while sub_start < span[1]:
                    sub_end = min(span[1], sub_start + self.max_chunk_chars)
                    if sub_end < span[1]:
                        space_idx = b_text.rfind(" ", sub_start, sub_end)
                        if space_idx > sub_start + 100:
                            sub_end = space_idx
                    c = emit_sentence_chunk([(sub_start, sub_end)], start_num + len(chunks))
                    if c:
                        chunks.append(c)
                    sub_start = sub_end
                continue

            if (curr_tokens + s_tokens > self.max_chunk_tokens or curr_chars + s_chars > self.max_chunk_chars) and curr_spans:
                c = emit_sentence_chunk(curr_spans, start_num + len(chunks))
                if c:
                    chunks.append(c)

                last_span = curr_spans[-1]
                last_text = b_text[last_span[0]:last_span[1]]
                last_tokens = self.estimate_tokens(last_text)
                if last_tokens <= self.chunk_overlap_tokens and len(last_text) + s_chars <= self.max_chunk_chars:
                    curr_spans = [last_span, span]
                    curr_tokens = last_tokens + s_tokens
                    curr_chars = len(last_text) + s_chars
                else:
                    curr_spans = [span]
                    curr_tokens = s_tokens
                    curr_chars = s_chars
            else:
                curr_spans.append(span)
                curr_tokens += s_tokens
                curr_chars += s_chars

        if curr_spans:
            c = emit_sentence_chunk(curr_spans, start_num + len(chunks))
            if c:
                chunks.append(c)

        return chunks

