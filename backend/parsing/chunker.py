import re
import uuid
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from backend.parsing.html_cleaner import CleanedDocument, ParsedSection

logger = logging.getLogger(__name__)


class ParsedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0


class SectionAwareChunker:
    """
    Contextual section-based chunking with token overlap.
    Hierarchy: Section -> Heading -> Paragraph -> Sentence -> Token Limit.
    """

    def __init__(self, max_chunk_tokens: int = 600, chunk_overlap_tokens: int = 80):
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimates token count (~1.3 tokens per word or whitespace split)."""
        words = text.split()
        return max(1, int(len(words) * 1.3))

    def chunk_document(self, doc_id: str, document: CleanedDocument) -> List[ParsedChunk]:
        """Chunks a cleaned document into section-aware chunks respecting max tokens & overlap."""
        chunks: List[ParsedChunk] = []
        chunk_idx = 0

        # If document has no pre-detected sections, treat the entire text as one section
        sections = document.sections
        if not sections and document.text.strip():
            sections = [
                ParsedSection(
                    title=document.title or "General",
                    content=document.text,
                    page=1,
                    char_start=0,
                    char_end=len(document.text)
                )
            ]

        for sec in sections:
            sec_chunks = self._chunk_section(doc_id=doc_id, section=sec, start_idx=chunk_idx)
            chunks.extend(sec_chunks)
            chunk_idx += len(sec_chunks)

        logger.info(f"[SectionAwareChunker] Generated {len(chunks)} chunks for doc {doc_id}.")
        return chunks

    def _chunk_section(self, doc_id: str, section: ParsedSection, start_idx: int) -> List[ParsedChunk]:
        """Splits a single section into chunks while keeping section context and page attribution."""
        content = section.content.strip()
        if not content:
            return []

        # Split section into paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        if not paragraphs:
            paragraphs = [content]

        section_chunks: List[ParsedChunk] = []
        curr_paras: List[str] = []
        curr_tokens = 0
        running_char_offset = section.char_start

        def flush_chunk(paras: List[str], chunk_num: int) -> Optional[ParsedChunk]:
            if not paras:
                return None
            chunk_text = "\n\n".join(paras).strip()
            if not chunk_text:
                return None
            t_count = self.estimate_tokens(chunk_text)
            c_start = section.char_start + content.find(paras[0]) if paras[0] in content else running_char_offset
            c_end = c_start + len(chunk_text)
            return ParsedChunk(
                chunk_id=f"chk_{doc_id}_{chunk_num:03d}",
                doc_id=doc_id,
                text=chunk_text,
                page=section.page or 1,
                section=section.title,
                char_start=c_start,
                char_end=c_end,
                token_count=t_count
            )

        for para in paragraphs:
            para_tokens = self.estimate_tokens(para)

            # If a single paragraph is larger than max_chunk_tokens, split it by sentences
            if para_tokens > self.max_chunk_tokens:
                # Flush existing buffer first
                if curr_paras:
                    c = flush_chunk(curr_paras, start_idx + len(section_chunks))
                    if c:
                        section_chunks.append(c)
                    curr_paras = []
                    curr_tokens = 0

                sub_chunks = self._chunk_large_paragraph(doc_id, section, para, start_idx + len(section_chunks))
                section_chunks.extend(sub_chunks)
                continue

            if curr_tokens + para_tokens > self.max_chunk_tokens and curr_paras:
                # Flush current chunk
                c = flush_chunk(curr_paras, start_idx + len(section_chunks))
                if c:
                    section_chunks.append(c)

                # Implement overlap: take trailing sentences or words from last paragraph
                overlap_text = self._get_overlap_text(curr_paras[-1])
                curr_paras = [overlap_text, para] if overlap_text else [para]
                curr_tokens = self.estimate_tokens("\n\n".join(curr_paras))
            else:
                curr_paras.append(para)
                curr_tokens += para_tokens

        # Flush final remaining paragraphs
        if curr_paras:
            c = flush_chunk(curr_paras, start_idx + len(section_chunks))
            if c:
                section_chunks.append(c)

        return section_chunks

    def _chunk_large_paragraph(self, doc_id: str, section: ParsedSection, para: str, start_num: int) -> List[ParsedChunk]:
        """Splits an oversized paragraph into sentence-level chunks."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        if not sentences:
            sentences = [para]

        chunks: List[ParsedChunk] = []
        curr_sentences: List[str] = []
        curr_tokens = 0

        for sent in sentences:
            s_tokens = self.estimate_tokens(sent)
            if curr_tokens + s_tokens > self.max_chunk_tokens and curr_sentences:
                chunk_text = " ".join(curr_sentences).strip()
                chunks.append(ParsedChunk(
                    chunk_id=f"chk_{doc_id}_{start_num + len(chunks):03d}",
                    doc_id=doc_id,
                    text=chunk_text,
                    page=section.page or 1,
                    section=section.title,
                    char_start=section.char_start,
                    char_end=section.char_start + len(chunk_text),
                    token_count=self.estimate_tokens(chunk_text)
                ))
                # Carry overlap
                overlap = curr_sentences[-1] if len(curr_sentences) > 1 else ""
                curr_sentences = [overlap, sent] if overlap else [sent]
                curr_tokens = self.estimate_tokens(" ".join(curr_sentences))
            else:
                curr_sentences.append(sent)
                curr_tokens += s_tokens

        if curr_sentences:
            chunk_text = " ".join(curr_sentences).strip()
            chunks.append(ParsedChunk(
                chunk_id=f"chk_{doc_id}_{start_num + len(chunks):03d}",
                doc_id=doc_id,
                text=chunk_text,
                page=section.page or 1,
                section=section.title,
                char_start=section.char_start,
                char_end=section.char_start + len(chunk_text),
                token_count=self.estimate_tokens(chunk_text)
            ))

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Extracts the tail portion of text matching roughly chunk_overlap_tokens."""
        words = text.split()
        target_words = int(self.chunk_overlap_tokens / 1.3)
        if len(words) <= target_words:
            return text
        return " ".join(words[-target_words:])
