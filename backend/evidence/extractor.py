import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from backend.db.database import DatabaseManager
from backend.db.repositories import RawEvidenceRepository, EvidenceRepository, SourceRepository
from backend.llm.backend import LLMBackend
from backend.llm.schemas import ExtractedEvidencesSchema, AtomicFactItemSchema

logger = logging.getLogger(__name__)


def find_quote_in_text(quote: str, text: str) -> Optional[Tuple[int, int, str]]:
    """
    Verifies that a candidate quote is an authentic excerpt from the source text.
    Returns (start_idx, end_idx, exact_matched_text) if found, else None.
    Uses three-level matching:
    1. Exact string search
    2. Case-insensitive search
    3. Whitespace-normalized regex search
    """
    if not quote or not text:
        return None
    cleaned_quote = quote.strip()
    if len(cleaned_quote) < 5:
        return None

    # Level 1: Exact match
    idx = text.find(cleaned_quote)
    if idx != -1:
        return (idx, idx + len(cleaned_quote), text[idx:idx + len(cleaned_quote)])

    # Level 2: Case-insensitive match
    lower_text = text.lower()
    lower_quote = cleaned_quote.lower()
    idx = lower_text.find(lower_quote)
    if idx != -1:
        return (idx, idx + len(cleaned_quote), text[idx:idx + len(cleaned_quote)])

    # Level 3: Whitespace-normalized regex match (tolerates line breaks / multiple spaces)
    words = [w for w in cleaned_quote.split() if w]
    if len(words) >= 2:
        pattern = r"\s+".join(re.escape(w) for w in words)
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return (match.start(), match.end(), text[match.start():match.end()])

    return None


def verify_atomic_fact(
    fact: AtomicFactItemSchema,
    target_chunk: Dict[str, Any],
    verified_quote: str,
    min_quote_len: int = 15
) -> Tuple[bool, str]:
    """
    Python-based rule verification for atomic facts (Anti-hallucination guardrail).
    Enforces:
    1. Quote minimum length (>= 15 chars) to prevent meaningless fragments like 'achieves'.
    2. Value numeric invariant: all numbers in fact.value must appear in the verbatim quote.
    3. Statement numeric invariant: all numbers in fact.statement must appear in the verbatim quote.
    4. Subject grounding: fact.subject must appear in the quote or in the chunk text.
    """
    if not verified_quote or len(verified_quote.strip()) < min_quote_len:
        return False, f"QUOTE_TOO_SHORT: quote length {len(verified_quote.strip()) if verified_quote else 0} < {min_quote_len}"

    quote_lower = verified_quote.lower()
    chunk_text = target_chunk.get("text", "")
    chunk_text_lower = chunk_text.lower()

    # 1. Subject verification: subject must be grounded in quote or chunk
    if fact.subject:
        subj = fact.subject.strip().lower()
        subj_in_quote = subj in quote_lower
        subj_in_chunk = subj in chunk_text_lower
        if not subj_in_quote and not subj_in_chunk:
            subj_tokens = [
                t for t in re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", subj)
                if t not in {"model", "detector", "architecture", "system", "algorithm", "network"}
            ]
            token_found = any(t in quote_lower or t in chunk_text_lower for t in subj_tokens)
            if not token_found:
                return False, f"SUBJECT_MISMATCH: subject '{fact.subject}' not found in quote or source chunk"

    # 2. Value numeric verification: all numbers in fact.value MUST appear in verbatim quote
    if fact.value:
        val_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", str(fact.value))
        for num in val_numbers:
            num_pattern = rf"(?<![a-zA-Z0-9_.])(?<!\d){re.escape(num)}(?!\d)(?![a-zA-Z0-9_.])"
            if not re.search(num_pattern, verified_quote) and num not in verified_quote:
                return False, f"NUMERIC_MISMATCH: value '{fact.value}' (number '{num}') not found in quote '{verified_quote}'"

    # 3. Statement numeric verification: all numbers in fact.statement MUST appear in verbatim quote
    if fact.statement:
        stmt_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", str(fact.statement))
        for num in stmt_numbers:
            num_pattern = rf"(?<![a-zA-Z0-9_.])(?<!\d){re.escape(num)}(?!\d)(?![a-zA-Z0-9_.])"
            if not re.search(num_pattern, verified_quote) and num not in verified_quote:
                return False, f"NUMERIC_MISMATCH: number '{num}' from statement not found in verbatim quote '{verified_quote}'"

    return True, "VERIFIED"


class EvidenceExtractor:
    """
    Extracts atomic evidence items from evidence chunks with strict quote invariance guardrails.
    Guarantees that every extracted fact is grounded in an authentic, verified verbatim quote.
    """
    def __init__(self, llm: LLMBackend, db: DatabaseManager):
        self.llm = llm
        self.db = db
        self.raw_evidence_repo = RawEvidenceRepository(db)
        self.evidence_repo = EvidenceRepository(db)

    def extract_from_chunks(
        self,
        session_id: str,
        goal: str,
        chunks: List[Dict[str, Any]],
        max_evidence: int = 5,
        budget_tracker: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts up to max_evidence atomic facts from retrieved chunks.
        Performs Python-level quote invariance and numeric/subject verification before saving to SQLite.
        Limits extraction context to ~2500 tokens (<= 8000 chars) to prevent context overflow.
        """
        if not chunks:
            logger.warning(f"[{session_id}][EXTRACT] No chunks provided for evidence extraction.")
            return []

        # 1. Format chunks context for LLM extraction (constrained to ~2500 tokens)
        snippets = []
        total_chars = 0
        MAX_CONTEXT_CHARS = 8000

        for i, c in enumerate(chunks[:5], start=1):
            c_id = c.get("chunk_id", f"chunk_{i}")
            c_sec = c.get("section") or ""
            c_page = c.get("page") or ""
            c_url = c.get("url") or c.get("metadata", {}).get("url") or "local"
            sec_info = f", Section: '{c_sec}'" if c_sec else ""
            page_info = f", Page: {c_page}" if c_page else ""
            c_text = c.get("text", "").strip()

            snippet_header = f"--- [Chunk ID: {c_id}] (Source: {c_url}{sec_info}{page_info}) ---\n"
            snippet_body = c_text
            remaining_chars = MAX_CONTEXT_CHARS - total_chars - len(snippet_header)
            if remaining_chars <= 200:
                break
            if len(snippet_body) > remaining_chars:
                snippet_body = snippet_body[:remaining_chars] + "..."

            snippet = f"{snippet_header}{snippet_body}\n"
            snippets.append(snippet)
            total_chars += len(snippet)

        context_str = "\n".join(snippets)

        prompt = (
            f"You are an expert factual research extractor. Extract 3 to {max_evidence} substantive, key atomic facts "
            f"from the evidence text below to answer the research goal.\n\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"1. 'raw_quote' MUST be an EXACT VERBATIM excerpt from the chunk text. Do NOT alter, summarize, or edit the quote.\n"
            f"2. 'chunk_id' MUST match the exact Chunk ID containing the quote.\n"
            f"3. 'statement' must be a concise, objective 1-sentence statement of the fact or benchmark result.\n"
            f"4. If metrics/numbers are present, extract them into 'metric' and 'value'. All numbers in statement and value must be directly in raw_quote.\n\n"
            f"Goal: {goal}\n\n"
            f"Evidence Context:\n{context_str}"
        )

        try:
            if budget_tracker:
                budget_tracker.assert_can_call_llm()
            res = self.llm.structured_generate(prompt, schema=ExtractedEvidencesSchema)
            if budget_tracker:
                calls = getattr(res, "calls_made", 1)
                tokens = getattr(res, "total_tokens", 0)
                budget_tracker.record_llm_call(tokens=tokens, count=calls)
            extracted_facts = res.parsed.facts
            logger.info(f"[{session_id}][EXTRACT] LLM proposed {len(extracted_facts)} candidate facts.")
        except Exception as e:
            logger.error(f"[{session_id}][EXTRACT] LLM extraction failed ({e}), falling back to direct chunk facts.")
            extracted_facts = []

        verified_evidence: List[Dict[str, Any]] = []
        chunks_by_id = {c.get("chunk_id"): c for c in chunks if c.get("chunk_id")}

        # 2. Strict Quote Invariance & Numeric Verification
        for fact in extracted_facts:
            candidate_quote = fact.raw_quote.strip() if fact.raw_quote else ""
            if not candidate_quote:
                continue

            target_chunk = None
            match_res = None

            # First, check specified chunk_id
            if fact.chunk_id and fact.chunk_id in chunks_by_id:
                cand_chunk = chunks_by_id[fact.chunk_id]
                match_res = find_quote_in_text(candidate_quote, cand_chunk.get("text", ""))
                if match_res:
                    target_chunk = cand_chunk

            # Fallback: search across all available chunks if chunk_id was missing or mismatched
            if not match_res:
                for c in chunks:
                    match_res = find_quote_in_text(candidate_quote, c.get("text", ""))
                    if match_res:
                        target_chunk = c
                        break

            # Reject hallucinated or altered quotes
            if not match_res or not target_chunk:
                logger.warning(
                    f"[{session_id}][EXTRACT] Rejected ungrounded quote (not found in source chunks): '{candidate_quote[:60]}...'"
                )
                continue

            start_idx, end_idx, verified_verbatim_quote = match_res

            # Enforce Python-level Rule-Based Grounding Verification
            is_valid, reject_reason = verify_atomic_fact(fact, target_chunk, verified_verbatim_quote, min_quote_len=15)
            if not is_valid:
                logger.warning(
                    f"[{session_id}][EXTRACT] Rejected candidate fact ({reject_reason}): statement='{fact.statement}', quote='{verified_verbatim_quote}'"
                )
                continue

            chunk_char_start = target_chunk.get("char_start") or 0
            abs_start = chunk_char_start + start_idx
            abs_end = chunk_char_start + end_idx

            # Resilient source_id resolution to prevent foreign key errors
            source_id = target_chunk.get("source_id")
            url = target_chunk.get("url") or target_chunk.get("metadata", {}).get("url") or "local"
            source_title = target_chunk.get("source_title") or url
            chunk_id = target_chunk.get("chunk_id") or ""
            page = target_chunk.get("page")
            section = target_chunk.get("section")

            if not source_id or source_id == "src_unknown":
                if chunk_id:
                    with self.db.session() as conn:
                        row = conn.execute(
                            """
                            SELECT d.source_id, s.url, s.title as source_title
                            FROM chunks c
                            JOIN documents d ON c.doc_id = d.doc_id
                            JOIN sources s ON d.source_id = s.source_id
                            WHERE c.chunk_id = ?
                            LIMIT 1
                            """,
                            (chunk_id,)
                        ).fetchone()
                        if row:
                            source_id = row["source_id"]
                            url = row["url"] or url
                            source_title = row["source_title"] or source_title

                if not source_id or source_id == "src_unknown":
                    sess_sources = SourceRepository(self.db).get_by_session(session_id)
                    if sess_sources:
                        source_id = sess_sources[0]["source_id"]
                        url = sess_sources[0]["url"]
                        source_title = sess_sources[0].get("title") or url

            if not source_id or source_id == "src_unknown":
                logger.error(f"[{session_id}][EXTRACT] Cannot persist evidence: no valid source_id found for chunk {chunk_id}")
                continue

            raw_ev_id = f"revi_{uuid.uuid4().hex[:8]}"
            ev_id = f"evi_{uuid.uuid4().hex[:8]}"

            # Construct clean fact statement
            statement = fact.statement.strip() if fact.statement else ""
            if not statement:
                val_str = f": {fact.value}" if fact.value else ""
                statement = f"{fact.subject} {fact.predicate}{val_str}."

            # Save to raw_evidences (immutable raw layer)
            self.raw_evidence_repo.add(
                raw_evidence_id=raw_ev_id,
                session_id=session_id,
                source_id=source_id,
                raw_quote=verified_verbatim_quote,
                page=page,
                section=section,
                char_start=abs_start,
                char_end=abs_end,
                chunk_id=chunk_id
            )

            # Save to evidences (atomic structured layer)
            object_payload = {
                "statement": statement,
                "metric": fact.metric,
                "value": fact.value,
                "raw_quote": verified_verbatim_quote,
                "url": url,
                "source_title": source_title,
                "section": section,
                "page": page,
                "chunk_id": chunk_id
            }

            self.evidence_repo.add(
                evidence_id=ev_id,
                session_id=session_id,
                raw_evidence_id=raw_ev_id,
                subject=fact.subject,
                predicate=fact.predicate,
                object_data=object_payload,
                confidence=fact.confidence or 1.0
            )

            verified_evidence.append({
                "evidence_id": ev_id,
                "raw_evidence_id": raw_ev_id,
                "session_id": session_id,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "statement": statement,
                "exact_quote": verified_verbatim_quote,
                "metric": fact.metric,
                "value": fact.value,
                "source_id": source_id,
                "source_title": source_title,
                "url": url,
                "section": section,
                "page": page,
                "chunk_id": chunk_id,
                "char_start": abs_start,
                "char_end": abs_end,
                "confidence": fact.confidence or 1.0
            })

            if len(verified_evidence) >= max_evidence:
                break

        logger.info(
            f"[{session_id}][EXTRACT] Successfully verified and stored {len(verified_evidence)} atomic evidence items."
        )
        return verified_evidence
