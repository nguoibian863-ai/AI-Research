import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from backend.db.database import DatabaseManager
from backend.db.repositories import RawEvidenceRepository, EvidenceRepository
from backend.llm.backend import LLMBackend
from backend.llm.schemas import ExtractedEvidencesSchema, AtomicFactItemSchema
from backend.core.coverage import extract_core_entities, check_entity_in_text

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


def expand_quote_to_sentence(start_idx: int, end_idx: int, text: str) -> Tuple[int, int, str]:
    """
    Expands a candidate quote match to the full sentence or line boundaries
    to provide complete context, subject attribution, and metric grounding.
    """
    if not text or (end_idx - start_idx) > 200 or text[start_idx:end_idx].count(".") > 1:
        return start_idx, end_idx, text[start_idx:end_idx]

    # Expand backwards to sentence start (previous period + space, newline, or start of text)
    sent_start = start_idx
    while sent_start > 0:
        prev_char = text[sent_start - 1]
        if prev_char == "\n":
            break
        if prev_char == "." and (sent_start == len(text) or text[sent_start].isspace()):
            break
        sent_start -= 1

    while sent_start < start_idx and text[sent_start].isspace():
        sent_start += 1

    # Expand forwards to sentence end (period + space/newline, newline, or end of text)
    sent_end = end_idx
    while sent_end < len(text):
        char = text[sent_end]
        if char == "\n":
            break
        if char == "." and (sent_end + 1 == len(text) or text[sent_end + 1].isspace()):
            sent_end += 1
            break
        sent_end += 1

    expanded = text[sent_start:sent_end].strip()
    if len(expanded) < (end_idx - start_idx):
        return start_idx, end_idx, text[start_idx:end_idx]

    actual_start = text.find(expanded, sent_start)
    if actual_start != -1:
        return actual_start, actual_start + len(expanded), expanded
    return sent_start, sent_end, expanded


def verify_atomic_fact(
    fact: AtomicFactItemSchema,
    target_chunk: Dict[str, Any],
    verified_quote: str,
    min_quote_len: int = 15,
    goal_entities: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Python-based rule verification for atomic facts (Anti-hallucination guardrail).
    Enforces:
    1. Quote minimum length (>= 15 chars) to prevent meaningless fragments like 'achieves'.
    2. Value numeric invariant: all numbers in fact.value must appear in the verbatim quote.
    3. Statement numeric invariant: all numbers in fact.statement must appear in the verbatim quote.
    4. Subject grounding: fact.subject must appear in the quote or in the chunk text.
    5. Comparative grounding: any secondary entity mentioned in fact.statement must appear in the verbatim quote.
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

    # 4. Comparative & cross-entity grounding:
    # A statement must not make ungrounded comparative claims or assert facts about
    # competitor entities not supported by the quote.
    if fact.statement:
        stmt_lower = fact.statement.lower()
        subj_lower = (fact.subject or "").strip().lower()

        # Check secondary goal entities: if statement mentions another entity from research goal,
        # that entity MUST appear in the verbatim quote.
        if goal_entities:
            for ent in goal_entities:
                if ent in subj_lower or subj_lower in ent:
                    continue
                if check_entity_in_text(ent, stmt_lower):
                    if not check_entity_in_text(ent, quote_lower):
                        return False, f"UNSUPPORTED_COMPARISON: foreign entity '{ent}' in statement not found in verbatim quote"

        # Check comparative syntax: "X is better than Y", "outperforms Y", "beating Y"
        is_comparative = bool(re.search(r"\b(than|outperforms?|beats?|beating|superior to|inferior to|ahead of)\b", stmt_lower))
        if is_comparative:
            stmt_entities = extract_core_entities(fact.statement)
            for ent in stmt_entities:
                if ent in subj_lower or subj_lower in ent:
                    continue
                if not check_entity_in_text(ent, quote_lower):
                    return False, f"UNSUPPORTED_COMPARISON: comparative statement mentions entity '{ent}' not found in verbatim quote"

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
        Processes chunks with focused, few-shot prompts to maximize small model accuracy.
        Performs quote sentence expansion and Python-level rule verification before saving to SQLite.
        """
        if not chunks:
            logger.warning(f"[{session_id}][EXTRACT] No chunks provided for evidence extraction.")
            return []

        verified_evidence: List[Dict[str, Any]] = []
        seen_quotes = set()
        goal_entities = extract_core_entities(goal) if goal else []

        # Iterate over candidate chunks (up to 5 chunks)
        for i, target_chunk in enumerate(chunks[:5], start=1):
            if len(verified_evidence) >= max_evidence:
                break
            if budget_tracker and not budget_tracker.can_call_llm():
                logger.warning(f"[{session_id}][EXTRACT] LLM budget exhausted. Stopping extraction.")
                break

            c_id = target_chunk.get("chunk_id", f"chunk_{i}")
            c_sec = target_chunk.get("section") or ""
            c_page = target_chunk.get("page") or ""
            c_url = target_chunk.get("url") or target_chunk.get("metadata", {}).get("url") or "local"
            sec_info = f", Section: '{c_sec}'" if c_sec else ""
            page_info = f", Page: {c_page}" if c_page else ""
            c_text = target_chunk.get("text", "").strip()

            if len(c_text) < 20:
                continue

            # Focused prompt with neutral few-shot example for high JSON compliance with small models
            prompt = (
                f"You are an expert factual research extractor. Extract key atomic facts from the chunk below.\n\n"
                f"NEUTRAL FEW-SHOT EXAMPLE:\n"
                f'Input Chunk: "RocksDB introduces a log-structured merge-tree architecture that achieves 150,000 writes/sec on NVMe storage."\n'
                f"Output:\n"
                f'{{\n'
                f'  "facts": [\n'
                f'    {{\n'
                f'      "statement": "RocksDB achieves 150,000 writes/sec on NVMe storage.",\n'
                f'      "subject": "RocksDB",\n'
                f'      "predicate": "achieves",\n'
                f'      "metric": "writes/sec",\n'
                f'      "value": "150,000",\n'
                f'      "raw_quote": "RocksDB introduces a log-structured merge-tree architecture that achieves 150,000 writes/sec on NVMe storage.",\n'
                f'      "chunk_id": "{c_id}",\n'
                f'      "confidence": 0.98\n'
                f'    }}\n'
                f'  ]\n'
                f'}}\n\n'
                f"Goal: {goal}\n\n"
                f"Evidence Chunk:\n"
                f"--- [Chunk ID: {c_id}] (Source: {c_url}{sec_info}{page_info}) ---\n"
                f"{c_text}\n"
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
                logger.info(f"[{session_id}][EXTRACT] LLM proposed {len(extracted_facts)} candidate facts for chunk {c_id}.")
            except Exception as e:
                logger.warning(f"[{session_id}][EXTRACT] Extraction call failed for chunk {c_id} ({e}).")
                continue

            for fact in extracted_facts:
                candidate_quote = fact.raw_quote.strip() if fact.raw_quote else ""
                if not candidate_quote:
                    continue

                match_res = find_quote_in_text(candidate_quote, c_text)
                if not match_res:
                    logger.warning(
                        f"[{session_id}][EXTRACT] Rejected ungrounded quote (not in source chunk): '{candidate_quote[:60]}...'"
                    )
                    continue

                start_idx, end_idx, matched_quote = match_res

                # Check if matched_quote itself satisfies rule verification
                is_valid, reject_reason = verify_atomic_fact(
                    fact, target_chunk, matched_quote, min_quote_len=15, goal_entities=goal_entities
                )
                verified_verbatim_quote = matched_quote
                final_start, final_end = start_idx, end_idx

                if not is_valid:
                    # Sentence expansion: expand quote to full sentence boundaries for rich context & grounding
                    exp_start, exp_end, expanded_quote = expand_quote_to_sentence(start_idx, end_idx, c_text)
                    is_valid_exp, reject_reason_exp = verify_atomic_fact(
                        fact, target_chunk, expanded_quote, min_quote_len=15, goal_entities=goal_entities
                    )
                    if is_valid_exp:
                        verified_verbatim_quote = expanded_quote
                        final_start, final_end = exp_start, exp_end
                        is_valid = True
                    else:
                        logger.warning(
                            f"[{session_id}][EXTRACT] Rejected candidate fact ({reject_reason}): statement='{fact.statement}', quote='{matched_quote}'"
                        )
                        continue

                # Deduplicate identical quotes
                if verified_verbatim_quote in seen_quotes:
                    continue

                # Strict source_id resolution: NEVER fallback to unrelated first source
                source_id = target_chunk.get("source_id")
                url = target_chunk.get("url") or target_chunk.get("metadata", {}).get("url") or "local"
                source_title = target_chunk.get("source_title") or url
                chunk_id = target_chunk.get("chunk_id") or c_id
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

                # If source_id cannot be verified, reject the evidence to prevent false attribution!
                if not source_id or source_id == "src_unknown":
                    logger.warning(
                        f"[{session_id}][EXTRACT] Rejected evidence: source_id could not be resolved for chunk {chunk_id}."
                    )
                    continue

                chunk_char_start = target_chunk.get("char_start") or 0
                abs_start = chunk_char_start + final_start
                abs_end = chunk_char_start + final_end

                raw_ev_id = f"revi_{uuid.uuid4().hex[:8]}"
                ev_id = f"evi_{uuid.uuid4().hex[:8]}"

                # Construct clean fact statement
                statement = fact.statement.strip() if fact.statement else ""
                if not statement or len(statement) < 10:
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

                seen_quotes.add(verified_verbatim_quote)
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
