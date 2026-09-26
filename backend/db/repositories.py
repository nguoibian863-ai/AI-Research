import json
import sqlite3
from typing import List, Optional, Dict, Any, Set
from backend.db.database import DatabaseManager


# Sentinel distinguishing "leave column unchanged" from an explicit NULL.
_UNSET: Any = object()


class SessionRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def create(self, session_id: str, goal: str, plan_json: Optional[str] = None, phase: str = "INIT", status: str = "PENDING") -> None:
        with self.db.session() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, goal, plan_json, open_questions_json, error_message, step, phase, status, search_calls, fetch_calls, llm_calls, tokens_consumed, updated_at)
                VALUES (?, ?, ?, '[]', NULL, 0, ?, ?, 0, 0, 0, 0, CURRENT_TIMESTAMP)
                """,
                (session_id, goal, plan_json, phase, status)
            )

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def update_state(
        self,
        session_id: str,
        phase: str,
        step: int,
        status: str,
        plan_json: Optional[str] = None,
        open_questions_json: Optional[str] = None,
        error_message: Optional[str] = _UNSET,
        search_calls: Optional[int] = None,
        fetch_calls: Optional[int] = None,
        llm_calls: Optional[int] = None,
        tokens_consumed: Optional[int] = None
    ) -> None:
        with self.db.session() as conn:
            updates = ["phase = ?", "step = ?", "status = ?", "updated_at = CURRENT_TIMESTAMP"]
            params: List[Any] = [phase, step, status]
            if plan_json is not None:
                updates.append("plan_json = ?")
                params.append(plan_json)
            if open_questions_json is not None:
                updates.append("open_questions_json = ?")
                params.append(open_questions_json)
            if error_message is not _UNSET:
                updates.append("error_message = ?")
                params.append(error_message)
            if search_calls is not None:
                updates.append("search_calls = ?")
                params.append(search_calls)
            if fetch_calls is not None:
                updates.append("fetch_calls = ?")
                params.append(fetch_calls)
            if llm_calls is not None:
                updates.append("llm_calls = ?")
                params.append(llm_calls)
            if tokens_consumed is not None:
                updates.append("tokens_consumed = ?")
                params.append(tokens_consumed)

            params.append(session_id)
            query = f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?"
            conn.execute(query, params)

    def update_phase_and_step(self, session_id: str, phase: str, step: int, status: str, plan_json: Optional[str] = None) -> None:
        self.update_state(session_id, phase, step, status, plan_json=plan_json)


class QueryRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def add(self, query_id: str, session_id: str, query_text: str, query_type: str = "discovery") -> None:
        with self.db.session() as conn:
            conn.execute(
                """
                INSERT INTO queries (query_id, session_id, query_text, query_type)
                VALUES (?, ?, ?, ?)
                """,
                (query_id, session_id, query_text, query_type)
            )

    def get_visited(self, session_id: str) -> List[str]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT query_text FROM queries WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
            return [row["query_text"] for row in cur.fetchall()]


class SourceRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def add(self, source_id: str, session_id: str, url: str, title: str, domain: str,
            source_type: str = "web", canonical_key: Optional[str] = None, authors: Optional[List[str]] = None,
            published_at: Optional[str] = None) -> None:
        authors_json = json.dumps(authors or [])
        with self.db.session() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sources (source_id, session_id, url, title, authors_json, published_at, source_type, domain, canonical_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (source_id, session_id, url, title, authors_json, published_at, source_type, domain, canonical_key)
            )

    def get_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT * FROM sources WHERE session_id = ? ORDER BY retrieved_at ASC", (session_id,))
            return [dict(row) for row in cur.fetchall()]

    def get_ids_by_session(self, session_id: str) -> List[str]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT source_id FROM sources WHERE session_id = ? ORDER BY retrieved_at ASC", (session_id,))
            return [row["source_id"] for row in cur.fetchall()]

    def get_urls_by_session(self, session_id: str) -> Set[str]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT url FROM sources WHERE session_id = ?", (session_id,))
            return {row["url"] for row in cur.fetchall() if row["url"]}

    def find_by_canonical_key(self, session_id: str, canonical_key: str) -> Optional[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT * FROM sources WHERE session_id = ? AND canonical_key = ?", (session_id, canonical_key))
            row = cur.fetchone()
            return dict(row) if row else None


class DocumentRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def add(self, doc_id: str, source_id: str, file_path: Optional[str], content_hash: str, raw_text: str) -> None:
        with self.db.session() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (doc_id, source_id, file_path, content_hash, raw_text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc_id, source_id, file_path, content_hash, raw_text)
            )

    def get_by_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT * FROM documents WHERE source_id = ?", (source_id,))
            row = cur.fetchone()
            return dict(row) if row else None


class ChunkRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def add(self, chunk_id: str, doc_id: str, text: str, page: Optional[int] = None,
            section: Optional[str] = None, char_start: Optional[int] = None,
            char_end: Optional[int] = None, token_count: Optional[int] = None) -> None:
        with self.db.session() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, page, section, char_start, char_end, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, doc_id, text, page, section, char_start, char_end, token_count)
            )

    def add_batch(self, chunks: List[Dict[str, Any]]) -> None:
        with self.db.session() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, page, section, char_start, char_end, token_count)
                VALUES (:chunk_id, :doc_id, :text, :page, :section, :char_start, :char_end, :token_count)
                """,
                chunks
            )

    def get(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_by_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY page ASC, char_start ASC", (doc_id,))
            return [dict(row) for row in cur.fetchall()]

    def get_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        with self.db.session() as conn:
            query = """
            SELECT c.*, s.url, s.source_id, s.title as source_title
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            JOIN sources s ON d.source_id = s.source_id
            WHERE s.session_id = ?
            ORDER BY c.page ASC, c.char_start ASC
            """
            cur = conn.execute(query, (session_id,))
            return [dict(row) for row in cur.fetchall()]

    def count_by_session(self, session_id: str) -> int:
        with self.db.session() as conn:
            query = """
            SELECT COUNT(*) as cnt
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            JOIN sources s ON d.source_id = s.source_id
            WHERE s.session_id = ?
            """
            cur = conn.execute(query, (session_id,))
            row = cur.fetchone()
            return row["cnt"] if row else 0


class RawEvidenceRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def add(self, raw_evidence_id: str, session_id: str, source_id: str, raw_quote: str,
            page: Optional[int] = None, section: Optional[str] = None,
            char_start: Optional[int] = None, char_end: Optional[int] = None,
            chunk_id: Optional[str] = None) -> None:
        with self.db.session() as conn:
            conn.execute(
                """
                INSERT INTO raw_evidences (raw_evidence_id, session_id, source_id, raw_quote, page, section, char_start, char_end, chunk_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (raw_evidence_id, session_id, source_id, raw_quote, page, section, char_start, char_end, chunk_id)
            )

    def get(self, raw_evidence_id: str) -> Optional[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT * FROM raw_evidences WHERE raw_evidence_id = ?", (raw_evidence_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT * FROM raw_evidences WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
            return [dict(row) for row in cur.fetchall()]

    def get_ids_by_session(self, session_id: str) -> List[str]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT raw_evidence_id FROM raw_evidences WHERE session_id = ?", (session_id,))
            return [row["raw_evidence_id"] for row in cur.fetchall()]


class EvidenceRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def add(self, evidence_id: str, session_id: str, raw_evidence_id: str,
            subject: str, predicate: str, object_data: Dict[str, Any], confidence: float = 1.0) -> None:
        object_json = json.dumps(object_data)
        with self.db.session() as conn:
            conn.execute(
                """
                INSERT INTO evidences (evidence_id, session_id, raw_evidence_id, subject, predicate, object_json, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (evidence_id, session_id, raw_evidence_id, subject, predicate, object_json, confidence)
            )

    def get_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT * FROM evidences WHERE session_id = ?", (session_id,))
            return [dict(row) for row in cur.fetchall()]

    def get_ids_by_session(self, session_id: str) -> List[str]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT evidence_id FROM evidences WHERE session_id = ?", (session_id,))
            return [row["evidence_id"] for row in cur.fetchall()]

    def get_full_evidence_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        with self.db.session() as conn:
            query = """
            SELECT e.evidence_id, e.session_id, e.raw_evidence_id, e.subject, e.predicate,
                   e.object_json, e.confidence, e.created_at,
                   r.raw_quote, r.page, r.section, r.char_start, r.char_end, r.chunk_id,
                   s.source_id, s.url, s.title as source_title
            FROM evidences e
            JOIN raw_evidences r ON e.raw_evidence_id = r.raw_evidence_id
            JOIN sources s ON r.source_id = s.source_id
            WHERE e.session_id = ?
            ORDER BY e.created_at ASC
            """
            cur = conn.execute(query, (session_id,))
            rows = []
            for row in cur.fetchall():
                item = dict(row)
                obj = json.loads(item["object_json"]) if item.get("object_json") else {}
                item["object_data"] = obj
                item["statement"] = obj.get("statement") or f"{item.get('subject', '')} {item.get('predicate', '')}"
                item["metric"] = obj.get("metric")
                item["value"] = obj.get("value")
                item["exact_quote"] = item.get("raw_quote", "")
                rows.append(item)
            return rows


class ClaimRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def add(self, claim_id: str, session_id: str, text: str, evidence_ids: List[str],
            verification: Dict[str, Any], status: str = "SUPPORTED") -> None:
        ev_json = json.dumps(evidence_ids)
        ver_json = json.dumps(verification)
        with self.db.session() as conn:
            conn.execute(
                """
                INSERT INTO claims (claim_id, session_id, text, evidence_ids_json, verification_json, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (claim_id, session_id, text, ev_json, ver_json, status)
            )

    def get_ids_by_session(self, session_id: str) -> List[str]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT claim_id FROM claims WHERE session_id = ?", (session_id,))
            return [row["claim_id"] for row in cur.fetchall()]

    def get_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT * FROM claims WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
            rows = []
            for row in cur.fetchall():
                item = dict(row)
                item["evidence_ids"] = json.loads(item["evidence_ids_json"]) if item.get("evidence_ids_json") else []
                item["verification"] = json.loads(item["verification_json"]) if item.get("verification_json") else {}
                rows.append(item)
            return rows


class ReportRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def create(self, report_id: str, session_id: str, title: str, content_markdown: str,
               claims_json: Optional[str] = None, status: str = "COMPLETED") -> None:
        with self.db.session() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reports (report_id, session_id, title, content_markdown, claims_json, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (report_id, session_id, title, content_markdown, claims_json or "[]", status)
            )

    def get_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT * FROM reports WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
            return dict(row) if row else None


class TrajectoryRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def add(self, trajectory_id: str, session_id: str, task_type: str, model_source: str,
            payload: Dict[str, Any], verified: bool = False, quality_score: float = 0.0,
            partition: str = "raw") -> None:
        payload_json = json.dumps(payload)
        with self.db.session() as conn:
            conn.execute(
                """
                INSERT INTO trajectories (trajectory_id, session_id, task_type, model_source, verified, quality_score, payload_json, partition)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (trajectory_id, session_id, task_type, model_source, 1 if verified else 0, quality_score, payload_json, partition)
            )

    def list_by_partition(self, partition: str) -> List[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT * FROM trajectories WHERE partition = ? ORDER BY created_at DESC", (partition,))
            return [dict(row) for row in cur.fetchall()]
