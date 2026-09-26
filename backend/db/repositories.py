import json
import sqlite3
from typing import List, Optional, Dict, Any
from backend.db.database import DatabaseManager


class SessionRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def create(self, session_id: str, goal: str, plan_json: Optional[str] = None, phase: str = "INIT", status: str = "PENDING") -> None:
        with self.db.session() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, goal, plan_json, step, phase, status, updated_at)
                VALUES (?, ?, ?, 0, ?, ?, CURRENT_TIMESTAMP)
                """,
                (session_id, goal, plan_json, phase, status)
            )

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def update_phase_and_step(self, session_id: str, phase: str, step: int, status: str, plan_json: Optional[str] = None) -> None:
        with self.db.session() as conn:
            if plan_json:
                conn.execute(
                    """
                    UPDATE sessions 
                    SET phase = ?, step = ?, status = ?, plan_json = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                    """,
                    (phase, step, status, plan_json, session_id)
                )
            else:
                conn.execute(
                    """
                    UPDATE sessions 
                    SET phase = ?, step = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                    """,
                    (phase, step, status, session_id)
                )


class SourceRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def add(self, source_id: str, session_id: str, url: str, title: str, domain: str,
            source_type: str = "web", canonical_key: Optional[str] = None, authors: List[str] = None,
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

    def find_by_canonical_key(self, session_id: str, canonical_key: str) -> Optional[Dict[str, Any]]:
        with self.db.session() as conn:
            cur = conn.execute("SELECT * FROM sources WHERE session_id = ? AND canonical_key = ?", (session_id, canonical_key))
            row = cur.fetchone()
            return dict(row) if row else None


class RawEvidenceRepository:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def add(self, raw_evidence_id: str, session_id: str, source_id: str, raw_quote: str,
            page: Optional[int] = None, section: Optional[str] = None,
            char_start: Optional[int] = None, char_end: Optional[int] = None) -> None:
        with self.db.session() as conn:
            conn.execute(
                """
                INSERT INTO raw_evidences (raw_evidence_id, session_id, source_id, raw_quote, page, section, char_start, char_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (raw_evidence_id, session_id, source_id, raw_quote, page, section, char_start, char_end)
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
