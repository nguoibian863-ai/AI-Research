import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager

DEFAULT_DB_PATH = Path("data/research.db")

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    plan_json TEXT,
    open_questions_json TEXT,
    error_message TEXT,
    step INTEGER DEFAULT 0,
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    search_calls INTEGER DEFAULT 0,
    fetch_calls INTEGER DEFAULT 0,
    llm_calls INTEGER DEFAULT 0,
    tokens_consumed INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    query_type TEXT DEFAULT 'discovery',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    url TEXT,
    title TEXT,
    authors_json TEXT,
    published_at TEXT,
    retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_type TEXT DEFAULT 'web',
    domain TEXT,
    canonical_key TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    file_path TEXT,
    content_hash TEXT,
    raw_text TEXT,
    parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    text TEXT NOT NULL,
    page INTEGER,
    section TEXT,
    char_start INTEGER,
    char_end INTEGER,
    token_count INTEGER,
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

-- Raw Evidence Layer (P0 - Immutable)
CREATE TABLE IF NOT EXISTS raw_evidences (
    raw_evidence_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    raw_quote TEXT NOT NULL,
    page INTEGER,
    section TEXT,
    char_start INTEGER,
    char_end INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);

-- Atomic Evidence (1 Fact = 1 Evidence)
CREATE TABLE IF NOT EXISTS evidences (
    evidence_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    raw_evidence_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_json TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY(raw_evidence_id) REFERENCES raw_evidences(raw_evidence_id) ON DELETE CASCADE
);

-- Verified Claims
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    text TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL,
    verification_json TEXT NOT NULL,
    status TEXT DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- Research Reports
CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content_markdown TEXT NOT NULL,
    claims_json TEXT,
    status TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- Trajectories for Phase 2 fine-tuning & evaluation (Partitioned: raw, candidate, gold, eval)
CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    model_source TEXT NOT NULL,
    verified BOOLEAN DEFAULT 0,
    quality_score REAL DEFAULT 0.0,
    payload_json TEXT NOT NULL,
    partition TEXT DEFAULT 'raw',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- Indexes for high-speed queries
CREATE INDEX IF NOT EXISTS idx_queries_session ON queries(session_id);
CREATE INDEX IF NOT EXISTS idx_sources_session ON sources(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_raw_ev_session ON raw_evidences(session_id);
CREATE INDEX IF NOT EXISTS idx_ev_session ON evidences(session_id);
CREATE INDEX IF NOT EXISTS idx_claims_session ON claims(session_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_partition ON trajectories(partition);
"""


class DatabaseManager:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @contextmanager
    def session(self):
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.session() as conn:
            conn.executescript(SCHEMA_SQL)
            # Safe migration for existing databases
            cursor = conn.execute("PRAGMA table_info(sessions)")
            existing_columns = {row["name"] for row in cursor.fetchall()}
            needed_columns = {
                "open_questions_json": "TEXT",
                "error_message": "TEXT",
                "search_calls": "INTEGER DEFAULT 0",
                "fetch_calls": "INTEGER DEFAULT 0",
                "llm_calls": "INTEGER DEFAULT 0",
                "tokens_consumed": "INTEGER DEFAULT 0",
            }
            for col_name, col_type in needed_columns.items():
                if col_name not in existing_columns:
                    conn.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_type}")
