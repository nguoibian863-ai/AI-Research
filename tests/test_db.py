import tempfile
from pathlib import Path
from backend.db.database import DatabaseManager
from backend.db.repositories import (
    SessionRepository,
    SourceRepository,
    RawEvidenceRepository,
    EvidenceRepository,
    TrajectoryRepository
)


def test_sqlite_schema_and_repositories():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_research.db"
        db = DatabaseManager(db_path=db_path)

        # 1. Session Repo
        session_repo = SessionRepository(db)
        session_repo.create(session_id="s123", goal="Evaluate 3D Detectors", phase="INIT", status="RUNNING")
        s = session_repo.get("s123")
        assert s is not None
        assert s["goal"] == "Evaluate 3D Detectors"
        assert s["phase"] == "INIT"

        session_repo.update_phase_and_step("s123", phase="SEARCH", step=1, status="RUNNING")
        s_updated = session_repo.get("s123")
        assert s_updated["phase"] == "SEARCH"
        assert s_updated["step"] == 1

        # 2. Source Repo
        source_repo = SourceRepository(db)
        source_repo.add(
            source_id="src_001",
            session_id="s123",
            url="https://arxiv.org/abs/2401.0001",
            title="Model X Paper",
            domain="arxiv.org",
            source_type="paper",
            canonical_key="arxiv:2401.0001"
        )
        sources = source_repo.get_by_session("s123")
        assert len(sources) == 1
        assert sources[0]["source_id"] == "src_001"

        # 3. Raw Evidence Repo (P0 - Immutable)
        raw_repo = RawEvidenceRepository(db)
        raw_repo.add(
            raw_evidence_id="raw_001",
            session_id="s123",
            source_id="src_001",
            raw_quote="Model X achieves 71.2 NDS on nuScenes val.",
            page=7,
            section="Experiments",
            char_start=1024,
            char_end=1068
        )
        raw_ev = raw_repo.get("raw_001")
        assert raw_ev is not None
        assert raw_ev["raw_quote"] == "Model X achieves 71.2 NDS on nuScenes val."
        assert raw_ev["page"] == 7

        # 4. Trajectory Partitioning
        traj_repo = TrajectoryRepository(db)
        traj_repo.add(
            trajectory_id="t_001",
            session_id="s123",
            task_type="evidence_extraction",
            model_source="qwen3:8b",
            payload={"quote": "71.2 NDS", "extracted": {"NDS": 71.2}},
            verified=True,
            quality_score=0.98,
            partition="gold"
        )
        traj_repo.add(
            trajectory_id="t_002",
            session_id="s123",
            task_type="evidence_extraction",
            model_source="qwen3:8b",
            payload={"quote": "noisy text"},
            verified=False,
            quality_score=0.50,
            partition="raw"
        )

        gold_samples = traj_repo.list_by_partition("gold")
        assert len(gold_samples) == 1
        assert gold_samples[0]["trajectory_id"] == "t_001"
        assert gold_samples[0]["quality_score"] == 0.98

        raw_samples = traj_repo.list_by_partition("raw")
        assert len(raw_samples) == 1
        assert raw_samples[0]["trajectory_id"] == "t_002"
