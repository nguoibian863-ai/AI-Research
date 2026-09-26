from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ResearchPhase(str, Enum):
    INIT = "INIT"
    PLAN = "PLAN"
    SEARCH = "SEARCH"
    FETCH = "FETCH"
    CLEAN = "CLEAN"
    RETRIEVE = "RETRIEVE"
    EXTRACT = "EXTRACT"
    EVALUATE = "EVALUATE"
    VERIFY = "VERIFY"
    WRITE = "WRITE"
    DONE = "DONE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ResearchStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class PlanTask(BaseModel):
    task_id: str
    description: str
    expected_evidence: str
    status: str = "PENDING"  # PENDING | IN_PROGRESS | COMPLETED | SKIPPED


class ResearchPlan(BaseModel):
    goal: str
    tasks: List[PlanTask] = Field(default_factory=list)
    key_hypotheses: List[str] = Field(default_factory=list)


class OpenQuestion(BaseModel):
    question_id: str
    text: str
    priority: int = 1
    derived_from_gap: Optional[str] = None


class ResearchState(BaseModel):
    session_id: str
    goal: str
    plan: Optional[ResearchPlan] = None

    visited_queries: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    claim_ids: List[str] = Field(default_factory=list)

    open_questions: List[OpenQuestion] = Field(default_factory=list)

    step: int = 0
    phase: ResearchPhase = ResearchPhase.INIT
    status: ResearchStatus = ResearchStatus.PENDING

    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
