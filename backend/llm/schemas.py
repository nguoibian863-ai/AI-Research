from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PlanTaskSchema(BaseModel):
    task_id: str = Field(description="Unique task id like task_1, task_2")
    description: str = Field(description="Actionable goal for this sub-task")
    expected_evidence: str = Field(description="Type of evidence or metric sought")


class ResearchPlanSchema(BaseModel):
    goal: str = Field(description="Original research question")
    tasks: List[PlanTaskSchema] = Field(description="List of structured research subtasks")
    key_hypotheses: List[str] = Field(default_factory=list, description="Core hypotheses to test")


class SearchQueryItemSchema(BaseModel):
    query: str = Field(description="Precise search query string")
    query_type: str = Field(
        default="discovery",
        description="Query category: discovery | evidence | verification | contradiction"
    )
    rationale: str = Field(description="Why this specific query is needed")


class GeneratedQueriesSchema(BaseModel):
    queries: List[SearchQueryItemSchema] = Field(description="List of targeted search queries")


class AtomicFactItemSchema(BaseModel):
    subject: str = Field(description="Entity or model being discussed, e.g. 'Model X'")
    predicate: str = Field(description="Action or relationship, e.g. 'achieved', 'uses', 'outperformed'")
    metric: Optional[str] = Field(None, description="Metric name if numeric, e.g. 'NDS', 'mAP', 'Accuracy'")
    value: Optional[str] = Field(None, description="Metric value, e.g. '71.2', '84.5%'")
    raw_quote: str = Field(description="Exact verbatim quote from the text chunk without alteration")
    confidence: float = Field(default=0.95, description="Confidence in this atomic extraction (0.0 to 1.0)")


class ExtractedEvidencesSchema(BaseModel):
    facts: List[AtomicFactItemSchema] = Field(description="List of atomic facts extracted from context")


class GapEvaluationSchema(BaseModel):
    is_sufficient: bool = Field(description="True if enough grounded evidence exists to write final report")
    open_questions: List[str] = Field(default_factory=list, description="List of specific missing details")
    next_search_queries: List[str] = Field(default_factory=list, description="Queries to resolve missing facts")
    coverage_score: float = Field(default=0.0, description="Estimated evidence coverage (0.0 to 1.0)")


class SemanticVerificationSchema(BaseModel):
    claim_text: str = Field(description="Claim being verified")
    status: str = Field(
        description="Relationship to evidence: SUPPORTED | PARTIALLY_SUPPORTED | NOT_SUPPORTED | CONTRADICTED"
    )
    explanation: str = Field(description="Brief justification of the logical entailment")
    has_numeric_mismatch: bool = Field(default=False, description="Flagged true if numbers contradict evidence")
