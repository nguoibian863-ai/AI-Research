from pydantic import BaseModel, Field


class TokenBudget(BaseModel):
    planner: int = Field(default=1800, description="Token budget for Research Plan creation")
    query_generator: int = Field(default=1000, description="Token budget for Search Query formulation")
    extractor: int = Field(default=3000, description="Token budget for Atomic Evidence Extraction")
    gap_evaluator: int = Field(default=2000, description="Token budget for Gap Evaluation")
    verifier: int = Field(default=2000, description="Token budget for Claim Entailment Verification")
    writer: int = Field(default=6000, description="Token budget for Grounded Report Writer")


class ExecutionBudgetTracker:
    def __init__(self, limits=None):
        from backend.core.limits import ResearchLimits
        self.limits = limits or ResearchLimits()
        self.step_count: int = 0
        self.search_calls: int = 0
        self.fetch_calls: int = 0
        self.llm_calls: int = 0
        self.total_tokens_consumed: int = 0

    def record_step(self) -> None:
        self.step_count += 1

    def record_search(self) -> None:
        self.search_calls += 1

    def record_fetch(self) -> None:
        self.fetch_calls += 1

    def record_llm_call(self, tokens: int = 0) -> None:
        self.llm_calls += 1
        self.total_tokens_consumed += tokens

    def can_search(self) -> bool:
        return self.search_calls < self.limits.max_search_calls

    def can_fetch(self) -> bool:
        return self.fetch_calls < self.limits.max_fetch_calls

    def can_call_llm(self) -> bool:
        return self.llm_calls < self.limits.max_llm_calls

    def has_exceeded_steps(self) -> bool:
        return self.step_count >= self.limits.max_research_steps
