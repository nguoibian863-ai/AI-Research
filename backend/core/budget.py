import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from backend.core.errors import BudgetExceededError


class TokenBudget(BaseModel):
    planner: int = Field(default=1800, description="Token budget for Research Plan creation")
    query_generator: int = Field(default=1000, description="Token budget for Search Query formulation")
    extractor: int = Field(default=3000, description="Token budget for Atomic Evidence Extraction")
    gap_evaluator: int = Field(default=2000, description="Token budget for Gap Evaluation")
    verifier: int = Field(default=2000, description="Token budget for Claim Entailment Verification")
    writer: int = Field(default=6000, description="Token budget for Grounded Report Writer")


class ExecutionBudgetTracker:
    def __init__(
        self,
        limits=None,
        search_calls: int = 0,
        fetch_calls: int = 0,
        llm_calls: int = 0,
        tokens_consumed: int = 0,
        step_count: int = 0
    ):
        from backend.core.limits import ResearchLimits
        self.limits = limits or ResearchLimits()
        self.start_time: float = time.time()
        self.step_count: int = step_count
        self.search_calls: int = search_calls
        self.fetch_calls: int = fetch_calls
        self.llm_calls: int = llm_calls
        self.total_tokens_consumed: int = tokens_consumed

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def check_runtime(self) -> None:
        if self.elapsed_seconds > self.limits.max_runtime_seconds:
            raise BudgetExceededError(
                f"Maximum research runtime of {self.limits.max_runtime_seconds}s exceeded ({self.elapsed_seconds:.1f}s)."
            )

    def assert_can_search(self) -> None:
        self.check_runtime()
        if self.search_calls >= self.limits.max_search_calls:
            raise BudgetExceededError(
                f"Maximum search calls limit reached ({self.search_calls}/{self.limits.max_search_calls})."
            )

    def assert_can_fetch(self) -> None:
        self.check_runtime()
        if self.fetch_calls >= self.limits.max_fetch_calls:
            raise BudgetExceededError(
                f"Maximum fetch calls limit reached ({self.fetch_calls}/{self.limits.max_fetch_calls})."
            )

    def assert_can_call_llm(self) -> None:
        self.check_runtime()
        if self.llm_calls >= self.limits.max_llm_calls:
            raise BudgetExceededError(
                f"Maximum LLM calls limit reached ({self.llm_calls}/{self.limits.max_llm_calls})."
            )

    def record_step(self) -> None:
        self.check_runtime()
        self.step_count += 1

    def record_search(self) -> None:
        self.assert_can_search()
        self.search_calls += 1

    def record_fetch(self) -> None:
        self.assert_can_fetch()
        self.fetch_calls += 1

    def record_llm_call(self, tokens: int = 0, count: int = 1) -> None:
        self.assert_can_call_llm()
        if isinstance(count, int):
            self.llm_calls += count
        if isinstance(tokens, int):
            self.total_tokens_consumed += tokens

    def can_search(self) -> bool:
        return self.search_calls < self.limits.max_search_calls and self.elapsed_seconds <= self.limits.max_runtime_seconds

    def can_fetch(self) -> bool:
        return self.fetch_calls < self.limits.max_fetch_calls and self.elapsed_seconds <= self.limits.max_runtime_seconds

    def can_call_llm(self) -> bool:
        return self.llm_calls < self.limits.max_llm_calls and self.elapsed_seconds <= self.limits.max_runtime_seconds

    def has_exceeded_steps(self) -> bool:
        return self.step_count >= self.limits.max_research_steps

    def summary(self) -> Dict[str, Any]:
        return {
            "steps": self.step_count,
            "search_calls": f"{self.search_calls}/{self.limits.max_search_calls}",
            "fetch_calls": f"{self.fetch_calls}/{self.limits.max_fetch_calls}",
            "llm_calls": f"{self.llm_calls}/{self.limits.max_llm_calls}",
            "tokens_consumed": self.total_tokens_consumed,
            "elapsed_seconds": round(self.elapsed_seconds, 2)
        }
