import logging
from typing import Set, Dict
from backend.core.state import ResearchPhase, ResearchState, ResearchStatus
from backend.core.limits import ResearchLimits
from backend.core.errors import StateTransitionError, BudgetExceededError

logger = logging.getLogger(__name__)

# Valid transitions map
VALID_TRANSITIONS: Dict[ResearchPhase, Set[ResearchPhase]] = {
    ResearchPhase.INIT: {ResearchPhase.PLAN, ResearchPhase.FAILED, ResearchPhase.CANCELLED},
    ResearchPhase.PLAN: {ResearchPhase.SEARCH, ResearchPhase.FAILED, ResearchPhase.CANCELLED},
    ResearchPhase.SEARCH: {ResearchPhase.FETCH, ResearchPhase.EVALUATE, ResearchPhase.PARTIAL, ResearchPhase.FAILED, ResearchPhase.CANCELLED},
    ResearchPhase.FETCH: {ResearchPhase.CLEAN, ResearchPhase.EVALUATE, ResearchPhase.WRITE, ResearchPhase.PARTIAL, ResearchPhase.FAILED, ResearchPhase.CANCELLED},
    ResearchPhase.CLEAN: {ResearchPhase.RETRIEVE, ResearchPhase.EXTRACT, ResearchPhase.EVALUATE, ResearchPhase.WRITE, ResearchPhase.PARTIAL, ResearchPhase.FAILED, ResearchPhase.CANCELLED},
    ResearchPhase.RETRIEVE: {ResearchPhase.EXTRACT, ResearchPhase.EVALUATE, ResearchPhase.WRITE, ResearchPhase.PARTIAL, ResearchPhase.FAILED, ResearchPhase.CANCELLED},
    ResearchPhase.EXTRACT: {ResearchPhase.EVALUATE, ResearchPhase.PARTIAL, ResearchPhase.FAILED, ResearchPhase.CANCELLED},
    ResearchPhase.EVALUATE: {ResearchPhase.SEARCH, ResearchPhase.VERIFY, ResearchPhase.WRITE, ResearchPhase.PARTIAL, ResearchPhase.FAILED, ResearchPhase.CANCELLED},
    ResearchPhase.VERIFY: {ResearchPhase.WRITE, ResearchPhase.PARTIAL, ResearchPhase.FAILED, ResearchPhase.CANCELLED},
    ResearchPhase.WRITE: {ResearchPhase.DONE, ResearchPhase.PARTIAL, ResearchPhase.FAILED, ResearchPhase.CANCELLED},
    ResearchPhase.DONE: set(),
    ResearchPhase.PARTIAL: set(),
    ResearchPhase.FAILED: set(),
    ResearchPhase.CANCELLED: set(),
}


class StateMachine:
    def __init__(self, limits: ResearchLimits | None = None):
        self.limits = limits or ResearchLimits()

    def transition(self, state: ResearchState, target_phase: ResearchPhase, reason: str = "") -> None:
        """Executes a validated transition on the ResearchState."""
        current_phase = state.phase

        # Terminal state check
        if current_phase in {ResearchPhase.DONE, ResearchPhase.PARTIAL, ResearchPhase.FAILED, ResearchPhase.CANCELLED}:
            raise StateTransitionError(
                f"Cannot transition from terminal phase {current_phase} to {target_phase}."
            )

        allowed = VALID_TRANSITIONS.get(current_phase, set())
        if target_phase not in allowed:
            raise StateTransitionError(
                f"Invalid state transition: {current_phase} -> {target_phase}. Allowed: {[p.value for p in allowed]}."
            )

        logger.info(f"[StateMachine] Session {state.session_id}: {current_phase.value} -> {target_phase.value} ({reason})")
        state.phase = target_phase
        state.touch()

        # Update research status corresponding to terminal phases
        if target_phase == ResearchPhase.DONE:
            state.status = ResearchStatus.COMPLETED
        elif target_phase == ResearchPhase.PARTIAL:
            state.status = ResearchStatus.PARTIAL
        elif target_phase == ResearchPhase.FAILED:
            state.status = ResearchStatus.FAILED
        elif target_phase == ResearchPhase.CANCELLED:
            state.status = ResearchStatus.CANCELLED
        elif state.status == ResearchStatus.PENDING:
            state.status = ResearchStatus.RUNNING

    def evaluate_next_step(self, state: ResearchState) -> ResearchPhase:
        """
        Determines next state after EVALUATE phase strictly via Python logic.
        No LLM determines loop termination.
        """
        if state.phase != ResearchPhase.EVALUATE:
            raise StateTransitionError(f"evaluate_next_step called in invalid phase: {state.phase}")

        state.step += 1

        # Check hard step limit
        if state.step >= self.limits.max_research_steps:
            logger.info(f"Max research steps reached ({state.step}/{self.limits.max_research_steps}). Proceeding to VERIFY.")
            return ResearchPhase.VERIFY

        # Check if there are unresolved open questions
        if state.open_questions:
            logger.info(f"Found {len(state.open_questions)} open questions. Looping back to SEARCH (Step {state.step}).")
            return ResearchPhase.SEARCH

        # All questions satisfied or no open questions left
        logger.info("No open questions remaining. Proceeding to VERIFY.")
        return ResearchPhase.VERIFY
