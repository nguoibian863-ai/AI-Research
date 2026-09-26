import pytest
from backend.core.state import ResearchState, ResearchPhase, ResearchStatus, OpenQuestion
from backend.core.limits import ResearchLimits
from backend.core.transitions import StateMachine
from backend.core.errors import StateTransitionError


def test_valid_transitions_lifecycle():
    limits = ResearchLimits(max_research_steps=3)
    sm = StateMachine(limits=limits)
    state = ResearchState(session_id="test_sess_1", goal="Test Research Goal")

    assert state.phase == ResearchPhase.INIT
    assert state.status == ResearchStatus.PENDING

    # INIT -> PLAN
    sm.transition(state, ResearchPhase.PLAN)
    assert state.phase == ResearchPhase.PLAN
    assert state.status == ResearchStatus.RUNNING

    # PLAN -> SEARCH
    sm.transition(state, ResearchPhase.SEARCH)
    assert state.phase == ResearchPhase.SEARCH

    # SEARCH -> FETCH
    sm.transition(state, ResearchPhase.FETCH)
    assert state.phase == ResearchPhase.FETCH

    # FETCH -> CLEAN
    sm.transition(state, ResearchPhase.CLEAN)
    assert state.phase == ResearchPhase.CLEAN

    # CLEAN -> RETRIEVE
    sm.transition(state, ResearchPhase.RETRIEVE)
    assert state.phase == ResearchPhase.RETRIEVE

    # RETRIEVE -> EXTRACT
    sm.transition(state, ResearchPhase.EXTRACT)
    assert state.phase == ResearchPhase.EXTRACT

    # EXTRACT -> EVALUATE
    sm.transition(state, ResearchPhase.EVALUATE)
    assert state.phase == ResearchPhase.EVALUATE


def test_invalid_transition_rejected():
    sm = StateMachine()
    state = ResearchState(session_id="test_sess_2", goal="Test Invalid Jump")

    # Jumping directly from INIT to WRITE is invalid
    with pytest.raises(StateTransitionError) as exc_info:
        sm.transition(state, ResearchPhase.WRITE)
    assert "Invalid state transition" in str(exc_info.value)


def test_evaluate_step_loops_on_open_questions():
    limits = ResearchLimits(max_research_steps=5)
    sm = StateMachine(limits=limits)
    state = ResearchState(
        session_id="test_sess_3",
        goal="Open questions check",
        phase=ResearchPhase.EVALUATE,
        step=1,
        open_questions=[OpenQuestion(question_id="q1", text="What is Model X mAP?")]
    )

    next_phase = sm.evaluate_next_step(state)
    assert next_phase == ResearchPhase.SEARCH
    assert state.step == 2


def test_evaluate_step_terminates_at_hard_limit():
    limits = ResearchLimits(max_research_steps=3)
    sm = StateMachine(limits=limits)
    state = ResearchState(
        session_id="test_sess_4",
        goal="Hard limit check",
        phase=ResearchPhase.EVALUATE,
        step=2,
        open_questions=[OpenQuestion(question_id="q1", text="Still unanswered question")]
    )

    # Step becomes 3 (>= max_research_steps 3) -> Must proceed to VERIFY, no infinite loops
    next_phase = sm.evaluate_next_step(state)
    assert next_phase == ResearchPhase.VERIFY
    assert state.step == 3
