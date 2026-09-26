from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from backend.core.engine import ResearchEngine
from backend.core.state import ResearchPhase, ResearchStatus

router = APIRouter(prefix="/api/research", tags=["Research"])


class CreateSessionRequest(BaseModel):
    goal: str = Field(description="The research question to investigate")


class SessionResponse(BaseModel):
    session_id: str
    goal: str
    phase: str
    status: str
    step: int


# Dependency placeholder - in main.py we will inject the singleton ResearchEngine
def get_engine() -> ResearchEngine:
    from backend.main import get_research_engine
    return get_research_engine()


@router.post("/session", response_model=SessionResponse)
def create_session(req: CreateSessionRequest, engine: ResearchEngine = Depends(get_engine)):
    state = engine.create_session(goal=req.goal)
    return SessionResponse(
        session_id=state.session_id,
        goal=state.goal,
        phase=state.phase.value,
        status=state.status.value,
        step=state.step
    )


@router.get("/session/{session_id}")
def get_session(session_id: str, engine: ResearchEngine = Depends(get_engine)):
    session_data = engine.session_repo.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    sources = engine.source_repo.get_by_session(session_id)
    raw_evidences = engine.raw_evidence_repo.get_by_session(session_id)

    return {
        "session": session_data,
        "sources": sources,
        "raw_evidences": raw_evidences
    }


@router.post("/session/{session_id}/plan")
def run_plan(session_id: str, engine: ResearchEngine = Depends(get_engine)):
    session_data = engine.session_repo.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    from backend.core.state import ResearchState
    state = ResearchState(
        session_id=session_id,
        goal=session_data["goal"],
        step=session_data["step"],
        phase=ResearchPhase(session_data["phase"]),
        status=ResearchStatus(session_data["status"])
    )
    engine.run_plan_phase(state)
    return {"session_id": session_id, "plan": state.plan.model_dump() if state.plan else None}


@router.post("/session/{session_id}/search")
def run_search(session_id: str, engine: ResearchEngine = Depends(get_engine)):
    session_data = engine.session_repo.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    from backend.core.state import ResearchState
    state = ResearchState(
        session_id=session_id,
        goal=session_data["goal"],
        step=session_data["step"],
        phase=ResearchPhase(session_data["phase"]),
        status=ResearchStatus(session_data["status"])
    )
    urls = engine.run_search_phase(state)
    return {"session_id": session_id, "found_urls": urls}


@router.get("/trajectories/{partition}")
def list_trajectories(partition: str, engine: ResearchEngine = Depends(get_engine)):
    if partition not in {"raw", "candidate", "gold", "eval"}:
        raise HTTPException(status_code=400, detail="Invalid partition. Must be raw, candidate, gold, or eval.")
    items = engine.trajectory_repo.list_by_partition(partition)
    return {"partition": partition, "count": len(items), "samples": items}
