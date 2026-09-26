from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from backend.core.engine import ResearchEngine
from backend.core.state import ResearchPhase, ResearchStatus
from backend.core.errors import SessionNotFoundError

router = APIRouter(prefix="/api/research", tags=["Research"])


class CreateSessionRequest(BaseModel):
    goal: str = Field(description="The research question to investigate")


class RunResearchRequest(BaseModel):
    goal: Optional[str] = Field(None, description="Research question for a new session")
    session_id: Optional[str] = Field(None, description="Existing session ID to run to completion")


class RetrieveRequest(BaseModel):
    query: Optional[str] = Field(None, description="Custom query to retrieve for; defaults to research goal")
    top_k: Optional[int] = Field(5, description="Number of top chunks to retrieve")


class SessionResponse(BaseModel):
    session_id: str
    goal: str
    phase: str
    status: str
    step: int


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
    try:
        state = engine.load_state(session_id)
    except (KeyError, SessionNotFoundError):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    sources = engine.source_repo.get_by_session(session_id)
    raw_evidences = engine.raw_evidence_repo.get_by_session(session_id)
    evidences = engine.evidence_repo.get_full_evidence_by_session(session_id)
    claims = engine.claim_repo.get_by_session(session_id)
    report = engine.report_repo.get_by_session(session_id)
    budget = engine.get_budget_tracker(session_id)
    chunks_count = engine.chunk_repo.count_by_session(session_id)

    return {
        "session": {
            "session_id": state.session_id,
            "goal": state.goal,
            "phase": state.phase.value,
            "status": state.status.value,
            "step": state.step,
            "plan": state.plan.model_dump() if state.plan else None,
            "visited_queries": state.visited_queries,
            "open_questions": [q.model_dump() for q in state.open_questions],
            "chunks_count": chunks_count
        },
        "sources": sources,
        "raw_evidences": raw_evidences,
        "evidences": evidences,
        "claims": claims,
        "report": report,
        "budget": budget.summary()
    }


@router.get("/session/{session_id}/evidence")
def get_session_evidence(session_id: str, engine: ResearchEngine = Depends(get_engine)):
    try:
        _ = engine.load_state(session_id)
    except (KeyError, SessionNotFoundError):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    evidence = engine.evidence_repo.get_full_evidence_by_session(session_id)
    return {
        "session_id": session_id,
        "evidence_count": len(evidence),
        "evidence": evidence
    }


@router.get("/session/{session_id}/claims")
def get_session_claims(session_id: str, engine: ResearchEngine = Depends(get_engine)):
    try:
        _ = engine.load_state(session_id)
    except (KeyError, SessionNotFoundError):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    claims = engine.claim_repo.get_by_session(session_id)
    return {
        "session_id": session_id,
        "claims_count": len(claims),
        "claims": claims
    }


@router.post("/session/{session_id}/extract")
def run_extract(session_id: str, engine: ResearchEngine = Depends(get_engine)):
    try:
        state = engine.load_state(session_id)
    except (KeyError, SessionNotFoundError):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    evidence_items = engine.run_extract_phase(state)
    return {
        "session_id": session_id,
        "phase": state.phase.value,
        "evidence_count": len(evidence_items),
        "evidence": evidence_items
    }


@router.post("/session/{session_id}/plan")
def run_plan(session_id: str, engine: ResearchEngine = Depends(get_engine)):
    try:
        state = engine.load_state(session_id)
    except (KeyError, SessionNotFoundError):
        raise HTTPException(status_code=404, detail="Session not found")

    engine.run_plan_phase(state)
    return {
        "session_id": session_id,
        "phase": state.phase.value,
        "plan": state.plan.model_dump() if state.plan else None
    }


@router.post("/session/{session_id}/search")
def run_search(session_id: str, engine: ResearchEngine = Depends(get_engine)):
    try:
        state = engine.load_state(session_id)
    except (KeyError, SessionNotFoundError):
        raise HTTPException(status_code=404, detail="Session not found")

    urls = engine.run_search_phase(state)
    return {
        "session_id": session_id,
        "phase": state.phase.value,
        "visited_queries": state.visited_queries,
        "found_urls": urls
    }


@router.post("/session/{session_id}/fetch")
def run_fetch(session_id: str, engine: ResearchEngine = Depends(get_engine)):
    try:
        state = engine.load_state(session_id)
    except (KeyError, SessionNotFoundError):
        raise HTTPException(status_code=404, detail="Session not found")

    docs = engine.run_fetch_phase(state)
    return {
        "session_id": session_id,
        "phase": state.phase.value,
        "documents_count": len(docs),
        "documents": [{"url": d.url, "title": d.title, "length": len(d.text)} for d in docs]
    }


@router.post("/session/{session_id}/clean")
def run_clean(session_id: str, engine: ResearchEngine = Depends(get_engine)):
    try:
        state = engine.load_state(session_id)
    except (KeyError, SessionNotFoundError):
        raise HTTPException(status_code=404, detail="Session not found")

    total_chunks = engine.run_clean_phase(state)
    return {
        "session_id": session_id,
        "phase": state.phase.value,
        "total_chunks": total_chunks
    }


@router.post("/session/{session_id}/retrieve")
def run_retrieve(session_id: str, req: Optional[RetrieveRequest] = None, engine: ResearchEngine = Depends(get_engine)):
    try:
        state = engine.load_state(session_id)
    except (KeyError, SessionNotFoundError):
        raise HTTPException(status_code=404, detail="Session not found")

    query = req.query if req and req.query else state.goal
    top_k = req.top_k if req and req.top_k else 5
    # If session is in CLEAN, advance state machine to RETRIEVE. Otherwise perform read-only retrieval.
    if state.phase == ResearchPhase.CLEAN:
        retrieved = engine.run_retrieve_phase(state, query=query, top_k=top_k)
    else:
        retrieved = engine.retrieve_chunks(session_id=session_id, query=query, top_k=top_k)

    return {
        "session_id": session_id,
        "phase": state.phase.value,
        "query": query,
        "results_count": len(retrieved),
        "results": [r.model_dump() for r in retrieved]
    }



@router.post("/session/{session_id}/answer")
def run_answer(session_id: str, engine: ResearchEngine = Depends(get_engine)):
    try:
        state = engine.load_state(session_id)
    except (KeyError, SessionNotFoundError):
        raise HTTPException(status_code=404, detail="Session not found")

    # Get fetched sources
    sources = engine.source_repo.get_by_session(session_id)
    docs = []
    for s in sources:
        doc = engine.doc_repo.get_by_source(s["source_id"])
        if doc:
            from backend.tools.web_fetch import FetchedWebContent
            docs.append(FetchedWebContent(
                url=s["url"],
                title=s.get("title"),
                text=doc["raw_text"],
                content_hash=doc["content_hash"]
            ))

    answer = engine.run_basic_answer(state, fetched_docs=docs)
    return {
        "session_id": session_id,
        "phase": state.phase.value,
        "status": state.status.value,
        "answer": answer
    }


@router.post("/run")
def run_end_to_end(req: RunResearchRequest, engine: ResearchEngine = Depends(get_engine)):
    """P0-3 & P0-4: End-to-End One-Click Execution (Question -> Plan -> Search -> Fetch -> Answer)"""
    if req.session_id:
        try:
            state = engine.load_state(req.session_id)
        except (KeyError, SessionNotFoundError):
            raise HTTPException(status_code=404, detail=f"Session {req.session_id} not found")
        result = engine.run_week1(state)
        return result
    elif req.goal:
        result = engine.run_week1(req.goal)
        return result
    else:
        raise HTTPException(status_code=400, detail="Either 'goal' or 'session_id' must be provided.")



@router.get("/trajectories/{partition}")
def list_trajectories(partition: str, engine: ResearchEngine = Depends(get_engine)):
    if partition not in {"raw", "candidate", "gold", "eval"}:
        raise HTTPException(status_code=400, detail="Invalid partition. Must be raw, candidate, gold, or eval.")
    items = engine.trajectory_repo.list_by_partition(partition)
    return {"partition": partition, "count": len(items), "samples": items}
