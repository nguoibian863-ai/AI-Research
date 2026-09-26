import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
import yaml
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.core.engine import ResearchEngine
from backend.core.limits import ResearchLimits
from backend.core.errors import (
    StateTransitionError,
    BudgetExceededError,
    ModelInferenceError,
    SessionNotFoundError,
    ResearchException
)
from backend.db.database import DatabaseManager
from backend.llm.ollama import OllamaBackend
from backend.llm.mock import MockLLMBackend
from backend.api.research import router as research_router, get_engine

LOGS_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

logger = logging.getLogger("ResearchAI")

# Lazy singleton engine instance
_engine_instance: Optional[ResearchEngine] = None


def get_research_engine() -> ResearchEngine:
    global _engine_instance
    if _engine_instance is None:
        CONFIG_PATH = Path("config/settings.yaml")
        settings = {}
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f)

        db_manager = DatabaseManager(db_path=settings.get("paths", {}).get("db_path", "data/research.db"))
        llm_provider = settings.get("llm", {}).get("default_provider", "ollama")
        ollama_cfg = settings.get("llm", {}).get("ollama", {})

        if llm_provider == "ollama":
            llm_backend = OllamaBackend(
                base_url=ollama_cfg.get("base_url", "http://127.0.0.1:11434"),
                model=ollama_cfg.get("model", "smollm3:3b"),
                timeout=ollama_cfg.get("timeout_seconds", 120.0),
                temperature=ollama_cfg.get("temperature", 0.1),
                context_window=ollama_cfg.get("context_window", 4096),
                max_output_tokens=ollama_cfg.get("max_output_tokens", 1024)
            )
        else:
            llm_backend = MockLLMBackend()

        _engine_instance = ResearchEngine(
            llm=llm_backend,
            db=db_manager,
            limits=ResearchLimits(**settings.get("limits", {}))
        )
    return _engine_instance


def setup_logging(logs_dir: Optional[Path] = None) -> List[logging.Handler]:
    """Attaches console + file handlers to the root logger without removing existing handlers."""
    logs_dir = logs_dir or LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(str(logs_dir / "research.log"), encoding="utf-8")
    handlers: List[logging.Handler] = [stream_handler, file_handler]

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    return handlers


def teardown_logging(handlers: List[logging.Handler]) -> None:
    root_logger = logging.getLogger()
    for handler in handlers:
        root_logger.removeHandler(handler)
        handler.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    handlers = setup_logging()
    try:
        yield
    finally:
        teardown_logging(handlers)


app = FastAPI(
    lifespan=lifespan,
    title="Local Deep Research Agent API",
    version="0.1.0",
    description="Local-first, grounded research system with provenance verification."
)

# Exception handlers mapping custom domain errors to appropriate HTTP status codes
@app.exception_handler(StateTransitionError)
async def state_transition_error_handler(request: Request, exc: StateTransitionError):
    logger.warning(f"State transition conflict: {exc}")
    return JSONResponse(
        status_code=409,
        content={"error": "StateTransitionError", "detail": str(exc)}
    )

@app.exception_handler(BudgetExceededError)
async def budget_exceeded_error_handler(request: Request, exc: BudgetExceededError):
    logger.warning(f"Budget exceeded limit: {exc}")
    return JSONResponse(
        status_code=429,
        content={"error": "BudgetExceededError", "detail": str(exc)}
    )

@app.exception_handler(ModelInferenceError)
async def model_inference_error_handler(request: Request, exc: ModelInferenceError):
    logger.error(f"LLM model error: {exc}")
    return JSONResponse(
        status_code=502,
        content={"error": "ModelInferenceError", "detail": str(exc)}
    )

@app.exception_handler(SessionNotFoundError)
async def not_found_error_handler(request: Request, exc: SessionNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": "NotFound", "detail": str(exc)}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(research_router)


@app.get("/health")
def health_check(engine: ResearchEngine = Depends(get_engine)):
    provider = "ollama" if isinstance(engine.llm, OllamaBackend) else "mock"
    return {
        "status": "healthy",
        "llm_provider": provider,
        "model": getattr(engine.llm, "model", "mock"),
        "database": str(engine.db.db_path)
    }

