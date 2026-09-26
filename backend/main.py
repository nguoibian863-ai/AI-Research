import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
import yaml
from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
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
                model=ollama_cfg.get("model", "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M"),
                timeout=ollama_cfg.get("timeout_seconds", 120.0),
                temperature=ollama_cfg.get("temperature", 0.1),
                context_window=ollama_cfg.get("context_window", 4096),
                max_output_tokens=ollama_cfg.get("max_output_tokens", 1024),
                system_prefix=ollama_cfg.get("system_prefix")
            )
        else:
            llm_backend = MockLLMBackend()

        # Configure Embedding Backend
        retrieval_cfg = settings.get("retrieval", {})
        emb_cfg = retrieval_cfg.get("embedding", {})
        emb_type = emb_cfg.get("backend", "fastembed")
        emb_model = emb_cfg.get("model", "BAAI/bge-small-en-v1.5")

        embedding_backend = None
        if emb_type == "fastembed":
            try:
                from backend.retrieval.embeddings import FastEmbedEmbeddingBackend
                embedding_backend = FastEmbedEmbeddingBackend(model_name=emb_model)
                logger.info(f"Loaded FastEmbedEmbeddingBackend with model: {emb_model}")
            except Exception as e:
                logger.error(f"Failed to load configured FastEmbed model '{emb_model}': {e}")
                raise RuntimeError(
                    f"Configured embedding backend 'fastembed' failed to load model '{emb_model}' ({e}). "
                    f"Check internet connection or pre-cached models in ~/.cache/fastembed."
                ) from e
        elif emb_type == "ollama":
            from backend.retrieval.embeddings import OllamaEmbeddingBackend
            embedding_backend = OllamaEmbeddingBackend(model=emb_model)
        elif emb_type in {"hash", "local_hash"}:
            from backend.retrieval.embeddings import LocalHashEmbeddingBackend
            embedding_backend = LocalHashEmbeddingBackend()
        else:
            raise ValueError(f"Unknown embedding backend type: '{emb_type}'")

        _engine_instance = ResearchEngine(
            llm=llm_backend,
            db=db_manager,
            limits=ResearchLimits(**settings.get("limits", {})),
            embedding_backend=embedding_backend
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
    # Eagerly initialize engine to validate embedding backend at startup
    from backend.api.research import get_engine
    if get_engine in app.dependency_overrides:
        _ = app.dependency_overrides[get_engine]()
    else:
        _ = get_research_engine()
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


UI_INDEX = Path(__file__).resolve().parent / "ui" / "index.html"


@app.get("/ui", include_in_schema=False)
def session_viewer():
    """Read-only Session Viewer (plan 29.0): static page that calls the /api/research read endpoints."""
    return FileResponse(UI_INDEX, media_type="text/html")


@app.get("/health")
def health_check(engine: ResearchEngine = Depends(get_engine)):
    provider = "ollama" if isinstance(engine.llm, OllamaBackend) else "mock"
    emb_name = engine.embedding_backend.__class__.__name__ if engine.embedding_backend else "None"
    return {
        "status": "healthy",
        "llm_provider": provider,
        "model": getattr(engine.llm, "model", "mock"),
        "embedding_backend": emb_name,
        "database": str(engine.db.db_path)
    }

