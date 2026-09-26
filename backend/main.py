import logging
from pathlib import Path
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.core.engine import ResearchEngine
from backend.core.limits import ResearchLimits
from backend.core.errors import StateTransitionError, BudgetExceededError, ModelInferenceError, ResearchException
from backend.db.database import DatabaseManager
from backend.llm.ollama import OllamaBackend
from backend.llm.mock import MockLLMBackend
from backend.api.research import router as research_router

# Setup local logging (both stdout and file)
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / "research.log"

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
file_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()
root_logger.addHandler(stream_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger("ResearchAI")

# Load settings
CONFIG_PATH = Path("config/settings.yaml")
settings = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

# Global instances
db_manager = DatabaseManager(db_path=settings.get("paths", {}).get("db_path", "data/research.db"))

# Select LLM backend
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

research_engine = ResearchEngine(
    llm=llm_backend,
    db=db_manager,
    limits=ResearchLimits(**settings.get("limits", {}))
)


def get_research_engine() -> ResearchEngine:
    return research_engine


app = FastAPI(
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

@app.exception_handler(KeyError)
async def not_found_error_handler(request: Request, exc: KeyError):
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
def health_check():
    return {
        "status": "healthy",
        "llm_provider": llm_provider,
        "model": getattr(llm_backend, "model", "mock"),
        "database": str(db_manager.db_path)
    }
