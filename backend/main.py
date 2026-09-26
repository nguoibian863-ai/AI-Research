import logging
from pathlib import Path
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.engine import ResearchEngine
from backend.core.limits import ResearchLimits
from backend.db.database import DatabaseManager
from backend.llm.ollama import OllamaBackend
from backend.llm.mock import MockLLMBackend
from backend.api.research import router as research_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
