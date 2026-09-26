from pydantic import BaseModel, Field


class ResearchLimits(BaseModel):
    max_research_steps: int = Field(default=5, description="Max iterative research loops")
    max_search_calls: int = Field(default=12, description="Max external search queries")
    max_fetch_calls: int = Field(default=20, description="Max web/PDF fetch requests")
    max_search_results: int = Field(default=8, description="Max search results per query")
    max_sources_per_run: int = Field(default=15, description="Max unique sources retained per session")
    max_total_chunks: int = Field(default=300, description="Max chunks processed per session")
    max_chunk_tokens: int = Field(default=600, description="Chunk length limit in tokens")
    max_evidence_items: int = Field(default=30, description="Max atomic evidence items stored")
    retrieval_top_k: int = Field(default=5, description="Number of chunks retrieved per query")
    max_llm_calls: int = Field(default=30, description="Max LLM inferences allowed per run")
    max_runtime_seconds: int = Field(default=600, description="Hard timeout for total research run in seconds")
    llm_context: int = Field(default=4096, description="Target LLM context window")
    max_output_tokens: int = Field(default=1024, description="Max token generation limit")
