import pytest
from pathlib import Path
from backend.retrieval.bm25 import BM25Index, tokenize_for_bm25
from backend.retrieval.embeddings import LocalHashEmbeddingBackend, MockEmbeddingBackend, FastEmbedEmbeddingBackend
from backend.retrieval.faiss_index import FaissVectorIndex
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import ScoreReranker
from backend.core.state import ResearchPhase, ResearchStatus


def test_bm25_exact_metric_match():
    """
    Acceptance Criteria 3: exact metric query BM25 tim duoc.
    Verifies BM25 ranks exact metric queries (numbers, model names, benchmarks) #1.
    """
    chunks = [
        {"chunk_id": "c1", "text": "CenterPoint achieves 71.2 NDS on the nuScenes 3D detection benchmark."},
        {"chunk_id": "c2", "text": "PointPillars achieves 59.2 mAP on KITTI dataset at 62 FPS."},
        {"chunk_id": "c3", "text": "FlashAttention-2 provides 2x speedup on NVIDIA A100 GPU."},
        {"chunk_id": "c4", "text": "General overview of autonomous driving perception sensors and cameras."}
    ]
    bm25 = BM25Index(chunks)

    # 1. Query for exact metric: 71.2 NDS
    res1 = bm25.search("71.2 NDS", top_k=2)
    assert len(res1) > 0
    assert res1[0][0]["chunk_id"] == "c1"
    assert res1[0][1] > 0.0

    # 2. Query for exact model and metric: PointPillars 59.2 mAP
    res2 = bm25.search("PointPillars 59.2 mAP", top_k=2)
    assert len(res2) > 0
    assert res2[0][0]["chunk_id"] == "c2"

    # 3. Query for FlashAttention-2 A100
    res3 = bm25.search("FlashAttention-2 A100", top_k=2)
    assert len(res3) > 0
    assert res3[0][0]["chunk_id"] == "c3"


def test_faiss_vector_semantic_search(tmp_path):
    """
    Acceptance Criteria 4: semantic query vector tim duoc.
    Verifies FAISS vector index performs semantic search and handles persistence.
    """
    emb = LocalHashEmbeddingBackend(dimension=128)
    faiss_idx = FaissVectorIndex(embedding_backend=emb, index_dir=tmp_path)

    chunks = [
        {"chunk_id": "c1", "text": "Accuracy comparison and evaluation of 3D object detection LiDAR pipelines."},
        {"chunk_id": "c2", "text": "Culinary recipes for Italian pasta carbonara with eggs and pecorino cheese."},
        {"chunk_id": "c3", "text": "Hardware cooling solutions and liquid cooling loops for desktop computers."}
    ]
    faiss_idx.add_chunks(chunks)

    # Semantic query without exact keywords
    results = faiss_idx.search("evaluation of 3D LiDAR perception accuracy", top_k=2)
    assert len(results) > 0
    assert results[0][0]["chunk_id"] == "c1"
    assert results[0][1] > 0.0

    # Test persistence (save and load)
    faiss_idx.save("test_session_index")
    loaded_idx = FaissVectorIndex(embedding_backend=emb, index_dir=tmp_path)
    assert loaded_idx.load("test_session_index")
    loaded_res = loaded_idx.search("3D perception evaluation", top_k=1)
    assert len(loaded_res) > 0
    assert loaded_res[0][0]["chunk_id"] == "c1"


def test_faiss_rebuild_from_sqlite(isolated_engine, tmp_path):
    """
    P0 Invariant: SQLite is Source of Truth.
    If FAISS vector index is wiped or corrupted, it must be 100% rebuildable from SQLite chunks table.
    """
    state = isolated_engine.create_session("Rebuild test")
    sid = state.session_id

    # 1. Add source, document, and chunks directly to SQLite
    src_id = "src_rebuild_01"
    doc_id = "doc_rebuild_01"
    isolated_engine.source_repo.add(
        source_id=src_id, session_id=sid, url="https://example.com/paper",
        title="Paper Title", domain="example.com"
    )
    isolated_engine.doc_repo.add(
        doc_id=doc_id, source_id=src_id, file_path=None,
        content_hash="hash123", raw_text="Paper content with 71.2 NDS nuScenes."
    )
    isolated_engine.chunk_repo.add(
        chunk_id="chk_rb_1", doc_id=doc_id, text="Model X achieved 71.2 NDS on nuScenes.",
        page=1, section="Results", char_start=0, char_end=38, token_count=10
    )

    # 2. Corrupt/Clear FAISS index
    faiss_idx = FaissVectorIndex(
        embedding_backend=isolated_engine.embedding_backend,
        index_dir=tmp_path
    )
    faiss_idx.clear()
    assert faiss_idx.index.ntotal == 0

    # 3. Rebuild from SQLite Source of Truth
    rebuilt_count = faiss_idx.rebuild_from_db(isolated_engine.db, session_id=sid)
    assert rebuilt_count == 1
    assert faiss_idx.index.ntotal == 1

    # Verify search works immediately after rebuild
    res = faiss_idx.search("nuScenes 71.2 NDS", top_k=1)
    assert len(res) == 1
    assert res[0][0]["chunk_id"] == "chk_rb_1"


def test_hybrid_retrieval_rrf():
    """
    Acceptance Criteria 5: hybrid retrieval chay duoc.
    Verifies HybridRetriever correctly merges BM25 keyword matching and Vector search via RRF.
    """
    chunks = [
        {"chunk_id": "c1", "text": "CenterPoint achieves 71.2 NDS on nuScenes.", "page": 7, "section": "Results"},
        {"chunk_id": "c2", "text": "PointPillars achieves 59.2 mAP at 62 FPS.", "page": 5, "section": "Benchmark"},
        {"chunk_id": "c3", "text": "General autonomous driving overview.", "page": 1, "section": "Intro"}
    ]
    retriever = HybridRetriever(
        bm25_index=BM25Index(chunks),
        vector_index=FaissVectorIndex(embedding_backend=LocalHashEmbeddingBackend(dimension=128))
    )
    retriever.vector_index.add_chunks(chunks)

    # Hybrid query containing exact metric
    results = retriever.retrieve(query="CenterPoint 71.2 NDS nuScenes", top_k=2)

    assert len(results) >= 1
    top_chunk = results[0]
    assert top_chunk.chunk_id == "c1"
    assert top_chunk.page == 7
    assert top_chunk.section == "Results"
    assert top_chunk.score > 0.0
    assert top_chunk.bm25_score is not None
    assert top_chunk.vector_score is not None


def test_api_clean_and_retrieve_endpoints(client, isolated_engine):
    """
    Integration Test for API Endpoints:
    - POST /api/research/session/{session_id}/clean
    - POST /api/research/session/{session_id}/retrieve
    """
    # 1. Create session
    create_res = client.post("/api/research/session", json={"goal": "nuScenes 3D benchmark"})
    sid = create_res.json()["session_id"]
    state = isolated_engine.load_state(sid)

    # 2. Add source and doc
    isolated_engine.source_repo.add(
        source_id="src_api_01", session_id=sid, url="https://example.com/test",
        title="Test Doc", domain="example.com"
    )
    isolated_engine.doc_repo.add(
        doc_id="doc_api_01", source_id="src_api_01", file_path=None,
        content_hash="h1", raw_text="CenterPoint achieved 71.2 NDS on nuScenes benchmark."
    )

    # Move phase to FETCH so transitioning to CLEAN is valid
    state.phase = ResearchPhase.FETCH
    isolated_engine.save_state(state)

    # 3. Call /clean
    clean_res = client.post(f"/api/research/session/{sid}/clean")
    assert clean_res.status_code == 200
    assert clean_res.json()["total_chunks"] >= 1
    assert clean_res.json()["phase"] == "CLEAN"

    # 4. Call /retrieve
    ret_res = client.post(
        f"/api/research/session/{sid}/retrieve",
        json={"query": "71.2 NDS nuScenes", "top_k": 3}
    )
    assert ret_res.status_code == 200
    data = ret_res.json()
    assert data["phase"] == "RETRIEVE"
    assert data["results_count"] >= 1
    assert "71.2 NDS" in data["results"][0]["text"]


@pytest.mark.slow
def test_fastembed_true_semantic_paraphrasing(tmp_path):
    """
    P0 Test: True semantic search with paraphrased query without shared vocabulary.
    Verifies FastEmbedEmbeddingBackend (BAAI/bge-small-en-v1.5) accurately matches
    'how fast does the model run' with 'Inference latency is 20 ms' over irrelevant text.
    Offline-safe: skips gracefully in air-gapped test containers if the model is not cached.
    """
    try:
        emb = FastEmbedEmbeddingBackend()
        _ = emb.embed_text("smoke test")
    except Exception as e:
        pytest.skip(f"FastEmbed model download not available offline: {e}")

    faiss_idx = FaissVectorIndex(embedding_backend=emb, index_dir=tmp_path)

    chunks = [
        {"chunk_id": "speed_chunk", "text": "Inference latency is 20 ms per frame on RTX hardware."},
        {"chunk_id": "recipe_chunk", "text": "Authentic Italian carbonara pasta requires eggs, guanciale, and pecorino."},
        {"chunk_id": "weather_chunk", "text": "Sunny skies and mild temperatures expected across the Mediterranean."}
    ]
    faiss_idx.add_chunks(chunks)

    # Paraphrased query with ZERO shared words with speed_chunk
    results = faiss_idx.search("how fast does the model run", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk["chunk_id"] == "speed_chunk"
    # Ensure significant margin over recipe chunk
    assert score > 0.5


def test_reranker_does_not_overpower_semantic_vector():
    """
    Issue #2 Regression Test:
    Reranker must NOT overpower RRF semantic ranking.
    Candidate A: 'Inference latency is 20 ms' (vec rank 1, RRF score 0.01639)
    Candidate B: 'We run the model on 8 GPUs' (vec rank 2, RRF score 0.01612)
    Query: 'how fast does the model run'
    Candidate A must remain Rank 1; Candidate B must not jump over Candidate A.
    """
    from backend.retrieval.hybrid import RetrievedChunk

    cand_a = RetrievedChunk(
        chunk_id="chunk_latency",
        doc_id="d1",
        text="Inference latency is 20 ms per frame on RTX hardware.",
        section="Performance",
        score=0.01639
    )
    cand_b = RetrievedChunk(
        chunk_id="chunk_gpus",
        doc_id="d1",
        text="We run the model on 8 GPUs for training.",
        section="Training Setup",
        score=0.01612
    )

    reranked = ScoreReranker.rerank(
        query="how fast does the model run",
        chunks=[cand_a, cand_b],
        top_k=2
    )

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "chunk_latency", (
        f"Expected chunk_latency to remain rank 1, but got {reranked[0].chunk_id} "
        f"with score {reranked[0].score} vs {reranked[1].score}"
    )
    assert reranked[0].score > reranked[1].score


def test_bm25_small_corpus_no_drop():
    """
    P1 Test: Small corpus (N <= 2) term matching.
    Ensures that when N=1 or N=2, chunks containing query terms are NOT discarded
    due to negative or zero IDF scores.
    """
    chunks = [
        {"chunk_id": "doc1", "text": "CenterPoint LiDAR point clouds 3D bounding boxes."},
    ]
    bm25 = BM25Index(chunks)

    # Query matching doc1 in a 1-document corpus
    results = bm25.search("CenterPoint", top_k=5)
    assert len(results) == 1
    assert results[0][0]["chunk_id"] == "doc1"
    assert results[0][1] > 0.0


def test_score_reranker_boosts():
    """
    P2 Test: ScoreReranker exact keyword and section boosts.
    Verifies that chunks in 'Results' sections and containing exact query phrases
    receive score boosts and rank at the top.
    """
    from backend.retrieval.hybrid import RetrievedChunk

    c1 = RetrievedChunk(
        chunk_id="c1", doc_id="d1",
        text="Background discussion on autonomous vehicles and cameras.",
        section="Introduction", score=0.015
    )
    c2 = RetrievedChunk(
        chunk_id="c2", doc_id="d1",
        text="CenterPoint achieved 71.2 NDS on nuScenes benchmark.",
        section="Results", score=0.014
    )

    reranked = ScoreReranker.rerank(query="CenterPoint nuScenes benchmark", chunks=[c1, c2], top_k=2)
    assert len(reranked) == 2
    # c2 should receive both exact query match boost and benchmark section boost!
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].score > reranked[1].score


def test_session_retriever_isolation(isolated_engine):
    """
    P1 Test: HybridRetriever is isolated per session.
    Verifies that indexing chunks for session A does not pollute or wipe session B.
    """
    sid_a = "sess_iso_a"
    sid_b = "sess_iso_b"

    retriever_a = isolated_engine.get_session_retriever(sid_a)
    retriever_b = isolated_engine.get_session_retriever(sid_b)

    assert retriever_a is not retriever_b

    chunks_a = [{"chunk_id": "ca1", "text": "Session A unique secret alpha."}]
    chunks_b = [{"chunk_id": "cb1", "text": "Session B unique secret beta."}]

    retriever_a.index_chunks(chunks_a)
    retriever_b.index_chunks(chunks_b)

    res_a = retriever_a.retrieve("secret", top_k=5)
    res_b = retriever_b.retrieve("secret", top_k=5)

    assert len(res_a) == 1
    assert res_a[0].chunk_id == "ca1"

    assert len(res_b) == 1
    assert res_b[0].chunk_id == "cb1"



def test_retrieval_does_not_reembed_unchanged_chunks(isolated_engine):
    """Real E2E showed every retrieve re-embedding all chunks (~30s each on CPU). Index must be reused."""
    from backend.retrieval.embeddings import LocalHashEmbeddingBackend

    class CountingEmbedding(LocalHashEmbeddingBackend):
        def __init__(self):
            super().__init__(dimension=64)
            self.embedded_texts = 0

        def embed_batch(self, texts):
            self.embedded_texts += len(texts)
            return super().embed_batch(texts)

    counting = CountingEmbedding()
    isolated_engine.embedding_backend = counting

    state = isolated_engine.create_session("Compare CenterPoint and PointPillars on nuScenes")
    source_id = "src_idx"
    isolated_engine.source_repo.add(source_id=source_id, session_id=state.session_id,
                                    url="https://arxiv.org/abs/2006.11275", title="CenterPoint", domain="arxiv.org")
    isolated_engine.doc_repo.add(doc_id="doc_idx", source_id=source_id, file_path=None, content_hash="h", raw_text="x")
    isolated_engine.chunk_repo.add_batch([
        {"chunk_id": f"chk_{i}", "doc_id": "doc_idx", "text": f"CenterPoint result number {i} on nuScenes.",
         "page": 1, "section": "Results", "char_start": 0, "char_end": 10, "token_count": 8}
        for i in range(3)
    ])

    for q in ["CenterPoint NDS", "PointPillars NDS", "nuScenes benchmark"]:
        assert isolated_engine.retrieve_chunks(state.session_id, q, top_k=2)
    assert counting.embedded_texts == 3  # embedded once, reused for 3 queries

    # New chunk appended -> only the new chunk is embedded
    isolated_engine.chunk_repo.add_batch([
        {"chunk_id": "chk_new", "doc_id": "doc_idx", "text": "PointPillars reaches 59.2 NDS.",
         "page": 2, "section": "Results", "char_start": 0, "char_end": 10, "token_count": 6}
    ])
    results = isolated_engine.retrieve_chunks(state.session_id, "PointPillars 59.2 NDS", top_k=4)
    assert counting.embedded_texts == 4
    assert any(r.chunk_id == "chk_new" for r in results)
