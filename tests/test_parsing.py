import pytest
from pathlib import Path
import pymupdf
from backend.parsing.html_cleaner import HTMLCleaner
from backend.parsing.pdf_parser import PDFParser
from backend.parsing.chunker import SectionAwareChunker
from backend.tools.pdf_fetch import PDFFetchTool


def test_html_cleaner_extracts_article_and_sections():
    """
    Acceptance Criteria 1: Web article parse dung.
    Ensures HTMLCleaner extracts clean text, metadata, and sections with character offsets.
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Benchmarking PointPillars vs CenterPoint</title>
        <meta name="author" content="Dr. Autonomous">
        <meta name="date" content="2026-03-15">
    </head>
    <body>
        <nav><a href="/home">Home</a></nav>
        <h1>1. Introduction</h1>
        <p>PointPillars is a fast point-cloud encoder for 3D object detection in autonomous vehicles.</p>
        <h2>2. Experiments and nuScenes Benchmark</h2>
        <p>CenterPoint achieved 71.2 NDS and 65.5 mAP on the official nuScenes test split.</p>
        <footer>Copyright 2026</footer>
    </body>
    </html>
    """
    cleaned = HTMLCleaner.clean(html, url="https://example.com/3d-benchmarks")

    assert cleaned.text != ""
    assert "71.2 NDS" in cleaned.text
    assert "CenterPoint" in cleaned.text
    assert "Copyright" not in cleaned.text  # Stripped boilerplates
    assert len(cleaned.sections) >= 1

    # Verify section character offsets and titles
    intro_sec = next((s for s in cleaned.sections if "Introduction" in s.title or "Benchmarking" in s.title), cleaned.sections[0])
    assert intro_sec.char_start >= 0
    assert intro_sec.char_end > intro_sec.char_start


def test_pdf_parser_page_aware_and_sections(tmp_path):
    """
    Acceptance Criteria 2: PDF page-aware.
    Builds a synthetic 2-page PDF with PyMuPDF, parses it, and verifies page attribution per section.
    """
    pdf_path = tmp_path / "synthetic_paper.pdf"
    doc = pymupdf.open()

    # Page 1: Abstract & Introduction
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Abstract\nWe present CenterPoint for 3D detection.")
    page1.insert_text((50, 150), "1. Introduction\nLiDAR-based 3D bounding box prediction is crucial.")

    # Page 2: Experiments & Benchmark
    page2 = doc.new_page()
    page2.insert_text((50, 50), "3. Experiments\nOur Model X achieved 71.2 NDS on nuScenes benchmark.")
    page2.insert_text((50, 150), "4. Conclusion\nWe have demonstrated state-of-the-art results.")

    doc.save(str(pdf_path))
    doc.close()

    cleaned = PDFParser.parse_file(pdf_path)

    assert cleaned.metadata["page_count"] == 2
    assert "71.2 NDS" in cleaned.text
    assert len(cleaned.sections) >= 2

    # Check page awareness
    pages_recorded = [s.page for s in cleaned.sections if s.page is not None]
    assert 1 in pages_recorded
    assert 2 in pages_recorded

    exp_section = next((s for s in cleaned.sections if "Experiments" in s.title), None)
    if exp_section:
        assert exp_section.page == 2
        assert "71.2 NDS" in exp_section.content


def test_section_aware_chunker():
    """
    Verifies section-aware hierarchical chunking respecting max_tokens and overlap.
    """
    chunker = SectionAwareChunker(max_chunk_tokens=50, chunk_overlap_tokens=10)
    html = """
    <h1>1. Methodology</h1>
    <p>We train the 3D pillar feature net with sparse convolutions. The voxel size is set to 0.1 meters across all dimensions.</p>
    <p>Data augmentation includes random horizontal flips, global rotations around the Z axis, and random scaling from 0.95 to 1.05.</p>
    <h2>2. Results</h2>
    <p>CenterPoint achieves 71.2 NDS on nuScenes test set with 24 FPS inference latency on an RTX 3050 Laptop GPU.</p>
    """
    cleaned = HTMLCleaner.clean(html)
    chunks = chunker.chunk_document(doc_id="doc_test_001", document=cleaned)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.doc_id == "doc_test_001"
        assert c.chunk_id.startswith("chk_doc_test_001_")
        assert c.token_count <= 80  # within bounded chunk limit
        assert c.section is not None
        assert c.char_end > c.char_start

    # Check that nuScenes metric exists in results chunk
    result_chunk = next((c for c in chunks if "71.2 NDS" in c.text), None)
    assert result_chunk is not None
    assert "Results" in (result_chunk.section or "")


def test_pdf_fetch_tool_local(tmp_path):
    """
    Verifies PDFFetchTool reads and caches local PDFs.
    """
    pdf_path = tmp_path / "test_local.pdf"
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((50, 50), "Local PDF test text with FlashAttention-2 speedup.")
    doc.save(str(pdf_path))
    doc.close()

    fetch_tool = PDFFetchTool(cache_dir=tmp_path / "cache")
    content = fetch_tool.fetch(str(pdf_path))

    assert content is not None
    assert "FlashAttention-2" in content.text
    assert content.is_cached
