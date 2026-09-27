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
        # Exact character offset invariant:
        assert cleaned.text[c.char_start:c.char_end] == c.text

    # Check that nuScenes metric exists in results chunk
    result_chunk = next((c for c in chunks if "71.2 NDS" in c.text), None)
    assert result_chunk is not None
    assert "Results" in (result_chunk.section or "")


def test_pdf_multipage_section_chunk_page_attribution(tmp_path):
    """
    Verifies that when a section spans multiple pages, chunks from page 2 get chunk.page == 2,
    not page 1, and exact character offsets match document.text.
    """
    pdf_path = tmp_path / "multipage_section.pdf"
    doc = pymupdf.open()

    # Page 1: Section starts here with paragraph 1
    page1 = doc.new_page()
    page1.insert_text((50, 50), "1. Experiments and Results\nThis is the first benchmark experiment on page 1.")

    # Page 2: Section continues here with paragraph 2
    page2 = doc.new_page()
    page2.insert_text((50, 50), "This is the second benchmark experiment on page 2 with 85.4 accuracy.")

    doc.save(str(pdf_path))
    doc.close()

    cleaned = PDFParser.parse_file(pdf_path)
    assert len(cleaned.sections) == 1
    sec = cleaned.sections[0]
    assert sec.page_start == 1
    assert sec.page_end == 2

    chunker = SectionAwareChunker(max_chunk_tokens=30, chunk_overlap_tokens=0)
    chunks = chunker.chunk_document(doc_id="doc_multi_page", document=cleaned)

    assert len(chunks) >= 2
    # Verify exact character slices for all chunks
    for c in chunks:
        assert cleaned.text[c.char_start:c.char_end] == c.text

    # The first chunk should have page 1
    p1_chunk = next((c for c in chunks if "page 1" in c.text), None)
    assert p1_chunk is not None
    assert p1_chunk.page == 1

    # The second chunk from page 2 MUST have page 2, NOT page 1!
    p2_chunk = next((c for c in chunks if "page 2" in c.text), None)
    assert p2_chunk is not None
    assert p2_chunk.page == 2



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


def test_section_aware_chunker_enforces_max_chunk_chars_limit():
    """
    Verifies that SectionAwareChunker strictly enforces max_chunk_chars <= 1800,
    even when blocks contain long sentences or paragraphs.
    """
    from backend.parsing.html_cleaner import CleanedDocument, ParsedSection

    long_sentence = "This is a detailed factual sentence about detection performance on COCO benchmark. " * 30
    very_long_block = (long_sentence + "\n\n") * 5  # > 6000 chars

    doc = CleanedDocument(
        text=very_long_block,
        title="Oversized Document Test",
        sections=[
            ParsedSection(
                title="Results",
                content=very_long_block,
                page=1,
                page_start=1,
                page_end=1,
                char_start=0,
                char_end=len(very_long_block),
                blocks=[{"text": very_long_block, "page": 1, "char_start": 0, "char_end": len(very_long_block)}]
            )
        ]
    )

    chunker = SectionAwareChunker(max_chunk_tokens=350, max_chunk_chars=1800)
    chunks = chunker.chunk_document(doc_id="doc_oversized", document=doc)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 1800, f"Chunk {c.chunk_id} length ({len(c.text)}) exceeds 1800 chars limit"
        assert doc.text[c.char_start:c.char_end] == c.text


def test_section_aware_chunker_counts_delimiter_whitespace_between_blocks():
    """
    Verifies that whitespace/newlines between blocks are accounted for,
    ensuring chunks never exceed max_chunk_chars even by 1 character.
    """
    from backend.parsing.html_cleaner import CleanedDocument, ParsedSection

    # Create multiple blocks of 500 chars separated by multi-newline gaps
    block_texts = ["A" * 500, "B" * 500, "C" * 500, "D" * 500]
    delimiter = "\n\n   \n\n"
    full_text = delimiter.join(block_texts)

    blocks = []
    offset = 0
    for bt in block_texts:
        start = full_text.find(bt, offset)
        end = start + len(bt)
        blocks.append({"text": bt, "page": 1, "char_start": start, "char_end": end})
        offset = end

    doc = CleanedDocument(
        text=full_text,
        title="Multi-block Whitespace Test",
        sections=[
            ParsedSection(
                title="Blocks",
                content=full_text,
                page=1,
                page_start=1,
                page_end=1,
                char_start=0,
                char_end=len(full_text),
                blocks=blocks
            )
        ]
    )

    chunker = SectionAwareChunker(max_chunk_tokens=1000, max_chunk_chars=1200)
    chunks = chunker.chunk_document(doc_id="doc_blocks", document=doc)

    for c in chunks:
        assert len(c.text) <= 1200, f"Chunk {c.chunk_id} length {len(c.text)} exceeds 1200 chars limit"


