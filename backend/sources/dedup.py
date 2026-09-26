import re
import urllib.parse
from typing import Optional


def compute_canonical_key(url: str, title: Optional[str] = None) -> str:
    """
    Computes a canonical deduplication key for a source:
    1. arXiv ID (e.g. 'arxiv:2006.11275') supporting /abs/, /pdf/, /html/, versions and .pdf/.html
    2. DOI (e.g. 'doi:10.1109/cvpr.2021.00123')
    3. Normalized title slug for cross-domain mirrors of substantive papers (>= 4 words, not repo/doc subpaths)
    4. Fallback: normalized URL (lowercase, stripped query/hash/trailing slash)
    """
    clean_url = (url or "").strip().lower()

    # 1. Check for arXiv ID (supporting abs, pdf, html)
    arxiv_match = re.search(r"arxiv(?:\.org)?/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?(?:\.pdf|\.html)?", clean_url)
    if not arxiv_match:
        arxiv_match = re.search(r"\barxiv:([0-9]{4}\.[0-9]{4,5})\b", clean_url)
    if arxiv_match:
        return f"arxiv:{arxiv_match.group(1)}"

    # 2. Check for DOI
    doi_match = re.search(r"(?:doi\.org/|doi:)(10\.[0-9]{4,9}/[-._;()/:a-z0-9]+)", clean_url)
    if doi_match:
        matched_doi = doi_match.group(1).rstrip("/")
        return f"doi:{matched_doi}"

    parsed = urllib.parse.urlparse(clean_url)
    domain = parsed.netloc.lower()

    # 3. Check for normalized title for cross-domain articles
    # Avoid collapsing distinct subpaths on repo/doc platforms (like GitHub, GitLab, ReadTheDocs)
    is_repo_or_docs = any(platform in domain for platform in ["github.com", "gitlab.com", "readthedocs", "gitbook"]) or domain.startswith("docs.")
    if title and not is_repo_or_docs:
        # Strip generic website branding prefixes & suffixes
        clean_title = re.sub(r"^(?:github\s*[-–:]\s*|arxiv\s*[-–:]\s*|docs?\s*[-–:]\s*)", "", title.lower())
        clean_title = re.sub(r"\s*[-–:|]\s*(?:github|arxiv|docs?|documentation).*$", "", clean_title)
        clean_title = re.sub(r"[^\w\s-]", "", clean_title).strip()
        words = clean_title.split()
        if len(words) >= 3:
            slug = "-".join(words[:8])
            return f"title:{slug}"

    # 4. Fallback: normalized URL
    norm_url = re.sub(r"^https?://(?:www\.)?", "", clean_url)
    norm_url = re.sub(r"[?#].*$", "", norm_url)
    norm_url = norm_url.rstrip("/")
    return f"url:{norm_url}"
