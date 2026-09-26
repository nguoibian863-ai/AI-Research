import re
import urllib.parse
from datetime import datetime
from typing import Dict, Any, Optional

TIER_1_ACADEMIC_DOMAINS = {
    "arxiv.org", "acm.org", "ieee.org", "nature.com", "openreview.net",
    "springer.com", "biorxiv.org", "usenix.org", "vldb.org", "aclweb.org",
    "neurips.cc", "iclr.cc", "cvpr.org", "icml.cc", "sciencedirect.com"
}

TIER_2_OFFICIAL_DOMAINS = {
    "github.com", "gitlab.com", "readthedocs.io", "gitbook.io",
    "postgresql.org", "kernel.org", "apache.org", "python.org",
    "research.google", "openai.com", "meta.com", "microsoft.com", "anthropic.com"
}


def score_source_credibility(
    url: str,
    title: Optional[str] = None,
    domain: Optional[str] = None,
    text: Optional[str] = None,
    published_at: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes a multi-dimensional credibility score for a research source according to Plan 14:
    1. authority: Domain academic/institutional prestige
    2. primary_source: Direct original research/docs vs secondary third-party blogs
    3. recency: Publication freshness
    4. directness: Concrete empirical metrics vs opinion commentary
    5. independence: First-party independent publication
    6. reproducibility: Open-source code, public dataset or artifacts available
    """
    clean_url = (url or "").strip().lower()
    parsed = urllib.parse.urlparse(clean_url)
    dom = (domain or parsed.netloc).lower()
    dom = re.sub(r"^www\.", "", dom.split(":")[0])

    # 1. Authority
    if any(ad in dom for ad in TIER_1_ACADEMIC_DOMAINS):
        authority = 1.0
        tier = "TIER_1_ACADEMIC"
    elif any(od in dom for od in TIER_2_OFFICIAL_DOMAINS) or dom.startswith("docs."):
        authority = 0.85
        tier = "TIER_2_OFFICIAL"
    elif dom.endswith(".edu") or dom.endswith(".gov"):
        authority = 0.90
        tier = "TIER_1_INSTITUTIONAL"
    elif any(td in dom for td in ["medium.com", "substack.com", "dev.to", "hashnode.com", "towardsdatascience.com"]):
        authority = 0.50
        tier = "TIER_3_COMMUNITY_BLOG"
    else:
        authority = 0.60
        tier = "TIER_3_GENERAL_WEB"

    # 2. Primary Source (1.0 for original papers / official docs / repositories)
    if tier in {"TIER_1_ACADEMIC", "TIER_2_OFFICIAL"} or "arxiv" in dom or "doi.org" in clean_url:
        primary_source = 1.0
    elif tier == "TIER_3_COMMUNITY_BLOG":
        primary_source = 0.45
    else:
        primary_source = 0.60

    # 3. Recency (Score freshness, default 0.85 if year cannot be deduced)
    recency = 0.85
    year = None
    arxiv_m = re.search(r"arxiv(?:\.org)?/(?:abs|pdf|html)/([0-9]{2})[0-9]{2}\.", clean_url)
    if arxiv_m:
        year = 2000 + int(arxiv_m.group(1))
    elif published_at:
        year_m = re.search(r"\b(19\d{2}|20\d{2})\b", published_at)
        if year_m:
            year = int(year_m.group(1))
    elif title:
        year_m = re.search(r"\b(19\d{2}|20\d{2})\b", title)
        if year_m:
            year = int(year_m.group(1))

    current_year = datetime.now().year
    if year:
        age = max(0, current_year - year)
        if age <= 2:
            recency = 1.0
        elif age <= 4:
            recency = 0.90
        elif age <= 7:
            recency = 0.80
        else:
            recency = 0.65

    # 4. Directness (Empirical benchmarks / results / code vs opinion)
    directness = 0.70
    if text:
        text_lower = text.lower()
        if any(w in text_lower for w in ["benchmark", "experiment", "evaluation", "results", "table", "latency", "accuracy", "map", "nds"]):
            directness = 0.95
        elif any(w in text_lower for w in ["tutorial", "guide", "documentation", "api reference"]):
            directness = 0.85
    elif "arxiv" in clean_url or "doi" in clean_url:
        directness = 0.95

    # 5. Independence (Non-sponsored, canonical peer-reviewed or open-source)
    independence = 0.85
    if "arxiv" in dom or ".edu" in dom or "github.com" in dom:
        independence = 1.0
    elif tier == "TIER_3_COMMUNITY_BLOG":
        independence = 0.60

    # 6. Reproducibility (Code repository, artifact evaluation, open benchmarks)
    reproducibility = 0.60
    if "github.com" in clean_url or "gitlab.com" in clean_url:
        reproducibility = 1.0
    elif text and any(w in text.lower() for w in ["github.com", "reproduce", "open source", "weights available", "code is available"]):
        reproducibility = 0.90
    elif "arxiv" in clean_url:
        reproducibility = 0.75

    composite_score = round(
        0.30 * authority +
        0.25 * primary_source +
        0.15 * recency +
        0.10 * directness +
        0.10 * independence +
        0.10 * reproducibility,
        3
    )

    return {
        "score": composite_score,
        "tier": tier,
        "authority": authority,
        "primary_source": primary_source,
        "recency": recency,
        "directness": directness,
        "independence": independence,
        "reproducibility": reproducibility
    }
