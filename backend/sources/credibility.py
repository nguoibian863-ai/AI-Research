import re
import urllib.parse
from datetime import datetime
from typing import Dict, Any, Iterable, Optional

from backend.core.coverage import check_entity_in_text

# Plan 14.1: source_score (0–100) weights
WEIGHTS = {
    "authority": 30,
    "primary_source": 30,
    "directness": 20,
    "recency": 10,
    "independence": 10,
}

# Plan 14.2: sources below this score are not fetched
MIN_SOURCE_SCORE = 40.0

TIER_1_ACADEMIC_DOMAINS = {
    "arxiv.org", "acm.org", "ieee.org", "nature.com", "openreview.net",
    "springer.com", "biorxiv.org", "usenix.org", "vldb.org", "aclweb.org",
    "aclanthology.org", "neurips.cc", "iclr.cc", "thecvf.com", "icml.cc",
    "sciencedirect.com", "doi.org", "semanticscholar.org", "openalex.org"
}

CODE_HOST_DOMAINS = {"github.com", "gitlab.com", "huggingface.co"}

TIER_2_OFFICIAL_DOMAINS = {
    "readthedocs.io", "gitbook.io", "postgresql.org", "kernel.org", "apache.org",
    "python.org", "pytorch.org", "tensorflow.org", "research.google", "openai.com",
    "meta.com", "microsoft.com", "anthropic.com", "nvidia.com", "ultralytics.com"
}

COMMUNITY_BLOG_DOMAINS = {
    "medium.com", "substack.com", "dev.to", "hashnode.com", "towardsdatascience.com"
}

FORUM_QA_DOMAINS = {"reddit.com", "quora.com", "stackoverflow.com", "stackexchange.com"}

SOCIAL_DOMAINS = {
    "facebook.com", "instagram.com", "twitter.com", "x.com", "tiktok.com",
    "pinterest.com", "linkedin.com", "youtube.com", "whatsapp.com"
}


def domain_matches(domain: str, candidates: Iterable[str]) -> bool:
    """Exact host or subdomain match only: 'github.com.evil.io' and 'pacm.org' do not match."""
    return any(domain == d or domain.endswith("." + d) for d in candidates)


def _normalize_domain(url: str, domain: Optional[str]) -> str:
    dom = (domain or urllib.parse.urlparse(url).netloc or "").lower()
    dom = dom.split(":")[0]
    return re.sub(r"^www\.", "", dom)


def _repo_matches_entities(url: str, entities: Iterable[str]) -> bool:
    """Code-host URL counts as primary only when owner/repo path names a goal entity."""
    path = urllib.parse.urlparse(url).path.lower()
    parts = [p for p in path.split("/") if p][:2]
    repo_path = " ".join(parts)
    return any(check_entity_in_text(e, repo_path) for e in entities)


def score_source_credibility(
    url: str,
    title: Optional[str] = None,
    domain: Optional[str] = None,
    text: Optional[str] = None,
    published_at: Optional[str] = None,
    goal_entities: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Computes source_score (0–100) per Plan 14.1:
      authority 30% · primary_source 30% · directness 20% · recency 10% · independence 10%
    Pure Python, deterministic. Each component is in [0, 1]; details are stored for explainability.
    """
    clean_url = (url or "").strip()
    url_lower = clean_url.lower()
    dom = _normalize_domain(url_lower, domain)
    entities = [e.lower() for e in (goal_entities or []) if e]

    # 1. Authority + tier
    if domain_matches(dom, SOCIAL_DOMAINS):
        tier, authority = "SOCIAL", 0.10
    elif domain_matches(dom, FORUM_QA_DOMAINS):
        tier, authority = "FORUM_QA", 0.35
    elif domain_matches(dom, TIER_1_ACADEMIC_DOMAINS):
        tier, authority = "TIER_1_ACADEMIC", 1.0
    elif dom.endswith(".edu") or dom.endswith(".gov") or ".ac." in f".{dom}":
        tier, authority = "TIER_1_INSTITUTIONAL", 0.90
    elif domain_matches(dom, CODE_HOST_DOMAINS):
        tier, authority = "CODE_HOST", 0.70
    elif domain_matches(dom, TIER_2_OFFICIAL_DOMAINS) or dom.startswith("docs."):
        tier, authority = "TIER_2_OFFICIAL", 0.85
    elif domain_matches(dom, COMMUNITY_BLOG_DOMAINS):
        tier, authority = "TIER_3_COMMUNITY_BLOG", 0.50
    else:
        tier, authority = "TIER_3_GENERAL_WEB", 0.55

    # 2. Primary source: original paper / official docs / the entity's own repository
    if tier in {"TIER_1_ACADEMIC", "TIER_1_INSTITUTIONAL", "TIER_2_OFFICIAL"}:
        primary_source = 1.0
    elif tier == "CODE_HOST":
        primary_source = 1.0 if entities and _repo_matches_entities(url_lower, entities) else 0.40
    elif tier == "TIER_3_COMMUNITY_BLOG":
        primary_source = 0.40
    elif tier in {"FORUM_QA", "SOCIAL"}:
        primary_source = 0.10
    else:
        primary_source = 0.50

    # 3. Directness: share of goal entities named in title/snippet/URL
    evidence_text = " ".join(filter(None, [title, text, url_lower]))
    if entities:
        matched = sum(1 for e in entities if check_entity_in_text(e, evidence_text))
        directness = matched / len(entities)
    else:
        directness = 0.50

    # 4. Recency
    recency = 0.70
    year = None
    arxiv_m = re.search(r"arxiv(?:\.org)?/(?:abs|pdf|html)/([0-9]{2})[0-9]{2}\.", url_lower)
    if arxiv_m:
        year = 2000 + int(arxiv_m.group(1))
    else:
        for field in (published_at, title):
            if field:
                year_m = re.search(r"\b(19\d{2}|20\d{2})\b", field)
                if year_m:
                    year = int(year_m.group(1))
                    break
    if year:
        age = max(0, datetime.now().year - year)
        recency = 1.0 if age <= 2 else 0.85 if age <= 4 else 0.70 if age <= 7 else 0.50

    # 5. Independence: third-party aggregation and social reposts are less independent
    if tier in {"TIER_3_COMMUNITY_BLOG", "FORUM_QA", "SOCIAL"}:
        independence = 0.50
    else:
        independence = 1.0

    components = {
        "authority": authority,
        "primary_source": primary_source,
        "directness": round(directness, 3),
        "recency": recency,
        "independence": independence,
    }
    score = round(sum(WEIGHTS[k] * v for k, v in components.items()), 1)

    return {
        "score": score,
        "tier": tier,
        "domain": dom,
        "year": year,
        **components,
    }
