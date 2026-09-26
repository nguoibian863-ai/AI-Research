import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class CitationVerifier:
    """
    Validates report citations against verified atomic evidence and appends a
    structured Evidence & Provenance Table for complete click-through verification.
    """

    @staticmethod
    def format_evidence_for_prompt(evidence_items: List[Dict[str, Any]]) -> str:
        """
        Formats evidence items with unambiguous citation indices [E1], [E2], ...
        for the LLM report synthesizer.
        """
        if not evidence_items:
            return "No verified atomic evidence available."

        lines = []
        for i, ev in enumerate(evidence_items, start=1):
            statement = ev.get("statement") or f"{ev.get('subject', '')} {ev.get('predicate', '')}"
            quote = ev.get("exact_quote", "")
            url = ev.get("url") or "local"
            sec = ev.get("section") or ""
            page = ev.get("page")
            loc_parts = []
            if sec:
                loc_parts.append(f"Section: {sec}")
            if page is not None and page != "":
                loc_parts.append(f"Page: {page}")
            loc_str = f" ({', '.join(loc_parts)})" if loc_parts else ""
            source_score = ev.get("source_score")
            if source_score is not None:
                loc_str += f" [Source score: {source_score:.0f}/100]"

            lines.append(
                f"[E{i}] {statement}\n"
                f"     Quote: \"{quote}\"\n"
                f"     Source: {url}{loc_str}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def verify_and_append_appendix(
        report_markdown: str,
        evidence_items: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Verifies all [E#] citations in the report against the evidence items.
        Appends a Markdown Evidence & Provenance Table to the report.
        """
        if not evidence_items:
            return report_markdown, {"valid_citations": [], "invalid_citations": [], "coverage_ratio": 0.0}

        # Find all citation tokens like [E1], [E2]
        found_tokens = re.findall(r"\[E(\d+)\]", report_markdown)
        found_indices = sorted(list({int(t) for t in found_tokens}))

        max_idx = len(evidence_items)
        valid_indices = [idx for idx in found_indices if 1 <= idx <= max_idx]
        invalid_indices = [idx for idx in found_indices if idx < 1 or idx > max_idx]

        if invalid_indices:
            logger.warning(f"Report contains invalid citation indices: {invalid_indices} (max valid is {max_idx})")

        # Build Markdown Evidence Table
        table_lines = [
            "\n\n---\n\n### Evidence & Provenance Table\n",
            "| Citation | Key Fact | Source | Location | Exact Verbatim Quote |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]

        for i, ev in enumerate(evidence_items, start=1):
            stmt = (ev.get("statement") or "").replace("|", "\\|")
            quote = (ev.get("exact_quote") or "").replace("|", "\\|").replace("\n", " ")
            url = ev.get("url") or "local"
            title = (ev.get("source_title") or url).replace("|", "\\|")
            sec = (ev.get("section") or "").replace("|", "\\|")
            page = ev.get("page")

            loc_str = sec if sec else ""
            if page is not None and str(page).strip() != "":
                loc_str = f"{loc_str} (Page {page})".strip()
            if not loc_str:
                loc_str = "-"

            source_link = f"[{title[:35]}]({url})" if url.startswith("http") else title[:35]
            quote_snippet = f"*\"{quote[:120]}...\"*" if len(quote) > 120 else f"*\"{quote}\"*"

            table_lines.append(
                f"| **[E{i}]** | {stmt} | {source_link} | {loc_str} | {quote_snippet} |"
            )

        appendix = "\n".join(table_lines)
        augmented_report = report_markdown + appendix

        stats = {
            "total_evidence": len(evidence_items),
            "cited_evidence": len(valid_indices),
            "valid_indices": valid_indices,
            "invalid_indices": invalid_indices,
            "citation_rate": len(valid_indices) / len(evidence_items) if evidence_items else 0.0
        }

        return augmented_report, stats
