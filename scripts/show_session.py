"""
Xem kết quả research đã lưu trong SQLite, không cần server hay công cụ ngoài.

    python scripts/show_session.py                 # liệt kê 10 session gần nhất
    python scripts/show_session.py sess_xxx        # chi tiết 1 session + xuất report ra data/reports/sess_xxx.md
    python scripts/show_session.py --latest        # chi tiết session mới nhất
    python scripts/show_session.py --db path.db    # dùng DB khác
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from backend.db.database import DatabaseManager  # noqa: E402
from backend.db.repositories import (  # noqa: E402
    ClaimRepository,
    EvidenceRepository,
    ReportRepository,
    SessionRepository,
    SourceRepository,
)

REPORTS_DIR = Path("data/reports")


def default_db_path() -> str:
    cfg = Path("config/settings.yaml")
    if cfg.exists():
        settings = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        return settings.get("paths", {}).get("db_path", "data/research.db")
    return "data/research.db"


def short(text: Optional[str], width: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def list_sessions(db: DatabaseManager, limit: int = 10) -> List[Dict[str, Any]]:
    with db.session() as conn:
        rows = conn.execute(
            "SELECT session_id, phase, status, goal, created_at FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def print_session_list(db: DatabaseManager) -> None:
    sessions = list_sessions(db)
    if not sessions:
        print("Chưa có session nào trong DB.")
        return
    print(f"{'SESSION':<20} {'PHASE':<9} {'CREATED':<20} GOAL")
    for s in sessions:
        print(f"{s['session_id']:<20} {s['phase']:<9} {str(s['created_at'])[:19]:<20} {short(s['goal'], 70)}")
    print("\nXem chi tiết: python scripts/show_session.py <SESSION>")


def render_session(db: DatabaseManager, session_id: str) -> str:
    session = SessionRepository(db).get(session_id)
    if not session:
        raise SystemExit(f"Không tìm thấy session '{session_id}'.")

    sources = SourceRepository(db).get_by_session(session_id)
    evidence = EvidenceRepository(db).get_full_evidence_by_session(session_id)
    claims = ClaimRepository(db).get_by_session(session_id)
    report = ReportRepository(db).get_by_session(session_id)

    out: List[str] = []
    out.append(f"# {session['goal']}")
    out.append("")
    out.append(f"- Session: `{session_id}`")
    out.append(f"- Phase / status: **{session['phase']} / {session['status']}**")
    if session.get("error_message"):
        out.append(f"- Lý do: {session['error_message']}")
    out.append(
        f"- Budget: search {session.get('search_calls') or 0}, fetch {session.get('fetch_calls') or 0}, "
        f"LLM {session.get('llm_calls') or 0}, tokens {session.get('tokens_consumed') or 0}"
    )
    out.append("")

    out.append(f"## Nguồn ({len(sources)})")
    out.append("")
    out.append("| Điểm | Loại | Tiêu đề | URL |")
    out.append("|---:|---|---|---|")
    for s in sorted(sources, key=lambda x: -(x.get("credibility_score") or 0)):
        score = s.get("credibility_score")
        score_str = f"{score:.0f}" if isinstance(score, (int, float)) else "-"
        out.append(f"| {score_str} | {s.get('source_type') or ''} | {short(s.get('title'), 60)} | {s.get('url')} |")
    out.append("")

    out.append(f"## Evidence đã kiểm chứng ({len(evidence)})")
    out.append("")
    for i, ev in enumerate(evidence, start=1):
        loc = ", ".join(p for p in [ev.get("section") or "", f"page {ev['page']}" if ev.get("page") else ""] if p)
        out.append(f"**[E{i}] {ev.get('subject') or ''}** — {ev.get('statement') or ''}")
        out.append(f"> \"{ev.get('raw_quote') or ''}\"")
        out.append(f"> {ev.get('url') or ''}{' (' + loc + ')' if loc else ''}")
        out.append("")

    out.append(f"## Claims trong report ({len(claims)})")
    out.append("")
    for c in claims:
        ver = c.get("verification") or {}
        if not ver and c.get("verification_json"):
            try:
                import json
                ver = json.loads(c["verification_json"])
            except Exception:
                ver = {}
        ver_label = ver.get("entailment") or ver.get("label")
        ver_conf = ver.get("entailment_confidence") if ver.get("entailment_confidence") is not None else ver.get("confidence", 0)
        ver_reason = ver.get("entailment_reason") or ver.get("reason")
        ver_type = ver.get("verifier_type", "")
        ver_prompt_v = ver.get("prompt_version") or ver.get("nli_prompt_version")
        ver_info = ""
        if ver_label:
            v_tag = f" {ver_prompt_v}" if ver_prompt_v else ""
            ver_info = f" [NLI: {ver_label} ({ver_conf:.2f}) by {ver_type}{v_tag}]"
        out.append(f"- **{c['status']}**{ver_info}: {short(c.get('text'), 160)}")
        if ver_reason:
            out.append(f"  > Lý do NLI: {ver_reason}")
    out.append("")

    out.append("## Report")
    out.append("")
    out.append(report["content_markdown"] if report else "_Chưa có report (session chưa chạy tới WRITE)._")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Xem session research đã lưu trong SQLite.")
    parser.add_argument("session_id", nargs="?", help="ID session (bỏ trống để liệt kê)")
    parser.add_argument("--latest", action="store_true", help="Xem session mới nhất")
    parser.add_argument("--db", default=None, help="Đường dẫn file SQLite")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    db = DatabaseManager(db_path=args.db or default_db_path())

    session_id = args.session_id
    if args.latest:
        sessions = list_sessions(db, limit=1)
        session_id = sessions[0]["session_id"] if sessions else None
    if not session_id:
        print_session_list(db)
        return

    text = render_session(db, session_id)
    print(text)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / f"{session_id}.md"
    out_file.write_text(text, encoding="utf-8")
    print(f"\n[Đã lưu] {out_file}  (mở bằng VS Code: code {out_file})")


if __name__ == "__main__":
    main()
