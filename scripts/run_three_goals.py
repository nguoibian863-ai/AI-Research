"""
Run 3 E2E research goals sequentially and print show_session results.
"""
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from backend.main import get_research_engine
from backend.db.database import DatabaseManager
from scripts.show_session import render_session, REPORTS_DIR

GOALS = [
    "Compare YOLOv8 and RT-DETR accuracy and latency on COCO",
    "Compare PostgreSQL and MySQL transaction throughput on TPC-C",
    "Compare PointPillars and CenterPoint 3D detection on nuScenes",
]

def run_goals():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    engine = get_research_engine()
    results = []

    print("=" * 80)
    print("STARTING 3 E2E RESEARCH GOALS")
    print("=" * 80)

    for i, goal in enumerate(GOALS, start=1):
        print(f"\n[{i}/{len(GOALS)}] Running: '{goal}'")
        t0 = time.time()
        try:
            res = engine.run_week1(goal)
            elapsed = time.time() - t0
            session_id = res.get("session_id")
            phase = res.get("phase")
            status = res.get("status")
            print(f"[{i}/{len(GOALS)}] Finished in {elapsed:.1f}s | Session: {session_id} | Phase: {phase} | Status: {status}")

            # Render session report
            rendered = render_session(engine.db, session_id)
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            out_file = REPORTS_DIR / f"{session_id}.md"
            out_file.write_text(rendered, encoding="utf-8")
            print(f"Report saved to: {out_file}")

            results.append({
                "goal": goal,
                "session_id": session_id,
                "phase": phase,
                "status": status,
                "elapsed": elapsed,
                "report_file": str(out_file),
                "rendered": rendered,
                "error": None
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{i}/{len(GOALS)}] FAILED in {elapsed:.1f}s: {e}")
            results.append({
                "goal": goal,
                "session_id": None,
                "phase": "ERROR",
                "status": "FAILED",
                "elapsed": elapsed,
                "error": str(e)
            })

    print("\n" + "=" * 80)
    print("ALL 3 GOALS FINISHED SUMMARY:")
    print("=" * 80)
    for r in results:
        err_msg = f" (Error: {r['error']})" if r.get('error') else ""
        print(f"- {r['goal'][:50]}... -> {r['status']} ({r['phase']}) in {r['elapsed']:.1f}s | ID: {r.get('session_id')}{err_msg}")

if __name__ == "__main__":
    run_goals()
