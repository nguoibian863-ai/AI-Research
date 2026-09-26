"""
Xuất toàn bộ trajectory mẫu từ SQLite ra các file JSONL theo partition phục vụ Phase 2 fine-tuning (Plan 43.1).

    python scripts/export_trajectories.py               # xuất partition 'raw' vào data/trajectories/raw
    python scripts/export_trajectories.py --partition candidate  # xuất partition khác
    python scripts/export_trajectories.py --out-dir data/trajectories
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402
from backend.db.database import DatabaseManager  # noqa: E402
from backend.db.repositories import TrajectoryRepository  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Export trajectories from SQLite to JSONL files (Plan 43.1).")
    parser.add_argument("--partition", default="raw", help="Partition to export (default: 'raw').")
    parser.add_argument("--out-dir", default="data/trajectories", help="Output directory (default: 'data/trajectories').")
    parser.add_argument("--db", default=None, help="Database path (default from config/settings.yaml).")
    args = parser.parse_args()

    db_path = args.db
    if not db_path:
        config_p = Path("config/settings.yaml")
        if config_p.exists():
            with open(config_p, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f)
                db_path = settings.get("paths", {}).get("db_path", "data/research.db")
        else:
            db_path = "data/research.db"

    db = DatabaseManager(db_path=db_path)
    repo = TrajectoryRepository(db)
    counts = repo.export_partition_jsonl(partition=args.partition, output_dir=args.out_dir)

    print(f"[Trajectory Exporter] Exported partition '{args.partition}' to '{args.out_dir}/{args.partition}':")
    if not counts:
        print("  (No trajectory records found for this partition)")
    else:
        for task_type, cnt in sorted(counts.items()):
            print(f"  - {task_type}.jsonl: {cnt} samples")


if __name__ == "__main__":
    main()
