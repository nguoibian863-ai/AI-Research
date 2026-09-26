"""
Trigger E2E Research Run via API and display the result.
"""
import sys
import json
import time
import httpx

GOAL = sys.argv[1] if len(sys.argv) > 1 else "Compare YOLOv8 and RT-DETR accuracy and latency on COCO"

print(f"[START] Triggering E2E Research for: {GOAL}")
start_time = time.time()

try:
    with httpx.Client(timeout=700.0) as client:
        resp = client.post("http://127.0.0.1:8000/api/research/run", json={"goal": GOAL})
        elapsed = time.time() - start_time
        print(f"[COMPLETED] Status {resp.status_code} in {elapsed:.1f}s")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Session ID: {data.get('session_id')}")
            print(f"Phase: {data.get('phase')}, Status: {data.get('status')}")
            print(f"Sources: {data.get('sources_count')}, Evidence: {data.get('evidence_count')}")
            report = data.get("report") or {}
            print(f"Report length: {len(report.get('content_markdown', ''))} chars")
        else:
            print(f"Error response: {resp.text[:500]}")
except Exception as e:
    elapsed = time.time() - start_time
    print(f"[FAILED] in {elapsed:.1f}s: {e}")
