#!/usr/bin/env python3
"""Run the full verifiable demo from the command line (no server, no API key needed).

    python scripts/run_demo.py [--no-fault] [--out demo_output]

Everything printed is real: each [exec] line is an actual sandbox execution and the metrics are computed from
the predictions the generated code produced. Mode (demo/LLM) and sandbox isolation are printed explicitly.
"""
import argparse, json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app.config import Settings
from app.services import Service


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fault", action="store_true"); ap.add_argument("--out", default="demo_output")
    a = ap.parse_args()
    data = tempfile.mkdtemp(prefix="p2p_demo_")
    s = Settings.from_env(); s = Settings(**{**s.__dict__, "data_dir": data, "db_path": os.path.join(data, "demo.db")})
    sv = Service(s, sync_jobs=True)
    print("integrations:", json.dumps(sv.integrations(), indent=1))
    pid = sv.create_demo(inject_fault=not a.no_fault)
    sv.approve_spec(pid, None)
    print("\n-- plan awaiting approval:", sv.get_project(pid)["state"]["plan"]["assumptions"])
    sv.approve_plan(pid, True)
    print("\n-- event log")
    for e in sv.events(pid):
        if e["stage"] != "state":
            print(f"[{e['stage']:9}] {e['level']:5} {e['message'][:140]}")
    st = sv.get_project(pid)
    print("\nFINAL STATUS:", st["status"], "| reproduction:", st["state"]["reproduction_status"])
    print("NOTE:", st["state"]["reproduction_note"])
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "REPORT.md"), "w").write(sv.report_md(pid))
    open(os.path.join(a.out, "reproducibility_manifest.json"), "w").write(json.dumps(sv.manifest(pid), indent=2))
    open(os.path.join(a.out, "project.zip"), "wb").write(sv.zip_bytes(pid))
    print("wrote", os.path.abspath(a.out))
    return 0 if st["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
