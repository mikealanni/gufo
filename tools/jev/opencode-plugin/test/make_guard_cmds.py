#!/usr/bin/env python3
"""Rebuilds the classification parity corpus from the opencode database.

Reads every bash/shell tool call from opencode's SQLite store (read only), skips commands
that contain secrets, and writes one row per command:
  {cmd, dur, sid, script_heavy, verdict}
where verdict is what tools/jev/guard.py classify() answers today and script_heavy says
whether script inspection would call it heavy. The corpus holds private command text, so it
stays out of git: the default location is ~/.gufo/jev-data/guard_cmds.json (next to
~/.gufo/jev-log), and parity.ts reads it from there (JEV_PARITY_DATA overrides).

usage: make_guard_cmds.py [--db ~/.local/share/opencode/opencode.db] [--out PATH]
"""
import argparse
import importlib.util
import json
import os
import sqlite3
import sys

GUARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "guard.py")


def load_guard():
    spec = importlib.util.spec_from_file_location("guard", GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.expanduser("~/.local/share/opencode/opencode.db"))
    ap.add_argument("--out", default=os.path.expanduser("~/.gufo/jev-data/guard_cmds.json"))
    args = ap.parse_args()
    g = load_guard()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows, seen, skipped = [], set(), 0
    cur = conn.execute(
        "SELECT session_id, data FROM part WHERE data LIKE '%\"type\":\"tool\"%' "
        "AND (data LIKE '%\"tool\":\"bash\"%' OR data LIKE '%\"tool\":\"shell\"%') "
        "ORDER BY time_created")
    for sid, data in cur:
        try:
            part = json.loads(data)
            cmd = part["state"]["input"]["command"]
            t = part["state"].get("time", {})
            dur = (t.get("end", 0) - t.get("start", 0)) / 1000.0
        except (ValueError, KeyError, TypeError):
            continue
        if not isinstance(cmd, str) or not cmd.strip() or cmd in seen:
            continue
        if g.contains_secret(cmd):
            skipped += 1
            continue
        seen.add(cmd)
        try:
            verdict = g.classify(cmd)[0]
            script_heavy = bool(g.inspect_scripts(cmd)[0])
        except Exception:  # a command the classifier cannot parse is still a data point
            verdict, script_heavy = "unknown", False
        rows.append({"cmd": cmd, "dur": round(dur, 3), "sid": sid,
                     "script_heavy": script_heavy, "verdict": verdict})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f)
    os.replace(tmp, args.out)
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(f"{len(rows)} commands ({skipped} skipped for secrets) -> {args.out}  {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
