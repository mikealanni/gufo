#!/usr/bin/env python3
"""Is the jev opencode plugin installed and logging? Read-only."""
from __future__ import annotations

import collections
import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path

HOME = Path.home()
LINK = HOME / ".config/opencode/plugins/jev.ts"
LOGDIR = Path(os.environ.get("JEV_LOG_DIR", HOME / ".gufo/jev-log"))


def age(sec: float) -> str:
    return f"{sec:.0f}s" if sec < 120 else f"{sec / 60:.0f}min" if sec < 7200 else f"{sec / 3600:.1f}h"


def service_start() -> float | None:
    """Start time of `opencode serve --service`, which keeps plugin code in memory."""
    out = subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "opencode serve --service" in line and "grep" not in line:
            parts = line.split(None, 2)
            return time.time() - int(parts[1])
    return None


def main() -> None:
    ok = LINK.is_symlink() and LINK.exists()
    print(f"installed: {'yes' if ok else 'NO'}  {LINK} -> {os.readlink(LINK) if LINK.is_symlink() else '(missing)'}")
    flag = LOGDIR / "plugin-pause"
    print(f"PAUSED: {'YES (rm ' + str(flag) + ' to resume)' if flag.exists() else 'no'}")
    print(f"kill switch JEV_PLUGIN: {os.environ.get('JEV_PLUGIN', '(unset, plugin active)')}")
    src = max((p.stat().st_mtime for p in Path(__file__).resolve().parent.glob("*.ts")), default=0)
    started = service_start()
    if started is not None and started < src:
        print("WARNING: the opencode service started before the plugin code was last edited; it is still running the OLD code.")
        print("         Fix: kill the `opencode serve --service` process by pid (the TUI starts a new one), then use opencode again.")
    elif started is None:
        print("opencode service: not running (plugin loads when opencode starts one)")
    else:
        print(f"opencode service: running since {age(time.time() - started)} ago, newer than the plugin code")
    files = sorted(LOGDIR.glob("plugin-2*.jsonl"))
    if not files:
        print("log: no plugin-YYYYMMDD.jsonl yet (plugin has not recorded anything)")
        return
    now = time.time()
    rows = []
    for line in files[-1].read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    last = max((r.get("ts", 0) for r in rows), default=0)
    print(f"log: {files[-1].name}, {len(rows)} records, last record {age(now - last)} ago")
    hour = [r for r in rows if now - r.get("ts", 0) < 3600]
    print("last hour by feature:", dict(collections.Counter(r.get("feature") for r in hour)))
    cmds = [r for r in rows if r.get("feature") == "command_gate" and r.get("dur_ms") is None]
    print("command verdicts today:", dict(collections.Counter(r["decision"]["verdict"] for r in cmds)),
          "| would stop:", sum(1 for r in cmds if r["decision"].get("would_stop")))
    errs = [r for r in rows if r.get("feature") == "error"]
    print(f"errors today: {len(errs)}" + (f"  last: {errs[-1].get('hook')}: {errs[-1].get('note')}" if errs else ""))
    print("sessions seen today:", len({r.get("session") for r in rows if r.get("session")}),
          "| providers:", sorted({r.get("provider") for r in rows if r.get("provider")}))
    try:
        c = sqlite3.connect(f"file:{LOGDIR / 'jev.sqlite3'}?mode=ro", uri=True)
        n, t = c.execute("select count(*), max(ts) from decisions where trigger='plugin-command'").fetchone()
        print(f"sqlite plugin-command rows: {n}" + (f", last {age(now - t)} ago" if t else ""))
    except sqlite3.Error as exc:
        print(f"sqlite: {exc}")
    if now - last > 1800:
        print("NOTE: nothing recorded for over 30 min; that is normal if opencode was idle or not using gufo.")


if __name__ == "__main__":
    main()
