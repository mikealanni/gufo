#!/usr/bin/env python3
"""Compare the opencode plugin's command-gate verdicts with the guard daemon's.

Reads ~/.gufo/jev-log/jev.sqlite3 (read-only). Plugin rows have trigger
'plugin-command', guard rows have trigger 'guard'. A plugin row is matched to
the nearest guard row within --window seconds whose command text shares a
prefix with it. 'heavy' means action 'stop'.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

PREFIX = 80


def load(conn: sqlite3.Connection, trigger: str, since: float) -> list[tuple[int, int, str, str, str]]:
    cur = conn.execute(
        "SELECT id, ts, action, question, model FROM decisions WHERE trigger = ? AND ts >= ? ORDER BY ts",
        (trigger, int(since)),
    )
    return [(i, ts, act or "", (q or ""), model or "") for i, ts, act, q, model in cur]


HOME = str(Path.home())


def norm(text: str) -> str:
    """The guard stores commands with ~ expanded; the plugin stores them as typed."""
    return text.replace(" ~/", f" {HOME}/").replace('"~/', f'"{HOME}/').removeprefix("~/").strip()


def same_command(a: str, b: str) -> bool:
    """The guard stores the heavy segment, the plugin the whole command: allow containment."""
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return short[:PREFIX] in long_


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(Path.home() / ".gufo/jev-log/jev.sqlite3"))
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--show", type=int, default=15, help="disagreements to print")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    since = time.time() - args.days * 86400
    plugin = load(conn, "plugin-command", since)
    guard = load(conn, "guard", since)
    used: set[int] = set()
    both_heavy = both_light = p_only = g_only = unmatched_p = 0
    disagreements: list[str] = []
    judge_only = 0
    for pid, pts, pact, pq, _pm in plugin:
        best: tuple[float, int, str, str, str] | None = None
        for gid, gts, gact, gq, gm in guard:
            if gid in used or abs(gts - pts) > args.window or not same_command(pq, gq):
                continue
            gap = abs(gts - pts)
            if best is None or gap < best[0]:
                best = (gap, gid, gact, gq, gm)
        if best is None:
            unmatched_p += 1
            continue
        used.add(best[1])
        ph, gh = pact == "stop", best[2] == "stop"
        if ph and gh:
            both_heavy += 1
        elif not ph and not gh:
            both_light += 1
        else:
            p_only += ph
            g_only += gh
            by_judge = best[4] != "guard-patterns"
            judge_only += by_judge and gh
            tag = "judge" if by_judge else "rules"
            disagreements.append(f"plugin={'heavy' if ph else 'light'} guard={'heavy' if gh else 'light'} (guard decided by {tag})  {pq[:100]!r}")
    unmatched_g = sum(1 for g in guard if g[0] not in used)
    matched = both_heavy + both_light + p_only + g_only
    print(f"window={args.window:.0f}s days={args.days:g} plugin rows={len(plugin)} guard rows={len(guard)}")
    print(f"matched={matched}  agree={both_heavy + both_light} (heavy {both_heavy}, light {both_light})")
    print(f"disagree={p_only + g_only}  plugin-heavy/guard-light={p_only}  guard-heavy/plugin-light={g_only}")
    print(f"of the guard-heavy disagreements, decided only by the judge (rules said unknown): {judge_only}")
    print(f"unmatched plugin rows={unmatched_p}  unmatched guard rows={unmatched_g}")
    if matched:
        print(f"agreement={100 * (both_heavy + both_light) / matched:.1f}%")
    for line in disagreements[: args.show]:
        print("  ", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
