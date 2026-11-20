#!/usr/bin/env python3
"""Extract a DFlash2 distillation corpus from opencode's SQLite store.

Reads the ``part`` table (content JSON in ``data``), keeps ``type=="text"``
parts, splits by author (message ``data.role``), splits into ~max-bytes shards
and writes plain .txt files. Reasoning/tool/patch parts are skipped. Runs
against a read-only copy of the live DB to avoid lock contention.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3


def parts(cur, role):
    sql = ("SELECT p.data FROM part p JOIN message m ON p.message_id=m.id "
           "WHERE m.session_id=p.session_id AND json_extract(p.data,'$.type')='text' "
           "AND json_extract(m.data,'$.role')=? ORDER BY m.session_id, m.time_created, p.rowid")
    for row in cur.execute(sql, (role,)):
        try:
            text = json.loads(row[0]).get("text", "")
        except (ValueError, TypeError):
            continue
        if text.strip():
            yield text


def _split(doc, max_bytes):
    """Yield pieces of a single document, each no longer than max_bytes."""
    start = 0
    n = len(doc)
    while n - start > max_bytes:
        cut = doc.rfind(" ", start, start + max_bytes)
        if cut <= start:
            cut = start + max_bytes
        yield doc[start:cut]
        start = cut + 1
    tail = doc[start:]
    if tail:
        yield tail


def emit(docs, out_dir, prefix, min_doc, max_bytes, max_shards=0):
    n = 0
    pending = ""
    def write(chunk):
        nonlocal n
        if max_shards and n >= max_shards:
            return False
        if len(chunk.strip()) >= min_doc:
            (out_dir / f"{prefix}_{n:04d}.txt").write_text(chunk)
            n += 1
        return not (max_shards and n >= max_shards)
    for doc in docs:
        for piece in _split(doc, max_bytes):
            if len(pending) + len(piece) + 2 <= max_bytes:
                pending += piece + "\n\n"
                continue
            if not write(pending):
                return n, 0
            pending = piece + "\n\n"
    write(pending)
    return n, 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="path to opencode.db (or a copy)")
    ap.add_argument("--out", required=True, help="output corpus directory")
    ap.add_argument("--min-doc", type=int, default=800,
                    help="skip text fragments shorter than this many chars")
    ap.add_argument("--max-bytes", type=int, default=15000, help="shard size")
    ap.add_argument("--limit", type=int, default=0, help="stop after N shards")
    ap.add_argument("--max-per-role", type=int, default=200,
                    help="cap shards written per role (assistant+user); 0 = no cap")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    cur = con.cursor()
    total = 0
    for role, prefix in (("assistant", "assistant"), ("user", "user")):
        n, _ = emit(parts(cur, role), out, prefix, args.min_doc, args.max_bytes,
                    args.max_per_role)
        print(f"{role}: {n} shard(s)")
        total += n
    print(f"total {total} shards -> {out}")
    print("bytes:", sum(f.stat().st_size for f in out.glob("*.txt")))


if __name__ == "__main__":
    main()
