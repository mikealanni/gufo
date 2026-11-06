#!/usr/bin/env python3
"""Export opencode sessions to JSONL for `gufo judge --goal-replay`.
usage: export_sessions.py OPENCODE_DB_COPY OUT_DIR
One file per top-level session; one JSON event per line:
 {"role":"user","text":...}
 {"role":"assistant","text":...,"calls":[{"id","name","command","path"}]}
 {"role":"tool","id":callID,"name":tool,"text":output,"status":"completed|error","exit":int|null}
"""
import json, os, sqlite3, sys

LIMIT = 3000

def clip(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = json.dumps(s)
    return s if len(s) <= LIMIT else s[: LIMIT // 2] + "\n...\n" + s[-LIMIT // 2 :]

def main():
    db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    sessions = db.execute(
        "select id, model, time_created from session where parent_id is null order by time_created"
    ).fetchall()
    n = 0
    for sid, model, tc in sessions:
        msgs = db.execute(
            "select id, data from message where session_id=? order by time_created, id", (sid,)
        ).fetchall()
        events = []
        for mid, mdata in msgs:
            m = json.loads(mdata)
            role = m.get("role")
            parts = [json.loads(p[0]) for p in db.execute(
                "select data from part where message_id=? order by id", (mid,))]
            if role == "user":
                text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text" and not p.get("synthetic"))
                if text.strip():
                    events.append({"role": "user", "text": text})
            elif role == "assistant":
                text, calls, tools = [], [], []
                for p in parts:
                    t = p.get("type")
                    if t == "text":
                        text.append(p.get("text", ""))
                    elif t == "tool":
                        st = p.get("state", {})
                        inp = st.get("input", {}) if isinstance(st.get("input"), dict) else {}
                        path = inp.get("filePath") or inp.get("file_path") or inp.get("path") or ""
                        calls.append({"id": p.get("callID", ""), "name": p.get("tool", ""),
                                      "command": inp.get("command", "") if isinstance(inp.get("command", ""), str) else "",
                                      "path": path if isinstance(path, str) else ""})
                        md = st.get("metadata") or {}
                        tools.append({"role": "tool", "id": p.get("callID", ""), "name": p.get("tool", ""),
                                      "text": clip(st.get("output") if st.get("status") == "completed" else st.get("error", st.get("output"))),
                                      "status": st.get("status", ""), "exit": md.get("exit") if isinstance(md, dict) else None})
                if text or calls:
                    events.append({"role": "assistant", "text": "\n".join(text), "calls": calls})
                    events.extend(tools)
        if not any(e["role"] == "user" for e in events):
            continue
        with open(os.path.join(out, f"{n:04d}_{sid}.jsonl"), "w") as f:
            f.write(json.dumps({"session": sid, "model": model, "created": tc}) + "\n")
            for e in events:
                f.write(json.dumps(e) + "\n")
        n += 1
    print(f"exported {n} sessions to {out}")

main()
