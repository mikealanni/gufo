#!/usr/bin/env python3
"""Logging HTTP pass-through for a gufo serve instance.

Stands between the client (opencode) and gufo serve, streams both directions
untouched, and appends one JSON record per request to a JSONL log. The log is
the input to replay.py, so a real coding session can be replayed as a
benchmark instead of a synthetic prompt.

Streaming responses are forwarded chunk-by-chunk and never buffered, so SSE
clients behave exactly as they do against the server directly.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import urllib.error
import urllib.request

# Hop-by-hop headers must not be forwarded.
SKIP_REQUEST = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}
SKIP_RESPONSE = SKIP_REQUEST | {"content-length", "content-encoding"}

# The provider config keeps its historical model id, so the proxy maps it onto
# whatever this serve actually offers. Both names are logged.
MODEL_ALIASES = {"qwen35": "Qwen3.8 Flash Next"}

_lock = threading.Lock()
_upstream = ""
_log_path: pathlib.Path | None = None


def _append(record: dict) -> None:
    if _log_path is None:
        return
    with _lock:
        with _log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            fh.flush()


def _summarise(body: bytes) -> dict:
    """Pull the fields a replay needs, plus the sampling knobs worth reporting."""
    out: dict = {}
    try:
        payload = json.loads(body or b"{}")
    except Exception:
        return {"_unparsed": True, "_bytes": len(body)}
    if not isinstance(payload, dict):
        return {"_unparsed": True, "_bytes": len(body)}
    for key in ("model", "temperature", "top_p", "top_k", "max_tokens",
                "stream", "seed", "presence_penalty", "frequency_penalty",
                "repetition_penalty", "logprobs", "n"):
        if key in payload:
            out[key] = payload[key]
    # Thinking / reasoning switches appear under several vendor spellings.
    for key in ("thinking", "reasoning", "reasoning_effort", "chat_template_kwargs",
                "enable_thinking", "reasoning_format"):
        if key in payload:
            out[key] = payload[key]
    out["_tools"] = len(payload.get("tools") or [])
    # Replay needs the real transcript, not a summary of it.
    out["_messages_replay"] = payload.get("messages")
    if payload.get("tools"):
        out["tools_replay"] = payload.get("tools")
    out["_tool_choice"] = payload.get("tool_choice")
    messages = payload.get("messages")
    if isinstance(messages, list):
        out["_messages"] = len(messages)
        # Rough context estimate: ~3.6 chars/token for English + code.
        chars = 0
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        chars += len(str(part.get("text", "")))
            chars += len(str(m.get("tool_calls") or ""))
        out["_approx_prompt_chars"] = chars
        if messages:
            out["_system"] = str(messages[0].get("content", ""))[:120]
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "gufo-logging-proxy/1"

    def log_message(self, fmt, *args):  # silence the default stderr spam
        pass

    def _proxy(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        injected = False
        aliased = False
        original_model = None
        if body:
            # Ask for usage so the log carries exact prompt/completion token
            # counts. This changes only what is reported, never the generation,
            # but it is recorded because the forwarded request is not byte
            # identical to the client's.
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    original_model = parsed.get("model")
                    alias = MODEL_ALIASES.get(original_model)
                    if alias is not None:
                        parsed["model"] = alias
                        aliased = True
                    if (parsed.get("stream")
                            and not (parsed.get("stream_options") or {})
                            .get("include_usage")):
                        opts = dict(parsed.get("stream_options") or {})
                        opts["include_usage"] = True
                        parsed["stream_options"] = opts
                        injected = True
                    if aliased or injected:
                        body = json.dumps(parsed).encode()
            except json.JSONDecodeError:
                pass
        url = _upstream.rstrip("/") + self.path
        req = urllib.request.Request(url, data=body or None, method=method)
        for key, value in self.headers.items():
            if key.lower() not in SKIP_REQUEST:
                req.add_header(key, value)
        if injected:
            req.add_header("Content-Length", str(len(body)))

        started = time.time()
        info = _summarise(body)
        info["model_original"] = original_model
        info["model_rewritten"] = aliased
        record: dict = {
            "t_start": started,
            "method": method,
            "path": self.path,
            "request": info,
            "injected_stream_options": injected,
            "model_aliased": aliased,
        }
        try:
            resp = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as exc:
            record["status"] = exc.code
            record["t_first_byte"] = time.time()
            record["t_end"] = time.time()
            record["error"] = str(exc)
            _append(record)
            payload = exc.read() or b""
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type",
                                                             "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        except Exception as exc:  # upstream refused / DNS / etc.
            record["status"] = 0
            record["t_end"] = time.time()
            record["error"] = repr(exc)
            _append(record)
            self.send_response(502)
            payload = json.dumps({"error": repr(exc)}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(resp.status)
        streaming = (resp.headers.get("Content-Type") or "").startswith(
            "text/event-stream")
        for key, value in resp.headers.items():
            if key.lower() not in SKIP_RESPONSE:
                self.send_header(key, value)
        if streaming:
            # No Content-Length: chunked until the generator ends.
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        first = None
        total = 0
        usage = None
        tail = b""
        try:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                if first is None:
                    first = time.time()
                total += len(chunk)
                if streaming:
                    # Watch the stream without altering it: gufo sends usage in
                    # a final data chunk, and replay needs exact token counts.
                    tail += chunk
                    if len(tail) > (1 << 20):
                        tail = tail[-(1 << 19):]
                    for raw in tail.split(b"\n")[:-1]:
                        raw = raw.strip()
                        if not raw.startswith(b"data:"):
                            continue
                        body = raw[5:].strip()
                        if not body or body == b"[DONE]":
                            continue
                        try:
                            parsed = json.loads(body)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(parsed, dict) and parsed.get("usage"):
                            usage = parsed["usage"]
                if streaming:
                    self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                    self.wfile.flush()
                else:
                    self.wfile.write(chunk)
        except Exception as exc:
            record["stream_error"] = repr(exc)
        finally:
            if streaming:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except Exception:
                    pass



        record["status"] = resp.status
        record["t_first_byte"] = first or time.time()
        record["t_end"] = time.time()
        record["response_bytes"] = total
        if usage is not None:
            record["usage"] = usage
        _append(record)

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")


def main() -> None:
    global _upstream, _log_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, required=True, help="port to listen on")
    ap.add_argument("--upstream", required=True,
                    help="gufo serve base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--log", default="proxy.jsonl")
    args = ap.parse_args()
    _upstream = args.upstream
    _log_path = pathlib.Path(args.log)
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.listen), Handler)
    server.daemon_threads = True
    print(f"proxy 127.0.0.1:{args.listen} -> {_upstream}  log={_log_path}",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
