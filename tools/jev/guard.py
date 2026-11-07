#!/usr/bin/env python3
"""jev-guard: keeps gufo out of the way of heavy tests an agent runs locally.

Sits on the public port, runs gufo on an internal port, and
  1. holds the final tool-call chunk of a response until gufo has been stopped
     when the command needs the GPU or most of the RAM,
  2. holds incoming requests (SSE keepalives) while the test runs and gufo is down,
  3. restarts gufo when the test is gone and forwards the held request,
  4. stops gufo on its own when free RAM collapses (unknown commands).
"""

import argparse
import asyncio
import json
import os
import re
import signal
import sys
import time

import aiohttp
from aiohttp import web

HEAVY = [re.compile(p) for p in (
    r"\bgufo(\s+\S+)*?\s+(bench|serve|judge|diagnose|eval|transcribe)\b",
    r"build/\S+/gufo\b",
    r"\bctest\b",
    r"\bcmake\s+--build\b",
    r"\bnix\s+(build|develop)\b",
    r"\brocprof\w*",
    r"model-bench\.py",
    r"tools/(bench|prof)/",
    r"\bpytest\b",
    r"\bpython3?\b.*\b(torch|hip|rocm)\b",
    r"run-gufo\.sh",
)]
LIGHT_WORDS = {
    "git", "ls", "cat", "rg", "grep", "sed", "head", "tail", "echo", "pwd", "cd",
    "find", "wc", "diff", "mkdir", "cp", "mv", "rm", "touch", "curl", "tr", "sort",
    "uniq", "awk", "stat", "file", "which", "date", "env", "printf", "test", "true",
}
SEGMENT_SPLIT = re.compile(r"&&|\|\||;|\|")
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "upgrade",
       "proxy-authenticate", "proxy-authorization", "trailers", "content-length",
       "host", "content-encoding"}
SHELL_TOOLS = {"bash", "shell", "run_command", "execute_command"}
JUDGE_PROMPT = ("Does this shell command need the whole GPU or most of the RAM "
                "of the machine? Command: ")


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "[jev-guard]", msg, flush=True)


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576
    return 0.0


def segments(command):
    return [s.strip() for s in SEGMENT_SPLIT.split(command) if s.strip()]


def first_word(segment):
    for word in segment.split():
        if "=" in word and not word.startswith("-"):
            continue
        return os.path.basename(word)
    return ""


def classify(command):
    """Returns (verdict, segment): heavy / light / unknown."""
    unknown = ""
    for seg in segments(command):
        if any(p.search(seg) for p in HEAVY):
            return "heavy", seg
        if first_word(seg) not in LIGHT_WORDS and not unknown:
            unknown = seg
    return ("unknown", unknown) if unknown else ("light", "")


def tool_commands(calls):
    out = []
    for call in calls.values():
        if call["name"].lower() not in SHELL_TOOLS:
            continue
        try:
            args = json.loads(call["args"] or "{}")
        except ValueError:
            continue
        cmd = args.get("command") or args.get("cmd")
        if isinstance(cmd, list):
            cmd = " ".join(map(str, cmd))
        if isinstance(cmd, str):
            out.append(cmd)
    return out


def find_pids(needle):
    me = os.getpid()
    hits = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if needle in cmd:
            hits.append(int(entry))
    return hits


def biggest_foreign_rss(skip_pgid):
    best, best_pid = 0, 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            if os.getpgid(int(entry)) == skip_pgid:
                continue
            with open(f"/proc/{entry}/statm") as f:
                rss = int(f.read().split()[1])
        except (OSError, ValueError):
            continue
        if rss > best:
            best, best_pid = rss, int(entry)
    return best_pid


class Judge:
    def __init__(self, binary, model, npu_dir):
        self.args = [binary, "judge", "--model", model, "--stdio"]
        libs = "/opt/xilinx/xrt/lib:/home/majed/npu-work/boostshim:/opt/rocm/lib:/opt/rocm/core-7.14/lib"
        self.env = dict(os.environ, GUFO_NPU_DIR=npu_dir, LD_LIBRARY_PATH=libs)
        self.proc = None
        self.lock = asyncio.Lock()

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(
            *self.args, env=self.env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        while True:
            line = await asyncio.wait_for(self.proc.stdout.readline(), 300)
            if not line:
                raise RuntimeError("judge exited while loading")
            if line.strip() == b"READY":
                return

    async def yes_probability(self, question):
        if self.proc is None or self.proc.returncode is not None:
            return None
        text = question.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", " ")
        async with self.lock:
            self.proc.stdin.write((text + "\t\n").encode())
            await self.proc.stdin.drain()
            while True:
                line = await asyncio.wait_for(self.proc.stdout.readline(), 20)
                parts = line.decode().split()
                if len(parts) == 4 and parts[0] in ("Yes", "No"):
                    return float(parts[1])
                if not line:
                    return None


class Guard:
    def __init__(self, a):
        self.a = a
        self.state = "down"
        self.ready = asyncio.Event()
        self.lock = asyncio.Lock()
        self.proc = None
        self.session = None
        self.judge = None
        self.watch_task = None
        self.stops = 0

    async def start_gufo(self):
        async with self.lock:
            if self.state == "up":
                return
            self.state = "starting"
            env = dict(os.environ, PORT=str(self.a.upstream_port), HOST="127.0.0.1")
            self.proc = await asyncio.create_subprocess_exec(
                "bash", self.a.run_script, env=env, start_new_session=True,
                stdin=asyncio.subprocess.DEVNULL)
            log(f"starting gufo pid={self.proc.pid} on :{self.a.upstream_port}")
            deadline = time.time() + self.a.start_timeout
            while time.time() < deadline:
                if self.proc.returncode is not None:
                    self.state = "down"
                    log(f"gufo exited during start rc={self.proc.returncode}")
                    return
                try:
                    async with self.session.get(self.a.upstream + "/v1/models") as r:
                        if r.status == 200:
                            self.state = "up"
                            self.ready.set()
                            log("gufo is up")
                            return
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(2)
            self.state = "down"
            log("gufo start timed out")

    async def stop_gufo(self, reason):
        async with self.lock:
            if self.proc is None or self.proc.returncode is not None:
                self.state = "down"
                self.ready.clear()
                return True
            self.state = "stopping"
            self.ready.clear()
            self.stops += 1
            log(f"stopping gufo ({reason})")
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self.proc.wait(), self.a.stop_timeout)
            except asyncio.TimeoutError:
                log("gufo did not exit; leaving it running (no SIGKILL on this box)")
                self.state = "up"
                self.ready.set()
                return False
            deadline = time.time() + 60
            while mem_available_gb() < self.a.free_target_gb and time.time() < deadline:
                await asyncio.sleep(1)
            self.state = "down"
            log(f"gufo stopped, {mem_available_gb():.0f} GB available")
            return True

    async def after_test(self, label, pids_fn):
        """Restart gufo once the test processes are gone for a quiet period."""
        started = time.time()
        seen = False
        quiet_since = None
        while time.time() - started < self.a.max_hold:
            alive = pids_fn()
            if alive:
                seen, quiet_since = True, None
            else:
                if quiet_since is None:
                    quiet_since = time.time()
                grace = self.a.quiet if seen else self.a.appear_grace
                if time.time() - quiet_since >= grace:
                    break
            await asyncio.sleep(2)
        log(f"test finished or timed out ({label}); restarting gufo")
        await self.start_gufo()

    def begin_watch(self, label, pids_fn):
        self.watch_task = asyncio.create_task(self.after_test(label, pids_fn))

    async def decide(self, commands):
        """Returns the heavy segment that needs gufo stopped, or None."""
        for cmd in commands:
            verdict, seg = classify(cmd)
            if verdict == "heavy":
                return seg
            if verdict == "unknown" and self.judge is not None:
                try:
                    p = await self.judge.yes_probability(JUDGE_PROMPT + seg)
                except (asyncio.TimeoutError, RuntimeError, OSError):
                    p = None
                log(f"judge on unknown command p_yes={p} cmd={seg[:80]!r}")
                if p is not None and p >= self.a.judge_cutoff:
                    return seg
        return None

    async def prepare_for_test(self, segment):
        if await self.stop_gufo(f"heavy command: {segment[:80]}"):
            needle = segment[:120]
            self.begin_watch(segment[:60], lambda: find_pids(needle))

    async def memory_watchdog(self):
        low = 0
        while True:
            await asyncio.sleep(2)
            if self.state != "up":
                low = 0
                continue
            low = low + 1 if mem_available_gb() < self.a.reactive_gb else 0
            if low >= 3:
                low = 0
                pid = biggest_foreign_rss(self.proc.pid if self.proc else -1)
                if await self.stop_gufo(f"memory pressure, top process pid={pid}"):
                    self.begin_watch(f"pid {pid}", lambda: [pid] if os.path.exists(f"/proc/{pid}") else [])

    async def wait_ready(self, resp):
        while not self.ready.is_set():
            if self.state == "down" and (self.watch_task is None or self.watch_task.done()):
                asyncio.create_task(self.start_gufo())
            try:
                await asyncio.wait_for(self.ready.wait(), 10)
            except asyncio.TimeoutError:
                if resp is not None:
                    await resp.write(b": jev-guard waiting for gufo\n\n")

    async def proxy(self, request):
        body = await request.read()
        try:
            stream = bool(json.loads(body or b"{}").get("stream"))
        except ValueError:
            stream = False
        resp = None
        if not self.ready.is_set():
            if stream:
                resp = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                                   "Cache-Control": "no-cache"})
                await resp.prepare(request)
            await self.wait_ready(resp)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        try:
            async with self.session.request(request.method, self.a.upstream + request.path_qs,
                                            data=body, headers=headers) as up:
                return await self.relay(request, up, resp)
        except aiohttp.ClientError as exc:
            log(f"upstream error: {exc}")
            if resp is not None:
                return resp
            return web.json_response({"error": {"message": f"gufo unavailable: {exc}"}}, status=503)

    async def relay(self, request, up, resp):
        ctype = up.headers.get("Content-Type", "")
        if "text/event-stream" not in ctype:
            data = await up.read()
            if up.status == 200 and request.path.endswith("/chat/completions"):
                try:
                    msg = json.loads(data)["choices"][0]["message"]
                    calls = {i: {"name": c["function"]["name"], "args": c["function"]["arguments"]}
                             for i, c in enumerate(msg.get("tool_calls") or [])}
                    seg = await self.decide(tool_commands(calls))
                    if seg:
                        await self.prepare_for_test(seg)
                except (ValueError, KeyError, IndexError, TypeError):
                    pass
            hdrs = {k: v for k, v in up.headers.items() if k.lower() not in HOP}
            return web.Response(body=data, status=up.status, headers=hdrs)
        if resp is None:
            hdrs = {k: v for k, v in up.headers.items() if k.lower() not in HOP}
            resp = web.StreamResponse(status=up.status, headers=hdrs)
            await resp.prepare(request)
        calls, held, holding, buf = {}, [], False, b""
        async for chunk in up.content.iter_any():
            buf += chunk
            while b"\n\n" in buf:
                event, buf = buf.split(b"\n\n", 1)
                event += b"\n\n"
                for line in event.splitlines():
                    if not line.startswith(b"data: ") or line.endswith(b"[DONE]"):
                        continue
                    try:
                        delta = json.loads(line[6:])["choices"][0]["delta"]
                    except (ValueError, KeyError, IndexError):
                        continue
                    for tc in delta.get("tool_calls") or []:
                        holding = True
                        slot = calls.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["args"] += fn.get("arguments") or ""
                if holding:
                    held.append(event)
                else:
                    await resp.write(event)
        if buf:
            held.append(buf)
        if holding:
            seg = await self.decide(tool_commands(calls))
            if seg:
                await self.prepare_for_test(seg)
        for event in held:
            await resp.write(event)
        await resp.write_eof()
        return resp

    async def status(self, request):
        return web.json_response({
            "state": self.state, "stops": self.stops,
            "mem_available_gb": round(mem_available_gb(), 1),
            "judge": self.judge is not None and self.judge.proc is not None
            and self.judge.proc.returncode is None})

    async def run(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        if self.a.judge_model:
            try:
                self.judge = Judge(self.a.gufo_bin, self.a.judge_model, self.a.npu_dir)
                await self.judge.start()
                log("judge sidecar ready")
            except Exception as exc:
                log(f"judge sidecar unavailable ({exc}); using patterns and memory watchdog only")
                self.judge = None
        asyncio.create_task(self.memory_watchdog())
        asyncio.create_task(self.start_gufo())
        app = web.Application(client_max_size=1 << 30)
        app.router.add_get("/jev-guard/status", self.status)
        app.router.add_route("*", "/{tail:.*}", self.proxy)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, self.a.host, self.a.port).start()
        log(f"listening on {self.a.host}:{self.a.port}, gufo on :{self.a.upstream_port}")
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        await stop.wait()
        await self.stop_gufo("guard exiting")
        await runner.cleanup()


def main():
    home = os.path.expanduser("~")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8900)
    p.add_argument("--upstream-port", type=int, default=8901)
    p.add_argument("--run-script", default=f"{home}/run-gufo.sh")
    p.add_argument("--gufo-bin", default=f"{home}/gufo/build/release/gufo")
    p.add_argument("--judge-model", default=f"{home}/jev-model/MiniCPM5-1B-SFT-Q6_K-fixed.gguf")
    p.add_argument("--npu-dir", default=f"{home}/gufo/tools/npu")
    p.add_argument("--judge-cutoff", type=float, default=0.85)
    p.add_argument("--free-target-gb", type=float, default=60)
    p.add_argument("--reactive-gb", type=float, default=2.5)
    p.add_argument("--quiet", type=float, default=8)
    p.add_argument("--appear-grace", type=float, default=30)
    p.add_argument("--max-hold", type=float, default=7200)
    p.add_argument("--start-timeout", type=float, default=420)
    p.add_argument("--stop-timeout", type=float, default=120)
    a = p.parse_args()
    a.upstream = f"http://127.0.0.1:{a.upstream_port}"
    asyncio.run(Guard(a).run())


if __name__ == "__main__":
    sys.exit(main())
