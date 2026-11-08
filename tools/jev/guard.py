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
import shlex
import signal
import sys
import time

import aiohttp
from aiohttp import web

HEAVY = [re.compile(p) for p in (
    r"\bgufo(\s+\S+)*?\s+(bench|serve|judge|diagnose|eval|transcribe|prompt)\b",
    r"build/\S+/gufo\b",
    r"\bctest\b",
    r"\bcmake\s+--build\b",
    r"\bnix\s+(build|develop)\b",
    r"\brocprof\w*",
    r"model-bench\.py",
    r"^(?:\S*/)?(?:python3?|bash|sh)\s+(?!-c\b)\S*tools/prof/prof\.py\s+run\b",
    r"^(?:\./)?tools/prof/prof\.py\s+run\b",
    r"^(?:\S*/)?(?:python3?|bash|sh)\s+(?!-c\b)\S*tools/bench/",
    r"^(?:\./)?tools/bench/",
    r"\bpytest\b",
    r"^(?:\S*/)?python3?\s+(?!-c\b)\S+.*\b(torch|hip|rocm)\b",
    r"run-gufo\.sh",
)]
LIGHT_WORDS = {
    "git", "ls", "cat", "rg", "grep", "sed", "head", "tail", "echo", "pwd", "cd",
    "find", "wc", "diff", "mkdir", "cp", "mv", "rm", "touch", "curl", "tr", "sort",
    "uniq", "awk", "stat", "file", "which", "date", "printf", "true", "less", "more",
    "tee", "xxd", "sha256sum", "md5sum", "sha1sum", "strings", "nm", "objdump", "readelf", "ldd", "nl", "cut", "basename", "dirname", "readlink", "realpath", "du",
    "df", "ps", "pgrep", "sleep", "wait", "kill", "tar", "unzip", "gh", "jj",
}
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
    out, cur, quote, i = [], [], "", 0
    while i < len(command):
        ch = command[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < len(command):
                cur.append(command[i:i + 2])
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch in ";|&\n":
            out.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    out.append("".join(cur))
    return [s.strip() for s in out if s.strip()]


def first_word(segment):
    for word in segment.split():
        if "=" in word and not word.startswith("-"):
            continue
        return os.path.basename(word)
    return ""


ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(\S*)$")
SUBST = re.compile(r"^[A-Za-z_]\w*=\$\((.*?)\)?$")
QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")
HOME = os.path.expanduser("~")


def expand(seg, env):
    for name, value in env.items():
        seg = seg.replace("${" + name + "}", value).replace("$" + name, value)
    return seg.replace("~/", HOME + "/")


LAUNCHER = re.compile(r"^(?:(?:nohup|setsid|time|nice|ionice|stdbuf|env|command|exec)(?:\s+-\S+)*\s+|[A-Za-z_]\w*=\S*\s+)+")


def strip_launchers(seg):
    m = LAUNCHER.match(seg.strip())
    if not m:
        return seg, False
    detached = bool(re.search(r"\b(nohup|setsid)\b", m.group(0)))
    return seg.strip()[m.end():], detached


def classify(command):
    """Returns (verdict, segment): heavy / light / unknown."""
    unknown = ""
    env = {}
    for seg in segments(command):
        seg = expand(seg, env)
        m = ASSIGN.match(seg.strip("\"'"))
        if m:
            env[m.group(1)] = m.group(2).strip("\"'")
            continue
        sub = SUBST.match(seg)
        if sub:
            verdict, inner = classify(sub.group(1))
            if verdict == "heavy":
                return verdict, inner
            continue
        seg, detached = strip_launchers(seg)
        if first_word(seg) in LIGHT_WORDS:
            continue
        if detached and seg and not re.search(r"\s(--help|-h|--version)(\s|$)", seg):
            return "heavy", seg
        if re.search(r"\s(--help|-h|--version)(\s|$)", seg):
            continue
        shell_c = first_word(seg) in ("bash", "sh", "eval", "xargs", "timeout", "nohup")
        if any(p.search(seg if shell_c else QUOTED.sub('""', seg)) for p in HEAVY):
            return "heavy", seg
        if not unknown:
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


SHELL_WORDS = {"do", "then", "else", "elif", "if", "while", "until", "time", "exec",
               "nohup", "{", "(", "!"}
REDIRECT = re.compile(r"\s\d*[<>]|\s&>")


def needle_for(segment):
    """What /proc/<pid>/cmdline shows: no env prefix, no redirects, no quotes."""
    seg = strip_launchers(REDIRECT.split(segment, 1)[0])[0]
    try:
        tokens = shlex.split(seg)
    except ValueError:
        tokens = seg.split()
    while tokens and (tokens[0] in SHELL_WORDS or
                      ("=" in tokens[0] and not tokens[0].startswith("-"))):
        tokens.pop(0)
    keep = []
    for token in tokens[:6]:
        if token.startswith("~") or "$" in token or "`" in token:
            break
        keep.append(token)
    return " ".join(keep) or segment[:120]


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

KFD = "/dev/kfd"
KFD_SAMPLES = 2
KFD_QUIET = 15
SCRIPT_RE = re.compile(r"(?:(?<=\s)|^)(~?/[^\s'\"<>|;&]+\.(?:py|sh))")
HEAVY_CONTENT = re.compile(
    r"import\s+torch|from\s+torch|\bhipMalloc|\bhipLaunch|libamdhip|\bcuda\b|rocm|"
    r"\bgufo\b|llama[._-]?cpp|\bvllm\b|safetensors|from_pretrained|\bcmake\s+--build|"
    r"\bninja\b|\bmake\s+-j|/dev/kfd|\bgguf\b", re.I)
SCRIPT_DEPTH = 2
SCRIPT_BYTES = 65536


def kfd_holders(skip_pgid, skip_pids):
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) in skip_pids:
            continue
        pid = int(entry)
        try:
            if os.getpgid(pid) == skip_pgid:
                continue
            for fd in os.listdir(f"/proc/{entry}/fd"):
                if os.readlink(f"/proc/{entry}/fd/{fd}") == KFD:
                    out.append(pid)
                    break
        except OSError:
            continue
    return out


INTERPRETERS = {"python", "python3", "bash", "sh", "zsh", "node"}


def executed_scripts(command):
    out = []
    for seg in segments(command):
        seg = strip_launchers(seg)[0]
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        if not tokens:
            continue
        head = os.path.basename(tokens[0]).rstrip("0123456789.")
        if head in INTERPRETERS:
            rest = tokens[1:]
            if rest and rest[0] == "-c":
                continue
            rest = [t for t in rest if not t.startswith("-")]
            target = rest[0] if rest else None
        else:
            target = tokens[0]
        if target and re.search(r"\.(py|sh)$", target) or (target and target.startswith(("./", "~/", "/"))):
            out.append(os.path.expanduser(target))
    return out


def inspect_scripts(command, depth=SCRIPT_DEPTH, seen=None):
    """Returns (heavy_reason, excerpt) from the script files a command executes."""
    seen = seen if seen is not None else set()
    excerpt = ""
    for path in executed_scripts(command):
        if path in seen:
            continue
        seen.add(path)
        try:
            with open(path, errors="replace") as f:
                text = f.read(SCRIPT_BYTES)
        except OSError:
            continue
        excerpt = excerpt or text[:1500]
        m = HEAVY_CONTENT.search(text)
        if m:
            return f"{os.path.basename(path)} mentions {m.group(0)!r}", excerpt
        if depth > 1:
            reason, _ = inspect_scripts(" ; ".join(SCRIPT_RE.findall(text)), depth - 1, seen)
            if reason:
                return f"{os.path.basename(path)} -> {reason}", excerpt
    return None, excerpt


MEM_RELEASE_QUIET = 30
MULTI_PASS_QUIET = 30


def biggest_foreign_rss(skip_pgid, skip_pids=()):
    best, best_pid = 0, 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) in skip_pids:
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
        self.inflight = 0
        self.starter = None
        self.last_fail = 0.0

    async def start_gufo(self):
        async with self.lock:
            if self.state == "up":
                return
            self.state = "starting"
            env = dict(os.environ, PORT=str(self.a.upstream_port), HOST="127.0.0.1", GUFO_GUARD="0")
            self.proc = await asyncio.create_subprocess_exec(
                "bash", self.a.run_script, env=env, start_new_session=True,
                stdin=asyncio.subprocess.DEVNULL)
            log(f"starting gufo pid={self.proc.pid} on :{self.a.upstream_port}")
            deadline = time.time() + self.a.start_timeout
            while time.time() < deadline:
                if self.proc.returncode is not None:
                    self.state = "down"
                    self.last_fail = time.time()
                    log(f"gufo exited during start rc={self.proc.returncode}")
                    return
                try:
                    async with self.session.get(self.a.upstream + "/v1/models") as r:
                        if r.status == 200:
                            self.state = "up"
                            self.ready.set()
                            log(f"gufo is up, start_took={time.time() - deadline + self.a.start_timeout:.1f}s")
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
            t_stop = time.time()
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
            log(f"gufo stopped, {mem_available_gb():.0f} GB available, stop_took={time.time() - t_stop:.1f}s")
            return True

    async def after_test(self, label, pids_fn, mem_release=False, quiet=None):
        """Restart gufo once the test processes are gone for a quiet period."""
        started = time.time()
        seen = False
        quiet_since = None
        free_since = None
        while time.time() - started < self.a.max_hold:
            alive = pids_fn()
            if mem_release:
                if mem_available_gb() >= self.a.free_target_gb:
                    free_since = free_since or time.time()
                    if time.time() - free_since >= MEM_RELEASE_QUIET:
                        log(f"memory recovered to {mem_available_gb():.0f} GB for {MEM_RELEASE_QUIET}s")
                        break
                else:
                    free_since = None
            if alive:
                seen, quiet_since = True, None
            else:
                if quiet_since is None:
                    quiet_since = time.time()
                grace = (quiet or self.a.quiet) if seen else self.a.appear_grace
                if time.time() - quiet_since >= grace:
                    break
            await asyncio.sleep(0.5)
        log(f"test finished or timed out ({label}) seen_process={seen} "
            f"watched={time.time() - started:.1f}s; restarting gufo")
        await self.start_gufo()

    def begin_watch(self, label, pids_fn, mem_release=False, quiet=None):
        self.watch_task = asyncio.create_task(self.after_test(label, pids_fn, mem_release, quiet))

    async def decide(self, commands):
        """Returns the heavy segment that needs gufo stopped, or None."""
        t0 = time.time()
        chosen = None
        for cmd in commands:
            verdict, seg = classify(cmd)
            p = None
            excerpt = ""
            if verdict == "unknown":
                reason, excerpt = inspect_scripts(cmd)
                if reason:
                    log(f"script inspection: {reason}")
                    verdict = "heavy"
            if verdict == "unknown" and self.judge is not None:
                try:
                    ctx = f"\nScript contents:\n{excerpt}" if excerpt else ""
                    p = await self.judge.yes_probability(JUDGE_PROMPT + seg + ctx)
                except (asyncio.TimeoutError, RuntimeError, OSError):
                    p = None
            by_judge = verdict == "unknown" and p is not None and p >= self.a.judge_cutoff
            log(f"decision pattern={verdict} judge_p_yes={p if p is None else round(p, 3)} "
                f"judge_would_stop={p is not None and p >= self.a.judge_cutoff} "
                f"acts={'stop' if verdict == 'heavy' or by_judge else 'none'} cmd={cmd[:100]!r}")
            if chosen is None and (verdict == "heavy" or by_judge):
                chosen = seg
        log(f"decision_ms={round((time.time() - t0) * 1000)}")
        return chosen

    async def prepare_for_test(self, segment):
        if await self.stop_gufo(f"heavy command: {segment[:80]}"):
            needle = needle_for(segment)
            log(f"watching processes matching {needle!r}")
            multi = bool(re.search(r"\.(py|sh)\b|&\s*$|\b(nohup|setsid)\b", segment))
            self.begin_watch(segment[:60], lambda: find_pids(needle), quiet=MULTI_PASS_QUIET if multi else None)

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
                own = {os.getpid()}
                if self.judge is not None and self.judge.proc is not None:
                    own.add(self.judge.proc.pid)
                pid = biggest_foreign_rss(self.proc.pid if self.proc else -1, own)
                if await self.stop_gufo(f"memory pressure, top process pid={pid}"):
                    self.begin_watch(f"pid {pid}", lambda: [pid] if os.path.exists(f"/proc/{pid}") else [], True)

    def own_pids(self):
        own = {os.getpid()}
        if self.judge is not None and self.judge.proc is not None:
            own.add(self.judge.proc.pid)
        return own

    def foreign_kfd(self):
        return kfd_holders(self.proc.pid if self.proc else -1, self.own_pids())

    async def kfd_watchdog(self):
        hits = 0
        while True:
            await asyncio.sleep(1)
            if self.state != "up":
                hits = 0
                continue
            found = await asyncio.get_running_loop().run_in_executor(None, self.foreign_kfd)
            hits = hits + 1 if found else 0
            if hits >= KFD_SAMPLES:
                hits = 0
                if await self.stop_gufo(f"foreign GPU process pid={found[0]}"):
                    self.begin_watch(f"kfd {found[0]}", self.foreign_kfd, quiet=KFD_QUIET)

    async def wait_ready(self, resp):
        t_wait = time.time()
        while not self.ready.is_set():
            idle = self.starter is None or self.starter.done()
            if (self.state == "down" and idle and time.time() - self.last_fail > 15
                    and (self.watch_task is None or self.watch_task.done())):
                self.starter = asyncio.create_task(self.start_gufo())
            try:
                await asyncio.wait_for(self.ready.wait(), 10)
            except asyncio.TimeoutError:
                if resp is not None:
                    await resp.write(b": jev-guard waiting for gufo\n\n")
        log(f"request held {time.time() - t_wait:.1f}s")

    async def proxy(self, request):
        self.inflight += 1
        try:
            return await self.proxy_inner(request)
        finally:
            self.inflight -= 1

    async def proxy_inner(self, request):
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
            t_hold = time.time()
            seg = await self.decide(tool_commands(calls))
            if seg:
                await self.prepare_for_test(seg)
            log(f"tool call held {time.time() - t_hold:.2f}s stopped={bool(seg)}")
        for event in held:
            await resp.write(event)
        await resp.write_eof()
        return resp

    async def status(self, request):
        return web.json_response({
            "state": self.state, "stops": self.stops, "inflight": self.inflight,
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
        asyncio.create_task(self.kfd_watchdog())
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
