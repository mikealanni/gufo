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
import hashlib
import json
import os
import queue
import re
import shlex
import signal
import sqlite3
import sys
import threading
import time
import uuid

import aiohttp
from aiohttp import web

HEAVY = [re.compile(p) for p in (
    r"\bgufo(\s+\S+)*?\s+(bench|serve|judge|diagnose|eval|transcribe|prompt)\b",
    r"build/\S+/gufo\b",
    r"\bctest\b",
    r"\brocprof\w*",
    r"model-bench\.py",
    r"^(?:\S*/)?(?:python3?|bash|sh)\s+(?!-c\b)\S*tools/prof/prof\.py\s+run\b",
    r"^(?:\./)?tools/prof/prof\.py\s+run\b",
    r"^(?:\S*/)?(?:python3?|bash|sh)\s+(?!-c\b)\S*tools/bench/",
    r"^(?:\./)?tools/bench/",
    r"\bpytest\b",
    r"^(?:\S*/)?python3?\s+(?!-c\b)\S+.*\b(torch|hip|rocm)\b",
    r"run-gufo\.sh",
    r"(?:^|\s)(?:\./)?build/[\w.-]+/(?:\S*/)?tests/\S*(?:_test|_tests|_bench|_probe)\b",
)]
LIGHT_SCRIPTS = {"chroma-helper.py"}
# Compiling uses CPU and RAM only, and swap covers the memory, so a build never
# stops gufo. Running the built tests still does (ctest and the test binaries).
BUILD_ONLY = re.compile(
    r"^(?:\S*/)?(?:cmake\s+--build|ninja|make|nix\s+build"
    r"|nix\s+develop\s+-c\s+(?:cmake\s+--build|ninja|make))\b")
LIGHT_WORDS = {
    "git", "ls", "cat", "rg", "grep", "sed", "head", "tail", "echo", "pwd", "cd",
    "find", "wc", "diff", "mkdir", "cp", "mv", "rm", "touch", "curl", "tr", "sort",
    "uniq", "awk", "stat", "file", "which", "date", "printf", "true", "less", "more",
    "tee", "xxd", "sha256sum", "md5sum", "sha1sum", "strings", "nm", "objdump", "readelf", "ldd", "nl", "cut", "basename", "dirname", "readlink", "realpath", "du",
    "df", "ps", "pgrep", "sleep", "wait", "kill", "tar", "unzip", "gh", "jj",
    "env", "printenv", "id", "whoami", "hostname", "uname", "nproc", "free", "uptime", "lsof", "ss",
    "read", "export", "test", "[", "[[", "local", "return", "exit", "break", "continue",
}
SHELL_KEYWORDS = ("do", "then", "else", "elif", "if", "while", "until", "!", "{", "(")
END_WORDS = {"done", "fi", "esac", "}", ")"}
LOOP_HEADER = re.compile(r"^(?:for\s+\w+\s+in\b|for\s+\(\()")


def strip_shell_keywords(seg):
    """Drop leading do/then/if/while/... so the command that follows is judged."""
    while True:
        word = seg.split(None, 1)[0] if seg.split() else ""
        if word in SHELL_KEYWORDS:
            seg = seg[len(word):].lstrip()
            continue
        return seg
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "upgrade",
       "proxy-authenticate", "proxy-authorization", "trailers", "content-length",
       "host", "content-encoding"}
SHELL_TOOLS = {"bash", "shell", "run_command", "execute_command"}
JUDGE_PROMPT = ("Does this shell command need the whole GPU or most of the RAM "
                "of the machine? Command: ")


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "[jev-guard]", msg, flush=True)


SECRET_KEYS = (
    "sk-", "ghp_", "gho_", "ghs_", "github_pat_", "AKIA", "xoxb-",
    "xoxp-", "AIza", "BEGIN PRIVATE KEY", "BEGIN OPENSSH PRIVATE KEY",
    "-----BEGIN", "Bearer eyJ", "eyJhbGciOiJ",
)


def contains_secret(text):
    return any(key in text for key in SECRET_KEYS)


# Same schema as src/cli/judge/jev_log.cpp; the two writers share one db file.
JEV_LOG_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY,
  ts INTEGER NOT NULL,
  model TEXT NOT NULL,
  battery TEXT,
  trigger TEXT NOT NULL,
  prompt_hash TEXT NOT NULL,
  score TEXT NOT NULL CHECK (score IN ('yes','no')),
  logit_gap REAL NOT NULL,
  duration_ms REAL NOT NULL,
  threshold TEXT,
  framing_ver TEXT,
  action TEXT,
  request_id TEXT,
  session_id TEXT,
  outcome TEXT CHECK (outcome IS NULL OR outcome IN
    ('confirmed','diverged','retry','rerun','harmless')),
  question TEXT
);
CREATE INDEX IF NOT EXISTS decisions_request ON decisions(request_id);
CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
  question, content='decisions', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS decisions_ai AFTER INSERT ON decisions BEGIN
  INSERT INTO decisions_fts(rowid, question)
    VALUES (new.id, new.question);
END;
CREATE TABLE IF NOT EXISTS corrections (
  id INTEGER PRIMARY KEY,
  decision_id INTEGER NOT NULL REFERENCES decisions(id),
  ts INTEGER NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('failure','correction','insight')),
  wrong_score TEXT,
  note TEXT
);
CREATE TABLE IF NOT EXISTS patterns (
  id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,
  pattern_hash TEXT NOT NULL,
  yes_rate REAL NOT NULL,
  hits INTEGER NOT NULL,
  last_ts INTEGER NOT NULL,
  UNIQUE(model, pattern_hash)
);
"""


AUTOEFFORT_OFF = os.path.expanduser("~/.gufo/jev-log/autoeffort-off")
EFFORT_LEVELS = {"minimal", "low", "medium", "high", "xhigh", "max"}
DEFAULT_PINNED_EFFORT = "low"
DRAIN_MAX = 1800
MIN_NEEDLE_CHARS = 5
WATCHDOG_DRAIN_MAX = 300
MEM_HARD_FLOOR_GB = 1.5


class GuardLog:
    """Queued best-effort writer: the event loop never waits on disk.

    A single thread owns the sqlite connection. A full queue drops the row
    (counted) instead of blocking the proxy; failures are logged only.
    """

    QUEUE_MAX = 2000

    def __init__(self, path):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.executescript(JEV_LOG_SCHEMA)
        conn.commit()
        conn.close()
        self.queue = queue.Queue(self.QUEUE_MAX)
        self.dropped = 0
        threading.Thread(target=self._writer, daemon=True, name="jev-log").start()

    def _put(self, item):
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _writer(self):
        conn = sqlite3.connect(self.path, timeout=5)
        conn.execute("PRAGMA foreign_keys=ON")
        while True:
            kind, args = self.queue.get()
            try:
                if kind == "decision":
                    conn.execute(
                        "INSERT INTO decisions(ts, model, battery, trigger, "
                        "prompt_hash, score, logit_gap, duration_ms, threshold, "
                        "framing_ver, action, request_id, session_id, question) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", args)
                else:
                    request_id, prompt_hash, outcome, note = args
                    conn.execute(
                        "UPDATE decisions SET outcome=? WHERE request_id=? "
                        "AND prompt_hash=? AND outcome IS NULL",
                        (outcome, request_id, prompt_hash))
                    if note:
                        conn.execute(
                            "INSERT INTO corrections(decision_id, ts, kind, wrong_score, note) "
                            "SELECT id, ?, 'failure', score, ? FROM decisions "
                            "WHERE request_id=? AND prompt_hash=?",
                            (int(time.time()), note, request_id, prompt_hash))
                conn.commit()
            except sqlite3.Error as exc:
                log(f"jev-log write failed: {exc}")

    def record(self, model, trigger, prompt_hash, score, logit_gap,
               duration_ms, threshold, framing_ver, action, request_id,
               session_id, question):
        self._put(("decision", (int(time.time()), model, "", trigger, prompt_hash,
                                score, logit_gap, duration_ms, threshold,
                                framing_ver, action, request_id, session_id,
                                question)))

    def set_outcome(self, request_id, prompt_hash, outcome, note=""):
        self._put(("outcome", (request_id, prompt_hash, outcome, note)))


LIGHT_TTL = 900
LEARNED_FILE = os.path.expanduser("~/.gufo/jev-log/learned-heavy.json")
LEARNED_MAX = 200
LEARNED_TTL = 30 * 86400
LEARN_SKIP = ("lemonade", "ollama", "llama-server", "xorg", "kwin", "gnome",
              "plasma", "pipewire", "claude", "opencode", "/usr/", "/bin/",
              "/sbin/", "/opt/rocm", "/proc/", "/sys/")
LEARN_SKIP_NAMES = {"gufo", "gufo-server", "guard.py", "run-gufo.sh", "run-gufo-jev.sh",
                    "run-jev-guard.sh", "jev-stop.sh", "jev-ctl.sh"}
GENERIC_TOKENS = {"python", "python3", "bash", "sh", "echo", "true", "false", "cat",
                  "grep", "head", "tail", "sleep", "time", "nohup", "setsid", "env"}
LEARNED = {}


def load_learned():
    try:
        with open(LEARNED_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return
    now = time.time()
    LEARNED.clear()
    LEARNED.update({k: v for k, v in data.items()
                    if isinstance(v, dict) and now - v.get("last", 0) < LEARNED_TTL})


def save_learned():
    try:
        os.makedirs(os.path.dirname(LEARNED_FILE), exist_ok=True)
        tmp = LEARNED_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(LEARNED, f, indent=1)
        os.replace(tmp, LEARNED_FILE)
    except OSError as exc:
        log(f"learned-heavy save failed: {exc}")


def proc_chain(pid, depth=4):
    """Command lines of pid and its parents, nearest first."""
    out = []
    for _ in range(depth):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                out.append(f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip())
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        if pid <= 1:
            break
    return out


def learn_key(cmdline, cwd=None):
    """A stable script or program path from a process that was seen using the GPU."""
    key = None
    for tok in cmdline.split():
        if re.search(r"\.(py|sh)$", tok) and not tok.startswith("-"):
            key = os.path.expanduser(tok)
            if not os.path.isabs(key) and cwd:
                key = os.path.normpath(os.path.join(cwd, key))
            break
    if key is None:
        first = cmdline.split(" ", 1)[0]
        if first.startswith(("/home/", "/mnt/", "/tmp/")):
            key = first
    if (key is None or not os.path.isabs(key) or len(key) < 8
            or os.path.basename(key) in LEARN_SKIP_NAMES
            or any(bad in key.lower() for bad in LEARN_SKIP)):
        return None
    return key


def learn_from_pid(pid):
    chain = proc_chain(pid, depth=3)
    key = None
    for cmd in chain:
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            cwd = None
        key = learn_key(cmd, cwd)
        if key:
            break
    if key is None:
        return None
    now = time.time()
    entry = LEARNED.setdefault(key, {"first": now, "hits": 0})
    entry.update(last=now, hits=entry["hits"] + 1, cmd=chain[0][:200])
    for old in sorted(LEARNED, key=lambda k: LEARNED[k]["last"])[:max(0, len(LEARNED) - LEARNED_MAX)]:
        del LEARNED[old]
    save_learned()
    return key


def light_matches(text, chain):
    """True when a process in the chain plausibly came from this light command."""
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    joined = " ".join(chain)
    for tok in tokens:
        tok = tok.strip("\"'")
        if len(tok) < 4 or tok in GENERIC_TOKENS or tok.startswith("-"):
            continue
        if tok in joined or os.path.basename(tok) in joined:
            return True
    return False


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576
    return 0.0


HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1[^\n]*\n(.*?)\n[ \t]*\2[ \t]*(?=\n|$)", re.S)
NL_MARK = "\x01"


def protect_heredocs(command):
    """Keep a here-document body inside the command that reads it."""
    return HEREDOC.sub(lambda m: m.group(0).replace("\n", NL_MARK), command)


def segments(command):
    command = protect_heredocs(command.replace("\\\n", " "))
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
        elif ch == "&" and ((cur and cur[-1] in "<>") or command[i + 1:i + 2] == ">"):
            pass  # 2>&1, >&2, &>file: a redirect, not a separator
        elif ch in ";|&\n":
            out.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    out.append("".join(cur))
    return [s.strip().replace(NL_MARK, "\n") for s in out if s.strip()]


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
    for key in LEARNED:
        if key in command:
            return "heavy", command
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
        seg = strip_shell_keywords(seg)
        if not seg or first_word(seg) in END_WORDS:
            continue
        if LOOP_HEADER.match(seg) and "$(" not in seg and "`" not in seg:
            continue
        if first_word(seg) in LIGHT_WORDS:
            continue
        if any(os.path.basename(x) in LIGHT_SCRIPTS for x in executed_scripts(seg)):
            continue
        if BUILD_ONLY.match(seg):
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
    r"(?<!\.)\bgufo\b|llama[._-]?cpp|\bvllm\b|safetensors|from_pretrained|\bcmake\s+--build|"
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
    cwd = None
    for seg in segments(command):
        seg = strip_launchers(seg)[0]
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        if not tokens:
            continue
        if tokens[0] == "cd":
            target_dir = next((t for t in tokens[1:] if not t.startswith("-")), None)
            cwd = os.path.expanduser(target_dir) if target_dir else None
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
            path = os.path.expanduser(target)
            if cwd and not os.path.isabs(path):
                path = os.path.normpath(os.path.join(cwd, path))
            out.append(path)
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
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    return best_pid, best * page / (1024 ** 3)


def foreign_rss_over_gb(skip_pgid, skip_pids, limit_gb):
    """Pid and RSS (GB) of a non-gufo process exceeding limit_gb, or (0, 0)."""
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) in skip_pids:
            continue
        try:
            pid = int(entry)
            if os.getpgid(pid) == skip_pgid:
                continue
            with open(f"/proc/{entry}/statm") as f:
                rss_gb = int(f.read().split()[1]) * page / (1024 ** 3)
        except (OSError, ValueError):
            continue
        if rss_gb > limit_gb:
            return pid, rss_gb
    return 0, 0.0


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
        self.jev_log = None
        self.pending_lights = []
        self.run_id = uuid.uuid4().hex[:12]
        self.last_activity = 0.0
        self.gen_inflight = 0
        self.heavy_waiting = 0
        self.idle_noted = False
        self.want_up = True  # the operator started the guard, which starts gufo once at boot
        self.guard_stopped = False  # True only while gufo is down because THIS guard stopped it
        # The model identity is fixed when the guard boots: every (re)start uses exactly this launch
        # environment, so a gufo the guard stopped always comes back as the same model.
        self.launch_env = {k: v for k, v in os.environ.items()
                           if k.startswith("GUFO_") or k in ("CTX", "JOBS", "KIND")}
        self.model_id = None  # id reported by gufo's /v1/models after the last successful start
        self.expected_model = None  # first model id seen; a restart must report the same id
        load_learned()
        if LEARNED:
            log(f"learned heavy keys: {len(LEARNED)}")
        if a.jev_log:
            try:
                self.jev_log = GuardLog(a.jev_log)
                log(f"jev-log enabled at {a.jev_log}")
            except Exception as exc:
                log(f"jev-log disabled ({exc})")
                self.jev_log = None

    def judge_label(self):
        stem = os.path.splitext(os.path.basename(self.a.judge_model or ""))[0]
        return stem.lower() or "guard-judge"

    def _record_decision(self, model, score, logit_gap, duration_ms, action,
                         request_id, session_id, segment, command):
        if self.jev_log is None:
            return
        raw = segment or command or ""
        text = raw[:500].replace("\n", " ").replace("\r", " ")
        if contains_secret(raw[:4000]):
            text = "[redacted-secret]"
        prompt_hash = hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()
        self.jev_log.record(
            model, "guard", prompt_hash, score, logit_gap, duration_ms,
            str(self.a.judge_cutoff), "guard-v2", action, request_id,
            session_id, text)
        if action == "none":
            self.pending_lights.append({"request_id": request_id, "hash": prompt_hash,
                                        "ts": time.time(), "text": raw[:500]})
            del self.pending_lights[:-100]

    def _attribute_catch(self, pid, note):
        """Blame a light decision only when the offending process matches it."""
        chain = proc_chain(pid)
        learned = learn_from_pid(pid)
        if learned:
            log(f"jev-log: learned heavy key {learned}")
        if self.jev_log is None:
            return
        now = time.time()
        self.pending_lights = [p for p in self.pending_lights if now - p["ts"] < LIGHT_TTL]
        for light in reversed(self.pending_lights):
            if light_matches(light["text"], chain):
                self.jev_log.set_outcome(light["request_id"], light["hash"], "diverged", note)
                self.pending_lights.remove(light)
                log(f"jev-log: light decision diverged ({note}) cmd={light['text'][:80]!r}")
                return
        log(f"jev-log: watchdog catch ({note}) not attributable to any recent light decision")

    async def start_gufo(self):
        async with self.lock:
            if self.state == "up":
                return
            self.state = "starting"
            env = dict(os.environ, **self.launch_env, PORT=str(self.a.upstream_port), HOST="127.0.0.1", GUFO_GUARD="0")
            self.proc = await asyncio.create_subprocess_exec(
                "bash", self.a.run_script, env=env, start_new_session=True,
                stdin=asyncio.subprocess.DEVNULL)
            log(f"starting gufo pid={self.proc.pid} on :{self.a.upstream_port} kind={self.launch_env.get('GUFO_KIND', 'flash-next')} expected_model={self.expected_model!r}")
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
                            try:
                                data = await r.json()
                                self.model_id = (data.get("data") or [{}])[0].get("id")
                            except Exception:
                                self.model_id = None
                            if self.expected_model is None:
                                self.expected_model = self.model_id
                            elif self.model_id != self.expected_model:
                                log(f"WARNING: gufo restarted as model {self.model_id!r}, expected {self.expected_model!r}")
                            self.state = "up"
                            self.guard_stopped = False
                            self.want_up = True
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
                if not self.guard_stopped:
                    # gufo was not running and we did not stop it: never bring it back on our own.
                    self.want_up = False
                return True
            self.state = "stopping"
            self.ready.clear()
            self.stops += 1
            t_stop = time.time()
            log(f"stopping gufo model={self.model_id!r} kind={self.launch_env.get('GUFO_KIND', 'flash-next')} ({reason})")
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
            self.guard_stopped = True
            self.want_up = True
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

    async def decide(self, commands, session_id=""):
        """Returns the heavy segment that needs gufo stopped, or None.

        Writes one jev-log row per command in the batch. Lights (acted as
        none) stay pending until a watchdog either blames one or the
        operator labels it.
        """
        t0 = time.time()
        request_id = uuid.uuid4().hex
        chosen = None
        for cmd in commands:
            cmd_t0 = time.time()
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
            heavy = verdict == "heavy" or by_judge
            log(f"decision pattern={verdict} judge_p_yes={p if p is None else round(p, 3)} "
                f"judge_would_stop={p is not None and p >= self.a.judge_cutoff} "
                f"acts={'stop' if heavy else 'none'} cmd={cmd[:100]!r}")
            if heavy:
                score = "yes"
                logit_gap = 1.0 if verdict == "heavy" else 2 * p - 1
            elif verdict == "light":
                score = "no"
                logit_gap = -1.0
            elif p is not None:
                score = "no"
                logit_gap = 2 * p - 1
            else:
                score = "no"
                logit_gap = 0.0
            self._record_decision(
                self.judge_label() if p is not None else "guard-patterns",
                score, logit_gap, (time.time() - cmd_t0) * 1000,
                "stop" if heavy else "none", request_id,
                session_id or self.run_id, seg, cmd)
            if chosen is None and heavy:
                chosen = seg
        log(f"decision_ms={round((time.time() - t0) * 1000)}")
        return chosen

    async def drain_for_watchdog(self, reason):
        t0 = time.time()
        logged = False
        while self.gen_inflight > 0 and time.time() - t0 < WATCHDOG_DRAIN_MAX:
            if mem_available_gb() < MEM_HARD_FLOOR_GB:
                log(f"memory below {MEM_HARD_FLOOR_GB} GB; not waiting for generations")
                return
            if not logged:
                log(f"waiting for in-flight generations before stopping ({reason})")
                logged = True
            await asyncio.sleep(0.5)
        if logged:
            log(f"generations done after {time.time() - t0:.1f}s")

    async def drain_generations(self, segment, own=True):
        self.heavy_waiting += 1 if own else 0
        t0 = time.time()
        logged = False
        try:
            while self.gen_inflight - self.heavy_waiting > 0 and time.time() - t0 < DRAIN_MAX:
                if not logged:
                    log(f"heavy command waits for other generations: {segment[:60]}")
                    logged = True
                await asyncio.sleep(0.5)
        finally:
            self.heavy_waiting -= 1 if own else 0
        if logged:
            log(f"other generations done after {time.time() - t0:.1f}s")

    async def prepare_for_test(self, segment, own=True):
        await self.drain_generations(segment, own)
        if await self.stop_gufo(f"heavy command: {segment[:80]}"):
            needle = needle_for(segment)
            log(f"watching processes matching {needle!r}")
            multi = bool(re.search(r"\.(py|sh)\b|&\s*$|\b(nohup|setsid)\b", segment))
            self.begin_watch(segment[:60], lambda: find_pids(needle), quiet=MULTI_PASS_QUIET if multi else None)

    async def memory_watchdog(self):
        low = 0
        rss_hits = 0
        while True:
            await asyncio.sleep(2)
            if self.state != "up":
                low = 0
                rss_hits = 0
                continue
            avail = mem_available_gb()
            low = low + 1 if avail < self.a.reactive_gb else 0
            own = {os.getpid()}
            if self.judge is not None and self.judge.proc is not None:
                own.add(self.judge.proc.pid)
            big_pid, big_rss = foreign_rss_over_gb(self.proc.pid if self.proc else -1, own, self.a.foreign_rss_gb)
            rss_hits = rss_hits + 1 if big_pid else 0
            reason = None
            if low >= 3:
                if not big_pid:
                    big_pid, big_rss = biggest_foreign_rss(self.proc.pid if self.proc else -1, own)
                reason = f"memory pressure avail={avail:.1f}GB (<{self.a.reactive_gb}GB), top pid={big_pid} rss={big_rss:.1f}GB"
            elif rss_hits >= 2:
                reason = f"foreign process rss pressure pid={big_pid} rss={big_rss:.1f}GB (>{self.a.foreign_rss_gb}GB)"
            if reason:
                low = 0
                rss_hits = 0
                await self.drain_for_watchdog(reason)
                if await self.stop_gufo(reason):
                    self._attribute_catch(big_pid, f"memory watchdog pid={big_pid}")
                    self.begin_watch(f"pid {big_pid}", lambda pid=big_pid: [pid] if os.path.exists(f"/proc/{pid}") else [], True)

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
            if not found:
                self.idle_noted = False
            if hits >= KFD_SAMPLES and not self.opencode_active():
                hits = 0
                if not self.idle_noted:
                    self.idle_noted = True
                    since = (f"for {time.time() - self.last_activity:.0f}s" if self.last_activity
                             else "since the guard started")
                    log(f"foreign GPU process pid={found[0]} seen but opencode has not used gufo "
                        f"{since}; leaving gufo running")
                continue
            if hits >= KFD_SAMPLES:
                hits = 0
                await self.drain_for_watchdog(f"foreign GPU process pid={found[0]}")
                if await self.stop_gufo(f"foreign GPU process pid={found[0]}"):
                    self._attribute_catch(found[0], f"kfd watchdog pid={found[0]}")
                    self.begin_watch(f"kfd {found[0]}", self.foreign_kfd, quiet=KFD_QUIET)

    def opencode_active(self):
        """True while a generation request is in flight or one finished recently."""
        return self.gen_inflight > 0 or time.time() - self.last_activity < self.a.active_window

    def session_key(self, request):
        for name in ("x-session-id", "x-opencode-session", "x-session-affinity"):
            value = request.headers.get(name)
            if value:
                return value[:64]
        return self.run_id

    async def supervisor(self):
        """Notice a dead gufo and bring it back, but only when it is safe.

        Safe means: no heavy-test watch is running, nothing foreign holds the
        GPU, memory has recovered. It never starts anything on the GPU
        beside a test, and never restarts the NPU judge while gufo serves.
        """
        while True:
            await asyncio.sleep(3)
            if self.state == "up" and self.proc is not None and self.proc.returncode is not None:
                log(f"gufo exited unexpectedly rc={self.proc.returncode}")
                self.state = "down"
                self.ready.clear()
                self.last_fail = time.time()
            idle = self.starter is None or self.starter.done()
            watching = self.watch_task is not None and not self.watch_task.done()
            if not (self.want_up and self.state == "down" and idle and not watching):
                continue
            if time.time() - self.last_fail < 15 or mem_available_gb() < self.a.free_target_gb:
                continue
            if await asyncio.get_running_loop().run_in_executor(None, self.foreign_kfd):
                continue
            log("supervisor: gufo is down with nothing blocking it; starting")
            self.starter = asyncio.create_task(self.start_gufo())

    async def wait_ready(self, resp):
        t_wait = time.time()
        while not self.ready.is_set():
            if not self.want_up and self.state == "down":
                log("gufo is not running and the guard did not stop it; not starting it for a request")
                return
            idle = self.starter is None or self.starter.done()
            if (self.want_up and self.state == "down" and idle and time.time() - self.last_fail > 15
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
        generating = request.method == "POST" and request.path.endswith(
            ("/chat/completions", "/completions", "/responses", "/messages"))
        if generating:
            self.gen_inflight += 1
            self.last_activity = time.time()
        try:
            return await self.proxy_inner(request)
        finally:
            self.inflight -= 1
            if generating:
                self.gen_inflight -= 1
                self.last_activity = time.time()

    def pinned_effort(self):
        try:
            with open(AUTOEFFORT_OFF) as fh:
                level = fh.read().strip().lower()
        except OSError:
            return None
        return level if level in EFFORT_LEVELS else DEFAULT_PINNED_EFFORT

    def pin_effort(self, request, body):
        if request.method != "POST" or not request.path.endswith("/chat/completions"):
            return body
        level = self.pinned_effort()
        if level is None:
            return body
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            return body
        kwargs = data.get("chat_template_kwargs")
        if not isinstance(data, dict) or "reasoning_effort" in data or "thinking" in data or (
                isinstance(kwargs, dict) and ("reasoning_effort" in kwargs or "enable_thinking" in kwargs)):
            return body
        data["reasoning_effort"] = level
        return json.dumps(data).encode()

    async def proxy_inner(self, request):
        body = self.pin_effort(request, await request.read())
        if request.method == "POST" and request.path.endswith("/chat/completions"):
            try:
                data = json.loads(body or b"{}")
                kw = data.get("chat_template_kwargs")
                log("request keys: effort=%s thinking=%s budget=%s kwargs=%s plugin=%s session=%s" % (
                    data.get("reasoning_effort"), data.get("thinking"),
                    data.get("reasoning_budget_tokens", data.get("thinking_budget_tokens")),
                    sorted(kw) if isinstance(kw, dict) else None,
                    request.headers.get("x-jev-plugin"), (request.headers.get("x-session-id") or "")[:16]))
            except (ValueError, AttributeError):
                pass
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
                    seg = None if self.plugin_owns_gate(request) else await self.decide(tool_commands(calls), self.session_key(request))
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
            seg = None if self.plugin_owns_gate(request) else await self.decide(tool_commands(calls), self.session_key(request))
            if seg:
                await self.prepare_for_test(seg)
            log(f"tool call held {time.time() - t_hold:.2f}s stopped={bool(seg)}")
        for event in held:
            await resp.write(event)
        await resp.write_eof()
        return resp

    def plugin_owns_gate(self, request):
        if request.headers.get("x-jev-plugin") != "1":
            return False
        return not os.path.exists(os.path.expanduser("~/.gufo/jev-log/feature-off.command"))

    async def classify_command(self, request):
        try:
            data = await request.json()
            command = str(data.get("command", ""))[:4000]
        except ValueError:
            return web.json_response({"error": "bad json"}, status=400)
        seg = await self.decide([command], str(data.get("session", ""))[:64])
        return web.json_response({"heavy": bool(seg), "segment": seg or ""})

    async def prepare(self, request):
        try:
            data = await request.json()
            segment = str(data.get("segment", ""))[:300]
        except ValueError:
            return web.json_response({"error": "bad json"}, status=400)
        if not segment:
            return web.json_response({"error": "segment required"}, status=400)
        if len(needle_for(segment)) < MIN_NEEDLE_CHARS:
            log(f"plugin asks to prepare for {segment[:60]!r}, but nothing specific can be watched; leaving gufo running")
            return web.json_response({"state": self.state, "skipped": "generic", "mem_available_gb": round(mem_available_gb(), 1)})
        log(f"plugin asks to prepare for: {segment[:80]} (session {str(data.get('session', ''))[:24]})")
        await self.prepare_for_test(segment, own=False)
        return web.json_response({"state": self.state, "mem_available_gb": round(mem_available_gb(), 1)})

    async def status(self, request):
        return web.json_response({
            "state": self.state, "stops": self.stops, "inflight": self.inflight,
            "opencode_active": self.opencode_active(), "prepare_api": 1, "default_effort": os.environ.get("GUFO_EFFORT", "low"), "pinned_effort": self.pinned_effort(),
            "idle_s": round(time.time() - self.last_activity) if self.last_activity else None,
            "mem_available_gb": round(mem_available_gb(), 1),
            "model": self.model_id, "kind": self.launch_env.get("GUFO_KIND", "flash-next"),
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
        asyncio.create_task(self.supervisor())
        asyncio.create_task(self.start_gufo())
        app = web.Application(client_max_size=1 << 30)
        app.router.add_get("/jev-guard/status", self.status)
        app.router.add_post("/jev-guard/prepare", self.prepare)
        app.router.add_post("/jev-guard/classify", self.classify_command)
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
        self.want_up = False
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
    p.add_argument("--reactive-gb", type=float, default=15.0,
                   help="stop gufo when available memory stays below this for ~6s")
    p.add_argument("--foreign-rss-gb", type=float, default=20.0,
                   help="stop gufo when any non-gufo process RSS exceeds this")
    p.add_argument("--quiet", type=float, default=8)
    p.add_argument("--appear-grace", type=float, default=30)
    p.add_argument("--max-hold", type=float, default=7200)
    p.add_argument("--start-timeout", type=float, default=420)
    p.add_argument("--stop-timeout", type=float, default=120)
    p.add_argument("--active-window", type=float, default=600,
                   help="seconds after the last generation request that opencode counts as using gufo")
    p.add_argument("--jev-log", default=f"{home}/.gufo/jev-log/jev.sqlite3",
                   help="jev-log db file; set empty to disable")
    a = p.parse_args()
    a.upstream = f"http://127.0.0.1:{a.upstream_port}"
    asyncio.run(Guard(a).run())


if __name__ == "__main__":
    sys.exit(main())
