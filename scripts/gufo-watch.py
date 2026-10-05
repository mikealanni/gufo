#!/usr/bin/env python3
"""Live gufo dashboard: server speeds, current request progress, recent requests, memory.

Usage: gufo-watch.py [--once] [--port 8900] [--log ~/.cache/gufo/gufo.log] [--rows 12]
Per-request details come from the server log (scripts/run-gufo.sh writes ~/.cache/gufo/gufo.log).
"""
import argparse
import os
import re
import sys
import time
import urllib.request
from collections import OrderedDict, deque

KV = re.compile(r"(\w+)=(\S+)")
DIM, BOLD, GREEN, YELLOW, RED, CYAN, RESET = (
    "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m")
GTT = "/sys/class/drm/card0/device/mem_info_gtt_used"


def num(fields, key, default=0.0):
    try:
        return float(fields.get(key, default))
    except ValueError:
        return default


class LogTail:
    def __init__(self, path, history_bytes=3_000_000):
        self.path, self.fh, self.inode = path, None, None
        self.history_bytes = history_bytes
        self.done = deque(maxlen=200)
        self.active = OrderedDict()

    def _open(self):
        try:
            st = os.stat(self.path)
        except OSError:
            self.fh = None
            return
        self.fh = open(self.path, "r", errors="replace")
        self.inode = st.st_ino
        self.fh.seek(max(0, st.st_size - self.history_bytes))
        if self.fh.tell():
            self.fh.readline()

    def poll(self):
        try:
            st = os.stat(self.path)
            if self.fh is None or st.st_ino != self.inode or st.st_size < self.fh.tell():
                self._open()
        except OSError:
            self.fh = None
        if self.fh is None:
            return
        for line in self.fh:
            self.feed(line)

    def feed(self, line):
        if "request=" not in line:
            return
        f = dict(KV.findall(line))
        rid = f.get("request")
        try:
            f["_ts"] = time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            f["_ts"] = time.time()
        if "[progress]" in line and rid:
            # progress ids are scheduler numbers; completions use HTTP ids (r123)
            self.active[rid] = f
            while len(self.active) > 16:
                self.active.popitem(last=False)
        elif f.get("event") == "completed" and "completions" in f.get("path", ""):
            self.active.pop(rid, None)
            gen = f.get("generated_tokens")
            for k, v in list(self.active.items()):
                if v.get("phase") == "decode" and v.get("tokens", "").split("/")[0] == gen:
                    del self.active[k]
                    break
            if f.get("status") == "200":
                f["_t"] = line[11:19] if len(line) > 19 else ""
                self.done.append(f)
        elif f.get("event") == "received" and rid:
            self.active.setdefault(rid, {"request": rid, "phase": "queued", "_ts": f["_ts"]})

    def live(self):
        """In-flight entries: progress seen in the last 30 s, queued within 15 min."""
        now = time.time()
        return [(k, v) for k, v in self.active.items()
                if now - v.get("_ts", 0) <= (900 if v.get("phase") == "queued" else 30)]


def metrics(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as r:
            text = r.read().decode()
    except Exception:
        return None
    out = {}
    for m in re.finditer(r"^llamacpp:(\w+) ([0-9.eE+-]+)$", text, re.M):
        out[m.group(1)] = float(m.group(2))
    return out


def system():
    mem = {}
    with open("/proc/meminfo") as fh:
        for ln in fh:
            k, v = ln.split(":")
            mem[k] = int(v.split()[0]) / 1048576
    try:
        with open(GTT) as fh:
            gtt = int(fh.read()) / 1073741824
        with open(GTT.replace("used", "total")) as fh:
            mem["GTT_TOTAL"] = int(fh.read()) / 1073741824
    except OSError:
        gtt = 0.0
    stuck = sum(1 for p in os.listdir("/proc") if p.isdigit() and _state(p) == "D")
    return mem, gtt, stuck, os.getloadavg()[0]


def _state(pid):
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0]
    except OSError:
        return ""


def gtt_color(gtt, mem):
    total = mem.get("GTT_TOTAL", 0) or 1
    return RED if gtt / total > 0.95 else YELLOW if gtt / total > 0.85 else GREEN


def color_avail(gb):
    return RED if gb < 2 else YELLOW if gb < 6 else GREEN


def summarize(done):
    pp_tok = pp_sec = tg_tok = tg_sec = 0.0
    pp_all, tg_all, acc_all = [], [], []
    for f in done:
        n, tps = num(f, "prefill_tokens"), num(f, "prefill_tps")
        if n > 0 and tps > 0:
            pp_tok += n
            pp_sec += n / tps
            pp_all.append(tps)
        g, dps = num(f, "generated_tokens"), num(f, "decode_tps")
        if g > 0 and dps > 0:
            tg_tok += g
            tg_sec += g / dps
            tg_all.append(dps)
        acc = f.get("acceptance_pct")
        if acc:
            try:
                acc_all.append(float(acc))
            except ValueError:
                pass
    mean = lambda v: sum(v) / len(v) if v else 0.0
    return (pp_tok / pp_sec if pp_sec else 0.0, mean(pp_all),
            tg_tok / tg_sec if tg_sec else 0.0, mean(tg_all), mean(acc_all), len(done))


HIT_FRAC = 0.80   # share of the prompt served from cache to count as a hit
MISS_FRAC = 0.10  # below this the cache did not really help


def cache_stats(done):
    hit = part = miss = 0
    for f in done:
        prompt = num(f, "prompt_tokens")
        frac = num(f, "cached_tokens") / prompt if prompt else 0.0
        if frac >= HIT_FRAC:
            hit += 1
        elif frac >= MISS_FRAC:
            part += 1
        else:
            miss += 1
    prompt = sum(num(f, "prompt_tokens") for f in done)
    cached = sum(num(f, "cached_tokens") for f in done)
    return hit, part, miss, (100.0 * cached / prompt if prompt else 0.0)


def summary_lines(label, done):
    pp_agg, pp_avg, tg_agg, tg_avg, acc, n = summarize(done)
    hit, part, miss, tok_hit = cache_stats(done)
    return [
        f"{BOLD}{label}{RESET} {DIM}n={n}{RESET}",
        f"  PP agg {CYAN}{pp_agg:5.0f}{RESET} avg {CYAN}{pp_avg:5.0f}{RESET}"
        f"  TG agg {CYAN}{tg_agg:4.1f}{RESET} avg {CYAN}{tg_avg:4.1f}{RESET}",
        f"  accept {CYAN}{acc:4.1f}%{RESET}  cache {tok_hit:4.1f}% tok",
        f"  hit {GREEN}{hit}{RESET} part {YELLOW if part else GREEN}{part}{RESET}"
        f" miss {RED if miss else GREEN}{miss}{RESET}",
    ]


def render(tail, port, rows):
    m = metrics(port)
    mem, gtt, stuck, load = system()
    lines = [f"{BOLD}gufo{RESET}  {DIM}{time.strftime('%H:%M:%S')}  port {port}{RESET}"]
    if m is None:
        lines.append(f"{RED}server not answering on :{port}{RESET}")
    else:
        lines.append(
            f"last request   PP {CYAN}{m.get('prompt_tokens_seconds', 0):7.0f}{RESET} tok/s"
            f"   TG {CYAN}{m.get('predicted_tokens_seconds', 0):6.1f}{RESET} tok/s"
            f"   {DIM}totals: {m.get('prompt_tokens_total', 0):,.0f} in / "
            f"{m.get('tokens_predicted_total', 0):,.0f} out{RESET}")
    avail = mem["MemAvailable"]
    swap_used = mem["SwapTotal"] - mem["SwapFree"]
    lines.append(
        f"memory         avail {color_avail(avail)}{avail:5.1f} GB{RESET}"
        f"   swap {swap_used:5.1f} GB   GPU pinned {gtt_color(gtt, mem)}{gtt:5.1f}/{mem.get('GTT_TOTAL', 0):.0f} GB{RESET}"
        f"   load {load:5.1f}   stuck tasks {RED if stuck else GREEN}{stuck}{RESET}")
    lines.append("")
    live_all = tail.live()
    open_reqs = sum(1 for _, v in live_all if v.get("phase") == "queued")
    live = [(k, v) for k, v in live_all if v.get("phase") != "queued"]
    waiting = max(0, open_reqs - len(live))
    lines.append(f"requests       open {open_reqs}   running {len(live)}"
                 f"   waiting for a session {YELLOW if waiting else GREEN}{waiting}{RESET}")
    lines.append("")
    if live:
        lines.append(f"{BOLD}running{RESET}")
        for rid, f in live:
            ph = f.get("phase", "?")
            lines.append(
                f"  {rid:>5}  {ph:<8} {f.get('tokens', ''):>14}  {f.get('percentage', ''):>6}%"
                f"  now {num(f, 'chunk_tps'):6.0f}  avg {num(f, 'avg_tps'):6.0f} tok/s"
                + (f"  accept {num(f, 'acceptance_percentage'):.0f}%"
                   if "acceptance_percentage" in f else ""))
        lines.append("")
    lines.append(f"{BOLD}recent requests{RESET}")
    lines.append(f"{DIM}  time      in(tok) cached  PP tok/s   out(tok)  TG tok/s  accept  TTFT s  total s{RESET}")
    for f in list(tail.done)[-rows:]:
        acc = f.get("acceptance_pct")
        lines.append(
            f"  {f.get('_t', ''):<8} {num(f, 'prompt_tokens'):8,.0f} {num(f, 'cached_tokens'):6,.0f}"
            f" {num(f, 'prefill_tps'):9.0f} {num(f, 'generated_tokens'):10,.0f}"
            f" {num(f, 'decode_tps'):9.1f} {(acc[:5] + '%') if acc else '     -':>7}"
            f" {num(f, 'ttft_ms') / 1000:7.1f} {num(f, 'duration_ms') / 1000:8.1f}")
    if not tail.done:
        lines.append(f"{DIM}  (no completed requests in {tail.path} yet){RESET}")
    else:
        lines.append("")
        lines.extend(summary_lines(f"last {min(rows, len(tail.done))}", list(tail.done)[-rows:]))
        lines.extend(summary_lines("all seen", list(tail.done)))
        lines.append(f"{DIM}  cache: hit >=80%, part 10-80%, miss <10%{RESET}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--log", default=os.path.expanduser("~/.cache/gufo/gufo.log"))
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    tail = LogTail(a.log)
    try:
        while True:
            tail.poll()
            out = render(tail, a.port, a.rows)
            if a.once:
                print(out)
                return
            sys.stdout.write("\033[H\033[J" + out + "\n")
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
