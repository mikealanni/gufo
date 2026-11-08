#!/usr/bin/env python3
"""Rank DFlash2 serving configs for Qwen3.8 Flash-Next by acceptance and tok/s.

Runs `gufo bench --speculative dflash2 -v` over a grid of drafter files,
--draft-tokens and --draft-policy values, parses the per-repetition
`Qwen3.8-Flash-Next tg ... drafted= accepted= ... output_sha256=` trace on stderr
plus the tg throughput row from the stdout table, compares greedy output
output_sha256 against the no-drafter baseline, and prints a ranked table plus the
winning env line. Every bench process exits before the next one starts, so peak
memory is one model image.
"""
from __future__ import annotations

import argparse
import itertools
import os
import re
import subprocess

# Per-repetition verbose trace, stderr:
#   Qwen3.8-Flash-Next tg depth=0 cycles=245 drafted=1400 accepted=712 widths=... \
#       output_sha256=<hex> text=...
DETAIL = re.compile(
    r"Qwen3\.8-Flash-Next tg depth=(?P<depth>\d+) cycles=(?P<cycles>\d+) "
    r"drafted=(?P<drafted>\d+) accepted=(?P<accepted>\d+)"
    r".*?output_sha256=(?P<sha>[0-9a-f]+)")
# stdout table row: | model | size | ROCm (HIP) | tg 256/0 | 21.34 ± 0.10 |
TABLE_ROW = re.compile(r"\|\s*ROCm \(HIP\)\s*\|\s*(?P<test>[^|]*?)\s*\|"
                       r"\s*(?P<mean>[\d.]+)\s*±\s*(?P<std>[\d.]+)\s*\|")


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def run_bench(binary, model, dflash, prompt_tokens, gen_tokens, depth,
              draft_tokens, policy, repetitions, vocab):
    env = dict(os.environ)
    if vocab:
        env["GUFO_DFLASH_VOCAB"] = str(vocab)
    command = [
        str(binary), "bench", "-m", str(model),
        "-p", str(prompt_tokens), "-n", str(gen_tokens), "-d", str(depth),
        "-r", str(repetitions), "-v",
    ]
    if dflash:
        command += ["--speculative", "dflash2", "--dflash-model", str(dflash),
                    "--draft-tokens", str(draft_tokens),
                    "--draft-policy", policy]
    print(f"# run p={prompt_tokens} d={depth} drafter={dflash or 'no-draft'} "
          f"dt={draft_tokens} policy={policy} vocab={vocab or 'default'}",
          flush=True)
    result = subprocess.run(command, env=env, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise SystemExit(f"bench failed ({result.returncode}):\n"
                         f"{result.stderr[-2000:]}")

    drafted = accepted = 0
    per_rep = []
    shas = set()
    for line in result.stderr.splitlines():
        match = DETAIL.search(line)
        if match:
            d = int(match["drafted"])
            a = int(match["accepted"])
            drafted += d
            accepted += a
            if d:
                per_rep.append(a / d)
            shas.add(match["sha"])
    throughput = throughput_std = None
    for line in result.stdout.splitlines():
        row = TABLE_ROW.search(line)
        if row and row["test"].strip().startswith("tg"):
            throughput = float(row["mean"])
            throughput_std = float(row["std"])

    if not shas:
        raise SystemExit(f"no tg trace line parsed (need -v output);\n"
                         f"stderr tail:\n{result.stderr[-2000:]}")
    if len(shas) > 1:
        print(f"WARN: {len(shas)} distinct output_sha256 across repetitions "
              f"(non-deterministic generation); greedy comparison unreliable")
    sha = next(iter(shas))
    if drafted:
        acceptance, acceptance_std = _mean_std(per_rep)
    else:
        acceptance = acceptance_std = None
    return (acceptance, acceptance_std, throughput, throughput_std, sha,
            drafted, accepted)


def busy() -> str | None:
    import shutil
    try:
        running = subprocess.run(["pgrep", "-x", "gufo"],
                                 capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        running = None
    if running and running.returncode == 0:
        return "a gufo model process is running; it holds the GPU model"
    if shutil.which("free"):
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    if available < 40 * 1024**3:
                        return (f"only {available / 1e9:.0f} GiB available; a model "
                                f"load needs a free machine")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", required=True, help="gufo binary")
    parser.add_argument("--model", required=True, help="target first shard")
    parser.add_argument("--dflash", action="append", default=[],
                        help="drafter GGUF; repeat for alternatives")
    parser.add_argument("--draft-tokens", type=int, nargs="*", default=[7])
    parser.add_argument("--policy", nargs="*", default=["adaptive"],
                        choices=["fixed", "adaptive"])
    parser.add_argument("--vocab", type=int, nargs="*", default=[0],
                        help="GUFO_DFLASH_VOCAB row counts; 0 = builder default")
    parser.add_argument("-p", "--prompt-tokens", type=int, default=2048)
    parser.add_argument("-n", "--gen-tokens", type=int, default=256)
    parser.add_argument("-d", "--depth", type=int, default=0)
    parser.add_argument("-r", "--repetitions", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="run even if the machine looks busy")
    args = parser.parse_args()

    if not (args.dry_run or args.force):
        blocked = busy()
        if blocked:
            raise SystemExit(f"refusing to load a model: {blocked}. "
                             "Stop it first (or pass --force).")

    baseline_sha = baseline_throughput = None
    if not args.dry_run:
        _, _, baseline_throughput, _, baseline_sha, _, _ = run_bench(
            args.bin, args.model, None, args.prompt_tokens, args.gen_tokens,
            args.depth, 0, "", args.repetitions, 0)
        baseline_sha = str(baseline_sha)
        print(f"baseline (no draft): {baseline_throughput:.2f} tok/s "
              f"sha={baseline_sha[:12]}", flush=True)

    verdicts = []
    for drafter, tokens, policy, vocab in itertools.product(
            args.dflash, args.draft_tokens, args.policy, args.vocab):
        label = (f"{os.path.basename(str(drafter))} d{tokens} {policy} "
                 f"vocab={vocab or 'default'}")
        if args.dry_run:
            print(f"would run {label}")
            continue
        (acc, acc_std, tput, tput_std, sha, drafted, accepted) = run_bench(
            args.bin, args.model, drafter, args.prompt_tokens,
            args.gen_tokens, args.depth, tokens, policy, args.repetitions,
            vocab)
        identical = (sha == baseline_sha) if baseline_sha else None
        speedup = (tput / baseline_throughput) if (tput and baseline_throughput) else None
        verdicts.append({
            "drafter": str(drafter), "draft_tokens": tokens, "policy": policy,
            "vocab": vocab, "acceptance": acc, "acceptance_std": acc_std,
            "drafted": drafted, "accepted": accepted,
            "tok_per_s": tput, "tok_per_s_std": tput_std,
            "speedup": speedup, "greedy_identical": identical, "sha": sha,
        })
        acc_text = f"{acc:.3f} +/- {acc_std:.3f}" if acc is not None else "n/a"
        print(f"{label}: acceptance={acc_text} "
              f"(accepted {accepted}/{drafted}) "
              f"tok/s={tput:.2f} speedup={speedup if speedup is None else f'{speedup:.2f}x'} "
              f"greedy_identical={identical} sha={sha[:12]}", flush=True)

    if args.dry_run or not verdicts:
        return
    ranked = [row for row in verdicts if row["tok_per_s"] is not None]
    ranked.sort(key=lambda row: row["tok_per_s"], reverse=True)
    print(f"\nbaseline no-draft: {baseline_throughput:.2f} tok/s sha={baseline_sha[:12]}")
    print("ranked by tok/s:")
    for row in ranked:
        acc = row["acceptance"]
        print(f"  {'%.3f' % acc if acc is not None else ' n/a '}  "
              f"{row['tok_per_s']:.2f} tok/s  "
              f"{row['speedup']:.2f}x  greedy={row['greedy_identical']}  "
              f"{row['drafter']} "
              f"-d {row['draft_tokens']} --draft-policy {row['policy']}"
              + (f"  GUFO_DFLASH_VOCAB={row['vocab']}" if row["vocab"] else ""))
    differs = [row for row in ranked if row["greedy_identical"] is False]
    if differs:
        print("\nNOTE: these drafters' greedy output_sha256 differs from the "
              "no-draft baseline. If the stock drafter also differs (it does on "
              "this build), the bench output_sha256 is not a valid cross-path "
              "identity check here - treat acceptance/tok/s as the signal.")
    winner = ranked[0] if ranked else None
    if winner:
        print(f"\nbest: GUFO_DFLASH={winner['drafter']} "
              f"--draft-tokens {winner['draft_tokens']} "
              f"--draft-policy {winner['policy']}"
              + (f" GUFO_DFLASH_VOCAB={winner['vocab']}" if winner["vocab"] else ""))


if __name__ == "__main__":
    main()
