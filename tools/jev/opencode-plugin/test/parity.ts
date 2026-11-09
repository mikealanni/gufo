#!/usr/bin/env bun
// Classification parity: TS classify() vs Python guard.classify (fresh).
// Also reports stored-file drift for transparency. Target: >=99.5% vs fresh
// Python, zero fresh-heavy -> TS-light.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { spawnSync } from "node:child_process";
import { classify } from "../classify.ts";

const dataPath = join(homedir(), "dflash-train", "guard_cmds.json");
const data = JSON.parse(readFileSync(dataPath, "utf8")) as Array<{ cmd: string; verdict: string }>;

const pyScript = `
import json, importlib.util
spec = importlib.util.spec_from_file_location("g", "/home/majed/gufo/tools/jev/guard.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
d = json.load(open("/home/majed/dflash-train/guard_cmds.json"))
print(json.dumps([g.classify(r["cmd"])[0] for r in d]))
`;
writeFileSync("/tmp/jev-parity-fresh.py", pyScript);
const out = spawnSync("python3", ["/tmp/jev-parity-fresh.py"], { encoding: "utf8", maxBuffer: 100 * 1024 * 1024 });
if (out.status !== 0) {
  console.log("PARITY FAIL: python fresh failed");
  console.log(out.stderr.slice(0, 500));
  process.exit(1);
}
const pyFresh = JSON.parse(out.stdout) as string[];

let agreeStored = 0, agreeFresh = 0, h2lStored = 0, h2lFresh = 0;
const misses: Array<{ i: number; cmd: string; py: string; ts: string }> = [];
const times: number[] = [];
for (let i = 0; i < data.length; i++) {
  const t0 = performance.now();
  const ts = classify(data[i].cmd).verdict;
  times.push(performance.now() - t0);
  if (ts === data[i].verdict) agreeStored++;
  else if (data[i].verdict === "heavy" && ts === "light") h2lStored++;
  if (ts === pyFresh[i]) agreeFresh++;
  else {
    if (pyFresh[i] === "heavy" && ts === "light") h2lFresh++;
    if (misses.length < 20) misses.push({ i, cmd: data[i].cmd.slice(0, 120), py: pyFresh[i], ts });
  }
}
times.sort((a, b) => a - b);
const median = times[Math.floor(times.length / 2)] ?? 0;
const p99 = times[Math.floor(times.length * 0.99)] ?? 0;
console.log(`rows=${data.length} stored=${(100 * agreeStored / data.length).toFixed(2)}% fresh=${(100 * agreeFresh / data.length).toFixed(2)}% h2l_fresh=${h2lFresh} h2l_stored=${h2lStored}`);
console.log(`median_ms=${median.toFixed(3)} p99_ms=${p99.toFixed(3)}`);
for (const d of misses) console.log(`MISS i=${d.i} py=${d.py} ts=${d.ts} cmd=${JSON.stringify(d.cmd)}`);
const ok = (100 * agreeFresh) / data.length >= 99.5 && h2lFresh === 0;
console.log(ok ? "PARITY PASS" : "PARITY FAIL");
process.exit(ok ? 0 : 1);
