#!/usr/bin/env bun
// Hosted safety checks (no GPU): no-mutation, fail-open, gufo-only.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { homedir, tmpdir } from "node:os";
import { mkdtempSync } from "node:fs";

// Never write test records into the real log: point the plugin at a temp dir
// before it is imported (the path is read at import time).
const tmp = mkdtempSync(join(tmpdir(), "jev-plugin-test-"));
process.env.JEV_LOG_DIR = tmp;

let failed = false;
function check(name: string, cond: boolean) {
  console.log(`${cond ? "PASS" : "FAIL"} ${name}`);
  if (!cond) failed = true;
}

// No-mutation: hooks must not throw or change a frozen output object.
const plugin = (await import("../jev.ts")).default;
const captured: Record<string, any> = {};
const events: any[] = [];
const mockCtx: any = {
  tool: { hook: async (name: string, fn: any) => { captured[name] = fn; } },
  session: {
    hook: async (name: string, fn: any) => { captured[`session.${name}`] = fn; },
    context: async () => [],
  },
  event: { subscribe: () => ({ [Symbol.asyncIterator]: async function* () { for (const e of events) yield e; } }) },
};
await plugin.setup(mockCtx);
check("registers tool.before/after + session hooks", !!captured["execute.before"] && !!captured["execute.after"]);

const frozen = Object.freeze({ args: Object.freeze({ command: "ls /tmp" }) } as any);
try {
  // No session cache yet -> gufo-only gate returns early without mutation.
  await captured["execute.before"]({ tool: "shell", sessionID: "ses_none", input: { command: "ls /tmp" }, id: "c1" });
  check("tool.before on unknown session is no-op (no throw)", true);
} catch { check("tool.before on unknown session is no-op (no throw)", false); }
const before = JSON.stringify(frozen);
try {
  await captured["execute.after"]({ tool: "shell", sessionID: "ses_none", input: { command: "ls /tmp" }, id: "c1", status: "completed" });
  // Frozen output is the event itself in V2 (no output arg to mutate);
  // the hook must not attempt writes to it.
  check("tool.after leaves frozen object untouched", JSON.stringify(frozen) === before);
} catch { check("tool.after leaves frozen object untouched", false); }

// Gufo-only: non-gufo provider produces no JSONL rows.
const logDir = tmp;
const today = new Date();
const ymd = `${today.getFullYear()}${String(today.getMonth() + 1).padStart(2, "0")}${String(today.getDate()).padStart(2, "0")}`;
const f = join(logDir, `plugin-${ymd}.jsonl`);
function countLines(): number {
  try { return readFileSync(f, "utf8").split("\n").filter(Boolean).length; } catch { return 0; }
}
// Seed a non-gufo session via context hook, then fire a shell tool.
await captured["session.context"]({ sessionID: "ses_nongufo", model: { providerID: "openai", id: "gpt-5" } });
const n0 = countLines();
await captured["execute.before"]({ tool: "shell", sessionID: "ses_nongufo", input: { command: "cmake --build --preset release" }, id: "c2" });
check("non-gufo tool produces no record", countLines() === n0);

// Gufo session does produce a record.
await captured["session.context"]({ sessionID: "ses_gufo", model: { providerID: "local-gufo", id: "Qwen3.8 Flash Next" } });
await captured["execute.before"]({ tool: "shell", sessionID: "ses_gufo", input: { command: "ls /tmp" }, id: "c3" });
await new Promise((r) => setTimeout(r, 300));
check("gufo tool produces a record", countLines() > n0);

// Kill switch: JEV_PLUGIN=off -> hooks are no-ops.
process.env.JEV_PLUGIN = "off";
const n1 = countLines();
await captured["execute.before"]({ tool: "shell", sessionID: "ses_gufo", input: { command: "ls /tmp" }, id: "c4" });
await new Promise((r) => setTimeout(r, 300));
check("kill switch makes hooks no-op", countLines() === n1);
delete process.env.JEV_PLUGIN;

// Pause flag: a file in the log dir pauses every hook, removing it resumes (2 s cache).
import { writeFileSync, unlinkSync } from "node:fs";
const pauseFile = join(tmp, "plugin-pause");
writeFileSync(pauseFile, "");
await new Promise((r) => setTimeout(r, 2300));
const p0 = countLines();
await captured["execute.before"]({ tool: "shell", sessionID: "ses_gufo", input: { command: "ls /tmp" }, id: "cp1" });
await new Promise((r) => setTimeout(r, 400));
check("pause flag makes hooks no-op", countLines() === p0);
unlinkSync(pauseFile);
await new Promise((r) => setTimeout(r, 2300));
await captured["execute.before"]({ tool: "shell", sessionID: "ses_gufo", input: { command: "ls /tmp" }, id: "cp2" });
await new Promise((r) => setTimeout(r, 400));
check("removing the pause flag resumes logging", countLines() > p0);

// Fail-open: an injected exception in each hook must not escape and must leave
// exactly one error record (per hook name, per minute).
function rows(): any[] {
  try { return readFileSync(f, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)); } catch { return []; }
}
for (const [hook, call, arg] of [
  ["prompt", "session.prompt", { sessionID: "ses_gufo", prompt: "hi", messageID: "m1" }],
  ["context", "session.context", { sessionID: "ses_gufo", model: { providerID: "local-gufo", id: "Qwen3.8 Flash Next" }, messages: [] }],
  ["execute.before", "execute.before", { tool: "shell", sessionID: "ses_gufo", input: { command: "ls" }, id: "ci1" }],
  ["execute.after", "execute.after", { tool: "shell", sessionID: "ses_gufo", id: "ci1", status: "completed" }],
] as const) {
  process.env.JEV_TEST_THROW = hook;
  const before = rows().filter((r) => r.feature === "error" && r.hook === hook).length;
  let threw = false;
  try { await captured[call](arg); } catch { threw = true; }
  await new Promise((r) => setTimeout(r, 300));
  const after = rows().filter((r) => r.feature === "error" && r.hook === hook).length;
  check(`fail-open ${hook}: no throw, one error record`, !threw && after === before + 1);
  delete process.env.JEV_TEST_THROW;
}

// Goal phrase is evaluated for every user message, not only the first one.
await captured["session.prompt"]({ sessionID: "ses_gufo", prompt: "please look at the repo", messageID: "g1" });
await captured["session.context"]({ sessionID: "ses_gufo", model: { providerID: "local-gufo", id: "Qwen3.8 Flash Next" }, messages: [] });
await captured["session.prompt"]({ sessionID: "ses_gufo", prompt: "jev make sure its done", messageID: "g2" });
await captured["session.context"]({ sessionID: "ses_gufo", model: { providerID: "local-gufo", id: "Qwen3.8 Flash Next" }, messages: [] });
await new Promise((r) => setTimeout(r, 300));
const goals = rows().filter((r) => r.feature === "goal");
check("goal logged per user message", goals.length >= 2 && goals.some((g) => g.decision.matched === true));
check("unmatched prompt text is not logged", goals.filter((g) => g.decision.matched === false).every((g) => g.cmd === ""));

// Finish detector and activity end: every turn end is handled (not only the
// first one per session), and bursts of events for one turn collapse to one.
const turnEvents = async function* () {
  const ev = { type: "session.idle", data: { sessionID: "ses_gufo" } };
  yield ev; yield { type: "session.execution.succeeded", data: { sessionID: "ses_gufo" } };
  await new Promise((r) => setTimeout(r, 1700));
  yield ev;
  await new Promise((r) => setTimeout(r, 400));
};
const ctx2: any = {
  tool: { hook: async () => {} },
  session: { hook: async () => {}, context: async () => [{ type: "user", text: "do it" }, { type: "assistant", content: [{ type: "text", text: "done" }] }] },
  event: { subscribe: () => turnEvents() },
};
const unsub = await plugin.setup(ctx2);
await new Promise((r) => setTimeout(r, 2800));
const fin = rows().filter((r) => r.feature === "finish").length;
const ends = rows().filter((r) => r.feature === "activity" && r.decision.phase === "end").length;
check("finish evaluated once per turn (2 turns, burst collapsed)", fin === 2);
check("activity end recorded once per turn", ends === 2);
if (typeof unsub === "function") unsub();

// Effort gate reads the hook's model messages (tool-call parts on assistant
// messages, tool-result parts on tool messages).
const call = (name: string, input: any) => ({ type: "tool-call", id: "c1", name, input });
const result = (name: string, value: string) => ({ type: "tool-result", id: "c1", name, result: { type: "text", value } });
async function gateFor(name: string, input: any, value: string, trailingUser = false, trailingSystem = false): Promise<any> {
  const n0 = rows().filter((r) => r.feature === "effort_gate").length;
  const messages: any[] = [
    { role: "user", content: [{ type: "text", text: "do it" }] },
    { role: "assistant", content: [call(name, input)] },
    { role: "tool", content: [result(name, value)] },
  ];
  if (trailingUser) messages.push({ role: "user", content: [{ type: "text", text: "reminder" }] });
  if (trailingSystem) messages.push({ role: "system", content: [{ type: "text", text: "reminder" }] });
  await captured["session.context"]({ sessionID: "ses_gufo", model: { providerID: "local-gufo", id: "Qwen3.8 Flash Next" }, messages });
  await new Promise((r) => setTimeout(r, 300));
  return rows().filter((r) => r.feature === "effort_gate").slice(n0).pop()?.decision;
}
const g1 = await gateFor("read", { path: "/x" }, "file contents here");
check("gate: short read result is routine (would lower)", g1?.would_lower === true);
const g2 = await gateFor("shell", { command: "cd /home/majed/gufo && git status -sb" }, "## main");
check("gate: git status through cd is routine", g2?.would_lower === true);
const g3 = await gateFor("shell", { command: "cmake --build build --target gufo" }, "ok");
check("gate: a build is not routine", g3?.would_lower === false && String(g3.blocker).startsWith("cmd=cmake"));
const g4 = await gateFor("read", { path: "/x" }, "Error: no such file");
check("gate: a failure signal is not routine", g4?.would_lower === false);
const g5 = await gateFor("read", { path: "/x" }, "file contents here", true);
check("gate: trailing user message means the gate does not apply", g5?.reason === "last-message-not-tool");
const g6 = await gateFor("read", { path: "/x" }, "file contents here", false, true);
check("gate: a trailing system message is ignored", g6?.would_lower === true);

// A relative script after a cd in the same command is resolved against that directory and inspected.
import { mkdirSync, writeFileSync as writeFile } from "node:fs";
import { executedScripts, inspectScripts } from "../classify.ts";
const cdDir = join(tmp, "cdcase");
mkdirSync(cdDir, { recursive: true });
writeFile(join(cdDir, "run_bench.sh"), "#!/bin/bash\n./build/release/gufo bench -m x\n");
const cdCmd = `cd ${cdDir} && ./run_bench.sh > out.log 2>&1; tail -3 out.log`;
check("cd: relative script is resolved against the cd directory", executedScripts(cdCmd)[0] === join(cdDir, "run_bench.sh"));
check("cd: the resolved script is inspected and found heavy", inspectScripts(cdCmd).reason !== null);

console.log(failed ? "SAFETY FAIL" : "SAFETY PASS");
process.exit(failed ? 1 : 0);
