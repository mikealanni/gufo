// Task 0 spike (docs/JEV-OPENCODE-PLUGIN.md): log hook names + input/output
// keys to ~/.gufo/jev-log/plugin-spike.jsonl. V2 ctx-registration.
// Kill switch: JEV_PLUGIN=off makes every hook an immediate no-op.
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const dir = join(homedir(), ".gufo", "jev-log");
mkdirSync(dir, { recursive: true });
const out = join(dir, "plugin-spike.jsonl");
function raw(obj: any) {
  if (process.env.JEV_PLUGIN === "off") return;
  appendFileSync(out, JSON.stringify({ ts: Date.now(), ...obj }) + "\n");
}
function rec(hook: string, input: any, output?: any) {
  raw({ hook, ik: Object.keys(input ?? {}), ok: Object.keys(output ?? {}) });
}

raw({ hook: "load", rt: process.versions });
try {
  const m: any = await import("bun:sqlite");
  raw({ hook: "bun-sqlite", hasDatabase: typeof m.Database === "function" });
} catch (e: any) {
  raw({ hook: "bun-sqlite", hasDatabase: false, error: String(e).slice(0, 120) });
}
export default {
  id: "jev-spike",
  async setup(ctx: any) {
    raw({ hook: "setup", ctx: Object.keys(ctx ?? {}) });
    const controller = new AbortController();
    void (async () => {
      try {
        for await (const ev of ctx.event.subscribe({ signal: controller.signal })) {
          raw({ hook: "event", type: (ev as any)?.type ?? "?", keys: Object.keys(ev ?? {}) });
        }
      } catch { /* ignore */ }
    })();
    await ctx.tool.hook("execute.before", async (e: any) => {
      rec("tool.before", e);
      raw({ hook: "tool.before.shape", tool: e.tool, inputKeys: Object.keys(e.input ?? {}), cmd: typeof e.input?.command === "string" ? e.input.command.slice(0, 80) : typeof e.input?.command, sid: typeof e.sessionID === "string" });
    });
    await ctx.tool.hook("execute.after", async (e: any) => {
      rec("tool.after", e);
      raw({ hook: "tool.after.shape", tool: e.tool, status: e.status ?? e.result?.status });
    });
    await ctx.session.hook("prompt", async (e: any) => rec("session.prompt", e));
    await ctx.session.hook("context", async (e: any) => {
      rec("session.context", e);
      raw({ hook: "session.context.shape", sessionID: typeof e.sessionID === "string", model: e.model ? { providerID: (e.model as any).providerID, id: (e.model as any).id ?? (e.model as any).modelID } : null, optionsKeys: Object.keys(e.options ?? {}), tools: Array.isArray(e.tools) ? e.tools.length : typeof e.tools, messages: Array.isArray(e.messages) ? e.messages.length : typeof e.messages });
    });
    return () => controller.abort();
  },
};
