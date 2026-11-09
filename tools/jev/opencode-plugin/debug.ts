// Debug: log message shapes on idle.
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const dir = join(homedir(), ".gufo", "jev-log");
mkdirSync(dir, { recursive: true });
const out = join(dir, "plugin-debug.jsonl");
function raw(o: any) { appendFileSync(out, JSON.stringify({ ts: Date.now(), ...o }) + "\n"); }
export default {
  id: "jev-debug",
  async setup(ctx: any) {
    const ctrl = new AbortController();
    void (async () => {
      for await (const ev of ctx.event.subscribe({ signal: ctrl.signal })) {
        const t = (ev as any)?.type;
        if (t !== "session.execution.succeeded" && t !== "session.idle") continue;
        const sid = (ev as any)?.data?.sessionID ?? (ev as any)?.sessionID;
        if (!sid) continue;
        try {
          const msgs = await ctx.session.context({ sessionID: sid });
          raw({ hook: "debug", sessionID: sid, n: Array.isArray(msgs) ? msgs.length : typeof msgs, sample: JSON.stringify(msgs).slice(0, 2000) });
        } catch (e: any) { raw({ hook: "debug-error", e: String(e).slice(0, 200) }); }
      }
    })();
    return () => ctrl.abort();
  },
};
