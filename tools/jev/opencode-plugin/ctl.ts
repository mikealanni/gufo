// /jev command logic: per-session pause, per-feature switches, autoeffort pin, live status.
import { mkdirSync, readFileSync, readdirSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"];
const FEATURES = ["command", "effort", "goal", "finish", "pre"];
const GUARD_URL = process.env.JEV_GUARD_URL ?? "http://127.0.0.1:8900";
const safe = (s: string): string => s.replace(/[^\w.-]/g, "_");
const exists = (p: string): boolean => { try { readFileSync(p); return true; } catch { return false; } };
const rm = (p: string): void => { try { unlinkSync(p); } catch { /* absent */ } };

export const HELP = [
  "jev control (type /jev <option>)",
  "  status                      plugin state, pause, features, autoeffort, guard (model, memory)",
  "  pause [all]                 pause jev (in the TUI this pauses all sessions)",
  "  resume [all]                resume jev",
  "  autoeffort on|off [level]   let jev pick reasoning effort, or pin it (minimal|low|medium|high|xhigh|max)",
  "  feature <name> on|off       switch one feature: command, effort, goal, finish, pre",
  "  help                        this list",
].join("\n");

export async function control(logDir: string, sid: string, argText: string): Promise<string> {
  mkdirSync(logDir, { recursive: true });
  const [cmd = "", a = "", b = ""] = argText.trim().split(/\s+/);
  const all = join(logDir, "plugin-pause");
  const one = join(logDir, `plugin-pause.${safe(sid)}`);
  switch (cmd) {
    case "pause":
      if (a === "all") { writeFileSync(all, ""); return "jev PAUSED for all sessions."; }
      writeFileSync(one, ""); return `jev PAUSED for this session (${sid}) only.`;
    case "resume":
      if (a === "all") {
        rm(all);
        for (const f of readdirSync(logDir)) if (f.startsWith("plugin-pause.")) rm(join(logDir, f));
        return "jev RESUMED for all sessions.";
      }
      rm(one);
      return `jev RESUMED for this session (${sid}).` + (exists(all) ? " A global pause is still active (/jev resume all)." : "");
    case "autoeffort":
      if (a === "off") {
        const level = LEVELS.includes(b) ? b : "low";
        writeFileSync(join(logDir, "autoeffort-off"), level);
        return `autoeffort OFF: requests without their own effort are pinned to '${level}' (all sessions).`;
      }
      if (a === "on") { rm(join(logDir, "autoeffort-off")); return "autoeffort ON: jev picks the reasoning effort per request."; }
      return exists(join(logDir, "autoeffort-off")) ? `autoeffort: OFF, pinned to ${String(readFileSync(join(logDir, "autoeffort-off"), "utf8")).trim()}` : "autoeffort: ON";
    case "feature": {
      if (!FEATURES.includes(a) || (b !== "on" && b !== "off")) return `usage: /jev feature ${FEATURES.join("|")} on|off`;
      if (b === "off") writeFileSync(join(logDir, `feature-off.${a}`), ""); else rm(join(logDir, `feature-off.${a}`));
      return `feature '${a}' ${b.toUpperCase()}.`;
    }
    case "help": return HELP;
    case "status": case "": return await status(logDir, sid);
    default: return `unknown option "${cmd}"\n\n${HELP}`;
  }
}

async function status(logDir: string, sid: string): Promise<string> {
  const lines: string[] = [];
  lines.push(`session ${sid || "(unknown)"}: ${exists(join(logDir, `plugin-pause.${safe(sid)}`)) ? "PAUSED" : "active"}` + (exists(join(logDir, "plugin-pause")) ? " (global pause on)" : ""));
  lines.push("features: " + FEATURES.map((f) => `${f}=${exists(join(logDir, `feature-off.${f}`)) || process.env[`JEV_${f.toUpperCase()}`] === "off" ? "off" : "on"}`).join(" "));
  lines.push(exists(join(logDir, "autoeffort-off")) ? `autoeffort: OFF, pinned to ${String(readFileSync(join(logDir, "autoeffort-off"), "utf8")).trim()}` : "autoeffort: ON");
  try {
    const r = await fetch(`${GUARD_URL}/jev-guard/status`, { signal: AbortSignal.timeout(2000) });
    const g = (await r.json()) as Record<string, unknown>;
    lines.push(`guard: state=${g.state} stops=${g.stops} default_effort=${g.default_effort} mem_available_gb=${g.mem_available_gb} judge=${g.judge}`);
  } catch { lines.push("guard: unreachable"); }
  return lines.join("\n");
}
