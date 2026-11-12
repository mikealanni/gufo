// /jev command logic: per-session pause, per-feature switches, autoeffort pin, live status.
import { mkdirSync, readFileSync, readdirSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const LEVELS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
const FEATURES = ["command", "effort", "goal", "finish", "pre"];
const GUARD_URL = process.env.JEV_GUARD_URL ?? "http://127.0.0.1:8900";
const safe = (s: string): string => s.replace(/[^\w.-]/g, "_");
const exists = (p: string): boolean => { try { readFileSync(p); return true; } catch { return false; } };
const rm = (p: string): void => { try { unlinkSync(p); } catch { /* absent */ } };

export const HELP = [
  "jev options",
  "",
  "/jev status",
  "   plugin and guard state",
  "/jev pause [all]",
  "   pause jev",
  "/jev resume [all]",
  "   resume jev",
  "/jev autoeffort on",
  "   jev picks the reasoning effort",
  "/jev autoeffort off [level]",
  "   pin the effort (default low)",
  "   levels: none (no thinking) minimal low medium high xhigh max",
  "/jev switch <model>",
  "   reload gufo as another model",
  "   models: flash-next 27b 35b glm flint laguna",
  "/jev goal [text]",
  "   keep the agent working until the task is done",
  "   (no text = the current task; max 3 pushes)",
  "/jev goal off",
  "   cancel the goal",
  "/jev feature <name> on|off",
  "   names: command effort goal finish pre",
  "/jev help",
  "   this list",
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
    case "switch": {
      if (!a) {
        try {
          const g = (await (await fetch(`${GUARD_URL}/jev-guard/status`, { signal: AbortSignal.timeout(2000) })).json()) as Record<string, unknown>;
          if (g.state === undefined) return "guard is not running; start it first";
          return `loaded: ${g.kind} (${g.model ?? "starting"})\nmodels: ${(g.models as string[]).join(" ")}\nuse /jev switch <model>`;
        } catch { return "guard is not running; start it first"; }
      }
      try {
        const r = await fetch(`${GUARD_URL}/jev-guard/switch`, {
          method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ kind: a }), signal: AbortSignal.timeout(130000),
        });
        const g = (await r.json()) as Record<string, unknown>;
        if (typeof g.error === "object") return "guard is not running; start it first";
        if (!r.ok) return `switch failed: ${g.error ?? r.status}${g.models ? `\nmodels: ${(g.models as string[]).join(" ")}` : ""}`;
        return g.note ? `${a}: ${g.note}` : `switching to ${a}\ngufo restarts now; /jev switch shows when it is loaded`;
      } catch { return "guard is not running; start it first"; }
    }
    case "goal": {
      const rest = argText.trim().slice(cmd.length).trim();
      if (rest.toLowerCase() === "off") { writeFileSync(join(logDir, "goal-off"), ""); return "goal cancelled for the next agent turn."; }
      writeFileSync(join(logDir, "goal-pending"), rest);
      return rest ? `goal armed: ${rest.slice(0, 80)}\nit takes effect when the agent next stops.` : "goal armed for the current task\nit takes effect when the agent next stops.";
    }
    case "help": return HELP;
    case "status": case "": return await status(logDir, sid);
    default: return `unknown option "${cmd}"\n\n${HELP}`;
  }
}

async function status(logDir: string, sid: string): Promise<string> {
  const paused = exists(join(logDir, `plugin-pause.${safe(sid)}`)) || exists(join(logDir, "plugin-pause"));
  const feats = FEATURES.map((f) => `${f} ${exists(join(logDir, `feature-off.${f}`)) || process.env[`JEV_${f.toUpperCase()}`] === "off" ? "off" : "on"}`).join(", ");
  const lines: string[] = ["jev status", "", `plugin: ${paused ? "PAUSED" : "active"}`, `features: ${feats}`];
  lines.push(exists(join(logDir, "autoeffort-off")) ? `autoeffort: off, pinned to ${String(readFileSync(join(logDir, "autoeffort-off"), "utf8")).trim()}` : "autoeffort: on");
  try {
    const r = await fetch(`${GUARD_URL}/jev-guard/status`, { signal: AbortSignal.timeout(2000) });
    const g = (await r.json()) as Record<string, unknown>;
    if (g.state === undefined) { lines.push("guard: not running (port 8900 is served without it)"); return lines.join("\n"); }
    lines.push(`guard: ${g.state}`, `model: ${g.model ?? "-"} (${g.kind ?? "-"})`, `free memory: ${g.mem_available_gb} GB`,
      `judge: ${g.judge ? "on" : "off"}`, `stops: ${g.stops}`);
  } catch { lines.push("guard: unreachable"); }
  return lines.join("\n");
}
