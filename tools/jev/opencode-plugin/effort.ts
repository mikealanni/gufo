// Effort plans: the context hook decides, the http.request hook writes them into the body.
import { readFileSync } from "node:fs";
import { join } from "node:path";

export interface Plan { effort?: string; thinking?: boolean; budget?: number; reason: string }

const LEVELS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
const DEFAULT_PINNED = "low";
const GUARD_URL = process.env.JEV_GUARD_URL ?? "http://127.0.0.1:8900";
const BASELINE_TTL_MS = 30_000;
const BASELINE_FALLBACK = "medium";

const plans = new Map<string, Plan>();
export function setPlan(sid: string, plan: Plan | null): void {
  if (plan) plans.set(sid, plan); else plans.delete(sid);
  if (plans.size > 200) plans.clear();
}
export function takePlan(sid: string): Plan | undefined {
  const p = plans.get(sid);
  plans.delete(sid);
  return p;
}

export function pinnedEffort(logDir: string): string | null {
  try {
    const level = String(readFileSync(join(logDir, "autoeffort-off"), "utf8")).trim().toLowerCase();
    return LEVELS.includes(level) ? level : DEFAULT_PINNED;
  } catch { return null; }
}

let baseline = BASELINE_FALLBACK;
let baselineAt = 0;
export async function baselineEffort(): Promise<string> {
  const now = Date.now();
  if (now - baselineAt < BASELINE_TTL_MS) return baseline;
  baselineAt = now;
  try {
    const r = await fetch(`${GUARD_URL}/jev-guard/status`, { signal: AbortSignal.timeout(1500) });
    const info = (await r.json()) as { default_effort?: string };
    if (info.default_effort && LEVELS.includes(info.default_effort)) baseline = info.default_effort;
  } catch { /* keep the last known value */ }
  return baseline;
}
export const effortRank = (e: string): number => LEVELS.indexOf(e);

// Adds only keys the request does not carry, like the engine: a caller's own choice wins.
export function applyPlan(body: any, plan: Plan): string[] {
  const added: string[] = [];
  if (!body || typeof body !== "object") return added;
  const kwargs = body.chat_template_kwargs;
  const hasEffort = "reasoning_effort" in body || (kwargs && typeof kwargs === "object" && "reasoning_effort" in kwargs);
  const hasThinking = "thinking" in body || (kwargs && typeof kwargs === "object" && "enable_thinking" in kwargs);
  // The engine rejects an effort together with thinking disabled; disabled thinking needs no effort.
  if (plan.effort && !hasEffort && plan.thinking !== false) { body.reasoning_effort = plan.effort; added.push("reasoning_effort"); }
  if (plan.thinking !== undefined && !hasThinking) { body.thinking = { type: plan.thinking ? "enabled" : "disabled" }; added.push("thinking"); }
  if (plan.budget !== undefined && !("reasoning_budget_tokens" in body) && !("thinking_budget_tokens" in body)) {
    body.reasoning_budget_tokens = plan.budget; added.push("reasoning_budget_tokens");
  }
  return added;
}
