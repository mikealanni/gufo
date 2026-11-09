// Cross-session gate for heavy commands (docs/JEV-OPENCODE-PLUGIN.md, heavy gate).
// Heavy commands from opencode sessions run one at a time. A session that does not use gufo
// asks the guard to drain gufo's generations and stop it before the command starts.
import { mkdirSync, readdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const GUARD_URL = process.env.JEV_GUARD_URL ?? "http://127.0.0.1:8900";
const LOCK_TTL_MS = 2 * 3600 * 1000;
const WAIT_MAX_MS = 30 * 60 * 1000;
const POLL_MS = 1000;
const PREPARE_TIMEOUT_MS = 12 * 60 * 1000;

type Lock = { sid: string; call: string; pid: number; ts: number; segment: string };

function alive(pid: number): boolean {
  try { process.kill(pid, 0); return true; } catch { return false; }
}

function lockName(sid: string, call: string): string {
  return `${sid}__${call}`.replace(/[^\w.-]/g, "_") + ".json";
}

function readLocks(dir: string): { name: string; lock: Lock }[] {
  const out: { name: string; lock: Lock }[] = [];
  let names: string[] = [];
  try { names = readdirSync(dir); } catch { return out; }
  for (const name of names) {
    try {
      const lock = JSON.parse(readFileSync(join(dir, name), "utf8")) as Lock;
      if (Date.now() - lock.ts > LOCK_TTL_MS || !alive(lock.pid)) {
        try { unlinkSync(join(dir, name)); } catch { /* raced */ }
        continue;
      }
      out.push({ name, lock });
    } catch { /* partial write: ignore this round */ }
  }
  return out;
}

export type HeavyResult = { waitedMs: number; prepared: boolean; note: string };

export async function acquireHeavy(logDir: string, sid: string, call: string, segment: string, gufoSession: boolean): Promise<HeavyResult> {
  const dir = join(logDir, "heavy-lock");
  const t0 = Date.now();
  let note = "";
  mkdirSync(dir, { recursive: true });
  const mine = lockName(sid, call);
  const lock: Lock = { sid, call, pid: process.pid, ts: t0, segment: segment.slice(0, 200) };
  writeFileSync(join(dir, mine), JSON.stringify(lock));
  const ahead = () => readLocks(dir).filter((x) => x.lock.sid !== sid && (x.lock.ts < t0 || (x.lock.ts === t0 && x.name < mine)));
  while (ahead().length > 0) {
    if (Date.now() - t0 > WAIT_MAX_MS) { note = "lock wait timed out"; break; }
    await new Promise((r) => setTimeout(r, POLL_MS));
  }
  let prepared = false;
  {
    try {
      const st = await fetch(`${GUARD_URL}/jev-guard/status`, { signal: AbortSignal.timeout(3000) });
      const info = st.ok ? await st.json() as { prepare_api?: number } : {};
      if (!info.prepare_api) throw new Error("guard has no prepare api");
      const resp = await fetch(`${GUARD_URL}/jev-guard/prepare`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ segment, session: sid }),
        signal: AbortSignal.timeout(PREPARE_TIMEOUT_MS),
      });
      prepared = resp.ok;
      if (!resp.ok) note = `guard prepare http ${resp.status}`;
    } catch (err) {
      note = `guard unreachable: ${String((err as Error)?.message ?? err).slice(0, 80)}`;
    }
  }
  return { waitedMs: Date.now() - t0, prepared, note };
}

export function releaseHeavy(logDir: string, sid: string, call: string): void {
  try { unlinkSync(join(logDir, "heavy-lock", lockName(sid, call))); } catch { /* already gone */ }
}

const CLASSIFY_TIMEOUT_MS = 4000;
// Commands the rules cannot place go to the guard's NPU judge; any failure means "not heavy".
export async function judgeCommand(command: string, sid: string): Promise<boolean> {
  try {
    const r = await fetch(`${GUARD_URL}/jev-guard/classify`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ command, session: sid }),
      signal: AbortSignal.timeout(CLASSIFY_TIMEOUT_MS),
    });
    if (!r.ok) return false;
    return !!((await r.json()) as { heavy?: boolean }).heavy;
  } catch { return false; }
}

// A cancelled or aborted tool call never reaches execute.after: drop the session's locks when its turn ends.
export function releaseSessionLocks(logDir: string, sid: string): void {
  const dir = join(logDir, "heavy-lock");
  let names: string[] = [];
  try { names = readdirSync(dir); } catch { return; }
  const prefix = `${sid}__`.replace(/[^\w.-]/g, "_");
  for (const n of names) if (n.startsWith(prefix)) { try { unlinkSync(join(dir, n)); } catch { /* gone */ } }
}
