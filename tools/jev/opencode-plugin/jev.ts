// Phase 0 command gate (docs/JEV-OPENCODE-PLUGIN.md Section 1, log-only).
// No GPU/NPU, no judge, no mutation: hook bodies only classify + enqueue.
// Entry point symlinked to ~/.config/opencode/plugins/jev.ts.
import { appendFileSync, mkdirSync, readdirSync, statSync, unlinkSync } from "node:fs";
import { createHash } from "node:crypto";
import { homedir } from "node:os";
import { join } from "node:path";
import { Database } from "bun:sqlite";
import { classify, containsSecret, inspectScripts } from "./classify.ts";
import { detectGoal } from "./goal.ts";
import { judgeFinish } from "./finish.ts";
import { control } from "./ctl.ts";
import { hasGoalTrigger } from "./goal.ts";
import { GOAL_MAX_CONTINUES, detectGoalSpec, evaluateGoal, extractItems, stripGoalTrigger, type GoalMessage } from "./goal-eval.ts";
import { acquireHeavy, judgeCommand, releaseHeavy, releaseSessionLocks } from "./heavy.ts";
import { classifyJevAgentTurn, classifyJevRequest, K_ROUTINE_BUDGET_TOKENS, type AgentTurnResult } from "./gate.ts";
import { applyPlan, baselineEffort, effortRank, pinnedEffort, setPlan, takePlan, type Plan } from "./effort.ts";

const PLUGIN_VERSION = "0.1.0";
const LOG_DIR = process.env.JEV_LOG_DIR ?? join(homedir(), ".gufo", "jev-log");
const RETAIN_DAYS = 14;
const MAX_DAY_BYTES = 64 * 1024 * 1024;
const JSONL = () => {
  const d = new Date();
  const ymd = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}`;
  return join(LOG_DIR, `plugin-${ymd}.jsonl`);
};
const DB_PATH = join(LOG_DIR, "jev.sqlite3");
const PROVIDERS = new Set((process.env.JEV_GUFO_PROVIDERS ?? "local-gufo,kairic-remote").split(",").map((s) => s.trim()).filter(Boolean));

function redact(cmd: string): string {
  const t = cmd.slice(0, 500).replace(/[\r\n]/g, " ");
  return containsSecret(t) ? "[redacted-secret]" : t;
}

type Rec = Record<string, unknown>;
const queue: Rec[] = [];
let flushedError = false;
let db: any = null;

let capped = false;
function flush(): void {
  if (!queue.length) return;
  try {
    mkdirSync(LOG_DIR, { recursive: true });
    const file = JSONL();
    let size = 0;
    try { size = statSync(file).size; } catch { /* new file */ }
    if (size > MAX_DAY_BYTES) {
      if (!capped) {
        capped = true;
        appendFileSync(file, JSON.stringify({ feature: "error", note: "daily size cap reached; dropping records" }) + "\n");
      }
      queue.length = 0;
      return;
    }
    capped = false;
    appendFileSync(file, queue.map((r) => JSON.stringify(r)).join("\n") + "\n");
  } catch {
    if (!flushedError) {
      flushedError = true;
      try { appendFileSync(JSONL(), JSON.stringify({ feature: "error", note: "flush failed" }) + "\n"); } catch { /* drop */ }
    }
  }
  queue.length = 0;
}
setInterval(flush, 250);

function pruneOldLogs(): void {
  try {
    const cutoff = Date.now() - RETAIN_DAYS * 86400 * 1000;
    for (const name of readdirSync(LOG_DIR)) {
      if (!/^plugin-\d{8}\.jsonl$/.test(name)) continue;
      const f = join(LOG_DIR, name);
      if (statSync(f).mtimeMs < cutoff) unlinkSync(f);
    }
  } catch { /* retention is best effort */ }
}

// One error record per hook name per minute: fail open, but leave a trace.
const lastError = new Map<string, number>();
function noteError(hook: string, err: unknown): void {
  const now = Date.now();
  if (now - (lastError.get(hook) ?? 0) < 60000) return;
  lastError.set(hook, now);
  queue.push({ ts: now / 1000, plugin_version: PLUGIN_VERSION, feature: "error", hook, note: String((err as any)?.message ?? err).slice(0, 200) });
}
// Diagnostic trace: create ~/.gufo/jev-log/plugin-trace.on to log every hook call
// (names, session, provider, key lists; never content). Checked at most every 2 s.
const TRACE_FLAG = join(LOG_DIR, "plugin-trace.on");
let traceOn = false;
let traceChecked = 0;
function trace(hook: string, e: any): void {
  try {
    const now = Date.now();
    if (now - traceChecked > 2000) {
      traceChecked = now;
      try { statSync(TRACE_FLAG); traceOn = true; } catch { traceOn = false; }
    }
    if (!traceOn) return;
    appendFileSync(join(LOG_DIR, "plugin-trace.jsonl"), JSON.stringify({
      ts: now / 1000, pid: process.pid, hook, sid: e?.sessionID ?? e?.data?.sessionID ?? null,
      provider: e?.model?.providerID ?? null, model: e?.model?.id ?? e?.model?.modelID ?? null,
      keys: e && typeof e === "object" ? Object.keys(e).slice(0, 20) : [], type: e?.type ?? null,
      nmsg: Array.isArray(e?.messages) ? e.messages.length : null, seen: seen.size,
      shape: Array.isArray(e?.messages) ? e.messages.slice(-12).map((m: any) => ({
        k: Object.keys(m ?? {}).slice(0, 14), type: m?.type ?? m?.role ?? null,
        parts: ((m?.content ?? m?.parts ?? []) as any[]).slice(0, 6).map((pp: any) => ({
          t: pp?.type ?? null, k: Object.keys(pp ?? {}).slice(0, 12),
          sk: pp?.state && typeof pp.state === "object" ? Object.keys(pp.state).slice(0, 10) : null,
          rk: pp?.result && typeof pp.result === "object" ? Object.keys(pp.result).slice(0, 8) : typeof pp?.result,
          ik: pp?.input && typeof pp.input === "object" ? Object.keys(pp.input).slice(0, 8) : typeof pp?.input,
        })),
      })) : null,
    }) + "\n");
  } catch { /* trace is best effort */ }
}

// Test-only fault injection (JEV_TEST_THROW=<hook name>); never set in production.
function injected(hook: string): void {
  if (process.env.JEV_TEST_THROW === hook) throw new Error(`injected failure in ${hook}`);
}

function sqliteInsert(row: Rec): void {
  if (!db) {
    try {
      db = new Database(DB_PATH);
      db.exec("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=2000;");
    } catch { db = null; return; }
  }
  try {
    const s = row.score as string;
    const d = row as any;
    db.prepare(
      "INSERT INTO decisions(ts, model, battery, trigger, prompt_hash, score, logit_gap, duration_ms, threshold, framing_ver, action, request_id, session_id, question) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    ).run(d.ts, d.model, d.battery, d.trigger, d.prompt_hash, s, d.logit_gap, d.duration_ms, d.threshold, d.framing_ver, d.action, d.request_id, d.session_id, d.question);
  } catch { /* busy_timeout / missing table: drop one row silently */ }
}

function enqueue(rec: Rec): void {
  queue.push(rec);
  if (queue.length >= 50) flush();
}

const GATE_ON = process.env.JEV_HEAVY_GATE !== "off";
const held = new Set<string>();
const seen = new Map<string, { provider: string; model: string }>();
const pending = new Map<string, { verdict: string; would_stop: boolean; segment: string; session: string; provider: string; model: string; cmd: string; t: number }>();

// Pause without restarting anything: create ~/.gufo/jev-log/plugin-pause to make every
// hook a no-op, delete it to resume (checked at most every 2 s). JEV_PLUGIN=off does the
// same but is only read when the opencode service starts.
const PAUSE_FLAG = join(LOG_DIR, "plugin-pause");
const pauseChecked = new Map<string, { at: number; on: boolean }>();
function flagOn(path: string, key: string): boolean {
  const now = Date.now();
  const c = pauseChecked.get(key);
  if (c && now - c.at < 2000) return c.on;
  let on = false;
  try { statSync(path); on = true; } catch { on = false; }
  pauseChecked.set(key, { at: now, on });
  return on;
}
const FEATURES = ["command", "effort", "goal", "finish", "pre"] as const;
function featureOn(name: (typeof FEATURES)[number]): boolean {
  return process.env[`JEV_${name.toUpperCase()}`] !== "off" && !flagOn(join(LOG_DIR, `feature-off.${name}`), `f:${name}`);
}
function disabled(sid = ""): boolean {
  if (process.env.JEV_PLUGIN === "off") return true;
  if (flagOn(PAUSE_FLAG, "")) return true;
  return sid ? flagOn(`${PAUSE_FLAG}.${sid.replace(/[^\w.-]/g, "_")}`, sid) : false;
}
function messageText(m: any): string {
  if (!m) return "";
  if (typeof m.content === "string") return m.content;
  const parts = (Array.isArray(m.content) ? m.content : Array.isArray(m.parts) ? m.parts : []) as any[];
  return parts.filter((p) => p?.type === "text").map((p) => String(p.text ?? "")).join("\n");
}

const FINISH_MAX_CONTINUES = 2;
const FINISH_TAIL_CHARS = 200;
const goals = new Map<string, { text: string; continues: number }>();
const finishes = new Map<string, { continues: number; lastReason: string; lastTail: string }>();
const lastMsgs = new Map<string, any[]>();
const injectedTexts = new Set<string>();
const armed = new Set<string>();

function resultTextOf(r: any): string {
  const v = r?.value ?? r;
  return typeof v === "string" ? v : v == null ? "" : JSON.stringify(v);
}
// Hook-shaped messages (assistant tool-call parts, tool-result parts) to the goal tracker's view.
function toGoalMessages(msgs: any[]): GoalMessage[] {
  const out: GoalMessage[] = [];
  for (const m of msgs) {
    const role = String(m?.role ?? m?.type ?? "");
    const parts = (Array.isArray(m?.content) ? m.content : []) as any[];
    if (role === "user") out.push({ role: "user", text: messageText(m) });
    else if (role === "assistant") {
      const calls = parts.filter((p) => p?.type === "tool-call").map((c) => ({
        id: String(c.id ?? ""), name: String(c.name ?? ""),
        command: String(c.input?.command ?? c.input?.cmd ?? ""),
        path: String(c.input?.filePath ?? c.input?.file_path ?? c.input?.path ?? ""),
      }));
      out.push({ role: "assistant", text: messageText(m), calls });
    } else if (role === "tool") {
      for (const p of parts) if (p?.type === "tool-result") out.push({ role: "tool", text: resultTextOf(p.result), toolCallId: String(p.id ?? "") });
    }
  }
  return out;
}
function isGufo(sessionID: string): boolean {
  const v = seen.get(sessionID);
  return !!v && PROVIDERS.has(v.provider);
}

export default {
  id: "jev",
  async setup(ctx: any) {
    // Buffer the last user text; provider/model only arrive in context, which
    // fires after prompt. The goal check runs there when the session is gufo.
    const lastPrompt = new Map<string, { text: string; messageID: string }>();
    const goalDone = new Map<string, string>();
    pruneOldLogs();
    await ctx.session.hook("prompt", async (e: any) => {
      try {
        const sid = String(e.sessionID ?? "");
        if (disabled(sid)) return;
        trace("session.prompt", e);
        injected("prompt");
        if (!sid) return;
        const text = typeof e.prompt === "string" ? e.prompt : (e.prompt?.text ?? "");
        if (typeof text === "string" && text) lastPrompt.set(sid, { text, messageID: String(e.messageID ?? "") });
        if (!isGufo(sid) || typeof text !== "string" || !text) return;
        const ours = injectedTexts.delete(`${sid}\n${text}`);
        if (ours) return;
        // A genuine user message: continuation budgets start over.
        finishes.delete(sid);
        const g = goals.get(sid);
        if (g) g.continues = 0;
      } catch (err) { noteError("prompt", err); }
    });
    await ctx.session.hook("context", async (e: any) => {
      try {
        const sid = String(e.sessionID ?? "");
        if (disabled(sid)) return;
        trace("session.context", e);
        injected("context");
        const pid = e.model?.providerID as string | undefined;
        const mid = (e.model?.id ?? e.model?.modelID) as string | undefined;
        if (sid && pid) seen.set(sid, { provider: pid, model: mid ?? "" });
        if (!sid || !isGufo(sid)) return;
        // Goal phrase: stripped from every user message at render time (like the engine did), armed once per message.
        if (featureOn("goal")) {
          const all = (e.messages ?? []) as any[];
          const raw = toGoalMessages(all);
          const hit = all.some((m: any) => String(m?.role ?? m?.type ?? "") === "user" && hasGoalTrigger(messageText(m)));
          if (hit) {
            const spec = detectGoalSpec(raw);
            const key = `${sid}:${spec.text.slice(0, 60)}`;
            if (!armed.has(key)) {
              armed.add(key);
              if (armed.size > 500) armed.clear();
              goals.set(sid, { text: spec.text, continues: 0 });
              const en = seen.get(sid) ?? { provider: "", model: "" };
              enqueue({
                ts: Date.now() / 1000, plugin_version: PLUGIN_VERSION, feature: "goal_arm",
                session: sid, call: "", provider: en.provider, model: en.model,
                decision: { items: spec.items, from_history: spec.fromHistory, goal: spec.text.slice(0, 120) }, segment: "", cmd: "", dur_ms: null, note: "",
              });
            }
            for (const m of all) {
              if (String(m?.role ?? m?.type ?? "") !== "user") continue;
              if (typeof m.content === "string") { if (hasGoalTrigger(m.content)) m.content = stripGoalTrigger(m.content); }
              else if (Array.isArray(m.content)) for (const part of m.content) if (part?.type === "text" && typeof part.text === "string" && hasGoalTrigger(part.text)) part.text = stripGoalTrigger(part.text);
            }
          }
        }
        lastMsgs.set(sid, (e.messages ?? []) as any[]);
        if (lastMsgs.size > 200) lastMsgs.clear();
        const entry = seen.get(sid)!;
        // Effort gate (log-only): rebuild turns from trailing history.
        try {
          // Hook message shape (model messages): assistant parts of type "tool-call"
          // carry {id, name, input}; "tool" messages carry "tool-result" parts
          // {id, name, result:{type,value}}. Mirrors ApplyJevAgentGate: a user message
          // clears the turns, and the gate only applies when the last message is a tool result.
          const msgs = (e.messages ?? []) as any[];
          const roleOf = (m: any): string => String(m?.role ?? m?.type ?? "");
          const resultText = (r: any): string => {
            const v = r?.value ?? r;
            return typeof v === "string" ? v : v == null ? "" : JSON.stringify(v);
          };
          const calls = new Map<string, { tool: string; command: string }>();
          const turns: AgentTurnResult[][] = [];
          for (const m of msgs) {
            const role = roleOf(m);
            if (role === "user") { turns.length = 0; continue; }
            const parts = Array.isArray(m?.content) ? (m.content as any[]) : [];
            if (role === "assistant") {
              const tcs = parts.filter((pp: any) => pp?.type === "tool-call");
              if (!tcs.length) continue;
              turns.push([]);
              for (const c of tcs) {
                calls.set(String(c.id), { tool: String(c.name ?? "tool"), command: String(c.input?.command ?? c.input?.cmd ?? "") });
              }
            } else if (role === "tool" && turns.length) {
              for (const pp of parts) {
                if (pp?.type !== "tool-result") continue;
                const hit = calls.get(String(pp.id));
                turns[turns.length - 1].push({ tool: hit?.tool ?? String(pp.name ?? "tool"), command: hit?.command ?? "", output: resultText(pp.result) });
              }
            }
          }
          // System messages are hoisted to the front when the request is rendered, so the
          // message that decides whether the gate applies is the last non-system one.
          const lastRole = roleOf([...msgs].reverse().find((m: any) => roleOf(m) !== "system"));
          const decision = lastRole !== "tool"
            ? { lower: false, reason: "last-message-not-tool", blocker: lastRole }
            : classifyJevAgentTurn(turns);
          enqueue({
            ts: Date.now() / 1000, plugin_version: PLUGIN_VERSION, feature: "effort_gate",
            session: sid, call: "", provider: entry.provider, model: entry.model,
            decision: { would_lower: decision.lower, reason: decision.reason, blocker: decision.blocker }, segment: "",
            cmd: "", dur_ms: null, note: "",
          });
          // Effort plan: written into the request body by the http.request hook.
          let plan: Plan | null = null;
          const pinned = pinnedEffort(LOG_DIR);
          if (pinned) plan = { effort: pinned, reason: "autoeffort-off" };
          else if (featureOn("effort")) {
            if (lastRole === "tool") {
              const base = await baselineEffort();
              if (decision.lower && effortRank(base) > effortRank("low")) plan = { budget: K_ROUTINE_BUDGET_TOKENS, reason: decision.reason };
            } else {
              const lastUser = [...msgs].reverse().find((m: any) => roleOf(m) === "user");
              const text = messageText(lastUser);
              if (text) {
                const rd = classifyJevRequest(text);
                plan = { effort: rd.effort, thinking: rd.needsThinking, reason: "request" };
              }
            }
          }
          setPlan(sid, plan);
        } catch (err) { noteError("context.effort_gate", err); }
        // Activity: one request record per model call (daemon reads start/end).
        enqueue({
          ts: Date.now() / 1000, plugin_version: PLUGIN_VERSION, feature: "activity",
          session: sid, call: "", provider: entry.provider, model: entry.model,
          decision: { phase: "request" }, segment: "", cmd: "", dur_ms: null, note: "start",
        });
        // Prefer the buffered prompt text; fall back to last user message.
        const buffered = lastPrompt.get(sid);
        const userMsg = [...(e.messages ?? [])].reverse().find((m: any) => (m.role ?? m.info?.role) === "user");
        const text = buffered?.text ?? (typeof userMsg?.content === "string" ? userMsg.content : "");
        if (!text) return;
        // Once per user message (a goal phrase can arrive at any point in a session).
        const msgKey = buffered?.messageID || createHash("sha1").update(text).digest("hex");
        if (goalDone.get(sid) === msgKey) return;
        goalDone.set(sid, msgKey);
        if (goalDone.size > 500) goalDone.clear();
        const hit = detectGoal(text);
        enqueue({
          ts: Date.now() / 1000, plugin_version: PLUGIN_VERSION, feature: "goal",
          session: sid, call: buffered?.messageID ?? "", provider: entry.provider, model: entry.model,
          decision: { matched: hit.matched, phrase: hit.phrase, bare: hit.bare }, segment: "",
          cmd: hit.matched ? redact(text) : "", dur_ms: null, note: "",
        });
      } catch (err) { noteError("context", err); }
    });
    await ctx.session.hook("http.request", async (e: any) => {
      const sid = String(e.sessionID ?? "");
      try {
        if (!sid || disabled(sid) || !isGufo(sid)) return;
        trace("http.request", { kind: e.kind, keys: Object.keys(e ?? {}), method: e.request?.method, url: e.request?.url });
        e.request.headers.set("x-session-id", sid);
        e.request.headers.set("x-jev-plugin", "1");
        if (e.kind !== "primary" || e.request.method !== "POST") return;
        const plan = takePlan(sid);
        if (!plan) return;
        const body = JSON.parse(await e.request.clone().text());
        const added = applyPlan(body, plan);
        if (added.length) {
          const headers = new Headers(e.request.headers);
          headers.delete("content-length");
          e.request = new Request(e.request.url, { method: "POST", headers, body: JSON.stringify(body) });
        }
        const entry = seen.get(sid) ?? { provider: "", model: "" };
        enqueue({
          ts: Date.now() / 1000, plugin_version: PLUGIN_VERSION, feature: "effort_apply",
          session: sid, call: "", provider: entry.provider, model: entry.model,
          decision: { plan, added }, segment: "", cmd: "", dur_ms: null, note: "",
        });
      } catch (err) { noteError("http.request", err); }
    });
    // The /jev command template runs jev-ctl.sh through a shell; give that shell the session id.
    try {
      await ctx.shell.hook("create.before", (e: any) => {
        trace("shell.create.before", { keys: Object.keys(e ?? {}), sessionID: e?.sessionID, envKeys: Object.keys(e?.env ?? {}).filter((k) => k.startsWith("OPENCODE")) });
        const sid = String(e?.sessionID ?? "");
        if (sid && e?.env && !e.env.OPENCODE_SESSION_ID) e.env.OPENCODE_SESSION_ID = sid;
      });
    } catch (err) { noteError("shell.hook", err); }
    await ctx.tool.hook("execute.before", async (e: any) => {
      const sid = String(e.sessionID ?? "");
      const call = String(e.id ?? e.callID ?? "");
      try {
        if (disabled(sid)) return;
        trace("tool.execute.before", e);
        injected("execute.before");
        if (e.tool !== "shell" && e.tool !== "bash") return;
        const gufoSession = isGufo(sid);
        const cmd = typeof e.input?.command === "string" ? e.input.command : "";
        const t0 = Date.now();
        let verdict = "unknown";
        let segment = "";
        try {
          const r = classify(cmd);
          verdict = r.verdict;
          segment = r.segment;
          if (verdict === "unknown") {
            const insp = inspectScripts(cmd);
            if (insp.reason) verdict = "heavy";
          }
        } catch { verdict = "unknown"; }
        let judged = false;
        if (verdict === "unknown" && GATE_ON && featureOn("command") && cmd) {
          judged = await judgeCommand(cmd, sid);
          if (judged) { verdict = "heavy"; segment = segment || cmd.slice(0, 200); }
        }
        const wouldStop = verdict === "heavy";
        let note = "";
        if (wouldStop && GATE_ON && featureOn("command")) {
          try {
            const g = await acquireHeavy(LOG_DIR, sid, call, segment || cmd, gufoSession);
            held.add(call);
            note = `gate waited ${g.waitedMs}ms prepared=${g.prepared}${g.note ? " " + g.note : ""}`;
          } catch (err) { noteError("heavy-gate", err); }
        }
        const entry = seen.get(sid) ?? { provider: "", model: "" };
        const safeCmd = redact(cmd);
        pending.set(call, { verdict, would_stop: wouldStop, segment, session: sid, provider: entry.provider, model: entry.model, cmd: safeCmd, t: t0 });
        const ts = Date.now() / 1000;
        enqueue({
          ts, plugin_version: PLUGIN_VERSION, feature: "command_gate",
          session: sid, call, provider: entry.provider, model: entry.model,
          decision: { verdict, would_stop: wouldStop, gufo_session: gufoSession }, segment: segment.slice(0, 500),
          cmd: safeCmd, dur_ms: null, note,
        });
        if (gufoSession) {
          const logit = verdict === "heavy" ? 1.0 : verdict === "light" ? -1.0 : 0.0;
          sqliteInsert({
            ts: Math.floor(ts), model: `${entry.provider}/${entry.model}`.toLowerCase(), battery: "",
            trigger: "plugin-command", prompt_hash: createHash("sha256").update(segment || cmd).digest("hex"),
            score: wouldStop ? "yes" : "no", logit_gap: logit, duration_ms: Date.now() - t0,
            threshold: "", framing_ver: "plugin-v0", action: wouldStop ? "stop" : "none",
            request_id: call, session_id: sid, question: safeCmd,
          });
        }
      } catch (err) { noteError("execute.before", err); /* fail open: never block the tool */ }
    });
    await ctx.tool.hook("execute.after", async (e: any) => {
      const sid = String(e.sessionID ?? "");
      const call = String(e.id ?? e.callID ?? "");
      if (held.delete(call)) releaseHeavy(LOG_DIR, sid, call);
      try {
        if (disabled(sid)) return;
        trace("tool.execute.after", e);
        injected("execute.after");
        if (e.tool !== "shell" && e.tool !== "bash") return;
        const p = pending.get(call);
        if (!p) return;
        pending.delete(call);
        const entry = seen.get(sid) ?? { provider: "", model: "" };
        enqueue({
          ts: Date.now() / 1000, plugin_version: PLUGIN_VERSION, feature: "command_gate",
          session: sid, call, provider: entry.provider, model: entry.model,
          decision: { verdict: p.verdict, would_stop: p.would_stop }, segment: p.segment.slice(0, 500),
          cmd: p.cmd, dur_ms: Date.now() - p.t, note: String(e.status ?? e.result?.status ?? "").slice(0, 100),
        });
      } catch (err) { noteError("execute.after", err); }
    });
    async function injectPrompt(sid: string, text: string): Promise<void> {
      injectedTexts.add(`${sid}\n${text}`);
      try { await ctx.session.prompt({ sessionID: sid, text, delivery: "queue" }); }
      catch (err) { injectedTexts.delete(`${sid}\n${text}`); noteError("inject", err); }
    }
    const subagent = new Map<string, boolean>();
    async function isSubagent(sid: string): Promise<boolean> {
      const known = subagent.get(sid);
      if (known !== undefined) return known;
      let child = false;
      try { const info: any = await ctx.session.get({ sessionID: sid }); child = !!(info?.parentID ?? info?.parentId ?? info?.parent); } catch { child = false; }
      subagent.set(sid, child);
      return child;
    }
    // Turn end: finish detector and goal tracker.
    const lastTurnEnd = new Map<string, number>();
    const ctrl = new AbortController();
    void (async () => {
      try {
        for await (const ev of ctx.event.subscribe({ signal: ctrl.signal })) {
          try {
            if (disabled()) continue;
            injected("event");
            trace("event", ev);
            const t = (ev as any)?.type as string;
            if (t !== "session.idle" && t !== "session.execution.succeeded" && t !== "session.step.ended") continue;
            const sid = (ev as any)?.data?.sessionID as string | undefined;
            if (sid && t !== "session.step.ended") releaseSessionLocks(LOG_DIR, sid);
            if (!sid || !isGufo(sid) || disabled(sid)) continue;
            // Several events fire for one turn end: handle the first, ignore the
            // rest for 1.5 s. A later turn in the same session is handled again.
            const nowMs = Date.now();
            const dkey = `${sid}:${t === "session.step.ended" ? "s" : "e"}`;
            if (nowMs - (lastTurnEnd.get(dkey) ?? 0) < 1500) continue;
            lastTurnEnd.set(dkey, nowMs);
            if (lastTurnEnd.size > 500) lastTurnEnd.clear();
            const entry = seen.get(sid)!;
            enqueue({
              ts: nowMs / 1000, plugin_version: PLUGIN_VERSION, feature: "activity",
              session: sid, call: "", provider: entry.provider, model: entry.model,
              decision: { phase: "end" }, segment: "", cmd: "", dur_ms: null, note: t,
            });
            let task = "";
            let response = "";
            let hadTool = false;
            try {
              const msgs = await ctx.session.context({ sessionID: sid });
              const arr = (msgs as any) ?? [];
              // Message shapes: V2 flattened {type, text/content} or V1 {info, parts}.
              const textOf = (m: any): { text: string; hasTool: boolean } => {
                const content = (m.content ?? m.parts ?? []) as any[];
                const texts = content.filter((p: any) => p?.type === "text").map((p: any) => String(p.text ?? ""));
                const hasTool = content.some((p: any) => String(p?.type ?? "").startsWith("tool"));
                return { text: texts.join("\n"), hasTool };
              };
              const roleOf = (m: any): string => String(m.type ?? m.role ?? m.info?.role ?? "");
              for (let i = arr.length - 1; i >= 0; i--) {
                const m = arr[i];
                const role = roleOf(m);
                if (role === "assistant") {
                  if (response) break;
                  const { text, hasTool } = textOf(m);
                  response = text;
                  hadTool = hadTool || hasTool || String(m.finish ?? "") === "tool" || Array.isArray(m.tool_calls) && (m.tool_calls as any[]).length > 0;
                } else if (role === "user" && !task) {
                  task = typeof m.text === "string" ? m.text : textOf(m).text;
                }
                if (response && task) break;
              }
              // If the last assistant turn itself carried a tool call, silence.
              // (Healthy: had_tool_call means unfinished never fires.)
            } catch { /* unreadable history: skip this idle */ continue; }
            const verdict = judgeFinish(task, response, hadTool, "stop");
            let acted = "";
            if (t !== "session.step.ended" && !hadTool && !(await isSubagent(sid))) {
              const g = goals.get(sid);
              if (g && featureOn("goal")) {
                const hist = toGoalMessages(lastMsgs.get(sid) ?? []);
                let taskIndex = 0;
                for (let i = hist.length - 1; i >= 0; i--) if (hist[i].role === "user" && hist[i].text.startsWith(g.text.slice(0, 60))) { taskIndex = i; break; }
                const spec = { active: true, text: g.text, index: taskIndex, taskIndex, items: extractItems(g.text).length, fromHistory: false };
                const dec = evaluateGoal(spec, hist, response, g.continues);
                enqueue({
                  ts: Date.now() / 1000, plugin_version: PLUGIN_VERSION, feature: "goal_eval",
                  session: sid, call: "", provider: entry.provider, model: entry.model,
                  decision: { action: dec.action, items: dec.items, open: dec.open, rule: dec.rule, reason: dec.reason, continues: g.continues },
                  segment: "", cmd: "", dur_ms: null, note: dec.message.slice(0, 200),
                });
                if (dec.action === "continue" || dec.action === "nudge") {
                  g.continues++;
                  acted = "goal";
                  await injectPrompt(sid, dec.message);
                }
              }
              if (!acted && featureOn("finish") && verdict.unfinished) {
                const fs = finishes.get(sid) ?? { continues: 0, lastReason: "", lastTail: "" };
                const tail = response.slice(-FINISH_TAIL_CHARS);
                if (fs.continues >= FINISH_MAX_CONTINUES || (verdict.reason === fs.lastReason && tail === fs.lastTail)) {
                  acted = "finish-loop-guard";
                } else {
                  fs.continues++; fs.lastReason = String(verdict.reason); fs.lastTail = tail;
                  finishes.set(sid, fs);
                  acted = "finish";
                  await injectPrompt(sid, "Continue; the task is not complete. " + verdict.reason);
                }
              }
            }
            enqueue({
              ts: Date.now() / 1000, plugin_version: PLUGIN_VERSION, feature: "finish",
              session: sid, call: "", provider: entry.provider, model: entry.model,
              decision: { would_continue: verdict.unfinished, rule: verdict.unfinished ? String(verdict.reason) : "", score: verdict.score, acted },
              segment: "", cmd: redact(response.slice(0, 500)), dur_ms: null, note: "",
            });
          } catch (err) { noteError("event", err); }
        }
      } catch { /* subscription closed */ }
    })();
    return () => { flush(); ctrl.abort(); };
  },
};
