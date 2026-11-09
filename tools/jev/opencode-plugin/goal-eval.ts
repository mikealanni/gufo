// Port of the goal tracker's rules (src/models/qwen38_flash_next/jev_goal.cpp): strip the
// trigger phrase, extract the asks, decide whether a finished turn left work open.
import { findTrigger, hasGoalTrigger } from "./goal.ts";

const RULE_OPEN_FILE = true; // last-failure, loop-nudge, open-test and open-push rules stay off, as in the engine
const MAX_ITEMS = 8;
const MAX_CLAUSE_CHARS = 240;
const ITEM_CHARS = 80;
const FAILURE_WINDOW = 4096;
const LOOP_WINDOW = 60;
const ANSWER_MIN_CHARS = 10;
const CLAIM_WINDOW = 400;
const INTENT_WINDOW = 220;
const USER_LOOKBACK = 6;
export const GOAL_MAX_CONTINUES = 3;
const ID_LOOKBACK = 64;
const GOAL_ENDS = 1500;
const TRIVIAL_MAX_CHARS = 32;

const isAlnum = (c: string | undefined): boolean => !!c && /[A-Za-z0-9]/.test(c);
const fold = (s: string): string => s.replace(/[A-Z]/g, (c) => c.toLowerCase());
const contains = (f: string, n: string): boolean => f.includes(n);
function containsWord(f: string, w: string): boolean {
  for (let pos = f.indexOf(w); pos !== -1; pos = f.indexOf(w, pos + 1)) {
    if ((pos === 0 || !isAlnum(f[pos - 1])) && (pos + w.length >= f.length || !isAlnum(f[pos + w.length]))) return true;
  }
  return false;
}
const tailOf = (t: string, n: number): string => (t.length > n ? t.slice(t.length - n) : t);
const trim = (t: string): string => t.replace(/^[\s,;]+|[\s,;]+$/g, "");

function stripOnce(text: string): [string, boolean] {
  let out = "";
  let from = 0;
  let changed = false;
  for (;;) {
    const [begin, end] = findTrigger(text, from);
    if (begin === -1) break;
    changed = true;
    out += text.slice(from, begin);
    let stop = end;
    while (stop < text.length && " \t,.;:!?".includes(text[stop])) stop++;
    from = stop;
  }
  return [out + text.slice(from), changed];
}

export function stripGoalTrigger(text: string): string {
  if (!hasGoalTrigger(text)) return text;
  let cur = text;
  for (let round = 0; round < 4; round++) {
    const [next, changed] = stripOnce(cur);
    cur = next;
    if (!changed) break;
  }
  const t = trim(cur);
  return t === "" ? "Continue." : t;
}

const TRIVIAL = ["continue", "ok", "okay", "go", "go on", "go ahead", "yes", "yep", "do it", "proceed", "keep going", "continue please", "thanks", "thank you", "status", "next"];
export function trivialTurn(stripped: string): boolean {
  if (stripped.length > TRIVIAL_MAX_CHARS) return false;
  const v = fold(stripped).trim();
  return v === "" || TRIVIAL.includes(v);
}

// ---- evidence
const HARD_FAILURE = ["traceback (most recent", "error:", "fatal:", "assertionerror", "panic:", "undefined reference", "command not found", "no such file or directory", "syntax error", "npm err", "make: ***", "segmentation fault"];
const COUNT_FAILURE = ["failed", "failure", "errors"];

function benignCount(f: string, pos: number, markerLen: number): boolean {
  let b = pos;
  while (b > 0 && f[b - 1] === " ") b--;
  if (b >= 1 && f[b - 1] === "0" && (b === 1 || !isAlnum(f[b - 2]))) return true;
  if (b >= 2 && f.slice(b - 2, b) === "no" && (b === 2 || !isAlnum(f[b - 3]))) return true;
  let a = pos + markerLen;
  while (a < f.length && (f[a] === ":" || f[a] === " " || f[a] === "=" || f[a] === "s")) a++;
  return a < f.length && f[a] === "0" && (a + 1 >= f.length || !isAlnum(f[a + 1]));
}
function hasMarker(f: string, marker: string, checkBenign: boolean): boolean {
  for (let pos = f.indexOf(marker); pos !== -1; pos = f.indexOf(marker, pos + 1)) {
    if (!checkBenign || !benignCount(f, pos, marker.length)) return true;
  }
  return false;
}
function looksFailed(output: string): boolean {
  const f = fold(tailOf(output, FAILURE_WINDOW));
  if (HARD_FAILURE.some((m) => contains(f, m))) return true;
  return COUNT_FAILURE.some((m) => hasMarker(f, m, true));
}
function firstFailureLine(output: string): string {
  for (const line of tailOf(output, FAILURE_WINDOW).split("\n")) {
    if (looksFailed(line)) return trim(line.slice(0, 120));
  }
  return "";
}

function stripCd(command: string): string {
  let cmd = command;
  for (;;) {
    cmd = cmd.replace(/^\s+/, "");
    if (!cmd.startsWith("cd ")) return cmd;
    const amp = cmd.indexOf("&&");
    const semi = cmd.indexOf(";");
    const sep = Math.min(amp === -1 ? Infinity : amp, semi === -1 ? Infinity : semi);
    if (sep === Infinity) return cmd;
    cmd = cmd.slice(sep + (sep === amp ? 2 : 1));
  }
}

type Kind = "other" | "test" | "build" | "commit" | "push";
function classifyCommand(command: string): Kind {
  const f = fold(stripCd(command));
  if (contains(f, "git commit")) return "commit";
  if (contains(f, "git push")) return "push";
  if (["pytest", "ctest", "jest", "vitest", "mocha", "rspec"].some((w) => containsWord(f, w)) ||
      ["unittest", "go test", "cargo test", "npm test", "npm run test", "pnpm test", "make test", "make check", "gradle test", "mvn test"].some((w) => contains(f, w)) ||
      containsWord(f, "test") || containsWord(f, "tests")) return "test";
  if (contains(f, "build") || contains(f, "cmake") || ["make", "ninja", "tsc", "gcc", "g++", "clang", "mvn", "gradle"].some((w) => containsWord(f, w)) ||
      contains(f, "cargo check") || contains(f, "compile")) return "build";
  return "other";
}
const isShell = (t: string): boolean => t === "bash" || t === "shell";
function shellWrites(command: string): boolean {
  const f = fold(stripCd(command));
  for (let i = 0; i < f.length; i++) {
    if (f[i] !== ">") continue;
    const fdDup = f[i + 1] === "&";
    const devnull = f.startsWith(">/dev/null", i) || f.startsWith("> /dev/null", i);
    if (!fdDup && !devnull && !(i > 0 && f[i - 1] === "-")) return true;
  }
  return ["sed -i", "tee ", "git apply", "pip install", "npm install", "cargo add", "git add", "git checkout", "git mv", "chmod"].some((w) => contains(f, w)) ||
    ["mv", "cp", "patch", "mkdir", "touch"].some((w) => containsWord(f, w));
}
const isDelegation = (t: string): boolean => t === "task" || t.startsWith("team_");
const WRITE_TOOLS = ["edit", "write", "multiedit", "patch", "apply_patch", "filesystem_write_file", "filesystem_edit_file"];
const isWriteTool = (t: string): boolean => WRITE_TOOLS.includes(t);

export interface GoalCall { id: string; name: string; command: string; path: string }
export interface GoalMessage {
  role: "user" | "assistant" | "tool";
  text: string;
  calls?: GoalCall[];
  toolCallId?: string;
}
interface Rec extends GoalCall { result: string; hasResult: boolean; kind: Kind; wrote: boolean }

function collectCalls(history: GoalMessage[], from: number): Rec[] {
  const records: Rec[] = [];
  for (let i = from; i < history.length; i++) {
    const m = history[i];
    if (m.role === "assistant") {
      for (const c of m.calls ?? []) {
        records.push({ ...c, result: "", hasResult: false,
          kind: isShell(c.name) ? classifyCommand(c.command) : "other",
          wrote: isWriteTool(c.name) || isDelegation(c.name) || (isShell(c.name) && shellWrites(c.command)) });
      }
    } else if (m.role === "tool") {
      const stop = Math.max(0, records.length - ID_LOOKBACK);
      for (let j = records.length; j-- > stop;) {
        if (records[j].id === m.toolCallId && !records[j].hasResult) { records[j].result = m.text; records[j].hasResult = true; break; }
      }
    }
  }
  return records;
}

// ---- items
type ItemKind = "change" | "file" | "test" | "build" | "commit" | "push" | "answer";
interface Item { kind: ItemKind; text: string; target: string; rule: string }
const CHANGE_VERBS = ["fix", "add", "implement", "create", "write", "change", "update", "modify", "refactor", "rename", "remove", "delete", "patch", "rewrite", "make", "build", "wire"];
const ANSWER_VERBS = ["explain", "describe", "tell", "summarize", "summarise", "why", "what", "how", "list", "show", "check", "review", "investigate", "find", "look", "read", "analyze", "analyse", "compare", "debug", "diagnose", "report", "which", "where"];
const FILE_EXTENSIONS = ["cpp", "hpp", "h", "c", "cc", "py", "js", "ts", "tsx", "jsx", "md", "json", "toml", "yaml", "yml", "sh", "txt", "rs", "go", "css", "html", "cmake", "hip", "java"];

const shortText = (t: string): string => trim(t.slice(0, ITEM_CHARS)).replace(/\n/g, " ");
function startsWithWordOf(f: string, words: string[]): boolean {
  const m = /^[^A-Za-z0-9]*([A-Za-z0-9]*)/.exec(f);
  return !!m && words.includes(m[1]);
}
const hasWordOf = (f: string, words: string[]): boolean => words.some((w) => containsWord(f, w));
function pathToken(clause: string): string {
  const ok = (c: string | undefined): boolean => !!c && (isAlnum(c) || c === "/" || c === "_" || c === "." || c === "-");
  let i = 0;
  while (i < clause.length) {
    while (i < clause.length && !ok(clause[i])) i++;
    let j = i;
    while (j < clause.length && ok(clause[j])) j++;
    let token = clause.slice(i, j);
    while (token.endsWith(".")) token = token.slice(0, -1);
    const dot = token.lastIndexOf(".");
    if (dot > 0 && dot + 1 < token.length && FILE_EXTENSIONS.includes(fold(token.slice(dot + 1)))) return token;
    i = j + 1;
  }
  return "";
}
const basename = (p: string): string => p.slice(p.lastIndexOf("/") + 1);
function splitClauses(text: string): string[] {
  const out: string[] = [];
  let cur = "";
  const flush = (): void => { const t = trim(cur); if (t.length >= 3) out.push(t); cur = ""; };
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    const sentenceEnd = (c === "." || c === "!" || c === "?") && (i + 1 >= text.length || /\s/.test(text[i + 1]));
    if (c === "\n" || c === ";" || sentenceEnd) { if (c === "?") cur += c; flush(); continue; }
    cur += c;
  }
  flush();
  return out;
}
function imperative(f: string, verb: string): boolean {
  if (f.startsWith(verb)) return true;
  return ["and ", "then ", "also ", "to ", ", ", "please ", "finally ", "&& "].some((b) => f.includes(b + verb));
}

export function extractItems(goalIn: string): Item[] {
  let goal = goalIn;
  if (goal.length > 2 * GOAL_ENDS) goal = goal.slice(0, GOAL_ENDS) + "\n" + goal.slice(goal.length - GOAL_ENDS);
  const items: Item[] = [];
  let test = false, build = false, commit = false, push = false, answer = false;
  for (const clause of splitClauses(goal)) {
    if (items.length >= MAX_ITEMS) break;
    if (clause.length > MAX_CLAUSE_CHARS) continue;
    const f = fold(clause);
    const negatedGit = ["no git", "not push", "no push", "dont push", "don't push", "not commit", "no commit", "dont commit", "don't commit", "without commit", "without push"].some((w) => contains(f, w));
    const wantsTest = contains(f, "test") && (["run", "rerun", "pass", "passes", "passing", "green", "verify", "ensure"].some((w) => containsWord(f, w)) || contains(f, "make sure") || f.startsWith("test"));
    const wantsBuild = (["build", "builds", "compile", "compiles"].some((w) => containsWord(f, w))) &&
      (["run", "ensure", "verify", "pass", "succeed", "succeeds", "works", "clean"].some((w) => containsWord(f, w)) || contains(f, "make sure") || f.startsWith("build") || f.startsWith("compile"));
    const wantsCommit = !negatedGit && imperative(f, "commit");
    const wantsPush = !negatedGit && imperative(f, "push");
    const change = startsWithWordOf(f, CHANGE_VERBS) || (containsWord(f, "please") && hasWordOf(f, CHANGE_VERBS) && !wantsTest && !wantsBuild);
    const reads = startsWithWordOf(f, ANSWER_VERBS) || clause.endsWith("?");
    const path = pathToken(clause);
    if (change && path) items.push({ kind: "file", text: shortText(clause), target: basename(path), rule: "open-file" });
    else if (change && !wantsTest && !wantsBuild && !wantsCommit && !wantsPush) items.push({ kind: "change", text: shortText(clause), target: "", rule: "open-change" });
    else if (reads && !change && !answer) { answer = true; items.push({ kind: "answer", text: shortText(clause), target: "", rule: "open-answer" }); }
    if (wantsTest && !test) { test = true; items.push({ kind: "test", text: "run the tests", target: "", rule: "open-test" }); }
    if (wantsBuild && !build) { build = true; items.push({ kind: "build", text: "build", target: "", rule: "open-build" }); }
    if (wantsCommit && !commit) { commit = true; items.push({ kind: "commit", text: "commit", target: "", rule: "open-commit" }); }
    if (wantsPush && !push) { push = true; items.push({ kind: "push", text: "push", target: "", rule: "open-push" }); }
  }
  return items;
}

const commandSucceeded = (r: Rec): boolean => r.hasResult && !looksFailed(r.result);

function itemClosed(item: Item, calls: Rec[], finalText: string): boolean {
  switch (item.kind) {
    case "change": return calls.some((c) => c.wrote);
    case "file": return calls.some((c) => (isWriteTool(c.name) && c.path.includes(item.target)) ||
      (c.wrote && !isWriteTool(c.name) && c.command.includes(item.target)) || isDelegation(c.name));
    case "test": case "build": {
      for (let i = calls.length - 1; i >= 0; i--) if (calls[i].kind === item.kind) return commandSucceeded(calls[i]);
      return false;
    }
    case "commit": case "push": {
      for (let i = calls.length - 1; i >= 0; i--) {
        const c = calls[i];
        if (c.kind !== item.kind || !c.hasResult) continue;
        if (!looksFailed(c.result) && !contains(fold(tailOf(c.result, FAILURE_WINDOW)), "nothing to commit")) return true;
      }
      return false;
    }
    case "answer": return finalText.length >= ANSWER_MIN_CHARS;
  }
}

const INTENT = ["i'll next", "i will next", "let me now", "next, i will", "next i will", "i'll now", "i will now", "now i will", "now i'll", "next, i'll", "next i'll", "i'm going to now", "let me start by"];
const DONE_WORDS = ["done", "complete", "completed", "finished", "implemented", "fixed", "all set", "ready", "working", "works", "passes", "passing", "successfully", "resolved", "created", "added", "updated"];
const ACK_PROBLEM = ["fail", "error", "can't", "cannot", "unable", "blocked", "couldn't", "could not", "not working", "doesn't", "does not", "still"];
const foldApostrophes = (t: string): string => fold(t.replace(/[’‘]/g, "'"));

export interface GoalSpec { active: boolean; text: string; index: number; taskIndex: number; items: number; fromHistory: boolean }

export function detectGoalSpec(messages: GoalMessage[]): GoalSpec {
  const spec: GoalSpec = { active: false, text: "", index: 0, taskIndex: 0, items: 0, fromHistory: false };
  let trigger = messages.length;
  for (let i = 0; i < messages.length; i++) if (messages[i].role === "user" && hasGoalTrigger(messages[i].text)) trigger = i;
  if (trigger === messages.length) return spec;
  spec.active = true; spec.index = trigger; spec.taskIndex = trigger;
  let haveText = false;
  let seen = 0;
  for (let i = trigger; i >= 0 && seen < USER_LOOKBACK; i--) {
    if (messages[i].role !== "user") continue;
    seen++;
    let view = messages[i].text;
    if (hasGoalTrigger(view)) view = stripGoalTrigger(view);
    if (view === "Continue." || trivialTurn(view)) continue;
    const n = extractItems(view).length;
    if (!haveText) { spec.text = view; haveText = true; }
    if (n > 0) { spec.text = view; spec.taskIndex = i; spec.items = n; spec.fromHistory = i !== trigger; break; }
  }
  return spec;
}

export interface GoalDecision { action: "none" | "done" | "continue" | "nudge"; items: number; open: number; rule: string; reason: string; message: string }

export function evaluateGoal(spec: GoalSpec, history: GoalMessage[], finalText: string, continues: number): GoalDecision {
  const d: GoalDecision = { action: "none", items: 0, open: 0, rule: "", reason: "", message: "" };
  if (!spec.active) return d;
  d.action = "done";
  const calls = collectCalls(history, spec.taskIndex);
  const items = extractItems(spec.text);
  d.items = items.length;
  const open: string[] = [];
  let openRule = "";
  for (const item of items) {
    if (item.kind === "test" || item.kind === "push" || (item.kind === "file" && !RULE_OPEN_FILE)) continue; // open-test / open-push rules are off
    if (!itemClosed(item, calls, finalText)) { open.push(item.text); if (!openRule) openRule = item.rule; }
  }
  d.open = open.length;
  let lastFailure = "";
  for (let i = calls.length - 1; i >= 0; i--) {
    if (calls[i].kind !== "test" && calls[i].kind !== "build") continue;
    if (calls[i].hasResult && looksFailed(calls[i].result)) lastFailure = firstFailureLine(calls[i].result);
    break;
  }
  const claim = foldApostrophes(tailOf(finalText, CLAIM_WINDOW));
  const intentText = foldApostrophes(tailOf(finalText, INTENT_WINDOW));
  const claimsDone = DONE_WORDS.some((w) => containsWord(claim, w) || contains(claim, w));
  const intent = INTENT.find((p) => contains(intentText, p)) ?? "";
  const emptyTurn = finalText.length < ANSWER_MIN_CHARS;
  if (continues >= GOAL_MAX_CONTINUES) return d;
  const noItems = items.length === 0;
  let rule = "";
  let reason = "";
  if (open.length && (claimsDone || emptyTurn)) { rule = openRule; reason = claimsDone ? "claims-done-without-evidence" : "empty-turn"; }
  else if (noItems && trim(finalText) === "") { rule = "empty-turn"; reason = "empty-turn"; }
  else if (intent) { rule = "intent"; reason = "announced-next-step"; }
  else { if (noItems) d.reason = "no-items-fallback"; return d; }
  if (noItems) reason += ",no-items-fallback";
  d.action = "continue"; d.rule = rule; d.reason = reason; d.message = "Continue.";
  if (open.length) d.message += " Still open: " + open.join("; ") + ".";
  if (lastFailure) d.message += ` Last failure: ${lastFailure}.`;
  if (!open.length && !lastFailure) d.message += " Do the step you announced instead of stopping.";
  return d;
}
