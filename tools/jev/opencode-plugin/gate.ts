// Port of jev_gate.cpp ClassifyJevAgentTurn + turn building (Phase 0, log-only).
// Only the per-turn policy is ported; the request-level ClassifyJevRequest
// (xhigh/low/medium pre-gate) is a separate engine feature and out of scope.
export interface AgentTurnResult { tool: string; command: string; output: string }
export interface AgentDecision { lower: boolean; reason: string; blocker: string }

const K_AGENT_SHELL_OUTPUT_MAX = 4000;
const K_AGENT_READ_HEAD = 160;
const K_AGENT_OTHER_HEAD = 400;
export const K_AGENT_MAX_RUN = 5;

const FAIL_WORDS = [
  "error", "failed", "failure", "traceback", "exception", "fatal", "panic",
  "segfault", "assertion", "no such file", "not found", "permission denied",
  "undefined reference", "timed out", "timeout", "command not found", "exit code",
];
const READ_TOOLS = new Set([
  "read", "filesystem_read_multiple_files", "glob", "grep", "list", "ls",
  "filesystem_directory_tree",
]);
const LIGHT_TOOLS = new Set([
  "todowrite", "todoread", "edit", "write", "memory_store_context",
  "memory_multi_search_context", "get_goal",
]);
const INSPECT = new Set([
  "ls", "cat", "head", "tail", "wc", "stat", "pwd", "echo",
  "find", "grep", "rg", "tree", "file", "nl", "du", "df",
  "which", "sleep", "test", "[", "date", "ps", "pgrep", "git",
  "sed", "awk", "sort", "uniq", "diff",
]);
const GIT_READ = new Set(["status", "log", "diff", "show", "branch", "ls-files", "rev-parse", "remote"]);

function agentWordChar(c: string): boolean { return /[a-zA-Z0-9_]/.test(c); }
function agentFold(t: string): string { return t.toLowerCase(); }
function agentHasFailure(folded: string): boolean {
  for (const word of FAIL_WORDS) {
    for (let pos = folded.indexOf(word, 0); pos !== -1; pos = folded.indexOf(word, pos + 1)) {
      const left = pos === 0 || !agentWordChar(folded[pos - 1]);
      const next = pos + word.length;
      const right = next >= folded.length || !agentWordChar(folded[next]);
      if (!left || !right) continue;
      if (word !== "exit code") return true;
      if (next + 1 < folded.length && folded[next] === " " && folded[next + 1] >= "1" && folded[next + 1] <= "9") return true;
    }
  }
  return false;
}
function firstWord(text: string): { word: string; end: number } {
  let i = 0;
  while (i < text.length && /\s/.test(text[i])) i++;
  if (i < text.length && text[i] === "[") return { word: "[", end: i + 1 };
  const start = i;
  while (i < text.length && agentWordChar(text[i])) i++;
  return { word: text.slice(start, i), end: i };
}
function agentInspectCommand(command: string): boolean {
  let cmd = command;
  let end = 0;
  let word = firstWord(cmd).word;
  end = firstWord(cmd).end;
  if (word === "cd") {
    const amp = cmd.indexOf("&&", end);
    const semi = cmd.indexOf(";", end);
    let sep = amp;
    let sepLen = 2;
    if (sep === -1 || (semi !== -1 && semi < sep)) { sep = semi; sepLen = 1; }
    if (sep === -1) return false;
    cmd = cmd.slice(sep + sepLen);
    word = firstWord(cmd).word;
    end = firstWord(cmd).end;
  }
  if (!INSPECT.has(word)) return false;
  if (word === "git") {
    const rest = cmd.slice(end);
    const sub = firstWord(rest).word;
    return GIT_READ.has(sub);
  }
  if (word === "sed") return cmd.slice(end).includes("-n");
  return true;
}
function agentIsShell(tool: string): boolean { return tool === "bash" || tool === "shell"; }
function agentBlocker(r: AgentTurnResult): string {
  const tool = r.tool;
  if (agentIsShell(tool)) {
    if (!agentInspectCommand(r.command)) {
      let command = r.command;
      let { word, end } = firstWord(command);
      if (word === "cd") {
        const amp = command.indexOf("&&", end);
        const semi = command.indexOf(";", end);
        const sep = Math.min(amp === -1 ? Infinity : amp, semi === -1 ? Infinity : semi);
        if (sep !== Infinity) {
          command = command.slice(sep + (amp === sep ? 2 : 1));
          word = firstWord(command).word;
        }
      }
      return "cmd=" + word.slice(0, 24);
    }
    if (r.output.length >= K_AGENT_SHELL_OUTPUT_MAX) return "output-large tool=" + tool;
    if (agentHasFailure(agentFold(r.output))) return "failure-signal tool=" + tool;
    return "";
  }
  if (READ_TOOLS.has(tool)) {
    return agentHasFailure(agentFold(r.output.slice(0, K_AGENT_READ_HEAD))) ? "failure-signal tool=" + tool : "";
  }
  if (LIGHT_TOOLS.has(tool)) {
    return agentHasFailure(agentFold(r.output.slice(0, K_AGENT_OTHER_HEAD))) ? "failure-signal tool=" + tool : "";
  }
  return "tool=" + tool;
}
function agentRoutine(r: AgentTurnResult): boolean { return agentBlocker(r) === ""; }
function agentTurnRoutine(turn: AgentTurnResult[]): boolean {
  return turn.length > 0 && turn.every(agentRoutine);
}

export function classifyJevAgentTurn(turns: AgentTurnResult[][]): AgentDecision {
  if (turns.length === 0) return { lower: false, reason: "no-tool-result", blocker: "" };
  if (!agentTurnRoutine(turns[turns.length - 1])) {
    const turn = turns[turns.length - 1];
    let blocked = 0;
    let first = "";
    for (const r of turn) {
      const why = agentBlocker(r);
      if (!why) continue;
      if (blocked++ === 0) first = why;
    }
    return { lower: false, reason: "tool-result-not-routine", blocker: `${first} blocked=${blocked}/${turn.length}` };
  }
  let run = 0;
  for (let i = turns.length - 1; i-- > 0;) {
    if (!agentTurnRoutine(turns[i])) break;
    run++;
  }
  if (run >= K_AGENT_MAX_RUN) return { lower: false, reason: "run-limit", blocker: `run=${run}` };
  return { lower: true, reason: "routine-tool-result", blocker: "" };
}

export const K_ROUTINE_BUDGET_TOKENS = 512;

// Port of ClassifyJevRequest (src/models/qwen38_flash_next/jev_gate.cpp): user-turn pre-gate.
const SCAN_LIMIT = 64 * 1024;
const TRIVIAL_MAX_CHARS = 160;
const HARD_MIN_CHARS = 2000;
const DIFF_FILES_FOR_XHIGH = 2;
const STACK_MARKERS = ["traceback (most recent call last)", "panic:", "segmentation fault", "core dumped", "abort()", "caused by:", "exception:", "assertionerror", "undefined reference", "runtime error", "stack trace", "segfault", "kernel panic", "error[", "fatal error", "\tat "];
const ERROR_LOCATIONS = [".cpp:", ".hpp:", ".cc:", ".py:", ".rs:", ".go:", ".java:", ".js:", ".ts:", ".rb:"];
const DEBUG_PHRASES = ["fix this", "fix the bug", "why is", "why does", "doesn't work", "does not work", "not working", "won't start", "root cause", "misbehav"];
const DEBUG_WORDS = ["debug", "broken", "regression", "repro", "failing", "crashing"];
// The engine's kWorkPhrases array has 5 slots and 4 entries; the empty fifth matches every text.
const WORK_PHRASES = ["add support", "change the", "update the", "write a", ""];
const WORK_WORDS = ["implement", "refactor", "migrate", "optimize", "optimise", "fix", "port", "rename"];
const LOOKUP_PHRASES = ["what is", "what's", "what are", "what does", "what do", "meaning of", "definition of", "how many", "how much"];
const LOOKUP_WORDS = ["where", "when", "which", "translate", "list", "show", "convert", "spell", "define", "summarize", "version"];
const SMALL_TALK = ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "yes", "no", "nice", "great", "cool"];

const isWordChar = (c: string): boolean => /[a-z0-9_]/.test(c);
const hasAny = (t: string, list: string[]): boolean => list.some((p) => t.includes(p));
function hasWord(t: string, w: string): boolean {
  for (let pos = t.indexOf(w); pos !== -1; pos = t.indexOf(w, pos + 1)) {
    const left = pos === 0 || !isWordChar(t[pos - 1]);
    const next = pos + w.length;
    const right = next >= t.length || !isWordChar(t[next]);
    if (left && right) return true;
  }
  return false;
}
const hasAnyWord = (t: string, list: string[]): boolean => list.some((w) => hasWord(t, w));
function count(t: string, needle: string): number {
  let n = 0;
  for (let pos = t.indexOf(needle); pos !== -1; pos = t.indexOf(needle, pos + needle.length)) n++;
  return n;
}
const hasCode = (t: string): boolean => t.includes("```") || t.includes("~~~") || t.includes("diff --git") || t.includes("<function=");
function isSmallTalk(t: string): boolean {
  const trimmed = t.replace(/^[ \t\n\r!?.]+|[ \t\n\r!?.]+$/g, "");
  return trimmed === "" || SMALL_TALK.includes(trimmed);
}
function isTrivial(f: string): boolean {
  if (f.length > TRIVIAL_MAX_CHARS || hasCode(f)) return false;
  return isSmallTalk(f) || f.includes("?") || hasAny(f, LOOKUP_PHRASES) || hasAnyWord(f, LOOKUP_WORDS);
}
function isHard(f: string, code: boolean): boolean {
  return f.length >= HARD_MIN_CHARS || hasAny(f, STACK_MARKERS) || hasAny(f, ERROR_LOCATIONS) || hasAny(f, DEBUG_PHRASES) ||
    hasAnyWord(f, DEBUG_WORDS) || count(f, "diff --git") >= DIFF_FILES_FOR_XHIGH ||
    (code && (hasAny(f, WORK_PHRASES) || hasAnyWord(f, WORK_WORDS)));
}

export interface RequestDecision { effort: "low" | "medium" | "xhigh"; needsThinking: boolean }

export function classifyJevRequest(text: string): RequestDecision {
  const folded = text.slice(0, SCAN_LIMIT).replace(/[A-Z]/g, (c) => c.toLowerCase());
  const code = hasCode(folded);
  if (isHard(folded, code)) return { effort: "xhigh", needsThinking: true };
  if (isTrivial(folded)) return { effort: "low", needsThinking: false };
  return { effort: "medium", needsThinking: true };
}
