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
