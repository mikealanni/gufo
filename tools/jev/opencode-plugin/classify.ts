// Port of tools/jev/guard.py classify() to TypeScript (Phase 0, log-only).
// Names and constants are kept identical to the Python source so the two
// implementations can be diffed. Do NOT call the NPU judge here: a command
// the rules cannot decide stays `unknown` (Phase 0 rule 2).
import { readFileSync } from "node:fs";
import { basename, isAbsolute, join, normalize } from "node:path";
import { homedir } from "node:os";

export const HEAVY: RegExp[] = [
  /\bgufo(\s+\S+)*?\s+(bench|serve|judge|diagnose|eval|transcribe|prompt)\b/,
  /build\/\S+\/gufo\b/,
  /\bctest\b/,
  /\brocprof\w*/,
  /model-bench\.py/,
  /^(?:\S*\/)?(?:python3?|bash|sh)\s+(?!-c\b)\S*tools\/prof\/prof\.py\s+run\b/,
  /^(?:\.\/)?tools\/prof\/prof\.py\s+run\b/,
  /^(?:\S*\/)?(?:python3?|bash|sh)\s+(?!-c\b)\S*tools\/bench\//,
  /^(?:\.\/)?tools\/bench\//,
  /\bpytest\b/,
  /^(?:\S*\/)?python3?\s+(?!-c\b)\S+.*\b(torch|hip|rocm)\b/,
  /run-gufo\.sh/,
  /(?:^|\s)(?:\.\/)?build\/[\w.-]+\/(?:\S*\/)?tests\/\S*(?:_test|_tests|_bench|_probe)\b/,
];
export const LIGHT_SCRIPTS = new Set(["chroma-helper.py"]);
// Compiling uses CPU and RAM only, and swap covers the memory, so a build never
// stops gufo. Running the built tests still does (ctest and the test binaries).
export const BUILD_ONLY = /^(?:\S*\/)?(?:cmake\s+--build|ninja|make|nix\s+build|nix\s+develop\s+-c\s+(?:cmake\s+--build|ninja|make))\b/;

export const LIGHT_WORDS = new Set([
  "git", "ls", "cat", "rg", "grep", "sed", "head", "tail", "echo", "pwd", "cd",
  "find", "wc", "diff", "mkdir", "cp", "mv", "rm", "touch", "curl", "tr", "sort",
  "uniq", "awk", "stat", "file", "which", "date", "printf", "true", "less", "more",
  "tee", "xxd", "sha256sum", "md5sum", "sha1sum", "strings", "nm", "objdump", "readelf", "ldd", "nl", "cut", "basename", "dirname", "readlink", "realpath", "du",
  "df", "ps", "pgrep", "sleep", "wait", "kill", "tar", "unzip", "gh", "jj",
  "read", "export", "test", "[", "[[", "local", "return", "exit", "break", "continue",
]);

export const SECRET_KEYS = [
  "sk-", "ghp_", "gho_", "ghs_", "github_pat_", "AKIA", "xoxb-",
  "xoxp-", "AIza", "BEGIN PRIVATE KEY", "BEGIN OPENSSH PRIVATE KEY",
  "-----BEGIN", "Bearer eyJ", "eyJhbGciOiJ",
];

export function containsSecret(text: string): boolean {
  return SECRET_KEYS.some((k) => text.includes(k));
}

const HEREDOC = /<<-?\s*(['"]?)([A-Za-z_]\w*)\1[^\n]*\n([\s\S]*?)\n[ \t]*\2[ \t]*(?=\n|$)/g;
const NL_MARK = "\x01";

function protectHeredocs(command: string): string {
  return command.replace(HEREDOC, (m) => m.split("\n").join(NL_MARK));
}

export function segments(commandIn: string): string[] {
  const command = protectHeredocs(commandIn.split("\\\n").join(" "));
  const out: string[] = [];
  let cur = "";
  let quote = "";
  for (let i = 0; i < command.length; i++) {
    const ch = command[i];
    if (quote) {
      if (ch === "\\" && quote === '"' && i + 1 < command.length) {
        cur += command.slice(i, i + 2);
        i += 1;
        continue;
      }
      if (ch === quote) quote = "";
    } else if (ch === "'" || ch === '"') {
      quote = ch;
    } else if (ch === "&" && ((cur.length > 0 && (cur[cur.length - 1] === "<" || cur[cur.length - 1] === ">")) || command[i + 1] === ">")) {
      // 2>&1, >&2, &>file: a redirect, not a separator
    } else if (ch === ";" || ch === "|" || ch === "&" || ch === "\n") {
      out.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out.map((s) => s.trim()).filter((s) => s.length > 0).map((s) => s.split(NL_MARK).join("\n"));
}

export function firstWord(segment: string): string {
  for (const word of segment.split(/\s+/)) {
    if (!word) continue;
    if (word.includes("=") && !word.startsWith("-")) continue;
    return basename(word);
  }
  return "";
}

const ASSIGN = /^([A-Za-z_][A-Za-z0-9_]*)=(\S*)$/;
const SUBST = /^[A-Za-z_]\w*=\$\((.*?)\)?$/;
const QUOTED = /"[^"]*"|'[^']*'/g;
const HOME = homedir();

export function expand(seg: string, env: Record<string, string>): string {
  for (const [name, value] of Object.entries(env)) {
    seg = seg.split("${" + name + "}").join(value).split("$" + name).join(value);
  }
  return seg.split("~/").join(HOME + "/");
}

const LAUNCHER = /^(?:(?:nohup|setsid|time|nice|ionice|stdbuf|env|command|exec)(?:\s+-\S+)*\s+|[A-Za-z_]\w*=\S*\s+)+/;

export function stripLaunchers(seg: string): [string, boolean] {
  const m = seg.trim().match(LAUNCHER);
  if (!m) return [seg, false];
  const detached = /\b(nohup|setsid)\b/.test(m[0]);
  return [seg.trim().slice(m[0].length), detached];
}

// Learned-heavy list (substring match, reload at most every 10 s).
export const LEARNED_FILE = `${HOME}/.gufo/jev-log/learned-heavy.json`;
const LEARNED_MAX = 200;
const LEARNED_TTL = 30 * 86400;
let LEARNED: Record<string, { last?: number }> = {};
let LEARNED_AT = 0;

export function loadLearned(): void {
  try {
    const data = JSON.parse(readFileSync(LEARNED_FILE, "utf8"));
    const now = Date.now() / 1000;
    LEARNED = {};
    for (const [k, v] of Object.entries<any>(data)) {
      if (v && typeof v === "object" && now - (v.last ?? 0) < LEARNED_TTL) LEARNED[k] = v;
    }
    LEARNED_AT = now;
  } catch { /* missing/invalid file = no learned list */ }
}

function maybeReloadLearned(): void {
  if (Date.now() / 1000 - LEARNED_AT > 10) {
    try { loadLearned(); } catch { /* ignore */ }
  }
}

const SHELL_KEYWORDS = new Set(["do", "then", "else", "elif", "if", "while", "until", "!", "{", "("]);
const END_WORDS = new Set(["done", "fi", "esac", "}", ")"]);
const LOOP_HEADER = /^(?:for\s+\w+\s+in\b|for\s+\(\()/;

function stripShellKeywords(segIn: string): string {
  let seg = segIn;
  for (;;) {
    const word = seg.split(/\s+/, 1)[0] ?? "";
    if (word && SHELL_KEYWORDS.has(word)) {
      seg = seg.slice(word.length).replace(/^\s+/, "");
      continue;
    }
    return seg;
  }
}

export function classify(command: string): { verdict: "heavy" | "light" | "unknown"; segment: string } {
  maybeReloadLearned();
  for (const key of Object.keys(LEARNED)) {
    if (key && command.includes(key)) return { verdict: "heavy", segment: command };
  }
  let unknown = "";
  const env: Record<string, string> = {};
  for (const rawSeg of segments(command)) {
    let seg = expand(rawSeg, env);
    const m = seg.trim().replace(/^["']|["']$/g, "").match(ASSIGN);
    if (m) {
      env[m[1]] = m[2].replace(/^["']|["']$/g, "");
      continue;
    }
    const sub = seg.match(SUBST);
    if (sub) {
      const inner = classify(sub[1]);
      if (inner.verdict === "heavy") return inner;
      continue;
    }
    const [stripped, detached] = stripLaunchers(seg);
    seg = stripShellKeywords(stripped);
    if (!seg || END_WORDS.has(firstWord(seg))) continue;
    if (LOOP_HEADER.test(seg) && !seg.includes("$(") && !seg.includes("`")) continue;
    if (LIGHT_WORDS.has(firstWord(seg))) continue;
    if (executedScripts(seg).some((x) => LIGHT_SCRIPTS.has(basename(x)))) continue;
    if (BUILD_ONLY.test(seg)) continue;
    if (detached && seg && !/\s(--help|-h|--version)(\s|$)/.test(seg)) {
      return { verdict: "heavy", segment: seg };
    }
    if (/\s(--help|-h|--version)(\s|$)/.test(seg)) continue;
    const shellC = ["bash", "sh", "eval", "xargs", "timeout", "nohup"].includes(firstWord(seg));
    const strippedQuotes = seg.replace(QUOTED, '""');
    if (HEAVY.some((p) => (shellC ? p.test(seg) : p.test(strippedQuotes)))) {
      return { verdict: "heavy", segment: seg };
    }
    if (!unknown) unknown = seg;
  }
  return unknown
    ? { verdict: "unknown", segment: unknown }
    : { verdict: "light", segment: "" };
}

// needle_for / script inspection (kept for parity; log-only still reads
// script files — bounded, fail-open).
export const SHELL_WORDS = new Set(["do", "then", "else", "elif", "if", "while", "until", "time", "exec", "nohup", "{", "(", "!"]);
const REDIRECT = /\s\d*[<>]|\s&>/;

function shlexSplit(text: string): string[] {
  const out: string[] = [];
  let cur = "";
  let quote = "";
  let has = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quote) {
      if (ch === "\\" && quote === '"' && i + 1 < text.length) {
        cur += text[i + 1];
        i += 1;
        continue;
      }
      if (ch === quote) { quote = ""; continue; }
      cur += ch;
      continue;
    }
    if (ch === "\\" && i + 1 < text.length) {
      cur += text[i + 1];
      i += 1;
      has = true;
      continue;
    }
    if (ch === "'" || ch === '"') { quote = ch; has = true; continue; }
    if (ch === " " || ch === "\t") {
      if (has || cur) out.push(cur);
      cur = "";
      has = false;
      continue;
    }
    cur += ch;
    has = true;
  }
  if (has || cur) out.push(cur);
  if (quote) throw new Error("No closing quotation");
  return out;
}

export function needleFor(segment: string): string {
  const seg = stripLaunchers(segment.split(REDIRECT)[0])[0];
  let tokens: string[];
  try { tokens = shlexSplit(seg); } catch { tokens = seg.split(/\s+/).filter(Boolean); }
  while (tokens.length && (SHELL_WORDS.has(tokens[0]) || (tokens[0].includes("=") && !tokens[0].startsWith("-")))) tokens.shift();
  const keep: string[] = [];
  for (const tok of tokens.slice(0, 6)) {
    if (tok.startsWith("~") || tok.includes("$") || tok.includes("`")) break;
    keep.push(tok);
  }
  return keep.join(" ") || segment.slice(0, 120);
}

export const SCRIPT_RE = /(?:(?<=\s)|^)(~?\/[^\s'"<>|;&]+\.(?:py|sh))/g;
export const HEAVY_CONTENT = /import\s+torch|from\s+torch|\bhipMalloc|\bhipLaunch|libamdhip|\bcuda\b|rocm|(?<!\.)\bgufo\b|llama[._-]?cpp|\bvllm\b|safetensors|from_pretrained|\bcmake\s+--build|\bninja\b|\bmake\s+-j|\/dev\/kfd|\bgguf\b/i;
const INTERPRETERS = new Set(["python", "python3", "bash", "sh", "zsh", "node"]);
const SCRIPT_BYTES = 65536;

export function executedScripts(command: string): string[] {
  const out: string[] = [];
  let cwd: string | null = null;
  for (const rawSeg of segments(command)) {
    const seg = stripLaunchers(rawSeg)[0];
    let tokens: string[];
    try { tokens = shlexSplit(seg); } catch { tokens = seg.split(/\s+/).filter(Boolean); }
    if (!tokens.length) continue;
    if (tokens[0] === "cd") {
      const dir = tokens.slice(1).find((t) => !t.startsWith("-"));
      cwd = dir ? (dir.startsWith("~") ? `${HOME}/${dir.slice(2)}` : dir) : null;
      continue;
    }
    const head = basename(tokens[0]).replace(/[\d.]+$/, "");
    let target: string | undefined;
    if (INTERPRETERS.has(head)) {
      const rest = tokens.slice(1);
      if (rest.length && rest[0] === "-c") continue;
      const filtered = rest.filter((t) => !t.startsWith("-"));
      target = filtered[0];
    } else {
      target = tokens[0];
    }
    if ((target && /\.(py|sh)$/.test(target)) || (target && (target.startsWith("./") || target.startsWith("~/") || (isAbsolute(target) && target.startsWith("/"))))) {
      let path = target.startsWith("~") ? `${HOME}/${target.slice(2)}` : target;
      if (cwd && !isAbsolute(path)) path = normalize(join(cwd, path));
      out.push(path);
    }
  }
  return out;
}

export function inspectScripts(command: string, depth = 2, seen = new Set<string>()): { reason: string | null; excerpt: string } {
  let excerpt = "";
  for (const path of executedScripts(command)) {
    if (seen.has(path)) continue;
    seen.add(path);
    let text: string;
    try {
      const buf = readFileSync(path);
      text = buf.slice(0, SCRIPT_BYTES).toString("utf8");
    } catch { continue; }
    excerpt = excerpt || text.slice(0, 1500);
    const m = HEAVY_CONTENT.exec(text);
    if (m) return { reason: `${basename(path)} mentions '${m[0]}'`, excerpt };
    if (depth > 1) {
      SCRIPT_RE.lastIndex = 0;
      const found: string[] = [];
      let r: RegExpExecArray | null;
      while ((r = SCRIPT_RE.exec(text)) !== null) found.push(r[1]);
      const inner = inspectScripts(found.join(" ; "), depth - 1, seen);
      if (inner.reason) return { reason: `${basename(path)} -> ${inner.reason}`, excerpt };
    }
  }
  return { reason: null, excerpt };
}
