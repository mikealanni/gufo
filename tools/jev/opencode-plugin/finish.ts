// Port of jev_finish.cpp (Phase 0, log-only). Constants and helpers mirror
// the C++ so the two can be diffed. Exported for the finish hook only.
export const K_MIN_PROSE_CHARS = 40;
export const K_MIN_WEAK_PROSE_CHARS = 80;
export const K_LONG_TASK_CHARS = 600;
export const K_TINY_ANSWER_CHARS = 20;
export const K_MIN_PLAN_ITEMS = 2;
export const K_SCORE_LENGTH = 1.0;
export const K_SCORE_EMPTY = 0.9;
export const K_SCORE_FENCE = 0.95;
export const K_SCORE_OPEN_END = 0.85;
export const K_SCORE_STOPWORD = 0.85;
export const K_SCORE_ANNOUNCE = 0.8;
export const K_SCORE_PLAN = 0.7;
export const K_SCORE_BRACE = 0.65;
export const K_SCORE_TINY = 0.45;
export const K_SCORE_BARE = 0.62;
export const K_SCORE_BARE_ITEM = 0.45;
const K_MIN_ITEM_WORDS = 10;
export const K_THRESHOLD = 0.6;

const DANGLING = new Set([
  "the", "a", "an", "and", "or", "but", "of", "to",
  "in", "for", "with", "that", "which", "is", "are", "as",
  "by", "on", "at", "from", "because", "if", "then", "so",
  "not", "this", "these", "those", "its", "their", "your", "my",
  "our", "will", "would", "can", "could", "should", "has", "have",
]);
const ANNOUNCEMENTS = [
  "i'll now ", "i will now ", "now i'll ", "now i will ",
  "next, i'll ", "next i'll ", "next, i will ", "next i will ",
  "now let me ", "let's now ", "i'm going to ", "i am going to ",
  "first, i'll ", "first i'll ", "i'll start ", "i'll begin ", "i'll go ahead ",
];
const LET_ME_BENIGN = ["know", "help", "be ", "say", "clarify", "just "];
const HANDOFF = new Set([
  "you", "your", "you're", "you've", "you'll", "please", "tell", "reply",
  "paste", "share", "send", "confirm", "provide", "try",
]);
const PLAN_HEADERS = [
  "steps:", "plan:", "the plan:", "here's the plan:", "here is the plan:",
  "my plan:", "approach:", "todo:",
];
const COMPLETION = [
  "done", "completed", "fixed", "implemented", "created", "updated",
  "added", "removed", "passes", "passed", "resolved", "finished",
];

function lower(s: string): string { return s.toLowerCase(); }
function upper(s: string): string { return s.toUpperCase(); }
function trimRight(s: string): string { return s.replace(/\s+$/, ""); }
function trimLeft(s: string): string { return s.replace(/^\s+/, ""); }
function startsWith(s: string, prefix: string): boolean { return s.startsWith(prefix); }
function isFenceLine(line: string): boolean { return trimLeft(line).startsWith("```"); }
function splitLines(text: string): string[] {
  if (text === "") return [];
  const parts = text.split("\n");
  // C++ splits keeping trailing empty handling simple: while !empty push substr(0,nl).
  return parts;
}
function countFences(lines: string[]): number { return lines.filter(isFenceLine).length; }
function stripMarker(line: string): { text: string; listy: boolean } {
  const t = trimLeft(line);
  if (t.length > 1 && (t[0] === "-" || t[0] === "*" || t[0] === "+") && t[1] === " ") {
    return { text: trimLeft(t.slice(2)), listy: true };
  }
  if (t.length && /[0-9]/.test(t[0])) {
    let i = 0;
    while (i < t.length && /[0-9]/.test(t[i])) i++;
    if (i + 1 < t.length && (t[i] === "." || t[i] === ")") && t[i + 1] === " ") {
      return { text: trimLeft(t.slice(i + 2)), listy: true };
    }
  }
  if (t.length && (t[0] === "#" || t[0] === "|" || t[0] === ">")) return { text: t, listy: true };
  return { text: t, listy: false };
}
function isTerminal(c: string): boolean {
  const code = c.charCodeAt(0);
  if (code >= 0x80) return true;
  return ".!?;:)]}\"'`*_~|>".includes(c);
}
function lastWord(text: string): string {
  const t = trimRight(text);
  let i = t.length;
  while (i > 0 && (/[a-zA-Z]/.test(t[i - 1]) || t[i - 1] === "'")) i--;
  return t.slice(i);
}
function countWords(text: string): number {
  let count = 0; let inWord = false;
  for (const c of text) { const w = !/\s/.test(c); count += w && !inWord ? 1 : 0; inWord = w; }
  return count;
}
function looksLikeToken(body: string): boolean {
  const t = body.replace(/\s+$/, "");
  const idx = t.search(/\S+$/);
  const word = idx === -1 ? "" : t.slice(idx);
  return /[0-9\/._\-@#]/.test(word);
}
function isDangling(word: string): boolean {
  if (word.length > 1 && word === upper(word)) return false;
  return DANGLING.has(lower(word));
}
function findTail(lines: string[]): { last_line: string; ends_in_code: boolean; empty: boolean } {
  for (let i = lines.length - 1; i >= 0; i--) {
    if (trimRight(lines[i]) === "") continue;
    return { last_line: trimRight(lines[i]), ends_in_code: isFenceLine(lines[i]), empty: false };
  }
  return { last_line: "", ends_in_code: false, empty: true };
}
function lastSentence(line: string): string {
  const lowered = lower(trimLeft(line));
  let start = 0;
  for (let i = 0; i + 1 < lowered.length; i++) {
    if ((lowered[i] === "." || lowered[i] === "!" || lowered[i] === "?") && lowered[i + 1] === " ") start = i + 2;
  }
  const sentence = lowered.slice(start).split("*").join("");
  return stripMarker(sentence).text;
}
function isHandoff(line: string): boolean {
  const lowered = lower(line);
  for (const imp of ["run ", "refresh ", "open ", "reply ", "restart ", "reload "]) {
    if (lowered.startsWith(imp)) return true;
  }
  let i = 0;
  while (i < lowered.length) {
    while (i < lowered.length && !/[a-z_]/.test(lowered[i]) && lowered[i] !== "'") i++;
    const start = i;
    while (i < lowered.length && (/[a-z_]/.test(lowered[i]) || lowered[i] === "'")) i++;
    const word = lowered.slice(start, i);
    if (!word) continue;
    if (HANDOFF.has(word)) return true;
  }
  return false;
}
function announces(sentence: string): boolean {
  for (const opener of ANNOUNCEMENTS) if (sentence.startsWith(opener)) return true;
  if (sentence.startsWith("let me ")) {
    const rest = sentence.slice(7);
    return !LET_ME_BENIGN.some((b) => rest.startsWith(b));
  }
  return false;
}
function openDepthOfLastBlock(lines: string[]): number {
  let close = lines.length;
  while (close > 0 && !isFenceLine(lines[close - 1])) close--;
  if (close === 0) return 0;
  let open = close - 1;
  while (open > 0 && !isFenceLine(lines[open - 1])) open--;
  if (open === 0) return 0;
  let depth = 0;
  for (let i = open; i + 1 < close; i++) {
    for (const c of lines[i]) depth += ((c === "{" || c === "(" || c === "[") ? 1 : 0) - ((c === "}" || c === ")" || c === "]") ? 1 : 0);
  }
  return depth;
}
function hasPlanWithoutExecution(lines: string[], response: string): boolean {
  const lowered = lower(response);
  for (const done of COMPLETION) if (lowered.includes(done)) return false;
  for (let i = 0; i < lines.length; i++) {
    const header = lower(trimRight(trimLeft(lines[i])));
    const bare = header.split("*").join("");
    const isHeader = PLAN_HEADERS.some((h) => bare.startsWith(h));
    if (!isHeader) continue;
    let items = 0; let firstPerson = false;
    for (let j = i + 1; j < lines.length; j++) {
      if (trimRight(lines[j]) === "") continue;
      const { text: item, listy } = stripMarker(lines[j]);
      if (!listy) return false;
      items++;
      const lo = lower(item);
      firstPerson = firstPerson || lo.startsWith("i will") || lo.startsWith("i'll") || lo.startsWith("we will");
    }
    return items >= K_MIN_PLAN_ITEMS && firstPerson;
  }
  return false;
}

export interface FinishHit { rule: string; reason: string; score: number }

export function evaluateFinishRules(task: string, response: string, hadToolCall: boolean, finishReason: string): FinishHit[] {
  const hits: FinishHit[] = [];
  const add = (rule: string, reason: string, score: number) => hits.push({ rule, reason, score });
  if (hadToolCall) return hits;
  if (finishReason === "length") add("length", "stopped at the token limit", K_SCORE_LENGTH);
  response = trimRight(response);
  if (trimLeft(response) === "") {
    if (finishReason === "stop") add("empty", "empty answer", K_SCORE_EMPTY);
    hits.sort((a, b) => b.score - a.score);
    return hits;
  }
  const lines = splitLines(response);
  if (countFences(lines) % 2 === 1) add("fence", "unclosed code fence", K_SCORE_FENCE);
  const tail = findTail(lines);
  if (tail.ends_in_code) {
    if (openDepthOfLastBlock(lines) >= 2) add("brace", "code block closes with unbalanced brackets", K_SCORE_BRACE);
  } else if (!tail.empty && countFences(lines) % 2 === 0) {
    const stripped = stripMarker(tail.last_line);
    const body = stripped.text;
    const listy = stripped.listy;
    const last = tail.last_line[tail.last_line.length - 1];
    const handoff = isHandoff(body);
    if (!handoff && (last === ":" || last === "," || last === "(" || last === "[" || last === "{")) {
      add("open_end", "last line ends in an open punctuation mark", K_SCORE_OPEN_END);
    } else if (!isTerminal(last)) {
      if (body.length >= K_MIN_PROSE_CHARS && isDangling(lastWord(body))) {
        add("stopword", "last sentence ends on a dangling word", K_SCORE_STOPWORD);
      } else if (looksLikeToken(body)) {
        // Ends on a path, number or name: stay silent.
      } else if (!listy && body.length >= K_MIN_WEAK_PROSE_CHARS) {
        add("bare", "last paragraph has no closing punctuation", K_SCORE_BARE);
      } else if (listy && countWords(body) >= K_MIN_ITEM_WORDS) {
        add("bare_item", "last list item has no closing punctuation", K_SCORE_BARE_ITEM);
      }
    }
    if (!handoff && announces(lastSentence(tail.last_line))) {
      add("announce", "announces a next action that never ran", K_SCORE_ANNOUNCE);
    }
    if (hasPlanWithoutExecution(lines, response)) {
      add("plan", "plan listed with no execution after it", K_SCORE_PLAN);
    }
  }
  if (task.length >= K_LONG_TASK_CHARS && response.length < K_TINY_ANSWER_CHARS) {
    add("tiny", "answer far shorter than the task", K_SCORE_TINY);
  }
  hits.sort((a, b) => b.score - a.score);
  return hits;
}

export function judgeFinish(task: string, response: string, hadToolCall: boolean, finishReason: string): { unfinished: boolean; reason: string; score: number } {
  const hits = evaluateFinishRules(task, response, hadToolCall, finishReason);
  if (!hits.length || hits[0].score < K_THRESHOLD) {
    return { unfinished: false, reason: "", score: hits.length ? hits[0].score : 0 };
  }
  return { unfinished: true, reason: hits[0].reason, score: hits[0].score };
}
