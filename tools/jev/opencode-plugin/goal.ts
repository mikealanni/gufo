// Port of jev_goal.cpp trigger detection (Phase 0, log-only).
// Only the phrase match + bare check is ported here; the scoring rules
// (open files/failures/loops) stay in the engine until Phase 2.
// Names/constants mirror the C++ so the two can be diffed.
export const TRIGGERS = [
  "jev make sure its done",
  "jev make sure it is done",
  "jev make sure its finished",
  "jev make sure it is finished",
  "jev finish it",
  "jev finish this",
  "jev keep going until done",
  "jev keep going until its done",
] as const;

function isAlnum(c: string): boolean {
  return (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || (c >= "0" && c <= "9");
}

function apostropheLen(text: string, i: number): number {
  const c = text[i];
  if (c === "'") return 1;
  // C++ checks UTF-8 E2 80 99/98 (right/left single quote); in JS these are
  // single UTF-16 code units.
  if (c === "’" || c === "‘") return 1;
  return 0;
}

function isSeparator(text: string, i: number): boolean {
  return !isAlnum(text[i]) && apostropheLen(text, i) === 0;
}

function lower(c: string): string {
  return c >= "A" && c <= "Z" ? String.fromCharCode(c.charCodeAt(0) | 0x20) : c;
}

function matchPhraseAt(text: string, pos: number, phrase: string): number {
  let i = pos;
  const words = phrase.split(" ");
  let first = true;
  for (const word of words) {
    if (!first) {
      const sepStart = i;
      while (i < text.length && isSeparator(text, i)) i++;
      if (i === sepStart) return -1;
    }
    first = false;
    for (let w = 0; w < word.length; w++) {
      let a = apostropheLen(text, i);
      while (a) { i += a; a = apostropheLen(text, i); }
      if (i >= text.length || lower(text[i]) !== word[w]) return -1;
      i++;
    }
  }
  if (i < text.length && isAlnum(text[i])) return -1;
  return i;
}

export interface GoalHit {
  matched: boolean;
  phrase: string;
  begin: number;
  end: number;
  bare: boolean;
}

export function detectGoal(text: string): GoalHit {
  const none: GoalHit = { matched: false, phrase: "", begin: -1, end: -1, bare: false };
  for (let pos = 0; pos < text.length; pos++) {
    if (text[pos].toLowerCase() !== "j") continue;
    if (pos > 0 && isAlnum(text[pos - 1])) continue;
    for (const phrase of TRIGGERS) {
      const end = matchPhraseAt(text, pos, phrase);
      if (end !== -1) {
        // Bare = nothing besides separators/whitespace on either side.
        const before = text.slice(0, pos).replace(/^[\s,.;:!?]+|[\s,.;:!?]+$/g, "").trim();
        const after = text.slice(end).replace(/^[\s,.;:!?]+/, "").trim();
        return { matched: true, phrase, begin: pos, end, bare: before.length === 0 && after.length === 0 };
      }
    }
  }
  return none;
}

export function findTrigger(text: string, from: number): [number, number] {
  for (let pos = from; pos < text.length; pos++) {
    if ((text[pos] ?? "").toLowerCase() !== "j") continue;
    if (pos > 0 && isAlnum(text[pos - 1])) continue;
    for (const phrase of TRIGGERS) {
      const end = matchPhraseAt(text, pos, phrase);
      if (end !== -1) return [pos, end];
    }
  }
  return [-1, -1];
}
export const hasGoalTrigger = (text: string): boolean => findTrigger(text, 0)[0] !== -1;
