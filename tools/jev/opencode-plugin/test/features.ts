import { classifyJevRequest } from "../gate.ts";
import { applyPlan } from "../effort.ts";
import { detectGoalSpec, evaluateGoal, stripGoalTrigger } from "../goal-eval.ts";
import { judgeFinish } from "../finish.ts";

let failed = 0;
const check = (name: string, ok: boolean): void => { console.log(`${ok ? "PASS" : "FAIL"} ${name}`); if (!ok) failed++; };

check("trivial question gets low, no thinking", classifyJevRequest("what is a mutex?").effort === "low");
check("stack trace gets xhigh", classifyJevRequest("Traceback (most recent call last):\n File x.py:3").effort === "xhigh");
check("plain request gets medium", classifyJevRequest("Please review this design for the cache layer in detail").effort === "medium");

const b1: any = { messages: [] };
check("disabled thinking carries no effort", applyPlan(b1, { effort: "low", thinking: false, reason: "t" }).join() === "thinking");
const b1b: any = {};
check("enabled thinking carries effort", applyPlan(b1b, { effort: "medium", thinking: true, reason: "t" }).join() === "reasoning_effort,thinking");
const b2: any = { reasoning_effort: "high" };
check("caller effort wins", applyPlan(b2, { effort: "low", reason: "t" }).length === 0 && b2.reasoning_effort === "high");
const b3: any = {};
check("budget added once", applyPlan(b3, { budget: 512, reason: "t" }).join() === "reasoning_budget_tokens" && applyPlan(b3, { budget: 256, reason: "t" }).length === 0);

check("phrase stripped", stripGoalTrigger("Fix foo.cpp. Jev, make sure it's done!") === "Fix foo.cpp.");
check("bare phrase becomes Continue.", stripGoalTrigger("jev finish it") === "Continue.");
const msgs: any[] = [{ role: "user", text: "Fix foo.cpp and run the build. jev finish it" }];
const spec = detectGoalSpec(msgs);
check("goal armed with two items", spec.active && spec.items === 2);
const early = evaluateGoal(spec, msgs, "All done, everything is fixed.", 0);
check("false completion claim continues", early.action === "continue" && early.open === 2 && early.message.startsWith("Continue. Still open:"));
const proven = evaluateGoal(spec, [...msgs,
  { role: "assistant", text: "", calls: [{ id: "1", name: "edit", command: "", path: "src/foo.cpp" }, { id: "2", name: "shell", command: "cmake --build build", path: "" }] },
  { role: "tool", text: "ok", toolCallId: "2" }], "All done.", 0);
check("evidence closes the goal", proven.action === "done");
check("continuation budget respected", evaluateGoal(spec, msgs, "All done.", 3).action === "done");
check("announced step continues", evaluateGoal(detectGoalSpec([{ role: "user", text: "jev finish it, explain the cache design" }]), [], "Let me now check the file.", 0).action === "continue");

check("unclosed fence is unfinished", judgeFinish("write code", "Here it is:\n```ts\nconst x = 1;\n", false, "stop").unfinished);
check("finished answer stays finished", !judgeFinish("what is 2+2", "It is 4.", false, "stop").unfinished);
console.log(failed ? `FEATURES FAIL (${failed})` : "FEATURES PASS");
if (failed) process.exit(1);
