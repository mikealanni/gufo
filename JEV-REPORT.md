# Jev report (2026-11-07)

Branch `feat/jevgate`, merged into `fork-enhancements` as `c546413f`. Nothing pushed.
Judge: MiniCPM5-1B-SFT (Q6_K) on the XDNA2 NPU. Workload that counts: opencode on port 8900.

## Bottom line

Jev is built and works, but the measured wall-time gain on opencode is small: well under 5%.
The one piece with a visible, working user-facing effect is the goal tracker ("jev finish it"),
and it has not yet had a chance to rescue a stopped turn. The gate and the finish detector
save little or nothing on agent traffic. Part B (answering the model's own questions mid-think)
was removed from the engine on 2026-11-08.

## What exists

| Piece | Flag | State |
|---|---|---|
| NPU judge (yes/no, batched 4 x 128 rows) | always | working, default MiniCPM5-1B |
| Agent effort gate (cap reasoning on routine tool results) | `--jev-gate agent` | working, small gain |
| Pre-route (single yes/no question, no tools) | `--jev-pre on` | works; irrelevant to agent traffic; cutoff not recalibrated |
| Unfinished-turn detector | `--jev-finish on` | built; cannot fire on agent clients |
| Goal tracker + magic phrase | `--jev-goal on` | working live |
| Inline self-question answering (Part B) | removed | judge said Yes too often and lacked context; flag, scanner and replay mode deleted 2026-11-08 |

## Measured

**Judge (NPU, merged tree, batched):** trivia 9/10, code 8/12, 31.7 to 36.5 ms per question
(127 ms for a batch of 4). 2B scores 9/10 and 9/12 at higher cost. Qwen3-1.7B (7/10, 11/12) was
dropped for quality. Misses are borderline cases (confidence 0.46 to 0.53) plus one confident
wrong "Yes" (git commit vs push).

**Agent gate (live, opencode):**
- Cache stays intact: capped turns re-read under 1,600 tokens. An earlier design that changed the
  effort directive forced a full 208k-token re-read; fixed by capping at decode time (budget 512).
- Capped vs uncapped turns were nearly the same length (about 402 vs 408 tokens), so the saving is small.
- Estimated saving: 3.7 to 8.7% of reasoning tokens today, 17% across all history. About 70% of
  output is reasoning and 24% is edit/write payload, which jev cannot reach.
- Latest session (13 decisions): 6 turns capped, 7 held back (git 2, large shell output 2,
  failure signal 2, python3 1).

**Goal tracker:**
- Phrase matched on 16/16 requests, including a bare "jev finish it" (task taken from earlier
  messages, `source=history items=4`). The phrase is stripped before the model sees it, so the prefix cache is unaffected.
- One evaluation in the latest session: 4 items, 0 open, action=done, 0.4 ms. The user confirmed the work was finished, so "done" was correct.
- It has not yet pushed a continuation, so a rescue of a stopped turn is not demonstrated.

**Scanner / detector replays:** scanner precision rewritten and replayed over logged traces;
the finish detector and goal tracker were replayed over saved corpora (`~/jev-finish`, `~/jev-goal`).
The finish detector never fires live because every request with tools is `constrained_tools`,
which returns early.

## Not measured

- Part A 3-arm run (effort routing vs baseline vs always-low). Needs the server down for hours.
- Pre-route cutoff (0.9) against the 1B judge's graded confidence (about 0.89): needs recalibration.
- Jev overhead on decode (judge runs on the NPU, expected small).
- Goal tracker rescue rate: needs turns where the model stops early.

## Recommendation

1. Keep the goal tracker and the agent gate on: cost is under 1 ms and a cap on routine turns, with the cache intact.
2. Leave the finish detector on only if you relax the `constrained_tools` guard; otherwise it is dead weight on opencode.
3. Part B is removed; do not bring it back.
4. Treat decode-2 (engine levers) as the real speed work: jev's ceiling on agent traffic is a few percent.
5. Run the Part A 3-arm test only if you want a number for non-agent chat use.
