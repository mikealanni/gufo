# Brief 1: port upstream decode wins into the fork (engine-wide)

Repo: `~/gufo`, branch `fork-enhancements`. Work in a NEW worktree off it (`git worktree add ~/gufo-decport -b perf/decode-upstream-port`), never in `~/gufo` itself. Upstream remote `upstream` (gufo-org/gufo) is already fetched: `git log HEAD..upstream/main` = 282 commits, merge-base `d0f7c3a0`.

## Rules
- Selective cherry-pick / hand-port only. A full merge was tried before and aborted (upstream rewrote Qwen serving onto a component cache; ~170 conflict hunks). Do NOT port the cache redesign (#573, #540 family).
- No competitor names in repo files or commits. Single-line Conventional Commits. Do not push. No `pkill -f` (kill by pid). No rocprof on servers. GPU jobs one at a time; ComfyUI/other agents may hold GTT (check `mem_info_gtt_used`). Builds `-j30` max. Configure with `-DGUFO_ENABLE_NPU=ON` plus the XRT include/lib dirs from `~/bench-compare/saved/integ_build.log` or timings read ~16% slow.
- Quality gate for every port: bit-identical (or documented numerics change) on greedy output for 16 prompts, needles >= 15/16 at 32k, plus the repo's focused tests. Timing: report per-request/phase at realistic depth (32k-128k), not empty context. Keep a result row for every port even if it is a null.
- Before each port: check whether it is ALREADY in HEAD (PR numbers are not in our squashed log; grep the changed symbols/files, not the commit message).

## Candidates (decode / engine-wide first), upstream hash -> what
Engine-wide (benefits every Qwen-family model, do these first):
1. `1c75887b` #995 K-quant warp dot specialization + one-row-per-warp K-quant GEMV (`src/families/qwen/hip_common/quant_ops.hpp`, `kernels/gemv_quant.hip`). Pure decode GEMV win for all K-quant models. Small diff.
2. `56ed8f4d` #990 step IQ4 small-batch K-quant tokens so ROCm spills do not occur.
3. `ae9ec691` #681 vector-load small projections (notes say ported on `merge/upstream-latest`; verify it is in `fork-enhancements`).
4. `95ba1eae` #650 fork recurrent state across two buffers; `d1830de7` #660 wave64 verify attention (27B/35B; notes say ported, verify).
5. `72dc1a09` #874 27B verification GEMM geometries at 2-8 rows; `e4a219ef` #705 and `49eeada6` #988 27B DFlash2 draft/verify speedups.
6. `3462aa96` #653 stall fixes (prompt and decode stalls), `31c3b3a0` #881 idle-spill polling back-off.
Flash-Next decode:
7. `eb1d52e6` #894 decode batch HC combine + Q8_0 mixer in one launch (kernels.hip.cpp, executor; includes hc_mix tests/bench).
8. `96c9a0f5` #961 and `d2b01f24` #992 verify long prompt-lookup copies in one wide pass (big on copy/code edits).
9. `5a1669a6` #626 and `baf5a1bb` #622 attention cost / compacted selections at depth. Conflict with our selector (SelectScoreWmmaKernel); port the idea, keep our selector.
10. `e060aeed` #699 reduce sampled decoding overhead; `d4eaaad4` #742 selector spills and sampled drafting.
11. `5d66de8c` #882 end MTP chains at first unconfident draft: our DFlash2 path already has a confidence cut (commit on fork-enhancements, selector softmax >= 0.7); compare, do not duplicate.
Skip (we ship DFlash2, not MTP): #603 #613 #655 #664 #703 #733 #739. DeepSeek `bdf800de` #897 only if ds4 is in our tree.

## Procedure
Order by (expected decode gain x low conflict). For each: show the diff, port, build the affected target, run the focused tests, run the matched A/B on `gufo bench` and the served decode suite (code, json, copy, math, prose) at depth, commit if non-negative, append a line to `docs/models/<model>/EXPERIMENTS.md` (even for nulls). Final report: table of port | status | decode delta by category | quality check.
