# Brief 3: Qwen3.8-27B decode (DSpine drafter) and upstream catch-up for 27B + prefill

## How to run this brief (read first)

Repo `~/gufo`, branch `fork-enhancements`. Work in a NEW worktree off it, never in `~/gufo` itself:
`git worktree add ~/gufo-27bport -b perf/qwen27b-upstream-port`. Upstream remote `upstream` is already fetched. Overlap with `DECODE-1-UPSTREAM-PORT.md`: items #874 #705 #988 #660 are listed there too; do each only once, and check HEAD first (PR numbers are not in our squashed log, grep the changed symbols and files).

Rules (same as Brief 1): selective cherry-pick or hand-port only, no full merge, do not port the cache redesign (#540 family) unless the user asks; no competitor names in repo files or commits; single-line Conventional Commits; do not push; no `pkill -f` (kill by pid); one GPU job at a time; quality gate for every port (greedy output identical on 16 prompts, repo focused tests) and matched A/B timing at realistic depth.

You are being served by the same gufo that this work disturbs. Expect jev-guard to stop gufo while a command needs the GPU or most of the RAM (27B benchmarks, GPU ctest, large builds) and to restart it afterwards; your request is held and resumes. That is normal. Do not rename, wrap, detach or background commands to avoid the guard. The jev opencode plugin only logs your commands in this phase; leave it alone.

Order of work: Part 2 first (27B-only commits: #705, #988, #874, #992, #660, #688, then the shared K-quant and weight-loading items, then prefill items), one commit at a time, each with a dry-run cherry-pick and a measured A/B. Part 1 (DSpine) starts only after Part 2 has landed, and only with its step 1 (torch reproduction of the card's claim); that step needs the GPU for a long batch, so say so first and run it as one job. Report measured and unmeasured results separately.

Status: future plan, written 2026-11-08. Nothing here is started. Same rules as `DECODE-1-UPSTREAM-PORT.md` and `DECODE-2-ENGINE-LEVERS.md`: no competitor names in repo files, no push, no `pkill -f`, one GPU job at a time, quality gate (greedy output must stay identical), depth-realistic timing (32k-128k), report measured vs unmeasured. Do not start without the user's go; the earlier decision "no more upstream PR rebases unless asked" still stands, so each port below is opt-in.

## Part 1: DSpine drafter for the 27B

Source: `tencent/Qwen3.8-27B-DSpine-b16` (Apache-2.0, safetensors BF16, ~2B params, public). Claims below are from the model card (SGLang, BF16, FA3, concurrency 8, 20 recorded conversations per agent subset). None are reproduced here.

| Benchmark | DSpine tokens/cycle | DFlash2 | Gain |
|---|---|---|---|
| SWE-bench Pro (low reasoning) | 4.99 | 4.19 | +19% |
| SWE-bench Pro (xhigh) | 4.77 | 4.00 | +19% |
| WorkBuddy Code (xhigh) | 4.92 | 4.06 | +21% |
| HumanEval | 5.71 | 4.87 | +17% |
| MT-Bench | 4.48 | 4.11 | +9% |

Card end-to-end speedup vs autoregressive on agent tasks: 2.7-3.2x (DFlash2 2.3-2.9x).

Design (card): 5 layers, block size 16, 8,192-token sliding window, each layer passes predicted predecessor features to adjacent positions. Training data not stated. The runtime documented is SGLang only, so gufo needs its own drafter path.

Why it is not a drop-in: block 16 (DFlash2 is 8), different draft mechanism, 8k window. The 27B engine in gufo already has a DFlash2 path (`src/models/qwen27b`, GGUF `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, MTP head); DSpine would be a second drafter kind behind the same speculative interface.

Tasks, in order, stop at the first failed gate:
1. **Reproduce the claim cheaply first.** Run the card's drafter in torch against the 27B (GGUF target features captured with the existing capture probe pattern, adapted to the 27B's tap layers) and score teacher-forced accepted tokens per cycle on the user's opencode sessions and on target-generated text. Use the protocol learned on Flash-Next: anchor = the sampled token, context features end BEFORE the anchor, labels start after the anchor (a one-token misalignment gave 1.3 instead of 3.4 there). Gate: >= +10% over DFlash2 on the same held-out text. If not, drop the port.
2. **Reference + converter.** Torch reference from the card's code, GGUF converter (Q8_0 matrices, F32 vectors, as for DFlash2), tensor map and config.
3. **Engine drafter.** Port the 5-layer block-16 draft pass with the predecessor-feature path and 8k window, behind the existing 27B speculative interface. Reuse the DFlash2 feature-capture path (#688 pinned-host capture is an upstream commit we lack, see Part 2).
4. **Verify width 16.** Verification then runs up to 16 rows. Upstream #874 tuned 27B verification GEMM geometries only for 2-8 rows and #779 covers 9-15-row K-quant GEMMs; measure 9-16 rows and tune what is slow. Confidence-cut verification (verify only the prefix whose cumulative draft confidence passes a threshold) is the proven lever on Flash-Next (prose 47.9 -> 51.8) and applies here.
5. **Measure.** Engine bench at 32k-128k depth: acceptance, tok/s per category (code, json, copy, math, prose), greedy output identical to the baseline drafter. Keep only if tok/s at depth is >= +10% with no regression in any category. Show the held-out and engine numbers side by side.
6. **Own fine-tune (optional).** Fine-tune on the 27B's own generations only (never other models' text). Lesson from Flash-Next: the old fine-tune lost acceptance because of the alignment bug and an unused selector, not because fine-tuning is bad; with both fixed a mixed run reached +10% on held-out sessions.

## Part 2: upstream commits we are behind (27B and prefill)

Method: `git fetch upstream`; `git cherry HEAD upstream/main` shows 293 of 309 upstream commits with no patch-equivalent in this fork (fork-enhancements has its own prefill and decode work, so some of these may already exist functionally under different patches). Before porting any item, do a dry-run cherry-pick on a scratch worktree, check overlap with the fork's existing change, and only then measure. Each port: one commit, quality gate, matched timing against current HEAD at depth, revert if not a clear win.

### 27B only (all missing)
| Commit | Subject |
|---|---|
| 90b9745f | split prompt passes by layers so other sessions keep decoding (#1079) |
| d2b01f24 | verify long prompt-lookup copies in one wide pass (#992) |
| 49eeada6 | draft DFlash2 through the shared prefix-plus-table output head (#988) |
| 72dc1a09 | tune verification GEMM geometries at 2-8 rows (#874) |
| e4a219ef | accelerate DFlash2 drafting and verification (#705) |
| b0eb5d47 | capture DFlash2 prompt features into bounded pinned host memory (#688) |
| da5d5a2b | Qwen 27B AR and DFlash2 cache continuation (#540, large: 83 files) |
| d1830de7 | verification attention rows > 1 as wave64 with shared per-head steps (#660, large) |
| e121a214 | CLI test fix (#674) |

Priority for single-user decode speed: #705, #988, #874, #992, #660, #688. #540 matters if the 27B is served with the disk/prefix cache. #1079 is a multi-session fairness change; low value for one user.

### Qwen-family shared (touch 27B decode/prefill too)
#779 (split 9-15-row K-quant prefill GEMM as one masked 16-row pass), #777 (Q8_0 prefill GEMM scratch size fix), #995 (K-quant warp dot, GEMV one row per warp), #990 (IQ4 small-batch stepping, avoids spills), #708 and #794 (weights stay in device memory when the heap has room; fill from shard file), #650 (fork recurrent state across two buffers instead of copying), #773/#815 (stable cache boundary after system instructions, with its revert: check the net state first).

### Prefill commits (mostly Flash-Next, fork has its own tuned prefill: compare, do not stack blindly)
#895 (fuse prefill DeltaNet norms/gates/recurrence/epilogue), #933 (16,384-row lone-prompt passes with grouped GEMM rasters), #890 (register-resident routed prefill experts over one 128-token map), #690 (wide prefill routes for every pass), #875 (prefill rows independent of pass route), #963 and #1016 (prompt checkpoints inside prefill passes; capacity-limited step end), #954 (4,096-row passes with cached requests), #877 (cache/prefill alignment), #866 (fused WMMA prefill attention past 262,144 tokens), #626/#622/#571 (attention cost at depth, compact sparse selections, wide prefill kernels). #758/#850 (chunked DeltaNet): the lossy version was removed upstream (#850), so port neither without reading both.

### Also missing, not 27B/prefill (note only)
Flash-Next decode: #894 (HC combine + Q8_0 mixer in one launch), #1048 (overlap shared expert with routed experts), #882 (end MTP chains at first unconfident draft), #742/#699 (selector spills, sampled decode overhead), weights move to Unsloth Dynamic 3.1 UD-Q4_K_XL (#1031; also see the new 4-shard UD-IQ4_XS on the Hub).

## Deliverable
One table: commit | applies cleanly? | measured gain at depth (decode and prefill, per category) | quality evidence | kept/reverted. Keep kept work as local single-line Conventional Commits, no push. Update the port-model skill `LEARNINGS.md` with every measured result.
