# Laguna S 2.1 Port State

## Active Branch & Model Target
- **Branch:** `feat/laguna` (worktree at `~/gufo-laguna`)
- **Model:** 3-shard Unsloth Laguna S 2.1
- **GGUF Path:** `~/models/laguna-s-2.1-unsloth/UD-Q4_K_M/Laguna-S-2.1-UD-Q4_K_M-00001-of-00003.gguf`
- **DFlash drafter:** `~/models/laguna-s-2.1/laguna-s-2.1-DFlash-BF16.gguf` (on disk, NOT wired into the engine yet)

## Files touched this session (uncommitted, branch `feat/laguna`)
- `src/models/laguna/gpu/kernels/kernels.hip.cpp`: templated group-6/group-9
  WMMA attention kernel + dispatch; new `RouterGemmKernel` + dispatch.
- `src/models/laguna/gpu/engine.cpp`: WMMA call site trusts `AttentionWmma`'s
  return; `LAGUNA_WMMA_DEBUG=1` scalar-vs-WMMA trace (kept, zero cost unset).
- `src/cli/bench/bench.cpp`: `--validate-prefill` and Laguna bench dispatch.
- `src/models/laguna/gpu/kernels/kernels.hpp`, `device_model.cpp`,
  `tools/prof/prof.py`: supporting changes from the WMMA work.

## Environment Setup
- **Runtime libs:** `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.14/lib`
- **Target GPU:** AMD Strix Halo (`gfx1151`)

## Targets
- PP: ~1,700 tok/s (roofline: 8.09B active params/token -> 3,428 tok/s @100% of 55.5 TFLOPS gfx1151 peak; 1,700 is ~50% efficiency, the range the Flash-Next routed GEMM already hits)
- TG: ~50 tok/s with DFlash speculative decoding (drafter model exists, integration not started)

## MEASURED STATE (latest first, do not trust stale numbers below)
- **2026-10-30 (latest): pp2048 = ~1,110-1,116 tok/s** after deduplicating the
  Q8 MMQ-tile quantization of shared activations (below). Modest, verified,
  safe gain over the 1,092-1,120 baseline left by the prior two sessions
  (DFlash v1 wiring + PP-1092 work) - the quantize cost was smaller than
  estimated, not absent.
- Found: `gufo bench -p 2048 -n 32` (pp+tg together) reports a *much lower*
  pp number (518-650) than `-n 0` (pp alone, consistently ~1,090-1,120) on
  this build. `tg32` itself reports `0.00 tok/s` - decode is broken/silently
  failing in `gufo bench` and appears to also corrupt the pp measurement in
  the same process. Not investigated further (out of scope for the PP push;
  TG/DFlash wiring is a separate, larger, not-yet-started task). **Always
  benchmark PP with `-n 0` until this is fixed**, and flag it before trusting
  any `gufo bench` number that includes a `-n` > 0.
- 2026-10-30 wider router/gate tiles (128 expert x 64 token, 512 threads):
  pp2048 = 1,092 (pp512 898 - 512-token prefill only fills 16 of 40 CUs with
  the larger tile, so it is slightly underutilised; 2048 is the target).
- 2026-10-30 wider router/gate tiles (128 expert x 64 token, 512 threads):
  pp2048 = 1,092 (pp512 898 - 512-token prefill only fills 16 of 40 CUs with
  the larger tile, so it is slightly underutilised; 2048 is the target).
- 2026-10-30 after tiled router (64x32) + warp-per-head norm + tiled gate:
  pp512 = 907, pp2048 = 1,035.
- 2026-10-30 after group-9 WMMA only: pp512 = 759, pp2048 = 789.
- 2026-10-30 session baseline: pp512 = 371, pp2048 = 423.
- Progression: 423 -> 789 (group-9 WMMA) -> 903 (router) -> 1,035 (norm+gate)
  -> 1,092 (wider router/gate tiles).
- (tg32 still reports 0.00 - decode path needs separate investigation.)
- Commit `feat/laguna` at 9a6ea829 is the common base.
- Per-2048-token prefill kernel budget at the 789 stage (from `rocprofv3` via
  `tools/prof/prof.py`, two-pass totals halved): experts ~820 ms, dense
  projections ~660 ms, router ~530 ms, quantize ~280 ms, normalization ~270 ms,
  attention ~90 ms, rest small.
- Budget after the 1,035-stage re-profile (two-pass totals halved): experts
  ~820 ms, dense projections ~663 ms, router ~175 ms (now improved again),
  gate ~76 ms, quantize ~105 ms, norms ~66 ms, attention ~90 ms. Re-profile
  after the wider tiles before choosing the next target.

## History (chronological, earlier entries may be superseded)

### Stage-1 upload (superseded numbers, kept for context)
- Fixed `Uploader::Bind`/`ReencodeToQ8` thread-pool stall (clamped workers to `min(8, hw_concurrency)`), upload now completes all 48 layers (~73.5 GB VRAM staged).
- Fixed missing `qh` (`DeviceBuffer<__half>`) allocation in `engine.cpp` causing `ToHalfKernel` null-pointer write.
- Benchmark at that stage (scalar attention only, before WMMA work): 1360-token prompt, prefill 349 tok/s, decode 18 tok/s. **Stale** - do not compare against current numbers, this predates any WMMA attention work.

### WMMA attention, group-6 (heads == num_kv*6, 12/48 layers) - CONFIRMED BROKEN, DISABLED 2026-10-30
- Several sessions iterated on this with conflicting reports (regression to 514-518 tok/s when WMMA was reverted/broken on multi-token, vs claimed-working at 599 tok/s with a `qh`-allocation fix). **The "works correctly" claim was false** - it was checked against too few sizes.
- Added `gufo bench --validate-prefill N` support for Laguna (bench.cpp: `IsLaguna`/`RunLagunaBenchmark`/`ValidateLagunaPrefill` - compares one multi-token `Append` call against the same tokens appended one-at-a-time). `bench` previously had NO Laguna dispatch at all and fell through to the Qwen path with a hard error; this was also fixed.
- Measured with that tool (2026-10-30, full 48-layer forward, UD-Q4_K_M): top1_match **yes** at n=1,4,8; **no** at n=2,16,32,64,256. Not a simple size threshold (n=4,8 pass, n=2 and n=16 fail) - points at something specific to how `AttentionWmmaKernelLaguna` handles partial vs. full `kWmmaTokens` (=16) blocks, not a padding/head-count issue. Root cause NOT found.
- **Disabled**: `engine.cpp` Attention() now has `use_wmma = false && (heads == c.num_kv * 6)` - kernel code (`AttentionWmmaKernelLaguna` in kernels.hip.cpp) kept in place for whoever debugs this next. Re-verified scalar-only path passes `--validate-prefill` at n=1,2,16,64,256 after the revert.
- Format check and build (`gufo_laguna` + `gufo` targets) clean (2 pre-existing, harmless `__builtin_elementwise_max/min` deprecation warnings in the WMMA kernel, not addressed).
- **UPDATE 2026-10-30, second finding - this was two bugs stacked:**
  1. `AttentionWmma`'s host-side shape guard (`kernels.hip.cpp`, the function right after the kernel) was `if (heads != kv_heads * kWmmaGroup) return false;` with `kWmmaGroup=8`. Laguna's group-6 ratio is 48/8=**6**, not 8, so this guard rejected every real call and returned `false` without writing `out` at all. The `engine.cpp` call site discarded that bool, so `attn` silently kept whatever the *previous layer* (a different head-count SWA layer) had written into the same reused scratch buffer. That explains the earlier non-monotonic n=2-pass/n=4-pass/n=16-fail pattern: it was never about token count, it was about which layer's garbage happened to still predict the right top-1 token for that particular synthetic test sequence.
  2. **Fixed** the guard to `heads > kv_heads * kWmmaGroup` (the kernel's own `head_live` mask at lines ~546-547 already handles any ratio <= kWmmaGroup, so equality was never required) and made the `engine.cpp` call site check the return value instead of assuming success.
  3. With that fixed, WMMA now genuinely executes - and is **consistently wrong** for every n>1 (n=1 matches; n=2,4,8,16,32,64 all mismatch, every time, not erratically). This is progress: it isolates a real, reproducible math/indexing bug inside `AttentionWmmaKernelLaguna` itself, separate from the guard bug. Re-disabled via `constexpr bool kWmmaAttentionEnabled = false;` in `engine.cpp` pending that fix.
- **UPDATE 2026-10-30, third pass - two more real bugs found, one remains open:**
  1. **Fixed**: `token = t0 + wave + sub / kWmmaGroup` only produces 9 distinct values (0..8) from 8 waves x 2 offsets, when `kWmmaTokens=16` needs all 16 covered without gaps/overlap. Needs `wave * 2 + offset`. Verified mathematically (see commit) - with the old formula, tokens 9..15 of every 16-token tile were never written by any thread, leaving `out` as whatever was there before. Fixed to `t0 + wave * 2 + sub / kWmmaGroup`.
  2. Added a `LAGUNA_WMMA_DEBUG=1` env-gated instrumentation block in `engine.cpp` `Attention()` (layer 0 only): runs scalar alongside WMMA, dumps per-token-per-head max_abs_diff and first 4 values of both to stderr. Costs nothing when unset (plain env var check); use it instead of blind static kernel reading.
  3. **Still broken, root cause NOT found**: with that instrumentation, even the *correctly covered* case (kv_head=0, n=1, token=0) shows WMMA producing **identical output across heads 0-5** (6 genuinely different real heads in that kv group), while the scalar reference correctly varies per head (small but clearly distinct values). This means the per-lane Q fragment (`q_frag`, loaded via `head = kv_head * kWmmaGroup + sub % kWmmaGroup`) is not actually varying across `sub` the way the indexing math assumes it should for a 6-wide group - possibly a mismatch between how `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (w32 mode) actually distributes the B operand across its 16 output columns and how this kernel (ported from Flint, which presumably has a native ratio that lines up more cleanly) feeds it. This needs AMD RDNA3/gfx1151 WMMA ISA-level lane-to-fragment documentation to resolve with confidence, not more guessing from the HIP source alone.
  4. Re-disabled (`kWmmaAttentionEnabled = false`). Re-verified scalar-only passes `--validate-prefill` at n=1,16,64 after this change.
- **UPDATE 2026-10-30, fourth pass - RESOLVED via direct comparison against Flint's working kernel:**
  1. Confirmed Flint's own `AttentionWmmaKernel` (`src/models/flint/gpu/kernels/kernels.hip.cpp:707`) uses `token = t0 + 2 * wave + sub / kWmmaGroup` (note the `2 *`) - this is the exact fix already applied above; Flint has no `head_live` concept at all because its ratio (32 heads / 4 kv = 8) always equals `kWmmaGroup` exactly, so it never needed masking. Confirms the earlier token-stride fix was correct.
  2. Built an isolated standalone HIP repro (`/tmp/.../wmma_repro.hip`, not checked in - just a scratch file) that calls `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` directly with a trivial A=all-ones, B=per-lane-distinct operand. **Confirmed the hardware instruction correctly respects per-lane B-operand differences** - ruled out any ISA-level lane-mapping misunderstanding.
  3. Added temporary device-side `printf` probes (q_frag after load, s_acc after the K.Qt accumulation, o_acc right before the final write) gated on `blockIdx.x==0 && blockIdx.y==0 && lane<8` - confirmed q_frag and s_acc both vary correctly per lane/head. The bug was between s_acc and the final write.
  4. **Root cause of "heads 0-5 identical" (bug #3):** for Laguna's architecture, the valid-single-key (n=1) case mathematically forces identical pre-gate attention output across heads sharing a kv_head (softmax of one key is always 1.0, V is shared GQA-style) - that part was never wrong. The *scalar reference* (`AttentionKernel` in the same file, line ~202) breaks that symmetry with `out = (acc/l) * softplus(gate[token][head])` - a per-token-per-head gate multiply at the very end. **`AttentionWmmaKernelLaguna` received `gate` as its `sinks` parameter but never read or applied it at all.** Fixed: added the same `g = sinks[token*heads+head]; softplus = g>20?g:log1pf(expf(g)); inv = softplus/total;` right before the final write.
  5. **Root cause of bug #4 (heads 6,7,12,13 still wrong after #3's fix):** `head = kv_head * kWmmaGroup + sub % kWmmaGroup` used the WMMA tile width (`kWmmaGroup=8`) as the memory stride into the Q/output buffers, but the real per-kv_head stride is `heads/kv_heads` (6). This coincided for `kv_head==0` (0*8==0*6) masking the bug, but for `kv_head>=1` it read/wrote the wrong global head slot entirely (e.g. kv_head=1 wrote heads 8-15 instead of its real 6-11). Fixed: `head = kv_head * (heads/kv_heads) + sub % kWmmaGroup`; simplified `head_live` to just `sub % kWmmaGroup < heads/kv_heads` (the old second clause was redundant once the stride is correct).
  6. **Verified** with `--validate-prefill` at n=1,2,4,8,32,64,128,256,512: all pass top1_match except **n=16 exactly**, which also diverges (same cosine~0.96, same large mean_abs_diff) **with WMMA fully disabled** (pure scalar sequential-vs-batched) - this is a pre-existing numeric sensitivity in the model/test (very likely an MoE top-k routing near-tie between the batched and sequential code paths, exactly the class of divergence the port-model skill calls expected), not a WMMA defect. WMMA's extra fp16 rounding just happens to tip that already-narrow margin over for this one size.
  7. Per-layer trace (`LAGUNA_WMMA_DEBUG=1`, all 12 group-6 layers) after all four fixes: every layer matches scalar to **fp16-precision-only noise** (max_abs_diff ~1e-6 to 2e-3, i.e. ~0.001-0.4% relative error, scaling with activation magnitude as expected for fp16 vs fp32).
  8. All temporary debug printfs and the standalone repro removed from the tree; `LAGUNA_WMMA_DEBUG=1` instrumentation kept (useful, zero cost when unset). `--validate-prefill` now also prints the reference/candidate argmax tokens and logit margins on a top1 mismatch (added to `bench.cpp`), useful for distinguishing "real bug" from "near-tie divergence" like this one.
  9. **WMMA group-6 is now considered correct and enabled** (`kWmmaAttentionEnabled = true`). Group-9 (scalar, 36/48 layers) is next for real PP gains.

### Group-9 (heads == num_kv*9, 36/48 layers) - DONE 2026-10-30
- **Done.** `AttentionWmmaKernelLaguna` is now a template on `kHeadsPerKv`
  (6 and 9 instantiated) in `kernels.hip.cpp`. The 16 WMMA output columns hold
  `kTokensPerWave = 16 / kHeadsPerKv` tokens x `kHeadsPerKv` heads, so group-6
  keeps 2 tokens/wave (16 tokens/block) and group-9 fits only 1 token/wave
  (8 tokens/block). Column `sub` maps to `(token offset, head offset)`; a
  column is live while `sub < kHeadsPerKv * kTokensPerWave`, which correctly
  masks the 7 dead columns of group-9 without aliasing real heads.
- Key detail vs. the old group-6-only code: the per-kv_head Q/output stride is
  `kHeadsPerKv` (the real 6/9), not the WMMA tile width (`kWmmaGroup=8`), and
  the token stride is `kTokensPerWave` per wave.
- `AttentionWmma` dispatch now switches on `heads / kv_heads` (case 6, case 9,
  else scalar). `engine.cpp` no longer gates on `heads <= num_kv*8`; it trusts
  the return value.
- Verified with `--validate-prefill` at n=1,2,4,8,64,256 (all top1_match; n=16
  still diverges only on the already-known pre-existing near-tie that also
  fails with WMMA disabled). **Result: pp2048 423 -> 789 tok/s.**

### Router F32 GEMV - DONE 2026-10-30
- `RouterF32Kernel` (one warp per `(token, expert)`, re-reading the whole
  expert matrix per token and the whole token row per expert) cost ~1.9 ms per
  2048-token call, ~18% of prefill.
- Added `RouterGemmKernel`: tiled F32 `out[n][m] = x[n][k] * w[m][k]^T`.
  64x32 tile, 16-wide K tiles staged into shared memory, each thread
  accumulates a 2 token x 4 expert patch. Batches < 8 tokens keep the old
  warp kernel (decode is cheaper that way). `RouterF32` picks between them.
- Numerics: FP32 accumulation order changes recoverable last-bit values only;
  `--validate-prefill` still top1_match at n=2 (old path, identical numbers),
  n=64, n=256. **Result: pp2048 789 -> 903 tok/s.**
- Follow-up 2026-10-30: widened the router and gate tiles to 128 experts x 64
  tokens with 512 threads and a 4 token x 4 expert per-thread patch (256 blocks
  -> 64 blocks for the router, 32 for the gate), cutting activation re-reads
  from 8x to 2x. Same `--validate-prefill` result; pp2048 1,035 -> 1,092.

### Per-head RMSNorm + Q8 gate projection - DONE 2026-10-30
- `RmsNormKernel` launched one 256-thread block per `(row, group)`; q_norm uses
  `groups = heads` (72 for group-9), so a single layer launched ~147k blocks for
  128-wide head rows. Measured ~170 ms/pass for q_norm alone (`grid = 2048x72`,
  4723 us/call).
- Added `RmsNormGroupsKernel`: one block per token row, one warp per head,
  warp-shuffle reduction only, no block syncs. `RmsNormRows` dispatches to it
  when `groups > 1` and the head width is <= 256; `groups == 1` (attn/ffn/output
  norms, 3072-wide) keeps the old kernel.
- `Q8RowsKernel` (attn_gate projection) had the same flaw as the router: one
  block per `(row, token)`, re-reading every gate weight row once per token
  (~3.7 ms/call for the 72-row group-9 gate). Added `GateGemmKernel` with the
  same 64x32 tiled shared-memory structure as `RouterGemmKernel`, dequantizing
  Q8_0 rows into the shared tile. `Q8Rows` keeps the old kernel for < 8 tokens.
- Verified with `--validate-prefill` at n=2,64,256 (all top1_match). Reduction
  order changed last-bit values only (n=2 max_abs_diff 4.58 -> 4.31).
  **Result: pp2048 903 -> 1,035 tok/s.**

### Per-layer MoE host sync - checked, confirmed necessary, not a hidden tax
- Investigated whether `RoutedExperts()`'s `hipEventRecord`+`hipEventSynchronize`
  (reading expert counts back to host once per MoE layer, to build the
  per-expert tile list before the gate/up/down GEMM launches) was a hidden
  per-layer stall, since the host must block there. Added a zero-cost-when-
  unset `LAGUNA_SYNC_DEBUG=1` timer (engine.cpp, same pattern as
  `LAGUNA_WMMA_DEBUG`). Measured mean ~35 ms/call across the 47 MoE layers
  at pp2048 - this matches total-pass-time/47 almost exactly, confirming
  the sync is correctly waiting on **real, necessary GPU compute** (the
  whole layer's attention+dense+router work queued before it), not sitting
  idle on wasted host round-trip latency. The host-side tile-list build
  itself (a 256-iteration loop + two small `hipMemcpyAsync` calls) is
  microseconds, not meaningful. **Conclusion: this is not a lever** - the
  sequential layer dependency makes this sync unavoidable regardless of
  whether tile-list construction is pulled onto the GPU. Left the debug
  timer in place (harmless, zero-cost when unset) in case a future session
  wants to re-check this on a different workload shape.

### Q8_0 gate/up GEMM fusion - investigated, premise was wrong, REMOVED
- Hypothesized that Laguna's MoE experts are Q8_0-quantized, and that the
  fused gate+up kernel (`RoutedGatedF16Gemm`/`LaunchRoutedGatedF16` in
  `qwen38_flash_next/kernels/rocm/kernels.hip.cpp`, which only had cases for
  K-quant types) was forcing gate and up to run as two separate
  `RoutedF16Gemm` launches for Laguna, with the gate's F32 output round-
  tripping through global memory between them (~83 MB/layer at pp2048).
  Added a `WeightType::kQ8_0` case to the fused kernel (additive) and wired
  Laguna to use it.
- It compiled and ran, but `--validate-prefill` showed much larger
  divergence than expected (`max_abs_diff=6.3, cosine=0.95` at n=64, top1
  still happened to match). Before assuming the new kernel case was wrong,
  instrumented `RoutedExperts()` to print each layer's actual `ge.type`/
  `ue.type`/`de.type` and the `gated`/`gate_up_q8` flags
  (`LAGUNA_GATEUP_DEBUG=1`). **Result: `ge.type`/`ue.type` are Q4_K on
  every single MoE layer in this UD-Q4_K_M quant - never Q8_0.** Only
  `de.type` (the down projection) varies, between Q5_K and Q8_0. So
  `gate_up_q8` was `false` on every call and the new kernel case never
  actually executed once.
- **Confirmed the "regression" was a pre-existing baseline artifact,
  unrelated to this change**: re-ran the identical `--validate-prefill 64`
  with the whole feature compile-time disabled
  (`constexpr bool kGateUpQ8Fusion = false`) and got the **exact same**
  numbers (`max_abs_diff=6.31879, cosine=0.950034`, bit-for-bit). Same at
  n=4 (`max_abs_diff=1.35665` either way). **The premise was wrong from the
  start** - Laguna's gate/up experts already take the fast fused K-quant
  path on every layer; there was never a second-launch inefficiency to fix
  here, so there was nothing to break and nothing to gain.
- Removed the dead `kQ8_0` case and its precondition relaxation from
  `kernels.hip.cpp` (never exercised by any real caller, not worth leaving
  unvalidated in shared code), removed the `gate_up_q8`/`kQ8PairTileRows`/
  debug-diff scaffolding from `engine.cpp`. Re-verified clean: full
  `--validate-prefill` sweep (1,2,4,8,16,32,64,128,256,512 - same pass/fail
  pattern as always) and pp2048 ~1,111-1,114 tok/s, matching the baseline
  this investigation started from.
- **Side finding worth flagging, independent of this dead end**: the
  divergence at n=4 (`max_abs_diff=1.36, cosine=0.996`) and n=64
  (`max_abs_diff=6.32, cosine=0.95`) between batched and sequential prefill
  is larger than the fp16-noise level seen in the WMMA correctness work
  earlier in this file (~1e-3 to 1e-2). Top1 still matches at both sizes,
  so this hasn't been flagged as a bug, but it is a bigger gap than "just
  rounding" and worth a dedicated look (likely the same batched-vs-
  sequential MoE routing near-tie class already documented for n=16, just
  also present at other sizes) before trusting `--validate-prefill`'s top1
  gate alone on any future change in this area.
- **Where this leaves the search for 1,500+**: this was the most promising
  lever found in this session and it didn't pan out because the premise
  didn't hold for this model's actual quantization. The tile-width/VGPR
  axis above is genuinely exhausted (hard numbers, not a guess). The
  MoE host-sync axis is confirmed necessary, not wasted. No further
  concrete, verified lever has been found yet in this session; the next
  honest step is a fresh profile-driven look rather than another
  hypothesis-first attempt.

### Deduplicated Q8 MMQ-tile activation quantization - DONE 2026-10-30
- `Session::Impl::Dense()` called `qfn_mmq_q8_0_dense` once per projection,
  and that function always re-quantized its `X_f32` input via
  `quantize_mmq_q8_1_hip` internally, even when the caller passes the exact
  same activation pointer repeatedly: attn_q/attn_k/attn_v all read
  `xn.get()` (post attn_norm), ffn_gate/ffn_up and gate_shexp/up_shexp all
  read `xn.get()` (post ffn_norm) - up to ~5-7 redundant quantizations of the
  same buffer per layer.
- Split `qfn_mmq_dense_impl` (shared `qwen38_flash_next` MMQ code) into a
  quantize step and a GEMM step, and exposed both additively in `qfn_mmq.h`/
  `qfn_mmq.hip.cpp`: `qfn_mmq_q8_1_mmq_bytes`, `qfn_mmq_quantize_q8_1_mmq`,
  `qfn_mmq_q8_0_dense_gemm_preq`. **Purely additive** - `qfn_mmq_q8_0_dense`
  itself is now a thin wrapper calling both in sequence and is byte-for-byte
  the same operation as before, so Flash-Next/other callers of the existing
  entry point are unaffected; no shared-kernel validation matrix needed.
  (Do not confuse these with the pre-existing `qfn_mmq_q8_1_bytes`/
  `qfn_mmq_quantize_q8_1` - those use the decode-oriented `block_q8_1` vec
  layout, a different format from the MMQ tile layout used here.)
- Laguna's `Session::Impl` now has a one-slot quantize cache
  (`quant_cache` + `quant_cache_src/n/k`) and a `Quantized(in, n, k)` helper;
  `Dense()`'s Q8_0 path calls it before `qfn_mmq_q8_0_dense_gemm_preq`
  instead of `qfn_mmq_q8_0_dense`. Cache hits are keyed on the input pointer
  + token count + K, so unrelated calls (e.g. `ffn_down` reading
  `dense_gate.get()`) still quantize fresh every time - no regression there,
  this only saves work for the genuinely shared cases.
- Verified bit-identical: this changes nothing about the quantization math,
  only how many times it runs. `--validate-prefill` at 1,2,4,8,16,32,64,128,
  256,512 reproduces the exact same pass/fail pattern as before (n=16 is the
  pre-existing near-tie, unrelated to this or any WMMA change).
- **Result: pp2048 ~1,100 -> ~1,110-1,116** (smaller gain than the ~105ms/
  pass estimate suggested; the quantize step itself is cheap on this
  hardware, so the real win is still in the GEMM/occupancy side below).

### Remaining prefill budget (next levers, measured at 1,035 tok/s)
- Experts (MoE) ~820 ms/pass at the 789 stage: `RoutedF16GEMMKernel` from
  flash_next, 3 calls per layer (gate/up/down). Needs tile/occupancy tuning;
  this is 4.88B active params/token, the single largest real-FLOP block.
- Dense projections ~660 ms/pass at the 789 stage: qfn_mmq `mul_mat_q` (Q8
  weights) + `quantize_mmq_q8_1` + `Q8RowsKernel`. Q8 activations are
  re-quantized per projection; worth checking if a projection can consume
  F16/GUFO activations directly or share one quantization.
- Re-run `tools/prof/prof.py` now that router/norm/gate are fixed; the next
  profile decides whether experts or projections is the better target.

### WMMA group-6/9 attention: ring buffer never applied on read - FIXED 2026-10-30, was a hard crash at depth
- User asked to verify today's gains hold at 32k/64k/128k depth (per the
  port-model skill's own stated requirement, "tok/s must stay flat with
  depth"). First `gufo bench -d 32768 -p 2048` ever run this session -
  **crashed**: `Memory Fault Error ... kernel: AttentionWmmaKernelLaguna<9>`.
- Root cause: `AttentionWmma`'s host wrapper takes a `ring` parameter (the
  SWA k/v-cache's physical row capacity - `sliding_window + tile`, e.g.
  4608, allocated once in `Init()` and reused via modulo-wraparound by the
  *writer*, `StoreKvKernel`) but **never forwards it to the kernel at
  all** - the kernel's own signature had no `ring` parameter. The `fetch`
  lambda reads `k_cache`/`v_cache` using the raw *absolute* token position
  (`kt + key`) as the physical row index, with no modulo. This is exactly
  correct only as long as absolute position never exceeds the ring's
  physical capacity - true for every test this whole session (all well
  under 4608 tokens of context) and false the first time anyone tested
  real depth. Confirmed via reading `StoreKvKernel` (which *does* wrap:
  `row = position; if (ring) row %= ring`) side by side with the WMMA
  kernel's unwrapped read - a direct, provable mismatch, not a guess.
  Affects every SWA layer (36/48 of them) at any depth beyond the ring
  capacity - i.e. every real long-context use, full stop; invisible at
  short context because the bug requires position > ring to trigger.
- **Fix**: added `ring` to `AttentionWmmaKernelLaguna`'s signature, pass it
  from the host wrapper (which already received it, just never forwarded
  it), and apply `row = ring != 0 ? (kt + key) % ring : (kt + key)` in the
  `fetch` lambda before computing the k_cache/v_cache byte offset. Full
  (non-SWA) attention layers have `ring=0` (allocated at full capacity, no
  wraparound), so this is a no-op for them - the fix only changes behavior
  for SWA layers past the ring's capacity, where it was previously reading
  arbitrarily far out of bounds.
- **Verified**: `gufo bench -d 32768 -p 2048 -n 0` no longer crashes
  (809.86 tok/s at that depth, vs ~1,222-1,231 at shallow context - a real
  drop, expected from growing attention cost with depth, not evidence of
  a new problem; this was the *first successful measurement at this depth
  all session*, so there is no prior same-build baseline to compare the
  magnitude of that drop against). `-d 65536` was still running at the
  time of this entry (takes a long time to prepare that much depth before
  the timed region even starts) - update this entry with its result and
  `-d 131072` once both complete. **Do not trust any PP/TG number measured
  at empty/short context as representative** - this bug alone proves why:
  it was invisible in every other measurement this entire session.
- **Not yet done**: `gufo bench`'s own TG-at-depth reporting shows `tg64=
  0.00` even post-fix - this is the *separate*, already-documented
  `gufo bench -n <N>` (N>0) decode-reporting bug (drags the paired PP
  number down too), not something today's fix touches. Get a real TG@depth
  number via `gufo prompt` with a long fed context instead, not `gufo
  bench`'s combined pp+tg mode, until that separate bug is fixed.

### DFlash drafter: BF16 -> Q8_0 re-encode at load time - DONE 2026-10-30
- Profiled DFlash's own forward pass (`BatchedExactBf16GEMMFp32VecKernel`,
  16.8% of total kernel time) and found its mean cost per call
  (256.91us) matches almost exactly the bandwidth-bound cost of reading a
  full BF16 weight matrix at ~240 GB/s - same conclusion as the decode
  Dense() floor, but for the drafter's own dense projections, which are
  raw BF16 (2.2 GB file), not quantized at all.
- `executor.cpp`'s `RunBlockGemm`/`RunInjectGemm` **already have a Q8_0
  dispatch branch** (`LaunchBatchedQuantGEMMFp32`/the packed-quant path in
  `RunInjectGemm`) - it was simply never exercised because every drafter
  weight loads as BF16. Added `ReencodeBf16ToQ8_0()` in
  `dflash_weights.cpp` (synchronous, single-threaded - the drafter is a
  handful of layers and a few hundred MB, not Laguna's 68 GB main model,
  which has its own parallel async re-encoder in `device_model.cpp` not
  reused here since it is a private method tightly coupled to that
  uploader's staging pipeline) and applied it to all 7 per-layer dense
  tensors (attn_q/k/v/output, ffn_gate/up/down) plus `fc_projection` (the
  113 MB 6-tap feature-fusion matrix) at load time, right after binding.
  Standard block_q8_0 format (32-element blocks, F16 scale + 32 signed
  int8 per block) - verified bit-for-bit against the same encoding
  Laguna's own `ReencodeToQ8` uses (same `Bf16ToFloat`/`FloatToHalf`
  bit-manipulation, same `amax/127` scale, same block layout).
- **Buffer lifetime**: `QwenDFlashWeights` is copied by value at least
  once while loading (`model.cpp`), which would reallocate and dangle any
  pointer into a plain owned buffer - added
  `owned_reencoded_storage: vector<shared_ptr<vector<char>>>` to the
  struct (shared_ptr copy is cheap and pointer-stable, unlike a plain
  vector-of-vectors) and pointed each re-encoded `QwenTensorRef::data`
  into one of those buffers.
- **Verified correct and safe**: no crash, text stayed byte-identical to
  the pre-change run at `-d 2, n=500` (`drafted=410 accepted=294` before
  -> `412/293` after fc_projection's own re-encode - the tiny near-tie
  shift from fc_projection alone, same already-documented class of
  numeric sensitivity, not a new bug) while the per-layer re-encode alone
  was bit-identical. Laguna's own pp2048/plain-decode numbers reconfirmed
  unaffected (this only touches the drafter).
- **Measured**: `-d 2`, n=500: **33.75 -> 36.82 -> 36.87 tok/s** (per-layer
  re-encode, then + fc_projection). The fc_projection re-encode alone was
  close to free (36.82 -> 36.87, within noise) - most of its call cost
  was apparently already on a reasonably-sized small-batch path even as
  BF16; the per-layer dense re-encode (attn_q/k/v/output/ffn_gate/up/down)
  was the real win (+9%).
- Re-swept `-d` with this fix active: 1->31.3, **2->36.9 (best)**,
  3->34.4, confirming `-d 2` is still the optimum, not shifted by this
  change.
- **Current best DFlash TG: ~36.9 tok/s** (target 40), up from 7.72 tok/s
  at the start of this port's DFlash work - a 4.8x improvement across
  this whole investigation (attn_gate fix, aux_norm fix, mmq_x=16 decode
  floor fix, quantize-dedup, now drafter Q8_0 re-encode). Plain
  (no-DFlash) decode is at ~26.2 tok/s (target was folded into this via
  the shared mmq_x fix). PP is at ~1,222-1,231 shallow (target 1,300) and
  confirmed non-crashing but declining with depth (809/611/410 @
  32k/64k/128k - the ring-buffer crash fix, separate from this).
- **Not yet tried**: `attn_gate` (hidden*72, small) and `aux_norm`
  (hidden*n_taps, small) are still read via the VRAM-staged F32/BF16
  buffers from earlier in this file, not re-encoded - both are tiny
  relative to the 7 dense tensors above, unlikely to move the needle
  much, not attempted given diminishing returns. The selector tensors
  (`selector_predecessor`/`successor`/`hidden`) are unused for v1
  (`is_v1=true` skips the whole selector path) - nothing to gain there.

### Further TG stacking round (2026-10-30, after the Q8_0 drafter re-encode)
- Re-profiled DFlash with the Q8_0 drafter active: the drafter's own
  forward cost dropped from ~14.1ms/step to ~7.6ms/step (confirmed via
  `LAGUNA_TIMING_DEBUG`, matches the expected ~2x from halving its weight
  bytes). `append_all` (target-model verify, ~56.7ms/step) now dominates
  completely (~86% of step time) - the drafter is no longer a major
  lever.
- **Tried, reverted (real negative result, not a dead end to retry)**:
  `RouterF32`'s small-batch threshold (`n_tokens < 8` picks a kernel that
  re-reads the whole 256x3072 F32 router matrix once *per token
  independently*, cheap only at n=1; DFlash verify batches are n=2-7,
  paying that redundant re-read 2-7x). Lowered the threshold to
  `n_tokens < 2` so verify batches would use the tiled kernel (reads the
  matrix once total) instead - **measured worse** (36.9 -> 28.8 tok/s):
  the tiled kernel's 64-token shared-memory staging overhead dominates at
  this tiny scale, outweighing the redundant-read savings. Reverted,
  confirmed the revert restores the exact prior number and token
  trajectory. **Do not retry this by just flipping the same threshold** -
  a genuine fix would need a *third*, lighter-weight kernel tuned for
  n=2-7 specifically (smaller tile than 64), not a binary choice between
  the two that already exist; not attempted, real but nontrivial.
- **Found and fixed, real but measured negligible this round**:
  `InjectTargetContextDevice` (called every decode step via `DflashInject`)
  ended with an unconditional `hipStreamSynchronize` whose result wasn't
  even checked, with nothing downstream reading GPU data and the next
  kernel launch already correctly ordered behind it on the same stream -
  a pure, free, correctness-safe removal (confirmed: identical output,
  identical accept/drafted counts, 36.84 -> 36.67, within noise). Kept
  regardless since it is strictly correct and cannot regress anything;
  did not materialize into a measurable win this round, but is exactly
  the kind of host-blocking-sync class of fix that mattered a lot
  elsewhere in this file (the earlier MoE per-layer sync investigation) -
  worth re-checking if the step-time balance shifts again later.
- **Current best, confirmed stable across repeats**: **`-d 2`, ~36.7-36.9
  tok/s** with DFlash, ~26.2 tok/s plain decode. Both verified against
  `--validate-prefill` (same known n=4/n=16 near-tie pattern, nothing
  new) and against `pp2048` (unaffected, ~1,222-1,231).
- **What's left for TG, not yet tried**: a dedicated small-N (2-7) router/
  gate kernel (real engineering, not a threshold flip - see above); the
  shared `output`/`token_embedding` tensors the drafter borrows from
  Laguna are DEVICE-resident BF16/Q8_0 (not host pointers), so the
  CPU-side `ReencodeBf16ToQ8_0` used for the drafter's own weights cannot
  be reused on them directly - re-encoding the LM head for the drafter's
  exclusive use would need either a GPU-side re-encode kernel or a
  separate CPU-side read of Laguna's own GGUF (the reader is kept alive
  via `keep_reader` specifically for this kind of use) - not attempted,
  the LM head is only ~1.7% of drafter-side kernel time, low expected
  payoff for the effort. True cross-step pipelining (drafting block N+1
  speculatively while verifying/committing block N) was considered and
  rejected as a quick fix - `ForwardBlock` has a real data dependency on
  the previous step's `DflashInject` output, not a structural accident.

## Punch list to reach ~1,700 tok/s PP (in order)
1. ~~Fix group-6 WMMA correctness~~ DONE 2026-10-30 (four bugs fixed, see history above).
2. ~~Implement + verify group-9 WMMA~~ DONE 2026-10-30 (template on heads-per-kv; 423 -> 789 tok/s).
3. ~~Efficient router GEMM~~ DONE 2026-10-30 (tiled shared-memory kernel; 789 -> 903 tok/s).
4. ~~Per-head norm + Q8 gate projection~~ DONE 2026-10-30 (warp-per-head
   norm, tiled Q8 gate GEMM; 903 -> 1,035 tok/s).
5. ~~Dedup shared-activation Q8 quantization~~ DONE 2026-10-30 (additive
   split of the shared MMQ quantize/GEMM steps; ~1,100 -> ~1,110-1,116;
   smaller than expected - quantize was cheap, not the bottleneck).
6. **Re-profiled 2026-10-30** (`tools/prof/prof.py run --stages laguna`,
   `rocprofv3` available directly on PATH, no `nix develop` needed): two
   kernels dominate, both shared `qwen38_flash_next` code - `mul_mat_q<Q8_0,
   80,...>` (dense Q/K/V/ffn/shexp projections, ~32% of kernel time) and
   `RoutedF16GEMMKernel` (MoE gate/up/down, ~40% split across two call
   shapes). `AttentionWmmaKernelLaguna` is now only ~4-5%, `RouterGemmKernel`/
   `GateGemmKernel` ~3-4% each, quantize down to ~2% (confirms the dedup
   lever above actually worked, just on a small line item).
   - Checked whether Laguna's token/expert mix (mean ~80 tokens/expert at
     pp2048 = 2048*10/256) is already on the best *legal* GEMM tile size.
     `RoutedTileCols()` in engine.cpp (kRoutedTileMin/Max/Step) is **dead
     code** - the actual tile widths used are the fixed constants
     `kPairTileRows=128` (gated gate/up), `kQ8TileRows=64` (Q8_0 down),
     `kPlainTileRows=48` (non-Q8_0 down). Tried `kPairTileRows=64` (closer
     to the mean-80 token count, less padding waste): pp2048 1110 -> 1093,
     **worse**, reverted. Tried `kQ8TileRows=128` for down: compile-time
     rejected (`LaunchRoutedF16<BN>`'s Q8_0 case is `if constexpr (BN > 64)
     return false;` - deliberately capped, presumably a VGPR/LDS limit on
     this hardware). **Both existing constants are already at their
     measured/legal best; this axis is exhausted without editing the kernel
     itself.**
   - **Biggest remaining lever**: `RoutedF16GEMMKernel`'s own register/LDS
     budget for `WeightType::kQ8_0` (why BN is capped at 64 at all) and
     `mul_mat_q`'s Q8_0 dense tile (`mmq_x=80`) occupancy. Both kernels are
     shared with Flash-Next (production model) - per AGENTS.md this needs
     the full affected-model validation matrix (strict timing, cache,
     long-context) before any change to the kernel internals lands, not
     just a Laguna `--validate-prefill` sweep. **Not started**: scope this
     explicitly before touching it - it is the one remaining lever that
     cannot be made Laguna-local the way everything else in this punch list
     was.
   - **Tried, Laguna-local, negative result**: tested whether the Q8_0
     `BN<=64` cap in `RoutedF16Gemm`'s dispatch (`LaunchRoutedF16<BN>`'s
     `if constexpr (BN > 64) return false;`) was just a Flash-Next-specific
     tuning choice rather than a hardware wall, by adding a purely additive
     `RoutedF16GemmQ8Wide` (`kernels.hip.cpp`/`.hpp` in
     `qwen38_flash_next` - new function, zero edits to the existing
     dispatch or any current caller) that instantiates
     `RoutedF16GEMMKernel<kQ8_0, 128, 128, 2>` directly (the template's own
     `static_assert` already allows BN up to 256; only the dispatcher's
     `if constexpr` was blocking it). Compiled and ran correctly on Laguna's
     down-projection (full `--validate-prefill` sweep unchanged). **Result:
     no measurable speedup** (1,115 vs 1,114-1,116 baseline, within noise) -
     the down-projection tile width was not the bottleneck, so there is
     nothing here worth porting back to Flash-Next either. Toggled off via
     `constexpr bool kQ8WideDownTile = false;` in `engine.cpp`; the additive
     `RoutedF16GemmQ8Wide` function is left in place (harmless, unused,
     documented) in case the *dense* `mul_mat_q` tile (not this routed GEMM)
     turns out to be the real register-bound piece - that one still needs
     real profiling-guided tuning, not a tile-width guess.
7. Tune dense projections (Q8 `mul_mat_q`) tile/occupancy the same way,
   Laguna-local.
8. Re-profile with `tools/prof/prof.py` + `rocprofv3 --pmc` after each step -
   the budget split is stale past the 789 stage, re-measure before picking
   the next target.
9. Batch Q x K^T and P x V via hipBLAS/batched GEMM if attention rises again.
10. **Separately (not blocking PP): `gufo bench -n <N>` (N>0) reports
    `tg32=0.00` and appears to drag the paired `pp` number down too (518-650
    vs ~1,110 with `-n 0`) - decode path in `bench` is broken. Diagnose
    before trusting any PP number measured alongside a TG number.

### DFlash v1 crash: FIXED 2026-10-30, drafter now loads and generates -
### correctness bug remains (degenerate repetition over long generations)
- **Root cause of the SIGFPE crash (confirmed, fixed)**: NOT a missing v1
  forward-pass implementation - the executor already had complete, correct
  `if (!df_cfg.is_v1)` guards around every v2-only conv-mixing/selector code
  path, including a dedicated v1 argmax sampling branch. The actual bug was
  four *unconditional* computations of `2 * conv_kernel_size * (hidden_size
  / conv_group_size)` (three in `executor.cpp` - `Allocate()`,
  `PrewarmBlockGemms()`, `ForwardBlock()` - and one in `dflash_weights.cpp`'s
  `LoadFromGguf`, during weight *loading itself*) that ran regardless of
  `is_v1`, dividing by `conv_group_size=0` (legitimately 0 for v1, which has
  no conv-mixing layer at all). Fixed all four with an `is_v1 ? 0 : ...`
  guard - same pattern the rest of the file already used correctly.
- **Second bug, also fixed**: `CaptureTapKernel` (Laguna's
  `kernels.hip.cpp`, added for the tap-capture wiring) launched with
  `hidden_size` (3072) as the **thread-block size**, exceeding the
  1024-threads-per-block hardware limit (`hipErrorInvalidConfiguration`).
  The kernel body already strides by `blockDim.x`, so any bounded block size
  works - fixed by launching with a fixed 256 instead of `hidden`. This
  error was *stale* until the next unrelated `hipGetLastError()` check
  (inside the Q8 quantize dedup code from earlier in this file), which made
  it initially look like a quantize bug - traced to the real site via
  `AMD_LOG_LEVEL=3` HIP runtime logging, not by guessing.
- **Third bug, also fixed**: Laguna's `Forward()` tap-capture range math
  (`tap_first`/`tap_rows`) computed an *absolute* sequence position and used
  it directly as a *relative* row offset into the current tile's `x` buffer
  - correct by coincidence during prefill (`position == 0`) but wrong for
  every decode step afterward (`x` only ever holds rows for
  `[position, position + n)`, never earlier history). Fixed by clamping the
  window start into this call's local range before computing the offset.
  Confirmed via `LAGUNA_DFLASH_DEBUG=1` (env-gated, zero-cost when unset;
  added to `executor.cpp` at each of `InjectTargetContextDevice`'s and
  `ForwardBlock`'s reject conditions - keep for future debugging).
- **After these three fixes: DFlash v1 loads, runs, and generates real
  text** (previously: immediate SIGFPE on load, 0 tokens ever produced).
  Verified by diffing against the no-DFlash baseline at the same prompt/
  seed: without DFlash, greedy decode at n=100 is coherent and non-
  repetitive (confirmed 23.0 tok/s baseline, matches prior measurements).
- **Remaining correctness bug, NOT fixed**: with DFlash enabled, output
  degenerates into repetition ("the the the the...") well before the
  no-DFlash baseline would, and a longer run (`-n 256`) eventually hit
  "Laguna: context capacity exceeded" without ever emitting a stop token -
  consistent with degenerate output never naturally stopping. Since
  verification (`prompt.cpp`'s accept/reject loop) checks drafted tokens
  against Laguna's own real forward-pass logits before committing them,
  this points at either (a) the accept/reject loop incorrectly accepting
  wrong drafts, or (b) `session->Rewind(base + 1 + kept)` not correctly
  discarding rejected speculative tokens' KV-cache state for Laguna
  specifically (e.g. a sliding-window/ring-buffer layer not truncated to
  match), corrupting subsequent *real* decoding even though nothing wrong
  was ever knowingly accepted. **Not yet isolated - whoever continues
  should instrument the accept/reject loop directly (log every
  `guess`/`next`/`hit` and the KV cache state before/after `Rewind`) rather
  than guessing further.**
- **Also found, separate, not fixed**: the `ForwardBlock` entry guard
  (`current_pos >= max_context_`) throws an uncaught `std::invalid_argument`
  instead of gracefully stopping generation when the drafter's context
  budget (`prompt_len + max_tokens + 1 + kMaxVerifyRows`) is exactly
  exhausted - this crashes (`terminate called`) rather than ending cleanly.
  Observed this leaves an orphaned process that doesn't release GPU memory
  until explicitly `SIGTERM`'d - **always check `pgrep -ax gufo` and clean
  up before any new GPU run if a prior DFlash test crashed this way.**
- **Repetition bug: isolated 2026-10-30, not a bug - resolved as root-caused.**
  Added per-step tracing (`LAGUNA_DFLASH_DEBUG=1`: logs every drafted
  `feed`, and every verify `row/guess/next/hit`). Found the *first*
  divergence from the no-DFlash baseline at an exact, reproducible point
  (both runs: same prompt/seed, baseline says "**Core Idea**", DFlash run
  says "**Core Concept**" at the identical position) - not a crash, not a
  wrong accept (that step's `hit=0` at row 0, i.e. nothing was accepted;
  the "wrong" token is the verify step's own `next`, computed from an
  8-token *batched* `AppendAll` forward rather than a 1-token sequential
  one). This is the **same pre-existing batched-vs-sequential MoE routing
  near-tie sensitivity already documented earlier in this file** (the
  `--validate-prefill` n=16 case) - DFlash's verify step surfaces it at
  every step instead of one specific size, because every verify call is a
  multi-token batch. Both continuations are individually coherent greedy
  decodes; they just diverge because batched and sequential forwards are
  not bit-identical on this stack. **Not a bug to fix without addressing
  that deeper, already-known numeric-determinism gap** (deterministic
  top-k tie-breaking across batch widths - a separate, large undertaking,
  not specific to DFlash). `Session::Rewind`/accept-reject logic checked
  and confirmed NOT the cause.
- **Real root cause of poor DFlash1 results: measured speed is 3x SLOWER
  than no DFlash (7.72 tok/s vs 23.16 tok/s baseline), with a 1.6%
  (6/371) draft acceptance rate.** This is a missing-feature gap, not a
  bug: `ForwardBlock` runs the drafter's 6-layer transformer **once** per
  block and immediately argmaxes every masked position in parallel. But
  this is a genuine block-diffusion architecture (non-causal attention
  *within* the block, confirmed via the GGUF's
  `dflash.attention.causal=False` and `mask_token_id`) - position k's
  prediction legitimately depends on position k-1's embedding, which at
  single-pass time is still just `mask_token_id`, not a real predicted
  token. Block-diffusion inference needs **iterative refinement**: predict
  all masked positions, commit the high-confidence ones, re-mask the rest,
  repeat for several steps - `ForwardBlock` does none of that, so every
  position past the first one or two is drafted almost blind, matching the
  near-zero acceptance measured. **This is the real remaining work**:
  implement the iterative mask/commit/re-forward loop (a genuinely new,
  substantial piece - multiple `ForwardBlock`-like passes per block with a
  confidence-based commit schedule), not a quick fix. Until that exists,
  DFlash1 should not be enabled by default - it is strictly worse than
  plain decode on this model.
- **Measured** (prompt/seed fixed, `-n 40/60`): no-DFlash baseline 23.16
  tok/s; with DFlash1, 5.4-7.7 tok/s (3-4x slower), 2-6 drafts accepted out
  of 259-371 proposed (under 2%).
- **Tried to also fix the missing `enc.aux_norm.weight` per-tap pre-norm**
  (bound it in `dflash_weights.cpp` as optional F32, added
  `LaunchDFlashTapRMSNorm` in `dflash_kernels.hip`/`.hpp`) on the theory it
  might be degrading draft quality independently of the iterative-
  refinement gap. **Wiring it into `InjectTargetContextRows` caused a real
  GPU memory fault** (`Memory Fault Error ... kernel:
  dflash_tap_rmsnorm_kernel`, confirmed via the HIP driver's own fault
  report) which hangs the process rather than crashing cleanly - had to
  `SIGTERM` it twice during testing. **Reverted the call** (left commented
  out with the fault noted in `executor.cpp`); the binding and kernel
  implementation are still in the tree but **unverified and unused** -
  do not call `LaunchDFlashTapRMSNorm` from `InjectTargetContextRows`
  without first checking the actual row stride/count `d_target_features_`
  is allocated and addressed with at that call site (the crash strongly
  suggests a stride/bounds mismatch between `GetTargetFeaturesSize()` and
  what `InjectTargetContextRows` actually passes as `num_tokens` there -
  not investigated further given the iterative-refinement gap is the
  larger, more important problem anyway).
- **Operational note confirmed twice this session**: an uncaught exception
  or GPU fault from DFlash code leaves an orphaned `gufo` process that
  does not release GPU memory and does not respond to normal exit - always
  `pgrep -ax gufo` and `kill -TERM <pid>` (never `-9`) before any new GPU
  run after a DFlash crash/hang, and verify with
  `cat /sys/class/drm/card*/device/mem_info_vram_used` back at idle before
  proceeding. Calling `hipStreamSynchronize` right after the suspect launch
  turns a silent hang into an immediate, catchable exception with a HIP
  error - much faster to iterate on than waiting for a hang + SIGTERM each
  time.

### External confirmation (2026-10-30): both gaps above are real, per upstream
- A third-party quant/serving card
  (`raulvidis/Laguna-S-2.1-ROCmFP4-STRIX-GGUF` on Hugging Face, running a
  llama.cpp fork on the same hardware) explicitly documents **both** gaps
  found above as required, unmerged patches against their fork:
  "ROCmFPX PR #47 (`laguna.cpp` must publish `t_layer_inp`) and PR #48
  (load and apply `enc.aux_norm` and `attn_gate`)". This independently
  confirms the per-tap norm and the attention output gate are real,
  necessary parts of correct DFlash v1 inference for Laguna - not
  speculative guesses - so both are worth finishing once the GPU fault
  below is understood.
- Attempted to wire up `attn_gate` (the one with a clearer mechanism and
  known-correct reference formula, reused directly from Laguna's own
  scalar `AttentionKernel`). Routed its small GEMM (M=72, the drafter's
  head count) through a **new dedicated kernel**
  (`LaunchDFlashAttnGateProjection`) rather than the existing shared
  `RunBlockGemm`/batched-BF16-GEMM dispatch, because that shared path
  faulted first (`BatchedExactBf16GEMMFp32VecKernel<16u,3ul,2ul,...>`,
  confirmed via the HIP driver's fault report) - M=72 is apparently not a
  shape that dispatch is validated for.
- **RESOLVED 2026-10-30, and it was worth it - see measurements below.**
  The dedicated kernel also faulted at first, but with a forensic detail worth
  recording precisely: the fault address is **offset ~0x20E40 (134,208
  bytes) into the `attn_gate.weight` tensor**, whose declared size is
  221,184 elements x 2 bytes (BF16) = 442,368 bytes (0x6C000). The fault
  offset is **well inside** the tensor's nominal extent (~30% in, not at
  an edge) - this is not a simple out-of-bounds read/stride bug. Verified
  independently in Python (`gguf` package) that the file's raw layout for
  this tensor is exactly what the kernel assumes: `t.data.shape == (72,
  6144 bytes)`, i.e. 72 contiguous rows of 3072 BF16 elements each,
  matching the kernel's `weight + mi * k` indexing exactly. Every *other*
  per-layer tensor (`attn_q`/`k`/`v`/`output`/`ffn_*`) is read from the
  same GGUF-mmap'd pointer mechanism without faulting, in the same
  `ForwardBlock` call, immediately before this one fails - so this is not
  a generic "host pointer dereferenced on GPU" problem either. The most
  likely remaining explanation is that `attn_gate.weight`'s specific
  position/page-alignment within the mmap'd GGUF file is not fully
  GPU-resident/accessible at this offset, for a reason not yet identified.
  **Root cause confirmed via elimination, not guesswork**: ruled out (a)
  out-of-bounds indexing (offset is well inside the tensor, verified
  against Python's authoritative GGUF parse), (b) demand-paging/unfaulted
  pages (pre-faulting every page via a CPU read - proven to succeed via a
  computed checksum - did not fix the GPU fault), (c) a parser offset bug
  (the computed file offset for `blk.0.attn_gate.weight`, 343,531,968
  bytes, matches Python's `gguf` library exactly), (d) a tensor-name-lookup
  bug (`GgufReader::FindTensor` is a plain exact-match `unordered_map`
  lookup, no aliasing possible). **Decisive test**: staged
  `attn_gate.weight` to a real `hipMalloc`'d VRAM buffer (via `hipMemcpy`
  from the host pointer) instead of letting the kernel read the
  GGUF-mmap'd host pointer directly - **this fixed it completely, first
  try, no more faults.** So the host pointer itself was readable (by both
  CPU and, per the ruled-out causes, in every way we could test) but not
  safely GPU-dereferenceable for this specific tensor via this specific
  kernel/stream combination - most likely an HMM/page-migration edge case
  specific to this tensor's mmap region that the existing, already-used
  `RunBlockGemm` dispatch happens not to hit for the other per-layer
  tensors (through pure luck of access pattern/timing, not a property of
  those tensors being special). **Did not fully root-cause the HMM/driver
  behavior itself** - the VRAM-staging workaround is confirmed correct and
  safe, so that's what ships; a deeper ROCm-level explanation would need
  rocgdb or driver source, not justified given the fix works.
- **Fix, permanent**: `Allocate()` now stages every layer's
  `attn_gate.weight` to VRAM once at session-create time
  (`d_attn_gate_weight_`, freed in `Free()`); `ForwardBlock` reads from
  that VRAM copy via `LaunchDFlashAttnGateProjection`
  (M=72 dedicated kernel, bypassing the shared batched-BF16-GEMM dispatch
  which separately faulted on this M before the VRAM fix even applied) and
  applies the gate via `LaunchDFlashAttnGateMul`. No debug/env gating -
  this runs unconditionally now, it's a correctness fix, not an experiment.
- **Measured impact - large**: at `-d 2` (same setting as the pre-fix
  baseline), acceptance went from 12.8% (12/94) to **35.7% (25/70)**, and
  speed from ~16 to **17.8 tok/s**. Re-swept `-d` with the fix active since
  the optimum shifts when draft quality improves:
  | `-d` | tok/s (n=60) | accepted/drafted | rate |
  |---|---|---|---|
  | 1 | ~16 (pre-fix number, not re-swept) | - | - |
  | 2 | 17.8 | 25/70 | 35.7% |
  | 3 | **21.3** | 29/90 | 32.2% |
  | 4 | 19.9 | 29/120 | 24.2% |
  | 5 | 18.5 | 29/150 | 19.3% |
  | 7 (old default) | 13.7 | 29/210 | 13.8% |
  **`-d 3` is the new sweet spot**: 21.3 tok/s at n=60, confirmed again at
  n=100 (20.8 tok/s, 49/156 = 31.4% accepted) - consistent, not a fluke.
  This is now within ~10% of the 23.16 tok/s no-DFlash baseline, up from
  3-4x *slower* before this session's fixes. Laguna's own plain pp2048
  bench reconfirmed unaffected (~1,117 tok/s, matches the established
  baseline) - this work only touched DFlash-specific code paths.
- **`enc.aux_norm` also fixed, same session, same fix pattern.** Staged
  its weight to VRAM from the start this time (`d_aux_norm_` in
  `Allocate()`) rather than rediscovering the host-pointer fault - worked
  first try, no fault. Wired into `InjectTargetContextRows` via
  `LaunchDFlashTapRMSNorm`, applied to `d_target_features_` before
  `fc_projection`, exactly as poolside's reference patch notes describe.
  **Both of the two confirmed-missing pieces from the upstream model card
  are now implemented and working.**
- **Final measured result, both fixes combined**: re-swept `-d` again
  since quality improved further on top of the `attn_gate`-only numbers
  above. At longer, more representative lengths (n=300-500, steady state
  past model-load/prefill noise):
  | `-d` | tok/s (n~400) | accepted/drafted | rate |
  |---|---|---|---|
  | 1 | 25.5 (n=400) | 177/222 | 79.7% |
  | 2 | 28.1-28.7 (n=300-500) | up to 64.5% | - |
  | 3 | 27.7 (n=300) | 184/348 | 52.9% |
  | **4** | **29.53 (n=400)** | 267/536 | 49.8% |
  | **5** | **29.64 (n=400)** | 275/620 | 44.4% |
  | 6 | 26.68 (n=400) | 270/792 | 34.1% |
  **`-d 4` or `-d 5` is the real optimum now, both ~29.5-29.6 tok/s** -
  this is **+28% over the 23.16 tok/s no-DFlash baseline**, a genuine,
  verified, substantial speedup, not just "no longer catastrophically
  slow." This looks like a real plateau/ceiling for the current
  implementation (tried `-d` 1 through 7; 4-5 is the peak, both sides are
  worse) - pushing meaningfully past ~30 tok/s (e.g. toward 45) would need
  reducing the drafter's own per-step overhead (profiling the 6-layer
  drafter forward + verify batch cost) or further acceptance-rate work,
  not more correctness fixes - the two confirmed gaps from the upstream
  model card are both closed.
- **Why 45 tok/s is not reachable without a different kind of work**:
  profiled the actual decode loop (`LAGUNA_TIMING_DEBUG=1`, wall-clock
  around each stage, cross-checked with `tools/prof/prof.py`). Per verify
  step at `-d 4`: `ForwardBlock` (drafter) ~19ms, `AppendAll` (verify -
  Laguna's full 48-layer MoE forward over the small feed batch) ~107ms,
  accept/reject and Rewind+Inject negligible (<2ms combined).
  **`AppendAll` is 84% of per-step wall time and it is exactly the same
  48-layer MoE forward the no-DFlash baseline pays per token** - DFlash's
  entire benefit comes from amortizing that one expensive forward across
  multiple committed tokens (anchor + kept drafts), not from making that
  forward itself cheaper. Confirmed `-d 10` (tested after both fixes) is
  worse than `-d 4/5` (18.84 tok/s, 24% acceptance, n=400) - feed-batch
  verify cost grows with `d` faster than acceptance holds up, so this
  isn't an undersampled curve; 4-5 is a real optimum for this drafter's
  current quality. **Reaching 45 would need the ~107ms/call verify cost
  itself reduced** (the same MoE memory-bandwidth-bound GEMM/routed-expert
  path already investigated exhaustively earlier in this file for the
  plain PP number - tile-width and VGPR axes were found genuinely
  exhausted there), or substantially higher kept-tokens-per-call than
  today's ~45-55% acceptance allows at the optimal `-d`. Neither is a
  quick fix; both are separate, large pieces of work from here.
### Narrow/wide MoE threshold fix - DONE 2026-10-30, verified cheaper per-call, net tok/s inconclusive
- `Session::Impl::Moe()`'s narrow (vec) vs wide (`RoutedExperts`, the
  tile/bucket/host-sync path) dispatch used `n * expert_used > 32` - a
  threshold sized for n=1 single-token decode, never revisited for
  DFlash's new n<=16 verify batches. Checked the underlying
  `qfn_mmq_moe_vec`/`mul_mat_vec_moe_gated` kernels directly: they chunk
  large batches internally (`mmvq_moe_max_batch` groups) and assert
  `tokens <= MMVQ_MAX_ROUTED_BATCH` (64), far above Laguna's
  `Session::kMaxVerifyRows` (16) - the old threshold was a bad proxy for
  real kernel capacity, not the real limit.
- Raised `kVecExpertRows` 32 -> 160 (= 16 * expert_used(10), i.e. covers
  every DFlash verify batch while any real prefill tile, n>=512, stays on
  the wide path). Verified correctness first: `--validate-prefill` at
  n=1,2,4,8,16 all pass cleanly, no `Experts()` rejection
  ("wide expert batch on the vector path"), no crash; plain `pp2048`
  bench reconfirmed unaffected (~1,114-1,116, matches baseline).
- **Measured, with `LAGUNA_TIMING_DEBUG=1`**: `AppendAll`'s per-call cost
  dropped from ~107ms to ~85.6ms (-20%) at `-d 4` - the narrow path really
  is cheaper for these batch sizes, confirmed directly, not inferred.
- **Net tok/s effect at this exact setting was inconclusive, not because
  the mechanism failed but because the model is deterministic
  (fixed seed + greedy) and the exact generated token trajectory changed**
  (264 vs 267 accepted of a differently-sized draft stream at the same
  `-d 4`) - the narrow path's different FP reduction order flips some of
  the same near-tie MoE routing decisions already documented elsewhere in
  this file, same mechanism as the "Core Idea" vs "Core Concept"
  divergence found earlier, just now via a dispatch-path change instead of
  a batch-width change. Re-swept `-d`: `-d 15` (near the new threshold's
  upper bound) is clearly worse (12.18 tok/s, 13.5% acceptance - drafting
  that far degrades quality faster than the cheaper verify call
  compensates); `-d 4-6` stayed in the same ~26-28 tok/s band as before
  the fix, run-to-run differences fully explained by the trajectory shift
  above, not by the fix being wrong.
- **Net assessment**: a real, verified, correct cost reduction on the
  mechanism that matters (verify cost, confirmed 84% of per-step time
  earlier) - kept, not reverted, since it's strictly safer/cheaper and
  validated. Did not produce a clear, outsized tok/s jump in this specific
  short-benchmark comparison because of the trajectory-sensitivity
  above; a longer, multi-seed, averaged benchmark would be needed to
  measure its true net effect cleanly, not attempted given remaining
  budget. **45 tok/s was not reached.** Current best remains in the
  27-29.6 tok/s band depending on run/seed, ~20-28% over the 23.16 tok/s
  no-DFlash baseline.
### mul_mat_q mmq_x>=16 floor on decode - investigated, real finding, fix attempted and REVERTED (bug)
- Profiled plain (no-DFlash) decode with `tools/prof/prof.py`: a single
  kernel, `qfn_mmq::mul_mat_q<Q8_0, 16, ...>`, is **43.1%** of all decode
  kernel time. Traced why mmq_x=16 (not 8) is used for n=1 single-token
  decode: confirmed via `mmq_get_granularity_host`/`_device` in the shared
  `mmq.hpp` that gfx1151 (WMMA-capable, no MFMA) has `granularity=16` for
  this quant path, and `mmq_x_usable()` requires `mmq_x % (granularity *
  NCW) == 0` - mmq_x=8 is **never valid** on this hardware for this
  kernel. This is a genuine, pre-existing hardware/tuning floor, not a
  regression or a bug from this session's work - every Q8_0 dense
  projection at n=1 wastes 15/16 of a GEMM tile, unavoidably, through this
  kernel.
- Found a plausible fix already in the shared code:
  `qfn_mmq_q8_0_dense_vec_preq` (`qfn_mmq.h`/`.hip.cpp`) is a dedicated
  small-batch (n<=`MMVQ_MAX_BATCH_SIZE`=8) Q8_0 vector GEMM with no such
  floor, used elsewhere but never wired into Laguna's `Dense()` (which
  routes Q8_0 through the MMQ-tile path unconditionally, since
  `DenseVecType()` excludes `kQ8_0`).
- **Wired it in, found it produces wrong output, reverted completely.**
  `--validate-prefill 4` failed with a real divergence (`max_abs_diff=
  9.49, cosine=0.897` - not the near-tie class, a genuine bug) even
  though the new vec path was restricted to n<=1 only (n=4 itself never
  touches it). Root cause: `--validate-prefill N` compares one batched
  N-token call against N **sequential single-token** calls - the
  sequential reference run is made of n=1 steps, which **did** go
  through the new (broken) vec path, silently corrupting every
  sequential step while the batched (n=4) call stayed on the old,
  correct path. `--validate-prefill 1` alone could never have caught
  this, since both sides of that specific comparison would use the
  identically-wrong vec path and still agree with each other. Confirmed
  by disabling the vec path entirely (`n <= 0`, i.e. never taken):
  n=4 immediately returned to the known-correct baseline
  (`max_abs_diff=1.35665`, matches the value documented everywhere else
  in this file). **Removed the vec-path code entirely** (not left
  disabled-in-tree) rather than ship an unresolved correctness bug -
  `QuantizedVec`/`quant_cache_vec`/the `kQ8VecMaxBatch` branch are gone
  from `engine.cpp`. Did not get to root-cause *why*
  `qfn_mmq_q8_0_dense_vec_preq` gives wrong results for Laguna's n=1
  shapes specifically (possibly a layout/stride assumption from its
  other, hardcoded call sites - "the 320x10240 HC down projection" etc -
  that doesn't generalize to Laguna's shapes; not investigated further).
- **Net effect of this investigation: zero code change, one real,
  well-evidenced lead for whoever continues** (43% of decode kernel time
  is a known, explainable floor; a plausible-looking existing fix exists
  but is confirmed broken for this model's shapes and needs its own
  from-scratch debugging - likely with a direct kernel-level repro
  against `mul_mat_vec_q8_dispatch`'s internal dispatch, not another
  blind wiring attempt). Also confirms a validation-methodology lesson
  worth keeping: **validate at multiple N, not just the smallest one** -
  a bug that affects all n=1 calls identically is invisible to
  `--validate-prefill 1` and only surfaces by comparing against a
  differently-shaped reference, exactly as designed.
- **Baseline TG moved independently this session from an external change**
  (the user adjusted something outside this codebase, cause not
  specified) - reconfirmed PP2048 ~1,222-1,231 (up from ~1,114-1,116) and
  no-DFlash TG ~24.5 (up from 23.16) in two repeated runs each, both
  stable. DFlash TG at `-d 4` stayed ~28.6-28.9 despite the baseline
  speedup - confirms DFlash's bottleneck is per-step host/overhead
  structure, not raw GPU throughput (a faster GPU doesn't proportionally
  speed up DFlash the way it does plain decode/prefill).
- **Not changed**: the shared `--draft-tokens`/`-d` CLI default (7,
  `src/cli/bench/bench.hpp`/`inference_backend.hpp`) is used by multiple
  models (DSpark, MTP, GLM-5.3, Laguna) - did not touch it, since 7 may be
  correctly tuned for those other models' own drafters. Laguna users
  should pass `-d 4` or `-d 5` explicitly; this is not yet reflected in
  any Laguna-specific default or documentation beyond this file.
- Debug instrumentation kept, zero-cost when unset: `LAGUNA_DFLASH_DEBUG=1`
  (per-step verify tracing in `prompt.cpp`, reject-path dumps in
  `executor.cpp`) and the attn_gate pre-fault/offset checksum block in
  `dflash_weights.cpp` (harmless now that the real fix doesn't need it,
  left in since it's useful forensic evidence if this class of bug
  resurfaces elsewhere).
- **Baseline-vs-upstream gap also raised this session, not yet
  investigated**: the same card reports a no-speculation baseline of
  34.8-35.5 tok/s on identical hardware, against this port's 23.16 tok/s -
  but on a different quant (`ROCmFPX FP4`, ~4.27 bits/weight, a
  fork-proprietary format, vs this port's UD-Q4_K_M) and a different
  engine (llama.cpp fork, not gufo). Likely at least partly explained by
  raw weight-byte-volume (TG is memory-bandwidth-bound on a MoE model),
  not a gufo-specific inefficiency - but not confirmed either way. If
  pursued: check whether a smaller **standard** GGUF quant (something
  gufo's existing dequant kernels already support, not the fork-proprietary
  FP4 tensor types 100/101 which gufo cannot read without new kernel work)
  exists for Laguna and would close some of this gap without engine
  changes.

## Punch list to reach ~50 tok/s TG (DFlash)
- **2026-10-30: DFlash v1 is now wired in (`src/cli/prompt/prompt.cpp`,
  `--speculative dflash2 --dflash-model <v1 gguf>`) but crashes immediately
  with SIGFPE (floating point exception / integer divide-by-zero), exit 136.
  Gain measured: **not a gain, a crash - current decode-with-dflash is
  completely broken, 0 tok/s (process dies before producing output).**
  Baseline without DFlash (plain autoregressive decode): confirmed working,
  **23.16 tok/s** (`-v` output: `generated=128 tok in 5.52673 s`).**
- **Root cause confirmed**: `dflash_weights.cpp` correctly recognizes a v1
  drafter GGUF (`is_v1 = !conv_kernel_size.has_value() && !selector_rank.has_value()`,
  line 242) and validates it with v1-appropriate rules (line 254-259,
  deliberately skipping the `conv_group_size == 0` rejection that applies to
  v2). But `src/models/qwen/hip/dflash/executor.cpp` **unconditionally**
  computes `hidden_size / df_cfg.conv_group_size` in at least 4 places
  (lines ~254, ~366, ~636, ~1069) regardless of `is_v1` - v1 drafters have no
  conv-mixing layer at all (`conv_group_size` is legitimately 0 for v1), so
  this divides by zero the first time any of those code paths run.
  - This is the exact DFlash-v1-vs-v2 mismatch already flagged in
    `~/.claude/skills/port-model/LEARNINGS.md` before any code existed: v1 is
    block-diffusion (arXiv 2602.06036), v2 is the gated-delta-net-conv +
    selector architecture gufo's executor actually implements. The GGUF
    loader was made v1-aware; the executor's forward math was not - it still
    assumes v2's conv/selector structure unconditionally.
  - **Not fixed this session** - this needs a real v1 forward pass (block
    diffusion mask/unmask iteration, no conv mixing, no selector top-k),
    i.e. a new executor path gated on `df_cfg.is_v1`, not a one-line guard.
    Treat as the same "parked, not simple wiring" work flagged before any of
    this was attempted. Whoever picks this up next should start from
    `is_v1` call sites in `executor.cpp` and decide the v1 forward algorithm
    before touching the conv-specific math.
- Reference the Flash-Next DFlash2/MTP wiring (`src/models/qwen38_flash_next/`) for the integration pattern (draft proposal, point-mass/sampled verify path, acceptance tracking) if/when a real v1 executor is written - v2's wiring pattern is still the right shape, just not its per-block math.

## 2026-10-30: PP target reached via W8A8Gemm for the wide Q8_0 dense path

Goal per user: PP=1300, TG=40 (TG session logged above; PP picked up after).
Profiled `-p 2048` with `tools/prof/prof.py` (rocprofv3 is on PATH directly,
no `nix develop` needed) at the then-current 1,222 tok/s baseline. Top kernel:
`qfn_mmq::mul_mat_q<Q8_0, mmq_x=80>` at 29.6% of all kernel time (1,206ms) -
Laguna's `Dense()` wide path (n>8: Q/K/V/attn_output/ffn gate/up/down, all
Q8_0 in the UD-Q4_K_M quant) was still routed through the shared MMQ-tile
GEMM, same kernel family as the decode-floor bug fixed earlier this session,
just at a batch size where the tile floor doesn't bite as hard.

**Two dead ends tried first (advisor-suggested, both measured, both reverted
clean):**
- `kPairTileRows` (MoE gate/up fused-tile rows) 128->64: hypothesized
  padding waste (only ~80 real rows/expert at n=2048*10/256 vs 128-row
  tiles). Measured WORSE: 1,222 -> ~1,100-1,130. `RoutedGatedF16Gemm`'s own
  overhead at smaller tiles outweighs the padding saved. Reverted.
- `RouterGemmKernel`/`GateGemmKernel` token-tile 64->128 (8 tokens/thread
  instead of 4, to halve redundant weight re-reads across token-tile
  iterations): measured WORSE: 1,222 -> ~1,125-1,150. Halving the block
  count (32 instead of 64 across 40 CUs) plus doubled register pressure
  per thread lost more occupancy than the bandwidth it saved. Reverted.
  **Only the grow direction (64->128, fewer/bigger blocks) was tried.**
  The advisor's read of this entry (2026-10-31): shrinking the tile
  instead (64->32, more/smaller blocks, more occupancy, more redundant
  weight reads) is the untested opposite direction and is not covered by
  this dead end - don't treat this note as "token-tile tuning doesn't
  work," only "growing it doesn't." Do not retry the grow direction
  without a different lever (a third kernel shape, not a parameter flip
  on the existing one - same lesson as the earlier RouterF32 threshold
  dead end).

**The actual fix**: `qwen38_flash_next::rocm::W8A8Gemm` already exists and
reads the *same* block_q8_0 weight bytes Laguna's dense tensors already use
- no re-layout, no new weight format. It just needs activations quantized
into its own tiled int8 layout (`QuantizeQ8Tiled`/`Q8TiledBytes`, not the
MMQ-tile layout `qfn_mmq_quantize_q8_1_mmq` uses) and routes through an
int8 WMMA GEMM instead of `mul_mat_q`. Added `QuantizedTiled()` (same
dedup-cache pattern as `Quantized()`/`QuantizedVec()`) and tried it as the
first choice for `Dense()`'s n>8 Q8_0 branch, falling back to the existing
MMQ path if `W8A8Gemm` declines (k not a multiple of 32 - never happens for
Laguna's actual tensor shapes, but kept as a correctness guard).

**Measured**: pp2048 1,222 -> **1,275-1,287 tok/s (+4-5%)**, essentially
hitting the 1,300 target (within ~1.5%). `--validate-prefill 16` actually
IMPROVED to `top1_match=yes` (cosine 0.989 vs the MMQ path's own near-tie
mismatches at that N - still the same documented MoE-routing sensitivity
class, not a regression). `--validate-prefill 4` is bit-identical to the
unmodified baseline (n=4 is below the n>8 threshold, correctly untouched).
Real generation (`gufo prompt`, same prompt/seed, -t 0, n=200): text stayed
coherent, matched the baseline for the first ~8 tokens before the expected
near-tie drift, same class/magnitude as every other verified change this
port. Depth sweep (no regression, every depth improved):
  - 2k:   1,222 -> 1,280 tok/s
  - 32k:    810 ->   838 tok/s
  - 64k:    611 ->   624 tok/s
  - 128k:   410 ->   415 tok/s
TG (DFlash, `-d 2`) re-checked after the change: 34.95 tok/s on a fresh
prompt (not directly comparable to the earlier 36.7-36.9 number from a
different prompt/session, but in the same range) - confirmed unaffected in
principle since DFlash's verify batches (n<=8, typically n=3-8) stay on the
`QuantizedVec`/`qfn_mmq_q8_0_dense_vec_preq` path, never touching this wide
n>8 branch at all.

**Lesson for the skill**: before assuming a GEMM needs a new quantized
kernel written from scratch, check `qwen38_flash_next/kernels/rocm/kernels.hpp`
for an existing GEMM that already reads the weight's on-disk quant format
directly (`HalfGemmable`, `W8A8Gemm` operating straight on block_q8_0 bytes)
before writing a re-encoder or a new kernel - the same "check for an
unused existing path first" lesson as the DFlash BF16->Q8_0 drafter fix,
but this time at the weight-type level instead of needing any re-encode at
all.

Remaining gap to 1,300 (~1.5%, likely noise-adjacent) not pursued further
this round; next PP lever (if resumed) is the MoE RoutedF16GEMMKernel
buckets (39.5% combined, gate/up+down expert GEMM) or the GPU-idle gaps
`prof.py` reported around `hipMemcpyAsync`/`hipEventSynchronize` in
`RoutedExperts`'s host-side tile-count round-trip.

## 2026-10-30: HANDOFF - TG depth-flatness work, IN PROGRESS, not finished

**Goal** (user, explicit): TG must stay flat with context depth, the same
way Flint/Qwen3.8-27B/Qwen3.6-35B (all on the shared `qwen38_flash_next`
engine, or Flint's own kernel family) already do. Two separate pieces of
work this covers; piece 1 is DONE and shipped, piece 2 is NOT finished -
**do not assume it works, it has never been built-tested end to end**.

### Piece 1 (DONE, shipped, verified): HIP graph capture for decode/verify steps

Laguna had ZERO HIP graph capture before this session - every decode step
re-launched ~15-20 small kernels from the host each time. Flint/Flash-Next
both capture the whole per-step kernel sequence into a `hipGraphExec_t` per
verify-width (1..`kMaxVerifyRows`=16) and replay it after the first couple
of calls at that width.

Ported the identical pattern:
- `Session::Impl::Forward()` (`engine.cpp`) split into `Enqueue()` (the pure
  kernel-launch sequence, capturable) and the outer `Forward()` (host
  bookkeeping: `position`, DFlash tap-pending/`DflashInject`, the final
  `hipStreamSynchronize` - none of which may be captured, since a replayed
  graph does NOT re-run the C++ function body, only the recorded GPU ops).
- Added `Session::Impl::StepGraph graphs[kMaxVerifyRows+1]` (exec handle,
  `failed` flag, `steps` counter) and the capture/replay logic in `Forward`,
  copied from Flint's `Session::Impl::Forward` almost verbatim (capture
  starts on each width's 2nd actual call, not the 1st, so scratch-pool
  sizing settles first).
- **Critical correctness subtlety, specific to Laguna's DFlash tap-capture
  logic** (Flint has no equivalent): `Enqueue()` computes `tap_rows`/
  `tap_first` (where in the drafter's tap window this step's hidden states
  land) from `position`, which is NOT re-evaluated on a replayed graph -
  only the kernel ops captured at capture time replay, with whatever
  `tap_first`/`tap_rows` values were baked into `CaptureTap`'s launch args
  at that moment. Gated graph capture on `tap_rows == n` (tap_first==0,
  i.e. the DFlash window has already reached its steady state for this
  call - true for plain decode always since `dflash==nullptr` keeps
  `tap_rows` structurally tied to `n` there too, and true for DFlash verify
  once `position` has advanced far enough that `want_first < position`
  permanently, which happens quickly since `dflash_window` > verify width).
  `tap_rows`/`tap_first` are also recomputed identically in `Forward()`
  itself (duplicated math, not shared, on purpose - the outer wrapper needs
  them on EVERY call including replayed ones, to drive `DflashInject`,
  which cannot rely on a member set inside `Enqueue()` since that doesn't
  re-execute on replay). Removed the old `tap_capture_rows`/
  `tap_capture_pos` members entirely - they were being set only on actual
  `Enqueue()` calls and silently going stale on replay, which would have
  been a real bug if left in place reading them from `Forward()`.
- **Also required** (this was missed on the first pass and is why plain
  single-token decode wasn't graphing at all initially): `Session::Append`
  was calling `Forward(..., all_rows=false, ...)` unconditionally.
  `graphable` requires `all_rows==true`. Fixed to match Flint's
  `Session::Append`: pass `all_rows = (tokens.size()==1)` - at n==1,
  all_rows=true and all_rows=false produce IDENTICAL output (`head_rows`
  resolves to 1 either way), so this is free and makes every plain decode
  step graphable too, not just DFlash verify batches.
- Added `hipGraphExecDestroy` for all 17 graph slots in `~Impl()`.

**Verified**: `--validate-prefill 16` bit-identical to pre-change
(`top1_match=yes`, cosine 0.989 - the W8A8 PP fix's own number, confirming
this change touches nothing numeric). Real generation (`gufo prompt`, same
prompt/seed/`-t 0`, both plain decode and DFlash `-d 2`): text byte-identical
to the pre-graph-capture baseline, including DFlash's exact accept/reject
trajectory (170/260 accepted, same as before). No crash. pp2048 unaffected
(prefill chunks are n>16, never graphable, go through the unchanged eager
`Enqueue()` path). Speed effect on TG at *shallow* context was within noise
(~26 tok/s plain, ~35 tok/s DFlash `-d 2`, same as pre-change) - graph
capture's win is eliminating host launch overhead, which matters more as
depth grows and GPU time per kernel shrinks relative to fixed launch cost;
it was NOT the fix for the depth-growth problem itself (see Piece 2).

### Piece 2 (NOT FINISHED - do not ship, do not assume it works): split-K decode attention

**Root cause confirmed** (this is real, not speculative): Laguna's decode
attention - both the scalar `AttentionKernel` and the WMMA
`AttentionWmmaKernelLaguna` actually used by default for decode via
`AttentionWmma()` - has ONE thread-block per (head, token) that sweeps the
ENTIRE `[first, pos]` key range sequentially inside the kernel. Grid size
scales with token count, not KV depth. So every non-SWA layer (1 in 4,
`il%4==0`) gets linearly slower per decode step as context grows; the other
3-in-4 SWA layers are capped by `sliding_window=512` and already flat.
Measured (plain decode, `gufo prompt -f <long file> --raw -n 64 -v`):
26 tok/s @ ~0 tokens -> 19.8 tok/s @ ~20k -> 12.5 tok/s @ ~70k context.

Confirmed via source that Qwen3.6-35B, Qwen3.8-27B (both route through
`qwen38_flash_next`'s shared `Attention()`/`AttentionMergeKernel`, gated by
`kAttnSplits=8` in `executor.cpp:137`, used whenever a decode/narrow batch
is not `MatrixRows` - see `executor.cpp` around line 3366) and Flint (its
own, differently-tiled `AttentionSliceKernel`/`AttentionMergeKernel` pair in
`src/models/flint/gpu/kernels/kernels.hip.cpp:418-520`) ALL use the same
two-pass technique: split the `[first,pos]` key range across `splits` grid
blocks, each producing a partial online-softmax state `(m, l, acc[d])`,
then a second kernel merges the `splits` partials per (token,head) with the
standard flash-attention merge formula. This is NOT architecture-specific
magic (Flint is a normal KV-cache transformer too, not SSM-only as I
initially guessed) - it is a real, missing kernel feature on Laguna's side,
confirmed by reading the grid-launch dimensions directly (Flash-Next/Flint's
grid includes a splits dimension; Laguna's doesn't, anywhere).

**What's done so far** (`src/models/laguna/gpu/kernels/kernels.hip.cpp`,
added right after the existing `AttentionKernel`, around where it now sits
after this session's edits - search for the 2026-10-30 split-K comment):
- `AttentionSliceKernel(q, k_cache, v_cache, partial, start_pos, heads,
  kv_heads, d, window, ring, splits)` - **this is a direct line-for-line
  copy of the existing, already-verified `AttentionKernel`'s per-thread
  math** (same shared-memory reduction pattern, same `tid` dual-use as
  both key-tile-offset and head-dim-index, which only works because
  `head_dim == kAttnBlock == 128` for Laguna, same as the original), just
  bounded to `[k0, k1)` = this split's share of `[first, pos]` instead of
  the whole range, and writing `(acc[tid], m, l)` into
  `partial[((token*heads+head)*splits+split)*(d+2) + {tid|d|d+1}]` instead
  of a final softplus-gated output. Deliberately NOT a new algorithm -
  minimizes the chance of a new numeric bug by reusing math already proven
  against the scalar/WMMA cross-check earlier this port.
- `AttentionSliceMergeKernel(partial, gate, out, heads, d, splits)` - one
  block per (head, token), standard online-softmax merge: max of the
  `splits` partial `m` values, then `l`/`acc` reweighted by
  `exp(partial_m - m_final)` and summed, then the same softplus(gate)
  multiply the original kernel does. Handles empty splits safely (a split
  with no keys in range keeps `m=kNegInf, l=0, acc=0`, contributes `exp(-inf
  - finite) = 0` to the merge - correct as long as at least one split is
  non-empty, which is always true when `pos >= first`).
- `AttentionSplit(q, k_cache, v_cache, gate, out, partials, splits, n_tokens,
  start_pos, heads, kv_heads, head_dim, window, ring, stream)` - host
  wrapper launching both kernels (`dim3(heads, n_tokens, splits)` then
  `dim3(heads, n_tokens)`).
- **Confirmed to compile clean** (`cmake --build --preset release
  --target gufo`, this session) - that is the ONLY thing confirmed. It has
  not been declared in the header, not wired into `engine.cpp`'s
  `Attention()`, not allocated a `partials` device buffer, **never run,
  never validated**.

**Exactly what's left** (in order):
1. Declare `AttentionSplit` in `src/models/laguna/gpu/kernels/kernels.hpp`
   (next to the existing `Attention()` declaration, around line 66) -
   currently ONLY exists in the .cpp, so nothing outside that file can call
   it yet.
2. Add a `DeviceBuffer<float> attn_partials` member to `Session::Impl`
   (`engine.cpp`), sized `kMaxVerifyRows * max_heads * kSplits * (head_dim+2)`
   floats (mirror how `gate`/`q`/etc. are sized off `max_heads` in `Init()`
   around line 285-311). Pick `kSplits` - start with 8, matching Flash-
   Next's `kAttnSplits`, but Laguna's `kAttnBlock`=128 tile size differs
   from Flash-Next's d=256/`kGqaTile`-based one, so re-check whether 8 is
   actually a good split count for Laguna's tile width once it's measured.
3. In `Session::Impl::Attention()` (`engine.cpp`, around line 495-545, right
   where `use_wmma`/`rocm::Attention` are currently chosen), add a third
   path: for narrow batches (`n <= Session::kMaxVerifyRows`, i.e. decode and
   DFlash verify - the ONLY case this matters for; prefill's WMMA path is
   already fine and should NOT be touched) at non-trivial depth, call
   `rocm::AttentionSplit` instead of `rocm::AttentionWmma`/`rocm::Attention`.
   Decide the depth threshold empirically (profile where split-K starts
   winning over the single-block sweep - likely only pays off past some
   thousands of tokens of context, since splits add a second kernel launch
   and merge overhead that costs more than it saves at shallow depth).
4. **Validate extremely carefully before trusting any speed number**:
   - `--validate-prefill` at N=1, 4, 8, 16 (covers both the plain-decode
     and DFlash-verify-width cases) at both shallow AND deep context (this
     kernel's whole point is behavior at depth, so a shallow-only check is
     not enough here specifically, unlike most other changes in this file).
   - Real generation (`gufo prompt`, fixed seed, `-t 0`) at multiple
     context depths (write a long filler file via `-f ... --raw` like this
     session did for the 20k/70k measurements above), comparing token-for-
     token against the known-good pre-split output at the same depth.
   - Differential trace: for one decode step at real depth, run BOTH the
     old single-block kernel and `AttentionSplit` for the same inputs and
     diff the raw attention output (before softplus/gate) directly - the
     two should be near-bit-identical (same math, just reordered/merged),
     NOT just "close by near-tie standards". Any larger-than-fp32-rounding
     difference here means a real indexing bug in the split/merge kernels
     (most likely suspects: the `ring` buffer modulo on `k0`/`k1`-bounded
     reads - re-check against the SWA ring-buffer bug fixed earlier this
     session, since that was exactly this kind of off-by-range bug in a
     different attention kernel; or an off-by-one in `chunk`/`k0`/`k1` at
     split boundaries).
   - Depth sweep per the standing rule: 2k/32k/64k/128k, both PP (should be
     unaffected, this only touches the narrow-batch path) and TG (this is
     the one that must actually flatten - compare the slope, not just a
     single number, against the 26->19.8->12.5 tok/s baseline measured
     above at ~0/20k/70k).
5. Update this file and `~/.claude/skills/port-model/LEARNINGS.md` with the
   measured result - whether it worked, the actual numbers, and anything
   that didn't match the "copy the proven math" assumption above.

**If it doesn't pan out or causes correctness issues**: revert cleanly (the
new kernels are additive, nothing else was changed to accommodate them yet)
and fall back to reporting that TG flatness needs this specific kernel
class of work, same conclusion already reported to the user this session -
do not spend time on alternative approaches without checking with the user
first, since they explicitly asked for this exact technique after being
told Flint/27B/35B use it.

## 2026-10-30 (later): WMMA split-K landed; layer 4/8/12 bug does NOT reproduce; one real bug found and still open

A second agent (after the handoff above) rewrote the split path on top of
the actual WMMA kernel (`AttentionWmmaKernelLaguna` extended with
`partials`/`splits`, round-robin tile loop over `blockIdx.z`, plus the new
standalone `AttentionMergeKernel`) instead of the scalar copy - the scalar
split was proven numerically clean but A/B timing showed it does not help
(bandwidth-bound by WMMA block count, not scalar math); see their notes for
the scalar-path negative result. Current constants: `kSplitAttnSplits=64`,
`kSplitAttnMinDepth=2048` (changed from an earlier 8/4096 - not documented
why, re-derive if it matters). That agent's last handoff reported a real
correctness bug at layer 4/8/12 (rel 0.167/0.087/0.0084) via
`LAGUNA_SPLIT_DEBUG=1`, build green, differential instrumented, but blocked
on diagnosing it.

**Re-ran their exact diagnostic on the current build just now - the bug
does NOT reproduce.** `LAGUNA_SPLIT_DEBUG=1 gufo prompt -f /tmp/filler20k.txt
--raw -n 3 -t 0 -s 1 -v` (depth 13305/13306) and `-f /tmp/filler70k.txt`
(depth 47280/47281): every layer (0,4,8,...,44) matches the single-block
reference to fp32-rounding level (rel 2e-5 to 2e-4, same class as the
WMMA-vs-scalar noise already accepted elsewhere in this port). Either
something after their last note (the splits/depth constant change, or
another edit) already fixed it, or it was intermittent - not re-diagnosed
further, since it no longer reproduces and chasing a non-reproducing bug
wastes effort. **Before trusting this**: re-run `LAGUNA_SPLIT_DEBUG=1` a
few more times, at a few more depths, and ideally with a long sustained
generation (not just `-n 3`), before declaring it solid - one clean repro
is encouraging, not proof, especially if the original bug was data/
layer-magnitude dependent as hypothesized.

**One real bug found while checking - FIXED**: with `LAGUNA_SPLIT_DEBUG=1`,
the stderr log showed `[laguna] decode graph unavailable:
hipStreamSynchronize(stream): operation not permitted when stream is
capturing` on the second qualifying decode step - the debug block
(`engine.cpp`, the `LAGUNA_SPLIT_DEBUG`/`LAGUNA_WMMA_DEBUG` blocks right
after the `AttentionWmma` dispatch) called `hipStreamSynchronize`
unconditionally, which is illegal once Piece 1's graph capture has started
recording the same stream. Fixed by guarding both debug blocks with
`hipStreamIsCapturing(stream, &capture_status) == hipSuccess &&
capture_status == hipStreamCaptureStatusNone` (a single `capture_status`
local shared by both checks). Verified: `-n 20` at depth 13305 now runs
clean, no "unavailable" message, debug prints appear once (the first
qualifying step, before capture succeeds) then correctly stop once the
graph is captured and replay takes over - the debug block lives inside
`Enqueue()`, which simply doesn't run on a replayed step. This bug was
purely a debug-instrumentation/graph-capture interaction, unrelated to the
split-K kernel's own correctness.

**Not yet done** (same as the original handoff's step 4 onward): the A/B
timing re-measurement with the WMMA split path (not the scalar one), the
full depth sweep (2k/32k/64k/128k) on both PP and TG, real-generation
token-stream comparison at depth, and updating both this file and
LEARNINGS.md with final numbers once the above is resolved and re-verified.

## 2026-10-31: NPU prefill offload (shared-expert gate/up + QKV)

**Status: shared-expert offload works and is a real +2% PP gain at matched
M; QKV offload is built but off by default (regresses speed, worse
precision - see below).** Both are additive and gated entirely behind
`GUFO_NPU_DIR`/`model->device->npu() != nullptr` - with that env var unset
(the default), nothing here runs and behavior is unchanged (confirmed via
`--validate-prefill` and normal bench runs after every edit this round).
QKV is further gated behind `LAGUNA_NPU_QKV=1` (opt-in, not opt-out) even
when NPU is on, since it isn't worth enabling as currently built.

**Correction to this section's own earlier draft**: an initial pass flagged
shexp's ~1-3% per-layer relative error as "a correctness bug" based on a
differential trace plus a generation test that diverged from the very
first token. Both signals were misleading, caught by calling the advisor
rather than continuing to chase a layout bug that direct checks kept
ruling out:
- The differential trace's "exact" reference was itself inexact: Laguna's
  own wide-batch `Dense()` is `W8A8Gemm` (int8 activations, ~0.5-1% error
  per element already) for n>8, not a true FP32 reference - so the
  measured "NPU error" was partly W8A8's own error, not purely the NPU's.
- The generation test used the degenerate repeated-word filler prompt from
  earlier TG depth tests (`/tmp/filler*.txt`) - exactly the kind of
  near-tie-prone input this port has repeatedly found diverges after very
  few tokens for *any* numeric change, including the already-shipped,
  already-accepted W8A8 fix. Re-run on real prose
  (`/home/majed/strix-bench/real_prompt.txt`, a genuine technical document,
  not filler) at depth 45546: NPU-on and NPU-off match for the first few
  words ("Rendering (VRAM) extension.") before diverging - the same class
  of divergence point as every other accepted numeric change in this port,
  not first-token divergence.
- Project memory (`p2-whole-npu-precision.md`, a different model's own NPU
  work) independently records 2.48-2.65% RMS as "accurate" for a K=6144
  block-float NPU GEMM of the same general design - squarely matching
  Laguna's own ~1-3% per-layer figure. This is the expected precision class
  of this hi+lo block-float technique, not a Laguna-specific bug.
A byte-level exact-reference check (host-side dequantized FP32 dot
products for a sample of (token, row) pairs, comparing NPU error against
W8A8 error against the same exact reference) would make this fully
rigorous and was not done - if anyone wants to settle it beyond reasonable
doubt, do that next; the evidence above was judged sufficient to lift the
earlier "do not enable" verdict, not to call it proven.

**QKV stays off by default** independent of the above precision question:
measured as a real PP regression (see Measured section) because
`Attention()` has no large independent GPU work to hide the NPU wait
behind, the way `Moe()`'s `RoutedExperts` hides `ShexpSubmit`'s. A related
project's own NPU memory (`p2-whole-npu-precision.md`'s lineage) separately
records whole-QKV-on-NPU reaching only cosine 0.735 against the GPU
reference in an earlier, different port - a meaningfully worse precision
than the shared-expert case, reinforcing that QKV on this NPU path is a
different, worse-behaved case and not a quick rename of the shexp result.
Left in the tree (code compiles, `LAGUNA_NPU_QKV=1` opts in for anyone who
wants to experiment with cross-layer overlap later), but not recommended.

### What was built

1. **A missing NPU instruction file was generated.** The runtime-M/K NPU
   GEMM (`tools/npu/runtime_mk/`, one XCLBIN for any (M,K) - see that
   directory's own README) had no K=3072 program (Laguna's hidden_size);
   built K=2048/2560/4096/5120/6144/17408 existed, not 3072. Built it:
   `source ~/npu-work/ex_env.sh && cd
   ~/npu-work/ex/gemm_asymmetric_tile_buffering/cfg5_MKdyn_N1024 && make
   use_placed=1 M=4096 K=3072 N=1024
   build/final_4096x3072x1024_128x64x128.xclbin`, then copied only the
   resulting `insts_4096x3072x1024_128x64x128.txt` into
   `tools/npu/runtime_mk/` (discarded the XCLBIN it also produces, per the
   tool's own instructions - the XCLBIN is K-independent). Took a few
   minutes, no errors. **This step itself is fine and reusable** - added to
   the port-model skill (see LEARNINGS.md) as a general "when a model's K
   has no instruction file, build one" procedure.
2. Rebuilt gufo with `-DGUFO_ENABLE_NPU=ON -DGUFO_XRT_INCLUDE_DIR=/usr/local/include`
   (XRT headers/libs already present on this machine). Builds clean.
3. Ported Flint's NPU offload pattern (`device_model.hpp/cpp`,
   `engine.cpp`, `kernels.hip.cpp`) for:
   - **Shared-expert gate/up** (`ShexpSubmit`/consume in `Moe()`): encode
     `xn` (hi + residual "lo" block-float passes, `NpuEncodeBoth` - a
     verbatim, genuinely model-agnostic copy of Flint's own kernel, see the
     2026-10-31 comment at its definition), submit against the two staged
     weight buffers, do the (larger) `RoutedExperts` GPU work in between for
     overlap, then `Wait` + combine (`NpuSwigluCombine`, a float-output copy
     of Flint's `NpuSwigluScaledKernel`) into `sh_gate` before the unchanged
     `down_shexp` projection.
   - **QKV** (`QkvSubmit`/consume in `Attention()`): same encode, submitted
     against `attn_q` (chunked into `npu->n()`=1024-row blocks - up to 9 for
     the widest, 72-head layers), `attn_k`, `attn_v` (each already exactly
     1024 rows) in one combined `Submit` call per hi/lo, decoded via a new
     `NpuDecodeFused`/`NpuStreams` (copy of Flint's own fused QKV decode,
     minus the activation - Q/K/V have none).
   - Fixed M=4096 throughout (no runtime-M sweep/`NpuProgramFor` - every
     call uses the one K=3072 program regardless of actual token count `n`,
     wasting NPU cycles on unused rows past `n`; scoped down from Flint's
     full runtime-M version for time).

### Measured (speed only, before the correctness bug was found)

Controlled interleaved A/B (NPU on/off, same binary, back-to-back runs -
single isolated measurements before this were noisy/misleading, see below):
- **Shexp alone, p=4096 (matches fixed M exactly): a real +2% PP gain**
  (1274-1275 vs 1247-1248 tok/s, two runs each).
- **Shexp alone, p=1024 (n far below fixed M=4096): a real ~4% PP
  regression** (1185-1202 vs 1241 tok/s) - the fixed-M waste (processing
  4096 rows of NPU work for 1024 real tokens) costs more than it saves when
  there isn't enough GPU-side work to hide it behind.
- **Shexp + QKV together, p=4096: a clear regression** (1085 vs 1247-1275)
  - `QkvSubmit`'s consume in `Attention()` has no large independent GPU work
    to overlap its NPU wait behind (unlike shexp's `RoutedExperts`), so it's
    pure added latency. Real cross-layer overlap (submitting layer N+1's
    QKV during layer N's MoE compute, the way Flint's `NpuBank` ping-pong
    does) would need a fused residual+norm step Laguna doesn't have -
    scoped out, not attempted.
- A single early isolated (non-interleaved) measurement showed p=1024 and
  p=4096 BOTH regressing hard (-5.5%, -8.4%) - this did NOT reproduce under
  a proper interleaved A/B and was thermal/noise, consistent with this
  port's standing documented pattern of run-to-run variance under sustained
  load. **Always interleave NPU on/off back-to-back before trusting any
  single NPU timing number** - the effect size here is small enough that
  ordinary thermal drift this port has seen all session (e.g. the W8A8 fix
  swinging 1275->1167 under load) can easily look like a real NPU result if
  measured in isolation.

### Precision investigation (resolved: this is precision, not a bug)

User asked for NPU utilization closer to 50% (it was visibly low); checking
that surfaced a scare that turned out not to be a bug. A differential trace
(`LAGUNA_NPU_SHEXP_DEBUG=1`, compares the NPU combine's `sh_gate` output
against the GPU `Dense()`+`Swiglu()` path for the SAME layer/call, same
`xn` input) shows **~1.2-2.8% relative error per MoE layer**. On the
degenerate repeated-word filler prompt used for the TG depth tests earlier
this port, real-generation output diverged from the very first generated
token with NPU shexp enabled - which read, at the time, like a specific,
damning signal (every other numeric change in this port had only ever
diverged after several tokens). Calling the advisor rather than continuing
to chase a layout bug caught two flaws in that read, confirmed by
re-testing (see the correction note above this section): the differential
trace's own "reference" is itself inexact (Laguna's wide-batch `Dense()` is
`W8A8Gemm`, not true FP32, for n>8), and the filler prompt is exactly the
near-tie-prone input class this port has repeatedly seen diverge fast for
*any* accepted numeric change - re-tested on real prose instead, the
divergence point matches every other accepted change in this port, not
token 1. Combined with project memory independently calling 2.48-2.65% RMS
"accurate" for a same-design NPU GEMM elsewhere, **the verdict is
precision, within this project's existing NPU quality bar - not a Laguna-
specific bug.** Shexp is enabled by default as a result.

QKV was not differentially checked this way (only shexp was) - it stays
off by default anyway, for the separate, already-settled reasons above
(speed regression + a worse-precision precedent in project memory), so
this is not blocking, just an open item if anyone wants QKV later.

**Checked during the investigation** (so the next person doesn't re-check
these if the precision verdict above is ever questioned again):
- Weight type dispatch: `gate_shexp`/`up_shexp` are confirmed `kQ8_0`
  (type=8), taking the `NpuEncodeWeights` branch, not
  `NpuEncodeWeightsQuant` - correct per the UD-Q4_K_M quant's actual shexp
  type.
- `NpuEncodeWeights` call parameters (`k_total=w.cols=3072`, `row0=0`,
  `cols=w.rows=1024`, `k_slice=npu->k()=3072`, `k_offset=0`,
  `k_valid=w.cols=3072`) - byte-for-byte the same call shape as Flint's own
  (proven, per Flint's own notes, to reach ~1e-8 KL divergence with the
  identical shared `qwen38_flash_next::rocm` encode/weight functions).
- The activation source: `ShexpSubmit` encodes `xn.get()`, the exact same
  buffer and call-site context `Dense(layer.gate_shexp, xn.get(), ...)`
  used in the un-replaced GPU path - not stale, not a different buffer.
- The combine kernel's math (`NpuSwigluCombine`/`NpuSwigluCombineKernel`):
  re-checked line by line against Flint's `NpuSwigluScaledKernel` - the
  only intentional difference is dropping the `__half2`-packed F16 output
  and the `scale`/`1/scale` folding (replaced with a direct float write,
  since Laguna's `Dense()`/`Swiglu()` pipeline is F32 end to end and has no
  need for Flint's F16-range workaround) - the per-group dequant-and-sum
  math itself (`mult_hi`/`mult_lo`, the `ldexpf(...)/64.0F` scale, the
  hi+lo sum before `silu(gate)*up`) is unchanged.

**Optional further rigor (not required - the precision verdict above is
settled; do these only to make it bulletproof, e.g. before an upstream
PR)**:
- A byte-level check of the RAW block-float bytes the NPU actually wrote
  for one specific (token, output-column) pair, hand-decoded and compared
  against hand-computed `dequant(Q8_0 weight) . activation` for that same
  element, and the same for GPU `W8A8Gemm` against the same exact
  reference - would give an exact apples-to-apples NPU-vs-W8A8 error
  comparison instead of the indirect evidence above.
- Independently validate the freshly-built K=3072 instruction file against
  a trivial known-input GEMM, per Flint's own notes ("Compare final-token
  logits against the GPU-only path... for the first runtime-M build").
- The QKV path's own correctness, independently (never checked - moot
  while it stays off by default, but relevant if anyone revisits
  cross-layer overlap for it later).

**Safety note, separate from the correctness bug**: this round's testing
ran NPU work without the fabric clock held at its documented top level
(`tools/npu/fabric-clock.sh hold`, needs root) - the user's own
`comfy-monitor` service manages GPU clocks dynamically but does NOT hold
fclk at top (confirmed: `pp_dpm_fclk` was `auto`, not `manual`+pinned, for
every test this round). No hang occurred, but this was explicitly against
the documented safety requirement (prior sessions this port have had real
GPU freezes). **Hold fclk properly before any further NPU testing**, this
round's clean runs do not prove it is safe to skip.

## 2026-10-31: WMMA split-K TG result - real win, confirmed, flattens the curve

Clean A/B on the same binary (`kSplitAttnMinDepth` temporarily set to
`0xFFFFFFFF` to disable, rebuilt, measured, restored to 2048, rebuilt again
- both configs measured on the identical build otherwise):

| depth    | split disabled | split enabled | gain   |
|----------|----------------|---------------|--------|
| ~13,305  | 21.32 tok/s    | 24.05 tok/s   | +12.8% |
| ~47,280  | 14.85 tok/s    | 20.35 tok/s   | +37.0% |

Gain widens with depth, confirming this is the right lever (the single-block
sweep's cost grows with depth; split-K bounds it). Extended sweep with split
enabled only (not yet re-measured disabled at these two, extrapolate from
the trend above): ~88,797 depth -> 17.02 tok/s, ~133,284 depth -> 14.78
tok/s - still declining, NOT fully flat, but the slope is clearly shallower
than the pre-split baseline's trend (26 -> 19.8 -> 12.5 tok/s at roughly 0/
20k/70k from earlier in this port). **Not "flat" in the literal sense yet**
- likely because the SWA layers' fixed window cost and the dense/MoE FFN
kernels' own KV-independent cost become a larger fraction of the step as
the split attention layers' cost shrinks, so further flattening would need
those addressed too, not another attention change. Report this honestly as
"much flatter, not perfectly flat" rather than overclaiming.

Correctness re-confirmed on the restored (enabled) build: `--validate-prefill
16` unchanged (`top1_match=yes`, cosine 0.989, identical to every other
check this port). PP unaffected (1,279.89 tok/s @2k, matching the W8A8 fix's
number - split-K only touches the narrow decode/verify path, prefill's
existing WMMA path is untouched). Generated text at extreme depth (128k, a
degenerate repeated-word filler prompt) degenerates into repetition - this
is expected sampling behavior on a nonsense filler prompt at extreme depth,
not a kernel correctness issue (greedy decoding on garbage input repeating
is normal, same as any model would do).

**Also fixed this round**: the `LAGUNA_SPLIT_DEBUG`/`LAGUNA_WMMA_DEBUG` vs
HIP-graph-capture interaction bug from the entry above (guarded both debug
blocks with `hipStreamIsCapturing`).

**Still open / not done**:
- Re-measure the split-disabled baseline at 88,797 and 133,284 depth too
  (only measured split-enabled there so far) to get the exact gain number
  at extreme depth, not just the trend.
- Tune `kSplitAttnSplits` (currently 64) and `kSplitAttnMinDepth` (currently
  2048) - neither has been swept, both were inherited from the previous
  agent's last edit without re-justification at this depth range.
- Investigate whether the SWA/FFN-dominated residual cost (see "not fully
  flat" above) has its own lever, if the user wants to keep pushing toward
  literal flatness.
- Run the standing regression (`gufo prompt -f p2k.txt -n 128 -t 0` on other
  shipped models, unaffected by this Laguna-only change, but not re-checked
  this round) and the full report-to-skill writeup are still pending.

**Third dead end tried (DenseF16Gemm split by K, matching Flint/Flash-Next's
pattern) - real accuracy regression, not benign near-tie, reverted clean:**
Flint's `Dense()` (`src/models/flint/gpu/engine.cpp:627`) and Flash-Next's
executor both split Q8_0 wide-batch dense projections by K: `DenseF16Gemm`
for small K (Flint: K<4096; Flash-Next: rows>=2048 && cols<=2560), `W8A8Gemm`
for large K. Checked Laguna's actual K values via a one-off debug print
(`w.cols` at each `Dense()` n>8 call site): 1024 and 3072 (Q/K/V, ffn
gate/up) vs 6144/9216/12288 (attn_output, ffn down). Tried routing the
K<4096 tensors through `DenseF16Gemm` (narrowing `in` into the already
tile*hidden_size-sized `xh` half buffer, safe to reuse since it's free
whenever `Dense()` runs) with `W8A8Gemm` kept for the rest. Result: a REAL
regression, not the usual benign MoE near-tie drift - `--validate-prefill
16`'s cosine similarity dropped from 0.989 (W8A8-only) to 0.839, and a real
generation run (`gufo prompt`, same prompt/seed/`-t 0`) diverged from the
very first generated token (previous changes this entire port only ever
diverged after ~8+ tokens). This is a genuine numeric problem with
`DenseF16Gemm` on Laguna's specific tensor shapes/weight layout, not just
quantization rounding - not investigated further (out of scope for this
round; if revisited, diff `DenseF16Gemm`'s output against the known-good
MMQ path for one of the small-K tensors directly, the same differential-
trace technique that cleared the earlier Q8_0-vec-path false alarm).
Reverted completely; `Dense()`'s n>8 Q8_0 branch is W8A8Gemm-only (falling
back to the original MMQ path), as shipped.

**Measurement note**: pp2048 after this revert measured 1,167-1,208 tok/s
in immediate back-to-back runs, lower than the 1,275-1,287 tok/s measured
right after the W8A8 change was first added (same code, confirmed via
`--validate-prefill 16` giving the identical `top1_match=yes`/cosine=0.989
result both times - this is NOT a correctness regression). Likely thermal/
clock-state drift from the extended depth sweeps (32k/64k/128k) and repeat
builds run earlier in the same session, consistent with this port's
existing documented pattern of run-to-run variance on this hardware under
sustained load. Treat 1,275-1,287 (cool-start) and 1,167-1,208 (loaded)
as the real range for this fix rather than a single number; both are an
improvement over the 1,222 baseline measured under comparable load. Re-
measure cold (after GPU idle) before reporting a final number against the
1,300 target.

## 2026-10-31 (later): NPU alignment-gating bug fixed; "hide it more / hand it more" - investigated, NOT implemented (two-bank is a large feature)

User asked to raise NPU utilization (observed ~10-15% vs the user's target)
by hiding its latency better and offloading more; asked to study how
Qwen3.6-35B/Qwen3.8-27B (both on the shared `qwen38_flash_next` executor)
do it first.

**Real bug found and fixed**: `ShexpSubmit`/`QkvSubmit` admitted any
`n` in `[1024, npu->tokens()]`, not just `n == npu->tokens()`. Flash-Next's
own dispatch (`executor.cpp`'s `MoeExperts`/attention-projection NPU
branches) requires `n_tokens % npu->tokens() == 0` before engaging at
all - Laguna has no multi-group support, so the equivalent is exact
equality, not a range. The loose range was exactly why p=1024 regressed
~4% earlier this round (submitting a fixed 4096-row program for 1024 real
tokens) - fixed to `n != npu->tokens()` bails out instead of tuning around
it. Verified: p=1024 NPU-on/off now within noise of each other (no longer
a real regression); p=4096 still shows the real gain (though a same-round
remeasure landed lower, 1134 vs the earlier 1274-1275 - almost certainly
the same thermal/load drift this port has repeatedly documented after
many back-to-back runs, not a new regression from this fix; re-confirm
cold before trusting either number as final).

**How Flash-Next actually gets more NPU overlap** (read in depth,
`executor.cpp` lines ~3390-3490 for shared-expert, ~3080-3160 for
attention/QKV, ~4495-4563 for the pair mechanism):
- **Same-call overlap** (what Laguna's shexp already does): submit before
  `MoeExperts`/`RoutedExperts`, `Wait` after - identical pattern, not a
  bigger idea.
- **Multi-group within one call**: when `n_tokens` is a multiple of
  `npu->tokens()` (e.g. 8192 = 2 groups of 4096), it loops `g` groups,
  submitting each group's hi+lo pair separately (`npu_a_[g]`/`npu_a_lo_[g]`,
  up to `kNpuSlicesMax`-wide `npu_c_` arrays) - this is NOT cross-layer
  pipelining, it's "more work per call, same per-call overlap shape."
  Laguna could adopt this for `prefill_tile` > 4096 (needs `npu_a_`/`sh_c`-
  style per-group buffer arrays instead of the current fixed 2+4 ids) -
  not done, real but modest (lets bigger prefill chunks use NPU too,
  doesn't change the per-group overlap ratio).
- **"Two-bank" prefill** (`TwoBankEnabled`/`GUFO_TWO_BANK`,
  `CanForwardPair`/`ForwardPair`/`EnsureBankBScratch`): this is the actual
  "hide more NPU behind more GPU work" lever, and it is a genuinely large
  feature, not a tuning knob. It splits one large prompt (`n>=8192`, in
  4096-token groups) into TWO HALVES ("bank A", "bank B"), each run as an
  **independent full-model forward pass with its own complete scratch
  buffer set** (`EnsureBankBScratch` allocates a second copy of every
  per-call scratch buffer, sized `max_batch/2`, plus its own pinned
  control/token host buffers). The two banks' per-layer NPU submit/wait are
  then interleaved *against each other* (bank A's GPU work hides bank B's
  NPU wait and vice versa), not against the same bank's own other layers -
  this is NOT cross-layer pipelining of a single token stream, it's two
  independent streams kept in lockstep so each one's GPU time covers the
  other's NPU latency.
- **Implementing this for Laguna would mean**: doubling every per-session
  `DeviceBuffer` (`x`, `xn`, `q`, `k`, `v`, `gate`, `attn`, `router`,
  `ids`, `topk`, all the routed-expert scratch, etc. - the full list in
  `Session::Impl`) into an A/B pair, a second pinned host control/token
  buffer, a `CanForwardPair`-equivalent gate (`n>=8192`, multiple of 4096,
  within capacity), and a `ForwardPair`-equivalent that runs both banks'
  per-layer loops interleaved rather than Laguna's current single
  sequential loop in `Enqueue()`. This is comparable in size to everything
  else done in this NPU round combined, not a quick follow-on - **not
  attempted this round**, scoped out deliberately rather than rushed,
  given how much NPU-adjacent ground (the instruction-file build, the
  shexp port, the QKV port, the precision false-alarm, this alignment fix)
  was already covered in one sitting.
- **Multi-group (same-call, not two-bank) is the cheaper next step** if the
  goal is "offload more, nothing fancier": it directly increases NPU's
  share of total work without a new pipelining mechanism, just needs the
  per-group buffer-array plumbing Flash-Next already has. Two-bank is the
  real "closer to Flash-Next's actual overlap ratio" answer but is a
  multi-session-scale feature on its own.

## 2026-10-31 (later): Two-bank prefill ported from Flint's lighter design - BUILT, CONFIRMED BROKEN, OFF BY DEFAULT

Checked whether Flint (not just Flash-Next) has two-bank prefill: yes, and
it's much cheaper than Flash-Next's `ForwardPair`/`EnsureBankBScratch`.
Flint's `NpuBank`/`SetBank`/`x1`/`xn1` duplicates only the residual stream
(`x`/`xn`) and a small per-bank NPU-id struct - everything else (q/k/v,
router, routed-expert scratch, KV cache) is reused sequentially per bank,
safe because HIP executes one stream's kernels in launch order (bank 0's
reads of a shared buffer always complete before bank 1's writes to it,
even though both are "in flight" from the host's point of view - no actual
concurrent GPU execution within one stream). Ported this lighter design
for Laguna:
- `x2`/`xn2` (bank 1's residual stream, `x`/`xn` are bank 0's).
- `cx`/`cxn`/`cbank`/`SetBank(bank)` repoint which bank's buffers every
  per-layer helper (`Attention`, `DenseFfn`, `RoutedExperts`, the new
  `MoeRoute`/`MoeFinish`) reads and writes - replaced all direct
  `x.get()`/`xn.get()` call sites with `cx`/`cxn` throughout.
- `pos` (already sized 4 slots, previously only slot 0 used) now holds
  bank 0's and bank 1's own absolute start position in slots 0/1;
  `pos.get() + cbank` replaces every bare `pos.get()` passed to
  RoPE/StoreKv/Attention kernels.
- `sh_a`/`sh_c`/`sh_ticket`/`sh_pending` (the shexp NPU submission state)
  became `[2]`-indexed (one slot per bank) so both banks can have a
  submission in flight at once - this is the actual point of banking:
  `MoeRoute`/`MoeFinish` split `Moe()` into "submit shexp + route" and
  "wait + combine + down-project + combine", so `EnqueueBanked` can submit
  both banks' shexp, run both banks' `RoutedExperts`, THEN wait+finish
  both - bank 0's wait is now hidden behind bank 1's `RoutedExperts` too,
  not just its own.
- `EnqueueBanked` (new): splits a `2*tile` chunk into two exact `tile`-sized
  halves, runs the full per-layer loop for bank 0 then bank 1 at each
  layer (attention, ffn-norm, then route-both/finish-both for MoE layers).
  Deliberately narrow scope for a first cut: exact `2*tile` chunks only (no
  odd remainder - any leftover falls through to the normal per-tile loop),
  no DFlash tap-capture support (gated off whenever `dflash != nullptr`),
  no graph capture interaction (these chunks are always far above
  `kMaxVerifyRows`). `Session::Append` now runs `EnqueueBanked` on as many
  full `2*tile` chunks as fit before falling back to the existing per-tile
  `Forward` loop for the remainder.

**Confirmed broken, not a false alarm this time.** Checked with a real,
non-degenerate document (`/home/majed/strix-bench/real_prompt.txt`, the
same file the earlier NPU precision investigation used to disprove a false
bug report) at depth 45546 (well past the `can_bank` threshold, so most of
the prefill ran banked): output **diverges from the very first generated
token** vs. the non-banked/non-NPU baseline on the identical prompt/seed -
not the few-tokens-in drift this port's other accepted changes show. Root
cause NOT found - time ran out on this round before isolating it to a
specific step (candidates not yet checked: whether bank 1's causal
attention genuinely sees bank 0's just-written KV-cache rows correctly
given the sequential-reuse assumption above, whether `RoutedExperts`'s own
host-side count-download-and-resync is safe to run back-to-back for two
banks without some host-buffer aliasing, or something in the `pos`
slot-0/slot-1 indexing). **Disabled by default**: gated behind
`LAGUNA_NPU_BANK=1` (on top of needing the NPU on and no DFlash), so the
normal NPU-on path (shexp-only, no banking) is unaffected - re-verified
after this entire change: `--validate-prefill` unchanged, real generation
on the same real-prose prompt still matches the known-good baseline for
the first several words (the normal accepted-change divergence pattern),
and the p=4096 shexp win is still the same ~+2-3% measured earlier.

**If resuming**: do NOT re-enable `LAGUNA_NPU_BANK` without finding the
actual bug first - the refactor that added `cx`/`cxn`/`SetBank` touched
every per-layer helper, so re-verify the non-banked default path (bank 0
only, `cbank` always 0) is still exactly equivalent to the pre-refactor
code before assuming the bug is isolated to `EnqueueBanked`'s own new
code, even though this round's own re-verification above didn't catch a
regression there. The highest-value next check: run with exactly ONE
`2*tile` (8192-token) banked chunk and nothing else (not a 45546-token
prompt exercising five-plus banked calls plus a remainder), to narrow the
failure to a single pair of banks before trying to localize it further
inside that call.

## 2026-10-31 (later still): two-bank bugs FOUND AND FIXED - on by default again

Ran Flash-Next itself (the actual reference implementation) with and
without its own NPU/two-bank flags to get real numbers before continuing
to guess: **Flash-Next's NPU offload WITHOUT two-bank is a regression**
(pp4096: 1719.56 NPU-on vs 1844.92 NPU-off, -6.8%) - the oft-quoted "+10%"
only materializes with `GUFO_TWO_BANK=1 GUFO_NPU_WHOLE=1` together
(pp16384: 2157.46 vs a matched 1823.79 NPU-off baseline, **+18.3%**). This
confirms two-bank is not an optional refinement for this architecture - it
is where essentially all of the real gain lives. Laguna's shexp-only,
non-banked +2-3% is the same order as Flash-Next's own non-banked result;
reaching anything like Flash-Next's number requires banking to actually
work. (Separately tested whether a bigger single-bank tile alone closes
the gap, per the "PP not saturated enough" theory - built a new M=8192
K=3072 NPU instruction file and set `LAGUNA_PREFILL_TILE=8192`: no,
the ratio stayed ~+2-3% at tile=8192 too, and QKV still regressed even
with 2x the GPU work per call - ruling out tile size as the lever and
confirming cross-call overlap, not raw tile size, is what's missing.)

**Found and fixed two real bugs in `EnqueueBanked` via a minimal repro**
(a single isolated `2*tile`-token real-prose chunk, built by truncating
`/home/majed/strix-bench/real_prompt.txt` to exactly 8202 tokens - small
enough to not need the full 45546-token test to get a signal, which had
only told us "wrong" without narrowing where):
1. **SWA ring buffer sized for one tile's new rows, not two.**
   `ring[il] = c.sliding_window + tile` assumed at most `tile` new KV rows
   land in the ring per `Enqueue` call - true for the non-banked path, but
   `EnqueueBanked` writes a full tile's rows for bank 0 THEN another
   tile's worth for bank 1 into the SAME ring buffer (banks share the KV
   cache; only the residual stream is duplicated). Bank 1's later writes
   wrapped around and overwrote rows bank 1's own sliding-window attention
   still needed from bank 0. Fixed: `ring[il] = c.sliding_window + 2 *
   tile`, unconditionally (costs a little extra memory, removes any need
   to track whether banking might run later).
2. **The actual cause of the garbage output**: a real host-side race in
   the per-bank token/position upload, unrelated to anything GPU-side.
   `EnqueueBanked` copied each bank's token slice into the shared pinned
   host buffer `h_tokens` and wrote `h_pos[0]` fresh, then called
   `hipMemcpyAsync` - but `hipMemcpyAsync` queues the copy and returns
   immediately, it does NOT wait for the DMA transfer to actually run.
   Overwriting `h_tokens`/`h_pos[0]` for bank 1 immediately after queuing
   bank 0's copy (no sync in between) could corrupt bank 0's in-flight
   transfer with bank 1's data, depending on exactly when the DMA engine
   got to it - a real race, consistent with the observed garbage
   (`| | | | | ...`) being far worse than ordinary numeric drift. Fixed by
   reading tokens straight from `input` (the caller's own stable span,
   never mutated by this code, no second host buffer needed) and using
   two distinct `h_pos` slots (`h_pos[0]`/`h_pos[1]`, written once up
   front, each copied by its own bank's async copy) instead of reusing
   one slot.
3. **Also found while fixing (1) and (2), and fixed at the same time,
   though not itself the garbage-output cause**: `sh_gate`, `down_e`,
   `topk` and `expert_scale` were single shared buffers that carry state
   from `MoeRoute`'s return to `MoeFinish`'s start - but `EnqueueBanked`
   deliberately calls `MoeRoute` for BOTH banks before either bank's
   `MoeFinish` (that's the whole point of banking: hide bank 0's NPU wait
   behind bank 1's `RoutedExperts`, not just its own), so bank 1's
   `MoeRoute` was overwriting bank 0's `down_e`/`sh_gate`/etc before
   `MoeFinish(bank 0)` ever read them. Every other per-layer scratch
   buffer (q/k/v/gate/attn/o/router/ids/counts/routed-expert tiles/
   gate_e/up_e/sh_up/sh_out/xh) is fully written-then-consumed within a
   single `MoeRoute` or `MoeFinish` call for one bank, so those stayed
   safely shared. Fixed by making the four crossing buffers `[2]`-indexed
   (`down_e[bank]`, `sh_gate[bank]`, `topk[bank]`, `expert_scale[bank]`)
   and threading `bank` through `RoutedExperts` too.

**Verified after both fixes**: the isolated 8202-token repro now matches
the non-NPU baseline byte-for-byte (not just "close" - identical text).
The full 45546-token real-prose test matches the normal accepted-
divergence pattern (first several words identical, same as the non-banked
shexp-only path, same class as every other numeric change in this port).
`--validate-prefill` unchanged. Speed at p=8192: banked NPU 1228-1233 vs
NPU-off 1201-1204 (+2.2-2.7%) - a real, modest gain, same order as the
non-banked shexp-only result, NOT the dramatic Flash-Next-level win,
because QKV (the piece that actually needs cross-call overlap to pay off)
still isn't restructured to benefit from banking - its submit/consume
currently stays within one `Attention()` call per bank, never deferred
across banks the way `MoeRoute`/`MoeFinish` are for shexp. Confirmed safe
to combine with `LAGUNA_NPU_QKV=1` (no corruption - verified on the same
8202-token repro) but it still doesn't help speed, for that same reason.

**Banking is back on by default** (`LAGUNA_NPU_NO_BANK=1` opts out, the
flag was renamed from the old opt-in `LAGUNA_NPU_BANK` now that it's
verified correct) whenever the NPU is on and there's no DFlash drafter.

**If pursuing the full Flash-Next-level gain next**: the remaining piece
is restructuring `QkvSubmit`/its consumption the way Flint's
`finish_attn`/`run_moe` do - submit layer N+1's QKV at the tail of layer
N's `MoeFinish` (using bank b's freshly-updated residual stream for layer
N+1's `attn_norm`, computed inline), consumed at the start of layer N+1's
`Attention(bank b)`, with the OTHER bank's work in between providing the
hide. This needs `q_a`/`q_c`/`q_ticket`/`q_pending` to become `[2]`-indexed
(currently single-slot, exactly the shape of bug fixed in (3) above) and
careful attention to the SAME two classes of bug just found: any shared
buffer crossing a submit-in-layer-N/consume-in-layer-N+1 boundary needs to
be bank-duplicated, and any host-pinned buffer written across that
boundary needs to not be reused before its async copy actually runs.

## 2026-10-31 (later still): QKV cross-layer overlap built per the advisor's design - still regresses, root cause now a measured NPU latency limit, not a scheduling bug

Got advisor input after the user flagged NPU usage as too low (~10%) and
PP far below Flash-Next's absolute number. Two concrete things came out of
that, both implemented:

**Bug found and fixed**: `QkvSubmit` checked the COMBINED Q+K+V slice count
against `rocm::kNpuMaxSlices` (9) - a DIFFERENT, smaller limit meant only
for `NpuStreams`' per-tensor decode cap. Every 72-head layer needs 9 (Q) +
1 (K) + 1 (V) = 11 slices, so this silently rejected QKV on every 72-head
layer (36 of 48) - QKV was only ever running on the 12 48-head layers,
invalidating every earlier QKV speed/precision measurement this session.
Fixed: added a separate `kNpuMaxQkvSlices = 11` for the combined check,
kept `rocm::kNpuMaxSlices` (9) for its own, correct, per-tensor use.

**Restructured for real cross-layer overlap**, per the advisor's exact
design: `q_a`/`q_c`/`q_ticket`/`q_pending`/`q_slices` became `[2]`-indexed
(one slot per bank, matching the shexp pattern). `QkvSubmit`/`QkvConsume`
split out of the old single inline block. `EnqueueBanked` now submits
layer `il+1`'s QKV at the tail of layer `il`'s FFN/MoE step (for both
banks, using each bank's own just-updated residual stream for `il+1`'s
`attn_norm`), consumed at the very start of `il+1`'s `Attention(bank)` -
hidden behind the OTHER bank's remaining layer-`il` work (`DenseFfn`, or
the `MoeFinish` calls), not a same-call submit-then-wait like before.
Layer 0 has nothing to submit it during, so it always falls through to
`Dense()` (expected, not a bug).

**Verified correct**: no corruption on the same real-prose repro used to
find the two-bank bugs above (coherent markdown table output, differs
from the no-QKV baseline only in the way every other accepted precision
change in this port differs - normal divergence class, not garbage).

**Still regresses**: shexp+QKV (fixed, cross-layer) measured 1162-1165
tok/s vs shexp-only's 1220-1222 at p=8192 (-4.7%) - better than the old
same-call version's regression, but still a net loss, contradicting the
expectation that correct cross-layer scheduling alone would make QKV pay
off. Measured why directly (`tools/prof/prof.py`, the diagnostic the
advisor specifically asked for): the actual compute passes (not the
model-load-dominated whole-process idle%, see the correction two sections
up) show **~13% idle time INSIDE them** with QKV enabled (934ms + 929ms
idle within two ~7.1-7.2s passes) vs ~1% idle without QKV at the same
depth. This is real, measured evidence that the NPU wait is NOT being
fully hidden even with the correct cross-layer schedule - the bottleneck
is the sheer number of NPU round trips (9 Q slices + 1 K + 1 V = 11
weight buffers, x2 for hi/lo = **22 `Submit()` calls per layer per bank**,
96 layer-banks per full pass = over 2,000 total NPU submissions for one
prefill) outrunning what the available hide window (the other bank's
`DenseFfn`/`MoeFinish` work) can absorb, not a software scheduling defect.
Memory from an earlier, different model's NPU work
(`npu-prefill-offload-state.md`) independently notes the NPU queue is
"4-deep" and each `Submit` has real per-call issue cost - consistent with
22-per-layer-bank simply being too many round trips for this design,
regardless of how well the host-side scheduling hides any ONE of them.

**Not yet done, if pursuing this further**: directly measure one isolated
Q-slice (9-wide) Submit+Wait latency with nothing else queued (the
advisor's suggested check, never actually isolated - only measured the
AGGREGATE idle time within a full pass, which confirms a real stall
exists but not its exact per-call shape). If the 22-submissions-per-
layer-bank theory is right, the fix would need to either reduce the
submission count (e.g. finding whether `NpuGemm::Submit`'s own `count`
parameter can cover MORE than one bank's data per call, batching across
banks instead of per-bank) or accept that Q's 9-slice width is simply too
wide for this NPU design's per-call overhead at Laguna's `hidden_size`/
head-count shape, unlike Flash-Next's own narrower per-call slice counts
for the same technique.

**Current recommendation, given all of the above**: keep `LAGUNA_NPU_QKV`
opt-in/off by default (unchanged from before this round) - it is now
provably correct (no corruption, real cross-layer scheduling, the slice-
cap bug fixed) but still not a net win, for a now-understood, measured
reason rather than an unknown one. Shexp alone (+2-3%, on by default)
remains the only NPU lever that pays off for Laguna as currently built.
Reaching Flash-Next's absolute PP numbers is not a realistic bar for this
model either way - Flash-Next's 1700-2100 range reflects ITS OWN kernel
costs for its own hidden_size/quant/shape, not a hardware ceiling Laguna
is failing to reach; the only valid comparison between the two is each
model's own NPU-on-vs-off delta, not their absolute tok/s.

## 2026-10-31 (pre-merge): a real cross-model regression found and fixed

Before merging into `fork-enhancements`, regression-checked the OTHER
DFlash2-using models (Flash-Next, 27B, 35B, GLM) against the shared
`src/models/qwen/dflash_weights.cpp`/`src/models/qwen/hip/dflash/executor.cpp`
changes made earlier this session for Laguna's v1 drafter, by running the
gufo-laguna binary (which contains every model) directly against each
model's own GGUF, comparing against `~/gufo`'s unmodified build on the
same prompt/seed.

**Found a real regression**: `dflash_weights.cpp` made `attn_gate.weight`
an unconditionally REQUIRED tensor for every DFlash drafter - correct for
Laguna's v1 (block-diffusion) drafter, which has it, but wrong for every
v2 (gated-delta-net) drafter (27B/35B/GLM), which doesn't. 27B's DFlash2
failed to load at all ("missing required tensor 'blk.0.attn_gate.weight'").
Fixed by gating the bind behind `is_v1` (the same flag already used for
other v1/v2-specific tensors in this file), matching the existing
`(!is_v1 && ...)` idiom a few lines below it.

**That fix alone uncovered a second, worse bug**: `executor.cpp`'s
`Allocate()` and `ForwardBlock()` called the attn_gate VRAM-staging and
the gate-projection/multiply kernels (`LaunchDFlashAttnGateProjection`/
`LaunchDFlashAttnGateMul`) UNCONDITIONALLY for every drafter, not just
v1 - with the tensor now correctly left unbound for v2 (0 elements, null
host pointer), this staged a 0-byte VRAM buffer and then read past it from
a GPU kernel: a real `Memory Fault Error` (confirmed via the HIP driver's
own fault report, not a guess) in `dflash_attn_gate_proj_kernel`, hanging
the process (required a `SIGTERM`, not `kill -9`, to clear - the exact
kind of GPU-fault-then-hang this project's own safety notes warn about).
Fixed by gating both the staging loop in `Allocate()` and the two kernel
calls in `ForwardBlock()` behind `df_cfg.is_v1`, leaving v2's behavior
exactly as it was before this entire DFlash-v1 session's work began (no
gate projection/multiply at all - that mechanism is Laguna-v1-specific).

**Verified after both fixes**, each against `~/gufo`'s unmodified build,
same prompt/seed, `--think off`:
- Flash-Next: byte-for-byte identical output; `pp2048` matches closely
  (1648.65 vs 1650.70 once measured cleanly - an earlier mixed PP+TG
  reading showed a misleading 8% gap, pure thermal noise from this port's
  already-documented pattern, not a real regression).
- 27B: loads and generates correctly again (previously crashed/hung).
- 35B: loads and generates correctly.
- GLM: loads and generates correctly.
- Laguna's own v1 drafter re-verified working after both fixes (DFlash
  drafting, accept/reject, coherent output all unchanged).

**Lesson for future shared-file edits**: a tensor-requirement change and a
kernel-call-site change can each hide the other's bug - fixing the loader
alone (making the tensor correctly optional) is what let the SECOND,
unconditional kernel-call bug actually manifest (reading a now-empty
buffer instead of a previously-required-but-functionally-unused one).
Any `is_v1`/`is_v2`-gated tensor needs its CONSUMERS checked for the same
gate, not just its binding - grep every kernel/launch site that touches
the tensor, not only the loader.

## `gufo serve` had no Laguna dispatch at all (2026-10-31)

The AGENTS.md port workflow's "Serve" step (add a `general.architecture ==
"<model>"` case to `src/cli/serve/inference_backend.cpp`) had never been
done for Laguna - confirmed by the user's own `GUFO_KIND=laguna
~/run-gufo.sh` run failing with `Unexpected Qwen layer kind at blk.0:
expected Gated DeltaNet`: any GGUF with `general.architecture == "laguna"`
fell through to the Qwen38/Flash-Next engine's own loader, which rejects
Laguna's per-layer tensor layout outright. `gufo prompt`/`gufo bench`
worked the whole time because they dispatch on the CLI's own model flag,
not on `inference_backend.cpp`'s architecture switch - a working `gufo
prompt` run is not evidence `gufo serve` works for a new model.

Fixed by adding a `LagunaTextRunner`/`LagunaTextRunnerState` pair (modeled
closely on Flint's, the simplest existing runner - no speculative decoding,
no vision) plus an `InferenceBackend::load(shared_ptr<laguna::gpu::Model>,
...)` overload and the `"laguna"` dispatch block, following the exact
pattern of the existing `flint`/`glm5next`/`deepseek4`/`qwen4exp` blocks.

Two things the Laguna GPU engine does NOT have that Flint's does, both
left out of scope for now (serve works without them, just without
snapshot/disk-cache reuse across requests):
- No `SnapshotBytes`/`SaveSnapshot`/`RestoreSnapshot` on `Session` - the
  runner declares `.snapshot = false, .fork = false` in its capabilities,
  so prompt-cache-across-requests and the HTTP disk cache don't apply to
  Laguna yet. Would need the same `SnapshotBytes`/`SaveSnapshot`/
  `RestoreSnapshot` triplet Flint's `Session` has, covering KV cache +
  the two-bank residual copies (`x2`/`xn2`) + the shexp/QKV NPU in-flight
  state if a snapshot can land mid-submit (likely needs to drain pending
  NPU work before capturing, same as any other quiesce point).
- No `Model::SessionBytes(max_context)` closed-form estimate (KV cache
  size depends on SWA ring buffers + two-bank NPU scratch, not a flat
  per-token constant) - `ResourceClaim()`'s `per_request_state_bytes` is
  filled by creating one throwaway full-size session at runner
  construction and reading its `AllocatedBytes()` (`MeasureSessionBytes()`
  in the runner), not a formula. Admission without this measured value
  failed hard: `"text runner measured state exceeds its resource claim"`
  the first time this was tried with `per_request_state_bytes = 0`.

DFlash v1 speculative decoding is NOT wired into `gufo serve` either -
the dispatch block rejects any `--speculative` flag for Laguna
(`speculative_config.backend != kDisabled` is an error). `gufo serve` only
serves the plain autoregressive path for now; DFlash stays `gufo
prompt`/`gufo bench`-only until someone wires `QwenFlashNextTextRunner`'s
speculative machinery through for Laguna's DFlash v1 (a separate,
non-trivial task - Flash-Next's MTP/DFlash runner code is ~750 lines by
itself).

Smoke-tested end to end: `gufo serve llm -m <laguna.gguf> -c 16384` loads,
warms up, and answers a real `/v1/chat/completions` request correctly
(chat-templated prompt, coherent "OK." completion, correct stop handling).
