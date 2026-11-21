---
name: iu4-prefill
description: Port a model's prefill to int4 x int4 WMMA on gfx1151 (rotated symmetric 4-bit artifact, per-128 exact scales). Everything learned on Qwen3.8-27B - format, kernels, tests, engine hookup, measured numbers, dead ends, and the checklist for Flash-Next and GLM.
metadata:
  origin: gufo
---

# int4 x int4 WMMA prefill

Use when a model's dense or expert prefill GEMMs should run on
`v_wmma_i32_16x16x16_iu4`. Qwen3.8-27B is done (GPU only); Flash-Next and
GLM-5.3 follow. Read `docs/plans/iu4-prefill.md` (kernel experiments) and
`docs/plans/iu4-27b-integration.md` (engine integration) first, then
`.agents/skills/optimize-kernel/SKILL.md` for the measurement rules.

## Rules from the project owner

- Weights come from artifacts already published by others. Never quantize our
  own models. Activations are int4 too.
- Mirrors (Q8 copies) go away once the int4 path is the default for a model.
- Order: Qwen3.8-27B (done), Qwen3.8 Flash-Next, GLM-5.3 Flash.
- Production paths keep quality; a successful optimization becomes the default
  (the env var below is a transitional switch, not the end state).
- No external project names in repo files or commits.

## Result on 27B (release build, bench mode on, GPU only)

pp2048 693 -> 1090 tok/s, pp8192 646 -> 1013, pp16384 624 -> 957 (GPU only), and
with the NPU 1118 at pp8192 / 1036 at pp16384 (the old base + NPU was 801 / 764).
Decode unchanged (still the base GGUF, 12.5 tok/s plain). Next-token NLL over 240 samples (2048-token contexts of
`~/ppl_corpus.txt`): 1.0215 base, 1.0205 int4, difference -0.001 +- 0.036,
top-1 agreement 92%. The NPU adds about +10% on top for prompts of 5000+ tokens (see NPU below).

## Hardware facts that matter

- iu4 WMMA peak on this chip: 111 TOPS with constant operands, but about 90 with
  real/random data (the matrix unit slows with data activity and the chip is
  held to 120 W, about 2.5-2.7 GHz). The old "108 TOPS" targets use constant
  data. Judge a kernel against ~90, not 108. The exact per-128 kernel reaches
  75 TOPS (83%); per-token scale reached 76.7 but is rejected (below).
- Benchmark only with `sudo ~/bench-mode.sh on` (fabric clock held, EC
  performance, 120 W limits, comfy-monitor stopped). `bench-mode off` applies
  85 W limits and comfy-monitor changes limits mid-run. Run one GPU job at a
  time: overlapping jobs corrupt timings.
- Measure with a release build; test builds keep assertions.
- `/home` is nearly full: put worktrees' build dirs and artifacts on
  `/mnt/raid0`. Worktree `~/gufo-iu4` (branch `iu4-27b`), builds in
  `/mnt/raid0/iu4-build/{gpu-test,release}`. NPU build needs
  `-DGUFO_ENABLE_NPU=ON -DGUFO_XRT_INCLUDE_DIR=/home/majed/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include -DGUFO_XRT_LIB_DIR=/opt/xilinx/xrt/lib`.
- Never `pkill -f` a pattern that matches your own shell command.

## Artifact format (mq4, type 44)

- 136 bytes per 256 K per row: `[sc0 zp0 sc1 zp1]` f16 for the two 128-K halves,
  then 128 payload bytes (byte j of a half holds K 2j low nibble, 2j+1 high).
  Weights are stored already rotated along K by `R = D2 H D1 / 16` (signed
  Hadamard over 256, sign seeds 42 and 1042 via
  `st = (st*1103515245+12345) & 0x7fffffff; sign = bit 16`).
- Symmetric: `zp == -8 * sc` for every group (the loader checks all of them), so
  XOR 0x88 on the payload gives signed codes and the zero point drops out.
- Activations: divide by the projection's `awq_scale` vector, rotate with the
  same signs, quantize each 128 group to signed int4 with scale `max|x|/7`
  (no clip search: it changes nothing after rotation). Projections reading the
  same input share one awq vector (loader verifies: q/k/v/qkv/z/a/b and
  gate/up).
- Per-128 weight scale and per-128 activation scale both change every group, so
  the epilogue must fold every 128 K: `out += sx * sw * float(int32 dot)`.
  Integer accumulators start at `0x4B400000` so one subtract of `12582912.0f`
  converts the int32 exactly.
- Per-token activation scale (one scale per token) is rejected: on real 27B
  FFN-input activations its relative RMS error is 0.42-0.82 against 0.157-0.215
  for per-128 without rotation and 0.116 with rotation + AWQ. Outlier channels
  (max/rms 36-61) kill it.

## Verify a new artifact before writing engine code (cheap, decisive)

1. Go/no-go on real activations: dump the post-norm FFN input of the base
   engine at a few layers (temporary env-gated dump in the prefill, never
   commit it), push it through awq divide + rotation + per-128 int4 and measure
   relative RMS error. Pass: about 0.12 (Gaussian int4 floor). Fail: about 0.2.
2. Seeds and orientation: correlate `artifact_weights . rot(x/awq)` with
   `base_gguf_weights . x` on real activations for a few rows. Correct seeds
   give 0.99; wrong order, no rotation or no signs give about 0.
3. Row/column permutations: for every projection compute the correlation
   matrix between base-engine output chunks and artifact output chunks (random
   x) and read the permutation off the argmax. For Qwen GDN: engine head e
   corresponds to artifact head `(e % 16) * 3 + e / 16` for in_proj z, a, b and
   the V rows of in_proj_qkv (rows from 4096 on, 128 rows per head); q/k rows are
   identical. The out_proj input columns cannot be permuted in the weights (the
   rotation blocks are 256 wide), so its producer reads the GDN output through
   the same mapping.

## Code map (27B, worktree `iu4-27b`)

- `src/models/qwen/hip/iu4_weights.{hpp,cpp}`: `Iu4Weights` loads an mq4
  artifact, repacks every projection to the tile-major layout (lossless),
  applies the GDN row permutation, uploads awq as f32, owns the rotation signs.
  `Find(layer, role)`, `LmHead()`. Loaded lazily by `QwenGpuModel::Iu4()` from
  `GUFO_IU4_ARTIFACT`.
- `src/models/qwen/hip/ops/iu4.hpp` + `kernels/iu4_repack.hip`: repack,
  `LaunchIu4NormQuantize` (RMSNorm + awq + rotate + quantize, one workgroup per
  token, K <= 8192), `LaunchIu4Quantize` (quantize-only; any K; optional GDN
  head remap), `LaunchIu4SwapHalfColumns` (NPU glue).
- `kernels/iu4_gemm.hip`: `LaunchIu4Gemm(tiles, scales, act_tiles, act_scales,
  out, accumulate, rows, row_begin, n, padded_n, k, stream)` and the scalar
  `LaunchIu4GemmSmall` for the 48-row GDN a/b gates.
- `prefill_chunk.cpp`: `iu4` flag (artifact loaded, batch >= 256, 27B dims);
  eight call sites switch to the producers + GEMM: pre-layer norm, q/k/v, o,
  GDN in-projections, GDN out, FFN norm, gate/up + SwiGLU, down. Residual-add
  GEMMs accumulate into `d_hidden`. The final-layer partial-row trick is off.
- Layouts. Weights: `(((e*(M/16) + row/16)*4 + q)*256 + (row%16)*16`, q = 16-byte
  chunk of the 64 payload bytes of 128 group e; scales `[e][row]` f16. Tokens:
  tiles of 16, per 128 group a 1024-byte block, byte
  `((chunk&1)*16 + t%16)*32 + (chunk>>1)*8 + within` for 8-byte chunk
  `chunk = byte/8`; scales `[e][padded_n]` f32. N is padded to 128; padded rows
  carry zero codes and scale 0; the GEMM epilogue masks t >= n.
- Tests (`tests/models/qwen27b/`, need `GUFO_IU4_ARTIFACT`, skip with 77):
  `qwen27b_iu4_weights_test` (payload/scales preserved, permutation, symmetric
  reading), `..._producer_test` (all four producers vs a double precision
  model, zero real mismatches), `..._gemm_test` (real projections, tails, row
  range, accumulate), `..._chain_test` (producer -> GEMM -> weights vs fp32
  activations: relative error about 0.10), and the `qwen27b_prefill_nll` tool
  (`<model.gguf> <corpus> <samples> <length> [seed]`, one prefill per sample,
  compare two runs with the same arguments).

## GEMM design (v13) and what was tried

Exact per-128 kernel, CTA 128 features x 128 tokens, 8 waves (wave tile 32
features x 64 tokens), 1 barrier per 256 K, weights global -> registers
(double buffered across the two 128 groups), tokens through LDS, half the
waves defer one fold so their VALU overlaps their neighbours' WMMA, 16-byte
token loads, CTA order grouped four feature tiles wide (L2 reuse), 216-234
VGPRs, 3 CTAs per WGP. 75 TOPS at 2048 tokens on every 27B MLP shape.
Gains stacked from 58: weights straight to registers +5, staggered fold +7,
no DPP row-share +11, tile-major layouts +4, K=256 per barrier +2, grouped order
+2, token loads +1.
Dead ends: wave priority (`s_setprio`), interleaving token loads across the WMMA
stream, 2 CTAs per WGP (72.8), 4 CTAs (needs 192 VGPRs: spills, 51), 64 x 32
wave tile (72.4: same fold work, more redundant weight loads), no-LDS design
(60), 128-bit fragment reads, removing the epilogue stores, Curve Optimizer on
CPU or GPU (no clock effect). The fold costs about 6 TOPS and cannot shrink
exactly (three VALU ops per output element per group; gfx11 has no packed fp32
fma; fp16 packing loses exactness).

## NPU (27B) - works, +10% at pp8192, FFN gate/up only

All NPU sites work on the int4 path (qkv/gate, o, ssm_out, gate/up, down): the
GPU runs `row_begin = NPU columns`, the NPU keeps the leading columns, FP16
activations are produced only where the NPU takes columns, and the NPU's FP16
SwiGLU columns are converted into the FP32 activation by
`LaunchIu4HalfColumnsToF32` (and the GPU columns into an FP16 copy by
`LaunchIu4F32ColumnsToHalf` only when the NPU also takes down columns).
Run it with `GUFO_NPU_DIR=<repo>/tools/npu`,
`LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/home/majed/npu-work/boostshim`, bench mode
on (fabric clock held), only inside binaries named `gufo*`.

Lessons (each cost a wrong number):
- The two chunks of a lockstep pair interleave mid-layer (they `co_yield` after
  an NPU begin). Any scratch the int4 path owns (activation tiles, scales, the
  NPU half buffer) must be per bank, indexed by the task's `bank`; a shared
  buffer silently corrupts the other chunk.
- The NPU only pays when the other chunk of a pair hides its hand-off: pairs
  start at 5000 tokens (`GUFO_QWEN_PAIR_MIN`) and need 4096-token chunks. A lone
  chunk waits for the NPU, so the int4 path leaves `npu_ffn` null unless
  `Paired()` (pp2048 stays 1094 with or without the NPU).
- The NPU does about 20 TOPS-equivalent (bfp16, K=5120: 2.5 ms per 1024 columns
  per 4096 tokens) against 75 on the int4 GPU GEMM, so its useful share is about
  a fifth. Only the FFN gate/up columns (same K as the attention inputs, one
  hardware context, output needed late) help. FFN slices at pp8192: 0 1033, 6
  1064, 8 1102, 10 1125, 12 1148, 14 1152, 16 1084 (NPU becomes the bottleneck).
  Adding qkv, gate, ssm_out, down or attention slices on top lowers throughput
  (down 827, qkv 952, ssm_out 1100) because the NPU queue is already full and
  those outputs are needed immediately or use another K (3.7 ms context switch).
  The loader sets 14 FFN slices and 0 elsewhere (explicit env still wins).
- Glue cost matters: converting the whole [tokens x 17408] activation per layer
  between FP16 and FP32 cost more than a slice gained; convert only the NPU
  strip.
- Result (release, bench mode on, GPU base 646 at pp8192): base + NPU 801, int4
  GPU 1013, int4 + NPU 1118 (pp16384: 624 / 764 / 957 / 1036). Ideal overlap of
  a 20-TOPS NPU on a 75-TOPS GPU is about +27% (about 1290); the gap is the
  NPU's 42% utilisation of its bfp16 peak and the paired schedule.
- Next NPU lever is the NPU kernel itself: it computes in bfp16 (9 bits per
  weight) at about 42% of the 50 TOPS peak. XDNA2 also lists int8 at 50 TOPS and
  int4 as a supported type (no published peak). An int8 activation x int4-weight
  K=5120 kernel (weights straight from the artifact, per-group scales applied on
  the host or in the epilogue) is a new AIE kernel project (30+ min builds per
  iteration); measure its realised TOPS on a standalone probe first.
- Validation numbers: `gufo bench --validate-prefill 4096` compares batched
  prefill logits with the sequential decode path on the base weights, so with the
  artifact it measures the weight difference too: base 0.008 rmse / cosine
  0.99999, int4 0.515 / 0.989, int4 + NPU 0.627 / 0.986, top-1 matches.

## Decode

Unchanged: it reads the base GGUF and the existing decode kernels (12.3-12.5
tok/s plain). The speculative drafts (MTP, DFlash2) keep working because chunks
under 256 tokens take the old path. A decode GEMV for the mq4 format (and then
dropping the base GGUF projections and the Q8 mirrors) is the remaining 27B
step; then the quality gate on code (HumanEval) and long contexts.

## Checklist for the next model

1. Artifact survey: shapes, per-projection awq, which tensors are mq4, which
   are not (embedding is Q8_0 layout). All output sizes must be multiples of 128
   (the GEMM runs in 128-feature tiles); smaller ones use the scalar kernel.
2. Run the three verification steps above. A model with only unrotated W4A16
   artifacts (GLM) fails step 2 by construction: measure step 1 without rotation
   first (raw per-128 error was 0.157-0.215 on 27B) before any engine work.
3. Loader: copy `Iu4Weights`, add the model's role table and permutations.
4. Producers: reuse; add fused variants only after correctness (one per input
   class: norm-fed, quantize-only, SwiGLU).
5. Flash-Next: dense attention/GDN layers follow 27B; routed experts need a
   grouped (per-expert row range) int4 GEMM over the same tile layout and the
   expert-token gather in the producer; shared experts reuse the dense path.
   Artifact is large (70.9 GB plus the PLE sidecar): keep it on `/mnt/raid0`.
6. Hook behind the weight-type routing, then qualify: unit tests, chain test,
   NLL tool against the model's default quant, then pp/tg at depth with bench
   mode on. Update this skill and the plan doc with every new finding.
