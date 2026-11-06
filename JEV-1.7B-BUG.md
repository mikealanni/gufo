# RESOLVED: Qwen3-1.7B garbage output through jev's NPU judge engine

## File

`src/models/qwen38_flash_next/jev_npu.hip.cpp` (the "gufo judge" small-model
NPU engine — separate from the main Flash-Next engine) and
`src/models/qwen38_flash_next/kernels/rocm/npu_gemm.cpp` (the NPU GEMM
executor it drives).

## Resolution (2026-11-05)

Two independent bugs, both fixed; verified against a torch fp32 CPU
reference (`~/jev-ref/`) element-by-element:

1. **`encode_one` weight encoding** — the fused qkv/gate-up buffers were
   built by calling `NpuEncodeWeightsQuant(pin, type, k_total, /*row0=*/0,
   cols, k_total, 0, /*k_valid=*/cols, ...)` once per 128-column chunk. The
   kernel advances through the tensor's rows via its own column-tile index,
   which is always 0 when `cols==128`: every weight tile ended up encoding
   rows 0..127 of the source duplicated across all tiles, and `k_valid=128`
   truncated the reduction length to the first 128 values. Fix: one
   full-width call per source (`cols=w, k_valid=k_total`) writing directly
   into the fused buffer at the source's `(col0/128)`-tile offset.
   Evidence at the time: per-tile decoded sums identical within
   q/k/v groups; encoded density 1/8–1/16 of expected.

2. **Attention softmax max reduction** (`JevAttnKernel` +
   `JevAttnBatchedKernel`) — after the per-warp butterfly, each thread kept
   its *partial* suffix max and folded in only `red[1..warps]`, never
   `red[0]`, so the softmax denominator's `m` was not the row max: every
   key got weight `exp(s_p - suffixmax_p)` instead of
   `exp(s_p - globalmax_p)`. Near-uniform early layers (scores spread < 1)
   hid it; the peaked deep layers diverged completely (per-layer hidden
   cosine vs torch fp32: 0.999 at L0–L2, 0.68 at L3, 0.23 at L27).
   Fix: adopt `m = max(red[0..warps])` in every thread (the `sum`
   reduction already summed all of `red[]` correctly).

Both were invisible before the prompt-framing fix because with the broken
encoding *everything* looked broken; the softmax bug then made 0.6B score
"reasonably" while 1.7B produced the Chinese-garbage logits (its sharper
attention is more sensitive to the max bias).

## Verification (all models, `GUFO_JEV_DEBUG=1` off / on, exclusive NPU)

- Elementwise vs torch fp32 (same no-think framing, n=54):
  qkv-l0 cos 0.9999, attn-l0 cos 0.9988, `s2qkv` at L13 2.485e6 vs torch
  2.486e6 (0.04%).
- Qwen3-1.7B-Q6_K battery: trivia 7/10, code 11/12 (CPU reference:
  8/10, 11/12 = 19/22). The NPU's three misses are the CPU model's own
  misses (sky-green / cheese / two-legs, positive CPU logit gaps).
- Qwen3-0.6B-Q6_K: 7/10 + 7/12 = 14/22 vs CPU 13/22 — CPU itself has
  `sep=NO` (every logit gap positive) on this base framing, so 1.7B is the
  production judge.
- No `Memory Fault`, no stale-output poisoning across a 4-call window.

## Ruled out during investigation — do not re-check

- `NpuGemm::NewBuffer`: fresh independent allocation per call; no aliasing.
- Instruction streams for 1.7B shapes (K-loop word = K/64 verified per
  file; files differ correctly between K=1024/2048 variants).
- Program indexing (`AddProgram` returns 1-based, `extra_programs[i-1]`),
  decode-kernel index bijection (proved bijective on a CPU re-implementation),
  `n_qkv_cap` (resolves to exact 4096 for both models).
- `q[1024]==q[0]`-style "column aliasing" in decoded q rows: it was
  duplicated *weights* (bug 1), not index math.
- Per-call `hipHostMalloc` for `qkv`/`gu` stays (caching them caused the
  call-2 memory fault; see plan doc).
- `rawC_t0_eq_t8≈0.9` raw-C tile probe: an artifact of row padding
  (54/512 real rows → 89.5% zero bytes match). The C tiles are distinct.
