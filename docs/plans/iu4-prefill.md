# int4 x int4 WMMA prefill for all models

Goal: use the gfx1151 `v_wmma_i32_16x16x16_iu4` instruction for dense and
expert prefill GEMMs on every model, then drop the Q8 mirrors.

Measured (this box, `gfx1151_peak`-style loop): fp16 WMMA 57 TFLOPS, iu8 57
TOPS, iu4 110 TOPS (1.92x). Qwen3.8-27B GPU-only prefill (pp8192, 644 tok/s)
spends about 85% of kernel time in the fp16 GEMM (`HalfPrefillGemmKernel`:
Q5_K 36%, Q4_K 17%, IQ4_XS 22%, Q6_K 7%, IQ4_NL 2%), attention 3.7%, GDN scan
3.4%. A perfect 1.92x on the GEMMs bounds GPU-only pp8192 at about 1,090 tok/s;
plan on 1,000 to 1,100 before the NPU split is retuned.

Rules from the project owner:
- Weights come from artifacts already quantized by others; we do not quantize
  our own models.
- Mirrors go away once the int4 path is the default for a model.
- Order: Qwen3.8-27B, Qwen3.8 Flash-Next, GLM-5.3 Flash, then the rest.
- Quality gate: NLL on the existing quality corpus against the current default
  quant of the same model; production paths keep quality.
- Kernels are written independently; no external project names in the repo.

What an int4 x int4 GEMM needs:
1. Weights stored as symmetric 4-bit groups (group scale, zero point -8).
2. Activations quantized per group to signed int4 at run time. This is only
   accurate when activations and weights share an orthogonal rotation along K
   (a signed Hadamard transform over 256-wide blocks), so the weights must be
   stored already rotated. An unrotated artifact (plain symmetric group-128
   W4A16 builds) needs a quality measurement before it can use this path.
3. Fused producer kernels: norm + rotate + int4 quantize, and
   silu*mul + rotate + int4 quantize, so activations are quantized once.
4. A chunked (WY form) delta-rule scan for the GDN layers.

Artifact survey:
- Qwen3.8-27B and Flash-Next: rotated symmetric 4-bit artifacts exist
  (136 bytes per 256 weights: two fp16 group headers plus 128 bytes of
  nibbles). Flash-Next base is 70.9 GB plus a PLE sidecar.
- GLM-5.3 Flash: only unrotated W4A16 symmetric group-128 builds (AWQ,
  AutoRound) and NVFP4 builds were found.
- Third-party IU4 GGUF builds need their own runtime and sidecar files and
  report lower prefill than we already have; not used.

Stages:
1. Kernel: standalone iu4 GEMM in `tools/bench` for the 27B shapes, checked
   against an fp32 reference, timed against the fp16 GEMM.
2. Producer kernels and rotation, checked bit-exactly against a host model.
3. Loader for the 27B artifact and the prefill path behind the existing
   quality tools; NLL gate.
4. Retune the NPU split on top; remove the mirrors for that model.
5. Repeat for Flash-Next (dense + routed experts), then GLM.

## Status (2026-11-20)

Kernel experiment: `tools/bench/iu4_gemm_bench.hip`. Verified against a CPU
model of the same arithmetic on all six distinct Qwen3.8-27B projection shapes
(relative error 7e-5 or better). Measured at 2048 tokens:

| Stage | TOPS (peak 110) |
| --- | --- |
| first version, operands read straight from global memory | 23.7 |
| LDS tiled, 128 x 128 block, 8 waves | 42 |
| signed weights (nibble ^ 0x88), register prefetch, non-unrolled group loop | 54-62 |
| core loop only (no staging, barriers or epilogue), pipelined fragments | 77.8 |

The activation producer (divide by AWQ scale, rotate, int4 quantize, integer
sum) runs in 0.25 ms for 2048 x 5120, bit-identical to the first version.

Target for production: 100 +/- 8 TOPS in the full GEMM.

Artifact facts needed by the integration (Qwen3.8-27B rotated int4 artifact):
- Every projection and the lm_head are 136-byte symmetric 4-bit groups; the
  embedding is 8-bit with f16 scales per 32 values (the GGUF Q8_0 layout);
  norms and small tensors are f16. A lossless container conversion into GGUF
  with a new tensor type for the 4-bit groups is enough; no requantization.
- The AWQ divide vector is identical for every projection that reads the same
  input (q/k/v, GDN qkv/z/a/b, gate/up), so for norm-fed inputs it folds into
  the RMSNorm weight. The o, out and down inputs need an explicit divide in
  their fused producer.
- The GDN output projection input follows an `hf-kmajor-vminor` head order that
  the engine's GDN output layout must match.
- Prefill call sites are the `LaunchBatchedQuantGEMM*Fp16` family in
  `src/models/qwen/hip/ops/prefill_fp16.hpp` (plain, residual-add, fused SwiGLU,
  row-range variants used by the NPU split); the new path plugs in behind the
  same signatures with a new `core::GgmlType`.

### Ceiling analysis (2026-11-20)

Isolated loop measurements (same 8-accumulator wave tile as the kernel):

| Loop | TOPS |
| --- | --- |
| WMMA only, operands in registers | 107 |
| plus pipelined LDS fragment loads | 83-90 |
| plus per-128-group scale epilogue (cvt, mul, fma per element) | 76 |
| full kernel with global staging and barriers | 54-62 |

The epilogue VALU work does not overlap the WMMAs on this chip, so it adds
directly; per element per group it is about 2 VALU cycles against 16 cycles of
matrix work per 8 elements. The scale epilogue is required because both the
weight scale and the activation scale change every 128 K values.

An independent hand-tuned implementation of the same weight format measured
57 to 67 TOPS on the same six shapes on this machine (N = 2048), and its
published 27B prefill of 1,150 tok/s at pp8192 implies GEMMs averaging about
65-73 TOPS. 100 TOPS has not been demonstrated for this format on this chip;
the realistic target is about 70 TOPS. Going beyond needs a weight format with
one scale per row (a single int32 accumulation over all of K), which is a
different artifact, not a kernel change.

### Power and overlap measurements (2026-11-20)

Clock under sustained load (`hwmon freq1_input`):

| Run | Power limit | GFX clock | TOPS |
| --- | --- | --- | --- |
| WMMA only, registers | default | 2.73 GHz | 103 |
| full kernel | default | about 2.04 GHz | 50 |
| WMMA only, registers | 120 W | 2.88 GHz | 107.6 |
| full kernel | 120 W | 2.59 GHz | 57-58 |

At the 120 W limit the clock is no longer the limiter; the full kernel runs at
about 59% of the clock-adjusted register-only rate.

VALU/WMMA overlap (8 accumulators, VALU work as independent fmaf chains):

| Placement of the VALU work | WMMA TOPS |
| --- | --- |
| none | 108.2 |
| same wave, 8 / 16 / 24 / 32 fmaf per 8 WMMA | 100.5 / 95.2 / 90.5 / 86.4 |
| other waves (half the waves), 24 or 64 fmaf per step | unchanged (51.0 vs 51.6 for the WMMA half) |

VALU work in the same wave serializes with its WMMAs (0.8% of WMMA time per
fmaf per 8 WMMA); VALU work in other waves is free. The per-group scale
epilogue therefore has to move to separate waves (MMA waves publish int32
results through LDS, epilogue waves convert, scale and accumulate). A first
prototype was occupancy-limited (60 KB of LDS per block leaves about 2 MMA
waves per SIMD, and the fragment loads need deeper prefetch at that
occupancy); not yet measured at full effect.

### Warp-specialized epilogue experiment (2026-11-20)

Kernel variant with MMA waves publishing int16 group results through LDS to
separate epilogue waves (which also stage global data); correct (max difference
1e-6 against the single-wave kernel) but slower:

| Variant | TOPS |
| --- | --- |
| single-wave epilogue (current best) | 57 |
| core structure only (staging, barriers, no handoff, no epilogue) | 76.8 |
| plus handoff writes, no waiting, no epilogue | 57.0 |
| plus handoff wait, no epilogue | 53.8 |
| complete warp-specialized kernel | 48 |

Draining each group's WMMA results, packing and storing them costs more than
the epilogue VALU work it removes. Isolated LDS-fragment loop at 120 W: 88 TOPS
at 2 waves per SIMD, 95 at 4, 97 at 12, so the matrix and LDS paths are not the
limit; the per-group handoff or epilogue is.

Conclusion: with per-128-group weight and activation scales the full kernel
tops out near 57-62 TOPS in HIP C++. Remaining untested levers: weights
straight to registers, double-buffered A, a 256 x 256 tile, and a per-token
activation scale (one cvt and one fma per element, at a quality cost).

## Scale-consumption experiments (clocks locked, 120 W, 27B gate_proj, N=2048)

| Variant | TOPS | Notes |
|---|---|---|
| v3 (128x128 tile, per-group activation and weight scales) | 58.5-58.9 | baseline |
| per-token activation scale (timing ablation, `EXP_PT`) | 61.7 | epilogue `acc += sc*float(d)`, one row multiply at the end; output differs by design |
| v5: 256x128 tile, 16 waves, exact semantics | 61.3 | correct (max diff 9.5e-7) |

The weight scale is already loaded once per group per lane (two LDS reads) and
held in registers across the WMMA block; the combined product costs one multiply
per element. Removing the activation scale entirely only gains 5%, so the
epilogue is not the dominant loss: staging and barriers are. The two gains are
roughly additive (about 65 TOPS); 100 TOPS is not reachable with this format.

### Stacked (256x128 tile + per-token activation scale), same conditions

| Variant | TOPS |
|---|---|
| v5 + `EXP_PT` | 65.7 |
| + no barriers (timing bound, incorrect output) | 70.5 |
| + no epilogue as well | 70.7 |

With the per-token scale the epilogue is free (removing it changes nothing),
barriers are worth about 7%, and the rest is global-load staging. The bound for
this design is about 70 TOPS.

### Reference-architecture kernel (v6) and where the remaining time goes

v6 follows the structure of the published production kernel for this format:
K128 epochs, two 16 KiB LDS slots (one barrier per epoch), fragment-major LDS
layout (each WMMA operand fragment is one contiguous 128-byte run, 64-bit
reads), 32-token x 64-feature wave tile, fold per token fragment with the
token scales shared through DPP row_share, weight scales prefetched one epoch
ahead, tile-major grid order. It is exact (max diff 9.5e-7) and uses 170 VGPRs.

| Variant | TOPS |
|---|---|
| v6 exact | 58.5-59.0 |
| v6 per-token scale | 61-63 |
| v6 without next-epoch loads | 68.5 |
| v6 without the fold | 69.0 |
| v6 without loads and fold | 74.6 |
| v6 with 128-bit fragment reads (dropped, 46) | 46 |
| v5 exact / v5 per-token | 61-63 / 65-67 |

Findings: the production-style design lands on the same 58-59 as v3; the
256x128 v5 tile is the better exact kernel. LDS co-residency (1, 2 or 3 CTAs per
WGP) changes nothing, so cross-CTA overlap of fold and WMMA is not what limits
it. Loads, fold and barriers each cost 6-8% and add up; even with loads and
fold removed the structure reaches 74.6, not 88-97. Best exact: v5 at about 62;
best with per-token scale: about 67.

### Shape sweep, v8, and clock under load

Shape sweep (N=2048, exact / per-token): gate_proj 58-63 / 62-66, down_proj
(K=17408) 59-64 / 62-67, up_proj 56-60 / 60-65, q_proj 57-61 / 60-64, in_proj_qkv
57-62 / 61-66. The kernels are not shape sensitive; v6 is better on the long-K
down_proj and v5 on K=5120. v8 (256x128 tile with the K128 double-buffered
structure) is correct and not better (56-62, 68.8 once on down_proj).

Clock under load: the register-only WMMA loop sustains 2.88 GHz (107.6 TOPS,
37.4 TOPS/GHz). The full kernels run at 2.50-2.52 GHz at 120 W (about 24.5
TOPS/GHz); the loads-and-fold-removed ablation runs at 2.65 GHz and 140 W
(28.1 TOPS/GHz). Even a perfect kernel at those clocks tops out near 94-99
TOPS (37.4 x 2.5-2.65), and every byte moved through LDS, VMEM and VALU costs
power and lowers the clock further. 92+ TOPS would need more than 95% of that
bound, so the practical ceiling for a kernel that stages real data is about 75.

### v9: weights straight from global memory into registers

Weights are repacked once at load time into `[epoch][row][64 B]` payload plus a
`[epoch][row]` scale array (a lossless relayout), so each lane reads its weight
fragment with four 16-byte loads per 128-K epoch and no LDS staging. Only the
token tile goes through LDS (8 KB slots, double buffered, one barrier per
epoch). The next epoch's weight registers are reloaded in place as each K32
pair is consumed. Wave tile 32 features x 64 tokens, 128x128 CTA.

| Variant (gate_proj, N=2048) | TOPS | VGPRs |
|---|---|---|
| v9 per-token activation scale | 69-71 | 256 |
| v5 per-token (previous best) | 66 | |
| v9 exact per-128 scales | 46 | 244, no spill |

The weights-to-registers change is worth about +5 TOPS. The exact fold is
slow here (46): all waves leave the barrier in lock-step, so every wave runs
its 240 fold VALU instructions at the same time and nothing overlaps the
WMMAs.

Per-token activation scale quality, same synthetic activations as the
bench (heavy channels, rotation within 256-blocks only): relative RMS 0.160
versus 0.115 for per-128 scales. That is 40% more activation error, so the
per-token scale is not free and needs the real-activation quality test before
it can ship.
