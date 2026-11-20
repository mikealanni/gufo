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
