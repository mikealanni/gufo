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
