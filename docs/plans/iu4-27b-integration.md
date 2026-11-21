# Qwen3.8-27B int4 prefill integration

Worktree `~/gufo-iu4`, branch `iu4-27b`, builds under `/mnt/raid0/iu4-build`
(`/home` is nearly full). Kernel and ceiling work is in
[iu4-prefill.md](iu4-prefill.md); this file tracks the engine integration.

## Go/no-go (2026-11-20): passed

Real 27B FFN-input activations (3,972 tokens, layers 3 to 59) pushed through the
artifact's per-layer AWQ divide, the signed Hadamard rotation over 256 and
per-128 symmetric int4 quantization: relative RMS error 0.1163 to 0.1171 on every
layer (0.157 to 0.215 without the rotation). A clip search over eight candidates
changes nothing. That is the Gaussian int4 noise floor, so the rotation removes the outlier
problem and the NLL gate is worth running.

## Artifact facts (qwen3.8-27b.mq4-xts, 64 layers, 1,347 tensors)

- Projections and lm_head: type 44, 136 bytes per 256 K. Per layer: full
  attention q (12288 x 5120), k and v (1024 x 5120), o (5120 x 6144); GDN
  in_proj_qkv (10240 x 5120), in_proj_z (6144 x 5120), in_proj_a and in_proj_b
  (48 x 5120), out_proj (5120 x 6144); MLP gate and up (17408 x 5120), down
  (5120 x 17408).
- One f16 `awq_scale` vector per projection input (5120, 6144 or 17408 long).
- Embedding is type 3 (the GGUF Q8_0 layout); norms, A_log, dt_bias, conv1d are f16/f32.
- The base GGUF used by the engine (UD-Q4_K_XL) has 65 blocks: block 64 is the MTP
  head and has no counterpart in the artifact.
- All M are multiples of 512 except in_proj_a/b (48): those two need a small
  fallback GEMM. All K are multiples of 256 with an even number of 128 groups.

## Design

1. Sidecar first. The engine keeps loading the base GGUF for everything it
   loads today; a new loader reads the artifact, relayouts each projection into
   the tile-major layout (`RelayoutW`) at load time (lossless) and keeps the AWQ
   vectors as f32. Prefill GEMMs read the sidecar; decode is untouched. This
   gets an NLL number without writing the decode GEMV. Later the sidecar becomes
   the only weight source and the base-GGUF projections and mirrors go away.
2. Routing is by weight type, with no flag: a projection that has a sidecar
   entry runs the int4 path.
3. Activation producers (one per input class), writing the tile-major token
   layout and the `[epoch][token]` float scales the GEMM reads:
   - norm-fed (q/k/v, GDN qkv/z/a/b, gate/up): RMSNorm, divide by AWQ, rotate, quantize
     (the AWQ vector is shared by every projection reading that input);
   - attention output (o): sigmoid gate multiply, AWQ divide, rotate, quantize;
   - GDN output (out): gated norm output, remapped to the artifact head order
     (`hf-kmajor-vminor`), AWQ divide, rotate, quantize;
   - SwiGLU (down): silu(gate) * up, AWQ divide, rotate, quantize.
4. GEMM: the exact per-128 kernel from the bench (v13), with N padded to a
   multiple of 128 in the producer and a masked epilogue; residual-add and
   fused variants as needed by the call sites.
5. Oracles: per layer, the int4 GEMM against the fp-activation result on the same
   dequantized weights; end to end, NLL against the base GGUF on the existing
   quality corpus. Report both before any speed claim.

## Steps

- [x] go/no-go on real activations
- [ ] artifact loader and weight relayout (C++), unit-tested against the bench CPU model
- [ ] producers (norm-fed first), checked against the bench host model
- [ ] GEMM port with N tail and masked epilogue
- [ ] hook the dense layers behind the weight-type routing, NLL run
- [ ] SwiGLU, o and out producers; lm_head; remaining layers
- [ ] speed at pp2048/8192 with bench mode on; NPU split retune; remove mirrors

## First end-to-end results (2026-11-20, GPU only, no NPU)

Release build, bench mode on, `gufo bench -p 2048,8192 -n 128 -r 2`, base GGUF
UD-Q4_K_XL with `GUFO_IU4_ARTIFACT` set or unset:

| | pp2048 | pp8192 | tg128 |
| --- | --- | --- | --- |
| base fp16 prefill | 687.5 | 643.1 | 12.47 |
| int4 prefill | 1076.9 | 1004.7 | 12.30 |

Prefill +57% at 2k and +56% at 8k; decode is unchanged (it still reads the base
GGUF and the plain decode path).

Quality (`qwen27b_prefill_nll`, 240 samples of a 2048-token context from
`ppl_corpus.txt`, next-token NLL): base 1.0215, int4 1.0205, paired difference
-0.0010 +- 0.0362 (standard error), top-1 agreement 92.1%. Within the resolution
of the sample no quality loss shows up; the base and the artifact are different
weight quantizations, so this is the combined effect of weights and activations.

## NPU on the int4 path (2026-11-20)

pp8192, release build, bench mode on: base GPU 646, base + NPU 801, int4 GPU
1013, int4 + NPU 1118; pp16384 624 / 764 / 957 / 1036; pp2048 693 / 736 / 1090 /
1094 (the NPU stays out of lone chunks). 14 FFN gate/up slices and nothing else
on the NPU is the best split; see the iu4-prefill skill for the sweep, the
per-bank scratch requirement of the lockstep pair, and the NPU kernel
opportunity.
