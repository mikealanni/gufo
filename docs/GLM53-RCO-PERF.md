# GLM-5.3-Flash RCO-152E performance worklog — prefill + decode + MTP

Machine: AMD Strix Halo (Ryzen AI MAX+ 395, gfx1151, 120 GB shared RAM).
Model: `GLM-5.3-Flash-Coder-RCO-152E-Q3_K_M` (69.91 GiB, 45 layers: 34 KDA +
11 DSA, 42 MoE layers with **152 experts, top-8**, 3 dense, experts Q3_K).
All work on branch `fork-enhancements` (uncommitted), binary
`build/release/gufo`. Guarded benchmarks via the jev guard (`:8900`).

## 0. Goals

1. **Prefill:** reach 1000 tok/s at pp2048/4096/8192/16384 using GPU + NPU.
2. **Decode:** reach ~40 tok/s, "explore all possibilities", get MTP working.

Measured starting points: pp ~640–720 (with NPU, after this work) and
**18.4 tok/s** decode (plain, single stream).

## 1. Measured hardware ceilings (this box)

`tools/bench/gfx1151_peak.hip`:
**fp16 54.5 TFLOPS · int8 54.9 TOPS · DRAM read 209 GB/s · copy 193 GB/s**
(20 CUs @ 2.9 GHz, 128-bit LPDDR5).

## 2. Prefill

### 2.1 What was built (all still in the working tree)

* **NPU (XDNA2) offload for prefill** — `Executor::SetupNpu`, two 4096-row
  banks of NPU pair path (`kPrefillChunk = 4096`; larger chunks measured
  slower), shared-expert + shared-down NPU submission in `FfnBlock`
  (`NpuShexpQueue/Finish`, `NpuShDownQueue/Finish`) with encoder/decoder
  kernels, enabled with `GUFO_NPU_DIR=/home/majed/gufo/tools/npu`.
* **Stage profiler** — `GUFO_GLM53_PROFILE=1` prints 16 wall buckets per run
  (`hc, kda:qkv, dsa, ffn-dense, ffn-moe, dsa:proj/select/headgemm/attend/
  scores, kda:conv/recur/wo, moe:routed/shexp/router`) via `Mark()` in
  `executor.cpp/.hpp`.
* **Bucket-attribution bug fixed** — `kda:qkv` was being timed inside
  `kda:conv`: the "2.28 s conv" that several optimizations chased was
  actually the qkv projections. Conv was always ~0.15 s/token-scale.
* **Q8_0/f16 mirrors kept** for the W8A8 GEMM path (7.4 GiB) — this is the
  "Q8 keep" fix that stopped the 228 tok/s requantization trap (requantizing
  per call collapsed prefill to 228 tok/s).
* Earlier in the port (see `~/.claude/skills/port-model/LEARNINGS.md`):
  parallel O_DIRECT weight upload (`gufo::hip::WeightUpload`) cut model load
  from ~74 s to ~21 s.

### 2.2 Prefill results (NPU two-bank, quiet machine)

| test | tok/s |
|---|---|
| pp2048 | 640 |
| pp4096 | **719** |
| pp8192 | 711 |
| pp16384 | 702 |
| tg32 (decode, same run) | 18.39 |

**NPU output is byte-identical to the GPU path** (quality constraint held:
speed never cost quality).

### 2.3 Why 1000 tok/s was not reached — the roofline

Each 4096-token prefill pass costs **118–130 TFLOP** (MoE experts dominate
even at 8/152 activation). At the measured **54.5 TFLOPS fp16** ceiling:

* 1000 tok/s = 4.07 s per 4096-pass ⇒ **≥53% sustained on every stage at
  once** — not achievable end to end.
* Even the MoE GEMMs alone at 100% of peak only reach **~910 tok/s**.

So pp ~720 is at the practical roofline for this architecture on this die;
further gains need structural changes (chunked KDA scan, DSA attend rewrite),
deferred by decision.

### 2.4 Prefill experiments tried and reverted (measured)

* Tiled fp32 KDA conv: 0.45 s vs 0.15 s baseline — reverted.
* `KdaRecurKernel` float4 + lookahead prefetch: 632.1/631.8 vs pristine
  631.4/634.2 — neutral; `kda.hip.cpp` back at HEAD.
* Gate-tile threshold tuning, W8A8 wave64: neutral-or-worse — reverted.

## 3. Decode

### 3.1 The decode roofline, measured for real

Per decode token the model streams (computed exactly from the GGUF tensor
table, batch-1 = every weight matrix read once):

| part | GB/token |
|---|---|
| Routed experts (42 MoE × 8 slots × 3 × 2048×4096 Q3_K) | **3.633** |
| KDA q/k/v + f-gate + beta (34 layers) | 2.139 |
| KDA wo + output gates | 0.699 |
| DSA/MLA + indexer (11 layers) | 0.697 |
| Shared expert (42 × gate/up Q4_K + down Q6_K) | 0.636 |
| lm_head (Q6_K) | 0.520 |
| token embedding (Q4_K) | 0.357 |
| Dense FFN layers 0–2 | 0.294 |
| hc + norms + router | 0.248 |
| **total** | **9.22** |

209 GB/s ⇒ **44 ms/token = 22.7 tok/s absolute batch-1 ceiling**, ~19–20 at
85% efficiency. **40 tok/s single-stream is physically unreachable for this
quant.** (An earlier 7.2 GB/token / 29 tok/s estimate was wrong: v-projection
is Q6_K, and q_b/wo/embd/lm_head are bigger than assumed.)

Type mapping note (gufo vendored enum, `ds4_ggml_stubs.h`):
`Q8_0=8, Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14`.

### 3.2 Decode stage profile (`GUFO_GLM53_PROFILE=1`, 200 decode tokens)

| stage | ms/token | GB | GB/s | % of 209 |
|---|---|---|---|---|
| moe:routed | 20.5 | 3.63 | **177** | 85% ✅ |
| kda:qkv | 12.2 | 2.14 | **175** | 84% ✅ |
| kda:wo | 5.4 | 0.70 | 130 | 62% |
| moe:shexp | 5.1 | 0.64 | 126 | 60% |
| dsa (+dsa:proj) | 7.2 | 0.70 | 97 | 46% |
| ffn-dense | 1.6 | 0.29 | 185 | 88% ✅ |
| moe:router | 1.8 | 0.11 | 58 | 28% |
| hc | 5.3 | 0.14 | launch/ALU bound | — |
| kda:recur | 2.4 | ~0.3 state r/w | — | — |

The two big GEMV buckets are **already at 84–85% of DRAM peak** (the earlier
"moe:routed at 142 GB/s" number was a graph-reading error). Remaining
single-stream headroom is in the small stages ≈ 8–10 ms/token combined →
realistic quality-free target **~20–21 tok/s**.

Kernel view (rocprof, decode): ~2126 launches/token; hot kernels
`mul_mat_vec_q_moe<Q3_K,2>` (126/tok), `mul_mat_vec_q<Q4_K,1>` (243/tok),
`mul_mat_vec_q<Q8_0,1>` (247/tok), `KdaRecurKernel` (35/tok),
`DsaAttendKernel` (11/tok), `HcMixKernel<float>` (90/tok — single 1024-thread
block running the 19×Sinkhorn per call, ~1.7 ms/tok of pure overhead).
HIP graph capture is already wired for decode, so launch count is largely
amortized.

## 4. MTP speculative decoding — economics on a top-8-of-152 MoE

`--speculative mtp --mtp-model ~/models/glm53-mtp/mtp-Q5_K_M-fixed.gguf`
(Q5 sidecar, 6.3 GiB; the Q8 one is 9.3 GiB). The draft width is fixed at one
token (`kMtpWidth`): verifying n tokens shares the dense reads but multiplies
routed experts (+22 ms per extra token), so break-even acceptance is ~0.49 at
d=1 and ~0.89 at d=2.

| config | acceptance | tg (512 tok, t=0) | GTT |
|---|---|---|---|
| plain | - | 18.1 | 83.1 GiB |
| MTP d=1, Q5 sidecar | 0.70-0.75 | **20.3-21.4** | 90.1 GiB |
| MTP d=1, Q8 sidecar | 0.61-0.75 | 15-20 (stalls under memory pressure) | 92.3 GiB |
| MTP d=2-4 (old policy) | 0.29-0.31 | 12-16 | - |
| DFlash2 on RCO | 0.00-0.06 | 8.8-14 | 83.3 GiB |
| DFlash2 on AJ-IQ2_XXS (old model) | 0.56 | 25-29 | 94.8 GiB |

The "402 ms/step" mystery was a one-time first-step stall: lazy `SaveState`
allocations (now one slab) plus the kernel swapping out idle GTT buffers when
the 90+ GiB working set leaves no headroom next to the host services.
Keep `vm.swappiness` high and stop other services during runs.

## 5. Memory accounting (new: `GUFO_MEM_REPORT=1`)

GTT accounting added to `executor.cpp` (`Gtt` enum, `GttScope`/`GttCount`,
printed at end of `Executor::Create` and `CreateSession`):

| config | weights | Q8/f16 mirrors | scratch | session | GTT total |
|---|---|---|---|---|---|
| RCO alone | 69.9 | 7.4 | 5.6 | 0.2 | **83.1 GiB** |
| RCO + MTP Q8 sidecar | 78.5 | 7.4 | 6.1 | 0.2 | **92.3 GiB** |

All pinned GTT → the 70 GB model legitimately needs 90+ GB of the 120 GB with
page cache; runs fail whenever another big model is resident. Mirrors = W8A8
GEMM Q8 copies + NPU staging (the "Q8 keep" fix); scratch = 4096-row prefill
chunk (7 × 134 MB KDA arrays, `scores`+`qabs` 1.07 GB, `parts` 0.5 GB,
logits); chunk shrink 4096→1024 would save ~4 GiB for decode-only use.

## 6. Measurement infrastructure lessons

* **jev guard:** `POST /jev-guard/prepare` must carry the **exact command
  line** that will run (it watches processes matching the segment). A
  descriptive segment stops serve once and the supervisor restarts the
  ~100 GB flash-next serve **mid-run** — that OOM-killed two runs this
  session. Helper: `/tmp/opencode/run1.sh`. Guard log: `~/.gufo/guard.log`.
* `gufo prompt` prints `Generated ... tok/s` and `[Speculative]:` only with
  `-v`.
* **Q5_K_M MTP sidecar won't load:** its `blk.45.indexer_compressor_*`
  tensors are F16/Q6_K (`unsupported tensor type`) where Q8 keeps F32.
* rocprof adds ~30 µs/dispatch: decode kernel ms/launch numbers are inflated
  (×2 wall); use traces for ratios, the internal stage profile for absolute.
* Host load skews results (xmrig once cost 11% of pp); check
  `ps -eo pid,pcpu --sort=-pcpu` and loadavg before trusting 10% deltas.
  SIGTERM only, never `kill -9`/`pkill -f`; one GPU job at a time.
* Run env: `LD_LIBRARY_PATH=/opt/rocm/lib:/opt/xilinx/xrt/lib:/home/majed/npu-work/boostshim`,
  NPU adds `GUFO_NPU_DIR=/home/majed/gufo/tools/npu`.

## 7. State of the tree

Uncommitted on `fork-enhancements`: NPU two-bank prefill + stage profiler
(+bucket fix) + Q8 mirrors + GTT accounting + MTP (engine/executor/kernels)
+ prefill matrix pp2048 640 / pp4096 719 / pp8192 711 / pp16384 702 with
byte-identical NPU output. **`engine.cpp` currently carries the temporary
experiment `kPlainBelow = 0.0F // TEMP EXPERIMENT: always draft`** — restore
to `0.8F` (or a principled d=1 break-even ≈ 0.5) before committing.
`tools/jev/guard.py` working-tree diff is the user's — never stage.
`glm53.kda` test passes; `check-format.py` PASS.

## 8. Next steps (ordered)

1. **Explain d=1's 400 ms/step** (rocprof `-n 12` d=1 trace; look for idle
   gaps and non-graphed draft launches; keep draft logits on device / graph
   the draft / overlap). If fixed → d=1 MTP ≈ 21 tok/s expected.
2. **shexp overlap:** shared expert (0.64 GB @ 60% BW) is independent of the
   routed experts — second stream hides most of 5.1 ms/token.
3. **hc fusion** (90 HcMix + 92 HcSink launches/token for a 4×4 Sinkhorn —
   one warp can do it), **router** fp32 GEMV (58 GB/s), **dsa** decode
   (46% BW). Combined target ~20–21 tok/s plain.
4. Restore `kPlainBelow` to a principled value; wire `ForceDraftWidth` or an
   env knob for experiments.
5. If MTP overhead is unfixable: document d=1 MTP as not-usable on RCO and
   retest DFlash2 acceptance on RCO (needs > ~0.85 here; prior 0.389).
6. Deferred prefill work (only if the 1000 tok/s goal is revisited): chunked
   KDA scan, DSA attend rewrite — needed because pp is at the roofline.

## 9. Future plan: shared fused K-quant prefill kernel, no Q8 mirrors

Goal: drop the 7.4 GiB Q8 mirrors (GTT 83.6 → 76.9 GiB) at ≤ parity prefill,
with one shared MMQ library for every model (today two diverged vendored
copies: `qwen38_flash_next/kernels/rocm/mmq` for Flash-Next/Laguna/Flint and
`deepseek_v4_flash/kernels/rocm/mmq` for DeepSeek V4/GLM).

Measured (pp2048, interleaved x5): mirrors 640.5 vs native Q4_K/Q6_K MMQ
570.5 (-11%). Q4_K MMQ 1.85 s/482 launches, Q6_K 0.47 s, per-GEMM activation
re-quantization 182 ms (723 launches). `DS4_ROCM_WMMA_MMQ_Y` 32/128 worse
than 64. `GUFO_GLM53_NO_MIRROR=1` skips the K-quant mirrors (Q8_0 keeps its
free alias).

Steps:
1. Finish the uncommitted activation reuse: `ds4_mmq_q{4,6}_K_dense_preq`,
   `ds4_mmq_quantize_q8_1_mmq`, lazy per-layout (DS4 for Q4_K, D4 for Q6_K)
   cache on `Act::mmq` in `Executor::Proj`. Built, NOT yet benchmarked or
   checked for output equality. Expected to recover ~half of the 11%.
2. Tune the Q4_K tile loader / tile shape for the GLM dense shapes
   (4096x8192, 8192x4096, 4096x12288).
3. Move more of prefill to the NPU if GPU parity is out of reach (dense
   q/k/v already there; check whether NO_MIRROR breaks NPU staging).
4. Consolidate into `src/kernels/rocm/mmq` with one adapter exposing a generic
   `dense_gemm(type, ...)`; port Flash-Next first (superset, has tests),
   repoint Laguna/Flint/DeepSeek/GLM, delete the duplicate, rerun every
   model's operator tests.
5. Make native the default once it matches, remove the env switch.
6. Q3 DFlash2 drafter: train on the RCO model's own hidden states
   (`tools/qwen-flash/dflash-*.sh`); the Q2-trained drafter accepts ~0 on RCO.
7. Prefill 800 target needs chunked KDA scan + DSA attend work (pp is at the
   roofline otherwise).
