# Raon-1-mini GPU port: process log

Date: 2026-11-26. Worktree: `/home/majed/gufo-raon`, branch `port/raon1` from
`fork-enhancements` at `7c51d474`. GPU target: gfx1151. Do not touch `/home/majed/gufo`
while another agent uses `gufo serve --port 8901`.

## Order followed

1. CPU reference in `src/models/raon1/reference.cpp` matched the patched reference
   runtime (oracle) before GPU work. Default comparison uses `GUFO_RAON1_NO_EFFORT=1` because
   the oracle ignores reasoning-effort tensors.
2. GPU executor and custom HIP kernels produced logits; validate first-token logits
   against CPU and oracle before CLI/serve integration.
3. After every kernel change: rebuild focused targets, dump logits, compare rms/max/corr
   and argmax against CPU, then bench the same shape. Keep the change only if parity
   holds and speed improves beyond noise.
4. Keep stage dumps while debugging; unset them for clean parity runs. Remove debug
   env hooks once parity is clean.
5. Work only in the worktree; download only into `/mnt/raid0/user_cache/huggingface`.

## Verified state

CPU reference vs oracle: full 25-layer greedy top-1 matches at both tested positions;
logit rms `0.20046954 / 0.050334502`, correlation `0.996087948 / 0.999486259`.

GPU vs CPU reference after the HC and RDA fixes (`gpu_check 100 200`):

| position | rms | max | corr | argmax |
| --- | --- | --- | --- | --- |
| 0 | 0.160658 | 0.811049 | 0.997447 | 166461 / 166461 |
| 1 | 0.053865 | 0.245774 | 0.999414 | 145 / 145 |

Bench progression at `-p 32 -n 8 -r 3`:

| stage | pp32 | tg8 |
| --- | --- | --- |
| baseline | 31.16-31.22 | 24.61-24.83 |
| HC warp reduction (`kHcThreads=640`) | 35.47 | 28.25 |
| RDA decay index + warp recurrence | 45.68 | 36.53 |
| current restored state | 45.48 | 36.34 |

Latest production-shape result: `pp32 45.48`, `tg128 35.31` tok/s. `pp2048` needs a
clean rerun.

## Kernel findings

- HC selection uses 640 threads: one warp per mix, 17 active warps for 17 mixes, with
  `WarpSum` across the wave. Every lane must call `WarpSum` before the lane-0 store.
  Calling it only from lane 0 makes each wave return lane 0's own value and silently
  breaks parity. This was the cause of the regression after the split experiment.
- The HC split (`HcMixKernel` + `HcFinishKernel`), both with lane-0 `WarpSum` and with
  a shared-memory `BlockSum32`, was slower (`pp32 43.68`, `tg8 34.25`) and was reverted.
  Keep the single integrated `HcSelectKernel`.
- RDA recurrence divides the summed squares by `hd`; the GPU now matches CPU
  `rda_scan_out`. The decay vector must be indexed as `decay + h * hd`, not from head 0.
  Warp-per-row recurrence (`kRdaThreads=128`, 4 items per lane) removed
  `RdaRecurKernel` from the top-eight profile.
- Q5_0 routed-expert down requires Q8_0 activations and high bits at `j + 16`, not
  the scalar dequant `j + 12`.
- F16 block-tensor conversion was tried and rejected: parity was similar but speed
  dropped (`pp32 29.15`, `tg8 19.05`). Keep only as a debugging dead end until removed.

## Current profile and next step

Top kernels from the last profile (total 61.7 ms): Q6_K matvec `18.316 ms/832`,
HC select `9.718 ms/100`, Q8_0 matvec `7.870 ms/686`, `quantize_q8_1` `6.198 ms/1666`,
Q4_K MoE `5.270 ms/96`, Q8_0 MoE `2.273 ms/24`, Q5_0 MoE `1.637 ms/24`.

All Raon quantized matvecs pass `n=1`. GLM53's DS4 wrapper launches one row per block
on RDNA3_5 (`grid=(m,1,1)`, `block=(32,1,1)`). An additive candidate already in the
tree is `qfn_mmq_dense_vec` from Flash-Next: it supports Q6_K, uses 8 rows per block
(`row_waves=8`, `block=256`), and reuses canonical `block_q8_1` scratch. Validate
numerics before retaining. Gate/up fusion is not a drop-in: Raon uses a custom
`tanh(g/beta)*beta*sigmoid(g)` gate, not SwiGLU.

## Commands

```sh
M=/mnt/raid0/user_cache/huggingface/hub/models--KRAFTON--Raon-1-mini-gguf/snapshots/84afe5c89be174639c1855aa0c7528ed34ec3d63/raon-1-mini-Q4_K_M.gguf

cmake --build build/gpu-test --target gufo_raon1_gpu_check gufo -j 16
cmake --build build/cpu --target gufo_raon1_ref_check -j 16

LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib GUFO_GPU_DUMP_LOGITS=/tmp/opencode/gpu_logits.bin \
  ./build/gpu-test/src/models/raon1/gufo_raon1_gpu_check $M 100 200
GUFO_RAON1_NO_EFFORT=1 GUFO_DUMP_LOGITS=/tmp/opencode/ref_logits.bin \
  ./build/cpu/src/models/raon1/gufo_raon1_ref_check $M 100 200

LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib ./build/gpu-test/gufo bench -m $M -p 32 -n 8 -r 3
```

Debug env hooks still present pending final cleanup: `GUFO_RAON1_GPU_TRACE`,
`GUFO_RAON1_GPU_DUMP`, `GUFO_RAON1_DUMP_FULL`, `GUFO_RAON1_GPU_PROF`,
`GUFO_RAON1_GPU_LAYERS`, `GUFO_RAON1_GPU_F16`, `GUFO_RAON1_GPU_F16_TRACE`.

## Batched forward (2026-11-26)

`Executor::EvaluateBatch(session, tokens[<=8])` runs the trunk for several tokens at
once; `Evaluate` calls it with one token. Dense projections use `Matvec` with n columns
(mmvq, n <= 8), routed experts use `MatvecExpertsTokens` in groups of 4 tokens (the
ids path fails at n >= 5), RDA conv/decay/recurrence and MLA attention loop per token.
Two kernel bugs fixed on the way: `HcApply` ignored the per-token `block` offset and
`ArgmaxRows` ignored the row offset.

Parity at 12 tokens, batch 4 and 8 vs sequential: rms 0.042, max 0.21, corr 0.9996,
same argmax (GPU vs CPU reference was corr 0.9994). `pp32` 45 -> 80 tok/s, `pp512` 81,
`tg32` 36 (unchanged).

Tensor types: attn q/k/v Q6_K (18 RDA layers), experts gate/up Q4_K, down alternating
Q8_0/Q5_0, shared expert gate/up Q6_K and down Q8_0, MLA k_b Q8_0, v_b Q6_K.

Next for 3k pp (see the port-model skill): F16-activation GEMMs (`DenseF16Gemm` covers
Q6_K/Q8_0/Q4_K), `RoutedF16Gemm` for Q4_K gate/up and Q8_0 down (re-encode Q5_0 down to
Q8_0 at load), a fused multi-token RDA scan kernel, batched MLA attention and k_b/v_b.
Decode: HIP graph or fusion (per layer 1.2 ms vs about 0.45 ms by bandwidth), then the
DFlash2 drafter (`dflash-raon-1-mini-Q8_0.gguf`) using the GLM53 Verify/tap-capture
executor as the template.

## Prefill path (2026-11-26, same day)

`pp2048` 27.5 -> 1430 tok/s, `tg` 36 -> 40; chunks of 1024 tokens (`kMaxBatch`;
512 gave 1000, 2048 was worse). Parity vs sequential n=1 on a 40-token batch: corr
0.9997, same argmax; a real 2.5k-token prompt summarizes coherently.

| step | pp2048 |
| --- | --- |
| sequential Evaluate | 27 |
| batched matvec n<=8 | 80 |
| mmq Gemm n>8, MLA k_b/v_b via `MatvecBatched` | 86 |
| routed WMMA experts (`RoutedF16Gemm` Q4_K gate/up, Q8_0 down, Q5_0 down re-encoded) | 115 |
| non-absorbed WMMA MLA attention (K/V expanded from the latent cache by F16 GEMM) | 381 |
| fused multi-token RDA kernel (state in registers, 256 threads, reduce-scatter) | 720 |
| tiled F32 GEMM for BF16/F32 weights (router, ssm gates) | 1068 |
| F16 copies of Q4_K/Q6_K/Q8_0 dense weights through `DenseHalfWeightGemm` (3.2 GiB) | 1365 |
| mHC mixes as a multi-token dot kernel + `HcSelectPre` | 1430 |

Findings: the Q5_0 -> Q8_0 expert re-encode is lossless (Q5_0 values are 5-bit
integers times the block scale). The MoE ids path of `MatvecExpertsTokens` fails
above 4 tokens. A lane-0-per-row recurrence with a WarpSum per row was latency
bound (about 18 us per token); 16 rows per warp with a 16-value reduce-scatter and
`readlane` broadcasts cut the shuffle count from 320 to 124 per token. 32-row
variants spill registers (VGPR 256). Benchmarks on this box vary 10-30% with host
load (load average above 40 from other agents); compare interleaved.

Remaining GPU time at 2048 tokens (about 1.5 s): dense F16 GEMMs 460 ms, RDA 274,
routed experts 255, HcApply 136, router/ssm F32 GEMM 71. Not yet done: tg, DFlash2,
NPU offload, depth (8k+) tuning of the K/V expansion.

## Decode and DFlash2 (2026-11-26)

Plain decode `tg` 36 -> 48 tok/s: mHC mixes as a chunked 20-block dot kernel plus
`HcSelectPre` (HcSelect was 4.9 ms per token, single block per token), split-K
multi-token MLA attention (`MlaAttendSplit`, was one block per head scoring one key
at a time: 345 us per call at 300 keys), small-n F32 GEMM for the router and ssm
gates. The step is now bandwidth bound (about 3.5 GB of Q6_K/Q8_0/Q4_K per token at
roughly 200 GB/s).

DFlash2 (`dflash-raon-1-mini-Q8_0.gguf`, block 8, taps 2/7/12/17/22) runs through the
shared drafter executor. Raon taps are the full four-stream residual (10240 per
tap, `fc.weight` 51200 -> 2560), so the drafter config gained `target_hidden_size`
(metadata `dflash.target_hidden_size`, default = hidden) in `dflash_weights`,
`model.cpp`, `executor.cpp` and `dflash.hpp`; Qwen/GLM drafters are unchanged. The
tap is the layer's output (`GUFO_RAON1_TAP_INPUT=1` selects the input instead).

Verification chain of k <= 8 tokens goes through `EvaluateBatch`. The RDA recurrent
state cannot be rewound, so `SaveState` copies conv history + state before the chain,
`SetRecording` keeps each layer's operands, and `Rollback(keep)` restores the state
and replays the first `keep` tokens with `RdaPrefill` (about 0.6 ms). Attention
caches need no repair. The drafter runs on its own stream, so the trunk stream must
be synchronized before `InjectTargetContextDevice` (a missing sync made acceptance
non-deterministic: 74 vs 113 cycles for the same prompt).

Greedy CSV-function prompt, 200 tokens, `gufo prompt --speculative dflash2`:
33 tok/s (CLI plain) -> 62.7 tok/s, 74 cycles, 126 of 234 drafts accepted. Text
matches plain decoding until a near-tie flip at character 436 (batched verify
numerics), then stays coherent. Per cycle: draft 6.5 ms, verify 38 ms at k=5 (plain
step 21 ms), accept+inject 3 ms.

Next: verify is launch-bound (about 1300 small kernels) and MoE-byte-bound (k tokens
touch about 3.5x the experts); HIP graph per k, fused small kernels, drafter head
over a reduced vocabulary, serve runner `DecodeStep`.

## Serve and status (2026-11-26, end of session)

`gufo serve llm -m <trunk> --speculative dflash2 --dflash-model <drafter>` works: the
runner's `DecodeStep` calls `Session::DecodeStep` (history lookup plus DFlash2, greedy
and sampled verification). A greedy 200-token request reports 74 draft rounds and 126
of 234 drafts accepted, identical to the CLI. Verify-time MLA `k_b`/`v_b` projections
are batched over tokens (`MatvecBatchedTokens`), and the residual streams ping-pong
instead of copying.

Numbers (noisy, see above): `pp2048` 1370 (quiet) / 965 (loaded), `pp8192` 1300 (loaded),
plain `tg` 48, DFlash2 greedy 62-64 (46 when the box is loaded). Targets 3k pp and 100 tg
are not reached.

Not done: snapshot/fork/disk cache for the runner, `gufo bench --speculative`, HIP graph
for verify, drafter head over a reduced vocabulary, depth (32k+) measurements, NPU
offload, regression runs of the other models after the shared drafter/matvec edits
(both additive: `target_hidden_size` defaults to the drafter hidden size,
`MatvecBatchedTokens` is new).

## Quiet-box measurements (xmrig stopped)

`pp2048` 1668, `pp8192` 1428, plain `tg` 52. DFlash2 greedy, 300 tokens, 5 drafts, thinking off/on:
CSV function 70.5/66.9, binary search tree 95.4/56.1, printing-press essay 51.5/51.9, JSON books
97.8/98.3, TCP explainer 50.7/54.4 tok/s. Verify cost is at its MoE-byte floor (plain 19.5 ms,
k=3 27.9 ms, k=4 30.6 ms: about 3.4 ms per extra token for the extra experts); the drafter costs
about 6 ms (Q8_0 trunk 3.4 ms + Q6_K head 2.7 ms). Speed therefore tracks acceptance: code and JSON
about 95, prose about 51 (break-even with plain decode). `HcApply` as one block per token with
float4 columns: pp2048 1589 -> 1668. Remaining prefill GPU time at 2048 tokens (about 1.3 s): dense
F16 GEMMs 430 ms, RDA 282, routed experts 252, HcApply 120, F32 GEMM 73.

## Prefill to 3k (2026-11-26, xmrig stopped)

`pp2048` 1668 -> 3017 tok/s, `pp4096` 2866, `pp8192` 2460, `pp16384` 2031 (3 banks from 2560 tokens, else 2).
DFlash2 greedy 300 tokens, 5 drafts, thinking off: binary search tree 106, JSON books 102, printing-press
essay 55 tok/s; plain `tg` 53.7.

| step | pp2048 |
| --- | --- |
| RDA with one state row per thread (128 registers, no cross-lane reduction, `sched_barrier` against LDS hoisting) | 1668 -> 2006 |
| device-side routed tile maps (no per-layer host sync) | neutral, but removes the sync that blocked overlap |
| `DenseHalfWeightGemmWide`: `<256,128,2,8,1,1>` plan (m = 2560: row-tile group 2) for 1024+ token batches | 2012 -> 2350 |
| two or three pipelined banks (own stream, scratch and host thread; bank k trails bank k-1 by a layer) | 2350 -> 2640-3030 |
| fused Q4_K gate/up + SiTU (`RoutedQ4KSituGemmF16`, 128-row tiles), tiled 4x4 F32 GEMM | +3% |

Findings:
- The shared dense plan hit 16 TFLOPS on K = 2560/5120 projections and 31 on K = 1088. Plans with eight row
  waves per weight fragment give 37-39 TFLOPS on the same shapes (micro-benchmark in `gpu_check`
  with `GUFO_GEMM_BENCH=1`). It is a new export; the existing plans for other models are untouched.
- Byte pitches that are multiples of 1-4 KiB (K = 1024, 2048, 4096) are 3x slower than 1088/2560.
- The old per-layer host sync (`ExpertCounts` readback to build tiles) cost about 15% by itself; with it gone
  `Session::Sync` must synchronize the stream before returning (the bench measured only enqueue time).
- Pipelined banks need a layer-skew event chain, `thread_local` bank selection and a non-blocking stream;
  drafter taps are captured per bank and injected in order. Contexts past 32768 tokens fall back to one bank.
- Rows of the prompt farther than the drafter window (2048) from the end of a `Sync` are skipped, not injected.
- Depth: K/V of every cached position is re-expanded per chunk (blocks of 2048 rows), so pp falls to 2031 at
  16k. Next: persist expanded K/V or an absorbed WMMA kernel, and heads-per-block decode attention
  (decode at 12k depth is 30 tok/s because every head re-reads the latents).

Tried and dropped (2026-11-26): 64-key tiles with DPP reductions in the WMMA prefill attention kernel
(pp 1765 -> 1690 at 12k depth, 808 -> 789 at 30k); 128-row Q8_0 tiles for the routed down projection (+0.4%);
more than three banks (pp2048 falls as chunks shrink below ~700 tokens).

## NPU banks and sparse long context (2026-11-26/27)

Build the NPU tree with `cmake --preset gpu-test -B build/gpu-npu -DGUFO_ENABLE_NPU=ON -DGUFO_XRT_INCLUDE_DIR=/usr/local/include`
and run with `GUFO_NPU_DIR=<repo>/tools/npu` (existing xclbin and `insts_1024x2560x1024_128x64x128.txt`, nothing rebuilt).
`GUFO_RAON1_NO_NPU=1` turns it off.

- The RDA layers' q/k/v/g projections (4 x 5 slices of 1024 outputs, K = 2560) run on the NPU for 1024-token chunks
  (`Executor::NpuQkvg`). pp4096: 2929 -> 3007 (2 banks), pp2048 2790 -> 2855. A single bank is slower with the NPU (2212 vs
  ~2700) because the bank waits; the other bank's GPU work hides the wait.
- `NpuGemm::Submit`/`Wait` use HIP host callbacks, which share one thread: with two banks a blocked `Wait` stalled the other
  bank's `Submit` and deadlocked. New additive `SubmitHost`/`WaitHost` run on the calling thread after the stream is
  synchronized.
- Pipelined prefill now runs for every group of `banks x 1024` tokens (before, only a prompt's last group). Two banks beat three
  at 2048-4096 tokens.
- Opt-in block-sparse attention for the 7 MLA layers (`GUFO_RAON1_SPARSE=k,window,min_depth`, default 24,1536,12288 when set):
  sink 256 + local window + top-k 64-row blocks by a Quest bound over 8 query-token groups; K/V are expanded only for the
  selected keys. Approximate. Needle test (90k tokens): exact finds the code; k=24 misses it, k=48 and up find it with some
  wording damage. pp4096 at 64k depth: 600 exact -> 2337 (k=48) / 2494 (k=32); tg at 64k 47.
- Remaining NPU candidates: attn_output (K = 5120 as two K halves with accumulate), shared-expert gate/up (N = 1088), MLA
  projections; each saves about 0.7 ms of GPU dense per chunk-layer but costs encode/decode kernels (NpuDecodeOutput was 44 us
  per slice).

### Whole-model NPU offload, measured (pp4096, 2 banks, NPU + GPU)

`GUFO_RAON1_NPU_PARTS` adds projections beyond the RDA q/k/v/g (which are always on): `out` (attn_output, K = 5120 as two
K halves with accumulate), `gate` (MLA attn_gate), `shared` (shared-expert gate/up, N = 1088 as 1024 + 64 padded to 128 rows).

| NPU parts | pp4096 |
| --- | --- |
| none (GPU only, 2 banks) | 2929 |
| q/k/v/g | 2980-3015 |
| q/k/v/g + gate | 3006 |
| q/k/v/g + shared | 2991 |
| q/k/v/g + out | 2877 |
| all (`out,gate,shared`) | 2518 |

One 1024 x 2560 x 1024 run takes 0.25 ms (20 runs = 5 ms per layer-chunk), the NPU is a single in-order queue, and each linear
blocks its bank's host thread (stream sync + `WaitHost`). With everything offloaded the NPU becomes the bottleneck and the
blocking waits sit on the critical path (attn_output, whose result feeds the next op directly, is the worst). The GPU saving
per offloaded GEMM is also eaten by `NpuEncodeActivations` / `NpuDecodeOutput` (44 us per 1024-wide slice, F32 writes the
GPU GEMM would have made anyway). Open: stream-ordered waits without the shared callback thread (one waiter per bank that
records an event), decoding the bfp stream inside the consumer kernel, and a dedicated NPU share (about 5 ms per layer-chunk).

### NPU with asynchronous waits (2026-11-27)

Host-thread waits replaced by a per-bank ring of helper threads: the encode kernels record an event, the helper waits for it,
submits (`SubmitHost`), waits for the NPU and raises a pinned-host flag; the stream stalls on a one-wave spin kernel
(`WaitFlag`), so the bank's host thread keeps enqueuing (the RDA gate/beta/f_a/f_b projections run while the q/k/v/g runs are
on the NPU, the shared-expert gate/up run during the routed experts). Interleaved pp4096 (2 banks): GPU only 2867-2907,
q/k/v/g 3035-3066, + shared 3125-3171, + MLA gate 3103-3144, + attn_output 2954-3015 (it has nothing to overlap with).
Default parts are now q/k/v/g + gate + shared (`GUFO_RAON1_NPU_PARTS=` empty keeps only q/k/v/g, `...=out,gate,shared` adds
attn_output). pp2048 3290, pp4096 3109, pp8192 2651, pp16384 2176 (about +9-10% over GPU only). The dense GEMMs are only about
a third of the GPU time here and already run at 37 TFLOPS, so the offloadable share is smaller than on Flash-Next.

### Fused F16 copies (2026-11-27)

`RmsNormRows` can write the F16 copy of its output (`y16`), kept as a per-bank hint (attn_in -> `a16`, ffn_in -> `f16`);
the dense F16 GEMMs, the NPU activation encoders and the routed experts read it instead of running their own `ToHalf`
(about 4 conversions per layer gone). Load-time host dequantization moved to the GPU (`DequantRowsToHalf`,
`EmbedTokensQuant`): load CPU 1450% -> 160%, load 4.5 s -> 2.4 s. Blocking device sync (`GUFO_RAON1_SPIN=1` restores spinning).

Interleaved, two banks: pp2048 GPU only 3193 -> default NPU (q/k/v/g + gate + shared) 3442; pp4096 3049 -> 3240; pp8192 2659 -> 2714;
pp16384 2193 -> 2245; plain tg 53.5. Earlier GPU-only reference (before the fusion) was about 3030 at pp2048.
Flat at depth with sparse k=32,window=1024: pp2048 @ 4k 2475, @ 64k 2613, @ 128k 2604; tg 51 / 47.6 / 46.8.
The 258000-deep run did not finish in 14 minutes and was stopped.

## 2026-11-27: serve at depth, sparse attention, concurrency, loader

What changed, in the order found:

1. **Decode attention at depth.** Verifying a draft re-ran the per-key absorbed MLA kernel once per
   row, so every extra verified token cost its full attention again (about 15 ms per row at 10K
   context) and speculation gained nothing at depth. The WMMA kernel stages each 16-key latent tile
   once and shares it across all rows (`MlaSplitWmmaKernel`): plain decode at 26K 35 to 46.6 tok/s,
   a 3-row verify 61 to 34 ms. It rounds q, k, p and v to F16, so greedy text diverges from the
   F32 kernel after a few hundred characters (`GUFO_RAON1_MLA_WMMA=0` restores the old kernel).
2. **Thinking turns slowed down.** At temperature 1 the sampler fully sorted the 260K-token
   vocabulary whenever the top 256 held under `top_p` of the mass, which high-entropy text does
   often; the decode thread saturated one core while the GPU idled. The nucleus now grows 8x per
   round against the exact normalizer (same result on 1,080 configurations): high-entropy prose at
   10K depth 26 to 35 tok/s.
3. **Sparse attention broke tool use.** With 256 protected tokens an agent replay picked the wrong
   tool in 4 of 8 samples at 46K tokens because the tool definitions sit in the first 6K tokens.
   Protected prefix of 4,096 or more: 0 of 8. Prefill keeps 256 (a larger prefix costs prefill speed
   in proportion to its size), decode uses 8,192. When the prefix plus the local window already
   cover the context, attention is exact. Needle retrieval did not catch this.
4. **Two requests at once.** `Bank` now owns the rollback records, the recording flag and the
   logit-row count, `stream()` follows the calling thread's bank, and the runner declares a
   width-2 batched plan that runs each request on its own bank from its own thread.
5. **Weights stayed mapped.** The loader copied quantized tensors from the GGUF mapping, which kept
   18.4 GB faulted into the process for its whole life. Reclaim stalled on those pages (memory
   pressure 25 to 30% of wall time), and shutdown took minutes. `hip::TensorUpload` plus
   `madvise(DONTNEED)` on host-read tensors fixed it.
6. **Reasoning effort.** Effort now maps to a thinking budget so the effort the jev plugin picks is
   effective; the launcher default for this model is uncapped.

Tried and reverted (interleaved A/B, keep only if it wins its row):

- tiled recurrent prefill scan, syncs once per 16 tokens: +1.4% at 4K, -4.8% at 8K;
- 16 tokens per hyper-connection dot block: -1.8%;
- stricter drafting threshold (0.65 against 0.45): 27 against 33 tok/s;
- choosing plain or drafted steps by measured tokens per millisecond: -8% on prose, no gain on code.

A prefill change is only kept after at least three interleaved runs per side: separate runs minutes
apart showed a false +5 to 7%.
