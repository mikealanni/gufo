# Inkling 2 Mini: prefill attention on the NPU

Task for an agent that builds an XDNA2 (AIE2P) prefill attention kernel for Inkling and wires it
into the Inkling engine. Everything needed is below or pointed to by path. Read it fully before
touching hardware.

## 1. Why

At 8K prompts on this machine (Strix Halo gfx1151 + XDNA2), GPU time per 4,096-token chunk is about
1.5 to 1.9 s. The `FlashRel` attention kernel is **23 to 27%** of that (about 700 ms per 8K pass,
8.5 ms per call at depth, 48 calls per chunk). The NPU is idle most of the time (about 32% busy,
only the q/gate/k/v/out projections run on it) and has 60%+ spare capacity. Moving attention to the
NPU removes up to ~400 ms of GPU time per chunk.

Expected effect if attention is fully hidden behind GPU work: roughly +15 to 20% prefill speed at
4K to 16K (default engine 8K with NPU is 2,382 tok/s; the opt-in int4 experts on branch `perf/inkling-pp3k` reach 2,822; both short of 3,000).
Attention must stay exact up to the same tolerance as the current NPU projections (see section 8).

Nothing in this task is lossy by design except the block-float rounding the NPU kernels already use.

## 2. Inkling attention, exactly

Source of truth: `src/models/inkling/reference.cpp` lines 436 to 498 (CPU, double accumulation) and
`FlashRel` in `src/models/inkling/gpu/engine.hip.cpp` (GPU, what the NPU path must match).

Model constants (from the GGUF): hidden 2,304; `num_heads` nh = 24; `num_kv` nkv = 8 (so gh = 3 query
heads per kv head); `head_dim` hd = 128; sliding window W = 1,024; relative basis length nrel = 16;
relative extent E = 1,024; 24 physical layers, the middle 12 run 3 times = **48 attention visits**
per token: **40 sliding-window visits (window 1,024) and 8 full-attention visits** (every sixth
layer). No RoPE.

For one query token i at absolute position p, head h, kv group g = h / 3, over keys j:

```
s_ij   = scale * dot(q_ih, k_jg)  +  bias(p - j) . r_ih  -  sink_ih        scale = 1 / sqrt(128)
bias(d) = table[d] (16 values) if 0 <= d < E else 0                       dot over the 16 basis values
keys   : j in [max(0, p - W + 1), p] for sliding-window visits, j in [0, p] for full visits
softmax: over { s_ij for those j }  plus ONE extra entry with value exactly 0 (the "sink";
         it has no value vector, it only adds exp(0 - m) to the denominator)
out_ih = sum_j softmax_ij * v_jg                                          (128 values)
```

- `q`, `r`, `sink` arrive already computed by earlier GPU kernels: q after per-head RMS, `r` after
  clipping (`attn_rel_clip`), sink = `attn_sink_proj . x`. On **full-attention visits** q and r are
  both multiplied by `tau = 1 + alpha * ln(max(1, (p+1)/floor))` before the kernel (log-scale
  attention; the engine already does this on the GPU, the NPU kernel does not see it).
- The result `out` (fp32, 24 x 128 per token, before the output gate) goes back to the GPU, which
  multiplies it by `sigmoid(attn_gate . x)` and runs the output projection. Keep that on the GPU.
- Sinks are per (token, head) and are **subtracted** from every score of that row.

## 3. Shapes and work

- A chunk is up to 4,096 query tokens (bank size); two banks pipeline. Context grows chunk by chunk
  (8K, 16K, 32K prompts; sparse block attention on full visits starts at 65,536 tokens and stays on
  the GPU, out of scope).
- Per 4K chunk the whole model does about 4.7 TFLOP of attention (8 full visits at ~6K average
  keys: ~2.5 TFLOP; 40 window visits: ~2.2 TFLOP). The NPU runs bfp16 GEMM-like work at 13 to 17
  TFLOPS effective in this system (measured with `gufo_inkling_npu_expert_probe`), so budget
  roughly 300 to 450 ms of NPU time per chunk for the whole model. This must overlap GPU work.
- KV memory per token per layer: K = 8 x 128 fp16 = 2 KB, V the same. Full visits at 64K tokens
  hold 8 layers x 4 KB x 64K = 2 GB (the engine holds these in device memory as fp16).

## 4. Layouts in the engine today

All in `engine.hip.cpp`, kernel `FlashRel` (search `AttnArgs`) and `LayerAttn`:

| Tensor | Layout | Notes |
| --- | --- | --- |
| `q` | fp32 `[token][24][128]` | after norm and tau; FlashRel converts to fp16 |
| `r` | fp32 `[token][24][16]` | relative basis coefficients (clipped, tau scaled on full visits) |
| `sink` | fp32 `[token][24]` | subtracted from every score of the row |
| K cache | fp16 `[ring or pos][8][128]` | window visits use a ring of `ring` rows (index `j % ring`) |
| V cache | fp16, 64-key blocks, transposed: `[g][vblock][128][64]` | `vc[((g*vblocks + (row>>6))*128 + d)*64 + (row & 63)]` |
| bias table | fp16 `[E + 1][16]` | `attn_rel_bias.weight.h`; row E is the "outside" row |
| out | fp32 `[token][24 x 128]` | then GPU sigmoid gate (`SigmoidMulH`, or fused in the FlashRel epilogue) |

The NPU needs these in its own block-float stream format, produced by GPU kernels into pinned
host buffers (see `NpuEncodeActivations`, `NpuEncodeWeights`, `NpuGemm` in
`src/models/qwen38_flash_next/kernels/rocm/`). K/V encoding should be done once per token and kept
(a persistent bfp16 copy of K and V^T per layer, about 1.1 B per value), not rebuilt every chunk.

## 5. Existing NPU attention kernel to extend (do not start from scratch)

Branch `wp11/atb-prefill` of the worktree `~/gufo-wp10` (local, unpushed). Files under
`tools/npu/decode_int5/`: `attn_kernel.h`, `attnp/attn_prefill.h`, `attnp/mm_attnp.cc`,
`attnp/n32_attnp.py` (program generator, env `ATTNP_NT`, `ATTNP_MC`), `attnp/sbtest_attnp.cpp`
(probe), host side `AttnPrefill` in `src/npu/runtime/runtime.hpp` (passes, double buffering,
`RunMany`). What it already does for another model (LFM3, measured 0.955 ms per 128-token pass at
4K context): causal prefill attention, **sliding window via a ring table (`a_win`, `a_tab`)**,
bfp16 K/V streamed in 16-token chunks (2 x K then V^T, 4,608 B per chunk), exact exp2 for short
contexts and hardware exp2 beyond, static softmax bound with a `bad_rows` fallback,
g = 2 packing of two token blocks. 32 cores = 8 columns x 4 rows; a core owns one 64-row query
block (16 tokens x 4 query heads) of one kv head.

Hard-wired constraints that must change for Inkling (documented in `attn_kernel.h`):

| Existing | Inkling needs |
| --- | --- |
| 4 kv heads (one per core row) | **8 kv heads** (two groups per pass, or remap rows) |
| 4 query heads per kv head | **3** (48 real rows of 64: 25% padding, or pack tokens like the g = 2 variant) |
| 16 heads total | 24 |
| causal / window mask only | plus **relative bias term** and **sink entry** |
| softmax bound from the host | must stay valid with bias and sink (see 6) |

## 6. The two new pieces

**Relative bias.** For a 16-token query block and a 16-key chunk the needed distances are 31
consecutive values. Compute `Zd = R[rows x 16] x Bias[16 x 31 distances]` with the block-float
matrix unit (the GPU does this with one tiny WMMA per 16 distances), then each row takes a 16-wide
slice of `Zd` shifted by its row index (unaligned vector load or shuffle) and adds it to the
scores before exp. Distances >= E (1,024) contribute 0, and window visits have W = E = 1,024, so
inside a window every distance is valid; only full visits cross E. Do **not** materialize a
`[rows x keys]` bias in DRAM (about 200 MB per layer; far too much traffic).

**Sink and softmax safety.** Score = `scale*q.k + bias - sink`; the softmax includes a
constant 0 entry. The GPU kernel starts each row with running max m = 0 and denominator l = 1.
The existing NPU kernel uses a static per-row bound `M_row` supplied by the host
(`p = exp2(s - M_row)`) so partials add with no rescaling, and returns "bad rows" when the mass is
unsafe so the host recomputes them. For Inkling choose one: (a) keep the static bound, derive
`M_row >= max(0, bound(s))` on the GPU from `|q|`, `max |k|`, `|r|` and the bias table norm, and
send rows whose mass underflows back to `FlashRel`; or (b) implement online softmax in the
kernel (more cores, more program memory; AIE program memory is 16 KB, which is the binding limit).
Measure how often (a) falls back before choosing.

## 7. Integration into the engine

- Opt-in env `INKLING_NPU_ATTN=1`, default off until it passes the gates. When the NPU, the program
  files or the shape are unavailable, fall back to `FlashRel` with no change in output.
- Hook in `LayerAttn` (the `FlashRel` launch for `n >= kWmmaMinTokens`, sliding and full visits
  with exact attention). Copy the pattern of `NpuQkvBegin` / `NpuQkvEnd` (submit from a host
  callback, ticket, wait on the stream) and the two-bank schedule `BodyBanks` so one bank's NPU
  attention overlaps the other bank's expert GEMMs.
- Do not change `Impl::AllocBytes` accounting: buffers created on first use must use `AllocLazy`
  (the server compares measured session bytes with a probe claim; see commit `395b2145`).
- Instruction streams are generated offline per (rows, context) shape; one xclbin serves many.
  Generation takes seconds with `gen_insts.sh`-style tooling (see `tools/npu/runtime_mk/README.md`
  for the pattern); a new xclbin needs the AIE toolchain (`~/npu-work/ex_env.sh`).

## 8. Gates (all must pass; measure, do not estimate)

Correctness, in order:
1. Kernel vs the CPU reference (`reference.cpp` formulas, fp64) on random data and on real
   q/r/sink/K/V dumps of one window and one full layer: relative rms error <= 1e-2 per layer
   (the GPU FlashRel is ~1e-3; bfp16 K/V costs a few e-3).
2. Layer outputs inside the engine vs `FlashRel`, same inputs, all 48 visits of one chunk.
3. End to end: logits KL against the default engine on the 48-window set (`gufo_inkling_gpu_check <gguf> --windows <text> 48 1024 <out.bin>`, compared with
   `tools/bench/inkling_kl.py`; noise floor 1.6e-3 to 4e-3) must stay <= 4e-3
   mean with top-1 >= 46/48 (default engine vs itself is 0 and 48/48). Do not use 6 windows: too
   noisy to rank anything.
4. HumanEval (`tools/bench/humaneval.py`, 164 tasks) is a sanity check only; 89.6% is the default
   engine's score without thinking.

Speed (single machine, `gufo_inkling_gpu_check`, `INKLING_BENCH=<n> INKLING_BENCH_REPEAT=3`,
warm numbers, interleave A/B runs): report 4K, 8K and 16K, with and without the NPU projections.
A change that wins at 4K and loses at 8K is not done.

## 9. Rules for working on this machine

- NPU binaries must be named `gufo*`; run NPU jobs through `~/npu-lock.sh <command>`; the fabric
  clock must be held (`sudo ~/bench-mode.sh on`; the user does this); never `kill -9` a GPU or NPU
  process (SIGTERM; reboot if stuck); never `pkill -f` with a pattern that matches your own shell.
- **Check for a running `gufo` or `gufo serve` first** (`ps aux | grep gufo`). The user's own
  server shares the same memory; running a test next to it caused a memory-pressure storm
  (kswapd at 90% on three cores) and a 27-minute stall. If it is up, wait.
- Disk: `/home` is nearly full. Put builds, caches and artifacts on `/mnt/raid0` (about 40 GB free).
- No third-party project names in repo files or commit messages. Conventional Commits, one line,
  `Co-Authored-By` trailer as in the log. Do not push; the user pushes.
- Build: `cmake --build build/release-npu --target gufo_inkling_gpu_check --parallel 8`
  (about 4 minutes after touching `engine.hip.cpp`; run long builds in the background).
- Measurement helpers: the `--windows` mode of `gufo_inkling_gpu_check` writes last-token logits;
  `INKLING_ALLOC_LOG=1` logs session allocation changes; per-kernel time with
  `rocprofv3 --kernel-trace` (does not work with the NPU run, use GPU-only for kernel tables).

## 10. Milestones

1. **M1, one pass on the device:** one layer, one 128-token pass, window visit, bias + sink,
   checked against the CPU reference. Probe tool in the style of `npu_expert_probe.cpp`.
2. **M2, full coverage:** 8 kv heads x 3 q heads, full visits (distances above E), tau-scaled
   inputs, ring windows across a chunk boundary, bad-row fallback rate measured on real data.
3. **M3, engine path:** `INKLING_NPU_ATTN=1`, KV encode kernels, two-bank overlap, fallback, gates
   2 and 3 above.
4. **M4, speed:** tune dispatch count (about 0.15 to 0.3 us per instruction-stream op and ~3 ms per
   context switch; one hardware context for GEMM and attention if possible), report the table.

Open questions for the agent to settle and write down: row packing for 3 query heads; where the
bias product runs (matrix unit vs vector unit); static bound vs online softmax; whether K/V bfp16
copies can be written by the existing GPU encode kernels or need a new one.
