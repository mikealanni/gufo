# Inkling 2 Mini port state

User-facing guide: [docs/models/inkling](../../../docs/models/inkling/README.md). This file records
the artifact facts, engine design and what was learned, for the next person touching the port.

## Artifact (GGUF, `general.architecture = "inkling2"`, version 3)

- 48 slots over 24 physical layers: `num_loops = 3`, `loop.core_start = 6`, `loop.core_end = 18`,
  `loop.gate_rank = 64`. Hidden 2304, dense FFN 1152, expert FFN 1152, 128 experts, 8 used,
  `expert_weights_scale 8.0`. Routed gate/up and down are NVFP4 with per-expert scales; slots 20-23
  are Q8_0 with no scale tensor.
- Attention: 24 Q heads, 8 KV heads, head dim 128, sliding window 1024 except every sixth slot,
  relative bias (length 16, extent 1024) plus a zero sink logit, no RoPE, short convolution `l_cache 4`
  (dynamic taps on full-attention slots), `qk_clip 22.63`, `relative_clip 8.0`, final softcap 30,
  SiTU `beta 4.0 / linear_beta 25.0`, `rms_eps ~1e-6`.
- Vocabulary 200058 (padded 201024), tokenizer `gpt2` / pre `inkling2`, eos 200006, chat template
  with an effort header and custom role/content tokens.
- Loop message queries: per core layer c, columns (attn, k, v, ffn) = (res[2c], kv[2c], kv[2c+1],
  res[2c+1]).

## Layout

- `runtime/` config + GGUF reader glue; `reference.cpp` CPU reference; `chat_template.cpp`.
- `gpu/engine.hip.cpp` single-file HIP engine: n=1 graph-captured decode with device-side position,
  few-row verify path (`Session::AppendAll`, `KeepRows`), WMMA prefill (routed NVFP4/Q8 expert GEMM,
  dense GEMM, relative-bias flash attention with sink), block-sparse attention, two-bank NPU prefill,
  DSpark drafter (`Model::LoadDrafter`, `Session::Draft`).
- `gpu_check.cpp` harness: `--text`, `--verify`, `--vtime`, `--draft`, `--windows`, `--dwindows`,
  `INKLING_BENCH=<n>`.
- Serve: `InklingTextRunner` in `src/cli/serve/inference_backend.cpp` (`DecodeStep` drafts, verifies
  and keeps the matching prefix).
- Build: plain `cmake --preset gpu-test` (or `release` with `-DGUFO_ENABLE_NPU=ON` for the NPU); run with
  `LD_LIBRARY_PATH` set to the ROCm libraries.

## Decode design

- Plain step: about 1.9 GB of Q8 dense weights (including the 492 MB output matrix) and about 1.7 GB of
  NVFP4 experts per token, both at the memory-bandwidth floor (about 223 GB/s). 29.7 ms per step.
- Verify of k rows touches the experts of every row (six rows: about 28 distinct per layer), so cost
  grows about 7 ms per row; expert reads are about 60% of a verify step.
- DSpark drafter (arch `dflash`, three dense layers, block 8, Markov head): four taps from the target
  (outputs of virtual layers 4, 22, 40 and the final hidden state after `output_norm`), fused by
  `dspark_fuse` (softmax rows 0-2 per drafter layer, row 3 plus `dspark_norm` for the carry feature)
  and projected into a drafter KV (two sliding layers, one full). Slot 0 is the target-embedded anchor
  token, slots 1-7 are `mask_embd`. The proposal vocabulary is the top 32,768 rows of `markov_w1` by
  L2 norm, quantised to NVFP4 (a first-N-token-ids shortlist cannot propose most chat and code tokens).
- Draft width is gated per step by the top-1 probability of each slot (>= 0.75); slots after the first
  weak one are skipped on the device; acceptance is measured on chat-templated output with the text
  channel forced, not on raw text.
- Deep context: few-row steps pack all rows and the three heads of a KV group into one 4-wave flash tile
  (K/V read once per group); sliding-window passes are split in 256-key parts; at >= 64k sessions block
  selection is per KV group.

## Prefill

- Routed experts use the shared NVFP4 GEMM with the expert scale folded in and F16 outputs; dense Q8 uses
  the engine's WMMA GEMM; elementwise work is fused (norm, half conversion, convolution-add).
- NPU: shared runtime-M/K block-float GEMM takes q/gate/k/v and the attention output per bank-layer; two
  token banks hide it behind the other bank's expert GEMMs (NPU traffic slows the expert GEMMs about 15%).
  Only the instruction streams for 4096xKx1024 were generated; the xclbin is shared.

## Learned (keep)

- Plain decode and verify are bandwidth-bound; kernel-count work pays about 2 microseconds per launch.
- Verify graph capture, expert-grouped GEMM, multi-row chunked attention and next-tile register prefetch
  were slower or equal.
- The model thinks at effort 0; benchmark code and JSON with the text channel forced.
- Greedy speculation can differ from plain decode at near ties because the verify path uses f32 batch
  kernels while decode uses int8 activations.
- Open: other-model regression for the shared-kernel edits (NVFP4 weight type, routed empty-tile return,
  tokenizer pattern), snapshots/disk cache, tools, reasoning split.
