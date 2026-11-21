# Inkling 2 Mini port state (port kickoff)

Source artifact: `unsloth/Inkling-2-Mini-GGUF` (`Inkling-2-Mini-UD-Q4_K_XL.gguf`,
17.63 GB, downloaded to `~/hfcache` via `hf download unsloth/Inkling-2-Mini-GGUF
--include Inkling-2-Mini-UD-Q4_K_XL.gguf`), on top of Thinking Machines
`thinkingmachines/Inkling-2-Mini` (Apache 2.0, 26.2B params / 3.4B active per pass,
looped 3x -> 5.4B effective active; 24 physical layers, middle core looped).

## Confirmed artifact read (GGUF, `general.architecture = "inkling2"`, version 3)

- `general.type = "model"`, name `Inkling-2-Mini`, size label `128x1.8B`.
- `inkling2.block_count = 48` slots, `inkling2.num_loops = 3`,
  `loop.core_start = 6`, `loop.core_end = 18`, `loop.gate_rank = 64`.
- Hidden/embedding `2304`, dense FFN `1152`, expert FF `1152`, expert_count `128`,
  expert_used `8`, expert_weights_scale `8.0`, routed expert gate/up and down are
  **NVFP4** with per-expert scales (`*_exps.scale` per expert).
- Attention: 24 Q heads / 8 KV heads (GQA), key/value length `128`,
  sliding window `1024`, SWA pattern True except every 6th slot (slots 5, 11, 17, ...)
  -> local-only layers; `attn_res.block_size = 6`; shortconv `l_cache = 4;
  qk_clip 22.63, relative_clip 8.0, relative_length 16, relative_extent 1024,
  log_scale_floor 8192, log_scale_alpha 0.1, final_logit_softcapping 30.0,
  situ_beta 4.0, situ_linear_beta 25.0, rms_eps ~1e-6, no RoPE (rope dim count 0).
- Context `1048576` (1M tokens), vocab unpadded 200058 / padded 201024.
- Tensors: 734 total; `blk.N.*` naming with the looped slots aliasing physical
  layers; loop mix tensors `loop_kv_res_query` / `loop_res_query` (+ Q/K/V/FFN
  control groups per upstream).
- Tokenizer: `tokenizer.ggml.model = "gpt2"`, `pre = "inkling2"`, BPE merges,
  eos/eot `200006`, add_bos/eos false, clean_spaces false, padding `199999`,
  chat_template embedded (6868 chars, effort + custom role/content tokens),
  sampling temp 1.0 / top_k 0 / top_p 1.0 / min_p 0.

## User targets for this port

- NPU prefill: sparse, or two banks, or all - to be resolved by measurement.
- Prefill (pp) target: ~2k tok/s on a 2k prompt (skill checklist baseline: Flint
  689 -> 4,045 with NPU).
- Decode (tg) target: 50+ tok/s with MTP, DFlash2 or DSpark if it works here.
- Weights: download `Inkling-2-Mini-UD-Q4_K_XL.gguf` (the Q4_K_XL the user picked;
  imatrix-calibrated, mean KLD vs UD-Q8_K_XL 0.0107, same top token 98.0%).

## Status snapshot (2026-11-21)

- User targets: pp ~3k tok/s with NPU, decode ~100 tok/s.
- Oracle route abandoned (upstream loop-msg query layout guessed; reemit scripts broken).
  Upstream graph source (`/tmp/opencode/llamacpp/src/models/inkling2{,-loop}.cpp`) used as the spec.
- CPU reference `reference.cpp` complete: full forward (attn_res block mixing, shortconv with
  dynamic taps on full layers, relative-bias + sink attention, NVFP4/Q8_0 SiTU MoE, loop
  passes with message control). Builds; `gufo_inkling_ref_check <gguf> <token...>` runs
  ~13 s/token, finite plausible logits. NOT yet validated against any reference.
- Reader now supports NVFP4 (ggml type 40). Slots 20-23 use Q8_0 experts with no `.scale`.
- Loop message queries: per core layer c, columns (attn, k, v, ffn) =
  (res[2c], kv[2c], kv[2c+1], res[2c+1]) - unverified guess.

## GPU engine stage 1 (2026-11-21, done)

- `gpu/engine.hip.cpp`: single-file HIP engine (own kernels, no shared Flash-Next kernels yet).
  Matches the CPU reference to ~4 digits on tokens 1000/2000/3000, batched prefill == token-by-token.
  Build: `cmake --preset gpu-test` then `--target gufo_inkling_gpu_check` (plain cmake, no nix on
  this box); run with `LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib`. `INKLING_BENCH=N` prints pp/tg.
- Measured: pp 89 tok/s (2k, naive per-item NVFP4 expert GEMM), tg 29 tok/s (graph-captured n=1).
- Decode profile (per token, 34 ms wall, 28 ms kernels, 2705 kernels): dense Q8 8.9 ms (~1.8 GB
  incl. 492 MB output.weight), NVFP4 experts 8.9 ms (~1.7 GB), Mix 2.4, Rms 1.6, F32 ctrl gemms 1.75.
  GEMMs already run near ~200 GB/s, so plain decode ceiling is ~55-60 tok/s; 100 needs fewer bytes
  (4-bit dense, check KL) and/or speculation (idea: self-draft with 1 loop pass instead of 3).
- Next: fuse Mix/Rms/ctrl kernels; WMMA routed experts + dense for prefill (3k target); serve runner.

## GPU prefill (2026-11-21)

- Added own WMMA kernels: routed NVFP4/Q8 expert gemm (sorted per-expert tiles), dense gemm for
  n>=32, flash attention with relative bias + sink. pp 2k: 89 -> 1295 tok/s; decode still 25-29.
- Prefill split at 2k (1.6 s): expert WMMA 0.57+0.27 s (~22 TFLOPs), dense F32/Q8 WMMA 0.09,
  flash attention 0.18, ~0.4 s in ~10 small kernels (Conv, Mix, Rms, Combine, Situ, Clip, Control).
- Tried, no gain: register prefetch of next K step, 128x64 tile with 256 threads, K step 64
  (slower, 1049), batch 2048 instead of 512. Gemm is neither dequant- nor wmma-bound alone
  (removing either gives only ~+17%), so it is sync/latency bound.
- FLOP budget: ~9.2 GFLOP/token -> 3k tok/s needs ~28 TFLOPs sustained; GPU-only is at ~12.
  Remaining levers: fuse the small kernels (-25% time), LDS double buffer / direct-to-register
  B fragments, NPU offload of dense + shared parts, then drafter for decode.

## Open questions / next steps (skill default order)

1. CPU reference (`src/models/inkling/reference.cpp`) + config parser.
2. GPU engine stage 1 (plain decode coherent) reusing shared Flash-Next kernels.
3. Drafter: MTP or DFlash2 or DSpark; measure acceptance on real prose.
4. NPU prefill (sparse / two banks / all).
5. Serve runner + dispatch (`general.architecture` == "inkling2").
6. Bench integration.
7. Docs, regression, merge from clean worktree.
