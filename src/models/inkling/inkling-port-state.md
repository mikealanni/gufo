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

## Open questions / next steps (skill default order)

1. CPU reference (`src/models/inkling/reference.cpp`) + config parser.
2. GPU engine stage 1 (plain decode coherent) reusing shared Flash-Next kernels.
3. Drafter: MTP or DFlash2 or DSpark; measure acceptance on real prose.
4. NPU prefill (sparse / two banks / all).
5. Serve runner + dispatch (`general.architecture` == "inkling2").
6. Bench integration.
7. Docs, regression, merge from clean worktree.
