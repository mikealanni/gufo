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

## Serve (2026-11-21)

- `gufo serve llm -m <gguf>` dispatches on `inkling2` (runner cloned from Laguna: no snapshot/fork,
  thinking effort header, sampling preset temp 1 / top_p 1 / top_k 0). `~/run-gufo.sh` kind `inkling`
  (backup `~/run-gufo.sh.bak-inkling`). curl chat completion works: reasoning then answer, 30.6 tok/s decode,
  prompt 103 tok/s on a 28-token prompt.
- Known gaps: control tokens (`<|content_thinking|>`, `<|end_message|>`) are returned as plain text,
  no reasoning-split; no snapshots/disk cache; tools unsupported; KV is full FP16 for all 48 slots
  (393 KB/token), so default context 16k.

## Serve (2026-11-21)

- `gufo serve llm -m Inkling-2-Mini-UD-Q4_K_XL.gguf -c 8192 -p 8911` works end to end (InklingTextRunner,
  no snapshot/fork yet). Chat template in `chat_template.cpp` (effort header + `<|message_model|>` opener).
  First request: model thinks, then answers "The capital of France is Paris."; 31 tok/s decode via serve.
- Open: control tokens (`<|content_thinking|>`, `<|end_message|>`, `<|message_model|>`, `<|content_text|>`) leak into
  the content and generation does not stop at the final `<|end_message|>`; needs reasoning split + stop handling.
  No `~/run-gufo.sh` KIND yet; no snapshots (RAM/disk cache), no tools.

## Depth and sparse attention (2026-11-21, real prose, GPU only)

| depth | dense pp | dense tg | sparse(top256) pp | sparse tg |
| 2k | 1289 | 24 | - | - |
| 32k | 996 | 14.6 | 957 | 17.0 |
| 64k | 809 | 10.8 | ~930 | ~17 |
| 128k | 545 | 7.2 | 905 | 16.8 |

- Sparse = sink 2 + local 16 + top-256 older 64-key blocks (Quest bound of the tile's mean query), only
  the 8 full-attention slots; exact until ~17.7k tokens; automatic when session context >= 64k
  (INKLING_SPARSE=0/1 forces). SWA slots use 3072-row ring buffers. Decode attention is split-K flash.
- Needle (passphrase) retrieval at 128k: 50% and 90% depth found by dense and sparse; 10% depth fails
  in BOTH (model/prompt limit, not sparse). Top 128 missed the 50% needle; 512 gave no gain and was slower.
- Shared Flash-Next `DenseF16Gemm` for Q8 dense = same speed as the own WMMA dense (A/B, real prose).
- GPU-only f16 ceiling: ~9.2 GFLOP/token at ~24 TFLOPs (best shared-kernel level) = ~2.6k tok/s at 2k;
  3k needs the NPU and/or int4 WMMA (iu4 skill) for experts. DSpark drafter (3 dense layers, block 8,
  markov head, arch `dflash`) downloaded to the hf cache; upstream spec: upstream src/models/dflash.cpp.

## Prefill push (2026-11-21, real prose, GPU only)

| ctx | pp tok/s | notes |
| 2k | 1492 | was 1289; shared NVFP4 routed gemm, fused rms/half/conv-add, WMMA rel-bias flash |
| 8k | 1393 | |
| 32k | 1107 | sparse top-256 automatic at ctx >= 64k sessions |
| 128k | 1035 | dense was 545 |

- Profile at 2k (busy 1.37 s): routed experts 0.50 s (shared kernel, new NVFP4 type), dense WMMA 0.31 s,
  flash ~0.10 s, ~0.38 s in bandwidth-bound elementwise passes (each [2k x 2304] f32 pass ~0.1 ms).
- NPU: runtime M/K xclbin is reused; only instruction streams were generated
  (tools/npu/runtime_mk/insts_2048x2304x1024 and insts_2048x3072x1024). Dense offload cannot carry the
  model: NPU ~6x slower per GEMM at 2k tokens (LEARNINGS, Flint), so expected gain <= ~5%.
- Reference point from the user: another Strix Halo engine runs this model at ~50 tg with its own drafter
  head and ~1,000 tok/s prompts (19.6 GB NVFP4/MXFP8/BF16 weights).
- Open: DSpark drafter (tg), F16 activation storage / more fusion (elementwise ~28%), dense gemm tile.

## NPU two-bank prefill (2026-11-21)

- NPU (shared runtime-M/K block-float GEMM, single pass, ~1.3% RMS per projection) takes q/gate/k/v
  (3+3+1+1 slices of 1024 rows, K=2304) and the attention output (3 slices, K=3072 program) per bank-layer.
  Weights encoded from the original Q8_0 at load. Probe: 3 slices at M=2048 = 1.6 ms (same as GPU).
- Two token banks (own scratch, interleaved per layer: A.qkv-begin | B.moe | A.qkv-end A.attn A.out |
  B.front B.qkv-begin | A.moe ...) so one bank's NPU work hides behind the other's MoE. Banks of 2048 rows
  cost the GPU path ~6% (expert tile fill) and the NPU gives ~9% back; 4096-row banks (M=4096 program,
  batches 3072-8192) remove the penalty. Batches < 3072 stay on the plain GPU path.
- Results (real prose, release-npu build, INKLING_NPU=1 INKLING_NPU_M=4096): 4k 1697, 8k ~1786, 32k 1656,
  64k ~1590, 128k 1584 tok/s; 2k stays ~1500 (GPU path). `gufo serve` (NPU env via ~/run-gufo.sh kind
  inkling, 8192 prefill chunks): 18.7k-token prompt 1631 tok/s, coherent.
- Quality: KL(GPU||NPU) of the last-token distribution over 12 prose windows of 4096 tokens: mean 0.0026,
  max 0.0149, top-1 agree 12/12. NPU memory traffic slows the expert GEMMs ~15% (why the net gain is ~9-10%).
- Env: INKLING_NPU=1, GUFO_NPU_DIR=<repo>/tools/npu, INKLING_NPU_M=2048|4096, INKLING_NPU_PARTS bitmask
  (1 q, 2 gate, 4 k, 8 v), INKLING_NPU_OUT=0 disables the output projection. Build: cmake -B build/release-npu
  --preset release -DGUFO_ENABLE_NPU=ON -DGUFO_XRT_INCLUDE_DIR=.../xrt/include.
- Not done: other-model regression for the shared-kernel edits (NVFP4 weight type, routed empty-tile
  return); merge.

## the reference runtime Inkling port as a reference (2026-11-22) and what it gave us

- `an external package`: GPU-only the reference runtime package (programs only, same Q4_K_XL GGUF),
  prefill 1.45k (1k) / 1.65-1.70k (2k) / 1.69-1.74k (8k) / 1.61-1.64k (16k) / 1.24-1.26k (32k), decode 46 tok/s
  with DSpark depth 1 (58-70% accepted), upstream ROCm 273 / 15 tok/s. Their kernel list (its program) showed:
  NVFP4 expert GEMMs with the expert scale folded in and F16 outputs (`g_moe_ggu_nvfp4_fused_sc_f16o`),
  Q8 activation planes out of the norms (int8 GEMMs), many fused small kernels, tile selection by token count.
- Taken: NVFP4 routed GEMM with folded expert scale and F16 output (scaled gate/up max 20, down max 21.6, far
  inside F16): GPU-only 4k 1643 -> 1694, 8k 1614 -> 1663. With the NPU: 8k 1864, 32k 1718, 128k 1647 tok/s.
- Not taken yet: fused gate+up+SiTU epilogue (needs split gate/up tensors), Q8 activation planes / W8A8 dense.

## DSpark drafter (not implemented) - spec recovered from the the reference runtime program (its program)

- Drafter GGUF `dspark-Inkling-2-Mini-Q8_0.gguf` (arch `dflash`, 54 tensors): 3 dense layers (hidden 2304, 24 heads /
  8 kv, ffn 9216, q/k norm 128), per-layer `dflare_norm`, `attn_k_ctx` / `attn_v_ctx` (+ `attn_k_ctx_norm`),
  `dspark_fuse` [4,4] (fuses 4 target tap features), `dspark_norm`, `mask_embd`, Markov RNN head
  (`markov_w1` [256, vocab], `joint_proj` [5120 -> 3x768?], `dspark_hidden_out`), block size 8, token embeddings and the
  output head are the target's (the reference runtime ships a Q4_0 shortlisted head).
- Flow per round: (1) warm: target tap features (`target_layers` of the looped model) -> 4x4 fuse -> per drafter layer
  `dflare_norm` -> K/V ctx projections -> drafter SWA KV cache; (2) propose: embed anchor token with target embeddings +
  mask embeddings for the other block slots, 3 decoder layers (non-causal over the block + ctx KV), head + Markov RNN
  chain per slot; (3) verify the draft with the target (n = 1 + depth rows) and accept the matching prefix. the reference runtime uses
  depth 1. Needs: tap capture in the engine, an n=2..8 verify path (batched decode kernels + graph per width),
  drafter engine, acceptance logic and serve integration.

## DSpark milestone 1 (2026-11-22): tap capture, verify path, keep-rows

- `Session::AppendAll(1..8 tokens)` keeps the logits of every row (`Logits(row)`); argmax of every row matches
  sequential single-token steps (max |dlogit| 0.4-1.4, int8-activation decode path vs f32 batch path).
- `Session::KeepRows(k)` rewinds a verification batch to its first k rows: the short-convolution histories keep a
  window of the batch inputs per channel (3 + 8 floats) and a fix-up kernel restores the history for k rows; cache rows
  past k are overwritten by the next append. Checked: keep j of k then +1 token matches j sequential steps + 1.
- Tap capture (INKLING_DRAFT=1): hidden state outputs of virtual layers 4, 22, 40, 47 (from drafter `target_layers`
  [5, 23, 41, 48]; the off-by-one convention is to be confirmed by drafter acceptance) stored as F16 per position
  (`feat[pos][tap][2304]`); works in the single, banked and graph paths.
- Measured (eager, no graph): verify 2 rows 42.5 ms, 4 rows 58 ms, 8 rows 89 ms vs 35.4 ms per plain token. With ~60%
  draft acceptance at depth 1 and a ~3 ms draft: ~35 tok/s eager, ~41 with a captured verify-2 graph; 50 needs a
  cheaper n=2..4 verify (batched Q8/NVFP4 kernels for few rows) and/or depth 2-3 with higher acceptance.
- Drafter recipe recovered from the reference runtime's `its program`: per drafter layer l, ctx feature = rmsnorm(sum_k softmax(
  dspark_fuse[l])_k * tap_k) * dflare_norm_l -> K = attn_k_ctx(.) (rmsnorm attn_k_ctx_norm, rope theta 1e6 neox),
  V = attn_v_ctx(.), stored in the drafter KV (layers 0 and 2 sliding 1024, layer 1 full); row 3 of dspark_fuse (+
  dspark_norm) gives the carry feature for the Markov RNN head. Propose: block slots = target-embedded anchor token +
  mask_embd for the rest, 3 layers (ffn 9216), output_norm, target lm head (the reference runtime uses a Q4_0 shortlist head) + Markov RNN
  (markov_w1 [256, vocab], joint_proj [5120 -> 768] over [state, w1[prev], 2x2304 features], dspark_hidden_out); the
  artifact has no markov_w2, so the vocab bias is probably tied to markov_w1 (to confirm in g_dsp_markov_*).
- Drafter engine (2026-11-22, `Session::Draft`, `Model::LoadDrafter`, `gpu_check --draft`): warm (fuse + K/V ctx
  into a drafter KV, rings for the two SWA layers), 3 layers over 8 slots, Markov RNN head over the first 65536 vocab
  rows (greedy). Tap definition settled by acceptance on prose: virtual layer outputs 4, 22, 40, 47 (mean accepted 1.72
  of 7 vs 0.3-1.5 for neighbours). First-slot acceptance ~64%, matching the reference runtime's depth-1 58-70%. Draft ~9.5 ms (7
  sequential Q8 head matvecs over 65536 rows dominate) vs verify-8 93 ms.
- Speculation verdict (2026-11-22): break-even. Draft 3.3 ms with an 8192-row head shortlist (acceptance same as 65536),
  verify-2 ~40 ms vs plain 32 ms (two tokens touch ~14 distinct experts), 1.4-2.3 tokens/step over widths 1-4 gives
  29-36 tok/s vs 31 plain. Not wired into serve. Only a cheaper plain step or a smarter verify (shared experts across
  rows, higher acceptance) changes this. Fusing the tiny per-layer GEMVs into one launch did not help (reverted):
  the step is bandwidth-bound, not launch-bound. The built-in GGUF has no draft/MTP tensors; DSpark is external only.
  (greedy first), captured verify graphs, serve integration.
- Acceptance fix (2026-11-22): the proposal vocabulary must be the top 65536 rows of markov_w1 by L2 norm (the drafter's
  own reduced vocabulary), not the first N token ids; most chat/code tokens have ids above 65536, so a first-N shortlist
  could never propose them (mean accepted 0.68 -> 2.15 of 3 on templated code, 32768 rows by norm, draft ~4 ms). Fourth tap
  is the final hidden state after output_norm (the reference runtime taps [5,23,41,48] = outputs of virtual layers 4, 22, 40 and the
  normed final state). Measured with templated chat prompts, not raw text: code harness 55 tok/s at width 3 (verify-4
  ~52 ms, plain 32); serve (thinking text) code 44.8 tok/s, JSON 35.9 tok/s, 52-60% of drafted tokens accepted.
  Verify cost per extra row is ~7 ms, so width 3 beats 5 and 7.
- Decode attention (2026-11-22): n=1 decode used the 128-query prefill flash kernel; a dedicated split-key kernel
  (DecodeAttn, 256 keys per block, 16 lanes per key row) raised plain decode 23.0 -> 28.2 tok/s at 4k, 19.6 -> 23.3 at
  16k. Full-attention layers still ~1 ms each at 16k (~60 GB/s), room left.
- Serve with thinking off (text channel forced after the model role token; the model otherwise thinks even at effort 0):
  code 67 tok/s, JSON 62 tok/s, 76-82% of drafted tokens accepted, draft width gated per step by top-1 confidence >= 0.75
  (up to 7). 37k context: decode 38.5 tok/s, prefill 1433. Tried and reverted: multi-row DecodeAttn for verify (slower
  than the prefill flash tile at every depth), verify graph capture (no change: GPU-side launch gaps are the same).
  Greedy argmax runs on the device per verify row; logits are downloaded lazily.
- GPU-only pp (bench, 2026-11-22): 2k 1652, 4k 1745, 8k 1715, 16k 1670, 32k 1536 (sparse 1619), 128k ~1035. At 32k the
  busy time is routed experts 5.4 s, flash attention 5.0 s, dense Q8 WMMA 4.2 s of 22.8 s.
- Verify cost (2026-11-22): verify-4 ~52 ms = routed NVFP4 experts 24 ms (4 rows touch ~22.7 distinct experts per
  layer, 5 GB at ~205 GB/s, same rate as the plain step's 8.4 ms) + dense Q8 9 ms (same bytes as plain) + ~10 ms of
  small kernels. The expert term is bandwidth-bound by distinct experts and cannot shrink; fusing the loop control step
  (6 launches -> 2) saved only ~0.7 ms. Each extra verify row costs ~7 ms, so width 3 is the optimum.

## Open questions / next steps (skill default order)

1. CPU reference (`src/models/inkling/reference.cpp`) + config parser.
2. GPU engine stage 1 (plain decode coherent) reusing shared Flash-Next kernels.
3. Drafter: MTP or DFlash2 or DSpark; measure acceptance on real prose.
4. NPU prefill (sparse / two banks / all).
5. Serve runner + dispatch (`general.architecture` == "inkling2").
6. Bench integration.
7. Docs, regression, merge from clean worktree.

- Deep-context decode (2026-11-22): packed few-row attention (all rows x the 3 heads of one KV group in one 4-wave flash
  tile, K/V read once per group), sliding-window verify split into 256-key parts, drafter attention with slots across
  lanes, and per-KV-group sparse block selection (Quest bound over the group's heads and rows) with the same packed
  tile: 62k verify-6 94 -> 73 ms, plain step 50 -> 36 ms (short context 33). Needle test at 10/50/90% in 63k:
  dense, per-head sparse and grouped sparse give identical answers. Draft slots after the first low-confidence one are
  skipped on the device. Serve decode at 63k: 54.2 / 52.1 / 54.2 tok/s (37k 56.9; short code 67.7, JSON 64-68).
  Tried and reverted: register prefetch of the next K/V tile (slower, register pressure), multi-row chunk kernel.
- Expert layers 20-23 (Q8_0 in the GGUF, 1.083 GB each) are converted to NVFP4 at load (nearest E2M1 with a 3-candidate
  UE4M3 scale search per 16 elements; INKLING_EXPERT_NVFP4=0 disables). The reference runtime's "fitted" final-layer banks
  are 574.7 MB each = one NVFP4 expert layer, so this matches what they ship. KL(orig||converted) over 12 mixed
  prose/code windows of 2048 tokens: mean 0.0013 nats, max 0.0048, top-1 12/12 (their figure 0.0095). Gains: 6-row
  verify 65.8 -> 63.0 ms, plain step -0.6 ms, GPU pp 2k 1652 -> 1692, 4k 1745 -> 1768. Serve (greedy, thinking off):
  code 70.7, JSON 68.9, 37k 59.3, 63k 55.4/53.9/55.4 tok/s.
- Tried and reverted: control step restructured to read weights once for all rows (verify 5 rows 61.6 -> 66.2 ms: 8 blocks
  is too little parallelism).
- Expert pruning for few-row steps (opt-in, INKLING_PRUNE=<share>, default off): routed experts whose normalised routing
  weight share is below the threshold are skipped on the device (n < 32 only; their output rows are zeroed). Measured
  over 256 decode-path positions, KL vs the unpruned engine / top-1 agreement / 6-row verify / plain step:
  0.03: 0.0016 / 98.8% / 63.5 / 31.2 ms; 0.06: 0.0067 / 96.9% / 60.8 / 30.3; 0.08 (renormalised) 0.026 / 95%; 0.10: 0.040 /
  94.5% / 48.9 / 27.7; 0.12 renormalised: 0.165 / 86% / 42.7 / 26.5 (baseline 63.5 / 31.5). Renormalising the kept
  weights made KL worse at every threshold. Drafter head is NVFP4 (32768 ranked rows, draft 5.4 -> 4.3 ms).
- Lossless stack status (2026-11-22, serve, greedy, thinking off): code 74.6, JSON 71.1 tok/s; 63k 53.7/60.9/55.1; 37k 54-59.
  Vectorised loop-control kernel (-1.6 ms per step), quantised-activation reuse across projections (-0.2 ms). Profile of a
  verify step: experts 60% (at ~223 GB/s, the bandwidth floor), dense Q8 ~14% (bandwidth), the rest ~25% is many 4-15 us
  kernels; each launch removed is worth only ~2 us. Remaining lossless items are 0.2-0.5 ms each; JSON 100 needs fewer
  expert bytes per row (pruning/lower-bit, both lossy).
