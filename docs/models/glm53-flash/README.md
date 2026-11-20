# GLM-5.3 Flash

GLM-5.3-Flash is a 321B-parameter mixture-of-experts model (about 14B active
per token): 45 hyper-connection layers (34 KDA linear-attention, 11 MLA layers
with the native DSA indexer), MoE from layer 3 with sigmoid top-8 routing.
The engine runs the trunk with the DSA selection intact (attention reads the
selected pooled keys plus the tail, never the full context), the KDA
recurrence per head, and the hyper-connection Sinkhorn mixing on the GPU.

## Artifacts

| Checkpoint | Size | Notes |
| --- | --- | --- |
| AJ-IQ2_XXS (45 blk) | 81.35 GiB | quality reference for this port; 288 experts |
| RCO-152E Q3_K_M (coder) | 69.91 GiB | 152 experts; native Q3_K WMMA routed GEMMs |

Both are sharded GGUFs; point `-m` at shard 00001.

## Run

```sh
gufo prompt -m .../GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf -p "..." -n 128 -t 0
gufo bench  -m .../GLM-5.3-Flash-Coder-RCO-152E-Q3_K_M-00001-of-00002.gguf -p 4096,8192 -n 64
gufo serve  llm -m <shard 00001> --served-model-name <name> -c 65536
```

Drafter: `--speculative dflash2 --dflash-model <GLM-5.3-Flash-DFlash2 GGUF>`
(plus lossless prompt-lookup drafts). An MTP sidecar exists for the coder
checkpoint (`--speculative mtp --mtp-model ~/models/glm53-mtp/mtp-Q5_K_M-fixed.gguf`);
it is deterministic and coherent but is still evaluated against DFlash2 before
being advertised.

## NPU prefill

With `GUFO_NPU_DIR=tools/npu` (binary built `-DGUFO_ENABLE_NPU=ON`), the KDA
q/k/v projections and the shared-expert gate/up run on the XDNA2 NPU in two
2048-token banks that alternate layer by layer through the runtime-M/K program
(`tools/npu/runtime_mk`). Every prompt of 4096 tokens or more pairs, at any
context depth; smaller prompts and decode run on the GPU at full speed. The
engine prints `NPU: KDA q/k/v of 34 layers` when engaged and `NPU off: ...`
(including a `MemAvailable < 32 GB` gate) otherwise — capture stderr in NPU
measurements.

| Env | Effect |
| --- | --- |
| `GUFO_GLM53_NPU_M=4096` | wide banks (equal at 8192, no gain elsewhere; default 2048) |
| `GUFO_NPU_WHOLE=1` | also run the shared-expert down projection on the NPU; measured zero net gain, off by default |
| `GUFO_GLM53_PROFILE=1` | stage timers, printed at exit (compare shares between runs, not absolute seconds) |
| `GUFO_GLM53_F32_RESIDUAL=1` | disable the F16 residual stream of wide batches |

## Speed (Strix Halo, warm, interleaved, tok/s)

RCO-152E Q3_K_M, `-p <len> -n 1 -r 2`; depth rows are `pp4096 @ -d <len>`:

| Prefill | GPU only | GPU + NPU | Δ |
| --- | --- | --- | --- |
| `-p 2048` | 632 | 636 | GPU path |
| `-p 4096` | 623 | 714 | +14.6% |
| `-p 8192` | 617 | 706 | +14.4% |
| `-p 16384` | 610 | 699 | +14.6% |
| `-d 32768` | 584 | 665 | +13.9% |
| `-d 65536` | 560 | 633 | +13.0% |
| `-d 131072` | 511 | 571 | +11.8% |

tg128 ≈ 17–18 tok/s (flat with NPU; decode is GPU-side). AJ-IQ2_XXS: pp4096
616 GPU vs 689 NPU, pp8192 613 vs ~690, pp2048 unchanged (previously a -1.3%
regression under the old pair path).

## Numerics and quality

- NPU two-bank prefill on RCO: greedy output byte-identical to the GPU-only
  path over 64 tokens on real prose, and deterministic across runs; `serve`
  first/repeat/multi-turn pass with the pair warm (RAM cache hits confirmed).
- AJ-IQ2_XXS two-bank: logits cosine 0.994 against the GPU-only chunked path,
  same top-1; GPU-only prefill matches sequential decode at cosine 0.989.
- `GUFO_NPU_WHOLE=1` (sh_down on NPU): text byte-identical, no speed change.
- IQ2_XXS capability: HumanEval 157/164 (reasoning off); 50/75 on the DS4
  capability set with reasoning.
- DSA selection is the model's semantics: all pooled keys are always scored;
  no approximate indexer skip is applied or needed — attention is already
  pool-bounded (top 512 pools + tail) at every depth.

## Status

Engine (trunk, sampler, chat template), DFlash2 drafter, serve runner with
RAM/disk cache, NPU two-bank prefill: working, measurements above. MTP graft:
functional (deterministic, sampled acceptance 0.28–0.31) but not yet a net
win at short depth on the 152-expert checkpoint; under evaluation.
