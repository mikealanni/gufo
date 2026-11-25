# Inkling 2 Mini

Mixture-of-experts text model (26.4B parameters, 3.4B active per pass) on gfx1151.
24 physical layers whose middle twelve run three times, so a token makes 48 layer
passes (5.4B effective active parameters). Hidden size 2,304; 128 experts with 8
routed per token (expert FFN 1,152); 24 query heads over 8 KV heads of 128; sliding
window 1,024 on every layer except every sixth (the eight full-attention passes);
relative-position bias with a sink logit instead of RoPE; short convolutions; loop
message control between passes. The tested weights are
`Inkling-2-Mini-UD-Q4_K_XL.gguf` (17.6 GB: NVFP4 routed experts, Q8_0 dense and
attention). The optional drafter is `dspark-Inkling-2-Mini-Q8_0.gguf`.

## Run

```sh
gufo prompt -m Inkling-2-Mini-UD-Q4_K_XL.gguf -f prompt.txt -n 128 -t 0
gufo serve llm -m Inkling-2-Mini-UD-Q4_K_XL.gguf -c 16384 \
  --speculative dspark --dspark-model dspark-Inkling-2-Mini-Q8_0.gguf
```

The default context is 16,384 tokens: the KV cache is FP16 for all 48 passes
(about 393 KB per token). Sessions of 65,536 tokens or more select block-sparse
attention for the full-attention passes automatically (`INKLING_SPARSE=0/1`
forces it).

- **Thinking.** The model thinks even at effort 0. `--think off` starts the answer
  in the text channel, which is what code and JSON requests want.
- **Speculation.** `--speculative dspark` drafts up to 7 tokens per step from the
  external drafter and verifies them in one pass. The draft is cut at the first
  slot whose top-1 probability is below 0.75, and steps with no confident draft
  run as plain decode. Output is exact: a draft is kept only if the sampler picks
  the same token.
- **NPU.** `INKLING_NPU=1 INKLING_NPU_M=4096 GUFO_NPU_DIR=<repo>/tools/npu` offloads
  the q/gate/k/v and attention-output projections for batches of 3,072 to 8,192
  rows (two token banks, the NPU hides behind the other bank's experts). Needs
  the fabric clock held and a binary named `gufo*`. `INKLING_NPU_PARTS` (bit mask
  1 q, 2 gate, 4 k, 8 v) and `INKLING_NPU_OUT=0` narrow it.
- **Fast profile (opt-in, lossy).** `INKLING_IU4=1` runs the routed gate/up experts
  on rotated symmetric int4 matrix cores (integer WMMA); `INKLING_IU4=2` runs the
  down projection on them too; `INKLING_F16_BLOCKS=1` keeps the prefill block states
  as FP16. The NVFP4 experts are repacked to int4 at first use of each layer
  (about 28 ms per layer, one-off), so the int4 copy costs extra memory (about
  340 MB per layer for gate/up, 190 MB for down; one copy per model, shared by all
  sessions). It applies to prefill batches of 32 rows
  or more and is off by default because it changes outputs, see
  [QUALITY.md](QUALITY.md). `INKLING_IU4_WCLIP=0` and `INKLING_IU4_ACLIP=<x>` tune the
  scale choice, `INKLING_IU4_SKIP=<layers>` keeps listed physical layers on NVFP4.
- **3K prefill profile (opt-in, lossy).** The fast profile plus GPTQ-rounded int4 gate/up
  weights and prefill expert pruning reaches about 3,000 tok/s at 8K prompts with the NPU
  (2,292 tok/s at 128K, against 1,935 for the lossless engine). Set
  `INKLING_IU4=2 INKLING_F16_BLOCKS=1 INKLING_PRUNE_PREFILL=0.10
  INKLING_GPTQ_DIR=<dir> INKLING_GPTQ_PARTS=1`; `run-gufo.sh` does this by default for
  Inkling (`GUFO_INKLING_FAST=0` serves the lossless engine). The GPTQ artifacts are built
  once per model file: `INKLING_GPTQ_CAPTURE=<layers> gufo_inkling_gpu_check <gguf> --gptq
  <text> <windows> 4096 <dir>` runs a lossless calibration prefill over the text, accumulates
  each expert's input Hessian for the listed layers (six layers at a time fit in memory
  next to the model) and writes `layer_<n>.gptq`; the whole model takes about 25 minutes
  and 13 GB. Without the artifacts the int4 experts fall back to plain rounding. Decode
  and speculative verification always use the exact kernels. Measurements are in
  [QUALITY.md](QUALITY.md) and [BENCHMARKS.md](BENCHMARKS.md).
- **Expert layers 20 to 23.** They are Q8_0 in the GGUF and are converted to NVFP4
  at load (`INKLING_EXPERT_NVFP4=0` keeps Q8_0). See [QUALITY.md](QUALITY.md).
- **Expert pruning** (`INKLING_PRUNE=<share>`, default off) skips routed experts
  whose routing weight share is below the threshold in few-row steps. It changes
  outputs; measurements are in [QUALITY.md](QUALITY.md).

## Chat, reasoning and tools

The chat template follows the model's own: effort header, tool declarations
(`tool_declare`), reasoning, tool calls and tool results. Generated channel tokens
are translated for the shared response layer: the thinking channel is returned as
`reasoning_content`, the text channel as `content`, and each
`<|content_invoke_tool_json|>` message as an OpenAI `tool_calls` entry (parallel
calls and streaming included). The server adds no tool grammar or instruction for
this model (`ToolsHandledByTemplate`). `tool_choice: required` cannot be enforced;
a turn without a call fails as unsatisfied.

## Status and gaps

Continuation snapshots work like the other models: a RAM cache (`--cache-ram-bytes`)
and a disk cache (`--cache-disk`) keyed by the GGUF and drafter fingerprints, context
size and expert format. A 46K-token snapshot is about 2-3 GB; a RAM hit restores in
tens of ms, a disk hit in 2-8 s (versus about 30 s to re-prefill). The request's
reasoning effort maps to the model's own levels (0.1, 0.2, 0.7, 0.9, 0.99).
There is no image input. Structured output
(`response_format`) constrains the raw channel text, so it only applies in the
text channel (`--think off`). Sampled (temperature above zero) speculation uses
the same exact-match rule as greedy and therefore accepts fewer drafts.

The CPU reference (`reference.cpp`, `gufo_inkling_ref_check`) is complete but runs
at about 13 s per token; the GPU engine matches it to about four digits.

## Planned

- Prefill attention on the NPU (attention is
  23 to 27% of GPU time at 8K): the full task description for a kernel and its engine hook is
  [../../plans/inkling-npu-attention-kernel.md](../../plans/inkling-npu-attention-kernel.md).
- Per-32 activation scales for the int4 fast profile (quality), see [QUALITY.md](QUALITY.md).
