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
- **Expert layers 20 to 23.** They are Q8_0 in the GGUF and are converted to NVFP4
  at load (`INKLING_EXPERT_NVFP4=0` keeps Q8_0). See [QUALITY.md](QUALITY.md).
- **Expert pruning** (`INKLING_PRUNE=<share>`, default off) skips routed experts
  whose routing weight share is below the threshold in few-row steps. It changes
  outputs; measurements are in [QUALITY.md](QUALITY.md).

## Status and gaps

Control tokens (`<|content_thinking|>`, `<|end_message|>`) are returned as plain
text; there is no reasoning split, no snapshot or disk cache, and no tool calling.
Sampled (temperature above zero) speculation uses the same exact-match rule as
greedy and therefore accepts fewer drafts.

The CPU reference (`reference.cpp`, `gufo_inkling_ref_check`) is complete but runs
at about 13 s per token; the GPU engine matches it to about four digits.
