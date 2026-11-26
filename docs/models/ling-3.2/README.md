# Ling 3.2 Air

Ling-3.2-air (inclusionAI/Ling-3.2-air) is a hybrid-linear mixture-of-experts vision-language model
with 124 billion total parameters and 5.5 billion activated per token. gufo runs its text side
(`bailingmoe3`; image input is rejected) on gfx1151: 42 trunk
layers (35 Kimi-delta-rule KDA layers and 7 multi-latent-attention layers), 2 dense lead layers,
512 routed experts with 8 used per token (sigmoid router, 8 groups of which 4 are searched) plus a
shared expert, and one multi-token-prediction layer. The tested weights are the bartowski
`Ling-3.2-air-Q4_K_M` GGUF (75 GB; Q8_0 attention and dense layers, Q4_K experts with Q6_K in some
layers, Q4_0 MTP layer).

## Run

```sh
gufo prompt -m Ling-3.2-air-Q4_K_M-00001-of-00003.gguf -f prompt.txt -n 300 -t 0 --think off -v
gufo prompt ... --speculative mtp --mtp-model embedded     # MTP, 2 drafts
GUFO_MODEL=ling32 ~/run-gufo.sh                             # serve, MTP on (GUFO_SPEC=off for plain)
```

`GUFO_NPU_DIR=<repo>/tools/npu` turns the NPU prefill on (needs the fabric clock held, a binary
named `gufo*`, and the XRT libraries on `LD_LIBRARY_PATH`). `gufo serve` drafts with MTP for plain greedy
requests (`--speculative mtp --mtp-model embedded`; sampled, penalised or grammar-constrained requests decode normally).

## What runs where

- **Weights.** Q8_0 dense tensors stay raw and run through the shared W8A8 prefill GEMM and the
  Q8 decode GEMV. Q6_K experts are re-encoded to Q8_0 at load, the MTP layer's Q4_0 tensors
  likewise. The KDA q/k/v/f_a/g_a and the shared gate/up weights are stacked at load.
- **Prefill** (4096-token tiles): batched KDA convolution and recurrence, routed experts through
  the shared grouped F16 WMMA GEMM, the router through an F16 hi/lo GEMM (`LING_F32_ROUTER=1`
  restores the exact F32 router), and non-absorbed MLA with block-sparse key selection.
- **Block-sparse MLA.** Keys are pooled in 64-row blocks (per-channel min/max), scored against the
  last query by a Quest bound, and the top 96 blocks plus a sink and the 4 most recent are gathered.
  Exact while the prefix fits the budget (6,144 rows). Needle retrieval is correct at 33k+ tokens.
- **Decode.** Absorbed MQA MLA (batched-head split kernels), wave-per-row KDA kernels, parallel
  router top-k.
- **MTP.** The model's MTP layer drafts 2 tokens per round (`LING_MTP_DRAFTS` changes it; 3 and 4
  drafts measured slower, see [BENCHMARKS.md](BENCHMARKS.md)); verification runs the trunk on the
  draft rows with KDA state checkpoints and absorbed MLA attention per row. Greedy only.

## NPU prefill

Two token banks of 4096 rows run layer-outer, bank-inner. The KDA q/k/v projections (12 slices of
1,024 columns, K = 2,560) are submitted for the next layer while the other bank computes, and the
shared-expert gate/up overlaps the routed experts. Weights are encoded on the GPU into pinned
memory and then moved to NPU-only pages. The activation and output buffers are released at the
first decode token and recreated before the next prefill of 4,096 tokens or more. A weight buffer
imported as a userptr mapping measurably stalls GPU decode (see the port notes), the private pages
do not. Prompts shorter than 8,192 tokens run the single-bank path (shared expert only).

## Quality

The NPU path uses block-float arithmetic; the final logits shift by 0.1 to 0.9 against the GPU-only
path and the top token is unchanged on the prompts tested. Greedy outputs diverge from the GPU-only
path at near-ties. No KL measurement has been made yet.

## Debug switches

`LING_VALIDATE=1` (finite/range checks), `LING_GUARD=1` (buffer canaries), `LING_PF_TIMING=1`,
`LING_PROF=1`, `LING_DEBUG_LAYERS=1`, `LING_NO_SYNC_SPARSE=1`, `LING_GRAPH=1`,
`LING_SPEC_PREFILL_MLA=1` (verify with the prefill MLA path), `LING_F32_ROUTER=1`.

## Known limits

- The first decode seconds after a long prefill include a one-time 0.4 to 0.8 s hiccup (NPU buffer
  release); plain decode at 16k to 64k tokens measures 30 to 34.5 tok/s over 300 tokens.
- Real-text prefill is 1,600 to 1,700 tok/s at 10k to 24k tokens; the 1,800+ figures are `gufo bench`
  with its repeating token pattern, which touches few experts.
- Serve MTP has no continuation snapshots yet (the Ling engine has none), so a repeated prefix is prefilled again.
