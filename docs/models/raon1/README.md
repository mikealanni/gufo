# Raon-1 mini

Raon-1 mini (KRAFTON) is a hybrid mixture-of-experts language model: 25 layers, of which 18 are
recurrent delta-rule layers (RDA, 40 heads of 128, per-channel decay) and 7 are multi-latent
attention (MLA, 512-wide latent plus 64 rotary dims, 40 heads). Hidden size 2,560 carried as four
hyper-connection streams, 128 routed experts with 6 used per token (expert width 1,088) plus one
shared expert, a 259,840-token vocabulary. gufo runs it on gfx1151 from the `Raon-1-mini` GGUF
(`Q4_K_M`, 19.7 GB: Q4_K expert gate/up, Q8_0 or Q5_0 expert down, Q6_K for the recurrent
projections and the output head) with the published DFlash2 drafter (`Q8_0`, 0.73 GB).

## Run

```sh
gufo prompt -m raon-1-mini-Q4_K_M.gguf --speculative dflash2 --dflash-model dflash-raon-1-mini-Q8_0.gguf \
    -f prompt.txt -n 300 -t 0 --think off
gufo bench  -m raon-1-mini-Q4_K_M.gguf -p 2048,8192 -n 128 -d 32768,131072 --speculative off
GUFO_MODEL=raon1 JOBS=2 ~/run-gufo.sh                     # serve, DFlash2 on (GUFO_SPEC=off for plain)
```

`GUFO_NPU_DIR=<repo>/tools/npu` turns the NPU prefill on (fabric clock held, a binary named `gufo*`,
XRT libraries on `LD_LIBRARY_PATH`). `gufo bench` supports plain decode only; use `gufo prompt` or
`gufo serve` for speculation.

## What runs where

- **Weights.** Quantized tensors stream disk-to-device through `hip::TensorUpload`, which never
  faults the GGUF mapping in. After load the mapping holds under 1 GB of resident pages and the
  process RSS is about 3 GB; a loader that copies from the mapping keeps the whole model faulted
  in as dead memory, which stalls reclaim and makes shutdown take minutes.
- **Prefill** (4096-token tiles, two banks): batched recurrent kernels, routed experts through the
  shared grouped F16 WMMA GEMM, non-absorbed MLA with an F16 WMMA flash kernel. The q/k/v/g
  projections of the 18 recurrent layers, the MLA output gate and the shared-expert gate/up run on
  the NPU (`GUFO_RAON1_NPU_PARTS`, default `gate,shared`). Two banks are the largest NPU gain; a
  third bank and the attention output projection on the NPU both lose.
- **MLA attention at depth.** Exact below 16,384 tokens of context. Above it, block-sparse
  attention (64-row blocks, Quest bound) keeps prefill and decode flat out to 256K:
  `GUFO_RAON1_SPARSE=k,window,min_depth,prefill_prefix,decode_prefix`, default
  `48,1536,16384,256,8192` (48 selected blocks, 1,536 local rows, a protected prefix of the first 256
  tokens while prefilling and the first 8,192 while decoding). `GUFO_SPARSE=0` in the launcher keeps
  attention exact at every depth. The prefix holds the system prompt and the tool definitions:
  with only 256 protected tokens at decode, a replayed agent request chose the wrong tool in 4 of 8
  samples at 46K tokens; 4,096 or more gave 0 of 8, the same as exact attention.
- **Decode attention.** Absorbed MLA on F16 WMMA: the 40 heads of every verified token are rows of
  one tile, so a draft block reads each latent key once. q, k, softmax weights and v are rounded to
  F16 as in prefill, so greedy output is not bit-identical to the older F32 kernel
  (`GUFO_RAON1_MLA_WMMA=0` restores it).
- **Speculation.** The DFlash2 drafter proposes up to 5 tokens; a running average of accepted
  drafts picks the width and falls back to plain steps when it stays low. Recurrent state is saved
  before a verified chain and replayed up to the accepted length on a partial accept.
- **Sampling.** At temperature 1 with `top_p` the nucleus is found from the top 256 candidates and
  the exact softmax normalizer, growing the sorted prefix 8x at a time when it falls short; the
  distribution is the full-sort result.

## Serve

- **Cache.** Prefix cache with snapshot and fork (recurrent state, latents, last logits; the sparse
  block summaries are rebuilt after a restore), RAM and disk tiers, persistent snapshots.
- **Two requests at once.** `JOBS=2` runs two sessions on the two executor banks from two threads
  (`plan=batched-w2` in the completion log). Per-bank state keeps the rollback records, the
  recording flag and the logit rows. Greedy outputs match a solo run.
- **Reasoning effort.** Raon's template only switches thinking on or off, so effort maps to a
  thinking budget: low 1,024, medium 4,096, high 16,384 tokens; minimal, xhigh and max are uncapped.
  The launcher's server default for this model is `xhigh`; the jev plugin picks an effort per
  request.
- **Memory.** A 63K-token session costs about 1.1 GB of snapshot plus the latent cache
  (16 KB per token across the 7 MLA layers).

## Numerics and quality

- The CPU reference (`reference.cpp`) matches the reference runtime on the first tokens; the GPU
  matches the CPU reference (correlation 0.997 to 0.999, same argmax).
- `GUFO_BENCH_TEXT=<file> GUFO_BENCH_NLL=600 gufo bench -m <model> --speculative off` prints a
  deterministic teacher-forced negative log-likelihood. Use it as the gate for any change that moves
  numerics.
- Sparse attention is approximate above the activation depth. Needle retrieval is not a sufficient
  gate: replay a captured agent request several times and compare tool choices against exact
  attention.
- Greedy output varies run to run after a few hundred characters on the same prompt (near-tie
  routing); compare logits or NLL, not text.

## Status

Prefill of fresh 4K to 8K real text is about 3,000 tok/s in `gufo bench` and 2,800 through serve;
decode is 45 to 70 tok/s depending on how predictable the text is. Open items: the recurrent scan
(about 15% of prefill), hyper-connection kernels (about 12%), routed-expert GEMM efficiency, and a
graph-captured single-token decode step. Measured numbers are in [BENCHMARKS.md](BENCHMARKS.md);
the process log is [PORTING.md](PORTING.md).
