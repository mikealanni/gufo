# Inkling 2 Mini benchmarks

Decode measured 2026-11-22 on one Strix Halo (gfx1151) machine, greedy, real text.
Prefill is `gufo_inkling_gpu_check` `INKLING_BENCH=<n>` (single request, no
cache); decode is `gufo serve` over HTTP with `--think off`, 400-token (short) or
300-token (deep) replies, speculation on unless stated.

## Prefill

Measured 2026-11-23 (`gufo_inkling_gpu_check`, `INKLING_BENCH=<n>`, warm, three
repeats, greedy text). The default profile is lossless against the engine before
these changes. The fast profile is opt-in (`INKLING_IU4=2 INKLING_F16_BLOCKS=1`) and
changes outputs, see [QUALITY.md](QUALITY.md).

| Prompt | Default GPU only | Default GPU + NPU | Fast GPU only | Fast GPU + NPU |
| --- | --- | --- | --- | --- |
| 2k | 2,065 | below the NPU batch | 2,499 | below the NPU batch |
| 4k | 2,144 | 2,342 | 2,580 | 2,737 |
| 8k | 2,075 | 2,382 | 2,466 | 2,822 |
| 16k | 1,930 | 2,170 | 2,254 | 2,516 |

Run-to-run spread is about 1% when nothing else uses the machine; background load
costs more. The 32k and 128k rows of the earlier table (1,536 to 1,689 and about
1,035 to 1,647 tok/s) have not been re-measured.

### 3K profile and 128K

Measured 2026-11-24 with `gufo_inkling_gpu_check` (one cold run per row, GPU + NPU, banks of 4,096):
the 3K profile (see [README.md](README.md)) prefills an 8K prompt at 3,018, 3,034 and 3,038 tok/s
against 2,545 to 2,807 for the fast profile on the same machine (a monitor was using a core; the
spread shows the noise). At 127,680 tokens of real text prefill is 1,935 tok/s for the default engine
and 2,292 for the 3K profile; plain decode at that depth is 28.5 tok/s in both. Speculative decode
at 127,680 tokens with the drafter, 400 generated tokens: 49.0 and 53.8 tok/s on code and
documentation text (3.7 and 4.1 accepted tokens per step), 38.1 and 30.3 on raw source dumps
(2.6 and 1.8 accepted); a step costs about 93 ms at that depth (12 ms draft, 81 ms verify)
against about 67 ms at 4K, so the rate follows the acceptance.

### Fast profile without pruning (default)

Measured 2026-11-25, GPU + NPU, banks of 4,096, warm 8K prefill, alternating runs: INT4 experts with
GPTQ and fp16 block states, no pruning, 2,851 and 2,853 tok/s (exact engine about 2,390; pruning 0.10,
3,010 and 3,030). Light pruning does not pay: 0.02, 0.04, 0.06 and 0.08 give 2,857, 2,862, 2,875 and
2,933 tok/s. `INKLING_PRUNE_PREFILL=0` gives logits identical to leaving it unset. GPU-only 4K profile
without pruning, window 1,368 ms: int4 gate/up 308 ms (54 TOPS), exact dense projections 304, attention 296,
int4 down 164, loop control 98, RMS 95, mixes 124, ConvT 60. Tried and not kept: fusing the NPU q decode with
the q clip (no change at 8K) and prefetching the next K/V tile in the attention kernel (attention 22% slower).

## Decode

| Workload | tok/s |
| --- | --- |
| Plain decode, short context | about 33 (29.7 ms per step) |
| Code, speculation | 74.6 (about 82% of drafts accepted) |
| JSON, speculation | 71.1 (about 82% of drafts accepted) |
| 37k context, code | 54 to 59 |
| 63k context, code | 54 to 61 (three runs: 53.7, 60.9, 55.1) |

A verified step of six rows takes 61.8 ms at short context and 73 ms at 62k;
a plain step takes 29.7 ms and 36 ms. At 62k the whole increase is attention,
about 7 ms for the eight full-attention passes with block-sparse selection.
Speculation gain depends on the text: prose accepts far fewer drafts than code or
JSON. Run-to-run spread at depth is about 2 tok/s because acceptance varies.

TODO: sampled (temperature above zero) decode, concurrent requests, 128k decode.

## Notes moved from the README speed section

Greedy, real text, one request; the weights are the UD-Q4_K_XL GGUF with the DSpark
drafter. Prefill is the single-request engine check (no cache).

Sampled requests (temperature above zero) accept fewer drafts, so they decode more
slowly than these greedy figures. Loads in about 15 s with the model file cached.
Details and method: [docs/models/inkling/BENCHMARKS.md](docs/models/inkling/BENCHMARKS.md).
