# Inkling 2 Mini benchmarks

Measured 2026-11-22 on one Strix Halo (gfx1151) machine, greedy, real text.
Prefill is `gufo_inkling_gpu_check` `INKLING_BENCH=<n>` (single request, no
cache); decode is `gufo serve` over HTTP with `--think off`, 400-token (short) or
300-token (deep) replies, speculation on unless stated.

## Prefill

| Prompt | GPU only | GPU + NPU |
| --- | --- | --- |
| 2k | 1,692 | 1,692 (below the 3,072-row NPU batch) |
| 4k | 1,768 | 1,837 |
| 8k | 1,715 | 1,851 |
| 16k | 1,670 | 1,791 |
| 32k | 1,536 (1,619 sparse) | 1,689 |
| 128k | about 1,035 | 1,647 |

The 8k to 128k rows predate the expert-layer conversion, which added about 2% at
2k and 4k.

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
