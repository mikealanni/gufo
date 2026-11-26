# Qwen3.6 35B-A3B benchmarks

UD-Q4_K_XL, gfx1151, fabric clock held, NPU in performance mode, nothing else
running. Prefill is `gufo bench -m MODEL -p N -n 128 -r 3`; GPU + NPU sets
`GUFO_NPU_DIR=tools/npu`. Decode on real text sends greedy chat requests to
`gufo serve llm` (500 tokens, three timed repeats after one warm-up).

## Prefill

| Tokens | GPU only | GPU + NPU |
| --- | --- | --- |
| 2048 | 3,034 | 3,066 |
| 4096 | 3,013 | 3,276 |
| 8192 | 2,862 | 3,134 |

## Decode, tok/s

`gufo bench` plain decode: 60.6 tok/s.

| Prompt | Plain | MTP | DFlash2 | Auto |
| --- | --- | --- | --- | --- |
| Code (LRU cache and tests) | 61 | 111 | 140 | 138 |
| Prose (500-word essay) | 61 | 72 | 59 | 68 |
| JSON (25 records) | 61 | 122 | 159 | 156 |

MTP drafts three tokens. `gufo bench` on its repetitive synthetic context
reports 69 to 80 tok/s for MTP, a poor guide to real text.

## Notes moved from the README speed section

`gufo bench -n 128` plain decode: 60.6 tok/s.
