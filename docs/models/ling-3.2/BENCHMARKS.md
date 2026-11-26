# Ling 3.2 Air benchmarks

Measured 2026-11-25 on one Strix Halo (gfx1151) machine with the bartowski `Ling-3.2-air-Q4_K_M`
GGUF, GPU + NPU as in [README.md](README.md), greedy, stock VM settings, nothing else using the GPU.

## Prefill

`gufo bench -p <n>`, three runs per cell, median shown, NPU through `GUFO_NPU_DIR`.

| Prefill, tok/s | GPU only | GPU + NPU |
| --- | --- | --- |
| `-p 2048` | 1,828 | 1,821 (below the 4,096-row NPU bank) (-0.4%) |
| `-p 4096` | 1,815 | 1,842 (+1.5%) |
| `-p 8192` | 1,701 | 1,854 (+9.0%) |
| `-p 16384` | 1,624 | 1,566 (-3.5%) |

`gufo bench` repeats a short token pattern, which touches few experts per layer. Real text routes
across all 512 experts and prefill is lower: 1,600 to 1,710 tok/s on real prompts of 10k to 24k
tokens (`gufo prompt`, one cold prefill each), 1,300 to 1,450 at 55k to 167k. The `-p 16384` NPU
cell is noisy (1,551, 1,768 and 1,566 in three runs) and not ahead of GPU-only; at 60k tokens of
depth the NPU is ahead (`-p 8192 -d 60000`: 1,693 against 1,553 GPU only).

## Decode, real text, depth

`gufo prompt` with a context built from this repository's own prose, C++ and JSON files, 300
generated tokens, `--think off`, NPU on. `gufo serve` has no MTP for Ling, so these are CLI runs.

| Context | Workload | Prompt tokens | Plain | MTP 2 drafts | MTP 3 | MTP 4 |
| --- | --- | --- | --- | --- | --- | --- |
| 4k | prose | 5,135 | 36.0 | 35.6 (2.04/round) | 31.5 (2.11/round) | 27.7 (2.16/round) |
| 4k | code | 2,964 | 36.3 | 40.0 (2.26/round) | 36.0 (2.36/round) | 31.4 (2.40/round) |
| 4k | json | 5,600 | 35.8 | 42.5 (2.48/round) | 39.9 (2.73/round) | 37.2 (2.97/round) |
| 16k | prose | 20,995 | 32.0 | 29.6 (1.81/round) | 25.3 (1.82/round) | 21.6 (1.82/round) |
| 16k | code | 13,659 | 34.5 | 32.1 (1.94/round) | 28.6 (2.03/round) | 24.8 (2.05/round) |
| 16k | json | 24,127 | 30.0 | 33.8 (2.08/round) | 30.4 (2.21/round) | 26.9 (2.29/round) |
| 64k | prose | 77,013 | 32.4 | 41.5 (2.61/round) | 34.9 (2.63/round) | 25.1 (2.63/round) |
| 64k | code | 55,241 | 30.8 | 33.8 (2.14/round) | 27.8 (2.22/round) | 24.5 (2.27/round) |
| 64k | json | 84,940 | 32.4 | 39.4 (2.48/round) | 33.7 (2.50/round) | 31.2 (2.73/round) |

MTP drafts 2 tokens per round by default. Three and four drafts measured slower on this engine,
because the single MTP layer is reused for every draft and acceptance falls quickly; the extra
verify rows cost more than the extra accepted tokens. `LING_MTP_DRAFTS` changes the count. The gain
over plain decode is largest on JSON-like text (+6 to +7 tok/s at 4k) and at 64k (+9 on prose); on
prose and code at 4k to 16k it is within 3 tok/s of plain decode or below it.

`gufo bench -n 128` plain decode: 37.0 tok/s (empty context, GPU only), 34.0 tok/s at 60k tokens
of depth with the NPU on (34.4 without it).

## Notes

- The NPU releases its activation buffers at the first decode token. Keeping them costs a 2 to 4 s
  stall in the first decode seconds (tok/s over 128 tokens: 25.7 kept, 34.0 released).
- Plain decode at 16k to 64k tokens measures 30 to 34.5 tok/s over 300 tokens (empty context 37). The
  gap is a one-time 0.4 to 0.8 s hiccup when the NPU buffers are released at the first decode token;
  a 128-token window shows it as 29 to 35 tok/s.
- Output differs from the GPU-only path at near-ties (block-float NPU arithmetic, F16 hi/lo router);
  no KL measurement has been made.

## Notes moved from the README speed section

Ling-3.2-air (inclusionAI/Ling-3.2-air) is a hybrid-linear mixture-of-experts vision-language model with 124 billion total parameters and 5.5 billion activated per token (35 delta-rule linear-attention layers and 7 latent attention layers, 512 experts with 8 routed plus a shared expert). gufo runs the text side only, in the 75 GB Q4_K_M quantization, with block-sparse attention for long context; image input is rejected.

Real-text prefill is lower than `gufo bench` (its repeating pattern touches few experts): 1,600 to
1,710 tok/s on 10k to 24k token prompts.

`gufo bench -n 128` plain decode: 37.0 tok/s. MTP (`gufo prompt --speculative mtp --mtp-model
embedded`) drafts 2 tokens per round; 3 and 4 drafts measured slower. Decode at 16k to 128k
tokens, the draft-count sweep and the method are in
[docs/models/ling-3.2](docs/models/ling-3.2/BENCHMARKS.md).
