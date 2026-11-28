# Raon-1 mini benchmarks

Hardware: Strix Halo (gfx1151), 128 GB. Weights `raon-1-mini-Q4_K_M.gguf`, drafter
`dflash-raon-1-mini-Q8_0.gguf`. Repeating-pattern prompts overstate speed: the routed experts
see few distinct inputs. Rows marked "real text" use `GUFO_BENCH_TEXT=<file>` with source code or
prose, which is what an agent sends. This box drifts about 5% between minutes, so compare
interleaved runs, and treat anything under 5% as noise.

## Prefill, `gufo bench`, real text, NPU

| Config | pp4096 | pp8192 |
| --- | --- | --- |
| one bank | 2,501 | 2,406 |
| **two banks, gate + shared expert on the NPU (default)** | **3,103** | **3,010** |
| two banks, shared expert only | 3,182 | 2,954 |
| two banks, plus the attention output projection | 3,027 | 2,779 |
| three banks | 2,753 | 2,792 |
| two banks, nothing on the NPU | 3,141 | 2,901 |

Two banks are worth +25%; the shared-expert and gate offload add 3 to 4% at 8K; more NPU work or
more banks loses because the added submissions delay the next layer's waits. The same binary on
the repeating-pattern prompt reads 3,537 (4K) and 3,409 (8K).

Through `gufo serve` a fresh 8K prompt prefills at about 2,840 tok/s with the drafter off and
about 2,650 with it on (the drafter's context injection).

## Context depth, `gufo bench -d`, plain decode

| Depth | exact: pp4096 | exact: tg | sparse: pp4096 | sparse: tg |
| --- | --- | --- | --- | --- |
| 8K | 2,830 | 50.8 | exact below 16K | |
| 26K | | 46.6 | | |
| 32K | 1,845 | | 2,539 | 44.4 |
| 64K | | 37.6 | 2,502 | 43.8 |
| 128K | | | 2,499 | 42.7 |

Sparse is the default above 16,384 tokens: `GUFO_RAON1_SPARSE=48,1536,16384,256,8192`
(prefill protects the first 256 tokens, decode the first 8,192). Both stay flat to 256K; a
256-token prefix at decode measured 2,454 pp and 44.5 tg at 253K, but it chose the wrong tool in an
agent replay, so it is not the default.

| Protected prefix (prefill = decode), 64K | pp4096 | tg | wrong tool in 8 replays |
| --- | --- | --- | --- |
| 256 | 2,510 | 48.2 | 4 |
| 4,096 | 2,121 | 46.2 | 0 |
| 8,192 | 1,766 | 43.7 | 0 |
| 16,384 | 1,364 | 39.9 | 0 |

The split (256 at prefill, 8,192 at decode) keeps prefill at the first row and still gave 0 wrong
tools in 8 replays.

## Decode

Per-step throughput of the speculative decode loop (`gufo prompt`, DFlash2, 5 drafts), measured
from the engine's own step timers, temperature 1:

| Text | tok/s | tokens per step |
| --- | --- | --- |
| source code, about 6K context | 63 to 73 | 3.2 to 4.3 |
| thinking prose, about 10K context | 36 to 39 | 1.4 to 1.8 |

Plain decode: 54 tok/s at empty context. Verifying a draft at about 10K context costs 34 ms for 3 rows
(61 ms before the WMMA attention kernel) and 53 ms for 6 rows (106 ms before).

## Two requests at once (`JOBS=2`, `plan=batched-w2`)

| Context | alone, each | together, each | aggregate vs one at a time |
| --- | --- | --- | --- |
| about 8K | 47 to 48 | 31 to 33 | 62 to 66 vs 48 |
| about 63K | 41 to 54 | 27 | 54 vs about 48 |

Greedy output of each request matched its solo run for the first 150 characters at 8K; at 63K both differ from solo by the
run-to-run variation measured between two solo runs. Host memory pressure stayed under 3% with two
63K sessions.

## Memory and load

| | before | after |
| --- | --- | --- |
| GGUF mapping resident in the server | 18.4 GB | 0.7 GB |
| server RSS | 29 GB | 2.9 GB |
| time from SIGTERM to exit | 210 s (or never) | 2 s |

## Method notes

- Prefill rows: `gufo bench -p 4096,8192 -n 0 -r 2` with `GUFO_BENCH_TEXT`.
- Depth rows: `gufo bench -d <depth> -p 4096 -n 64 -r 1 --speculative off`.
- Quality gate: `GUFO_BENCH_NLL=600` (teacher-forced NLL, deterministic).
- Agent replay: capture the request body an agent harness sends, replay it with different seeds
  against the server, count tool choices.
