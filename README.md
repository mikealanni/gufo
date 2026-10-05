# Gufo Flash-Next (fork)

A fork of [gufo-org/gufo](https://github.com/gufo-org/gufo), a local LLM inference engine
built for AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S, `gfx1151`, up to 128 GiB unified memory).
This fork focuses on **Qwen3.8 Flash-Next** with MTP speculative decoding and on the
snapshot/serving layer.

Released under the MIT license, same as upstream. `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`
and `licenses/` are kept unchanged. Credit for the engine goes to the gufo contributors.

## Model

| | |
| --- | --- |
| Model | Qwen3.8 Flash-Next (hybrid recurrent/QSA mixture-of-experts, text + image) |
| Weights | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), both supported: `UD-IQ4_XS` (3 shards, about 89 GiB; our main model) and `UD-Q4_K_XL` (4 shards) |
| Speculative decoding | Unsloth shared-Q8_0 MTP predictor, `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2.6 GiB). Other quants or sources fail to load |
| Vision | Optional `mmproj-BF16.gguf` |
| Context | 131072 in our setup (native 262144) |
| Memory | About 84 / 87 / 91 GB of free RAM for 1 / 2 / 3 concurrent sessions at 131072 context |
| Hardware | Strix Halo `gfx1151` only, Linux x86-64, 128 GiB unified memory |

## What this fork improves

Kernels and decode (Flash-Next):
- IQ3_S codebook and sparse-attention selection bits staged in LDS.
- IQ3_S gate and up projections paired with a fused SwiGLU; 128-token routed tile for IQ3_S gate/up and IQ4_NL down.
- Two state rows per lane in the GDN row-split kernel; F16 `ssm_out` activation pitch padded off a 4 KiB multiple.
- Prefill tile is env-tunable and defaults to 4096.
- MTP drafts from the first 131072 vocabulary rows plus special tokens.
- The next prefill chunk's n-gram rows are read while the GPU runs the current chunk.
- K-quant dense weights kept off the batched SmallGemm route.
- `GUFO_MTP_LOG` prints draft rejection and cycle timing.

Serving and core:
- Disk snapshots are indexed from their headers, checksummed as a parallel block tree and read with parallel preads; page cache is dropped after I/O.
- New options for disk snapshot stride and in-RAM snapshot entries, plus `--cache-ram-bytes` to cap the in-RAM snapshot budget.
- The request with the fewest remaining prefill tokens is prefilled first.
- Vectorized no-penalty softmax normalizer and top-p nucleus gather.
- Pooled snapshot streams and staging buffers; the device lock is held through a new graph's first launch.
- Fixes for graph capture, pinned-buffer use-after-free and rollback trim.

## Benchmarks

Strix Halo `gfx1151`, 128 GB. All benchmarks below are for `UD-IQ4_XS`, our main model, with the shared-Q8_0 MTP sidecar, built from this
fork (`scripts/run-gufo.sh`: 131072 context, 2 sessions, prefill chunk 4096, thinking off, greedy,
HTTP streaming). Measured October 5, 2026. pp is prompt tokens over time to first token; tg is
decode tok/s after that.

### Prefill speed by prompt size

`UD-IQ4_XS`, no MTP, 1 session, prefill tile and chunk 4096, EC performance profile (about 120 W).
pp is prompt tokens over time to first token, mean of 3 to 5 cold requests after one warm-up.

| Prompt (tokens) | Chunks | pp (tok/s) |
| ---: | ---: | ---: |
| 1,024 | 1 | 1,185 |
| 1,536 | 1 | 1,301 |
| 2,040 | 1 | 1,404 |
| 3,072 | 1 | 1,475 |
| 4,096 | 1 | 1,566 |
| 4,108 | 2 | 1,507 |
| 6,155 | 2 | 1,458 |
| 8,203 | 3 | 1,506 to 1,519 |

Prefill reaches 1,500 tok/s and above from about 3k tokens up, and peaks at 1,566 when a prompt fits one tile.
Short prompts pay a fixed cost per request of roughly 0.2 s. A tiny tail chunk past a 4,096 boundary costs
about 4%. The balanced EC profile (about 85 W) is about 10% slower: about 1,290 at 2k and 1,370 at 8k.
The tables below were measured before the profile was recorded.

### HumanEval prompts, AR vs MTP

All 164 HumanEval problems as chat requests (real code prompts, about 100 prompt tokens), greedy,
thinking off, up to 512 generated tokens, one request at a time. tg is decode tok/s per request,
averaged over the problems.

| Mode | tg mean (tok/s) | tg median (tok/s) | Time to first token (median) |
| --- | ---: | ---: | ---: |
| AR | 27.89 | 27.92 | 0.41 s |
| MTP | 75.75 | 76.07 | 0.42 s |

MTP is 2.7x faster on this workload, and all 164 completions are byte-identical between the two modes
(same 38,335 generated tokens). Code is short and structured, so MTP drafts are accepted often.

### Long prompts, MTP

Mean of 2 cold runs each (unique prompt prefix, no snapshot hits), 128 generated tokens. pp and TG
are the server's own `prefill_tps` and `decode_tps` from its request log.

| Prompt (tokens) | pp (tok/s) | TG (tok/s) | MTP acceptance |
| ---: | ---: | ---: | ---: |
| 8,205 | 1,470 | 48.4 | 75% |
| 32,780 | 1,454 | 48.6 | 73% |
| 65,548 | 1,433 | 49.7 | 74% |
| 116,405 | 1,406 | 49.8 | 72% |

Without MTP (AR), decode is 26.9 tok/s at 32,787 tokens and 26.0 at 116,411, so depth costs little on
its own. The 65k and 116k TG ranges are 41.4 to 58.0 and 47.6 to 51.9 tok/s between runs, driven by how many
drafts are accepted.

Two concurrent requests, 8,205-token prompts: aggregate decode 68.6 tok/s (33.3 + 28.5 and
38.5 + 36.9 in the two rounds), per-request pp about 1,466 tok/s. The second request prefills
while the first is already decoding, so the aggregate is a lower bound.

Kernel-level results, `UD-IQ4_XS` MTP decode:

| Change | Before | After |
| --- | ---: | ---: |
| Correct MMVQ vdr for IQ3_S and IQ4_XS | 22.16 | 41.68 |
| Row-grouped tall/narrow Q8 decode | 41.68 | 46.79 |

## Install

### Requirements

- Linux x86-64 with an AMD Strix Halo (`gfx1151`) and ROCm
- Either Nix, or a C++20 compiler, CMake 3.21+, Ninja, pkg-config and the ROCm development libraries (hipBLAS, hipBLASLt, rocBLAS)
- About 92 GiB of disk for the model files and 84 GB or more of free RAM to serve

### Build

```sh
git clone https://github.com/mikealanni/gufo.git
cd gufo
git switch fork-enhancements

# With Nix
nix build
./result/bin/gufo diagnose

# Without Nix
cmake --preset release
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
```

Dependency details are in the upstream notes: [build from source](https://github.com/gufo-org/gufo#build-from-source).

### Download the model

```sh
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 83cadfda58d30be06c110518208d1bb918b33f10 \
  --include "UD-IQ4_XS/*" \
  --local-dir models/qwen3.8-flash-next
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" \
  --local-dir models/qwen3.8-flash-next
```

### Run

`scripts/run-gufo.sh` starts the OpenAI-compatible server on port 8900 with MTP, a 131072-token
context, 2 sessions and a disk snapshot cache.

```sh
export GUFO_MODEL=$PWD/models/qwen3.8-flash-next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export GUFO_MTP=$PWD/models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
scripts/run-gufo.sh
```

Settings (environment variables): `PORT` (8900), `JOBS` sessions (2), `GUFO_CONTEXT` (131072),
`GUFO_EFFORT` reasoning effort (low), `GUFO_BIN` binary path, `GUFO_CACHE_DIR` and
`GUFO_CACHE_BYTES` (32 GiB) for disk snapshots, `GUFO_LOG`.
Extra arguments are passed to `gufo serve llm`.

Live dashboard with speeds, per-request progress and memory (reads the server log):

```sh
scripts/gufo-watch.py            # --port 8900 --log ~/.cache/gufo/gufo.log --once
```

Plain terminal chat or a manual server:

```sh
./result/bin/gufo chat --model "$GUFO_MODEL"
./result/bin/gufo serve llm --model "$GUFO_MODEL" --speculative mtp \
  --mtp-model "$GUFO_MTP" --sessions 2 --context 131072
```

Leave out the `--speculative` options for plain autoregressive decoding.
`--draft-tokens` caps MTP proposals at 1 to 7. Server options are in [docs/SERVER.md](docs/SERVER.md).

## Docs

- [Flash-Next model guide](docs/models/qwen3.8-flash-next/README.md)
- [Benchmarks](docs/models/qwen3.8-flash-next/BENCHMARKS.md)
- [Quality](docs/models/qwen3.8-flash-next/QUALITY.md)
- [Testing](docs/TESTING.md)

## License

MIT. See `LICENSE`. Model weights have their own licenses.
