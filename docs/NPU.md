# NPU build: options and environment variables

The XDNA2 NPU path is optional. It is compiled in with `-DGUFO_ENABLE_NPU=ON`
(see the [README](../README.md)) and switched on at run time by pointing
`GUFO_NPU_DIR` at `tools/npu`. Without `GUFO_NPU_DIR` the same binary runs
GPU-only.

## Prerequisites at run time

| Requirement | Command |
| --- | --- |
| GPU fabric clock held at its top level | `sudo tools/npu/fabric-clock.sh hold` (`status`, `release`) |
| NPU power mode `balanced`, `performance` or `turbo` (turbo is stable with the fabric clock held and measured identical to `performance`) | `sudo xrt-smi configure --pmode performance` |
| NPU kept out of runtime suspend | `echo on \| sudo tee /sys/bus/pci/devices/<npu pci id>/power/control` |
| XRT libraries on the loader path | `LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH` |

The fabric clock changing speed while the NPU is busy can hang the machine, so
hold it before starting. The clock and power mode reset at boot.

## Easiest start

```sh
# Qwen3.8 Flash-Next (default)
GUFO_MODEL=<first shard .gguf> GUFO_MTP=<mtp sidecar .gguf> scripts/run-gufo.sh
# Qwen3.8-27B: DFlash2 and/or an MTP file
GUFO_KIND=27b GUFO_MODEL=<27b.gguf> GUFO_DFLASH=<dflash2.gguf> scripts/run-gufo.sh
# Qwen3.6-35B-A3B: embedded MTP head, plus DFlash2 for the auto mode
GUFO_KIND=35b GUFO_MODEL=<35b.gguf> GUFO_DFLASH=<dflash2.gguf> scripts/run-gufo.sh
```

`GUFO_KIND` picks the model family. For the 27B and 35B-A3B the NPU shares each
projection with the GPU (no two-bank mode) and the speculative mode follows
what is set: both drafts select `auto`, one selects that draft.

The script enables the NPU only while the fabric clock is held (any NPU power mode,
turbo included), and turns the two-bank prefill on with it (`GUFO_TWO_BANK=0` opts
out). `GUFO_PREFILL_TILE=24576` (two 12288-token banks) prefills 24K-token prompts about 2.7% faster for about 6 GB more RAM.

## Environment variables

### NPU on and off

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_NPU_DIR` | unset (NPU off) | Directory with the NPU kernel files (`tools/npu`). Setting it turns the NPU on. |
| `GUFO_NPU` | `1` | Read by `scripts/run-gufo.sh` only. `0` forces GPU-only. |

### What the NPU computes (Qwen3.8 Flash-Next)

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_NPU_SLICES` | `6` | 1024-row slices of the linear-attention `ssm_in` projection (z rows) computed on the NPU, 1 to 12. |
| `GUFO_NPU_GATE_SLICES` | `12`, `0` with `GUFO_NPU_WHOLE=1` | 1024-row slices of the attention `attn_qkv` projection (query and gate rows, two heads per slice) on the NPU, 0 to 12. `0` turns this part off. |
| `GUFO_NPU_SHEXP` | on | `0` keeps the shared-expert gate and up projections on the GPU. |
| `GUFO_NPU_LAYERS` | all | Limit how many linear-attention layers use the NPU. |

### Prefill

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_PREFILL_TILE` | `16384` | Prefill chunk in tokens (128 to 65536). Larger fills the routed-expert tiles better and is faster: 4096 gives 1533 tok/s, 8192 1582, 16384 1606 on a 40k cold prompt (older build). Scratch memory grows with it: about 3 GiB at 8192, about 9 GiB at 16384. |

The server's `--prefill-chunk` is a separate scheduling setting (how much a
prefill runs before decoding gets a turn); `scripts/run-gufo.sh` keeps it at 4096.

### Two-bank prefill (experimental, not bit-identical)

Recommended recipe for Flash-Next: `GUFO_NPU_DIR=tools/npu GUFO_TWO_BANK=1 GUFO_NPU_WHOLE=1` with the GPU fabric clock held (`scripts/run-gufo.sh` sets all of it). `GUFO_TWO_BANK=2` splits unevenly and was faster and clean in third-party runs; it keeps the F32 residual unless `GUFO_F16_RESIDUAL=1`. The plain slice mode (no `GUFO_NPU_WHOLE`) returned non-finite logits on the UD-IQ4_XS pack in a third-party run and is not recommended there.

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_TWO_BANK` | unset | `1` splits a prefill chunk of 8192 tokens or more (a multiple of 8192, or 8192 + 4096 groups) into two equal banks that run layer by layer. While one bank runs on the GPU, the NPU computes the other bank's next projection. Needs `GUFO_NPU_WHOLE=1`. `scripts/run-gufo.sh` sets both when the NPU is enabled (`GUFO_TWO_BANK=0` turns it off). |
| `GUFO_NPU_WHOLE` | unset | `1` moves the complete `ssm_in` (16 slices) and `attn_qkv` (13 slices) projections to the NPU in the two-bank path. The GPU reads the NPU output streams directly: F16 convolution rows, a chunked (64-token) gated delta rule, and fused attention preparation. Adds about 3.3 GiB of NPU weight buffers and about 240 MB of shared scratch. |

pp16384 rises from about 1,965 to about 2,120 tok/s (+8%) at the cost of block-float
rounding: final-token KL against GPU-only prefill averages 0.013 against 0.017
for the slice mode. See `EXPERIMENTS.md` for what was tried and rejected.

### Reliability and diagnostics

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_NPU_TIMEOUT_MS` | `8000` | Time a single NPU run may take before it is treated as failed. |
| (automatic) | on | A prefill that returns non-finite logits is discarded with an error and the NPU stays off until the process restarts, as after a timeout. |
| `GUFO_NPU_KEEPALIVE` | on | `0` stops the 1 s heartbeat that keeps the NPU awake. |
| `GUFO_NPU_WARMUP_MS` | `0` | Keep the NPU out of the first N ms of each prefill burst so the GPU's idle-to-full step happens alone. |
| `GUFO_NPU_ISOLATE` | unset | `1` gives the NPU private buffers and copies through the CPU. Slower; for debugging. |
| `GUFO_NPU_STATS` | unset | Print wall-clock accounting of the NPU path. |
| `GUFO_NPU_TRACE` | unset | Path of a synchronous event log, one line per NPU event. |

## Qwen3.8-27B and Qwen3.6-35B-A3B

These models use one runtime-K program for every reduction length
(`tools/npu/runtime_k`, K = 2048, 2560, 4096, 5120, 6144 and 17408). The NPU
computes the leading output columns of a projection in 1024-column slices while
the GPU computes the rest, and the GPU adds or reads the result in place. A chunk
needs 1536 to 4096 tokens; the program always runs 4096 rows, so a shorter chunk
gets proportionally fewer slices.

| Variable | 27B | 35B-A3B | Effect |
| --- | --- | --- | --- |
| `GUFO_QWEN_NPU_FFN_SLICES` | 7 (6 for a lone chunk) | 0 | FFN gate/up slices on the NPU, 0 to 16 |
| `GUFO_QWEN_NPU_DOWN_SLICES` | 1 | 0 | `ffn_down` slices, 0 to 5 |
| `GUFO_QWEN_NPU_QKV_SLICES` | 4 | 6 | linear-attention `attn_qkv` slices, 0 to 10 |
| `GUFO_QWEN_NPU_GATE_SLICES` | 2 | 2 | linear-attention `attn_gate` slices, 0 to 6 |
| `GUFO_QWEN_NPU_SSM_OUT_SLICES` | 2 | 1 | linear-attention output projection slices, 0 to 5 |
| `GUFO_QWEN_NPU_ATTN_Q_SLICES` | 4 | 3 | full-attention `attn_q` slices, 0 to 12 |
| `GUFO_QWEN_NPU_ATTN_O_SLICES` | 2 | 1 | full-attention `attn_output` slices, 0 to 5 |
| `GUFO_QWEN_NPU_ATTN` | on | on | `0` keeps the full-attention layers on the GPU |

### Chunk pairing

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_QWEN_LOCKSTEP` | on | `0` runs chunks one at a time. Pairs of chunks advance layer by layer so the NPU waits of one overlap the GPU work of the other (bit-identical to the one-at-a-time order). |
| `GUFO_QWEN_LOCKSTEP_YIELDS` | 4 (27B), 3 (MoE) | Stages that hand over to the other chunk: 1 attention-linear, 2 output projection, 4 FFN gate/up, 8 down. |
| `GUFO_QWEN_PAIR_MIN` | 5000 | A prompt of at least this many tokens is split into a pair of halves (or two full chunks) for lock-step. |

### 35B-A3B prefill and speculation

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_QWEN_MOE_PREFILL` | on | `0` runs prefill through the per-token decode kernels, the slow reference path. |
| `GUFO_QWEN_ROUTER_SPLIT` | on | `0` computes the router logits with the plain fp32 GEMM instead of the split-bf16 one. |
| `GUFO_MTP_VOCAB` | 65536 | Token ids the MTP draft head scores on the 35B-A3B (the first ids are the most frequent); `0` scores the full vocabulary. |

## Qwen3.8 Flash-Next decode and residual precision

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_DECODE_COPY` | `balanced` | Decode reads Q4_1 copies of `ssm_in`, `ssm_out`, `attn_qkv` and the shared expert (+19% plain, +21% MTP decode, perplexity +1.3%). `q4` copies every large matrix, `q5` uses Q5_1, `off` decodes from the exact Q8_0 weights. |
| `GUFO_DECODE_COPY_CLASSES` | per mode | Comma list of `ssm_in`, `ssm_out`, `attn_qkv`, `attn_out`, `hc`, `shexp` to copy. |
| `GUFO_F16_RESIDUAL` | on with `GUFO_TWO_BANK=1`, else off | Keeps the prefill residual in F16 between combines (perplexity +0.5%; +1.5% prefill with the NPU, no gain GPU-only). `0` or `1` overrides. |

## Examples

NPU on, benchmark:

```sh
GUFO_NPU_DIR=tools/npu build/release/gufo bench -m <model.gguf> -p 16384 -n 1 -r 3
```

Same, GPU-only: leave `GUFO_NPU_DIR` unset.

Only the z rows on the NPU:

```sh
GUFO_NPU_DIR=tools/npu GUFO_NPU_GATE_SLICES=0 GUFO_NPU_SHEXP=0 build/release/gufo serve llm ...
```

Smaller prefill tile for less memory:

```sh
GUFO_PREFILL_TILE=4096 scripts/run-gufo.sh
```
