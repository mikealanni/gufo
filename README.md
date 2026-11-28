# gufo on Strix Halo: GPU + XDNA2 NPU inference

Fast local inference for AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S
`gfx1151`), with prefill shared between the GPU and the Ryzen AI NPU and
speculative decoding on top, for Qwen, Inkling, Flint, GLM-5.3 Flash, Laguna, Ling and Raon models. Fork of [gufo-org/gufo](https://github.com/gufo-org/gufo) (MIT).

## What's new

- **jev: agent guard and opencode plugin.** A proxy in front of gufo that stops
  the server before heavy GPU/NPU/RAM jobs, restarts the same model afterwards,
  switches models with `/jev switch`, picks reasoning effort per request (or pins
  it), and recovers turns that fail when the server restarts. The plugin adds
  `/jev` to opencode and works with `opencode --standalone`, so one session
  leaves no background service behind. A server started with `GUFO_THINK=off`
  keeps thinking off for every request. See [docs/JEV.md](docs/JEV.md) and the
  [setup guide](docs/JEV-SETUP.md).
- **Raon-1 mini** on the GPU with DFlash2 speculation (default) and NPU prefill. Two
  banks pipeline the prefill, attention is block-sparse above 16K tokens with the system
  prompt and tool definitions always attended, and two requests (`JOBS=2`) decode at the
  same time. Reasoning effort maps to a thinking budget. `gufo bench`: prefill 2,935 tok/s
  at 4K and 2,770 at 8K, plain decode 52.6 tok/s at 4K depth and 42.7 at 128K. See
  [docs/models/raon1](docs/models/raon1/README.md).
- **Inkling 2 Mini** (26.4B MoE, 3.4B active, looped middle layers) on the GPU with
  DFlash2 speculation (default) and NPU prefill. Up to 256K context with block-sparse
  attention, tool calls, a reasoning channel, and the same RAM and disk
  continuation cache as the other models. Prefill 2,851 tok/s at 8K with the NPU; decode 60.6 tok/s on code and 58.3
  on JSON at temperature 1. See [docs/models/inkling](docs/models/inkling/README.md).
- **Qwen3.8-27B on int4 WMMA plus NPU.** The dense projections run as int4 x int4
  WMMA on the GPU with the FFN split across the NPU: prefill 1,335 tok/s at 8K
  (806 with the Q4_K_XL GGUF and the NPU), plain decode 14.7 tok/s, HumanEval
  156/164 against 158/164 for the GGUF, equal NLL. DFlash2 decode as before.
- **Standalone NPU engine.** `gufo-gen` runs dense and hybrid models and the jev
  decision model on the NPU alone, with no GPU involved
  (see [Pure NPU engine](#pure-npu-engine)).
- **Faster loads.** Loaders upload tensors through one shared pinned staging
  pipeline (`hip::TensorUpload`); Inkling loads in about 15 s warm.

| Model | Weights tested | NPU prefill | Speculative decode |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | UD-Q4_K_XL | linear-attention and attention projections | MTP (embedded head), DFlash2, auto |
| Qwen3.8-27B | UD-Q4_K_XL | FFN, attention and linear-attention projections | DFlash2, MTP, auto |
| Qwen3.8 Flash-Next | UD-IQ4_XS | whole projections, two-bank lock-step | DFlash2 (default), MTP |
| Inkling 2 Mini | UD-Q4_K_XL | q/gate/k/v and attention output, two-bank | DFlash2 (default), DSpark |
| Flint | UD-Q4_K_XL | shared expert and QKV, two-bank pipeline | prompt lookup (no draft model) |
| GLM-5.3 Flash | AJ-IQ2_XXS / RCO-152E Q3_K_M | KDA q/k/v and shared expert, two-bank lock-step | DFlash2 with prompt-lookup drafts |
| Laguna S 2.1 | UD-Q4_K_M | shared expert, two-bank pipeline | DFlash v1 (block-diffusion) |
| Ling 3.2 Air | Q4_K_M | KDA q/k/v and shared expert, two-bank pipeline | MTP (embedded layer) |
| Raon-1 mini | Q4_K_M | q/k/v/g, MLA gate and shared expert, two-bank pipeline | DFlash2 |

## Speed

Measured with `gufo bench` (three runs per prefill cell) and, for decode on real
text, with greedy chat requests to `gufo serve`. NPU numbers need the setup in
[Try it](#try-it); GPU-only numbers use the same binary without `GUFO_NPU_DIR`.

### Inkling 2 Mini

| Prefill, tok/s | GPU only | GPU + NPU |
| --- | --- | --- |
| `-p 2048` | 2,499 | **2,499** |
| `-p 4096` | 2,580 | **2,737** (+6.1%) |
| `-p 8192` | 2,466 | **2,822** (+14.4%) |
| `-p 16384` | 2,254 | **2,516** (+11.6%) |

| Decode, tok/s (temperature 1) | plain | DFlash2 (default) |
| --- | --- | --- |
| code | 33 | **60.6** |
| JSON | 33 | **58.3** |
| prose | 33 | **36.4** |

Details: [docs/models/inkling](docs/models/inkling/)

### Qwen3.6-35B-A3B

| Prefill, tok/s | GPU only | GPU + NPU |
| --- | --- | --- |
| `-p 2048` | 3,034 | **3,066** (+1.1%) |
| `-p 4096` | 3,013 | **3,276** (+8.7%) |
| `-p 8192` | 2,862 | **3,134** (+9.5%) |

| Decode, tok/s | plain | MTP | DFlash2 | auto |
| --- | --- | --- | --- | --- |
| code | 61 | 111 | **140** | 138 |
| prose | 61 | **72** | 59 | 68 |
| JSON | 61 | 122 | **159** | 156 |

Details: [docs/models/qwen3.6-35b-a3b](docs/models/qwen3.6-35b-a3b/)

### Qwen3.8-27B

| Prefill, tok/s | GPU only | GPU + NPU |
| --- | --- | --- |
| `-p 2048` | 654 | **715** (+9.3%) |
| `-p 4096` | 636 | **795** (+25.0%) |
| `-p 8192` | 629 | **806** (+28.2%) |

| Prefill, tok/s (rotated int4) | Q4_K_XL GGUF + NPU | Rotated int4 + NPU |
| --- | --- | --- |
| `-p 8192` | 806 | **1,335** (+66%) |

| Decode, tok/s | plain | DFlash2 | auto |
| --- | --- | --- | --- |
| code | 12.3 | **47.6** | 43 |
| prose | 12.3 | **24.8** | 24 |
| JSON | 12.3 | **62.2** | 53 |

Details: [docs/models/qwen3.8-27b](docs/models/qwen3.8-27b/)

### Qwen3.8 Flash-Next

| Prefill, tok/s | GPU only | GPU + NPU (two-bank) |
| --- | --- | --- |
| `-p 8192` | 1,814 | **2,128** (+17.3%) |
| `-p 16384` | 1,786 | **2,142** (+19.9%) |
| `-p 32768` | 1,760 | **2,109** (+19.8%) |
| `-p 65536` | not measured | **2,046** |
| `-p 131072` | not measured | **1,990** |
| full 1,048,044-token prompt | not measured | **1,633** |

| Decode, tok/s | plain | MTP | DFlash2 |
| --- | --- | --- | --- |
| code | 36.6 | 76.5 | **109.6** |
| prose | 36.5 | 47.2 | **51.8** |
| JSON | 36.7 | 99.8 | **118.7** |

Details: [docs/models/qwen3.8-flash-next](docs/models/qwen3.8-flash-next/)

### Flint

| Prefill, tok/s | GPU only | GPU + NPU |
| --- | --- | --- |
| `-p 2048` | 3,882 | **4,150** (+6.9%) |
| `-p 4096` | 3,933 | **4,007** (+1.9%) |
| `-p 8192` | 3,724 | **4,143** (+11.3%) |

| Decode, tok/s | plain | prompt lookup |
| --- | --- | --- |
| code | 72.6 | **74.9** |
| prose | 72.6 | 72.3 |
| JSON | 72.6 | **73.3** |

Details: [docs/models/flint](docs/models/flint/)

### GLM-5.3 Flash

| Prefill, tok/s (RCO-152E) | GPU only | GPU + NPU (two-bank) |
| --- | --- | --- |
| `-p 2048` | 632 | 636 |
| `-p 4096` | 623 | **714** (+14.6%) |
| `-p 8192` | 617 | **706** (+14.4%) |
| `-p 16384` | 610 | **699** (+14.6%) |
| `-p 4096 -d 32768` | 584 | **665** (+13.9%) |
| `-p 4096 -d 65536` | 560 | **633** (+13.0%) |
| `-p 4096 -d 131072` | 511 | **571** (+11.8%) |

| Prefill, tok/s (AJ-IQ2_XXS, 81 GiB) | GPU only | GPU + NPU (two-bank) |
| --- | --- | --- |
| `-p 4096` | 616 | **689** (+11.9%) |
| `-p 8192` | 613 | **690** (+12.6%) |

| Decode, tok/s | plain | DFlash2 |
| --- | --- | --- |
| code | 18.9 | **29.7** |
| prose | 18.9 | **22.5** |
| JSON | 18.9 | **26.0** |

Details: [docs/models/glm53-flash](docs/models/glm53-flash/)

### Laguna S 2.1

| Prefill, tok/s | GPU only | GPU + NPU |
| --- | --- | --- |
| `-p 2048` | 1,145 | **1,277** (+11.5%) |
| `-p 4096` | 1,241 | **1,253** (+1.0%) |
| `-p 8192` | 1,193 | **1,204** (+0.9%) |

| Decode, tok/s | plain | DFlash v1 |
| --- | --- | --- |
| code | 25.9 | **44.5** |
| prose | 25.9 | 27.9 |
| JSON | 25.9 | **45.4** |

Details: [docs/models/laguna](docs/models/laguna/)

### Ling 3.2 Air

| Prefill, tok/s | GPU only | GPU + NPU |
| --- | --- | --- |
| `-p 2048` | 1,828 | 1,821 (-0.4%) |
| `-p 4096` | 1,815 | 1,842 (+1.5%) |
| `-p 8192` | 1,701 | 1,854 (+9.0%) |
| `-p 16384` | 1,624 | 1,566 (-3.5%) |

| Decode, tok/s | plain | MTP, 2 drafts |
| --- | --- | --- |
| code, 4k context | 36.3 | **40.0** |
| JSON, 4k context | 35.8 | **42.5** |
| prose, 4k context | 36.0 | 35.6 |

| Decode via `gufo serve`, tok/s | plain | MTP, 2 drafts |
| --- | --- | --- |
| code, 4k context | 32.7 | 34.3 |
| JSON, 4k context | 35.9 | **40.5** |
| prose, 4k context | 36.1 | 34.4 |

Details: [docs/models/ling-3.2](docs/models/ling-3.2/)

### Raon-1 mini

| `gufo bench`, real text | pp, tok/s | tg, tok/s (plain decode) |
| --- | --- | --- |
| 4K | 2,935 | 52.6 |
| 8K | 2,770 | 51.4 |
| 16K | 2,762 | 48.9 |
| 32K | 2,539 | 44.4 |
| 64K | 2,502 | 43.8 |
| 128K | 2,499 | 42.7 |

pp is a fresh prompt of that length on GPU + NPU with two banks; tg is decode at that depth.
Speculative decoding is not measured by `gufo bench` for this model.

Above 16K tokens attention is block-sparse, with the system prompt and tool definitions always
attended while decoding. Two simultaneous requests (`JOBS=2`) run at about two thirds of the
single-request speed each. Details: [docs/models/raon1](docs/models/raon1/README.md)

## Quality

Output is exact by default, with two documented exceptions, and the NPU is
opt-in. Numbers below compare each build with llama.cpp on the CPU (next-token
KL divergence and perplexity of the true continuation) or with the same build
without the NPU.

| Path | Measured effect |
| --- | --- |
| 27B, NPU prefill | KL against GPU-only 6e-5 to 3e-4, top-1 agreement 99.0% to 99.6%, perplexity change under 0.01% |
| 35B-A3B, NPU prefill | perplexity change under 0.03% against llama.cpp; routing near-ties give 0.002 to 0.003 KL for any numeric change, NPU included |
| Flash-Next, NPU slice and two-bank modes | KL against GPU-only 0.003 to 0.02 over real-text windows; needle-in-a-haystack 21 of 21 |
| Flash-Next decode copies, `balanced` (default) | perplexity +1.3% against llama.cpp, decode +19% (plain) and +21% (MTP) |
| Flash-Next decode copies, `off` | exact Q8_0 weights; plain 30.5 tok/s, MTP 44.9 tok/s |
| Flash-Next F16 residual (on with the two-bank NPU mode, `GUFO_F16_RESIDUAL=0/1` overrides) | perplexity +0.5%, prefill +1.5% with the NPU, no gain on GPU-only |
| GLM-5.3 Flash, NPU two-bank prefill | logits cosine 0.994 against the GPU-only chunked path, same top-1; GPU-only prefill matches sequential decode at cosine 0.989; RCO-152E two-bank output is byte-identical to the GPU-only path over 64 greedy tokens on real prose |
| GLM-5.3 Flash, IQ2_XXS weights | HumanEval 157 of 164 (reasoning off); 50 of 75 on the DS4 capability set with reasoning (GPQA 13/24, SuperGPQA 18/25, AIME 18/25); not yet compared with llama.cpp |
| Speculative decoding | the target verifies every draft; greedy text can differ from plain decode only at near-tie tokens |

Details and methods are in the per-model `QUALITY.md` files under
[docs/models](docs/models/README.md).

## How the NPU is used

The NPU runs block-float 8-bit GEMMs for the leading output columns of a
projection while the GPU computes the rest of the same projection, then the GPU
reads the NPU result directly. One runtime-K program serves every reduction length
(2048 to 17408), so the same kernel works for all three models.

- **Qwen3.8-27B:** FFN gate/up and down, attention q and output, and the
  linear-attention qkv, gate and output projections. Pairs of chunks run in
  lock-step so one chunk's NPU waits overlap the other's GPU work; prompts of
  5000 tokens or more split into two halves for the same reason, and chunks of
  1536 to 4095 tokens use proportionally fewer slices.
- **Qwen3.6-35B-A3B:** the same for the dense projections. The routed experts
  stay on the GPU, which runs them as grouped F16 matrix-core GEMMs. The router
  runs in fp32 accuracy through one split-bf16 GEMM.
- **Qwen3.8 Flash-Next:** whole `ssm_in`/`attn_qkv` projections and the shared
  expert, in two banks of 8192 tokens that run layer by layer in lock-step;
  passes of 1,024 to 4,095 tokens run in groups of their own size through the
  runtime-M program in `tools/npu/runtime_mk`.
- **GLM-5.3 Flash:** the KDA q/k/v projections and the shared expert gate/up, in
  two 2048-token banks that run layer by layer in lock-step; the runtime-M/K
  program in `tools/npu/runtime_mk` serves them, so every prompt of 4096 tokens
  or more pairs at any depth.
- **Flint:** the shared expert and QKV projections, in two banks of up to 8192
  tokens; the runtime-M/K kernel in `tools/npu/runtime_mk` serves any chunk size
  without rebuilding the xclbin.

## Pure NPU engine

Separate from the GPU+NPU offload above, `tools/npu` carries a GPU-free engine
that runs small models entirely on the XDNA2 NPU. The host only dispatches
instruction streams and does token bookkeeping; the GPU is not used at all.

- **`gufo-gen`** (`tools/npu/gen`) loads a dense GQA transformer straight from a
  GGUF: the config comes from the metadata, weights are dequantized from any GGUF
  type the repo can decode, decode runs RTN int5 programs and prefill runs
  block-float 16 programs at 1024 rows per pass. Missing instruction streams are
  generated on first use. It also runs the Qwen3.5-family decision model used by
  jev.
- **LFM3** has its own HIP-free build (`gufo-npu`, `GUFO_BUILD_NPU_ENGINE=ON`,
  `ENGINE_ENABLE_HIP=OFF`) with fused-FFN chains and draft heads.

| Model (Q6_K GGUF) | Prefill 1k / 4k (tok/s) | Decode at 1k (tok/s) |
| --- | --- | --- |
| MiniCPM5-1B | 2323 / 3204 | 35 (43 at short context) |
| Qwen3-0.6B | 1918 / 2198 | 29 |
| Qwen3-1.7B | 1567 / 1812 | 20 |
| MiniCPM5-2B | 1228 / 1626 | 17 |

LFM3 1.7B decodes at about 30 tok/s plain and 36 with draft heads at short
context.

Limits: FFN width must be a multiple of 512 and hidden size of 64; partial rope
and attention bias are unsupported; NPU attention covers 16 heads of 128
dimensions with 1, 2 or 4 KV heads, and other shapes (and contexts up to 256
tokens) use host attention. Decode weights are int5 round-to-nearest from the
GGUF, which is the weak spot for quality and decode speed.

The goal is every tensor operation on the NPU in both prefill and decode. Norms,
rope and some attention paths still run on the host as marked scaffolds, each
with a named NPU replacement task; a stage only counts as done once it has no
host tensor math. All NPU runs go through the shared lock script and binaries
must be named `gufo*` so the clock service recognises them. Instruction
streams, xclbins and kernels live in `tools/npu/decode_int5` and `tools/npu/gen`.

## Speculative decoding

| Flag | Effect |
| --- | --- |
| `--speculative mtp --mtp-model embedded` | MTP head stored in the checkpoint (35B-A3B), or a separate MTP file (`--mtp-model PATH`) |
| `--speculative dflash2 --dflash-model PATH` | DFlash2 draft model |
| `--speculative auto --dflash-model PATH --mtp-model embedded` | both drafts; each request uses whichever commits more tokens per second, with sampled requests on DFlash2 |

Flash-Next defaults to DFlash2 (`--speculative dflash2 --dflash-model <Qwen3.8-Flash-Next-DFlash2 GGUF>`),
which drafts a block of up to seven tokens in one pass and needs no MTP priming of the prompt; MTP
(`--speculative mtp --mtp-model <shared Q8_0 sidecar>`) remains available. The DFlash2 head covers the
first 163,840 vocabulary rows plus the special tokens (`GUFO_DFLASH_VOCAB=0` reads the whole vocabulary).
For greedy requests the drafter always drafts the full block and the target verifies
only the prefix whose running product of selector confidences stays at or above 0.7
(at least one token): +8% to +9% on prose and math against the adaptive length
alone, with identical output. Verifying all seven every round without the cut is slower (prose about 38 tok/s).
Sampled requests keep the adaptive length.

GLM-5.3 Flash uses a DFlash2 drafter (`--speculative dflash2 --dflash-model <GLM-5.3-Flash-DFlash2 GGUF>`)
plus lossless prompt-lookup drafts from the context; the checkpoint has no MTP head.
Each extra verified row reads another set of experts, so the draft length adapts
to the acceptance it sees.

The 35B-A3B runs MTP at three draft tokens (verifying more rows reads more
experts); DFlash2 wins on code and JSON, MTP on prose, and auto lands within a few
percent of the better one on each. On the 27B DFlash2 is far better than its MTP
head, so use `--speculative dflash2` there (auto costs 5% to 15% against it).

## Try it

1. Build with the NPU enabled:
   `cmake --preset release -DGUFO_ENABLE_NPU=ON -DGUFO_XRT_INCLUDE_DIR=<xrt include dir>`
   then `cmake --build --preset release --parallel 4`. The compiled NPU kernel
   and its instruction streams are in [tools/npu](tools/npu), so nothing else has
   to be built.
2. Hold the GPU fabric clock at its top level (2000 MHz on this chip). This is
   required: without it the GPU and the NPU together can hang the machine. It is
   not a guarantee: [ROCm/TheRock#8709](https://github.com/ROCm/TheRock/issues/8709)
   is still open, and some hosts (a Beelink GTR9 Pro, a CachyOS workstation) hung
   with the clock held, with no machine-check record in the journal. A hang needs a
   power cycle, so start with short runs and save your work.
   `sudo tools/npu/fabric-clock.sh hold` sets `power_dpm_force_performance_level`
   to `manual` and `pp_dpm_fclk` to the top level; `tools/npu/fabric-clock.sh status`
   must print `held` (`release` undoes it, a reboot also resets it). To hold it
   at every boot and after resume, run that command from a systemd oneshot unit
   wanted by `multi-user.target` and the `suspend`, `hibernate` and `hybrid-sleep`
   targets.
3. Put the NPU in performance mode:
   `sudo /opt/xilinx/xrt/bin/xrt-smi configure --pmode performance`
   (`balanced` works too and is within 1%; do not use `turbo`). The mode resets
   on every driver load, so set it after each boot.
4. Keep the NPU out of runtime suspend so its power messages cannot collide with
   other power-state writes:
   `echo on | sudo tee /sys/bus/pci/devices/<NPU PCI address>/power/control`
   (the address is the one `xrt-smi examine` prints, e.g. `0000:c7:00.1`).
5. Measure. With nothing else running:

   ```sh
   export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/opt/rocm/lib
   # Qwen3.8-27B or Qwen3.6-35B-A3B
   GUFO_NPU_DIR=tools/npu build/release/gufo bench -m <model.gguf> -p 4096,8192 -n 128 -r 3
   # Qwen3.8 Flash-Next
   GUFO_NPU_DIR=tools/npu GUFO_TWO_BANK=1 GUFO_NPU_WHOLE=1 build/release/gufo bench -m <model.gguf> -p 16384 -n 1 -r 3
   build/release/gufo bench -m <model.gguf> -p 4096 -n 128 -r 3   # NPU off
   ```

   The NPU run prints which projections it took while loading. For the
   Flash-Next server, run
   `GUFO_MODEL=<first shard> GUFO_DFLASH=<dflash2 draft> scripts/run-gufo.sh`
   (`GUFO_MTP=<mtp sidecar> GUFO_FN_SPEC=mtp` selects MTP). It
   enables the NPU only when the fabric clock is held and the NPU is not in turbo
   mode (`GUFO_NPU=0` forces GPU-only) and reads the options listed at the top of
   the script.
6. Serve with speculation, for example the 35B-A3B:

   ```sh
   GUFO_NPU_DIR=tools/npu build/release/gufo serve llm --model <35b.gguf> \
     --speculative auto --dflash-model <dflash2.gguf> --mtp-model embedded
   ```

Every option is described in [docs/NPU.md](docs/NPU.md). Without
`GUFO_NPU_DIR` the build runs GPU-only.

## jev

`tools/jev/` holds a guard proxy and an opencode plugin that keep a local gufo
server usable next to heavy work: it stops gufo before heavy commands, restarts
the same model afterwards, switches models with `/jev switch`, and controls
reasoning effort. See [docs/JEV.md](docs/JEV.md); install steps, flags and every
environment variable are in [docs/JEV-SETUP.md](docs/JEV-SETUP.md).

## Build from source

```sh
cmake --preset release
cmake --build --preset release --parallel 4
./build/release/gufo serve llm --model /path/to/model.gguf
```

Needs a C++20 compiler, CMake 3.21+, Ninja, ROCm (HIP, hipBLAS, hipBLASLt,
rocBLAS, hipCUB, rocPRIM, rocWMMA) and ICU, libcurl, OpenSSL, libpng, libjpeg,
libwebp. `nix build` builds the same thing with a pinned toolchain.

## License

MIT, see [LICENSE](LICENSE). Adapted code and dependencies keep their notices in
[NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model
weights are not bundled.
