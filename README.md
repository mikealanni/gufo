# gufo on Strix Halo: GPU + XDNA2 NPU inference

Fast local inference for AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S
`gfx1151`), with prefill shared between the GPU and the Ryzen AI NPU and
speculative decoding on top, for Qwen, Flint and GLM-5.3 Flash models. Fork of [gufo-org/gufo](https://github.com/gufo-org/gufo) (MIT).

| Model | Weights tested | NPU prefill | Speculative decode |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | UD-Q4_K_XL | linear-attention and attention projections | MTP (embedded head), DFlash2, auto |
| Qwen3.8-27B | UD-Q4_K_XL | FFN, attention and linear-attention projections | DFlash2, MTP, auto |
| Qwen3.8 Flash-Next | UD-IQ4_XS | whole projections, two-bank lock-step | DFlash2 (default), MTP |
| Flint | UD-Q4_K_XL | shared expert and QKV, two-bank pipeline | prompt lookup (no draft model) |
| GLM-5.3 Flash | AJ-IQ2_XXS | KDA q/k/v and shared expert, two-bank lock-step | DFlash2 with prompt-lookup drafts |

## Speed

Measured with `gufo bench` (three runs per prefill cell) and, for decode on real
text, with greedy chat requests to `gufo serve`. NPU numbers need the setup in
[Try it](#try-it); GPU-only numbers use the same binary without `GUFO_NPU_DIR`.

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

`gufo bench -n 128` plain decode: 60.6 tok/s.

### Qwen3.8-27B

| Prefill, tok/s | GPU only | GPU + NPU |
| --- | --- | --- |
| `-p 2048` | 654 | **715** (+9.3%) |
| `-p 4096` | 636 | **795** (+25.0%) |
| `-p 8192` | 629 | **806** (+28.2%) |

| Decode, tok/s | plain | DFlash2 | auto |
| --- | --- | --- | --- |
| code | 12.3 | **47.6** | 43 |
| prose | 12.3 | **24.8** | 24 |
| JSON | 12.3 | **62.2** | 53 |

### Qwen3.8 Flash-Next

| Prefill, tok/s | GPU only | GPU + NPU (two-bank) |
| --- | --- | --- |
| `-p 8192` | 1,814 | **2,128** (+17.3%) |
| `-p 16384` | 1,786 | **2,142** (+19.9%) |
| `-p 32768` | 1,760 | **2,109** (+19.8%) |

| Decode, tok/s | plain | MTP | DFlash2 |
| --- | --- | --- | --- |
| code | 36.6 | 76.5 | **109.6** |
| prose | 36.5 | 47.2 | **51.8** |
| JSON | 36.7 | 99.8 | **118.7** |

The DFlash2 column was re-measured with the confidence cut described under
speculative decoding (greedy, the project's probe prompts: 4 code, 6 prose and the
JSON request). The plain and MTP columns come from an earlier prompt set, so read
differences under about 5% as noise. DFlash2 also skips the MTP head's prompt
priming, which is why its prefill is about 3% faster. Math prompts decode at
about 95 tok/s and copy-heavy prompts at about 127 tok/s.

Grammar-constrained turns (tool calls) decode at 90 to 93 tok/s, up from 59 to 61
before their draft verification stayed on the GPU.

Prompt passes of 1,024 to 4,095 tokens (short prompts and the tail of a long one)
also run on the NPU, through the runtime-M program: prefill is +1% to +5% from
about 3,000 tokens upward and unchanged below, and the text matches the previous
build on 27 of 28 regression prompts (the other is a paraphrase at a near tie).

`gufo bench -n 96`: 35.9 tok/s plain, 54.2 tok/s with MTP (on the benchmark's
own repetitive context, where drafts are accepted easily).

In the two-bank NPU mode the prefill residual stays in F16, worth about 1.5%
(+0.5% perplexity); `GUFO_F16_RESIDUAL=0` keeps it in F32 (about 2,085 tok/s at
`-p 16384`). GPU-only prefill stays exact. The first NPU run after the machine
has been idle can be much slower, so discard it when benchmarking.

Prefill depends on the text as well as the length: code, markdown and prose
route to experts differently, so the same length can vary by 20% or more.

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

`gufo bench -n 128` plain decode: 73.6 tok/s; `gufo serve`: 72.8 tok/s. Flint has
no MTP head and no drafter yet, so only prompt lookup (lossless) applies, and it
helps when the output copies earlier text. Long context: 131k-token prefill takes
89 s dense and 48 s (2,716 tok/s) with `FLINT_SPARSE_ATTN=1`, which is
approximate beyond about 19k tokens. Details, numerics and the prefill history
are in [docs/models/flint](docs/models/flint/README.md).

### GLM-5.3 Flash

A 321B-parameter mixture-of-experts model (about 14B active per token; 45 layers,
34 KDA and 11 DSA, 288 experts with 8 routed) in the 81 GiB AJ-IQ2_XXS quantization.

| Prefill, tok/s | GPU only | GPU + NPU (two-bank) |
| --- | --- | --- |
| `-p 2048` | 634 | 626 (-1.3%) |
| `-p 4096` | 635 | 631 (-0.6%) |
| `-p 8192` | 630 | **740** (+17.5%) |

The NPU pair path needs two full 4096-token banks, so prompts under 8192 tokens run
the GPU path and match GPU-only speed; 16384 tokens measure 735 tok/s with the NPU.

| Decode, tok/s | plain | DFlash2 |
| --- | --- | --- |
| code | 18.9 | **29.7** |
| prose | 18.9 | **22.5** |
| JSON | 18.9 | **26.0** |

Greedy chat requests to `gufo serve`, 256 tokens. `gufo bench -n 128` plain decode:
18.8 tok/s. Prose is the expository prompt; a creative-writing prompt gains nothing
(18.7). Decode reads about 9 GB of weights per token, mostly the Q6_K attention and
shared-expert projections, so plain decode is bandwidth-bound; drafting helps when
the next tokens are predictable. Serving uses a 131,072-token context and one
session at a time. Start it with `GUFO_KIND=glm scripts/run-gufo.sh`.

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
| GLM-5.3 Flash, NPU two-bank prefill | logits cosine 0.994 against the GPU-only chunked path, same top-1; GPU-only prefill matches sequential decode at cosine 0.989 |
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
  two 4096-token banks that run layer by layer in lock-step; the runtime-K program
  in `tools/npu/runtime_k` serves them.
- **Flint:** the shared expert and QKV projections, in two banks of up to 8192
  tokens; the runtime-M/K kernel in `tools/npu/runtime_mk` serves any chunk size
  without rebuilding the xclbin.

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
