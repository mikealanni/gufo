# Qwen3.8 27B

Dense text/image model on gfx1151. Production target GGUFs are **UD-Q4_K_XL**
(16.35 GiB weights) and **UD-Q8_K_XL** (29.30 GiB); request state and an optional
draft/projector need additional memory. BF16 targets are reference-only.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf \
  --local-dir models/qwen3.8-27b
nix develop -c hf download z-lab/Qwen3.8-27B-DFlash2-GGUF \
  Qwen3.8-27B-DFlash2-Q4_K_M.gguf --local-dir models/qwen3.8-27b
nix build
MODEL=models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
DRAFT=models/qwen3.8-27b/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative dflash2 \
  --dflash-model "$DRAFT" --sessions 2 --context 32768
```

On a Strix Halo with the XDNA2 NPU, build with `-DGUFO_ENABLE_NPU=ON` and set
`GUFO_NPU_DIR=tools/npu` to share prefill between the GPU and the NPU (prefill
636 to 806 tok/s, see [Benchmarks](BENCHMARKS.md#npu-prefill) and the
[setup](../../../README.md#try-it)).

Substitute `Qwen3.8-27B-UD-Q8_K_XL.gguf` for Q8. Targets and drafts need not
have matching precision. Use the Q4_K_M DFlash2 draft with either target.

| Mode | Selection | Behavior |
| --- | --- | --- |
| AR | No speculative option | Target-only generation. |
| DFlash2 | `--speculative dflash2 --dflash-model PATH` | Adaptive default; `--draft-tokens` caps proposals, `--draft-policy fixed` is a comparison mode. |
| MTP | `--speculative mtp --mtp-model PATH` | The Q4_0 MTP head; accepted drafts are rare on this model, so DFlash2 is far faster. |
| Auto | `--speculative auto --dflash-model PATH --mtp-model PATH` | Both drafts stay in step; each request proposes with whichever commits more tokens per second. |

Greedy speculation must match AR. Sampled modes preserve the target policy,
but AR and speculation need not share the same-seed sequence. See the
[quality contract](QUALITY.md). Thinking/template controls and HTTP
sampling defaults are described in [the server guide](../../SERVER.md#reasoning-controls).

## Images

Use the matching `mmproj-BF16.gguf` beside the target or pass `--mmproj PATH`.
Vision weights upload lazily. Repeat `--image` in `prompt` or `chat`; chat
attaches them to the first user turn. Images count toward context capacity.

```sh
./result/bin/gufo prompt --model "$MODEL" --image photo.jpg \
  --prompt "Describe this image." --temperature 0
```

Send an image to the server:

```python
import base64, json, urllib.request
image = base64.b64encode(open("photo.jpg", "rb").read()).decode()
body = {"model": "vision-test", "temperature": 0, "max_tokens": 64,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}},
            {"type": "text", "text": "Describe this image."}]}]}
request = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(request).read().decode())
```

Start that server with `--served-model-name vision-test`. Ordered text/image
parts, multiple images and later turns work with AR/DFlash2 and caches.
PNG/JPEG data URLs and public HTTPS are supported (`detail: auto`); requests
share a 20 MiB encoded-byte, 16-image and 15-second download budget. Each image
is capped at 32 megapixels; private/loopback/link-local destinations are rejected.
Official dynamic resizing allows 64–16384 merged image tokens.

Image numbering is off by default; `--add-vision-id` or
`chat_template_kwargs: {"add_vision_id": true}` adds `Picture N:` prefixes.
Cache identity includes processed pixels, placement, preprocessing version and
projector contents. See [vision checks and limits](QUALITY.md#vision).

## Int4 prefill (rotated 4-bit artifact)

With the rotated int4 artifact loaded (`GUFO_IU4_ARTIFACT`, found automatically by
`~/run-gufo.sh`), the dense projections run as int4 x int4 WMMA and the FFN gate/up
slices go to the NPU. Measured 2026-11-21 on the main binary: prefill 1,335 tok/s
at 8K (806 with the Q4_K_XL GGUF and the NPU), plain decode 14.7 tok/s. HumanEval
156/164 against 158/164 for the GGUF; equal NLL. See `.agents/skills/iu4-prefill`.

## NPU prefill

With the XDNA2 NPU set up as in the [repository README](../../../README.md#try-it),
`GUFO_NPU_DIR=tools/npu` moves the leading output columns of the FFN, attention
and linear-attention projections to the NPU while the GPU computes the rest.
Pairs of 4096-token chunks (or two halves of a prompt of 5000 tokens or more)
advance in lock-step so the NPU waits of one chunk overlap the GPU work of the
other. Prefill rises by 10% to 28%; decode is unchanged. Options are in
[docs/NPU.md](../../NPU.md#qwen38-27b-and-qwen36-35b-a3b).

## Block-sparse attention (opt-in)

`GUFO_QWEN_SPARSE_ATTN=1` replaces full attention beyond 8192 tokens with block-sparse
attention: each query reads 2 sink blocks, a 32-block local window and the 256
best-scoring older 64-token blocks (per-block Quest key bounds, scored per head).
Prefill, plain decode and speculative verification use it. It is approximate:
`GUFO_QWEN_SPARSE_TOPK`, `GUFO_QWEN_SPARSE_LOCAL`, `GUFO_QWEN_SPARSE_SINK` and
`GUFO_QWEN_SPARSE_MIN_CONTEXT` tune it, and it is off unless the flag is set.
`scripts/run-gufo.sh` enables it for the 27B and 35B-A3B (`GUFO_SPARSE=0` disables).
Needle-in-a-haystack at 32k, 64k and 120k matches dense attention on every case dense passes.

| 27B at 128k depth, NPU on | Dense | Sparse |
| --- | --- | --- |
| pp4096 | 318 | 545 |
| tg, plain decode | 8.5 | 10.7 |
| tg, MTP | 8.7 | 10.5 |
| tg, DFlash2 | 18.0 | 17.7 |
