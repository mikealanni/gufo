# Laguna S 2.1

Mixture-of-experts text model (48 layers, hidden 3,072, 8 KV heads of 128, heads
per layer [48,72,72,72] repeating, 256 experts with 10 routed plus a shared
expert, sliding-window attention of 512 on 3 of every 4 layers) on gfx1151. The
tested weights are `Laguna-S-2.1-UD-Q4_K_M.gguf` (68 GB; Q8_0 attention,
Q4_K/Q5_K experts).

NPU kernel: `tools/npu/runtime_mk/` (K=3,072 instruction file built this port -
the shared XCLBIN needed no changes, see its README's "Adding a shape").

## Run

```sh
gufo prompt -m Laguna-S-2.1-UD-Q4_K_M.gguf -f prompt.txt -n 128 -t 0 -v
```

`--dflash-model laguna-s-2.1-DFlash-BF16.gguf --speculative dflash2 -d 3` adds
the v1 (block-diffusion) drafter. `GUFO_NPU_DIR=<repo>/tools/npu` uses the NPU
(shared-expert gate/up, banked by default); needs the fabric clock held and a
binary named `gufo*`.

## Speed (GPU, interleaved runs, one machine)

| | Prefill | Decode |
| --- | --- | --- |
| 2,048-token prompt | 1,275 tok/s (1,277 with NPU) | 26.0 (plain), 36.2 (DFlash `-d 3`) |
| 4,096 tokens | 1,249 (1,272 with NPU) | |
| 8,192 tokens | 1,198 (1,224 with NPU) | |
| 13,305 tokens | | 24.0 (plain, split-K attention) |
| 47,280 tokens | | 20.3 (plain, split-K attention) |
| 133,284 tokens | | 14.8 (plain, split-K attention) |

NPU gain is +0.2% to +2.4% depending on depth - shared-expert gate/up is a
small fraction of total per-layer FLOPs for this model's shape (72 query
heads), not a flat/large win the way it is for models with fewer heads. QKV
on the NPU is built (cross-layer submit/consume, bank-aware) but measured as
a net loss for Laguna specifically - the widest layers need 22 NPU round
trips per layer per bank (9 Q-slices + 1 K + 1 V, x2 for the hi/lo residual
pass), more than the available hide window absorbs. Left off by default
(`LAGUNA_NPU_QKV=1` opts in, not recommended). Full investigation, every
dead end, and the exact numbers behind each of the above:
[laguna-port-state.md](../../../src/models/laguna/laguna-port-state.md).

Decode flattened substantially at depth this port (split-K decode attention,
ported from Flash-Next/Flint's technique): 26 -> 19.8 -> 12.5 tok/s before,
26 -> 24.0 -> 20.3 -> 14.8 tok/s after, at comparable depths. Not literally
flat - SWA-layer and FFN/MoE kernel cost becomes a larger share of the step
once the split non-SWA attention layers' own cost shrinks.

## Block-sparse attention (opt-in)

`LAGUNA_SPARSE_ATTN=1` replaces full attention beyond the sink + local
window with block-sparse attention on the 12 full-attention layers only
(the 36 sliding-window layers are already bounded by `sliding_window=512`
and don't need it): each query reads 2 sink blocks, a 32-block local
window, and the 256 best-scoring older 64-token blocks (Quest bound,
per-block channel min/max). Exact below ~18.5k tokens of context (every
candidate fits inside the 256-block budget); approximate above it, same
accepted-divergence class as this port's other approximations.
Needle-in-a-haystack (a unique fact buried mid-document) matches dense on
every depth tested:

| Depth | Dense | Sparse | Gain |
| --- | --- | --- | --- |
| 41,327 tok | 937.6 tok/s, correct | 999.5 tok/s, correct | +6.6% |
| 61,108 tok | 835.3 tok/s, correct | 973.2 tok/s, correct | +16.5% |
| 100,616 tok | 694.6 tok/s, correct | 958.0 tok/s, correct | +37.9% |

Gain grows with depth (the candidate pool to prune grows with position).
Verified stacking cleanly with the NPU and two-bank pipeline (995.7 tok/s
at 61,108 tokens, all three on, no corruption) and across a 2,600-token
generation past the decode HIP-graph capture boundary (a real, separate
bug found and fixed this port - see
[laguna-port-state.md](../../../src/models/laguna/laguna-port-state.md)
for the full account, including three real selection bugs found when
this was first turned on). Off by default; `LAGUNA_SPARSE_TOPK=N`
overrides the 256-block budget.

## Options (environment)

| Variable | Effect |
| --- | --- |
| `GUFO_NPU_DIR=<repo>/tools/npu` | Use the NPU (shared-expert gate/up, banked by default) |
| `LAGUNA_NPU_NO_BANK=1` | NPU without the two-bank pipeline (same-call overlap only) |
| `LAGUNA_NPU_QKV=1` | Also offload QKV to the NPU - measured net loss, off by default |
| `LAGUNA_SPARSE_ATTN=1` | Block-sparse attention on the 12 full-attention layers - off by default |
| `LAGUNA_SPARSE_TOPK=N` | Candidate-block budget for sparse attention (default 256) |
| `LAGUNA_PREFILL_TILE=N` | Prefill chunk size (default 4,096); needs a matching NPU instruction
file if `GUFO_NPU_DIR` is set and N != 4,096 |

## Numerical notes

- Routed-expert activations are stored in F32 divided by 64 (the combine
  multiplies it back).
- The NPU path is not bit-identical to the GPU path (block-float, same as
  every other model's NPU offload in this project); real-generation output
  diverges from the GPU-only path after several tokens on real prose, the
  same accepted-divergence class as this port's other numeric changes (W8A8
  dense projections, the Q8_0 vector path, etc.) - confirmed NOT a bug after
  an earlier false alarm (see laguna-port-state.md's precision investigation
  section) found the "evidence" was itself measuring against an inexact
  reference and a near-tie-prone filler prompt.

## Status

Prefill and decode verified against greedy text and `--validate-prefill`.
DFlash v1 drafter working (coherent generation, real acceptance rate). NPU
shared-expert offload on by default; QKV offload built but off by default
(see Speed section). `gufo serve llm` works (smoke-tested: load, warm-up,
a real `/v1/chat/completions` request) but has no snapshot/disk-cache
reuse yet (the engine's `Session` has no save/restore) and rejects
`--speculative` (DFlash v1 is `gufo prompt`/`gufo bench`-only for now).
Block-sparse attention built and verified (needle-in-haystack at three
depths, two-bank+NPU interaction, decode graph-capture boundary) but off
by default (`LAGUNA_SPARSE_ATTN=1` opts in; see Block-sparse attention
above). Not done: a CPU-reference regression
script, model tests under `tests/`, and the standard cross-model
regression run for the shared `dflash_weights.cpp`/`qfn_mmq.hip.cpp` edits
(done manually this session against Flash-Next/27B/35B/GLM, not yet
automated).
