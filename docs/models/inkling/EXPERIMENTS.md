# Inkling 2 Mini experiments

Retained:

- Compact expert tile list built on the device (launch grid bounded by rows per
  tile plus experts instead of experts times tiles).
- NVFP4 expert weights and scales loaded straight into registers by each wave (no
  LDS staging); dense Q8 rows the same way; routed GEMM runs the dense projections.
- NPU token banks on two GPU streams, with NPU submits and waits ordered through one
  control stream.
- Fused loop-control GEMM, SiTU in the gate/up epilogue, attention epilogue with
  the output gate, exp2 softmax, wide RMS norm, warp-per-row clip and norms.
- Attention: no per-element mask on tiles inside the causal window, wave barrier
  before the PV step, longest full-attention blocks first (all bit-identical).
- Split-K for the small F32 GEMMs (router, loop-control down projection).

- Confidence-gated draft width (up to 7) instead of a fixed width.
- Proposal vocabulary ranked by the drafter's own `markov_w1` row norms (32,768
  rows, NVFP4), not the first token ids.
- Greedy argmax on the device per verify row; logits downloaded lazily.
- Packed few-row attention: all rows and the three heads of a KV group in one
  4-wave tile; sliding-window passes split for verify rows; per-group sparse block
  selection at 64k and above.
- Drafter attention with the eight slots across lanes.
- NVFP4 routed GEMM with the expert scale folded in and F16 outputs; expert layers
  20 to 23 converted to NVFP4 at load.
- Two NPU token banks of 4,096 rows.

Rejected (no gain or worse):

- Expert-grouped GEMM, verify graph capture, fused tiny GEMVs, multi-row chunked
  decode kernel, register prefetch of the next K/V tile, one-block-per-all-rows
  loop control: slower or unchanged.
- Q8 activation fusion into norms: about 2 microseconds saved per launch removed.
- Wider drafts than the confidence gate allows: each extra verify row costs about
  7 ms because it touches more distinct experts.

Open:

- NPU prefill attention (relative bias, sink, 24 heads over 8 KV heads): not started,
  specification in [../../plans/inkling-npu-attention-kernel.md](../../plans/inkling-npu-attention-kernel.md).
- Opt-in int4 matrix-core experts (prefill 2,466 GPU only and 2,822 with the NPU at 8K,
  logits KL about 0.02): on branch `perf/inkling-pp3k`, not merged.

Server notes:

- The serve runner compares each session's measured device bytes with a claim taken
  from a probe session that never ran a forward pass. Buffers created on first use
  (and anything allocated per prompt length) must not count as session state
  (`AllocLazy`), or every request fails with HTTP 500 once they exceed the claim.
- Newly allocated device buffers are zeroed with `hipMemset`; synchronize before a
  stream writes into them.
- Disk cache directories grow to the full budget (240 GB by default); give test runs a
  separate `GUFO_CACHE_DIR` and a small `GUFO_CACHE_BYTES`.

