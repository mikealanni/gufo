# Inkling 2 Mini experiments

Retained:

- Compact expert tile list built on the device (launch grid bounded by rows per
  tile plus experts instead of experts times tiles): bit-identical, small gain.
- NVFP4 expert weights and scales loaded straight into registers by each wave (no
  LDS staging); dense Q8 rows the same way.
- Attention: no per-element mask on tiles inside the causal window, wave barrier
  before the PV step, longest full-attention blocks first (all bit-identical).
- Split-K for the small F32 GEMMs (router, loop-control down projection).
- Prefill gate/up and down projections on rotated int4 matrix cores (opt-in, see
  README): register-direct weights, expert-aware tiles, gate and up in the two
  feature sub-tiles of one wave, SiTU applied after an LDS round trip (the epilogue
  math inside the accumulator loop made the K loop spill registers), matrix work
  and fold skipped for dead 16-token sub-tiles.

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

- Prefill: fused down projection and combine with atomics (slower, and the output
  order becomes nondeterministic), a prefill-only F16 expert copy (49 GB), wider
  expert tiles (BM 256, BN 64 to 256), deeper weight prefetch, 16-wave attention
  (needs more than the 64 KB of LDS per block), software-pipelined K/V loads,
  two accumulator chains for QK, removing the block barriers, F16 residual-stream
  states beyond the block states (KL 0.0042 for a few percent).
- NPU for routed experts: about 0.95 ms per expert at 512 rows against about 70 us
  on the GPU int4 kernel, so at most a sixth of the experts fit in the NPU's idle
  time (about 2% net).

- Expert-grouped GEMM, verify graph capture, fused tiny GEMVs, multi-row chunked
  decode kernel, register prefetch of the next K/V tile, one-block-per-all-rows
  loop control: slower or unchanged.
- Q8 activation fusion into norms: about 2 microseconds saved per launch removed.
- Wider drafts than the confidence gate allows: each extra verify row costs about
  7 ms because it touches more distinct experts.

Open:

- NPU prefill attention (relative bias, sink, 24 heads over 8 KV heads): not started,
  specification in [../../plans/inkling-npu-attention-kernel.md](../../plans/inkling-npu-attention-kernel.md).

Server notes learned while testing:

- The serve runner compares each session's measured device bytes with a claim taken
  from a probe session that never ran a forward pass. Buffers created on first use
  (and anything allocated per prompt length) must not count as session state
  (`AllocLazy`), or every request fails with HTTP 500 once they exceed the claim.
- Newly allocated device buffers are zeroed with `hipMemset`; synchronize before a
  stream writes into them, or results differ from run to run.
- Disk cache directories grow to the full budget (240 GB by default); give test runs a
  separate `GUFO_CACHE_DIR` and a small `GUFO_CACHE_BYTES`.

