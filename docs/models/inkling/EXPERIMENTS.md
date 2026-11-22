# Inkling 2 Mini experiments

Retained:

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
