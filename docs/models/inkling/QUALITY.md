# Inkling 2 Mini quality

KL is the divergence of the modified engine's next-token distribution from the
reference engine's, in nats (lower is better; 0 is identical). Top-1 is the share
of positions whose most likely token is unchanged.

| Check | Scope | Result |
| --- | --- | --- |
| GPU engine vs CPU reference | tokens 1000, 2000, 3000 | logits agree to about 4 digits; batched prefill equals token-by-token |
| Verify rows vs single steps | 5 rows, argmax | identical argmax; max logit difference 0.2 to 6 (int8 activation decode path against f32 batch path) |
| NPU offload (q/gate/k/v/out) | 12 prose windows of 4,096 tokens, last token | KL 0.0026 mean, 0.0149 max; top-1 12 of 12 |
| Expert layers 20 to 23 Q8_0 to NVFP4 | 12 prose/code windows of 2,048 tokens | KL 0.0013 mean, 0.0048 max; top-1 12 of 12 |
| Sparse attention (63k context) | passphrase at 10%, 50%, 90% depth | dense, per-head and per-group sparse give identical answers |
| Greedy speculation | exact-match verification | reproduces plain decode up to logit noise at near ties |

## Expert pruning (opt-in)

`INKLING_PRUNE=<share>` drops routed experts whose normalised routing weight share
is below the threshold during few-row steps (256 decode-path positions, KL and
top-1 against the unpruned engine; verify is a six-row step):

| Share | KL | Top-1 | Verify | Plain step |
| --- | --- | --- | --- | --- |
| off | 0 | 100% | 63.5 ms | 31.5 ms |
| 0.03 | 0.0016 | 98.8% | 63.5 | 31.2 |
| 0.06 | 0.0067 | 96.9% | 60.8 | 30.3 |
| 0.10 | 0.040 | 94.5% | 48.9 | 27.7 |

Rescaling the kept weights made KL worse at every threshold. Pruning is off by
default because it changes outputs.

Known near-tie behaviour: greedy output with speculation can differ from plain
decode at positions where the top two logits are within about 0.2.

## Fast profile (int4 experts, opt-in)

The expert weights are NVFP4 in the GGUF, so the int4 form is a second 4-bit
quantization (rotated, symmetric, one scale per 128 values; activations are
rotated and quantized per 128 as well). Measured as mean KL and top-1 of the last
token of 48 windows of 1,024 tokens against the default engine (default against
itself: KL 0, 48 of 48); the six-window versions of these runs were too noisy to
rank settings.

| Setting | Mean KL | Max KL | Top-1 |
| --- | --- | --- | --- |
| `INKLING_IU4=1` (gate/up) | 0.016 | 0.080 | 42/48 |
| `INKLING_IU4=2` (gate/up and down) | 0.022 | 0.100 | 44/48 |
| `INKLING_IU4=2`, weight scale by absmax / 7, activation scale 1.0 | 0.033 | 0.209 | 39/48 |
| plus `INKLING_F16_BLOCKS=1` | 0.019 | 0.059 | 43/48 |

The defaults of `INKLING_IU4` are the MSE-optimal weight scale (`INKLING_IU4_WCLIP`)
and an activation scale of 0.9 times absmax / 7 (`INKLING_IU4_ACLIP`; 0.85 and 0.9
are equal within noise, 0.8 is worse). Keeping any single layer on NVFP4
(`INKLING_IU4_SKIP`) changes the KL by less than the run-to-run spread, so the error
is spread over all layers. Next steps for quality: per-32 activation scales (the
gate/up output error roughly halves in an emulation but the accumulator fold runs
four times as often) and an int4 artifact made from higher-precision weights, which
the model release does not provide.

Other changes this round: split-K for the F32 GEMMs and the compact tile list keep
the default engine at KL 0.0016 (floating-point reassociation) and 0 (bit-identical)
respectively.

HumanEval (164 tasks, greedy, thinking off, 1,024 new tokens, `gufo serve` on GPU
only with speculation; int4 applies to the prefill of each prompt and decode stays
NVFP4): default engine 147/164 (89.6%); fast profile (`INKLING_IU4=2
INKLING_F16_BLOCKS=1`) 149/164 (90.9%). Only 56 of the 164 completions are
identical, so the numerics differ clearly, but four tasks flip (one lost, three
gained), which is within the spread of a 164-task score.

With thinking on (default reasoning, 6,000 new tokens, about 1,340 generated tokens
per task, 63 minutes for the set) the fast profile scores 152/164 (92.7%). The
default engine was not run with thinking on, so this is a check that long
reasoning decodes after an int4 prefill stay coherent, not a comparison.
