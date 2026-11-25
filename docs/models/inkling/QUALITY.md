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

### Prefill pruning (opt-in)

`INKLING_PRUNE_PREFILL=<share>` applies the same threshold to prefill batches on the
shared routed path. KL against the lossless engine, 48 windows of 1,024 tokens
(`inkling_kl.py`); the fast profile alone is the first row:

| Setting | KL | Top-1 | 8K prefill, fast + NPU (tok/s) |
| --- | --- | --- | --- |
| fast profile (`INKLING_IU4=2 INKLING_F16_BLOCKS=1`) | 0.0191 | 43/48 | about 2,650 to 2,845 |
| + prune 0.04 | 0.0208 | 42/48 | no gain |
| + prune 0.06 | 0.0260 | 41/48 | +2% |
| + prune 0.08 | 0.0355 | 41/48 | +4% |
| + prune 0.10 | 0.0709 | 42/48 | +10% (about 2,930 to 2,970 in the same noisy runs) |

The speed column was measured while a monitor used a CPU core, which lowered every
baseline by about 5%; compare rows within the same run only.

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

Method: `gufo_inkling_gpu_check <gguf> --windows <text> 48 1024 <out.bin>` writes the
last-token logits of 48 windows; `tools/bench/inkling_kl.py base.bin test.bin` reports KL
and top-1 agreement.


### GPTQ feasibility for the int4 experts (2026-11-24)

The fast profile's error is almost all the rounding of the int4 gate/up weights (gate/up alone
KL 0.016 of the 0.019; activations about 0.001, fp16 blocks 0.002; the error is spread evenly over all
24 layers, so exempting layers does not help). Check on layer 12, expert 71 (rows 2,304, K 2,304; NVFP4
weights, Hadamard-rotated, per-128 scales exactly as the repack), plain rounding against GPTQ with the
expert's own input Hessian in the rotated basis, error measured on held-out tokens:

| calibration tokens | damping | RTN output error | GPTQ output error |
| --- | --- | --- | --- |
| 6,300 | 0.01 | 0.0774 | 0.0571 |
| 7,550 | 0.01 | 0.0777 | 0.0529 |
| 7,550 | 0.05 | 0.0777 | 0.0522 |
| 7,550 | 0.20 | 0.0777 | 0.0521 |

Output error falls by about a third, so the squared error (which KL follows) falls by about 55%:
gate/up KL should go from 0.016 to about 0.007, and the fast profile from 0.019 to about 0.011, with no
change to the kernel or its speed (the codes and scales are produced offline). Held-out error is still
well above the train error (0.043), so a larger calibration set helps; rare experts see few tokens and
would need stronger damping or plain rounding. Cost: a calibration pass over text that captures each
expert's inputs, one Hessian and Cholesky per expert matrix (6,144 matrices over 24 layers), a blocked
GPTQ solve on the GPU, and a cached repack artifact. The numpy reference takes about 4 s per matrix.

### GPTQ artifacts and the 3K profile (2026-11-24)

Calibration: 48 windows of 4,096 tokens of source and documentation (about 196,000 tokens),
lossless prefill, Hessians in the rotated basis, damping 0.05 (0.2 for experts with fewer
tokens than inputs), plain rounding for experts with under 64 calibration tokens. Evaluation:
1,000 windows of 512 tokens from files never used in calibration, KL against the lossless
engine on the last token, mean with standard error:

| Setting | KL | Top-1 |
| --- | --- | --- |
| gate/up int4 only, plain rounding | 0.0134 +- 0.0013 | 97.3% |
| gate/up int4 only, GPTQ | 0.0115 +- 0.0013 | 97.6% |
| fast profile, plain rounding | 0.0166 +- 0.0026 | 96.7% |
| fast profile, GPTQ gate/up and down | 0.0147 +- 0.0022 | 96.7% |
| fast profile, GPTQ gate/up only (`INKLING_GPTQ_PARTS=1`) | 0.0143 +- 0.0014 | 96.6% |
| fast profile + prune 0.10 | 0.0375 +- 0.0031 | 95.0% |
| fast profile + GPTQ gate/up + prune 0.095 | 0.0319 +- 0.0028 | 95.3% |
| fast profile + GPTQ gate/up + prune 0.10 (3K profile) | 0.0351 +- 0.0040 | 95.7% |
| gate/up int4 + exact down, GPTQ, prune 0.10 | 0.0329 +- 0.0036 | 94.7% |
| gate/up int4 + exact down, GPTQ, prune 0.12 | 0.0658 +- 0.0046 | 93.2% |

GPTQ lowers the error by about 11 to 14%; it does nothing for the down projection, whose error
is the large SiTU activations. Pruning is the main cost and has a cliff between 0.10 and 0.12.
Pruning only one group of six layers at a time adds 0.0058, 0.0075, 0.0067 and 0.0044 to the
no-pruning 0.0143, so no layers are cheap to prune.

HumanEval (164 tasks, greedy, thinking off, 1,024 new tokens, `gufo serve` with the drafter),
3K profile: 146/164 (89.0%), against 147/164 for the default engine and 149/164 for the fast
profile. Four tasks the default engine passes fail and three it fails pass.
