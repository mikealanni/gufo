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
