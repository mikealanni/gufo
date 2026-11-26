# Laguna S 2.1 benchmarks

## Notes moved from the README speed section

A 118B-ish mixture-of-experts model (48 layers, 256 experts with 10 routed
plus a shared expert, sliding-window attention on 3 of every 4 layers) in
the 68 GiB UD-Q4_K_M quantization.

`gufo bench -n 128` plain decode: 25.9 tok/s. DFlash v1's gain is sharply
content-dependent - a short/trivial prompt can land below plain decode
(25% accepted, 24.0 tok/s on a filler prompt); measure on real target
text, not a filler prompt. Measured via `gufo prompt`
(`--speculative dflash2 --dflash-model ...`); `gufo serve` doesn't support
`--speculative` for Laguna yet. Block-sparse attention
(`LAGUNA_SPARSE_ATTN=1`) gives a depth-growing prefill win -
+6.6%/+16.5%/+37.9% at 41k/61k/100k tokens, needle-in-haystack-verified
correct at every depth tested. Fast HumanEval check (20/20, pass@1 = 1.0)
with sparse attention on. Details, every bug found turning sparse
attention on, and the full numbers are in
[docs/models/laguna](docs/models/laguna/README.md).
