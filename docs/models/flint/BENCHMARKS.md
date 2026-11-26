# Flint benchmarks

## Notes moved from the README speed section

`gufo bench -n 128` plain decode: 73.6 tok/s; `gufo serve`: 72.8 tok/s. Flint has
no MTP head and no drafter yet, so only prompt lookup (lossless) applies, and it
helps when the output copies earlier text. Long context: 131k-token prefill takes
89 s dense and 48 s (2,716 tok/s) with `FLINT_SPARSE_ATTN=1`, which is
approximate beyond about 19k tokens. Details, numerics and the prefill history
are in [docs/models/flint](docs/models/flint/README.md).
