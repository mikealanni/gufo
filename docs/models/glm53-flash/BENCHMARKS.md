# GLM-5.3 Flash benchmarks

## Notes moved from the README speed section

A 321B-parameter mixture-of-experts model (about 14B active per token; 45 layers,
34 KDA and 11 DSA, 288 experts with 8 routed) in the 81 GiB AJ-IQ2_XXS quantization,
and its coder variant RCO-152E (152 experts with 8 routed, 69.9 GiB Q3_K_M, served
by native Q3_K WMMA expert GEMMs).

Every prompt of 4096 tokens or more runs a two-bank pair of 2048-token banks
through the runtime-M/K program, at any context depth; smaller prompts and
decode stay on the GPU at full speed. `GUFO_GLM53_NPU_M=4096` selects wide
banks (same speed at 8192, no gain elsewhere). Moving the shared-expert down
projection to the NPU as well (`GUFO_NPU_WHOLE=1`, implemented) measures zero
net gain: the NPU queue is already balanced at 28 slices per bank.

Greedy chat requests to `gufo serve`, 256 tokens. `gufo bench -n 128` plain decode:
18.8 tok/s. Prose is the expository prompt; a creative-writing prompt gains nothing
(18.7). Decode reads about 9 GB of weights per token, mostly the Q6_K attention and
shared-expert projections, so plain decode is bandwidth-bound; drafting helps when
the next tokens are predictable. Serving uses a 131,072-token context and one
session at a time. Start it with `GUFO_KIND=glm scripts/run-gufo.sh`.
