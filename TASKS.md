# Tasks

- [x] WP6: model spec and neutral runtime (worktree `/home/majed/gufo-wp6`, branch `wp6/model-spec-neutral-runtime`)
  - commits: 658d45df
  - remaining: hardware gate validation (93k NLL/spec==plain), fuller spec-driven program/tensor resolution
- [~] WP0: fast CPU attention (bf16 KV, SIMD) — in progress in same worktree
  - commits: e418f21e, (pending)
  - done: bf16 KV cache, AVX-512 dot/accum, no per-(t,h) alloc, reusable buffers, chunked split-K over context with online-softmax partial reduction for long contexts
  - remaining: on-target perf validation (prefill/decode tok/s @1k/32k, plain+heads vs baseline) and NLL/spec==plain within noise; possible tuning of chunk size/threshold

