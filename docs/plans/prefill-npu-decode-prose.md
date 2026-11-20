# Plan: 2k+ Prefill to 256k (Flash-Next), NPU Utilization, and Real-World Prose Decode

**Status:** draft / review only — no implementation in this plan  
**Scope focus:** `qwen3.8-flash-next` (UD-IQ4_XS / UD-Q4_K_XL as applicable) on AMD Strix Halo `gfx1151` with XDNA2 NPU. Generalize only after Flash-Next evidence.  
**Primary goals:**
1. Achieve **≥ 2000 tok/s prompt processing (pp)** across depths **0 → 256k** (within native 262,144 context) on representative, NPU-eligible workloads.
2. Determine whether **more NPU work and/or better NPU utilization** can close the prefill gap safely.
3. Investigate why **prose decode looks ~50–55+ tok/s in tests** but is **slower in opencode / real usage**, and define fixes/experiments that improve real-world decode without regressing correctness.

**Non-goals (initially):** full model sweeps; changing default quality semantics without explicit qualification; unsafe/unstable NPU experiments without validation; optimizing unrelated models before Flash-Next path is understood.

---

## Baseline facts (from repo docs, verify before acting)

- README/NPU/BENCHMARKS show Flash-Next two-bank NPU whole (`GUFO_NPU_DIR=tools/npu GUFO_TWO_BANK=1 GUFO_NPU_WHOLE=1`) reaches ~**2.14k pp16384 @ d0**, ~**2.04k @ 32k**, ~**2.05k @ 65k (README)**, ~**1.92k @ 131k** (table varies by row/mode), and drops further at extreme length. Target gap is mainly **128k–256k** + workload-shape consistency.
- Prose decode (README, project probe prompts): plain **~36.5**, MTP **~47.2**, DFlash2 **~51.8** tok/s (greedy). Grammar-constrained/tool-call turns improved to **90–93 tok/s** after draft verification stayed on GPU (different regime). Real usage can differ due to sampling, speculative acceptance, context depth, cache behavior, request fragmentation, tool loops, penalties/grammar, and client/server overhead.
- NPU offloads dense projections in Flash-Next whole two-bank mode: `ssm_in`/`attn_qkv` whole projections + shared expert (layer-by-layer lock-step, banks of 8192 / 4096 groups). Routed experts and attention internals remain GPU-side. `NpuGemm` uses XRT, HIP-stream ordered submit/wait, **max 4 in-flight runs**. NPU is opt-in; non-finite logits or timeout disables NPU for the process.

**Safety (mandatory for NPU work):** keep NPU out of runtime suspend, use a validated NPU power/runtime setup for your host, start with short runs, save work, and validate stability before long sweeps. Non-finite logits or timeouts can disable NPU for the process.

---

## Phase 0 — Environment & measurement discipline (before profiling)

**Objective:** make results comparable and attributable.

- [ ] Confirm artifacts: exact model GGUF(s) (UD-IQ4_XS/UD-Q4_K_XL), DFlash2/MTP drafts if used, checksums/revisions recorded.
- [ ] Confirm binary: measure `result/bin/gufo` (Nix) or matched release build with `-DGUFO_ENABLE_NPU=ON` and XRT paths as needed; record `gufo --version`/commit, toolchain, ROCm/XRT, kernel/driver.
- [ ] Record machine state: `xrt-smi` pmode, NPU runtime PM control, CPU/GPU temps/power profile, no competing GPU/NPU workloads. Discard first NPU run after idle.
- [ ] Define canonical workloads (same text, same tokenization path):
  - **Prefill depth sweep:** pp at depths `0, 4096, 8192, 16384, 32768, 65536, 131072, 196608, 262144` (cap/adjust to memory/context). Use **NPU-eligible chunk shapes** first (multiples of 4096/8192, >=8192 for pair path). Capture both `bench` and `serve` (HTTP) where behavior differs.
  - **Prompt sizes:** at least `8192`, `16384`, `32768`; include one representative long prompt near target depth. Note text sensitivity (code/prose/markdown route experts differently).
  - **Decode prose (realistic):** fixed prose prompts from opencode-like usage (system prompt + user turn + tool history shape if relevant), greedy and sampled configs used in practice, with/without speculative modes.
- [ ] Use repo methodology: timings per request/phase (not averaged across requests), 5%/3ms gate mindset for regressions; retain all runs; correctness before speed claims.

**Commands (adapt paths/models):**
```bash
# Example prefill sweep (NPU)
GUFO_NPU_DIR=tools/npu GUFO_TWO_BANK=1 GUFO_NPU_WHOLE=1 \
  build/release/gufo bench -m <flash-next-first-shard.gguf> \
  -p 8192,16384,32768 -n 1 -r 3

# GPU-only control
build/release/gufo bench -m <flash-next-first-shard.gguf> -p 8192,16384,32768 -n 1 -r 3
```
Adjust for depth/context flags supported by `bench` in this build; for serving, use `scripts/run-gufo.sh`/`gufo serve llm` with matched settings.

---

## Phase 1 — Profiling (start here)

**Objective:** attribute <2k pp at 128k–256k and identify NPU idle/overlap gaps.

### 1A. Prefill decomposition by depth
- [ ] Collect pp tok/s, wall time, and phase timings for depths `0, 32k, 131k, 196k/256k` (feasible max).
- [ ] Confirm **which path executes**: two-bank pair (`CanForwardPair`/`ForwardPair`) vs single/chunk path; log/env evidence (`GUFO_NPU_DIR` active, NPU projections reported on load, no NPU-disable due to NaN/timeout).
- [ ] Check chunk-shape effects: multiples of 4096/8192 vs misaligned tails; compare `GUFO_PREFILL_TILE` 8192/16384/24576 where memory allows.
- [ ] Separate content variance: run same lengths with controlled text classes (code/prose/markdown) to isolate MoE routing noise.

### 1B. NPU utilization / overlap
- [ ] Enable diagnostics: `GUFO_NPU_STATS=1`, optionally `GUFO_NPU_TRACE=<path>` (NPU trace is O_DSYNC; inspect after runs/hangs). Capture per run/wait.
- [ ] From `npu_gemm`/executor behavior, attribute: submit latency, `launch_us`, `run_wait_us`, in-flight depth usage (max 4), wait bubbles, host-callback scheduling effects, and any timeouts/failures.
- [ ] Compare GPU vs NPU critical path during `ForwardPair` whole mode: are GPU stages waiting on NPU, or NPU waiting on GPU/data deps? Identify largest non-overlapped segments.
- [ ] Test sensitivity knobs (profile-guided, one at a time): `GUFO_TWO_BANK` modes where safe, warmup/keepalive behavior, tile/group coverage (`npu_whole_groups_`/eligible shapes). Record stability.

### 1C. Depth scaling bottlenecks
- [ ] Profile/inspect cost growth in: **full-attention layers** (every 4th, GQA + QSA/indexer selection, `indexer_top_k=2048`, `compress_ratio=4`), **KV/index memory traffic**, **linear-attention/SSM projection+state**, **routed experts** (MMQ tiles, routing skew).
- [ ] Validate long-depth correctness/safety: finite logits, no NPU disable, quality sanity against known gates (Flash-Next has strict numerical/functional tests; run focused relevant tests for touched behavior later).
- [ ] Memory: measure peak HIP/global + scratch impact (`GUFO_PREFILL_TILE`, two-bank whole adds ~3.3 GiB NPU weights + ~240 MB shared scratch per docs). 256k may hit bandwidth/residency limits before compute.

**Deliverable:** a short profiling report (can live under `docs/plans/` or `artifacts/` outside git as appropriate) with: table of pp by depth/shape/path, NPU stats/trace summary, top 3 bottlenecks, and “cheap win” candidates.

---

## Phase 2 — Cheap / low-risk improvements (after profiling)

Pick only items supported by Phase 1 data. Prefer config/workload alignment and proven-safe toggles before code changes.

- [ ] **Workload alignment:** ensure serving/bench uses NPU-eligible chunking; fix non-multiple/short-tail paths falling to GPU-only where possible *without* breaking semantics. Validate `--prefill-chunk` vs `GUFO_PREFILL_TILE` vs two-bank requirements.
- [ ] **Validated config sweep (small matrix):** `GUFO_PREFILL_TILE` (8192/16384/24576) × two-bank mode × text class × depth checkpoints; keep power/runtime state stable; reject unstable or NaN-producing combos.
- [ ] **NPU overlap hygiene:** if traces show avoidable waits, try minimal scheduling adjustments (diagnosed only). No speculative refactors.
- [ ] **Runtime inst coverage:** if dominant production `(M,K)` shapes lack optimal `tools/npu/runtime_{k,mk}` inst streams, generate/add insts per `tools/npu/runtime_mk/README.md` (no XCLBIN rebuild). Test only via `gufo*` binaries with safety checks; compare logits vs GPU-only/fixed baseline.
- [ ] **Residual/precision toggles (already documented):** `GUFO_F16_RESIDUAL` behavior in two-bank mode; only if Phase 1 shows residual/combine cost material and quality remains acceptable.

**Acceptance for prefill step:** reproducible ≥2k tok/s at agreed depth checkpoints (at minimum 0/32k/65k/131k/196k/256k as feasible) on the canonical NPU-eligible workload, with no correctness regressions on focused Flash-Next checks.

---

## Phase 3 — Larger prefill/NPU changes (only if Phase 2 insufficient)

Evaluate with cost/risk/expected gain from profiling:

- [ ] **Offload additional dense, BFP-GEMM-compatible projections** in Flash-Next two-bank whole schedule (candidates: uncovered dense projections in SSM/attention/indexer/shared-expert adjacency) — requires weight packing, inst streams, dependency-safe overlap, finite-logit + KL/top-1/functional qualification.
- [ ] **Improve lock-step overlap / group scheduling** in `ForwardPair` whole path (executor): reduce critical-path bubbles while respecting 4 in-flight max and callback ordering.
- [ ] **Full-attention depth path:** optimize QSA/indexer selection, mask/list compaction, attention kernels, and KV/index bandwidth behavior at 256k. This is likely required for stable 256k 2k+ even with better NPU.
- [ ] **Routed-expert/GPU prefill efficiency:** routing/tile/MMQ improvements (GPU) to reduce variance and depth pressure; preserve correctness.
- [ ] **New NPU shapes/programs:** only if current tile/N=1024/MK coverage blocks a proven offload; follow runtime_mk safety/process.

---

## Phase 4 — Decode prose “55+ in tests, slow in opencode/real usage” (review, then targeted fixes)

**Objective:** explain the gap between synthetic/probe decode speed and real interactive usage, then improve real-world prose decode.

> Repo baseline (Flash-Next, greedy project probe prompts): plain **~36.5**, MTP **~47.2**, DFlash2 **~51.8** tok/s prose; grammar-constrained/tool-call turns have shown different behavior after targeted fixes. Real opencode usage can diverge due to cache reuse, speculative acceptance, constraints, fragmentation, depth, sampling, and overhead.

### 4A. Reproduce and measure the real path

- [ ] Capture a sanitized **opencode/real session trace** per turn: request shape, system prompt size, conversation/tool history, input tokens, cached/reused tokens, **new prefill tokens**, generated tokens, **TTFT**, **decode-only tok/s** (exclude prefill), finish reason, context depth, sampling params, `--speculative` mode, grammar/JSON/tool constraints, and speculative stats (drafts proposed/accepted, acceptance rate, tokens/round) when available.
- [ ] Compare identical model/settings across: `gufo bench` / `gufo prompt`, `gufo serve` HTTP (OpenAI-compatible client/curl), and the opencode integration path. Isolate **client/streaming/protocol overhead** from model time.
- [ ] Run a small matrix on the representative prose trace: `plain` / `MTP` / `DFlash2` / `auto` × `greedy` / sampled-used-in-practice × shallow/deep context × unconstrained / constrained (only if constraints apply). Sampled requests may use different speculative/adaptive behavior than greedy.

### 4B. Review checklist (classify the bottleneck)

Do not treat “slow” as one metric. Classify each case:

| Signal | Likely cause | What to check |
| --- | --- | --- |
| High TTFT, low new useful decode | Hidden prefill / cache miss | New prefill tokens per turn, prefix/session reuse, continuation/KV cache hits, repeated system/tool history processing |
| Decode tok/s drops with speculative on | Speculative mismatch | Acceptance rate on real prose, draft overhead, `auto` choice, DFlash2 vs MTP fit for prose, priming/catch-up costs |
| Slow only on tool/JSON/grammar turns | Constrained decoding overhead | Constraint type/frequency, constrained vs unconstrained A/B on identical text, whether prose turns are unnecessarily constrained |
| Many fast/slow tiny turns | Fragmentation / overhead | Tokens/turn distribution, tool-loop round trips, per-request setup, streaming/client overhead; speculative gains need enough continuous tokens |
| Slower at long context | Depth/KV pressure | Same prompt at shallow vs opencode-real depth; decode path memory/KV reads, attention/cache behavior |
| Different from greedy probe | Sampling/config path | temp/top_p/penalties, sampled speculative behavior, server config differences |
| Direct fast, serve/opencode slow | Integration/server path | Scheduling, chunk alignment, request batching/fragmentation, protocol/tokenization/streaming overhead |

### 4C. Gain strategy (prioritized, evidence-driven)

1. **Cache/prefill reuse first (often largest real-world gain):** eliminate repeated prefill in multi-turn/tool loops. If unchanged prefixes are reprocessed, users perceive “slow decode” but the fix is TTFT/prefill/session reuse.
2. **Speculative policy for real prose:** choose `DFlash2`/`MTP`/`auto`/plain based on measured acceptance and decode-only tok/s on the actual trace; do not assume probe-prompt acceptance transfers.
3. **Grammar/tool constraint scope:** avoid unnecessary constraints on plain prose; optimize or isolate constrained paths only where needed.
4. **Fragmentation/interaction overhead:** reduce avoidable round trips and fixed per-turn costs; ensure measurement separates overhead from model decode.
5. **Depth/sampling alignment:** test at real context depth and real sampling; if depth dominates, gains require decode/attention/KV-path work (kernel-level, higher risk/cost).
6. **Kernel-level decode work last:** after A–E are controlled, profile projections, attention/KV reads, MoE, speculative verify/draft, sampling. Apply focused optimizations with correctness checks.

**Deliverable:** table comparing opencode trace vs bench/serve (TTFT, prefill/cached/new tokens, decode tok/s, speculative acceptance), bottleneck classification per case, and ranked experiments with expected gain + correctness risk. **Target:** ≥55 tok/s decode-only on a clearly defined representative opencode prose workflow, or a measured explanation of the gap and the highest-value fixes.

---

## References (repo)

- `README.md`, `docs/NPU.md`, `docs/BENCHMARKS.md`
- `docs/models/qwen3.8-flash-next/{README,BENCHMARKS,QUALITY}.md`
- `src/models/qwen38_flash_next/kernels/rocm/executor.cpp` — prefill/`ForwardPair`/two-bank/NPU whole flow
- `src/models/qwen38_flash_next/kernels/rocm/npu_gemm.{hpp,cpp}` — XRT submit/wait, 4 in-flight, stats/trace hooks
- `src/models/qwen38_flash_next/config.hpp` — hybrid layers (`full_attention_interval=4`), QSA indexer params
- `tools/npu/runtime_{k,mk}/README.md` — XCLBIN/insts, shape constraints, safety
- `scripts/run-gufo.sh` — NPU enablement/runtime checks
- `AGENTS.md` — focused checks, quality strict, timing methodology, NPU/port/optimize-kernel skills guidance

---

## Initial next action

Start **Phase 0 + Phase 1A/1B** on Flash-Next: run a depth/shape sweep (GPU-only control + NPU two-bank whole) with `GUFO_NPU_STATS=1` and a fixed prompt corpus, confirm `ForwardPair` eligibility, then produce the profiling report before any code changes.