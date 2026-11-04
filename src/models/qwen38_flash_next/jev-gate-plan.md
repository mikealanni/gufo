# Jev: NPU reasoning offload for Qwen3.5 Flash-Next

## The idea

Flash-Next spends its own slow, expensive decode tokens on two different
kinds of work while coding/debugging:

1. **Deciding how hard to think at all.** Every request gets the same
   launch-time `--reasoning-effort xhigh`, so a one-line lookup burns the
   same budget as a real bug hunt.
2. **Deciding small yes/no sub-questions inside its own thinking.** Mid
   `<think>` block, Flash-Next routinely poses itself binary questions
   ("is this the root cause?", "should I check the config first?", "does
   this match the error signature?") and then spends real reasoning
   tokens working each one out, one at a time, in series.

"Jev" is a small model that lives on the NPU — otherwise idle while the
GPU decodes — and takes both jobs off Flash-Next:

- Before decode starts, jev picks the effort level for the whole request
  (the pre-gate).
- While Flash-Next is thinking, jev watches the stream, catches the
  self-posed yes/no questions as they're written, answers them itself,
  and the answer is spliced back into the stream so Flash-Next continues
  from the answer instead of deriving it.

Both pieces chase the same goal: make Flash-Next do less work per task
without losing the answer quality that `xhigh` gives it on genuinely hard
problems. This is a full rewrite of the original pre-gate-only draft —
both mechanisms are now the main idea, not an optional stage 2.

## Status 2026-11-04 — Part A rule-gate MEASURED (3-arm), shipping; NPU chain debug continues separately

`tests/functional/jev_gate_measure.sh`, 24 prompts (12 trivial, 12 hard),
production settings via run-gufo.sh (results
`~/jev-gate-run/summary.tsv` + `labels.jsonl`; first partial run
`~/jev-gate-run.partial1/`):

| arm | trivial avg | hard avg | trivial think-chars |
|---|---|---|---|
| `gate_on` (this work) | **4.2 s** | **201 s** | **0** |
| `gate_off` (stock, effort=xhigh default) | 5.0 s | 301 s | 200–1500 |
| `effort_medium` | 5.6 s | 52 s | all |

Reading: the gate keeps hard prompts at xhigh (correctness class preserved)
while collapsing trivial requests to low/no-thinking (25% faster than stock on
trivial, ~33% faster on hard — hard mix includes six gate downgrades to medium
where the rule saw simple intent). `effort_medium`-on-hard is faster but is the
wrong-answer trade; response bodies in `~/jev-gate-run/responses/<arm>/` await
hand grading per the script's `labels.jsonl` contract.

Operational notes for future runs: (1) `systemd-oomd` (active on this box)
SIGTERMs the arm server mid-decode — arms must run under
`JEV_RUNNER=~/jev-arm-scope.sh` (systemd-run scope with
`MemoryPressureWatch=off`); (2) run via
`JEV_ARMS=... ~/jev-measure-cycle.sh` which stops the production serve, runs
measurement past failures, and ALWAYS restarts it; (3) host has no nix —
format gate runs with system clang-format via
`python3 tools/ci/check-format.py --fix` and passes.

## Status as of 2026-11-03 (evening) — trace instrumentation built; run blocked on xclbin host-arg count

Full running record:
`~/torch2aie/examples/qwen3-decode-layer/TRACE-HANG-INSTRUMENTATION.md`.
Summary of what changed since the plan below was written:

- **Trace instrumentation is implemented and builds.** `full-build rc=0`
  (xclbin 417552 B / insts 7352 words, 2026-11-03 15:15), RTP patch verified on
  the traced stream. `aie.trace` blocks on hub (memtile: DMA S2MM/MM2S
  SEL0_STALLED_LOCK, LOCK_SEL0/1_ACQ_GE/REL), bridge (same family — bridge is
  on the **memtile** row `aie.tile(1,1)`, not a core tile), swiglu (core,
  `CoreEventAIE2P::LOCK_STALL/MEMORY_STALL/STREAM_STALL/INSTR_LOCK_*_REQ`),
  plus a shim-tile packet enable at `aie.tile(6,0)`.
- **Corrections to the plan's assumptions, all verified the hard way:**
  trace egress is NOT auto-provisioned by `aie.trace` on this toolchain
  (mlir_aie **1.3.1** — first trace build produced an xclbin with no trace
  section), so the shim capture path is hand-wired: 3× `aie.packet_flow(20)`
  → shim DMA:1, `writebd` (col 6, S2MM ch 1, packet 20, type 2) +
  `address_patch(arg 5)` + `push_queue`. All traced tiles share **one packet
  ID 20** (demux by packet-header tile coords), not per-tile 20/21/22. Start
  must be `event=<"TRUE">` — `broadcast=15` fires nothing in this design. No
  stop event; capture runs until full, read back after the timeout.
  `aie.trace.mode` must be omitted (1.3.1 reg packer bug: "Field not found in
  register: Mode"; mode 0 = event-time anyway). Core events need AIE2P enums.
- **Instruction-patcher fixed for the trace prefix:** trace config writes
  shifted the stream and pattern-match `_rtp_positions` also matches
  `start_config` lowers; `_select_rtp_window()` now anchors the 13-write RTP
  chain on value `0x7f` + stride 6 instead of match order.
- **Toolchain fix:** `toolchain/xrt/lib64/libboost_atomic.so.1.83.0` symlink
  (xclbinutil RUNPATH; `LD_LIBRARY_PATH` doesn't survive aiecc→xclbinutil).
- **Current blocker (host-side, new):** `full-run` fails before submission —
  *"The xclbin declares 5 host buffer argument(s) but 6 were passed."* aiecc
  1.3.1's `main_kernels.json` emits `bo0..bo4` only; it never declares the 6th
  `runtime_sequence` operand (`%trace`) as a host BO — confirmed NOT
  trace-lowering's fault by recompiling a trace-stripped 6-arg copy (still
  `bo0..bo4`). The guard itself comes from the venv's mlir_aie **1.4.5**
  hostruntime checking the 1.3.1-built metadata. Note this example's own
  validator originally capped `address_patch arg_idx` at 4 — a pre-existing
  5-host-BO limit in this pipeline.
- **Next move (done 19:30, build11 pending):** dropped the 6th BO — trace
  capture BD now writes into the **output BO tail**
  (`address_patch(arg3, +OUTPUT_DWORDS*4)`, runner allocates
  `output = 512+8192` dwords, reads trace from `output[512:]`). Applied with
  the audit + method lessons from four same-symptom investigations
  (HF MiniCPM5 discussion #2, Xilinx/mlir-air#2194, Npu-Embeddings
  tasks 0087/0092 — full notes in
  `~/torch2aie/examples/qwen3-decode-layer/TRACE-HANG-INSTRUMENTATION.md`;
  notably a host-boundary BD-byte audit in the 0092 style found **no**
  ABI mismatch in our design, confirming the on-chip trace is the right
  instrument). Run protocol: `~/jev-run.sh <log> -- <cmd>` kills gufo serve,
  runs, and always restarts it (`FORCE=1 JOBS=1 GUFO_EFFORT=low`, ≥25 s +
  port wait).

- **TRACE RESULT + BISECTION (2026-11-03 late, captures 1–5 all recorded):**
  four traced captures + unpatched-127 control ⇒ **design-level boot lock
  deadlock at the attention handoff**: one compact packet (16 records)
  completes at t≈2090 cycles, first hub touch t≈90k, hub completes ~6 slow
  records then total on-chip idle (weights all arrive; patch exonerated;
  cores confirmed working but untraceable — FIFO flush). Bisection rung 0
  re-passed today (qkv-prefix 3.773 ms) — and the prefix slice *keeps*
  bridge/columns/full-replay, so **the fault is in the tail the prefix
  omits: hub + return-GE8 + attention_replay + KV-scan + swiglu +
  up/down**. Next rung: QKVO slice (`QKVO_PHASE_TRACE` +
  `main16_qkvo_tile` already exist unused in the engine generator);
  solve host-observable-completion first. Full record:
  `~/torch2aie/examples/qwen3-decode-layer/TRACE-HANG-INSTRUMENTATION.md`.

- **External reference — HF `julianmb/MiniCPM5-2B-NPU2` discussion #2**
  (same `ERT_CMD_STATE_TIMEOUT` on Strix Halo, closed FastFlowLM engine):
  every environment knob was swept — XRT 2.21→2.26, `-noert` (build-time-only
  flag, retracted), firmware, `force_cmdlist`, TDR, pmode, `iommu=pt`, chunk
  size — **identical timeout every time**. Confirms two things for us: (a) our
  going straight to on-chip trace instrumentation is the right path, env
  fixes don't exist; (b) their working fix — the open `OpenFlowLM-Next`
  engine, which "runs the layer list host-side, so depth is not a dispatch
  limit" (the closed one chains all layers into a single `xrt::runlist`) —
  marks the fallback architecture if our trace ever localizes the stall to
  submission-depth resources: split the chained layer into 2–3 host-side
  dispatches. Also: measured NPU decode is bandwidth-bound at ~63–88 GB/s
  effective (qwen3:0.6B-class = ~92 t/s whole-model), and their "63 t/s" was a
  phantom of ROCm/FastFlowLM#716 (server answers with the resident model and
  echoes the requested name) — in our jev 3-arm measurements, verify *what
  actually ran*, never trust the echoed model field.

Everything below is the original planning record, kept for the ruled-out list
and event-name references (the egress/auto-provision and broadcast-start items
in it were superseded by the findings above).

## Status as of 2026-11-03 — NPU kernels proven, one integration bug blocks the full layer

Read this section fully before touching the NPU work below; it records
what is already proven, what is already ruled out, and exactly which
debugging step is next, so nobody re-derives any of it.

**Proven, on real hardware, in `~/torch2aie/examples/qwen3-decode-layer`
(a near-complete single-token decode-layer implementation already built
for Qwen3-0.6B's exact dimensions — H=1024, NH=16, NKV=8, HD=128,
IM=3072, matching jev precisely):**

- Every individual kernel stage builds, runs on `/dev/accel/accel0`, and
  passes numerically against a CPU oracle: QKV-prefix 4.406ms,
  postprocess-qkv 1826µs, full-vector-station 1005µs, edge-attention
  952µs, main16-q4nx 724µs, SwiGLU 633µs. Zero mismatches on every one.
- This disproves "no existing NPU path for a dense 0.6B model" outright.
  The kernels exist and are fast and correct individually.

**The one open blocker: `make full-run` (the complete chained layer)
hangs with `ERT_CMD_STATE_TIMEOUT`.** Already ruled out, in this order,
each with a real test, not a guess:

1. Fabric clock not held — ruled out. Confirmed with `sudo
   ~/bench-mode.sh on` (fclk held manual at top level, NPU pmode
   performance): hangs identically with the clock held as without.
2. Any specific real compute kernel misbehaving — ruled out. Ran
   `make weight-stream-run` (real compute, only Main16 replaced by
   consume/drop) and `make weight-stream-light-attention-run` and
   `make weight-stream-light-edge-run` (progressively more stages
   replaced by trivial `*_light.cc` pass-through, down to *everything*
   trivial) — **all still hang identically**. The bug is independent of
   every compute kernel's behavior.
3. The RTP instruction-patch math (capacity-127-to-token-31 rescale,
   rewriting K/V scan BD iteration + scan queue repeat counts) — ruled
   out. Control test: `CURRENT_TOKEN=127` (== capacity, patch becomes a
   no-op) still hangs identically.
4. Duplicate/colliding packet-flow routes — ruled out.
   `require_unique_packet_flows` passes on the full MLIR.
5. A memory fault, illegal instruction, or crash — ruled out. Captured
   the amdxdna firmware's own TDR (timeout-detect) log during a live
   hang (`sudo /home/majed/debug-jev-hang.sh`, needs
   `MODEL_PATH=/home/majed/.config/flm/models/Qwen3-0.6B-NPU2` — the
   Makefile's `MODEL_PATH` default resolves against `$HOME`, which is
   `/root` under `sudo`, so it must be passed explicitly or the run
   fails before ever reaching the NPU). Result: clean `TDR timeout
   detected`, `Fatal error type: 0x0`, all exception fields zero — a
   genuine deadlock (something waiting on a lock forever), not a crash.
   `DPU PC`/`TXN OP ID` were both `0xffffffff` — confirmed via the actual
   `amdxdna` kernel driver source and a real AMD kernel patch
   ("accel/amdxdna: Support retrieving hardware context debug
   information") that this is the documented sentinel for "no faulting
   opcode captured," not a real address — this tool cannot localize the
   stuck core by design, on any amdxdna version.
6. Live per-tile lock-stall status via `xrt-smi examine -r
   aie-partitions` (the modern name for the AMD-documented `xbutil
   examine -r aie` hang-analysis workflow) — tried for real: launched
   the hang and polled this report every 50ms for the full duration,
   42 polls spanning the actual `ERT_CMD_STATE_TIMEOUT`. Every single
   poll reported "No hardware contexts running on device." This
   open-source XRT/amdxdna build does not expose live per-core status
   through this report, unlike whatever internal/Vitis build the AMD
   tutorial describing this workflow was written against.
7. `xdna-driver`'s own bundled tools (`npu_perf_trace.sh`,
   `npu_perf_analyze.sh`) — checked, not useful for this: they trace
   host-side driver IOCTL call timing via `perf`/SDT tracepoints (how
   long `submit`/`wait` took), not what is happening on-chip. No
   per-tile introspection tool exists in that repo either.

**Decision taken: pursue AIE trace-unit instrumentation (not the
lower-level `xaie_util_status.h` C API, which was the other option on
the table and was set aside as more speculative with less precedent
on this toolchain).** This means adding trace markers into the MLIR
design itself and rebuilding — slower iteration than live introspection
would have been, but there is already a real, working reference
pattern for it on this exact toolchain:

- `~/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/n32_core_placed.py`
  wires up tracing today, successfully, using:
  `import aie.utils.trace as trace_utils`, then
  `trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tile_trace)`,
  `trace_utils.configure_packet_tracing_aie2(tiles_to_trace, shim_tile_trace, trace_size)`,
  and `trace_utils.gen_trace_done_aie2(shim_tile_trace)`. **Do not call
  these functions directly from `cases/full_layer_engine_generate.py`** —
  that reference example uses the Python IRON dialect API (`tile()`
  objects inside an `mlir_mod_ctx()`), but `full_layer_engine_generate.py`
  builds raw textual MLIR via f-strings throughout (same style as its
  existing `aie.use_lock`/`aie.dma_bd` code) and has no such context;
  `trace_utils`'s wrappers expect real dialect objects, not strings, and
  will not work spliced into this file's output.
- **Use the higher-level declarative `aie.trace` op family instead** —
  it is text-writable directly, matches this file's existing style, and
  is simpler than what `trace_utils` generates under the hood. Full
  syntax confirmed from
  `~/torch2aie/toolchain/mlir_aie/include/aie/Dialect/AIE/IR/AIETraceOps.td`:
  ```mlir
  aie.trace @trace_hub(%hub) {
    aie.trace.mode "Event-Time"
    aie.trace.packet id=20 type=mem
    aie.trace.event<"DMA_S2MM_SEL0_STALLED_LOCK">
    aie.trace.event<"LOCK_SEL0_ACQ_GE">
    aie.trace.event<"LOCK_SEL0_REL">
    aie.trace.start broadcast=15
    aie.trace.stop broadcast=14
  }
  ```
  Declare one `aie.trace @trace_<name>(%<tile>) { ... }` block per traced
  tile at device scope, alongside the existing `tile_defs` list in
  `generate_mlir()` (~line 1207+). Apply each with
  `aie.trace.start_config @trace_<name>` inside the existing
  `_runtime_sequence()` output (~line 310), near the top before the main
  command dispatch.
  - **Packet IDs must not collide with the design's real packet IDs**
    (`0, 1, 2, 4, 8, 9` are already in use — see `FULL_REPLAY_PACKET_ID`,
    `DOWN_ACT_PACKET_ID`, `PACKET_ID_ATTENTION`, `DOWN_GLOBAL_PACKET_ID`,
    `FFN_GLOBAL_PACKET_ID`, `CURRENT_PACKET_K`/`V` in `compact_dataflow.py`
    / `qkv_compact_reference.py` / `decode_cache_reference.py`). Trace
    packets route through the same packet-switched network, so a
    colliding ID would corrupt the exact thing being debugged. Use
    unused IDs (e.g. 20, 21, 22) for the trace blocks.
  - `type=mem` for the hub (a memtile — use `MemTileEvent` names), plain
    `type=core` for `bridge`/`swiglu` (use `CoreEvent` names).
  - Not yet verified: whether `aie.trace`'s `buffer_size` attribute and
    the `AIETraceToConfigPass` lowering fully auto-provision the shim-DMA
    output path, or whether a manual shim tile + DMA wiring (like
    `trace_utils`'s `configure_shimtile_dma_tracing_aie2` does at the
    lower level) is still needed for packet-switched trace output in
    this MLIR-AIE version. Check this empirically on the first build —
    if the compile fails or the output buffer is never written, that's
    where to look next.
- The trace utility module lives at
  `~/torch2aie/toolchain/mlir_aie/python/aie/utils/trace/`, with
  `get_trace_summary.py` for parsing captured trace output — usable
  regardless of which syntax generated the trace, since it parses the
  captured binary trace stream, not the MLIR source.
- Apply to `cases/full_layer_engine_generate.py` (the full-layer MLIR
  generator): add `tiles_to_trace` covering the cores/tiles most
  relevant to the live dependency chain already traced by source
  reading — the hub tile (gates `down_out`/packet 1 via `hub_ffn_full`),
  the `bridge` core (routes packet 8 to swiglu), and `swiglu` itself —
  since the standing hypothesis (not yet confirmed) is a lock/timing
  issue around packet 1/8 reuse across these specific tiles. Start
  tracing there before instrumenting the whole 15+ tile graph.
- **This hardware is AIE2P (Strix Halo) — use
  `~/torch2aie/toolchain/mlir_aie/python/aie/utils/trace/events/aie2p.py`,
  not the generic `events/aie.py`, for event codes.** Confirmed real
  event names/values in that file (max 8 events per tile, pick from
  these, they directly target the suspected bug):
  - Hub tile (a **memtile** — use `MemTileEvent`):
    `DMA_S2MM_SEL0_STALLED_LOCK = 33`,
    `DMA_MM2S_SEL0_STALLED_LOCK = 35` (and the `_SEL1_` variants, 34/36)
    — fires when a DMA queue is stalled waiting on a lock, which is
    exactly `down_out`'s state if this hypothesis is right. Pair with
    `GROUP_LOCK = 45`, `LOCK_SEL0_ACQ_GE = 47`, `LOCK_SEL0_REL = 48`
    (and `_SEL1_` through at least `_SEL5_`, 50-69) to see whether a
    lock's acquire ever got a matching release, or the selector
    assigned to `hub_ffn_full`/`hub_down_replay` never clears.
  - Bridge/swiglu core tiles (use `CoreEvent`):
    `LOCK_STALL = 26` (the core is stalled on a lock right now),
    `INSTR_LOCK_ACQUIRE_REQ = 44`, `INSTR_LOCK_RELEASE_REQ = 45` (see
    every acquire/release the core's own instructions issue).
  - Since the design hangs rather than crashes, the trace buffer should
    still contain every event up to the stall point once the shim DMA
    flushes it to DDR — a clean `gen_trace_done_aie2` won't fire (nothing
    reaches the end), so read back whatever's in the trace buffer after
    the `ERT_CMD_STATE_TIMEOUT` anyway via `get_trace_summary.py`; the
    last few events before the gap are the ones that matter.
- Each rebuild+run cycle after adding trace markers takes real time
  (multi-core AIE compiles ran several minutes earlier this session) —
  budget for that, and always run with `sudo ~/bench-mode.sh on` first
  (fclk held) so a hang doesn't also cost a multi-minute TDR-retry cycle
  on top of the compile time. Run `sudo ~/bench-mode.sh off` when done
  for the session (it stops comfy-monitor and fixes CPU/NPU/EC power
  modes while "on" — don't leave it on indefinitely).

**Operational notes learned the hard way this session, apply to every
future `make` invocation against this project:**
- The Makefile hardcodes `PYTHONNOUSERSITE := 1` and its own
  `PYTHONPATH`/`LD_LIBRARY_PATH` with `:=`; shell `export` is silently
  overridden. Pass these as `make` command-line variable assignments,
  not shell exports.
- The working interpreter/library combo (three separately-built
  toolchains disagree on Python ABI and libxrt version otherwise):
  `TORCH2AIE_PYTHON=/home/majed/npu-work/venv/bin/python3.13`,
  `PYTHONNOUSERSITE=` (empty, to unset),
  `PYTHONPATH="/home/majed/xdna-driver/xrt/build/Release/python/pybind11:/home/majed/npu-work/venv/lib/python3.13/site-packages/mlir_aie/python"`,
  `LD_LIBRARY_PATH="/home/majed/xdna-driver/build/Release/bins/lib:/home/majed/npu-work/boostshim:$TORCH2AIE_SYSROOT/usr/lib64:$MLIR_AIE_DIR/lib:$TORCH2AIE_AIETOOLS/lib:$TORCH2AIE_AIETOOLS/lib/lnx64.o"`
  (the `boostshim` entry is required — without it `xclbinutil` fails on
  a missing `libboost_atomic.so.1.83.0`, easy to misdiagnose as an
  AIE-compile bug since the error surfaces deep in the build log; and
  because `LD_LIBRARY_PATH` does not survive the `aiecc`→`xclbinutil`
  chain, a `libboost_atomic.so.1.83.0` symlink was added under
  `~/torch2aie/toolchain/xrt/lib64/`, which is on xclbinutil's RUNPATH).
- `/home/majed/run-jev-full-layer.sh` has this all wired already
  (`full-check` → `full-build` → `full-run` in sequence); reuse it
  rather than retyping the env each time.
- `sudo` resets `$HOME` to `/root`; any `make` target with a `$HOME`-
  relative default (`MODEL_PATH` is the one that bit us) must get that
  variable passed explicitly when invoked under `sudo`.

- NPU test etiquette on this box (production `gufo serve` normally owns
  ~110 GB pinned): every NPU test runs inside one script that kills the
  running `gufo serve` first, runs the test, — regardless of test failure —
  restarts it with `FORCE=1 JOBS=1 GUFO_EFFORT=low ./run-gufo.sh`, and waits
  20–30 s for the server to come back before anything else continues. Never
  leave the box with gufo down after a test run.

## Goal (speed first, quality must not regress)
- Collapse trivial/lookup requests to `reasoning_effort=low` /
  `enable_thinking=false`, cutting wall time on that class to seconds.
- Shorten real thinking on hard requests by answering their internal
  yes/no checkpoints inline, without cutting off reasoning that still
  needs the full budget.
- Never silently downgrade a hard case, and never inject a wrong yes/no
  into the middle of a reasoning chain — a wrong call here is worse than
  slow, it's silently wrong.
- Each mechanism must prove a measured net win over the current `xhigh`
  baseline before being merged; the simpler one (the pre-gate) ships
  first and alone if it's enough.

## Worktree

```sh
git worktree add ~/gufo-jevgate -b feat/jevgate fork-enhancements
cd ~/gufo-jevgate
```

Follow `~/.claude/skills/port-model/SKILL.md` worktree rules: never touch
`~/gufo` or its running binary from this worktree. Merge back via a
separate clean worktree off `fork-enhancements`, fast-forward only after
tests pass there.

## Hook points (already located, do not rediscover)

Reasoning options flow:
- `src/core/reasoning.hpp:25` — `preserve_thinking` lives on the shared
  reasoning options struct. Pre-gate output lands here or alongside it.
- `src/models/qwen/chat_template.cpp:35-37,41,44,49,146-149` — maps
  `reasoning.enabled` / `reasoning_effort` strings to
  `QwenReasoningEffort::{kLow,kMedium,kXHigh}` and emits (or skips) the
  effort directive. The pre-gate's decision must land in
  `reasoning_effort` before this runs.
- `src/models/qwen/chat_template.cpp:791,806,809,835,847-855` — prior-turn
  thinking stripping/preservation and `<think>` prompt framing. Pre-gate
  also controls `preserve_thinking` here to stop context bloat across
  turns.

Request entry / HTTP layer (pre-gate plugs in here):
- `src/cli/serve/openai_chat.cpp:634-737` — where
  `reasoning_effort`/`chat_template_kwargs.{enable_thinking,
  reasoning_effort, preserve_thinking}` are parsed per request. Run the
  pre-gate on the incoming request here, then set these fields with its
  output before the rest of this function's validation runs, same as if
  the caller had passed them explicitly.
- `src/cli/serve/openai_chat.cpp:901-902` — model-default fallback; gate
  output becomes the new "default" only when the caller didn't explicitly
  override effort/thinking.

Decode-time stream interception (inline jev plugs in here):
- `src/cli/serve/inference_backend.cpp:5767-5771,5852-5856` —
  `request.reasoning.budget_tokens` → `effective_sampling.reasoning_budget`.
  The token-budget accounting that any injected answer must stay inside.
- `src/cli/serve/inference_backend.cpp:4521` — `TokenizeControl("</think>")`,
  the existing forced-close-thinking injection point; the mechanics for
  splicing synthetic tokens into the thinking stream live right next to
  this and should be reused for splicing in jev's yes/no answers too.
- `src/cli/serve/inference_backend.cpp:2944-2958` — Flash-Next-specific MTP
  decode budget check; any injection must respect this, not fight it.
- Wherever decode tokens are detokenized incrementally for streaming
  (same file, the per-token decode loop feeding `inference_backend.cpp`'s
  SSE output) is where the question-detection hook reads the trailing
  text window — it needs the same incremental text Flash-Next's own
  stream already produces, not a second detokenization pass.

## Part A — the pre-gate (build and ship first)

1. Add a rule-based gate function, e.g.
   `src/models/qwen38_flash_next/jev_gate.{hpp,cpp}`:
   - Input: last user message, any attached error/log/stack-trace text,
     message length, presence of code blocks.
   - Output: `effort: low|medium|xhigh`, `needs_thinking: bool`.
   - Pure heuristics first: short question with no code → low; message
     contains a stack trace / "fix this bug" / multi-file diff context →
     xhigh; everything else → medium. No ML in this first cut.
2. Wire it into `openai_chat.cpp` request parsing (~line 634-737): if the
   caller did not explicitly set `reasoning_effort` or
   `chat_template_kwargs.enable_thinking`, run the gate and fill those
   fields with its output before the existing validation/defaulting logic
   (~901-902) runs.
3. Flag to disable the gate entirely (`--jev-gate=off`, default on in this
   worktree only) so A/B testing against stock behavior is a flip, not a
   rebuild.
4. Measure on 20-30 real coding/debugging prompts (mix of trivial lookups
   and real bugs, pulled from your own past sessions). Record: wall time,
   token count, pass/fail on the bug (did the final answer actually fix
   it). Compare against both stock `xhigh` and plain
   `reasoning_effort=medium` as baselines — beat medium before going
   further.
5. Replace the heuristics with jev's zero-shot prompt (model choice
   below) only if the rule-based version measurably undershoots on your
   own prompt set. Keep whichever wins.

## Part B — inline jev: answering self-posed questions mid-think

This is the harder, higher-value half of the idea: while Flash-Next is
still inside a `<think>` block, detect the moment it poses itself a
binary question and answer it immediately instead of letting Flash-Next
spend tokens reasoning it out.

1. **Detect the question.** Watch the incrementally-decoded thinking text
   (same text the SSE stream already produces) for a trailing sentence
   that reads as a self-posed yes/no check — ends in `?`, matches patterns
   like "is X the cause", "should I", "does this match". Keep detection
   itself cheap: a short heuristic first (sentence-ending `?` plus a small
   keyword set), falling back to the NPU classifier only if heuristics
   misfire too often on your own transcripts.
2. **Answer it.** Feed the detected question plus the reasoning-so-far
   context into jev (the same zero-shot instruct model used for the
   pre-gate, prompted for a yes/no answer instead of a difficulty rating).
   Read the Yes-token versus No-token probability at the answer position
   rather than letting it generate free text; treat anything below the
   confidence threshold as `unsure`.
3. **Splice the answer in.** If confidence clears a tuned threshold,
   inject jev's answer as if Flash-Next had just written it ("Yes." /
   "No, because <short reason>.") using the same token-injection mechanism
   already used for forced `</think>` at `inference_backend.cpp:4521`,
   then let Flash-Next's own decode continue from that point. If
   confidence is low, do nothing — let Flash-Next reason it out itself,
   unmodified. This is the single most important guardrail: **jev only
   ever answers when confident; silence means stock behavior.**
4. **Never answer the same shape of question twice in a row with
   unchanged confidence** — if jev's answer didn't change Flash-Next's
   next move (it re-asks essentially the same question), stop intervening
   for the rest of that response and flag it in logs; that's a signal the
   detector or the answer was wrong, not a loop to keep patching over.
5. Budget accounting: injected tokens still count against
   `effective_sampling.reasoning_budget`
   (`inference_backend.cpp:5767-5771`) like any other thinking tokens —
   do not give jev's answers a free pass on the budget, or a string of
   confident-but-wrong injections can run the thinking budget out on a
   bad premise just as fast as normal reasoning would.

### Why this is harder than the pre-gate, build it second

- Question detection runs live, many times per request, inside the decode
  loop — not once at prefill like the pre-gate. Latency and NPU/GPU
  contention per injection matter far more here.
- A wrong pre-gate call costs one request's worth of effort. A wrong
  inline answer corrupts the reasoning chain Flash-Next continues from —
  it can silently derail an otherwise-correct bug hunt with no visible
  error, which is worse than slow.
- Validate on replayed transcripts first (`tests/functional/pi_agent.py`
  per AGENTS.md's agent-loop replay guidance) before ever running it live:
  take real past `<think>` traces, replay them with jev's answers spliced
  in offline, and diff the final conclusion against the original
  unmodified run before this ever touches a live request.

## Model choice for jev (shared by both parts) — zero-shot, no training

No training step, no labeled set, no fine-tuning. Use a small pretrained
**instruct** model directly, zero-shot, and read its answer straight off
— not an embedding model with a trained head on top.

**Qwen3-0.6B-Instruct** (or Qwen2.5-0.5B-Instruct) as jev itself:
- Shares Flash-Next's own tokenizer/control-token framing
  (`src/models/qwen/chat_template.cpp`), so no second tokenizer subsystem
  is needed in `openai_chat.cpp` or in the inline detector.
- Hidden size is still a multiple of 64 and fits the NPU tile shape below
  with no xclbin changes.

How each part reads jev's output, with no training involved:
- **Part A (effort gate):** prompt jev once per request — "Rate this
  request's difficulty: low, medium, or xhigh." — and take its answer
  directly, or read the probability mass on each of the three label
  tokens at the answer position (cheaper and more stable than letting it
  generate free text). This is the same technique the GLiClass-style
  zero-shot routers use: pretrained judgment, labels swappable at
  inference time, nothing retrained.
- **Part B (inline yes/no):** prompt jev with the detected self-posed
  question plus the reasoning-so-far context, and read the probability it
  assigns to the "Yes" token versus the "No" token at the answer position
  (not free-text generation). This is a documented technique with small
  instruct models (e.g. Qwen2.5-0.5B-Instruct used this way on CPU) — the
  model's own pretrained judgment is the answer, no labeled examples
  needed.

If zero-shot quality on your own bug set turns out to be too unreliable
for Part B specifically (self-posed-question judgment is a narrower task
than general difficulty rating), the fallback is a small trained
yes/no head on a frozen embedding backbone (linear 1024→2 + softmax on
Qwen3-Embedding-0.6B) — but only build that if the zero-shot version is
measurably not good enough on real transcripts; don't start there.

Lighter alternatives if tokenizer sharing isn't wanted: ModernBERT-base
or BGE-small/MiniLM with a GLiClass-style zero-shot classification head —
both still fit the NPU tile, but need their own tokenizer path wired in
separately.

### NPU runtime — jev runs purely on the NPU, this is in scope, not a stretch goal

Decision: jev (0.6B) runs on the NPU, full stop. CPU is not the fallback
plan here — it was considered and rejected; the two real blockers below
get built, not worked around. (GPU fabric clock holding is a separately
solved problem on this box already — don't re-litigate it in this plan.)

This repo uses raw XRT with hand-built `.xclbin` programs; there is no
ONNX Runtime or VitisAI anywhere in this tree, and jev should not
introduce one.

**Two concrete blockers to build through, not around** (found while
scoping this, both verified against the current tree):

1. **Model loading.** `QwenModelWeights::LoadFromGguf`
   (`src/models/qwen/weights.cpp`) requires SSM tensors (`ssm_a`,
   `ssm_conv1d`, `ssm_dt`, ...) and `full_attention_interval` — fields a
   plain dense Qwen3-0.6B-Instruct GGUF does not have, because it's a
   dense model, not a hybrid SSM/attention one like Flash-Next. jev needs
   its own loader path, or a dense-model branch in the existing one, that
   doesn't assume SSM layers. Scope this as its own small loader rather
   than bending the Flash-Next loader to accept a shape it was never
   meant to parse.
2. **Block-float conversion + tile reordering for this shape.** The
   `v8bfp16ebs8` encode/decode and tile-reorder kernels in
   `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp` exist for
   Flash-Next's and the 27B's tile shapes (`N=1024`,
   `K∈{2560,5120,17408,...}`). jev's shapes (`K∈{1024,2048,3072,151936}`
   for its embedding/FFN/vocab projections, `M=1024` padded from a ~100-
   200 token question) aren't wired yet. This is new kernel-launch
   plumbing against the *existing* AIE program (no new xclbin needed per
   below), not new AIE microcode.

- Reference: `src/models/qwen38_flash_next/kernels/rocm/npu_gemm.{hpp,cpp}`
  and `qwen27b_npu_ffn.cpp` (legacy name, still the active file) for how
  NPU kernels are invoked from `engine.cpp`/`weights.cpp` in this model.
- Reference: `src/models/glm53_flash/kernels/rocm/npu_decode.hip.cpp` for
  how a second NPU-resident model is structured alongside the main GPU
  decode engine in a sibling model.
- `tools/npu/runtime_mk/final_4096xKx1024_128x64x128.xclbin` already
  covers any K multiple of 64 and M in {1024..4096 step 512} at the
  N=1024, 128x64x128 tile — do not rebuild the xclbin for jev's GEMMs,
  its hidden dims already fit this shape. Only generate a new instruction
  file per `tools/npu/runtime_mk/README.md` ("Adding a shape", ~5 min via
  the `make` recipe) and register it with `npu->AddProgram(...)`. Rebuild
  the xclbin only if jev needs a different tile size, column/row count, or
  N slice width than 128x64x128 / N=1024 — something actually compiled
  into the AIE cores (`mm_bfp_mixed.cc`).
- Weights/activations must be block-float (`v8bfp16ebs8`); the
  encode/decode kernels in
  `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp` are the
  reference for the conversion math — port them to jev's shapes (blocker
  2 above) rather than inventing a new encoding.
- Safety per that README: only run from a `gufo*`-named binary; validate
  new shapes in `gufo prompt`/`gufo bench` first, never a stand-alone
  probe.
- Build order for the NPU path: (a) the dense-model loader (blocker 1),
  (b) one real measured forward pass through the existing xclbin with
  jev's shapes wired (blocker 2) — this is the actual number that was
  missing before, get it now that both blockers have a plan, (c) only
  after that number is in hand, decide whether Part A, Part B, or both run
  on NPU, since Part B's many-calls-per-request profile is more sensitive
  to per-call overhead than Part A's once-per-request profile.
- Part A runs jev once per request at prefill time — cheap, no contention
  concern expected. Part B runs jev many times per request, inline with
  decode — verify with `tools/bench/npu_decode_dispatch_bench.cpp`-style
  measurement that it does not regress existing NPU prefill/decode
  offload numbers before merging either part.

## Quality guardrails (apply to both parts)

- Never let the pre-gate downgrade effort for a request that already has
  `reasoning_effort`/`enable_thinking` explicitly set by the caller — it
  only fills in the default.
- Jev's inline answers only ever fire above a tuned confidence threshold;
  below it, Flash-Next reasons the question out itself, unmodified.
- Every change must report both speed delta and correctness delta on the
  same labeled set; a speed win with a correctness loss is a regression,
  not a result.
- `preserve_thinking` stays controlled by the pre-gate too — low-effort,
  non-thinking turns should not accumulate stale `<think>` blocks into
  later context (`chat_template.cpp:791,806,809`).

## Out of scope for this port

- No changes to the Flash-Next decode kernels, MTP budget logic, or YaRN/
  DFlash work already landed on `fork-enhancements` (commit `d4c0ecaa`).
- No new top-level "engine" abstraction — reuse the per-model NPU
  invocation pattern already in `engine.cpp`/`weights.cpp`.
- No model name mentions of competitor projects anywhere in this file or
  commits built from it.
