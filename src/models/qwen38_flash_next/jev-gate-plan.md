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

## Status 2026-11-05c — architecture decision: NPU+light-GPU-glue hybrid confirmed as production, NOT full-NPU microcode

Reconciles a contradiction between this file and
`~/torch2aie/examples/qwen3-decode-layer/TRACE-HANG-INSTRUMENTATION.md`:
this file's "Status as of 2026-11-03" section below says the full-layer
hang is still the open blocker — **that's stale.** It was resolved on
2026-11-04 (root cause was the test oracle's stale 8B-model GQA
constants, not an NPU bug) and that project went on to a full 28-layer
pure-NPU Qwen3-0.6B decode at 3.87 tok/s (zero GPU, zero CPU compute).
Full history in that doc, not repeated here.

Given that real alternative exists, the question "should jev be
pure-NPU (zero GPU) or NPU-GEMMs+light-GPU-glue" was asked and answered
with real numbers from both paths:
- **Current hybrid** (`jev_npu.hip.cpp`, this worktree): NPU GEMMs are
  ~98% of FLOPs per decision (36-43ms); the entire GPU glue (embed,
  RMSNorm, RoPE, attention, block-float encode/decode, final Yes/No
  scoring) is one ~9.7ms burst (encode 6.0 + decode 1.4 + attention
  2.3) — not sustained GPU load. **61-77ms/decision end to end.**
- **torch2aie fully-NPU path**: zero GPU, zero CPU compute, but ~258ms/
  decision (3.87 tok/s for the full 28-layer forward) — **~4x slower**.

**Decision: keep the NPU+light-GPU-glue hybrid as production jev.**
Accept the bounded, short GPU-glue burst for 4x the speed. The
torch2aie fully-NPU decode layer is not being adopted for jev; it
remains a proven, working artifact (in case GPU contention turns out
to matter more than measured once jev runs concurrently with real
Flash-Next decode load — re-open this decision if that measurement
ever shows real contention, don't re-litigate it without one).

**Current real status (2026-11-05, from the live punch list), supersedes
scattered status entries below where they conflict:**

| Model | Quant | NPU runnable | Battery | Notes |
|---|---|---|---|---|
| Qwen3-0.6B | Q6_K | yes, live judge | 6/10 trivia, 8/12 code | current default |
| Qwen3-1.7B | Q6_K | yes (insts generated) | frozen all-Yes (explained: thinking-model artifact, see below) | CPU ref: 8/10, 11/12 with no-think suffix |
| MiniCPM5-1B/2B | Q6_K (download `mradermacher`, not the local Q4_K_M) | not yet | — | Llama arch; no new dequant kernels needed now that a Q6_K build exists; see scoping below (arch-gate/qk-norm-guard/tokenizer-preset items) |
| Qwen3-{0.6B,1.7B}-hf (bf16) | — | CPU reference only (`/tmp/refenv`) | — | used to validate NPU faithfulness, not for production |

**The big finding (unblocks 1.7B):** Qwen3 small models answer bare
yes/no prompts with Yes saturated (+9 logit gap, every framing tried
including few-shot) — reproduced on torch CPU, so this was never an NPU
bug. Fix: Qwen3's official no-think suffix
(`<|im_start|>assistant\n<think>\n\n</think>\n\n`) after the assistant
turn. CPU-verified: 1.7B goes from useless (frozen Yes) to 11/12 code,
8/10 trivia.

**Punch list (current), in priority order:**
1. Port the no-think suffix into the judge's C++ prompt framing
   (`JevNpuModel::Impl::Run`/`RunBatch` in `jev_npu.hip.cpp`) — the
   unlock for 1.7B.
2. NPU A/B: 1.7B vs 0.6B with the new framing → pick the production
   judge model.
3. Calibrate on real coding-agent traffic, not just the trivia/code
   battery.
4. MiniCPM5-1B/2B port — download the Q6_K GGUFs (`mradermacher`, see
   below; no new dequant kernels needed, that was dropped once a Q6_K
   build was found) plus the
   architecture-gate/qk-norm-guard/tokenizer-preset items already
   scoped below.
5. Serve integration: `--jev-pre` exists already
   (`EnsureJevAnswer`/`TryJevPreRoute` in `inference_backend.cpp`/
   `openai_chat.cpp`) but is gated off in production because of the
   pre-fix calibration bug; repoint it at the fixed Q6_K/no-think judge
   and re-run the 3-arm measurement (`tests/functional/jev_gate_measure.sh`).
6. Chase the NPU time floor: measured 36-43ms vs. ~21ms theoretical
   per-decision NPU time, and the 61 vs 76ms call-to-call variance —
   not yet root-caused.
7. Cleanup: debug prints still in `jev_npu.hip.cpp`, format + commit
   (`feat/jevgate`, currently all uncommitted).

## Next: MiniCPM5-1B/2B port (scoped, not started)

MiniCPM5-2B GGUF found at `/mnt/raid0/user_cache/huggingface/hub/models--bartowski--MiniCPM5-2B-GGUF`
(`MiniCPM5-2B-Q4_K_M.gguf`; no 1B found in this cache, only 2B). Checked
its metadata with `gguf-dump` (`~/.local/bin/gguf-dump --no-tensors`,
much more reliable than hand-parsing the GGUF header -- don't redo that):
`general.architecture = 'llama'`, `block_count=42`,
`embedding_length=2048`, `feed_forward_length=6144`,
`attention.head_count=16`, `attention.head_count_kv=2`,
`rope.freq_base=5000000.0`, `rope.dimension_count=128`,
`attention.layer_norm_rms_epsilon≈1e-6`. Same architecture family
`jev_npu.hip.cpp` already supports generically (RMSNorm + RoPE + GQA +
SwiGLU) -- this is a near drop-in, not a kernel rewrite, but three real
things block it, found by reading `JevNpuModel::LoadFromGguf`
(`jev_npu.hip.cpp:1462+`) directly, not guessed:

1. **Architecture gate.** Line ~1465 hardcodes
   `*architecture != "qwen3"` and rejects anything else. The rest of the
   loader already derives its metadata key prefix from the architecture
   string itself (`prefix = *architecture + "."`), so allowing `"llama"`
   too should make the `llama.block_count` etc. lookups work without
   further changes to that part.
2. **Q/K-norm tensors are assumed to exist, unconditionally.**
   `L.q_norm = BindTensor(reader, b + "attn_q_norm.weight")` /
   `k_norm` likewise (`jev_npu.hip.cpp:1513-1514`), then
   `DequantizeRow(impl->layers[l].q_norm, ...)` at load time
   (`:1669-1672`) and `rocm::RmsNormRows(d_q, h_qnw, ...)` /
   `RmsNormRows(d_k, h_knw, ...)` at runtime in **both** `Run()` and the
   `RunBatch()` added above. Per-head Q/K RMSNorm before RoPE is a Qwen3
   architectural feature, not a standard Llama-architecture one --
   MiniCPM5 (plain `llama` arch) almost certainly has neither tensor.
   `BindTensor` on a missing tensor returns one with `data == nullptr`
   (`JevTensor::empty()` checks exactly this,
   `jev_npu.hip.cpp:53-60`); `DequantizeRow` does
   `tensor.data + row * row_bytes` with no null check
   (`jev_npu.hip.cpp:112-117`) -- this **will crash** on load for a
   model without these tensors, not fail gracefully. Fix: guard the two
   `DequantizeRow` calls with `!tensor.empty()`, add a
   `bool has_qk_norm` to `JevConfig` set at load time, and skip the
   `RmsNormRows(d_q/d_k, ...)` step at runtime when false (feed `d_q`/
   `d_k` straight into RoPE unnormalized instead of through `d_qn`/
   `d_kn`) -- applying RMSNorm with a zero/garbage gamma would silently
   corrupt output rather than just fail, so this must be skipped, not
   defaulted.
3. **Tokenizer preset unverified.** MiniCPM5's GGUF declares
   `tokenizer.ggml.pre = 'minicpm5'`, a custom pre-tokenizer regex
   preset name. `QwenTokenizer::CreateFromGguf` (used unconditionally by
   this loader, `jev_npu.hip.cpp:1519`) needs to actually support that
   preset name in its regex table -- check this *before* trusting any
   MiniCPM5 output, since an unrecognized preset is the kind of failure
   that can produce plausible-looking but wrong tokenization silently,
   not a loud error. Check `tokenization::QwenTokenizer`'s preset table
   for `"minicpm5"` (or whatever fallback it uses for an unknown preset)
   before doing anything else in this port -- if that's wrong, the rest
   of the port isn't worth starting yet.

**Model source changed (2026-11-05d): download Q6_K instead of adding
Q4_K/Q8_0 dequant kernels.** The only GGUF found locally
(`bartowski/MiniCPM5-2B-GGUF`, Q4_K_M) and the official `openbmb` repos
only ship F16/Q4_K_M/Q8_0 -- neither is a quant type `jev_npu.hip.cpp`
currently dequantizes (only `Q4_0`/`Q6_K` are implemented). Adding
Q4_K/Q8_0 dequant kernels was the original plan but is real, unnecessary
engineering when a Q6_K build already exists: `mradermacher` mirrors the
full standard llama.cpp quant matrix (confirmed via the HF API, not
guessed) for both sizes, instruct-tuned (not `-Base`) variants:
- 2B: `mradermacher/MiniCPM5-2B-GGUF` → `MiniCPM5-2B.Q6_K.gguf`
- 1B: `mradermacher/MiniCPM5-1B-SFT-GGUF` → `MiniCPM5-1B-SFT.Q6_K.gguf`
  (`-SFT` is the instruct checkpoint name for the 1B release; the 1B
  i1-GGUF variant also has `Q4_0` if Q6_K turns out to need
  requantization-level accuracy checking, but start with Q6_K)

Download these instead of starting point 2's Q4_K/Q8_0 dequant work --
that work is now out of scope for this port entirely, not just
deferred, since there's no remaining reason to need it.

Build order once resumed: (1) download the Q6_K GGUFs above, (2)
tokenizer preset check (cheapest to verify, blocks everything else if
wrong), (3) architecture gate (point 1), (4) q/k-norm guard at load +
both runtime paths (point 2), (5) build, then validate with `gufo judge
--model .../MiniCPM5-2B.Q6_K.gguf --battery` and `--battery --batched`
against known-answer verdicts the same way the Qwen3 batching change was
validated above -- bit-identical verdicts between single and batched
paths is the correctness bar, not just "it doesn't crash."

## Status 2026-11-05e — MiniCPM5 port: mechanically done, output blocked by the same suspected bug as Qwen3-1.7B

Done (builds clean, `MiniCPM5-2B-Q6_K.gguf` loads and runs end to end, no crash):
- Q6_K GGUFs downloaded to `~/jev-model/` (`MiniCPM5-2B-Q6_K.gguf`,
  `MiniCPM5-1B-SFT-Q6_K.gguf`), no new dequant kernels needed.
- Architecture gate accepts `llama`; `has_qk_norm` guards the load-time
  dequant and the runtime Q/K RMSNorm (copy `d_q`->`d_qn`, `d_k`->`d_kn`
  when absent).
- Tokenizer: `minicpm5` preset now maps to `kQwen35` (was silently `kNone` =
  no pre-split regex at all). Known gap: minicpm5 groups digits `\p{N}{1,3}`,
  `Qwen35PieceEnd` emits one digit at a time. NOT yet byte-verified against
  the HF tokenizer.
- Prompt: MiniCPM5's template is ChatML like Qwen (confirmed from the GGUF
  `chat_template`) and its no-think suffix is byte-identical, but it also
  prepends `bos_token` (`<s>`); added `Impl::bos_prefix`, set only for the
  `llama` architecture.
- N=2560 (2B qkv) has no instruction file and `aiecc` aborts in
  `generateCdoArtifacts` generating it. Worked around by padding: `n_qkv_cap`
  = smallest existing N >= n_qkv (4096), `encode_one` now sizes the weight
  buffer to the padded width (its 4th arg was previously ignored), and
  `JevSplitQkvKernel` takes the padded row stride. No effect on models with
  an exact match.

Result: 2B battery 6/10 trivia, 7/12 code, every answer 0.94-0.99 Yes; all
"No" cases miss. Re-run after the Qwen3-1.7B fixes landed (see
`JEV-1.7B-BUG.md` Resolution: broken weight encoding + softmax max
reduction, NOT `q_dim == hidden` as first suspected) gave the same result,
so the engine is no longer the prime suspect. Remaining suspects, none
verified: (a) the model genuinely saturates to Yes on this bare framing (as
Qwen3-0.6B does on CPU: `sep=NO`); (b) tokenizer mismatch (digit grouping,
unverified vs HF); (c) the n_qkv 2560->4096 padding path (first model to use
it). Next step is a CPU reference: download `openbmb/MiniCPM5-2B` (HF
safetensors) and run it with `~/jev-ref/nothink_battery.py`-style framing
(`<s>` + ChatML + no-think). If CPU also saturates, it is (a) and the model
is a poor judge; if CPU separates, bisect (b) vs (c).

MiniCPM5-1B (`hidden=1536`) still needs new instruction files (K=1536 base,
qkv n_qkv=2560 padded to 4096, o K=2048/N=1536, gu K=1536/N=9216, down
K=4608/N=1536); N=1536/9216 may hit the same `aiecc` crash.

## Status 2026-11-05b — batched AnswerBatch: ~2.2-2.6x throughput, verified bit-identical

Added `JevNpuModel::AnswerBatch` (`jev_npu.hip.cpp`'s `Impl::RunBatch`,
`kJevBatchMax`/`kJevSeqCap` constants) — answers up to 4 independent
questions in one NPU pass instead of one question per pass. The NPU
program and every staging buffer were already fixed at M=512 rows
regardless of how many are real tokens (single-question `Run()` already
pads one ~30-token question out to 512 rows every call), so packing 4
questions into 4x128-row slots instead of 1 question into mostly-padding
uses capacity that was already being spent and thrown away. Verified via
`gufo judge --battery --batched` against the existing `--battery`
baseline: **verdicts bit-identical** (same yes/no probabilities to 2
decimals, same 6/10 score, same misses) across all 10 trivia questions —
batching is numerically transparent. Measured: **29.9 ms/question
batched vs 61-77 ms/question single-question — ~2.2-2.6x throughput.**

Implementation: each question gets a fixed `kJevSeqCap=128`-row slot
(`kJevBatchMax=4`, `4*128=M=512`); `JevRopeKernel` gained a `seq_cap`
param (position resets per slot, `row % seq_cap` not global `row`);
new `JevStoreKvBatchedKernel`/`JevAttnBatchedKernel` variants added
rather than modifying the single-question kernels in place (`AttnBatched`
takes `seq_cap` + a `d_lens` device array so causal attention never
crosses a slot boundary or reads past that question's own real length);
GEMM projections (qkv/o/gu/down) needed no changes at all since they're
purely row-independent. `judge.cpp` gained `--batched` wired through a
new `print_battery_batched` using `AnswerBatch` directly, for exactly
this kind of regression check going forward.

Not yet done: wiring this into the real Part A/B call sites (still
single-question `Answer()` there) — `AnswerBatch` exists and is proven,
but nothing outside the judge CLI calls it yet. For Part A (one pre-gate
call per request) batching doesn't apply directly.

## Status 2026-11-05 (late) — 1.7B CORRECT: two engine bugs fixed; A/B settled (1.7B wins)

The "garbage output" bug was TWO real engine bugs (full write-up +
ruled-out list in `JEV-1.7B-BUG.md`, now RESOLVED):

1. `encode_one` encoded every 128-col weight tile from rows 0..127 with
   K truncated to 128 (`row0`/`k_valid` args to `NpuEncodeWeightsQuant`).
   Fixed: one full-width encode call per source into the fused buffer.
2. `JevAttnKernel`/`JevAttnBatchedKernel` softmax max reduction kept each
   thread's butterfly *suffix* max and skipped `red[0]`, weighting keys by
   `exp(s_p − suffixmax_p)`. Near-uniform early layers hid it (cos 0.999
   at L0–L2), peaked deep layers diverged (0.23 by L27). Fixed: adopt
   `max(red[0..warps])`. (The `sum` reduction was already correct.)

Validation vs torch fp32 CPU reference (same no-think framing, n=54):
qkv-l0 elementwise cos 0.9999; `s2qkv` L13 = 2.485e6 vs torch 2.486e6
(0.04%). **A/B settled — 1.7B is the production judge:** NPU 1.7B
trivia 7/10 + code 11/12 = **18/22** (CPU 19/22; its 3 misses are the
CPU model's own), NPU 0.6B 14/22 (CPU 13/22 — 0.6B base framing has
`sep=NO`: every logit gap positive, so 0.6B cannot veto negatives at all).
Timings (exclusive NPU, single calls): **1.7B ≈ 140 ms**, 0.6B ≈ 76 ms;
batched (4 q/pass) 44.6 ms/question. `--batched` (RunBatch) verified
identical scores — that's what serve would batch into later.

`--jev-pre` serve wiring: `EnsureJevAnswer` now prefers the NPU judge on
GUFO_NPU builds (GUFO_JEV_NPU=0 forces CPU), default model
`$HOME/jev-model/Qwen3-1.7B-Q6_K.gguf`; CPU 0.6B snapshot stays the
fallback. End-to-end serve test under `GUFO_JEV_PRE` in progress below.

Calibration note: with the no-think framing the verdicts are saturated
(0.00/1.00 on 19 of 22 battery questions); the useful calibration axis
for real traffic is `kJevPreConfidence` in `openai_chat.cpp` (0.9) + the
yes/no prompt detection, not per-question scores.

## Status 2026-11-05 — standalone `gufo judge` fast + healthy; yes-bias root-caused; porting no-think framing

Standalone judge (`gufo judge`, `src/cli/judge/`) on Qwen3-0.6B-Q6_K:
**61–77 ms/call** (from 82–118), verdicts stable across repeats, battery
trivia 6/10 + code 8/12. Perf work landed in `jev_npu.hip.cpp`: decode
`n` rows instead of the 512 padding, activation encode window `enc_rows`
(n rounded to 128) instead of 512, and NPU staging buffers (`a_hidden`,
`c_qkv`, …) allocated once in `Impl` instead of per call (that alone was
11–15 ms of `pre`). A partial-buffer `Submit` sync was tried and reverted:
it isn't needed and (like the buffer-reuse mistake below) faults.

**Call-2 `Memory Fault Error` in `NpuDecodeOutputKernel` — RESOLVED.** Root
cause was allocating `qkv`/`gu` pinned host buffers once per process
(cached in `h_qkv`/`h_gu`); allocating them per `Run()` call is clean
(proved by A/B in an exclusive NPU window). Persistent NPU *staging*
buffers are fine and stay. Post-fix battery shows no stale poisoning
(`yes=0.000 no=0.000` symptom gone). Health check rule: validate every
perf build with a multi-question battery in ONE process; repeats of one
question hide stale-data poisoning. `~/jev-test.sh` wraps the exclusive
window (stop supervisor+serve → test → `setsid` restart of
`FORCE=1 JOBS=1 GUFO_EFFORT=low ./run-gufo.sh` → wait `/v1/models`; all
fds of the restart must be redirected or the caller hangs on the pipe).

**Judge intelligence (the "smarter for code" ask) — root-caused, not a
compute bug.** Qwen3 models are thinking models: asked "Answer Yes or No"
directly, raw Qwen3-1.7B on torch CPU bf16 answers p(Yes)=1.000 on ALL 22
battery questions (+9 to +11 logit gap) for every single-shot AND few-shot
framing tried; 0.6B separates weakly (13/22). The frozen all-Yes verdicts
seen from the NPU 1.7B judge therefore mirror real model behavior —
NPU math validated as faithful. The lever is Qwen3's **no-think suffix**:
framing the assistant turn as `<|im_start|>assistant\n<think>\n\n</think>\n\n`
(official `enable_thinking=False` template) gives **1.7B 19/22 total,
11/12 code** on the CPU reference (sweep scripts in `~/jev-ref/`:
`nothink_battery.py`, `prompt_sweep.py`; `/tmp/refenv/bin/python`,
`Qwen3-{0.6B,1.7B}-hf` + `-GGUF` in `~/jev-model/`). Next: append that
suffix (token ids via the judge tokenizer) in `JevNpuModel::Run` framing
and adopt 1.7B (Q6_K) as the judge model; 0.6B stays the fallback.
MiniCPM5-1B/2B ports follow.

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

Decode-time hook points:
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

## Part B removed (2026-11-08)

Part B (inline answering of the model's own yes/no questions inside
`<think>`) was dropped completely: the engine flag `--jev-inline`, the
decode-time injection hook in the scheduler, `JevInlineScanner`
(`jev_inline.{hpp,cpp}`) and the offline `gufo judge --replay` mode no longer
exist. Reason: the judge answered Yes too often and lacked the context to be
safe (replay evidence in git history before this change). The pre-gate, the
agent gate, the pre-route, the finish detector and the goal tracker remain.
These are the pieces being ported to an opencode plugin
(see `docs/JEV-OPENCODE-PLUGIN.md`).

## Model choice for jev — zero-shot, no training

No training step, no labeled set, no fine-tuning. Use a small pretrained
**instruct** model directly, zero-shot, and read its answer straight off
— not an embedding model with a trained head on top.

**Qwen3-0.6B-Instruct** (or Qwen2.5-0.5B-Instruct) as jev itself:
- Shares Flash-Next's own tokenizer/control-token framing
  (`src/models/qwen/chat_template.cpp`), so no second tokenizer subsystem
  is needed in `openai_chat.cpp`.
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

If zero-shot quality on your own set turns out to be too unreliable, the
fallback is a small trained yes/no head on a frozen embedding backbone (linear
1024→2 + softmax on Qwen3-Embedding-0.6B) — but only build that if the
zero-shot version is measurably not good enough on real transcripts; don't
start there.

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
  after that number is in hand, decide whether Part A runs on the NPU.
- Part A runs jev once per request at prefill time — cheap, no contention
  concern expected; verify with `tools/bench/npu_decode_dispatch_bench.cpp`-style
  measurement that it does not regress existing NPU prefill/decode
  offload numbers before merging.

## Quality guardrails (apply to both parts)

- Never let the pre-gate downgrade effort for a request that already has
  `reasoning_effort`/`enable_thinking` explicitly set by the caller — it
  only fills in the default.
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

## Status 2026-11-06 — MiniCPM5 port WORKS (root causes found)

CPU reference (`~/jev-ref/minicpm_battery.py`): 2B 18/22, 1B 19/22.
NPU after fixes: **2B 9/10 + 9/12 = 18/22 (identical misses to CPU), 1B 9/10 + 8/12 = 17/22**; Qwen3-1.7B unchanged (7/10, 11/12). Batched 2B ~68-81 ms/q, 1B ~31-36 ms/q.

Root causes (the earlier "q_dim/saturation/padding" theories were all wrong):
1. **BOS**: prepending the text `"<s>"` tokenizes to ordinary pieces (`<`, `s>`), not BOS id 0. Now the BOS token id (from the tokenizer) is inserted into the token list (`EncodePrompt`).
2. **RoPE layout**: llama.cpp's llama-arch converter permutes q/k rows so adjacent pairs (2i, 2i+1) rotate together (NORM rope). Qwen3 uses half-split NEOX. `JevRopeKernel` now takes `interleaved`, set for arch `llama`. This was the all-Yes bug on 2B.
3. **1B shapes**: N=1536 / N=9216 instruction streams (generated fine by `gen_insts.sh`) compute WRONG results; N must be a whole number of 1024-wide tiles. The engine now pads o/down/gate-up outputs (`n_o_cap`, `n_down_cap`, `n_gu_cap`: 1536->2048, 9216->10240) like qkv (2560->3072); weight buffers zero-padded, decode uses `valid_cols`. New insts: 512x1536x{1024,3072,10240}, 512x4608x2048 (512x2048x2048 already existed).
4. **1B GGUF vocab is broken**: `MiniCPM5-1B-SFT-Q6_K.gguf` ships a vocab list that does not match its own embedding rows (id 22 = `<tr>` instead of `!`, newline = 65324). Embedding rows match HF ids. Workaround: `GUFO_JEV_TOKENIZER_GGUF=<2B gguf>` borrows the 2B tokenizer (identical to HF). Production use of 1B needs that env or a re-converted GGUF.

Recommendation: 2B (18/22, same as Qwen3-1.7B, ~2x the NPU time per question) is not better than Qwen3-1.7B (18/22); 1B is ~2x faster per batched question (31-36 ms vs 59) at 17/22. Next: A/B on the replay traces before choosing; judge defaults unchanged.

## Finish detector (rule-based "did the model stop unfinished?")

`jev_finish.{hpp,cpp}`: `EvaluateFinishRules` / `JudgeFinish(task, response, had_tool_call, finish_reason)`.
Silence means finished; a turn with a tool call is never flagged. Rules (score): length (1.0), unclosed
code fence (0.95), empty answer on `stop` (0.9), open end `:` `,` `(` (0.85), last word dangling
(0.85, lowercase only), announced next action in the last sentence (0.8), plan with no execution
(0.7), unbalanced closed code block (0.65), unpunctuated last paragraph (0.62), tiny answer to a long
task (0.45) and unpunctuated last list item (0.45). Threshold `kJevFinishThreshold = 0.6`.
Handoffs to the reader ("tell me:", "run ...", lines containing you/please) are not flagged.

Replay: `gufo judge --finish-replay <dir> [--finish-verbose]`. Corpus builder `~/jev-finish/build_corpus.py`
(7,230 turns in `~/jev-finish/corpus`): opencode assistant turns (finish stop/length/tool-calls),
`~/jev-gate-run` responses, the AIME #15 stop, and 600 synthetic mid-sentence truncations of finished
answers. Labels: finished = natural stop or tool call; unfinished = finish=length, AIME #15, a stop
that the user's next message answered with "continue"-style text (152 real cases), or a synthetic cut.

Result at threshold 0.6: precision 0.796, recall 0.343 (TP 296, FP 76, FN 567, TN 6291).
- Real user stops (opencode-cont, n=152): recall 0.18 (28), precision 1.0.
- Synthetic mid-sentence cuts (n=600): recall 0.34. AIME #15 (empty answer, finish=stop): caught.
- Per rule precision: length 1.0, stopword 1.0 (83), bare 1.0 (61), fence 0.93, open_end 0.93,
  announce 0.56, empty 0.28.
- The 76 "false positives" are label noise, not wrong flags: 60 are genuinely empty `stop` turns (0 or
  reasoning-only output) and 15 of the other 16 were read by hand and are real stalls ("Let me start
  by understanding the project structure." then stop, tool call written as text, answer cut inside a
  code fence). One is borderline (status report ending "Next I'll do the same pass ..."). Nothing in
  the 6,291 other finished turns was flagged. The "finished" label cannot see a model that stopped
  early without the user saying so, so the strict precision understates the real one.
- Unpunctuated list items (bare_item) fired on 98 finished turns, so it stays below the threshold.

Integration (not wired): at the turn-end hook in `inference_backend.cpp`, when the model finishes with
`finish=stop` and no tool call, call `JudgeFinish`; if `unfinished`, append a continuation user turn
("Continue; the task is not complete." plus the reason) and resume decoding on the cached prefix.
Loop guard: at most 2 auto-continues per request, never twice on the same reason with an unchanged
tail, and never when `finish=length` is the cause (the budget, not the model, stopped it). Recall
is the weak side (real stops are mostly "announce then stop" or empty, which it catches, but also
finished-sounding summaries it cannot see); a jev yes/no on the task plus the tail would add recall
only if a labeled set shows the judge separates them.

## Status 2026-11-06 — MiniCPM5-1B is the default judge; tokenizer fixed

- **Fixed GGUF:** `~/jev-model/MiniCPM5-1B-SFT-Q6_K-fixed.gguf`. HF 1B and 2B `tokenizer.json` are identical (vocab 130072 entries, merges, added tokens), so the 2B GGUF's tokenizer keys (`tokens`, `token_type`, `merges`, `model`, `pre`, ids, chat template) replace the 1B's SPM-style ones (it had `scores`, no `merges`). Tensors are byte-identical (219/219 verified). Patcher: gguf-py (`/mnt/raid0/llama.cpp/gguf-py`), copy of KVs + raw tensors.
- **Scores, no env var:** fixed 1B 9/10 trivia + 8/12 code; batched 32-38 ms/question. Original 1B + `GUFO_JEV_TOKENIZER_GGUF=<2B>`: identical 9/10 + 8/12. 2B 9/10 + 9/12, Qwen3-1.7B 7/10 + 11/12 (unchanged).
- **Engine guard:** `LoadFromGguf` refuses a GGUF whose tokenizer has no BPE merges (the SPM-style conversion; its ids do not line up with the embedding rows) and one where vocab < embedding rows or Yes/No/!/Answer fail to round-trip. The error names the remedy; `GUFO_JEV_TOKENIZER_GGUF` still overrides. The original broken 1B now fails loudly instead of scoring 6/10.
- **Serve default:** `EnsureJevAnswer` tries fixed 1B -> Qwen3-1.7B-Q6_K -> CPU; `GUFO_JEV_MODEL` overrides (single path, no fallback).
- **Calibration needed (no thresholds changed):** `TryJevPreRoute` (openai_chat.cpp) answers a whole "yes or no" request from jev when `max(yes,no) >= kJevPreConfidence (0.9)`. With the graded 1B, ~0% of replay answers clear 0.9, so the pre-route will almost never fire (safe, but its gain disappears); with saturated Qwen3 it fired on ~70% including wrong answers. Part A effort routing (`jev_gate`) does not use this cutoff. Re-pick the pre-route cutoff (likely 0.7-0.8) from labeled yes/no requests; the inline Part B 0.6 cutoff is unmeasured against truth and stays off.

## Status 2026-11-06 — serve check of the 1B default

- `--jev-pre on` serve (GUFO_DIR=gufo-jevgate): fixed MiniCPM5-1B loads on the NPU, 99 ms first call, `[jev-pre] yes=0.89 conf=0.89` (below kJevPreConfidence 0.9, so recalibrate with the 3-arm data).
- Production launcher uses GUFO_NPU_DIR=$GUFO_DIR/tools/npu; a tree without the new `insts_512x*` files (e.g. ~/gufo before this branch merges) silently fell back to the CPU 0.6B judge. Now logged as `[jev] event=npu_judge_load_failed` / `npu_judge_unavailable action=cpu_fallback`.
- MERGE ORDER: decode-1 upstream ports land first, then merge feat/jevgate (carries the insts files + xclbin programs), then decode-2.

## Status 2026-11-06 — `--jev-finish on|off` wired (default off)

- Chat Completions only (`HandleOpenAiChat`): after a turn with `finish=stop`, no tool call, no structured output, `JudgeFinish` runs on (last user message, response text). Unfinished at >= 0.6: re-submit the conversation + assistant text + user turn "Continue; the task is not complete. <reason>" via `backend.start_chat`; the cached prefix covers everything before the appended assistant turn.
- Non-stream: continuation text is appended to one message (`\n\n` separated), reasoning appended, completion/reasoning/prefill tokens summed into usage. Stream: same SSE response continues (new filter state per round, `\n\n` delta between rounds), one terminal chunk at the end.
- Guards: `finish=length` never continues; max 2 continues per request; same reason + same 200-byte tail never continues twice; any exception while continuing ends the response as it stands. Logs `[jev-finish] decision=continue|finished reason=… score=…`.
- Not covered: Responses and Anthropic endpoints (`CreateCompatibilityResponse`), reasoning (`<think>`) is not judged.
- Verified: `tests/cli/openai_chat_test` (off = 1 call and unchanged text; on = 2 calls, appended text, continuation turn shape; finished / length / identical-tail / two-continue cap; streaming). Live: serve with `--jev-finish on` answered 3 normal requests (stream and non-stream) with `decision=finished`. Not verified live: a real model producing an unfinished stop (cannot be provoked on demand).
- Also fixed `openai_chat_test` link (needed jev_gate.cpp, jev_finish.cpp sources).

## Status 2026-11-06 — agent mode for the gate (`--jev-gate agent`)

Why: on real opencode traffic (150k-token tool loops) none of the jev features fired — the request gate bypasses callers that set reasoning controls and only reads the last user message, pre-route needs a 1-message/no-tools/<512 B prompt, and the finish detector skips turns that end in a tool call.

**What opencode sends (could not capture the body):** an `opencode run` through a capture server on :18900 never reached it (the global provider config wins over `OPENCODE_CONFIG`/`OPENCODE_CONFIG_CONTENT`), and server logs do not record bodies. Known: provider `kairic-direct` is `@ai-sdk/openai-compatible` with no `reasoningEffort`/variant in `~/.config/opencode/opencode.json`; the opencode binary contains `chat_template_kwargs`/`enable_thinking`/`reasoning_effort` strings; the serve log showed no `[jev-gate]` line for any opencode request while the same server logged one for a bare curl request, so opencode sets `reasoning.effort` or `reasoning.enabled` itself. The agent policy does not depend on which: it reads the effective effort (caller's, else server default) and only lowers it. TODO: confirm the field once a body is captured.

**Policy** (`ClassifyJevAgentTurn`, jev_gate.cpp; wired in `ApplyJevAgentGate`, openai_chat.cpp): only when the request's last message is a tool result. The pending turn is lowered to `low` only if EVERY tool result is routine: `read`/`glob`/`grep`/`list`/directory-tree results and `edit`/`write`/todo/memory results with no failure word in their first 160/400 chars; `bash` only when the command is read-only inspection (ls, cat, head, tail, wc, stat, pwd, echo, find, grep, rg, tree, file, git status/log/diff/show/branch/ls-files/rev-parse/remote, sed -n, awk, sort, uniq, diff, nl, du, df, which, sleep, test, date, ps, pgrep, optionally behind `cd X &&`), output < 4000 B, no failure word (error, failed, traceback, exception, fatal, panic, segfault, assertion, no such file, not found, permission denied, undefined reference, timed out, timeout, command not found, exit code N). Build/test/run commands, web fetches, skills, questions and unknown tools keep the caller's effort. A user message resets the run; after 5 consecutive routine turns the effort is kept (run-limit). Effort is never raised; `enabled=false` or effort already <= low is untouched. Log: `[jev-gate] mode=agent role=tool turns=N effort_in=.. effort_out=.. reason=..`. In agent mode the chat-style request gate is skipped for tool-result turns.

**Offline replay** over `~/.local/share/opencode/opencode.db` (extract `~/jev-agent/extract.py`, tune `~/jev-agent/classify.py`; 5,852 Qwen3.8 Flash Next steps, 168 sessions, 3.35 M output tokens, ~1.90 M reasoning tokens at 4 chars/token):
- 34.4% of steps would be lowered (2,013), carrying 35.9% of reasoning chars (~0.68 M tokens, ~20% of all output tokens). If low removes 100/70/50% of those: 683k / 478k / 342k tokens = about 81 / 57 / 41 s per session at 50 tok/s (agent loops at 150k depth decode ~38 tok/s, so ~1.3x those seconds).
- Harm proxy (tool failure in the same or next 2 steps): lowered 21.4% vs kept 23.3%, no excess. Caveat: observational; the logged model did NOT run at low effort, so this cannot show a quality effect. The real test is the live opencode A/B.
- 338 lowered steps (16.8%) are ones where the model chose >2000 chars of reasoning; these are the at-risk cases (65 followed by a failure).
- C++ vs Python parity on 5,330 tool-result turns: 5,311 identical (99.6%); 19 differ on run-limit edge cases. C++ has no exit codes (not in the OpenAI message), only failure words; with exit info Python changes 146/5,330 decisions, so ~2.7% of lowered turns would be kept.
- Unit test: `TestJevAgentGateLowersRoutineToolTurns` (tests/cli/openai_chat_test.cpp).

**Finish detector for tool-capable turns:** no code change needed. `JudgeFinish` already runs on any `finish=stop` turn without a tool call, including requests with `tools`; the empty-answer (0.9) and announce (0.8) rules catch an agent stalling. On opencode traffic it does not log because almost every turn ends in a tool call and returns before the log line. Pre-route stays single-turn by design.

**Run it:** `./run-gufo.sh --jev-gate agent --jev-finish on` (plus `--jev-pre on`).

## Incident 2026-11-06 — `--jev-gate agent` broke the prefix cache (fixed)

**Symptom** (/tmp/run-gufo.log 16:19:41): r212 had prompt 207,738 tokens with cached_tokens=207,377 (prefill 361). The next turn was lowered (`mode=agent ... effort_out=low reason=routine-tool-result`) and request=10 prefilled 208,275 tokens from zero at ~1,700 tok/s (>2 min). Flipping back to the caller's effort on the following turn would miss again.

**Cause:** `AppendReasoningInstruction` (src/models/qwen/chat_template.cpp, called at the system prefix) writes the effort directive (`kLowReasoningInstruction` / `kXHighReasoningInstruction`) at the START of the rendered prompt. Any effort change rewrites the first bytes, so every cached token after it is dead. Test `TestJevAgentGateLowersRoutineToolTurns` now asserts that rendering the same conversation at low vs the caller's effort yields different prompts.

**Fix (70099195 -> next commit):** agent mode no longer touches effort/enabled/preserve_thinking. A routine tool-result turn now only caps `reasoning.budget_tokens` at `kJevAgentRoutineBudgetTokens` = 512 (min with the caller's budget, never raised); the cap is enforced at decode time by the existing reasoning-budget machinery (forced `</think>`, inference_backend.cpp BuildReasoningBudget), so the rendered prompt is byte-identical to the ungated render (asserted in the test). Log format: `[jev-gate] mode=agent role=tool turns=N effort=<caller> budget=<cap|0> reason=...`.

**Savings re-estimate** (offline, 5,852 steps / 168 sessions, reasoning tokens ~ chars/4, lowered = 2,013 steps carrying ~683k reasoning tokens; saved = max(0, reasoning - budget) on lowered steps; total reasoning ~1.90M tokens):
| budget | tokens saved | % of all reasoning | lowered steps actually capped | s/session @50 / @38 tok/s |
|---|---|---|---|---|
| 256 | 401k | 21.1% | 627 (31%) | 47.7 / 62.8 |
| 512 (shipped) | 285k | 15.0% | 332 (16%) | 33.9 / 44.7 |
| 1024 | 172k | 9.1% | 140 (7%) | 20.5 / 27.0 |
Plus: no prefill penalty. Risk: a cap truncates thinking abruptly (stock budget semantics); only 16% of lowered steps are touched at 512.

**Same-class issues checked (not changed):**
- Plain `--jev-gate on` sets effort/enabled/preserve_thinking from the last user message only when the caller left them unset. In one conversation each new user turn can classify differently (low vs medium vs xhigh), rewriting the system-prefix directive and missing the cache for the whole conversation at that turn. Also combined with `agent` + caller-unset effort: turn 1 renders at the gate's effort, later tool-result turns at the server default, a miss. Opencode sets its own effort so it is unaffected; for effort-unset clients prefer a budget cap there too (FLAGGED, not changed).
- `--jev-finish` continuation appends assistant+user turns. The prefix up to the previous generation prompt stays cached; the re-rendered assistant turn strips its thinking (it is no longer after the last user message), so only that turn's own tokens re-prefill (small), not the whole prompt.
- Release binary must be rebuilt/restarted for the fix to run live; 8900 was not touched.

## Status 2026-11-06 — agent gate tuned from real opencode data

Data: opencode.db now stores sessions in `session_message` (v2: assistant message = one step; `content` array of `reasoning`/`text`/`tool`; tool `state.input.command`, `state.content[].text`, `state.metadata.exit`). The earlier replay read the legacy `message`/`part` tables, which stop on 2026-11-03; the corrected extract has 8,155 Flash-Next steps in 181 sessions up to 2026-11-06 16:51 (scripts were scratch-only). opencode sends `variant: high` on every request.

**Root cause of "42 of 48 not routine":** newer opencode names its shell tool `shell`, older `bash`. The classifier only knew `bash`, so every `shell` result was "unknown tool -> keep". Tool inventory (all history): read 3841, bash 3519, shell 1756 (143 of the 174 tools in the last hour), edit 699, grep 122, write 99, filesystem_read_multiple_files 88, task 79, glob 56, filesystem_directory_tree 52, todowrite 26, fetch_fetch 26, memory_store_context 19, question 15, get_goal 13, others <10.

**Change:** `shell` is an alias of `bash` (same `command` argument). The `[jev-gate] mode=agent` line now carries `blocker=` (`tool=<name>` unknown tool, `cmd=<word>` non-inspect command, `failure-signal tool=<name>`, `output-large tool=<name>`, each + `blocked=<k>/<n>`; `run=<n>` for the 5-in-a-row back-off; `none` when lowered). One non-routine result among parallel results blocks the whole turn (all_of); 159 of 1,434 multi-result turns in history are mixed.

**Not loosened (measured no gain):** read-only MCP tools (+0.0 pt), graded 1536 cap for passing build/test (+0.0 pt), narrow failure scan for inspect commands (+0.2 pt of reasoning, harm proxy gets worse: 24.8% vs 23.7%). Failure words inside file text (error/failed/exception) are what block 'inspect' results (334 'error' hits), but relaxing it was not clearly safe.

**Offline replay (cap = min(actual reasoning tok, 512), reasoning tok = chars/4):**
| window | rules | steps capped | cap binds | reasoning removed |
|---|---|---|---|---|
| all history (8,155 steps) | bash only | 29.9% | 391 | 11.9% (335k tok) |
| all history | + shell alias | 38.3% | 579 | 16.9% (474k tok, ~58 s/session @45 tok/s) |
| since Nov 6 (199 steps) | bash only | 9.5% | 0 | 0.0% |
| since Nov 6 | + shell alias | 48.2% | 12 | 8.7% (~27 s/session) |
| live window 16:33+ (74 steps) | + shell alias | 51.4% | 2 | 3.7% (~6 s/session) |
Harm proxy (tool failure within next 2 steps): capped 22.9% vs kept 25.1% (history); 19.8% vs 26.4% (today). Weak evidence: capped turns ran uncapped in the data.

**Where the long turns come from (live window, top steps):** 2117 tok = 942 reasoning + 1140 edit/write payload; 1850 = 404 + 1176 payload; 1703 = 1021 reasoning + 604 final text; 1370 (jevgate session) = 137 + 1097 final text; 1367 = 284 + 1058 payload. Overall output composition (all history): reasoning 70.4%, tool-call arguments 23.8%, text 5.7%. Today's reasoning is short (median 163 tok, p90 648, max 1,963), so a 512 cap rarely binds now; the big savings are historical rambles (steps >2000 chars = 18% of steps, 69% of reasoning; ten 10k+ token reasoning steps after trivial ls/read results).

**Levers outside the agent gate (not implemented):** a global reasoning budget on every turn: cap 4096 removes 4.0% of reasoning (41 steps), 2048 removes 10.6% (197 steps = 2.4% of steps, 241k of 297k of it on tool-result turns), 1024 removes 24.5% (701 steps) — quality risk on hard turns, needs a live check. opencode's constant `variant: high` is cache-safe to change per session if medium is enough for a task class.

## Goal tracker ("jev make sure its done") — 2026-11-06

Off by default: `--jev-goal on`. Library `jev_goal.{hpp,cpp}`; replay
`gufo judge --goal-replay <dir>` (+ `--goal-verbose`); exporter
`tools/qwen-flash/export-opencode-sessions.py <opencode.db copy> <out dir>`.

Design
- Trigger table (constexpr): "jev make sure its done / it is done / its
  finished / it is finished", "jev finish it / this", "jev keep going until
  done / until its done"; case, punctuation and apostrophes are ignored.
- The phrase (and the separators after it) is stripped from EVERY user message
  at parse time, before the model or template sees it. Stripping is a pure
  function of each message, so a conversation renders identical prefix bytes on
  every request (test: turn N vs N+1 prefix equality). A message that is only
  the phrase becomes "Continue.". Mode is on if any user message carries it;
  serve is stateless, so it is re-derived per request.
- Goal = stripped text of the latest trigger message (previous non-empty user
  message if that one is only the phrase). Items by rule: change verb + file
  path -> needs an edit/write call (or shell write, or delegation) touching the
  file; change verb -> any write; commit/push/test/build asks -> a matching
  shell command with no failure text; question/investigate -> a non-trivial
  answer. Pasted logs (clauses > 240 chars) are ignored.
- On a turn that ends with finish=stop and no tool call: continue when items
  are open AND the model claims done (or the turn is empty), or when the last
  sentence announces a step ("let me now", "I'll next") that never ran. A turn
  that reports a blocker or hands a question back stays finished. Continue
  message: "Continue. Still open: ...". Guards: max 3 per request chain, never
  on finish=length, any error -> stock behavior.
- Reuses the --jev-finish continuation path (re-submit with appended
  assistant + user turn; streaming and non-streaming). Log:
  `[jev-goal] mode=on items=N open=M action=continue|done|nudge reason=.. us=..`.
- FINDING: `constrained_tools` is true for every request that carries tools
  (all agent clients). The existing --jev-finish guard
  `chat.constrained_tools -> return` therefore disables --jev-finish on all
  opencode traffic. The goal tracker bypasses that guard; --jev-finish is left
  unchanged (it still returns early with tools) — decide whether to relax it.

Offline replay (550 opencode sessions, 7,359 user episodes, 4,957 turn ends;
every user message treated as if it carried the trigger; truth = what the user
sent next: strong = "continue/go/you didn't finish/re-sent the same ask",
other = a new unrelated request, eof = session ended)
- Final rules: 79 continues (1.6% of turn ends): strong 39, loose 1, other 32,
  eof 7. precision(strong vs strong+loose+other) 0.54 against a 0.18 base rate
  of "next message was a continue"; recall 4.5%. Many of the 32 "other" cases
  are real stalls ("Writing the complete file now." then no file; "Let me
  start by understanding the project" then stop) the label cannot see.
- Per rule (fires: strong/other): intent 20: 9/8; open-change 28: 13/14;
  open-commit 10: 6/4; open-build 9: 4/5; open-answer 11: 7/0; open-file 1.
- Switched off (precision at or below base rate): last-failure (5 of 42 were
  followed by a continue), loop nudge (5 of 21), open-test (6 of 22),
  open-push. The toggles are `kRule*` constants in jev_goal.cpp.
- Speed: mean 56 us, p50 36 us, p99 264 us, max 1.1 ms per decision over 10,000
  evaluations (history up to thousands of messages; goal text capped to its two
  ends). Strip: 4 us per 1 KB message (~1 ms per 370 KB pasted message).
- Not verified live: no real opencode run with the flag on; Responses and
  Anthropic endpoints are not wired; recall is low by design (precision first).

### Goal tracker: bare phrase uses the earlier task — 2026-11-06 (evening)

Live finding (serve log 18:01:44): a bare "jev finish it" logged
`items=0 open=0 action=done` — the phrase matched but the trigger message had no
task text and the tracker did not look back.

Now:
- `DetectJevGoal` walks back over up to 6 user messages from the trigger
  message. It skips the bare phrase, `Continue.`, and ok/continue/go-on-style
  turns (<= 32 chars from a small list). The goal is the most recent message
  that yields >= 1 checklist item; its index is `spec.task_index` and evidence
  (edits, commands, results) is read from there on, not from the trigger
  message. No cross-message combining (precision first: stale items would nag).
- No items anywhere: `items == 0` fallback. Only existing precision rules apply
  (announced-but-never-run step) plus a truly empty turn; a plain "All done."
  is left alone. Log reason carries `no-items-fallback` (`reason=no-items-fallback`
  on done, `<rule>,no-items-fallback` on a continue).
- Parse-time log, once per request when the phrase is present:
  `[jev-goal] trigger=detected source=this-message|history items=N`.
- Strip behavior and prefix stability unchanged. Tests:
  `TestJevGoalTrackerBarePhraseUsesEarlierTask` (goal from history, evidence
  after the task counts, trivial turns skipped, no-task fallback for a done-claim
  and for an announced step). `openai_chat_test` passes.
- New replay mode `gufo judge --goal-replay DIR --goal-bare`: at every decision
  point the latest user message is replaced by a bare "jev finish it".

Replay (550 sessions, 4,957 turn ends):
- Baseline (trigger carries the task text): 85 fires, strict precision 0.482,
  loose 0.494, strict recall 0.048 (was 79 fires / 0.54 before the empty-turn
  fallback added 6 low-value fires).
- Bare phrase: goal from earlier turns at 3,343 points, no items at 1,614
  (fallback), 15 fallback fires. 77 fires, strict precision 0.338, recall 0.030.
  Per-rule: open-change 28 fires (13 strong), intent 20 (9), empty-turn 14 (2
  strong, 11 other), open-commit 10 (6), open-answer 11 (7), open-build 9 (4).
- Speed: normal path mean 59 us / p50 38 us / p99 276 us; bare path (detect +
  evaluate, evidence from an older task message so more history) mean 844 us,
  p50 143 us, p99 7.8 ms (long sessions with large tool results). Decision runs
  once per turn end, not per token, but p99 is above the 1 ms goal.
- Weak spot: the empty-turn fallback fires 14 times with 2 strong labels; kept
  because an empty assistant turn with no tool call is always a stall, but it is
  the first rule to drop if it nags.

## jev-guard (2026-11-07) — keeps gufo out of the way of heavy local tests

`tools/jev/guard.py`, started by `~/run-jev-guard.sh`. Public :8900 -> guard -> gufo on :8901.
`gufo judge --stdio` serves the NPU judge to the guard as a child process.

Behaviour: holds the final tool-call chunk of a response until gufo is stopped when the command
is heavy (`gufo bench`, `ctest`, `cmake --build`, `prof.py run`, ...); holds incoming requests
(SSE keepalives) while the test runs; restarts gufo about 8 s after the test process ends.
Memory watchdog (free RAM < 2.5 GB for 3 samples) stops gufo for unknown commands. SIGTERM only.

First live session (decode-2 work in opencode), measured:

- stop 1.0 s; heavy tool call held 1.09 s; gufo load 28 s; guard overhead per test about 36 s.
- unknown/light commands: hold 0.00-0.10 s (judge query 83-108 ms for unknown ones).
- 10 requests held 340-400 s during the 6-minute build; none failed.
- NPU judge vs patterns: judge p(Yes) sat in 0.37-0.47 for every command (heavy and harmless), so it
  does not separate them; the pattern list decides, the judge is a tiebreak only (cutoff 0.85).
- False stops found and fixed: `|` inside quotes split the command; read-only commands that mention
  `tools/prof/`; `which rocprofv3`; `python3 -c` with a path in its source; `prof.py show`.
- Process matcher fixed: the needle must be what /proc shows (no env prefix, no redirect, no quotes,
  no `~`/`$` tokens, no shell keywords like `do`). Before the fix, a run that started with
  `LD_LIBRARY_PATH=... ./gufo prompt` was never "seen" and gufo would have restarted at 30 s.
- Two requests were cut off by my own guard swaps (stale request-id match, bad timing). A swap must
  wait for `inflight == 0` on /jev-guard/status.

Open: the pattern list is the weak part (judge cannot back it up); `after_test` falls back to a 30 s
"never seen" restart; restart-on-start relies on run-gufo.sh refusing when RAM is short; opencode
retries a dropped request by itself but a guard cannot type "continue" into the TUI.

## Future: own NPU LLM engine (decided 2026-11-07: no closed-source runtime)

Goal: a small LLM that generates on our NPU engine and serves chat/light turns while the big gufo
is stopped for a test. Reference point (FastFlowLM page, closed source, not used): LFM3-1.7B Q4_0 on
the same chip decodes 41 tok/s at 1k and 24 tok/s at 32k context.

Measured here: NPU streams 30.3 GB/s of weights (97 us per 2.95 MB slice, pipelined depth 2+),
~170 us dispatch latency unpipelined (`tools/bench/npu_decode_dispatch_bench.cpp`).
Ceiling = bandwidth / bytes per token: MiniCPM5-1B Q6 block-float (~0.9 GB/token) about 30 tok/s,
realistic 15-25; 4-bit weights roughly double that.

Milestone 1 — MiniCPM5-1B generates on the NPU (current weights, current xclbin).
  - decode loop with a growing per-sequence KV cache (today's engine rewrites KV per call);
  - sampler (greedy + temperature/top-p), stop tokens, MiniCPM chat template;
  - `gufo judge --serve` OpenAI-style /v1/chat/completions with streaming;
  - measure tok/s at 1k/8k/32k context and correctness against the CPU reference.
  Done when: coherent chat, >= 15 tok/s at 1k, text matches the CPU reference on a fixed prompt set.

Milestone 2 — 4-bit weight streaming kernel.
  - new AIE kernel that dequantizes Q4 in-core; new xclbin + instruction files;
  - target about 2x the Milestone 1 speed (toward 40 tok/s).

Milestone 3 — LFM3 support.
  - GGUF arch + gated short-convolution layer (no such layer in the engine today);
  - compare accuracy against the PyTorch reference (perplexity, GSM8K subset).

Milestone 4 — guard integration.
  - guard routes requests to the NPU server while the big gufo is down;
  - limit it to light turns (status / "waiting for the test"); decide separately whether it may
    attempt full agent turns (1B-class models are weak at tool calls and code edits).

Milestone 5 — "continue" after a failure.
  - check whether opencode exposes an API the guard can use to send a prompt after a dropped request.

## Plugin-era milestones (added 2026-11-08, after the opencode plugin spike)

Context: jev's decisions are moving into the opencode plugin (see `docs/JEV-OPENCODE-PLUGIN.md`).
The plugin cannot run the NPU judge itself, so these milestones improve the one judge process the
daemon owns and how the plugin uses it. Everything here is automatic: nothing is exposed as a
user-facing API or endpoint, and nothing needs the user to start, call or configure it. Inference
stays on the NPU; only fine-tuning (Milestone 6) runs elsewhere, offline, with gufo stopped.

Milestone 6 — a trained decision model instead of a zero-shot one.
  - Evidence: off-the-shelf small models answer "Yes" to nearly everything (Qwen3-0.6B and
    Qwen3Guard-Gen-0.6B both failed the battery); MiniCPM5-1B-SFT is the best zero-shot judge
    (9/10 trivia, 8/12 code) but gave a false "heavy" (0.918) on a harmless memory CLI command.
    Small models trained specifically for decisions are reported to do far better.
  - Fine-tune MiniCPM5-1B (dense, loads in the existing NPU path) on logged decisions: guard rows
    and plugin rows with outcomes (`diverged`, `retry`, `rerun`, ...) from `jev.sqlite3`, plus the
    rule verdicts as weak labels. Hold out whole sessions for evaluation.
  - The training target is a typed decision: one of N options with a probability, not only
    Yes/No. Keep the Yes/No gap as the two-option special case so the current code keeps working.
  - Done when: on held-out sessions the tuned judge beats the zero-shot judge on false-heavy and
    false-light rates at the current 0.85 cutoff, the 22-case battery does not regress, and the
    exported GGUF loads in the NPU path unchanged.
  - LFM3-d1-2B (a published decision model with typed, calibrated outputs) is a candidate only
    after Milestone 3, because its hybrid convolution/attention layers do not load today.

Milestone 7 — one read of a turn, several questions.
  - The finish detector, the effort gate and the goal check each ask several questions about the
    same turn text. Today each question re-reads its own context (a batch of 4 shares nothing).
  - Read the turn once, keep its KV, and answer each question as a short suffix; an extension of a
    previously read text reuses the earlier work. Bounded cache (a few recent turns), results
    must equal the unshared runs.
  - Done when: answers are bit-identical to the unshared path on the battery and the replay sets,
    and the time for 4 questions about one 1,000-token turn drops by at least half.

Milestone 8 — automatic plugin-to-judge channel (private, not an API).
  - The plugin and the daemon talk over a private local channel (a Unix socket or the judge's
    stdio), started and supervised by the daemon with the rest of jev. No network port, no
    documentation for users, no keys. The contract is internal: a turn, a list of questions, and
    for each question the option probabilities.
  - Exactly one NPU judge process exists; the plugin never starts one. If the daemon or judge is
    down, the plugin falls back to rules only and keeps logging.
  - Done when: the plugin's finish, gate and goal checks can use the judge with no manual step, a
    killed judge is restarted automatically, and a dead channel never blocks or delays a tool call
    or a model request (fail open, 50 ms budget).

Milestone 9 — measure the judge's cost to decode.
  - JEV-REPORT lists "jev overhead on decode" as not measured. Measure gufo decode tok/s at 32k
    to 128k depth with the judge idle against the judge answering questions at the real rate
    (about 150 command checks and a few turn-level questions per hour).
  - The GPU and NPU share the memory bus and power budget, and GPU+NPU at the same time needs the
    fabric clock held at top speed (already a service on this box): confirm it is held, and
    record any hang or corrupted NPU result as a failure.
  - Done when: per-phase decode numbers (5% and 3 ms thresholds, per request, never averaged)
    show the penalty, or show none, with the clock held and not held.

Milestone 10 — observe-only streaming watcher.
  - The plugin can see streamed reasoning and text deltas (events such as
    `session.reasoning.delta`). Add a watcher that reads them and only logs what it would flag
    (reasoning that loops, the same question asked repeatedly, a reply that stops mid-sentence).
    It never alters the stream, the prompt or the history, which is why the removed inline
    answering (Part B) is not coming back. A small safety model such as Qwen3Guard-Stream-0.6B
    is the reference for this kind of watcher.
  - Done when: a week of logs shows the flags are rare and mostly correct. Any later action on a
    flag (for example an automatic continue) is a separate decision that needs its own evidence.
