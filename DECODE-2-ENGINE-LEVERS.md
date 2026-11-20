# Brief 2: engine-level decode levers (all models, no quality loss)

Repo `~/gufo`, new worktree `~/gufo-deceng` (branch `perf/decode-engine`). Same rules as `DECODE-1-UPSTREAM-PORT.md` (no competitor names, no push, no `pkill -f`, one GPU job at a time, NPU-ON build config, quality gate, depth-realistic timing). Read first: `.agents/skills/optimize-kernel/SKILL.md`, the port-model skill LEARNINGS.md, `~/gufo-private/decode-ideas.md` (never copy it into the repo), docs/models/qwen3.8-flash-next/EXPERIMENTS.md.

## Facts to build on (measured, do not re-derive)
- Flash-Next DFlash2 decode today: code ~102, json ~118, copy ~126, math ~84-95, prose ~45-52 tok/s. Prose is the weak category (code/json/copy at parity or ahead).
- Prose profile: GPU 97.3% busy (host overhead is NOT a lever). Round = ~44 ms, ~2.0 tok/round, per-slot acceptance 0.69/0.55/0.56/0.60 (reference server: 2.25 tok/round at 38.7 ms, acceptance 0.59).
- Verify base cost ~27.8 ms/round; gate/up rows-per-wave change gave ~0 end to end. Drafter round ~7.5 ms; trim ceiling <1%.
- Confidence-cut verify (draft 7 blocks, verify prefix with cumulative selector confidence >= 0.7, greedy only) took prose 47.9 -> 51.8, math 87 -> 95.
- A drafter fine-tune on session text LOWERED acceptance: labels must be the target's own generations.
- Smaller quants are NOT an option (quality); do not propose them.

## Tasks, in order
A. **Drafter trained on the target's own output (acceptance).** Generate target outputs (greedy and sampled at the production temperature) on diverse prose, explanation, agent-chat and code prompts (~use corpora from `tools/qwen-flash/extract-opencode-corpus.py`, `dflash-capture.sh`, `dflash-finetune.py`; vendor tooling already exists under `tools/qwen-flash/`). Fine-tune the DFlash2 drafter on (target features -> target next tokens). Measure per-slot acceptance and tok/s on prose/code/math/json/copy; accept only if prose improves and nothing regresses. Check the exact-output guarantee (lossless verification) still holds.
B. **Tree / multi-branch verification.** Prototype verifying the top-2 draft branches per slot in one wide pass (the verify pass is weight-streaming bound, so extra rows are cheap up to the GEMM geometry limit). Estimate expected tokens/round from the logged per-slot scores in `~/bench-compare/saved/rounds_prose.txt` BEFORE building anything; build only if the estimate >= +10% on prose.
C. **Verify cost per round, engine-wide.** Profile one verify round with `tools/prof/prof.py` (no rocprof on servers): per-kernel table for rows 2-8. Targets: attention at depth, HC/mixer fusion (see upstream #894), routed expert GEMV at small row counts, launch count. Only changes that are bit-identical or quality-neutral.
D. **Generalize to all models.** For every gain in A-C, check whether the same hook exists in the other Qwen-family engines (27B, 35B, Laguna, GLM) and apply it through shared code under `src/families/qwen` where possible. Per-model code stays with its model; shared helpers go in the family layer.
E. **jev side-effect measure.** With `--jev-pre` on, record the thinking-token reduction per request class next to the decode numbers (the user-visible win is time-to-answer, not tok/s). Uses `tests/functional/jev_gate_measure.sh` (stops the production server for hours: do not run without the user's OK).

## Deliverable
One report in `docs/models/qwen3.8-flash-next/EXPERIMENTS.md` style: lever | measured gain per category at depth | quality evidence | kept/reverted. Commit kept work locally, single-line Conventional Commits, no push.
