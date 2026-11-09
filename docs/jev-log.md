# JEV-LOG: decision log + feedback loop for the jev judge

**Status:** plan v2 (2026-11-08), code through T5 built and locally tested.
**Local-only** until the user says to merge.
**Origin idea:** [lpb-stack/lpb-memory](https://github.com/lpb-stack/lpb-memory)
(MIT) — persistent memory, failure/correction memory, background learning,
FTS5 session search, two-tier storage, auto-consolidation — applied to the
**jev NPU judge** (`src/cli/judge/judge.cpp`,
`src/models/qwen38_flash_next/jev_npu.hip.cpp`).

## Naming: decision log, not memory

lpb-memory's "memory" oversells what this is. It is a **decision log with a
feedback loop**: every judge decision is appended with enough context to join
it to what happened next, and whatever observes the consequence labels it.
`gufo judge`, no `--memory-*` flags — `--jev-log`, `--jev-log-search`,
`--jev-log-correct`, `--jev-log-outcome`.

## The idea in one paragraph

Today every jev decision — effort pre-gate or mid-think yes/no splice — is
ephemeral: the 61-77 ms NPU call answers and the outcome vanishes. Calibration
is the one-time 22-case trivia/code battery, and each hard-won lesson (Qwen3
"all-Yes" saturation, per-model framing differences) was discovered manually
and lives only in a markdown status table. JEV-LOG appends each decision
off-hot-path with `request_id`/`session_id`, so ground truth (guard
watchdogs, goal tracker, the user) can attach `outcome`/`correction` rows, and
the log can be searched ("did jev say yes about X while framing v1 was live?")

This directly serves jev-gate-plan.md punch list item 3: "calibrate on real
coding-agent traffic" — this log *is* that capture, replayable regression
evidence included.

## Ground truth first (review gap 1)

Review findings: the previous design had no producer of labels; `yes_rate` in
`patterns` would just record the judge's own answers back to itself, and T5-T7
would have had nothing to learn from. Fixed by schema v2:

- `decisions.request_id`, `decisions.session_id` — join keys, indexed.
- `decisions.outcome` CHECK'd to `confirmed|diverged|retry|rerun|harmless`.
- CLI `--jev-log-outcome "ID OUTCOME [NOTE]"` — SetOutcome.
- The battery auto-labels its own decisions (expected answer = ground truth).

Who produces outcomes per trigger:

| Trigger | Ground truth | Who calls SetOutcome |
| --- | --- | --- |
| `effort-pregate` | did a capped turn cause a retry / tool failure / rerun? | serve (Part A/B epoch) — after clearance |
| `mid-think` | did the model's later thinking contradict the splice? (guard "guilty-check" / silence guard) | serve + jev-guard watchdogs |
| `guard` | **first customer: jev-guard.** The kfd and memory watchdogs know when a "light" classification wrongly opened the GPU. Logging each of those as `outcome=diverged` on the guard decision produces a self-built heavy-pattern list — exactly the labeled data that has been missing and that forces repeat guard patching today | jev-guard (not in this checkout; contract below) |
| `goal` | re-prompt after "done" detected | goal tracker replay |
| `battery` | expected answer | automatic |

### jev-guard corrections contract (what that codebase calls)

jev-guard does not live in this checkout, so the hookup is a contract:

```
Call one of, per watchdog catch (kfd open under light classification,
memory watchdog event):
  gufo judge --jev-log DB --jev-log-outcome "ID diverged NOTE"
Programmatic equivalent (needs link in jev-guard's deploy unit):
  gufo::cli::jev_log::JevLog::Open(db, &log, &err);
  log.SetOutcome(decision_id, "diverged", note, &err);
It can also spawn a subprocess per catch — serialized, no in-process lib
required. The guard records its classification decision first:
  trigger="guard", request_id=<its own request id>, action="opened-gpu".
```

Expectation: each watchdog catch labels a past "light" decision wrong;
corrections accumulate the heavy-pattern list automatically instead of
patches. Until the linkage exists, guard outcomes can be attached by id from
operator runs (`--jev-log-outcome`) — no code change in that repo needed for
the first labeled batch.

## Missing columns fixed (review gap 4)

Schema v2 `decisions` now carries: `threshold` (cutoff in effect, e.g. "0.5",
"0.6"), `framing_ver` (prompt framing/template version — boundary replay and
recalibration need it), `action` (what the decision changed: "spliced",
"low-effort", "opened-gpu", "battery-seed"). Without these a decision is
unreplayable; with them each row can be re-judged under a new threshold.

## Privacy and retention (review gap 3)

- `jev_log::ContainsSecret` — token-shaped secret filter (sk-/ghp_/AKIA/xox
  /JWT/AIza/private-key blocks). Callers redact before persisting; battery
  questions are fixed strings, so recorded battery text is safe by
  construction. **Live traffic (T8) stays disabled until a real policy
  review passes**; the filter is a first line, not a guarantee.
- Retention: unanswered decisions are informational, but unbounded growth in
  `~/.gufo/jev-log/` is refused by design here since the CLI records only in
  explicit `--jev-log` mode. T8 (live traffic, if ever cleared) MUST ship
  with: max rows cap (default 50k), prune-on-open of decisions older than
  30 days without an outcome, and a `--jev-log-strip-shared` mode that
  stores only `prompt_hash` + size bucket, never plaintext question text,
  for server-shared deployments.

## Architecture

### Layers

| Layer | Content |
| --- | --- |
| Decisions | append-only `decisions` table (SQLite, WAL) |
| Outcomes | ground truth column + SetOutcome API |
| Corrections | `corrections` table — failure/correction/insight |
| Patterns | `patterns` table — per-model pattern validity, seed from battery only |
| Search | FTS5 index over the recorded question text |
| ~~RAM cache~~ | **dropped (T7 removed)** |

### Storage

- Database file, user-selectable with `--jev-log PATH`; default
  `~/.gufo/jev-log/jev.sqlite3` for battery only.
- Two tiers: a per-model tier key (`model` column: judge model id + quant,
  e.g. `qwen3-0.6b-q6_k`) — a pattern valid on one quant is not served for
  another.

### Off-hot-path rule (hard constraint)

The 61-77 ms decision budget stays untouched. Writes are queued in a
lock-protected ring buffer and flushed by a drain thread (or at run end in
CLI mode); no SQLite ever blocks the NPU call. CLI mode flushes on exit.

## Pass/fail bar before anything learns from the log

The report put jev's whole-system gain under 5%. So the bar is now defined
before T5+ instead of after:

- **Bar:** on ≥ 200 labeled decisions per trigger family, labeled agreement
  (judge answer vs. outcome) ≥ 0.85 on battery-family triggers AND ≥ 0.70 on
  guard-family triggers, with logit_gap drawn from a calibration fit that
  beats the fixed cutoff on holdout decisions by ≥ 2/10 battery score.
- T5 (review subprocess) may *report* these numbers. It may not change
  thresholds until the bar is met.
- If the bar is not reached after the guard's labeled batch exists, the work
  stops at T4 permanently: log stays useful as searchable history and
  regression evidence, nothing more.
- **T7 (RAM pattern cache) is dropped entirely** — the judge's own weak
  agreement (p 0.37-0.5 on commands) means a cache of its answers is a cache
  of confident mistakes. If the bar is ever met, the *fit* over logit_gap is
  the only suggested consumer.

## Subcommands / flags

| Interface | Behavior |
| --- | --- |
| `gufo judge --jev-log PATH` (with `--battery`/`--code`/`--question`) | every decision recorded; battery verdicts auto-labeled + pattern-seeded |
| `gufo judge --jev-log PATH --jev-log-search "query"` | FTS5 query over text, prints rows; exit |
| `gufo judge --jev-log PATH --jev-log-correct ID kind [NOTE]` | attach a correction |
| `gufo judge --jev-log PATH --jev-log-outcome ID outcome [NOTE]` | set ground truth |
| `gufo judge --jev-log PATH --jev-log-review` (T5) | background review subprocess: serialized, computes per-model labeled agreement — does not change thresholds |

The coding-agent integration (recording live `--jev-pre` traffic) touches
`inference_backend.cpp`/`openai_chat.cpp`, which the other agent owns — that
integration is explicitly deferred until the user says Part A/B files are safe
to touch. T8 requires the privacy setup above first.

## Tasks

Local-only, done in this worktree on top of `feat/jevgate`; files stay
untracked/uncommitted until the user says to merge.

- [x] T0 — write this plan doc (`docs/jev-log.md`; local-only)
- [x] T1 — `src/cli/judge/jev_log.{hpp,cpp}`: SQLite store (WAL), schema v2
      with outcome/request_id/session_id/threshold/framing_ver/action,
      append-only record API, FTS5 search, correction + outcome writers,
      `ContainsSecret` filter. Link `sqlite3`
- [x] T2 — wire recording into `judge.cpp` (`--jev-log`, prompt hashing,
      battery auto-label + per-model pattern seed)
- [x] T3 — `--jev-log-search`, `--jev-log-correct` and `--jev-log-outcome`
- [x] T4 — hosted functional test `tests/functional/jev_log.py` (CPU-only:
      schema, FTS5 round trip, correction FK/kind validation, outcome set)
      — PASS on this build; clang-format clean; `gufo` links with `sqlite3`
- [x] T5 — review subprocess (`--jev-log-review`): serialized, reports
      labeled agreement per trigger family; refuses to change thresholds
      until the bar above is met
- [x] T Guard — guard producer (`~/gufo/tools/jev/guard.py`, separate repo):
      `GuardLog` writes one `trigger=guard` row per classification in
      `decide()` (binary `--jev-log <db>`, default
      `~/.gufo/jev-log/jev.sqlite3`), queues `action=none` lights, and
      kfd/memory watchdog catches mark the most recent pending light
      `outcome=diverged` + `corrections(failure, note)`. Pure-Python
      sqlite3, same schema — C++ `--jev-log-review` reads its rows
      (verified cross-compat). Secrets redacted via `contains_secret`.
- [ ] T6 — **parked per recommendation** (consolidation: merge
      near-duplicate patterns when `patterns` exceeds a cap; stale entries
      age out by `last_ts`). Do not start.
- [ ] T7 — **dropped** (RAM pattern cache: the judge's weak agreement
      makes it a cache of confident mistakes)
- [ ] T8 — deferred: live-traffic hook in serve (Part A/B files — needs
      user clearance AND privacy: secret policy review, row cap, prune, and
      hash-only recording for shared deployments)

## Week 1 operation (collect, then review)

Guard is now the producer. Restart it so traffic accumulates:

```sh
# default db ~/.gufo/jev-log/jev.sqlite3, via --jev-log override as needed
bash ~/run-jev-guard.sh            # or the normal guard start path

# During the week: light decisions are written (outcome NULL), watchdog
# catches flip the blamed light to diverged. No operator labeling needed.
# Heavy decisions (action=stop) are also written but stay unlabeled.

# After ~7 days:
gufo judge --jev-log ~/.gufo/jev-log/jev.sqlite3 --jev-log-review
# apply the pass/fail bar above. >=200 per family? battery>=0.85,
# guard>=0.70? If not, stop at T4/T5: keep the log, do not fit anything.
```

`<db>` must be the same file for the battery seeder, the guard, and the
reviewer — one shared file, tiers split by `model`/`trigger` columns.

## Next move (enhancement order, best first)

1. **Guard is now landed as the first customer** (T Guard).
   Restart the guard so rows accumulate; watchdog catches already label
   the blamed light `diverged`. Collect a week, then `--jev-log-review`.
2. With labels, fit a small calibration over `logit_gap` and the cutoff —
   cheaper and safer than any cache. Only if the bar passes.
3. Hold T5+ against the pass/fail bar; stop at T4 if it is not reached.
   T6 stays parked until the bar says consolidation has something real
   to merge.

## What we deliberately did NOT take from lpb-memory

- Secret scanning as an afterthought — filter is built (ContainsSecret) and
  the stronger T8 privacy setup is specified as a prerequisite.
- Markdown two-file storage — SQLite alone; gufo is C++, no Markdown index
  layer needed.
- In-band footer status — gufo is a headless inference engine.
- Every-N-turns in-process review — CLI runs finish; review is an explicit
  subprocess (T5), which is lpb-memory's own safer transport anyway.
- Auto-serving cached answers — dropped (T7) for the reasons above.

## Test protocol

- Hosted checks only (SQLite + CLI logic need no GPU): build `gufo` and run
  the functional round-trip test; no NPU/GPU needed for T1-T4.
- GPU-touching evidence (battery recording from a real judge run) waits for
  a user-approved window.
- Real calibration comparisons reuse `tests/functional/jev_gate_measure.sh`
  per its 3-arm protocol once labels and the fit exist.
