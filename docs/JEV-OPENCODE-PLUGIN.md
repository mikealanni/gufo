# JEV as an opencode plugin: port plan (Phase 0 = log only)

Status: plan, written 2026-11-08. Nothing here is built yet. Phase 0 is the only
phase to build now; later phases are outlined so the design does not paint
itself into a corner.

Audience: the local LLM (gufo, through opencode) that will build Phase 0, and the
human who reviews it. Read the whole file first. Follow "Rules" and "Do not"
literally.

## Why

jev today is split between a Python proxy (`tools/jev/guard.py`, sits in the
token stream on :8900) and C++ features inside the engine (`--jev-gate agent`,
`--jev-pre`, `--jev-finish`, `--jev-goal`). The proxy only learns about a shell
command after the model has streamed it, and it has to hold the response while
it decides. opencode plugin hooks run before the tool executes, see the exact
command, working directory and session, and are async, so they can wait for the
GPU to be free before the command starts. They also see the whole message
history, which is what the goal tracker, finish detector and effort gate need.

Decision (user, 2026-11-08): move jev into an opencode plugin step by step,
starting with logging only, then enhance. Part B (inline self-question
answering inside `<think>`) is removed from the engine and is not part of this
port.

## Rules (non-negotiable)

1. **No GPU use, ever.** The GPU belongs to the main LLM. The plugin is plain
   TypeScript running in the opencode process.
2. **No NPU use in Phase 0.** There is exactly one NPU judge in the system and
   the guard daemon owns it (`gufo judge --stdio`). The plugin must never start
   a second NPU process. Phase 0 is rules only. Any later judge call goes to the
   daemon over HTTP.
3. **Log only.** In Phase 0 no hook may change its `output` argument, throw,
   block, stop or start gufo, send a prompt, or write to the session. A hook
   that is disabled, broken or slow must leave opencode behaving exactly as if
   the plugin were not installed.
4. **Fail open.** Wrap every hook body in try/catch; on any error log one line
   and return. No hook may take longer than 50 ms of synchronous work; file and
   database writes go through an in-memory queue drained asynchronously.
5. **Only for gufo.** Act only when the call is to the gufo provider (see
   "Detecting gufo"). Calls to every other provider are ignored entirely: no
   logging of their content.
6. **Never edit earlier messages.** Editing history breaks gufo's prefix cache.
   (Phase 0 edits nothing. This rule matters for later phases.)
7. **Secrets.** Never write raw command output, file contents or tool results to
   the log. Commands are logged truncated to 500 characters and replaced with
   `[redacted-secret]` when they contain a secret marker (same list as
   `SECRET_KEYS` in `tools/jev/guard.py`).
8. Global kill switch: environment variable `JEV_PLUGIN=off` makes every hook an
   immediate no-op.

## What is verified about the opencode plugin API

Verified locally (primary source): installed opencode v2.2.6; package
`@opencode-ai/plugin` 1.14.28 under `~/.opencode/node_modules/@opencode-ai/plugin/dist/index.d.ts`
(also `.../dist/v2/`). Read that file first, it is the ground truth for the
installed version. The V1 `Hooks` interface contains (all async,
`Promise<void>`):

| Hook | Input (summary) | Output you may mutate |
|---|---|---|
| `event` | `{ event }` | none |
| `chat.message` | `sessionID, agent, model{providerID,modelID}, messageID` | `message`, `parts` (user message) |
| `chat.params` | `sessionID, agent, model, provider, message` | `temperature, topP, topK, maxOutputTokens, options` |
| `chat.headers` | same | `headers` |
| `tool.execute.before` | `tool, sessionID, callID` | `args` (for bash: `args.command`) |
| `tool.execute.after` | `tool, sessionID, callID, args` | `title, output, metadata` |
| `shell.env` | `cwd, sessionID?, callID?` | `env` |
| `permission.ask` | permission | `status` |
| `experimental.chat.messages.transform` | `{}` | `messages[]` (full history) |
| `experimental.chat.system.transform` | `sessionID?, model` | `system[]` |
| `experimental.session.compacting`, `experimental.compaction.autocontinue`, `experimental.text.complete`, `tool.definition` | see d.ts | see d.ts |

Documentation (https://opencode.ai/docs/build/plugins/server/) also describes a
newer V2 style (`ctx.session.hook("prompt" | "context" | "model.request" ...)`,
`ctx.tool.hook("execute.before" | "execute.after")`, `ctx.shell.hook("create.before")`,
`ctx.event.subscribe(...)`, `ctx.session.prompt/context/interrupt/...`,
`ctx.storage`). The installed package still ships the V1 hook object. **Task 0
(below) decides which style to use.** Do not guess.

Plugin loading (from https://opencode.ai/docs/plugins/): `.ts`/`.js` files under
`.opencode/plugins/` or the global `~/.config/opencode/plugins/` are loaded
automatically, and entries can also be listed in the config `plugins` array
(npm names, local directories, `file://` URLs). `~/.config/opencode/opencode.json`
currently has no plugin entries and `~/.config/opencode/plugins/` does not exist.

Not verified (must be checked in Task 0, do not assume):
- Which style (V1 object or V2 `Plugin.define`) opencode v2.2.6 actually loads.
- The runtime (opencode ships under `~/.opencode/bin/opencode`; Bun is likely).
  If Bun, `bun:sqlite` and `fetch` are available; confirm in a plugin.
- The exact event names (`session.idle` and friends) and the event payload shape.
- Whether `chat.params` `output.options` reach the HTTP body for the gufo
  provider (needed later for the effort gate).
- How to read a session's messages from inside a hook (SDK client method names).

## Detecting gufo

The gufo providers in `~/.config/opencode/opencode.json` are `local-gufo` (base URL
`http://127.0.0.1:8900/v1`, renamed from `kairic-direct` on 2026-11-08) and `kairic-remote`
(`http://192.168.6.138:8900/v1`), model `Qwen3.8 Flash Next`. Sessions created before the
rename carry the old provider id in opencode's database and are not logged until they are
started again with `local-gufo`.
Make the provider ids and the model id configurable
(`JEV_GUFO_PROVIDERS`, default `local-gufo,kairic-remote`) and match
`model.providerID` from the hook input. Tool hooks (`tool.execute.*`) have no
model in their input: look up the session's active model from the most recent
`chat.message`/`chat.params` for that `sessionID` and cache it in memory.

## Architecture

```
opencode  --(hooks)-->  jev plugin (TS, in process)   Phase 0: log only
                              |  (later: HTTP, localhost)
                              v
                     jev daemon = today's tools/jev/guard.py, slimmed down
                     (owns: gufo start/stop/restart, NPU judge, memory and
                      GPU watchdogs, jev-log sqlite)
                              |
                              v
                       gufo serve (engine; keeps --jev-gate/--jev-pre/
                       --jev-finish/--jev-goal until their plugin versions
                       are proven)
```

Phase 0 touches only the plugin. The proxy stays in the token path and keeps
doing its job unchanged; the plugin runs beside it and writes what it *would*
have decided so the two can be compared.

## Phase 0 scope: what to build

Location in the repo: `tools/jev/opencode-plugin/` (TypeScript, strict mode, no
runtime dependencies unless Task 0 shows one is required). Install by
symlinking the entry file into `~/.config/opencode/plugins/jev.ts`. Do not edit
`opencode.json`.

### 0. Task 0: spike (do this first, report before building the rest)

Write a 20-line plugin that logs `hook name + keys of input/output` for
`chat.message`, `chat.params`, `tool.execute.before`, `tool.execute.after` and
`event` to `~/.gufo/jev-log/plugin-spike.jsonl`, run one short opencode session
against gufo (a prompt that makes the model run `ls /tmp` through the bash
tool), and record: loader style that worked, runtime (`process.versions`),
event names seen (print a list of distinct `event.type`), the exact shape of
`args` for bash, whether `chat.params.output.options` can be set, and how a hook
can read session messages. Write the findings into a section "Spike results" at
the end of this file. Everything below is conditional on those results.

### 1. Command gate (log only)

Hook: `tool.execute.before` (V2: `ctx.tool.hook("execute.before")`) when the tool is `shell` (the spike showed opencode v2.2.6 names it `shell`; accept `bash` too) and the session is on gufo.

- Port `classify(command)` from `tools/jev/guard.py` to TypeScript:
  `HEAVY` patterns, `LIGHT_WORDS`, `segments`, `strip_launchers`/`LAUNCHER`,
  `expand`/`ASSIGN`/`SUBST`, `needle_for`, `executed_scripts`,
  `inspect_scripts` (reads the script file, depth 2, 64 KiB cap,
  `HEAVY_CONTENT` regex) and the learned-heavy list from
  `~/.gufo/jev-log/learned-heavy.json` (substring match on the command; reload
  the file at most every 10 s). Keep names and constants identical so the two
  implementations can be diffed. Do NOT call the NPU judge: a command the rules
  cannot decide is logged with verdict `unknown`.
- Record one row per call (see schema): `verdict`, `would_stop` (heavy), the
  matched segment, and `callID`.
- In `tool.execute.after`, record `duration_ms` (since `before`), and a coarse
  `outcome` (`completed`/`error`) when available.

### 2. Goal tracker (log only)

Hook: V2 `ctx.session.hook("prompt")` buffers the user text, and `ctx.session.hook("context")` (which knows the provider) evaluates it once per user message; `chat.message` is the V1 name for the same step. Detect the same phrases as
`src/models/qwen38_flash_next/jev_goal.cpp` ("jev make sure its done", "jev
finish it", the variants listed there). Log `matched`, which phrase, and whether
the message was a bare phrase (then the task would come from earlier user
turns). **Do not modify the message.** (Later phases strip the phrase here.)

### 3. Finish detector (log only)

On the session-idle event (name from Task 0), read the last assistant message
and apply the rule-based detector ported from
`src/models/qwen38_flash_next/jev_finish.cpp` (empty answer, announced next step
with no tool call, etc.). Log `would_continue` and the rule that fired. Never
send a prompt.

### 4. Effort gate (log only)

Hook: `experimental.chat.messages.transform` (read the history, do not mutate it)
plus `chat.params` for the model/session. When the trailing messages are tool
results, port `ClassifyJevAgentTurn` from `src/models/qwen38_flash_next/jev_gate.cpp`
(and the turn-building loop in `ApplyJevAgentGate`,
`src/cli/serve/openai_chat.cpp` around line 875) and log `would_lower` with
`reason` and `blocker`. The engine's own cap is 512 reasoning tokens
(`kJevAgentRoutineBudgetTokens`) applied while decoding; it does not touch the
prompt, which is why it keeps the prefix cache. The engine already accepts a
request field `reasoning_budget_tokens` or `thinking_budget_tokens`
(`openai_chat.cpp` ~line 754). Whether a plugin can send it is open question
Q3.

### 5. Activity tracking

Record request start/end per session for gufo calls (from `chat.params` and
`event` hooks, whatever Task 0 shows). The daemon's rule "stop for foreign GPU
processes only while opencode is using gufo" will eventually read this.

## Log format

Append JSON lines to `~/.gufo/jev-log/plugin-YYYYMMDD.jsonl` (local date, one
file per day, create the directory if missing). Required fields on every record:

```json
{"ts": 1794200000.123, "plugin_version": "0.1.0", "feature": "command_gate",
 "session": "ses_...", "call": "call_...", "provider": "local-gufo",
 "model": "Qwen3.8 Flash Next", "decision": {"verdict": "heavy", "would_stop": true},
 "cmd": "<first 500 chars, secrets redacted>", "dur_ms": null, "note": ""}
```

`feature` is one of `command_gate`, `goal`, `finish`, `effort_gate`, `activity`,
`error`. Writes are buffered (flush every 250 ms or 50 records) and failures are
dropped silently after one `error` record. If `bun:sqlite` is confirmed in the
spike, also insert command-gate rows into `~/.gufo/jev-log/jev.sqlite3` table
`decisions` with `trigger = 'plugin-command'` and `framing_ver = 'plugin-v0'`
(schema in `tools/jev/guard.py`, `JEV_LOG_SCHEMA`); use `INSERT` only, WAL mode,
and `busy_timeout` 2000. Never create or alter tables.

## Tests and acceptance criteria

All must pass before Phase 0 is considered done.

1. **Classification parity.** A script `tools/jev/opencode-plugin/test/parity.ts`
   loads a JSON list of `{cmd, verdict}` (generated from the opencode database
   with `tools/jev/guard.py`'s `classify`; the repo does not contain the data, a
   local copy exists at `~/dflash-train/guard_cmds.json`, 42,570 commands) and
   compares the TypeScript `classify` result. Target: at least 99.5% identical
   verdicts, and zero cases where Python says `heavy` and TypeScript says
   `light`. List every disagreement in the output.
2. **No mutation.** A test calls each hook with a deep-frozen `output` object;
   the hook must not throw and must not change it.
3. **Fail open.** Inject an exception inside each hook body (test flag); opencode
   must behave exactly as without the plugin and one `error` record must be
   written.
4. **Overhead.** Median hook time under 1 ms and p99 under 10 ms on the 42,570
   commands (synchronous part only).
5. **Gufo only.** Requests to a non-gufo provider produce no records.
6. **Live check.** Run one real opencode session on gufo that includes a light
   command, a heavy command (for example `python3 -c "import time; time.sleep(1)"`
   is light; a `cmake --build` is heavy) and a goal phrase; show the resulting
   JSONL lines and confirm that the jev-guard daemon log shows the same verdicts
   for the same commands.
7. **Comparison report.** `tools/jev/opencode-plugin/compare.py` joins the plugin
   log to `/tmp`-independent guard decisions in `~/.gufo/jev-log/jev.sqlite3`
   (match on command text and a 30 s window) and prints agree / disagree
   counts. After a week of use this is the evidence for Phase 1.

## Do not

- Do not start any process other than the plugin's own code; never call
  `gufo`, `jev-stop.sh`, `run-gufo.sh`, the judge, or anything on the GPU/NPU.
- Do not modify `tools/jev/guard.py`, `~/run-gufo.sh` or the gufo engine.
- Do not change opencode's config files; use the symlink install.
- Do not persist tool outputs or file contents.
- Do not commit generated logs or the parity data file.
- Do not push. Commit locally with a single-line Conventional Commit.

## Later phases (outline only, not for now)

Each phase starts only after the previous one has run for at least a week with
the plugin log agreeing with the daemon log.

- **Phase 1: command gate active.** The daemon grows localhost endpoints
  (`/jev/precheck`, `/jev/done`); the plugin awaits the daemon in
  `tool.execute.before` so gufo is stopped before a heavy command starts and is
  restarted after `tool.execute.after`. The proxy stops holding tool-call events.
- **Phase 2: goal tracker.** Strip the phrase in `chat.message` (current message
  only, never earlier ones) and continue on session idle.
- **Phase 3: finish detector.** Continue unfinished turns on session idle, with a
  per-session limit.
- **Phase 4: effort gate.** Send the decode-time budget via request options
  (needs Q3). Never rewrite the prompt or the history.
- **Phase 5: retire the proxy.** The daemon stays (gufo lifecycle, memory and
  GPU watchdogs, the single NPU judge); only the proxying goes away.
- **Judge calls.** Added through the daemon only, so there is one NPU judge.

## Open questions (answer in the spike)

- Q1. V1 hook object or V2 `Plugin.define` for opencode v2.2.6?
- Q2. Runtime, and is `bun:sqlite` available inside the plugin?
- Q3. Do values in `chat.params` `output.options` reach the request body for the
  gufo (OpenAI-compatible) provider? If not, is there another route
  (`chat.headers`, provider options in `opencode.json`)?
- Q4. Exact event names for session idle / message completion and how to read a
  session's last messages.
- Q5. Does `tool.execute.before` run for every bash call, including calls made
  by subagents and background commands?

## Spike results

Done (2026-11-08, three `--standalone` runs on gufo; shared-service old code
is superseded — use `--standalone` or restart the service after edits,
because the file-plugin revision is keyed by target string, not content hash).

- **Q1 loader — V2 wins (corrects the static guess above).** The installed
  `@opencode-ai/plugin` 1.14.x d.ts describes V1 Hooks, but the v2.2.6
  loader for `~/.config/opencode/plugins/jev.ts` rejects a bare function
  (`Plugin must export a default definition with an id and an effect or
  setup function`) and *ignores* a `setup()` that returns a hooks object
  (loads active, hooks never fire). What works is `export default {
  id: "jev-spike", async setup(ctx) { await ctx.tool.hook(...);
  await ctx.session.hook(...); return () => {} } }`. `ctx` keys on this
  build: `app, location, options, agent, aisdk, command, credential, event,
  experimental, generate, model, provider, integration, mcp, permission,
  plugin, reference, rpc, skill, storage, tool, vcs, websearch, worktree,
  session, shell`. Spike: `tools/jev/opencode-plugin/spike.ts` (now V2),
  symlinked to `~/.config/opencode/plugins/jev.ts`, `JEV_PLUGIN=off` kill
  switch, JSONL to `~/.gufo/jev-log/plugin-spike.jsonl`.
- **Runtime (from the live `load` line, `process.versions`):** node 26.3.0,
  bun 1.4.4, sqlite 3.53.4, typescript 7.0.2 (host `bun` 1.3.13 / node
  22.22.2 are *not* the runtime — do not use host numbers for the plugin).
  `await import("bun:sqlite")` inside `setup` → `hasDatabase: true`
  (host `bun -e` also returns `function`). **`bun:sqlite` is available**
  in-plugin; the log-format note about INSERT-only into `jev.sqlite3`
  stands (see Log format).
- **Providers:** `~/.config/opencode/opencode.json` currently has
  `local-gufo` (`Qwen3.8 Flash Next` @ `http://127.0.0.1:8900/v1`) and
  `kairic-remote` (provider ids as of the spike; `kairic-direct` was renamed `local-gufo` afterwards).
  Default `JEV_GUFO_PROVIDERS` should be `local-gufo,kairic-remote` until
  the config says otherwise.
- **Parity data:** `~/dflash-train/guard_cmds.json` exists, 42,570 rows,
  keys `{cmd, dur, sid, script_heavy, verdict}` — ready for the parity test.
- **Bash tool args (live):** `tool.execute.before` fires with `tool:
  "shell"` — **not `"bash"`** (fix Section 1's `tool === "bash"` check to
  accept `shell`) and `input` keys `["command"]` with a string command
  (`"ls /tmp"`). `tool.execute.after` adds `status: "completed"`.
  Hook event keys: `tool.before: tool, sessionID, agent, messageID, id,
  input`; `tool.after` adds `status, result`. `sessionID` **is** present on
  tool hooks (plan's lookup-if-missing fallback not needed, but keep it).
- **Session hooks (live):** `session.prompt` keys `sessionID, messageID,
  prompt, metadata, delivery`; `session.context` keys `sessionID, model,
  system, messages, options, agent, tools` with `model: {providerID:
  "local-gufo", id: "Qwen3.8 Flash Next"}`, `options` keys `["maxTokens"]`,
  `messages` count 1 → 3 across the run, `tools` as `object` (not an array).
  So a hook reads history via `event.messages` (and `ctx.session.context`
  per the API docs) — no SDK-client guess needed.
- **Q3 (effort-gate route) — partially answered, not proven.** `options` is
  a mutable object (`maxTokens` present); per the server docs, request
  overrides merge into the outgoing call and non-typed keys pass as provider
  options. Whether a plugin-set `reasoning_budget_tokens`/`thinking_budget_tokens`
  actually reaches the gufo OpenAI-compatible HTTP body is **still unverified**
  — log-only forbids the mutation test. Needs a Phase-1 mutation test
  against `openai_chat.cpp` ~line 754 before Phase 4 design.
- **Event names (live, 63 events, 35 distinct `event.type`):** `agent.updated,
  command.updated, integration.updated, location.shutdown,
  mcp.resources.changed, mcp.status.changed, model.updated, plugin.updated,
  provider.updated, reference.updated, session.execution.started,
  session.execution.succeeded, session.inbox.delivered, session.inbox.enqueued,
  session.instructions.updated, session.reasoning.delta,
  session.reasoning.ended, session.reasoning.started, session.renamed,
  session.step.ended, session.step.started, session.step.streamed,
  session.text.delta, session.text.ended, session.text.started,
  session.tool.called, session.tool.input.ended, session.tool.input.started,
  session.tool.progress, session.tool.success, session.usage.updated,
  shell.created, shell.exited, skill.updated, vcs.branch.updated,
  websearch.updated`. **No `session.idle`** in short runs — the finish/goal
  idle hook still needs a longer-idle run to name the exact event (Q4/Q5
  remain open on idle-naming and subagent coverage; both are now cheap:
  extend the `ctx.event.subscribe` loop and re-run).
- **Q5 (subagents/background):** `tool.before` fired for the regular bash run;
  subagent/background coverage needs a run that actually spawns them.

Task 0 is complete pending the idle/subagent naming run. Sections 1–5 may
now be built against these facts — with the `shell` tool name and the
`local-gufo,kairic-remote` default above.


## Implementation status (2026-11-08, reviewed)

Code: `tools/jev/opencode-plugin/` (`jev.ts` entry, `classify.ts`, `goal.ts`,
`finish.ts`, `gate.ts`, `test/`, `compare.py`). Installed by symlink as
`~/.config/opencode/plugins/jev.ts`. Uncommitted in the jevgate worktree.

| Section | State |
|---|---|
| Spike (Task 0) | Done; results above. Q1, Q2 closed; Q3 (budget route), Q4 (idle name), Q5 (subagents) open. |
| 1 command gate | Done: `classify.ts` port, JSONL plus `decisions` rows (`plugin-command`). Accepts `shell` and `bash`. |
| 2 goal tracker | Done: 8 phrases, bare detection. Evaluated once per user message. Prompt text is logged only when a phrase matched. |
| 3 finish detector | Done: evaluated once per turn end (events inside 1.5 s collapse). `session.idle` was not seen in short runs; the subscriber also uses `session.execution.succeeded` and `session.step.ended`. |
| 4 effort gate | Port done (`gate.ts`); the turn builder is approximate (live blocker was `tool=tool`, not the shell command). Q3 unproven. |
| 5 activity | Start record per model call; end record per turn end (`phase: "end"`). |
| Retention | Daily JSONL files older than 14 days are deleted at startup; a day file stops growing at 64 MiB (one `error` record, then drops). `JEV_LOG_DIR` overrides the directory (tests use a temp dir). |

Acceptance (section "Tests and acceptance criteria"):

1. Parity: pass. 42,570 commands, fresh Python agreement 100.00%, zero heavy-to-light; median 0.014 ms, p99 0.117 ms.
2. No mutation: pass (`test/safety.ts`).
3. Fail-open: pass. `JEV_TEST_THROW=<hook>` injects an exception in `prompt`, `context`, `execute.before`, `execute.after`; each leaves one `error` record and does not throw.
4. Overhead: pass (same numbers as 1).
5. Gufo only: pass.
6. Live: light, heavy and goal phrase seen in one session. First live comparison with the guard (`compare.py`, `~` normalised): 40 matched command pairs, 39 agree (2 heavy, 37 light), 1 disagreement, 15 plugin rows and 14 guard rows unmatched. The one disagreement is a judge false positive, not a plugin bug: the rules say `unknown` for `python3 ~/.shokunin/scripts/chroma-helper.py session continue ...`, the guard's NPU judge scored it 0.918 (cutoff 0.85) and the guard stopped gufo for a harmless command. The unmatched rows are mostly my own `ls /tmp` pings from before the plugin was installed and commands whose text differs between proxy and plugin.
7. `compare.py`: built. The week of data it is meant to judge has not run yet.

Review notes (fixed in this pass): the finish detector used to be permanently
debounced per session (only the first idle was ever logged); the goal check
ran only for the first prompt of a session; unmatched prompts were logged with
their text; tests wrote into the real log directory; there was no activity end
record, no retention and no fail-open test.

Known remaining issues: the turn builder for the effort gate (above); the
unmatched rows in `compare.py`; Q3/Q4/Q5; the daemon does not read the plugin
log yet.


## Operations: reloading the plugin (learned 2026-11-08)

opencode keeps one long-running service (`opencode serve --service`) that holds
the plugin module in memory. Editing `tools/jev/opencode-plugin/*.ts` does not
reload it: the file-plugin revision is keyed by the target path string, not the
content, so the service keeps running the old code (observed: a live session
logged command rows but none of the goal, effort-gate, activity or finish
records until the service was restarted).

After every edit: kill the `opencode serve --service` process by pid (never
`pkill -f`), then use opencode again (the TUI starts a new service and resumes
open sessions). `status.py` warns when the service is older than the newest
`.ts` file. For a hook-by-hook trace, create `~/.gufo/jev-log/plugin-trace.on`;
the plugin then writes `plugin-trace.jsonl` (hook names, session, provider, key
lists, never content). Delete the flag file to stop.

Verified after the restart on a one-message session ("Reply with the single
word ok. jev finish it"): `goal` matched `jev finish it`, `effort_gate`,
`activity` start and end, and `finish` were all recorded.

### Pausing and stopping (added 2026-11-08)

- **Pause only the plugin** (opencode and gufo keep running): `touch ~/.gufo/jev-log/plugin-pause`.
  Every hook becomes a no-op within 2 seconds; `rm ~/.gufo/jev-log/plugin-pause` resumes.
  `status.py` prints `PAUSED: YES` while the flag exists. `JEV_PLUGIN=off` also disables it but is
  only read when the opencode service starts, so it cannot pause a service that is already running.
- **Stop the current opencode turn without quitting opencode:** use opencode's interrupt key in the
  TUI (Esc); the session stays open and can be continued.
- **Stop gufo:** `~/jev-stop.sh` stops the guard and gufo and waits for the memory to come back.
  opencode stays open, but its requests fail until gufo is started again with `~/run-gufo.sh`
  (the guard only holds requests while it is itself stopping gufo for a heavy test).

### The /jev command (added 2026-11-08)

`/jev pause`, `/jev resume` and `/jev status` in the opencode prompt. It is a markdown command
(`tools/jev/opencode-plugin/jev-command.md`, symlinked to `~/.config/opencode/commands/jev.md`)
whose shell block runs `~/.gufo/jev-ctl.sh '$1'` (symlink to `tools/jev/opencode-plugin/jev-ctl.sh`,
which accepts only `pause`, `resume` or `status`). The shell block runs when the command is
evaluated, before the prompt is submitted, then gufo is asked to repeat the output, so each use
costs one short model turn. Pause creates the flag file described above and resume deletes it.

Limits seen in testing: `/jev status` and `/jev pause` worked end to end on a throwaway session;
`/jev resume` timed out once because gufo was stopped for a heavy test and the request waited. When
gufo is down or busy use the file directly (`rm ~/.gufo/jev-log/plugin-pause` or
`~/.gufo/jev-ctl.sh resume`), which does not need a model.

## Production port (2026-11-09)

The plugin now acts, not just logs. Engine flags `--jev-gate/--jev-finish/--jev-goal` are off in
`run-gufo.sh` (`JEV_ENGINE=on` restores them); the engine code stays until the live test passes.

| Feature | Where it runs now | Switch |
|---|---|---|
| Command gate (heavy lock, drain, stop) | plugin `execute.before` -> guard `/jev-guard/prepare`; unknown commands -> `/jev-guard/classify` (NPU judge) | `/jev feature command off` |
| Effort gate (user turns + routine tool results) | plugin `context` hook decides, `http.request` hook writes `reasoning_effort` / `thinking` / `reasoning_budget_tokens` into the body | `/jev feature effort off` |
| `/jev autoeffort off [level]` | plugin plan first, guard pin as fallback | file `autoeffort-off` |
| Goal tracker | `prompt` hook strips the phrase; turn end evaluates and queues a continuation prompt | `/jev feature goal off` |
| Finish detector | turn end, text-only turns, 2 continuations max, loop guard | `/jev feature finish off` |
| Session identity | `http.request` adds `x-session-id`, `x-jev-plugin: 1` | pause the session |
| `/jev` command | native plugin command (`ctx.command.transform`), session id from `execute({ sessionID })`; output is a synthetic message with `resume: false`, so no model turn | n/a |

Stays outside the plugin: memory and kfd watchdogs, gufo supervisor, the single NPU judge (all need
process and RAM visibility across sessions), and `--jev-pre` (answers only tool-less single-turn
yes/no requests; opencode traffic always carries tools).

Checks for the live test: `plugin-<date>.jsonl` features `effort_apply`, `goal_arm`, `goal_eval`,
`finish` (field `acted`), `command_gate` (note `gate waited`); guard log lines `request keys:`.

### Verified on 2026-11-09 (native `/jev`)

`GET /api/command` lists `jev` once the plugin loads. `POST /api/session/<id>/command` with
`{"name":"jev","text":"pause"}` creates `plugin-pause.<sessionID>` for that session only; `status`
and `resume` work the same way. Output lands in the session inbox as a synthetic message and is
delivered with the next turn. The `~/.config/opencode/plugins/jev.ts` symlink must point at the
repo copy (`tools/jev/opencode-plugin/jev.ts`); a stale symlink to a deleted worktree makes the
service skip the plugin silently.

The guard refuses to stop gufo for a command whose process needle is shorter than 5 characters
(a bare `env` matched `/usr/bin/env` wrappers and held gufo down for the full hold time).
