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

From the opencode database the gufo providers are `kairic-direct` and
`gufo-local`, model `Qwen3.8 Flash Next` (base URLs `http://127.0.0.1:8900/v1`
and `http://192.168.6.138:8900/v1` in `~/.config/opencode/opencode.json`).
Make the provider ids and the model id configurable
(`JEV_GUFO_PROVIDERS`, default `kairic-direct,gufo-local`) and match
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

Hook: `tool.execute.before` when `tool === "bash"` and the session is on gufo.

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

Hook: `chat.message` for user messages on gufo. Detect the same phrases as
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
 "session": "ses_...", "call": "call_...", "provider": "kairic-direct",
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

(to be filled in by Task 0)
