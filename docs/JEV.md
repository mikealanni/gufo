# jev: guard and opencode plugin

jev keeps a local gufo server usable while other heavy work runs on the same
machine. It has two parts.

- **Guard** (`tools/jev/guard.py`): a proxy on `:8900` in front of gufo on
  `:8901`. It stops gufo before heavy commands, restarts the same model
  afterwards, and tracks which model is loaded.
- **opencode plugin** (`tools/jev/opencode-plugin/`): classifies shell commands
  before they run, controls reasoning effort, and adds the `/jev` command.

## What it does

- **Heavy-command stop.** A command that needs the GPU, the NPU or much RAM
  stops gufo first. Resource classes (`gpu`, `npu`, `ram`) decide which
  commands conflict with the loaded model; small models that leave RAM free
  are not stopped for RAM-only work.
- **Same-model restart.** The guard restarts gufo only if it stopped it, with
  the model it stopped. A fresh guard start loads the model named by `GUFO_MODEL`
  (default `flash-next`); a model remembered from an earlier run never overrides it.
  A server started with `GUFO_THINK=off` keeps thinking off for every request.
- **Watchdogs.** Memory and KFD watchdogs stop gufo before the host runs out
  of memory or the GPU driver wedges.
- **Model switch.** `/jev switch <model>` reloads gufo as another model;
  clients keep their connection and are pointed at the new model.
- **Effort control.** The plugin picks the reasoning effort per request, or
  pins it.
- **Recover.** When a turn fails because gufo restarted or the connection dropped,
  the plugin waits for the guard to report gufo up and sends "Continue" (at most
  3 times per 30 minutes per session).
- **Goal, finish, pre.** Keep the agent working until a task is done, detect
  early finishes, and screen requests before the model sees them.

## Run

```sh
python3 tools/jev/guard.py --gufo-bin build/release/gufo --run-script ~/run-gufo.sh
```

Point clients at `http://127.0.0.1:8900/v1`. The run script starts gufo and
reads `GUFO_MODEL` (`flash-next`, `27b`, `35b`, `glm`, `glm-coder`, `flint`, `laguna`, `inkling`) and
optionally `GUFO_MODEL_PATH` for the weights file. `GUFO_KIND` is still read
as the old name.

Guard endpoints: `GET /jev-guard/status`, `POST /jev-guard/prepare`,
`/jev-guard/classify`, `/jev-guard/switch`.

## Judge

Commands the rules cannot classify go to one resident NPU decision model, owned
by the guard (`gufo judge --model <gguf> --stdio`, one `READY` line, then one
question per line, answer `Yes|No <p_yes> <p_no> ...`). The plugin never starts a
second NPU process. A command is treated as heavy when `p_yes` reaches
`--judge-cutoff`.

- **Model.** `~/run-gufo.sh` picks the judge from the loaded model: pplx for
  `27b`, `35b`, `flint` and `inkling`; d1 for `flash-next`, `glm`, `glm-coder`
  and `laguna`. An explicit `GUFO_JUDGE` overrides that. Both run on the general
  NPU engine (`gufo-judge`, built from `tools/npu/gen`). d1 (d1-2B, Q8_0) uses
  about 2.3 GB of RAM; pplx (Qwen3.5-4B, Q8_0) about 370 ms per request and 4.5 GB
  real RAM (measured as the drop in `MemAvailable`; NPU buffers do not show in
  RSS).
- **Question.** Each judge gets its own framing from `guard.py`: pplx asks
  whether the command *actively occupies* the GPU or most RAM, explicitly
  excluding downloads, copies and reads; d1 asks whether the command uses the
  GPU *right now* and says that downloading a model file does not. The generic
  "needs the whole GPU or most of the RAM" wording scored `lemonade pull
  Qwen3.8-Flash-Next-Gufo 2>&1` as heavy on both judges.
- **Cutoff.** Probabilities are not comparable between judges. The launcher
  uses `0.2` for pplx and `0.30` for d1; `JUDGE_CUTOFF` overrides either.
  With the framing above, the lemonade command scores 0.029 on pplx and 0.126
  on d1 (both light), while 34/38 and 11/13 labeled heavy commands still stop
  gufo respectively. Each false stop costs a gufo reload (about 70 s), so tune
  from `~/.gufo/jev-log/jev.sqlite3` before relying on it.

## Start-up check

`~/run-gufo.sh` checks free RAM before it starts the guard and exits when the
model does not fit. `FORCE=1` skips the check. The guard cannot free memory; it
only stops a running gufo.

## opencode plugin

The full setup, flags and environment variables are in [JEV-SETUP.md](JEV-SETUP.md).

Install the server plugin and the TUI plugin from the opencode config
directory:

```sh
ln -s $PWD/tools/jev/opencode-plugin/jev.ts ~/.config/opencode/plugins/jev.ts
mkdir -p ~/.config/opencode/plugins/jev-tui
ln -s $PWD/tools/jev/opencode-plugin/jev-tui.ts ~/.config/opencode/plugins/jev-tui/tui.ts
```

opencode loads a TUI plugin only from a directory that contains `tui.ts`.
Restart opencode after changing either file.

## /jev commands

| Command | Effect |
| --- | --- |
| `/jev status` | plugin and guard state |
| `/jev pause [all]`, `/jev resume [all]` | pause or resume jev |
| `/jev autoeffort on` | jev picks the reasoning effort |
| `/jev autoeffort off [level]` | pin the effort (default `low`); levels `none minimal low medium high xhigh max`, `none` disables thinking |
| `/jev switch <model>` | reload gufo as another model |
| `/jev goal [text]`, `/jev goal off` | keep working until done (max 3 pushes), or cancel |
| `/jev feature <name> on\|off` | `command effort goal finish pre recover` |
| `/jev help` | list the options |

The same controls work from a terminal with `jev-ctl.sh`.

## Notes

- A heavy stop interrupts requests in flight, including long compaction
  prompts. NPU test commands stop a gufo that holds the NPU and reload the
  model afterwards.
- Pinned effort applies only to requests that carry no effort of their own.
- Decisions are logged to `~/.gufo/jev-log/jev.sqlite3`; the plugin writes
  `plugin-<date>.jsonl` beside it.

Design history and phases: [JEV-OPENCODE-PLUGIN.md](JEV-OPENCODE-PLUGIN.md).
