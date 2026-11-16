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
  the model it stopped. The last loaded model is kept across guard restarts.
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
reads `GUFO_MODEL` (`flash-next`, `27b`, `35b`, `glm`, `flint`, `laguna`) and
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

- **Model.** The launcher `~/run-jev-guard.sh` runs the pplx decider
  (Qwen3.5-4B, Q8_0) on the general NPU engine (`gufo-judge`, built from
  `tools/npu/gen`). About 370 ms per request, 4.5 GB real RAM (measured as the
  drop in `MemAvailable`; NPU buffers do not show in RSS).
- **Cutoff.** Probabilities are not comparable between judges. The old 0.85
  cutoff flagged none of the 239 unknown commands in a replay of real sessions.
  pplx uses `0.05` (flags 18% of unknown commands, 3 of 4 labeled heavy ones);
  the d1 decision model uses about `0.3`. Each false stop costs a gufo reload
  (about 70 s), so tune it from `~/.gufo/jev-log/jev.sqlite3` before relying on it.
- **Roll back.** `run-jev-guard.sh.bak-pre-pplx` restores the d1 launcher.

## Start-up check

`~/run-gufo.sh` checks free RAM before it starts the guard and exits when the
model does not fit. `FORCE=1` skips the check. The guard cannot free memory; it
only stops a running gufo.

## opencode plugin

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
