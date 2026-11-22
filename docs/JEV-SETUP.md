# Set up jev, the opencode plugin and the run script

Step-by-step install and every flag and environment variable. Behaviour is in
[JEV.md](JEV.md).

## What you need

- Linux x86-64, AMD Strix Halo (`gfx1151`), gufo built (`build/release/gufo`; see
  [Build from source](../README.md#build-from-source)).
- Python 3.10+ with `aiohttp` for the guard (`pip install aiohttp`).
- [opencode](https://opencode.ai) for the plugin (v2 or later).
- Optional, for the NPU judge and NPU prefill: XRT, the held fabric clock
  (`sudo tools/npu/fabric-clock.sh hold`) and the NPU build (see [NPU.md](NPU.md)).

## 1. Start gufo behind the guard

The guard listens on `:8900` and proxies to gufo on `:8901`. Clients (opencode,
curl) talk to `:8900` only.

```sh
python3 tools/jev/guard.py --run-script ./scripts/run-gufo.sh --gufo-bin build/release/gufo
```

The guard starts gufo by running `--run-script` with the guard's environment, so
`GUFO_MODEL=inkling python3 tools/jev/guard.py ...` loads that model. The model
named at the first launch is the one that loads; a model remembered from an earlier
run never overrides it. Check it:

```sh
curl -s http://127.0.0.1:8900/jev-guard/status
```

### Guard flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Address the guard listens on. Use `127.0.0.1` on a shared network. |
| `--port` | `8900` | Public port clients use. |
| `--upstream-port` | `8901` | Port gufo listens on behind the guard. |
| `--run-script` | `~/run-gufo.sh` | Script that starts gufo (see section 3). |
| `--gufo-bin` | `~/gufo/build/release/gufo` | The gufo binary the guard supervises. |
| `--judge-model` | `~/jev-model/...gguf` | GGUF of the resident NPU decision model. |
| `--npu-dir` | `~/gufo/tools/npu` | NPU kernels and instruction streams. |
| `--judge-cutoff` | `0.85` | Probability above which an unknown command counts as heavy. |
| `--free-target-gb` | `60` | RAM the guard tries to leave free before a heavy command. |
| `--reactive-gb` | `8.0` | Free RAM below which the memory watchdog stops gufo. |
| `--foreign-rss-gb` | `20.0` | Size of a foreign process that counts as heavy. |
| `--quiet` | `8` s | Quiet time after a heavy command before gufo restarts. |
| `--appear-grace` | `30` s | Wait for a heavy process to show up after its command starts. |
| `--max-hold` | `7200` s | Longest a request is held while gufo is stopped. |
| `--start-timeout` | `420` s | Time allowed for gufo to become ready. |
| `--stop-timeout` | `120` s | Time allowed for gufo to stop. |
| `--active-window` | `600` s | How long after the last request opencode counts as active. |
| `--jev-log` | `~/.gufo/jev-log/jev.sqlite3` | Decision log. |

Run `python3 tools/jev/guard.py --help` for the full list on your version.

### Guard environment

| Variable | Effect |
| --- | --- |
| `GUFO_MODEL` | Model to load: `flash-next`, `27b`, `35b`, `glm`, `flint`, `laguna`, `inkling`. |
| `GUFO_MODEL_PATH` | Weights file for that model. |
| `GUFO_THINK=off` | Thinking is off for every request; clients cannot turn it back on. |
| `GUFO_EFFORT` | Default reasoning effort (`low` if unset). |
| `GUFO_JUDGE` | Decision model: `pplx` (default) or `d1`. |
| `JUDGE_CUTOFF` | Overrides the judge cutoff. |

## 2. Install the opencode plugin

From the repository root:

```sh
mkdir -p ~/.config/opencode/plugins/jev-tui ~/.gufo
ln -s $PWD/tools/jev/opencode-plugin/jev.ts ~/.config/opencode/plugins/jev.ts
ln -s $PWD/tools/jev/opencode-plugin/jev-tui.ts ~/.config/opencode/plugins/jev-tui/tui.ts
ln -s $PWD/tools/jev/opencode-plugin/jev-ctl.sh ~/.gufo/jev-ctl.sh
mkdir -p ~/.local/bin && ln -s ~/.gufo/jev-ctl.sh ~/.local/bin/jev
```

- opencode loads a TUI plugin only from a directory that contains `tui.ts`.
- The last two links add the `jev` command (`jev status`, `jev goal off`, ...).
- Restart opencode after changing either plugin file.

Point opencode at the guard in `~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "local-gufo": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "gufo (:8900)",
      "options": { "baseURL": "http://127.0.0.1:8900/v1", "apiKey": "not-needed" },
      "models": {
        "Inkling 2 Mini": { "name": "Inkling 2 Mini", "limit": { "context": 262144, "output": 32768 } }
      }
    }
  }
}
```

The model key must match the id gufo serves (`curl :8900/v1/models`).

### Start opencode without a background service

opencode normally connects to a shared background server (`opencode serve
--service`) that keeps running sessions after you close the window. For one
session that ends with its window:

```sh
opencode --standalone                  # private server, ends with the window
opencode run "your prompt"             # one prompt, then exit
opencode service status | stop         # inspect or stop the background server
alias opencode='opencode --standalone' # make it the default (~/.bashrc)
```

### Plugin environment

| Variable | Effect |
| --- | --- |
| `JEV_PLUGIN=off` | Every hook becomes a no-op. |
| `JEV_GUARD_URL` | Guard address (default `http://127.0.0.1:8900`). |
| `JEV_LOG_DIR` | Directory for plugin state (default `~/.gufo/jev-log`). |
| `JEV_GUFO_PROVIDERS` | Provider ids treated as gufo (default `local-gufo`). |
| `JEV_HEAVY_GATE` | Turns the heavy-command gate on or off. |
| `JEV_PLUGIN_COMMAND` | Path of the `/jev` command helper. |

### `/jev` and `jev` commands

`/jev status`, `pause [all]`, `resume [all]`, `autoeffort on | off [level]`,
`switch <model>`, `goal [text]`, `goal off`, `feature <name> on|off`, `help`.
A goal is opt-in per use and makes at most three pushes. Running `jev goal`
with no text arms one, so do not use it to look at the state.

## 3. The run script

The run script starts gufo for the guard. `scripts/run-gufo.sh` is the minimal
launcher in this repository (flash-next, 27b, 35b, glm; `GUFO_KIND` picks the
model and `GUFO_MODEL` is the weights file). A multi-model script reads the
variables below; keep your own copy beside your weights paths.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GUFO_MODEL` | `flash-next` | Which model to serve. |
| `GUFO_MODEL_PATH` | per model | Weights file (first shard for sharded models). |
| `GUFO_DFLASH`, `GUFO_MTP` | per model | DFlash2 draft GGUF, MTP head or sidecar. |
| `GUFO_SPEC` | `on` | `off` disables speculation (Inkling DSpark). |
| `GUFO_FN_SPEC`, `GUFO_FN_DFLASH` | `mtp` | Flash-Next draft choice and DFlash2 file. |
| `GUFO_INKLING_DSPARK` | found in the HF cache | DSpark drafter for Inkling. |
| `GUFO_INKLING_NPU` | `1` | `0` turns the Inkling NPU prefill off. |
| `GUFO_IU4_ARTIFACT` | auto | Rotated int4 artifact for the 27B. |
| `GUFO_THINK` | unset | `off` answers in the text channel. |
| `GUFO_EFFORT` | `low` | Default reasoning effort. |
| `GUFO_TEMP` | unset | Default sampling temperature. |
| `CTX` / `GUFO_CONTEXT` | model default | Context tokens (Inkling 262144, Flash-Next 131072 in the repo script). |
| `JOBS` | `2` | Concurrent sessions; each holds its own KV memory. |
| `PORT` | `8900` | Public port (the guard moves gufo to `:8901`). |
| `GUFO_NPU` | `1` | `0` forces GPU-only. |
| `GUFO_TWO_BANK` | `1` | `0` turns the two-bank NPU prefill off. |
| `GUFO_SPARSE` | `1` | `0` keeps exact dense attention on 27b and 35b. |
| `GUFO_PREFILL_CHUNK` | per model | Pass size while another request is decoding. |
| `GUFO_PREFILL_TILE` | `16384` | Prefill tile; `24576` is faster for 24K+ prompts, +6 GB RAM. |
| `GUFO_CACHE_DIR` | `~/gufo-cache` | Disk continuation cache. |
| `GUFO_CACHE_BYTES` | 240 GB | Disk cache capacity. |
| `GUFO_CACHE_RAM_BYTES` | per model | RAM cache capacity. |
| `GUFO_CACHE_RAM_ENTRIES` | `3` | RAM cache entries. |
| `GUFO_CACHE_STAGING_BYTES` | per model | Staging area for disk writes. |
| `GUFO_CACHE_STRIDE` | `32768` | Disk checkpoint spacing in tokens. |
| `GUFO_BIN`, `GUFO_DIR` | the repo build | Binary and checkout to run. |
| `GUFO_LOG` | `~/gufo.log` | Server log. |
| `GUFO_GUARD`, `GUFO_JEV` | `1` | `0` runs without the guard or without jev. |
| `FORCE=1` | unset | Skips the free-RAM check before starting. |

Memory per model is the weights plus the KV cache for every session, so `JOBS`
and `CTX` together decide whether it fits. A session of Inkling at 256K holds
about 18 GB, a Flash-Next session about 83 GB in total with its weights.

## Stop and troubleshoot

- Stop everything with `~/jev-stop.sh` (the supervisor and gufo), then start again.
- Requests that never answer while a heavy command runs are held by the guard
  until gufo is back (`--max-hold`).
- If the model loads slowly, check that the weights are in the page cache and
  that nothing else is filling it; see [Performance](PERFORMANCE.md).
- Closing an opencode window does not stop its sessions unless you started it
  with `--standalone`.
