#!/usr/bin/env bash
# End-to-end DFlash2 drafter distillation from your OpenCode session data.
#
# Stage 2 (capture) MUST run while gufo serve is DOWN - it uses the GPU.
# Stage 3 (train) runs on CPU or GPU; see DEVICE below.
#
#   pkill -f run-gufo.sh; pkill -x gufo            # free the GPU for capture
#   until [ -z "$(pgrep -x gufo)" ]; do sleep 1; done
#   ./dflash-train.sh                              # assemble->capture->train->export
#   DEVICE=cpu ./dflash-train.sh                   # train on CPU (default; works now)
#   DEVICE=gpu ./dflash-train.sh                   # train on GPU (needs working torch)
#   ./dflash-test.sh                               # measure (AUTO_FT=0)
#
# Tunables (env): DEVICE=cpu|gpu  THREADS=24  EPOCHS=4  LR=2e-5  CONTEXT=256
#   BLOCKS=2  OPTIM=adamw|sgd  ONLY=<comma substrings>  STEP=0 (cap optimizer steps).
#   GPU example: DEVICE=gpu OPTIM=adamw ./dflash-train.sh
#   Quick CPU:   CONTEXT=128 STEP=4000 OPTIM=sgd ./dflash-train.sh
#
# Stages are idempotent: each writes a marker under $OUT and is skipped if done.
# Delete a marker (or $OUT) to redo that stage. Knobs are env-overridable.
set -euo pipefail

D=${D:-$HOME/dflash-train}
OPENCODE_DB=${OPENCODE_DB:-$HOME/.local/share/opencode/opencode.db}
VENV=${VENV:-$HOME/venvs/dflash-gpu}

ROOT=/home/majed/gufo
BIN=${BIN:-$ROOT/build/release/gufo}
PROBE=${PROBE:-$ROOT/build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_dflash_capture_probe}
SNAP=/mnt/raid0/user_cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots
MODEL=${MODEL:-$SNAP/83cadfda58d30be06c110518208d1bb918b33f10/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
HEAD=${HEAD:-$SNAP/83cadfda58d30be06c110518208d1bb918b33f10/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf}
DRAFT_DIR=${DRAFT_DIR:-$HOME/models/flash-next-dflash2}
DRAFT_GGUF=${DRAFT_GGUF:-$DRAFT_DIR/Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf}
CONFIG=${CONFIG:-$DRAFT_DIR/config.json}
WEIGHTS=${WEIGHTS:-$DRAFT_DIR/model.safetensors}
UPSTREAM=${UPSTREAM:-$ROOT/tools/qwen-flash/vendor/dflash2_model_pinned.py}
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/core-7.14/lib

# training hyper-params.
#   DEVICE=cpu : run on CPU. Works on this machine; full-param ~10 h/epoch @ctx256.
#   DEVICE=gpu : run on gfx1151 GPU. Currently torch segfaults on gfx1151 compute;
#                a preflight below will fall back to CPU unless torch is fixed.
EPOCHS=${EPOCHS:-4}; LR=${LR:-2e-5}; CONTEXT=${CONTEXT:-256}; BLOCKS=${BLOCKS:-2}
DTYPE=${DTYPE:-fp32}
DEVICE=${DEVICE:-cpu}; THREADS=${THREADS:-24}; OPTIM=${OPTIM:-adamw}; ONLY=${ONLY:-}
STEP=${STEP:-0}
[[ "$DEVICE" == cpu ]] && CPU_FLAG="--cpu" || CPU_FLAG=""
[[ "$STEP" -gt 0 ]] && STEP_FLAG="--steps $STEP" || STEP_FLAG=""
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"

CORPUS="$D/corpus"; CAP="$D/cap"; CKPT="$D/ckpt"; EXPORT="$D/new-drafter.gguf"
VPY="$VENV/bin/python"

echo "== guard =="
if [[ "$DEVICE" != cpu ]] && pgrep -x gufo >/dev/null; then
  echo "ERROR: gufo model process is running (pid $(pgrep -x gufo|tr '\n' ' ')); it holds the GPU." >&2
  echo "Stop it: pkill -f run-gufo.sh; pkill -x gufo   (or set DEVICE=cpu)" >&2; exit 2
fi
[[ -x "$VPY" ]] || { echo "ERROR: train venv missing. Run: $ROOT/tools/qwen-flash/setup-train-env.sh" >&2; exit 1; }

# Device preflight. DEVICE=gpu needs torch with *working* compute on gfx1151.
if [[ "$DEVICE" == gpu ]]; then
  if "$VPY" -c 'import torch;t=torch.arange(1024,device="cuda");(t+1).sum().item();torch.cuda.synchronize()' 2>/dev/null; then
    echo "preflight: GPU compute OK - training on GPU."
  else
    echo "WARN: torch GPU compute is unavailable/broken on this device (segfault on arange)." >&2
    echo "      Falling back to DEVICE=cpu. Re-run with DEVICE=gpu once torch is fixed." >&2
    DEVICE=cpu; CPU_FLAG="--cpu"
  fi
else
  echo "preflight: DEVICE=cpu (threads=$THREADS). CPU training is slow: full-param ~10 h/epoch @ctx256;"
  echo "           for a quick pass set CONTEXT=128 STEP=4000."
fi
mkdir -p "$CORPUS" "$CAP" "$CKPT"

echo "== stage 1: assemble corpus from the OpenCode database =="
if [[ ! -f "$CORPUS/.done" ]]; then
  # Assistant replies + user messages from opencode.db (your real session text).
  # The live DB is being written, so extract from a consistent snapshot copy.
  MAX_SHARDS=${MAX_SHARDS_PER_ROLE:-150}
  rm -f "$CORPUS"/*.txt
  TMPDB="$D/.opencode-snapshot.db"
  trap 'rm -f "$TMPDB" "$TMPDB-wal" "$TMPDB-shm"' EXIT
  cp "$OPENCODE_DB" "$TMPDB"
  cp "$OPENCODE_DB-wal" "$TMPDB-wal" 2>/dev/null || true
  "$VPY" "$ROOT/tools/qwen-flash/extract-opencode-corpus.py" \
    --db "$TMPDB" --out "$CORPUS" --max-per-role "$MAX_SHARDS" --max-bytes 8000
  rm -f "$TMPDB" "$TMPDB-wal" "$TMPDB-shm"; trap - EXIT
  : > "$CORPUS/.done"
  echo "assembled $(ls "$CORPUS"/*.txt 2>/dev/null|wc -l) shard(s), $(cat "$CORPUS"/*.txt|wc -c) bytes"
else echo "corpus already assembled ($CORPUS)"; fi

echo "== stage 2: capture target features (GPU) =="
if [[ ! -f "$CAP/.done" ]]; then
  [[ -x "$PROBE" ]] || { echo "ERROR: capture probe not built: $PROBE" >&2; exit 1; }
  GUFO_CAPTURE_FORCE=1 "$ROOT/tools/qwen-flash/dflash-capture.sh" \
    --corpus "$CORPUS" --out "$CAP" --context "${CAPTURE_CONTEXT:-8192}"
  : > "$CAP/.done"
else echo "features already captured ($CAP)"; fi

echo "== stage 3: distill the drafter =="
ls "$CAP"/*.meta.json >/dev/null 2>&1 || { echo "ERROR: no capture shards in $CAP" >&2; exit 1; }
"$VPY" "$ROOT/tools/qwen-flash/dflash-finetune.py" \
  --upstream "$UPSTREAM" --config "$CONFIG" --weights "$WEIGHTS" \
  --head-gguf "$HEAD" --data "$CAP" --out "$CKPT" \
  $CPU_FLAG --dtype "$DTYPE" --optimizer "$OPTIM" --lr "$LR" \
  --epochs "$EPOCHS" --context "$CONTEXT" --blocks "$BLOCKS" $STEP_FLAG \
  ${ONLY:+--only "$ONLY"} --log-every 10

LAST=$(ls -1t "$CKPT"/draft-epoch*.safetensors | head -1)
echo "checkpoint: $LAST"

echo "== stage 4: export to Gufo drafter GGUF =="
[[ -f "$DRAFT_GGUF" ]] || { echo "ERROR: template drafter GGUF missing: $DRAFT_GGUF" >&2; exit 1; }
python3 "$ROOT/tools/qwen-flash/dflash-export-gguf.py" \
  --template "$DRAFT_GGUF" --weights "$LAST" --out "$EXPORT"

echo "== done =="
echo "trained drafter: $EXPORT"
echo "measure it:      AUTO_FT=0 DRAFTERS='$EXPORT' $HOME/dflash-test.sh"
