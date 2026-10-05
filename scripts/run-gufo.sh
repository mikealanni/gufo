#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next UD-IQ4_XS with MTP drafting and a 131072-token context.
#
# Required: GUFO_MODEL = first shard of UD-IQ4_XS (...-00001-of-00003.gguf)
# Optional: GUFO_MTP (shared Q8_0 sidecar), GUFO_BIN, PORT, JOBS, GUFO_CONTEXT, GUFO_EFFORT,
#           GUFO_CACHE_DIR, GUFO_CACHE_BYTES, GUFO_LOG
set -euo pipefail

GUFO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GUFO_BIN=${GUFO_BIN:-}
if [ -z "$GUFO_BIN" ]; then
  for candidate in "$GUFO_DIR/build/release/gufo" "$GUFO_DIR/result/bin/gufo"; do
    [ -x "$candidate" ] && GUFO_BIN=$candidate && break
  done
fi
[ -x "$GUFO_BIN" ] || { echo "gufo binary not found; build it or set GUFO_BIN" >&2; exit 1; }

: "${GUFO_MODEL:?set GUFO_MODEL to the first UD-IQ4_XS shard (...-00001-of-00003.gguf)}"
[ -f "$GUFO_MODEL" ] || { echo "model not found: $GUFO_MODEL" >&2; exit 1; }

PORT=${PORT:-8900}
JOBS=${JOBS:-2}
CONTEXT=${GUFO_CONTEXT:-131072}
MTP=${GUFO_MTP:-}

export LD_LIBRARY_PATH="${ROCM_PATH:-/opt/rocm}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if pgrep -f "^$GUFO_BIN serve" > /dev/null; then
  echo "gufo is already running" >&2
  exit 1
fi

CACHE_DIR=${GUFO_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/gufo/snapshots}
CACHE_BYTES=${GUFO_CACHE_BYTES:-34359738368}
LOG=${GUFO_LOG:-${XDG_CACHE_HOME:-$HOME/.cache}/gufo/gufo.log}
mkdir -p "$CACHE_DIR" "$(dirname "$LOG")"
[ -f "$LOG" ] && [ "$(stat -c%s "$LOG")" -gt 104857600 ] && mv -f "$LOG" "$LOG.1"

if [ -n "$MTP" ] && [ -f "$MTP" ]; then
  SPEC=(--speculative mtp --mtp-model "$MTP")
else
  echo "WARNING: no MTP sidecar (set GUFO_MTP to the unsloth shared Q8_0 file): running without speculative decoding." >&2
  SPEC=()
fi

"$GUFO_BIN" serve llm \
  -m "$GUFO_MODEL" \
  --served-model-name "Qwen3.8 Flash Next" \
  --port "$PORT" \
  -c "$CONTEXT" \
  -n 16384 \
  -j "$JOBS" \
  --prefill-chunk 4096 \
  "${SPEC[@]}" \
  --cache-disk "$CACHE_DIR" \
  --cache-disk-bytes "$CACHE_BYTES" \
  --cache-disk-staging-bytes "${GUFO_CACHE_STAGING_BYTES:-4294967296}" \
  --cache-ram-bytes "${GUFO_CACHE_RAM_BYTES:-3758096384}" \
  --cache-ram-entries "${GUFO_CACHE_RAM_ENTRIES:-3}" \
  --cache-disk-stride-tokens "${GUFO_CACHE_STRIDE:-16384}" \
  --reasoning-effort "${GUFO_EFFORT:-low}" \
  --log-progress \
  "$@" 2>&1 | tee -a "$LOG"
