#!/usr/bin/env bash
# Capture DFlash2 training features for a corpus directory.
#
#   GUFO_MODEL=... ./dflash-capture.sh --corpus DIR --out DIR [--limit N]
#                   [--skip N] [--batch N] [--context N] [--raw]
#
# Each corpus file becomes one PREFIX.{tokens.i32,features.f16,meta.json}.
# The repository's dflash_capture_probe runs once per file and exits, so peak
# device memory is one model image per invocation.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)
GUFO_BIN=${GUFO_BIN:-$ROOT/build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_dflash_capture_probe}
MODEL=${GUFO_MODEL:-/mnt/raid0/user_cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/83cadfda58d30be06c110518208d1bb918b33f10/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}

CORPUS=
OUT=
LIMIT=0
SKIP=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --corpus) shift; CORPUS=$1 ;;
    --out) shift; OUT=$1 ;;
    --limit) shift; LIMIT=$1 ;;
    --skip) shift; SKIP=$1 ;;
    --batch) shift; EXTRA+=(--batch "$1") ;;
    --context) shift; EXTRA+=(--context "$1") ;;
    --raw) EXTRA+=(--raw) ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
[ -d "$CORPUS" ] || { echo "corpus directory required" >&2; exit 2; }
[ -n "$OUT" ] || { echo "--out directory required" >&2; exit 2; }
[ -x "$GUFO_BIN" ] || { echo "probe not found: $GUFO_BIN" >&2; exit 1; }
mkdir -p "$OUT"

if [ "${GUFO_CAPTURE_FORCE:-0}" != 1 ]; then
  if pgrep -af gufo 2>/dev/null | grep -q "gufo serve"; then
    echo "refusing to capture: a gufo serve is running and holds the GPU model." >&2
    echo "stop it first, or set GUFO_CAPTURE_FORCE=1." >&2
    exit 1
  fi
fi

NAME=$(basename "$CORPUS")
STEM=$(printf '%s' "$NAME" | tr -c 'a-zA-Z0-9_' _)
COUNT=0
if [ -f "$OUT/.done-$STEM" ]; then
  echo "already captured $CORPUS"
  exit 0
fi
for file in "$CORPUS"/*; do
  [ -f "$file" ] || continue
  COUNT=$((COUNT + 1))
  if [ "$SKIP" -gt 0 ] && [ "$COUNT" -le "$SKIP" ]; then continue; fi
  if [ "$LIMIT" -gt 0 ] && [ "$COUNT" -gt $((SKIP + LIMIT)) ]; then break; fi
  PREFIX="$OUT/${STEM}_$(printf '%04d' "$COUNT")"
  OUT_PREFIX=$(basename "$file")
  [ -f "$PREFIX.meta.json" ] && { echo "skip done $file"; continue; }
  echo "capture $file"
  "$GUFO_BIN" --model "$MODEL" --prompt-file "$file" --out "$PREFIX" \
    "${EXTRA[@]+"${EXTRA[@]}"}" \
    || echo "WARN: capture failed for $file" >&2
done
touch "$OUT/.done-$STEM"
echo "done $CORPUS -> $OUT"
