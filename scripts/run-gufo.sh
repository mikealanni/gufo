#!/usr/bin/env bash
# Serve a Qwen model with speculative decoding and, when set up, the NPU.
#
# GUFO_KIND selects the model family (default flash-next):
#   flash-next  Qwen3.8 Flash-Next UD-IQ4_XS, DFlash2 draft (or MTP sidecar), two-bank NPU prefill
#   27b         Qwen3.8-27B, DFlash2 and/or MTP
#   35b         Qwen3.6-35B-A3B, embedded MTP head and/or DFlash2
#   glm         GLM-5.3-Flash AJ-IQ2_XXS, DFlash2 draft, two-bank NPU prefill
#   ling32      Ling 3.2 Air Q4_K_M, no draft model, GPU-only hybrid KDA/MLA engine
#
# Required: GUFO_MODEL = the GGUF (for flash-next the first shard, ...-00001-of-00003.gguf)
# Optional: GUFO_MTP (flash-next: shared Q8_0 sidecar; 27b: MTP GGUF; 35b: "embedded" by default),
#           GUFO_DFLASH (DFlash2 draft GGUF. flash-next uses it alone when set, unless GUFO_FN_SPEC=mtp;
#                        27b/35b: with an MTP head the server picks per request),
#           GUFO_PREFILL_CHUNK (default 8192; the pass size while another request is decoding or waiting),
#           GUFO_BIN, PORT, JOBS, GUFO_CONTEXT (default 131072 for flash-next, 262144 for 27b and 35b), GUFO_EFFORT,
#           GUFO_CACHE_DIR, GUFO_CACHE_BYTES, GUFO_LOG, GUFO_PREFILL_TILE (default 16384; 24576 is faster for 24K+ prompts, +6 GB RAM),
#           GUFO_TWO_BANK (default 1; 0 turns the two-bank NPU prefill off)
# NPU build: GUFO_NPU=0 forces GPU-only; XRT_DIR (default /opt/xilinx/xrt) holds the XRT libraries.
#            The NPU is used only while `tools/npu/fabric-clock.sh status` reports held
#            (sudo tools/npu/fabric-clock.sh hold).
set -euo pipefail

GUFO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
KIND=${GUFO_KIND:-flash-next}
case "$KIND" in
  flash-next) SERVED_NAME="Qwen3.8 Flash Next"; DEFAULT_CONTEXT=131072 ;;
  27b) SERVED_NAME="Qwen3.8 27B"; DEFAULT_CONTEXT=262144 ;;
  35b) SERVED_NAME="Qwen3.6 35B-A3B"; DEFAULT_CONTEXT=262144 ;;
  glm) SERVED_NAME="GLM-5.3 Flash"; DEFAULT_CONTEXT=131072 ;;
  ling32) SERVED_NAME="Ling 3.2 Air"; DEFAULT_CONTEXT=262144 ;;
  *) echo "GUFO_KIND must be flash-next, 27b, 35b, glm or ling32 (got: $KIND)" >&2; exit 1 ;;
esac
GUFO_BIN=${GUFO_BIN:-}
if [ -z "$GUFO_BIN" ]; then
  for candidate in "$GUFO_DIR/build/release/gufo" "$GUFO_DIR/result/bin/gufo"; do
    [ -x "$candidate" ] && GUFO_BIN=$candidate && break
  done
fi
[ -x "$GUFO_BIN" ] || { echo "gufo binary not found; build it or set GUFO_BIN" >&2; exit 1; }

: "${GUFO_MODEL:?set GUFO_MODEL to the model GGUF (for flash-next the first UD-IQ4_XS shard, ...-00001-of-00003.gguf)}"
[ -f "$GUFO_MODEL" ] || { echo "model not found: $GUFO_MODEL" >&2; exit 1; }

PORT=${PORT:-8900}
JOBS=${JOBS:-2}
# One executor serves every GLM session, so more than one job only queues.
[ "$KIND" = glm ] && JOBS=${JOBS_GLM:-1}
CONTEXT=${GUFO_CONTEXT:-$DEFAULT_CONTEXT}
MTP=${GUFO_MTP:-}
DFLASH=${GUFO_DFLASH:-}

export LD_LIBRARY_PATH="${ROCM_PATH:-/opt/rocm}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# GUFO_PREFILL_TILE=24576 (two 12288-token banks) prefills prompts of 24K tokens and
# more about 2.7% faster than 16384 but needs about 6 GB more RAM.
export GUFO_PREFILL_TILE=${GUFO_PREFILL_TILE:-16384}

NPU_ENV=()
# Block-sparse attention keeps 27b / 35b prefill and decode nearly flat with depth
# (approximate; GUFO_SPARSE=0 restores exact dense attention).
SPARSE_ENV=()
if [ "$KIND" != flash-next ] && [ "$KIND" != glm ] && [ "${GUFO_SPARSE:-1}" != 0 ]; then
  SPARSE_ENV=(GUFO_QWEN_SPARSE_ATTN=1)
fi
if [ "${GUFO_NPU:-1}" != 0 ] && [ -e /dev/accel/accel0 ] && [ -d "$GUFO_DIR/tools/npu" ]; then
  XRT_DIR=${XRT_DIR:-/opt/xilinx/xrt}
  if ! "$GUFO_DIR/tools/npu/fabric-clock.sh" status > /dev/null 2>&1; then
    echo "WARNING: GPU fabric clock not held (sudo $GUFO_DIR/tools/npu/fabric-clock.sh hold): running without the NPU." >&2
  else
    NPU_ENV=(GUFO_NPU_DIR="$GUFO_DIR/tools/npu")
    # Flash-Next: two-bank prefill with the whole projections on the NPU (about
    # +17%, not bit-identical to the single-bank path; see docs/NPU.md). The 27B
    # and 35B-A3B share each projection between the GPU and the NPU instead.
    if [ "$KIND" = flash-next ] && [ "${GUFO_TWO_BANK:-1}" != 0 ]; then
      NPU_ENV+=(GUFO_TWO_BANK=1 GUFO_NPU_WHOLE=1)
    fi
    export LD_LIBRARY_PATH="$XRT_DIR/lib:$LD_LIBRARY_PATH"
  fi
fi

if pgrep -f "^$GUFO_BIN serve" > /dev/null; then
  echo "gufo is already running" >&2
  exit 1
fi

CACHE_DIR=${GUFO_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/gufo/snapshots}
CACHE_BYTES=${GUFO_CACHE_BYTES:-34359738368}
LOG=${GUFO_LOG:-${XDG_CACHE_HOME:-$HOME/.cache}/gufo/gufo.log}
mkdir -p "$CACHE_DIR" "$(dirname "$LOG")"
[ -f "$LOG" ] && [ "$(stat -c%s "$LOG")" -gt 104857600 ] && mv -f "$LOG" "$LOG.1"

SPEC=()
if [ "$KIND" = glm ]; then
  if [ -n "$DFLASH" ] && [ -f "$DFLASH" ]; then
    SPEC=(--speculative dflash2 --dflash-model "$DFLASH")
  else
    echo "WARNING: no DFlash2 draft (set GUFO_DFLASH): running without the drafter." >&2
  fi
elif [ "$KIND" = ling32 ]; then
  :
elif [ "$KIND" = flash-next ]; then
  if [ "${GUFO_FN_SPEC:-dflash2}" = dflash2 ] && [ -n "$DFLASH" ] && [ -f "$DFLASH" ]; then
    SPEC=(--speculative dflash2 --dflash-model "$DFLASH")
  elif [ -n "$MTP" ] && [ -f "$MTP" ]; then
    SPEC=(--speculative mtp --mtp-model "$MTP")
  else
    echo "WARNING: no DFlash2 draft or MTP sidecar (set GUFO_DFLASH or GUFO_MTP): running without speculative decoding." >&2
  fi
else
  if [ "$KIND" = 35b ] && [ -z "$MTP" ]; then MTP=embedded; fi
  HAVE_MTP=0; [ -n "$MTP" ] && { [ "$MTP" = embedded ] || [ -f "$MTP" ]; } && HAVE_MTP=1
  HAVE_DFLASH=0; [ -n "$DFLASH" ] && [ -f "$DFLASH" ] && HAVE_DFLASH=1
  if [ $HAVE_MTP = 1 ] && [ $HAVE_DFLASH = 1 ]; then
    SPEC=(--speculative auto --dflash-model "$DFLASH" --mtp-model "$MTP")
  elif [ $HAVE_DFLASH = 1 ]; then
    SPEC=(--speculative dflash2 --dflash-model "$DFLASH")
  elif [ $HAVE_MTP = 1 ]; then
    SPEC=(--speculative mtp --mtp-model "$MTP")
  else
    echo "WARNING: no draft model (set GUFO_DFLASH and/or GUFO_MTP): running without speculative decoding." >&2
  fi
fi

env "${NPU_ENV[@]}" "${SPARSE_ENV[@]}" "$GUFO_BIN" serve llm \
  -m "$GUFO_MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  -c "$CONTEXT" \
  -n 16384 \
  -j "$JOBS" \
  --prefill-chunk "${GUFO_PREFILL_CHUNK:-8192}" \
  "${SPEC[@]}" \
  --cache-disk "$CACHE_DIR" \
  --cache-disk-bytes "$CACHE_BYTES" \
  --cache-disk-staging-bytes "${GUFO_CACHE_STAGING_BYTES:-4294967296}" \
  --cache-ram-bytes "${GUFO_CACHE_RAM_BYTES:-3758096384}" \
  --reasoning-effort "${GUFO_EFFORT:-low}" \
  --log-progress \
  "$@" 2>&1 | tee -a "$LOG"
