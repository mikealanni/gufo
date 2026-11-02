#!/usr/bin/env bash
# tg-at-depth comparison: dflash2 vs AR, and an optional stage profile.
# Usage:
#   tools/tg_depth_test.sh all    [depth]   # dflash2 then AR (default depth 983040)
#   tools/tg_depth_test.sh dflash [depth]
#   tools/tg_depth_test.sh ar     [depth]
#   tools/tg_depth_test.sh prof   [depth]   # profiled dflash2, small n; default depth 131072
set -u
cd "$(dirname "$0")/.."

if pgrep -f "gufo serve" >/dev/null; then
  echo "[tg] a 'gufo serve' is running — stop it first (GPU contention)"
  exit 1
fi

export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib/hip:/opt/rocm/lib/hipblas:${LD_LIBRARY_PATH:-}"
export GUFO_TWO_BANK="${GUFO_TWO_BANK:-1}"
export GUFO_NPU_WHOLE="${GUFO_NPU_WHOLE:-1}"
export GUFO_NPU_DIR="$HOME/gufo-yarn/tools/npu"

MODEL=/mnt/raid0/user_cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/83cadfda58d30be06c110518208d1bb918b33f10/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
DFLASH="$HOME/models/flash-next-dflash2/Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf"

MODE="${1:-all}"
case "$MODE" in
  prof) DEPTH="${2:-131072}"; NGEN=32 ;;
  *)    DEPTH="${2:-983040}"; NGEN=128 ;;
esac
DIR=/tmp/tg_depth
mkdir -p "$DIR"
STAMP=$(date +%Y%m%d_%H%M%S)

bench() { ./build/release/gufo bench -m "$MODEL" -d "$DEPTH" -p 8192 -n "$NGEN" -v "$@"; }

run_bench() {
  local tag=$1; shift
  local log="$DIR/${tag}_${STAMP}.log"
  echo "[tg] === $tag depth=$DEPTH -> $log"
  bench "$@" 2>&1 | tee "$log"
}

case "$MODE" in
  dflash) run_bench dflash2 --speculative dflash2 --dflash-model "$DFLASH" ;;
  ar)     run_bench ar --speculative off ;;
  all)    run_bench dflash2 --speculative dflash2 --dflash-model "$DFLASH"
          run_bench ar --speculative off ;;
  prof)
    echo "[tg] === profiled dflash2 depth=$DEPTH n=$NGEN (profiling offloads to rocprofv3; expect several minutes)"
    tools/prof/prof.py run -o "$DIR/prof_df" --stages qwen -- \
      ./build/release/gufo bench -m "$MODEL" -d "$DEPTH" -p 8192 -n "$NGEN" -v \
        --speculative dflash2 --dflash-model "$DFLASH"
    tools/prof/prof.py show "$DIR/prof_df" --stages qwen --passes 500 --top 20 | tee "$DIR/prof_df_show_${STAMP}.log"
    echo "[tg] passes table above: identify the decode pass (small tokens, many rounds), then:"
    echo "[tg]   tools/prof/prof.py show $DIR/prof_df --stages qwen --passes 500 --pass N --top 20"
    ;;
  *) echo "usage: $0 [all|dflash|ar|prof] [depth]"; exit 2 ;;
esac

echo "[tg] logs in $DIR"
