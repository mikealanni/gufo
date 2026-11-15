#!/bin/bash
# usage: run_gen.sh interactive MODEL.gguf            (type messages, multi-turn)
#        run_gen.sh chat MODEL.gguf "prompt" [max_new]
#        run_gen.sh bench MODEL.gguf PROMPT_TOKENS NEW_TOKENS
# Prefill uses the 1024-row path by default (GEN_KD=1 GUFO_ATB=1); set GEN_KD= to disable.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
export LD_LIBRARY_PATH=$HOME/npu-work/boostshim:/opt/xilinx/xrt/lib
export GEN_KD=${GEN_KD-1} GUFO_ATB=${GUFO_ATB-1}
[ -z "$GEN_KD" ] && unset GEN_KD
cmd=$1; model=$2; shift 2
exec ~/npu-lock.sh "$ROOT/build/npu/gufo-gen" "$cmd" "$model" "$ROOT/tools/npu/decode_int5/runtime" "$@"
