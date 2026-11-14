#!/bin/bash
# Chat with LFM3-1.7B on the NPU: exact Q8 head, exact attention kernel up to 4k context, draft heads + prompt lookup.
# usage: run_chat.sh                      interactive chat (empty line quits)
#        run_chat.sh "prompt text"        one answer, then exit
#        run_chat.sh --bench              fixed prompt, prints tok/s
# env:   CTX (default 8192), MAXNEW (default 512)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
export LD_LIBRARY_PATH=$HOME/npu-work/boostshim:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}
export GUFO_ATB=1 GUFO_ATB_KDYN=1
ARGS=(--gguf "$HOME/models/lfm3/LFM3-1.7B-Q8_0.gguf"
      --bf16 "$HOME/halogen-models/LFM3-1.7B/model.safetensors"
      --programs "$ROOT/tools/npu/decode_int5/runtime"
      --heads "$HOME/lfm3-draft/heads_i5.bin" --head-k 2
      --max-new "${MAXNEW:-512}" --context "${CTX:-8192}")
BIN="$ROOT/build/npu/gufo-npu"
cd "$ROOT"
if [ "${1:-}" = "--bench" ]; then
  exec ~/npu-lock.sh "$BIN" generate "${ARGS[@]}" --prompt "Explain how a CPU cache works." --max-new 200
elif [ $# -gt 0 ]; then
  printf '%s\n\n' "$*" | exec ~/npu-lock.sh "$BIN" chat "${ARGS[@]}" 2>&1 | grep -v "^npu engine: packed\|^$"
else
  exec ~/npu-lock.sh "$BIN" chat "${ARGS[@]}" 2> >(grep -v "^npu engine: packed" >&2)
fi
