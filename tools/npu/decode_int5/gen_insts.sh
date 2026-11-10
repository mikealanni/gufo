#!/bin/bash
# usage: gen_insts.sh   (needs aiecc and the aie python package; AIETOOLS set)
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
gen() {
  python3 "$dir/n32_core_placed.py" -m 16 -k 64 -n 128 -M 16 -K "$1" -N "$2" --mode "$3" >"$work/a.mlir"
  aiecc --aietools="${AIETOOLS:?}" --alloc-scheme=basic-sequential --no-compile --no-link \
    --no-xchesscc --aie-generate-npu-insts --unified --dynamic-objFifos \
    --npu-insts-name="$work/i.txt" "$work/a.mlir" >/dev/null
  mv "$work/i.txt" "$dir/runtime/$4"
}
gen 2048 15360 ffn insts_ffn.txt
gen 2048 6144 plain insts_plain_2048x6144.txt
gen 2048 2048 plain insts_plain_2048x2048.txt
gen 2048 3072 plain insts_plain_2048x3072.txt
