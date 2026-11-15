#!/bin/bash
# usage: [TILE_M=64] gen_insts.sh M K N [M K N ...]   (TILE_M defaults to 128)
# Writes insts_<M>x<K>x<N>_128x64x128.txt next to this script. Needs aiecc and the
# aie python package on PATH (source the toolchain environment first).
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
while [ $# -ge 3 ]; do
  m=$1 k=$2 n=$3
  shift 3
  tm=${TILE_M:-128}
  python3 "$dir/n32_core_placed.py" -m "$tm" -k 64 -n 128 -M "$m" -K "$k" -N "$n" >"$work/a.mlir"
  aiecc --aietools="${AIETOOLS:?}" --alloc-scheme=basic-sequential --no-compile \
    --no-link --no-xchesscc --aie-generate-npu-insts --unified --dynamic-objFifos \
    --npu-insts-name="$work/i.txt" "$work/a.mlir" >/dev/null
  mv "$work/i.txt" "$dir/insts_${m}x${k}x${n}_${tm}x64x128.txt"
done
