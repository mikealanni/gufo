#!/bin/bash
# usage: gen_insts.sh M K N [M K N ...]
# Writes insts_<M>x<K>x<N>_128x64x128.txt next to this script. Needs aiecc and the
# aie python package on PATH (source the toolchain environment first).
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
while [ $# -ge 3 ]; do
  m=$1 k=$2 n=$3
  shift 3
  python3 "$dir/n32_core_placed.py" -m 128 -k 64 -n 128 -M "$m" -K "$k" -N "$n" >"$work/a.mlir"
  aiecc --aietools="${AIETOOLS:?}" --alloc-scheme=basic-sequential --no-compile \
    --no-link --no-xchesscc --aie-generate-npu-insts --unified --dynamic-objFifos \
    --npu-insts-name="$work/i.txt" "$work/a.mlir" >/dev/null
  mv "$work/i.txt" "$dir/insts_${m}x${k}x${n}_128x64x128.txt"
done
