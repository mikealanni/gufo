#!/bin/bash
# usage: gen_kd.sh <out_name> <INTER> <n32_core_placed args...>   -> ../runtime/<out_name>.txt (kd 1024-row instruction stream)
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
out=$1; inter=$2; shift 2
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
INTER=$inter python3 "$dir/n32_core_placed.py" -m 128 -k 64 -n 128 "$@" >"$work/a.mlir"
aiecc --aietools="${AIETOOLS:?}" --alloc-scheme=basic-sequential --no-compile --no-link --no-xchesscc --aie-generate-npu-insts --unified --dynamic-objFifos --npu-insts-name="$work/i.txt" "$work/a.mlir" >/dev/null
mv "$work/i.txt" "$dir/../runtime/$out.txt"
echo "generated $out.txt"
