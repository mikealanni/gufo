#!/bin/bash
# usage: gen_kdh.sh <out_name> <INTER> <n32_kdh.py args...>  -> ../runtime/<out_name>.txt (instruction stream for the hybrid kd program)
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
out=$1; inter=$2; shift 2
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
INTER=$inter python3 "$dir/n32_kdh.py" -m 128 -k 64 -n 128 "$@" >"$work/a.mlir"
aiecc --aietools="${AIETOOLS:?}" --alloc-scheme=basic-sequential --no-compile --no-link --no-xchesscc --aie-generate-npu-insts --unified --dynamic-objFifos --npu-insts-name="$work/i.txt" "$work/a.mlir" >/dev/null
mv "$work/i.txt" "$dir/../runtime/$out.txt"
echo "generated $out.txt"
