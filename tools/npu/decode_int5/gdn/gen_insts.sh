#!/bin/bash
# usage: gen_insts.sh NE...   (tokens per head = 4*NE) -> ../runtime/insts_gdn_<NE>.txt
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for ne in "$@"; do
  GDN_NE=$ne python3 "$dir/n32_gdn.py" -m 16 -k 64 -n 128 >"$work/a.mlir"
  aiecc --aietools="${AIETOOLS:?}" --alloc-scheme=basic-sequential --no-compile --no-link --no-xchesscc --aie-generate-npu-insts --unified --dynamic-objFifos --npu-insts-name="$work/i.txt" "$work/a.mlir" >/dev/null
  mv "$work/i.txt" "$dir/../runtime/insts_gdn_$ne.txt"
  echo "generated insts_gdn_$ne.txt ($((ne * 4)) tokens per head)"
done
