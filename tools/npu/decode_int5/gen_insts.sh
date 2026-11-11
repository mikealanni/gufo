#!/bin/bash
# Generate instruction streams for model spec program needs.
# usage:
#   gen_insts.sh                    # default LFM3 set
#   gen_insts.sh plain K N          # one plain program -> insts_plain_KxN.txt
#   gen_insts.sh ffn K1 N1 K2 N2    # fused ffn program -> insts_ffn.txt (or named)
#   gen_insts.sh spec <spec.json>   # (future) emit all distinct (K,N,mode) from spec
# Needs aiecc and the aie python package; AIETOOLS set.
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

gen_plain() {
  local K="$1" N="$2"
  local out="${3:-insts_plain_${K}x${N}.txt}"
  python3 "$dir/n32_core_placed.py" -m 16 -k 64 -n 128 -M 16 -K "$K" -N "$N" --mode plain --k2 "$K" --n2 "$N" >"$work/a.mlir"
  aiecc --aietools="${AIETOOLS:?}" --alloc-scheme=basic-sequential --no-compile --no-link \
    --no-xchesscc --aie-generate-npu-insts --unified --dynamic-objFifos \
    --npu-insts-name="$work/i.txt" "$work/a.mlir" >/dev/null
  mv "$work/i.txt" "$dir/runtime/$out"
  echo "generated $out (plain ${K}x${N})"
}

gen_ffn() {
  local K1="$1" N1="$2" K2="$3" N2="$4" out="${5:-insts_ffn.txt}"
  python3 "$dir/n32_core_placed.py" -m 16 -k 64 -n 128 -M 16 -K "$K1" -N "$N1" --mode ffn \
    --k1 "$K1" --n1 "$N1" --k2 "$K2" --n2 "$N2" >"$work/a.mlir"
  aiecc --aietools="${AIETOOLS:?}" --alloc-scheme=basic-sequential --no-compile --no-link \
    --no-xchesscc --aie-generate-npu-insts --unified --dynamic-objFifos \
    --npu-insts-name="$work/i.txt" "$work/a.mlir" >/dev/null
  mv "$work/i.txt" "$dir/runtime/$out"
  echo "generated $out (ffn ${K1}x${N1} -> ${K2}x${N2})"
}

gen_conv() {
  local H="$1" N1="$2" out="${3:-insts_conv.txt}"
  python3 "$dir/n32_core_placed.py" -m 16 -k 64 -n 128 -M 16 -K "$H" -N "$H" --mode conv \
    --k1 "$H" --n1 "$N1" --k2 "$H" --n2 "$H" >"$work/a.mlir"
  aiecc --aietools="${AIETOOLS:?}" --alloc-scheme=basic-sequential --no-compile --no-link \
    --no-xchesscc --aie-generate-npu-insts --unified --dynamic-objFifos \
    --npu-insts-name="$work/i.txt" "$work/a.mlir" >/dev/null
  mv "$work/i.txt" "$dir/runtime/$out"
  echo "generated $out (conv ${H}: in_proj ${N1} -> out_proj ${H})"
}

case "${1:-default}" in
  conv)
    gen_conv "$2" "$3" "${4:-}"
    ;;
  plain)
    gen_plain "$2" "$3" "${4:-}"
    ;;
  ffn)
    gen_ffn "$2" "$3" "$4" "$5" "${6:-}"
    ;;
  spec)
    # Minimal: emit LFM3 defaults; full spec-driven enumeration is in ModelSpec tooling.
    echo "spec-driven generation: using model spec program needs (LFM3 default set)"
    gen_ffn 2048 15360 7680 2048 insts_ffn.txt
    gen_conv 2048 6144
    gen_plain 2048 2048
    gen_plain 2048 3072
    ;;
  default|*)
    # LFM3-1.7B (WP6): shapes explicit, no literals in generator.
    gen_ffn 2048 15360 7680 2048 insts_ffn.txt
    gen_conv 2048 6144
    gen_plain 2048 2048
    gen_plain 2048 3072
    ;;
esac
