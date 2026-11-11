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
  local MR="${3:-1}"
  local out
  if [[ -n "${4:-}" ]]; then out="$4"; elif [[ "$MR" == 1 ]]; then out="insts_plain_${K}x${N}.txt"; else out="insts_plain_${K}x${N}_m64.txt"; fi
  python3 "$dir/n32_core_placed.py" -m 16 -k 64 -n 128 -M 16 -K "$K" -N "$N" --mode plain --k2 "$K" --n2 "$N" --mrows "$MR" >"$work/a.mlir"
  aiecc --aietools="${AIETOOLS:?}" --alloc-scheme=basic-sequential --no-compile --no-link \
    --no-xchesscc --aie-generate-npu-insts --unified --dynamic-objFifos \
    --npu-insts-name="$work/i.txt" "$work/a.mlir" >/dev/null
  mv "$work/i.txt" "$dir/runtime/$out"
  echo "generated $out (plain ${K}x${N})"
}

gen_ffn() {
  local K1="$1" N1="$2" K2="$3" N2="$4" MR="${5:-1}"
  local out
  if [[ -n "${6:-}" ]]; then out="$6"; elif [[ "$MR" == 1 ]]; then out="insts_ffn.txt"; else out="insts_ffn_m64.txt"; fi
  python3 "$dir/n32_core_placed.py" -m 16 -k 64 -n 128 -M 16 -K "$K1" -N "$N1" --mode ffn \
    --k1 "$K1" --n1 "$N1" --k2 "$K2" --n2 "$N2" --mrows "$MR" >"$work/a.mlir"
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

gen_attn() {
  local MC="$1"
  python3 "$dir/n32_core_placed.py" -m 16 -k 64 -n 128 -M 16 -K 2048 -N 2048 --mode attn --k1 "$MC" >"$work/a.mlir"
  aiecc --aietools="${AIETOOLS:?}" --alloc-scheme=basic-sequential --no-compile --no-link \
    --no-xchesscc --aie-generate-npu-insts --unified --dynamic-objFifos \
    --npu-insts-name="$work/i.txt" "$work/a.mlir" >/dev/null
  mv "$work/i.txt" "$dir/runtime/insts_attn_${MC}.txt"
  echo "generated insts_attn_${MC}.txt (${MC} chunks per column, $((MC * 128)) tokens)"
}

gen_attn_all() {
  for mc in 1 2 3 4 8 16 32 64 128 256 512; do gen_attn "$mc"; done
}

gen_argmax() {
  local K="$1"
  local N="$2"
  local out="${3:-insts_argmax_${K}x${N}.txt}"
  python3 "$dir/n32_core_placed.py" -m 16 -k 64 -n 128 -M 16 -K "$K" -N "$K" --mode argmax \
    --k2 "$K" --n2 "$N" >"$work/a.mlir"
  aiecc --aietools="${AIETOOLS:?}" --alloc-scheme=basic-sequential --no-compile --no-link     --no-xchesscc --aie-generate-npu-insts --unified --dynamic-objFifos     --npu-insts-name="$work/i.txt" "$work/a.mlir" >/dev/null
  mv "$work/i.txt" "$dir/runtime/$out"
  echo "generated $out (argmax ${K}x${N})"
}

case "${1:-default}" in
  argmax)
    gen_argmax "$2" "$3" "${4:-}"
    ;;
  attn)
    gen_attn "$2"
    ;;
  conv)
    gen_conv "$2" "$3" "${4:-}"
    ;;
  plain)
    gen_plain "$2" "$3" "${4:-1}" "${5:-}"
    ;;
  ffn)
    gen_ffn "$2" "$3" "$4" "$5" "${6:-1}" "${7:-}"
    ;;
  spec)
    # Minimal: emit LFM3 defaults; full spec-driven enumeration is in ModelSpec tooling.
    echo "spec-driven generation: using model spec program needs (LFM3 default set)"
    gen_ffn 2048 15360 7680 2048 1 insts_ffn.txt
    gen_conv 2048 6144
    gen_plain 2048 2048
    gen_plain 2048 3072
    gen_argmax 2048 128000
    gen_attn_all
    ;;
  default|*)
    # LFM3-1.7B (WP6): shapes explicit, no literals in generator.
    gen_ffn 2048 15360 7680 2048 1 insts_ffn.txt
    gen_conv 2048 6144
    gen_plain 2048 2048
    gen_plain 2048 3072
    gen_argmax 2048 128000
    gen_attn_all
    # WP10 m64 prefill variants (ffn split: gu-only + plain down).
    gen_ffn 2048 15360 0 0 4 insts_ffngu64.txt
    gen_plain 7680 2048 4
    gen_plain 2048 2048 4
    gen_plain 2048 3072 4
    ;;
esac
