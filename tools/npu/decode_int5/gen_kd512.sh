#!/bin/bash
# 512-row instruction streams for the decision-model (judge) passes. Same kd xclbin as the 1024-row prefill
# programs, generated from the shared generator (cfg35_norm). Not generated: conv (its core program consumes two
# 512-row groups, so the 1024-row conv stream is reused on a 1024-row sized wire) and swi_b at 7168 columns (the fused
# swiglu needs an even number of column groups, so swi_b covers 8192 columns from offset 56).
cd ~/npu-work/ex/gemm_asymmetric_tile_buffering/cfg35_norm
source ~/npu-work/ex_env.sh >/dev/null 2>&1
export AIETOOLS=$HOME/torch2aie/toolchain/aietools
OUT=${1:-$HOME/gufo-wp10/tools/npu/decode_int5/runtime}
mkdir -p stage512 && cd stage512
gen() { local out=$1; shift
  python3 ../n32_core_placed.py -m 128 -k 64 -n 128 "$@" > $out.mlir || return 1
  aiecc --aietools=$AIETOOLS --alloc-scheme=basic-sequential --no-compile --no-link --no-xchesscc --aie-generate-npu-insts --unified --dynamic-objFifos --npu-insts-name=$out.txt $out.mlir >/dev/null 2>$out.err && cp $out.txt $OUT/ && echo "ok $out"; }
for s in "2048 2048" "2048 3072" "2048 6144" "7680 2048"; do set -- $s; gen insts_kd_512x$1x$2 -M 512 -K $1 -N $2 & done
gen insts_kd_swi_a_512x2048x8192 -M 512 -K 2048 -N 8192 --mode swigu --noff 0 &
gen insts_kd_swi_b_512x2048x8192 -M 512 -K 2048 -N 8192 --mode swigu --noff 56 &
gen insts_kd_dn2_512x7680x2048 -M 512 -K 7680 -N 2048 --a2 1 &
wait
