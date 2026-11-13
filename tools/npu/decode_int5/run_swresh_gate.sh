#!/usr/bin/env bash
set -euo pipefail

N="${1:-6}"
PASS8=0
FAIL8=0
PASS10=0
FAIL10=0

run_once() {
  local mode="$1"
  local swi="$2"
  local env_extra="${3:-}"
  local output
  output=$(cd ~/gufo-wp10 && env \
    LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/home/majed/npu-work/boostshim \
    OMP_NUM_THREADS=8 \
    GUFO_PREFILL_SWI="$swi" \
    $env_extra \
    ~/npu-lock.sh timeout 300 ./build/sbtest-prefill tools/npu/decode_int5/prefill_probe 1024 1 2>&1) || true
  echo "$output"
}

check_mode8() {
  local output="$1"
  echo "$output" | grep -q "reshape BYTES.*PASS" && \
  echo "$output" | grep -q "verify dn.*PASS"
}

check_mode10() {
  echo "$1" | grep -q "verify dn(swresh-A).*PASS" && echo "$2" | grep -q "zero tiles: 0 "
}

for i in $(seq 1 "$N"); do
  echo "=== Run $i/$N ==="

  out8=$(run_once 8 8 "")
  if check_mode8 "$out8"; then
    echo "mode 8: PASS"
    PASS8=$((PASS8 + 1))
  else
    echo "mode 8: FAIL"
    FAIL8=$((FAIL8 + 1))
  fi

  out10=$(run_once 10 10 "")
  out10z=$(run_once 10 10 "SWI31_ZMAP=1")
  if check_mode10 "$out10" "$out10z"; then
    echo "mode 10: PASS"
    PASS10=$((PASS10 + 1))
  else
    echo "mode 10: FAIL"
    FAIL10=$((FAIL10 + 1))
  fi
done

echo ""
echo "Summary: mode 8: ${PASS8}/${N} PASS, mode 10: ${PASS10}/${N} PASS"

if [ "$FAIL8" -gt 0 ] || [ "$FAIL10" -gt 0 ]; then
  exit 1
fi
