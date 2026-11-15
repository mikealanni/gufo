#!/bin/bash
# usage: sudo tools/npu/fclk-sweep.sh   (needs root; restores fclk level 7 on exit)
# Measures NPU weight-stream bandwidth (ffnbench) at fixed fabric clock levels, one level per run, NPU idle between changes.
set -u
USER_NAME=${SUDO_USER:-majed}
ROOT=/home/$USER_NAME/gufo-wp10
for dev in /sys/class/drm/card*/device; do [ -e "$dev/pp_dpm_fclk" ] && break; done
restore() { echo manual >"$dev/power_dpm_force_performance_level"; echo 7 >"$dev/pp_dpm_fclk"; echo "restored fclk 7"; }
trap restore EXIT
echo manual >"$dev/power_dpm_force_performance_level"
for lvl in 7 6 5 4 3; do
  echo "$lvl" >"$dev/pp_dpm_fclk"
  sleep 2
  echo "== fclk level $lvl: $(grep '\*' "$dev/pp_dpm_fclk")"
  runuser -u "$USER_NAME" -- bash -c "cd $ROOT && GOMP_SPINCOUNT=1000 OMP_NUM_THREADS=2 /home/$USER_NAME/npu-lock.sh build/npu/gufo-gen ffnbench tools/npu/decode_int5/runtime 5120 17408 q4 2>&1 | tail -4"
  sleep 2
done
