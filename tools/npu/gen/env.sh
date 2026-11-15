# Toolchain environment for generating NPU instruction streams (override with GUFO_NPU_ENV).
if [ -n "${GUFO_NPU_ENV:-}" ]; then source "$GUFO_NPU_ENV"; else
  source ~/npu-work/ex_env.sh >/dev/null 2>&1
  export AIETOOLS=$HOME/torch2aie/toolchain/aietools
fi
