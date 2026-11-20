#!/usr/bin/env bash
# One-time setup of the PyTorch venv for DFlash2 drafter distillation.
#
#   ./setup-train-env.sh           # installs torch (CPU) + deps into ~/venvs/dflash
#
# NOTE: torch training runs on **CPU**. The ROCm/gfx1151 torch wheels (incl. the
# 2.12 rocm7.14 preview) segfault on any autograd `.backward()`, so we install a
# clean CPU-only torch. GPU capture is gufo's own HIP C++ and needs no torch.
# torchvision is pinned to the CPU wheel that matches torch (avoids the
# "operator torchvision::nms does not exist" mismatch). Requires internet, ~3 GB.
# Safe to re-run; idempotent. Uses uv if present, else python venv + pip.
set -euo pipefail

VENV=${VENV:-$HOME/venvs/dflash}
PY=${PY:-python3.13}
command -v "$PY" >/dev/null || PY=python3
command -v "$PY" >/dev/null || { echo "no python3" >&2; exit 1; }
VPY="$VENV/bin/python"

if command -v uv >/dev/null; then
  rm -rf "$VENV"
  uv venv "$VENV" --python "$PY"
  pip_install() { uv pip install --python "$VPY" "$@"; }
else
  "$PY" -m venv "$VENV"
  "$VPY" -m pip install --upgrade pip
  pip_install() { "$VPY" -m pip install "$@"; }
fi

CPU_INDEX="https://download.pytorch.org/whl/cpu"
echo "==> installing CPU torch + matching torchvision from $CPU_INDEX"
# uv resolves a torch<->torchvision-compatible pair from the CPU index.
pip_install torch torchvision --index-url "$CPU_INDEX" 2>&1 | tail -3

pip_install safetensors numpy gguf 'transformers>=4.47' 2>&1 | tail -2
echo "--- verify (forward + backward must work) ---"
"$VPY" - <<'PY'
import torch, safetensors, numpy, transformers  # noqa
x = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(8, 6, dtype=torch.bfloat16, requires_grad=True)
(x @ w).sum().backward()                       # proves autograd is healthy
assert w.grad is not None
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
      "backward OK", "| transformers", transformers.__version__)
print("NOTE: torch training runs on CPU (DEVICE=cpu). GPU autograd is broken on gfx1151.")
PY
echo "done. Train venv: $VPY"
