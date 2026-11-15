#!/bin/bash
# Runs gufo-gen chat + bench on every dense GGUF found in the HF cache (patterns below), logs to $OUT.
# usage: test_hf_cache.sh [OUT_DIR]   (default: /tmp/gen_hf_test)
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-/tmp/gen_hf_test}
CACHE=/mnt/raid0/user_cache/huggingface/hub
mkdir -p "$OUT"
MODELS=(
  "$CACHE"/models--unsloth--Qwen3-0.6B-GGUF/snapshots/*/Qwen3-0.6B-Q4_0.gguf
  "$CACHE"/models--unsloth--Qwen3-1.7B-GGUF/snapshots/*/Qwen3-1.7B-Q4_0.gguf
  "$CACHE"/models--bartowski--MiniCPM5-2B-GGUF/snapshots/*/MiniCPM5-2B-Q4_K_M.gguf
  ~/jev-model/MiniCPM5-1B-SFT-Q6_K-fixed.gguf
)
: >"$OUT/summary.txt"
for m in "${MODELS[@]}"; do
  [ -f "$m" ] || continue
  name=$(basename "$m" .gguf)
  echo "== $name" | tee -a "$OUT/summary.txt"
  OMP_NUM_THREADS=16 timeout 1800 "$HERE/run_gen.sh" chat "$m" "What is 17 times 3? Answer in one short sentence." 60 >"$OUT/$name.chat.txt" 2>&1
  echo "chat exit $?: $(grep -v '^gen:' "$OUT/$name.chat.txt" | head -3 | tr '\n' ' ')" | tee -a "$OUT/summary.txt"
  grep -E "^gen: [a-z0-9]+ layers|prefill|rror|not supported|unsupported" "$OUT/$name.chat.txt" | tee -a "$OUT/summary.txt"
  OMP_NUM_THREADS=16 timeout 1800 "$HERE/run_gen.sh" bench "$m" 1024 64 >"$OUT/$name.bench.txt" 2>&1
  OMP_NUM_THREADS=16 timeout 1800 "$HERE/run_gen.sh" bench "$m" 4096 32 >"$OUT/$name.bench4k.txt" 2>&1
  grep -E "prefill [0-9]" "$OUT/$name.bench4k.txt" | tee -a "$OUT/summary.txt"
  echo "bench exit $?" | tee -a "$OUT/summary.txt"
  grep -E "prefill|rror" "$OUT/$name.bench.txt" | tee -a "$OUT/summary.txt"
done
echo DONE | tee -a "$OUT/summary.txt"
