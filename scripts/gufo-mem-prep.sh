#!/usr/bin/env bash
# Prepare host memory before gufo loads a model (run as root; see docs/SERVER.md or the
# install lines below). Keeps the kernel from compacting and swapping while the GPU
# allocations are made, and gives the load clean, unfragmented memory.
#
#   sudo install -m 755 scripts/gufo-mem-prep.sh /usr/local/sbin/gufo-mem-prep
#   echo "$USER ALL=(root) NOPASSWD: /usr/local/sbin/gufo-mem-prep" | sudo tee /etc/sudoers.d/gufo-mem
#
# Environment: GUFO_MEM_SWAP_MARGIN_MB (default 8192) is the free headroom that must remain
# before swap is emptied; GUFO_MEM_DROP_CACHE=0 keeps the page cache.
set -u
[ "$(id -u)" = 0 ] || { echo "gufo-mem-prep: run as root" >&2; exit 1; }

sysctl -qw vm.compaction_proactiveness=0 vm.extfrag_threshold=1000 \
  vm.compact_unevictable_allowed=0 vm.swappiness=10 vm.watermark_boost_factor=0
for f in enabled defrag; do
  echo never > "/sys/kernel/mm/transparent_hugepage/$f" 2> /dev/null || true
done
echo 0 > /sys/kernel/mm/transparent_hugepage/khugepaged/defrag 2> /dev/null || true

sync
[ "${GUFO_MEM_DROP_CACHE:-1}" = 0 ] || echo 3 > /proc/sys/vm/drop_caches
echo 1 > /proc/sys/vm/compact_memory

# Empty swap only when the pages fit back into RAM with a safety margin.
swap_used=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print int((t-f)/1024)}' /proc/meminfo)
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
margin=${GUFO_MEM_SWAP_MARGIN_MB:-8192}
if [ "$swap_used" -gt 0 ] && [ $((avail - swap_used)) -ge "$margin" ]; then
  swapoff -a && swapon -a
fi

free_mb=$(awk '/MemFree/{print int($2/1024)}' /proc/meminfo)
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
echo "gufo-mem-prep: free ${free_mb} MB, available ${avail} MB, swap used ${swap_used} MB before"
