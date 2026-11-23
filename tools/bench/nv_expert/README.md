# NVFP4 routed expert GEMM harness

Standalone benchmark for the production `RoutedF16GEMMKernel` (Inkling NVFP4 experts). Generate
`kernel_extract.inc` from `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp` lines 5668-6772 plus the
header in this folder's history, then:

    hipcc -O2 --offload-arch=gfx1151 -DBMV=128 -DBNV=192 -o bench bench.cpp
    ./bench 2304 2304 save     # gate/up shape (K, M), stores the baseline output
    ./bench 2304 2304 cmp      # compares a modified kernel against it

`wmma_peak.cpp` measures the fp16 WMMA ceiling with real (non-constant) data: 41-46 TFLOPS, not 57.

Measured on Strix Halo (4096 tokens, top-8, 128 experts, K=M=2304): 11.8 ms, 29.5 TFLOPS useful. Ablations: removing the
global fetch alone -1.7 ms; removing fetch+commit (LDS writes) 8.6 ms (-27%); removing all WMMA 9.5 ms; no dequant -0.3 ms;
no activation fragment reads -0.2 ms; one barrier less -0.1 ms. BN 96..256 are within 5%, BM=256 slower, double-buffered
LDS slower (occupancy), waves_per_eu=7 +2% (removes a 32 B/lane spill). So the kernel is bound by its load/commit pipeline,
not by dequantization or matrix math.

Later findings (pair mode: add -DPAIRV; kernel_nvfp4_regdirect.inc = production register-direct variant):
weights-only fetch removal -2.2 ms, activations-only removal -0.3 ms, so the weight stream (not activations) is the exposed cost.
Prefetching weights 2 stages deep, or splitting the two 16-byte chunks across lane halves with a lane exchange: no gain / slower.
Timings in this harness are bimodal; compare min of 5 runs, interleaved.
