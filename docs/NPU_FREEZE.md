# NPU Integration Freeze — Root Cause Summary

Read-only investigation, 2026-10-10. Applies to the XDNA2 NPU offload on
`feat/npu-dense` (`src/models/qwen38_flash_next/kernels/rocm/npu_gemm.cpp`)
and the research stack on `origin/perf/qwen27b-dflash2-npu`.

## Verdict

The freeze is **not a gufo bug**. It is an AMD Strix Halo **Data Fabric**
firmware/AGESA defect that triggers when the ROCm iGPU and the XDNA2 NPU run
concurrently. gufo's design sits on the worst point of that failure envelope,
so it reproduces reliably.

## Chain of failure

1. `amdxdna` runtime-PM **suspends the NPU after ~5 s idle**.
2. The next request's first `Submit` resumes it: firmware, SMU, PSP and
   mailbox re-initialize **while the iGPU is running a wide dense GEMM**.
3. Two masters hammer the Data Fabric during a fabric/power transition.
4. Fabric logs a **deferred machine check, bank MC28**, syndrome
   `0x5b000000` (`Ext Err Code: 4`, `mem-tx: GEN`, `part-proc: SRC`).
5. Errors escalate (some with `Overflow`) until IO stops:
   `Previous system reset reason [0x08000800]: an uncorrected error caused a
   data fabric sync flood event`.
6. Hard host freeze, no panic, power cycle required.

This is why commit `93af3ab` reads *"parked: machine check on **second**
request"*: the first request warms the NPU while the GPU is still light; the
idle gap suspends it; the second request resumes it under full GPU load.

## External evidence (October 2026)

- **ROCm/TheRock#8709** — open, label `status: assessed`. An AMD engineer
  (harkgill-amd, 2026-10-06) reproduced the NPU+GPU Data Fabric failures and
  wrote that updating AGESA StrixHaloPI-FP11 from 1.0.0.2 to 1.0.0.2b resolved
  it on their setup (SMU 100.9.0). Later reports there do **not** confirm it on
  Linux: hangs persist on AGESA 1.0.0.2b with SMU 100.7.0 (ASUS ROG Flow Z13,
  Bosgame M5), AMD asked whether the PMFW version matters, and no fix is
  confirmed. Holding FCLK also failed on a Beelink GTR9 Pro and a CachyOS host.
  A BIOS "Eco Mode = Disabled" workaround was reported on the GTR9 Pro. Newer
  kernels and linux-firmware have not helped.
- **amd/xdna-driver#1605** — same scenario, worse outcome: *"Data Fabric
  deferred-error storm + MC28 machine check + hard system hang (power cycle
  required)"*, reproduced on Ubuntu 7.0.0-38 **and mainline 7.2.6**. Notes
  *"the first NPU request after an engine start seems to be a reliable
  trigger while the GPU is busy"* and *"shrinking the load does not help"*.
- Field data: hang in **2 s** with modest combined load; time-to-hang
  **shrinks** as GPU sclk/power rises. NPU alone and GPU alone are both clean.

## Mitigations, and why none are trustworthy yet

| Mitigation | Status |
| --- | --- |
| BIOS/AGESA ≥ 1.0.0.2b | AMD reports it fixes their reproducer, but Linux hosts on 1.0.0.2b still hung (SMU 100.7.0). Not a guarantee; vendor rollouts lag. |
| Hold `pp_dpm_fclk` at top | Works on some hosts; **fails on others** (hang with FCLK held, 3 s at 140 W). |
| Higher GPU clocks | Makes it **worse**. |
| **Strict serialization — no GPU/NPU overlap** | Only field-proven software mitigation. 5 h clean where FCLK-hold failed. |
| Newer kernel | Does **not** help. |

## Why gufo amplifies it

The root cause is platform-level, but these choices raise trigger probability
or blast radius:

1. **Maximal overlap by design.** `LinearAttention()` calls `Submit()` and
   then launches `DenseF16SsmGemm` on the same stream — the NPU DMA/compute
   runs exactly while the iGPU is busiest. This is the worst-case workload.
2. **Keepalive thread (`Heartbeat`, 1 s).** Injects a real NPU job every
   second for the object's lifetime — including during GPU-only decode —
   turning a burst hazard into a permanent one. It exists because of the
   suspend/wake race, but it does not fix the fabric trigger.
3. **`GUFO_NPU_ISOLATE=1` is counterproductive.** Adds 3 helper threads of
   full-buffer `memcpy` (A ≈ 11 MiB, C ≈ 4.5 MiB each) through the *same*
   fabric. More traffic on a traffic-threshold bug.
4. **Unbounded `xrt::run::wait()` inside a HIP host callback.** The driver's
   own firmware timeout is ~94 s; during that window the GPU stream is
   stalled too. `docs/NPU_BACKEND.md` requires finite timeouts and
   quarantine-before-reset — the implementation has neither.
5. **Unvalidated memory path.** `hipHostMalloc` pages are imported into XRT
   as a userptr (`XRT_BO_FLAGS_HOST_ONLY`). `docs/NPU_RESEARCH.md` flagged
   this exact XRT/`amdxdna` ↔ HIP combination as unproven; `amd/IRON#103`
   shows `aie2_map_host_buf` failing on this firmware generation.
6. **No platform gate.** `NpuGemm::Create()` never checks AGESA, FCLK policy,
   or GPU activity. The branch does not meet its own promotion policy in
   `docs/NPU_RESEARCH.md` (sustained concurrency soak, transactional
   GPU-only fallback).

## Consequences

- Keep `GUFO_ENABLE_NPU` **OFF** by default; treat `feat/npu-dense` as parked.
- If re-testing on unfixed AGESA: serialize GPU and NPU (never both in
  flight), drop the keepalive, force `GUFO_NPU_ISOLATE=0`, bound every
  `run.wait()`, and watch `journalctl -kf | grep -i "hardware error"` plus
  `dmesg | grep "Previous system reset reason"` — `[0x08000800]` is the
  DF sync-flood signature.
- Re-enable only with serialization or a verified-clean soak on the target host (AGESA ≥ 1.0.0.2b alone is not sufficient), plus the soak and
  fallback gates already written in `docs/NPU_RESEARCH.md`.

## References

- https://github.com/ROCm/TheRock/issues/8709
- https://github.com/amd/xdna-driver/issues/1605
- https://github.com/amd/IRON/issues/103
- `docs/NPU_BACKEND.md`, `docs/NPU_RESEARCH.md` (on `origin/perf/qwen27b-dflash2-npu`)
- Commits `93af3ab`, `24a6406`, `4cde9a9` (on `feat/npu-dense`)
