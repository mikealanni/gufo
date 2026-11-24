// Prefill wrappers for the Inkling attention kernel. Per core the A fifo carries:
// element 0: header, 1..6: Q row-blocks, 7: basis rows (bf16), 8..(7+mc): shifted bias table slices (one per chunk),
// then the B fifo carries mc KV chunks of 4608 B. Output is 24 O tiles + 2 L tiles.
#ifndef GUFO_INKLING_ATTN_PREFILL_H_
#define GUFO_INKLING_ATTN_PREFILL_H_

__attribute__((noinline)) static void AttnBeginP(const uint8_t *hdr) { AttnBegin(hdr); }

__attribute__((noinline)) static void AttnQP(const uint8_t *pB, int idx) { AttnQ(pB, idx); }

__attribute__((noinline)) static void AttnRP(const uint8_t *pB) { AttnR(pB); }

__attribute__((noinline)) static void AttnBiasP(const uint8_t *pB, int idx) { AttnBias(pB, idx); }

__attribute__((noinline)) static void AttnOutP(uint8_t *pC, int idx) { AttnOut(pC, idx); }

#endif
