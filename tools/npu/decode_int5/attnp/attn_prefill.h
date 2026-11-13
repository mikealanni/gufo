// Prefill attention: every core owns one 64-row query block (16 tokens x 4 q heads) of one kv head and
// streams the whole causal prefix itself, so a pass covers 128 tokens and nothing is merged across columns.
// Per-core input stream (1536 B elements on the B fifo): 1 header+M, 6 Q, 9 tail (3 blocks x 3), then KV chunks.
#ifndef GUFO_ATTN_PREFILL_H_
#define GUFO_ATTN_PREFILL_H_

__attribute__((noinline)) static void AttnBeginP(const uint8_t *hdr) { AttnBegin(hdr, 0); }

__attribute__((noinline)) static void AttnQP(const uint8_t *pB, int idx) {
  CopyBytes((uint8_t *)a_q + idx * 1536, pB, 1536);
}

__attribute__((noinline)) static void AttnTailP(const uint8_t *pB, int idx) {
  const int tb = idx / 3, e = idx % 3;
  CopyBytes(a_chunk + e * 1536, pB, 1536);
  if (e == 2) AttnTail(tb);
}

__attribute__((noinline)) static void AttnFeedP(const uint8_t *pB) {
  CopyBytes(a_chunk + a_feed * 1536, pB, 1536);
  if (++a_feed < 3) return;
  a_feed = 0;
  const int tokbase = a_chunk_idx * 16;
  a_chunk_idx++;
  if (tokbase <= a_pos + a_n - 1) AttnChunk(tokbase);
}

#endif
