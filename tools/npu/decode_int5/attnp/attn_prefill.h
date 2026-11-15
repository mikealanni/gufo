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
  const int slot = a_chunk_idx++;
  if (a_win) {
    const int d = a_tab[slot];
    if (d >= 0) AttnChunk((a_qhi - d) * 16);
    return;
  }
  const int tokbase = slot * 16;
  if (tokbase <= a_pos + a_n - 1) AttnChunk(tokbase);
}

// Whole 4608 B chunk element (K then V^T) consumed in place from the broadcast fifo.
__attribute__((noinline)) static void AttnFeedP2(const uint8_t *pB) {
  const int slot = a_chunk_idx++;
  int tokbase;
  if (a_win) {
    const int d = a_tab[slot];
    if (d < 0) return;
    tokbase = (a_qhi - d) * 16;
  } else {
    tokbase = slot * 16;
    if (tokbase > a_pos + a_n - 1) return;
  }
  a_kvp = pB;
  AttnChunk(tokbase);
  a_kvp = a_chunk;
}

#endif
