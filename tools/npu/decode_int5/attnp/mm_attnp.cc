#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

extern "C" void __cxa_finalize(void *) {}

alignas(128) uint8_t arena[42496];

#include "attn_kernel.h"
#include "attn_prefill.h"

extern "C" {
void attn_begin_p(uint8_t *hdr) { AttnBeginP(hdr); }
void attn_q_p(uint8_t *pB, int32_t idx) { AttnQP(pB, idx); }
void attn_tail_p(uint8_t *pB, int32_t idx) { AttnTailP(pB, idx); }
void attn_feed_p(uint8_t *pB) { AttnFeedP(pB); }
void attn_out(uint8_t *pC, int32_t idx) { AttnOut(pC, idx); }
}
