#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
extern "C" void __cxa_finalize(void *) {}
extern "C" void gdn_begin() {}
extern "C" void gdn_chunk(const uint8_t *in, uint8_t *out) {
  for (int t = 0; t < 4; ++t)
    for (int i = 0; i < 512; ++i) out[t * 512 + i] = in[t * 1024 + i];
}
