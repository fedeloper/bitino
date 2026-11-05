// int8_avx2 VPDPBUSD GEMM micro-kernel (AVX-VNNI, VEX-encoded 256-bit;
// compiled with -mavx2 -mavxvnni). Exactness of the u8 x s8 product with the
// +128 activation shift: see int8_avx2.h.
#include "kernels/int8_avx2/int8_avx2.h"

#if !defined(__AVX2__) || !defined(__AVXVNNI__)
#error "int8_gemm_avxvnni.cc must be compiled with -mavx2 -mavxvnni (see CMakeLists.txt)"
#endif

#include <immintrin.h>

#include "kernels/int8_avx2/int8_avx2_tile.h"

namespace bitino {
namespace int8_avx2 {
namespace {

struct VnniDot {
  // a: 4 bytes of qx + 128 (u8) broadcast; w: lane j = W[j][4c..4c+3] (s8).
  // VPDPBUSD (not VPDPBUSDS): the int32 accumulation wraps, see int8_avx2.h.
  static __m256i dot(__m256i acc, __m256i a, __m256i w) { return _mm256_dpbusd_avx_epi32(acc, a, w); }
};

}  // namespace

void gemm_vnni(const GemmArgs& g, int64_t r0, int64_t rows, int64_t p0, int64_t p1) { gemm<VnniDot>(g, r0, rows, p0, p1); }

}  // namespace int8_avx2
}  // namespace bitino
