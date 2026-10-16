#pragma once

#include "simd_inst.h"

namespace V = simd_inst;
using namespace AscendC;

/*!
 * \brief ND→NZ scatter reorder in UB using vsstb (DATA_BLOCK_COPY).
 *
 * Reorders [ROWS, COLS] data from ND (row-major) to NZ (fractal) layout
 * within UB.  Cast is fused when SrcT ≠ DstT.
 *
 * The scatter stride is ROWS + 1: one extra row of padding avoids bank
 * conflicts in the UB tmp buffer.  The caller's post-copy (codegen side)
 * accounts for the global L1 NZ layout (full_rows, M-split offset, etc.).
 *
 * Key trick: vsstb writes 16-element (32B) blocks with blockStride in
 * the upper 16 bits of its int32_t stride parameter.  blockStride =
 * ROWS + 1 means blocks span (ROWS+1) × 32B = one NZ D-group stride
 * including the pad row.
 *
 * The standalone entry is __simd_vf__ (runs on AIV vector pipe).  Code that
 * is already inside a SimdVF calls the __simd_callee__ entry instead.  The
 * caller must issue asc_copy_ub2l1() separately after this returns.
 *
 * For single-fractal tiles (ROWS <= 16 && COLS <= 16), this is a no-op
 * since ND == NZ.  The caller should just use raw copy in that case.
 *
 * \tparam ROWS   Number of rows in this sub-tile (e.g. Br_HALF after M-split)
 * \tparam COLS   Number of columns
 * \tparam SrcT   Source element type (ND data)
 * \tparam DstT   Destination element type (NZ data); same as SrcT for pure
 *                reorder, different for fused cast
 */

// ── Primary template (fallback / unsupported) ────────────────────────

template <int ROWS, int COLS, typename SrcT, typename DstT, typename = void>
struct ascend_nd2nz_scatter_impl {
  __simd_callee__ static inline void apply(__ubuf__ SrcT *src_ub,
                                           __ubuf__ DstT *dst_nz_ub) {
    static_assert(sizeof(SrcT) == 0,
                  "unsupported (SrcT, DstT) pair or COLS alignment for "
                  "ascend_nd2nz_scatter");
  }
};

// ── (T, T): same-type scatter, no cast ──────────────────────────────

// E4M3 uses separate full/tail loops; other types retain the updating-mask
// loop.
template <int ROWS, int COLS, typename T>
struct ascend_nd2nz_scatter_impl<
    ROWS, COLS, T, T, std::enable_if_t<(COLS * sizeof(T)) % 32 == 0>> {
  __simd_callee__ static inline void apply(__ubuf__ T *src_ub,
                                           __ubuf__ T *dst_nz_ub) {
    static_assert(ROWS % 16 == 0, "ROWS must be a multiple of 16");
    constexpr int STRIDE = ((ROWS + 1) << 16) | 1;

    if constexpr (std::is_same<T, fp8_e4_t>::value) {
      auto *src = reinterpret_cast<__ubuf__ uint8_t *>(src_ub);
      auto *dst = reinterpret_cast<__ubuf__ uint8_t *>(dst_nz_ub);
      if constexpr (COLS >= 256) {
        const auto full_mask = asc_create_mask_b8(PAT_ALL);
        for (int pass = 0; pass < COLS / 256; ++pass) {
          auto *srcBase = src + pass * 256;
          auto *dp = dst + pass * (ROWS + 1) * 256;
          for (int r = 0; r < ROWS; ++r) {
            auto data = V::vlds_norm(srcBase + r * COLS, 0);
            dp = V::vsstb(data, dp, STRIDE, full_mask, POST_UPDATE);
          }
        }
      }

      if constexpr (COLS % 256 != 0) {
        constexpr int tail = COLS % 256;
        constexpr int boundary_rows = 1 + (255 - tail) / COLS;
        constexpr int safe_rows =
            tail == 32 || ROWS < boundary_rows ? 0 : ROWS - boundary_rows;
        if constexpr (safe_rows > 0) {
          uint32_t count = tail;
          const auto mask = asc_update_mask_b8(count);
          auto *srcBase = src + (COLS / 256) * 256;
          auto *dp = dst + (COLS / 256) * (ROWS + 1) * 256;
          // These complete vector loads stay within the source allocation.
          for (int r = 0; r < safe_rows; ++r) {
            auto data = V::vlds_norm(srcBase + r * COLS, 0);
            dp = V::vsstb(data, dp, STRIDE, mask, POST_UPDATE);
          }
        }
        uint32_t count = 32;
        const auto tail_mask = asc_update_mask_b8(count);
        // Read only complete data blocks in the tail; a store mask cannot
        // prevent a full-vector source load from crossing the allocation.
        for (int block = 0; block < (COLS % 256) / 32; ++block) {
          const int col = (COLS / 256) * 256 + block * 32;
          auto *sp = src + col;
          auto *dp = dst + col * (ROWS + 1) + safe_rows * 32;
          for (int r = safe_rows; r < ROWS; ++r) {
            auto data = V::vlds_brc_datablock(sp + r * COLS, 0);
            dp = V::vsstb(data, dp, STRIDE, tail_mask, POST_UPDATE);
          }
        }
      }
      return;
    }

    constexpr int VL_T = 256 / sizeof(T);

    uint32_t count = COLS;
    int pass = 0;
    while (count > 0) {
      vector_bool mask;
      if constexpr (std::is_same<T, fp8_e4_t>::value) {
        mask = asc_update_mask_b8(count);
      } else if constexpr (sizeof(T) == 2) {
        mask = asc_update_mask_b16(count);
      } else if constexpr (sizeof(T) == 4) {
        mask = asc_update_mask_b32(count);
      } else {
        static_assert(sizeof(T) == 0,
                      "unsupported element size for same-type scatter");
      }
      __ubuf__ T *dstPtr = dst_nz_ub + pass * (ROWS + 1) * VL_T;
      __ubuf__ T *srcBase = src_ub + pass * VL_T;
      for (int r = 0; r < ROWS; ++r) {
        auto reg = V::vlds_norm(srcBase + r * COLS, 0);
        dstPtr = V::vsstb(reg, dstPtr, STRIDE, mask, POST_UPDATE);
      }
      ++pass;
    }
  }
};

// ── (float, bfloat16_t): f32 → bf16 cast + scatter ──────────────────

// Full VL: COLS multiple of 64 (one f32 VL → 64 bf16)
template <int ROWS, int COLS>
struct ascend_nd2nz_scatter_impl<ROWS, COLS, float, bfloat16_t,
                                 std::enable_if_t<COLS % 64 == 0>> {
  __simd_callee__ static inline void apply(__ubuf__ float *src_ub,
                                           __ubuf__ bfloat16_t *dst_nz_ub) {
    static_assert(ROWS % 16 == 0, "ROWS must be a multiple of 16");

    constexpr int NUM_PASSES = COLS / 64;
    constexpr int STRIDE = ((ROWS + 1) << 16) | 1;

    auto f32Mask = asc_create_mask_b32(PAT_ALL);
    auto bf16Mask = asc_create_mask_b16(PAT_VL64);

    for (int pass = 0; pass < NUM_PASSES; ++pass) {
      __ubuf__ bfloat16_t *dstPtr = dst_nz_ub + pass * (ROWS + 1) * 64;
      __ubuf__ float *srcBase = src_ub + pass * 64;
      for (int r = 0; r < ROWS; ++r) {
        auto srcReg = V::vlds_norm(srcBase + r * COLS, 0);
        auto castReg = V::vcvt<bfloat16_t>(srcReg, f32Mask, ROUND_R, RS_DISABLE,
                                           PART_EVEN, MODE_ZEROING);
        auto packReg = V::vpack<uint16_t>((V::vec_t<uint32_t> &)castReg, LOWER);
        dstPtr = V::vsstb((V::vec_t<bfloat16_t> &)packReg, dstPtr, STRIDE,
                          bf16Mask, POST_UPDATE);
      }
    }
  }
};

// Partial: COLS not a multiple of 64, but still 32B-aligned (COLS % 8 == 0)
template <int ROWS, int COLS>
struct ascend_nd2nz_scatter_impl<
    ROWS, COLS, float, bfloat16_t,
    std::enable_if_t<(COLS % 64 != 0) && ((COLS * 4) % 32 == 0)>> {
  __simd_callee__ static inline void apply(__ubuf__ float *src_ub,
                                           __ubuf__ bfloat16_t *dst_nz_ub) {
    static_assert(ROWS % 16 == 0, "ROWS must be a multiple of 16");
    static_assert(COLS == -1, "partial-VL f32→bf16 scatter: TODO");
  }
};

// ── (bfloat16_t, float): bf16 → f32 cast + scatter ──────────────────

// Full VL: COLS multiple of 128 (one bf16 VL → 128 f32 = 2 f32 VLs)
template <int ROWS, int COLS>
struct ascend_nd2nz_scatter_impl<ROWS, COLS, bfloat16_t, float,
                                 std::enable_if_t<COLS % 128 == 0>> {
  __simd_callee__ static inline void apply(__ubuf__ bfloat16_t *src_ub,
                                           __ubuf__ float *dst_nz_ub) {
    static_assert(ROWS % 16 == 0, "ROWS must be a multiple of 16");

    constexpr int NUM_PASSES = COLS / 128;
    constexpr int STRIDE = ((ROWS + 1) << 16) | 1;

    auto bf16Mask = asc_create_mask_b16(PAT_ALL);
    auto f32Mask = asc_create_mask_b32(PAT_ALL);

    for (int pass = 0; pass < NUM_PASSES; ++pass) {
      __ubuf__ float *dstPtr0 = dst_nz_ub + pass * (ROWS + 1) * 128;
      __ubuf__ float *dstPtr1 = dstPtr0 + (ROWS + 1) * 64;

      for (int r = 0; r < ROWS; ++r) {
        auto src = V::vlds_norm(src_ub + r * COLS + pass * 128, 0);
        auto dst_lo = V::vcvt<float>(src, bf16Mask, PART_EVEN, MODE_ZEROING);
        auto dst_hi = V::vcvt<float>(src, bf16Mask, PART_ODD, MODE_ZEROING);
        auto [dst_half0, dst_half1] = V::vintlv(dst_lo, dst_hi);
        dstPtr0 = V::vsstb(dst_half0, dstPtr0, STRIDE, f32Mask, POST_UPDATE);
        dstPtr1 = V::vsstb(dst_half1, dstPtr1, STRIDE, f32Mask, POST_UPDATE);
      }
    }
  }
};

// Partial: COLS not a multiple of 128, but still 32B-aligned (COLS % 16 == 0)
template <int ROWS, int COLS>
struct ascend_nd2nz_scatter_impl<
    ROWS, COLS, bfloat16_t, float,
    std::enable_if_t<(COLS % 128 != 0) && ((COLS * 2) % 32 == 0)>> {
  __simd_callee__ static inline void apply(__ubuf__ bfloat16_t *src_ub,
                                           __ubuf__ float *dst_nz_ub) {
    static_assert(ROWS % 16 == 0, "ROWS must be a multiple of 16");
    static_assert(COLS == -1, "partial-VL bf16→f32 scatter: TODO");
  }
};

// ── Thin function wrappers ──────────────────────────────────────────

template <int ROWS, int COLS, typename SrcT, typename DstT>
__simd_callee__ inline void
ascend_nd2nz_scatter_callee(__ubuf__ SrcT *src_ub, __ubuf__ DstT *dst_nz_ub) {
  ascend_nd2nz_scatter_impl<ROWS, COLS, SrcT, DstT>::apply(src_ub, dst_nz_ub);
}

template <int ROWS, int COLS, typename SrcT, typename DstT>
__simd_vf__ inline void ascend_nd2nz_scatter(__ubuf__ SrcT *src_ub,
                                             __ubuf__ DstT *dst_nz_ub) {
  ascend_nd2nz_scatter_callee<ROWS, COLS, SrcT, DstT>(src_ub, dst_nz_ub);
}
