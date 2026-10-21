/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <hute/config.hpp>

#include <hute/arch/copy.hpp>

// Config

namespace hute
{
/*************************************************/ 
/*not completed,to do ...*/
/*************************************************/ 
template <class TS, class TD = TS>
struct GFX928_CP_GLOBAL_DIRECT_TO_LDS
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == sizeof(TD), "cp.async requires sizeof(src_value_type) == sizeof(dst_value_type)");
  static_assert(sizeof(TS) == 4 || sizeof(TS) == 8 || sizeof(TS) == 16, "cp.async sizeof(TS) is not supported");

  HUTE_HOST_DEVICE static void
  copy(TS const& gmem_src,
       TD      & smem_dst) {
  }
};


struct GFX928_DS_READ_DS_M32x8_B32
{
  using SRegisters = uint128_t[1];
  using DRegisters = float[4];
  
  HUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       float& dst0, 
       float& dst1, 
       float& dst2, 
       float& dst3)
  {
    #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
    v4f d;

    d = __builtin_hcu_ds_read_m32x8f32((float *)&smem_src, short(0));
    float * dst = reinterpret_cast<float *>(&d);
    dst0 = dst[0];
    dst1 = dst[1];
    dst2 = dst[2];
    dst3 = dst[3];
    #endif
  }
};


struct GFX928_DS_READ_DS_M32x16_B16
{
  using SRegisters = uint128_t[1];
  using DRegisters = half_t[8];

  HUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       half_t& dst0, half_t& dst1, half_t& dst2, half_t& dst3,
       half_t& dst4, half_t& dst5, half_t& dst6, half_t& dst7)
  {
    #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
    __fp16x8_t d;
    d = __builtin_hcu_ds_read_m32x16f16((__fp16*)&smem_src, short(0));
    half_t * dst = reinterpret_cast<half_t *>(&d);
    dst0 = dst[0];
    dst1 = dst[1];
    dst2 = dst[2];
    dst3 = dst[3];
    dst4 = dst[4];
    dst5 = dst[5];
    dst6 = dst[6];
    dst7 = dst[7];
    #endif
  }

};

struct GFX928_DS_READ_DS_M32x16_B16_ALT
{
  using SRegisters = uint128_t[1];
  using DRegisters = half_t[8];

  HUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       half_t& dst0, half_t& dst1, half_t& dst2, half_t& dst3,
       half_t& dst4, half_t& dst5, half_t& dst6, half_t& dst7)
  {
    #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
    __fp16x8_t d;
    d = __builtin_hcu_ds_read_m32x16f16_alt((__fp16*)&smem_src, short(0));
    half_t * dst = reinterpret_cast<half_t *>(&d);
    dst0 = dst[0];
    dst1 = dst[1];
    dst2 = dst[2];
    dst3 = dst[3];
    dst4 = dst[4];
    dst5 = dst[5];
    dst6 = dst[6];
    dst7 = dst[7];
    #endif
  }
};
/*************************************************/ 
struct GFX928_DS_READ_DS_M32x32_B8
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint8_t[16];

  HUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint8_t& dst00, uint8_t& dst01, uint8_t& dst02, uint8_t& dst03,
       uint8_t& dst04, uint8_t& dst05, uint8_t& dst06, uint8_t& dst07,
       uint8_t& dst08, uint8_t& dst09, uint8_t& dst10, uint8_t& dst11,
       uint8_t& dst12, uint8_t& dst13, uint8_t& dst14, uint8_t& dst15)
  {
    #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
    v4i d;
    d = __builtin_hcu_ds_read_m32x32u8((int*)&smem_src, short(0));
    uint8_t * dst = reinterpret_cast<uint8_t *>(&d);
    dst00 = dst[0];
    dst01 = dst[1];
    dst02 = dst[2];
    dst03 = dst[3];
    dst04 = dst[4];
    dst05 = dst[5];
    dst06 = dst[6];
    dst07 = dst[7];
    dst08 = dst[8];
    dst09 = dst[9];
    dst10 = dst[10];
    dst11 = dst[11];
    dst12 = dst[12];
    dst13 = dst[13];
    dst14 = dst[14];
    dst15 = dst[15];
    #endif
  }
};

template <class T>
HUTE_HOST_DEVICE
void
copy_ds_read_m(uint128_t const* const smem_ptr,
               T* rmem_ptr)
{
#if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
  if constexpr (std::is_same_v<T, float>) {
    GFX928_DS_READ_DS_M32x8_B32::copy(
      *smem_ptr,
      rmem_ptr[0], rmem_ptr[1], rmem_ptr[2], rmem_ptr[3]
    );
  } else if constexpr (std::is_same_v<T, half_t>) {
    GFX928_DS_READ_DS_M32x16_B16::copy(
      *smem_ptr,
      rmem_ptr[0], rmem_ptr[1], rmem_ptr[2], rmem_ptr[3],
      rmem_ptr[4], rmem_ptr[5], rmem_ptr[6], rmem_ptr[7]
    );
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    GFX928_DS_READ_DS_M32x32_B8::copy(
      *smem_ptr,
      rmem_ptr[0], rmem_ptr[1], rmem_ptr[2], rmem_ptr[3],
      rmem_ptr[4], rmem_ptr[5], rmem_ptr[6], rmem_ptr[7],
      rmem_ptr[8], rmem_ptr[9], rmem_ptr[10], rmem_ptr[11],
      rmem_ptr[12], rmem_ptr[13], rmem_ptr[14], rmem_ptr[15]
    );
  } else {
    static_assert(sizeof(T) == 0, "Unsupported type for DS_READ_M");
  }
#else
  HUTE_INVALID_CONTROL_PATH("Trying to use DS_READ_M without support");
#endif
}
/////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace hute
