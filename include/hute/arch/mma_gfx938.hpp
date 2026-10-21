/***************************************************************************************************
 * Copyright (c) 2026 Hygon Information Technology Co., Ltd. All rights reserved.
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hute/arch/mma.hpp>
#include <hute/numeric/numeric_types.hpp>

namespace hute {

  struct GFX938_16x32x32_F32F8F8F32E4M3E4M3_NT_LIT
  {
    using DRegisters = float[8];
    using ARegisters = float_e4m3_t[8];
    using BRegisters = intx2_t[2];
    using CRegisters = float[8];
    HUTE_HOST_DEVICE static void
    fma(float& d0, float& d1, float& d2, float& d3,
        float& d4, float& d5, float& d6, float& d7,
        float_e4m3_t const& a0, float_e4m3_t const& a1, float_e4m3_t const& a2, float_e4m3_t const& a3,
        float_e4m3_t const& a4, float_e4m3_t const& a5, float_e4m3_t const& a6, float_e4m3_t const& a7,
        intx2_t const& b0, intx2_t const& b1,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7)
    {
      #if (defined(__gfx938__)) && defined(__HIPCC__)
        v4f C0, C1;
        v4f D0, D1;
        C0.x = c0; C0.y = c1; C0.z = c2; C0.w = c3;
        C1.x = c4; C1.y = c5; C1.z = c6; C1.w = c7;
        hytlass::Array<float_e4m3_t, 8> a;
        a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3;
        a[4] = a4; a[5] = a5; a[6] = a6; a[7] = a7;
        intx2_t A;
        A = *(reinterpret_cast<intx2_t *>(&a));
        D0 = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(A, b0, C0, true, false);
        D1 = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(A, b1, C1, true, false);
        d0 = D0.x; d1 = D0.y; d2 = D0.z; d3 = D0.w;
        d4 = D1.x; d5 = D1.y; d6 = D1.z; d7 = D1.w;
      #elif defined(__gfx92a__) && defined(__HIPCC__)
        HUTE_INVALID_CONTROL_PATH("GFX938 LIT MMAC requires gfx938");
      #endif
    }
  };

  struct GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT
  {
    using DRegisters = float[16];
    using ARegisters = float[4];
    using BRegisters = intx2_t[8];
    using CRegisters = float[16];
    HUTE_HOST_DEVICE static void
    fma(float& d0,  float& d1,  float& d2,  float& d3,
        float& d4,  float& d5,  float& d6,  float& d7,
        float& d8,  float& d9,  float& d10, float& d11,
        float& d12, float& d13, float& d14, float& d15,
        float const& a0, float const& a1, float const& a2, float const& a3,
        intx2_t const& b0, intx2_t const& b1, intx2_t const& b2, intx2_t const& b3,
        intx2_t const& b4, intx2_t const& b5, intx2_t const& b6, intx2_t const& b7,
        float const& c0,  float const& c1,  float const& c2,  float const& c3,
        float const& c4,  float const& c5,  float const& c6,  float const& c7,
        float const& c8,  float const& c9,  float const& c10, float const& c11,
        float const& c12, float const& c13, float const& c14, float const& c15)
    {
      #if (defined(__gfx938__)) && defined(__HIPCC__)
        v4f C0, C1, C2, C3;
        v4f D0, D1, D2, D3;
        C0.x = c0;  C1.x = c1;  C0.y = c2;  C1.y = c3;
        C0.z = c4;  C1.z = c5;  C0.w = c6;  C1.w = c7;
        C2.x = c8;  C3.x = c9;  C2.y = c10; C3.y = c11;
        C2.z = c12; C3.z = c13; C2.w = c14; C3.w = c15;
        hytlass::Array<float, 4> A0, A1;
        A0[0] = a0; A0[1] = a1;
        A1[0] = a2; A1[1] = a3;
        intx2_t A_0, A_1;
        A_0 = *(reinterpret_cast<intx2_t *>(&A0));
        A_1 = *(reinterpret_cast<intx2_t *>(&A1));
        D0 = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(A_0, b0, C0, true, false);
        D1 = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(A_0, b2, C1, true, false);
        D2 = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(A_0, b4, C2, true, false);
        D3 = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(A_0, b6, C3, true, false);
        D0 = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(A_1, b1, D0, true, false);
        D1 = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(A_1, b3, D1, true, false);
        D2 = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(A_1, b5, D2, true, false);
        D3 = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(A_1, b7, D3, true, false);
        d0 = D0.x;  d1 = D1.x;  d2 = D0.y;  d3 = D1.y;
        d4 = D0.z;  d5 = D1.z;  d6 = D0.w;  d7 = D1.w;
        d8 = D2.x;  d9 = D3.x;  d10 = D2.y; d11 = D3.y;
        d12 = D2.z; d13 = D3.z; d14 = D2.w; d15 = D3.w;
      #elif defined(__gfx92a__) && defined(__HIPCC__)
        HUTE_INVALID_CONTROL_PATH("GFX938 LIT MMAC requires gfx938");
      #endif
    }
  };

} // namespace hute
