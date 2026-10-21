#include "hip/hip_runtime.h"
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

#include <hute/arch/mma.hpp>

namespace hute
{

struct GFX926_16x16x4_F32F32F32F32_NT
{
  using DRegisters = float[4];
  using ARegisters = float[1];
  using BRegisters = float[1];
  using CRegisters = float[4];

  // Register asm fma
  HUTE_HOST_DEVICE static void
  fma(float      & d0, float      & d1, float      & d2, float      & d3, 
      float const& a0,
      float const& b0,
      float const& c0, float const& c1, float const& c2, float const& c3)
  {
#if defined(__gfx926__) && defined(__HIPCC__)
    v4f c;
    c.x = c0;
    c.y = c1;
    c.z = c2;
    c.w = c3;
    v4f d;
    asm volatile("v_mmac_16x16x4_f32 %0, %1, %2, %3, vstep:0\n\t"
                      : "+v"(d)
                      : "v"(a0), "v"(b0), "v"(c));

    d0 = d.x;
    d1 = d.y;
    d2 = d.z;
    d3 = d.w;
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct GFX926_32x32x4_F32F32F32F32_NT
{
  using DRegisters = float[2];
  using ARegisters = float[2];
  using BRegisters = float[2];
  using CRegisters = float[2];

  // Register asm fma
  HUTE_HOST_DEVICE static void
  fma(float      & d0, float      & d1,
      float const& a0, float const& a1,
      float const& b0, float const& b1,
      float const& c0, float const& c1)
  {

  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct GFX926_64x64x4_F32F32F32F32_NT
{
  using DRegisters = float[4];
  using ARegisters = float[4];
  using BRegisters = float[4];
  using CRegisters = float[4];

  // Register asm fma
  HUTE_HOST_DEVICE static void
  fma(float      & d0, float      & d1, float      & d2, float      & d3,
      float const& a0, float const& a1, float const& a2, float const& a3,
      float const& b0, float const& b1, float const& b2, float const& b3,
      float const& c0, float const& c1, float const& c2, float const& c3)
  {

  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace hute
