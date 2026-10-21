/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
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

#include "hytlass/arch/mmac_gfx9.h"

#include <hute/config.hpp>

#include <hute/arch/mma.hpp>


namespace hute
{
/////////////////////////////////////v_mmac_f32_f32/////////////////////////////////////
   struct GFX928_16x16x8_F32F32F32F32_NT
  {
    using DRegisters = float[4];
    using ARegisters = float[2];
    using BRegisters = float[2];
    using CRegisters = float[4];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3, 
        float const& a0, float const& a1,
        float const& b0, float const& b1,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
      // printf("a:%f %f b:%f %f\n",a0,a1,b0,b1);
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
          v4f c;
          v4f d;
          c.x = c0;
          c.y = c1;
          c.z = c2;
          c.w = c3;
          v2f a;
          v2f b;
          a.x = a0;
          a.y = a1;
          b.x = b0;
          b.y = b1;

          d = hytlass::arch::detail::mmac_f32_16x16x8_f32(a, b, c);

          d0 = d.x;
          d1 = d.y;
          d2 = d.z;
          d3 = d.w;
      #endif
    }
  };

/////////////////////////////////////v_mmac_f32_tf32/////////////////////////////////////
  struct GFX928_16x16x8_F32TF32TF32F32_NT
  {
    using DRegisters = float[4];
    using ARegisters = float[2];
    using BRegisters = float[2];
    using CRegisters = float[4];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3, 
        float const& a0, float const& a1,
        float const& b0, float const& b1,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
          v4f c;
          v4f d;
          c.x = c0;
          c.y = c1;
          c.z = c2;
          c.w = c3;

          v2f a;
          v2f b;
          a.x = a0;
          a.y = a1;
          b.x = b0;
          b.y = b1;

          d = hytlass::arch::detail::mmac_f32_16x16x8_tf32(a, b, c);

          d0 = d.x;
          d1 = d.y;
          d2 = d.z;
          d3 = d.w;
      #endif
    }
  };

/////////////////////////////////////v_mmac_f32_f16/////////////////////////////////////
  struct GFX928_16x16x16_F32F16F16F32_NT
  {
    using DRegisters = float[4];
    using ARegisters = half_t[4];
    using BRegisters = half_t[4];
    using CRegisters = float[4];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3, 
        half_t const& a0, half_t const& a1,half_t const& a2, half_t const& a3,
        half_t const& b0, half_t const& b1,half_t const& b2, half_t const& b3,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
        v4f c;
        v4f d;
        c.x = c0;
        c.y = c1;
        c.z = c2;
        c.w = c3;



        __fp16x4_t A,B;
        A.x = a0; A.y = a1; A.z = a2; A.w = a3;
        B.x = b0; B.y = b1; B.z = b2; B.w = b3;
        d = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B,c);

        d0 = d.x;
        d1 = d.y;
        d2 = d.z;
        d3 = d.w;
      #endif
    }
  
  };

  struct GFX928_16x16x16_F32F16F16F32_NT_FOR_GEMM1
  {
    using DRegisters = float[4];
    using ARegisters = half_t[4];
    using BRegisters = half_t[4];
    using CRegisters = float[4];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3, 
        half_t const& a0, half_t const& a1,half_t const& a2, half_t const& a3,
        half_t const& b0, half_t const& b1,half_t const& b2, half_t const& b3,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
        v4f c;
        v4f d;
        c.x = c0;
        c.y = c1;
        c.z = c2;
        c.w = c3;

        __fp16x4_t A,B;
        A.x = a0; A.y = a1; A.z = a2; A.w = a3;
        B.x = b0; B.y = b1; B.z = b2; B.w = b3;
        d = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B,c);

        d0 = d.x;
        d1 = d.y;
        d2 = d.z;
        d3 = d.w;
      #endif
    }
  
  };

  struct GFX928_16x32x16_F32F16F16F32_NT
  {
    using DRegisters = float[8];
    using ARegisters = half_t[4];
    using BRegisters = half_t[8];
    using CRegisters = float[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        float      & d4, float      & d5, float      & d6, float      & d7,
        half_t const& a0, half_t const& a1,half_t const& a2, half_t const& a3,
        half_t const& b0, half_t const& b1,half_t const& b2, half_t const& b3,
        half_t const& b4, half_t const& b5,half_t const& b6, half_t const& b7,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      v4f C0,C1;
      v4f D0,D1;

      C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
      C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;

      __fp16x4_t A, B, B0;

      A.x  = a0; A.y  = a1; A.z  = a2; A.w  = a3;
      B.x  = b0; B.y  = b1; B.z  = b2; B.w  = b3;
      B0.x = b4; B0.y = b5; B0.z = b6; B0.w = b7;

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B,C0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B0,C1);

      d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
      d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
  #endif
    }
  };

  struct GFX928_32x16x16_F32F16F16F32_NT
  {
    using DRegisters = float[8];
    using ARegisters = half_t[8];
    using BRegisters = half_t[4];
    using CRegisters = float[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        float      & d4, float      & d5, float      & d6, float      & d7,
        half_t const& a0, half_t const& a1,half_t const& a2, half_t const& a3,
        half_t const& a4, half_t const& a5,half_t const& a6, half_t const& a7,
        half_t const& b0, half_t const& b1,half_t const& b2, half_t const& b3,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
        v4f C0,C1;
        v4f D0,D1;

        C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
        C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;

        __fp16x4_t A,A0,B;

        A.x  = a0; A.y  = a1; A.z  = a2; A.w  = a3;
        A0.x = a4; A0.y = a5; A0.z = a6; A0.w = a7;
        B.x  = b0; B.y  = b1; B.z  = b2; B.w  = b3;

        D0 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B,C0);
        D1 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A0,B,C1);

        d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
        d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
      #endif
    }
  };

  struct GFX928_32x32x16_F32F16F16F32_NT
  {
    using DRegisters = float[16];
    using ARegisters = half_t[8];
    using BRegisters = half_t[8];
    using CRegisters = float[16];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        float      & d4, float      & d5, float      & d6, float      & d7,
        float      & d8, float      & d9, float      & d10, float      & d11,
        float      & d12, float      & d13, float      & d14, float      & d15,
        half_t const& a0, half_t const& a1,half_t const& a2, half_t const& a3,
        half_t const& a4, half_t const& a5,half_t const& a6, half_t const& a7,
        half_t const& b0, half_t const& b1,half_t const& b2, half_t const& b3,
        half_t const& b4, half_t const& b5,half_t const& b6, half_t const& b7,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7,
        float const& c8, float const& c9, float const& c10, float const& c11,
        float const& c12, float const& c13, float const& c14, float const& c15)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      v4f C0,C1,C2,C3;
      v4f D0,D1,D2,D3;

      C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
      C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;
      C2.x = c8;  C2.y = c9;  C2.z = c10; C2.w = c11;
      C3.x = c12; C3.y = c13; C3.z = c14; C3.w = c15;

      __fp16x4_t A,B,A0,B0;

      A.x  = a0; A.y  = a1; A.z  = a2; A.w  = a3;
      A0.x = a4; A0.y = a5; A0.z = a6; A0.w = a7;
      B.x  = b0; B.y  = b1; B.z  = b2; B.w  = b3;
      B0.x = b4; B0.y = b5; B0.z = b6; B0.w = b7;

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B,C0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B0,C1);
      D2 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A0,B,C2);
      D3 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A0,B0,C3);

      d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
      d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
      d8 = D2.x;  d9 = D2.y;    d10 = D2.z; d11 = D2.w;
      d12 = D3.x; d13 = D3.y;   d14 = D3.z; d15 = D3.w;
      #endif
    }
  };

  struct GFX928_32x32x16_F32F16F16F32_NT_ALT
  {
    using DRegisters = float[16];
    using ARegisters = half_t[8];
    using BRegisters = half_t[8];
    using CRegisters = float[16];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3, 
        float      & d4, float      & d5, float      & d6, float      & d7, 
        float      & d8, float      & d9, float      & d10, float      & d11, 
        float      & d12, float      & d13, float      & d14, float      & d15, 
        half_t const& a0, half_t const& a1,half_t const& a2, half_t const& a3,
        half_t const& a4, half_t const& a5,half_t const& a6, half_t const& a7,
        half_t const& b0, half_t const& b1,half_t const& b2, half_t const& b3,
        half_t const& b4, half_t const& b5,half_t const& b6, half_t const& b7,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7,
        float const& c8, float const& c9, float const& c10, float const& c11,
        float const& c12, float const& c13, float const& c14, float const& c15)
    {

      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)

        v4f C0,C1,C2,C3;
        v4f D0,D1,D2,D3;

        C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
        C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;
        C2.x = c8;  C2.y = c9;  C2.z = c10; C2.w = c11;
        C3.x = c12; C3.y = c13; C3.z = c14; C3.w = c15;

        __fp16x4_t A,B,A0,B0;

        A.x  = a0; A.y  = a1; A.z  = a2; A.w  = a3;
        A0.x = a4; A0.y = a5; A0.z = a6; A0.w = a7;
        B.x  = b0; B.y  = b1; B.z  = b2; B.w  = b3;
        B0.x = b4; B0.y = b5; B0.z = b6; B0.w = b7;
        
        D0 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B,C0);
        D1 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B0,C1);
        D2 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A0,B,C2);
        D3 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A0,B0,C3);
       
        // asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %3\n\t"
        //           : "+v"(D0)
        //           : "v"(A), "v"(B), "v"(C0));
        // asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %3\n\t"
        //           : "+v"(D1)
        //           : "v"(A), "v"(B0), "v"(C1));
        // asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %3\n\t"
        //           : "+v"(D2)
        //           : "v"(A0), "v"(B), "v"(C2));
        // asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %3\n\t"
        //           : "+v"(D3)
        //           : "v"(A0), "v"(B0), "v"(C3));

        d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
        d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
        d8 = D2.x;  d9 = D2.y;    d10 = D2.z; d11 = D2.w;
        d12 = D3.x; d13 = D3.y;   d14 = D3.z; d15 = D3.w;


        #endif
    
    }
  
  };

//////////////////////////// v_mmac_16x16x16 concatenate to depthx32 /////////////////////////////////////////////
  struct GFX928_16x16x32_F32F16F16F32_NT {
    using DRegisters = float[4];
    using ARegisters = half_t[8];
    using BRegisters = half_t[8];
    using CRegisters = float[4];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float &d0, float &d1, float &d2, float &d3,
        half_t const &a0, half_t const &a1, half_t const &a2, half_t const &a3,
        half_t const &a4, half_t const &a5, half_t const &a6, half_t const &a7,
        half_t const &b0, half_t const &b1, half_t const &b2, half_t const &b3,
        half_t const &b4, half_t const &b5, half_t const &b6, half_t const &b7,
        float const &c0, float const &c1, float const &c2, float const &c3)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      v4f c;
      v4f d;
      c.x = c0;
      c.y = c1;
      c.z = c2;
      c.w = c3;

      __fp16x4_t A,B;
      A.x = a0; A.y = a1; A.z = a2; A.w = a3;
      B.x = b0; B.y = b1; B.z = b2; B.w = b3;
      d = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B,c);

      A.x = a4; A.y = a5; A.z = a6; A.w = a7;
      B.x = b4; B.y = b5; B.z = b6; B.w = b7;
      d = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B,d);

      d0 = d.x;
      d1 = d.y;
      d2 = d.z;
      d3 = d.w;
      #endif
    }
  };

  struct GFX928_16x32x32_F32F16F16F32_NT
  {
    using DRegisters = float[8];
    using ARegisters = half_t[8];
    using BRegisters = half_t[16];
    using CRegisters = float[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        float      & d4, float      & d5, float      & d6, float      & d7,
        half_t const& a0, half_t const& a1,half_t const& a2, half_t const& a3,
        half_t const& a4, half_t const& a5,half_t const& a6, half_t const& a7,
        half_t const& b0, half_t const& b1,half_t const& b2, half_t const& b3,
        half_t const& b4, half_t const& b5,half_t const& b6, half_t const& b7,
        half_t const& b8, half_t const& b9,half_t const& b10, half_t const& b11,
        half_t const& b12, half_t const& b13,half_t const& b14, half_t const& b15,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7)
    {
#if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      v4f C0,C1;
      v4f D0,D1;

      C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
      C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;

      __fp16x4_t A,B,B0;

      A.x  = a0; A.y  = a1; A.z  = a2; A.w  = a3;
      B.x  = b0; B.y  = b1; B.z  = b2; B.w  = b3;
      B0.x = b8; B0.y = b9; B0.z = b10; B0.w = b11;

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B,C0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B0,C1);

      A.x  = a4; A.y  = a5; A.z  = a6; A.w  = a7;
      B.x  = b4; B.y  = b5; B.z  = b6; B.w  = b7;
      B0.x = b12; B0.y = b13; B0.z = b14; B0.w = b15;

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B,D0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B0,D1);

      d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
      d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
#endif
    }
  };

  struct GFX928_32x16x32_F32F16F16F32_NT
  {
    using DRegisters = float[8];
    using ARegisters = half_t[16];
    using BRegisters = half_t[8];
    using CRegisters = float[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        float      & d4, float      & d5, float      & d6, float      & d7,
        half_t const& a0, half_t const& a1,half_t const& a2, half_t const& a3,
        half_t const& a4, half_t const& a5,half_t const& a6, half_t const& a7,
        half_t const& a8, half_t const& a9,half_t const& a10, half_t const& a11,
        half_t const& a12, half_t const& a13,half_t const& a14, half_t const& a15,
        half_t const& b0, half_t const& b1,half_t const& b2, half_t const& b3,
        half_t const& b4, half_t const& b5,half_t const& b6, half_t const& b7,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      v4f C0,C1;
      v4f D0,D1;

      C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
      C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;

      __fp16x4_t A, A0, B;

      A.x  = a0; A.y  = a1; A.z  = a2; A.w  = a3;
      A0.x = a8; A0.y = a9; A0.z = a10; A0.w = a11;
      B.x  = b0; B.y  = b1; B.z  = b2; B.w  = b3;

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B,C0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A0,B,C1);

      A.x  = a4;  A.y  = a5;  A.z  = a6;  A.w  = a7;
      A0.x = a12; A0.y = a13; A0.z = a14; A0.w = a15;
      B.x  = b4;  B.y  = b5;  B.z  = b6;  B.w  = b7;

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B,D0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A0,B,D1);

      d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
      d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
      #endif
    }
  };

  struct GFX928_32x32x32_F32F16F16F32_NT
  {
    using DRegisters = float[16];
    using ARegisters = half_t[16];
    using BRegisters = half_t[16];
    using CRegisters = float[16];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        float      & d4, float      & d5, float      & d6, float      & d7,
        float      & d8, float      & d9, float      & d10, float      & d11,
        float      & d12, float      & d13, float      & d14, float      & d15,
        half_t const& a0, half_t const& a1,half_t const& a2, half_t const& a3,
        half_t const& a4, half_t const& a5,half_t const& a6, half_t const& a7,
        half_t const& a8, half_t const& a9,half_t const& a10, half_t const& a11,
        half_t const& a12, half_t const& a13,half_t const& a14, half_t const& a15,
        half_t const& b0, half_t const& b1,half_t const& b2, half_t const& b3,
        half_t const& b4, half_t const& b5,half_t const& b6, half_t const& b7,
        half_t const& b8, half_t const& b9,half_t const& b10, half_t const& b11,
        half_t const& b12, half_t const& b13,half_t const& b14, half_t const& b15,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7,
        float const& c8, float const& c9, float const& c10, float const& c11,
        float const& c12, float const& c13, float const& c14, float const& c15)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)

      v4f C0,C1,C2,C3;
      v4f D0,D1,D2,D3;

      C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
      C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;
      C2.x = c8;  C2.y = c9;  C2.z = c10; C2.w = c11;
      C3.x = c12; C3.y = c13; C3.z = c14; C3.w = c15;

      __fp16x4_t A,B,A0,B0;

      A.x  = a0; A.y  = a1; A.z  = a2; A.w  = a3;
      A0.x = a8; A0.y = a9; A0.z = a10; A0.w = a11;
      B.x  = b0; B.y  = b1; B.z  = b2; B.w  = b3;
      B0.x = b8; B0.y = b9; B0.z = b10; B0.w = b11;

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B,C0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B0,C1);
      D2 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A0,B,C2);
      D3 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A0,B0,C3);

      A.x  = a4;  A.y  = a5;  A.z  = a6; A.w  = a7;
      A0.x = a12; A0.y = a13; A0.z = a14; A0.w = a15;
      B.x  = b4;  B.y  = b5;  B.z  = b6; B.w  = b7;
      B0.x = b12; B0.y = b13; B0.z = b14; B0.w = b15;

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B,D0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A,B0,D1);
      D2 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A0,B,D2);
      D3 = hytlass::arch::detail::mmac_f32_16x16x16_f16(A0,B0,D3);

      d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
      d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
      d8 = D2.x;  d9 = D2.y;    d10 = D2.z; d11 = D2.w;
      d12 = D3.x; d13 = D3.y;   d14 = D3.z; d15 = D3.w;
      #endif
    }
  };

/////////////////////////////////////v_mmac_f32_bf16/////////////////////////////////////
  struct GFX928_16x16x16_F32BF16BF16F32_NT
  {
    using DRegisters = float[4];
    using ARegisters = bfloat16_t[4];
    using BRegisters = bfloat16_t[4];
    using CRegisters = float[4];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        bfloat16_t const& a0, bfloat16_t const& a1,bfloat16_t const& a2, bfloat16_t const& a3,
        bfloat16_t const& b0, bfloat16_t const& b1,bfloat16_t const& b2, bfloat16_t const& b3,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      v4f c;
      v4f d;
      c.x = c0;
      c.y = c1;
      c.z = c2;
      c.w = c3;
      hytlass::Array<bfloat16_t,4> a;
      a[0] = a0;
      a[1] = a1;
      a[2] = a2;
      a[3] = a3;
      hytlass::Array<bfloat16_t,4> b;
      b[0] = b0;
      b[1] = b1;
      b[2] = b2;
      b[3] = b3;
      __bf16x4_t A,B;
      A = *(reinterpret_cast<__bf16x4_t *>(&a));
      B = *(reinterpret_cast<__bf16x4_t *>(&b));

      d = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A, B, c);

      d0 = d.x;
      d1 = d.y;
      d2 = d.z;
      d3 = d.w;
  #endif
    }
  };

  struct GFX928_16x16x16_F32BF16BF16F32_NT_FOR_GEMM1
  {
    using DRegisters = float[4];
    using ARegisters = bfloat16_t[4];
    using BRegisters = bfloat16_t[4];
    using CRegisters = float[4];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        bfloat16_t const& a0, bfloat16_t const& a1,bfloat16_t const& a2, bfloat16_t const& a3,
        bfloat16_t const& b0, bfloat16_t const& b1,bfloat16_t const& b2, bfloat16_t const& b3,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
          v4f c;
          v4f d;
          c.x = c0;
          c.y = c1;
          c.z = c2;
          c.w = c3;
          hytlass::Array<bfloat16_t,4> a;
          a[0] = a0;
          a[1] = a1;
          a[2] = a2;
          a[3] = a3;
          hytlass::Array<bfloat16_t,4> b;
          b[0] = b0;
          b[1] = b1;
          b[2] = b2;
          b[3] = b3;

          __bf16x4_t A,B;
          A = *(reinterpret_cast<__bf16x4_t *>(&a));
          B = *(reinterpret_cast<__bf16x4_t *>(&b));

          d = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A, B, c);

          d0 = d.x;
          d1 = d.y;
          d2 = d.z;
          d3 = d.w;
      #endif
    }

  };

  struct GFX928_16x32x16_F32BF16BF16F32_NT
  {
    using DRegisters = float[8];
    using ARegisters = bfloat16_t[4];
    using BRegisters = bfloat16_t[8];
    using CRegisters = float[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        float      & d4, float      & d5, float      & d6, float      & d7,
        bfloat16_t const& a0, bfloat16_t const& a1,bfloat16_t const& a2, bfloat16_t const& a3,
        bfloat16_t const& b0, bfloat16_t const& b1,bfloat16_t const& b2, bfloat16_t const& b3,
        bfloat16_t const& b4, bfloat16_t const& b5,bfloat16_t const& b6, bfloat16_t const& b7,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      v4f C0,C1;
      v4f D0,D1;

      C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
      C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;

      hytlass::Array<bfloat16_t, 4> array_a;
      hytlass::Array<bfloat16_t, 4> array_b0, array_b1;

      array_a[0] = a0; array_a[1] = a1; array_a[2] = a2; array_a[3] = a3;

      array_b0[0] = b0; array_b0[1] = b1; array_b0[2] = b2; array_b0[3] = b3;
      array_b1[0] = b4; array_b1[1] = b5; array_b1[2] = b6; array_b1[3] = b7;

      __bf16x4_t A,B,B0;
      A = *reinterpret_cast<__bf16x4_t*>(&array_a);
      B = *reinterpret_cast<__bf16x4_t*>(&array_b0);
      B0 = *reinterpret_cast<__bf16x4_t*>(&array_b1);

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B,C0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B0,C1);

      d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
      d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
  #endif
    }
  };

  struct GFX928_32x16x16_F32BF16BF16F32_NT
  {
    using DRegisters = float[8];
    using ARegisters = bfloat16_t[8];
    using BRegisters = bfloat16_t[4];
    using CRegisters = float[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        float      & d4, float      & d5, float      & d6, float      & d7,
        bfloat16_t const& a0, bfloat16_t const& a1,bfloat16_t const& a2, bfloat16_t const& a3,
        bfloat16_t const& a4, bfloat16_t const& a5,bfloat16_t const& a6, bfloat16_t const& a7,
        bfloat16_t const& b0, bfloat16_t const& b1,bfloat16_t const& b2, bfloat16_t const& b3,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      v4f C0,C1;
      v4f D0,D1;

      C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
      C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;

      hytlass::Array<bfloat16_t, 4> array_a0, array_a1;
      hytlass::Array<bfloat16_t, 4> array_b;

      array_a0[0] = a0; array_a0[1] = a1; array_a0[2] = a2; array_a0[3] = a3;
      array_a1[0] = a4; array_a1[1] = a5; array_a1[2] = a6; array_a1[3] = a7;

      array_b[0] = b0; array_b[1] = b1; array_b[2] = b2; array_b[3] = b3;

      __bf16x4_t A, A0, B;
      A = *reinterpret_cast<__bf16x4_t*>(&array_a0);
      A0 = *reinterpret_cast<__bf16x4_t*>(&array_a1);
      B = *reinterpret_cast<__bf16x4_t*>(&array_b);

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B,C0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A0,B,C1);

      d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
      d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
  #endif
    }
  };

  struct GFX928_32x32x16_F32BF16BF16F32_NT
  {
    using DRegisters = float[16];
    using ARegisters = bfloat16_t[8];
    using BRegisters = bfloat16_t[8];
    using CRegisters = float[16];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        float      & d4, float      & d5, float      & d6, float      & d7,
        float      & d8, float      & d9, float      & d10, float      & d11,
        float      & d12, float      & d13, float      & d14, float      & d15,
        bfloat16_t const& a0, bfloat16_t const& a1, bfloat16_t const& a2, bfloat16_t const& a3,
        bfloat16_t const& a4, bfloat16_t const& a5, bfloat16_t const& a6, bfloat16_t const& a7,
        bfloat16_t const& b0, bfloat16_t const& b1, bfloat16_t const& b2, bfloat16_t const& b3,
        bfloat16_t const& b4, bfloat16_t const& b5, bfloat16_t const& b6, bfloat16_t const& b7,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7,
        float const& c8, float const& c9, float const& c10, float const& c11,
        float const& c12, float const& c13, float const& c14, float const& c15)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      v4f C0,C1,C2,C3;
      v4f D0,D1,D2,D3;

      C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
      C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;
      C2.x = c8;  C2.y = c9;  C2.z = c10; C2.w = c11;
      C3.x = c12; C3.y = c13; C3.z = c14; C3.w = c15;

      hytlass::Array<bfloat16_t, 4> array_a0, array_a1;
      hytlass::Array<bfloat16_t, 4> array_b0, array_b1;

      array_a0[0] = a0; array_a0[1] = a1; array_a0[2] = a2; array_a0[3] = a3;
      array_a1[0] = a4; array_a1[1] = a5; array_a1[2] = a6; array_a1[3] = a7;

      array_b0[0] = b0; array_b0[1] = b1; array_b0[2] = b2; array_b0[3] = b3;
      array_b1[0] = b4; array_b1[1] = b5; array_b1[2] = b6; array_b1[3] = b7;

      __bf16x4_t A,B,A0,B0;
      A = *reinterpret_cast<__bf16x4_t*>(&array_a0);
      A0 = *reinterpret_cast<__bf16x4_t*>(&array_a1);
      B = *reinterpret_cast<__bf16x4_t*>(&array_b0);
      B0 = *reinterpret_cast<__bf16x4_t*>(&array_b1);

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B,C0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B0,C1);
      D2 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A0,B,C2);
      D3 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A0,B0,C3);

      d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
      d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
      d8 = D2.x;  d9 = D2.y;    d10 = D2.z; d11 = D2.w;
      d12 = D3.x; d13 = D3.y;   d14 = D3.z; d15 = D3.w;
      #endif
    }
  };

  struct GFX928_32x32x16_F32BF16BF16F32_NT_ALT
  {
    using DRegisters = float[16];
    using ARegisters = bfloat16_t[8];
    using BRegisters = bfloat16_t[8];
    using CRegisters = float[16];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        float      & d4, float      & d5, float      & d6, float      & d7,
        float      & d8, float      & d9, float      & d10, float      & d11,
        float      & d12, float      & d13, float      & d14, float      & d15,
        bfloat16_t const& a0, bfloat16_t const& a1,bfloat16_t const& a2, bfloat16_t const& a3,
        bfloat16_t const& a4, bfloat16_t const& a5,bfloat16_t const& a6, bfloat16_t const& a7,
        bfloat16_t const& b0, bfloat16_t const& b1,bfloat16_t const& b2, bfloat16_t const& b3,
        bfloat16_t const& b4, bfloat16_t const& b5,bfloat16_t const& b6, bfloat16_t const& b7,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7,
        float const& c8, float const& c9, float const& c10, float const& c11,
        float const& c12, float const& c13, float const& c14, float const& c15)
    {

      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)

        v4f C0,C1,C2,C3;
        v4f D0,D1,D2,D3;

        C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
        C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;
        C2.x = c8;  C2.y = c9;  C2.z = c10; C2.w = c11;
        C3.x = c12; C3.y = c13; C3.z = c14; C3.w = c15;

        hytlass::Array<bfloat16_t, 4> array_a0, array_a1;
        hytlass::Array<bfloat16_t, 4> array_b0, array_b1;

        array_a0[0] = a0; array_a0[1] = a1; array_a0[2] = a2; array_a0[3] = a3;
        array_a1[0] = a4; array_a1[1] = a5; array_a1[2] = a6; array_a1[3] = a7;

        array_b0[0] = b0; array_b0[1] = b1; array_b0[2] = b2; array_b0[3] = b3;
        array_b1[0] = b4; array_b1[1] = b5; array_b1[2] = b6; array_b1[3] = b7;

        __bf16x4_t A,B,A0,B0;
        A = *reinterpret_cast<__bf16x4_t*>(&array_a0);
        A0 = *reinterpret_cast<__bf16x4_t*>(&array_a1);
        B = *reinterpret_cast<__bf16x4_t*>(&array_b0);
        B0 = *reinterpret_cast<__bf16x4_t*>(&array_b1);

        D0 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B,C0);
        D1 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B0,C1);
        D2 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A0,B,C2);
        D3 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A0,B0,C3);

        d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
        d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
        d8 = D2.x;  d9 = D2.y;    d10 = D2.z; d11 = D2.w;
        d12 = D3.x; d13 = D3.y;   d14 = D3.z; d15 = D3.w;

        #endif

    }

  };

  struct GFX928_16x16x32_F32BF16BF16F32_NT {
    using DRegisters = float[4];
    using ARegisters = bfloat16_t[8];
    using BRegisters = bfloat16_t[8];
    using CRegisters = float[4];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float &d0, float &d1, float &d2, float &d3,
        bfloat16_t const &a0, bfloat16_t const &a1, bfloat16_t const &a2, bfloat16_t const &a3,
        bfloat16_t const &a4, bfloat16_t const &a5, bfloat16_t const &a6, bfloat16_t const &a7,
        bfloat16_t const &b0, bfloat16_t const &b1, bfloat16_t const &b2, bfloat16_t const &b3,
        bfloat16_t const &b4, bfloat16_t const &b5, bfloat16_t const &b6, bfloat16_t const &b7,
        float const &c0, float const &c1, float const &c2, float const &c3)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      v4f c;
      v4f d;
      c.x = c0;
      c.y = c1;
      c.z = c2;
      c.w = c3;

      hytlass::Array<bfloat16_t, 4> array_a0;
      hytlass::Array<bfloat16_t, 4> array_b0;

      array_a0[0] = a0; array_a0[1] = a1; array_a0[2] = a2; array_a0[3] = a3;
      array_b0[0] = b0; array_b0[1] = b1; array_b0[2] = b2; array_b0[3] = b3;

      __bf16x4_t A,B;
      A = *reinterpret_cast<__bf16x4_t*>(&array_a0);
      B = *reinterpret_cast<__bf16x4_t*>(&array_b0);

      d = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B,c);

      array_a0[0] = a4; array_a0[1] = a5; array_a0[2] = a6; array_a0[3] = a7;
      array_b0[0] = b4; array_b0[1] = b5; array_b0[2] = b6; array_b0[3] = b7;

      A = *reinterpret_cast<__bf16x4_t*>(&array_a0);
      B = *reinterpret_cast<__bf16x4_t*>(&array_b0);

      d = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B,d);

      d0 = d.x;
      d1 = d.y;
      d2 = d.z;
      d3 = d.w;
  #endif
    }
  };

  struct GFX928_16x32x32_F32BF16BF16F32_NT
  {
    using DRegisters = float[8];
    using ARegisters = bfloat16_t[8];
    using BRegisters = bfloat16_t[16];
    using CRegisters = float[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        float      & d4, float      & d5, float      & d6, float      & d7,
        bfloat16_t const& a0, bfloat16_t const& a1,bfloat16_t const& a2, bfloat16_t const& a3,
        bfloat16_t const& a4, bfloat16_t const& a5,bfloat16_t const& a6, bfloat16_t const& a7,
        bfloat16_t const& b0, bfloat16_t const& b1,bfloat16_t const& b2, bfloat16_t const& b3,
        bfloat16_t const& b4, bfloat16_t const& b5,bfloat16_t const& b6, bfloat16_t const& b7,
        bfloat16_t const& b8, bfloat16_t const& b9,bfloat16_t const& b10, bfloat16_t const& b11,
        bfloat16_t const& b12, bfloat16_t const& b13,bfloat16_t const& b14, bfloat16_t const& b15,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      v4f C0,C1;
      v4f D0,D1;

      C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
      C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;

      hytlass::Array<bfloat16_t, 4> array_a0;
      hytlass::Array<bfloat16_t, 4> array_b0, array_b1;

      array_a0[0] = a0; array_a0[1] = a1; array_a0[2] = a2; array_a0[3] = a3;

      array_b0[0] = b0; array_b0[1] = b1; array_b0[2] = b2; array_b0[3] = b3;
      array_b1[0] = b8; array_b1[1] = b9; array_b1[2] = b10; array_b1[3] = b11;

      __bf16x4_t A,B,B0;
      A = *reinterpret_cast<__bf16x4_t*>(&array_a0);
      B = *reinterpret_cast<__bf16x4_t*>(&array_b0);
      B0 = *reinterpret_cast<__bf16x4_t*>(&array_b1);

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B,C0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B0,C1);

      array_a0[0] = a4; array_a0[1] = a5; array_a0[2] = a6; array_a0[3] = a7;

      array_b0[0] = b4; array_b0[1] = b5; array_b0[2] = b6; array_b0[3] = b7;
      array_b1[0] = b12; array_b1[1] = b13; array_b1[2] = b14; array_b1[3] = b15;

      A = *reinterpret_cast<__bf16x4_t*>(&array_a0);
      B = *reinterpret_cast<__bf16x4_t*>(&array_b0);
      B0 = *reinterpret_cast<__bf16x4_t*>(&array_b1);

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B,D0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B0,D1);

      d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
      d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
  #endif
    }
  };

  struct GFX928_32x16x32_F32BF16BF16F32_NT
  {
    using DRegisters = float[8];
    using ARegisters = bfloat16_t[16];
    using BRegisters = bfloat16_t[8];
    using CRegisters = float[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        float      & d4, float      & d5, float      & d6, float      & d7,
        bfloat16_t const& a0, bfloat16_t const& a1,bfloat16_t const& a2, bfloat16_t const& a3,
        bfloat16_t const& a4, bfloat16_t const& a5,bfloat16_t const& a6, bfloat16_t const& a7,
        bfloat16_t const& a8, bfloat16_t const& a9,bfloat16_t const& a10, bfloat16_t const& a11,
        bfloat16_t const& a12, bfloat16_t const& a13,bfloat16_t const& a14, bfloat16_t const& a15,
        bfloat16_t const& b0, bfloat16_t const& b1,bfloat16_t const& b2, bfloat16_t const& b3,
        bfloat16_t const& b4, bfloat16_t const& b5,bfloat16_t const& b6, bfloat16_t const& b7,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      v4f C0,C1;
      v4f D0,D1;

      C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
      C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;

      hytlass::Array<bfloat16_t, 4> array_a0, array_a1;
      hytlass::Array<bfloat16_t, 4> array_b0;

      array_a0[0] = a0; array_a0[1] = a1; array_a0[2] = a2; array_a0[3] = a3;
      array_a1[0] = a8; array_a1[1] = a9; array_a1[2] = a10; array_a1[3] = a11;

      array_b0[0] = b0; array_b0[1] = b1; array_b0[2] = b2; array_b0[3] = b3;

      __bf16x4_t A,B,A0;
      A = *reinterpret_cast<__bf16x4_t*>(&array_a0);
      A0 = *reinterpret_cast<__bf16x4_t*>(&array_a1);
      B = *reinterpret_cast<__bf16x4_t*>(&array_b0);

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B,C0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A0,B,C1);

      array_a0[0] = a4; array_a0[1] = a5; array_a0[2] = a6; array_a0[3] = a7;
      array_a1[0] = a12; array_a1[1] = a13; array_a1[2] = a14; array_a1[3] = a15;

      array_b0[0] = b4; array_b0[1] = b5; array_b0[2] = b6; array_b0[3] = b7;

      A = *reinterpret_cast<__bf16x4_t*>(&array_a0);
      A0 = *reinterpret_cast<__bf16x4_t*>(&array_a1);
      B = *reinterpret_cast<__bf16x4_t*>(&array_b0);

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B,D0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A0,B,D1);

      d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
      d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
  #endif
    }
  };

  struct GFX928_32x32x32_F32BF16BF16F32_NT
  {
    using DRegisters = float[16];
    using ARegisters = bfloat16_t[16];
    using BRegisters = bfloat16_t[16];
    using CRegisters = float[16];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3,
        float      & d4, float      & d5, float      & d6, float      & d7,
        float      & d8, float      & d9, float      & d10, float      & d11,
        float      & d12, float      & d13, float      & d14, float      & d15,
        bfloat16_t const& a0, bfloat16_t const& a1,bfloat16_t const& a2, bfloat16_t const& a3,
        bfloat16_t const& a4, bfloat16_t const& a5,bfloat16_t const& a6, bfloat16_t const& a7,
        bfloat16_t const& a8, bfloat16_t const& a9,bfloat16_t const& a10, bfloat16_t const& a11,
        bfloat16_t const& a12, bfloat16_t const& a13,bfloat16_t const& a14, bfloat16_t const& a15,
        bfloat16_t const& b0, bfloat16_t const& b1,bfloat16_t const& b2, bfloat16_t const& b3,
        bfloat16_t const& b4, bfloat16_t const& b5,bfloat16_t const& b6, bfloat16_t const& b7,
        bfloat16_t const& b8, bfloat16_t const& b9,bfloat16_t const& b10, bfloat16_t const& b11,
        bfloat16_t const& b12, bfloat16_t const& b13,bfloat16_t const& b14, bfloat16_t const& b15,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7,
        float const& c8, float const& c9, float const& c10, float const& c11,
        float const& c12, float const& c13, float const& c14, float const& c15)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)

      v4f C0,C1,C2,C3;
      v4f D0,D1,D2,D3;

      C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
      C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;
      C2.x = c8;  C2.y = c9;  C2.z = c10; C2.w = c11;
      C3.x = c12; C3.y = c13; C3.z = c14; C3.w = c15;

      hytlass::Array<bfloat16_t, 4> array_a0, array_a1;
      hytlass::Array<bfloat16_t, 4> array_b0, array_b1;

      array_a0[0] = a0; array_a0[1] = a1; array_a0[2] = a2; array_a0[3] = a3;
      array_a1[0] = a8; array_a1[1] = a9; array_a1[2] = a10; array_a1[3] = a11;

      array_b0[0] = b0; array_b0[1] = b1; array_b0[2] = b2; array_b0[3] = b3;
      array_b1[0] = b8; array_b1[1] = b9; array_b1[2] = b10; array_b1[3] = b11;

      __bf16x4_t A,B,A0,B0;
      A = *reinterpret_cast<__bf16x4_t*>(&array_a0);
      A0 = *reinterpret_cast<__bf16x4_t*>(&array_a1);
      B = *reinterpret_cast<__bf16x4_t*>(&array_b0);
      B0 = *reinterpret_cast<__bf16x4_t*>(&array_b1);

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B,C0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B0,C1);
      D2 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A0,B,C2);
      D3 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A0,B0,C3);

      array_a0[0] = a4; array_a0[1] = a5; array_a0[2] = a6; array_a0[3] = a7;
      array_a1[0] = a12; array_a1[1] = a13; array_a1[2] = a14; array_a1[3] = a15;

      array_b0[0] = b4; array_b0[1] = b5; array_b0[2] = b6; array_b0[3] = b7;
      array_b1[0] = b12; array_b1[1] = b13; array_b1[2] = b14; array_b1[3] = b15;

      A = *reinterpret_cast<__bf16x4_t*>(&array_a0);
      A0 = *reinterpret_cast<__bf16x4_t*>(&array_a1);
      B = *reinterpret_cast<__bf16x4_t*>(&array_b0);
      B0 = *reinterpret_cast<__bf16x4_t*>(&array_b1);

      D0 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B,D0);
      D1 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A,B0,D1);
      D2 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A0,B,D2);
      D3 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(A0,B0,D3);

      d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
      d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
      d8 = D2.x;  d9 = D2.y;    d10 = D2.z; d11 = D2.w;
      d12 = D3.x; d13 = D3.y;   d14 = D3.z; d15 = D3.w;
      #endif
    }
  };

/////////////////////////////////////v_mmac_i32_i8/////////////////////////////////////
  struct GFX928_16x16x32_I32I8I8I32_NT
  {
    using DRegisters = int[4];
    using ARegisters = int8_t[8];
    using BRegisters = int8_t[8];
    using CRegisters = int[4];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d0, int      & d1, int      & d2, int      & d3,
        int8_t const& a0, int8_t const& a1,int8_t const& a2, int8_t const& a3,
        int8_t const& a4, int8_t const& a5,int8_t const& a6, int8_t const& a7,
        int8_t const& b0, int8_t const& b1,int8_t const& b2, int8_t const& b3,
        int8_t const& b4, int8_t const& b5,int8_t const& b6, int8_t const& b7,
        int const& c0, int const& c1, int const& c2, int const& c3)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      intx4_t c;
      intx4_t d;
      c.x = c0;
      c.y = c1;
      c.z = c2;
      c.w = c3;
      hytlass::Array<int8_t,8> a;
      a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3;
      a[4] = a4; a[5] = a5; a[6] = a6; a[7] = a7;

      hytlass::Array<int8_t,8> b;
      b[0] = b0;b[1] = b1;b[2] = b2;b[3] = b3;
      b[4] = b4;b[5] = b5;b[6] = b6;b[7] = b7;
      long A, B;
      A = *(reinterpret_cast<long *>(&a));
      B = *(reinterpret_cast<long *>(&b));
      d = hytlass::arch::detail::mmac_i32_16x16x32_i8(A,B,c);

      d0 = d.x;
      d1 = d.y;
      d2 = d.z;
      d3 = d.w;
  #endif
    }
  };

  struct GFX928_16x32x32_I32I8I8I32_NT
  {
    using DRegisters = int[8];
    using ARegisters = int8_t[8];
    using BRegisters = int8_t[16];
    using CRegisters = int[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d0, int      & d1, int      & d2, int      & d3,
        int      & d4, int      & d5, int      & d6, int      & d7,
        int8_t const& a0, int8_t const& a1,int8_t const& a2, int8_t const& a3,
        int8_t const& a4, int8_t const& a5,int8_t const& a6, int8_t const& a7,
        int8_t const& b0, int8_t const& b1,int8_t const& b2, int8_t const& b3,
        int8_t const& b4, int8_t const& b5,int8_t const& b6, int8_t const& b7,
        int8_t const& b8, int8_t const& b9,int8_t const& b10, int8_t const& b11,
        int8_t const& b12, int8_t const& b13,int8_t const& b14, int8_t const& b15,
        int const& c0, int const& c1, int const& c2, int const& c3,
        int const& c4, int const& c5, int const& c6, int const& c7)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      intx4_t C, C0;
      intx4_t D, D0;
      C.x = c0;
      C.y = c1;
      C.z = c2;
      C.w = c3;
      hytlass::Array<int8_t,8> a;
      a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3;
      a[4] = a4; a[5] = a5; a[6] = a6; a[7] = a7;

      hytlass::Array<int8_t,8> b;
      b[0] = b0;b[1] = b1;b[2] = b2;b[3] = b3;
      b[4] = b4;b[5] = b5;b[6] = b6;b[7] = b7;
      long A, B;
      A = *(reinterpret_cast<long *>(&a));
      B = *(reinterpret_cast<long *>(&b));
      D = hytlass::arch::detail::mmac_i32_16x16x32_i8(A,B,C);

      C0.x = c4;
      C0.y = c5;
      C0.z = c6;
      C0.w = c7;
      b[0] = b8;b[1] = b9;b[2] = b10;b[3] = b11;
      b[4] = b12;b[5] = b13;b[6] = b14;b[7] = b15;
      B = *(reinterpret_cast<long *>(&b));
      D0 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A,B,C0);

      d0 = D.x;
      d1 = D.y;
      d2 = D.z;
      d3 = D.w;
      d4 = D0.x;
      d5 = D0.y;
      d6 = D0.z;
      d7 = D0.w;
  #endif
    }
  };

  struct GFX928_32x16x32_I32I8I8I32_NT
  {
    using DRegisters = int[8];
    using ARegisters = int8_t[16];
    using BRegisters = int8_t[8];
    using CRegisters = int[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d0, int      & d1, int      & d2, int      & d3,
        int      & d4, int      & d5, int      & d6, int      & d7,
        int8_t const& a0, int8_t const& a1,int8_t const& a2, int8_t const& a3,
        int8_t const& a4, int8_t const& a5,int8_t const& a6, int8_t const& a7,
        int8_t const& a8, int8_t const& a9, int8_t const& a10, int8_t const& a11,
        int8_t const& a12, int8_t const& a13, int8_t const& a14, int8_t const& a15,
        int8_t const& b0, int8_t const& b1,int8_t const& b2, int8_t const& b3,
        int8_t const& b4, int8_t const& b5,int8_t const& b6, int8_t const& b7,
        int const& c0, int const& c1, int const& c2, int const& c3,
        int const& c4, int const& c5, int const& c6, int const& c7)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      intx4_t C, C0;
      intx4_t D, D0;
      C.x = c0;
      C.y = c1;
      C.z = c2;
      C.w = c3;
      hytlass::Array<int8_t,8> a;
      a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3;
      a[4] = a4; a[5] = a5; a[6] = a6; a[7] = a7;

      hytlass::Array<int8_t,8> b;
      b[0] = b0;b[1] = b1;b[2] = b2;b[3] = b3;
      b[4] = b4;b[5] = b5;b[6] = b6;b[7] = b7;
      long A, B;
      A = *(reinterpret_cast<long *>(&a));
      B = *(reinterpret_cast<long *>(&b));
      D = hytlass::arch::detail::mmac_i32_16x16x32_i8(A,B,C);

      C0.x = c4;
      C0.y = c5;
      C0.z = c6;
      C0.w = c7;
      a[0] = a8;a[1] = a9;a[2] = a10;a[3] = a11;
      a[4] = a12;a[5] = a13;a[6] = a14;a[7] = a15;
      A = *(reinterpret_cast<long *>(&a));
      D0 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A,B,C0);

      d0 = D.x;
      d1 = D.y;
      d2 = D.z;
      d3 = D.w;
      d4 = D0.x;
      d5 = D0.y;
      d6 = D0.z;
      d7 = D0.w;
  #endif
    }
  };

  struct GFX928_32x32x32_I32I8I8I32_NT
  {
    using DRegisters = int[16];
    using ARegisters = int8_t[16];
    using BRegisters = int8_t[16];
    using CRegisters = int[16];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d00, int      & d01, int      & d02, int      & d03,
        int      & d04, int      & d05, int      & d06, int      & d07,
        int      & d08, int      & d09, int      & d10, int      & d11,
        int      & d12, int      & d13, int      & d14, int      & d15,
        int8_t const& a00, int8_t const& a01, int8_t const& a02, int8_t const& a03,
        int8_t const& a04, int8_t const& a05, int8_t const& a06, int8_t const& a07,
        int8_t const& a08, int8_t const& a09, int8_t const& a10, int8_t const& a11,
        int8_t const& a12, int8_t const& a13, int8_t const& a14, int8_t const& a15,
        int8_t const& b00, int8_t const& b01, int8_t const& b02, int8_t const& b03,
        int8_t const& b04, int8_t const& b05, int8_t const& b06, int8_t const& b07,
        int8_t const& b08, int8_t const& b09, int8_t const& b10, int8_t const& b11,
        int8_t const& b12, int8_t const& b13, int8_t const& b14, int8_t const& b15,
        int const& c00, int const& c01, int const& c02, int const& c03,
        int const& c04, int const& c05, int const& c06, int const& c07,
        int const& c08, int const& c09, int const& c10, int const& c11,
        int const& c12, int const& c13, int const& c14, int const& c15)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
        intx4_t C0, C1, C2, C3;
        intx4_t D0, D1, D2, D3;
        C0.x = c00; C0.y = c01; C0.z = c02; C0.w = c03;
        C1.x = c04; C1.y = c05; C1.z = c06; C1.w = c07;
        C2.x = c08; C2.y = c09; C2.z = c10; C2.w = c11;
        C3.x = c12; C3.y = c13; C3.z = c14; C3.w = c15;

        hytlass::Array<int8_t, 16> a;
        a[0] = a00; a[1] = a01; a[2] = a02; a[3] = a03;
        a[4] = a04; a[5] = a05; a[6] = a06; a[7] = a07;
        a[8] = a08; a[9] = a09; a[10] = a10; a[11] = a11;
        a[12] = a12; a[13] = a13; a[14] = a14; a[15] = a15;

        hytlass::Array<int8_t, 16> b;
        b[0] = b00; b[1] = b01; b[2] = b02; b[3] = b03;
        b[4] = b04; b[5] = b05; b[6] = b06; b[7] = b07;
        b[8] = b08; b[9] = b09; b[10] = b10; b[11] = b11;
        b[12] = b12; b[13] = b13; b[14] = b14; b[15] = b15;
        long A0, A1, B0, B1;
        A0 = *(reinterpret_cast<long *>(&a));
        B0 = *(reinterpret_cast<long *>(&b));
        A1 = *(reinterpret_cast<long *>(&a) + 1);
        B1 = *(reinterpret_cast<long *>(&b) + 1);
        D0 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A0, B0, C0);
        D1 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A0, B1, C1);
        D2 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A1, B0, C2);
        D3 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A1, B1, C3);

        d00 = D0.x; d01 = D0.y; d02 = D0.z; d03 = D0.w;
        d04 = D1.x; d05 = D1.y; d06 = D1.z; d07 = D1.w;
        d08 = D2.x; d09 = D2.y; d10 = D2.z; d11 = D2.w;
        d12 = D3.x; d13 = D3.y; d14 = D3.z; d15 = D3.w;
      #endif
    }

  };

  struct GFX928_16x16x64_I32I8I8I32_NT
  {
    using DRegisters = int[4];
    using ARegisters = int8_t[16];
    using BRegisters = int8_t[16];
    using CRegisters = int[4];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d0, int      & d1, int      & d2, int      & d3,
        int8_t const& a0, int8_t const& a1,int8_t const& a2, int8_t const& a3,
        int8_t const& a4, int8_t const& a5,int8_t const& a6, int8_t const& a7,
        int8_t const& a8, int8_t const& a9,int8_t const& a10, int8_t const& a11,
        int8_t const& a12, int8_t const& a13,int8_t const& a14, int8_t const& a15,
        int8_t const& b0, int8_t const& b1,int8_t const& b2, int8_t const& b3,
        int8_t const& b4, int8_t const& b5,int8_t const& b6, int8_t const& b7,
        int8_t const& b8, int8_t const& b9,int8_t const& b10, int8_t const& b11,
        int8_t const& b12, int8_t const& b13,int8_t const& b14, int8_t const& b15,
        int const& c0, int const& c1, int const& c2, int const& c3)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      intx4_t c;
      intx4_t d;
      c.x = c0;
      c.y = c1;
      c.z = c2;
      c.w = c3;
      hytlass::Array<int8_t,8> a;
      a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3;
      a[4] = a4; a[5] = a5; a[6] = a6; a[7] = a7;

      hytlass::Array<int8_t,8> b;
      b[0] = b0;b[1] = b1;b[2] = b2;b[3] = b3;
      b[4] = b4;b[5] = b5;b[6] = b6;b[7] = b7;
      long A, B;
      A = *(reinterpret_cast<long *>(&a));
      B = *(reinterpret_cast<long *>(&b));
      d = hytlass::arch::detail::mmac_i32_16x16x32_i8(A,B,c);

      a[0] = a8; a[1] = a9; a[2] = a10; a[3] = a11;
      a[4] = a12; a[5] = a13; a[6] = a14; a[7] = a15;
      b[0] = b8; b[1] = b9; b[2] = b10; b[3] = b11;
      b[4] = b12; b[5] = b13; b[6] = b14; b[7] = b15;

      A = *(reinterpret_cast<long *>(&a));
      B = *(reinterpret_cast<long *>(&b));
      d = hytlass::arch::detail::mmac_i32_16x16x32_i8(A,B,d);

      d0 = d.x;
      d1 = d.y;
      d2 = d.z;
      d3 = d.w;
  #endif
    }

  };

  struct GFX928_16x32x64_I32I8I8I32_NT
  {
    using DRegisters = int[8];
    using ARegisters = int8_t[16];
    using BRegisters = int8_t[32];
    using CRegisters = int[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d00, int      & d01, int      & d02, int      & d03,
        int      & d04, int      & d05, int      & d06, int      & d07,
        int8_t const& a00, int8_t const& a01, int8_t const& a02, int8_t const& a03,
        int8_t const& a04, int8_t const& a05, int8_t const& a06, int8_t const& a07,
        int8_t const& a08, int8_t const& a09, int8_t const& a10, int8_t const& a11,
        int8_t const& a12, int8_t const& a13, int8_t const& a14, int8_t const& a15,
        int8_t const& b00, int8_t const& b01, int8_t const& b02, int8_t const& b03,
        int8_t const& b04, int8_t const& b05, int8_t const& b06, int8_t const& b07,
        int8_t const& b08, int8_t const& b09, int8_t const& b10, int8_t const& b11,
        int8_t const& b12, int8_t const& b13, int8_t const& b14, int8_t const& b15,
        int8_t const& b16, int8_t const& b17, int8_t const& b18, int8_t const& b19,
        int8_t const& b20, int8_t const& b21, int8_t const& b22, int8_t const& b23,
        int8_t const& b24, int8_t const& b25, int8_t const& b26, int8_t const& b27,
        int8_t const& b28, int8_t const& b29, int8_t const& b30, int8_t const& b31,
        int const& c00, int const& c01, int const& c02, int const& c03,
        int const& c04, int const& c05, int const& c06, int const& c07)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
        intx4_t C0, C1;
        intx4_t D0, D1;
        C0.x = c00; C0.y = c01; C0.z = c02; C0.w = c03;
        C1.x = c04; C1.y = c05; C1.z = c06; C1.w = c07;

        hytlass::Array<int8_t, 8> a;
        a[0] = a00; a[1] = a01; a[2] = a02; a[3] = a03;
        a[4] = a04; a[5] = a05; a[6] = a06; a[7] = a07;

        hytlass::Array<int8_t, 16> b;
        b[0] = b00; b[1] = b01; b[2] = b02; b[3] = b03;
        b[4] = b04; b[5] = b05; b[6] = b06; b[7] = b07;
        b[8] = b16; b[9] = b17; b[10] = b18; b[11] = b19;
        b[12] = b20; b[13] = b21; b[14] = b22; b[15] = b23;

        long A, B0, B1;
        A = *(reinterpret_cast<long *>(&a));
        B0 = *(reinterpret_cast<long *>(&b));
        B1 = *(reinterpret_cast<long *>(&b) + 1);
        D0 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A, B0, C0);
        D1 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A, B1, C1);

        a[0] = a08; a[1] = a09; a[2] = a10; a[3] = a11;
        a[4] = a12; a[5] = a13; a[6] = a14; a[7] = a15;

        b[0] = b08; b[1] = b09; b[2] = b10; b[3] = b11;
        b[4] = b12; b[5] = b13; b[6] = b14; b[7] = b15;
        b[8] = b24; b[9] = b25; b[10] = b26; b[11] = b27;
        b[12] = b28; b[13] = b29; b[14] = b30; b[15] = b31;

        A = *(reinterpret_cast<long *>(&a));
        B0 = *(reinterpret_cast<long *>(&b));
        B1 = *(reinterpret_cast<long *>(&b) + 1);
        D0 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A, B0, D0);
        D1 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A, B1, D1);

        d00 = D0.x; d01 = D0.y; d02 = D0.z; d03 = D0.w;
        d04 = D1.x; d05 = D1.y; d06 = D1.z; d07 = D1.w;
      #endif
    }
  };

  struct GFX928_32x16x64_I32I8I8I32_NT
  {
    using DRegisters = int[8];
    using ARegisters = int8_t[32];
    using BRegisters = int8_t[16];
    using CRegisters = int[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d00, int      & d01, int      & d02, int      & d03,
        int      & d04, int      & d05, int      & d06, int      & d07,
        int8_t const& a00, int8_t const& a01, int8_t const& a02, int8_t const& a03,
        int8_t const& a04, int8_t const& a05, int8_t const& a06, int8_t const& a07,
        int8_t const& a08, int8_t const& a09, int8_t const& a10, int8_t const& a11,
        int8_t const& a12, int8_t const& a13, int8_t const& a14, int8_t const& a15,
        int8_t const& a16, int8_t const& a17, int8_t const& a18, int8_t const& a19,
        int8_t const& a20, int8_t const& a21, int8_t const& a22, int8_t const& a23,
        int8_t const& a24, int8_t const& a25, int8_t const& a26, int8_t const& a27,
        int8_t const& a28, int8_t const& a29, int8_t const& a30, int8_t const& a31,
        int8_t const& b00, int8_t const& b01, int8_t const& b02, int8_t const& b03,
        int8_t const& b04, int8_t const& b05, int8_t const& b06, int8_t const& b07,
        int8_t const& b08, int8_t const& b09, int8_t const& b10, int8_t const& b11,
        int8_t const& b12, int8_t const& b13, int8_t const& b14, int8_t const& b15,
        int const& c00, int const& c01, int const& c02, int const& c03,
        int const& c04, int const& c05, int const& c06, int const& c07)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
        intx4_t C0, C1;
        intx4_t D0, D1;
        C0.x = c00; C0.y = c01; C0.z = c02; C0.w = c03;
        C1.x = c04; C1.y = c05; C1.z = c06; C1.w = c07;

        hytlass::Array<int8_t, 16> a;
        a[0] = a00; a[1] = a01; a[2] = a02; a[3] = a03;
        a[4] = a04; a[5] = a05; a[6] = a06; a[7] = a07;
        a[8] = a16; a[9] = a17; a[10] = a18; a[11] = a19;
        a[12] = a20; a[13] = a21; a[14] = a22; a[15] = a23;

        hytlass::Array<int8_t, 8> b;
        b[0] = b00; b[1] = b01; b[2] = b02; b[3] = b03;
        b[4] = b04; b[5] = b05; b[6] = b06; b[7] = b07;

        long A0, A1, B;
        A0 = *(reinterpret_cast<long *>(&a));
        A1 = *(reinterpret_cast<long *>(&a) + 1);
        B = *(reinterpret_cast<long *>(&b));
        D0 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A0, B, C0);
        D1 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A1, B, C1);

        a[0] = a08; a[1] = a09; a[2] = a10; a[3] = a11;
        a[4] = a12; a[5] = a13; a[6] = a14; a[7] = a15;
        a[8] = a24; a[9] = a25; a[10] = a26; a[11] = a27;
        a[12] = a28; a[13] = a29; a[14] = a30; a[15] = a31;

        b[0] = b08; b[1] = b09; b[2] = b10; b[3] = b11;
        b[4] = b12; b[5] = b13; b[6] = b14; b[7] = b15;

        A0 = *(reinterpret_cast<long *>(&a));
        A1 = *(reinterpret_cast<long *>(&a) + 1);
        B = *(reinterpret_cast<long *>(&b));
        D0 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A0, B, D0);
        D1 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A1, B, D1);

        d00 = D0.x; d01 = D0.y; d02 = D0.z; d03 = D0.w;
        d04 = D1.x; d05 = D1.y; d06 = D1.z; d07 = D1.w;
      #endif
    }
  };

  struct GFX928_32x32x64_I32I8I8I32_NT
  {
    using DRegisters = int[16];
    using ARegisters = int8_t[32];
    using BRegisters = int8_t[32];
    using CRegisters = int[16];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d00, int      & d01, int      & d02, int      & d03,
        int      & d04, int      & d05, int      & d06, int      & d07,
        int      & d08, int      & d09, int      & d10, int      & d11,
        int      & d12, int      & d13, int      & d14, int      & d15,
        int8_t const& a00, int8_t const& a01, int8_t const& a02, int8_t const& a03,
        int8_t const& a04, int8_t const& a05, int8_t const& a06, int8_t const& a07,
        int8_t const& a08, int8_t const& a09, int8_t const& a10, int8_t const& a11,
        int8_t const& a12, int8_t const& a13, int8_t const& a14, int8_t const& a15,
        int8_t const& a16, int8_t const& a17, int8_t const& a18, int8_t const& a19,
        int8_t const& a20, int8_t const& a21, int8_t const& a22, int8_t const& a23,
        int8_t const& a24, int8_t const& a25, int8_t const& a26, int8_t const& a27,
        int8_t const& a28, int8_t const& a29, int8_t const& a30, int8_t const& a31,
        int8_t const& b00, int8_t const& b01, int8_t const& b02, int8_t const& b03,
        int8_t const& b04, int8_t const& b05, int8_t const& b06, int8_t const& b07,
        int8_t const& b08, int8_t const& b09, int8_t const& b10, int8_t const& b11,
        int8_t const& b12, int8_t const& b13, int8_t const& b14, int8_t const& b15,
        int8_t const& b16, int8_t const& b17, int8_t const& b18, int8_t const& b19,
        int8_t const& b20, int8_t const& b21, int8_t const& b22, int8_t const& b23,
        int8_t const& b24, int8_t const& b25, int8_t const& b26, int8_t const& b27,
        int8_t const& b28, int8_t const& b29, int8_t const& b30, int8_t const& b31,
        int const& c00, int const& c01, int const& c02, int const& c03,
        int const& c04, int const& c05, int const& c06, int const& c07,
        int const& c08, int const& c09, int const& c10, int const& c11,
        int const& c12, int const& c13, int const& c14, int const& c15)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
        intx4_t C0, C1, C2, C3;
        intx4_t D0, D1, D2, D3;
        C0.x = c00; C0.y = c01; C0.z = c02; C0.w = c03;
        C1.x = c04; C1.y = c05; C1.z = c06; C1.w = c07;
        C2.x = c08; C2.y = c09; C2.z = c10; C2.w = c11;
        C3.x = c12; C3.y = c13; C3.z = c14; C3.w = c15;

        hytlass::Array<int8_t, 16> a;
        a[0] = a00; a[1] = a01; a[2] = a02; a[3] = a03;
        a[4] = a04; a[5] = a05; a[6] = a06; a[7] = a07;
        a[8] = a16; a[9] = a17; a[10] = a18; a[11] = a19;
        a[12] = a20; a[13] = a21; a[14] = a22; a[15] = a23;

        hytlass::Array<int8_t, 16> b;
        b[0] = b00; b[1] = b01; b[2] = b02; b[3] = b03;
        b[4] = b04; b[5] = b05; b[6] = b06; b[7] = b07;
        b[8] = b16; b[9] = b17; b[10] = b18; b[11] = b19;
        b[12] = b20; b[13] = b21; b[14] = b22; b[15] = b23;

        long A0, A1, B0, B1;
        A0 = *(reinterpret_cast<long *>(&a));
        B0 = *(reinterpret_cast<long *>(&b));
        A1 = *(reinterpret_cast<long *>(&a) + 1);
        B1 = *(reinterpret_cast<long *>(&b) + 1);
        D0 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A0, B0, C0);
        D1 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A0, B1, C1);
        D2 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A1, B0, C2);
        D3 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A1, B1, C3);

        a[0] = a08; a[1] = a09; a[2] = a10; a[3] = a11;
        a[4] = a12; a[5] = a13; a[6] = a14; a[7] = a15;
        a[8] = a24; a[9] = a25; a[10] = a26; a[11] = a27;
        a[12] = a28; a[13] = a29; a[14] = a30; a[15] = a31;

        b[0] = b08; b[1] = b09; b[2] = b10; b[3] = b11;
        b[4] = b12; b[5] = b13; b[6] = b14; b[7] = b15;
        b[8] = b24; b[9] = b25; b[10] = b26; b[11] = b27;
        b[12] = b28; b[13] = b29; b[14] = b30; b[15] = b31;

        A0 = *(reinterpret_cast<long *>(&a));
        B0 = *(reinterpret_cast<long *>(&b));
        A1 = *(reinterpret_cast<long *>(&a) + 1);
        B1 = *(reinterpret_cast<long *>(&b) + 1);
        D0 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A0, B0, D0);
        D1 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A0, B1, D1);
        D2 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A1, B0, D2);
        D3 = hytlass::arch::detail::mmac_i32_16x16x32_i8(A1, B1, D3);

        d00 = D0.x; d01 = D0.y; d02 = D0.z; d03 = D0.w;
        d04 = D1.x; d05 = D1.y; d06 = D1.z; d07 = D1.w;
        d08 = D2.x; d09 = D2.y; d10 = D2.z; d11 = D2.w;
        d12 = D3.x; d13 = D3.y; d14 = D3.z; d15 = D3.w;
      #endif
    }
  };

/////////////////////////////////////v_mmac_i32_u8/////////////////////////////////////
  struct GFX928_16x16x32_I32U8U8I32_NT
  {
    using DRegisters = int[4];
    using ARegisters = uint8_t[8];
    using BRegisters = uint8_t[8];
    using CRegisters = int[4];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d0, int      & d1, int      & d2, int      & d3,
        uint8_t const& a0, uint8_t const& a1,uint8_t const& a2, uint8_t const& a3,
        uint8_t const& a4, uint8_t const& a5,uint8_t const& a6, uint8_t const& a7,
        uint8_t const& b0, uint8_t const& b1,uint8_t const& b2, uint8_t const& b3,
        uint8_t const& b4, uint8_t const& b5,uint8_t const& b6, uint8_t const& b7,
        int const& c0, int const& c1, int const& c2, int const& c3)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      intx4_t c;
      intx4_t d;
      c.x = c0;
      c.y = c1;
      c.z = c2;
      c.w = c3;
      hytlass::Array<uint8_t,8> a;
      a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3;
      a[4] = a4; a[5] = a5; a[6] = a6; a[7] = a7;

      hytlass::Array<uint8_t,8> b;
      b[0] = b0;b[1] = b1;b[2] = b2;b[3] = b3;
      b[4] = b4;b[5] = b5;b[6] = b6;b[7] = b7;
      long A, B;
      A = *(reinterpret_cast<long *>(&a));
      B = *(reinterpret_cast<long *>(&b));
      d = hytlass::arch::detail::mmac_i32_16x16x32_u8(A,B,c);

      d0 = d.x;
      d1 = d.y;
      d2 = d.z;
      d3 = d.w;
  #endif
    }
  };

  struct GFX928_16x32x32_I32U8U8I32_NT
  {
    using DRegisters = int[8];
    using ARegisters = uint8_t[8];
    using BRegisters = uint8_t[16];
    using CRegisters = int[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d0, int      & d1, int      & d2, int      & d3,
        int      & d4, int      & d5, int      & d6, int      & d7,
        uint8_t const& a0, uint8_t const& a1,uint8_t const& a2, uint8_t const& a3,
        uint8_t const& a4, uint8_t const& a5,uint8_t const& a6, uint8_t const& a7,
        uint8_t const& b0, uint8_t const& b1,uint8_t const& b2, uint8_t const& b3,
        uint8_t const& b4, uint8_t const& b5,uint8_t const& b6, uint8_t const& b7,
        uint8_t const& b8, uint8_t const& b9,uint8_t const& b10, uint8_t const& b11,
        uint8_t const& b12, uint8_t const& b13,uint8_t const& b14, uint8_t const& b15,
        int const& c0, int const& c1, int const& c2, int const& c3,
        int const& c4, int const& c5, int const& c6, int const& c7)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      intx4_t C, C0;
      intx4_t D, D0;
      C.x = c0;
      C.y = c1;
      C.z = c2;
      C.w = c3;
      hytlass::Array<uint8_t,8> a;
      a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3;
      a[4] = a4; a[5] = a5; a[6] = a6; a[7] = a7;

      hytlass::Array<uint8_t,8> b;
      b[0] = b0;b[1] = b1;b[2] = b2;b[3] = b3;
      b[4] = b4;b[5] = b5;b[6] = b6;b[7] = b7;
      long A, B;
      A = *(reinterpret_cast<long *>(&a));
      B = *(reinterpret_cast<long *>(&b));
      D = hytlass::arch::detail::mmac_i32_16x16x32_u8(A,B,C);

      C0.x = c4;
      C0.y = c5;
      C0.z = c6;
      C0.w = c7;
      b[0] = b8;b[1] = b9;b[2] = b10;b[3] = b11;
      b[4] = b12;b[5] = b13;b[6] = b14;b[7] = b15;
      B = *(reinterpret_cast<long *>(&b));
      D0 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A,B,C0);

      d0 = D.x;
      d1 = D.y;
      d2 = D.z;
      d3 = D.w;
      d4 = D0.x;
      d5 = D0.y;
      d6 = D0.z;
      d7 = D0.w;
  #endif
    }
  };

  struct GFX928_32x16x32_I32U8U8I32_NT
  {
    using DRegisters = int[8];
    using ARegisters = uint8_t[16];
    using BRegisters = uint8_t[8];
    using CRegisters = int[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d0, int      & d1, int      & d2, int      & d3,
        int      & d4, int      & d5, int      & d6, int      & d7,
        uint8_t const& a0, uint8_t const& a1,uint8_t const& a2, uint8_t const& a3,
        uint8_t const& a4, uint8_t const& a5,uint8_t const& a6, uint8_t const& a7,
        uint8_t const& a8, uint8_t const& a9, uint8_t const& a10, uint8_t const& a11,
        uint8_t const& a12, uint8_t const& a13, uint8_t const& a14, uint8_t const& a15,
        uint8_t const& b0, uint8_t const& b1,uint8_t const& b2, uint8_t const& b3,
        uint8_t const& b4, uint8_t const& b5,uint8_t const& b6, uint8_t const& b7,
        int const& c0, int const& c1, int const& c2, int const& c3,
        int const& c4, int const& c5, int const& c6, int const& c7)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      intx4_t C, C0;
      intx4_t D, D0;
      C.x = c0;
      C.y = c1;
      C.z = c2;
      C.w = c3;
      hytlass::Array<uint8_t,8> a;
      a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3;
      a[4] = a4; a[5] = a5; a[6] = a6; a[7] = a7;

      hytlass::Array<uint8_t,8> b;
      b[0] = b0;b[1] = b1;b[2] = b2;b[3] = b3;
      b[4] = b4;b[5] = b5;b[6] = b6;b[7] = b7;
      long A, B;
      A = *(reinterpret_cast<long *>(&a));
      B = *(reinterpret_cast<long *>(&b));
      D = hytlass::arch::detail::mmac_i32_16x16x32_u8(A,B,C);

      C0.x = c4;
      C0.y = c5;
      C0.z = c6;
      C0.w = c7;
      a[0] = a8;a[1] = a9;a[2] = a10;a[3] = a11;
      a[4] = a12;a[5] = a13;a[6] = a14;a[7] = a15;
      A = *(reinterpret_cast<long *>(&a));
      D0 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A,B,C0);

      d0 = D.x;
      d1 = D.y;
      d2 = D.z;
      d3 = D.w;
      d4 = D0.x;
      d5 = D0.y;
      d6 = D0.z;
      d7 = D0.w;
  #endif
    }
  };

  struct GFX928_32x32x32_I32U8U8I32_NT
  {
    using DRegisters = int[16];
    using ARegisters = uint8_t[16];
    using BRegisters = uint8_t[16];
    using CRegisters = int[16];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d00, int      & d01, int      & d02, int      & d03,
        int      & d04, int      & d05, int      & d06, int      & d07,
        int      & d08, int      & d09, int      & d10, int      & d11,
        int      & d12, int      & d13, int      & d14, int      & d15,
        uint8_t const& a00, uint8_t const& a01, uint8_t const& a02, uint8_t const& a03,
        uint8_t const& a04, uint8_t const& a05, uint8_t const& a06, uint8_t const& a07,
        uint8_t const& a08, uint8_t const& a09, uint8_t const& a10, uint8_t const& a11,
        uint8_t const& a12, uint8_t const& a13, uint8_t const& a14, uint8_t const& a15,
        uint8_t const& b00, uint8_t const& b01, uint8_t const& b02, uint8_t const& b03,
        uint8_t const& b04, uint8_t const& b05, uint8_t const& b06, uint8_t const& b07,
        uint8_t const& b08, uint8_t const& b09, uint8_t const& b10, uint8_t const& b11,
        uint8_t const& b12, uint8_t const& b13, uint8_t const& b14, uint8_t const& b15,
        int const& c00, int const& c01, int const& c02, int const& c03,
        int const& c04, int const& c05, int const& c06, int const& c07,
        int const& c08, int const& c09, int const& c10, int const& c11,
        int const& c12, int const& c13, int const& c14, int const& c15)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
        intx4_t C0, C1, C2, C3;
        intx4_t D0, D1, D2, D3;
        C0.x = c00; C0.y = c01; C0.z = c02; C0.w = c03;
        C1.x = c04; C1.y = c05; C1.z = c06; C1.w = c07;
        C2.x = c08; C2.y = c09; C2.z = c10; C2.w = c11;
        C3.x = c12; C3.y = c13; C3.z = c14; C3.w = c15;

        hytlass::Array<uint8_t, 16> a;
        a[0] = a00; a[1] = a01; a[2] = a02; a[3] = a03;
        a[4] = a04; a[5] = a05; a[6] = a06; a[7] = a07;
        a[8] = a08; a[9] = a09; a[10] = a10; a[11] = a11;
        a[12] = a12; a[13] = a13; a[14] = a14; a[15] = a15;

        hytlass::Array<uint8_t, 16> b;
        b[0] = b00; b[1] = b01; b[2] = b02; b[3] = b03;
        b[4] = b04; b[5] = b05; b[6] = b06; b[7] = b07;
        b[8] = b08; b[9] = b09; b[10] = b10; b[11] = b11;
        b[12] = b12; b[13] = b13; b[14] = b14; b[15] = b15;
        long A0, A1, B0, B1;
        A0 = *(reinterpret_cast<long *>(&a));
        B0 = *(reinterpret_cast<long *>(&b));
        A1 = *(reinterpret_cast<long *>(&a) + 1);
        B1 = *(reinterpret_cast<long *>(&b) + 1);
        D0 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A0, B0, C0);
        D1 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A0, B1, C1);
        D2 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A1, B0, C2);
        D3 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A1, B1, C3);

        d00 = D0.x; d01 = D0.y; d02 = D0.z; d03 = D0.w;
        d04 = D1.x; d05 = D1.y; d06 = D1.z; d07 = D1.w;
        d08 = D2.x; d09 = D2.y; d10 = D2.z; d11 = D2.w;
        d12 = D3.x; d13 = D3.y; d14 = D3.z; d15 = D3.w;
      #endif
    }

  };

  struct GFX928_16x16x64_I32U8U8I32_NT
  {
    using DRegisters = int[4];
    using ARegisters = uint8_t[16];
    using BRegisters = uint8_t[16];
    using CRegisters = int[4];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d0, int      & d1, int      & d2, int      & d3,
        uint8_t const& a0, uint8_t const& a1,uint8_t const& a2, uint8_t const& a3,
        uint8_t const& a4, uint8_t const& a5,uint8_t const& a6, uint8_t const& a7,
        uint8_t const& a8, uint8_t const& a9,uint8_t const& a10, uint8_t const& a11,
        uint8_t const& a12, uint8_t const& a13,uint8_t const& a14, uint8_t const& a15,
        uint8_t const& b0, uint8_t const& b1,uint8_t const& b2, uint8_t const& b3,
        uint8_t const& b4, uint8_t const& b5,uint8_t const& b6, uint8_t const& b7,
        uint8_t const& b8, uint8_t const& b9,uint8_t const& b10, uint8_t const& b11,
        uint8_t const& b12, uint8_t const& b13,uint8_t const& b14, uint8_t const& b15,
        int const& c0, int const& c1, int const& c2, int const& c3)
    {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
      intx4_t c;
      intx4_t d;
      c.x = c0;
      c.y = c1;
      c.z = c2;
      c.w = c3;
      hytlass::Array<uint8_t,8> a;
      a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3;
      a[4] = a4; a[5] = a5; a[6] = a6; a[7] = a7;

      hytlass::Array<uint8_t,8> b;
      b[0] = b0;b[1] = b1;b[2] = b2;b[3] = b3;
      b[4] = b4;b[5] = b5;b[6] = b6;b[7] = b7;
      long A, B;
      A = *(reinterpret_cast<long *>(&a));
      B = *(reinterpret_cast<long *>(&b));
      d = hytlass::arch::detail::mmac_i32_16x16x32_u8(A,B,c);

      a[0] = a8; a[1] = a9; a[2] = a10; a[3] = a11;
      a[4] = a12; a[5] = a13; a[6] = a14; a[7] = a15;
      b[0] = b8; b[1] = b9; b[2] = b10; b[3] = b11;
      b[4] = b12; b[5] = b13; b[6] = b14; b[7] = b15;

      A = *(reinterpret_cast<long *>(&a));
      B = *(reinterpret_cast<long *>(&b));
      d = hytlass::arch::detail::mmac_i32_16x16x32_u8(A,B,d);

      d0 = d.x;
      d1 = d.y;
      d2 = d.z;
      d3 = d.w;
  #endif
    }

  };

  struct GFX928_16x32x64_I32U8U8I32_NT
  {
    using DRegisters = int[8];
    using ARegisters = uint8_t[16];
    using BRegisters = uint8_t[32];
    using CRegisters = int[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d00, int      & d01, int      & d02, int      & d03,
        int      & d04, int      & d05, int      & d06, int      & d07,
        uint8_t const& a00, uint8_t const& a01, uint8_t const& a02, uint8_t const& a03,
        uint8_t const& a04, uint8_t const& a05, uint8_t const& a06, uint8_t const& a07,
        uint8_t const& a08, uint8_t const& a09, uint8_t const& a10, uint8_t const& a11,
        uint8_t const& a12, uint8_t const& a13, uint8_t const& a14, uint8_t const& a15,
        uint8_t const& b00, uint8_t const& b01, uint8_t const& b02, uint8_t const& b03,
        uint8_t const& b04, uint8_t const& b05, uint8_t const& b06, uint8_t const& b07,
        uint8_t const& b08, uint8_t const& b09, uint8_t const& b10, uint8_t const& b11,
        uint8_t const& b12, uint8_t const& b13, uint8_t const& b14, uint8_t const& b15,
        uint8_t const& b16, uint8_t const& b17, uint8_t const& b18, uint8_t const& b19,
        uint8_t const& b20, uint8_t const& b21, uint8_t const& b22, uint8_t const& b23,
        uint8_t const& b24, uint8_t const& b25, uint8_t const& b26, uint8_t const& b27,
        uint8_t const& b28, uint8_t const& b29, uint8_t const& b30, uint8_t const& b31,
        int const& c00, int const& c01, int const& c02, int const& c03,
        int const& c04, int const& c05, int const& c06, int const& c07)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
        intx4_t C0, C1;
        intx4_t D0, D1;
        C0.x = c00; C0.y = c01; C0.z = c02; C0.w = c03;
        C1.x = c04; C1.y = c05; C1.z = c06; C1.w = c07;

        hytlass::Array<uint8_t, 8> a;
        a[0] = a00; a[1] = a01; a[2] = a02; a[3] = a03;
        a[4] = a04; a[5] = a05; a[6] = a06; a[7] = a07;

        hytlass::Array<uint8_t, 16> b;
        b[0] = b00; b[1] = b01; b[2] = b02; b[3] = b03;
        b[4] = b04; b[5] = b05; b[6] = b06; b[7] = b07;
        b[8] = b16; b[9] = b17; b[10] = b18; b[11] = b19;
        b[12] = b20; b[13] = b21; b[14] = b22; b[15] = b23;

        long A, B0, B1;
        A = *(reinterpret_cast<long *>(&a));
        B0 = *(reinterpret_cast<long *>(&b));
        B1 = *(reinterpret_cast<long *>(&b) + 1);
        D0 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A, B0, C0);
        D1 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A, B1, C1);

        a[0] = a08; a[1] = a09; a[2] = a10; a[3] = a11;
        a[4] = a12; a[5] = a13; a[6] = a14; a[7] = a15;

        b[0] = b08; b[1] = b09; b[2] = b10; b[3] = b11;
        b[4] = b12; b[5] = b13; b[6] = b14; b[7] = b15;
        b[8] = b24; b[9] = b25; b[10] = b26; b[11] = b27;
        b[12] = b28; b[13] = b29; b[14] = b30; b[15] = b31;

        A = *(reinterpret_cast<long *>(&a));
        B0 = *(reinterpret_cast<long *>(&b));
        B1 = *(reinterpret_cast<long *>(&b) + 1);
        D0 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A, B0, D0);
        D1 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A, B1, D1);

        d00 = D0.x; d01 = D0.y; d02 = D0.z; d03 = D0.w;
        d04 = D1.x; d05 = D1.y; d06 = D1.z; d07 = D1.w;
      #endif
    }
  };

  struct GFX928_32x16x64_I32U8U8I32_NT
  {
    using DRegisters = int[8];
    using ARegisters = uint8_t[32];
    using BRegisters = uint8_t[16];
    using CRegisters = int[8];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d00, int      & d01, int      & d02, int      & d03,
        int      & d04, int      & d05, int      & d06, int      & d07,
        uint8_t const& a00, uint8_t const& a01, uint8_t const& a02, uint8_t const& a03,
        uint8_t const& a04, uint8_t const& a05, uint8_t const& a06, uint8_t const& a07,
        uint8_t const& a08, uint8_t const& a09, uint8_t const& a10, uint8_t const& a11,
        uint8_t const& a12, uint8_t const& a13, uint8_t const& a14, uint8_t const& a15,
        uint8_t const& a16, uint8_t const& a17, uint8_t const& a18, uint8_t const& a19,
        uint8_t const& a20, uint8_t const& a21, uint8_t const& a22, uint8_t const& a23,
        uint8_t const& a24, uint8_t const& a25, uint8_t const& a26, uint8_t const& a27,
        uint8_t const& a28, uint8_t const& a29, uint8_t const& a30, uint8_t const& a31,
        uint8_t const& b00, uint8_t const& b01, uint8_t const& b02, uint8_t const& b03,
        uint8_t const& b04, uint8_t const& b05, uint8_t const& b06, uint8_t const& b07,
        uint8_t const& b08, uint8_t const& b09, uint8_t const& b10, uint8_t const& b11,
        uint8_t const& b12, uint8_t const& b13, uint8_t const& b14, uint8_t const& b15,
        int const& c00, int const& c01, int const& c02, int const& c03,
        int const& c04, int const& c05, int const& c06, int const& c07)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
        intx4_t C0, C1;
        intx4_t D0, D1;
        C0.x = c00; C0.y = c01; C0.z = c02; C0.w = c03;
        C1.x = c04; C1.y = c05; C1.z = c06; C1.w = c07;

        hytlass::Array<uint8_t, 16> a;
        a[0] = a00; a[1] = a01; a[2] = a02; a[3] = a03;
        a[4] = a04; a[5] = a05; a[6] = a06; a[7] = a07;
        a[8] = a16; a[9] = a17; a[10] = a18; a[11] = a19;
        a[12] = a20; a[13] = a21; a[14] = a22; a[15] = a23;

        hytlass::Array<uint8_t, 8> b;
        b[0] = b00; b[1] = b01; b[2] = b02; b[3] = b03;
        b[4] = b04; b[5] = b05; b[6] = b06; b[7] = b07;

        long A0, A1, B;
        A0 = *(reinterpret_cast<long *>(&a));
        A1 = *(reinterpret_cast<long *>(&a) + 1);
        B = *(reinterpret_cast<long *>(&b));
        D0 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A0, B, C0);
        D1 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A1, B, C1);

        a[0] = a08; a[1] = a09; a[2] = a10; a[3] = a11;
        a[4] = a12; a[5] = a13; a[6] = a14; a[7] = a15;
        a[8] = a24; a[9] = a25; a[10] = a26; a[11] = a27;
        a[12] = a28; a[13] = a29; a[14] = a30; a[15] = a31;

        b[0] = b08; b[1] = b09; b[2] = b10; b[3] = b11;
        b[4] = b12; b[5] = b13; b[6] = b14; b[7] = b15;

        A0 = *(reinterpret_cast<long *>(&a));
        A1 = *(reinterpret_cast<long *>(&a) + 1);
        B = *(reinterpret_cast<long *>(&b));
        D0 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A0, B, D0);
        D1 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A1, B, D1);

        d00 = D0.x; d01 = D0.y; d02 = D0.z; d03 = D0.w;
        d04 = D1.x; d05 = D1.y; d06 = D1.z; d07 = D1.w;
      #endif
    }
  };

  struct GFX928_32x32x64_I32U8U8I32_NT
  {
    using DRegisters = int[16];
    using ARegisters = uint8_t[32];
    using BRegisters = uint8_t[32];
    using CRegisters = int[16];

    // Register asm fma
    HUTE_HOST_DEVICE static void
    fma(int      & d00, int      & d01, int      & d02, int      & d03,
        int      & d04, int      & d05, int      & d06, int      & d07,
        int      & d08, int      & d09, int      & d10, int      & d11,
        int      & d12, int      & d13, int      & d14, int      & d15,
        uint8_t const& a00, uint8_t const& a01, uint8_t const& a02, uint8_t const& a03,
        uint8_t const& a04, uint8_t const& a05, uint8_t const& a06, uint8_t const& a07,
        uint8_t const& a08, uint8_t const& a09, uint8_t const& a10, uint8_t const& a11,
        uint8_t const& a12, uint8_t const& a13, uint8_t const& a14, uint8_t const& a15,
        uint8_t const& a16, uint8_t const& a17, uint8_t const& a18, uint8_t const& a19,
        uint8_t const& a20, uint8_t const& a21, uint8_t const& a22, uint8_t const& a23,
        uint8_t const& a24, uint8_t const& a25, uint8_t const& a26, uint8_t const& a27,
        uint8_t const& a28, uint8_t const& a29, uint8_t const& a30, uint8_t const& a31,
        uint8_t const& b00, uint8_t const& b01, uint8_t const& b02, uint8_t const& b03,
        uint8_t const& b04, uint8_t const& b05, uint8_t const& b06, uint8_t const& b07,
        uint8_t const& b08, uint8_t const& b09, uint8_t const& b10, uint8_t const& b11,
        uint8_t const& b12, uint8_t const& b13, uint8_t const& b14, uint8_t const& b15,
        uint8_t const& b16, uint8_t const& b17, uint8_t const& b18, uint8_t const& b19,
        uint8_t const& b20, uint8_t const& b21, uint8_t const& b22, uint8_t const& b23,
        uint8_t const& b24, uint8_t const& b25, uint8_t const& b26, uint8_t const& b27,
        uint8_t const& b28, uint8_t const& b29, uint8_t const& b30, uint8_t const& b31,
        int const& c00, int const& c01, int const& c02, int const& c03,
        int const& c04, int const& c05, int const& c06, int const& c07,
        int const& c08, int const& c09, int const& c10, int const& c11,
        int const& c12, int const& c13, int const& c14, int const& c15)
    {
      #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)) && defined(__HIPCC__)
        intx4_t C0, C1, C2, C3;
        intx4_t D0, D1, D2, D3;
        C0.x = c00; C0.y = c01; C0.z = c02; C0.w = c03;
        C1.x = c04; C1.y = c05; C1.z = c06; C1.w = c07;
        C2.x = c08; C2.y = c09; C2.z = c10; C2.w = c11;
        C3.x = c12; C3.y = c13; C3.z = c14; C3.w = c15;

        hytlass::Array<uint8_t, 16> a;
        a[0] = a00; a[1] = a01; a[2] = a02; a[3] = a03;
        a[4] = a04; a[5] = a05; a[6] = a06; a[7] = a07;
        a[8] = a16; a[9] = a17; a[10] = a18; a[11] = a19;
        a[12] = a20; a[13] = a21; a[14] = a22; a[15] = a23;

        hytlass::Array<uint8_t, 16> b;
        b[0] = b00; b[1] = b01; b[2] = b02; b[3] = b03;
        b[4] = b04; b[5] = b05; b[6] = b06; b[7] = b07;
        b[8] = b16; b[9] = b17; b[10] = b18; b[11] = b19;
        b[12] = b20; b[13] = b21; b[14] = b22; b[15] = b23;

        long A0, A1, B0, B1;
        A0 = *(reinterpret_cast<long *>(&a));
        B0 = *(reinterpret_cast<long *>(&b));
        A1 = *(reinterpret_cast<long *>(&a) + 1);
        B1 = *(reinterpret_cast<long *>(&b) + 1);
        D0 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A0, B0, C0);
        D1 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A0, B1, C1);
        D2 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A1, B0, C2);
        D3 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A1, B1, C3);

        a[0] = a08; a[1] = a09; a[2] = a10; a[3] = a11;
        a[4] = a12; a[5] = a13; a[6] = a14; a[7] = a15;
        a[8] = a24; a[9] = a25; a[10] = a26; a[11] = a27;
        a[12] = a28; a[13] = a29; a[14] = a30; a[15] = a31;

        b[0] = b08; b[1] = b09; b[2] = b10; b[3] = b11;
        b[4] = b12; b[5] = b13; b[6] = b14; b[7] = b15;
        b[8] = b24; b[9] = b25; b[10] = b26; b[11] = b27;
        b[12] = b28; b[13] = b29; b[14] = b30; b[15] = b31;

        A0 = *(reinterpret_cast<long *>(&a));
        B0 = *(reinterpret_cast<long *>(&b));
        A1 = *(reinterpret_cast<long *>(&a) + 1);
        B1 = *(reinterpret_cast<long *>(&b) + 1);
        D0 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A0, B0, D0);
        D1 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A0, B1, D1);
        D2 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A1, B0, D2);
        D3 = hytlass::arch::detail::mmac_i32_16x16x32_u8(A1, B1, D3);

        d00 = D0.x; d01 = D0.y; d02 = D0.z; d03 = D0.w;
        d04 = D1.x; d05 = D1.y; d06 = D1.z; d07 = D1.w;
        d08 = D2.x; d09 = D2.y; d10 = D2.z; d11 = D2.w;
        d12 = D3.x; d13 = D3.y; d14 = D3.z; d15 = D3.w;
      #endif
    }
  };
////////////////////////////////////////////////////////////////////////////////////////////////////

  struct GFX928_16x64x16_F32F16F16F32_NT
  {
    using DRegisters = float[16];
    using ARegisters = half_t[4];
    using BRegisters = half_t[16];
    using CRegisters = float[16];
    HUTE_HOST_DEVICE static void
    fma(float& d0,  float& d1,  float& d2,  float& d3,
        float& d4,  float& d5,  float& d6,  float& d7,
        float& d8,  float& d9,  float& d10, float& d11,
        float& d12, float& d13, float& d14, float& d15,
        half_t const& a0,  half_t const& a1,  half_t const& a2,  half_t const& a3,
        half_t const& b0,  half_t const& b1,  half_t const& b2,  half_t const& b3,
        half_t const& b4,  half_t const& b5,  half_t const& b6,  half_t const& b7,
        half_t const& b8,  half_t const& b9,  half_t const& b10, half_t const& b11,
        half_t const& b12, half_t const& b13, half_t const& b14, half_t const& b15,
        float const& c0,  float const& c1,  float const& c2,  float const& c3,
        float const& c4,  float const& c5,  float const& c6,  float const& c7,
        float const& c8,  float const& c9,  float const& c10, float const& c11,
        float const& c12, float const& c13, float const& c14, float const& c15)
    {
      #if (defined(__gfx928__) || defined(__gfx936__)) && defined(__HIPCC__)
        v4f C0, C1, C2, C3;
        v4f D0, D1, D2, D3;
        C0.x = c0; C0.y = c4; C0.z = c8;  C0.w = c12;
        C1.x = c1; C1.y = c5; C1.z = c9;  C1.w = c13;
        C2.x = c2; C2.y = c6; C2.z = c10; C2.w = c14;
        C3.x = c3; C3.y = c7; C3.z = c11; C3.w = c15;
        __fp16x4_t A, B0, B1, B2, B3;
        A.x = a0;   A.y = a1;   A.z = a2;   A.w = a3;
        B0.x = b0;  B0.y = b1;  B0.z = b2;  B0.w = b3;
        B1.x = b4;  B1.y = b5;  B1.z = b6;  B1.w = b7;
        B2.x = b8;  B2.y = b9;  B2.z = b10; B2.w = b11;
        B3.x = b12; B3.y = b13; B3.z = b14; B3.w = b15;
        D0 = __builtin_hcu_mmac_f32_16x16x16f16(A, B0, C0);
        D1 = __builtin_hcu_mmac_f32_16x16x16f16(A, B1, C1);
        D2 = __builtin_hcu_mmac_f32_16x16x16f16(A, B2, C2);
        D3 = __builtin_hcu_mmac_f32_16x16x16f16(A, B3, C3);

        d0 = D0.x;  d1 = D1.x;  d2 = D2.x;  d3 = D3.x;
        d4 = D0.y;  d5 = D1.y;  d6 = D2.y;  d7 = D3.y;
        d8 = D0.z;  d9 = D1.z;  d10 = D2.z; d11 = D3.z;
        d12 = D0.w; d13 = D1.w; d14 = D2.w; d15 = D3.w;
      #elif (defined(__gfx938__)) && defined(__HIPCC__)
        v4f C0, C1, C2, C3;
        v4f D0, D1, D2, D3;
        C0.x = c0; C0.y = c4; C0.z = c8;  C0.w = c12;
        C1.x = c1; C1.y = c5; C1.z = c9;  C1.w = c13;
        C2.x = c2; C2.y = c6; C2.z = c10; C2.w = c14;
        C3.x = c3; C3.y = c7; C3.z = c11; C3.w = c15;
        __fp16x4_t A, B0, B1, B2, B3;
        A.x = a0;   A.y = a1;   A.z = a2;   A.w = a3;
        B0.x = b0;  B0.y = b1;  B0.z = b2;  B0.w = b3;
        B1.x = b4;  B1.y = b5;  B1.z = b6;  B1.w = b7;
        B2.x = b8;  B2.y = b9;  B2.z = b10; B2.w = b11;
        B3.x = b12; B3.y = b13; B3.z = b14; B3.w = b15;
        D0 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(A, B0, C0, false, false);
        D1 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(A, B1, C1, false, false);
        D2 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(A, B2, C2, false, false);
        D3 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(A, B3, C3, false, false);

        d0 = D0.x;  d1 = D1.x;  d2 = D2.x;  d3 = D3.x;
        d4 = D0.y;  d5 = D1.y;  d6 = D2.y;  d7 = D3.y;
        d8 = D0.z;  d9 = D1.z;  d10 = D2.z; d11 = D3.z;
        d12 = D0.w; d13 = D1.w; d14 = D2.w; d15 = D3.w;
      #endif
    }
  };

  struct GFX928_16x64x16_F32BF16BF16F32_NT
  {
    using DRegisters = float[16];
    using ARegisters = bfloat16_t[4];
    using BRegisters = bfloat16_t[16];
    using CRegisters = float[16];
    HUTE_HOST_DEVICE static void
    fma(float& d0,  float& d1,  float& d2,  float& d3,
        float& d4,  float& d5,  float& d6,  float& d7,
        float& d8,  float& d9,  float& d10, float& d11,
        float& d12, float& d13, float& d14, float& d15,
        bfloat16_t const& a0,  bfloat16_t const& a1,  bfloat16_t const& a2,  bfloat16_t const& a3,
        bfloat16_t const& b0,  bfloat16_t const& b1,  bfloat16_t const& b2,  bfloat16_t const& b3,
        bfloat16_t const& b4,  bfloat16_t const& b5,  bfloat16_t const& b6,  bfloat16_t const& b7,
        bfloat16_t const& b8,  bfloat16_t const& b9,  bfloat16_t const& b10, bfloat16_t const& b11,
        bfloat16_t const& b12, bfloat16_t const& b13, bfloat16_t const& b14, bfloat16_t const& b15,
        float const& c0,  float const& c1,  float const& c2,  float const& c3,
        float const& c4,  float const& c5,  float const& c6,  float const& c7,
        float const& c8,  float const& c9,  float const& c10, float const& c11,
        float const& c12, float const& c13, float const& c14, float const& c15)
    {
      #if (defined(__gfx928__) || defined(__gfx936__)) && defined(__HIPCC__)
        v4f C0, C1, C2, C3;
        v4f D0, D1, D2, D3;
        C0.x = c0; C0.y = c4; C0.z = c8;  C0.w = c12;
        C1.x = c1; C1.y = c5; C1.z = c9;  C1.w = c13;
        C2.x = c2; C2.y = c6; C2.z = c10; C2.w = c14;
        C3.x = c3; C3.y = c7; C3.z = c11; C3.w = c15;
        hytlass::Array<bfloat16_t, 4> array_a0;
        hytlass::Array<bfloat16_t, 4> array_b0, array_b1, array_b2, array_b3;
        array_a0[0] = a0;  array_a0[1] = a1;  array_a0[2] = a2;  array_a0[3] = a3;
        array_b0[0] = b0;  array_b0[1] = b1;  array_b0[2] = b2;  array_b0[3] = b3;
        array_b1[0] = b4;  array_b1[1] = b5;  array_b1[2] = b6;  array_b1[3] = b7;
        array_b2[0] = b8;  array_b2[1] = b9;  array_b2[2] = b10; array_b2[3] = b11;
        array_b3[0] = b12; array_b3[1] = b13; array_b3[2] = b14; array_b3[3] = b15;
        __bf16x4_t A, B0, B1, B2, B3;
        A = *reinterpret_cast<__bf16x4_t*>(&array_a0);
        B0 = *reinterpret_cast<__bf16x4_t*>(&array_b0);
        B1 = *reinterpret_cast<__bf16x4_t*>(&array_b1);
        B2 = *reinterpret_cast<__bf16x4_t*>(&array_b2);
        B3 = *reinterpret_cast<__bf16x4_t*>(&array_b3);
        D0 = __builtin_hcu_mmac_f32_16x16x16bf16(A, B0, C0);
        D1 = __builtin_hcu_mmac_f32_16x16x16bf16(A, B1, C1);
        D2 = __builtin_hcu_mmac_f32_16x16x16bf16(A, B2, C2);
        D3 = __builtin_hcu_mmac_f32_16x16x16bf16(A, B3, C3);

        d0 = D0.x;  d1 = D1.x;  d2 = D2.x;  d3 = D3.x;
        d4 = D0.y;  d5 = D1.y;  d6 = D2.y;  d7 = D3.y;
        d8 = D0.z;  d9 = D1.z;  d10 = D2.z; d11 = D3.z;
        d12 = D0.w; d13 = D1.w; d14 = D2.w; d15 = D3.w;
      #elif (defined(__gfx938__)) && defined(__HIPCC__)
        v4f C0, C1, C2, C3;
        v4f D0, D1, D2, D3;
        C0.x = c0; C0.y = c4; C0.z = c8;  C0.w = c12;
        C1.x = c1; C1.y = c5; C1.z = c9;  C1.w = c13;
        C2.x = c2; C2.y = c6; C2.z = c10; C2.w = c14;
        C3.x = c3; C3.y = c7; C3.z = c11; C3.w = c15;
        hytlass::Array<bfloat16_t, 4> array_a0;
        hytlass::Array<bfloat16_t, 4> array_b0, array_b1, array_b2, array_b3;
        array_a0[0] = a0;  array_a0[1] = a1;  array_a0[2] = a2;  array_a0[3] = a3;
        array_b0[0] = b0;  array_b0[1] = b1;  array_b0[2] = b2;  array_b0[3] = b3;
        array_b1[0] = b4;  array_b1[1] = b5;  array_b1[2] = b6;  array_b1[3] = b7;
        array_b2[0] = b8;  array_b2[1] = b9;  array_b2[2] = b10; array_b2[3] = b11;
        array_b3[0] = b12; array_b3[1] = b13; array_b3[2] = b14; array_b3[3] = b15;
        __bf16x4_t A, B0, B1, B2, B3;
        A = *reinterpret_cast<__bf16x4_t*>(&array_a0);
        B0 = *reinterpret_cast<__bf16x4_t*>(&array_b0);
        B1 = *reinterpret_cast<__bf16x4_t*>(&array_b1);
        B2 = *reinterpret_cast<__bf16x4_t*>(&array_b2);
        B3 = *reinterpret_cast<__bf16x4_t*>(&array_b3);
        D0 = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(A, B0, C0, false, false);
        D1 = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(A, B1, C1, false, false);
        D2 = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(A, B2, C2, false, false);
        D3 = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(A, B3, C3, false, false);

        d0 = D0.x;  d1 = D1.x;  d2 = D2.x;  d3 = D3.x;
        d4 = D0.y;  d5 = D1.y;  d6 = D2.y;  d7 = D3.y;
        d8 = D0.z;  d9 = D1.z;  d10 = D2.z; d11 = D3.z;
        d12 = D0.w; d13 = D1.w; d14 = D2.w; d15 = D3.w;
      #endif
    }
  };

  struct GFX928_16x64x32_F32F16F16F32_NT
  {
    using DRegisters = float[16];
    using ARegisters = half_t[8];
    using BRegisters = half_t[32];
    using CRegisters = float[16];
    HUTE_HOST_DEVICE static void
    fma(float& d0,  float& d1,  float& d2,  float& d3,
        float& d4,  float& d5,  float& d6,  float& d7,
        float& d8,  float& d9,  float& d10, float& d11,
        float& d12, float& d13, float& d14, float& d15,
        half_t const& a0,  half_t const& a1,  half_t const& a2,  half_t const& a3,
        half_t const& a4,  half_t const& a5,  half_t const& a6,  half_t const& a7,
        half_t const& b0,  half_t const& b1,  half_t const& b2,  half_t const& b3,
        half_t const& b4,  half_t const& b5,  half_t const& b6,  half_t const& b7,
        half_t const& b8,  half_t const& b9,  half_t const& b10, half_t const& b11,
        half_t const& b12, half_t const& b13, half_t const& b14, half_t const& b15,
        half_t const& b16, half_t const& b17, half_t const& b18, half_t const& b19,
        half_t const& b20, half_t const& b21, half_t const& b22, half_t const& b23,
        half_t const& b24, half_t const& b25, half_t const& b26, half_t const& b27,
        half_t const& b28, half_t const& b29, half_t const& b30, half_t const& b31,
        float const& c0,  float const& c1,  float const& c2,  float const& c3,
        float const& c4,  float const& c5,  float const& c6,  float const& c7,
        float const& c8,  float const& c9,  float const& c10, float const& c11,
        float const& c12, float const& c13, float const& c14, float const& c15)
    {
      #if (defined(__gfx928__) || defined(__gfx936__)) && defined(__HIPCC__)
        v4f C0, C1, C2, C3;
        v4f D0, D1, D2, D3;
        C0.x = c0; C0.y = c4; C0.z = c8;  C0.w = c12;
        C1.x = c1; C1.y = c5; C1.z = c9;  C1.w = c13;
        C2.x = c2; C2.y = c6; C2.z = c10; C2.w = c14;
        C3.x = c3; C3.y = c7; C3.z = c11; C3.w = c15;
        __fp16x4_t A0, A1, B0, B1, B2, B3, B4, B5, B6, B7;
        A0.x = a0;  A0.y = a1;  A0.z = a2;  A0.w = a3;
        A1.x = a4;  A1.y = a5;  A1.z = a6;  A1.w = a7;
        B0.x = b0;  B0.y = b1;  B0.z = b2;  B0.w = b3;
        B4.x = b4;  B4.y = b5;  B4.z = b6;  B4.w = b7;
        B1.x = b8;  B1.y = b9;  B1.z = b10; B1.w = b11;
        B5.x = b12; B5.y = b13; B5.z = b14; B5.w = b15;
        B2.x = b16; B2.y = b17; B2.z = b18; B2.w = b19;
        B6.x = b20; B6.y = b21; B6.z = b22; B6.w = b23;
        B3.x = b24; B3.y = b25; B3.z = b26; B3.w = b27;
        B7.x = b28; B7.y = b29; B7.z = b30; B7.w = b31;
        D0 = __builtin_hcu_mmac_f32_16x16x16f16(A0, B0, C0);
        D1 = __builtin_hcu_mmac_f32_16x16x16f16(A0, B1, C1);
        D2 = __builtin_hcu_mmac_f32_16x16x16f16(A0, B2, C2);
        D3 = __builtin_hcu_mmac_f32_16x16x16f16(A0, B3, C3);
        D0 = __builtin_hcu_mmac_f32_16x16x16f16(A1, B4, D0);
        D1 = __builtin_hcu_mmac_f32_16x16x16f16(A1, B5, D1);
        D2 = __builtin_hcu_mmac_f32_16x16x16f16(A1, B6, D2);
        D3 = __builtin_hcu_mmac_f32_16x16x16f16(A1, B7, D3);

        d0 = D0.x;  d1 = D1.x;  d2 = D2.x;  d3 = D3.x;
        d4 = D0.y;  d5 = D1.y;  d6 = D2.y;  d7 = D3.y;
        d8 = D0.z;  d9 = D1.z;  d10 = D2.z; d11 = D3.z;
        d12 = D0.w; d13 = D1.w; d14 = D2.w; d15 = D3.w;
      #elif (defined(__gfx938__)) && defined(__HIPCC__)
        v4f C0, C1, C2, C3;
        v4f D0, D1, D2, D3;
        C0.x = c0; C0.y = c4; C0.z = c8;  C0.w = c12;
        C1.x = c1; C1.y = c5; C1.z = c9;  C1.w = c13;
        C2.x = c2; C2.y = c6; C2.z = c10; C2.w = c14;
        C3.x = c3; C3.y = c7; C3.z = c11; C3.w = c15;
        __fp16x4_t A0, A1, B0, B1, B2, B3, B4, B5, B6, B7;
        A0.x = a0;  A0.y = a1;  A0.z = a2;  A0.w = a3;
        A1.x = a4;  A1.y = a5;  A1.z = a6;  A1.w = a7;
        B0.x = b0;  B0.y = b1;  B0.z = b2;  B0.w = b3;
        B4.x = b4;  B4.y = b5;  B4.z = b6;  B4.w = b7;
        B1.x = b8;  B1.y = b9;  B1.z = b10; B1.w = b11;
        B5.x = b12; B5.y = b13; B5.z = b14; B5.w = b15;
        B2.x = b16; B2.y = b17; B2.z = b18; B2.w = b19;
        B6.x = b20; B6.y = b21; B6.z = b22; B6.w = b23;
        B3.x = b24; B3.y = b25; B3.z = b26; B3.w = b27;
        B7.x = b28; B7.y = b29; B7.z = b30; B7.w = b31;
        D0 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(A0, B0, C0, false, false);
        D1 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(A0, B1, C1, false, false);
        D2 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(A0, B2, C2, false, false);
        D3 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(A0, B3, C3, false, false);
        D0 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(A1, B4, D0, false, false);
        D1 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(A1, B5, D1, false, false);
        D2 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(A1, B6, D2, false, false);
        D3 = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(A1, B7, D3, false, false);

        d0 = D0.x;  d1 = D1.x;  d2 = D2.x;  d3 = D3.x;
        d4 = D0.y;  d5 = D1.y;  d6 = D2.y;  d7 = D3.y;
        d8 = D0.z;  d9 = D1.z;  d10 = D2.z; d11 = D3.z;
        d12 = D0.w; d13 = D1.w; d14 = D2.w; d15 = D3.w;
      #endif
    }
  };

  struct GFX928_16x64x32_F32BF16BF16F32_NT
  {
    using DRegisters = float[16];
    using ARegisters = bfloat16_t[8];
    using BRegisters = bfloat16_t[32];
    using CRegisters = float[16];
    HUTE_HOST_DEVICE static void
    fma(float& d0,  float& d1,  float& d2,  float& d3,
        float& d4,  float& d5,  float& d6,  float& d7,
        float& d8,  float& d9,  float& d10, float& d11,
        float& d12, float& d13, float& d14, float& d15,
        bfloat16_t const& a0,  bfloat16_t const& a1,  bfloat16_t const& a2,  bfloat16_t const& a3,
        bfloat16_t const& a4,  bfloat16_t const& a5,  bfloat16_t const& a6,  bfloat16_t const& a7,
        bfloat16_t const& b0,  bfloat16_t const& b1,  bfloat16_t const& b2,  bfloat16_t const& b3,
        bfloat16_t const& b4,  bfloat16_t const& b5,  bfloat16_t const& b6,  bfloat16_t const& b7,
        bfloat16_t const& b8,  bfloat16_t const& b9,  bfloat16_t const& b10, bfloat16_t const& b11,
        bfloat16_t const& b12, bfloat16_t const& b13, bfloat16_t const& b14, bfloat16_t const& b15,
        bfloat16_t const& b16, bfloat16_t const& b17, bfloat16_t const& b18, bfloat16_t const& b19,
        bfloat16_t const& b20, bfloat16_t const& b21, bfloat16_t const& b22, bfloat16_t const& b23,
        bfloat16_t const& b24, bfloat16_t const& b25, bfloat16_t const& b26, bfloat16_t const& b27,
        bfloat16_t const& b28, bfloat16_t const& b29, bfloat16_t const& b30, bfloat16_t const& b31,
        float const& c0,  float const& c1,  float const& c2,  float const& c3,
        float const& c4,  float const& c5,  float const& c6,  float const& c7,
        float const& c8,  float const& c9,  float const& c10, float const& c11,
        float const& c12, float const& c13, float const& c14, float const& c15)
    {
      #if (defined(__gfx928__) || defined(__gfx936__)) && defined(__HIPCC__)
        v4f C0, C1, C2, C3;
        v4f D0, D1, D2, D3;
        C0.x = c0; C0.y = c4; C0.z = c8;  C0.w = c12;
        C1.x = c1; C1.y = c5; C1.z = c9;  C1.w = c13;
        C2.x = c2; C2.y = c6; C2.z = c10; C2.w = c14;
        C3.x = c3; C3.y = c7; C3.z = c11; C3.w = c15;
        hytlass::Array<bfloat16_t, 4> array_a0, array_a1;
        hytlass::Array<bfloat16_t, 4> array_b0, array_b1, array_b2, array_b3;
        hytlass::Array<bfloat16_t, 4> array_b4, array_b5, array_b6, array_b7;
        array_a0[0] = a0;  array_a0[1] = a1;  array_a0[2] = a2;  array_a0[3] = a3;
        array_a1[0] = a4;  array_a1[1] = a5;  array_a1[2] = a6;  array_a1[3] = a7;
        array_b0[0] = b0;  array_b0[1] = b1;  array_b0[2] = b2;  array_b0[3] = b3;
        array_b4[0] = b4;  array_b4[1] = b5;  array_b4[2] = b6;  array_b4[3] = b7;
        array_b1[0] = b8;  array_b1[1] = b9;  array_b1[2] = b10; array_b1[3] = b11;
        array_b5[0] = b12; array_b5[1] = b13; array_b5[2] = b14; array_b5[3] = b15;
        array_b2[0] = b16; array_b2[1] = b17; array_b2[2] = b18; array_b2[3] = b19;
        array_b6[0] = b20; array_b6[1] = b21; array_b6[2] = b22; array_b6[3] = b23;
        array_b3[0] = b24; array_b3[1] = b25; array_b3[2] = b26; array_b3[3] = b27;
        array_b7[0] = b28; array_b7[1] = b29; array_b7[2] = b30; array_b7[3] = b31;
        __bf16x4_t A0, A1, B0, B1, B2, B3, B4, B5, B6, B7;
        A0 = *reinterpret_cast<__bf16x4_t*>(&array_a0);
        A1 = *reinterpret_cast<__bf16x4_t*>(&array_a1);
        B0 = *reinterpret_cast<__bf16x4_t*>(&array_b0);
        B1 = *reinterpret_cast<__bf16x4_t*>(&array_b1);
        B2 = *reinterpret_cast<__bf16x4_t*>(&array_b2);
        B3 = *reinterpret_cast<__bf16x4_t*>(&array_b3);
        B4 = *reinterpret_cast<__bf16x4_t*>(&array_b4);
        B5 = *reinterpret_cast<__bf16x4_t*>(&array_b5);
        B6 = *reinterpret_cast<__bf16x4_t*>(&array_b6);
        B7 = *reinterpret_cast<__bf16x4_t*>(&array_b7);
        D0 = __builtin_hcu_mmac_f32_16x16x16bf16(A0, B0, C0);
        D1 = __builtin_hcu_mmac_f32_16x16x16bf16(A0, B1, C1);
        D2 = __builtin_hcu_mmac_f32_16x16x16bf16(A0, B2, C2);
        D3 = __builtin_hcu_mmac_f32_16x16x16bf16(A0, B3, C3);
        D0 = __builtin_hcu_mmac_f32_16x16x16bf16(A1, B4, D0);
        D1 = __builtin_hcu_mmac_f32_16x16x16bf16(A1, B5, D1);
        D2 = __builtin_hcu_mmac_f32_16x16x16bf16(A1, B6, D2);
        D3 = __builtin_hcu_mmac_f32_16x16x16bf16(A1, B7, D3);

        d0 = D0.x;  d1 = D1.x;  d2 = D2.x;  d3 = D3.x;
        d4 = D0.y;  d5 = D1.y;  d6 = D2.y;  d7 = D3.y;
        d8 = D0.z;  d9 = D1.z;  d10 = D2.z; d11 = D3.z;
        d12 = D0.w; d13 = D1.w; d14 = D2.w; d15 = D3.w;
      #elif (defined(__gfx938__)) && defined(__HIPCC__)
        v4f C0, C1, C2, C3;
        v4f D0, D1, D2, D3;
        C0.x = c0; C0.y = c4; C0.z = c8;  C0.w = c12;
        C1.x = c1; C1.y = c5; C1.z = c9;  C1.w = c13;
        C2.x = c2; C2.y = c6; C2.z = c10; C2.w = c14;
        C3.x = c3; C3.y = c7; C3.z = c11; C3.w = c15;
        hytlass::Array<bfloat16_t, 4> array_a0, array_a1;
        hytlass::Array<bfloat16_t, 4> array_b0, array_b1, array_b2, array_b3;
        hytlass::Array<bfloat16_t, 4> array_b4, array_b5, array_b6, array_b7;
        array_a0[0] = a0;  array_a0[1] = a1;  array_a0[2] = a2;  array_a0[3] = a3;
        array_a1[0] = a4;  array_a1[1] = a5;  array_a1[2] = a6;  array_a1[3] = a7;
        array_b0[0] = b0;  array_b0[1] = b1;  array_b0[2] = b2;  array_b0[3] = b3;
        array_b4[0] = b4;  array_b4[1] = b5;  array_b4[2] = b6;  array_b4[3] = b7;
        array_b1[0] = b8;  array_b1[1] = b9;  array_b1[2] = b10; array_b1[3] = b11;
        array_b5[0] = b12; array_b5[1] = b13; array_b5[2] = b14; array_b5[3] = b15;
        array_b2[0] = b16; array_b2[1] = b17; array_b2[2] = b18; array_b2[3] = b19;
        array_b6[0] = b20; array_b6[1] = b21; array_b6[2] = b22; array_b6[3] = b23;
        array_b3[0] = b24; array_b3[1] = b25; array_b3[2] = b26; array_b3[3] = b27;
        array_b7[0] = b28; array_b7[1] = b29; array_b7[2] = b30; array_b7[3] = b31;
        __bf16x4_t A0, A1, B0, B1, B2, B3, B4, B5, B6, B7;
        A0 = *reinterpret_cast<__bf16x4_t*>(&array_a0);
        A1 = *reinterpret_cast<__bf16x4_t*>(&array_a1);
        B0 = *reinterpret_cast<__bf16x4_t*>(&array_b0);
        B1 = *reinterpret_cast<__bf16x4_t*>(&array_b1);
        B2 = *reinterpret_cast<__bf16x4_t*>(&array_b2);
        B3 = *reinterpret_cast<__bf16x4_t*>(&array_b3);
        B4 = *reinterpret_cast<__bf16x4_t*>(&array_b4);
        B5 = *reinterpret_cast<__bf16x4_t*>(&array_b5);
        B6 = *reinterpret_cast<__bf16x4_t*>(&array_b6);
        B7 = *reinterpret_cast<__bf16x4_t*>(&array_b7);
        D0 = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(A0, B0, C0, false, false);
        D1 = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(A0, B1, C1, false, false);
        D2 = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(A0, B2, C2, false, false);
        D3 = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(A0, B3, C3, false, false);
        D0 = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(A1, B4, D0, false, false);
        D1 = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(A1, B5, D1, false, false);
        D2 = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(A1, B6, D2, false, false);
        D3 = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(A1, B7, D3, false, false);

        d0 = D0.x;  d1 = D1.x;  d2 = D2.x;  d3 = D3.x;
        d4 = D0.y;  d5 = D1.y;  d6 = D2.y;  d7 = D3.y;
        d8 = D0.z;  d9 = D1.z;  d10 = D2.z; d11 = D3.z;
        d12 = D0.w; d13 = D1.w; d14 = D2.w; d15 = D3.w;
      #endif
    }
  };

} // end namespace hute
