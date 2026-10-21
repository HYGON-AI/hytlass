/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hute/arch/mma_gfx928.hpp>

#include <hute/atom/mma_traits.hpp>
#include <hute/layout.hpp>

namespace hute
{
/////////////////////////////////////v_mmac_f32_f32/////////////////////////////////////
template <>
struct MMA_Traits<GFX928_16x16x8_F32F32F32F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = float;
  using ValTypeB = float;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16,_16,_8>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>,_2>,
                  Stride<Stride<_1,_32>,_16>>;
  
  using BLayout = Layout<Shape <Shape <_16, _4>,_2>,
                         Stride<Stride< _1,_32>,_16>>;
  using CLayout = Layout<Shape <Shape <_16, _4>, _4>,
                         Stride<Stride< _1,_16>,_64>>;
};

/////////////////////////////////////v_mmac_f32_tf32/////////////////////////////////////
template <>
struct MMA_Traits<GFX928_16x16x8_F32TF32TF32F32_NT>
     : MMA_Traits<GFX928_16x16x8_F32F32F32F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;
};

/////////////////////////////////////v_mmac_f32_f16/////////////////////////////////////
template <>
struct MMA_Traits<GFX928_16x16x16_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;


  using Shape_MNK = Shape<_16,_16,_16>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>,_4>,
                          Stride<Stride<_1,_64>,_16>>;
  
  using BLayout = Layout<Shape <Shape <_16, _4>,_4>,
                         Stride<Stride< _1,_64>,_16>>;
  using CLayout = Layout<Shape <Shape <_16, _4>, _4>,
                         Stride<Stride< _1,_16>,_64>>;
};

template <>
struct MMA_Traits<GFX928_16x16x16_F32F16F16F32_NT_FOR_GEMM1>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;


  using Shape_MNK = Shape<_16,_16,_16>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape <Shape <_16, _4>, _4>,
                         Stride<Stride< _1,_16>,_64>>;
  
  using BLayout = Layout<Shape <Shape <_16, _4>, _4>,
                         Stride<Stride< _1,_16>,_64>>;
  using CLayout = Layout<Shape <Shape <_16, _4>, _4>,
                         Stride<Stride< _1,_16>,_64>>;
};

template <>
struct MMA_Traits<GFX928_16x32x16_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16,_32,_16>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape <Shape <_16, _4>,_4>,
                         Stride<Stride< _1,_64>,_16>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2>>,
                         Stride<Stride<_1, _128>, Stride<_32, _16>>>;
  using CLayout = Layout<Shape <Shape <_16, _4>, Shape<_4, _2>>,
                         Stride<Stride< _1,_16>, Stride<_64, _256>>>;
};

template <>
struct MMA_Traits<GFX928_32x16x16_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_32,_16,_16>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2>>,
                         Stride<Stride<_1, _128>, Stride<_32, _16>>>;
  using BLayout = Layout<Shape <Shape <_16, _4>,_4>,
                         Stride<Stride< _1,_64>,_16>>;
  using CLayout = Layout<Shape <Shape <_16, _4>, Shape<_4, _2>>,
                         Stride<Stride< _1,_32>, Stride<_128, _16>>>;
};

template <>
struct MMA_Traits<GFX928_32x32x16_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_32, _32, _16>;
  using ThrID = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2>>,
                         Stride<Stride<_1, _128>, Stride<_32, _16>>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2>>,
                         Stride<Stride<_1, _128>, Stride<_32, _16>>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2, _2>>,
                         Stride<Stride<_1, _32>, Stride<_128, _512, _16>>>;
};

template <>
struct MMA_Traits<GFX928_32x32x16_F32F16F16F32_NT_ALT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;


  using Shape_MNK = Shape<_32,_32,_16>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>,Shape<_2,_4>>,
                          Stride<Stride<_2,_128>,Stride<_1,_32>>>;
  
  using BLayout = Layout<Shape<Shape<_16, _4>,Shape<_2,_4>>,
                          Stride<Stride<_2,_128>,Stride<_1,_32>>>;
                          
  using CLayout = Layout<Shape <Shape <_16, _4>, Shape<_4,_2,_2>>,
                         Stride<Stride< _2,_64>, Stride<_256,_32,_1>>>;
};

template <>
struct MMA_Traits<GFX928_16x16x32_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>, _8>,
                         Stride<Stride<_1, _128>, _16>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, _8>,
                         Stride<Stride<_1, _128>, _16>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, _4>,
                         Stride<Stride<_1, _16>, _64>>;
};


template <>
struct MMA_Traits<GFX928_16x32x32_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _32, _32>;
  using ThrID = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>, _8>,
                         Stride<Stride<_1, _128>, _16>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, Shape<_8, _2>>,
                         Stride<Stride<_1, _256>, Stride<_32, _16>>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2>>,
                         Stride<Stride<_1, _16>, Stride<_64, _256>>>;
};


template <>
struct MMA_Traits<GFX928_32x16x32_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_32, _16, _32>;
  using ThrID = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>, Shape<_8, _2>>,
                         Stride<Stride<_1, _256>, Stride<_32, _16>>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, _8>,
                         Stride<Stride<_1, _128>, _16>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2>>,
                         Stride<Stride<_1, _32>, Stride<_128, _16>>>;
};

template <>
struct MMA_Traits<GFX928_32x32x32_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_32,_32,_32>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>, Shape<_8, _2>>,
                         Stride<Stride<_1, _256>, Stride<_32, _16>>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, Shape<_8, _2>>,
                         Stride<Stride<_1, _256>, Stride<_32, _16>>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2, _2>>,
                         Stride<Stride<_1, _32>, Stride<_128, _512, _16>>>;
};

/////////////////////////////////////v_mmac_i32_i8/////////////////////////////////////
template<>
struct MMA_Traits<GFX928_16x16x32_I32I8I8I32_NT> {
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;
  using Shape_MNK = Shape<_16, _16, _32>;
  using ThrID = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>, _8>,
                         Stride<Stride<_1, _128>, _16>>;

  using BLayout = Layout<Shape<Shape<_16, _4>, _8>,
                         Stride<Stride<_1, _128>, _16>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, _4>,
                         Stride<Stride<_1, _16>, _64>>;
};

template<>
struct MMA_Traits<GFX928_16x32x32_I32I8I8I32_NT> {
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16, _32, _32>;
  using ThrID = Layout<_64>;

  using ALayout = Layout<Shape<Shape<_16, _4>, _8>,
                         Stride<Stride<_1, _128>, _16>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, Shape<_8, _2>>,
                         Stride<Stride<_1, _256>, Stride<_32, _16>>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2>>,
                         Stride<Stride<_1, _16>, Stride<_64, _256>>>;
};

template<>
struct MMA_Traits<GFX928_32x16x32_I32I8I8I32_NT> {
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_32, _16, _32>;
  using ThrID = Layout<_64>;

  using ALayout = Layout<Shape<Shape<_16, _4>, Shape<_8, _2>>,
                         Stride<Stride<_1, _256>, Stride<_32, _16>>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, _8>,
                         Stride<Stride<_1, _128>, _16>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2>>,
                         Stride<Stride<_1, _32>, Stride<_128, _16>>>;
};

template<>
struct MMA_Traits<GFX928_32x32x32_I32I8I8I32_NT> {
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID = Layout<_64>;

  using ALayout = Layout<Shape<Shape<_16, _4>, Shape<_8, _2>>,
                         Stride<Stride<_1, _256>, Stride<_32, _16>>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, Shape<_8, _2>>,
                         Stride<Stride<_1, _256>, Stride<_32, _16>>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2, _2>>,
                         Stride<Stride<_1, _32>, Stride<_128, _512, _16>>>;
};

template<>
struct MMA_Traits<GFX928_16x16x64_I32I8I8I32_NT> {
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16, _16, _64>;
  using ThrID = Layout<_64>;

  using ALayout = Layout<Shape<Shape<_16, _4>, _16>,
                         Stride<Stride<_1, _256>, _16>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, _16>,
                         Stride<Stride<_1, _256>, _16>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, _4>,
                         Stride<Stride<_1, _16>, _64>>;
};

template<>
struct MMA_Traits<GFX928_16x32x64_I32I8I8I32_NT> {
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_16, _32, _64>;
  using ThrID = Layout<_64>;

  using ALayout = Layout<Shape<Shape<_16, _4>, _16>,
                         Stride<Stride<_1, _256>, _16>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, Shape<_16, _2>>,
                         Stride<Stride<_1, _512>, Stride<_32, _16>>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2>>,
                         Stride<Stride<_1, _16>, Stride<_64, _256>>>;
};

template<>
struct MMA_Traits<GFX928_32x16x64_I32I8I8I32_NT> {
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_32, _16, _64>;
  using ThrID = Layout<_64>;

  using ALayout = Layout<Shape<Shape<_16, _4>, Shape<_16, _2>>,
                         Stride<Stride<_1, _512>, Stride<_32, _16>>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, _16>,
                         Stride<Stride<_1, _256>, _16>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2>>,
                         Stride<Stride<_1, _32>, Stride<_128, _16>>>;
};

template<>
struct MMA_Traits<GFX928_32x32x64_I32I8I8I32_NT> {
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID = Layout<_64>;

  using ALayout = Layout<Shape<Shape<_16, _4>, Shape<_16, _2>>,
                         Stride<Stride<_1, _512>, Stride<_32, _16>>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, Shape<_16, _2>>,
                         Stride<Stride<_1, _512>, Stride<_32, _16>>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2, _2>>,
                         Stride<Stride<_1, _32>, Stride<_128, _512, _16>>>;
};

/////////////////////////////////////v_mmac_f32_bf16/////////////////////////////////////
template<>
struct MMA_Traits<GFX928_16x16x16_F32BF16BF16F32_NT>
  : MMA_Traits<GFX928_16x16x16_F32F16F16F32_NT> {
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template<>
struct MMA_Traits<GFX928_32x32x16_F32BF16BF16F32_NT_ALT>
  : MMA_Traits<GFX928_32x32x16_F32F16F16F32_NT_ALT> {
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template<>
struct MMA_Traits<GFX928_16x16x16_F32BF16BF16F32_NT_FOR_GEMM1>
  : MMA_Traits<GFX928_16x16x16_F32F16F16F32_NT_FOR_GEMM1> {
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template<>
struct MMA_Traits<GFX928_16x32x16_F32BF16BF16F32_NT>
  : MMA_Traits<GFX928_16x32x16_F32F16F16F32_NT> {
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template<>
struct MMA_Traits<GFX928_32x16x16_F32BF16BF16F32_NT>
  : MMA_Traits<GFX928_32x16x16_F32F16F16F32_NT> {
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template<>
struct MMA_Traits<GFX928_32x32x16_F32BF16BF16F32_NT>
  : MMA_Traits<GFX928_32x32x16_F32F16F16F32_NT> {
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template<>
struct MMA_Traits<GFX928_16x16x32_F32BF16BF16F32_NT>
  : MMA_Traits<GFX928_16x16x32_F32F16F16F32_NT> {
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template<>
struct MMA_Traits<GFX928_16x32x32_F32BF16BF16F32_NT>
  : MMA_Traits<GFX928_16x32x32_F32F16F16F32_NT> {
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template<>
struct MMA_Traits<GFX928_32x16x32_F32BF16BF16F32_NT>
  : MMA_Traits<GFX928_32x16x32_F32F16F16F32_NT> {
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template<>
struct MMA_Traits<GFX928_32x32x32_F32BF16BF16F32_NT>
  : MMA_Traits<GFX928_32x32x32_F32F16F16F32_NT> {
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

/////////////////////////////////////v_mmac_i32_u8/////////////////////////////////////
template<>
struct MMA_Traits<GFX928_16x16x32_I32U8U8I32_NT>
  : MMA_Traits<GFX928_16x16x32_I32I8I8I32_NT> {
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template<>
struct MMA_Traits<GFX928_16x32x32_I32U8U8I32_NT>
  : MMA_Traits<GFX928_16x32x32_I32I8I8I32_NT>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template<>
struct MMA_Traits<GFX928_32x16x32_I32U8U8I32_NT>
  : MMA_Traits<GFX928_32x16x32_I32I8I8I32_NT>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template<>
struct MMA_Traits<GFX928_32x32x32_I32U8U8I32_NT>
  : MMA_Traits<GFX928_32x32x32_I32I8I8I32_NT>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template<>
struct MMA_Traits<GFX928_16x16x64_I32U8U8I32_NT>
  : MMA_Traits<GFX928_16x16x64_I32I8I8I32_NT>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template<>
struct MMA_Traits<GFX928_16x32x64_I32U8U8I32_NT>
  : MMA_Traits<GFX928_16x32x64_I32I8I8I32_NT>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template<>
struct MMA_Traits<GFX928_32x16x64_I32U8U8I32_NT>
  : MMA_Traits<GFX928_32x16x64_I32I8I8I32_NT>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

template<>
struct MMA_Traits<GFX928_32x32x64_I32U8U8I32_NT>
  : MMA_Traits<GFX928_32x32x64_I32I8I8I32_NT>
{
  using ValTypeD = int32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;
};

// Composed MMAC operations with a 16x64 output tile.
template <>
struct MMA_Traits<GFX928_16x64x16_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;
  using Shape_MNK = Shape<_16, _64, _16>;
  using ThrID = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>, _4>,
                         Stride<Stride<_1, _64>, _16>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _4>>,
                         Stride<Stride<_4, _256>, Stride<_64, _1>>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _4>>,
                         Stride<Stride<_1, _64>, Stride<_16, _256>>>;
};

template <>
struct MMA_Traits<GFX928_16x64x16_F32BF16BF16F32_NT>
    : MMA_Traits<GFX928_16x64x16_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

template <>
struct MMA_Traits<GFX928_16x64x32_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;
  using Shape_MNK = Shape<_16, _64, _32>;
  using ThrID = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>, _8>,
                         Stride<Stride<_1, _128>, _16>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, Shape<_8, _4>>,
                         Stride<Stride<_4, _512>, Stride<_64, _1>>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _4>>,
                         Stride<Stride<_1, _64>, Stride<_16, _256>>>;
};

template <>
struct MMA_Traits<GFX928_16x64x32_F32BF16BF16F32_NT>
    : MMA_Traits<GFX928_16x64x32_F32F16F16F32_NT>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;
};

} // namespace hute
