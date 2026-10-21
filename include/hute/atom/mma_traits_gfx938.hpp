/***************************************************************************************************
 * Copyright (c) 2026 Hygon Information Technology Co., Ltd. All rights reserved.
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hute/arch/mma_gfx938.hpp>
#include <hute/atom/mma_traits.hpp>
#include <hute/layout.hpp>

namespace hute {

template <>
struct MMA_Traits<GFX938_16x32x32_F32F8F8F32E4M3E4M3_NT_LIT> {
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;
  using Shape_MNK = Shape<_16, _32, _32>;
  using ThrID = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>, _8>,
                         Stride<Stride<_1, _128>, _16>>;
  // Each packed B register contains eight K values for one N coordinate.
  using BLayout = Layout<Shape<Shape<_16, _4>, Shape<_8, _2>>,
                         Stride<Stride<_1, _256>, Stride<_32, _16>>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_4, _2>>,
                         Stride<Stride<_1, _64>, Stride<_16, _256>>>;
};

template <>
struct MMA_Traits<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;
  using Shape_MNK = Shape<_16, _64, _64>;
  using ThrID = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>, _16>,
                         Stride<Stride<_1, _256>, _16>>;
  using BLayout = Layout<Shape<Shape<_16, _4>, Shape<_16, Shape<_2, _2>>>,
                         Stride<Stride<_2, _1024>, Stride<_64, Stride<_1, _32>>>>;
  using CLayout = Layout<Shape<Shape<_16, _4>, Shape<_8, _2>>,
                         Stride<Stride<_1, _128>, Stride<_16, _512>>>;
};

} // namespace hute
