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

#include "hute/tensor.hpp"
#include "hute/atom/mma_atom.hpp"
#include "hute/atom/copy_atom.hpp"

#include "hytlass/hytlass.h"
#include "hytlass/gemm/gemm.h"
#include "hytlass/arch/arch.h"
#include "hytlass/arch/mma.h"
#include "hytlass/layout/layout.h"
#include "hytlass/gemm/dispatch_policy.hpp"
#include "hytlass/gemm/gemm.h"
#include "hytlass/gemm/collective/collective_mma.hpp"
#include "hytlass/epilogue/collective/collective_builder.hpp"

#include "hytlass/epilogue/collective/default_epilogue.hpp"
#include "hytlass/epilogue/thread/linear_combination.h"

namespace hytlass {
namespace gemm {
namespace device {
using namespace hute;

// This type is only intended to demonstrate porting 2.x kernels to 3.0
template<
  class OperatorClass, class ArchTag,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToHytlass3Types {
  static_assert(sizeof(ElementA) == 0, "No valid DefaultGemmConfigurationToHytlass3Types configuration exists.");
};

///////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpGfx928_OperandA;

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpGfx928_OperandB;

//
// F16: 128-by-128-by-32
//

/// Operand A - Row-major (K-Major)
template <>
struct DefaultGemm_TensorOpGfx928_OperandA<half_t, layout::RowMajor, 8, 32>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,2,4>{},
                Layout<Shape<Shape<_2, _8>, _32>,
                       Stride<Stride<_32, _64>, _1>>{}));
  using SmemCopyAtom = Copy_Atom<UniversalCopy<hute::uint64_t>, half_t>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint128_t>, half_t>{},
                    Layout<Shape <_64, _4>,
                           Stride< _4, _1>>{},
                    Layout<Shape < _1, _8>>{}));
};

/// Operand A - Column-major (M-major)
template <int SizeK>
struct DefaultGemm_TensorOpGfx928_OperandA<half_t, layout::ColumnMajor, 8, SizeK>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape <_64, _16>,
                       Stride< _1, _64>>{}));
  using SmemCopyAtom = Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, half_t>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint128_t>, half_t>{},
                    Layout<Shape <_16, _16>,
                           Stride< _1, _16>>{},
                    Layout<Shape < _8, _1>>{}));
};

// Because the F32F16 TiledMMA is A-B symmetric, we can reuse the DefaultOperands

// Operand B - Column-Major (K-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpGfx928_OperandB<half_t, layout::ColumnMajor, Alignment, SizeK>
     : DefaultGemm_TensorOpGfx928_OperandA<half_t, layout::RowMajor,    Alignment, SizeK>
{};

// Operand B - Row-Major (N-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpGfx928_OperandB<half_t, layout::RowMajor,    Alignment, SizeK>
     : DefaultGemm_TensorOpGfx928_OperandA<half_t, layout::ColumnMajor, Alignment, SizeK>
{};

//
// F16: 128-by-128-by-16 (small k-block)
//

/// Operand A - Row-major (K-Major)
template <>
struct DefaultGemm_TensorOpGfx928_OperandA<half_t, layout::RowMajor, 8, 16>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<2,2,4>{},
                Layout<Shape<Shape<_4, _4>, _16>,
                       Stride<Stride<_16, _64>, _1>>{}));
  using SmemCopyAtom = Copy_Atom<UniversalCopy<hute::uint64_t>, half_t>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint128_t>, half_t>{},
                    Layout<Shape <_128, _2>,
                           Stride< _2, _1>>{},
                    Layout<Shape < _1, _8>>{}));
};

}

///////////////////////////////////////////////////////////////////////////////

// Gfx928 MMA F32F16
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToHytlass3Types<
    arch::OpClassTensorOp, arch::Gfx928,
    half_t, LayoutA,
    half_t, LayoutB,
    float, LayoutC,
    float>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopDispatch<2, arch::Gfx928>;

  using UnderlyingStrideA = hute::remove_pointer_t<TagToStrideA_t<LayoutA>>;
  static constexpr bool IsGroupedGemmKernel =
      !hute::is_same_v<UnderlyingStrideA, TagToStrideA_t<LayoutA>>;
  static constexpr auto GemmLayoutType =
      hytlass::detail::get_gemm_layout<TagToStrideA_t<LayoutA>,
                                       TagToStrideB_t<LayoutB>,
                                       IsGroupedGemmKernel>();

  static constexpr int kAlignmentA = 8;
  using DefaultOperandA = detail::DefaultGemm_TensorOpGfx928_OperandA<
    half_t, LayoutA, kAlignmentA, 32>;

  static constexpr int kAlignmentB = 8;
  using DefaultOperandB = detail::DefaultGemm_TensorOpGfx928_OperandB<
    half_t, LayoutB, kAlignmentB, 32>;
  using MmaType = std::conditional_t<
    GemmLayoutType == hytlass::detail::GemmLayout::NT,
    GFX928_32x32x16_F32F16F16F32_NT_ALT,
    GFX928_32x32x16_F32F16F16F32_NT
  >;
  using SmemCopyAtomNT = Copy_Atom<GFX928_DS_READ_DS_M32x16_B16_ALT, half_t>;
  using SmemCopyTypeA = std::conditional_t<
    GemmLayoutType == hytlass::detail::GemmLayout::NT,
    SmemCopyAtomNT,
    typename DefaultOperandA::SmemCopyAtom
  >;
  using SmemCopyTypeB = std::conditional_t<
    GemmLayoutType == hytlass::detail::GemmLayout::NT,
    SmemCopyAtomNT,
    typename DefaultOperandB::SmemCopyAtom
  >;
  using TiledMma = TiledMMA<
      MMA_Atom<MmaType>,
      Layout<Shape<_2,_2,_1>>>;       

  // A
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = SmemCopyTypeA;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = SmemCopyTypeB;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // C
  using StrideType = TagToStrideC_t<LayoutC>;
  static constexpr bool is_mn_major_result = hytlass::gemm::detail::is_mn_major<StrideType>();
  using SmemLayoutAtomC = std::conditional_t<
      is_mn_major_result,
      decltype(composition(
          Swizzle<2, 3, 3>{},
          make_layout(make_shape(hute::tile_size<0>(TiledMma{}), hute::tile_size<1>(TiledMma{})),
                      make_stride(Int<1>{}, hute::tile_size<0>(TiledMma{}))))),
      decltype(composition(
          Swizzle<2, 3, 3>{},
          make_layout(make_shape(hute::tile_size<0>(TiledMma{}), hute::tile_size<1>(TiledMma{})),
                      make_stride(hute::tile_size<1>(TiledMma{}), Int<1>{}))))>;
  using R2SCopyAtomC = Copy_Atom<DefaultCopy, float>;
  using S2RTiledCopyC = std::conditional_t<
      is_mn_major_result,
      decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                               Layout<Shape<_16, _16>, Stride<_1, _16>>{},
                               Layout<Shape<_4, _1>>{})),
      decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                               Layout<Shape<_64, _4>, Stride<_4, _1>>{},
                               Layout<Shape<_16, _4>>{}))>;
  using R2GCopyAtomC = Copy_Atom<UniversalCopy<float>, float>;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    half_t, TagToStrideA_t<LayoutA>,
    half_t, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, hute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, hute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::Epilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<float, 1, float, float>,
    SmemLayoutAtomC,
    R2SCopyAtomC, 
    S2RTiledCopyC,
    R2GCopyAtomC>;
};

///////////////////////////////////////////////////////////////////////////////

namespace detail {

//
// TF32: 128-by-128-by-kblock (kBlock = 8, 16)
//

/// Operand A - Row-major  (K-major) (kBlock = 16)
template <>
struct DefaultGemm_TensorOpGfx928_OperandA<tfloat32_t, layout::RowMajor, 4, 16>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,2,3>{},
                Layout<Shape <_32, _8>,
                       Stride<_1, _32>>{}));
  using SmemCopyAtom = Copy_Atom<UniversalCopy<float>, float>;

  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint128_t>, tfloat32_t>{},
                    Layout<Shape <_64, _4>,
                           Stride< _4, _1>>{},
                    Layout<Shape < _1, _4>>{}));
};

/// Operand A - Row-major  (K-major) (kBlock = 8)
template <>
struct DefaultGemm_TensorOpGfx928_OperandA<tfloat32_t, layout::RowMajor, 4, 8>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,2,3>{},
                Layout<Shape <_32, _8>,
                       Stride<_1, _32>>{}));
  using SmemCopyAtom = Copy_Atom<UniversalCopy<float>, tfloat32_t>;
  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint128_t>, tfloat32_t>{},
                    Layout<Shape <_128, _2>,
                           Stride< _2, _1>>{},
                    Layout<Shape < _1, _4>>{}));
};

/// Operand A - Column-major  (M-major)
template <int SizeK>
struct DefaultGemm_TensorOpGfx928_OperandA<tfloat32_t, layout::ColumnMajor, 4, SizeK>
{
  // Smem
  using SmemLayoutAtom = decltype(
    composition(Swizzle<3,2,3>{},
                Layout<Shape <_32, _8>,
                       Stride< _1, _32>>{}));
  using SmemCopyAtom = Copy_Atom<UniversalCopy<float>, tfloat32_t>;
  // Gmem
  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint128_t>, tfloat32_t>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1, _32>>{},
                    Layout<Shape < _4, _1>>{}));
};

// Because the TF32 TiledMMA is A-B symmetric, we can reuse the DefaultOperands

// Operand B - Column-Major  (K-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpGfx928_OperandB<tfloat32_t, layout::ColumnMajor, Alignment, SizeK>
     : DefaultGemm_TensorOpGfx928_OperandA<tfloat32_t, layout::RowMajor,    Alignment, SizeK>
{};

// Operand B - Row-Major  (N-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpGfx928_OperandB<tfloat32_t, layout::RowMajor,    Alignment, SizeK>
     : DefaultGemm_TensorOpGfx928_OperandA<tfloat32_t, layout::ColumnMajor, Alignment, SizeK>
{};

}

///////////////////////////////////////////////////////////////////////////////

// Gfx928 MMA F32TF32
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToHytlass3Types<
    arch::OpClassTensorOp, arch::Gfx928,
    tfloat32_t, LayoutA,
    tfloat32_t, LayoutB,
    float, LayoutC,
    float>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopDispatch<2, arch::Gfx928>;
  using TiledMma = TiledMMA<
      MMA_Atom<GFX928_16x16x8_F32TF32TF32F32_NT>,
      Layout<Shape<_2,_2,_1>>>;                          

  // A
  static constexpr int kAlignmentA = 4;
  using DefaultOperandA = detail::DefaultGemm_TensorOpGfx928_OperandA<
    tfloat32_t, LayoutA, kAlignmentA, 16>;
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
  using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 4;
  using DefaultOperandB = detail::DefaultGemm_TensorOpGfx928_OperandB<
    tfloat32_t, LayoutB, kAlignmentB, 16>;
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
  using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // C
  using StrideType = TagToStrideC_t<LayoutC>;
  static constexpr bool is_mn_major_result = hytlass::gemm::detail::is_mn_major<StrideType>();
  using SmemLayoutAtomC = std::conditional_t<
      is_mn_major_result,
      decltype(composition(
          Swizzle<2, 3, 3>{},
          make_layout(make_shape(hute::tile_size<0>(TiledMma{}), hute::tile_size<1>(TiledMma{})),
                      make_stride(Int<1>{}, hute::tile_size<0>(TiledMma{}))))),
      decltype(composition(
          Swizzle<2, 3, 3>{},
          make_layout(make_shape(hute::tile_size<0>(TiledMma{}), hute::tile_size<1>(TiledMma{})),
                      make_stride(hute::tile_size<1>(TiledMma{}), Int<1>{}))))>;
  using R2SCopyAtomC = Copy_Atom<DefaultCopy, float>;
  using S2RTiledCopyC = std::conditional_t<
      is_mn_major_result,
      decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                               Layout<Shape<_16, _16>, Stride<_1, _16>>{},
                               Layout<Shape<_4, _1>>{})),
      decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                               Layout<Shape<_32, _8>, Stride<_8, _1>>{},
                               Layout<Shape<_16, _4>>{}))>;
  using R2GCopyAtomC = Copy_Atom<UniversalCopy<uint128_t>, float>;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    tfloat32_t, TagToStrideA_t<LayoutA>,
    tfloat32_t, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, hute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, hute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::Epilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<float, 4, float, float>,
    SmemLayoutAtomC,
    R2SCopyAtomC, 
    S2RTiledCopyC,
    R2GCopyAtomC>;
};

///////////////////////////////////////////////////////////////////////////////

// Gfx928 MMA INT32INT8
template <typename LayoutC>
struct DefaultGemmConfigurationToHytlass3Types<
    arch::OpClassTensorOp, arch::Gfx928,
    int8_t, hytlass::layout::RowMajor,
    int8_t, hytlass::layout::ColumnMajor,
    int32_t, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopDispatch<2, arch::Gfx928>;
  using TiledMma = TiledMMA<
      MMA_Atom<GFX928_32x32x32_I32I8I8I32_NT>,
      Layout<Shape<_2,_2,_1>>>; 

  // A (M,K)  K-major
  using SmemLayoutAtomA = decltype(
    composition(
      Swizzle<2,3,4>{},
      Layout<Shape<Shape<_4, _4>, _32>,
             Stride<Stride<_32, _128>, _1>>{}));
  static constexpr int kAlignmentA = 16;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint128_t>, int8_t>{},
                    Layout<Shape <_128, _2>,
                           Stride< _2, _1>>{},
                    Layout<Shape<_1, Int<kAlignmentA>>>{}));
  using SmemCopyAtomA = Copy_Atom<UniversalCopy<uint64_t>, int8_t>;  // ds_read_M works

  // B (N,K)  K-major
  using SmemLayoutAtomB = decltype(
    composition(
      Swizzle<2,3,4>{},
      Layout<Shape<Shape<_4, _4>, _32>,
             Stride<Stride<_32, _128>, _1>>{}));
  static constexpr int kAlignmentB = 16;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint128_t>, int8_t>{},
                    Layout<Shape <_128, _2>,
                           Stride< _2, _1>>{},
                    Layout<Shape<_1, Int<kAlignmentB>>>{}));
  using SmemCopyAtomB = Copy_Atom<UniversalCopy<uint64_t>, int8_t>;  // ds_read_M works

  // C (M, N) 
  using StrideType = TagToStrideC_t<LayoutC>;
  static constexpr bool is_mn_major_result = hytlass::gemm::detail::is_mn_major<StrideType>();
  using SmemLayoutAtomC = std::conditional_t<
      is_mn_major_result,
      decltype(composition(
          Swizzle<2, 3, 3>{},
          make_layout(make_shape(hute::tile_size<0>(TiledMma{}), hute::tile_size<1>(TiledMma{})),
                      make_stride(Int<1>{}, hute::tile_size<0>(TiledMma{}))))),
      decltype(composition(
          Swizzle<2, 3, 3>{},
          make_layout(make_shape(hute::tile_size<0>(TiledMma{}), hute::tile_size<1>(TiledMma{})),
                      make_stride(hute::tile_size<1>(TiledMma{}), Int<1>{}))))>;
  using R2SCopyAtomC = Copy_Atom<DefaultCopy, int32_t>;
  using S2RTiledCopyC = std::conditional_t<
      is_mn_major_result,
      decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, int32_t>{},
                               Layout<Shape<_2, _128>, Stride<_1, _2>>{},
                               Layout<Shape<_16, _1>>{})),
      decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, int32_t>{},
                               Layout<Shape<_128, _2>, Stride<_2, _1>>{},
                               Layout<Shape<_1, _16>>{}))>;
  using R2GCopyAtomC = Copy_Atom<UniversalCopy<int32_t>, int32_t>;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    int8_t, TagToStrideA_t<hytlass::layout::RowMajor>,
    int8_t, TagToStrideB_t<hytlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, hute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, hute::identity   // B
  >;

  using CollectiveEpilogue = epilogue::collective::Epilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
    SmemLayoutAtomC,
    R2SCopyAtomC, 
    S2RTiledCopyC,
    R2GCopyAtomC>;
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////////// SIMT TWO STAGE ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element, typename Layout, int ThreadCount, int ShapeM, int ShapeK>
struct DefaultGemm_Simt_OperandA;

///////////////////////////////////////////////////////////////////////////////

template <typename Element>
struct DefaultGemm_Simt_OperandA<Element, layout::ColumnMajor, 256, 128, 8>
{
  using SmemLayoutAtom = Layout<Shape <_128,  _8>,
                                Stride<  _8, _1>>;

  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<Element>, Element>{},
                    Layout<Shape <_32, _8>,
                           Stride< _1,_32>>{},
                    Layout<Shape<_1,_1>>{}));
};

template <typename Element>
struct DefaultGemm_Simt_OperandA<Element, layout::RowMajor, 256, 128, 8>
{
  using SmemLayoutAtom = Layout<Shape <_128,          _8>,
                                Stride<  Int<8 + 4>, _1>>;   // Padded

  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<Element>, Element>{},
                    Layout<Shape <_32, _8>,
                           Stride< _8, _1>>{},
                    Layout<Shape<_1,_1>>{}));

};

template <typename Element, typename Layout, int ThreadCount, int ShapeN, int ShapeK>
struct DefaultGemm_Simt_OperandB;

template <typename Element, int ThreadCount, int ShapeN, int ShapeK>
struct DefaultGemm_Simt_OperandB<Element, layout::ColumnMajor, ThreadCount, ShapeN, ShapeK>
     : DefaultGemm_Simt_OperandA<Element, layout::RowMajor,    ThreadCount, ShapeN, ShapeK> {};

template <typename Element, int ThreadCount, int ShapeN, int ShapeK>
struct DefaultGemm_Simt_OperandB<Element, layout::RowMajor,    ThreadCount, ShapeN, ShapeK>
     : DefaultGemm_Simt_OperandA<Element, layout::ColumnMajor, ThreadCount, ShapeN, ShapeK> {};

} // end namespace detail

// SIMT Two Stage
template <
  class ArchTag,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator>
struct DefaultGemmConfigurationToHytlass3Types<
    arch::OpClassSimt, ArchTag,
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator>
{
  using TileShape = Shape<_128, _128, _8>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopDispatch<2, ArchTag>;
  using TiledMma = TiledMMA<
      MMA_Atom<UniversalFMA<ElementAccumulator, ElementA, ElementB, ElementC>>,
      Layout<Shape<_16, _16, _1>>>;

  // A
  static constexpr int kAlignmentA = 1;
  using DefaultOperandA = detail::DefaultGemm_Simt_OperandA<ElementA, LayoutA, ThreadCount, 128, 8>;
  using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
  using SmemCopyAtomA   = typename DefaultOperandA::SmemCopyAtom;
  using GmemTiledCopyA  = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 1;
  using DefaultOperandB = detail::DefaultGemm_Simt_OperandB<ElementB, LayoutB, ThreadCount, 128, 8>;
  using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
  using SmemCopyAtomB   = typename DefaultOperandB::SmemCopyAtom;
  using GmemTiledCopyB  = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<LayoutA>,
    ElementB, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, hute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, hute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    hytlass::gemm::EpilogueDefault>;
};


//
// DP4A - int8    Proof-of-concept
//

// SIMT Two Stage TN - idp4a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToHytlass3Types<
    arch::OpClassSimt, ArchTag,
    int8_t, hytlass::layout::RowMajor,
    int8_t, hytlass::layout::ColumnMajor,
    ElementC, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopDispatch<2, ArchTag>;
  // NOTE: permuting MMA M mode lets us generate 128b smem loads (LDS.128) but has worst case bank conflicts
  using TiledMma = TiledMMA<
      MMA_Atom<GFX906_DP4A>,
      Layout<Shape<_16,_16,_1>>>;  // Tile of atoms (threads)

  // A (M,K)  K-major
  using ElementA = int8_t;
  // 40% from regular M and N major layout
  // using SmemLayoutAtomA = Layout<Shape <_128,_32>,
  //                                Stride<  _1,_128>>;
  // 80% from interleaved layouts
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;

  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 4;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint32_t>, ElementA>{},
                    Layout<Shape <_32, _8>,
                           Stride< _8, _1>>{},
                    Layout<Shape < _1, Int<kAlignmentA>>>{}));

  // B (N,K)  K-major
  using ElementB = int8_t;
  // 40% from regular M and N major layout
  // using SmemLayoutAtomB = Layout<Shape <_128,_32>,
  //                                Stride<  _1,_128>>;
  // 80% from interleaved layouts
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;

  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 4;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint32_t>, ElementB>{},
                    Layout<Shape <_32, _8>,
                           Stride< _8, _1>>{},
                    Layout<Shape < _1, Int<kAlignmentB>>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<hytlass::layout::RowMajor>,
    ElementB, TagToStrideB_t<hytlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, hute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, hute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, int32_t, int32_t>,
    hytlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Two Stage NN - idp4a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToHytlass3Types<
    arch::OpClassSimt, ArchTag,
    int8_t, hytlass::layout::ColumnMajor,
    int8_t, hytlass::layout::ColumnMajor,
    ElementC, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;

  using DispatchPolicy = MainloopDispatch<2, ArchTag>;

  using TiledMma = TiledMMA<
      MMA_Atom<GFX906_DP4A>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  M-major
  using ElementA = int8_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint8_t>, ElementA>{},
                    Layout<Shape <_8, _32>,
                           Stride< _1, _8>>{},
                    Layout<Shape < _1, Int<kAlignmentA>>>{}));

  // B (N,K)  K-major
  using ElementB = int8_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 4;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint32_t>, ElementB>{},
                    Layout<Shape <_32, _8>,
                           Stride< _8, _1>>{},
                    Layout<Shape < _1, Int<kAlignmentB>>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<hytlass::layout::ColumnMajor>,
    ElementB, TagToStrideB_t<hytlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, hute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, hute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, int32_t, int32_t>,
    hytlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Two Stage NT - idp4a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToHytlass3Types<
    arch::OpClassSimt, ArchTag,
    int8_t, hytlass::layout::ColumnMajor,
    int8_t, hytlass::layout::RowMajor,
    ElementC, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopDispatch<2, ArchTag>;
  using TiledMma = TiledMMA<
      MMA_Atom<GFX906_DP4A>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  M-major
  using ElementA = int8_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint8_t>, ElementA>{},
                    Layout<Shape <_8, _32>,
                           Stride< _1, _8>>{},
                    Layout<Shape < _1, Int<kAlignmentA>>>{}));

  // B (N,K)  N-major
  using ElementB = int8_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint8_t>, ElementB>{},
                    Layout<Shape <_8, _32>,
                           Stride< _1, _8>>{},
                    Layout<Shape < _1, Int<kAlignmentB>>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<hytlass::layout::ColumnMajor>,
    ElementB, TagToStrideB_t<hytlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, hute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, hute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, int32_t, int32_t>,
    hytlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Two Stage TT - idp4a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToHytlass3Types<
    arch::OpClassSimt, ArchTag,
    int8_t, hytlass::layout::RowMajor,
    int8_t, hytlass::layout::RowMajor,
    ElementC, LayoutC,
    int32_t>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopDispatch<2, ArchTag>;
  using TiledMma = TiledMMA<
      MMA_Atom<GFX906_DP4A>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  K-major
  using ElementA = int8_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 4;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint32_t>, ElementA>{},
                    Layout<Shape <_32, _8>,
                           Stride< _8, _1>>{},
                    Layout<Shape < _1, Int<kAlignmentA>>>{}));

  // B (N,K)  N-major
  using ElementB = int8_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_4,  _8>>,
                                 Stride<  _4, Stride<_1,_512>>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint8_t>, ElementB>{},
                    Layout<Shape <_8, _32>,
                           Stride< _1, _8>>{},
                    Layout<Shape < _1, Int<kAlignmentB>>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<hytlass::layout::RowMajor>,
    ElementB, TagToStrideB_t<hytlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, hute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, hute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, int32_t, int32_t>,
    hytlass::gemm::EpilogueDefault>;
};


//
// DP2A - fp16    Proof-of-concept
//

// SIMT Two Stage TN - idp2a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToHytlass3Types<
    arch::OpClassSimt, ArchTag,
    half_t, hytlass::layout::RowMajor,
    half_t, hytlass::layout::ColumnMajor,
    ElementC, LayoutC,
    float>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopDispatch<2, ArchTag>;
  // NOTE: permuting MMA M mode lets us generate 128b smem loads (LDS.128) but has worst case bank conflicts
  using TiledMma = TiledMMA<
      MMA_Atom<GFX906_DP2A>,
      Layout<Shape<_16,_16,_1>>>;  // Tile of atoms (threads)

  // A (M,K)  K-major
  using ElementA = half_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_2,  _16>>,
                                 Stride<  _2, Stride<_1,_256>>>;

  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 2;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint32_t>, ElementA>{},
                    Layout<Shape <_16, _16>,
                           Stride< _16, _1>>{},
                    Layout<Shape < _1, Int<kAlignmentA>>>{}));

  // B (N,K)  K-major
  using ElementB = half_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_2,  _16>>,
                                 Stride<  _2, Stride<_1,_256>>>;

  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 2;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint32_t>, ElementB>{},
                    Layout<Shape <_16, _16>,
                           Stride< _16, _1>>{},
                    Layout<Shape < _1, Int<kAlignmentB>>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<hytlass::layout::RowMajor>,
    ElementB, TagToStrideB_t<hytlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, hute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, hute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, float, float>,
    hytlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Two Stage NN - idp2a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToHytlass3Types<
    arch::OpClassSimt, ArchTag,
    half_t, hytlass::layout::ColumnMajor,
    half_t, hytlass::layout::ColumnMajor,
    ElementC, LayoutC,
    float>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;

  using DispatchPolicy = MainloopDispatch<2, ArchTag>;

  using TiledMma = TiledMMA<
      MMA_Atom<GFX906_DP2A>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  M-major
  using ElementA = half_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_2,  _16>>,
                                 Stride<  _2, Stride<_1,_256>>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::half_t>, ElementA>{},
                    Layout<Shape <_8, _32>,
                           Stride< _1, _8>>{},
                    Layout<Shape < _1, Int<kAlignmentA>>>{}));

  // B (N,K)  K-major
  using ElementB = half_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_2,  _16>>,
                                 Stride<  _2, Stride<_1,_256>>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 2;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint32_t>, ElementB>{},
                    Layout<Shape <_16, _16>,
                           Stride< _16, _1>>{},
                    Layout<Shape < _1, Int<kAlignmentB>>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<hytlass::layout::ColumnMajor>,
    ElementB, TagToStrideB_t<hytlass::layout::ColumnMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, hute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, hute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, float, float>,
    hytlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Two Stage NT - idp2a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToHytlass3Types<
    arch::OpClassSimt, ArchTag,
    half_t, hytlass::layout::ColumnMajor,
    half_t, hytlass::layout::RowMajor,
    ElementC, LayoutC,
    float>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopDispatch<2, ArchTag>;
  using TiledMma = TiledMMA<
      MMA_Atom<GFX906_DP2A>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  M-major
  using ElementA = half_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_2,  _16>>,
                                 Stride<  _2, Stride<_1,_256>>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 1;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::half_t>, ElementA>{},
                    Layout<Shape <_8, _32>,
                           Stride< _1, _8>>{},
                    Layout<Shape < _1, Int<kAlignmentA>>>{}));

  // B (N,K)  N-major
  using ElementB = half_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_2,  _16>>,
                                 Stride<  _2, Stride<_1,_256>>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::half_t>, ElementB>{},
                    Layout<Shape <_8, _32>,
                           Stride< _1, _8>>{},
                    Layout<Shape < _1, Int<kAlignmentB>>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<hytlass::layout::ColumnMajor>,
    ElementB, TagToStrideB_t<hytlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, hute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, hute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, float, float>,
    hytlass::gemm::EpilogueDefault>;
};

///////////////////////////////////////////////////////////////////////////////

// SIMT Two Stage TT - idp2a
template <
  class ArchTag,
  class ElementC, class LayoutC>
struct DefaultGemmConfigurationToHytlass3Types<
    arch::OpClassSimt, ArchTag,
    half_t, hytlass::layout::RowMajor,
    half_t, hytlass::layout::RowMajor,
    ElementC, LayoutC,
    float>
{
  using TileShape = Shape<_128, _128, _32>;
  static constexpr int ThreadCount = 256;
  using DispatchPolicy = MainloopDispatch<2, ArchTag>;
  using TiledMma = TiledMMA<
      MMA_Atom<GFX906_DP2A>,
      Layout<Shape<_16, _16, _1>>>;

  // A (M,K)  K-major
  using ElementA = half_t;
  using SmemLayoutAtomA = Layout<Shape <_128, Shape <_2,  _16>>,
                                 Stride<  _2, Stride<_1,_256>>>;
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;
  static constexpr int kAlignmentA = 2;
  using GmemTiledCopyA = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::uint32_t>, ElementA>{},
                    Layout<Shape <_16, _16>,
                           Stride< _16, _1>>{},
                    Layout<Shape < _1, Int<kAlignmentA>>>{}));

  // B (N,K)  N-major
  using ElementB = half_t;
  using SmemLayoutAtomB = Layout<Shape <_128, Shape <_2,  _16>>,
                                 Stride<  _2, Stride<_1,_256>>>;
  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  static constexpr int kAlignmentB = 1;
  using GmemTiledCopyB = decltype(
    make_tiled_copy(Copy_Atom<UniversalCopy<hute::half_t>, ElementB>{},
                    Layout<Shape <_8, _32>,
                           Stride< _1, _8>>{},
                    Layout<Shape < _1, Int<kAlignmentB>>>{}));

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, TagToStrideA_t<hytlass::layout::RowMajor>,
    ElementB, TagToStrideB_t<hytlass::layout::RowMajor>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, hute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, hute::identity   // B
  >;

  // Epilogue
  using CollectiveEpilogue = epilogue::collective::DefaultEpilogue<
    TagToStrideC_t<LayoutC>,
    TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<ElementC, 1, float, float>,
    hytlass::gemm::EpilogueDefault>;
};


///////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace hytlass
