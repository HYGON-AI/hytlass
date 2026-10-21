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
/*! \file
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "hytlass/hytlass.h"
#include "hytlass/gemm/device/gemm.h"
#include "hytlass/numeric_types.h"

#include "../../common/hytlass_unit_test.h"

#include "hytlass/util/host_tensor.h"
#include "hytlass/util/tensor_view_io.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/gemm.h"

#include "testbed.h"

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   16
//     Warps / Block:   1 x   2
//       Threadblock:  16 x  64 x  8
HYTLASS_TEST_L0(GFX906_device_igemm_tt, 16x64x8_16x32x1_4x2_4x16_1x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<16, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<16, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   16
//     Warps / Block:   1 x   1
//       Threadblock:  16 x  64 x  8
HYTLASS_TEST_L1(GFX906_device_igemm_tt, 16x64x8_16x64x1_4x4_4x16_1x1, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<16, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<16, 64, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   2
//    Threads / Warp:   4 x   16
//     Warps / Block:   1 x   2
//       Threadblock:  32 x  64 x  8
HYTLASS_TEST_L1(GFX906_device_igemm_tt, 32x64x8_32x32x1_8x2_4x16_1x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<32, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   16
//     Warps / Block:   1 x   1
//       Threadblock:  32 x  64 x  8
HYTLASS_TEST_L1(GFX906_device_igemm_tt, 32x64x8_32x64x1_8x4_4x16_1x1, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<32, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 64, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   8 x   8
//     Warps / Block:   1 x   2
//       Threadblock:  64 x  64 x  8
HYTLASS_TEST_L1(GFX906_device_igemm_tt, 64x64x8_64x32x1_8x4_8x8_1x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<64, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   2 x   2
//    Threads / Warp:   4 x   16
//     Warps / Block:   1 x   2
//       Threadblock:   8 x  64 x  8
HYTLASS_TEST_L1(GFX906_device_igemm_tt, 8x64x8_8x32x1_2x2_4x16_1x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<8, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<8, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   16
//     Warps / Block:   1 x   2
//       Threadblock:  16 x  64 x  8
HYTLASS_TEST_L2(GFX906_device_igemm_tt, 16x64x8_16x32x1_4x2_4x16_1x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<16, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<16, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   16
//     Warps / Block:   1 x   2
//       Threadblock:  16 x 128 x  8
HYTLASS_TEST_L1(GFX906_device_igemm_tt, 16x128x8_16x64x1_4x4_4x16_1x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<16, 128, 8>;
    using WarpShape = hytlass::gemm::GemmShape<16, 64, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   8 x   8
//     Warps / Block:   1 x   4
//       Threadblock:  32 x  64 x  8
HYTLASS_TEST_L2(GFX906_device_igemm_tt, 32x64x8_32x16x1_4x2_8x8_1x4, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<32, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   2
//    Threads / Warp:   4 x   16
//     Warps / Block:   1 x   2
//       Threadblock:  32 x  64 x  8
HYTLASS_TEST_L2(GFX906_device_igemm_tt, 32x64x8_32x32x1_8x2_4x16_1x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<32, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   16
//     Warps / Block:   1 x   2
//       Threadblock:  32 x 128 x  8
HYTLASS_TEST_L1(GFX906_device_igemm_tt, 32x128x8_32x64x1_8x4_4x16_1x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<32, 128, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 64, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   16
//     Warps / Block:   2 x   2
//       Threadblock:  32 x  64 x  8
HYTLASS_TEST_L2(GFX906_device_igemm_tt, 32x64x8_16x32x1_4x2_4x16_2x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<32, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<16, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   2
//    Threads / Warp:   4 x   16
//     Warps / Block:   2 x   2
//       Threadblock:  64 x  64 x  8
HYTLASS_TEST_L2(GFX906_device_igemm_tt, 64x64x8_32x32x1_8x2_4x16_2x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   16
//     Warps / Block:   2 x   1
//       Threadblock:  64 x  64 x  8
HYTLASS_TEST_L2(GFX906_device_igemm_tt, 64x64x8_32x64x1_8x4_4x16_2x1, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 64, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   4
//    Threads / Warp:   4 x   16
//     Warps / Block:   2 x   2
//       Threadblock:  32 x 128 x  8
HYTLASS_TEST_L2(GFX906_device_igemm_tt, 32x128x8_16x64x1_4x4_4x16_2x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<32, 128, 8>;
    using WarpShape = hytlass::gemm::GemmShape<16, 64, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   8 x   8
//     Warps / Block:   2 x   2
//       Threadblock:  64 x  32 x  8
HYTLASS_TEST_L2(GFX906_device_igemm_tt, 64x32x8_32x16x1_4x2_8x8_2x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   16
//     Warps / Block:   2 x   2
//       Threadblock:  64 x 128 x  8
HYTLASS_TEST_L1(GFX906_device_igemm_tt, 64x128x8_32x64x1_8x4_4x16_2x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<64, 128, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 64, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   8 x   8
//     Warps / Block:   2 x   2
//       Threadblock: 128 x  64 x  8
HYTLASS_TEST_L0(GFX906_device_igemm_tt, 128x64x8_64x32x1_8x4_8x8_2x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<128, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<64, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   8 x   8
//     Warps / Block:   2 x   4
//       Threadblock:  64 x  64 x  8
HYTLASS_TEST_L2(GFX906_device_igemm_tt, 64x64x8_32x16x1_4x2_8x8_2x4, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 16, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   2
//    Threads / Warp:   4 x   16
//     Warps / Block:   2 x   4
//       Threadblock:  64 x 128 x  8
HYTLASS_TEST_L2(GFX906_device_igemm_tt, 64x128x8_32x32x1_8x2_4x16_2x4, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<64, 128, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   16
//     Warps / Block:   2 x   4
//       Threadblock:  64 x 256 x  8
HYTLASS_TEST_L1(GFX906_device_igemm_tt, 64x256x8_32x64x1_8x4_4x16_2x4, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<64, 256, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 64, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   8 x   8
//     Warps / Block:   2 x   4
//       Threadblock: 128 x 128 x  8
HYTLASS_TEST_L1(GFX906_device_igemm_tt, 128x128x8_64x32x1_8x4_8x8_2x4, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 8>;
    using WarpShape = hytlass::gemm::GemmShape<64, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   4 x   2
//    Threads / Warp:   4 x   16
//     Warps / Block:   4 x   2
//       Threadblock:  64 x  64 x  8
HYTLASS_TEST_L2(GFX906_device_igemm_tt, 64x64x8_16x32x1_4x2_4x16_4x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<16, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   2
//    Threads / Warp:   4 x   16
//     Warps / Block:   4 x   2
//       Threadblock: 128 x  64 x  8
HYTLASS_TEST_L1(GFX906_device_igemm_tt, 128x64x8_32x32x1_8x2_4x16_4x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<128, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   4 x   16
//     Warps / Block:   4 x   2
//       Threadblock: 128 x 128 x  8
HYTLASS_TEST_L2(GFX906_device_igemm_tt, 128x128x8_32x64x1_8x4_4x16_4x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 64, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   4
//    Threads / Warp:   8 x   8
//     Warps / Block:   4 x   2
//       Threadblock: 256 x  64 x  8
HYTLASS_TEST_L1(GFX906_device_igemm_tt, 256x64x8_64x32x1_8x4_8x8_4x2, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<256, 64, 8>;
    using WarpShape = hytlass::gemm::GemmShape<64, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

////////////////////////////////////////////////////////////////////////////////
// Elements / Thread:   8 x   2
//    Threads / Warp:   4 x   16
//     Warps / Block:   4 x   4
//       Threadblock: 128 x 128 x  8
HYTLASS_TEST_L2(GFX906_device_igemm_tt, 128x128x8_32x32x1_8x2_4x16_4x4, {
    using precision = int;
    using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 8>;
    using WarpShape = hytlass::gemm::GemmShape<32, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = hytlass::gemm::device::Gemm<
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision, hytlass::layout::RowMajor,
        precision,
        hytlass::arch::OpClassSimt,
        hytlass::arch::Gfx906,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;
    EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )


////////////////////////////////////////////////////////////////////////////////

// dp4a
TEST(GFX906_Device_Gemm_s8t_s8t_simt_op_dp4a, 64x64x16_64x64x4) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::RowMajor,
    ElementOutput,
    hytlass::layout::RowMajor,
    int32_t,
    hytlass::arch::OpClassSimt,
    hytlass::arch::Gfx906,
    hytlass::gemm::GemmShape<64, 64, 16>,
    hytlass::gemm::GemmShape<64, 64, 16>,
    hytlass::gemm::GemmShape<1, 1, 4>,
    hytlass::epilogue::thread::LinearCombinationClamp<                \
      ElementOutput,                                                  \
      1,                                                              \
      ElementAccumulator,                                             \
      ElementCompute                                                  \
    >, 
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(GFX906_Device_Gemm_s8t_s8t_simt_op_dp4a, 128x256x32_64x128x4) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::RowMajor,
    ElementOutput,
    hytlass::layout::RowMajor,
    int32_t,
    hytlass::arch::OpClassSimt,
    hytlass::arch::Gfx906,
    hytlass::gemm::GemmShape<128, 256, 32>,
    hytlass::gemm::GemmShape<64, 128, 32>,
    hytlass::gemm::GemmShape<1, 1, 4>,
    hytlass::epilogue::thread::LinearCombinationClamp<                \
      ElementOutput,                                                  \
      1,                                                              \
      ElementAccumulator,                                             \
      ElementCompute                                                  \
    >, 
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(GFX906_Device_Gemm_s8t_s8t_simt_op_dp4a, 256x128x64_64x64x4) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::RowMajor,
    ElementOutput,
    hytlass::layout::RowMajor,
    int32_t,
    hytlass::arch::OpClassSimt,
    hytlass::arch::Gfx906,
    hytlass::gemm::GemmShape<256, 128, 64>,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<1, 1, 4>,
    hytlass::epilogue::thread::LinearCombinationClamp<                \
      ElementOutput,                                                  \
      1,                                                              \
      ElementAccumulator,                                             \
      ElementCompute                                                  \
    >, 
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(GFX906_Device_Gemm_s8t_s8t_simt_op_dp4a, 256x256x16_128x128x4) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::RowMajor,
    ElementOutput,
    hytlass::layout::RowMajor,
    int32_t,
    hytlass::arch::OpClassSimt,
    hytlass::arch::Gfx906,
    hytlass::gemm::GemmShape<256, 256, 16>,
    hytlass::gemm::GemmShape<128, 128, 16>,
    hytlass::gemm::GemmShape<1, 1, 4>,
    hytlass::epilogue::thread::LinearCombinationClamp<                \
      ElementOutput,                                                  \
      1,                                                              \
      ElementAccumulator,                                             \
      ElementCompute                                                  \
    >, 
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

// dp4a sliced_k
TEST(GFX906_Device_Gemm_s8t_s8t_simt_op_dp4a_sliced_k, 32x32x128_32x32x4) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = hytlass::gemm::device::Gemm<
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::RowMajor,
    ElementOutput,
    hytlass::layout::RowMajor,
    int32_t,
    hytlass::arch::OpClassSimt,
    hytlass::arch::Gfx906,
    hytlass::gemm::GemmShape<32, 64, 128>,
    hytlass::gemm::GemmShape<32, 64, 32>,
    hytlass::gemm::GemmShape<1, 1, 4>,
    hytlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(GFX906_Device_Gemm_s8t_s8t_simt_op_dp4a_sliced_k, 32x64x128_32x32x4) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Gemm = hytlass::gemm::device::Gemm<
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::RowMajor,
    ElementOutput,
    hytlass::layout::RowMajor,
    int32_t,
    hytlass::arch::OpClassSimt,
    hytlass::arch::Gfx906,
    hytlass::gemm::GemmShape<32, 64, 128>,
    hytlass::gemm::GemmShape<32, 64, 64>,
    hytlass::gemm::GemmShape<1, 1, 4>,
    hytlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

