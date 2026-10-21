/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma warning(disable : 4068) /* disable unknown pragma warnings for visual studio */

#pragma nv_diag_suppress boolean_controlling_expr_is_constant
#include <gtest/gtest.h>
#pragma nv_diag_warning boolean_controlling_expr_is_constant
#pragma warning(disable : 4503)

#include <cstdlib>
#include <string>

#include <hip/hip_runtime_api.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gets a Hip device
hipDeviceProp_t GetHipDevice();

/// Prints device properties
std::ostream &operator<<(std::ostream &out, hipDeviceProp_t const &device);

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Sets flags for Unit test
void FilterArchitecture();

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Reads environment variable `HYTLASS_UNIT_TEST_PROBLEM_COUNT` to control the number and order
//  of problem sizes run by HYTLASS unit tests
int HytlassUnitTestProblemCount();

/////////////////////////////////////////////////////////////////////////////////////////////////

// active test macro
#define HYTLASS_TEST_LEVEL_ACTIVE(LEVEL, NAME_STATIC, NAME_DYNAMIC, ...) \
  TEST(NAME_STATIC, L##LEVEL##_##NAME_DYNAMIC)                           \
  __VA_ARGS__

// disabled test macro
#define HYTLASS_TEST_LEVEL_DISABLED(LEVEL, NAME_STATIC, NAME_DYNAMIC, ...) \
  TEST(NAME_STATIC, DISABLED_L##LEVEL##_##NAME_DYNAMIC) {}

#if HYTLASS_TEST_LEVEL == 0
#define HYTLASS_TEST_L0(NAME_STATIC, NAME_DYNAMIC, ...) HYTLASS_TEST_LEVEL_ACTIVE(0, NAME_STATIC, NAME_DYNAMIC, __VA_ARGS__)
#define HYTLASS_TEST_L1(NAME_STATIC, NAME_DYNAMIC, ...) HYTLASS_TEST_LEVEL_DISABLED(1, NAME_STATIC, NAME_DYNAMIC, __VA_ARGS__)
#define HYTLASS_TEST_L2(NAME_STATIC, NAME_DYNAMIC, ...) HYTLASS_TEST_LEVEL_DISABLED(2, NAME_STATIC, NAME_DYNAMIC, __VA_ARGS__)
#elif HYTLASS_TEST_LEVEL == 1
#define HYTLASS_TEST_L0(NAME_STATIC, NAME_DYNAMIC, ...) HYTLASS_TEST_LEVEL_ACTIVE(0, NAME_STATIC, NAME_DYNAMIC, __VA_ARGS__)
#define HYTLASS_TEST_L1(NAME_STATIC, NAME_DYNAMIC, ...) HYTLASS_TEST_LEVEL_ACTIVE(1, NAME_STATIC, NAME_DYNAMIC, __VA_ARGS__)
#define HYTLASS_TEST_L2(NAME_STATIC, NAME_DYNAMIC, ...) HYTLASS_TEST_LEVEL_DISABLED(2, NAME_STATIC, NAME_DYNAMIC, __VA_ARGS__)
#else
#define HYTLASS_TEST_L0(NAME_STATIC, NAME_DYNAMIC, ...) HYTLASS_TEST_LEVEL_ACTIVE(0, NAME_STATIC, NAME_DYNAMIC, __VA_ARGS__)
#define HYTLASS_TEST_L1(NAME_STATIC, NAME_DYNAMIC, ...) HYTLASS_TEST_LEVEL_ACTIVE(1, NAME_STATIC, NAME_DYNAMIC, __VA_ARGS__)
#define HYTLASS_TEST_L2(NAME_STATIC, NAME_DYNAMIC, ...) HYTLASS_TEST_LEVEL_ACTIVE(2, NAME_STATIC, NAME_DYNAMIC, __VA_ARGS__)
#endif

#if !defined(HYTLASS_TEST_UNIT_ENABLE_WARNINGS)
#define HYTLASS_TEST_UNIT_ENABLE_WARNINGS false
#endif

#include <hytlass/hytlass.h>
#include <hytlass/numeric_types.h>
#include <hytlass/trace.h>

/////////////////////////////////////////////////////////////////////////////////////////////////
