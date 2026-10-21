# Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

find_package(Python3 3.5 COMPONENTS Interpreter REQUIRED)

#
# Sources for HYTLASS Profiler Tool
#
cmake_policy(SET CMP0112 NEW)
set(HYTLASS_TOOLS_PROFILER_SOURCES
  src/main.cpp
  src/hytlass_profiler.cu
  src/options.cu
  src/performance_report.cpp
  src/enumerated_types.cpp
  src/gpu_timer.cpp
  src/device_allocation.cu
  src/device_context.cu
  src/hipblas_helpers.cu                          
  src/problem_space.cpp
  src/operation_profiler.cu
  src/gemm_operation_profiler.cu
  src/conv2d_operation_profiler.cu
)

#
# Build target
#

hytlass_add_executable(
  hytlass_profiler
  ${HYTLASS_TOOLS_PROFILER_SOURCES}
)
add_executable(hygon::hytlass::profiler ALIAS hytlass_profiler)
set_target_properties(hytlass_profiler PROPERTIES EXPORT_NAME profiler)

#
# Include paths
#

target_include_directories(
  hytlass_profiler
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/include
  )

#
# Library dependencies
#

target_link_libraries(
  hytlass_profiler
  PRIVATE 
  hytlass_lib
  hytlass_tools_util_includes
  $<$<BOOL:${HYTLASS_ENABLE_HIPBLAS}>:hip::hipblas>
  # TODO: add dnn support
  # $<$<BOOL:${HYTLASS_ENABLE_HIPDNN}>:hygon::hipdnn>
  hip::galaxyhip
  # hip::hip_driver
  )

install(
  TARGETS hytlass_profiler
  EXPORT HygonHytlass
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

set(HYTLASS_PROFILER_TEST_COMMAND_OPTIONS_GEMM   --operation=Gemm       --providers=hytlass --verification-providers=hipblas,device      --junit-output=test_hytlass_profiler_gemm    --print-kernel-before-running=true)

hytlass_add_executable_tests(
  test_profiler hytlass_profiler
  DEPENDEES test_all
  TEST_COMMAND_OPTIONS
    GEMM
    CONV2D
  TEST_COMMAND_OPTIONS_PREFIX
    HYTLASS_PROFILER_TEST_COMMAND_OPTIONS_
  DISABLE_EXECUTABLE_INSTALL_RULE
  )

