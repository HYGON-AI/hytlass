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


set(HYTLASS_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_custom_target(hytlass_examples)
add_custom_target(test_examples)

function(hytlass_example_add_executable NAME)

  set(options)
  set(oneValueArgs DISABLE_TESTS)
  set(multiValueArgs DEPENDS DEPENDEES TEST_COMMAND_OPTIONS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED __DISABLE_TESTS)
    set(__DISABLE_TESTS OFF)
  endif()

  hytlass_add_executable(${NAME} ${__UNPARSED_ARGUMENTS} BATCH_SOURCES OFF)

  add_dependencies(hytlass_examples ${NAME})

  target_link_libraries(
    ${NAME}
    PRIVATE
    HYTLASS
    hytlass_tools_util_includes
    $<$<BOOL:${HYTLASS_ENABLE_HIPBLAS}>:hip::hipblas>
    hip
    )

  target_include_directories(
    ${NAME}
    PRIVATE
    ${HYTLASS_EXAMPLES_COMMON_SOURCE_DIR}
    ${HYTLASS_EXAMPLES_UTILS_DIR}
    )

  install(
    TARGETS ${NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

  hytlass_add_executable_tests(
    test_examples_${NAME} ${NAME}
    DEPENDS ${__DEPENDS}
    DEPENDEES test_examples ${__DEPENDEES}
    TEST_COMMAND_OPTIONS ${__TEST_COMMAND_OPTIONS}
    DISABLE_EXECUTABLE_INSTALL_RULE
    DISABLE_TESTS ${__DISABLE_TESTS}
    )

endfunction()

foreach(EXAMPLE
        00_hytlass_basic_gemm
        01_hytlass_serial_splitk_gemm
        02_hytlass_parallel_splitk_gemm
        03_hytlass_streamk_gemm
        04_hytlass_batch_gemm
        05_hytlass_group_gemm
  06_hute_streamk   
  07_hute_batch_gemm
        08_hytlass_fused_gemm
        09_hytlass_tensorop_conv2d
        10_hytlass_tensorop_wgrad_split_k
        11_hytlass_tensorop_group_conv
  12_depthwise_simt_conv2dfprop
        13_hytlass_tensorop_fused_conv2d_fprop
  14_gather_scatter_fusion
  15_hute_group_gemm
        16_hytlass_gemm_softmax
  17_ell_block_sparse_gemm
  18_gemm_with_abs_max
  hute
  )

  add_subdirectory(${EXAMPLE})

endforeach()
