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

include(CTest)

set(HYTLASS_UNIT_TEST_COMMON_DIR ${CMAKE_CURRENT_LIST_DIR}/common)

hytlass_add_library(
  hytlass_test_unit_infra 
  OBJECT
  common/filter_architecture.cpp
  )

target_link_libraries(
  hytlass_test_unit_infra
  PUBLIC
  HYTLASS
  hytlass_tools_util_includes
  $<$<BOOL:${HYTLASS_ENABLE_HIPBLAS}>:hip::hipblas>
  GTest::gtest
  hip::hiprtc
  hip::galaxyhip
  )

hytlass_add_library(
  hytlass_test_unit_infra_lib
  OBJECT
  test_unit.cpp
  )

target_link_libraries(
  hytlass_test_unit_infra_lib
  PUBLIC
  hytlass_test_unit_infra
  )

set(HYTLASS_TEST_UNIT_RESULTS_CACHE_DIR ${CMAKE_CURRENT_LIST_DIR}/data/hashes)

function(hytlass_test_unit_add_executable NAME)

  set(options WITHOUT_HIP)
  set(oneValueArgs)
  set(multiValueArgs TEST_SETS_SUPPORTED EXTRA_INCLUDE_DIRS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  hytlass_add_executable(${NAME} ${__UNPARSED_ARGUMENTS} BATCH_SOURCES OFF)

  target_compile_definitions(${NAME} PUBLIC HYTLASS_TARGET_NAME="${NAME}")

  target_include_directories(
    ${NAME}
    PRIVATE
    ${HYTLASS_UNIT_TEST_COMMON_DIR}
    ${__EXTRA_INCLUDE_DIRS}
  )
  if (__WITHOUT_HIP)
    # Avoid HIP dependencies for host-only unit tests that provide the
    # WITHOUT_HIP argument.
    target_link_libraries(
      ${NAME}
      PUBLIC
      GTest::gtest
    )
  else()
    target_link_libraries(
      ${NAME}
      PRIVATE
      hytlass_test_unit_infra
      hytlass_test_unit_infra_lib
    )
  endif()

  if (HYTLASS_ENABLE_OPENMP_TESTS AND OpenMP_CXX_FOUND)
    target_link_libraries(${NAME} PRIVATE OpenMP::OpenMP_CXX)
  endif()

  target_compile_options(${NAME} PUBLIC -I${CMAKE_SOURCE_DIR}/googletest/googletest/include)
  string(REGEX REPLACE hytlass_ "" NAME_STEM ${NAME})

  set(RESULT_CACHE_FILE "${HYTLASS_TEST_UNIT_RESULTS_CACHE_DIR}/cached_results_${NAME}.txt")

  if (EXISTS ${RESULT_CACHE_FILE})
    set(RESULT_CACHE_FILE_ARGS RESULT_CACHE_FILE ${RESULT_CACHE_FILE})
  endif()

  set(HYTLASS_TEST_UNIT_TEST_COMMAND_OPTIONS --gtest_output=xml:${NAME_STEM}.gtest.xml)

  hytlass_add_executable_tests(
    ${NAME_STEM} ${NAME}
    TEST_SETS_SUPPORTED ${__TEST_SETS_SUPPORTED}
    TEST_COMMAND_OPTIONS HYTLASS_TEST_UNIT_TEST_COMMAND_OPTIONS
    ${RESULT_CACHE_FILE_ARGS}
    )

endfunction()
add_custom_target(hytlass_test_unit)
add_custom_target(test_unit)

set(SUBDIRS
  core
  hute
  gemm
  conv
  layout
  transform
  epilogue
  reduction
  util
  substrate
  )


if(TARGET hip::hiprtc AND TARGET hip::hip_driver)
 set(HYTLASS_HIPRTC_ENABLE_INIT ON)
 message(STATUS "hiprtc enbale")
else()
 set(HYTLASS_HIPRTC_ENABLE_INIT OFF)
 message(STATUS "hiprtc not enbale")
endif()  

set(HYTLASS_HIPRTC_ENABLE ${HYTLASS_HIPRTC_ENABLE_INIT} CACHE BOOL "Enable HIPRTC support")

if (HYTLASS_HIPRTC_ENABLE)
 list(APPEND SUBDIRS nvrtc)
endif()


foreach(SUBDIR ${SUBDIRS})

  add_subdirectory(${SUBDIR})
  add_dependencies(hytlass_test_unit hytlass_test_unit_${SUBDIR})
  add_dependencies(test_unit test_unit_${SUBDIR})

endforeach()

