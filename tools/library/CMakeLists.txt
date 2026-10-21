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

cmake_policy(SET CMP0112 NEW)

include(GNUInstallDirs)

################################################################################

set(HYTLASS_BUILD_MONO_LIBRARY OFF CACHE BOOL 
  "Determines whether the hytlass library is generated as a single file or multiple files.")

################################################################################

add_library(hytlass_library_includes INTERFACE)
add_library(hygon::hytlass::library::includes ALIAS hytlass_library_includes)
set_target_properties(hytlass_library_includes PROPERTIES EXPORT_NAME library::includes)

target_include_directories(
  hytlass_library_includes
  INTERFACE
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

target_link_libraries(
  hytlass_library_includes 
  INTERFACE 
  HYTLASS
  hytlass_tools_util_includes
  )

install(
  TARGETS hytlass_library_includes
  EXPORT HygonHytlass
  )

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
  )

add_library(hytlass_library_internal_interface INTERFACE)
add_library(hygon::hytlass::library::obj_interface ALIAS hytlass_library_internal_interface)

target_include_directories(
  hytlass_library_internal_interface
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  )

target_link_libraries(
  hytlass_library_internal_interface
  INTERFACE
  hytlass_library_includes
  )

################################################################################

function(hytlass_add_hytlass_library)
#
# Generates static and shared libraries with the given SOURCES. The public CMake
# targets produces will be hytlass_library(_${SUFFIX})? and 
# hytlass_library(_${SUFFIX})?_static.
# 
# SUFFIX: An additional string to be joined to the default names. If suffix is given,
#   the generated libraries will be linked as a dependency of the main hytlass library.

  set(options)
  set(oneValueArgs SUFFIX)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(DEFAULT_NAME hytlass_library)

  set(__NAME ${DEFAULT_NAME})
  set(__OUTPUT_NAME hytlass)
  set(__EXPORT_NAME library)

  if (__SUFFIX)
    string(APPEND __NAME _${__SUFFIX})
    string(APPEND __OUTPUT_NAME _${__SUFFIX})
    string(APPEND __EXPORT_NAME _${__SUFFIX})
  endif()

  hytlass_add_library(
    ${__NAME}_objs
    OBJECT
    ${__UNPARSED_ARGUMENTS}
    )  

  target_link_libraries(${__NAME}_objs
    PUBLIC hytlass_library_includes
    PRIVATE hytlass_library_internal_interface
    )

  if (HYTLASS_BUILD_MONO_LIBRARY AND __SUFFIX)

    # If we're only building a single monolithic library then we
    # simply link the generated object files to the default library. 

    target_link_libraries(${DEFAULT_NAME} PRIVATE $<BUILD_INTERFACE:${__NAME}_objs>)
    target_link_libraries(${DEFAULT_NAME}_static PRIVATE $<BUILD_INTERFACE:${__NAME}_objs>)

  else()

    hytlass_add_library(
      ${__NAME} 
      SHARED
      EXPORT_NAME ${__EXPORT_NAME}
      ""
      )

    target_compile_features(${__NAME} INTERFACE cxx_std_17)
    
    set_target_properties(
      ${__NAME}
      PROPERTIES
      OUTPUT_NAME ${__OUTPUT_NAME}
      WINDOWS_EXPORT_ALL_SYMBOLS 1
      )
    
    target_link_libraries(
      ${__NAME}
      PUBLIC hytlass_library_includes
      PRIVATE $<BUILD_INTERFACE:${__NAME}_objs>
      # hip::hip_driver
      hip::galaxyhip
      )
    
    set_target_properties(${__NAME} PROPERTIES DEBUG_POSTFIX "${HYTLASS_LIBRARY_DEBUG_POSTFIX}")
    
    hytlass_add_library(
      ${__NAME}_static
      STATIC
      EXPORT_NAME ${__EXPORT_NAME}_static
      ""
      )

    target_compile_features(${__NAME}_static INTERFACE cxx_std_17)
    
    if (WIN32)
      set(STATIC_OUTPUT_NAME ${__OUTPUT_NAME}.static)
    else()
      set(STATIC_OUTPUT_NAME ${__OUTPUT_NAME})
    endif()
    
    set_target_properties(
      ${__NAME}_static
      PROPERTIES
      OUTPUT_NAME ${STATIC_OUTPUT_NAME}
      WINDOWS_EXPORT_ALL_SYMBOLS 1
      )
    
    target_link_libraries(
      ${__NAME}_static
      PUBLIC hytlass_library_includes
      PRIVATE $<BUILD_INTERFACE:${__NAME}_objs>
      hip::galaxyhip
      # hip::hip_driver
      )
    
    set_target_properties(${__NAME}_static PROPERTIES DEBUG_POSTFIX "${HYTLASS_LIBRARY_DEBUG_POSTFIX}")
    
    install(
      TARGETS ${__NAME} ${__NAME}_static
      EXPORT HygonHytlass
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      )
    
    if (__SUFFIX)
    
      # The partial libraries generated will be registered as linked libraries
      # to the main hytlass library so users automatically get the necessary link
      # commands to pull in all kernels by default.
    
      target_link_libraries(${DEFAULT_NAME} PUBLIC ${__NAME})
      target_link_libraries(${DEFAULT_NAME}_static PUBLIC ${__NAME}_static)
    
    endif()

  endif()

endfunction()

################################################################################

hytlass_add_hytlass_library(

  src/handle.cu
  src/manifest.cpp
  src/operation_table.cu
  src/singleton.cu
  src/util.cu

  # files split for parallel compilation
  src/reference/gemm_int8_canonical.cu
  src/reference/gemm_int8_interleaved_32.cu
  src/reference/gemm_int8_interleaved_64.cu
  src/reference/gemm_fp32out.cu
  src/reference/gemm_fp_other.cu
  src/reference/initialize_reference_operations.cu

  # hytlass reduction instances in hytlass library

  src/reduction/reduction_device.cu
  src/reduction/init_reduction_operations.cu
  
  # hytlass conv reference instances in hytlass library

  src/reference/conv2d.cu

  )

# For backward compatibility with the old name
add_library(hytlass_lib ALIAS hytlass_library)
add_library(hytlass_lib_static ALIAS hytlass_library_static)

################################################################################

file(GLOB_RECURSE GENERATOR_PYTHON_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.py)

#
# auto-instantiation of HYTLASS kernels
#

# set hytlass generator compiler version to filter kernels in the generator not supported by a specific toolkit.
set(HYTLASS_LIBRARY_GENERATED_KERNEL_LIST_FILE ${CMAKE_CURRENT_BINARY_DIR}/generated_kernels.txt CACHE STRING "Generated kernel listing file")

# --log-level is set to DEBUG to enable printing information about which kernels were excluded
# from generation in /python/hytlass_library/manifest.py. To avoid having this information appear
# in ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log, set this parameter to INFO
execute_process(
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../python/hytlass_library
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${HYTLASS_LIBRARY_PACKAGE_DIR}
    ${Python3_EXECUTABLE} ${HYTLASS_SOURCE_DIR}/python/hytlass_library/generator.py
    --operations "${HYTLASS_LIBRARY_OPERATIONS}" 
    --build-dir ${PROJECT_BINARY_DIR}
    --curr-build-dir ${CMAKE_CURRENT_BINARY_DIR}
    --generator-target library
    --architectures "${HYTLASS_HIPCC_ARCHS_ENABLED}"
    --kernels "${HYTLASS_LIBRARY_KERNELS}"
    --ignore-kernels "${HYTLASS_LIBRARY_IGNORE_KERNELS}"
    --kernel-filter-file "${KERNEL_FILTER_FILE}"
    --selected-kernel-list "${HYTLASS_LIBRARY_GENERATED_KERNEL_LIST_FILE}"
    --dtk-version "${HYTLASS_GENERATOR_HIP_COMPILER_VERSION}"
    --log-level DEBUG
    --problem-size-path "${HYTLASS_PROBLEM_SIZE_PATH}"
    --disable-hytlass-package-imports
  RESULT_VARIABLE hytlass_lib_INSTANCE_GENERATION_RESULT
  OUTPUT_VARIABLE hytlass_lib_INSTANCE_GENERATION_OUTPUT
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log
)

if(NOT hytlass_lib_INSTANCE_GENERATION_RESULT EQUAL 0)
  message(FATAL_ERROR "Error generating library instances. See ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log")
endif()

message(STATUS "Completed generation of library instances. See ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log for more information.")

# include auto-instantiated kernels in he HYTLASS Deliverables Library
set(HYTLASS_LIBRARY_MANIFEST_CMAKE_FILE ${CMAKE_CURRENT_BINARY_DIR}/generated/manifest.cmake)
if(EXISTS "${HYTLASS_LIBRARY_MANIFEST_CMAKE_FILE}")
  include(${HYTLASS_LIBRARY_MANIFEST_CMAKE_FILE})
else()
  message(STATUS "auto-generated library manifest cmake file (${HYTLASS_LIBRARY_MANIFEST_CMAKE_FILE}) not found.")
endif()

################################################################################

install(
  FILES ${HYTLASS_LIBRARY_GENERATED_KERNEL_LIST_FILE}
  DESTINATION ${CMAKE_INSTALL_INFODIR}/hytlass/
  )

