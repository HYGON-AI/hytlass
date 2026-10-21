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

cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
cmake_policy(SET CMP0112 NEW)

find_program(HIP_COMPILER_PATH hipcc)
if(HIP_COMPILER_PATH)
    get_filename_component(DTK_ROOT_DIR "${HIP_COMPILER_PATH}" DIRECTORY)
    get_filename_component(DTK_ROOT_DIR "${DTK_ROOT_DIR}/.." REALPATH)
    message(STATUS "DTK_ROOT_DIR is set to ${DTK_ROOT_DIR}")
else()
    message(FATAL_ERROR "hipcc not found in the environment path.")
endif()
# use hipcc as default compiler
set(CMAKE_CXX_COMPILER "${HIP_COMPILER_PATH}")

# for hipcomplex support
add_definitions(-DROCM_MATHLIBS_API_USE_HIP_COMPLEX=1)

if(hytlass_LOADED)
  # If HYTLASS has been previously fetched and loaded, don't do it again.
  return()
else()
  set(hytlass_LOADED ON)
  set(HYTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "HYTLASS Repository Directory")
endif()

message(STATUS "CMake Version: ${CMAKE_VERSION}")
set(IMPLICIT_CMAKE_CXX_STANDARD OFF CACHE BOOL "Do not explicitly specify -std=c++17 if set")

# To reduce duplicate version locations, parse the version out of the
# main versions.h file and reuse it here.

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/include/hytlass/version.h VERSION_FILE_CONTENTS)
string(REGEX MATCH "#define HYTLASS_MAJOR ([0-9]+)" _HYTLASS_VERSION_MAJOR "${VERSION_FILE_CONTENTS}")
set(_HYTLASS_VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define HYTLASS_MINOR ([0-9]+)" _HYTLASS_VERSION_MINOR "${VERSION_FILE_CONTENTS}")
set(_HYTLASS_VERSION_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define HYTLASS_PATCH ([0-9]+)" _HYTLASS_VERSION_PATCH "${VERSION_FILE_CONTENTS}")
set(_HYTLASS_VERSION_PATCH ${CMAKE_MATCH_1})

message(STATUS "HYTLASS ${_HYTLASS_VERSION_MAJOR}.${_HYTLASS_VERSION_MINOR}.${_HYTLASS_VERSION_PATCH}")

## HYTLASS PROJECT #############################################################

project(HYTLASS VERSION ${_HYTLASS_VERSION_MAJOR}.${_HYTLASS_VERSION_MINOR}.${_HYTLASS_VERSION_PATCH} LANGUAGES CXX)

################################################################################

include(${CMAKE_CURRENT_SOURCE_DIR}/HIP.cmake)

# enable __shfl_sync for dtk24.x
find_file(AMD_WARP_SYNC_PATH amd_warp_sync_functions.h
  PATHS ${DTK_ROOT_DIR}/hip/include/hip/amd_detail
)
if(AMD_WARP_SYNC_PATH)
  message(STATUS "Enable HIP_ENABLE_WARP_SYNC_BUILTINS")
  add_definitions(-DHIP_ENABLE_WARP_SYNC_BUILTINS)
endif()

if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--version" OUTPUT_VARIABLE CXX_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH "[A-Za-z]* ?clang version" TMP_CXX_VERSION ${CXX_OUTPUT})
  string(REGEX MATCH "[A-Za-z]+" CXX_VERSION_STRING ${TMP_CXX_VERSION})
endif()

# add compiler check
if( CXX_VERSION_STRING MATCHES "clang" )
  message( STATUS "Use hip-clang to build for HCU backend" )
  set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_HCC_COMPAT_MODE__=1" )
elseif( CXX_VERSION_STRING MATCHES "hipcc" )
  message(FATAL_ERROR "Don't support for hipcc")
else()
  message(FATAL_ERROR "Unsupport compiler ${CMAKE_CXX_COMPILER}. Only support for hip-clang")
endif()
find_package(Doxygen QUIET)

################################################################################

#
# HYTLASS 3.x requires C++17
#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


  
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX install CACHE PATH "Default installation location." FORCE)
endif()

message(STATUS "Default Install Location: ${CMAKE_INSTALL_PREFIX}")

set(HYTLASS_TEST_LEVEL "0" CACHE STRING "Level of tests to compile.")
# 0 - Sanity, 1 - Release-Quality, 2 - Exhaustive

find_package(Python3 3.5 COMPONENTS Interpreter REQUIRED)

################################################################################
set(HYTLASS_ENABLE_HEADERS_ONLY OFF CACHE BOOL "Enable only the header library")

if(HYTLASS_ENABLE_HEADERS_ONLY)
  set(HYTLASS_ENABLE_EXAMPLES_INIT OFF)
  set(HYTLASS_ENABLE_TOOLS_INIT ON)
  set(HYTLASS_ENABLE_LIBRARY_INIT OFF)
  set(HYTLASS_ENABLE_TESTS_INIT OFF)
else()
  set(HYTLASS_ENABLE_EXAMPLES_INIT ON)
  set(HYTLASS_ENABLE_TOOLS_INIT ON)
  set(HYTLASS_ENABLE_LIBRARY_INIT ON)
  if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    set(HYTLASS_ENABLE_TESTS_INIT ON)
  else()
    set(HYTLASS_ENABLE_TESTS_INIT OFF)
  endif()
  set(HYTLASS_ENABLE_HIPBLAS ON)
endif()

set(HYTLASS_TEST_UNIT_ENABLE_WARNINGS OFF CACHE BOOL "Enable warnings on waived unit tests.")

set(HYTLASS_ENABLE_EXAMPLES ${HYTLASS_ENABLE_EXAMPLES_INIT} CACHE BOOL "Enable HYTLASS Examples")
set(HYTLASS_ENABLE_TOOLS ${HYTLASS_ENABLE_TOOLS_INIT} CACHE BOOL "Enable HYTLASS Tools")
set(HYTLASS_ENABLE_LIBRARY ${HYTLASS_ENABLE_LIBRARY_INIT} CACHE BOOL "Enable HYTLASS Library")
set(HYTLASS_ENABLE_PROFILER ${HYTLASS_ENABLE_LIBRARY} CACHE BOOL "Enable HYTLASS Profiler")
set(HYTLASS_ENABLE_PERFORMANCE ${HYTLASS_ENABLE_PROFILER} CACHE BOOL "Enable HYTLASS Performance")

set(HYTLASS_ENABLE_TESTS ${HYTLASS_ENABLE_TESTS_INIT} CACHE BOOL "Enable HYTLASS Tests")
set(HYTLASS_ENABLE_GTEST_UNIT_TESTS ${HYTLASS_ENABLE_TESTS} CACHE BOOL "Enable HYTLASS GTest-based Unit Tests")
set(HYTLASS_USE_SYSTEM_GOOGLETEST OFF CACHE BOOL "Use system/external installation of GTest")
################################################################################


# Enable all arch for now unless some archs were specified
if (NOT DEFINED HYTLASS_HIPCC_ARCHS_SUPPORTED)
  set(HYTLASS_HIPCC_ARCHS_SUPPORTED)
  if(DEFINED ENV{AMDGPU_TARGETS})
    set(AMDGPU_TARGETS_LIST "$ENV{AMDGPU_TARGETS}")
    foreach(target ${AMDGPU_TARGETS_LIST})
      string(REGEX REPLACE "gfx([0-9]+)" "\\1" number ${target})
      list(APPEND HYTLASS_HIPCC_ARCHS_SUPPORTED "${number}")
    endforeach()
  elseif()
    list(APPEND HYTLASS_HIPCC_ARCHS_SUPPORTED 906 926 908 928 936)
  endif()
endif()

set(HYTLASS_HIPCC_ARCHS ${HYTLASS_HIPCC_ARCHS_SUPPORTED} CACHE STRING "The Gfx architectures requested.")
set(HYTLASS_HIPCC_ARCHS_ENABLED ${HYTLASS_HIPCC_ARCHS} CACHE STRING "The Gfx architectures to build code for.")

# Find unsupported and deprecated compute capabilities
if (HYTLASS_HIPCC_ARCHS_SUPPORTED)
  set(HYTLASS_HIPCC_ARCHS_UNSUPPORTED ${HYTLASS_HIPCC_ARCHS})
  list(REMOVE_ITEM HYTLASS_HIPCC_ARCHS_UNSUPPORTED ${HYTLASS_HIPCC_ARCHS_SUPPORTED})
  if (HYTLASS_HIPCC_ARCHS_UNSUPPORTED)
    message(WARNING "Using unsupported or deprecated compute capabilities ${HYTLASS_HIPCC_ARCHS_UNSUPPORTED}. Support may be removed in future versions.")
  endif()
else()
  message(WARNING "No supported compute capabilities")
endif()


# Special policy introduced in CMake 3.13
if (POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif()

include(GNUInstallDirs)

###################################################################################################
#
# Configure CMake variables
#
###################################################################################################

message(STATUS "HIP Compilation Architectures: ${HYTLASS_HIPCC_ARCHS_ENABLED}")

if (NOT (CMAKE_BUILD_TYPE OR CONFIGURATION_TYPES))
  # By default we want to build in Release mode to ensure that we're getting best performance.
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build level" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo" "Release")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (DEFINED CMAKE_DEBUG_POSTFIX)
  set(HYTLASS_LIBRARY_DEBUG_POSTFIX_INIT ${CMAKE_DEBUG_POSTFIX})
else()
  set(HYTLASS_LIBRARY_DEBUG_POSTFIX_INIT .debug)
endif()
set(HYTLASS_LIBRARY_DEBUG_POSTFIX ${HYTLASS_LIBRARY_DEBUG_POSTFIX_INIT} CACHE STRING "Default postfix value for debug libraries")

if(WIN32)
  # On Windows we link against the shared (DLL) runtime. Change gtest settings to match this.
  set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib even when Google Test is built as static lib" FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHYTLASS_VERSIONS_GENERATED")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -DHYTLASS_VERSIONS_GENERATED")

if (WIN32)
  # Enable more warnings.  Add "-Xcompiler=/WX" to enable warnings as errors.
  list(APPEND HYTLASS_HIP_HIPCC_FLAGS -Xcompiler=/W3)

  # Disable warning on Unicode characters
  list(APPEND HYTLASS_HIP_HIPCC_FLAGS -Xcompiler=/wd4819)

  # Disable excess x86 floating point precision that can lead to results being labeled incorrectly
  list(APPEND HYTLASS_HIP_HIPCC_FLAGS -Xcompiler=/fp:strict)
endif(WIN32)

if (${HYTLASS_HIPCC_VERBOSE})
  list(APPEND HYTLASS_HIP_HIPCC_FLAGS -v)
endif()


#
# HYTLASS NAMESPACE 
#
set(HYTLASS_NAMESPACE "hytlass" CACHE STRING "Top level namespace of HYTLASS")


set(HYTLASS_HIPCC_KEEP OFF CACHE BOOL "Keep intermediate files generated by HIPCC.")
set(HYTLASS_ENABLE_F16C OFF CACHE BOOL "Enable F16C x86 extensions in host code.")

################################################################################
#
# HYTLASS generator cmake configuration
#

# Kernel unified filter file

set(KERNEL_FILTER_FILE "" CACHE STRING "KERNEL FILTER FILE FULL PATH")

if (KERNEL_FILTER_FILE AND NOT HYTLASS_LIBRARY_KERNELS) 
  # If a kernel filter file is specified, we want to generate and then
  # filter on the entire kernel set, not the default kernel
  # (sub)set. The user may overried HYTLASS_LIBRRARY_KERNELS, in which
  # case the resulting kernel set will be the intersection of the two
  # options differenced against HYTLASS_LIBRARY_IGNORE_KERNELS.
  set(HYTLASS_LIBRARY_KERNELS_INIT "*")
else() 
  set(HYTLASS_LIBRARY_KERNELS_INIT "") 
endif()

if (KERNEL_FILTER_FILE)
  get_filename_component(KERNEL_FILTER_FILE "${KERNEL_FILTER_FILE}" ABSOLUTE)
  set(KERNEL_FILTER_FILE "${KERNEL_FILTER_FILE}" CACHE STRING "KERNEL FILTER FILE FULL PATH" FORCE)
endif()

set(SELECTED_KERNEL_LIST "selected" CACHE STRING "Name of the filtered kernel list")

if(KERNEL_FILTER_FILE)
  message(STATUS "Full path of filter file: ${KERNEL_FILTER_FILE}")
endif()

set(HYTLASS_LIBRARY_OPERATIONS "all" CACHE STRING "Comma delimited list of operation name filters. Default '' means all operations are enabled.")
set(HYTLASS_LIBRARY_KERNELS ${HYTLASS_LIBRARY_KERNELS_INIT} CACHE STRING "Comma delimited list of kernel name filters. If unspecified, only the largest tile size is enabled. If 'all' is specified, all kernels are enabled.")
set(HYTLASS_LIBRARY_IGNORE_KERNELS "" CACHE STRING "Comma delimited list of kernel names to exclude from build.")
set(HYTLASS_PROBLEM_SIZE_PATH "" CACHE STRING "Comma defined from which path data is loaded")

################################################################################

set(HYTLASS_TEST_ENABLE_CACHED_RESULTS OFF CACHE BOOL "Enable caching and reuse of test results in unit tests")

set_property(CACHE HYTLASS_TEST_LEVEL PROPERTY STRINGS 0 1 2)

list(APPEND HYTLASS_HIP_HIPCC_FLAGS -DHYTLASS_TEST_LEVEL=${HYTLASS_TEST_LEVEL})

if (HYTLASS_TEST_ENABLE_CACHED_RESULTS)
  message(STATUS "Enable caching of reference results in conv unit tests")
  list(APPEND HYTLASS_HIP_HIPCC_FLAGS -DHYTLASS_TEST_ENABLE_CACHED_RESULTS=1)
endif()

set(HYTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED ON CACHE BOOL "Enable/Disable rigorous conv problem sizes in conv unit tests")

if (HYTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED)
  message(STATUS "Enable rigorous conv problem sizes in conv unit tests")
  list(APPEND HYTLASS_HIP_HIPCC_FLAGS -DHYTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1)
endif()

################################################################################

# Trace levels for debugging
set(HYTLASS_DEBUG_TRACE_LEVEL "0" CACHE STRING "Level of debug tracing to perform.")
list(APPEND HYTLASS_HIP_HIPCC_FLAGS -DHYTLASS_DEBUG_TRACE_LEVEL=${HYTLASS_DEBUG_TRACE_LEVEL})


#
# NOTE: running with asan and HIP requires the following environment variable:
#
#  ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0
#
# without the above environment setting, an error like the following may be generated:
#
#  *** Error: Could not detect active GPU device ID [out of memory]
#  ...
#  ==9149==ERROR: LeakSanitizer: detected memory leaks
#  ...
#
if(ENABLE_ASAN)  # https://github.com/google/sanitizers/wiki/AddressSanitizer
  list(APPEND HYTLASS_HIP_HIPCC_FLAGS --compiler-options=-fsanitize=address --compiler-options=-fno-omit-frame-pointer)
  string(APPEND CMAKE_EXE_LINKER_FLAGS " -fsanitize=address")
endif()

# Enable double VGPRs for grid size 512
list(APPEND HYTLASS_HIP_HIPCC_FLAGS -mllvm -enable-num-vgprs-512=true)

###################################################################################################
#
# Configure HIP build options
#
###################################################################################################


# Warnings-as-error exceptions and warning suppressions for Clang builds
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=implicit-int-conversion ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pass-failed ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=inconsistent-missing-override ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-conversion ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-return-type ")
endif()

if (NOT MSVC AND HYTLASS_HIPCC_KEEP)
  # MSVC flow handles caching already, but for other generators we handle it here.
  set(HYTLASS_HIPCC_KEEP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp CACHE PATH "Location to store HIPCC scratch files")
  file(MAKE_DIRECTORY ${HYTLASS_HIPCC_KEEP_DIR})
  list(APPEND HYTLASS_HIP_HIPCC_FLAGS -save-temps=${HYTLASS_HIPCC_KEEP_DIR} -v)
endif()

if (HYTLASS_ENABLE_F16C AND NOT CMAKE_CROSSCOMPILING)
  list(APPEND HYTLASS_HIP_FLAGS -DHYTLASS_ENABLE_F16C=1)
  if ((CMAKE_CXX_COMPILER_ID MATCHES "MSVC"))
    list(APPEND HYTLASS_HIP_HIPCC_FLAGS -Xcompiler=/arch:AVX2)
  else()
    list(APPEND HYTLASS_HIP_HIPCC_FLAGS -mf16c)
  endif()
endif()

if (HYTLASS_ENABLE_OPENMP_TESTS)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    list(APPEND HYTLASS_HIP_HIPCC_FLAGS -Xcompiler=${OpenMP_CXX_FLAGS})
  else()
    message(WARNING "HYTLASS_ENABLE_OPENMP_TESTS set but OpenMP not found.")
  endif()
endif()

if(UNIX)
  list(APPEND HYTLASS_HIP_HIPCC_FLAGS -Wconversion)
  list(APPEND HYTLASS_HIP_HIPCC_FLAGS -fno-strict-aliasing)
endif()

# Don't leak lineinfo in release builds
if (NOT CMAKE_BUILD_TYPE MATCHES "Release")
  list(APPEND HYTLASS_HIP_HIPCC_FLAGS -lineinfo)
endif()

list(APPEND HYTLASS_HIP_HIPCC_FLAGS -Wno-sign-conversion -Wno-shorten-64-to-32 -Wno-implicit-float-conversion -Wno-implicit-int-conversion -Wno-return-type)

if(HYTLASS_HIP_HIPCC_FLAGS)
  message(STATUS "Using hipcc flags: ${HYTLASS_HIP_HIPCC_FLAGS}")
endif()

# Support for 128-bit integers if using HYGON C++ compiler 
# if (${CMAKE_CXX_COMPILER_ID} MATCHES "PGI" OR ${CMAKE_CXX_COMPILER_ID} MATCHES "NVHPC")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Mint128 ")
# endif()

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  # CMake 3.18 added support for HIP_ARCHITECTURES target property. We will use this
  # property for CMake 3.18+, so we request the NEW behavior for correct compatibility.
  # https://cmake.org/cmake/help/v3.18/policy/CMP0104.html#policy:CMP0104 
  cmake_policy(SET CMP0104 NEW)
endif()

if (MSVC)
  
  # MSVC by default does not apply the correct __cplusplus version as specified by the C++ standard
  # because MSVC is not a completely compliant implementation. This option forces MSVC to use the 
  # appropriate value given the requested --std option. This fixes a compilation issue mismatch
  # between GCC/Clang and MSVC.
  #
  # error : a constexpr function cannot have a nonliteral return type "dim3"
  # 
  # See https://developercommunity.visualstudio.com/t/msvc-incorrectly-defines-cplusplus/139261

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Xcompiler  /Zc:__cplusplus")
  
endif()

# Some tests require this build option in order to link.
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Xcompiler /bigobj")
endif()

function(hytlass_apply_hip_gencode_flags TARGET)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs SM_ARCHS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (__SM_ARCHS)
    set(ARCHS_ENABLED ${__SM_ARCHS})
  else()
    set(ARCHS_ENABLED ${HYTLASS_HIPCC_ARCHS_ENABLED})
  endif()

  set(HIPCC_FLAGS)

  foreach(ARCH ${ARCHS_ENABLED})
    list(APPEND HIPCC_FLAGS --offload-arch=gfx${ARCH})
  endforeach()
 
  if (NOT __SM_ARCHS)
    target_compile_options(
      ${TARGET}
      PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${HIPCC_FLAGS}>
      )
  else()
    list(JOIN HIPCC_FLAGS " " STR_HIPCC_FLAGS)
    if(${TARGET} MATCHES ".*\.cu")
      set_source_files_properties(${TARGET} PROPERTIES COMPILE_FLAGS ${STR_HIPCC_FLAGS})
    endif()
  endif()

endfunction()

# Cache the flags so they are available when the function below is called anywhere globally. 

set(__HYTLASS_HIP_FLAGS ${HYTLASS_HIP_FLAGS} CACHE INTERNAL "")
set(__HYTLASS_HIP_FLAGS_RELEASE ${HYTLASS_HIP_FLAGS_RELEASE} CACHE INTERNAL "")
set(__HYTLASS_HIP_FLAGS_RELWITHDEBINFO ${HYTLASS_HIP_FLAGS_RELWITHDEBINFO} CACHE INTERNAL "")
set(__HYTLASS_HIP_FLAGS_DEBUG ${HYTLASS_HIP_FLAGS_DEBUG} CACHE INTERNAL "")

set(__HYTLASS_HIP_HIPCC_FLAGS ${HYTLASS_HIP_HIPCC_FLAGS} CACHE INTERNAL "")
set(__HYTLASS_HIP_HIPCC_FLAGS_RELEASE ${HYTLASS_HIP_HIPCC_FLAGS_RELEASE} CACHE INTERNAL "")
set(__HYTLASS_HIP_HIPCC_FLAGS_RELWITHDEBINFO ${HYTLASS_HIP_HIPCC_FLAGS_RELWITHDEBINFO} CACHE INTERNAL "")
set(__HYTLASS_HIP_HIPCC_FLAGS_DEBUG ${HYTLASS_HIP_HIPCC_FLAGS_DEBUG} CACHE INTERNAL "")

function(hytlass_apply_standard_compile_options TARGET)

  set(HIP_COMPILE_LANGUAGE CXX)
  set(_FLAGS ${__HYTLASS_HIP_FLAGS} ${__HYTLASS_HIP_HIPCC_FLAGS})
  set(_FLAGS_RELEASE ${__HYTLASS_HIP_FLAGS_RELEASE} ${__HYTLASS_HIP_HIPCC_FLAGS_RELEASE})
  set(_FLAGS_RELWITHDEBINFO ${__HYTLASS_HIP_FLAGS_RELWITHDEBINFO} ${__HYTLASS_HIP_HIPCC_FLAGS_RELWITHDEBINFO})
  set(_FLAGS_DEBUG ${__HYTLASS_HIP_FLAGS_DEBUG} ${__HYTLASS_HIP_HIPCC_FLAGS_DEBUG})

  target_link_libraries(${TARGET} PRIVATE HYTLASS)

  target_compile_options(
    ${TARGET}
    PRIVATE
    $<$<COMPILE_LANGUAGE:${HIP_COMPILE_LANGUAGE}>:${_FLAGS}>
    $<$<COMPILE_LANGUAGE:${HIP_COMPILE_LANGUAGE}>:$<$<CONFIG:RELEASE>:${_FLAGS_RELEASE}>>
    $<$<COMPILE_LANGUAGE:${HIP_COMPILE_LANGUAGE}>:$<$<CONFIG:RELWITHDEBINFO>:${_FLAGS_RELWITHDEBINFO}>>
    $<$<COMPILE_LANGUAGE:${HIP_COMPILE_LANGUAGE}>:$<$<CONFIG:DEBUG>:${_FLAGS_DEBUG}>>
    )

endfunction()

#
# The following items should eventually be pushed into hytlass/CMakeLists.txt
#

# GLOB for HYTLASS header files. Should we use a static list instead?
file(GLOB_RECURSE HYTLASS_INCLUDE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/hytlass/*.h)
file(GLOB_RECURSE HYTLASS_HYTLASS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include include/hytlass/*.h include/hytlass/*.hpp include/hytlass/*.inl)
file(GLOB_RECURSE HYTLASS_HUTE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include include/hute/*.h*)

###################################################################################################
#
# Define build targets
#
###################################################################################################

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include REGULAR_EXPRESSION ".*\.h")

add_library(HYTLASS INTERFACE)
add_library(hygon::hytlass::hytlass ALIAS HYTLASS)
set_target_properties(HYTLASS PROPERTIES EXPORT_NAME hytlass)

set(HYTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE PATH "HYTLASS Header Library")

set(HYTLASS_GENERATOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/library CACHE INTERNAL "Location of generator scripts")

# The following utility directory is needed even if the tools build is disabled, so it exists here.
set(HYTLASS_TOOLS_UTIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/util/include CACHE INTERNAL "")

include_directories(${HYTLASS_INCLUDE_DIR})

target_compile_features(HYTLASS INTERFACE cxx_std_11)

if (NOT HYTLASS_NAMESPACE STREQUAL "hytlass")
  target_compile_definitions(HYTLASS INTERFACE HYTLASS_NAMESPACE=${HYTLASS_NAMESPACE})
endif()

if (NOT DEFINED HYTLASS_REVISION)

  find_package(Git QUIET)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    RESULT_VARIABLE HYTLASS_REVISION_RESULT
    OUTPUT_VARIABLE HYTLASS_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if (HYTLASS_REVISION_RESULT)
    message(STATUS "HYTLASS Revision: Unable to detect, Git returned code ${HYTLASS_REVISION_RESULT}.")
  else()
    message(STATUS "HYTLASS Revision: ${HYTLASS_REVISION}")
  endif()

endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_extended.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/hytlass/version_extended.h
  @ONLY)

target_include_directories(
  HYTLASS
  INTERFACE
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${HYTLASS_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${hute_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${hute_SOURCE_DIR}/examples>
  )

# Mark DTK headers as system to supress warnings from them
target_include_directories(
  HYTLASS
  SYSTEM INTERFACE
  $<BUILD_INTERFACE:${DTK_ROOT_DIR}/include>
  )

install(
  DIRECTORY
  ${HYTLASS_INCLUDE_DIR}/
  ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(
  TARGETS HYTLASS
  EXPORT HygonHytlass
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

################################################################################

# Doxygen is available. Generate documentation
if (DOXYGEN_FOUND)
    # DOT is available. Enable graph generation in the documentation
    if (DOXYGEN_DOT_EXECUTABLE)
        set(HYTLASS_ENABLE_DOXYGEN_DOT ON CACHE BOOL "Use dot to generate graphs in the doxygen documentation.")
    else()
        set(HYTLASS_ENABLE_DOXYGEN_DOT OFF CACHE BOOL "Use dot to generate graphs in the doxygen documentation." FORCE)
    endif()

    if (HYTLASS_ENABLE_DOXYGEN_DOT)
        set(HAVE_DOT "YES")
    else()
        set(HAVE_DOT "NO")
    endif()

    # Add custom target for Doxygen.
    add_custom_target(hytlass_docs ${CMAKE_COMMAND} -E env
        "DOT_PATH=${DOXYGEN_DOT_EXECUTABLE}"
        "HAVE_DOT=${HAVE_DOT}"
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM
    )
endif()

if(NOT WIN32)
  # Add common library search paths so executables and libraries can load and run
  # without LD_LIBRARY_PATH being set.
  link_libraries(
    "-Wl,-rpath,'$ORIGIN'"
    "-Wl,-rpath,'$ORIGIN/../lib64'"
    "-Wl,-rpath,'$ORIGIN/../lib'"
    "-Wl,-rpath,'${DTK_ROOT_DIR}/lib64'"
    "-Wl,-rpath,'${DTK_ROOT_DIR}/lib'"
    )
endif()

################################################################################

include(CTest)
enable_testing()


if (HYTLASS_ENABLE_GTEST_UNIT_TESTS)
  if (HYTLASS_USE_SYSTEM_GOOGLETEST)
    find_package(GTest REQUIRED)
  else()
    # include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/googletest.cmake)
    add_subdirectory(googletest)
    include_directories(googletest/googletest/include)
  endif()
endif()

if (NOT TARGET test_all)
  add_custom_target(test_all)
endif()

set(HYTLASS_INSTALL_TESTS ON CACHE BOOL "Install test executables")
set(HYTLASS_TEST_EXECUTION_ENVIRONMENT "" CACHE BOOL "Environment in which to invoke unit test executables")

set(CMAKE_TEST_INSTALL_PREFIX test CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(HYTLASS_TEST_INSTALL_PREFIX ${CMAKE_TEST_INSTALL_PREFIX}/hytlass CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(HYTLASS_TEST_INSTALL_BINDIR ${HYTLASS_TEST_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR} CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(HYTLASS_TEST_INSTALL_LIBDIR ${HYTLASS_TEST_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")

install(DIRECTORY DESTINATION ${HYTLASS_TEST_INSTALL_PREFIX})
install(DIRECTORY DESTINATION ${HYTLASS_TEST_INSTALL_BINDIR})
install(DIRECTORY DESTINATION ${HYTLASS_TEST_INSTALL_LIBDIR})
install(DIRECTORY DESTINATION ${HYTLASS_TEST_INSTALL_PREFIX}/ctest)

################################################################################
# use hipBlas
include(${CMAKE_CURRENT_SOURCE_DIR}/hipBLAS.cmake)

if (HYTLASS_ENABLE_HIPBLAS)
  target_compile_definitions(HYTLASS INTERFACE HYTLASS_ENABLE_HIPBLAS=1)
endif()

################################################################################

set(HYTLASS_DEFAULT_ACTIVE_TEST_SETS "default" CACHE STRING "Default
  activated test sets. In `make test` mode, this string determines the
  active set of tests. In `ctest` mode, this value can be overriden
  with HYTLASS_TEST_SETS environment variable when running the ctest
  executable.")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(HYTLASS_CTEST_TEMPLATE_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/CTestTestfile.configure.cmake)
set(HYTLASS_CTEST_GENERATED_FILES "" CACHE INTERNAL "")

function(hytlass_add_executable_tests NAME TARGET)
# 
# Generates test rules for `make test`, `make test_all`, and `ctest` invoked from either the 
# <CMAKE_BINARY_DIR> or the <CMAKE_INSTALL_PREFIX>/<HYTLASS_TEST_INSTALL_PREFIX> after installation.
# 
# NAME: The base name for the test. Can be run with `make <NAME>` or `ctest -R 'c<NAME>'`.
# TARGET: The target corresponding to the executable under test.
# DISABLE_EXECUTABLE_INSTALL_RULE: An option, if given, that disables creating an install rule for TARGET.
# DEPENDS: A list of targets or files on which this test is dependent.
# DEPENDEES: A list of targets which should depend on this test.
# TEST_COMMAND_OPTIONS: A list of variables (i.e. by reference params) which contain command line arguments
#   to pass to the test executable. A unique test is generated for each set of 
#   options given. If this option is not used, a single test with no arguments is generated.
# TEST_COMMAND_OPTIONS_PREFIX: If provided, is added as a prefix to each TEST_COMMAND_OPTIONS value for 
#   generating the full variable name to be referenced.
# RESULT_CACHE_FILE: A file to be installed alongside the test executable with pre-computed
#   test results to speed up test runtime.
# TEST_SETS_SUPPORTED: A list of test set names these tests support. 
# 

  set(options DISABLE_EXECUTABLE_INSTALL_RULE)
  set(oneValueArgs DISABLE_TESTS RESULT_CACHE_FILE TEST_COMMAND_OPTIONS_PREFIX)
  set(multiValueArgs DEPENDS DEPENDEES TEST_COMMAND_OPTIONS TEST_SETS_SUPPORTED)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED __DISABLE_TESTS)
    set(__DISABLE_TESTS OFF)
  endif()

  set(TEST_EXE $<TARGET_FILE_NAME:${TARGET}>)
  set(TEST_EXE_WORKING_DIRECTORY ./${CMAKE_INSTALL_BINDIR})

  if (NOT DEFINED __TEST_SETS_SUPPORTED)
    set(__TEST_SETS_SUPPORTED ${HYTLASS_DEFAULT_ACTIVE_TEST_SETS})
  endif()

  set(TEST_SETS_SUPPORTED ${__TEST_SETS_SUPPORTED})

  if (__RESULT_CACHE_FILE)

    add_custom_command(
      TARGET ${TARGET}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E copy ${__RESULT_CACHE_FILE} "$<TARGET_FILE_DIR:${TARGET}>"
      )

  endif()

  if (NOT __DISABLE_EXECUTABLE_INSTALL_RULE AND HYTLASS_INSTALL_TESTS)
  
    # file(RELATIVE_PATH CMAKE_CURRENT_BINARY_RELATIVE_DIR ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  
    install(
      TARGETS ${TARGET}
      RUNTIME DESTINATION ${HYTLASS_TEST_INSTALL_BINDIR}
      )

    if (__RESULT_CACHE_FILE)

     install(
       FILES ${__RESULT_CACHE_FILE}
       DESTINATION ${HYTLASS_TEST_INSTALL_BINDIR}/
       )

    endif()
  
  endif()

  if (NOT __TEST_COMMAND_OPTIONS)
    set(__TEST_COMMAND_OPTIONS " ")
  endif()

  list(LENGTH __TEST_COMMAND_OPTIONS CMD_COUNT)

  if (CMD_COUNT GREATER 1)
    add_custom_target(${NAME} DEPENDS ${TARGET} ${__DEPENDS})
    foreach(DEPENDEE ${__DEPENDEES})
      add_dependencies(${DEPENDEE} ${NAME})
    endforeach()
  endif()

  if (HYTLASS_INSTALL_TESTS)

    set(_INLINE_PER_TEST_CODE)

    file(READ "${PROJECT_SOURCE_DIR}/cmake/CTestTestfile.test.configure.cmake" _INLINE_PER_TEST_CODE_TEMPLATE)

  endif()

  set(TEST_GROUP_NAME ${NAME})

  foreach(CMD_OPTIONS_VAR IN LISTS __TEST_COMMAND_OPTIONS)

    if (CMD_COUNT GREATER 1)
      string(TOLOWER "${NAME}_${CMD_OPTIONS_VAR}" TEST_NAME)
    else()
      string(TOLOWER "${NAME}" TEST_NAME)
    endif()

    # The following rigmarole is needed to deal with spaces and possible quotes in 
    # command line arguments. The options are passed "by reference" as the actual
    # variable names holding the real options. We then expand these in a way that
    # preserves any quotes. Note, they have to be in this order for it to work for 
    # all the use cases below.

    set(TEST_COMMAND_OPTIONS ${${__TEST_COMMAND_OPTIONS_PREFIX}${CMD_OPTIONS_VAR}})
    list(JOIN TEST_COMMAND_OPTIONS " " TEST_COMMAND_OPTIONS)
    separate_arguments(TEST_COMMAND_OPTIONS)

    add_custom_target(
      ${TEST_NAME}
      COMMAND
      ${HYTLASS_TEST_EXECUTION_ENVIRONMENT} $<TARGET_FILE:${TARGET}> ${TEST_COMMAND_OPTIONS}
      DEPENDS
      ${TARGET}
      )

    if (CMD_COUNT GREATER 1)
      add_dependencies(${NAME} ${TEST_NAME})
    endif()

    foreach(DEPENDEE ${__DEPENDEES})
      add_dependencies(${DEPENDEE} ${TEST_NAME})
    endforeach()

    set(TEST_NAME c${TEST_NAME})
    string(CONFIGURE "${_INLINE_PER_TEST_CODE_TEMPLATE}" _TEST_CODE @ONLY)
    string(APPEND _INLINE_PER_TEST_CODE "${_TEST_CODE}")

  endforeach()

  # To run the tests from an install package with tests enabled, we need to generate test files
  # that don't rely on the current directory structure in build.  

  set(TEST_NAME c${NAME})
  set(TEST_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/ctest/${TEST_NAME})
  file(MAKE_DIRECTORY ${TEST_GEN_DIR})

  set(TEST_EXE_PATH $<TARGET_FILE:${TARGET}>)
  set(TEST_USE_EXTENDED_FORMAT ON)
  configure_file("${HYTLASS_CTEST_TEMPLATE_FILE}" "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.cmake" @ONLY)

  set(TEST_EXE_PATH $<TARGET_FILE_NAME:${TARGET}>)
  set(TEST_USE_EXTENDED_FORMAT OFF) # ctest does not support extended add_test format.
  configure_file("${HYTLASS_CTEST_TEMPLATE_FILE}" "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake.in" @ONLY)

  # The following line imports the tests for immediate run via `make test`.

  include(${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.cmake)
 
  set(HYTLASS_CTEST_GENERATED_FILES ${HYTLASS_CTEST_GENERATED_FILES};ctest/${TEST_NAME}/CTestTestfile.${TEST_NAME}.cmake CACHE INTERNAL "")

    if (HYTLASS_INSTALL_TESTS)

    file(GENERATE 
      OUTPUT "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake" 
      INPUT "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake.in" 
      )

    install(
      FILES "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake"
      DESTINATION ${HYTLASS_TEST_INSTALL_PREFIX}/ctest/${TEST_NAME}
      RENAME CTestTestfile.${TEST_NAME}.cmake
      )

    endif()

endfunction()

if (HYTLASS_ENABLE_TOOLS)
  add_subdirectory(tools)
  if (HYTLASS_ENABLE_PROFILER)
    add_dependencies(test_all test_profiler)
  endif()
endif()

if (HYTLASS_ENABLE_EXAMPLES)
  add_subdirectory(examples)
  add_dependencies(test_all test_examples)
endif()

if (HYTLASS_ENABLE_TESTS)
  add_subdirectory(test)
  if (HYTLASS_ENABLE_GTEST_UNIT_TESTS)
  add_dependencies(test_all test_unit)
  endif()
endif()

if (HYTLASS_INSTALL_TESTS)

  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/ctest")

  file(WRITE "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "# Generated File\n\n")
  file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "cmake_policy(SET CMP0057 NEW) # Allow IN_LIST for if()\n\n")
  file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "if (NOT DEFINED ENV{HYTLASS_TEST_SETS})\n")
  file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "  set(ENV{HYTLASS_TEST_SETS} ${HYTLASS_DEFAULT_ACTIVE_TEST_SETS})\n")
  file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "endif()\n\n")

  foreach(GENERATED_FILE ${HYTLASS_CTEST_GENERATED_FILES})
    file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "include(${GENERATED_FILE})\n")
  endforeach()

  install(
    FILES "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake"
    DESTINATION "${HYTLASS_TEST_INSTALL_PREFIX}/"
    )

endif()

################################################################################

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/HygonHytlassConfigVersion.cmake 
  COMPATIBILITY AnyNewerVersion)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HygonHytlassConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/HygonHytlassConfig.cmake  
  @ONLY
  )

install(
  FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/HygonHytlassConfig.cmake  
    ${CMAKE_CURRENT_BINARY_DIR}/HygonHytlassConfigVersion.cmake 
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HygonHytlass/
  )

install(
  EXPORT HygonHytlass
  NAMESPACE hygon::hytlass::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HygonHytlass/
  FILE HygonHytlassTargets.cmake
  )

################################################################################

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HygonHytlassPackageConfig.cmake)

