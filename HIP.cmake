# Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
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

find_package(HIP REQUIRED CONFIG PATHS ${DTK_ROOT_DIR})

################# RT #########################
find_library(
  GALAXY_HIP galaxyhip
  ${DTK_ROOT_DIR}/lib
  NO_DEFAULT_PATH
)

if(NOT TARGET hip::galaxyhip AND GALAXY_HIP)
  message(STATUS "Found galaxyhip: True")
  add_library(galaxyhip SHARED IMPORTED GLOBAL)
  add_library(hip::galaxyhip ALIAS hipgalaxy)
  set_property(
    TARGET galaxyhip
    PROPERTY IMPORTED_LOCATION
    ${GALAXY_HIP}
  )
elseif(TARGET hip::galaxyhip)
  message(STATUS "Found galaxyhip: True")
else()
  message(STATUS "Found galaxyhip: True")
endif()


find_library(
  HIPRTC_LIBRARY hiprtc
  PATHS
  ${DTK_ROOT_DIR}/lib
  NO_DEFAULT_PATH
  )

if(NOT TARGET hiprtc AND HIPRTC_LIBRARY)
  message(STATUS "Found hiprtc: True")
  add_library(hiprtc SHARED IMPORTED GLOBAL)
  add_library(hip::hiprtc ALIAS hiprtc)
  set_property(
    TARGET hiprtc
    PROPERTY IMPORTED_LOCATION
    ${HIPRTC_LIBRARY}
    )
elseif(TARGET hiprtc)
  message(STATUS "Found hiprtc: True")
else()
  message(STATUS "Found hiprtc: False")
endif()

include_directories(SYSTEM "${DTK_ROOT_DIR}/include")

# set hip property as *.cu
function(hytlass_correct_source_file_language_property)
  foreach(File ${ARGN})
    # add compile option -xhip while using clang++
    if(File MATCHES ".*\.cu$")
    #   set_source_files_properties(${File} PROPERTIES COMPILE_FLAGS "-x hip")
    set_source_files_properties(${File} PROPERTIES LANGUAGE CXX)
    endif()
  endforeach()
endfunction()

set(HYTLASS_UNITY_BUILD_ENABLED_INIT OFF)
set(HYTLASS_UNITY_BUILD_ENABLED ${HYTLASS_UNITY_BUILD_ENABLED_INIT} CACHE BOOL "Enable combined source compilation")

set(HYTLASS_UNITY_BUILD_BATCH_SIZE_INIT 16)
set(HYTLASS_UNITY_BUILD_BATCH_SIZE ${HYTLASS_UNITY_BUILD_BATCH_SIZE_INIT} CACHE STRING "Batch size for unified source files")

# set unify 
function(hytlass_unify_source_files TARGET_ARGS_VAR)

  set(options)
  set(oneValueArgs BATCH_SOURCES BATCH_SIZE)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED TARGET_ARGS_VAR)
    message(FATAL_ERROR "TARGET_ARGS_VAR parameter is required")
  endif()

  if (__BATCH_SOURCES AND NOT DEFINED __BATCH_SIZE)
    set(__BATCH_SIZE ${HYTLASS_UNITY_BUILD_BATCH_SIZE})
  endif()

  if (HYTLASS_UNITY_BUILD_ENABLED AND DEFINED __BATCH_SIZE AND __BATCH_SIZE GREATER 1)
    set(HIP_FILE_ARGS)
    set(TARGET_SOURCE_ARGS)
    
    foreach(ARG ${__UNPARSED_ARGUMENTS})
      if(${ARG} MATCHES ".*\.cu$")
        list(APPEND HIP_FILE_ARGS ${ARG})
      else()
        list(APPEND TARGET_SOURCE_ARGS ${ARG})
      endif()
    endforeach()
    
    list(LENGTH HIP_FILE_ARGS NUM_HIP_FILE_ARGS)
    while(NUM_HIP_FILE_ARGS GREATER 0)
      list(SUBLIST HIP_FILE_ARGS 0 ${__BATCH_SIZE} HIP_FILE_BATCH)
      string(SHA256 HIP_FILE_BATCH_HASH "${HIP_FILE_BATCH}")
      string(SUBSTRING ${HIP_FILE_BATCH_HASH} 0 12 HIP_FILE_BATCH_HASH)
      set(BATCH_FILE ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.unity.${HIP_FILE_BATCH_HASH}.cu)
      message(STATUS "Generating ${BATCH_FILE}")
      file(WRITE ${BATCH_FILE} "// Unity File - Auto Generated!\n")
      foreach(HIP_FILE ${HIP_FILE_BATCH})
        get_filename_component(HIP_FILE_ABS_PATH ${HIP_FILE} ABSOLUTE)
        file(APPEND ${BATCH_FILE} "#include \"${HIP_FILE_ABS_PATH}\"\n")
      endforeach()
      list(APPEND TARGET_SOURCE_ARGS ${BATCH_FILE})
      if (NUM_HIP_FILE_ARGS LESS_EQUAL __BATCH_SIZE)
        break()
      endif()
      list(SUBLIST HIP_FILE_ARGS ${__BATCH_SIZE} -1 HIP_FILE_ARGS)
      list(LENGTH HIP_FILE_ARGS NUM_HIP_FILE_ARGS)
    endwhile()
  else()
    set(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})
  endif()
  set(${TARGET_ARGS_VAR} ${TARGET_SOURCE_ARGS} PARENT_SCOPE)
endfunction()


# unify -> set property -> add library
function(hytlass_add_library NAME)

  set(options SKIP_GENCODE_FLAGS)
  set(oneValueArgs EXPORT_NAME)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  hytlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  hytlass_correct_source_file_language_property(${TARGET_SOURCE_ARGS})
  add_library(${NAME} ${TARGET_SOURCE_ARGS} "")


  hytlass_apply_standard_compile_options(${NAME})
  if (NOT __SKIP_GENCODE_FLAGS)
  hytlass_apply_hip_gencode_flags(${NAME})
  endif()

  target_compile_features(
   ${NAME}
   INTERFACE
   cxx_std_11
   )

  if(__EXPORT_NAME)
    add_library(hygon::hytlass::${__EXPORT_NAME} ALIAS ${NAME})
    set_target_properties(${NAME} PROPERTIES EXPORT_NAME ${__EXPORT_NAME})
  endif()

endfunction()

function(hytlass_add_executable NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  hytlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  hytlass_correct_source_file_language_property(${TARGET_SOURCE_ARGS})
  add_executable(${NAME} ${TARGET_SOURCE_ARGS})

  hytlass_apply_standard_compile_options(${NAME})
  hytlass_apply_hip_gencode_flags(${NAME})

  target_compile_features(
   ${NAME}
   INTERFACE
   cxx_std_11
   )

endfunction()

function(hytlass_target_sources NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  hytlass_unify_source_files(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})
  hytlass_correct_source_file_language_property(${TARGET_SOURCE_ARGS})
  target_sources(${NAME} ${TARGET_SOURCE_ARGS})

endfunction()
