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

message(STATUS "Configuring hipblas ...")

if((DEFINED HYTLASS_ENABLE_HIPBLAS AND NOT HYTLASS_ENABLE_HIPBLAS) OR
   (DEFINED HIPBLAS_ENABLED AND NOT HIPBLAS_ENABLED))
  
  # Don't add hipBLAS if it's defined and false, assume it's not found.

  set(HIPBLAS_FOUND OFF)
  message(STATUS "hipBLAS Disabled.")

elseif(NOT TARGET hipblas)
 
  find_path(
    _HIPBLAS_INCLUDE_DIR
    NAMES hipblas.h
    HINTS
      ${DTK_ROOT_DIR}/hipblas
    PATH_SUFFIXES
      include
    )

  find_library(
    _HIPBLAS_LIBRARY
    NAMES hipblas
    HINTS
      ${DTK_ROOT_DIR}/hipblas
    PATH_SUFFIXES
      lib64
      lib/x64
      lib
    )

  if(_HIPBLAS_INCLUDE_DIR AND _HIPBLAS_LIBRARY)

    message(STATUS "hipBLAS: ${_HIPBLAS_LIBRARY}")
    message(STATUS "hipBLAS: ${_HIPBLAS_INCLUDE_DIR}")
    
    set(HIPBLAS_FOUND ON CACHE INTERNAL "hipblas Library Found")
    set(HIPBLAS_LIBRARY ${_HIPBLAS_LIBRARY})
    set(HIPBLAS_INCLUDE_DIR ${_HIPBLAS_INCLUDE_DIR})

  else()

    message(STATUS "hipblas not found.")
    set(HIPBLAS_FOUND OFF CACHE INTERNAL "hipblas Library Found")

  endif()

endif()

set(HYTLASS_ENABLE_HIPBLAS ${HIPBLAS_FOUND} CACHE BOOL "Enable HYTLASS to build with hipBLAS library.")

if(HYTLASS_ENABLE_HIPBLAS AND NOT HIPBLAS_FOUND)
  message(FATAL_ERROR "HYTLASS_ENABLE_HIPBLAS enabled but hipBLAS library could not be found.")
endif()

if(HYTLASS_ENABLE_HIPBLAS AND NOT TARGET hipblas)

  if(WIN32)
    add_library(hipblas STATIC IMPORTED GLOBAL)
  else()
    add_library(hipblas SHARED IMPORTED GLOBAL)
  endif()

  add_library(hip::hipblas ALIAS hipblas)

  set_property(
    TARGET hipblas
    PROPERTY IMPORTED_LOCATION
    ${HIPBLAS_LIBRARY})
    
  target_include_directories(
    hipblas
    INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${HIPBLAS_INCLUDE_DIR}>)

  find_library(
    _HIPBLASLT_LIBRARY
    NAMES hipblasLt
    HINTS
      ${DTK_ROOT_DIR}/hipblaslt
    PATH_SUFFIXES
      lib64
      lib/x64
      lib
    )

  if(_HIPBLASLT_LIBRARY AND NOT TARGET hipblasLt)

    add_library(hipblasLt SHARED IMPORTED GLOBAL)

    set_property(
      TARGET hipblasLt
      PROPERTY IMPORTED_LOCATION
      ${_HIPBLASLT_LIBRARY})
  
    add_library(hip::hipblasLt ALIAS hipblasLt)

    target_link_libraries(hipblas INTERFACE hipblasLt)

  endif()

endif()

message(STATUS "Configuring hipBLAS ... done.")
