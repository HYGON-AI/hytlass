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

/*! \file
    \brief Contains code for debugging hytlass code
*/

#pragma once

#include "device_dump.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

/******************************************************************************
 * Debug and logging macros
 ******************************************************************************/

/**
 * Formats and prints the given message to stdout
 */
#if !defined(HIP_LOG)
#if !defined(__HIP_DEVICE_COMPILE__)
#define HIP_LOG(format, ...) printf(format, __VA_ARGS__)
#else
#define HIP_LOG(format, ...)                              \
  printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format, \
         blockIdx.x,                                       \
         blockIdx.y,                                       \
         blockIdx.z,                                       \
         threadIdx.x,                                      \
         threadIdx.y,                                      \
         threadIdx.z,                                      \
         __VA_ARGS__);
#endif
#endif

/**
 * Formats and prints the given message to stdout only if DEBUG is defined
 */
#if !defined(HIP_LOG_DEBUG)
#ifdef DEBUG
#define HIP_LOG_DEBUG(format, ...) HIP_LOG(format, __VA_ARGS__)
#else
#define HIP_LOG_DEBUG(format, ...)
#endif
#endif

/**
 * \brief The corresponding error message is printed to \p stderr (or \p stdout in device code)
 * along with the supplied source context.
 *
 * \return The HIP error.
 */
__host__ HYTLASS_DEVICE hipError_t hip_perror_impl(hipError_t error,
                                                     const char* expression,
                                                     const char* filename,
                                                     int line) {
  (void)filename;
  (void)line;
  if (error) {
#if !defined(__HIP_DEVICE_COMPILE__)
    fprintf(
        stderr, "HIP error %d [%s, %d] in expression '%s': %s\n", error, filename, line, expression, hipGetErrorString(error));
    fflush(stderr);
#else
    printf("HIP error %d [%s, %d] in expression '%s'\n", error, filename, line, expression);
#endif
  }
  return error;
}

/**
 * \brief Perror macro
 */
#ifndef HIP_PERROR
#define HIP_PERROR(e) hip_perror_impl((hipError_t)(e), #e, __FILE__, __LINE__)
#endif

/**
 * \brief Perror macro with exit
 */
#ifndef HIP_PERROR_EXIT
#define HIP_PERROR_EXIT(e)                                     \
  do { if (hip_perror_impl((hipError_t)(e), #e, __FILE__, __LINE__)) { \
    exit(1);                                                    \
  } } while (0)
#endif

/**
 * \brief Perror macro only if DEBUG is defined
 */
#ifndef HIP_PERROR_DEBUG
#ifdef DEBUG
#define HIP_PERROR_DEBUG(e) HIP_PERROR(e)
#else
#define HIP_PERROR_DEBUG(e) (e)
#endif
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// A small helper class to dump a type at compile time
// Usage:: DumpType<Class>::Class
template <typename T>
struct DebugType {};

template <typename T>
void DebugTypeFunc(T const& t) {
  T::t;
}

// A small helper class to dump a compile time constant at compile time
// Usage: DumpValue<Class::kConstant>::kConstant
template <int Value>
struct DebugValue {};
