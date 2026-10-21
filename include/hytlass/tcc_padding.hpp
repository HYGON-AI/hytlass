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
#pragma once

#include "hytlass/detail/layout.hpp"
#include "hytlass/gemm/kernel/padding_kernel.hpp"

#define PADDING_A 96
#define PADDING_B 32
#define PADDING_BLOCK_SIZE 256

namespace hytlass {

using hytlass::detail::GemmLayout;

static bool 
check_padding_arch() {
  int device_id, eco_info;
  hipDeviceAttribute_t attr = hipDeviceAttributeEcoInfo;

  if (hipGetDevice(&device_id) != hipSuccess) {
    return false;
  }
  if (hipDeviceGetAttribute(&eco_info, attr, device_id) != hipSuccess) {
    return false;
  }
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, device_id) != hipSuccess) {
    return false;
  }
  return ((strncmp(prop.gcnArchName, "gfx928", strlen("gfx928")) == 0) && (eco_info == 0));
}

template <typename ElementType>
HYTLASS_HOST_DEVICE
constexpr bool is_padding_required(int memory_order_length) {
  constexpr int tcc_lane_bytes = 2048;
  size_t total_bytes = sizeof(ElementType) * memory_order_length;
  if (total_bytes % tcc_lane_bytes == 0) {
    return true;
  }
  return false;
}

template <typename Element>
void launch_padding_kernel(const void *input, void *output, int lda, int row, int col, int ldb, hipStream_t stream = nullptr) {
  constexpr int scale = sizeof(float4) / sizeof(Element);
  const int padding_stride = PADDING_BLOCK_SIZE * scale;
  dim3 grid((row + padding_stride - 1) / padding_stride, col, 1);
  dim3 block(PADDING_BLOCK_SIZE, 1, 1);
  hytlass::gemm::kernel::device_padding_kernel<Element><<<grid, block>>>(input, output, lda, row, col, ldb);
  hipError_t status = hipGetLastError();
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("Failed to launch padding kernel: ") + hipGetErrorString(status));
  }
}

template <typename Arguments, typename ElementA, typename ElementB, typename StrideA, typename StrideB>
size_t get_padding_workspace_size(Arguments const& args) {
  size_t workspace_size = 0;
  constexpr GemmLayout layout = detail::get_gemm_layout<StrideA, StrideB>();
  auto M = hute::shape<0>(args.problem_shape);
  auto N = hute::shape<1>(args.problem_shape);
  auto K = hute::shape<2>(args.problem_shape);

  if constexpr (layout == GemmLayout::TN) {
    if (is_padding_required<ElementA>((int)K) || is_padding_required<ElementB>(int(K))) {
      workspace_size += (PADDING_A + K) * M * sizeof(ElementA);
      workspace_size += (PADDING_B + K) * N * sizeof(ElementB);
    }
  } else if constexpr (layout == GemmLayout::NN) {
    if (is_padding_required<ElementA>((int)M)) {
      workspace_size += (PADDING_A + M) * K * sizeof(ElementA);
    }
    if (is_padding_required<ElementB>((int)K)) {
      workspace_size += (PADDING_B + K) * N * sizeof(ElementB);
    }
  } else if constexpr (layout == GemmLayout::NT) {
    if (is_padding_required<ElementA>((int)M)) {
      workspace_size += (PADDING_A + M) * K * sizeof(ElementA);
    }
    if (is_padding_required<ElementB>((int)N)) {
      workspace_size += (PADDING_B + N) * K * sizeof(ElementB);
    }
  } else if constexpr (layout == GemmLayout::TT) {
    if (is_padding_required<ElementA>((int)K)) {
      workspace_size += (PADDING_A + K) * M * sizeof(ElementA);
    }
    if (is_padding_required<ElementB>((int)N)) {
      workspace_size += (PADDING_B + N) * K * sizeof(ElementB);
    }
  }

  return workspace_size;
}

template <typename Arguments, typename ElementA, typename ElementB, typename StrideA, typename StrideB>
hytlass::Status initialize_padding_workspace(Arguments const& args, void *workspace, hipStream_t stream) {
  constexpr GemmLayout layout = detail::get_gemm_layout<StrideA, StrideB>();
  auto M = hute::shape<0>(args.problem_shape);
  auto N = hute::shape<1>(args.problem_shape);
  auto K = hute::shape<2>(args.problem_shape);
  uint64_t ptr_offset = 0;

  if constexpr (layout == GemmLayout::TN) {
    if (is_padding_required<ElementA>((int)K) || is_padding_required<ElementB>(int(K))) {
      launch_padding_kernel<ElementA>((const void*)(args.mainloop.ptr_A), 
                                      workspace, 
                                      (int)K, 
                                      (int)K, 
                                      (int)M, 
                                      int(PADDING_A + K), 
                                      stream);
      ptr_offset += (PADDING_A + K) * M * sizeof(ElementA);
      launch_padding_kernel<ElementB>((const void*)(args.mainloop.ptr_B), 
                                      (void *)(reinterpret_cast<uint8_t *>(workspace) + ptr_offset), 
                                      (int)K, 
                                      (int)K, 
                                      (int)N, 
                                      (int)(PADDING_B + K), 
                                      stream);
    }
  }
  else if constexpr (layout == GemmLayout::NN) {
    if (is_padding_required<ElementA>((int)M)) {
      launch_padding_kernel<ElementA>((const void*)(args.mainloop.ptr_A), 
                                      workspace, 
                                      (int)M, 
                                      (int)M, 
                                      (int)K, 
                                      (int)(PADDING_A + M), 
                                      stream);
      ptr_offset += (PADDING_A + M) * K * sizeof(ElementA);
    }
    if (is_padding_required<ElementB>((int)K)) {
      launch_padding_kernel<ElementB>((const void*)(args.mainloop.ptr_B),
                                      (void *)(reinterpret_cast<uint8_t *>(workspace) + ptr_offset),
                                      (int)K, 
                                      (int)K, 
                                      (int)N, 
                                      (int)(PADDING_B + K), 
                                      stream);
    }
  }
  else if constexpr (layout == GemmLayout::NT) {
    if (is_padding_required<ElementA>((int)M)) {
      launch_padding_kernel<ElementA>((const void*)(args.mainloop.ptr_A), 
                                      workspace, 
                                      (int)M, 
                                      (int)M, 
                                      (int)K, 
                                      (int)(PADDING_A + M), 
                                      stream);
      ptr_offset += (PADDING_A + M) * K * sizeof(ElementA);
    }
    if (is_padding_required<ElementB>((int)N)) {
      launch_padding_kernel<ElementB>((const void*)(args.mainloop.ptr_B),
                                      (void *)(reinterpret_cast<uint8_t *>(workspace) + ptr_offset),
                                      (int)N, 
                                      (int)N, 
                                      (int)K, 
                                      (int)(PADDING_B + N), 
                                      stream);
    }
  }
  else if constexpr (layout == GemmLayout::TT) {
    if (is_padding_required<ElementA>((int)K)) {
      launch_padding_kernel<ElementA>((const void*)(args.mainloop.ptr_A), 
                                      workspace, 
                                      (int)K, 
                                      (int)K, 
                                      (int)M, 
                                      (int)(PADDING_A + K), 
                                      stream);

      ptr_offset += (PADDING_A + K) * M * sizeof(ElementA);
    }
    if (is_padding_required<ElementB>((int)N)) {
      launch_padding_kernel<ElementB>((const void*)(args.mainloop.ptr_B),
                                      (void *)(reinterpret_cast<uint8_t *>(workspace) + ptr_offset),
                                      (int)N, 
                                      (int)N, 
                                      (int)K, 
                                      (int)(PADDING_B + N), 
                                      stream);
    }
  }
  return hytlass::Status::kSuccess;
}

} // namespace hytlass