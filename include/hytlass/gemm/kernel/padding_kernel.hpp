// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include "hip/hip_runtime.h"

#define PADDING_BLOCK_SIZE 256

namespace hytlass::gemm::kernel
{

template <typename T>
__global__
void device_padding_kernel(const void *input, void *ouput, int lda, int row, int col, int ldb, T *dumy = nullptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x; // rowIdx
  int y = blockIdx.y * blockDim.y + threadIdx.y; // colIdx

  float4 *tempIn = (float4 *)input;
  float4 *tempOut = (float4 *)ouput;
  static constexpr int scale = sizeof(float4) / sizeof(T);
  {
    if (y < col && x < row / scale) {
      tempOut[y * ldb / scale + x] = tempIn[y * lda / scale + x];
    }
  }
}

} // namespace hytlass::gemm::kernel