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

#include "hytlass_unit_test.h"

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <hute/tensor.hpp>

#include <hute/atom/copy_traits_gfx928.hpp>

#define ELEMENTS_SIZE_PER_COL 32
#define ELEMENTS_PER_ROW 32

using namespace hute;

template <class T>
__global__ void
DS_READ_M_test_device(T* g_in, T* g_out)
{
  using RegType = T;
  constexpr int count = ELEMENTS_SIZE_PER_COL / sizeof(T);
  constexpr int reg_count = 16 / sizeof(T); // 2 * 8B / sizeof
  int tid = threadIdx.x;
  int row = tid >> 4;
  int col = tid & 15;

  // load input gmem -> smem
  __shared__ T smem[ELEMENTS_PER_ROW * count];
  for (int i = 0; i < count; ++i) {
    smem[tid * count + i] = g_in[tid * count + i];
  }

  __syncthreads();

  RegType reg[count];
  for (int i = 0; i < count; ++i) {
    reg[i] = 0;
  }

  // load smem -> rmem using DS_READ_M
  uint128_t* smem_ptr = reinterpret_cast<uint128_t*>(smem) + tid;

  hute::copy_ds_read_m(smem_ptr, reg);

  // store output rmem -> gmem
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half_t> || std::is_same_v<T, uint8_t>) {
    constexpr int rows_per_thread = 2;
    constexpr int elements_per_row = reg_count / rows_per_thread;
    constexpr int col_stride = ELEMENTS_PER_ROW / rows_per_thread;
    
    for (int i = 0; i < rows_per_thread; ++i) {
      for (int j = 0; j < elements_per_row; ++j) {
        int reg_index = i * elements_per_row + j;
        int target_row = row * elements_per_row + j;
        int target_col = col + i * col_stride;
        g_out[target_row * ELEMENTS_PER_ROW + target_col] = reg[reg_index];
      }
    }
  } else {
    static_assert(sizeof(T) == 0, "Unsupported type for DS_READ_M");
  }
}

template <class TiledCopy, class SmemLayout, class T>
__global__ void
DS_READ_M_test_device_hute(T* g_in, T* g_out,
                      TiledCopy tiled_copy, SmemLayout smem_layout)
{

  using namespace hute;

  __shared__ T smem[size(smem_layout)];

  auto t_g_in  = make_tensor(make_gmem_ptr(g_in),  smem_layout);
  auto t_g_out = make_tensor(make_gmem_ptr(g_out), smem_layout);
  auto t_smem  = make_tensor(make_smem_ptr(smem),  smem_layout);

  int tid = threadIdx.x;

  // Load input gmem -> smem
  for (int i = tid; i < size(t_smem); i += size(tiled_copy)) {
    t_smem(i) = t_g_in(i);
  }

  __syncthreads();

  auto thr_copy = tiled_copy.get_thread_slice(tid);

  auto tXsX = thr_copy.partition_S(t_smem);   // (V,M,N)
  auto tXgX = thr_copy.partition_D(t_g_out);  // (V,M,N)

  auto tXrX = make_tensor<T>(shape(tXgX)); // (V,M,N)
  clear(tXrX);  // Just to make sure

  // Copy smem -> rmem via tiled_copy (DS_READ_M, LDS)
  copy(tiled_copy, tXsX, tXrX);

  // Output rmem -> gmem
  copy(tXrX, tXgX);

}

template<class T>
__host__ void DS_READ_M_test() {
  constexpr int count = ELEMENTS_PER_ROW * ELEMENTS_SIZE_PER_COL / sizeof(T);
  thrust::host_vector<T> h_in(count);
  for (int i = 0; i < count; ++i) {
    h_in[i] = T(i);
  }
  thrust::device_vector<T> d_in = h_in;

  thrust::device_vector<T> d_out(count);
  DS_READ_M_test_device<T><<<1, 64>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()));
  thrust::host_vector<T> h_out = d_out;
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  HYTLASS_TRACE_HOST("DS_READ_M" << typeid(T).name() << "DS_READ_M_test_device SUCCESS\n");
}

template<class T, class TiledCopy, class Layout>
__host__ void DS_READ_M_test_hute(TiledCopy tiled_copy, Layout smem_layout) {
  constexpr int count = ELEMENTS_PER_ROW * ELEMENTS_SIZE_PER_COL / sizeof(T);
  thrust::host_vector<T> h_in(count);
  for (int i = 0; i < count; ++i) {
    h_in[i] = T(i);
  }
  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(count);

  DS_READ_M_test_device_hute<<<1, int(size(tiled_copy))>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    tiled_copy,
    smem_layout);
  
  thrust::host_vector<T> h_out = d_out;
  for (int i = 0; i < size(smem_layout); ++i) {
    EXPECT_EQ(h_out[i], h_in[i]);
  }
  HYTLASS_TRACE_HOST("HuTe DS_READ_M " << typeid(T).name() << " SUCCESS\n");
}

TEST(GFX928_HuTe, DS_READ_M)
{
  //
  // DS_READ_M float 
  //
  
  DS_READ_M_test<float>();

  //
  // DS_READ_M half_t
  //

  DS_READ_M_test<half_t>();

  //
  // DS_READ_M uint8_t
  //

  DS_READ_M_test<uint8_t>();

  //
  // HuTe DS_READ_M
  //

  {
  auto smem_layout = Layout<Shape <_32, _8>,
                            Stride< _1, _32>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<GFX928_DS_READ_DS_M32x8_B32, float>{},
                                    Layout<Shape<_8, _8>>{},
                                    Layout<Shape<_1, _4>>{});

  DS_READ_M_test_hute<float>(tiled_copy, smem_layout);
  }

  {
  auto smem_layout = Layout<Shape <_32, _16>,
                            Stride< _1, _32>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, half_t>{},
                                    Layout<Shape<_16, _4>>{},
                                    Layout<Shape<_1, _8>>{});

  DS_READ_M_test_hute<half_t>(tiled_copy, smem_layout);
  }

  {
  auto smem_layout = Layout<Shape <_32,_32>,
                            Stride< _1,_32>>{};
  auto tiled_copy = make_tiled_copy(Copy_Atom<GFX928_DS_READ_DS_M32x32_B8, uint8_t>{},
                                    Layout<Shape<_32, _2>>{},
                                    Layout<Shape<_1, _16>>{});

  DS_READ_M_test_hute<uint8_t>(tiled_copy, smem_layout);
  }

  HYTLASS_TRACE_HOST("PASS");
}
