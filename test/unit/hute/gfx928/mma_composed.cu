// Copyright (c) 2026 Hygon Information Technology Co., Ltd. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "hytlass_unit_test.h"
#include <hip/hip_runtime.h>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <hute/tensor.hpp>

namespace {

template <class Op>
__global__ void composed_mma_kernel(
    typename hute::MMA_Traits<Op>::ValTypeA const* a,
    typename hute::MMA_Traits<Op>::ValTypeB const* b,
    float const* c, float* d) {
  using namespace hute;
  using Traits = MMA_Traits<Op>;
  using ALayout = typename Traits::ALayout;
  using BLayout = typename Traits::BLayout;
  using CLayout = typename Traits::CLayout;
  auto ra = make_tensor<typename Traits::ValTypeA>(make_shape(size<1>(ALayout{})));
  auto rb = make_tensor<typename Traits::ValTypeB>(make_shape(size<1>(BLayout{})));
  auto rc = make_tensor<float>(make_shape(size<1>(CLayout{})));
  auto rd = make_tensor_like(rc);
  for (int i = 0; i < size(ra); ++i) ra(i) = a[ALayout{}(threadIdx.x, i)];
  for (int i = 0; i < size(rb); ++i) rb(i) = b[BLayout{}(threadIdx.x, i)];
  for (int i = 0; i < size(rc); ++i) rc(i) = c[CLayout{}(threadIdx.x, i)];
  MMA_Atom<Op>{}.call(rd, ra, rb, rc);
  for (int i = 0; i < size(rd); ++i) d[CLayout{}(threadIdx.x, i)] = rd(i);
}

template <class Op>
void check_composed_mma() {
  using namespace hute;
  using Traits = MMA_Traits<Op>;
  using A = typename Traits::ValTypeA;
  using B = typename Traits::ValTypeB;
  constexpr int m = size<0>(typename Traits::Shape_MNK{});
  constexpr int n = size<1>(typename Traits::Shape_MNK{});
  constexpr int k = size<2>(typename Traits::Shape_MNK{});
  thrust::host_vector<A> a(m * k);
  thrust::host_vector<B> b(n * k);
  thrust::host_vector<float> c(m * n);
  for (int seed : {1, 7}) {
    for (int i = 0; i < m * k; ++i) a[i] = A(float((i * 7 + i / 5 + seed) % 7 - 3));
    for (int i = 0; i < n * k; ++i) b[i] = B(float((i * 3 + i / 7 + seed) % 5 - 2));
    for (int i = 0; i < m * n; ++i) c[i] = float((i + seed) % 7 - 3);
    thrust::device_vector<A> da = a;
    thrust::device_vector<B> db = b;
    thrust::device_vector<float> dc = c;
    thrust::device_vector<float> dd(m * n, -12345.0f);
    composed_mma_kernel<Op><<<1, 64>>>(
        thrust::raw_pointer_cast(da.data()),
        thrust::raw_pointer_cast(db.data()), thrust::raw_pointer_cast(dc.data()),
        thrust::raw_pointer_cast(dd.data()));
    ASSERT_EQ(hipSuccess, hipGetLastError());
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    thrust::host_vector<float> actual = dd;
    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < m; ++row) {
        float expected = c[row + m * col];
        for (int inner = 0; inner < k; ++inner)
          expected += float(a[row + m * inner]) * float(b[col + n * inner]);
        ASSERT_EQ(expected, actual[row + m * col])
            << "M=" << m << " N=" << n << " K=" << k << " seed=" << seed
            << " row=" << row << " col=" << col;
      }
    }
  }
}

TEST(GFX936_HuTe, ExistingMma16x16x8F32) {
  check_composed_mma<hute::GFX928_16x16x8_F32F32F32F32_NT>();
}
TEST(GFX936_HuTe, ExistingMma16x16x8TF32) {
  check_composed_mma<hute::GFX928_16x16x8_F32TF32TF32F32_NT>();
}

TEST(GFX936_HuTe, ExistingMma16x16x16F16) {
  check_composed_mma<hute::GFX928_16x16x16_F32F16F16F32_NT>();
}
TEST(GFX936_HuTe, ExistingMma16x16x16BF16) {
  check_composed_mma<hute::GFX928_16x16x16_F32BF16BF16F32_NT>();
}
TEST(GFX936_HuTe, ExistingMma16x32x16F16) {
  check_composed_mma<hute::GFX928_16x32x16_F32F16F16F32_NT>();
}
TEST(GFX936_HuTe, ExistingMma16x32x16BF16) {
  check_composed_mma<hute::GFX928_16x32x16_F32BF16BF16F32_NT>();
}

TEST(GFX936_HuTe, ComposedMma16x64x16F16) {
  check_composed_mma<hute::GFX928_16x64x16_F32F16F16F32_NT>();
}
TEST(GFX936_HuTe, ComposedMma16x64x16BF16) {
  check_composed_mma<hute::GFX928_16x64x16_F32BF16BF16F32_NT>();
}
TEST(GFX936_HuTe, ComposedMma16x64x32F16) {
  check_composed_mma<hute::GFX928_16x64x32_F32F16F16F32_NT>();
}
TEST(GFX936_HuTe, ComposedMma16x64x32BF16) {
  check_composed_mma<hute::GFX928_16x64x32_F32BF16BF16F32_NT>();
}
TEST(GFX938_HuTe, ComposedMma16x32x32E4M3) {
  if (std::string(GetHipDevice().gcnArchName).find("gfx938") != 0)
    GTEST_SKIP() << "Requires gfx938 LIT MMAC";
  check_composed_mma<hute::GFX938_16x32x32_F32F8F8F32E4M3E4M3_NT_LIT>();
}
TEST(GFX938_HuTe, ComposedMma16x64x64E4M3) {
  if (std::string(GetHipDevice().gcnArchName).find("gfx938") != 0)
    GTEST_SKIP() << "Requires gfx938 LIT MMAC";
  check_composed_mma<hute::GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT>();
}

} // namespace
