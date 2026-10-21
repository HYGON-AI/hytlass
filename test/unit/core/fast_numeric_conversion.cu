/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Unit tests for conversion operators.
*/
#include "hip/hip_runtime.h"
#include "../common/hytlass_unit_test.h"

#include "hytlass/numeric_conversion.h"

#include "hytlass/layout/matrix.h"
#include "hytlass/util/host_tensor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace core {
namespace kernel {

/// Simple conversion function
template <typename Destination, typename Source, int Count>
__global__ void convert(
  hytlass::Array<Destination, Count> *destination,
  hytlass::Array<Source, Count> const *source) {

  hytlass::FastNumericArrayConverter<Destination, Source, Count> convert;

  *destination = convert(*source);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Destination, typename Source, int Count>
void run_test_integer_range_limited() {
  const int kN = Count;

  dim3 grid(1, 1);
  dim3 block(1, 1);

  hytlass::HostTensor<Destination, hytlass::layout::RowMajor> destination({1, kN});
  hytlass::HostTensor<Source, hytlass::layout::RowMajor> source({1, kN});

  for (int i = 0; i < kN; ++i) {
    source.host_view().at({0, i}) = Source(i % 4);
  }

  source.sync_device();

  convert<Destination, Source, kN><<< grid, block >>>(
    reinterpret_cast<hytlass::Array<Destination, kN> *>(destination.device_data()),
    reinterpret_cast<hytlass::Array<Source, kN> const *>(source.device_data())
  );

  destination.sync_host();

  for (int i = 0; i < kN; ++i) {
    EXPECT_TRUE(float(destination.host_view().at({0, i})) == float(source.host_view().at({0, i})));
  }
}


template <typename Destination, typename Source, int Count>
void run_test_integer_range_all() {
  const int kN = Count;

  dim3 grid(1, 1);
  dim3 block(1, 1);

  hytlass::HostTensor<Destination, hytlass::layout::RowMajor> destination({1, kN});
  hytlass::HostTensor<Source, hytlass::layout::RowMajor> source({1, kN});

  int const kIntSourceMin = hytlass::platform::numeric_limits<Source>::lowest();
  int const kIntSourceMax = hytlass::platform::numeric_limits<Source>::max();
  int const kIntRange = kIntSourceMax - kIntSourceMin + 1;

  for (int i = 0; i < kN; ++i) {
    source.host_view().at({0, i}) = Source(kIntSourceMin + (i % kIntRange));
  }

  source.sync_device();

  convert<Destination, Source, kN><<< grid, block >>>(
    reinterpret_cast<hytlass::Array<Destination, kN> *>(destination.device_data()),
    reinterpret_cast<hytlass::Array<Source, kN> const *>(source.device_data())
  );

  destination.sync_host();

  // Verify conversion
  bool passed = true;
  for (int i = 0; i < kN; ++i) {
    if(!(float(destination.host_view().at({0, i})) == float(source.host_view().at({0, i})))) {
      passed = false;
      break;
    }
  }
  EXPECT_TRUE(passed) << " FastNumericArrayConverter failed";

   // Print out results for the failed conversion.
   if (!passed) {
    for (int i = 0; i < kN; ++i) {
        std::cout << "source(" << float(source.host_view().at({0, i})) << ") -> "
                  << "destination ("<< float(destination.host_view().at({0, i})) << ")" << std::endl;
    }
   }
   std::flush(std::cout);
}

} // namespace kernel
} // namespace core
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////
TEST(FastNumericConversion, s32_to_f32) {
  int const kN = 4;
  using Source = int;
  using Destination = float;
  test::core::kernel::run_test_integer_range_limited<Destination, Source, kN>();
}

TEST(FastNumericConversion, s8_to_f32_array) {
  int const kN = 256;
  using Source = int8_t;
  using Destination = float;
  test::core::kernel::run_test_integer_range_all<Destination, Source, kN>();
}

TEST(FastNumericConversion, u8_to_f32_array) {
  int const kN = 256;
  using Source = uint8_t;
  using Destination = float;
  test::core::kernel::run_test_integer_range_all<Destination, Source, kN>();
}

TEST(FastNumericConversion, s8_to_f16_array) {
  int const kN = 256;
  using Source = int8_t;
  using Destination = hytlass::half_t;
  test::core::kernel::run_test_integer_range_all<Destination, Source, kN>();
}

TEST(FastNumericConversion, u8_to_f16_array) {
  int const kN = 256;
  using Source = uint8_t;
  using Destination = hytlass::half_t;
  test::core::kernel::run_test_integer_range_all<Destination, Source, kN>();
}

TEST(FastNumericConversion, u8_to_bf16_array) {
  int const kN = 256;
  using Source = uint8_t;
  using Destination = hytlass::bfloat16_t;
  test::core::kernel::run_test_integer_range_all<Destination, Source, kN>();
}

TEST(FastNumericConversion, s8_to_bf16_array) {
  int const kN = 256;
  using Source = int8_t;
  using Destination = hytlass::bfloat16_t;
  test::core::kernel::run_test_integer_range_all<Destination, Source, kN>();
}

