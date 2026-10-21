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

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Simple conversion function
template <typename Destination, typename Source, int Count>
__global__ void convert(
  hytlass::Array<Destination, Count> *destination,
  hytlass::Array<Source, Count> const *source) {

  hytlass::NumericArrayConverter<Destination, Source, Count> convert;

  *destination = convert(*source);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Destination, typename Source, int Count>
void run_test(const char dest_name[], const char source_name[], const int range = 4, const int offset = 0) {
  const int kN = Count;

  dim3 grid(1, 1);
  dim3 block(1, 1);

  hytlass::HostTensor<Destination, hytlass::layout::RowMajor> destination({1, kN});
  hytlass::HostTensor<Source, hytlass::layout::RowMajor> source({1, kN});
  auto source_ref = source.host_ref();
  auto destination_ref = destination.host_ref();

  for (int i = 0; i < kN; ++i) {
    source_ref.at({0, i}) = Source(i % range + offset);
  }

  source.sync_device();

  convert<Destination, Source, kN><<< grid, block >>>(
    reinterpret_cast<hytlass::Array<Destination, kN> *>(destination.device_data()),
    reinterpret_cast<hytlass::Array<Source, kN> const *>(source.device_data())
  );

  destination.sync_host();

  for (int i = 0; i < kN; ++i) {
    EXPECT_TRUE(float(destination_ref.at({0, i})) == float(source_ref.at({0, i})))
      << "Destination type: " << dest_name << " "<< float(destination_ref.at({0, i}))
      << ", Source type: " << source_name << " " << float(source_ref.at({0, i}))
      << ", Count: " << Count;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Destination, typename Source, typename ScaleFactor, int Count>
__global__ void convert_with_scale_factor(
  hytlass::Array<Destination, Count> *destination,
  hytlass::Array<Source, Count> const *source,
  hytlass::Array<ScaleFactor, Count> const *scale_factor) {

  hytlass::NumericArrayConverter<Destination, Source, Count> convert;

  *destination = convert(*source, *scale_factor);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Destination, typename Source, typename ScaleFactor,  int Count>
void run_test_with_scalefactor(const char dest_name[], const char source_name[], const char scale_factor_name[], const int range = 4, const int offset = 0) {
  const int kN = Count;

  dim3 grid(1, 1);
  dim3 block(1, 1);

  hytlass::HostTensor<Destination, hytlass::layout::RowMajor> destination({1, kN});
  hytlass::HostTensor<Source, hytlass::layout::RowMajor> source({1, kN});
  hytlass::HostTensor<ScaleFactor, hytlass::layout::RowMajor> scale_factor({1, kN});
  auto source_ref = source.host_ref();
  auto destination_ref = destination.host_ref();
  auto scale_factor_ref = scale_factor.host_ref();


  for (int i = 0; i < kN; ++i) {
    source_ref.at({0, i}) = Source(i % range + offset);
  }

  for (int i = 0; i < kN; ++i) {
    scale_factor_ref.at({0, i}) = ScaleFactor(1 + i % 8);
  }

  source.sync_device();
  scale_factor.sync_device();

  convert_with_scale_factor<Destination, Source, ScaleFactor, kN><<< grid, block >>>(
    reinterpret_cast<hytlass::Array<Destination, kN> *>(destination.device_data()),
    reinterpret_cast<hytlass::Array<Source, kN> const *>(source.device_data()),
    reinterpret_cast<hytlass::Array<ScaleFactor, kN> const *>(scale_factor.device_data())
  );

  destination.sync_host();

  for (int i = 0; i < kN; ++i) {
    float ref = float(source_ref.at({0, i})) / float(scale_factor_ref.at({0, i}));
    bool pass = float(destination_ref.at({0, i})) == ref;
    EXPECT_TRUE(pass) 
      << "Destination type: " << dest_name << " "<< float(destination_ref.at({0, i})) << std::endl
      << ", Source type: " << source_name << " " << float(source_ref.at({0, i})) << std::endl
      << ", Scalefactor type: " << source_name << " " << float(scale_factor_ref.at({0, i})) << std::endl
      << ", idx: " << i << std::endl;
  }
}

} // namespace kernel
} // namespace core
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(NumericConversion, f32_to_f16_rn) {
  constexpr int kN = 1;
  using Source = float;
  const char source_name[] = "float";
  using Destination = hytlass::half_t;
  const char dest_name[] = "half_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f32x2_to_f16x2_rn) {
  constexpr int kN = 2;
  using Source = float;
  const char source_name[] = "float";
  using Destination = hytlass::half_t;
  const char dest_name[] = "half_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f32x8_to_f16x8_rn) {
  constexpr int kN = 8;
  using Source = float;
  const char source_name[] = "float";
  using Destination = hytlass::half_t;
  const char dest_name[] = "half_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(NumericConversion, f16_to_f32_rn) {  
  int const kN = 1;
  using Source = hytlass::half_t;
  const char source_name[] = "half_t";
  using Destination = float;
  const char dest_name[] = "float";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

TEST(NumericConversion, f16x8_to_f32x8_rn) {
  int const kN = 8;
  using Source = hytlass::half_t;
  const char source_name[] = "half_t";
  using Destination = float;
  const char dest_name[] = "float";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

/////////////////////////////////////////////////////////////////////////////////////////////////


TEST(NumericConversion, f32x8_to_s8x8_rn) {

  int const kN = 8;
  using Source = float;
  const char source_name[] = "float";
  using Destination = int8_t;
  const char dest_name[] = "int8_t";
  test::core::kernel::run_test<Destination, Source, kN>(dest_name, source_name);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct GetName {
  static constexpr char name[] = "UNSUPPORTED";
};

template <>
struct GetName<uint8_t> {
  static constexpr char name[] = "uint8_t";
};

template <>
struct GetName<int8_t> {
  static constexpr char name[] = "int8_t";
};

template <>
struct GetName<hytlass::half_t> {
  static constexpr char name[] = "half_t";
};

template <>
struct GetName<hytlass::bfloat16_t> {
  static constexpr char name[] = "bfloat16_t";
};

template <>
struct GetName<float> {
  static constexpr char name[] = "float";
};

template <typename Result_, typename Source_>
struct ResultSourcePair {
  using Result = Result_;
  using Source = Source_;
};

using VectorConvertTypes = ::testing::Types<
  ResultSourcePair<float, int8_t>,
  ResultSourcePair<float, uint8_t>,

  ResultSourcePair<hytlass::half_t, int8_t>,
  ResultSourcePair<hytlass::half_t, uint8_t>,

  ResultSourcePair<hytlass::bfloat16_t, uint8_t>,
  ResultSourcePair<hytlass::bfloat16_t, int8_t>
>;

/////////////////////////////////////////////////////////////////////////////////////////////////
