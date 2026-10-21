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
  \brief Functor performing linear combination operations used by epilogues.
*/

#pragma once

#include <hip/hip_fp16.h>

#include "hytlass/hytlass.h"
#include "hytlass/numeric_types.h"
#include "hytlass/array.h"
#include "hytlass/functional.h"
#include "hytlass/numeric_conversion.h"
#include "hytlass/epilogue/thread/activation.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element, int ElementsPerAccess>
struct ArrayMaximum {

  HYTLASS_HOST_DEVICE
  Array<Element, ElementsPerAccess> operator()(
    Array<Element, ElementsPerAccess>  const &lhs,
    Array<Element, ElementsPerAccess>  const &rhs) const {

    Array<Element, ElementsPerAccess> result;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      result[i] = platform::max(lhs[i].get(), rhs[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  Array<Element, ElementsPerAccess> operator()(
    Array<Element, ElementsPerAccess>  const &lhs,
    Element                                   rhs) const {

    Array<Element, ElementsPerAccess> result;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      result[i] = platform::max(lhs[i].get(), rhs);
    }

    return result;
  }
};


/// Partial specialization: Element=float
template <int ElementsPerAccess>
struct ArrayMaximum<float, ElementsPerAccess> {

  HYTLASS_HOST_DEVICE
  Array<float, ElementsPerAccess> operator()(
    Array<float, ElementsPerAccess>  const &lhs,
    Array<float, ElementsPerAccess>  const &rhs) const {

    Array<float, ElementsPerAccess> result;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      result[i] = fmax(lhs[i], rhs[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  Array<float, ElementsPerAccess> operator()(
    Array<float, ElementsPerAccess>  const &lhs,
    float rhs) const {

    Array<float, ElementsPerAccess> result;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      result[i] = fmax(lhs[i], rhs);
    }

    return result;
  }
};

/// Partial specialization: Element=half
template <int ElementsPerAccess>
struct ArrayMaximum<half_t, ElementsPerAccess> {

  HYTLASS_DEVICE
  Array<half_t, ElementsPerAccess> operator()(
    Array<half_t, ElementsPerAccess>  const &lhs,
    Array<half_t, ElementsPerAccess>  const &rhs) const {

    Array<half_t, ElementsPerAccess> result;

    #if __HIP_DEVICE_COMPILE__ >= 800
    int const kVectorCount = ElementsPerAccess / 2;


    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(lhs.raw_data());
    __half2 const *rhs_ptr = reinterpret_cast<__half2 const *>(rhs.raw_data());
    __half2       *res_ptr = reinterpret_cast<__half2 *>(result.raw_data());

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVectorCount; ++i) {
      res_ptr[i] = __hmax2(lhs_ptr[i], rhs_ptr[i]);
    }

    static_assert(!(ElementsPerAccess % 2), "Output array must be divisible by vector length.");

    #else
    __half const *lhs_ptr = reinterpret_cast<__half const *>(lhs.raw_data());
    __half const *rhs_ptr = reinterpret_cast<__half const *>(rhs.raw_data());
    __half       *res_ptr = reinterpret_cast<__half       *>(result.raw_data());

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      res_ptr[i] = ((lhs_ptr[i] < rhs_ptr[i]) ? rhs_ptr[i] : lhs_ptr[i]);
    }

    #endif

    return result;
  }

  HYTLASS_DEVICE
  Array<half_t, ElementsPerAccess> operator()(
    Array<half_t, ElementsPerAccess>  const &lhs,
    half_t const &rhs) const {

    Array<half_t, ElementsPerAccess> result;

    #if __HIP_DEVICE_COMPILE__ >= 800
    int const kVectorCount = ElementsPerAccess / 2;


    __half rhs_raw = reinterpret_cast<__half const &>(rhs);
    __half2 rhs_pair = __half2half2(rhs_raw);

    __half2 const *lhs_ptr = reinterpret_cast<__half2 const *>(lhs.raw_data());
    __half2       *res_ptr = reinterpret_cast<__half2 *>(result.raw_data());

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVectorCount; ++i) {
      res_ptr[i] = __hmax2(lhs_ptr[i], rhs_pair);
    }

    static_assert(!(ElementsPerAccess % 2), "Output array must be divisible by vector length.");

    #else

    __half const *lhs_ptr = reinterpret_cast<__half const *>(lhs.raw_data());
    __half const  rhs_raw = reinterpret_cast<__half const &>(rhs);
    __half       *res_ptr = reinterpret_cast<__half       *>(result.raw_data());

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      res_ptr[i] = ((lhs_ptr[i] < rhs_raw) ? rhs_raw : lhs_ptr[i]);
    }

    #endif

    return result;
  }
};

/// Partial specialization: Element=bfloat16_t
template <int ElementsPerAccess>
struct ArrayMaximum<bfloat16_t, ElementsPerAccess> {

  using HipType   = __hip_bfloat16;
  using HipTypeV2 = __hip_bfloat162;

  HYTLASS_DEVICE
  Array<bfloat16_t, ElementsPerAccess> operator()(
    Array<bfloat16_t, ElementsPerAccess>  const &lhs,
    Array<bfloat16_t, ElementsPerAccess>  const &rhs) const {

    Array<bfloat16_t, ElementsPerAccess> result;

    #if __HIP_DEVICE_COMPILE__ >= 800
    int const kVectorCount = ElementsPerAccess / 2;


    HipTypeV2 const *lhs_ptr = reinterpret_cast<HipTypeV2 const *>(lhs.raw_data());
    HipTypeV2 const *rhs_ptr = reinterpret_cast<HipTypeV2 const *>(rhs.raw_data());
    HipTypeV2       *res_ptr = reinterpret_cast<HipTypeV2 *>(result.raw_data());

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVectorCount; ++i) {
      res_ptr[i] = __hmax2(lhs_ptr[i], rhs_ptr[i]);
    }

    #else
    HipType const *lhs_ptr = reinterpret_cast<HipType const *>(lhs.raw_data());
    HipType const *rhs_ptr = reinterpret_cast<HipType const *>(rhs.raw_data());
    HipType       *res_ptr = reinterpret_cast<HipType       *>(result.raw_data());

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      res_ptr[i] = ((lhs_ptr[i] < rhs_ptr[i]) ? rhs_ptr[i] : lhs_ptr[i]);
    }

    #endif

    return result;
  }

  HYTLASS_DEVICE
  Array<bfloat16_t, ElementsPerAccess> operator()(
    Array<bfloat16_t, ElementsPerAccess>  const &lhs,
    bfloat16_t                                   rhs) const {

    Array<bfloat16_t, ElementsPerAccess> result;

    #if __HIP_DEVICE_COMPILE__ >= 800
    int const kVectorCount = ElementsPerAccess / 2;


    HipType rhs_raw = reinterpret_cast<HipType const &>(rhs);
    HipTypeV2 rhs_pair = __bfloat162bfloat162(rhs_raw);

    HipTypeV2 const *lhs_ptr = reinterpret_cast<HipTypeV2 const *>(lhs.raw_data());
    HipTypeV2       *res_ptr = reinterpret_cast<HipTypeV2 *>(result.raw_data());

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVectorCount; ++i) {
      res_ptr[i] = __hmax2(lhs_ptr[i], rhs_pair);
    }

    static_assert(!(ElementsPerAccess % 2), "Output array must be divisible by vector length.");

    #else

    HipType const *lhs_ptr = reinterpret_cast<HipType const *>(lhs.raw_data());
    HipType const  rhs_raw = reinterpret_cast<HipType const &>(rhs);
    HipType       *res_ptr = reinterpret_cast<HipType       *>(result.raw_data());

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      res_ptr[i] = ((lhs_ptr[i] < rhs_raw) ? rhs_raw : lhs_ptr[i]);
    }

    #endif

    return result;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, int ElementsPerAccess>
struct ReluConditional {

  HYTLASS_HOST_DEVICE
  void operator()(
    bool conditional[],
    Array<Element, ElementsPerAccess> const &fragment, 
    Element threshold) const {

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      conditional[i] = !(fragment[i] < threshold);
    }
  }
};

template <int ElementsPerAccess>
struct ReluConditional<half_t, ElementsPerAccess> {

  HYTLASS_DEVICE
  void operator()(
    bool conditional[],
    Array<half_t, ElementsPerAccess> const &fragment, 
    half_t threshold) const {

    __half y = reinterpret_cast<__half const &>(threshold);
    __half const *x = reinterpret_cast<__half const *>(fragment.raw_data());

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      conditional[i] = !__hlt(x[i], y);
    }
  }
};

template <int ElementsPerAccess>
struct ReluConditional<bfloat16_t, ElementsPerAccess> {

  HYTLASS_DEVICE
  void operator()(
    bool conditional[],
    Array<bfloat16_t, ElementsPerAccess> const &fragment,
    bfloat16_t threshold) const {

    __hip_bfloat16 y = reinterpret_cast<__hip_bfloat16 const &>(threshold);
    __hip_bfloat16 const *x = reinterpret_cast<__hip_bfloat16 const *>(fragment.raw_data());

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      conditional[i] = !__hlt(x[i], y);
    }
  }
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This is a partial specialization for fused Bias and ReLU. It supports the option of packing
/// ReLU conditionals in a bit vector that may be used by backwards passes as an optimization.
///
/// This class can only be used with hytlass::epilogue::threadblock::EpilogueWithBroadcast<>.
///
/// This base class is meant to define the concept required of the
/// EpilogueWithBroadcast::OutputOp
template <
  typename ElementC_,
  typename ElementAccumulator_,
  typename ElementCompute_,
  typename ElementZ_,
  int ElementsPerAccess,
  bool StoreT_ = true,
  typename ElementVector_ = ElementC_
>
class LinearCombinationBiasRelu {
public:

  using ElementOutput = ElementC_;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementZ = ElementZ_;
  using ElementVector = ElementVector_;

  using ElementT = uint1b_t;

  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;

  using ElementwiseOp = ReLu<ElementCompute>;
  using BinaryOp = plus<ElementCompute>;

  // Indicates that this epilogue applies only one binary operation
  static bool const kIsSingleSource = true;

  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = Array<ElementOutput, kElementsPerAccess>;
  using FragmentZ = Array<ElementZ, kElementsPerAccess>;
  using FragmentT = Array<ElementT, kElementsPerAccess>;

  /// If true, the 'Z' tensor is stored
  static bool const kStoreZ = true;

  /// If true, the 'T' tensor is stored
  static bool const kStoreT = StoreT_;

  /// Host-constructable parameters structure
  struct Params {

    ElementCompute alpha;                  ///< scales accumulators
    ElementCompute beta;                   ///< scales source tensor
    ElementCompute const *alpha_ptr;       ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const *beta_ptr;        ///< pointer to source scalar - if not null, loads it from memory
    ElementZ threshold;                    ///< ReLu threshold

    //
    // Methods
    //
    //
    // Methods
    //

    HYTLASS_HOST_DEVICE
    Params(): 
      alpha(ElementCompute(1)), 
      beta(ElementCompute()), 
      alpha_ptr(nullptr), 
      beta_ptr(nullptr),
      threshold(ElementCompute()) { }

    HYTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute beta,
      ElementCompute threshold_ = ElementCompute()
    ): 
      alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) {

      NumericConverter<ElementZ, ElementCompute> convert_threshold;

      threshold = convert_threshold(threshold_);
    }

    HYTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha
    ): alpha(alpha), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr), threshold(ElementZ()) {

    }

    HYTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr,
      ElementCompute const *beta_ptr,
      ElementCompute threshold_ = ElementCompute()
    ): alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {

      NumericConverter<ElementZ, ElementCompute> convert_threshold;

      threshold = convert_threshold(threshold_);
    }

    HYTLASS_HOST_DEVICE
    Params(
      ElementCompute const *alpha_ptr
    ): alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(nullptr), threshold(ElementZ()) {
    }

  };

private:

  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;
  ElementZ threshold_;

public:

  //
  // Methods
  //

  /// Constructor from Params
  HYTLASS_HOST_DEVICE
  LinearCombinationBiasRelu(Params const &params) {

    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    threshold_ = params.threshold;
  }

  /// Returns true if source is needed
  HYTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  HYTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }

    if (k_partition != k_partition_count - 1) {
      // set to NaN to make ReLU no-op for all except last k partitions
      int64_t allones = -1;
      threshold_ = reinterpret_cast<ElementZ const &>(allones);
    }
  }

  /// Applies the operation when is_source_needed() is true
  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z, 
    FragmentT &frag_T, 
    FragmentAccumulator const &AB,
    FragmentC const &frag_C,
    FragmentCompute const &V) const {

    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_C = NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    FragmentCompute result_Z;

    bool conditions[kElementsPerAccess];

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {

      ElementCompute z = alpha_ * tmp_Accum[i];
      z += beta_ * tmp_C[i];

      z = binary_op(z, V[i]);
      result_Z[i] = z;
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    //
    // Compute condition
    //

    detail::ReluConditional<ElementZ, kElementsPerAccess> relu_conditional;
    relu_conditional(conditions, frag_Z, threshold_);

    detail::ArrayMaximum<ElementZ, kElementsPerAccess> maximum_op;
    frag_Z = maximum_op(frag_Z, threshold_);

    if (kStoreT) {
      PackPredicates<kElementsPerAccess> pack_predicates;
      frag_T = pack_predicates(conditions); 
    }
  }

  /// Applies the operation when is_source_needed() is false
  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z, 
    FragmentT &frag_T, 
    FragmentAccumulator const &AB,
    FragmentCompute const &V) const {

    BinaryOp binary_op;

    FragmentCompute tmp_Accum = NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute result_Z;

    bool conditions[kElementsPerAccess];

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute z = binary_op(alpha_ * tmp_Accum[i], V[i]);
      result_Z[i] = z;
    }

    NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    //
    // Compute condition
    //

    detail::ReluConditional<ElementZ, kElementsPerAccess> relu_conditional;
    relu_conditional(conditions, frag_Z, threshold_);

    detail::ArrayMaximum<ElementZ, kElementsPerAccess> maximum_op;
    frag_Z = maximum_op(frag_Z, threshold_);

    // 
    // Compute conditions
    //

    //
    // Store
    //
    if (kStoreT) {
      PackPredicates<kElementsPerAccess> pack_predicates;
      frag_T = pack_predicates(conditions);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace hytlass

/////////////////////////////////////////////////////////////////////////////////////////////////
