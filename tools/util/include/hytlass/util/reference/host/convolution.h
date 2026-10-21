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
    \brief Reference implementation for convolution in host-side code.
*/

#pragma once

#include "hytlass/coord.h"
#include "hytlass/functional.h"
#include "hytlass/layout/tensor.h"
#include "hytlass/numeric_conversion.h"
#include "hytlass/numeric_types.h"
#include "hytlass/tensor_ref.h"
#include "hytlass/tensor_view.h"
#include "hytlass/conv/convolution.h"
#include "hytlass/conv/conv2d_problem_size.h"
#include <iostream>

namespace hytlass {
namespace reference {
namespace host {

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Forward propagation
////////////////////////////////////////////////////////////////////////////////////////////////////

/// y = conv2d(x, w)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ElementD = ElementC,
  typename ConvertOp = NumericConverter<ElementD, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>
>
void Conv2dFprop(
  conv::Conv2dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_x,
  TensorRef<ElementB, LayoutB> tensor_w,
  TensorRef<ElementC, LayoutC> tensor_y_in,
  TensorRef<ElementD, LayoutC> tensor_y_out,
  ElementCompute alpha,
  ElementCompute beta) {

  ConvertOp convert_op;
  InnerProductOp inner_product_op;

  // Apply MMA and accumulate ElementAccumulator
  for (int n = 0; n < problem_size.N; ++n) {
    for (int p = 0; p < problem_size.P; ++p) {
      for (int q = 0; q < problem_size.Q; ++q) {
        for (int k = 0; k < problem_size.K; ++k) {

          int group_idx = k / (problem_size.K / problem_size.groups);
          int channels_per_group = problem_size.C / problem_size.groups;

          ElementAccumulator acc = ElementAccumulator();

          for (int r = 0; r < problem_size.R; ++r) {
            for (int s = 0; s < problem_size.S; ++s) {
              for (int c = 0; c < channels_per_group; ++c) {

                int filter_r = r;
                int filter_s = s;

                if (problem_size.mode == hytlass::conv::Mode::kConvolution) {
                  filter_r = problem_size.R - 1 - r;
                  filter_s = problem_size.S - 1 - s;
                }

                int h = p * problem_size.stride_h - problem_size.pad_h + filter_r * problem_size.dilation_h;
                int w = q * problem_size.stride_w - problem_size.pad_w + filter_s * problem_size.dilation_w;

                if (h >= 0 && h < problem_size.H && w >= 0 && w < problem_size.W) {

                  ElementA a = tensor_x.at({n, h, w, c + group_idx * channels_per_group});
                  ElementB b = tensor_w.at({k, r, s, c});

                  acc = inner_product_op(ElementAccumulator(a), ElementAccumulator(b), acc);

                }
              }
            }
          }

          // Apply Epilogue, compute ElementCompute, convert and store ElementC
          ElementC c_ref = ElementC();

          if (beta != ElementCompute()) {
            c_ref = tensor_y_in.at(hytlass::make_Coord(n, p, q, k));
          }

          tensor_y_out.at(hytlass::make_Coord(n, p, q, k)) =
              convert_op(alpha * ElementCompute(acc) + beta * ElementCompute(c_ref));
        }
      }
    }
  }
}

/// Depthwise-separable convolution
template <typename ElementA,
          typename LayoutA,
          typename ElementB,
          typename LayoutB,
          typename ElementC,
          typename LayoutC,
          typename ElementCompute,
          typename ElementAccumulator = ElementCompute,
          typename ElementD = ElementC,
          typename ConvertOp = NumericConverter<ElementD, ElementCompute>,
          typename InnerProductOp = multiply_add<ElementAccumulator>>
void Depsep_Fprop(hytlass::TensorView<ElementA, LayoutA> tensor_A,
                  hytlass::TensorView<ElementB, LayoutB> tensor_B,
                  hytlass::TensorView<ElementC, LayoutC> tensor_C,
                  hytlass::TensorView<ElementD, LayoutC> tensor_D,
                  ElementCompute alpha,
                  ElementCompute beta,
                  hytlass::Tensor4DCoord padding = hytlass::Tensor4DCoord(),
                  hytlass::Coord<2> conv_stride = hytlass::Coord<2>(),
                  hytlass::Coord<2> dilation = hytlass::Coord<2>(),
                  hytlass::conv::Mode mode = hytlass::conv::Mode::kCrossCorrelation) {

  ConvertOp convert_op;
  InnerProductOp inner_product_op;

  // Apply MMA and accumulate ElementAccumulator
  for (int n = 0; n < tensor_C.extent().n(); ++n) {
    for (int p = 0; p < tensor_C.extent().h(); ++p) {
      for (int q = 0; q < tensor_C.extent().w(); ++q) {
        for (int g = 0; g < tensor_C.extent().c(); ++g) {
          ElementAccumulator acc = ElementAccumulator();
          for (int r = 0; r < tensor_B.extent().h(); ++r) {
            for (int s = 0; s < tensor_B.extent().w(); ++s) {
              
              // input activation H and W
              int h = p * conv_stride[0] - padding[0] + r * dilation[0];
              int w = q * conv_stride[1] - padding[2] + s * dilation[1];

              if (h < tensor_A.extent().h() && h >= 0 && w < tensor_A.extent().w() && w >= 0) {
                ElementA a = tensor_A.at(hytlass::make_Coord(n, h, w, g));

                ElementB b = (mode == hytlass::conv::Mode::kCrossCorrelation)
                                   ? tensor_B.at(hytlass::make_Coord(g, r, s, 0))
                                   : tensor_B.at(hytlass::make_Coord(
                                         g, tensor_B.extent().h() - r - 1, tensor_B.extent().w() - s - 1, 0));

                acc = inner_product_op(ElementAccumulator(a), ElementAccumulator(b), acc);
              }
            }
          }

          // Apply Epilogue, compute ElementCompute, convert and store ElementC
          ElementC c_ref = tensor_C.at(hytlass::make_Coord(n, p, q, g));
          tensor_D.at(hytlass::make_Coord(n, p, q, g)) =
              convert_op(alpha * ElementCompute(acc) + beta * ElementCompute(c_ref));
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Dgrad / Deconv
////////////////////////////////////////////////////////////////////////////////////////////////////

/// dx = dgrad(dy, w)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ElementD = ElementC,
  typename ConvertOp = NumericConverter<ElementD, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>
>
void Conv2dDgrad(
  hytlass::conv::Conv2dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_dy,
  TensorRef<ElementB, LayoutB> tensor_w,
  TensorRef<ElementC, LayoutC> tensor_dx_in,
  TensorRef<ElementD, LayoutC> tensor_dx_out,
  ElementCompute alpha,
  ElementCompute beta,
  bool is_deconv = false) {

  ConvertOp convert_op;
  InnerProductOp inner_product_op;

  // Apply MMA and accumulate ElementAccumulator
  for (int n = 0; n < problem_size.N; ++n) {
    for (int h = 0; h < problem_size.H; ++h) {
      for (int w = 0; w < problem_size.W; ++w) {
        for (int c = 0; c < problem_size.C; ++c) {

          ElementAccumulator acc = ElementAccumulator();

          for (int r = 0; r < problem_size.R; ++r) {
            for (int s = 0; s < problem_size.S; ++s) {
              for (int k = 0; k < problem_size.K; ++k) {

                int filter_r = r;
                int filter_s = s;

                if (problem_size.mode == hytlass::conv::Mode::kConvolution) {
                  filter_r = problem_size.R - 1 - r;
                  filter_s = problem_size.S - 1 - s;
                }

                int p = h + problem_size.pad_h - filter_r * problem_size.dilation_h;
                int q = w + problem_size.pad_w - filter_s * problem_size.dilation_w;

                if (p >= 0 && (p % problem_size.stride_h) == 0 && 
                    q >= 0 && (q % problem_size.stride_w) == 0) {

                  p = p / problem_size.stride_h;
                  q = q / problem_size.stride_w;
#if 0
                  std::cout << "row:" 
                  << n * problem_size.H * problem_size.W +
                    h * problem_size.W +
                    w << " "
                  << "n, p, q: (" 
                  << n << ", "
                  << p << ", "
                  << q << ") * "
                  << "r, s: (" 
                  << r << ", "
                  << s << ") [" 
                  << ((p < problem_size.P && q < problem_size.Q) ? "true":"false") << "]"        
                  << std::endl;
#endif
                  if (p < problem_size.P && q < problem_size.Q) {

                    ElementA a = tensor_dy.at(hytlass::make_Coord(n, p, q, k));
                    ElementB b = is_deconv ? tensor_w.at(hytlass::make_Coord(c, r, s, k))
                        : tensor_w.at(hytlass::make_Coord(k, r, s, c));

                    acc = inner_product_op(ElementAccumulator(a), ElementAccumulator(b), acc);
                  }
                }

              } // for (K)
            } // for (S)
          } // for (R)

          // Apply Epilogue, compute ElementCompute, convert and store ElementC
          ElementC c_ref = ElementC();

          if (beta != ElementCompute()) {
            c_ref = tensor_dx_in.at(hytlass::make_Coord(n, h, w, c));
          }

          tensor_dx_out.at(hytlass::make_Coord(n, h, w, c)) =
              convert_op(alpha * ElementCompute(acc) + beta * ElementCompute(c_ref));

        } // for (C)
      } // for (W)
    } // for (H)
  } // for (N)
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Wgrad
////////////////////////////////////////////////////////////////////////////////////////////////////

/// dw = wgrad(dy, x)
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ElementD = ElementC,
  typename ConvertOp = NumericConverter<ElementD, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>
>
void Conv2dWgrad(
  hytlass::conv::Conv2dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_dy,
  TensorRef<ElementB, LayoutB> tensor_x,
  TensorRef<ElementC, LayoutC> tensor_dw_in,
  TensorRef<ElementD, LayoutC> tensor_dw_out,
  ElementCompute alpha,
  ElementCompute beta) {
  
  InnerProductOp inner_product_op;
  ConvertOp convert_op;

  // Apply MMA and accumulate ElementAccumulator
  for (int k = 0; k < problem_size.K; ++k) {
    for (int r = 0; r < problem_size.R; ++r) {
      for (int s = 0; s < problem_size.S; ++s) {
        for (int c = 0; c < problem_size.C; ++c) {

          ElementAccumulator acc = ElementAccumulator();

          for (int n = 0; n < problem_size.N; ++n) {
            for (int p = 0; p < problem_size.P; ++p) {
              for (int q = 0; q < problem_size.Q; ++q) {
                  
                hytlass::Tensor4DCoord b_coord;
                
                int filter_r = r;
                int filter_s = s; 

                if (problem_size.mode == hytlass::conv::Mode::kConvolution) {
                  filter_r = problem_size.R - 1 - r;
                  filter_s = problem_size.S - 1 - s;
                }

                b_coord = make_Coord(
                    n,
                    p * problem_size.stride_h - problem_size.pad_h + filter_r * problem_size.dilation_h,
                    q * problem_size.stride_w - problem_size.pad_w + filter_s * problem_size.dilation_w,
                    c);

                if (b_coord.h() < problem_size.H && b_coord.h() >= 0 &&
                    b_coord.w() < problem_size.W && b_coord.w() >= 0) {

                  ElementAccumulator a = ElementAccumulator(tensor_dy.at(hytlass::make_Coord(n, p, q, k)));
                  ElementAccumulator b = ElementAccumulator(tensor_x.at(b_coord));
                  acc = inner_product_op(a, b, acc);
                }
              }
            }
          }

          // Apply Epilogue, compute ElementCompute, convert and store ElementC
          ElementC c_ref = ElementC();

          if (beta != ElementCompute()) {
            c_ref = tensor_dw_in.at(hytlass::make_Coord(k, r, s, c));
          }

          tensor_dw_out.at(hytlass::make_Coord(k, r, s, c)) =
              convert_op(alpha * ElementCompute(acc) + beta * ElementCompute(c_ref));

        } // for (C)
      } // for (S)
    } // for (R)
  } // for (K)
}

/// Generic 2D convolution targeting Conv2dFprop, Conv2dDgrad, and Conv2dWgrad.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementCompute,
  typename ElementAccumulator = ElementCompute,
  typename ElementD = ElementC,
  typename ConvertOp = NumericConverter<ElementD, ElementCompute>,
  typename InnerProductOp = multiply_add<ElementAccumulator>
>
void Conv2d(
  conv::Operator convolutional_operator,
  conv::Conv2dProblemSize problem_size,
  TensorRef<ElementA, LayoutA> tensor_A,
  TensorRef<ElementB, LayoutB> tensor_B,
  TensorRef<ElementC, LayoutC> tensor_C,
  TensorRef<ElementD, LayoutC> tensor_D,
  ElementCompute alpha,
  ElementCompute beta) {

  switch (convolutional_operator) {
  case conv::Operator::kFprop:
    Conv2dFprop<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementCompute,
      ElementAccumulator,
      ElementD,
      ConvertOp, InnerProductOp
    >(problem_size, tensor_A, tensor_B, tensor_C, tensor_D, alpha, beta);
    break;

  case conv::Operator::kDeconv:
  case conv::Operator::kDgrad:
    Conv2dDgrad<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementCompute,
      ElementAccumulator,
      ElementD,
      ConvertOp, InnerProductOp
    >(problem_size, tensor_A, tensor_B, tensor_C, tensor_D, alpha, beta, (convolutional_operator == conv::Operator::kDeconv));
    break;

  case conv::Operator::kWgrad:
    Conv2dWgrad<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementCompute,
      ElementAccumulator,
      ElementD,
      ConvertOp, InnerProductOp
    >(problem_size, tensor_A, tensor_B, tensor_C, tensor_D, alpha, beta);
    break;

  default:
    break;  
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace host
}  // namespace reference
}  // namespace hytlass

/////////////////////////////////////////////////////////////////////////////////////////////////

