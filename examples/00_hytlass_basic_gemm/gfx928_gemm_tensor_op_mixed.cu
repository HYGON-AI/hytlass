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

/**
This example shows how to run matrix multiplication kernels using functions and data structures
provided by HYTLASS using tensor cores.

Writing a single high performance matrix multiplication kernel is hard but do-able. Whereas writing
high performance kernels at scale which works for multiple problem sizes with good abstractions is
really hard. HYTLASS solves this problem by providing simplified abstractions to compose
multiple sections of gemm kernel. When used properly, the kernels can hit peak performance of GPU
easily.

HYTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp
and thread-block level, they compute on their own tile-size with higher level of tile sizes being
composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used
to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute
threadblock-tile (tile size computed by a threadblock).

In thie example, we split variable initialization into
1. Setting up data properties : describes how matrices are laid out in the memory and how the kernel
can view them (logical to physical mapping)
2. Setting up computation properties : describes how the above set matrices will be used to compute
output of matrix multiplication.

First, we setup the data types of matrices A, B, C and D along with alpha, beta as the equation for
GEMM is D = alpha * A * B + beta * C. In HYTLASS, the kernels first compute A * B and leaves the
rest of the computation to end of the kernel as alpha * X + beta * C is a simple element-wise
operation on X (A * B) and C. We call this as epilogue of kernel. Hence, we setup data types for
alpha and beta to be equal to ElementComputeEpilogue = int32_t. As we want to use MMA instructions
on Turing and they support 8-bit signed integer (int8_t), we use data type for elements in input
matrix A and B as int8_t. Volta also supports accumulation of partial dot product to int32_t, which
can store wider range of numbers, we use it as data type of output matrix elements and accumulation.
We convey this to HYTLASS kernel by initializing template variables ElementAccumulator (int32_t),
ElementComputeEpilogue (int32_t), ElementInputA (int8_t), ElementInputB (int8_t), ElementOutput
(int32_t). Communicating just the data type is not enough. As the data is laid out linearly in
memory, we have to convey the layout of matrices. We do that by initializing template variable
LayoutInputA to column major hytlass variable, LayoutInputB to row major and LayoutOutput to row
major. Next, we setup rules to comptue alpha * X + beta * C which is called epilogue of the kernel.
We initialize template variable EpilogueOp, which takes the data type of output ElementOutput
(int32_t), the number of elements per vector memory access (16), data type of accumulator (int32_t)
and data type of computation of linear combination (alpha * X + beta * C).

Now that we setup the properties of data, we have to setup properties of computation.

Second, we create template variables of tile sizes for thread-block, warp and mma-op to 128x256x64,
64x64x16, 8x8x16 (MxNxK) respectively. When passed to instantiate HYTLASS GEMM kernel, it internally
deduce the amount of threads needed per thread-block, amount of shared memory, storing data in
bank-conflict free manner, and ton of other variables required to compose, initialize and launch a
high performance GEMM kernel. This is the beauty of HYTLASS, it relieves developer from
understanding and coding complicated hardware optimizations which can easily go wrong.

HYTLASS also supports multiple MMA pipelines in a threadblock. What are MMA pipelines? MMA pipelines
constitute the whole process of loading input data from global memory to shared memory, loading data
from shared memory to registers, doing matrix multiplication, store to global memory. The below flow
sequence shows a typical mma pipeline.

matrix in global memory -> registers -> tile in shared memory -> registers -> mma -> registers ->
output to global memory

The problem with single pipeline is, each stage is synchronous which means, each stage has to wait
until the previous finished executing. There are stages in the pipeline which do not have fixed
latency, for example, the loads from global memory and shared memory. Therefore, we can add one more
pipeline with a phase shift in mma kernel to hide latency from global and shared memory loads.
Finally, the pipeline in a kernel looks like

(1) matrix in global memory -> (2) registers -> (3) tile in shared memory -> (4) registers -> (5)
mma -> (6) registers -> (7) output to global memory (1) <null> -> (2) <null> -> (3) matrix in global
memory -> (4) registers -> (5) tile in shared memory -> (6) registers -> (7) mma -> (8) registers ->
(9) output to global memory

This way, you can hide the second global memoroy load latency by doing computation on already loaded
input data.

There are few more template variables initialized such as, which threadblock tile of output matrix
is done which threadblock launched on an SM, GFX architecture of GPU you want to run on.

These are all put together to create a template variable which describes HYTLASS GEMM kernel using
hytlass::gemm::device::Gemm template.

The next step is to initialize physical data, instantiate and initialize HYTLASS kernel and run it.
We use HYTLASS utilities to initialize, fill, compare matrices as they are simple and doesn't come
in the way of learning HYTLASS.

Once all the matrices are initialized and filled with data, create arguments tuple to launch HYTLASS
kernel which takes problem size (M = 5120, N = 4096 and K = 4096), matrices, alpha, beta and the
important one, split k-dimension factor. Along with that, we query HYTLASS if any scratch-space
memory required by the kernel we instantiated. If yes, we create it and pass it along with other
arguments created to initialize HYTLASS kernel then, the kernel is launched.

In this example, we later on launch a reference gemm kernel (from HYTLASS utilities) to compare if
the output from HYTLASS kernel is same as reference GEMM kernel.
*/

#include <iostream>

#include "hytlass/hytlass.h"
#include "hytlass/gemm/device/gemm.h"
#include "hytlass/util/host_tensor.h"
#include "hytlass/util/reference/device/gemm.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/tensor_view_io.h"
#include "helper.h"
#include "hytlass/util/GPU_Clock.hpp"

#include "hipblas.h"

#include "hytlass/blas3.h"

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                 // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = hytlass::tfloat32_t;                       // <- data type of elements in input matrix A
using ElementInputB = hytlass::tfloat32_t;                       // <- data type of elements in input matrix B
using ElementOutput = hytlass::half_t;                      // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = hytlass::layout::ColumnMajor;
using LayoutInputB = hytlass::layout::RowMajor;
using LayoutOutput = hytlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = hytlass::arch::OpClassTensorOp;

// This code section describes GFX architecture number
using SmArch = hytlass::arch::Gfx928;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock = hytlass::gemm::GemmShape<128, 128, 16>;
// This code section describes tile size a warp will compute
using ShapeMMAWarp = hytlass::gemm::GemmShape<64, 128, 16>;
// This code section describes the size of MMA op
using ShapeMMAOp = hytlass::gemm::GemmShape<16, 16, 8>;

constexpr int kAlignmentA = 128 / hytlass::sizeof_bits<ElementInputA>::value;
constexpr int kAlignmentB = 128 / hytlass::sizeof_bits<ElementInputA>::value;
constexpr int kAlignmentC = 128 / hytlass::sizeof_bits<ElementOutput>::value;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = hytlass::epilogue::thread::LinearCombination<
  ElementOutput,                                     // <- data type of output matrix
  kAlignmentC,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
  ElementAccumulator,                                // <- data type of accumulator
  ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 2;


using Gemm = hytlass::gemm::device::Gemm<
  ElementInputA,
  LayoutInputA,
  ElementInputB,
  LayoutInputB,
  ElementOutput,
  LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ShapeMMAThreadBlock,
  ShapeMMAWarp,
  ShapeMMAOp,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages, kAlignmentA, kAlignmentB>;

                              
void reference_hipblas(int m, int n, int k, 
                      const ElementInputA* a, const ElementInputA* b, float* c) {
  hipblasHandle_t handle;
  hipblasCreate(&handle);
  ElementComputeEpilogue one = ElementComputeEpilogue(1);
  ElementComputeEpilogue zero = ElementComputeEpilogue(0);

  float *_a;
  float *_b;  
  (void)hipMalloc((void **)(&_a), sizeof(float) * m * k);
  for (int i = 0; i < m * k; i++) {
      _a[i] = static_cast<float>(a[i]);
  }  
  (void)hipMalloc((void **)(&_b), sizeof(float) * k * n);
  for (int i = 0; i < n * k; i++) {
      _b[i] = static_cast<float>(b[i]);
  }

  hipblasOperation_t blas_trans_a = std::is_same<LayoutInputA, 
                                        hytlass::layout::ColumnMajor>::value ? 
                                        HIPBLAS_OP_N : 
                                        HIPBLAS_OP_T;

  hipblasOperation_t blas_trans_b = std::is_same<LayoutInputB, 
                                        hytlass::layout::ColumnMajor>::value ? 
                                        HIPBLAS_OP_N : 
                                        HIPBLAS_OP_T;
  int lda = std::is_same<LayoutInputA, 
                        hytlass::layout::ColumnMajor>::value ? m: k;
  int ldb = std::is_same<LayoutInputB, 
                        hytlass::layout::ColumnMajor>::value ? k: n;

  auto err_ = hipblasGemmEx(handle, blas_trans_a, blas_trans_b,
                           m, n, k,
                           &one,
                           _a, HIPBLAS_R_32F, lda,
                           _b, HIPBLAS_R_32F, ldb,
                           &zero,
                           c, HIPBLAS_R_32F, m,
                           HIPBLAS_R_32F,
                           HIPBLAS_GEMM_DEFAULT);  
  (void)hipDeviceSynchronize();
  if (err_ != 0) {
      printf("error code is %d\n", err_);
      exit(-1);
  }                       
}


int run(int length_m_, int length_n_, int length_k_) {

  const int length_m = length_m_;
  const int length_n = length_n_;
  const int length_k = length_k_;

  // Create a tuple of problem size for matrix multiplication
  hytlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // Initialize tensors using HYTLASS helper functions
  hytlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
    problem_size.mk());    // <- Create matrix A with dimensions M x K
  hytlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
    problem_size.kn());    // <- Create matrix B with dimensions K x N
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
    problem_size.mn());    // <- Create matrix C with dimensions M x N
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
    problem_size.mn());    // <- Create matrix D with dimensions M x N used to store output from
                           // HYTLASS kernel
  hytlass::HostTensor<float, hytlass::layout::ColumnMajor> tensor_ref_d(
    problem_size.mn());    // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Fill input and output matrices on host using HYTLASS helper functions
  hytlass::reference::host::TensorFillRandomUniform(
    tensor_a.host_view(),
    1,
    ElementInputA(8),
    ElementInputA(-8),
    hytlass::MantissaInBits<ElementOutput>::bits);  // <- Fill matrix A on host with uniform-distribution random data
  hytlass::reference::host::TensorFillRandomUniform(
    tensor_b.host_view(),
    2,
    ElementInputB(8),
    ElementInputB(-8),
    hytlass::MantissaInBits<ElementOutput>::bits);  // <- Fill matrix B on host with uniform-distribution random data


  hytlass::reference::host::TensorFill(
    tensor_c.host_view());
  hytlass::reference::host::TensorFill(
    tensor_d.host_view());  // <- fill matrix D on host with zeros
  hytlass::reference::host::TensorFill(
    tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated HYTLASS kernel
  typename Gemm::Arguments arguments {
    problem_size,  // <- problem size of matrix multiplication
    tensor_a.device_ref(),  // <- reference to matrix A on device
    tensor_b.device_ref(),  // <- reference to matrix B on device
    tensor_c.device_ref(),  // <- reference to matrix C on device
    tensor_d.device_ref(),  // <- reference to matrix D on device
    {alpha, beta},          // <- tuple of alpha and beta
    split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  hytlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate HYTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not 
  hytlass::Status status = gemm_op.can_implement(arguments);
  HYTLASS_CHECK(status);

  // Initialize HYTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  HYTLASS_CHECK(status);


  // Launch initialized HYTLASS kernel
  status = gemm_op();
  HYTLASS_CHECK(status);

  tensor_d.sync_host();

  // printf("result is :\n");
  // for(int i=0;i<length_m;i++){
  //   for(int j=0;j<length_n;j++){
  //       printf("%.2f,", float(tensor_d.host_data()[i*length_n+j]));
  //   }
  //   printf("\n");
  // }

  // Create instantiation for device reference gemm kernel

//   hytlass::reference::device::Gemm<ElementInputA,
//                                    LayoutInputA,
//                                    ElementInputB,
//                                    LayoutInputB,
//                                    ElementOutput,
//                                    LayoutOutput,
//                                    ElementOutput,
//                                    ElementOutput>
//       gemm_device;

// //   Launch device reference gemm kernel
//   gemm_device(problem_size,
//               alpha,
//               tensor_a.device_ref(),
//               tensor_b.device_ref(),
//               beta,
//               tensor_c.device_ref(),
//               tensor_ref_d.device_ref());



  reference_hipblas(length_m, length_n, length_k, 
                   tensor_a.device_data(),
                   tensor_b.device_data(),
                   tensor_ref_d.device_data());

//   // Wait for kernels to finish
  (void)hipDeviceSynchronize();

  // Copy output data from HYTLASS and reference kernel to host for comparison
  tensor_ref_d.sync_host();

  // Check if output from HYTLASS kernel and reference kernel are equal or not
  ElementOutput eps(1e-3f);
  if (std::is_same<ElementInputA, hytlass::bfloat16_t>::value) {
      eps = 0.05f;
  }


  float max_error_v = 1e-9;
  int cnt = 0;
  for (int i = 0; i < length_m; i++) {
    for (int j = 0; j < length_n; j++) {
      float factor = float(1);
        if (tensor_ref_d.host_data()[i + length_m * j] != 0) {
          factor = std::abs(tensor_ref_d.host_data()[i + length_m * j]);
        }
        if (std::abs(tensor_ref_d.host_data()[i + length_m * j]) <= 1) {
          factor = 1;
        }
        if (std::abs((tensor_ref_d.host_data()[i + length_m * j]) - float(tensor_d.host_data()[i * length_n + j])) / factor > eps) {
          printf("error at (%d %d) expected %f got %f abs err is %f and Relative error is %f\n", i, j,
                float(tensor_ref_d.host_data()[i + length_m * j]), float(tensor_d.host_data()[i * length_n + j]),
                float(std::abs((tensor_ref_d.host_data()[i + length_m * j]) - float(tensor_d.host_data()[i * length_n + j]))),
                float(std::abs((tensor_ref_d.host_data()[i + length_m * j]) - float(tensor_d.host_data()[i * length_n + j])) / factor));

        if (max_error_v < std::abs((tensor_ref_d.host_data()[i + length_m * j]) - (tensor_d.host_data()[i * length_n + j])) / factor){
          max_error_v = (std::abs((tensor_ref_d.host_data()[i + length_m * j]) - (tensor_d.host_data()[i * length_n + j])) / factor);
        }
        cnt++;
      }
    }
  }
  if (cnt > 0) {
    printf("faild,");
    printf("%f\n", max_error_v);
    return -1;
  }
    printf("success\n");

  return 0;
}

int main() {
  run(1024, 1024, 1024);

}
