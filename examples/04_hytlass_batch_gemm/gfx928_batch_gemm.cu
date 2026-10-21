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

#include <iostream>
#include <vector>
#include "hytlass/util/command_line.h"
#include "hip/hip_runtime.h"

#include "hytlass/hytlass.h"
#include "hytlass/layout/matrix.h"
#include "hytlass/gemm/device/gemm_batched.h"

#pragma warning( disable : 4503)

/*
This example demonstrates how to use hytlass to compute a batched strided gemm in two different ways:
  1. By specifying pointers to the first matrices of the batch and the stride between the consecutive
     matrices of the batch (this is called a strided batched gemm).
  2. By copying pointers to all matrices of the batch to the device memory (this is called an array gemm).
In this example, both A and B matrix are non-transpose and column major matrix
batched_C = batched_A x batched_B
As an example, matrix C can be seen as
-----------------------------------------------------------
(0,0,0) | (0,0,1) | (0,0,2) | (1,0,0) | (1,0,1) | (1,0,2) |
-----------------------------------------------------------
(0,1,0) | (0,1,1) | (0,1,2) | (1,1,0) | (1,1,1) | (1,1,2) |
-----------------------------------------------------------
(0,2,0) | (0,2,1) | (0,2,2) | (1,2,0) | (1,2,1) | (1,2,2) |
-----------------------------------------------------------
(0,3,0) | (0,3,1) | (0,3,2) | (1,3,0) | (1,3,1) | (1,3,2) |
-----------------------------------------------------------
(0,4,0) | (0,4,1) | (0,4,2) | (1,4,0) | (1,4,1) | (1,4,2) |
-----------------------------------------------------------
(0,5,0) | (0,5,1) | (0,5,2) | (1,5,0) | (1,5,1) | (1,5,2) |
-----------------------------------------------------------
           batch 0          |           batch 1
where we denote each element with (batch_idx, row_idx, column_idx)
In this example, batch size is 2, M is 6 and N is 3
The stride (batch_stride_C) between the first element of two batches is ldc * n

matrix A can be seen as
---------------------------------------
(0,0,0) | (0,0,1) | (1,0,0) | (1,0,1) |
---------------------------------------
(0,1,0) | (0,1,1) | (1,1,0) | (1,1,1) |
---------------------------------------
(0,2,0) | (0,2,1) | (1,2,0) | (1,2,1) |
---------------------------------------
(0,3,0) | (0,3,1) | (1,3,0) | (1,3,1) |
---------------------------------------
(0,4,0) | (0,4,1) | (1,4,0) | (1,4,1) |
---------------------------------------
(0,5,0) | (0,5,1) | (1,5,0) | (1,5,1) |
---------------------------------------
     batch 0      |      batch 1
, where batch size is 2, M is 6 and K is 2
The stride (batch_stride_A) between the first element of two batches is lda * k

matrix B can be seen as
-----------------------------
(0,0,0) | (0,0,1) | (0,0,2) |
----------------------------- batch 0
(0,1,0) | (0,1,1) | (0,1,2) |
-------------------------------------
(1,0,0) | (1,0,1) | (1,0,2) |
----------------------------- batch 1
(1,1,0) | (1,1,1) | (1,1,2) |
-----------------------------
, where the batch size is 2, N is 3 and K is 2
The stride (batch_stride_B) between the first element of two batches is k


*/
///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  hytlass::gemm::GemmCoord problem_size;
  int batch_count;
  float alpha;
  float beta;

  bool reference_check;
  int iterations;

  Options():
    help(false),
    problem_size({1024, 1024, 1024}),
    batch_count(2),
    reference_check(false),
    alpha(1),
    beta() 
  {}

  bool valid() {
    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    hytlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("batch_count", batch_count);

  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "04_hytlass_batch_gemm example\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --batch_count=<int>         Batch number\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/04_hytlass_batch_gemm/gfx928_batch_gemm --m=1024 --n=512 --k=1024 \\\n"
      << "     --alpha=2 --beta=0.707 --batch_count=2 \n\n";

    return out;
  }
  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds
    int64_t fmas = problem_size.product() * batch_count;

    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

hipError_t hytlass_strided_batched_sgemm(
  int m, 
  int n,
  int k,
  float alpha,
  float const *A,
  int lda,
  long long int batch_stride_A,
  float const *B,
  int ldb,
  long long int batch_stride_B,
  float *C,
  int ldc,
  long long int batch_stride_C,
  float beta,
  int batch_count) {

using Gemm = hytlass::gemm::device::GemmBatched<
  float, hytlass::layout::ColumnMajor, float, hytlass::layout::ColumnMajor,
  float, hytlass::layout::ColumnMajor, float,
  hytlass::arch::OpClassTensorOp, hytlass::arch::Gfx928,
  hytlass::gemm::GemmShape<128, 128, 32>,
  hytlass::gemm::GemmShape<64, 64, 32>, hytlass::gemm::GemmShape<16, 16, 8>,
  hytlass::epilogue::thread::LinearCombination<float, 1, float, float>,
  hytlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 1, 4,
  4, hytlass::arch::OpMultiplyAdd
  >;

  Gemm gemm_op;

  hytlass::Status status = gemm_op({
    {m, n, k},
    {A, lda}, 
    batch_stride_A,
    {B, ldb}, 
    batch_stride_B,
    {C, ldc}, 
    batch_stride_C,
    {C, ldc}, 
    batch_stride_C,
    {alpha, beta},
    batch_count
  });

  if (status != hytlass::Status::kSuccess) {
    return hipErrorUnknown;
  }

  return hipSuccess;
}

template<typename T> 
hipError_t strided_batched_gemm_nn_reference(
  int m,
  int n,
  int k,
  T alpha,
  std::vector<T> const &A, 
  int lda,
  long long int batch_stride_A,
  std::vector<T> const &B, 
  int ldb,
  long long int batch_stride_B,
  std::vector<T> &C, 
  int ldc,
  long long int batch_stride_C,
  T beta,
  int batch_count) {
  /*
  strided batched gemm NN
  */
  
  hipError_t result = hipSuccess;

  if (A.size() < size_t(lda * k * batch_count)) {
    std::cout << "the size of A is too small" << std::endl;
    return hipErrorInvalidValue;
  }
  if (B.size() < size_t(ldb * n)) {
    std::cout << "the size of B is too small" << std::endl;
    return hipErrorInvalidValue;
  }
  if (C.size() < size_t(ldc * n * batch_count)) {
    std::cout << "the size of C is too small" << std::endl;
    return hipErrorInvalidValue;
  }
  
  for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
    for (int n_idx = 0; n_idx < n; n_idx++) {
      for (int m_idx = 0; m_idx < m; m_idx++) {
        T accum = beta * C[batch_idx * batch_stride_C + n_idx * ldc + m_idx];
        for (int k_idx = 0; k_idx < k; k_idx++) {
          accum += alpha 
            * A[batch_idx * batch_stride_A + k_idx * lda + m_idx]
            * B[batch_idx * batch_stride_B + n_idx * ldb + k_idx];
        }
        C[batch_idx * batch_stride_C + n_idx * ldc + m_idx] = accum;
      }
    }
  }

  return result;
}


hipError_t run_batched_gemm(Options &options) {

  std::cout << "Running strided batched gemm" << std::endl;

  // Arbitrary problem size
  int m = options.problem_size.m();
  int n = options.problem_size.n();
  int k = options.problem_size.k();
  int batch_count = options.batch_count;
  // alpha and beta
  float alpha = options.alpha;
  float beta = options.beta;

  // A, B are non-transpose, column major
  int const lda = m;
  int const ldb = k * batch_count;
  int const ldc = m;

  int const count_A = batch_count * lda * k;
  int const count_B = ldb * n;
  int const count_C = batch_count * ldc * n;

  // the memory is batched along K dimension
  long long int batch_stride_A = static_cast<long long int>(lda) * static_cast<long long int>(k);
  long long int batch_stride_B = static_cast<long long int>(k);
  long long int batch_stride_C = static_cast<long long int>(ldc) * static_cast<long long int>(n);

  hipError_t result = hipSuccess;

  // allocate the host memory
  std::vector<float> host_A(count_A);
  std::vector<float> host_B(count_B);
  std::vector<float> host_C(count_C);
  std::vector<float> result_C(count_C);

  // allocate the device memory
  float *A;
  float *B;
  float *C;

  result = hipMalloc(&A, count_A * sizeof(float));
  if (result != hipSuccess) {
    std::cerr << "hipMalloc result = " << result << std::endl;
    return result;
  }
  result = hipMalloc(&B, count_B * sizeof(float));
  if (result != hipSuccess) {
    std::cerr << "hipMalloc result = " << result << std::endl;
    return result;
  }
  result = hipMalloc(&C, count_C * sizeof(float));
  if (result != hipSuccess) {
    std::cerr << "hipMalloc result = " << result << std::endl;
    return result;
  }

  // Limit range to avoid floating-point errors
  int const kRange = 8;

  // fill A
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < k; col_idx++) {
      for (int row_idx = 0; row_idx < m; row_idx++) {
        host_A[row_idx + col_idx * lda + b_idx * lda * k] = static_cast<float>((row_idx + col_idx * lda + b_idx * lda * k) % kRange);
      }
    }
  }
  // fill B
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < n; col_idx++) {
      for (int row_idx = 0; row_idx < k; row_idx++) {
        host_B[row_idx + col_idx * ldb + b_idx * k] = static_cast<float>(((n + k * ldb + batch_count * k) - (row_idx + col_idx * ldb + b_idx * k)) % kRange);
      }
    }
  }
  // fill C
  for (int b_idx = 0; b_idx < batch_count; b_idx++) {
    for (int col_idx = 0; col_idx < n; col_idx++) {
      for (int row_idx = 0; row_idx < m; row_idx++) {
        host_C[row_idx + col_idx * ldc + b_idx * ldc * n] = 1.f;
      }
    }
  }

  // ref memory
  std::vector<float> ref_A(host_A);
  std::vector<float> ref_B(host_B);
  std::vector<float> ref_C(host_C);
  // copy host memory to device
  result = hipMemcpy(A, host_A.data(), count_A * sizeof(float), hipMemcpyHostToDevice);
  if (result != hipSuccess) {
    std::cerr << "hipMemcpy result = " << result << std::endl;
    return result;
  }
  result = hipMemcpy(B, host_B.data(), count_B * sizeof(float), hipMemcpyHostToDevice);
  if (result != hipSuccess) {
    std::cerr << "hipMemcpy result = " << result << std::endl;
    return result;
  }
  result = hipMemcpy(C, host_C.data(), count_C * sizeof(float), hipMemcpyHostToDevice);
  if (result != hipSuccess) {
    std::cerr << "hipMemcpy result = " << result << std::endl;
    return result;
  }

  result = hytlass_strided_batched_sgemm(
    m, n, k, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B, C, ldc, batch_stride_C,
    beta, batch_count);
  if (result != hipSuccess)
    return result;
  

  // copy device memory to host
  result = hipMemcpy(result_C.data(), C, count_C * sizeof(float), hipMemcpyDeviceToHost);
  if (result != hipSuccess) {
    std::cerr << "hipMemcpy result = " << result << std::endl;
    return result;
  }

  //compare with reference code
  result = strided_batched_gemm_nn_reference(m, n, k, alpha, ref_A, lda, batch_stride_A, ref_B, ldb, batch_stride_B, ref_C, ldc, batch_stride_C,
    beta, batch_count);
  if (result != 0)
    return result;

  // Expect bit-level accuracy for this simple example
  if (ref_C != result_C) {
    std::cout << "HYTLASS strided batched gemm does not run correctly" << std::endl;
    return hipErrorUnknown;
  }

  // free memory
  result = hipFree(A);
  if (result != hipSuccess) {
    std::cerr << "hipFree result = " << result << std::endl;
    return result;
  }
  result = hipFree(B);
  if (result != hipSuccess) {
    std::cerr << "hipFree result = " << result << std::endl;
    return result;
  }
  result = hipFree(C);
  if (result != hipSuccess) {
    std::cerr << "hipFree result = " << result << std::endl;
    return result;
  }

  return result;
}

int main(int argc, const char **argv) {

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }
  printf("%d x %d x %d x %d tensor op Matrix Multiply\n", \
  options.problem_size.m(), options.problem_size.n(), options.problem_size.k(), options.batch_count);

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  hipError_t result = hipSuccess;

    result = run_batched_gemm(options);
    if (result == hipSuccess) {
      std::cout << "Passed." << std::endl;
    } 
  // Exit.
  return result == hipSuccess ? 0 : -1;
}
