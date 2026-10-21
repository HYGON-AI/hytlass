[README](../../README.md#documentation) > **Quick Start**

# Quickstart

## Prerequisites

HYTLASS requires:
- DTK (DCC 25.10 or later required), see [DCC](https://download.sourcefind.cn:65024/1/main/compiler)
- CMake 3.19+
- host compiler supporting C++17 or greater
- Python 3.5+

HYTLASS may be optionally compiled and linked with
- hipBLAS

## Initial build steps

Construct a build directory and run CMake.
```bash
$ export DTK_ROOT_DIR=/opt/dtk
$ source "$DTK_ROOT_DIR/env.sh"

$ mkdir build && cd build

$ cmake .. -DHYTLASS_HIPCC_ARCHS=928
```

If your goal is strictly to build only the HYTLASS Profiler and to minimize compilation time, we suggest
executing the following CMake command in an empty `build/` directory.
```bash
$ cmake .. -DHYTLASS_HIPCC_ARCHS=928 -DHYTLASS_ENABLE_TESTS=OFF -DHYTLASS_UNITY_BUILD_ENABLED=ON
```

This reduces overall compilation time by excluding unit tests and enabling the unity build.

You may reduce build times by compiling only certain operations by setting the `HYTLASS_LIBRARY_OPERATIONS` flag as shown below,
executed from an empty `build/` directory. This only compiles 2-D convolution kernels.

```bash
$ cmake .. -DHYTLASS_HIPCC_ARCHS=928 -DHYTLASS_LIBRARY_OPERATIONS=conv2d
```

You may also filter kernels by name by supplying a filter string with flag `HYTLASS_LIBRARY_KERNELS`. For example the below command selects only HYTLASS-3 kernels.

```bash
$ cmake .. -DHYTLASS_HIPCC_ARCHS=928 -DHYTLASS_LIBRARY_KERNELS=hytlass3x*
```

## Build and run the HYTLASS Profiler

From the `build/` directory created above, compile the HYTLASS Profiler.
```bash
$ make hytlass_profiler -j12
```

Then execute the HYTLASS Profiler computing GEMM, execute the following command.
```bash
$ ./tools/profiler/hytlass_profiler --kernels=sgemm --m=4352 --n=4096 --k=4096

=============================
  Problem ID: 1

        Provider: HYTLASS
   OperationKind: gemm
       Operation: hytlass3x_simt_s8_igemm_s8_32x32x32_1x1x1_x2_nn_align_a16_b16_c16

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed
          hipBLAS: Not run

       Arguments: --gemm_kind=universal --m=4352 --n=4096 --k=4096 --lda=4352 --ldb=4096 --ldc=4352 --A=s8:column --B=s8:column  \
                  --C=s8:column --D=s8:column --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1 --batch_count=1  \
                  --raster_order=heuristic --stagger_k=1 --stagger_k_stride=0 --swizzle_size=1 --op_class=simt --accum=s32  \
                  --cta_m=32 --cta_n=32 --cta_k=32 --cluster_m=1 --cluster_n=1 --cluster_k=1 --stages=2 --warps_m=2 --warps_n=1  \
                  --warps_k=1 --inst_m=1 --inst_n=1 --inst_k=4 --min_cc=906 --max_cc=1024

           Bytes: 36896768  bytes
           FLOPs: 72491728896  flops
           FLOPs/Byte: 1964

         Runtime: xxx  ms
          Memory: xxx GiB/s

            Math: xxx GFLOP/s
```

To execute the HYTLASS Profiler for convolution, run the following example.
```bash
$ ./tools/profiler/hytlass_profiler --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --pad_h=1 --pad_w=1
```

To execute all HYTLASS 2-D convolution operators, execute the following.
```bash
$ ./tools/profiler/hytlass_profiler --operation=conv2d --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3


=============================
  Problem ID: 1

        Provider: HYTLASS
   OperationKind: conv2d
       Operation: hytlass_tensorop_s16168fprop_optimized_32x32x16_1x1x1_x1_nhwc_align_a4_b4_c4

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed

       Arguments: --conv_kind=fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --p=224 --q=224 --g=1 --pad_h=1  \
                  --pad_w=1 --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1 --Activation=f32:nhwc --Filter=f32:nhwc  \
                  --Output=f32:nhwc --conv_mode=cross --iterator_algorithm=optimized --alpha=1 --beta=0 --split_k_mode=serial  \
                  --split_k_slices=1 --eq_gemm_provider=none --op_class=tensorop --accum=f32 --cta_m=32 --cta_n=32 --cta_k=16  \
                  --cluster_m=1 --cluster_n=1 --cluster_k=1 --stages=1 --warps_m=1 --warps_n=1 --warps_k=1 --inst_m=16  \
                  --inst_n=16 --inst_k=8 --min_cc=928 --max_cc=1024

           Bytes: 2055798784  bytes
           FLOPs: 118482796544  flops
           FLOPs/Byte: 57

         Runtime: xxx  ms
          Memory: xxx GiB/s

            Math: xxx GFLOP/s

```
## Build and run HYTLASS Unit Tests

From the `build/` directory created above, simply build the target `test_unit` to compile and run
all unit tests.

```bash
$ make test_unit -j
...
...
...
[----------] Global test environment tear-down
[==========] 60 tests from 24 test suites ran. (22159 ms total)
[  PASSED  ] 60 tests.
```
The exact number of tests run is subject to change as we add more functionality.

No tests should fail. Unit tests automatically construct the appropriate runtime filters
to avoid executing on architectures that do not support all features under test.

The unit tests are arranged hierarchically mirroring the HYTLASS Template Library. This enables
parallelism in building and running tests as well as reducing compilation times when a specific
set of tests are desired.

For example, the following executes strictly the hytlass 2.x tensorop GEMM tests.
```bash
$ make test_unit_gemm_device_tensorop_gfx928 -j
...
...
[----------] 16 tests from GFX928_Device_Gemm_bf16n_bf16n_f32t_tensor_op_f32
[ RUN      ] GFX928_Device_Gemm_bf16n_bf16n_f32t_tensor_op_f32.128x256x64_64x64x64
[       OK ] GFX928_Device_Gemm_bf16n_bf16n_f32t_tensor_op_f32.128x256x64_64x64x64 (0 ms)
[ RUN      ] GFX928_Device_Gemm_bf16n_bf16n_f32t_tensor_op_f32.256x128x64_64x64x64
[       OK ] GFX928_Device_Gemm_bf16n_bf16n_f32t_tensor_op_f32.256x128x64_64x64x64 (0 ms)
...
[----------] 16 tests from GFX928_Device_Gemm_bf16t_bf16t_bf16t_tensor_op_f32 (788 ms total)
...
...
[----------] Global test environment tear-down
[==========] 371 tests from 26 test suites ran. (106656 ms total)
[  PASSED  ] 371 tests.
[100%] Built target test_unit_gemm_device_tensorop_gfx928
```

## Building for Multiple Architectures

To minimize compilation time, specific GPU architectures can be enabled via the CMake command.

**Hygon BW Architecture.**
```bash
$ cmake .. -DHYTLASS_HIPCC_ARCHS=936              # compiles for Hygon BW architecture
```

**Hygon KM-ECO Architecture.**
```bash
$ cmake .. -DHYTLASS_HIPCC_ARCHS=928              # compiles for Hygon KM-ECO architecture
```

## Launching a GEMM kernel

Refer to [examples](/examples) 00 ~ 07 for the usage of HYTLASS 2.x API and the collective builders.

# HYTLASS Library

The [HYTLASS Library](/tools/library) defines an API for managing and executing collections of compiled
kernel instances and launching them from host code without template instantiations in client code.

The host-side launch API is designed to be analogous to BLAS implementations for convenience, though its 
kernel selection procedure is intended only to be functionally sufficient. It may not launch the 
optimal tile size for a given problem. It chooses the first available kernel whose data types, 
layouts, and alignment constraints satisfy the given problem. Kernel instances and a data structure
describing them are completely available to client applications which may choose to implement their
own selection logic.
