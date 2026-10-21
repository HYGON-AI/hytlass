# HYTLASS 0.1.0

_HYTLASS 0.1.0 - 2025年12月_

HYTLASS(HYGON HCU Templates for Linear Algebra Subroutines)是一个用于在海光HCU架构下实现高性能矩阵乘法(GEMM)及其衍生计算的C++模板库，其设计思路来源于CUTLASS，将数据搬运、层次化结构等可动部件分解为可复用、模块化的软件组件，
并通过C++模板类进行抽象。

HYTLASS同时兼容了基于CUTLASS 2.x实现的基于C++模版的矩阵乘和卷积，以及基于CUTLASS 3.x引入的CuTe编程模型。

你可以参考[快速入门指南](media/docs/quickstart.md)来快速入门使用HYTLASS。

## Upstream source and license

Upstream: [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass.git), `v3.5.0`,
commit `7d49e6c7e2f8896c47f586706e67e1fb215529dc`.
License: BSD-3-Clause (see [LICENSE.txt](LICENSE.txt)).
Modified by Hygon Information Technology Co., Ltd.

# HYTLASS 0.1.0新增

HYTLASS 0.1.0是HYTLASS的首次提交，其实现基于CUTLASS 3.5.0，具体地，提供了：

- HYTLASS兼容和支持CUTLASS 2.x实现：

  - 支持BW(GFX936)及更早架构下的指令特性，包括支持TensorCore的mmac及ds_read_matrix原语。

  - 支持基于这些架构指令特性下的矩阵乘法实现。

  - 支持基于矩阵乘法模版实现的基于隐式矩阵乘法的卷积实现。

- HYTLASS兼容和支持CUTLASS 3.x及CuTe编程模型：

  - 适配基于HCU架构对CUTLASS 3.x中的CuTe编程模型HuTe，目前已支持至BW平台的指令原语。

  - 支持基于HuTe模型的矩阵乘法实现，包括MMA部分和Epilogue部分。

  - 支持基于HuTe模型的Kernel调度、Tiling调度等计算任务调度架构，支持多种线程块调度优化策略。

- 十余个计算示例支持(详见[示例](examples))：

  - 支持基于2.x的多种数据类型(TF32/FP16/BF16/I8/U8)的矩阵乘法、卷积及其融合算子实现。

  - 支持基于Split-K、Stream-K等计算优化算法的矩阵乘法示例。

  - 支持基于访问者模式的自定义尾声处理示例。

  - 支持基于HuTe的矩阵乘法示例，包括BatchedGemm、GroupGemm等示例。

  - 支持使用TensorCore加速的基于Block Ell格式的稀疏矩阵乘法示例。

- 工具链支持：

  - 支持hytlass_profiler，用于细粒度问题参数下的kernel tuning。

# 编译HYTLASS

在其他项目中，HYTLASS作为头文件库不需要单独编译，用户将`include/`目录指定至头文件路径即可。

HYTLASS的单元测试、示例及工具链通过CMake编译，其所需编译环境要求如下：

- DTK (DCC 25.10或更新版本)，详见[DCC编译器](https://download.sourcefind.cn:65024/1/main/compiler)

- CMake 3.19+

- 支持C++17或更高版本的主机端编译器

- Python 3.5+

部分示例和单测中，HYTLASS可能需要编译链接hipBLAS库。

在HYTLASS项目根目录中创建一个build目录，然后执行cmake，你可以通过CMAKE选项`HYTLASS_HIPCC_ARCHS`指定编译的架构。

```bash
$ mkdir build && cd build

$ cmake .. -DHYTLASS_HIPCC_ARCHS=936               # compiles for HCU BW Architecture
```

你可以通过在build目录下使用make编译并执行`test_unit`来执行HYTLASS的单元测试。可以通过`-j`选项来并行执行make的流程。

```bash
$ make test_unit -j
...
...
...
[----------] Global test environment tear-down
[==========] 946 tests from 57 test cases ran. (10812 ms total)
[  PASSED  ] 946 tests.
```

在所支持的硬件架构下，所有测试都应当通过，尽管不同硬件架构下的单元测试数量可能有所不同。

你也可以通过在build目录下，使用make编译`test_examples*`来执行所有示例或某个示例。

```bash
$ make test_examples -j
...
...
...
[100%] Built target test_examples_xxx
[100%] Built target test_examples
```

或在编译完成后，在`build/examples`目录下单独执行某个示例的可执行文件。

```bash
$ cd build/examples && ./00_hytlass_basic_gemm/gfx928_gemm_tensor_op
```