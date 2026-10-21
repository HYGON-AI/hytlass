# HYGON HYTLASS Changelog

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

- 十余个计算示例支持：

    - 支持基于2.x的多种数据类型(TF32/FP16/BF16/I8/U8)的矩阵乘法、卷积及其融合算子实现。

    - 支持基于Split-K、Stream-K等计算优化算法的矩阵乘法示例。

    - 支持基于访问者模式的自定义尾声处理示例。

    - 支持基于HuTe的矩阵乘法示例，包括BatchedGemm、GroupGemm等示例。

    - 支持使用TensorCore加速的基于Block Ell格式的稀疏矩阵乘法示例。

- 工具链支持：

    - 支持hytlass_profiler，用于细粒度问题参数下的kernel tuning。

