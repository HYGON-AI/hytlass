# profiler helper

用于从profiler的结果中提取最优kernel。至少需要python3.6支持


## 准备参数文件
在当前路径下准备形式如**params_nt.csv**所示的csv参数文件。其中列名为profiler的命令行参数，根据需要可按照profiler的命令行参数自行添加，需要注意必须包含 operation 列，当前支持`gemm` 和`conv2d`两种operation。
## 执行脚本，提取kernel
```bash
# 查看参数
python3 profiler_helper.py --help

# -I or --inputs    指定输入的参数文件  
# -O or --output    输出目录名称  
# -R or --start_idx 从第几个参数开始  

# 参数文件为 params_nt.csv, 输出到 ./profiler_result/cutlass_nt 路径下 从第二个参数开始
python3 profiler_helper.py --input=params_nt.csv --output=cutlass_nt --start_idx=2
```
## 查看结果
结果将会保存在 ./profiler_result/save_path路径下
best_kernel.csv会记录各个参数选择的最优kernel以及性能。对于执行出现异常的情况，会在该文件中保留一行的占位符。  
具体执行情况和错误信息可以看同目录下的profiler_helper.log文件。各个参数的选优结果，错误信息都会在此记录。  
detail目录则保存各组参数的具体执行情况和标准错误输出，以供进一步查看选优情况。
```
|-profiler_helper
| |-output_path
| | |-best_kernel.csv       // 提取出的最优kernel
| | |-profiler_helper.log   // 日志文件
| | |-detail
| | | |-profiler_gens       // profiler 生成 csv 文件的保存路径
| | | | |-report_1.gemm.csv // 第一个参数对应的kernel 数据
| | | |-stderr              // 标准错误输出路径
```

