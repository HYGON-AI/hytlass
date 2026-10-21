#################################################################################################
# Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################
import numpy as np
import pandas as pd
import argparse
import os
import logging
from pathlib import Path
import subprocess

def operation_check(value):
  value = str(value)
  if not ((value.lower() == "gemm" or value.lower() == "conv2d")):
      raise argparse.ArgumentTypeError(f"only gemm and conv2d operations are supported")
  return value

def parseArgs():
  parser = argparse.ArgumentParser(description ="Extract best kernel from profiler",
                                     epilog =
'''
EXAMPLE:
python3 profiler_helper.py --input=params_nt.csv --output=hytlass_nt --start_idx=2
python3 profiler_helper.py --input=params_res50_fprop.csv --output=res50_fprop
''',
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument("-I", "--input", type=str, help="Parameter file for profiler analysis", required=True)

  parser.add_argument("-R", "--start_idx", type=int, default=0, help="Execution starts from the first line of the parameter file")

  parser.add_argument("-O", "--output", type=str, default= "best_kernel", help="Best kernel saving path")

  args = parser.parse_args()

  return args



def init_csv_header(op: str, output_path: str):
  if not hasattr(init_csv_header, "header_line"):
    init_csv_header.header_line = ""

  if not init_csv_header.header_line:
    if not op:
      raise ValueError(f"Do not provide operation in input csv\n")
    if op.lower() == "gemm":
      init_csv_header.header_line = (
          "Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,lda,ldb,ldc,A,B,C,D,alpha,beta,"
          "split_k_mode,split_k_slices,batch_count,raster_order,stagger_k,stagger_k_stride,swizzle_size,op_class,accum,cta_m,cta_n,cta_k,"
          "cluster_m,cluster_n,cluster_k,stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,min_cc,max_cc,Bytes,"
          "Flops,Flops/Byte,Runtime,GB/s,GFLOPs\n"
      )
    elif op.lower() == "conv2d":
      init_csv_header.header_line = (
          "Problem,Provider,OperationKind,Operation,Disposition,Status,conv_kind,n,h,w,c,k,r,s,p,q,g,pad_h,pad_w,"
          "stride_h,stride_w,dilation_h,dilation_w,Activation,Filter,Output,conv_mode,iterator_algorithm,alpha,beta,"
          "split_k_mode,split_k_slices,eq_gemm_provider,op_class,accum,cta_m,cta_n,cta_k,cluster_m,cluster_n,cluster_k,"
          "stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,min_cc,max_cc,Bytes,Flops,Flops/Byte,Runtime,GB/s,GFLOPs\n"
      )
    else:
      raise ValueError(f"Unsupported operation type: {op}")

    if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
      with open(output_path, "w") as best_kernel_file:
        if op.lower() == "gemm":
          best_kernel_file.write(init_csv_header.header_line)
        elif op.lower():
          best_kernel_file.write(init_csv_header.header_line)

  return init_csv_header.header_line


def main():
  args = parseArgs()

  current_dir = os.getcwd()

  params_file = f"{current_dir}/{args.input}"
  save_path = f"{current_dir}/profiler_result/{args.output}"
  detail_profile_result_path_name = f"{save_path}/detail/profiler_gens"
  detail_stderr_path_name = f"{save_path}/detail/stderr"
  best_kernel_file_name = f"{save_path}/best_kernel.csv"

  folder_path = Path(save_path)
  folder_path.mkdir(parents=True, exist_ok=True)

  detail_stdout_path = Path(detail_profile_result_path_name)
  detail_stdout_path.mkdir(parents=True, exist_ok=True)

  detail_stderr_path = Path(detail_stderr_path_name)
  detail_stderr_path.mkdir(parents=True, exist_ok=True)

  hytlass_profiler = "../../build/tools/profiler/hytlass_profiler"

  logger = logging.getLogger("profiler_helper_logger")
  logger.setLevel(logging.DEBUG)

  file_handler = logging.FileHandler(f"{save_path}/profiler_helper.log", mode='a', encoding='utf-8')
  file_handler.setLevel(logging.DEBUG)

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.DEBUG)

  logger.addHandler(file_handler)
  logger.addHandler(console_handler)

  if not os.path.exists(params_file):
      logger.fatal(f"input csv file do not found at {params_file}")
      raise FileNotFoundError(f"input csv file: {params_file} do not found")    

  if not os.path.exists(hytlass_profiler):
      logger.fatal(f"hytlass profiler do not found at {hytlass_profiler}")
      raise FileNotFoundError(f"hytlass_profiler: {hytlass_profiler} do not found")

  total_lines = 0
  op = ""

  with open(params_file, "r") as f:
    total_lines = sum(1 for line in f)
  total_data_lines = total_lines - 1 

  current_line = 0

  for chunk in pd.read_csv(params_file, chunksize=1, skipinitialspace=True):
    current_line += 1

    if args.start_idx > current_line :
        continue

    chunk.columns = chunk.columns.str.strip()
    row_dict: dict = chunk.iloc[0].to_dict()
    profiler_args: list = []
    for key, value in row_dict.items():
      if key == "output":
          print(f"op is {op}")
      profiler_args.append(f"--{key}={value} ")

    op = row_dict.get("operation", "")
    try:
      init_csv_header(op, best_kernel_file_name)
    except Exception as e:
      logger.fatal(f"{e}")

    output_file = f"{detail_profile_result_path_name}/report_{current_line}.csv"

    profiler_args.append(f"--output={output_file}")
        
    # The profiler adds the operation to the generated filename.
    output_file = output_file.replace('.csv', f'.{op.lower()}.csv')

    disable_verification = row_dict.get("verification-enabled", "true") == "false"

    command = [hytlass_profiler] + profiler_args
    logger.info("+===============================================================================================================================+")
    logger.info(f"exec profiler [{current_line}/{total_data_lines}]")
    logger.info(f"{' '.join(command)}")

    profiler_status = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


    if profiler_status.returncode != 0 :
      logger.error(profiler_status.stderr)
      logger.error(f"status: [faild: {profiler_status.returncode}]")

      with open(f"{detail_stderr_path_name}/profier_stderr_{current_line}.csv", "w", encoding='utf-8') as _file:
              _file.write(profiler_status.stderr)

      with open(best_kernel_file_name, "a") as best_kernel_file:
        if op.lower() == "gemm":
          _m = row_dict.get("m", 1024)
          _n = row_dict.get("n", 1024)
          _k = row_dict.get("k", 1024)
          best_kernel_file.write(f"-,hytlass,Gemm,-,failed,failed,-,{_m},{_n},{_k},-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-\n")
        elif op.lower() == "conv2d":
          _n = row_dict.get("n", 1)
          _h = row_dict.get("h", 16)
          _w = row_dict.get("w", 16)
          _c = row_dict.get("c", 64)
          _k = row_dict.get("k", 64)
          _r = row_dict.get("r", 3)
          _s = row_dict.get("s", 3)
          _g = row_dict.get("g", 1)
          _pad_h = row_dict.get("pad_h", 1)
          _pad_w = row_dict.get("pad_w", 1)
          _stride_h = row_dict.get("stride_h", 1)
          _stride_w = row_dict.get("stride_w", 1)
          _dilation_h = row_dict.get("dilation_h", 1)
          _dilation_w = row_dict.get("dilation_w", 1)

          _p = row_dict.get("p", (_h + 2 * _pad_h - ((_r - 1) * _dilation_h + 1)) / (_stride_h) + 1)
          _q = row_dict.get("q", (_w + 2 * _pad_w - ((_s - 1) * _dilation_w + 1)) / (_stride_w) + 1)

          best_kernel_file.write(f"-,hytlass,Conv2d,-,failed,failed,-,{_n},{_h},{_w},{_c},{_k},{_r},{_s},{_p},{_q},{_g},{_pad_h},{_pad_w},"+
                                       f"{_stride_h},{_stride_w},{_dilation_h},{_dilation_w},-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-\n")
      continue

    kernel_file = pd.read_csv(output_file)

    if kernel_file.empty :
      logger.warning("Failed to generate valid data, possibly because there is no kernel available")

      # best_kernel_file = open(best_kernel_file_name, "a")

      with open(best_kernel_file_name, "a") as best_kernel_file:
        if op.lower() == "gemm":
          _m = row_dict.get("m", 1024)
          _n = row_dict.get("n", 1024)
          _k = row_dict.get("k", 1024)
          best_kernel_file.write(f"-,hytlass,Gemm,-,failed,no_res,-,{_m},{_n},{_k},-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-\n")
        elif op.lower() == "conv2d":
          _n = row_dict.get("n", 1)
          _h = row_dict.get("h", 16)
          _w = row_dict.get("w", 16)
          _c = row_dict.get("c", 64)
          _k = row_dict.get("k", 64)
          _r = row_dict.get("r", 3)
          _s = row_dict.get("s", 3)
          _g = row_dict.get("g", 1)
          _pad_h = row_dict.get("pad_h", 1)
          _pad_w = row_dict.get("pad_w", 1)
          _stride_h = row_dict.get("stride_h", 1)
          _stride_w = row_dict.get("stride_w", 1)
          _dilation_h = row_dict.get("dilation_h", 1)
          _dilation_w = row_dict.get("dilation_w", 1)

          _p = row_dict.get("p", (_h + 2 * _pad_h - ((_r - 1) * _dilation_h + 1)) / (_stride_h) + 1)
          _q = row_dict.get("q", (_w + 2 * _pad_w - ((_s - 1) * _dilation_w + 1)) / (_stride_w) + 1)

          best_kernel_file.write(f"-,hytlass,Conv2d,-,failed,no_res,-,{_n},{_h},{_w},{_c},{_k},{_r},{_s},{_p},{_q},{_g},{_pad_h},{_pad_w}," + 
                                        f"{_stride_h},{_stride_w},{_dilation_h},{_dilation_w},-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-\n")
      continue

    if not disable_verification:
      # An unverified result is not an error.
      failed_row = kernel_file[kernel_file["Disposition"] == "incorrect"]
      if not failed_row.empty:
        logger.error("kernel result errors")
        logger.error(failed_row)

    best_row = kernel_file.loc[kernel_file["GFLOPs"].idxmax()]

    best_row_df = pd.DataFrame([best_row], index=[0])

    best_row_df.to_csv(best_kernel_file_name, mode='a', index=False, header=False, encoding='utf-8', float_format='%.3f')

    logger.info(f"best kernel: {best_row_df['Operation'].values[0]}")
    logger.info(f"runtime: {round(best_row_df['Runtime'].values[0], 3)}")
    logger.info(f"Gflops: {round(best_row_df['GFLOPs'].values[0], 1)}")
    logger.info(f"status: [success]")

main()