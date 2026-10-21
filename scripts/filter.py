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
import sys
import datetime
#log path
file_path=sys.argv[1]
#kernel counts
kernel_sip=sys.argv[2]
contents=[]
with open(file_path, 'r') as file:
    for line in file:
        if '1,HYTLASS,gemm,hytlass_simt_f16_sgemm' in line:
            contents.append(line)
# print(len(contents))
save_index = 0
perf_max = 0
filter_save_max=[]
inner_loop = 0
for i in range(len(contents)):
    inner_loop += 1
    perf = contents[i].split(',')[-1]
    if perf_max < float(perf):
        save_index = i
        perf_max = float(perf)
    if inner_loop == int(kernel_sip):
        # print(contents[save_index])
        filter_save_max.append(contents[save_index])
        perf_max = 0
        inner_loop =0
# print(len(filter_save_max))  
save_path='./'+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_gemm.csv';     
with open(save_path, 'w') as file:
    file.write('Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,A,B,C,D,alpha,beta,split_k_mode,split_k_slices,batch_count,raster_order,op_class,accum,cta_m,cta_n,cta_k,cluster_m,cluster_n,cluster_k,stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,min_cc,max_cc,Bytes,Flops,Flops/Byte,Runtime,GB/s,GFLOPs\n')
    for i in filter_save_max:
        file.write(i)
            