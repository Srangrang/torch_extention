//   Copyright (c) 2024 Xiaobai Team (Ao Dong, Linjin Li, Haishuai Zhang)
//   RV-MTVM(RISC-V Multi-thread Vector Martix) is licensed under Mulan PSL v2.
//   You can use this software according to the terms and conditions of the Mulan PSL v2. 
//   You may obtain a copy of Mulan PSL v2 at:
//            http://license.coscl.org.cn/MulanPSL2 
//   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
//   See the Mulan PSL v2 for more details.  
#ifndef GEMM_H
#define GEMM_H

#include <riscv_vector.h>
#include <malloc.h>
#include "gemm_ncopy8_g.h"
#include "gemm_tcopy16_g.h"
#include "hgemm.h"

int gemm(unsigned int M, unsigned int N, unsigned int K, _Float16 alpha, _Float16* A, _Float16* B, _Float16* C);

#endif
