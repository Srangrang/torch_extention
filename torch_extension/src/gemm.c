//   Copyright (c) 2024 Xiaobai Team (Ao Dong, Linjin Li, Haishuai Zhang)
//   RV-MTVM(RISC-V Multi-thread Vector Martix) is licensed under Mulan PSL v2.
//   You can use this software according to the terms and conditions of the Mulan PSL v2. 
//   You may obtain a copy of Mulan PSL v2 at:
//            http://license.coscl.org.cn/MulanPSL2 
//   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
//   See the Mulan PSL v2 for more details.  

#include "gemm.h"

int gemm(unsigned int M, unsigned int N, unsigned int K, _Float16 alpha, _Float16* A, _Float16* B, _Float16* C)

{
	_Float16 *B_trans = (_Float16 *)malloc(K * N * sizeof(_Float16));
	_Float16 *A_trans = (_Float16 *)malloc(K * M * sizeof(_Float16));

	gemm_ncopy(K, N, B, K, B_trans);

	gemm_tcopy_16(K, M, A, M, A_trans);
	
	hgemm(M, N, K, alpha, A_trans, B_trans, C, M);
	
	free(B_trans);
	free(A_trans);
	return 0;
}
