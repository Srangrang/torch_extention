//   Copyright (c) 2024 Xiaobai Team (Ao Dong, Linjin Li, Haishuai Zhang)
//   RV-MTVM(RISC-V Multi-thread Vector Martix) is licensed under Mulan PSL v2.
//   You can use this software according to the terms and conditions of the Mulan PSL v2. 
//   You may obtain a copy of Mulan PSL v2 at:
//            http://license.coscl.org.cn/MulanPSL2 
//   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
//   See the Mulan PSL v2 for more details.  


#include "hgemv.h"

int gemv(unsigned int m, unsigned int n, unsigned int dummy1, _Float16 alpha, _Float16 *a, unsigned int lda, _Float16 *x, unsigned int inc_x, _Float16 *y, unsigned int inc_y) {
	hgemv( m, n, dummy1, alpha, a, lda, x, inc_x, y, inc_y);
	return 0;
}