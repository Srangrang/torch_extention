//   Copyright (c) 2024 Xiaobai Team (Ao Dong, Linjin Li, Haishuai Zhang)
//   RV-MTVM(RISC-V Multi-thread Vector Martix) is licensed under Mulan PSL v2.
//   You can use this software according to the terms and conditions of the Mulan PSL v2. 
//   You may obtain a copy of Mulan PSL v2 at:
//            http://license.coscl.org.cn/MulanPSL2 
//   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
//   See the Mulan PSL v2 for more details.  

#include <riscv_vector.h>
#include <stdio.h>

#define VSETVL(n)               __riscv_vsetvl_e16m4(n)	// 设置向量长度
#define FLOAT_V_T               vfloat16m4_t	// 向量类型定义
#define VLEV_FLOAT              __riscv_vle16_v_f16m4	// 向量载入函数
#define VLSEV_FLOAT             __riscv_vlse16_v_f16m4	// 向量载入函数（支持步长）
#define VSEV_FLOAT              __riscv_vse16_v_f16m4	// 向量存储函数
#define VSSEV_FLOAT             __riscv_vsse16_v_f16m4	// 向量存储函数（支持步长）
#define VFMACCVF_FLOAT          __riscv_vfmacc_vf_f16m4	// 向量乘加函数

// hgemv函数执行矩阵-向量乘法运算
// m, n 分别是矩阵的行数和列数
// dummy1 是一个未使用的参数（用于后续扩展）
// alpha 是乘法因子
// a 是矩阵的指针
// lda 是矩阵的列跨度（leading dimension）
// x 是向量的指针
// inc_x 是向量x的增量
// y 是结果向量的指针
// inc_y 是结果向量y的增量
// buffer 是一个未使用的参数（用于后续扩展）

int hgemv(unsigned int m, unsigned int n, unsigned int dummy1, _Float16 alpha, _Float16 *a, unsigned int lda, _Float16 *x, unsigned int inc_x, _Float16 *y, unsigned int inc_y, _Float16 *buffer)
{
    if(n < 0)  return 0;	// 如果列数n为负，直接返回

    _Float16 *a_ptr, *x_ptr;	// 指针用于遍历矩阵和向量
    unsigned int i;	// 循环变量
    FLOAT_V_T va, vy;	// 向量变量
// 当y向量的增量inc_y为1时，使用连续存储的方式处理
    if(inc_y == 1) {

        for (size_t vl; m > 0; ) {
            vl = VSETVL(m);// 设置向量长度
            a_ptr = a;// 初始化a_ptr指向矩阵a的当前行
            x_ptr = x;// 初始化x_ptr指向向量x的当前元素
            vy = VLEV_FLOAT(y, vl);// 从y向量载入向量数据到vy
            // 遍历每一列
            for(i = 0; i < n; i++) {
                va = VLEV_FLOAT(a_ptr, vl);// 从矩阵a载入向量数据到va
                // 执行向量乘加操作，vy = vy + alpha * x_ptr[i] * va
                vy = VFMACCVF_FLOAT(vy, (alpha * (*x_ptr)), va, vl);

                a_ptr += lda;// 移动到矩阵a的下一列
                x_ptr += inc_x;// 移动到向量x的下一元素
            }
            VSEV_FLOAT(y, vy, vl);// 将计算结果存回y向量
             m -= vl;	// 更新剩余的m值
             y += vl; 	// 移动到y向量的下一元素
             a += vl;	// 移动到矩阵a的下一初始元素
        }
 
    } else {
	// 当y向量的增量inc_y不为1时，使用步长存储的方式处理
        unsigned int stride_y = inc_y * sizeof(_Float16);// 计算y向量的步长
	
        for (size_t vl; m > 0; m -= vl, y += vl*inc_y, a += vl) {
            vl = VSETVL(m);
            a_ptr = a;
            x_ptr = x;
            vy = VLSEV_FLOAT(y, stride_y, vl);
            for(i = 0; i < n; i++) {
                va = VLEV_FLOAT(a_ptr, vl);
                vy = VFMACCVF_FLOAT(vy, (alpha * (*x_ptr)), va, vl);

                a_ptr += lda;
                x_ptr += inc_x;
            }
            VSSEV_FLOAT(y, stride_y, vy, vl);
        }

    }
    return 0;
}
