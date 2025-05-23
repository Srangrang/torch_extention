//   Copyright (c) 2024 Xiaobai Team (Ao Dong, Linjin Li, Haishuai Zhang)
//   RV-MTVM(RISC-V Multi-thread Vector Martix) is licensed under Mulan PSL v2.
//   You can use this software according to the terms and conditions of the Mulan PSL v2. 
//   You may obtain a copy of Mulan PSL v2 at:
//            http://license.coscl.org.cn/MulanPSL2 
//   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
//   See the Mulan PSL v2 for more details.  



// hgemm函数执行半精度浮点数的矩阵乘法运算
// M, N, K 是矩阵的维度
// alpha 是乘法因子
// A, B 是输入矩阵的指针
// C 是输出矩阵的指针
// ldc 是C矩阵的列跨度
#include "hgemm.h"

int hgemm(unsigned int M, unsigned int N, unsigned int K, _Float16 alpha, _Float16* A, _Float16* B, _Float16* C, unsigned int ldc)

{
    unsigned int gvl = 0;	// 用于向量操作的变量长度
    unsigned int m_top = 0;	// 用于跟踪当前处理的M维度的索引
    unsigned int n_top = 0;	// 用于跟踪当前处理的N维度的索引


	// 主循环处理N维度
    for (unsigned int j=0; j<N/8; j+=1) {
        m_top = 0;
        unsigned int gvl = __riscv_vsetvl_e16m1(8);	// 设置向量长度为8


	    // 处理M维度，每次处理16个元素
        for (unsigned int i=0; i<M/16; i+=1) {
            // 以下代码块处理A和B矩阵的乘法
            // 它使用了RISC-V的向量指令来加速计算
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            _Float16 B0 = B[bi+0];
            _Float16 B1 = B[bi+1];
            _Float16 B2 = B[bi+2];
            _Float16 B3 = B[bi+3];
            _Float16 B4 = B[bi+4];
            _Float16 B5 = B[bi+5];
            _Float16 B6 = B[bi+6];
            _Float16 B7 = B[bi+7];
            bi += 8;

            vfloat16m1_t A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
            vfloat16m1_t A1 = __riscv_vle16_v_f16m1( &A[ai+1*gvl], gvl );
            ai += 16;

            vfloat16m1_t result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
            vfloat16m1_t result1 = __riscv_vfmul_vf_f16m1( A1, B0, gvl);
            vfloat16m1_t result2 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
            vfloat16m1_t result3 = __riscv_vfmul_vf_f16m1( A1, B1, gvl);
            vfloat16m1_t result4 = __riscv_vfmul_vf_f16m1( A0, B2, gvl);
            vfloat16m1_t result5 = __riscv_vfmul_vf_f16m1( A1, B2, gvl);
            vfloat16m1_t result6 = __riscv_vfmul_vf_f16m1( A0, B3, gvl);
            vfloat16m1_t result7 = __riscv_vfmul_vf_f16m1( A1, B3, gvl);
            vfloat16m1_t result8 = __riscv_vfmul_vf_f16m1( A0, B4, gvl);
            vfloat16m1_t result9 = __riscv_vfmul_vf_f16m1( A1, B4, gvl);
            vfloat16m1_t result10 = __riscv_vfmul_vf_f16m1( A0, B5, gvl);
            vfloat16m1_t result11 = __riscv_vfmul_vf_f16m1( A1, B5, gvl);
            vfloat16m1_t result12 = __riscv_vfmul_vf_f16m1( A0, B6, gvl);
            vfloat16m1_t result13 = __riscv_vfmul_vf_f16m1( A1, B6, gvl);
            vfloat16m1_t result14 = __riscv_vfmul_vf_f16m1( A0, B7, gvl);
            vfloat16m1_t result15 = __riscv_vfmul_vf_f16m1( A1, B7, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                B2 = B[bi+2];
                B3 = B[bi+3];
                B4 = B[bi+4];
                B5 = B[bi+5];
                B6 = B[bi+6];
                B7 = B[bi+7];
                bi += 8;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                A1 = __riscv_vle16_v_f16m1( &A[ai+1*gvl], gvl );
                ai += 16;

                result0 = __riscv_vfmacc_vf_f16m1( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f16m1( result1, B0, A1, gvl);
                result2 = __riscv_vfmacc_vf_f16m1( result2, B1, A0, gvl);
                result3 = __riscv_vfmacc_vf_f16m1( result3, B1, A1, gvl);
                result4 = __riscv_vfmacc_vf_f16m1( result4, B2, A0, gvl);
                result5 = __riscv_vfmacc_vf_f16m1( result5, B2, A1, gvl);
                result6 = __riscv_vfmacc_vf_f16m1( result6, B3, A0, gvl);
                result7 = __riscv_vfmacc_vf_f16m1( result7, B3, A1, gvl);
                result8 = __riscv_vfmacc_vf_f16m1( result8, B4, A0, gvl);
                result9 = __riscv_vfmacc_vf_f16m1( result9, B4, A1, gvl);
                result10 = __riscv_vfmacc_vf_f16m1( result10, B5, A0, gvl);
                result11 = __riscv_vfmacc_vf_f16m1( result11, B5, A1, gvl);
                result12 = __riscv_vfmacc_vf_f16m1( result12, B6, A0, gvl);
                result13 = __riscv_vfmacc_vf_f16m1( result13, B6, A1, gvl);
                result14 = __riscv_vfmacc_vf_f16m1( result14, B7, A0, gvl);
                result15 = __riscv_vfmacc_vf_f16m1( result15, B7, A1, gvl);
            }


            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c2 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c3 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c4 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c5 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c6 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c7 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c8 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c9 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c10 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c11 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c12 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c13 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c14 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c15 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f16m1( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f16m1( c1, alpha, result1, gvl );
            c2 = __riscv_vfmacc_vf_f16m1( c2, alpha, result2, gvl );
            c3 = __riscv_vfmacc_vf_f16m1( c3, alpha, result3, gvl );
            c4 = __riscv_vfmacc_vf_f16m1( c4, alpha, result4, gvl );
            c5 = __riscv_vfmacc_vf_f16m1( c5, alpha, result5, gvl );
            c6 = __riscv_vfmacc_vf_f16m1( c6, alpha, result6, gvl );
            c7 = __riscv_vfmacc_vf_f16m1( c7, alpha, result7, gvl );
            c8 = __riscv_vfmacc_vf_f16m1( c8, alpha, result8, gvl );
            c9 = __riscv_vfmacc_vf_f16m1( c9, alpha, result9, gvl );
            c10 = __riscv_vfmacc_vf_f16m1( c10, alpha, result10, gvl );
            c11 = __riscv_vfmacc_vf_f16m1( c11, alpha, result11, gvl );
            c12 = __riscv_vfmacc_vf_f16m1( c12, alpha, result12, gvl );
            c13 = __riscv_vfmacc_vf_f16m1( c13, alpha, result13, gvl );
            c14 = __riscv_vfmacc_vf_f16m1( c14, alpha, result14, gvl );
            c15 = __riscv_vfmacc_vf_f16m1( c15, alpha, result15, gvl );

            ci=n_top*ldc+m_top;

	        // 将计算结果存储回C矩阵
            __riscv_vse16_v_f16m1( &C[ci], c0, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c1, gvl); ci += ldc-gvl*1;
            __riscv_vse16_v_f16m1( &C[ci], c2, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c3, gvl); ci += ldc-gvl*1;
            __riscv_vse16_v_f16m1( &C[ci], c4, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c5, gvl); ci += ldc-gvl*1;
            __riscv_vse16_v_f16m1( &C[ci], c6, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c7, gvl); ci += ldc-gvl*1;
            __riscv_vse16_v_f16m1( &C[ci], c8, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c9, gvl); ci += ldc-gvl*1;
            __riscv_vse16_v_f16m1( &C[ci], c10, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c11, gvl); ci += ldc-gvl*1;
            __riscv_vse16_v_f16m1( &C[ci], c12, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c13, gvl); ci += ldc-gvl*1;
            __riscv_vse16_v_f16m1( &C[ci], c14, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c15, gvl);
            m_top += 16;
        }



        // 处理M维度的剩余部分（如果M不是16的倍数）
        // 这部分代码处理了8, 4, 2, 1个元素的情况

        if( M & 8 ) {
            gvl = __riscv_vsetvl_e16m1(8);

            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            _Float16 B0 = B[bi+0];
            _Float16 B1 = B[bi+1];
            _Float16 B2 = B[bi+2];
            _Float16 B3 = B[bi+3];
            _Float16 B4 = B[bi+4];
            _Float16 B5 = B[bi+5];
            _Float16 B6 = B[bi+6];
            _Float16 B7 = B[bi+7];
            bi += 8;

            vfloat16m1_t A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
            ai += 8;

            vfloat16m1_t result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
            vfloat16m1_t result1 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
            vfloat16m1_t result2 = __riscv_vfmul_vf_f16m1( A0, B2, gvl);
            vfloat16m1_t result3 = __riscv_vfmul_vf_f16m1( A0, B3, gvl);
            vfloat16m1_t result4 = __riscv_vfmul_vf_f16m1( A0, B4, gvl);
            vfloat16m1_t result5 = __riscv_vfmul_vf_f16m1( A0, B5, gvl);
            vfloat16m1_t result6 = __riscv_vfmul_vf_f16m1( A0, B6, gvl);
            vfloat16m1_t result7 = __riscv_vfmul_vf_f16m1( A0, B7, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                B2 = B[bi+2];
                B3 = B[bi+3];
                B4 = B[bi+4];
                B5 = B[bi+5];
                B6 = B[bi+6];
                B7 = B[bi+7];
                bi += 8;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                ai += 8;

                result0 = __riscv_vfmacc_vf_f16m1( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f16m1( result1, B1, A0, gvl);
                result2 = __riscv_vfmacc_vf_f16m1( result2, B2, A0, gvl);
                result3 = __riscv_vfmacc_vf_f16m1( result3, B3, A0, gvl);
                result4 = __riscv_vfmacc_vf_f16m1( result4, B4, A0, gvl);
                result5 = __riscv_vfmacc_vf_f16m1( result5, B5, A0, gvl);
                result6 = __riscv_vfmacc_vf_f16m1( result6, B6, A0, gvl);
                result7 = __riscv_vfmacc_vf_f16m1( result7, B7, A0, gvl);
            }


            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c2 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c3 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c4 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c5 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c6 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c7 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f16m1( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f16m1( c1, alpha, result1, gvl );
            c2 = __riscv_vfmacc_vf_f16m1( c2, alpha, result2, gvl );
            c3 = __riscv_vfmacc_vf_f16m1( c3, alpha, result3, gvl );
            c4 = __riscv_vfmacc_vf_f16m1( c4, alpha, result4, gvl );
            c5 = __riscv_vfmacc_vf_f16m1( c5, alpha, result5, gvl );
            c6 = __riscv_vfmacc_vf_f16m1( c6, alpha, result6, gvl );
            c7 = __riscv_vfmacc_vf_f16m1( c7, alpha, result7, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse16_v_f16m1( &C[ci], c0, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c1, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c2, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c3, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c4, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c5, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c6, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c7, gvl);
            m_top += 8;
        }


        if( M & 4 ) {
            gvl = __riscv_vsetvl_e16m1(4);

            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            _Float16 B0 = B[bi+0];
            _Float16 B1 = B[bi+1];
            _Float16 B2 = B[bi+2];
            _Float16 B3 = B[bi+3];
            _Float16 B4 = B[bi+4];
            _Float16 B5 = B[bi+5];
            _Float16 B6 = B[bi+6];
            _Float16 B7 = B[bi+7];
            bi += 8;

            vfloat16m1_t A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
            ai += 4;

            vfloat16m1_t result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
            vfloat16m1_t result1 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
            vfloat16m1_t result2 = __riscv_vfmul_vf_f16m1( A0, B2, gvl);
            vfloat16m1_t result3 = __riscv_vfmul_vf_f16m1( A0, B3, gvl);
            vfloat16m1_t result4 = __riscv_vfmul_vf_f16m1( A0, B4, gvl);
            vfloat16m1_t result5 = __riscv_vfmul_vf_f16m1( A0, B5, gvl);
            vfloat16m1_t result6 = __riscv_vfmul_vf_f16m1( A0, B6, gvl);
            vfloat16m1_t result7 = __riscv_vfmul_vf_f16m1( A0, B7, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                B2 = B[bi+2];
                B3 = B[bi+3];
                B4 = B[bi+4];
                B5 = B[bi+5];
                B6 = B[bi+6];
                B7 = B[bi+7];
                bi += 8;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                ai += 4;

                result0 = __riscv_vfmacc_vf_f16m1( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f16m1( result1, B1, A0, gvl);
                result2 = __riscv_vfmacc_vf_f16m1( result2, B2, A0, gvl);
                result3 = __riscv_vfmacc_vf_f16m1( result3, B3, A0, gvl);
                result4 = __riscv_vfmacc_vf_f16m1( result4, B4, A0, gvl);
                result5 = __riscv_vfmacc_vf_f16m1( result5, B5, A0, gvl);
                result6 = __riscv_vfmacc_vf_f16m1( result6, B6, A0, gvl);
                result7 = __riscv_vfmacc_vf_f16m1( result7, B7, A0, gvl);
            }


            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c2 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c3 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c4 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c5 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c6 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c7 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f16m1( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f16m1( c1, alpha, result1, gvl );
            c2 = __riscv_vfmacc_vf_f16m1( c2, alpha, result2, gvl );
            c3 = __riscv_vfmacc_vf_f16m1( c3, alpha, result3, gvl );
            c4 = __riscv_vfmacc_vf_f16m1( c4, alpha, result4, gvl );
            c5 = __riscv_vfmacc_vf_f16m1( c5, alpha, result5, gvl );
            c6 = __riscv_vfmacc_vf_f16m1( c6, alpha, result6, gvl );
            c7 = __riscv_vfmacc_vf_f16m1( c7, alpha, result7, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse16_v_f16m1( &C[ci], c0, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c1, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c2, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c3, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c4, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c5, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c6, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c7, gvl);
            m_top += 4;
        }


        if( M & 2 ) {
            _Float16 result0 = 0;
            _Float16 result1 = 0;
            _Float16 result2 = 0;
            _Float16 result3 = 0;
            _Float16 result4 = 0;
            _Float16 result5 = 0;
            _Float16 result6 = 0;
            _Float16 result7 = 0;
            _Float16 result8 = 0;
            _Float16 result9 = 0;
            _Float16 result10 = 0;
            _Float16 result11 = 0;
            _Float16 result12 = 0;
            _Float16 result13 = 0;
            _Float16 result14 = 0;
            _Float16 result15 = 0;
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;

            for(unsigned int k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                result1+=A[ai+1]*B[bi+0];
                result2+=A[ai+0]*B[bi+1];
                result3+=A[ai+1]*B[bi+1];
                result4+=A[ai+0]*B[bi+2];
                result5+=A[ai+1]*B[bi+2];
                result6+=A[ai+0]*B[bi+3];
                result7+=A[ai+1]*B[bi+3];
                result8+=A[ai+0]*B[bi+4];
                result9+=A[ai+1]*B[bi+4];
                result10+=A[ai+0]*B[bi+5];
                result11+=A[ai+1]*B[bi+5];
                result12+=A[ai+0]*B[bi+6];
                result13+=A[ai+1]*B[bi+6];
                result14+=A[ai+0]*B[bi+7];
                result15+=A[ai+1]*B[bi+7];
                ai+=2;
                bi+=8;
            }

            unsigned int ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            C[ci+0*ldc+1] += alpha * result1;
            C[ci+1*ldc+0] += alpha * result2;
            C[ci+1*ldc+1] += alpha * result3;
            C[ci+2*ldc+0] += alpha * result4;
            C[ci+2*ldc+1] += alpha * result5;
            C[ci+3*ldc+0] += alpha * result6;
            C[ci+3*ldc+1] += alpha * result7;
            C[ci+4*ldc+0] += alpha * result8;
            C[ci+4*ldc+1] += alpha * result9;
            C[ci+5*ldc+0] += alpha * result10;
            C[ci+5*ldc+1] += alpha * result11;
            C[ci+6*ldc+0] += alpha * result12;
            C[ci+6*ldc+1] += alpha * result13;
            C[ci+7*ldc+0] += alpha * result14;
            C[ci+7*ldc+1] += alpha * result15;
            m_top+=2;
        }


        if( M & 1 ) {
            _Float16 result0 = 0;
            _Float16 result1 = 0;
            _Float16 result2 = 0;
            _Float16 result3 = 0;
            _Float16 result4 = 0;
            _Float16 result5 = 0;
            _Float16 result6 = 0;
            _Float16 result7 = 0;
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;

            for(unsigned int k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                result1+=A[ai+0]*B[bi+1];
                result2+=A[ai+0]*B[bi+2];
                result3+=A[ai+0]*B[bi+3];
                result4+=A[ai+0]*B[bi+4];
                result5+=A[ai+0]*B[bi+5];
                result6+=A[ai+0]*B[bi+6];
                result7+=A[ai+0]*B[bi+7];
                ai+=1;
                bi+=8;
            }

            unsigned int ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            C[ci+1*ldc+0] += alpha * result1;
            C[ci+2*ldc+0] += alpha * result2;
            C[ci+3*ldc+0] += alpha * result3;
            C[ci+4*ldc+0] += alpha * result4;
            C[ci+5*ldc+0] += alpha * result5;
            C[ci+6*ldc+0] += alpha * result6;
            C[ci+7*ldc+0] += alpha * result7;
            m_top+=1;
        }

        n_top += 8;
    }




    // 如果N不是8的倍数，处理N维度的剩余部分（4）
    if( N & 4 ) {
        gvl = __riscv_vsetvl_e16m1(8);
        m_top = 0;

        for (unsigned int i=0; i<M/16; i+=1) {
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            _Float16 B0 = B[bi+0];
            _Float16 B1 = B[bi+1];
            _Float16 B2 = B[bi+2];
            _Float16 B3 = B[bi+3];
            bi += 4;

            vfloat16m1_t A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
            vfloat16m1_t A1 = __riscv_vle16_v_f16m1( &A[ai+1*gvl], gvl );
            ai += 16;

            vfloat16m1_t result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
            vfloat16m1_t result1 = __riscv_vfmul_vf_f16m1( A1, B0, gvl);
            vfloat16m1_t result2 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
            vfloat16m1_t result3 = __riscv_vfmul_vf_f16m1( A1, B1, gvl);
            vfloat16m1_t result4 = __riscv_vfmul_vf_f16m1( A0, B2, gvl);
            vfloat16m1_t result5 = __riscv_vfmul_vf_f16m1( A1, B2, gvl);
            vfloat16m1_t result6 = __riscv_vfmul_vf_f16m1( A0, B3, gvl);
            vfloat16m1_t result7 = __riscv_vfmul_vf_f16m1( A1, B3, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                B2 = B[bi+2];
                B3 = B[bi+3];
                bi += 4;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                A1 = __riscv_vle16_v_f16m1( &A[ai+1*gvl], gvl );
                ai += 16;

                result0 = __riscv_vfmacc_vf_f16m1( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f16m1( result1, B0, A1, gvl);
                result2 = __riscv_vfmacc_vf_f16m1( result2, B1, A0, gvl);
                result3 = __riscv_vfmacc_vf_f16m1( result3, B1, A1, gvl);
                result4 = __riscv_vfmacc_vf_f16m1( result4, B2, A0, gvl);
                result5 = __riscv_vfmacc_vf_f16m1( result5, B2, A1, gvl);
                result6 = __riscv_vfmacc_vf_f16m1( result6, B3, A0, gvl);
                result7 = __riscv_vfmacc_vf_f16m1( result7, B3, A1, gvl);
            }


            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c2 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c3 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c4 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c5 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c6 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c7 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f16m1( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f16m1( c1, alpha, result1, gvl );
            c2 = __riscv_vfmacc_vf_f16m1( c2, alpha, result2, gvl );
            c3 = __riscv_vfmacc_vf_f16m1( c3, alpha, result3, gvl );
            c4 = __riscv_vfmacc_vf_f16m1( c4, alpha, result4, gvl );
            c5 = __riscv_vfmacc_vf_f16m1( c5, alpha, result5, gvl );
            c6 = __riscv_vfmacc_vf_f16m1( c6, alpha, result6, gvl );
            c7 = __riscv_vfmacc_vf_f16m1( c7, alpha, result7, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse16_v_f16m1( &C[ci], c0, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c1, gvl); ci += ldc-gvl*1;
            __riscv_vse16_v_f16m1( &C[ci], c2, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c3, gvl); ci += ldc-gvl*1;
            __riscv_vse16_v_f16m1( &C[ci], c4, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c5, gvl); ci += ldc-gvl*1;
            __riscv_vse16_v_f16m1( &C[ci], c6, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c7, gvl);
            m_top += 16;
        }


        if( M & 8 ) {
            gvl = __riscv_vsetvl_e16m1(8);

            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            _Float16 B0 = B[bi+0];
            _Float16 B1 = B[bi+1];
            _Float16 B2 = B[bi+2];
            _Float16 B3 = B[bi+3];
            bi += 4;

            vfloat16m1_t A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
            ai += 8;

            vfloat16m1_t result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
            vfloat16m1_t result1 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
            vfloat16m1_t result2 = __riscv_vfmul_vf_f16m1( A0, B2, gvl);
            vfloat16m1_t result3 = __riscv_vfmul_vf_f16m1( A0, B3, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                B2 = B[bi+2];
                B3 = B[bi+3];
                bi += 4;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                ai += 8;

                result0 = __riscv_vfmacc_vf_f16m1( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f16m1( result1, B1, A0, gvl);
                result2 = __riscv_vfmacc_vf_f16m1( result2, B2, A0, gvl);
                result3 = __riscv_vfmacc_vf_f16m1( result3, B3, A0, gvl);
            }


            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c2 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c3 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f16m1( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f16m1( c1, alpha, result1, gvl );
            c2 = __riscv_vfmacc_vf_f16m1( c2, alpha, result2, gvl );
            c3 = __riscv_vfmacc_vf_f16m1( c3, alpha, result3, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse16_v_f16m1( &C[ci], c0, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c1, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c2, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c3, gvl);
            m_top += 8;
        }


        if( M & 4 ) {
            gvl = __riscv_vsetvl_e16m1(4);

            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            _Float16 B0 = B[bi+0];
            _Float16 B1 = B[bi+1];
            _Float16 B2 = B[bi+2];
            _Float16 B3 = B[bi+3];
            bi += 4;

            vfloat16m1_t A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
            ai += 4;

            vfloat16m1_t result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
            vfloat16m1_t result1 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
            vfloat16m1_t result2 = __riscv_vfmul_vf_f16m1( A0, B2, gvl);
            vfloat16m1_t result3 = __riscv_vfmul_vf_f16m1( A0, B3, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                B2 = B[bi+2];
                B3 = B[bi+3];
                bi += 4;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                ai += 4;

                result0 = __riscv_vfmacc_vf_f16m1( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f16m1( result1, B1, A0, gvl);
                result2 = __riscv_vfmacc_vf_f16m1( result2, B2, A0, gvl);
                result3 = __riscv_vfmacc_vf_f16m1( result3, B3, A0, gvl);
            }


            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c2 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c3 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f16m1( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f16m1( c1, alpha, result1, gvl );
            c2 = __riscv_vfmacc_vf_f16m1( c2, alpha, result2, gvl );
            c3 = __riscv_vfmacc_vf_f16m1( c3, alpha, result3, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse16_v_f16m1( &C[ci], c0, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c1, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c2, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c3, gvl);
            m_top += 4;
        }


        if( M & 2 ) {
            _Float16 result0 = 0;
            _Float16 result1 = 0;
            _Float16 result2 = 0;
            _Float16 result3 = 0;
            _Float16 result4 = 0;
            _Float16 result5 = 0;
            _Float16 result6 = 0;
            _Float16 result7 = 0;
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;

            for(unsigned int k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                result1+=A[ai+1]*B[bi+0];
                result2+=A[ai+0]*B[bi+1];
                result3+=A[ai+1]*B[bi+1];
                result4+=A[ai+0]*B[bi+2];
                result5+=A[ai+1]*B[bi+2];
                result6+=A[ai+0]*B[bi+3];
                result7+=A[ai+1]*B[bi+3];
                ai+=2;
                bi+=4;
            }

            unsigned int ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            C[ci+0*ldc+1] += alpha * result1;
            C[ci+1*ldc+0] += alpha * result2;
            C[ci+1*ldc+1] += alpha * result3;
            C[ci+2*ldc+0] += alpha * result4;
            C[ci+2*ldc+1] += alpha * result5;
            C[ci+3*ldc+0] += alpha * result6;
            C[ci+3*ldc+1] += alpha * result7;
            m_top+=2;
        }


        if( M & 1 ) {
            _Float16 result0 = 0;
            _Float16 result1 = 0;
            _Float16 result2 = 0;
            _Float16 result3 = 0;
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;

            for(unsigned int k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                result1+=A[ai+0]*B[bi+1];
                result2+=A[ai+0]*B[bi+2];
                result3+=A[ai+0]*B[bi+3];
                ai+=1;
                bi+=4;
            }

            unsigned int ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            C[ci+1*ldc+0] += alpha * result1;
            C[ci+2*ldc+0] += alpha * result2;
            C[ci+3*ldc+0] += alpha * result3;
            m_top+=1;
        }

        n_top += 4;
    }




    // 如果N不是4的倍数，处理N维度的剩余部分（2）
    if( N & 2 ) {
        gvl = __riscv_vsetvl_e16m1(8);
        m_top = 0;

        for (unsigned int i=0; i<M/16; i+=1) {
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            _Float16 B0 = B[bi+0];
            _Float16 B1 = B[bi+1];
            bi += 2;

            vfloat16m1_t A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
            vfloat16m1_t A1 = __riscv_vle16_v_f16m1( &A[ai+1*gvl], gvl );
            ai += 16;

            vfloat16m1_t result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
            vfloat16m1_t result1 = __riscv_vfmul_vf_f16m1( A1, B0, gvl);
            vfloat16m1_t result2 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
            vfloat16m1_t result3 = __riscv_vfmul_vf_f16m1( A1, B1, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                bi += 2;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                A1 = __riscv_vle16_v_f16m1( &A[ai+1*gvl], gvl );
                ai += 16;

                result0 = __riscv_vfmacc_vf_f16m1( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f16m1( result1, B0, A1, gvl);
                result2 = __riscv_vfmacc_vf_f16m1( result2, B1, A0, gvl);
                result3 = __riscv_vfmacc_vf_f16m1( result3, B1, A1, gvl);
            }


            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c2 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c3 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f16m1( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f16m1( c1, alpha, result1, gvl );
            c2 = __riscv_vfmacc_vf_f16m1( c2, alpha, result2, gvl );
            c3 = __riscv_vfmacc_vf_f16m1( c3, alpha, result3, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse16_v_f16m1( &C[ci], c0, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c1, gvl); ci += ldc-gvl*1;
            __riscv_vse16_v_f16m1( &C[ci], c2, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c3, gvl);
            m_top += 16;
        }


        if( M & 8 ) {
            gvl = __riscv_vsetvl_e16m1(8);

            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            _Float16 B0 = B[bi+0];
            _Float16 B1 = B[bi+1];
            bi += 2;

            vfloat16m1_t A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
            ai += 8;

            vfloat16m1_t result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
            vfloat16m1_t result1 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                bi += 2;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                ai += 8;

                result0 = __riscv_vfmacc_vf_f16m1( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f16m1( result1, B1, A0, gvl);
            }


            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f16m1( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f16m1( c1, alpha, result1, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse16_v_f16m1( &C[ci], c0, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c1, gvl);
            m_top += 8;
        }


        if( M & 4 ) {
            gvl = __riscv_vsetvl_e16m1(4);

            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            _Float16 B0 = B[bi+0];
            _Float16 B1 = B[bi+1];
            bi += 2;

            vfloat16m1_t A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
            ai += 4;

            vfloat16m1_t result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
            vfloat16m1_t result1 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                bi += 2;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                ai += 4;

                result0 = __riscv_vfmacc_vf_f16m1( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f16m1( result1, B1, A0, gvl);
            }


            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f16m1( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f16m1( c1, alpha, result1, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse16_v_f16m1( &C[ci], c0, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c1, gvl);
            m_top += 4;
        }


        if( M & 2 ) {
            _Float16 result0 = 0;
            _Float16 result1 = 0;
            _Float16 result2 = 0;
            _Float16 result3 = 0;
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;

            for(unsigned int k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                result1+=A[ai+1]*B[bi+0];
                result2+=A[ai+0]*B[bi+1];
                result3+=A[ai+1]*B[bi+1];
                ai+=2;
                bi+=2;
            }

            unsigned int ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            C[ci+0*ldc+1] += alpha * result1;
            C[ci+1*ldc+0] += alpha * result2;
            C[ci+1*ldc+1] += alpha * result3;
            m_top+=2;
        }


        if( M & 1 ) {
            _Float16 result0 = 0;
            _Float16 result1 = 0;
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            for(unsigned int k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                result1+=A[ai+0]*B[bi+1];
                ai+=1;
                bi+=2;
            }

            unsigned int ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            C[ci+1*ldc+0] += alpha * result1;
            m_top+=1;
        }

        n_top += 2;
    }


    // 如果N不是2的倍数，处理N维度的剩余部分（1）
    if( N & 1 ) {
        gvl = __riscv_vsetvl_e16m1(8);
        m_top = 0;

        for (unsigned int i=0; i<M/16; i+=1) {
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            _Float16 B0 = B[bi+0];
            bi += 1;

            vfloat16m1_t A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
            vfloat16m1_t A1 = __riscv_vle16_v_f16m1( &A[ai+1*gvl], gvl );
            ai += 16;

            vfloat16m1_t result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
            vfloat16m1_t result1 = __riscv_vfmul_vf_f16m1( A1, B0, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                bi += 1;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                A1 = __riscv_vle16_v_f16m1( &A[ai+1*gvl], gvl );
                ai += 16;

                result0 = __riscv_vfmacc_vf_f16m1( result0, B0, A0, gvl);
                result1 = __riscv_vfmacc_vf_f16m1( result1, B0, A1, gvl);
            }


            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f16m1( c0, alpha, result0, gvl );
            c1 = __riscv_vfmacc_vf_f16m1( c1, alpha, result1, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse16_v_f16m1( &C[ci], c0, gvl); ci += gvl;
            __riscv_vse16_v_f16m1( &C[ci], c1, gvl);
            m_top += 16;
        }


        if( M & 8 ) {
            gvl = __riscv_vsetvl_e16m1(8);

            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            _Float16 B0 = B[bi+0];
            bi += 1;

            vfloat16m1_t A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
            ai += 8;

            vfloat16m1_t result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                bi += 1;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                ai += 8;

                result0 = __riscv_vfmacc_vf_f16m1( result0, B0, A0, gvl);
            }


            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f16m1( c0, alpha, result0, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse16_v_f16m1( &C[ci], c0, gvl);
            m_top += 8;
        }


        if( M & 4 ) {
            gvl = __riscv_vsetvl_e16m1(4);

            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            _Float16 B0 = B[bi+0];
            bi += 1;

            vfloat16m1_t A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
            ai += 4;

            vfloat16m1_t result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                bi += 1;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                ai += 4;

                result0 = __riscv_vfmacc_vf_f16m1( result0, B0, A0, gvl);
            }


            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            c0 = __riscv_vfmacc_vf_f16m1( c0, alpha, result0, gvl );

            ci=n_top*ldc+m_top;

            __riscv_vse16_v_f16m1( &C[ci], c0, gvl);
            m_top += 4;
        }


        if( M & 2 ) {
            _Float16 result0 = 0;
            _Float16 result1 = 0;
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;

            for(unsigned int k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                result1+=A[ai+1]*B[bi+0];
                ai+=2;
                bi+=1;
            }

            unsigned int ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            C[ci+0*ldc+1] += alpha * result1;
            m_top+=2;
        }


        if( M & 1 ) {
            _Float16 result0 = 0;
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            for(unsigned int k=0; k<K; k++) {
                result0+=A[ai+0]*B[bi+0];
                ai+=1;
                bi+=1;
            }

            unsigned int ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] += alpha * result0;
            m_top+=1;
        }

        n_top += 1;
    }

    return 0;
}
