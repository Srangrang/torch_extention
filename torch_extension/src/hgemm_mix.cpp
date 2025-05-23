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
#include "hgemm_mix.hpp"

int hgemm_mix(unsigned int M, unsigned int N, unsigned int K, _Float16 alpha, _Float16* A, _Float16* B, _Float16* C, unsigned int ldc)

{

    unsigned int gvl = 0;	// 向量长度变量
    unsigned int m_top = 0;	// M维度的顶部索引
    unsigned int n_top = 0;	// N维度的顶部索引

    // -- MAIN PASS
    // 主循环处理N维度，每次处理8个元素
    for (unsigned int j=0; j<N/8; j+=1) {
        m_top = 0;
        unsigned int gvl = __riscv_vsetvl_e16m1(8);// 设置向量长度为8

        for (unsigned int i=0; i<M/16; i+=1) {
            unsigned int ai=m_top*K;	// A矩阵的当前行索引
            unsigned int bi=n_top*K;	// B矩阵的当前列索引
            // 加载B矩阵的8个元素
            _Float16 B0 = B[bi+0];
            _Float16 B1 = B[bi+1];
            _Float16 B2 = B[bi+2];
            _Float16 B3 = B[bi+3];
            _Float16 B4 = B[bi+4];
            _Float16 B5 = B[bi+5];
            _Float16 B6 = B[bi+6];
            _Float16 B7 = B[bi+7];
            bi += 8;

		// 加载A矩阵的16个元素，并与B矩阵元素相乘
            vfloat16m1_t A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
            vfloat16m1_t A1 = __riscv_vle16_v_f16m1( &A[ai+1*gvl], gvl );
            ai += 16;
		// 执行乘法运算，并转换为32位浮点数进行累加
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
             // 将16位半精度浮点数结果转换为32位单精度浮点数
            vfloat32m2_t result0_32 = __riscv_vfwcvt_f_f_v_f32m2( result0, gvl);
            vfloat32m2_t result1_32 = __riscv_vfwcvt_f_f_v_f32m2( result1, gvl);
            vfloat32m2_t result2_32 = __riscv_vfwcvt_f_f_v_f32m2( result2, gvl);
            vfloat32m2_t result3_32 = __riscv_vfwcvt_f_f_v_f32m2( result3, gvl);
            vfloat32m2_t result4_32 = __riscv_vfwcvt_f_f_v_f32m2( result4, gvl);
            vfloat32m2_t result5_32 = __riscv_vfwcvt_f_f_v_f32m2( result5, gvl);
            vfloat32m2_t result6_32 = __riscv_vfwcvt_f_f_v_f32m2( result6, gvl);
            vfloat32m2_t result7_32 = __riscv_vfwcvt_f_f_v_f32m2( result7, gvl);
            vfloat32m2_t result8_32 = __riscv_vfwcvt_f_f_v_f32m2( result8, gvl);
            vfloat32m2_t result9_32 = __riscv_vfwcvt_f_f_v_f32m2( result9, gvl);
            vfloat32m2_t result10_32 = __riscv_vfwcvt_f_f_v_f32m2( result10, gvl);
            vfloat32m2_t result11_32 = __riscv_vfwcvt_f_f_v_f32m2( result11, gvl);
            vfloat32m2_t result12_32 = __riscv_vfwcvt_f_f_v_f32m2( result12, gvl);
            vfloat32m2_t result13_32 = __riscv_vfwcvt_f_f_v_f32m2( result13, gvl);
            vfloat32m2_t result14_32 = __riscv_vfwcvt_f_f_v_f32m2( result14, gvl);
            vfloat32m2_t result15_32 = __riscv_vfwcvt_f_f_v_f32m2( result15, gvl);
		// 循环处理K维度的剩余部分
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
		// 执行乘法和累加运算
		vfloat16m1_t mul_result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
                vfloat16m1_t mul_result1 = __riscv_vfmul_vf_f16m1( A1, B0, gvl);
                vfloat16m1_t mul_result2 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
                vfloat16m1_t mul_result3 = __riscv_vfmul_vf_f16m1( A1, B1, gvl);
                vfloat16m1_t mul_result4 = __riscv_vfmul_vf_f16m1( A0, B2, gvl);
                vfloat16m1_t mul_result5 = __riscv_vfmul_vf_f16m1( A1, B2, gvl);
                vfloat16m1_t mul_result6 = __riscv_vfmul_vf_f16m1( A0, B3, gvl);
                vfloat16m1_t mul_result7 = __riscv_vfmul_vf_f16m1( A1, B3, gvl);
                vfloat16m1_t mul_result8 = __riscv_vfmul_vf_f16m1( A0, B4, gvl);
                vfloat16m1_t mul_result9 = __riscv_vfmul_vf_f16m1( A1, B4, gvl);
                vfloat16m1_t mul_result10 = __riscv_vfmul_vf_f16m1( A0, B5, gvl);
                vfloat16m1_t mul_result11 = __riscv_vfmul_vf_f16m1( A1, B5, gvl);
                vfloat16m1_t mul_result12 = __riscv_vfmul_vf_f16m1( A0, B6, gvl);
                vfloat16m1_t mul_result13 = __riscv_vfmul_vf_f16m1( A1, B6, gvl);
                vfloat16m1_t mul_result14 = __riscv_vfmul_vf_f16m1( A0, B7, gvl);
                vfloat16m1_t mul_result15 = __riscv_vfmul_vf_f16m1( A1, B7, gvl);
		// 将16位半精度浮点数结果转换为32位单精度浮点数
                vfloat32m2_t mul_result0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result0, gvl);
                vfloat32m2_t mul_result1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result1, gvl);
                vfloat32m2_t mul_result2_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result2, gvl);
                vfloat32m2_t mul_result3_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result3, gvl);
                vfloat32m2_t mul_result4_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result4, gvl);
                vfloat32m2_t mul_result5_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result5, gvl);
                vfloat32m2_t mul_result6_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result6, gvl);
                vfloat32m2_t mul_result7_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result7, gvl);
                vfloat32m2_t mul_result8_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result8, gvl);
                vfloat32m2_t mul_result9_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result9, gvl);
                vfloat32m2_t mul_result10_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result10, gvl);
                vfloat32m2_t mul_result11_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result11, gvl);
                vfloat32m2_t mul_result12_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result12, gvl);
                vfloat32m2_t mul_result13_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result13, gvl);
                vfloat32m2_t mul_result14_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result14, gvl);
                vfloat32m2_t mul_result15_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result15, gvl);
                
                result0_32 = __riscv_vfadd_vv_f32m2( result0_32, mul_result0_32, gvl);
                result1_32 = __riscv_vfadd_vv_f32m2( result1_32, mul_result1_32, gvl);
                result2_32 = __riscv_vfadd_vv_f32m2( result2_32, mul_result2_32, gvl);
                result3_32 = __riscv_vfadd_vv_f32m2( result3_32, mul_result3_32, gvl);
                result4_32 = __riscv_vfadd_vv_f32m2( result4_32, mul_result4_32, gvl);
                result5_32 = __riscv_vfadd_vv_f32m2( result5_32, mul_result5_32, gvl);
                result6_32 = __riscv_vfadd_vv_f32m2( result6_32, mul_result6_32, gvl);
                result7_32 = __riscv_vfadd_vv_f32m2( result7_32, mul_result7_32, gvl);
                result8_32 = __riscv_vfadd_vv_f32m2( result8_32, mul_result8_32, gvl);
                result9_32 = __riscv_vfadd_vv_f32m2( result9_32, mul_result9_32, gvl);
                result10_32 = __riscv_vfadd_vv_f32m2( result10_32, mul_result10_32, gvl);
                result11_32 = __riscv_vfadd_vv_f32m2( result11_32, mul_result11_32, gvl);
                result12_32 = __riscv_vfadd_vv_f32m2( result12_32, mul_result12_32, gvl);
                result13_32 = __riscv_vfadd_vv_f32m2( result13_32, mul_result13_32, gvl);
                result14_32 = __riscv_vfadd_vv_f32m2( result14_32, mul_result14_32, gvl);
                result15_32 = __riscv_vfadd_vv_f32m2( result15_32, mul_result15_32, gvl);
            }
		// 将最终的32位单精度浮点数结果转换回16位半精度浮点数
            result0 = __riscv_vfncvt_f_f_w_f16m1( result0_32, gvl);
            result1 = __riscv_vfncvt_f_f_w_f16m1( result1_32, gvl);
            result2 = __riscv_vfncvt_f_f_w_f16m1( result2_32, gvl);
            result3 = __riscv_vfncvt_f_f_w_f16m1( result3_32, gvl);
            result4 = __riscv_vfncvt_f_f_w_f16m1( result4_32, gvl);
            result5 = __riscv_vfncvt_f_f_w_f16m1( result5_32, gvl);
            result6 = __riscv_vfncvt_f_f_w_f16m1( result6_32, gvl);
            result7 = __riscv_vfncvt_f_f_w_f16m1( result7_32, gvl);
            result8 = __riscv_vfncvt_f_f_w_f16m1( result8_32, gvl);
            result9 = __riscv_vfncvt_f_f_w_f16m1( result9_32, gvl);
            result10 = __riscv_vfncvt_f_f_w_f16m1( result10_32, gvl);
            result11 = __riscv_vfncvt_f_f_w_f16m1( result11_32, gvl);
            result12 = __riscv_vfncvt_f_f_w_f16m1( result12_32, gvl);
            result13 = __riscv_vfncvt_f_f_w_f16m1( result13_32, gvl);
            result14 = __riscv_vfncvt_f_f_w_f16m1( result14_32, gvl);
            result15 = __riscv_vfncvt_f_f_w_f16m1( result15_32, gvl);
	 // 加载C矩阵的元素，并与计算结果相加
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
            	// 将C矩阵元素转换为32位单精度浮点数，并与计算结果相加
            vfloat32m2_t c0_32 = __riscv_vfwcvt_f_f_v_f32m2( c0, gvl);
            vfloat32m2_t c1_32 = __riscv_vfwcvt_f_f_v_f32m2( c1, gvl);
            vfloat32m2_t c2_32 = __riscv_vfwcvt_f_f_v_f32m2( c2, gvl);
            vfloat32m2_t c3_32 = __riscv_vfwcvt_f_f_v_f32m2( c3, gvl);
            vfloat32m2_t c4_32 = __riscv_vfwcvt_f_f_v_f32m2( c4, gvl);
            vfloat32m2_t c5_32 = __riscv_vfwcvt_f_f_v_f32m2( c5, gvl);
            vfloat32m2_t c6_32 = __riscv_vfwcvt_f_f_v_f32m2( c6, gvl);
            vfloat32m2_t c7_32 = __riscv_vfwcvt_f_f_v_f32m2( c7, gvl);
            vfloat32m2_t c8_32 = __riscv_vfwcvt_f_f_v_f32m2( c8, gvl);
            vfloat32m2_t c9_32 = __riscv_vfwcvt_f_f_v_f32m2( c9, gvl);
            vfloat32m2_t c10_32 = __riscv_vfwcvt_f_f_v_f32m2( c10, gvl);
            vfloat32m2_t c11_32 = __riscv_vfwcvt_f_f_v_f32m2( c11, gvl);
            vfloat32m2_t c12_32 = __riscv_vfwcvt_f_f_v_f32m2( c12, gvl);
            vfloat32m2_t c13_32 = __riscv_vfwcvt_f_f_v_f32m2( c13, gvl);
            vfloat32m2_t c14_32 = __riscv_vfwcvt_f_f_v_f32m2( c14, gvl);
            vfloat32m2_t c15_32 = __riscv_vfwcvt_f_f_v_f32m2( c15, gvl);
            	// 将alpha因子应用到计算结果上，并与C矩阵元素相加
            vfloat16m1_t mul_c0 = __riscv_vfmul_vf_f16m1( result0, alpha, gvl);
            vfloat16m1_t mul_c1 = __riscv_vfmul_vf_f16m1( result1, alpha, gvl);
            vfloat16m1_t mul_c2 = __riscv_vfmul_vf_f16m1( result2, alpha, gvl);
            vfloat16m1_t mul_c3 = __riscv_vfmul_vf_f16m1( result3, alpha, gvl);
            vfloat16m1_t mul_c4 = __riscv_vfmul_vf_f16m1( result4, alpha, gvl);
            vfloat16m1_t mul_c5 = __riscv_vfmul_vf_f16m1( result5, alpha, gvl);
            vfloat16m1_t mul_c6 = __riscv_vfmul_vf_f16m1( result6, alpha, gvl);
            vfloat16m1_t mul_c7 = __riscv_vfmul_vf_f16m1( result7, alpha, gvl);
            vfloat16m1_t mul_c8 = __riscv_vfmul_vf_f16m1( result8, alpha, gvl);
            vfloat16m1_t mul_c9 = __riscv_vfmul_vf_f16m1( result9, alpha, gvl);
            vfloat16m1_t mul_c10 = __riscv_vfmul_vf_f16m1( result10, alpha, gvl);
            vfloat16m1_t mul_c11 = __riscv_vfmul_vf_f16m1( result11, alpha, gvl);
            vfloat16m1_t mul_c12 = __riscv_vfmul_vf_f16m1( result12, alpha, gvl);
            vfloat16m1_t mul_c13 = __riscv_vfmul_vf_f16m1( result13, alpha, gvl);
            vfloat16m1_t mul_c14 = __riscv_vfmul_vf_f16m1( result14, alpha, gvl);
            vfloat16m1_t mul_c15 = __riscv_vfmul_vf_f16m1( result15, alpha, gvl);
            // 将16位半精度浮点数结果转换为32位单精度浮点数
            vfloat32m2_t mul_c0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c0, gvl);
            vfloat32m2_t mul_c1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c1, gvl);
            vfloat32m2_t mul_c2_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c2, gvl);
            vfloat32m2_t mul_c3_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c3, gvl);
            vfloat32m2_t mul_c4_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c4, gvl);
            vfloat32m2_t mul_c5_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c5, gvl);
            vfloat32m2_t mul_c6_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c6, gvl);
            vfloat32m2_t mul_c7_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c7, gvl);
            vfloat32m2_t mul_c8_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c8, gvl);
            vfloat32m2_t mul_c9_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c9, gvl);
            vfloat32m2_t mul_c10_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c10, gvl);
            vfloat32m2_t mul_c11_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c11, gvl);
            vfloat32m2_t mul_c12_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c12, gvl);
            vfloat32m2_t mul_c13_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c13, gvl);
            vfloat32m2_t mul_c14_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c14, gvl);
            vfloat32m2_t mul_c15_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c15, gvl);
            
            c0_32 = __riscv_vfadd_vv_f32m2( c0_32, mul_c0_32, gvl);
            c1_32 = __riscv_vfadd_vv_f32m2( c1_32, mul_c1_32, gvl);
            c2_32 = __riscv_vfadd_vv_f32m2( c2_32, mul_c2_32, gvl);
            c3_32 = __riscv_vfadd_vv_f32m2( c3_32, mul_c3_32, gvl);
            c4_32 = __riscv_vfadd_vv_f32m2( c4_32, mul_c4_32, gvl);
            c5_32 = __riscv_vfadd_vv_f32m2( c5_32, mul_c5_32, gvl);
            c6_32 = __riscv_vfadd_vv_f32m2( c6_32, mul_c6_32, gvl);
            c7_32 = __riscv_vfadd_vv_f32m2( c7_32, mul_c7_32, gvl);
            c8_32 = __riscv_vfadd_vv_f32m2( c8_32, mul_c8_32, gvl);
            c9_32 = __riscv_vfadd_vv_f32m2( c9_32, mul_c9_32, gvl);
            c10_32 = __riscv_vfadd_vv_f32m2( c10_32, mul_c10_32, gvl);
            c11_32 = __riscv_vfadd_vv_f32m2( c11_32, mul_c11_32, gvl);
            c12_32 = __riscv_vfadd_vv_f32m2( c12_32, mul_c12_32, gvl);
            c13_32 = __riscv_vfadd_vv_f32m2( c13_32, mul_c13_32, gvl);
            c14_32 = __riscv_vfadd_vv_f32m2( c14_32, mul_c14_32, gvl);
            c15_32 = __riscv_vfadd_vv_f32m2( c15_32, mul_c15_32, gvl);
            // 将最终的32位单精度浮点数结果转换回16位半精度浮点数，并存储回C矩阵
            c0 = __riscv_vfncvt_f_f_w_f16m1( c0_32, gvl);
            c1 = __riscv_vfncvt_f_f_w_f16m1( c1_32, gvl);
            c2 = __riscv_vfncvt_f_f_w_f16m1( c2_32, gvl);
            c3 = __riscv_vfncvt_f_f_w_f16m1( c3_32, gvl);
            c4 = __riscv_vfncvt_f_f_w_f16m1( c4_32, gvl);
            c5 = __riscv_vfncvt_f_f_w_f16m1( c5_32, gvl);
            c6 = __riscv_vfncvt_f_f_w_f16m1( c6_32, gvl);
            c7 = __riscv_vfncvt_f_f_w_f16m1( c7_32, gvl);
            c8 = __riscv_vfncvt_f_f_w_f16m1( c8_32, gvl);
            c9 = __riscv_vfncvt_f_f_w_f16m1( c9_32, gvl);
            c10 = __riscv_vfncvt_f_f_w_f16m1( c10_32, gvl);
            c11 = __riscv_vfncvt_f_f_w_f16m1( c11_32, gvl);
            c12 = __riscv_vfncvt_f_f_w_f16m1( c12_32, gvl);
            c13 = __riscv_vfncvt_f_f_w_f16m1( c13_32, gvl);
            c14 = __riscv_vfncvt_f_f_w_f16m1( c14_32, gvl);
            c15 = __riscv_vfncvt_f_f_w_f16m1( c15_32, gvl);

            ci=n_top*ldc+m_top;

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



        // -- tails for main pass
	// 处理M维度的剩余部分（如果M不是16的倍数）
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
            
            vfloat32m2_t result0_32 = __riscv_vfwcvt_f_f_v_f32m2( result0, gvl);
            vfloat32m2_t result1_32 = __riscv_vfwcvt_f_f_v_f32m2( result1, gvl);
            vfloat32m2_t result2_32 = __riscv_vfwcvt_f_f_v_f32m2( result2, gvl);
            vfloat32m2_t result3_32 = __riscv_vfwcvt_f_f_v_f32m2( result3, gvl);
            vfloat32m2_t result4_32 = __riscv_vfwcvt_f_f_v_f32m2( result4, gvl);
            vfloat32m2_t result5_32 = __riscv_vfwcvt_f_f_v_f32m2( result5, gvl);
            vfloat32m2_t result6_32 = __riscv_vfwcvt_f_f_v_f32m2( result6, gvl);
            vfloat32m2_t result7_32 = __riscv_vfwcvt_f_f_v_f32m2( result7, gvl);

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
                
		vfloat16m1_t mul_result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
                vfloat16m1_t mul_result1 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
                vfloat16m1_t mul_result2 = __riscv_vfmul_vf_f16m1( A0, B2, gvl);
                vfloat16m1_t mul_result3 = __riscv_vfmul_vf_f16m1( A0, B3, gvl);
                vfloat16m1_t mul_result4 = __riscv_vfmul_vf_f16m1( A0, B4, gvl);
                vfloat16m1_t mul_result5 = __riscv_vfmul_vf_f16m1( A0, B5, gvl);
                vfloat16m1_t mul_result6 = __riscv_vfmul_vf_f16m1( A0, B6, gvl);
                vfloat16m1_t mul_result7 = __riscv_vfmul_vf_f16m1( A0, B7, gvl);
                
                vfloat32m2_t mul_result0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result0, gvl);
                vfloat32m2_t mul_result1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result1, gvl);
                vfloat32m2_t mul_result2_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result2, gvl);
                vfloat32m2_t mul_result3_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result3, gvl);
                vfloat32m2_t mul_result4_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result4, gvl);
                vfloat32m2_t mul_result5_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result5, gvl);
                vfloat32m2_t mul_result6_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result6, gvl);
                vfloat32m2_t mul_result7_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result7, gvl);
                
                result0_32 = __riscv_vfadd_vv_f32m2( result0_32, mul_result0_32, gvl);
                result1_32 = __riscv_vfadd_vv_f32m2( result1_32, mul_result1_32, gvl);
                result2_32 = __riscv_vfadd_vv_f32m2( result2_32, mul_result2_32, gvl);
                result3_32 = __riscv_vfadd_vv_f32m2( result3_32, mul_result3_32, gvl);
                result4_32 = __riscv_vfadd_vv_f32m2( result4_32, mul_result4_32, gvl);
                result5_32 = __riscv_vfadd_vv_f32m2( result5_32, mul_result5_32, gvl);
                result6_32 = __riscv_vfadd_vv_f32m2( result6_32, mul_result6_32, gvl);
                result7_32 = __riscv_vfadd_vv_f32m2( result7_32, mul_result7_32, gvl);
            }
            
            result0 = __riscv_vfncvt_f_f_w_f16m1( result0_32, gvl);
            result1 = __riscv_vfncvt_f_f_w_f16m1( result1_32, gvl);
            result2 = __riscv_vfncvt_f_f_w_f16m1( result2_32, gvl);
            result3 = __riscv_vfncvt_f_f_w_f16m1( result3_32, gvl);
            result4 = __riscv_vfncvt_f_f_w_f16m1( result4_32, gvl);
            result5 = __riscv_vfncvt_f_f_w_f16m1( result5_32, gvl);
            result6 = __riscv_vfncvt_f_f_w_f16m1( result6_32, gvl);
            result7 = __riscv_vfncvt_f_f_w_f16m1( result7_32, gvl);

            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c2 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c3 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c4 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c5 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c6 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c7 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            
            vfloat32m2_t c0_32 = __riscv_vfwcvt_f_f_v_f32m2( c0, gvl);
            vfloat32m2_t c1_32 = __riscv_vfwcvt_f_f_v_f32m2( c1, gvl);
            vfloat32m2_t c2_32 = __riscv_vfwcvt_f_f_v_f32m2( c2, gvl);
            vfloat32m2_t c3_32 = __riscv_vfwcvt_f_f_v_f32m2( c3, gvl);
            vfloat32m2_t c4_32 = __riscv_vfwcvt_f_f_v_f32m2( c4, gvl);
            vfloat32m2_t c5_32 = __riscv_vfwcvt_f_f_v_f32m2( c5, gvl);
            vfloat32m2_t c6_32 = __riscv_vfwcvt_f_f_v_f32m2( c6, gvl);
            vfloat32m2_t c7_32 = __riscv_vfwcvt_f_f_v_f32m2( c7, gvl);
            
            vfloat16m1_t mul_c0 = __riscv_vfmul_vf_f16m1( result0, alpha, gvl);
            vfloat16m1_t mul_c1 = __riscv_vfmul_vf_f16m1( result1, alpha, gvl);
            vfloat16m1_t mul_c2 = __riscv_vfmul_vf_f16m1( result2, alpha, gvl);
            vfloat16m1_t mul_c3 = __riscv_vfmul_vf_f16m1( result3, alpha, gvl);
            vfloat16m1_t mul_c4 = __riscv_vfmul_vf_f16m1( result4, alpha, gvl);
            vfloat16m1_t mul_c5 = __riscv_vfmul_vf_f16m1( result5, alpha, gvl);
            vfloat16m1_t mul_c6 = __riscv_vfmul_vf_f16m1( result6, alpha, gvl);
            vfloat16m1_t mul_c7 = __riscv_vfmul_vf_f16m1( result7, alpha, gvl);
            
            vfloat32m2_t mul_c0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c0, gvl);
            vfloat32m2_t mul_c1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c1, gvl);
            vfloat32m2_t mul_c2_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c2, gvl);
            vfloat32m2_t mul_c3_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c3, gvl);
            vfloat32m2_t mul_c4_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c4, gvl);
            vfloat32m2_t mul_c5_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c5, gvl);
            vfloat32m2_t mul_c6_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c6, gvl);
            vfloat32m2_t mul_c7_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c7, gvl);
            
            c0_32 = __riscv_vfadd_vv_f32m2( c0_32, mul_c0_32, gvl);
            c1_32 = __riscv_vfadd_vv_f32m2( c1_32, mul_c1_32, gvl);
            c2_32 = __riscv_vfadd_vv_f32m2( c2_32, mul_c2_32, gvl);
            c3_32 = __riscv_vfadd_vv_f32m2( c3_32, mul_c3_32, gvl);
            c4_32 = __riscv_vfadd_vv_f32m2( c4_32, mul_c4_32, gvl);
            c5_32 = __riscv_vfadd_vv_f32m2( c5_32, mul_c5_32, gvl);
            c6_32 = __riscv_vfadd_vv_f32m2( c6_32, mul_c6_32, gvl);
            c7_32 = __riscv_vfadd_vv_f32m2( c7_32, mul_c7_32, gvl);
            
            c0 = __riscv_vfncvt_f_f_w_f16m1( c0_32, gvl);
            c1 = __riscv_vfncvt_f_f_w_f16m1( c1_32, gvl);
            c2 = __riscv_vfncvt_f_f_w_f16m1( c2_32, gvl);
            c3 = __riscv_vfncvt_f_f_w_f16m1( c3_32, gvl);
            c4 = __riscv_vfncvt_f_f_w_f16m1( c4_32, gvl);
            c5 = __riscv_vfncvt_f_f_w_f16m1( c5_32, gvl);
            c6 = __riscv_vfncvt_f_f_w_f16m1( c6_32, gvl);
            c7 = __riscv_vfncvt_f_f_w_f16m1( c7_32, gvl);

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
            
            vfloat32m2_t result0_32 = __riscv_vfwcvt_f_f_v_f32m2( result0, gvl);
            vfloat32m2_t result1_32 = __riscv_vfwcvt_f_f_v_f32m2( result1, gvl);
            vfloat32m2_t result2_32 = __riscv_vfwcvt_f_f_v_f32m2( result2, gvl);
            vfloat32m2_t result3_32 = __riscv_vfwcvt_f_f_v_f32m2( result3, gvl);
            vfloat32m2_t result4_32 = __riscv_vfwcvt_f_f_v_f32m2( result4, gvl);
            vfloat32m2_t result5_32 = __riscv_vfwcvt_f_f_v_f32m2( result5, gvl);
            vfloat32m2_t result6_32 = __riscv_vfwcvt_f_f_v_f32m2( result6, gvl);
            vfloat32m2_t result7_32 = __riscv_vfwcvt_f_f_v_f32m2( result7, gvl);

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
                
		vfloat16m1_t mul_result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
                vfloat16m1_t mul_result1 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
                vfloat16m1_t mul_result2 = __riscv_vfmul_vf_f16m1( A0, B2, gvl);
                vfloat16m1_t mul_result3 = __riscv_vfmul_vf_f16m1( A0, B3, gvl);
                vfloat16m1_t mul_result4 = __riscv_vfmul_vf_f16m1( A0, B4, gvl);
                vfloat16m1_t mul_result5 = __riscv_vfmul_vf_f16m1( A0, B5, gvl);
                vfloat16m1_t mul_result6 = __riscv_vfmul_vf_f16m1( A0, B6, gvl);
                vfloat16m1_t mul_result7 = __riscv_vfmul_vf_f16m1( A0, B7, gvl);
                
                vfloat32m2_t mul_result0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result0, gvl);
                vfloat32m2_t mul_result1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result1, gvl);
                vfloat32m2_t mul_result2_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result2, gvl);
                vfloat32m2_t mul_result3_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result3, gvl);
                vfloat32m2_t mul_result4_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result4, gvl);
                vfloat32m2_t mul_result5_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result5, gvl);
                vfloat32m2_t mul_result6_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result6, gvl);
                vfloat32m2_t mul_result7_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result7, gvl);
                
                result0_32 = __riscv_vfadd_vv_f32m2( result0_32, mul_result0_32, gvl);
                result1_32 = __riscv_vfadd_vv_f32m2( result1_32, mul_result1_32, gvl);
                result2_32 = __riscv_vfadd_vv_f32m2( result2_32, mul_result2_32, gvl);
                result3_32 = __riscv_vfadd_vv_f32m2( result3_32, mul_result3_32, gvl);
                result4_32 = __riscv_vfadd_vv_f32m2( result4_32, mul_result4_32, gvl);
                result5_32 = __riscv_vfadd_vv_f32m2( result5_32, mul_result5_32, gvl);
                result6_32 = __riscv_vfadd_vv_f32m2( result6_32, mul_result6_32, gvl);
                result7_32 = __riscv_vfadd_vv_f32m2( result7_32, mul_result7_32, gvl);
            }

            result0 = __riscv_vfncvt_f_f_w_f16m1( result0_32, gvl);
            result1 = __riscv_vfncvt_f_f_w_f16m1( result1_32, gvl);
            result2 = __riscv_vfncvt_f_f_w_f16m1( result2_32, gvl);
            result3 = __riscv_vfncvt_f_f_w_f16m1( result3_32, gvl);
            result4 = __riscv_vfncvt_f_f_w_f16m1( result4_32, gvl);
            result5 = __riscv_vfncvt_f_f_w_f16m1( result5_32, gvl);
            result6 = __riscv_vfncvt_f_f_w_f16m1( result6_32, gvl);
            result7 = __riscv_vfncvt_f_f_w_f16m1( result7_32, gvl);

            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c2 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c3 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c4 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c5 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c6 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c7 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            
            vfloat32m2_t c0_32 = __riscv_vfwcvt_f_f_v_f32m2( c0, gvl);
            vfloat32m2_t c1_32 = __riscv_vfwcvt_f_f_v_f32m2( c1, gvl);
            vfloat32m2_t c2_32 = __riscv_vfwcvt_f_f_v_f32m2( c2, gvl);
            vfloat32m2_t c3_32 = __riscv_vfwcvt_f_f_v_f32m2( c3, gvl);
            vfloat32m2_t c4_32 = __riscv_vfwcvt_f_f_v_f32m2( c4, gvl);
            vfloat32m2_t c5_32 = __riscv_vfwcvt_f_f_v_f32m2( c5, gvl);
            vfloat32m2_t c6_32 = __riscv_vfwcvt_f_f_v_f32m2( c6, gvl);
            vfloat32m2_t c7_32 = __riscv_vfwcvt_f_f_v_f32m2( c7, gvl);
            
            vfloat16m1_t mul_c0 = __riscv_vfmul_vf_f16m1( result0, alpha, gvl);
            vfloat16m1_t mul_c1 = __riscv_vfmul_vf_f16m1( result1, alpha, gvl);
            vfloat16m1_t mul_c2 = __riscv_vfmul_vf_f16m1( result2, alpha, gvl);
            vfloat16m1_t mul_c3 = __riscv_vfmul_vf_f16m1( result3, alpha, gvl);
            vfloat16m1_t mul_c4 = __riscv_vfmul_vf_f16m1( result4, alpha, gvl);
            vfloat16m1_t mul_c5 = __riscv_vfmul_vf_f16m1( result5, alpha, gvl);
            vfloat16m1_t mul_c6 = __riscv_vfmul_vf_f16m1( result6, alpha, gvl);
            vfloat16m1_t mul_c7 = __riscv_vfmul_vf_f16m1( result7, alpha, gvl);
            
            vfloat32m2_t mul_c0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c0, gvl);
            vfloat32m2_t mul_c1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c1, gvl);
            vfloat32m2_t mul_c2_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c2, gvl);
            vfloat32m2_t mul_c3_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c3, gvl);
            vfloat32m2_t mul_c4_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c4, gvl);
            vfloat32m2_t mul_c5_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c5, gvl);
            vfloat32m2_t mul_c6_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c6, gvl);
            vfloat32m2_t mul_c7_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c7, gvl);
            
            c0_32 = __riscv_vfadd_vv_f32m2( c0_32, mul_c0_32, gvl);
            c1_32 = __riscv_vfadd_vv_f32m2( c1_32, mul_c1_32, gvl);
            c2_32 = __riscv_vfadd_vv_f32m2( c2_32, mul_c2_32, gvl);
            c3_32 = __riscv_vfadd_vv_f32m2( c3_32, mul_c3_32, gvl);
            c4_32 = __riscv_vfadd_vv_f32m2( c4_32, mul_c4_32, gvl);
            c5_32 = __riscv_vfadd_vv_f32m2( c5_32, mul_c5_32, gvl);
            c6_32 = __riscv_vfadd_vv_f32m2( c6_32, mul_c6_32, gvl);
            c7_32 = __riscv_vfadd_vv_f32m2( c7_32, mul_c7_32, gvl);
            
            c0 = __riscv_vfncvt_f_f_w_f16m1( c0_32, gvl);
            c1 = __riscv_vfncvt_f_f_w_f16m1( c1_32, gvl);
            c2 = __riscv_vfncvt_f_f_w_f16m1( c2_32, gvl);
            c3 = __riscv_vfncvt_f_f_w_f16m1( c3_32, gvl);
            c4 = __riscv_vfncvt_f_f_w_f16m1( c4_32, gvl);
            c5 = __riscv_vfncvt_f_f_w_f16m1( c5_32, gvl);
            c6 = __riscv_vfncvt_f_f_w_f16m1( c6_32, gvl);
            c7 = __riscv_vfncvt_f_f_w_f16m1( c7_32, gvl);

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

            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;

            float result0_32 = 0;
            float result1_32 = 0;
            float result2_32 = 0;
            float result3_32 = 0;
            float result4_32 = 0;
            float result5_32 = 0;
            float result6_32 = 0;
            float result7_32 = 0;
            float result8_32 = 0;
            float result9_32 = 0;
            float result10_32 = 0;
            float result11_32 = 0;
            float result12_32 = 0;
            float result13_32 = 0;
            float result14_32 = 0;
            float result15_32 = 0;

            for(unsigned int k=0; k<K; k++) {
                result0_32+=(float)(A[ai+0]*B[bi+0]);
                result1_32+=(float)(A[ai+1]*B[bi+0]);
                result2_32+=(float)(A[ai+0]*B[bi+1]);
                result3_32+=(float)(A[ai+1]*B[bi+1]);
                result4_32+=(float)(A[ai+0]*B[bi+2]);
                result5_32+=(float)(A[ai+1]*B[bi+2]);
                result6_32+=(float)(A[ai+0]*B[bi+3]);
                result7_32+=(float)(A[ai+1]*B[bi+3]);
                result8_32+=(float)(A[ai+0]*B[bi+4]);
                result9_32+=(float)(A[ai+1]*B[bi+4]);
                result10_32+=(float)(A[ai+0]*B[bi+5]);
                result11_32+=(float)(A[ai+1]*B[bi+5]);
                result12_32+=(float)(A[ai+0]*B[bi+6]);
                result13_32+=(float)(A[ai+1]*B[bi+6]);
                result14_32+=(float)(A[ai+0]*B[bi+7]);
                result15_32+=(float)(A[ai+1]*B[bi+7]);
                ai+=2;
                bi+=8;
            }
            
            _Float16 result0 = (_Float16)result0_32;
            _Float16 result1 = (_Float16)result1_32;
            _Float16 result2 = (_Float16)result2_32;
            _Float16 result3 = (_Float16)result3_32;
            _Float16 result4 = (_Float16)result4_32;
            _Float16 result5 = (_Float16)result5_32;
            _Float16 result6 = (_Float16)result6_32;
            _Float16 result7 = (_Float16)result7_32;
            _Float16 result8 = (_Float16)result8_32;
            _Float16 result9 = (_Float16)result9_32;
            _Float16 result10 = (_Float16)result10_32;
            _Float16 result11 = (_Float16)result11_32;
            _Float16 result12 = (_Float16)result12_32;
            _Float16 result13 = (_Float16)result13_32;
            _Float16 result14 = (_Float16)result14_32;
            _Float16 result15 = (_Float16)result15_32;
            
            unsigned int ci=n_top*ldc+m_top;
            
            C[ci+0*ldc+0] = (_Float16)((float)(alpha * result0) + (float)C[ci+0*ldc+0]);
            C[ci+0*ldc+1] = (_Float16)((float)(alpha * result1) + (float)C[ci+0*ldc+1]);
            C[ci+1*ldc+0] = (_Float16)((float)(alpha * result2) + (float)C[ci+1*ldc+0]);
            C[ci+1*ldc+1] = (_Float16)((float)(alpha * result3) + (float)C[ci+1*ldc+1]);
            C[ci+2*ldc+0] = (_Float16)((float)(alpha * result4) + (float)C[ci+2*ldc+0]);
            C[ci+2*ldc+1] = (_Float16)((float)(alpha * result5) + (float)C[ci+2*ldc+1]);
            C[ci+3*ldc+0] = (_Float16)((float)(alpha * result6) + (float)C[ci+3*ldc+0]);
            C[ci+3*ldc+1] = (_Float16)((float)(alpha * result7) + (float)C[ci+3*ldc+1]);
            C[ci+4*ldc+0] = (_Float16)((float)(alpha * result8) + (float)C[ci+4*ldc+0]);
            C[ci+4*ldc+1] = (_Float16)((float)(alpha * result9) + (float)C[ci+4*ldc+1]);
            C[ci+5*ldc+0] = (_Float16)((float)(alpha * result10) + (float)C[ci+5*ldc+0]);
            C[ci+5*ldc+1] = (_Float16)((float)(alpha * result11) + (float)C[ci+5*ldc+1]);
            C[ci+6*ldc+0] = (_Float16)((float)(alpha * result12) + (float)C[ci+6*ldc+0]);
            C[ci+6*ldc+1] = (_Float16)((float)(alpha * result13) + (float)C[ci+6*ldc+1]);
            C[ci+7*ldc+0] = (_Float16)((float)(alpha * result14) + (float)C[ci+7*ldc+0]);
            C[ci+7*ldc+1] = (_Float16)((float)(alpha * result15) + (float)C[ci+7*ldc+1]);
            m_top+=2;
        }


        if( M & 1 ) {
            
            float result0_32 = 0;
            float result1_32 = 0;
            float result2_32 = 0;
            float result3_32 = 0;
            float result4_32 = 0;
            float result5_32 = 0;
            float result6_32 = 0;
            float result7_32 = 0;
            
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;

            for(unsigned int k=0; k<K; k++) {
                result0_32+=(float)(A[ai+0]*B[bi+0]);
                result1_32+=(float)(A[ai+0]*B[bi+1]);
                result2_32+=(float)(A[ai+0]*B[bi+2]);
                result3_32+=(float)(A[ai+0]*B[bi+3]);
                result4_32+=(float)(A[ai+0]*B[bi+4]);
                result5_32+=(float)(A[ai+0]*B[bi+5]);
                result6_32+=(float)(A[ai+0]*B[bi+6]);
                result7_32+=(float)(A[ai+0]*B[bi+7]);
                ai+=1;
                bi+=8;
            }

            _Float16 result0 = (_Float16)result0_32;
            _Float16 result1 = (_Float16)result1_32;
            _Float16 result2 = (_Float16)result2_32;
            _Float16 result3 = (_Float16)result3_32;
            _Float16 result4 = (_Float16)result4_32;
            _Float16 result5 = (_Float16)result5_32;
            _Float16 result6 = (_Float16)result6_32;
            _Float16 result7 = (_Float16)result7_32;

            unsigned int ci=n_top*ldc+m_top;
            C[ci+0*ldc+0] = (_Float16)((float)(alpha * result0) + (float)C[ci+0*ldc+0]);
            C[ci+1*ldc+0] = (_Float16)((float)(alpha * result1) + (float)C[ci+1*ldc+0]);
            C[ci+2*ldc+0] = (_Float16)((float)(alpha * result2) + (float)C[ci+2*ldc+0]);
            C[ci+3*ldc+0] = (_Float16)((float)(alpha * result3) + (float)C[ci+3*ldc+0]);
            C[ci+4*ldc+0] = (_Float16)((float)(alpha * result4) + (float)C[ci+4*ldc+0]);
            C[ci+5*ldc+0] = (_Float16)((float)(alpha * result5) + (float)C[ci+5*ldc+0]);
            C[ci+6*ldc+0] = (_Float16)((float)(alpha * result6) + (float)C[ci+6*ldc+0]);
            C[ci+7*ldc+0] = (_Float16)((float)(alpha * result7) + (float)C[ci+7*ldc+0]);
            m_top+=1;
        }

        n_top += 8;
    }



    // -- tails for N=4
	// 处理N维度的剩余部分（如果N不是8的倍数）
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
            
            vfloat32m2_t result0_32 = __riscv_vfwcvt_f_f_v_f32m2( result0, gvl);
            vfloat32m2_t result1_32 = __riscv_vfwcvt_f_f_v_f32m2( result1, gvl);
            vfloat32m2_t result2_32 = __riscv_vfwcvt_f_f_v_f32m2( result2, gvl);
            vfloat32m2_t result3_32 = __riscv_vfwcvt_f_f_v_f32m2( result3, gvl);
            vfloat32m2_t result4_32 = __riscv_vfwcvt_f_f_v_f32m2( result4, gvl);
            vfloat32m2_t result5_32 = __riscv_vfwcvt_f_f_v_f32m2( result5, gvl);
            vfloat32m2_t result6_32 = __riscv_vfwcvt_f_f_v_f32m2( result6, gvl);
            vfloat32m2_t result7_32 = __riscv_vfwcvt_f_f_v_f32m2( result7, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                B2 = B[bi+2];
                B3 = B[bi+3];
                bi += 4;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                A1 = __riscv_vle16_v_f16m1( &A[ai+1*gvl], gvl );
                ai += 16;
                
		vfloat16m1_t mul_result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
                vfloat16m1_t mul_result1 = __riscv_vfmul_vf_f16m1( A1, B0, gvl);
                vfloat16m1_t mul_result2 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
                vfloat16m1_t mul_result3 = __riscv_vfmul_vf_f16m1( A1, B1, gvl);
                vfloat16m1_t mul_result4 = __riscv_vfmul_vf_f16m1( A0, B2, gvl);
                vfloat16m1_t mul_result5 = __riscv_vfmul_vf_f16m1( A1, B2, gvl);
                vfloat16m1_t mul_result6 = __riscv_vfmul_vf_f16m1( A0, B3, gvl);
                vfloat16m1_t mul_result7 = __riscv_vfmul_vf_f16m1( A1, B3, gvl);
                
                vfloat32m2_t mul_result0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result0, gvl);
                vfloat32m2_t mul_result1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result1, gvl);
                vfloat32m2_t mul_result2_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result2, gvl);
                vfloat32m2_t mul_result3_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result3, gvl);
                vfloat32m2_t mul_result4_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result4, gvl);
                vfloat32m2_t mul_result5_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result5, gvl);
                vfloat32m2_t mul_result6_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result6, gvl);
                vfloat32m2_t mul_result7_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result7, gvl);
                
                result0_32 = __riscv_vfadd_vv_f32m2( result0_32, mul_result0_32, gvl);
                result1_32 = __riscv_vfadd_vv_f32m2( result1_32, mul_result1_32, gvl);
                result2_32 = __riscv_vfadd_vv_f32m2( result2_32, mul_result2_32, gvl);
                result3_32 = __riscv_vfadd_vv_f32m2( result3_32, mul_result3_32, gvl);
                result4_32 = __riscv_vfadd_vv_f32m2( result4_32, mul_result4_32, gvl);
                result5_32 = __riscv_vfadd_vv_f32m2( result5_32, mul_result5_32, gvl);
                result6_32 = __riscv_vfadd_vv_f32m2( result6_32, mul_result6_32, gvl);
                result7_32 = __riscv_vfadd_vv_f32m2( result7_32, mul_result7_32, gvl);
                
            }
            
            result0 = __riscv_vfncvt_f_f_w_f16m1( result0_32, gvl);
            result1 = __riscv_vfncvt_f_f_w_f16m1( result1_32, gvl);
            result2 = __riscv_vfncvt_f_f_w_f16m1( result2_32, gvl);
            result3 = __riscv_vfncvt_f_f_w_f16m1( result3_32, gvl);
            result4 = __riscv_vfncvt_f_f_w_f16m1( result4_32, gvl);
            result5 = __riscv_vfncvt_f_f_w_f16m1( result5_32, gvl);
            result6 = __riscv_vfncvt_f_f_w_f16m1( result6_32, gvl);
            result7 = __riscv_vfncvt_f_f_w_f16m1( result7_32, gvl);

            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c2 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c3 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c4 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c5 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c6 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c7 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            
            vfloat32m2_t c0_32 = __riscv_vfwcvt_f_f_v_f32m2( c0, gvl);
            vfloat32m2_t c1_32 = __riscv_vfwcvt_f_f_v_f32m2( c1, gvl);
            vfloat32m2_t c2_32 = __riscv_vfwcvt_f_f_v_f32m2( c2, gvl);
            vfloat32m2_t c3_32 = __riscv_vfwcvt_f_f_v_f32m2( c3, gvl);
            vfloat32m2_t c4_32 = __riscv_vfwcvt_f_f_v_f32m2( c4, gvl);
            vfloat32m2_t c5_32 = __riscv_vfwcvt_f_f_v_f32m2( c5, gvl);
            vfloat32m2_t c6_32 = __riscv_vfwcvt_f_f_v_f32m2( c6, gvl);
            vfloat32m2_t c7_32 = __riscv_vfwcvt_f_f_v_f32m2( c7, gvl);
            
            vfloat16m1_t mul_c0 = __riscv_vfmul_vf_f16m1( result0, alpha, gvl);
            vfloat16m1_t mul_c1 = __riscv_vfmul_vf_f16m1( result1, alpha, gvl);
            vfloat16m1_t mul_c2 = __riscv_vfmul_vf_f16m1( result2, alpha, gvl);
            vfloat16m1_t mul_c3 = __riscv_vfmul_vf_f16m1( result3, alpha, gvl);
            vfloat16m1_t mul_c4 = __riscv_vfmul_vf_f16m1( result4, alpha, gvl);
            vfloat16m1_t mul_c5 = __riscv_vfmul_vf_f16m1( result5, alpha, gvl);
            vfloat16m1_t mul_c6 = __riscv_vfmul_vf_f16m1( result6, alpha, gvl);
            vfloat16m1_t mul_c7 = __riscv_vfmul_vf_f16m1( result7, alpha, gvl);
            
            vfloat32m2_t mul_c0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c0, gvl);
            vfloat32m2_t mul_c1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c1, gvl);
            vfloat32m2_t mul_c2_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c2, gvl);
            vfloat32m2_t mul_c3_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c3, gvl);
            vfloat32m2_t mul_c4_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c4, gvl);
            vfloat32m2_t mul_c5_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c5, gvl);
            vfloat32m2_t mul_c6_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c6, gvl);
            vfloat32m2_t mul_c7_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c7, gvl);
            
            c0_32 = __riscv_vfadd_vv_f32m2( c0_32, mul_c0_32, gvl);
            c1_32 = __riscv_vfadd_vv_f32m2( c1_32, mul_c1_32, gvl);
            c2_32 = __riscv_vfadd_vv_f32m2( c2_32, mul_c2_32, gvl);
            c3_32 = __riscv_vfadd_vv_f32m2( c3_32, mul_c3_32, gvl);
            c4_32 = __riscv_vfadd_vv_f32m2( c4_32, mul_c4_32, gvl);
            c5_32 = __riscv_vfadd_vv_f32m2( c5_32, mul_c5_32, gvl);
            c6_32 = __riscv_vfadd_vv_f32m2( c6_32, mul_c6_32, gvl);
            c7_32 = __riscv_vfadd_vv_f32m2( c7_32, mul_c7_32, gvl);
            
            c0 = __riscv_vfncvt_f_f_w_f16m1( c0_32, gvl);
            c1 = __riscv_vfncvt_f_f_w_f16m1( c1_32, gvl);
            c2 = __riscv_vfncvt_f_f_w_f16m1( c2_32, gvl);
            c3 = __riscv_vfncvt_f_f_w_f16m1( c3_32, gvl);
            c4 = __riscv_vfncvt_f_f_w_f16m1( c4_32, gvl);
            c5 = __riscv_vfncvt_f_f_w_f16m1( c5_32, gvl);
            c6 = __riscv_vfncvt_f_f_w_f16m1( c6_32, gvl);
            c7 = __riscv_vfncvt_f_f_w_f16m1( c7_32, gvl);

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
            
            vfloat32m2_t result0_32 = __riscv_vfwcvt_f_f_v_f32m2( result0, gvl);
            vfloat32m2_t result1_32 = __riscv_vfwcvt_f_f_v_f32m2( result1, gvl);
            vfloat32m2_t result2_32 = __riscv_vfwcvt_f_f_v_f32m2( result2, gvl);
            vfloat32m2_t result3_32 = __riscv_vfwcvt_f_f_v_f32m2( result3, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                B2 = B[bi+2];
                B3 = B[bi+3];
                bi += 4;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                ai += 8;
                
		vfloat16m1_t mul_result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
                vfloat16m1_t mul_result1 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
                vfloat16m1_t mul_result2 = __riscv_vfmul_vf_f16m1( A0, B2, gvl);
                vfloat16m1_t mul_result3 = __riscv_vfmul_vf_f16m1( A0, B3, gvl);
                
                vfloat32m2_t mul_result0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result0, gvl);
                vfloat32m2_t mul_result1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result1, gvl);
                vfloat32m2_t mul_result2_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result2, gvl);
                vfloat32m2_t mul_result3_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result3, gvl);
                
                result0_32 = __riscv_vfadd_vv_f32m2( result0_32, mul_result0_32, gvl);
                result1_32 = __riscv_vfadd_vv_f32m2( result1_32, mul_result1_32, gvl);
                result2_32 = __riscv_vfadd_vv_f32m2( result2_32, mul_result2_32, gvl);
                result3_32 = __riscv_vfadd_vv_f32m2( result3_32, mul_result3_32, gvl);
                
            }

            result0 = __riscv_vfncvt_f_f_w_f16m1( result0_32, gvl);
            result1 = __riscv_vfncvt_f_f_w_f16m1( result1_32, gvl);
            result2 = __riscv_vfncvt_f_f_w_f16m1( result2_32, gvl);
            result3 = __riscv_vfncvt_f_f_w_f16m1( result3_32, gvl);

            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c2 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c3 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            
            vfloat32m2_t c0_32 = __riscv_vfwcvt_f_f_v_f32m2( c0, gvl);
            vfloat32m2_t c1_32 = __riscv_vfwcvt_f_f_v_f32m2( c1, gvl);
            vfloat32m2_t c2_32 = __riscv_vfwcvt_f_f_v_f32m2( c2, gvl);
            vfloat32m2_t c3_32 = __riscv_vfwcvt_f_f_v_f32m2( c3, gvl);
            
            vfloat16m1_t mul_c0 = __riscv_vfmul_vf_f16m1( result0, alpha, gvl);
            vfloat16m1_t mul_c1 = __riscv_vfmul_vf_f16m1( result1, alpha, gvl);
            vfloat16m1_t mul_c2 = __riscv_vfmul_vf_f16m1( result2, alpha, gvl);
            vfloat16m1_t mul_c3 = __riscv_vfmul_vf_f16m1( result3, alpha, gvl);
            
            vfloat32m2_t mul_c0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c0, gvl);
            vfloat32m2_t mul_c1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c1, gvl);
            vfloat32m2_t mul_c2_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c2, gvl);
            vfloat32m2_t mul_c3_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c3, gvl);
            
            c0_32 = __riscv_vfadd_vv_f32m2( c0_32, mul_c0_32, gvl);
            c1_32 = __riscv_vfadd_vv_f32m2( c1_32, mul_c1_32, gvl);
            c2_32 = __riscv_vfadd_vv_f32m2( c2_32, mul_c2_32, gvl);
            c3_32 = __riscv_vfadd_vv_f32m2( c3_32, mul_c3_32, gvl);
            
            c0 = __riscv_vfncvt_f_f_w_f16m1( c0_32, gvl);
            c1 = __riscv_vfncvt_f_f_w_f16m1( c1_32, gvl);
            c2 = __riscv_vfncvt_f_f_w_f16m1( c2_32, gvl);
            c3 = __riscv_vfncvt_f_f_w_f16m1( c3_32, gvl);

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
            
            vfloat32m2_t result0_32 = __riscv_vfwcvt_f_f_v_f32m2( result0, gvl);
            vfloat32m2_t result1_32 = __riscv_vfwcvt_f_f_v_f32m2( result1, gvl);
            vfloat32m2_t result2_32 = __riscv_vfwcvt_f_f_v_f32m2( result2, gvl);
            vfloat32m2_t result3_32 = __riscv_vfwcvt_f_f_v_f32m2( result3, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                B2 = B[bi+2];
                B3 = B[bi+3];
                bi += 4;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                ai += 4;

		vfloat16m1_t mul_result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
                vfloat16m1_t mul_result1 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
                vfloat16m1_t mul_result2 = __riscv_vfmul_vf_f16m1( A0, B2, gvl);
                vfloat16m1_t mul_result3 = __riscv_vfmul_vf_f16m1( A0, B3, gvl);
                
                vfloat32m2_t mul_result0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result0, gvl);
                vfloat32m2_t mul_result1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result1, gvl);
                vfloat32m2_t mul_result2_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result2, gvl);
                vfloat32m2_t mul_result3_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result3, gvl);
                
                result0_32 = __riscv_vfadd_vv_f32m2( result0_32, mul_result0_32, gvl);
                result1_32 = __riscv_vfadd_vv_f32m2( result1_32, mul_result1_32, gvl);
                result2_32 = __riscv_vfadd_vv_f32m2( result2_32, mul_result2_32, gvl);
                result3_32 = __riscv_vfadd_vv_f32m2( result3_32, mul_result3_32, gvl);
                
            }

            result0 = __riscv_vfncvt_f_f_w_f16m1( result0_32, gvl);
            result1 = __riscv_vfncvt_f_f_w_f16m1( result1_32, gvl);
            result2 = __riscv_vfncvt_f_f_w_f16m1( result2_32, gvl);
            result3 = __riscv_vfncvt_f_f_w_f16m1( result3_32, gvl);

            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c2 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c3 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            
            vfloat32m2_t c0_32 = __riscv_vfwcvt_f_f_v_f32m2( c0, gvl);
            vfloat32m2_t c1_32 = __riscv_vfwcvt_f_f_v_f32m2( c1, gvl);
            vfloat32m2_t c2_32 = __riscv_vfwcvt_f_f_v_f32m2( c2, gvl);
            vfloat32m2_t c3_32 = __riscv_vfwcvt_f_f_v_f32m2( c3, gvl);
            
            vfloat16m1_t mul_c0 = __riscv_vfmul_vf_f16m1( result0, alpha, gvl);
            vfloat16m1_t mul_c1 = __riscv_vfmul_vf_f16m1( result1, alpha, gvl);
            vfloat16m1_t mul_c2 = __riscv_vfmul_vf_f16m1( result2, alpha, gvl);
            vfloat16m1_t mul_c3 = __riscv_vfmul_vf_f16m1( result3, alpha, gvl);
            
            vfloat32m2_t mul_c0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c0, gvl);
            vfloat32m2_t mul_c1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c1, gvl);
            vfloat32m2_t mul_c2_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c2, gvl);
            vfloat32m2_t mul_c3_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c3, gvl);

            c0_32 = __riscv_vfadd_vv_f32m2( c0_32, mul_c0_32, gvl);
            c1_32 = __riscv_vfadd_vv_f32m2( c1_32, mul_c1_32, gvl);
            c2_32 = __riscv_vfadd_vv_f32m2( c2_32, mul_c2_32, gvl);
            c3_32 = __riscv_vfadd_vv_f32m2( c3_32, mul_c3_32, gvl);
            
            c0 = __riscv_vfncvt_f_f_w_f16m1( c0_32, gvl);
            c1 = __riscv_vfncvt_f_f_w_f16m1( c1_32, gvl);
            c2 = __riscv_vfncvt_f_f_w_f16m1( c2_32, gvl);
            c3 = __riscv_vfncvt_f_f_w_f16m1( c3_32, gvl);

            ci=n_top*ldc+m_top;

            __riscv_vse16_v_f16m1( &C[ci], c0, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c1, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c2, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c3, gvl);
            m_top += 4;
        }


        if( M & 2 ) {
            
            float result0_32 = 0;
            float result1_32 = 0;
            float result2_32 = 0;
            float result3_32 = 0;
            float result4_32 = 0;
            float result5_32 = 0;
            float result6_32 = 0;
            float result7_32 = 0;
            
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;

            for(unsigned int k=0; k<K; k++) {
                result0_32+=(float)(A[ai+0]*B[bi+0]);
                result1_32+=(float)(A[ai+1]*B[bi+0]);
                result2_32+=(float)(A[ai+0]*B[bi+1]);
                result3_32+=(float)(A[ai+1]*B[bi+1]);
                result4_32+=(float)(A[ai+0]*B[bi+2]);
                result5_32+=(float)(A[ai+1]*B[bi+2]);
                result6_32+=(float)(A[ai+0]*B[bi+3]);
                result7_32+=(float)(A[ai+1]*B[bi+3]);
                ai+=2;
                bi+=4;
            }

	    _Float16 result0 = (_Float16)result0_32;
	    _Float16 result1 = (_Float16)result1_32;
	    _Float16 result2 = (_Float16)result2_32;
	    _Float16 result3 = (_Float16)result3_32;
	    _Float16 result4 = (_Float16)result4_32;
	    _Float16 result5 = (_Float16)result5_32;
	    _Float16 result6 = (_Float16)result6_32;
	    _Float16 result7 = (_Float16)result7_32;

            unsigned int ci=n_top*ldc+m_top;
            
            C[ci+0*ldc+0] = (_Float16)((float)(alpha * result0) + (float)C[ci+0*ldc+0]);
            C[ci+0*ldc+1] = (_Float16)((float)(alpha * result1) + (float)C[ci+0*ldc+1]);
            C[ci+1*ldc+0] = (_Float16)((float)(alpha * result2) + (float)C[ci+1*ldc+0]);
            C[ci+1*ldc+1] = (_Float16)((float)(alpha * result3) + (float)C[ci+1*ldc+1]);
            C[ci+2*ldc+0] = (_Float16)((float)(alpha * result4) + (float)C[ci+2*ldc+0]);
            C[ci+2*ldc+1] = (_Float16)((float)(alpha * result5) + (float)C[ci+2*ldc+1]);
            C[ci+3*ldc+0] = (_Float16)((float)(alpha * result6) + (float)C[ci+3*ldc+0]);
            C[ci+3*ldc+1] = (_Float16)((float)(alpha * result7) + (float)C[ci+3*ldc+1]);

            m_top+=2;
        }


        if( M & 1 ) {
        
            float result0_32 = 0;
            float result1_32 = 0;
            float result2_32 = 0;
            float result3_32 = 0;
            
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;

            for(unsigned int k=0; k<K; k++) {
                result0_32+=(float)(A[ai+0]*B[bi+0]);
                result1_32+=(float)(A[ai+0]*B[bi+1]);
                result2_32+=(float)(A[ai+0]*B[bi+2]);
                result3_32+=(float)(A[ai+0]*B[bi+3]);
                ai+=1;
                bi+=4;
            }
            
            _Float16 result0 = (_Float16)result0_32;
	    _Float16 result1 = (_Float16)result1_32;
	    _Float16 result2 = (_Float16)result2_32;
	    _Float16 result3 = (_Float16)result3_32;

            unsigned int ci=n_top*ldc+m_top;
            
            C[ci+0*ldc+0] = (_Float16)((float)(alpha * result0) + (float)C[ci+0*ldc+0]);
            C[ci+1*ldc+0] = (_Float16)((float)(alpha * result1) + (float)C[ci+1*ldc+0]);
            C[ci+2*ldc+0] = (_Float16)((float)(alpha * result2) + (float)C[ci+2*ldc+0]);
            C[ci+3*ldc+0] = (_Float16)((float)(alpha * result3) + (float)C[ci+3*ldc+0]);
            
            m_top+=1;
        }

        n_top += 4;
    }



    // -- tails for N=2

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
            
            vfloat32m2_t result0_32 = __riscv_vfwcvt_f_f_v_f32m2( result0, gvl);
            vfloat32m2_t result1_32 = __riscv_vfwcvt_f_f_v_f32m2( result1, gvl);
            vfloat32m2_t result2_32 = __riscv_vfwcvt_f_f_v_f32m2( result2, gvl);
            vfloat32m2_t result3_32 = __riscv_vfwcvt_f_f_v_f32m2( result3, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                bi += 2;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                A1 = __riscv_vle16_v_f16m1( &A[ai+1*gvl], gvl );
                ai += 16;

		vfloat16m1_t mul_result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
                vfloat16m1_t mul_result1 = __riscv_vfmul_vf_f16m1( A1, B0, gvl);
                vfloat16m1_t mul_result2 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
                vfloat16m1_t mul_result3 = __riscv_vfmul_vf_f16m1( A1, B1, gvl);
                
                vfloat32m2_t mul_result0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result0, gvl);
                vfloat32m2_t mul_result1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result1, gvl);
                vfloat32m2_t mul_result2_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result2, gvl);
                vfloat32m2_t mul_result3_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result3, gvl);
                
                result0_32 = __riscv_vfadd_vv_f32m2( result0_32, mul_result0_32, gvl);
                result1_32 = __riscv_vfadd_vv_f32m2( result1_32, mul_result1_32, gvl);
                result2_32 = __riscv_vfadd_vv_f32m2( result2_32, mul_result2_32, gvl);
                result3_32 = __riscv_vfadd_vv_f32m2( result3_32, mul_result3_32, gvl);
                
            }

            result0 = __riscv_vfncvt_f_f_w_f16m1( result0_32, gvl);
            result1 = __riscv_vfncvt_f_f_w_f16m1( result1_32, gvl);
            result2 = __riscv_vfncvt_f_f_w_f16m1( result2_32, gvl);
            result3 = __riscv_vfncvt_f_f_w_f16m1( result3_32, gvl);

            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*1;
            vfloat16m1_t c2 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c3 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            
            vfloat32m2_t c0_32 = __riscv_vfwcvt_f_f_v_f32m2( c0, gvl);
            vfloat32m2_t c1_32 = __riscv_vfwcvt_f_f_v_f32m2( c1, gvl);
            vfloat32m2_t c2_32 = __riscv_vfwcvt_f_f_v_f32m2( c2, gvl);
            vfloat32m2_t c3_32 = __riscv_vfwcvt_f_f_v_f32m2( c3, gvl);
            
            vfloat16m1_t mul_c0 = __riscv_vfmul_vf_f16m1( result0, alpha, gvl);
            vfloat16m1_t mul_c1 = __riscv_vfmul_vf_f16m1( result1, alpha, gvl);
            vfloat16m1_t mul_c2 = __riscv_vfmul_vf_f16m1( result2, alpha, gvl);
            vfloat16m1_t mul_c3 = __riscv_vfmul_vf_f16m1( result3, alpha, gvl);
            
            vfloat32m2_t mul_c0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c0, gvl);
            vfloat32m2_t mul_c1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c1, gvl);
            vfloat32m2_t mul_c2_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c2, gvl);
            vfloat32m2_t mul_c3_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c3, gvl);
            
            c0_32 = __riscv_vfadd_vv_f32m2( c0_32, mul_c0_32, gvl);
            c1_32 = __riscv_vfadd_vv_f32m2( c1_32, mul_c1_32, gvl);
            c2_32 = __riscv_vfadd_vv_f32m2( c2_32, mul_c2_32, gvl);
            c3_32 = __riscv_vfadd_vv_f32m2( c3_32, mul_c3_32, gvl);
            
            c0 = __riscv_vfncvt_f_f_w_f16m1( c0_32, gvl);
            c1 = __riscv_vfncvt_f_f_w_f16m1( c1_32, gvl);
            c2 = __riscv_vfncvt_f_f_w_f16m1( c2_32, gvl);
            c3 = __riscv_vfncvt_f_f_w_f16m1( c3_32, gvl);

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
            
            vfloat32m2_t result0_32 = __riscv_vfwcvt_f_f_v_f32m2( result0, gvl);
            vfloat32m2_t result1_32 = __riscv_vfwcvt_f_f_v_f32m2( result1, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                bi += 2;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                ai += 8;

		vfloat16m1_t mul_result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
                vfloat16m1_t mul_result1 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
                
                vfloat32m2_t mul_result0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result0, gvl);
                vfloat32m2_t mul_result1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result1, gvl);
                
                result0_32 = __riscv_vfadd_vv_f32m2( result0_32, mul_result0_32, gvl);
                result1_32 = __riscv_vfadd_vv_f32m2( result1_32, mul_result1_32, gvl);
            }

            result0 = __riscv_vfncvt_f_f_w_f16m1( result0_32, gvl);
            result1 = __riscv_vfncvt_f_f_w_f16m1( result1_32, gvl);

            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            
            vfloat32m2_t c0_32 = __riscv_vfwcvt_f_f_v_f32m2( c0, gvl);
            vfloat32m2_t c1_32 = __riscv_vfwcvt_f_f_v_f32m2( c1, gvl);
            
            vfloat16m1_t mul_c0 = __riscv_vfmul_vf_f16m1( result0, alpha, gvl);
            vfloat16m1_t mul_c1 = __riscv_vfmul_vf_f16m1( result1, alpha, gvl);
            
            vfloat32m2_t mul_c0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c0, gvl);
            vfloat32m2_t mul_c1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c1, gvl);
            
            c0_32 = __riscv_vfadd_vv_f32m2( c0_32, mul_c0_32, gvl);
            c1_32 = __riscv_vfadd_vv_f32m2( c1_32, mul_c1_32, gvl);
            
            c0 = __riscv_vfncvt_f_f_w_f16m1( c0_32, gvl);
            c1 = __riscv_vfncvt_f_f_w_f16m1( c1_32, gvl);

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
            
            vfloat32m2_t result0_32 = __riscv_vfwcvt_f_f_v_f32m2( result0, gvl);
            vfloat32m2_t result1_32 = __riscv_vfwcvt_f_f_v_f32m2( result1, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                B1 = B[bi+1];
                bi += 2;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                ai += 4;

		vfloat16m1_t mul_result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
                vfloat16m1_t mul_result1 = __riscv_vfmul_vf_f16m1( A0, B1, gvl);
                
                vfloat32m2_t mul_result0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result0, gvl);
                vfloat32m2_t mul_result1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result1, gvl);
                
                result0_32 = __riscv_vfadd_vv_f32m2( result0_32, mul_result0_32, gvl);
                result1_32 = __riscv_vfadd_vv_f32m2( result1_32, mul_result1_32, gvl);
                
            }

            result0 = __riscv_vfncvt_f_f_w_f16m1( result0_32, gvl);
            result1 = __riscv_vfncvt_f_f_w_f16m1( result1_32, gvl);

            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += ldc-gvl*0;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            
            vfloat32m2_t c0_32 = __riscv_vfwcvt_f_f_v_f32m2( c0, gvl);
            vfloat32m2_t c1_32 = __riscv_vfwcvt_f_f_v_f32m2( c1, gvl);
            
            vfloat16m1_t mul_c0 = __riscv_vfmul_vf_f16m1( result0, alpha, gvl);
            vfloat16m1_t mul_c1 = __riscv_vfmul_vf_f16m1( result1, alpha, gvl);
            
            vfloat32m2_t mul_c0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c0, gvl);
            vfloat32m2_t mul_c1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c1, gvl);
            
            c0_32 = __riscv_vfadd_vv_f32m2( c0_32, mul_c0_32, gvl);
            c1_32 = __riscv_vfadd_vv_f32m2( c1_32, mul_c1_32, gvl);
            
            c0 = __riscv_vfncvt_f_f_w_f16m1( c0_32, gvl);
            c1 = __riscv_vfncvt_f_f_w_f16m1( c1_32, gvl);

            ci=n_top*ldc+m_top;

            __riscv_vse16_v_f16m1( &C[ci], c0, gvl); ci += ldc-gvl*0;
            __riscv_vse16_v_f16m1( &C[ci], c1, gvl);
            m_top += 4;
        }


        if( M & 2 ) {

            float result0_32 = 0;
            float result1_32 = 0;
            float result2_32 = 0;
            float result3_32 = 0;
            
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;

            for(unsigned int k=0; k<K; k++) {
                result0_32+=(float)(A[ai+0]*B[bi+0]);
                result1_32+=(float)(A[ai+1]*B[bi+0]);
                result2_32+=(float)(A[ai+0]*B[bi+1]);
                result3_32+=(float)(A[ai+1]*B[bi+1]);
                ai+=2;
                bi+=2;
            }

	    _Float16 result0 = (_Float16)result0_32;
	    _Float16 result1 = (_Float16)result1_32;
	    _Float16 result2 = (_Float16)result2_32;
	    _Float16 result3 = (_Float16)result3_32;

            unsigned int ci=n_top*ldc+m_top;
            
            C[ci+0*ldc+0] = (_Float16)((float)(alpha * result0) + (float)C[ci+0*ldc+0]);
            C[ci+0*ldc+1] = (_Float16)((float)(alpha * result1) + (float)C[ci+0*ldc+1]);
            C[ci+1*ldc+0] = (_Float16)((float)(alpha * result2) + (float)C[ci+1*ldc+0]);
            C[ci+1*ldc+1] = (_Float16)((float)(alpha * result3) + (float)C[ci+1*ldc+1]);

            m_top+=2;
        }


        if( M & 1 ) {
        
            float result0_32 = 0;
            float result1_32 = 0;
            
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            for(unsigned int k=0; k<K; k++) {
                result0_32+=(float)(A[ai+0]*B[bi+0]);
                result1_32+=(float)(A[ai+0]*B[bi+1]);
                ai+=1;
                bi+=2;
            }
            
            _Float16 result0=(_Float16)result0_32;
            _Float16 result1=(_Float16)result1_32;

            unsigned int ci=n_top*ldc+m_top;
            
            C[ci+0*ldc+0] = (_Float16)((float)(alpha * result0) + (float)C[ci+0*ldc+0]);
            C[ci+1*ldc+0] = (_Float16)((float)(alpha * result1) + (float)C[ci+1*ldc+0]);
            

            m_top+=1;
        }

        n_top += 2;
    }



    // -- tails for N=1

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
            
            vfloat32m2_t result0_32 = __riscv_vfwcvt_f_f_v_f32m2( result0, gvl);
            vfloat32m2_t result1_32 = __riscv_vfwcvt_f_f_v_f32m2( result1, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                bi += 1;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                A1 = __riscv_vle16_v_f16m1( &A[ai+1*gvl], gvl );
                ai += 16;

		vfloat16m1_t mul_result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
                vfloat16m1_t mul_result1 = __riscv_vfmul_vf_f16m1( A1, B0, gvl);
                
                vfloat32m2_t mul_result0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result0, gvl);
                vfloat32m2_t mul_result1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result1, gvl);
                
                result0_32 = __riscv_vfadd_vv_f32m2( result0_32, mul_result0_32, gvl);
                result1_32 = __riscv_vfadd_vv_f32m2( result1_32, mul_result1_32, gvl);
            }

            result0 = __riscv_vfncvt_f_f_w_f16m1( result0_32, gvl);
            result1 = __riscv_vfncvt_f_f_w_f16m1( result1_32, gvl);

            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl); ci += gvl;
            vfloat16m1_t c1 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            
            vfloat32m2_t c0_32 = __riscv_vfwcvt_f_f_v_f32m2( c0, gvl);
            vfloat32m2_t c1_32 = __riscv_vfwcvt_f_f_v_f32m2( c1, gvl);
            
            vfloat16m1_t mul_c0 = __riscv_vfmul_vf_f16m1( result0, alpha, gvl);
            vfloat16m1_t mul_c1 = __riscv_vfmul_vf_f16m1( result1, alpha, gvl);
            
            vfloat32m2_t mul_c0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c0, gvl);
            vfloat32m2_t mul_c1_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c1, gvl);
            
            c0_32 = __riscv_vfadd_vv_f32m2( c0_32, mul_c0_32, gvl);
            c1_32 = __riscv_vfadd_vv_f32m2( c1_32, mul_c1_32, gvl);
            
            c0 = __riscv_vfncvt_f_f_w_f16m1( c0_32, gvl);
            c1 = __riscv_vfncvt_f_f_w_f16m1( c1_32, gvl);
            
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
            
            vfloat32m2_t result0_32 = __riscv_vfwcvt_f_f_v_f32m2( result0, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                bi += 1;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                ai += 8;

                vfloat16m1_t mul_result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
                
                vfloat32m2_t mul_result0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result0, gvl);
                
                result0_32 = __riscv_vfadd_vv_f32m2( result0_32, mul_result0_32, gvl);
                
            }

	    result0 = __riscv_vfncvt_f_f_w_f16m1( result0_32, gvl);

            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            
            vfloat32m2_t c0_32 = __riscv_vfwcvt_f_f_v_f32m2( c0, gvl);
            
            vfloat16m1_t mul_c0 = __riscv_vfmul_vf_f16m1( result0, alpha, gvl);
            
            vfloat32m2_t mul_c0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c0, gvl);
            
            c0_32 = __riscv_vfadd_vv_f32m2( c0_32, mul_c0_32, gvl);
            
            c0 = __riscv_vfncvt_f_f_w_f16m1( c0_32, gvl);

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
            
            vfloat32m2_t result0_32 = __riscv_vfwcvt_f_f_v_f32m2( result0, gvl);

            for(unsigned int k=1; k<K; k++) {
                B0 = B[bi+0];
                bi += 1;

                A0 = __riscv_vle16_v_f16m1( &A[ai+0*gvl], gvl );
                ai += 4;

                vfloat16m1_t mul_result0 = __riscv_vfmul_vf_f16m1( A0, B0, gvl);
                
                vfloat32m2_t mul_result0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_result0, gvl);
                
                result0_32 = __riscv_vfadd_vv_f32m2( result0_32, mul_result0_32, gvl);
            }

	    result0 = __riscv_vfncvt_f_f_w_f16m1( result0_32, gvl);

            unsigned int ci=n_top*ldc+m_top;

            vfloat16m1_t c0 = __riscv_vle16_v_f16m1( &C[ci], gvl);
            
            vfloat32m2_t c0_32 = __riscv_vfwcvt_f_f_v_f32m2( c0, gvl);
            
            vfloat16m1_t mul_c0 = __riscv_vfmul_vf_f16m1( result0, alpha, gvl);
            
            vfloat32m2_t mul_c0_32 = __riscv_vfwcvt_f_f_v_f32m2( mul_c0, gvl);
            
            c0_32 = __riscv_vfadd_vv_f32m2( c0_32, mul_c0_32, gvl);
            
            c0 = __riscv_vfncvt_f_f_w_f16m1( c0_32, gvl);

            ci=n_top*ldc+m_top;

            __riscv_vse16_v_f16m1( &C[ci], c0, gvl);
            m_top += 4;
        }


        if( M & 2 ) {
        
            float result0_32 = 0;
            float result1_32 = 0;
            
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;

            for(unsigned int k=0; k<K; k++) {
                result0_32+=(float)(A[ai+0]*B[bi+0]);
                result1_32+=(float)(A[ai+1]*B[bi+0]);
                ai+=2;
                bi+=1;
            }
            
            _Float16 result0 = (_Float16)result0_32;
	    _Float16 result1 = (_Float16)result1_32;
	    
            unsigned int ci=n_top*ldc+m_top;
            
            C[ci+0*ldc+0] = (_Float16)((float)(alpha * result0) + (float)C[ci+0*ldc+0]);
            C[ci+0*ldc+1] = (_Float16)((float)(alpha * result1) + (float)C[ci+0*ldc+1]);
            
            m_top+=2;
        }


        if( M & 1 ) {
        
            float result0_32 = 0;
            
            unsigned int ai=m_top*K;
            unsigned int bi=n_top*K;
            
            for(unsigned int k=0; k<K; k++) {
                result0_32+=(float)(A[ai+0]*B[bi+0]);
                ai+=1;
                bi+=1;
            }

	    _Float16 result0 = (_Float16)result0_32;

            unsigned int ci=n_top*ldc+m_top;
            
            C[ci+0*ldc+0] = (_Float16)((float)(alpha * result0) + (float)C[ci+0*ldc+0]);
            
            m_top+=1;
        }

        n_top += 1;
    }
    

    return 0;
}
