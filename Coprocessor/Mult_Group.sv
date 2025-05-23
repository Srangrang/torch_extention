`timescale 1ns / 1ps
//   Copyright (c) 2024 Xiaobai Team (Ao Dong, Linjin Li, Haishuai Zhang)
//   RV-MTVM(RISC-V Multi-thread Vector Martix) is licensed under Mulan PSL v2.
//   You can use this software according to the terms and conditions of the Mulan PSL v2. 
//   You may obtain a copy of Mulan PSL v2 at:
//            http://license.coscl.org.cn/MulanPSL2 
//   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
//   See the Mulan PSL v2 for more details.  
module Mult_Group(      //乘法器组实现
    input [15:0] in1[127:0],
    input [15:0] in2[127:0],
    output [15:0] out[127:0]
    );
    genvar i;
    generate 
        for(i=0;i<128;i=i+1)
        begin:loop
            floatMult fm(.floatA(in1[i]),.floatB(in2[i]),.product(out[i]));
        end
    endgenerate 
endmodule
