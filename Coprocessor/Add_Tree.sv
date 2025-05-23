`timescale 1ns / 1ps
//   Copyright (c) 2024 Xiaobai Team (Ao Dong, Linjin Li, Haishuai Zhang)
//   RV-MTVM(RISC-V Multi-thread Vector Martix) is licensed under Mulan PSL v2.
//   You can use this software according to the terms and conditions of the Mulan PSL v2. 
//   You may obtain a copy of Mulan PSL v2 at:
//            http://license.coscl.org.cn/MulanPSL2 
//   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
//   See the Mulan PSL v2 for more details.  
module Add_Tree(        //加法树实现
    input clk,
    input rst,
    input [15:0] in[127:0],
    output [15:0] out
    );
    reg[15:0] add_in[253:0];
    wire[15:0] add_out[125:0];
    genvar i1;
    generate
        for(i1=0;i1<64;i1=i1+1)
        begin:loop1
            floatAdd fa(.floatA(add_in[2*i1]),.floatB(add_in[2*i1+1]),.sum(add_out[i1]));
        end
    endgenerate
    genvar i2;
    generate
        for(i2=0;i2<32;i2=i2+1)
        begin:loop2
            floatAdd fa(.floatA(add_in[128+2*i2]),.floatB(add_in[129+2*i2]),.sum(add_out[64+i2]));
        end
    endgenerate
    genvar i3;
    generate
        for(i3=0;i3<16;i3=i3+1)
        begin:loop3
            floatAdd fa(.floatA(add_in[192+2*i3]),.floatB(add_in[193+2*i3]),.sum(add_out[96+i3]));
        end
    endgenerate
    genvar i4;
    generate
        for(i4=0;i4<8;i4=i4+1)
        begin:loop4
            floatAdd fa(.floatA(add_in[224+2*i4]),.floatB(add_in[225+2*i4]),.sum(add_out[112+i4]));
        end
    endgenerate
    genvar i5;
    generate
        for(i5=0;i5<4;i5=i5+1)
        begin:loop5
            floatAdd fa(.floatA(add_in[240+2*i5]),.floatB(add_in[241+2*i5]),.sum(add_out[120+i5]));
        end
    endgenerate
    genvar i6;
    generate
        for(i6=0;i6<2;i6=i6+1)
        begin:loop6
            floatAdd fa(.floatA(add_in[248+2*i6]),.floatB(add_in[249+2*i6]),.sum(add_out[124+i6]));
        end
    endgenerate
    floatAdd fa(.floatA(add_in[252]),.floatB(add_in[253]),.sum(out));
    always @(posedge clk,posedge rst)
    begin
        if(rst)
        begin
            add_in<='{default:'0};
        end
        else
        begin 
            add_in[127:0]<=in;
            add_in[191:128]<=add_out[63:0];
            add_in[223:192]<=add_out[95:64];
            add_in[239:224]<=add_out[111:96];
            add_in[247:240]<=add_out[119:112];
            add_in[251:248]<=add_out[123:120];
            add_in[253:252]<=add_out[125:124];
        end
    end
endmodule
