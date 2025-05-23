`timescale 1ns / 1ps
//   Copyright (c) 2024 Xiaobai Team (Ao Dong, Linjin Li, Haishuai Zhang)
//   RV-MTVM(RISC-V Multi-thread Vector Martix) is licensed under Mulan PSL v2.
//   You can use this software according to the terms and conditions of the Mulan PSL v2. 
//   You may obtain a copy of Mulan PSL v2 at:
//            http://license.coscl.org.cn/MulanPSL2 
//   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
//   See the Mulan PSL v2 for more details.  

parameter rowsA=1,colsA=128,rowsB=128,colsB=128;        //矩阵乘法规模
parameter none=0,readB=1,readA_and_readB=2,readA_and_readB_and_writeC=3;        //DMA事件定义

module top(
    input clk,  //时钟信号
    input rst,  //复位信号
    output reg A_bank,  //在哪个Abank中进行计算
    output reg[3:0] B_bank,     //在哪个Bbank中进行计算
    output reg[3:0] incident,       //DMA事件
    output reg done     //完成信号
    );
    
    reg[15:0] in1[127:0];   //乘法器组的输入1
    reg[15:0] in2[127:0];   //乘法器组的输入2
    wire[15:0] group_out[127:0];    //乘法器组的输出
    
    (*DONT_TOUCH = "yes"*)reg[15:0] A_cache[255:0]; //A缓存
    (*DONT_TOUCH = "yes"*)reg[15:0] B_cache[16384:0]; //B缓存
    (*DONT_TOUCH = "yes"*)reg[15:0] C_cache[colsB-1]; //C缓存
    
    reg[31:0] C_cache_ptr;  //C缓存的指针
    reg[31:0] A_cols_ptr,C_rows_ptr;    //A计算到拿一列，C计算到哪一行
    reg[15:0] count;    //一个bank的128循环计数器
    wire [15:0] tree_out;   //加法器的输出
    wire [15:0] sum;    //累加和
    reg [15:0] tree_in[127:0];  //加法器的输入
    reg [15:0] add_in1,add_in2;     //加法器的输入
    
    Mult_Group group(.in1(in1),.in2(in2),.out(group_out));  //128个乘法器组
    Add_Tree at(.clk(clk),.rst(rst),.in(tree_in),.out(tree_out));   //宽度为64的加法树
    floatAdd fa(.floatA(add_in1),.floatB(add_in2),.sum(sum));   //累加和加法器
    
    always @(posedge clk,posedge rst)
    begin
        if(rst)
        begin
            A_cache<='{default:'0};
            B_cache<='{default:'0};
            C_cache<='{default:'0};
            A_bank<=0;
            B_bank<=0;
            incident<=0;
            done<=0;
        end
        else
        begin
            C_cache[C_cache_ptr]<=sum;
            if((count==127)&(C_cache_ptr==colsB-1))
            begin
                A_bank<=!A_bank;
                if(A_cols_ptr+128>=colsA-1)
                begin
                    incident<=readA_and_readB_and_writeC;
                    C_cache<='{default:'0};
                    if(C_rows_ptr==rowsA-1)
                        done<=1;
                end
                else
                    incident<=readA_and_readB;
            end
            else if(count==127)
            begin
                incident<=readB;
                if(B_bank==7)
                    B_bank<=0;
                else
                    B_bank<=B_bank+1;
            end
            else
                incident<=0;
        end
    end
    
    always@(posedge clk,posedge rst)
    begin
        if(rst)
        begin
            in1<='{default:'0};
            in2<='{default:'0};
        end
        else
        begin
            in1<=A_cache[128*A_bank+:128];
            in2<=B_cache[count*128+:128];
        end
    end
    
    always @(posedge clk,posedge rst)
    begin
        if(rst)
        begin
            A_cols_ptr<=0;
            C_rows_ptr<=0;
            count<=0;
        end
        else
        begin
            if(count==127)
                count<=0;
            else
                count<=count+1;
            if(count==127&(C_cache_ptr==colsB-1))
            begin
                if(A_cols_ptr+128>=colsA-1)
                begin
                    A_cols_ptr<=0;
                    C_rows_ptr<=C_rows_ptr+1;
                end
                else
                    A_cols_ptr<=A_cols_ptr+128;
            end
        end
    end
    
    always @(posedge clk,posedge rst)
    begin
        if(rst)
        begin
            C_cache_ptr<=0;
        end
        else
        begin
            if(C_cache_ptr==colsB-1)
                C_cache_ptr<=0;
            else
                C_cache_ptr++;
        end
    end

    always @(posedge clk,posedge rst)
    begin
        if(rst)
        begin
            tree_in<='{default:'0};
            add_in1<=0;
            add_in2<=0;
        end
        else
        begin
            tree_in<=group_out;
            add_in1<=tree_out;
            add_in2<=C_cache[C_cache_ptr];
        end
    end
    
endmodule
