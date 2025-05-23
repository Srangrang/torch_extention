`timescale 1ns / 1ps
//   Copyright (c) 2024 Xiaobai Team (Ao Dong, Linjin Li, Haishuai Zhang)
//   RV-MTVM(RISC-V Multi-thread Vector Martix) is licensed under Mulan PSL v2.
//   You can use this software according to the terms and conditions of the Mulan PSL v2. 
//   You may obtain a copy of Mulan PSL v2 at:
//            http://license.coscl.org.cn/MulanPSL2 
//   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.  
//   See the Mulan PSL v2 for more details.  


module sim(     //行为仿真测试

    );
    
    reg clk;
    reg rst;
    
    wire A_bank;
    wire[3:0] B_bank;
    wire[3:0] incident;
    wire done;
    
    always #10 clk=~clk;
    
    initial
    begin
        clk=0;
        rst=0;
        #5 rst=1;
        #10 rst =0;
    end
    
    top top(.clk(clk),.rst(rst),.A_bank(A_bank),.B_bank(B_bank),.incident(incident),.done(done));
    
endmodule
