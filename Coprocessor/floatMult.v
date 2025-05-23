module floatMult (floatA,floatB,product);   //16位乘法器实现
//版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
//本文链接：https://blog.csdn.net/weixin_58275336/article/details/136738605
input [15:0] floatA, floatB;
output reg [15:0] product;
 
reg sign;
reg signed [5:0] exponent; //6th bit is the sign
reg [9:0] mantissa;
reg [10:0] fractionA, fractionB;	//fraction = {1,mantissa}
reg [21:0] fraction;
 
 
always @ (floatA or floatB) begin
	if (floatA == 0 || floatB == 0) begin
		product = 0;
	end else begin
		sign = floatA[15] ^ floatB[15];//异或，相异为1
		exponent = floatA[14:10] + floatB[14:10] - 5'd15 ;
	
		fractionA = {1'b1,floatA[9:0]};//显示出隐藏位1来计算
		fractionB = {1'b1,floatB[9:0]};
		fraction = fractionA * fractionB;
		
		if (fraction[21] == 1'b1) begin//规格化，将隐藏位再次隐藏起来
			fraction = fraction << 1;
			exponent = exponent + 1; 
		end else if (fraction[20] == 1'b1) begin
			fraction = fraction << 2;
			exponent = exponent + 0;
		end else if (fraction[19] == 1'b1) begin
			fraction = fraction << 3;
			exponent = exponent - 1;
		end else if (fraction[18] == 1'b1) begin
			fraction = fraction << 4;
			exponent = exponent - 2;
		end else if (fraction[17] == 1'b1) begin
			fraction = fraction << 5;
			exponent = exponent - 3;
		end else if (fraction[16] == 1'b1) begin
			fraction = fraction << 6;
			exponent = exponent - 4;
		end else if (fraction[15] == 1'b1) begin
			fraction = fraction << 7;
			exponent = exponent - 5;
		end else if (fraction[14] == 1'b1) begin
			fraction = fraction << 8;
			exponent = exponent - 6;
		end else if (fraction[13] == 1'b1) begin
			fraction = fraction << 9;
			exponent = exponent - 7;
		end else if (fraction[12] == 1'b0) begin
			fraction = fraction << 10;
			exponent = exponent - 8;
		end 
	
		mantissa = fraction[21:12];
		if(exponent[5]==1'b1) begin //exponent is negative 1.exponent太大了溢出；2.exponent太小了，忽略，直接等于0
			product=16'b0000000000000000;
		end
		else begin
			product = {sign,exponent[4:0],mantissa};
		end
	end
end
 
endmodule