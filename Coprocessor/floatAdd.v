module floatAdd (floatA,floatB,sum);    //16位浮点数加法器实现
//版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
//本文链接：https://blog.csdn.net/weixin_58275336/article/details/136738605

input [15:0] floatA, floatB;
output reg [15:0] sum;
 
reg sign;
reg signed [5:0] exponent; //fifth bit is sign
reg [9:0] mantissa;
reg [4:0] exponentA, exponentB;
reg [10:0] fractionA, fractionB, fraction;	//fraction = {1,mantissa}
reg [7:0] shiftAmount;
reg cout;
 
always @ (floatA or floatB) begin
	exponentA = floatA[14:10];
	exponentB = floatB[14:10];
	fractionA = {1'b1,floatA[9:0]};//将隐藏位表示出来进行计算
	fractionB = {1'b1,floatB[9:0]}; //同理
	
	exponent = exponentA;
 
	if (floatA == 0) begin						//special case (floatA = 0)
		sum = floatB;
	end else if (floatB == 0) begin					//special case (floatB = 0)
		sum = floatA;
	end else if (floatA[14:0] == floatB[14:0] && floatA[15]^floatB[15]==1'b1) begin
		sum=0;
	end else begin
		if (exponentB > exponentA) begin//对阶：将阶数(exponent)化为相同的，小阶化为大阶。好比6.6x10^(6)和8.8x10^(4)相加时，我们会化成6.6x10^(6)和0.088x10^(6)进行运算
			shiftAmount = exponentB - exponentA;
			fractionA = fractionA >> (shiftAmount);//要将floatA化为大阶，则尾数要右移，尾数减小
			exponent = exponentB;
		end else if (exponentA > exponentB) begin //同理
			shiftAmount = exponentA - exponentB;
			fractionB = fractionB >> (shiftAmount);
			exponent = exponentA;
		end
		if (floatA[15] == floatB[15]) begin			//same sign
			{cout,fraction} = fractionA + fractionB;
			if (cout == 1'b1) begin//'相加后的值'的小数点前的第二位如果是1，则要向右移一位，确保小数点前只有一位
				{cout,fraction} = {cout,fraction} >> 1;
				exponent = exponent + 1;
			end
			sign = floatA[15];
		end else begin						//different signs
			if (floatA[15] == 1'b1) begin
				{cout,fraction} = fractionB - fractionA;//如果B比A小，则相减后得到补码，cout为符号位
			end else begin
				{cout,fraction} = fractionA - fractionB;//同理
			end
			sign = cout;
			if (cout == 1'b1) begin
				fraction = -fraction;//将补码转化为原码
			end else begin
			end
			if (fraction [10] == 0) begin
				if (fraction[9] == 1'b1) begin//规格化，将隐藏位再次隐藏起来
					fraction = fraction << 1;
					exponent = exponent - 1;
				end else if (fraction[8] == 1'b1) begin
					fraction = fraction << 2;
					exponent = exponent - 2;
				end else if (fraction[7] == 1'b1) begin
					fraction = fraction << 3;
					exponent = exponent - 3;
				end else if (fraction[6] == 1'b1) begin
					fraction = fraction << 4;
					exponent = exponent - 4;
				end else if (fraction[5] == 1'b1) begin
					fraction = fraction << 5;
					exponent = exponent - 5;
				end else if (fraction[4] == 1'b1) begin
					fraction = fraction << 6;
					exponent = exponent - 6;
				end else if (fraction[3] == 1'b1) begin
					fraction = fraction << 7;
					exponent = exponent - 7;
				end else if (fraction[2] == 1'b1) begin
					fraction = fraction << 8;
					exponent = exponent - 8;
				end else if (fraction[1] == 1'b1) begin
					fraction = fraction << 9;
					exponent = exponent - 9;
				end else if (fraction[0] == 1'b1) begin
					fraction = fraction << 10;
					exponent = exponent - 10;
				end 
			end
		end
		mantissa = fraction[9:0];//
		if(exponent[5]==1'b1) begin //exponent is negative：1.exponent太大了溢出；2.规格化后发现exponent太小了，忽略，直接等于0
			sum = 16'b0000000000000000;
		end
		else begin
			sum = {sign,exponent[4:0],mantissa};
		end		
	end		
end
 
endmodule