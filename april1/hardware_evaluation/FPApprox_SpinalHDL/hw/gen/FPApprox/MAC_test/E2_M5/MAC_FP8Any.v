// Generator : SpinalHDL v1.10.2a    git head : a348a60b7e8b6a455c72e1536ec3d74a2ea16935
// Component : MAC_FP8Any
// Git hash  : f1981db022e2d0e3b777f608e60cec9faa0d42d7

`timescale 1ns/1ps

module MAC_FP8Any (
  input  wire [7:0]    Iact,
  input  wire [7:0]    Wght,
  output wire [10:0]   Result,
  input  wire          PassResult,
  input  wire [10:0]   ResultCIN,
  input  wire          clk,
  input  wire          resetn
);

  wire       [7:0]    Mult_Z;
  wire       [2:0]    fPAny_LNS_Mult_Comp_DnSp_1_MantX;
  wire       [2:0]    fPAny_LNS_Mult_Comp_DnSp_1_MantY;
  wire       [7:0]    Mult_Sum;
  wire       [2:0]    fPAny_LNS_Mult_Comp_DnSp_1_MantComp;
  wire       [10:0]   FPAnyAdd_r;
  wire       [4:0]    tmp_Z;
  wire       [2:0]    tmp_Z_1;
  reg        [10:0]   AccuReg;

  assign tmp_Z_1 = fPAny_LNS_Mult_Comp_DnSp_1_MantComp;
  assign tmp_Z = {2'd0, tmp_Z_1};
  (* DONT_TOUCH="yes" *) TernaryAdderFP8_Hard Mult (
    .X      (Iact[7:0]    ), //i
    .Y      (Wght[7:0]    ), //i
    .Z      (Mult_Z[7:0]  ), //i
    .Cin    (1'b0         ), //i
    .Sum    (Mult_Sum[7:0]), //o
    .clk    (clk          ), //i
    .resetn (resetn       )  //i
  );
  FPAny_LNS_Mult_Comp_DnSp fPAny_LNS_Mult_Comp_DnSp_1 (
    .MantX    (fPAny_LNS_Mult_Comp_DnSp_1_MantX[2:0]   ), //i
    .MantY    (fPAny_LNS_Mult_Comp_DnSp_1_MantY[2:0]   ), //i
    .MantComp (fPAny_LNS_Mult_Comp_DnSp_1_MantComp[2:0])  //o
  );
  fpany_adder_widen #(
    .EXPO_WIDTH_MULT (2),
    .MANT_WIDTH_MULT (5),
    .EXPO_WIDTH_PSUM (5),
    .MANT_WIDTH_PSUM (5)
  ) FPAnyAdd (
    .a (Mult_Sum[7:0]   ), //i
    .b (AccuReg[10:0]   ), //i
    .r (FPAnyAdd_r[10:0])  //o
  );
  assign fPAny_LNS_Mult_Comp_DnSp_1_MantX = Iact[4 : 2];
  assign fPAny_LNS_Mult_Comp_DnSp_1_MantY = Wght[4 : 2];
  assign Mult_Z = {3'b111,tmp_Z};
  assign Result = AccuReg;
  always @(posedge clk or negedge resetn) begin
    if(!resetn) begin
      AccuReg <= 11'h0;
    end else begin
      if(PassResult) begin
        AccuReg <= ResultCIN;
      end else begin
        AccuReg <= FPAnyAdd_r;
      end
    end
  end


endmodule

module FPAny_LNS_Mult_Comp_DnSp (
  input  wire [2:0]    MantX,
  input  wire [2:0]    MantY,
  output reg  [2:0]    MantComp
);

  wire                LUTs_0_O;
  wire                LUTs_1_O;
  wire                LUTs_2_O;
  wire       [5:0]    LUT_input;

  LUT6_Hard LUTs_0 (
    .I (LUT_input[5:0]), //i
    .O (LUTs_0_O      )  //o
  );
  LUT6_Hard_1 LUTs_1 (
    .I (LUT_input[5:0]), //i
    .O (LUTs_1_O      )  //o
  );
  LUT6_Hard_2 LUTs_2 (
    .I (LUT_input[5:0]), //i
    .O (LUTs_2_O      )  //o
  );
  assign LUT_input = {MantY,MantX};
  always @(*) begin
    MantComp[0] = LUTs_0_O;
    MantComp[1] = LUTs_1_O;
    MantComp[2] = LUTs_2_O;
  end


endmodule

module TernaryAdderFP8_Hard (
  input  wire [7:0]    X,
  input  wire [7:0]    Y,
  input  wire [7:0]    Z,
  input  wire          Cin,
  output wire [7:0]    Sum,
  input  wire          clk,
  input  wire          resetn
);

  wire       [7:0]    TerAdd_sum;
  reg        [7:0]    Result;

  ternary_adder_noincr TerAdd (
    .x   (X[7:0]         ), //i
    .y   (Y[7:0]         ), //i
    .z   (Z[7:0]         ), //i
    .cin (Cin            ), //i
    .sum (TerAdd_sum[7:0])  //o
  );
  assign Sum = Result;
  always @(posedge clk or negedge resetn) begin
    if(!resetn) begin
      Result <= 8'h0;
    end else begin
      Result <= TerAdd_sum;
    end
  end


endmodule

module LUT6_Hard_2 (
  input  wire [5:0]    I,
  output wire          O
);

  wire                LUT6_inst_I0;
  wire                LUT6_inst_I1;
  wire                LUT6_inst_I2;
  wire                LUT6_inst_I3;
  wire                LUT6_inst_I4;
  wire                LUT6_inst_I5;
  wire                LUT6_inst_O;

  LUT6 #(
    .INIT (64'h0000040c1c380000)
  ) LUT6_inst (
    .I0 (LUT6_inst_I0), //i
    .I1 (LUT6_inst_I1), //i
    .I2 (LUT6_inst_I2), //i
    .I3 (LUT6_inst_I3), //i
    .I4 (LUT6_inst_I4), //i
    .I5 (LUT6_inst_I5), //i
    .O  (LUT6_inst_O )  //o
  );
  assign LUT6_inst_I0 = I[0];
  assign LUT6_inst_I1 = I[1];
  assign LUT6_inst_I2 = I[2];
  assign LUT6_inst_I3 = I[3];
  assign LUT6_inst_I4 = I[4];
  assign LUT6_inst_I5 = I[5];
  assign O = LUT6_inst_O;

endmodule

module LUT6_Hard_1 (
  input  wire [5:0]    I,
  output wire          O
);

  wire                LUT6_inst_I0;
  wire                LUT6_inst_I1;
  wire                LUT6_inst_I2;
  wire                LUT6_inst_I3;
  wire                LUT6_inst_I4;
  wire                LUT6_inst_I5;
  wire                LUT6_inst_O;

  LUT6 #(
    .INIT (64'h001e3a7262467c00)
  ) LUT6_inst (
    .I0 (LUT6_inst_I0), //i
    .I1 (LUT6_inst_I1), //i
    .I2 (LUT6_inst_I2), //i
    .I3 (LUT6_inst_I3), //i
    .I4 (LUT6_inst_I4), //i
    .I5 (LUT6_inst_I5), //i
    .O  (LUT6_inst_O )  //o
  );
  assign LUT6_inst_I0 = I[0];
  assign LUT6_inst_I1 = I[1];
  assign LUT6_inst_I2 = I[2];
  assign LUT6_inst_I3 = I[3];
  assign LUT6_inst_I4 = I[4];
  assign LUT6_inst_I5 = I[5];
  assign O = LUT6_inst_O;

endmodule

module LUT6_Hard (
  input  wire [5:0]    I,
  output wire          O
);

  wire                LUT6_inst_I0;
  wire                LUT6_inst_I1;
  wire                LUT6_inst_I2;
  wire                LUT6_inst_I3;
  wire                LUT6_inst_I4;
  wire                LUT6_inst_I5;
  wire                LUT6_inst_O;

  LUT6 #(
    .INIT (64'h0f634b17a894f2f0)
  ) LUT6_inst (
    .I0 (LUT6_inst_I0), //i
    .I1 (LUT6_inst_I1), //i
    .I2 (LUT6_inst_I2), //i
    .I3 (LUT6_inst_I3), //i
    .I4 (LUT6_inst_I4), //i
    .I5 (LUT6_inst_I5), //i
    .O  (LUT6_inst_O )  //o
  );
  assign LUT6_inst_I0 = I[0];
  assign LUT6_inst_I1 = I[1];
  assign LUT6_inst_I2 = I[2];
  assign LUT6_inst_I3 = I[3];
  assign LUT6_inst_I4 = I[4];
  assign LUT6_inst_I5 = I[5];
  assign O = LUT6_inst_O;

endmodule
