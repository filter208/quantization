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
  wire       [7:0]    Mult_Sum;
  wire       [10:0]   FPAnyAdd_r;
  reg        [10:0]   AccuReg;

  (* DONT_TOUCH="yes" *) TernaryAdderFP8_Hard Mult (
    .X      (Iact[7:0]    ), //i
    .Y      (Wght[7:0]    ), //i
    .Z      (Mult_Z[7:0]  ), //i
    .Cin    (1'b0         ), //i
    .Sum    (Mult_Sum[7:0]), //o
    .clk    (clk          ), //i
    .resetn (resetn       )  //i
  );
  fpany_adder_widen #(
    .EXPO_WIDTH_MULT (5),
    .MANT_WIDTH_MULT (2),
    .EXPO_WIDTH_PSUM (8),
    .MANT_WIDTH_PSUM (2)
  ) FPAnyAdd (
    .a (Mult_Sum[7:0]   ), //i
    .b (AccuReg[10:0]   ), //i
    .r (FPAnyAdd_r[10:0])  //o
  );
  assign Mult_Z = {6'h31,2'b00};
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
