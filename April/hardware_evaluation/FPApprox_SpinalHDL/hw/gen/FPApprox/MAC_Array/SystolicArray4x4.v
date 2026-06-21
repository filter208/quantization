// Generator : SpinalHDL v1.10.2a    git head : a348a60b7e8b6a455c72e1536ec3d74a2ea16935
// Component : SystolicArray4x4
// Git hash  : 651771dac71183b2470213a29207390adcecc770

`timescale 1ns/1ps

module SystolicArray4x4 (
  input  wire [7:0]    io_i_act_top_0,
  input  wire [7:0]    io_i_act_top_1,
  input  wire [7:0]    io_i_act_top_2,
  input  wire [7:0]    io_i_act_top_3,
  input  wire [7:0]    io_i_wght_left_0,
  input  wire [7:0]    io_i_wght_left_1,
  input  wire [7:0]    io_i_wght_left_2,
  input  wire [7:0]    io_i_wght_left_3,
  input  wire          io_shift_en,
  output wire [10:0]   io_o_res_right_0,
  output wire [10:0]   io_o_res_right_1,
  output wire [10:0]   io_o_res_right_2,
  output wire [10:0]   io_o_res_right_3,
  input  wire          clk,
  input  wire          resetn
);

  wire       [7:0]    peGrid_0_0_io_o_act;
  wire       [7:0]    peGrid_0_0_io_o_wght;
  wire       [10:0]   peGrid_0_0_io_o_pass;
  wire       [7:0]    peGrid_0_1_io_o_act;
  wire       [7:0]    peGrid_0_1_io_o_wght;
  wire       [10:0]   peGrid_0_1_io_o_pass;
  wire       [7:0]    peGrid_0_2_io_o_act;
  wire       [7:0]    peGrid_0_2_io_o_wght;
  wire       [10:0]   peGrid_0_2_io_o_pass;
  wire       [7:0]    peGrid_0_3_io_o_act;
  wire       [7:0]    peGrid_0_3_io_o_wght;
  wire       [10:0]   peGrid_0_3_io_o_pass;
  wire       [7:0]    peGrid_1_0_io_o_act;
  wire       [7:0]    peGrid_1_0_io_o_wght;
  wire       [10:0]   peGrid_1_0_io_o_pass;
  wire       [7:0]    peGrid_1_1_io_o_act;
  wire       [7:0]    peGrid_1_1_io_o_wght;
  wire       [10:0]   peGrid_1_1_io_o_pass;
  wire       [7:0]    peGrid_1_2_io_o_act;
  wire       [7:0]    peGrid_1_2_io_o_wght;
  wire       [10:0]   peGrid_1_2_io_o_pass;
  wire       [7:0]    peGrid_1_3_io_o_act;
  wire       [7:0]    peGrid_1_3_io_o_wght;
  wire       [10:0]   peGrid_1_3_io_o_pass;
  wire       [7:0]    peGrid_2_0_io_o_act;
  wire       [7:0]    peGrid_2_0_io_o_wght;
  wire       [10:0]   peGrid_2_0_io_o_pass;
  wire       [7:0]    peGrid_2_1_io_o_act;
  wire       [7:0]    peGrid_2_1_io_o_wght;
  wire       [10:0]   peGrid_2_1_io_o_pass;
  wire       [7:0]    peGrid_2_2_io_o_act;
  wire       [7:0]    peGrid_2_2_io_o_wght;
  wire       [10:0]   peGrid_2_2_io_o_pass;
  wire       [7:0]    peGrid_2_3_io_o_act;
  wire       [7:0]    peGrid_2_3_io_o_wght;
  wire       [10:0]   peGrid_2_3_io_o_pass;
  wire       [7:0]    peGrid_3_0_io_o_act;
  wire       [7:0]    peGrid_3_0_io_o_wght;
  wire       [10:0]   peGrid_3_0_io_o_pass;
  wire       [7:0]    peGrid_3_1_io_o_act;
  wire       [7:0]    peGrid_3_1_io_o_wght;
  wire       [10:0]   peGrid_3_1_io_o_pass;
  wire       [7:0]    peGrid_3_2_io_o_act;
  wire       [7:0]    peGrid_3_2_io_o_wght;
  wire       [10:0]   peGrid_3_2_io_o_pass;
  wire       [7:0]    peGrid_3_3_io_o_act;
  wire       [7:0]    peGrid_3_3_io_o_wght;
  wire       [10:0]   peGrid_3_3_io_o_pass;

  SystolicPE peGrid_0_0 (
    .io_i_act    (io_i_act_top_0[7:0]       ), //i
    .io_i_wght   (io_i_wght_left_0[7:0]     ), //i
    .io_i_casc   (11'h0                     ), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_0_0_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_0_0_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_0_0_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_0_1 (
    .io_i_act    (io_i_act_top_1[7:0]       ), //i
    .io_i_wght   (peGrid_0_0_io_o_wght[7:0] ), //i
    .io_i_casc   (peGrid_0_0_io_o_pass[10:0]), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_0_1_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_0_1_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_0_1_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_0_2 (
    .io_i_act    (io_i_act_top_2[7:0]       ), //i
    .io_i_wght   (peGrid_0_1_io_o_wght[7:0] ), //i
    .io_i_casc   (peGrid_0_1_io_o_pass[10:0]), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_0_2_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_0_2_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_0_2_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_0_3 (
    .io_i_act    (io_i_act_top_3[7:0]       ), //i
    .io_i_wght   (peGrid_0_2_io_o_wght[7:0] ), //i
    .io_i_casc   (peGrid_0_2_io_o_pass[10:0]), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_0_3_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_0_3_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_0_3_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_1_0 (
    .io_i_act    (peGrid_0_0_io_o_act[7:0]  ), //i
    .io_i_wght   (io_i_wght_left_1[7:0]     ), //i
    .io_i_casc   (11'h0                     ), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_1_0_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_1_0_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_1_0_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_1_1 (
    .io_i_act    (peGrid_0_1_io_o_act[7:0]  ), //i
    .io_i_wght   (peGrid_1_0_io_o_wght[7:0] ), //i
    .io_i_casc   (peGrid_1_0_io_o_pass[10:0]), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_1_1_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_1_1_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_1_1_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_1_2 (
    .io_i_act    (peGrid_0_2_io_o_act[7:0]  ), //i
    .io_i_wght   (peGrid_1_1_io_o_wght[7:0] ), //i
    .io_i_casc   (peGrid_1_1_io_o_pass[10:0]), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_1_2_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_1_2_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_1_2_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_1_3 (
    .io_i_act    (peGrid_0_3_io_o_act[7:0]  ), //i
    .io_i_wght   (peGrid_1_2_io_o_wght[7:0] ), //i
    .io_i_casc   (peGrid_1_2_io_o_pass[10:0]), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_1_3_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_1_3_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_1_3_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_2_0 (
    .io_i_act    (peGrid_1_0_io_o_act[7:0]  ), //i
    .io_i_wght   (io_i_wght_left_2[7:0]     ), //i
    .io_i_casc   (11'h0                     ), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_2_0_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_2_0_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_2_0_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_2_1 (
    .io_i_act    (peGrid_1_1_io_o_act[7:0]  ), //i
    .io_i_wght   (peGrid_2_0_io_o_wght[7:0] ), //i
    .io_i_casc   (peGrid_2_0_io_o_pass[10:0]), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_2_1_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_2_1_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_2_1_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_2_2 (
    .io_i_act    (peGrid_1_2_io_o_act[7:0]  ), //i
    .io_i_wght   (peGrid_2_1_io_o_wght[7:0] ), //i
    .io_i_casc   (peGrid_2_1_io_o_pass[10:0]), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_2_2_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_2_2_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_2_2_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_2_3 (
    .io_i_act    (peGrid_1_3_io_o_act[7:0]  ), //i
    .io_i_wght   (peGrid_2_2_io_o_wght[7:0] ), //i
    .io_i_casc   (peGrid_2_2_io_o_pass[10:0]), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_2_3_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_2_3_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_2_3_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_3_0 (
    .io_i_act    (peGrid_2_0_io_o_act[7:0]  ), //i
    .io_i_wght   (io_i_wght_left_3[7:0]     ), //i
    .io_i_casc   (11'h0                     ), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_3_0_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_3_0_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_3_0_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_3_1 (
    .io_i_act    (peGrid_2_1_io_o_act[7:0]  ), //i
    .io_i_wght   (peGrid_3_0_io_o_wght[7:0] ), //i
    .io_i_casc   (peGrid_3_0_io_o_pass[10:0]), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_3_1_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_3_1_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_3_1_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_3_2 (
    .io_i_act    (peGrid_2_2_io_o_act[7:0]  ), //i
    .io_i_wght   (peGrid_3_1_io_o_wght[7:0] ), //i
    .io_i_casc   (peGrid_3_1_io_o_pass[10:0]), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_3_2_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_3_2_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_3_2_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  SystolicPE peGrid_3_3 (
    .io_i_act    (peGrid_2_3_io_o_act[7:0]  ), //i
    .io_i_wght   (peGrid_3_2_io_o_wght[7:0] ), //i
    .io_i_casc   (peGrid_3_2_io_o_pass[10:0]), //i
    .io_shift_en (io_shift_en               ), //i
    .io_o_act    (peGrid_3_3_io_o_act[7:0]  ), //o
    .io_o_wght   (peGrid_3_3_io_o_wght[7:0] ), //o
    .io_o_pass   (peGrid_3_3_io_o_pass[10:0]), //o
    .clk         (clk                       ), //i
    .resetn      (resetn                    )  //i
  );
  assign io_o_res_right_0 = peGrid_0_3_io_o_pass;
  assign io_o_res_right_1 = peGrid_1_3_io_o_pass;
  assign io_o_res_right_2 = peGrid_2_3_io_o_pass;
  assign io_o_res_right_3 = peGrid_3_3_io_o_pass;

endmodule

//SystolicPE_15 replaced by SystolicPE

//SystolicPE_14 replaced by SystolicPE

//SystolicPE_13 replaced by SystolicPE

//SystolicPE_12 replaced by SystolicPE

//SystolicPE_11 replaced by SystolicPE

//SystolicPE_10 replaced by SystolicPE

//SystolicPE_9 replaced by SystolicPE

//SystolicPE_8 replaced by SystolicPE

//SystolicPE_7 replaced by SystolicPE

//SystolicPE_6 replaced by SystolicPE

//SystolicPE_5 replaced by SystolicPE

//SystolicPE_4 replaced by SystolicPE

//SystolicPE_3 replaced by SystolicPE

//SystolicPE_2 replaced by SystolicPE

//SystolicPE_1 replaced by SystolicPE

module SystolicPE (
  input  wire [7:0]    io_i_act,
  input  wire [7:0]    io_i_wght,
  input  wire [10:0]   io_i_casc,
  input  wire          io_shift_en,
  output wire [7:0]    io_o_act,
  output wire [7:0]    io_o_wght,
  output wire [10:0]   io_o_pass,
  input  wire          clk,
  input  wire          resetn
);

  wire       [10:0]   mac_Result;
  reg        [7:0]    act_reg;
  reg        [7:0]    wght_reg;

  MAC_FP8Any_15 mac (
    .Iact       (act_reg[7:0]    ), //i
    .Wght       (wght_reg[7:0]   ), //i
    .Result     (mac_Result[10:0]), //o
    .PassResult (io_shift_en     ), //i
    .ResultCIN  (io_i_casc[10:0] ), //i
    .clk        (clk             ), //i
    .resetn     (resetn          )  //i
  );
  assign io_o_act = act_reg;
  assign io_o_wght = wght_reg;
  assign io_o_pass = mac_Result;
  always @(posedge clk or negedge resetn) begin
    if(!resetn) begin
      act_reg <= 8'h0;
      wght_reg <= 8'h0;
    end else begin
      act_reg <= io_i_act;
      wght_reg <= io_i_wght;
    end
  end


endmodule

//MAC_FP8Any replaced by MAC_FP8Any_15

//MAC_FP8Any_1 replaced by MAC_FP8Any_15

//MAC_FP8Any_2 replaced by MAC_FP8Any_15

//MAC_FP8Any_3 replaced by MAC_FP8Any_15

//MAC_FP8Any_4 replaced by MAC_FP8Any_15

//MAC_FP8Any_5 replaced by MAC_FP8Any_15

//MAC_FP8Any_6 replaced by MAC_FP8Any_15

//MAC_FP8Any_7 replaced by MAC_FP8Any_15

//MAC_FP8Any_8 replaced by MAC_FP8Any_15

//MAC_FP8Any_9 replaced by MAC_FP8Any_15

//MAC_FP8Any_10 replaced by MAC_FP8Any_15

//MAC_FP8Any_11 replaced by MAC_FP8Any_15

//MAC_FP8Any_12 replaced by MAC_FP8Any_15

//MAC_FP8Any_13 replaced by MAC_FP8Any_15

//MAC_FP8Any_14 replaced by MAC_FP8Any_15

module MAC_FP8Any_15 (
  input  wire [7:0]    Iact,
  input  wire [7:0]    Wght,
  output wire [10:0]   Result,
  input  wire          PassResult,
  input  wire [10:0]   ResultCIN,
  input  wire          clk,
  input  wire          resetn
);

  wire       [7:0]    Mult_Z;
  wire       [2:0]    fPAny_LNS_Mult_Comp_DnSp_16_MantX;
  wire       [2:0]    fPAny_LNS_Mult_Comp_DnSp_16_MantY;
  wire       [7:0]    Mult_Sum;
  wire       [0:0]    fPAny_LNS_Mult_Comp_DnSp_16_MantComp;
  wire       [10:0]   FPAnyAdd_r;
  wire       [2:0]    tmp_Z;
  wire       [0:0]    tmp_Z_1;
  reg        [10:0]   AccuReg;

  assign tmp_Z_1 = fPAny_LNS_Mult_Comp_DnSp_16_MantComp;
  assign tmp_Z = {2'd0, tmp_Z_1};
  (* DONT_TOUCH="yes" *) TernaryAdderFP8_Hard_15 Mult (
    .X      (Iact[7:0]    ), //i
    .Y      (Wght[7:0]    ), //i
    .Z      (Mult_Z[7:0]  ), //i
    .Cin    (1'b0         ), //i
    .Sum    (Mult_Sum[7:0]), //o
    .clk    (clk          ), //i
    .resetn (resetn       )  //i
  );
  FPAny_LNS_Mult_Comp_DnSp_15 fPAny_LNS_Mult_Comp_DnSp_16 (
    .MantX    (fPAny_LNS_Mult_Comp_DnSp_16_MantX[2:0]), //i
    .MantY    (fPAny_LNS_Mult_Comp_DnSp_16_MantY[2:0]), //i
    .MantComp (fPAny_LNS_Mult_Comp_DnSp_16_MantComp  )  //o
  );
  fpany_adder_widen #(
    .EXPO_WIDTH_MULT (4),
    .MANT_WIDTH_MULT (3),
    .EXPO_WIDTH_PSUM (7),
    .MANT_WIDTH_PSUM (3)
  ) FPAnyAdd (
    .a (Mult_Sum[7:0]   ), //i
    .b (AccuReg[10:0]   ), //i
    .r (FPAnyAdd_r[10:0])  //o
  );
  assign fPAny_LNS_Mult_Comp_DnSp_16_MantX = Iact[2 : 0];
  assign fPAny_LNS_Mult_Comp_DnSp_16_MantY = Wght[2 : 0];
  assign Mult_Z = {5'h19,tmp_Z};
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

//FPAny_LNS_Mult_Comp_DnSp replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard replaced by TernaryAdderFP8_Hard_15

//FPAny_LNS_Mult_Comp_DnSp_1 replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard_1 replaced by TernaryAdderFP8_Hard_15

//FPAny_LNS_Mult_Comp_DnSp_2 replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard_2 replaced by TernaryAdderFP8_Hard_15

//FPAny_LNS_Mult_Comp_DnSp_3 replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard_3 replaced by TernaryAdderFP8_Hard_15

//FPAny_LNS_Mult_Comp_DnSp_4 replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard_4 replaced by TernaryAdderFP8_Hard_15

//FPAny_LNS_Mult_Comp_DnSp_5 replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard_5 replaced by TernaryAdderFP8_Hard_15

//FPAny_LNS_Mult_Comp_DnSp_6 replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard_6 replaced by TernaryAdderFP8_Hard_15

//FPAny_LNS_Mult_Comp_DnSp_7 replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard_7 replaced by TernaryAdderFP8_Hard_15

//FPAny_LNS_Mult_Comp_DnSp_8 replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard_8 replaced by TernaryAdderFP8_Hard_15

//FPAny_LNS_Mult_Comp_DnSp_9 replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard_9 replaced by TernaryAdderFP8_Hard_15

//FPAny_LNS_Mult_Comp_DnSp_10 replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard_10 replaced by TernaryAdderFP8_Hard_15

//FPAny_LNS_Mult_Comp_DnSp_11 replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard_11 replaced by TernaryAdderFP8_Hard_15

//FPAny_LNS_Mult_Comp_DnSp_12 replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard_12 replaced by TernaryAdderFP8_Hard_15

//FPAny_LNS_Mult_Comp_DnSp_13 replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard_13 replaced by TernaryAdderFP8_Hard_15

//FPAny_LNS_Mult_Comp_DnSp_14 replaced by FPAny_LNS_Mult_Comp_DnSp_15

//TernaryAdderFP8_Hard_14 replaced by TernaryAdderFP8_Hard_15

module FPAny_LNS_Mult_Comp_DnSp_15 (
  input  wire [2:0]    MantX,
  input  wire [2:0]    MantY,
  output wire [0:0]    MantComp
);

  wire                LUTs_0_O;
  wire       [5:0]    LUT_input;

  LUT6_Hard_15 LUTs_0 (
    .I (LUT_input[5:0]), //i
    .O (LUTs_0_O      )  //o
  );
  assign LUT_input = {MantY,MantX};
  assign MantComp[0] = LUTs_0_O;

endmodule

module TernaryAdderFP8_Hard_15 (
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

//LUT6_Hard replaced by LUT6_Hard_15

//LUT6_Hard_1 replaced by LUT6_Hard_15

//LUT6_Hard_2 replaced by LUT6_Hard_15

//LUT6_Hard_3 replaced by LUT6_Hard_15

//LUT6_Hard_4 replaced by LUT6_Hard_15

//LUT6_Hard_5 replaced by LUT6_Hard_15

//LUT6_Hard_6 replaced by LUT6_Hard_15

//LUT6_Hard_7 replaced by LUT6_Hard_15

//LUT6_Hard_8 replaced by LUT6_Hard_15

//LUT6_Hard_9 replaced by LUT6_Hard_15

//LUT6_Hard_10 replaced by LUT6_Hard_15

//LUT6_Hard_11 replaced by LUT6_Hard_15

//LUT6_Hard_12 replaced by LUT6_Hard_15

//LUT6_Hard_13 replaced by LUT6_Hard_15

//LUT6_Hard_14 replaced by LUT6_Hard_15

module LUT6_Hard_15 (
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
    .INIT (64'h000c3e3e7c783000)
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
