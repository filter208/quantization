// Generator : SpinalHDL v1.10.2    git head : 279867b771fb50fc0aec21d8a20d8fdad0f87e3f
// Component : MyAdder
// Git hash  : cde6059b17f4382be3f53ff8c3068e3b72d7f1db

`timescale 1ns/1ps

module MyAdder (
  input  wire [7:0]    io_a,
  input  wire [7:0]    io_b,
  output wire [7:0]    io_result
);


  assign io_result = (io_a + io_b);

endmodule
