module MACUnit(
  input         clock,
  input         reset,
  input  [7:0]  io_a, // @[src/main/scala/MACUnit.scala 7:14]
  input  [7:0]  io_b, // @[src/main/scala/MACUnit.scala 7:14]
  input         io_clear, // @[src/main/scala/MACUnit.scala 7:14]
  output [31:0] io_out // @[src/main/scala/MACUnit.scala 7:14]
);
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_0;
  reg [31:0] _RAND_1;
`endif // RANDOMIZE_REG_INIT
  reg [31:0] accReg; // @[src/main/scala/MACUnit.scala 16:23]
  reg [15:0] mulResult; // @[src/main/scala/MACUnit.scala 18:26]
  wire [31:0] _GEN_1 = {{16'd0}, mulResult}; // @[src/main/scala/MACUnit.scala 26:22]
  wire [31:0] _accReg_T_1 = accReg + _GEN_1; // @[src/main/scala/MACUnit.scala 26:22]
  assign io_out = accReg; // @[src/main/scala/MACUnit.scala 30:10]
  always @(posedge clock) begin
    if (reset) begin // @[src/main/scala/MACUnit.scala 16:23]
      accReg <= 32'h0; // @[src/main/scala/MACUnit.scala 16:23]
    end else if (io_clear) begin // @[src/main/scala/MACUnit.scala 20:18]
      accReg <= 32'h0; // @[src/main/scala/MACUnit.scala 23:12]
    end else begin
      accReg <= _accReg_T_1; // @[src/main/scala/MACUnit.scala 26:12]
    end
    mulResult <= io_a * io_b; // @[src/main/scala/MACUnit.scala 18:32]
  end
// Register and memory initialization
`ifdef RANDOMIZE_GARBAGE_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_INVALID_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_REG_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_MEM_INIT
`define RANDOMIZE
`endif
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE_MEM_INIT
  integer initvar;
`endif
`ifndef SYNTHESIS
`ifdef FIRRTL_BEFORE_INITIAL
`FIRRTL_BEFORE_INITIAL
`endif
initial begin
  `ifdef RANDOMIZE
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      `ifdef RANDOMIZE_DELAY
        #`RANDOMIZE_DELAY begin end
      `else
        #0.002 begin end
      `endif
    `endif
`ifdef RANDOMIZE_REG_INIT
  _RAND_0 = {1{`RANDOM}};
  accReg = _RAND_0[31:0];
  _RAND_1 = {1{`RANDOM}};
  mulResult = _RAND_1[15:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
