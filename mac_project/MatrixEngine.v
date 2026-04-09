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
module Buffer(
  input        clock,
  input        io_we, // @[src/main/scala/Buffer.scala 5:14]
  input  [7:0] io_addr, // @[src/main/scala/Buffer.scala 5:14]
  input  [7:0] io_din, // @[src/main/scala/Buffer.scala 5:14]
  output [7:0] io_dout // @[src/main/scala/Buffer.scala 5:14]
);
`ifdef RANDOMIZE_MEM_INIT
  reg [31:0] _RAND_0;
`endif // RANDOMIZE_MEM_INIT
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_1;
  reg [31:0] _RAND_2;
`endif // RANDOMIZE_REG_INIT
  reg [7:0] mem [0:255]; // @[src/main/scala/Buffer.scala 14:24]
  wire  mem_io_dout_MPORT_en; // @[src/main/scala/Buffer.scala 14:24]
  wire [7:0] mem_io_dout_MPORT_addr; // @[src/main/scala/Buffer.scala 14:24]
  wire [7:0] mem_io_dout_MPORT_data; // @[src/main/scala/Buffer.scala 14:24]
  wire [7:0] mem_MPORT_data; // @[src/main/scala/Buffer.scala 14:24]
  wire [7:0] mem_MPORT_addr; // @[src/main/scala/Buffer.scala 14:24]
  wire  mem_MPORT_mask; // @[src/main/scala/Buffer.scala 14:24]
  wire  mem_MPORT_en; // @[src/main/scala/Buffer.scala 14:24]
  reg  mem_io_dout_MPORT_en_pipe_0;
  reg [7:0] mem_io_dout_MPORT_addr_pipe_0;
  assign mem_io_dout_MPORT_en = mem_io_dout_MPORT_en_pipe_0;
  assign mem_io_dout_MPORT_addr = mem_io_dout_MPORT_addr_pipe_0;
  assign mem_io_dout_MPORT_data = mem[mem_io_dout_MPORT_addr]; // @[src/main/scala/Buffer.scala 14:24]
  assign mem_MPORT_data = io_din;
  assign mem_MPORT_addr = io_addr;
  assign mem_MPORT_mask = 1'h1;
  assign mem_MPORT_en = io_we;
  assign io_dout = mem_io_dout_MPORT_data; // @[src/main/scala/Buffer.scala 23:11]
  always @(posedge clock) begin
    if (mem_MPORT_en & mem_MPORT_mask) begin
      mem[mem_MPORT_addr] <= mem_MPORT_data; // @[src/main/scala/Buffer.scala 14:24]
    end
    mem_io_dout_MPORT_en_pipe_0 <= 1'h1;
    if (1'h1) begin
      mem_io_dout_MPORT_addr_pipe_0 <= io_addr;
    end
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
`ifdef RANDOMIZE_MEM_INIT
  _RAND_0 = {1{`RANDOM}};
  for (initvar = 0; initvar < 256; initvar = initvar+1)
    mem[initvar] = _RAND_0[7:0];
`endif // RANDOMIZE_MEM_INIT
`ifdef RANDOMIZE_REG_INIT
  _RAND_1 = {1{`RANDOM}};
  mem_io_dout_MPORT_en_pipe_0 = _RAND_1[0:0];
  _RAND_2 = {1{`RANDOM}};
  mem_io_dout_MPORT_addr_pipe_0 = _RAND_2[7:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
module MatrixEngine(
  input         clock,
  input         reset,
  input         io_start, // @[src/main/scala/MatrixEngine.scala 7:14]
  output        io_done, // @[src/main/scala/MatrixEngine.scala 7:14]
  input  [7:0]  io_aIn, // @[src/main/scala/MatrixEngine.scala 7:14]
  input  [7:0]  io_bIn, // @[src/main/scala/MatrixEngine.scala 7:14]
  output [31:0] io_result // @[src/main/scala/MatrixEngine.scala 7:14]
);
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_0;
  reg [31:0] _RAND_1;
  reg [31:0] _RAND_2;
`endif // RANDOMIZE_REG_INIT
  wire  mac_clock; // @[src/main/scala/MatrixEngine.scala 16:19]
  wire  mac_reset; // @[src/main/scala/MatrixEngine.scala 16:19]
  wire [7:0] mac_io_a; // @[src/main/scala/MatrixEngine.scala 16:19]
  wire [7:0] mac_io_b; // @[src/main/scala/MatrixEngine.scala 16:19]
  wire  mac_io_clear; // @[src/main/scala/MatrixEngine.scala 16:19]
  wire [31:0] mac_io_out; // @[src/main/scala/MatrixEngine.scala 16:19]
  wire  bufA_clock; // @[src/main/scala/MatrixEngine.scala 17:20]
  wire  bufA_io_we; // @[src/main/scala/MatrixEngine.scala 17:20]
  wire [7:0] bufA_io_addr; // @[src/main/scala/MatrixEngine.scala 17:20]
  wire [7:0] bufA_io_din; // @[src/main/scala/MatrixEngine.scala 17:20]
  wire [7:0] bufA_io_dout; // @[src/main/scala/MatrixEngine.scala 17:20]
  wire  bufB_clock; // @[src/main/scala/MatrixEngine.scala 18:20]
  wire  bufB_io_we; // @[src/main/scala/MatrixEngine.scala 18:20]
  wire [7:0] bufB_io_addr; // @[src/main/scala/MatrixEngine.scala 18:20]
  wire [7:0] bufB_io_din; // @[src/main/scala/MatrixEngine.scala 18:20]
  wire [7:0] bufB_io_dout; // @[src/main/scala/MatrixEngine.scala 18:20]
  reg [1:0] state; // @[src/main/scala/MatrixEngine.scala 23:22]
  reg [7:0] count; // @[src/main/scala/MatrixEngine.scala 24:22]
  wire  readingValid = state == 2'h2 & count < 8'h10; // @[src/main/scala/MatrixEngine.scala 41:43]
  reg  mulValid; // @[src/main/scala/MatrixEngine.scala 45:25]
  wire [7:0] _count_T_1 = count + 8'h1; // @[src/main/scala/MatrixEngine.scala 75:22]
  wire [1:0] _GEN_3 = count == 8'h12 ? 2'h3 : state; // @[src/main/scala/MatrixEngine.scala 85:28 86:15 23:22]
  wire [1:0] _GEN_4 = ~io_start ? 2'h0 : state; // @[src/main/scala/MatrixEngine.scala 23:22 92:{23,31}]
  wire [1:0] _GEN_5 = 2'h3 == state ? _GEN_4 : state; // @[src/main/scala/MatrixEngine.scala 63:17 23:22]
  wire [7:0] _GEN_9 = 2'h1 == state ? io_aIn : 8'h0; // @[src/main/scala/MatrixEngine.scala 34:16 63:17 72:20]
  wire [7:0] _GEN_10 = 2'h1 == state ? io_bIn : 8'h0; // @[src/main/scala/MatrixEngine.scala 35:16 63:17 73:20]
  MACUnit mac ( // @[src/main/scala/MatrixEngine.scala 16:19]
    .clock(mac_clock),
    .reset(mac_reset),
    .io_a(mac_io_a),
    .io_b(mac_io_b),
    .io_clear(mac_io_clear),
    .io_out(mac_io_out)
  );
  Buffer bufA ( // @[src/main/scala/MatrixEngine.scala 17:20]
    .clock(bufA_clock),
    .io_we(bufA_io_we),
    .io_addr(bufA_io_addr),
    .io_din(bufA_io_din),
    .io_dout(bufA_io_dout)
  );
  Buffer bufB ( // @[src/main/scala/MatrixEngine.scala 18:20]
    .clock(bufB_clock),
    .io_we(bufB_io_we),
    .io_addr(bufB_io_addr),
    .io_din(bufB_io_din),
    .io_dout(bufB_io_dout)
  );
  assign io_done = state == 2'h3; // @[src/main/scala/MatrixEngine.scala 56:23]
  assign io_result = mac_io_out; // @[src/main/scala/MatrixEngine.scala 57:13]
  assign mac_clock = clock;
  assign mac_reset = reset;
  assign mac_io_a = mulValid ? bufA_io_dout : 8'h0; // @[src/main/scala/MatrixEngine.scala 49:18]
  assign mac_io_b = mulValid ? bufB_io_dout : 8'h0; // @[src/main/scala/MatrixEngine.scala 50:18]
  assign mac_io_clear = state == 2'h0 | state == 2'h1; // @[src/main/scala/MatrixEngine.scala 53:36]
  assign bufA_clock = clock;
  assign bufA_io_we = 2'h0 == state ? 1'h0 : 2'h1 == state; // @[src/main/scala/MatrixEngine.scala 30:16 63:17]
  assign bufA_io_addr = count; // @[src/main/scala/MatrixEngine.scala 32:16]
  assign bufA_io_din = 2'h0 == state ? 8'h0 : _GEN_9; // @[src/main/scala/MatrixEngine.scala 34:16 63:17]
  assign bufB_clock = clock;
  assign bufB_io_we = 2'h0 == state ? 1'h0 : 2'h1 == state; // @[src/main/scala/MatrixEngine.scala 30:16 63:17]
  assign bufB_io_addr = count; // @[src/main/scala/MatrixEngine.scala 33:16]
  assign bufB_io_din = 2'h0 == state ? 8'h0 : _GEN_10; // @[src/main/scala/MatrixEngine.scala 35:16 63:17]
  always @(posedge clock) begin
    if (reset) begin // @[src/main/scala/MatrixEngine.scala 23:22]
      state <= 2'h0; // @[src/main/scala/MatrixEngine.scala 23:22]
    end else if (2'h0 == state) begin // @[src/main/scala/MatrixEngine.scala 63:17]
      if (io_start) begin // @[src/main/scala/MatrixEngine.scala 66:22]
        state <= 2'h1; // @[src/main/scala/MatrixEngine.scala 66:30]
      end
    end else if (2'h1 == state) begin // @[src/main/scala/MatrixEngine.scala 63:17]
      if (count == 8'hf) begin // @[src/main/scala/MatrixEngine.scala 76:28]
        state <= 2'h2; // @[src/main/scala/MatrixEngine.scala 77:15]
      end
    end else if (2'h2 == state) begin // @[src/main/scala/MatrixEngine.scala 63:17]
      state <= _GEN_3;
    end else begin
      state <= _GEN_5;
    end
    if (reset) begin // @[src/main/scala/MatrixEngine.scala 24:22]
      count <= 8'h0; // @[src/main/scala/MatrixEngine.scala 24:22]
    end else if (2'h0 == state) begin // @[src/main/scala/MatrixEngine.scala 63:17]
      count <= 8'h0; // @[src/main/scala/MatrixEngine.scala 65:13]
    end else if (2'h1 == state) begin // @[src/main/scala/MatrixEngine.scala 63:17]
      if (count == 8'hf) begin // @[src/main/scala/MatrixEngine.scala 76:28]
        count <= 8'h0; // @[src/main/scala/MatrixEngine.scala 78:15]
      end else begin
        count <= _count_T_1; // @[src/main/scala/MatrixEngine.scala 75:13]
      end
    end else if (2'h2 == state) begin // @[src/main/scala/MatrixEngine.scala 63:17]
      count <= _count_T_1; // @[src/main/scala/MatrixEngine.scala 83:13]
    end
    if (reset) begin // @[src/main/scala/MatrixEngine.scala 45:25]
      mulValid <= 1'h0; // @[src/main/scala/MatrixEngine.scala 45:25]
    end else begin
      mulValid <= readingValid; // @[src/main/scala/MatrixEngine.scala 45:25]
    end
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
  state = _RAND_0[1:0];
  _RAND_1 = {1{`RANDOM}};
  count = _RAND_1[7:0];
  _RAND_2 = {1{`RANDOM}};
  mulValid = _RAND_2[0:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
