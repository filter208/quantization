FIRRTL version 1.2.0
circuit MatrixEngine :
  module MACUnit :
    input clock : Clock
    input reset : UInt<1>
    input io_a : UInt<8> @[src/main/scala/MACUnit.scala 7:14]
    input io_b : UInt<8> @[src/main/scala/MACUnit.scala 7:14]
    input io_clear : UInt<1> @[src/main/scala/MACUnit.scala 7:14]
    output io_out : UInt<32> @[src/main/scala/MACUnit.scala 7:14]

    reg accReg : UInt<32>, clock with :
      reset => (UInt<1>("h0"), accReg) @[src/main/scala/MACUnit.scala 16:23]
    node _mulResult_T = mul(io_a, io_b) @[src/main/scala/MACUnit.scala 18:32]
    reg mulResult : UInt<16>, clock with :
      reset => (UInt<1>("h0"), mulResult) @[src/main/scala/MACUnit.scala 18:26]
    node _accReg_T = add(accReg, mulResult) @[src/main/scala/MACUnit.scala 26:22]
    node _accReg_T_1 = tail(_accReg_T, 1) @[src/main/scala/MACUnit.scala 26:22]
    node _GEN_0 = mux(io_clear, UInt<1>("h0"), _accReg_T_1) @[src/main/scala/MACUnit.scala 20:18 23:12 26:12]
    io_out <= accReg @[src/main/scala/MACUnit.scala 30:10]
    accReg <= mux(reset, UInt<32>("h0"), _GEN_0) @[src/main/scala/MACUnit.scala 16:{23,23}]
    mulResult <= _mulResult_T @[src/main/scala/MACUnit.scala 18:26]

  module Buffer :
    input clock : Clock
    input reset : UInt<1>
    input io_we : UInt<1> @[src/main/scala/Buffer.scala 5:14]
    input io_addr : UInt<8> @[src/main/scala/Buffer.scala 5:14]
    input io_din : UInt<8> @[src/main/scala/Buffer.scala 5:14]
    output io_dout : UInt<8> @[src/main/scala/Buffer.scala 5:14]

    mem mem : @[src/main/scala/Buffer.scala 14:24]
      data-type => UInt<8>
      depth => 256
      read-latency => 1
      write-latency => 1
      reader => io_dout_MPORT
      writer => MPORT
      read-under-write => undefined
    node _GEN_0 = validif(io_we, io_addr) @[src/main/scala/Buffer.scala 17:15]
    node _GEN_1 = validif(io_we, clock) @[src/main/scala/Buffer.scala 17:15]
    node _GEN_2 = mux(io_we, UInt<1>("h1"), UInt<1>("h0")) @[src/main/scala/Buffer.scala 17:15 14:24]
    node _GEN_3 = validif(io_we, UInt<1>("h1")) @[src/main/scala/Buffer.scala 17:15]
    node _GEN_4 = validif(io_we, io_din) @[src/main/scala/Buffer.scala 17:15]
    node _GEN_5 = validif(UInt<1>("h1"), io_addr) @[src/main/scala/Buffer.scala 23:{22,22}]
    node _io_dout_WIRE = _GEN_5 @[src/main/scala/Buffer.scala 23:22]
    node _io_dout_T = or(_io_dout_WIRE, UInt<8>("h0")) @[src/main/scala/Buffer.scala 23:22]
    node _io_dout_T_1 = bits(_io_dout_T, 7, 0) @[src/main/scala/Buffer.scala 23:22]
    node _GEN_6 = mux(UInt<1>("h1"), UInt<1>("h1"), UInt<1>("h0")) @[src/main/scala/Buffer.scala 23:{22,22} 14:24]
    node _GEN_7 = validif(UInt<1>("h1"), _io_dout_T_1) @[src/main/scala/Buffer.scala 23:{22,22}]
    node _GEN_8 = validif(UInt<1>("h1"), clock) @[src/main/scala/Buffer.scala 23:{22,22}]
    io_dout <= mem.io_dout_MPORT.data @[src/main/scala/Buffer.scala 23:11]
    mem.io_dout_MPORT.addr <= _GEN_7
    mem.io_dout_MPORT.en <= _GEN_6
    mem.io_dout_MPORT.clk <= _GEN_8
    mem.MPORT.addr <= _GEN_0
    mem.MPORT.en <= _GEN_2
    mem.MPORT.clk <= _GEN_1
    mem.MPORT.data <= _GEN_4
    mem.MPORT.mask <= _GEN_3

  module MatrixEngine :
    input clock : Clock
    input reset : UInt<1>
    input io_start : UInt<1> @[src/main/scala/MatrixEngine.scala 7:14]
    output io_done : UInt<1> @[src/main/scala/MatrixEngine.scala 7:14]
    input io_aIn : UInt<8> @[src/main/scala/MatrixEngine.scala 7:14]
    input io_bIn : UInt<8> @[src/main/scala/MatrixEngine.scala 7:14]
    output io_result : UInt<32> @[src/main/scala/MatrixEngine.scala 7:14]

    inst mac of MACUnit @[src/main/scala/MatrixEngine.scala 16:19]
    inst bufA of Buffer @[src/main/scala/MatrixEngine.scala 17:20]
    inst bufB of Buffer @[src/main/scala/MatrixEngine.scala 18:20]
    reg state : UInt<2>, clock with :
      reset => (UInt<1>("h0"), state) @[src/main/scala/MatrixEngine.scala 23:22]
    reg count : UInt<8>, clock with :
      reset => (UInt<1>("h0"), count) @[src/main/scala/MatrixEngine.scala 24:22]
    node _readingValid_T = eq(state, UInt<2>("h2")) @[src/main/scala/MatrixEngine.scala 41:29]
    node _readingValid_T_1 = lt(count, UInt<5>("h10")) @[src/main/scala/MatrixEngine.scala 41:53]
    node readingValid = and(_readingValid_T, _readingValid_T_1) @[src/main/scala/MatrixEngine.scala 41:43]
    reg mulValid : UInt<1>, clock with :
      reset => (UInt<1>("h0"), mulValid) @[src/main/scala/MatrixEngine.scala 45:25]
    node _mac_io_a_T = mux(mulValid, bufA.io_dout, UInt<1>("h0")) @[src/main/scala/MatrixEngine.scala 49:18]
    node _mac_io_b_T = mux(mulValid, bufB.io_dout, UInt<1>("h0")) @[src/main/scala/MatrixEngine.scala 50:18]
    node _mac_io_clear_T = eq(state, UInt<2>("h0")) @[src/main/scala/MatrixEngine.scala 53:26]
    node _mac_io_clear_T_1 = eq(state, UInt<2>("h1")) @[src/main/scala/MatrixEngine.scala 53:45]
    node _mac_io_clear_T_2 = or(_mac_io_clear_T, _mac_io_clear_T_1) @[src/main/scala/MatrixEngine.scala 53:36]
    node _io_done_T = eq(state, UInt<2>("h3")) @[src/main/scala/MatrixEngine.scala 56:23]
    node _T = eq(UInt<2>("h0"), state) @[src/main/scala/MatrixEngine.scala 63:17]
    node _GEN_0 = mux(io_start, UInt<2>("h1"), state) @[src/main/scala/MatrixEngine.scala 23:22 66:{22,30}]
    node _T_1 = eq(UInt<2>("h1"), state) @[src/main/scala/MatrixEngine.scala 63:17]
    node _count_T = add(count, UInt<1>("h1")) @[src/main/scala/MatrixEngine.scala 75:22]
    node _count_T_1 = tail(_count_T, 1) @[src/main/scala/MatrixEngine.scala 75:22]
    node _T_2 = eq(count, UInt<4>("hf")) @[src/main/scala/MatrixEngine.scala 76:18]
    node _GEN_1 = mux(_T_2, UInt<2>("h2"), state) @[src/main/scala/MatrixEngine.scala 76:28 77:15 23:22]
    node _GEN_2 = mux(_T_2, UInt<1>("h0"), _count_T_1) @[src/main/scala/MatrixEngine.scala 75:13 76:28 78:15]
    node _T_3 = eq(UInt<2>("h2"), state) @[src/main/scala/MatrixEngine.scala 63:17]
    node _count_T_2 = add(count, UInt<1>("h1")) @[src/main/scala/MatrixEngine.scala 83:22]
    node _count_T_3 = tail(_count_T_2, 1) @[src/main/scala/MatrixEngine.scala 83:22]
    node _T_4 = eq(count, UInt<5>("h12")) @[src/main/scala/MatrixEngine.scala 85:18]
    node _GEN_3 = mux(_T_4, UInt<2>("h3"), state) @[src/main/scala/MatrixEngine.scala 85:28 86:15 23:22]
    node _T_5 = eq(UInt<2>("h3"), state) @[src/main/scala/MatrixEngine.scala 63:17]
    node _T_6 = eq(io_start, UInt<1>("h0")) @[src/main/scala/MatrixEngine.scala 92:12]
    node _GEN_4 = mux(_T_6, UInt<2>("h0"), state) @[src/main/scala/MatrixEngine.scala 23:22 92:{23,31}]
    node _GEN_5 = mux(_T_5, _GEN_4, state) @[src/main/scala/MatrixEngine.scala 63:17 23:22]
    node _GEN_6 = mux(_T_3, _count_T_3, count) @[src/main/scala/MatrixEngine.scala 63:17 83:13 24:22]
    node _GEN_7 = mux(_T_3, _GEN_3, _GEN_5) @[src/main/scala/MatrixEngine.scala 63:17]
    node _GEN_8 = mux(_T_1, UInt<1>("h1"), UInt<1>("h0")) @[src/main/scala/MatrixEngine.scala 30:16 63:17 70:20]
    node _GEN_9 = mux(_T_1, io_aIn, UInt<1>("h0")) @[src/main/scala/MatrixEngine.scala 34:16 63:17 72:20]
    node _GEN_10 = mux(_T_1, io_bIn, UInt<1>("h0")) @[src/main/scala/MatrixEngine.scala 35:16 63:17 73:20]
    node _GEN_11 = mux(_T_1, _GEN_2, _GEN_6) @[src/main/scala/MatrixEngine.scala 63:17]
    node _GEN_12 = mux(_T_1, _GEN_1, _GEN_7) @[src/main/scala/MatrixEngine.scala 63:17]
    node _GEN_13 = mux(_T, UInt<1>("h0"), _GEN_11) @[src/main/scala/MatrixEngine.scala 63:17 65:13]
    node _GEN_14 = mux(_T, _GEN_0, _GEN_12) @[src/main/scala/MatrixEngine.scala 63:17]
    node _GEN_15 = mux(_T, UInt<1>("h0"), _GEN_8) @[src/main/scala/MatrixEngine.scala 30:16 63:17]
    node _GEN_16 = mux(_T, UInt<1>("h0"), _GEN_9) @[src/main/scala/MatrixEngine.scala 34:16 63:17]
    node _GEN_17 = mux(_T, UInt<1>("h0"), _GEN_10) @[src/main/scala/MatrixEngine.scala 35:16 63:17]
    io_done <= _io_done_T @[src/main/scala/MatrixEngine.scala 56:13]
    io_result <= mac.io_out @[src/main/scala/MatrixEngine.scala 57:13]
    mac.clock <= clock
    mac.reset <= reset
    mac.io_a <= _mac_io_a_T @[src/main/scala/MatrixEngine.scala 49:12]
    mac.io_b <= _mac_io_b_T @[src/main/scala/MatrixEngine.scala 50:12]
    mac.io_clear <= _mac_io_clear_T_2 @[src/main/scala/MatrixEngine.scala 53:16]
    bufA.clock <= clock
    bufA.reset <= reset
    bufA.io_we <= _GEN_15
    bufA.io_addr <= count @[src/main/scala/MatrixEngine.scala 32:16]
    bufA.io_din <= _GEN_16
    bufB.clock <= clock
    bufB.reset <= reset
    bufB.io_we <= _GEN_15
    bufB.io_addr <= count @[src/main/scala/MatrixEngine.scala 33:16]
    bufB.io_din <= _GEN_17
    state <= mux(reset, UInt<2>("h0"), _GEN_14) @[src/main/scala/MatrixEngine.scala 23:{22,22}]
    count <= mux(reset, UInt<8>("h0"), _GEN_13) @[src/main/scala/MatrixEngine.scala 24:{22,22}]
    mulValid <= mux(reset, UInt<1>("h0"), readingValid) @[src/main/scala/MatrixEngine.scala 45:{25,25,25}]
