FIRRTL version 1.2.0
circuit MatrixEngine :
  module MACUnit :
    input clock : Clock
    input reset : Reset
    output io : { flip a : UInt<8>, flip b : UInt<8>, flip clear : UInt<1>, out : UInt<32>} @[src/main/scala/MACUnit.scala 7:14]

    reg accReg : UInt<32>, clock with :
      reset => (reset, UInt<32>("h0")) @[src/main/scala/MACUnit.scala 16:23]
    node _mulResult_T = mul(io.a, io.b) @[src/main/scala/MACUnit.scala 18:32]
    reg mulResult : UInt, clock with :
      reset => (UInt<1>("h0"), mulResult) @[src/main/scala/MACUnit.scala 18:26]
    mulResult <= _mulResult_T @[src/main/scala/MACUnit.scala 18:26]
    when io.clear : @[src/main/scala/MACUnit.scala 20:18]
      accReg <= UInt<1>("h0") @[src/main/scala/MACUnit.scala 23:12]
    else :
      node _accReg_T = add(accReg, mulResult) @[src/main/scala/MACUnit.scala 26:22]
      node _accReg_T_1 = tail(_accReg_T, 1) @[src/main/scala/MACUnit.scala 26:22]
      accReg <= _accReg_T_1 @[src/main/scala/MACUnit.scala 26:12]
    io.out <= accReg @[src/main/scala/MACUnit.scala 30:10]

  module Buffer :
    input clock : Clock
    input reset : Reset
    output io : { flip we : UInt<1>, flip addr : UInt<8>, flip din : UInt<8>, dout : UInt<8>} @[src/main/scala/Buffer.scala 5:14]

    smem mem : UInt<8> [256] @[src/main/scala/Buffer.scala 14:24]
    when io.we : @[src/main/scala/Buffer.scala 17:15]
      write mport MPORT = mem[io.addr], clock
      MPORT <= io.din
    wire _io_dout_WIRE : UInt @[src/main/scala/Buffer.scala 23:22]
    _io_dout_WIRE is invalid @[src/main/scala/Buffer.scala 23:22]
    when UInt<1>("h1") : @[src/main/scala/Buffer.scala 23:22]
      _io_dout_WIRE <= io.addr @[src/main/scala/Buffer.scala 23:22]
      node _io_dout_T = or(_io_dout_WIRE, UInt<8>("h0")) @[src/main/scala/Buffer.scala 23:22]
      node _io_dout_T_1 = bits(_io_dout_T, 7, 0) @[src/main/scala/Buffer.scala 23:22]
      read mport io_dout_MPORT = mem[_io_dout_T_1], clock @[src/main/scala/Buffer.scala 23:22]
    io.dout <= io_dout_MPORT @[src/main/scala/Buffer.scala 23:11]

  module Buffer_1 :
    input clock : Clock
    input reset : Reset
    output io : { flip we : UInt<1>, flip addr : UInt<8>, flip din : UInt<8>, dout : UInt<8>} @[src/main/scala/Buffer.scala 5:14]

    smem mem : UInt<8> [256] @[src/main/scala/Buffer.scala 14:24]
    when io.we : @[src/main/scala/Buffer.scala 17:15]
      write mport MPORT = mem[io.addr], clock
      MPORT <= io.din
    wire _io_dout_WIRE : UInt @[src/main/scala/Buffer.scala 23:22]
    _io_dout_WIRE is invalid @[src/main/scala/Buffer.scala 23:22]
    when UInt<1>("h1") : @[src/main/scala/Buffer.scala 23:22]
      _io_dout_WIRE <= io.addr @[src/main/scala/Buffer.scala 23:22]
      node _io_dout_T = or(_io_dout_WIRE, UInt<8>("h0")) @[src/main/scala/Buffer.scala 23:22]
      node _io_dout_T_1 = bits(_io_dout_T, 7, 0) @[src/main/scala/Buffer.scala 23:22]
      read mport io_dout_MPORT = mem[_io_dout_T_1], clock @[src/main/scala/Buffer.scala 23:22]
    io.dout <= io_dout_MPORT @[src/main/scala/Buffer.scala 23:11]

  module MatrixEngine :
    input clock : Clock
    input reset : UInt<1>
    output io : { flip start : UInt<1>, done : UInt<1>, flip aIn : UInt<8>, flip bIn : UInt<8>, result : UInt<32>} @[src/main/scala/MatrixEngine.scala 7:14]

    inst mac of MACUnit @[src/main/scala/MatrixEngine.scala 16:19]
    mac.clock <= clock
    mac.reset <= reset
    inst bufA of Buffer @[src/main/scala/MatrixEngine.scala 17:20]
    bufA.clock <= clock
    bufA.reset <= reset
    inst bufB of Buffer_1 @[src/main/scala/MatrixEngine.scala 18:20]
    bufB.clock <= clock
    bufB.reset <= reset
    reg state : UInt<2>, clock with :
      reset => (reset, UInt<2>("h0")) @[src/main/scala/MatrixEngine.scala 23:22]
    reg count : UInt<8>, clock with :
      reset => (reset, UInt<8>("h0")) @[src/main/scala/MatrixEngine.scala 24:22]
    bufA.io.we <= UInt<1>("h0") @[src/main/scala/MatrixEngine.scala 30:16]
    bufB.io.we <= UInt<1>("h0") @[src/main/scala/MatrixEngine.scala 31:16]
    bufA.io.addr <= count @[src/main/scala/MatrixEngine.scala 32:16]
    bufB.io.addr <= count @[src/main/scala/MatrixEngine.scala 33:16]
    bufA.io.din <= UInt<1>("h0") @[src/main/scala/MatrixEngine.scala 34:16]
    bufB.io.din <= UInt<1>("h0") @[src/main/scala/MatrixEngine.scala 35:16]
    node _readingValid_T = eq(state, UInt<2>("h2")) @[src/main/scala/MatrixEngine.scala 41:29]
    node _readingValid_T_1 = lt(count, UInt<5>("h10")) @[src/main/scala/MatrixEngine.scala 41:53]
    node readingValid = and(_readingValid_T, _readingValid_T_1) @[src/main/scala/MatrixEngine.scala 41:43]
    reg mulValid : UInt<1>, clock with :
      reset => (reset, UInt<1>("h0")) @[src/main/scala/MatrixEngine.scala 45:25]
    mulValid <= readingValid @[src/main/scala/MatrixEngine.scala 45:25]
    node _mac_io_a_T = mux(mulValid, bufA.io.dout, UInt<1>("h0")) @[src/main/scala/MatrixEngine.scala 49:18]
    mac.io.a <= _mac_io_a_T @[src/main/scala/MatrixEngine.scala 49:12]
    node _mac_io_b_T = mux(mulValid, bufB.io.dout, UInt<1>("h0")) @[src/main/scala/MatrixEngine.scala 50:18]
    mac.io.b <= _mac_io_b_T @[src/main/scala/MatrixEngine.scala 50:12]
    node _mac_io_clear_T = eq(state, UInt<2>("h0")) @[src/main/scala/MatrixEngine.scala 53:26]
    node _mac_io_clear_T_1 = eq(state, UInt<2>("h1")) @[src/main/scala/MatrixEngine.scala 53:45]
    node _mac_io_clear_T_2 = or(_mac_io_clear_T, _mac_io_clear_T_1) @[src/main/scala/MatrixEngine.scala 53:36]
    mac.io.clear <= _mac_io_clear_T_2 @[src/main/scala/MatrixEngine.scala 53:16]
    node _io_done_T = eq(state, UInt<2>("h3")) @[src/main/scala/MatrixEngine.scala 56:23]
    io.done <= _io_done_T @[src/main/scala/MatrixEngine.scala 56:13]
    io.result <= mac.io.out @[src/main/scala/MatrixEngine.scala 57:13]
    node _T = eq(UInt<2>("h0"), state) @[src/main/scala/MatrixEngine.scala 63:17]
    when _T : @[src/main/scala/MatrixEngine.scala 63:17]
      count <= UInt<1>("h0") @[src/main/scala/MatrixEngine.scala 65:13]
      when io.start : @[src/main/scala/MatrixEngine.scala 66:22]
        state <= UInt<2>("h1") @[src/main/scala/MatrixEngine.scala 66:30]
    else :
      node _T_1 = eq(UInt<2>("h1"), state) @[src/main/scala/MatrixEngine.scala 63:17]
      when _T_1 : @[src/main/scala/MatrixEngine.scala 63:17]
        bufA.io.we <= UInt<1>("h1") @[src/main/scala/MatrixEngine.scala 70:20]
        bufB.io.we <= UInt<1>("h1") @[src/main/scala/MatrixEngine.scala 71:20]
        bufA.io.din <= io.aIn @[src/main/scala/MatrixEngine.scala 72:20]
        bufB.io.din <= io.bIn @[src/main/scala/MatrixEngine.scala 73:20]
        node _count_T = add(count, UInt<1>("h1")) @[src/main/scala/MatrixEngine.scala 75:22]
        node _count_T_1 = tail(_count_T, 1) @[src/main/scala/MatrixEngine.scala 75:22]
        count <= _count_T_1 @[src/main/scala/MatrixEngine.scala 75:13]
        node _T_2 = eq(count, UInt<4>("hf")) @[src/main/scala/MatrixEngine.scala 76:18]
        when _T_2 : @[src/main/scala/MatrixEngine.scala 76:28]
          state <= UInt<2>("h2") @[src/main/scala/MatrixEngine.scala 77:15]
          count <= UInt<1>("h0") @[src/main/scala/MatrixEngine.scala 78:15]
      else :
        node _T_3 = eq(UInt<2>("h2"), state) @[src/main/scala/MatrixEngine.scala 63:17]
        when _T_3 : @[src/main/scala/MatrixEngine.scala 63:17]
          node _count_T_2 = add(count, UInt<1>("h1")) @[src/main/scala/MatrixEngine.scala 83:22]
          node _count_T_3 = tail(_count_T_2, 1) @[src/main/scala/MatrixEngine.scala 83:22]
          count <= _count_T_3 @[src/main/scala/MatrixEngine.scala 83:13]
          node _T_4 = eq(count, UInt<5>("h12")) @[src/main/scala/MatrixEngine.scala 85:18]
          when _T_4 : @[src/main/scala/MatrixEngine.scala 85:28]
            state <= UInt<2>("h3") @[src/main/scala/MatrixEngine.scala 86:15]
        else :
          node _T_5 = eq(UInt<2>("h3"), state) @[src/main/scala/MatrixEngine.scala 63:17]
          when _T_5 : @[src/main/scala/MatrixEngine.scala 63:17]
            node _T_6 = eq(io.start, UInt<1>("h0")) @[src/main/scala/MatrixEngine.scala 92:12]
            when _T_6 : @[src/main/scala/MatrixEngine.scala 92:23]
              state <= UInt<2>("h0") @[src/main/scala/MatrixEngine.scala 92:31]


