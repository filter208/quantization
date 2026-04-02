FIRRTL version 1.2.0
circuit MACUnit :
  module MACUnit :
    input clock : Clock
    input reset : UInt<1>
    input io_a : UInt<8> @[src/main/scala/MACUnit.scala 7:14]
    input io_b : UInt<8> @[src/main/scala/MACUnit.scala 7:14]
    input io_clear : UInt<1> @[src/main/scala/MACUnit.scala 7:14]
    output io_out : UInt<32> @[src/main/scala/MACUnit.scala 7:14]

    reg accReg : UInt<32>, clock with :
      reset => (UInt<1>("h0"), accReg) @[src/main/scala/MACUnit.scala 16:23]
    node _accReg_T = mul(io_a, io_b) @[src/main/scala/MACUnit.scala 25:30]
    node _accReg_T_1 = add(accReg, _accReg_T) @[src/main/scala/MACUnit.scala 25:22]
    node _accReg_T_2 = tail(_accReg_T_1, 1) @[src/main/scala/MACUnit.scala 25:22]
    node _GEN_0 = mux(io_clear, UInt<1>("h0"), _accReg_T_2) @[src/main/scala/MACUnit.scala 19:18 22:12 25:12]
    io_out <= accReg @[src/main/scala/MACUnit.scala 29:10]
    accReg <= mux(reset, UInt<32>("h0"), _GEN_0) @[src/main/scala/MACUnit.scala 16:{23,23}]
