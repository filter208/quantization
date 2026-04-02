FIRRTL version 1.2.0
circuit MACUnit :
  module MACUnit :
    input clock : Clock
    input reset : UInt<1>
    output io : { flip a : UInt<8>, flip b : UInt<8>, flip clear : UInt<1>, out : UInt<32>} @[src/main/scala/MACUnit.scala 7:14]

    reg accReg : UInt<32>, clock with :
      reset => (reset, UInt<32>("h0")) @[src/main/scala/MACUnit.scala 16:23]
    when io.clear : @[src/main/scala/MACUnit.scala 19:18]
      accReg <= UInt<1>("h0") @[src/main/scala/MACUnit.scala 22:12]
    else :
      node _accReg_T = mul(io.a, io.b) @[src/main/scala/MACUnit.scala 25:30]
      node _accReg_T_1 = add(accReg, _accReg_T) @[src/main/scala/MACUnit.scala 25:22]
      node _accReg_T_2 = tail(_accReg_T_1, 1) @[src/main/scala/MACUnit.scala 25:22]
      accReg <= _accReg_T_2 @[src/main/scala/MACUnit.scala 25:12]
    io.out <= accReg @[src/main/scala/MACUnit.scala 29:10]

