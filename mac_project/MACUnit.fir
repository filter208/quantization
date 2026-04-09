FIRRTL version 1.2.0
circuit MACUnit :
  module MACUnit :
    input clock : Clock
    input reset : UInt<1>
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

