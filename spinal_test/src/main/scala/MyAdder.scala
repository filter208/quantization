import spinal.core._

// 定义一个硬件模块：8位加法器
class MyAdder extends Component {
  val io = new Bundle {
    val a = in UInt(8 bits)
    val b = in UInt(8 bits)
    val result = out UInt(8 bits)
  }
  // 硬件逻辑：将 a 和 b 相加赋值给 result
  io.result := io.a + io.b
}

// 运行入口：将上面的逻辑翻译成 Verilog
object MyAdderVerilog {
  def main(args: Array[String]): Unit = {
    SpinalVerilog(new MyAdder)
    println("🎉 恭喜！Verilog 底层硬件代码生成成功！")
  }
}
