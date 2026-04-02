import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

// 继承 AnyFlatSpec 和 ChiselScalatestTester，这是 Scala 测试框架的标准写法
class MACUnitTest extends AnyFlatSpec with ChiselScalatestTester {

  "MACUnit" should "correctly multiply and accumulate" in {
    // test(...) 会自动为你实例化被测模块（DUT: Device Under Test），并自动连接隐式时钟
    test(new MACUnit) { dut =>
      
      // 1. 初始化：清零寄存器
      // poke 意思是“给输入端口强制赋值”（相当于 Verilog 里的给 reg 赋值）
      dut.io.clear.poke(true.B) 
      // step(1) 意思是“时钟往前走一个周期”（相当于 Verilog 里的 #10 或者 @(posedge clk)）
      dut.clock.step(1)         

      // 2. 第一拍：关闭清零，输入 3 * 4
      dut.io.clear.poke(false.B)
      dut.io.a.poke(3.U)
      dut.io.b.poke(4.U)
      dut.clock.step(1) // 走一个时钟，硬件内部算出 3*4=12，并在下一个上升沿打入寄存器

      // expect 意思是“断言/检查输出端口的值”。如果不是 12，测试就会在此报错停止！
      dut.io.out.expect(12.U)

      // 3. 第二拍：保持累加，输入 5 * 2
      dut.io.a.poke(5.U)
      dut.io.b.poke(2.U)
      dut.clock.step(1) // 走一个时钟，算出 5*2=10，加上之前的 12，结果变成 22

      // 检查累加结果是否正确
      dut.io.out.expect(22.U)

      // 4. 第三拍：测试清零功能
      dut.io.clear.poke(true.B)
      dut.clock.step(1)
      
      // 检查是否成功清零
      dut.io.out.expect(0.U)
      
      println("🎉 所有硬件逻辑测试通过！")
    }
  }
}