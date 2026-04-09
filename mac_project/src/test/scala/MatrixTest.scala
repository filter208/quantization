import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.language.reflectiveCalls

class MatrixTest extends AnyFlatSpec with ChiselScalatestTester {

  "MatrixEngine" should "correctly compute 16 elements with proper handshake" in {
    test(new MatrixEngine) { dut =>
      
      // --- 1. 硬件复位与初始化 ---
      dut.io.start.poke(false.B)
      dut.clock.step(1)

      // --- 2. 模拟外部 DDR 加载数据 (sLoad) ---
      println("--- 开始从外部接口搬运数据 ---")
      dut.io.start.poke(true.B) // 发送启动脉冲
      dut.clock.step(1)         // 等待状态机跳入 sLoad

      for (i <- 0 until 16) {
        dut.io.aIn.poke(2.U)
        dut.io.bIn.poke(3.U)
        dut.clock.step(1)
      }

      // --- 3. 硬件进入高速计算 (sCompute) ---
      println("--- 搬运完毕，开始纯内部流水线计算 ---")
      
      // ✅ 修复点：千万不要在这里提前撤销 start！
      // 保持 start 为 true，等待硬件把 done 信号交给我们。
      dut.io.aIn.poke(99.U) // 制造脏数据干扰，验证 Buffer 的隔离性
      
      for (cycle <- 0 until 20) {
        dut.clock.step(1)
        val currentResult = dut.io.result.peek().litValue
        println(s"周期 $cycle | 累加结果: $currentResult")
      }

      // --- 4. 握手与自动化质检 ---
      // 此时 20 拍走完，硬件必然已经停在 sDone 状态等待指示
      dut.io.result.expect(96.U)
      
      // ✅ 修复点：此时检查 done，它一定是 true.B
      dut.io.done.expect(true.B) 
      println("✅ 硬件已发出计算完成信号 (done = 1)")

      // --- 5. 完成握手，释放状态机 ---
      dut.io.start.poke(false.B) // 主机收到 done，撤销 start
      dut.clock.step(1)          // 硬件感受到 start 变低，跳回 sIdle
      dut.io.done.expect(false.B) // 检查 done 是否随之熄灭
      
      println("🎉 握手结束！测试完美通过。")
    }
  }
}