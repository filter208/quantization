import chisel3._

// 这是一个模块类，代表你要造的一个芯片模块
class MACUnit extends Module {
  
  // io (Input/Output) 就是你这块芯片外面的“引脚”
  val io = IO(new Bundle {
    val a     = Input(UInt(8.W))    // 8根线的输入 A (8-bit 无符号整数)
    val b     = Input(UInt(8.W))    // 8根线的输入 B
    val clear = Input(Bool())       // 1根线的输入：清零开关 (布尔值：真/假)
    val out   = Output(UInt(32.W))  // 32根线的输出 (用来放结果，防止溢出)
  })

  // val 的意思是声明一个不可变的名字。
  // RegInit 声明了一个有记忆功能的“寄存器”，括号里的 0.U(32.W) 意思是它初始值是 32位宽的 0。
  val accReg = RegInit(0.U(32.W))

  // when 就像硬件里的多路选择器 (MUX)
  when(io.clear) {
    // := 在这里不是数学里的等于，它是“硬件连线”的意思！
    // 这句话的意思是：当 clear 通电时，把 0 连到寄存器的输入端。
    accReg := 0.U
  } .otherwise {
    // 否则，把 (当前的寄存器值 + A乘B的结果) 连到寄存器的输入端。
    accReg := accReg + (io.a * io.b)
  }

  // 最后，把寄存器里的值，连一根线到芯片的输出引脚上
  io.out := accReg
}

// ==========================================
// 下面这一小段是“启动器”，作用是把上面的代码翻译成底层的 Verilog 硬件语言
object Main extends App {
  (new chisel3.stage.ChiselStage).emitVerilog(new MACUnit())
}