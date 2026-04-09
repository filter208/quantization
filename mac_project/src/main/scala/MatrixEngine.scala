import chisel3._
import chisel3.util._
import scala.language.reflectiveCalls

class MatrixEngine extends Module {
  // Bundle：打包各种引脚，相当于 Verilog 里的 port list
  val io = IO(new Bundle {
    val start  = Input(Bool())      
    val done   = Output(Bool())     
    val aIn    = Input(UInt(8.W))   
    val bIn    = Input(UInt(8.W))   
    val result = Output(UInt(32.W)) 
  })

  // Module(...)：实例化其他硬件模块（芯片）
  val mac = Module(new MACUnit)
  val bufA = Module(new Buffer(256))
  val bufB = Module(new Buffer(256))

  // Enum(4)：生成 4 个互斥的硬件状态码
  val sIdle :: sLoad :: sCompute :: sDone :: Nil = Enum(4)
  // RegInit：带有初始值的寄存器。通电瞬间复位为括号内的值。
  val state = RegInit(sIdle)
  val count = RegInit(0.U(8.W))

  // ==========================================
  // 基础连线与默认值
  // ==========================================
  // := 表示硬件连线。左边是接收端，右边是发送端。
  bufA.io.we   := false.B
  bufB.io.we   := false.B
  bufA.io.addr := count
  bufB.io.addr := count
  bufA.io.din  := 0.U
  bufB.io.din  := 0.U

  // ==========================================
  // 流水线保护逻辑
  // ==========================================
  // readingValid 是一个组合逻辑信号（Wire），判断当前是否在合法读取区间
  val readingValid = (state === sCompute) && (count < 16.U)
  
  // RegNext(in, init)：延迟寄存器，带初始值 false.B。
  // 它把 readingValid 往后推迟了 1 个时钟周期，匹配 Buffer 的读取延迟。
  val mulValid = RegNext(readingValid, false.B)

  // Mux(条件, 真值, 假值)：硬件的多路选择器。
  // 拦截器：当 mulValid 有效时，导通 Buffer 数据；无效时，强制灌入 0。
  mac.io.a := Mux(mulValid, bufA.io.dout, 0.U)
  mac.io.b := Mux(mulValid, bufB.io.dout, 0.U)
  
  // 逻辑或 (||) 生成一个或门：在空闲或加载时，持续压住 MAC 的复位键
  mac.io.clear := (state === sIdle || state === sLoad)

  // 顶层输出映射
  io.done   := (state === sDone)
  io.result := mac.io.out

  // ==========================================
  // 状态机 (FSM) 核心控制逻辑
  // ==========================================
  // switch 会被综合器翻译成 Verilog 的 case 语句
  switch(state) {
    is(sIdle) {
      count := 0.U
      when(io.start) { state := sLoad }
    }
    
    is(sLoad) {
      bufA.io.we   := true.B            
      bufB.io.we   := true.B
      bufA.io.din  := io.aIn            
      bufB.io.din  := io.bIn 
      
      count := count + 1.U
      when(count === 15.U) { 
        state := sCompute               
        count := 0.U 
      }
    }
    
    is(sCompute) {
      count := count + 1.U
      // 等待计数器达到 18，确保流水线中最后两拍数据完成计算
      when(count === 18.U) { 
        state := sDone 
      }
    }
    
    is(sDone) {
      // 严格的握手协议：只有当外界撤销 start 时，才跳回空闲状态
      when(!io.start) { state := sIdle } 
    }
  }
}