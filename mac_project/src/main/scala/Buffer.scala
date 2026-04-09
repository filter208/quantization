import chisel3._

// 定义一个参数化的 Buffer 类，depth 表示缓存的大小（深度）
class Buffer(depth: Int) extends Module {
  val io = IO(new Bundle {
    val we    = Input(Bool())       // Write Enable: 写使能信号。为 1 时数据存入，为 0 时只读。
    val addr  = Input(UInt(8.W))    // Address: 地址线。决定数据存/取到哪个位置。
    val din   = Input(UInt(8.W))    // Data In: 要写入的 8-bit 数据（如量化后的权重）。
    val dout  = Output(UInt(8.W))   // Data Out: 从缓存读取出的 8-bit 数据。
  })

  // SyncReadMem 会被 Vivado 识别并映射为 FPGA 的 Block RAM (BRAM) 资源
  // 它的特点是：读操作是同步的，即这一拍给出地址，下一拍才能拿到数据。
  val mem = SyncReadMem(depth, UInt(8.W))
  
  // 硬件连线逻辑
  when(io.we) {
    // 当写使能开启时，在时钟上升沿将 din 写入 addr 指定的位置
    mem.write(io.addr, io.din)
  }
  
  // 读取操作：根据当前地址输出数据（注意：输出会比地址延迟一个时钟周期）
  io.dout := mem.read(io.addr)
}