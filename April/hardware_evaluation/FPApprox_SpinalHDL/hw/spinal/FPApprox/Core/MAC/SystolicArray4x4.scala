package FPApprox.Core.MAC

import spinal.core._

case class SystolicArray4x4(ExpoWidth: Int, MantWidth: Int) extends Component {
  val N = 4
  val TotalWidth = 1 + ExpoWidth + MantWidth
  val TotalWidthPSum = 1 + (ExpoWidth + 3) + MantWidth

  val io = new Bundle {
    val i_act_top   = in Vec(Bits(TotalWidth bits), N)
    val i_wght_left = in Vec(Bits(TotalWidth bits), N)
    val shift_en    = in Bool()
    val o_res_right = out Vec(Bits(TotalWidthPSum bits), N) 
  }

  // 生成 4x4 PE 网格
  val peGrid = Seq.fill(N, N)(SystolicPE(ExpoWidth, MantWidth))

  for (row <- 0 until N) {
    for (col <- 0 until N) {
      val pe = peGrid(row)(col)
      
      // 全局移位信号
      pe.io.shift_en := io.shift_en

      // 垂直连线 (Iact 向下流动)
      if (row == 0) pe.io.i_act := io.i_act_top(col)
      else pe.io.i_act := peGrid(row - 1)(col).io.o_act

      // 水平连线 (Wght 向右流动)
      if (col == 0) pe.io.i_wght := io.i_wght_left(row)
      else pe.io.i_wght := peGrid(row)(col - 1).io.o_wght

      // 移位连线 (ResultCIN 从左向右流动)
      if (col == 0) pe.io.i_casc := 0
      else pe.io.i_casc := peGrid(row)(col - 1).io.o_pass
    }
    
    io.o_res_right(row) := peGrid(row)(N - 1).io.o_pass
  }
}