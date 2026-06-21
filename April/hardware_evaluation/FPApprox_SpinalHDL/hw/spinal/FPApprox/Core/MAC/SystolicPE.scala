package FPApprox.Core.MAC

import spinal.core._

// 根据你的 MAC_FP8Any，改为传入具体的位宽参数
case class SystolicPE(ExpoWidth: Int, MantWidth: Int) extends Component {
  val TotalWidth = 1 + ExpoWidth + MantWidth
  val ExpoWidthPSum = ExpoWidth + 3
  val MantWidthPSum = MantWidth
  val TotalWidthPSum = 1 + ExpoWidthPSum + MantWidthPSum

  val io = new Bundle {
    val i_act  = in Bits(TotalWidth bits)
    val i_wght = in Bits(TotalWidth bits)
    val i_casc = in Bits(TotalWidthPSum bits) 
    val shift_en = in Bool()      
    val o_act  = out Bits(TotalWidth bits)
    val o_wght = out Bits(TotalWidth bits)
    val o_pass = out Bits(TotalWidthPSum bits)
  }

  // 实例化你工程里原有的 MAC_FP8Any
  val mac = MAC_FP8Any(ExpoWidth, MantWidth, ExpoWidthPSum, MantWidthPSum, UseTerAddOpt = true, MultWithFF = true)
  
  // 仅仅为激活值和权重添加流水线寄存器
  val act_reg  = RegNext(io.i_act) init(0)
  val wght_reg = RegNext(io.i_wght) init(0)
  
  io.o_act  := act_reg
  io.o_wght := wght_reg

  mac.io.Iact := act_reg
  mac.io.Wght := wght_reg
  
  // 完美利用你原有的控制端口进行累加和移位！
  mac.io.PassResult := io.shift_en
  mac.io.ResultCIN  := io.i_casc

  io.o_pass := mac.io.Result
}