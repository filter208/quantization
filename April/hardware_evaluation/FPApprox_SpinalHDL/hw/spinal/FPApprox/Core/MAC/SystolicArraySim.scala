package FPApprox.Core.MAC

import spinal.core._
import FPApprox.Config

object SystolicArrayGen extends App {
  // 我们使用 E4M3 (Expo=4, Mant=3) 作为配置
  val ExpoWidth = 4
  val MantWidth = 3

  // 设置生成的 Verilog 存放的子目录
  Config.setGenSubDir("/MAC_Array")
  
  // 生成 Verilog 代码
  Config.spinal.generateVerilog(
    SystolicArray4x4(ExpoWidth, MantWidth)
  ).printRtl().mergeRTLSource()
  
  println("✅ Verilog 阵列代码已成功生成！")
}