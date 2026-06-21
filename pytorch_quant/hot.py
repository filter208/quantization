import numpy as np
import matplotlib.pyplot as plt

def generate_hardware_lut_and_plot(wm=4, k=4, save_path='heatmapk4.png'):
    """
    核心功能：计算 Mitchell 近似误差、生成下采样 LUT 常数并绘制热力图。
    
    【输入参数】
    :param wm: 尾数位宽 (Mantissa width)，如 E3M4 中 wm=4。它定义了原始误差空间的精度。
    :param k:  补偿因子 (Compensation factor)，决定了硬件 LUT 的输入地址位数。
               例如 k=3 时，LUT 大小为 2^3 * 2^3 = 64 词，适配一个 LUT6 单元 [cite: 1, 155]。
    :param save_path: 图片保存路径，方便在远程服务器通过 VS Code 侧边栏查看。
    """
    
    # --- 步骤 1：建立高分辨率的原始误差矩阵 ---
    num_steps = 2 ** wm  # 原始尾数组合总数
    exact_error_map = np.zeros((num_steps, num_steps))
    
    for mx in range(num_steps):
        for my in range(num_steps):
            # 将整数索引还原为 [1.0, 2.0) 之间的浮点值
            val_x = 1.0 + mx / num_steps
            val_y = 1.0 + my / num_steps
            
            # 计算理论对数结果与 Mitchell 近似结果的差值 (单位: ulp)
            # Mitchell 近似逻辑：log2(1+m) ≈ m [cite: 1, 46]
            golden_log = np.log2(val_x * val_y) * num_steps
            approx_log = (mx / num_steps + my / num_steps) * num_steps
            exact_error_map[mx, my] = golden_log - approx_log

    # --- 步骤 2：执行下采样 (Down-sampling) 提取硬件常数 ---
    # 核心逻辑：利用误差的局部相似性，用区域平均值代替单点误差 [cite: 1, 126, 129]
    lut_size = 2 ** k
    window_size = 2 ** (wm - k)
    hardware_lut = np.zeros((lut_size, lut_size), dtype=int)

    for i in range(lut_size):
        for j in range(lut_size):
            # 锁定下采样窗口范围
            window = exact_error_map[i*window_size : (i+1)*window_size, 
                                     j*window_size : (j+1)*window_size]
            # 计算该区域的平均补偿值并四舍五入
            hardware_lut[i, j] = int(np.round(np.mean(window)))

    # --- 步骤 3：绘制并保存热力图 ---
    plt.figure(figsize=(8, 6))
    
    # 使用 'Reds' 色谱：误差越大，红色越深
    # origin='lower' 保证坐标 (0,0) 在左下角，符合硬件地址递增逻辑
    im = plt.imshow(hardware_lut, cmap='Reds', interpolation='nearest', origin='lower')
    
    # 颜色条说明 (ulp)
    cbar = plt.colorbar(im)
    cbar.set_label('Compensation Value (ulp)')
    
    # 设置坐标轴：代表尾数的高 k 位索引 [cite: 1, 115]
    plt.xlabel(f'MSB {k}-bits of X_mant')
    plt.ylabel(f'MSB {k}-bits of Y_mant')
    
    # 关键：针对服务器环境，保存为文件而非直接显示
    plt.savefig(save_path, dpi=300)
    print(f"热力图已保存至: {save_path}")
    
    return hardware_lut.flatten()

if __name__ == "__main__":
    # 配置：E?M4 格式，使用 k=3 的补偿因子
    # 运行后，请在 VS Code 左侧文件树中查找 heatmap.png 并打开
    lut_constants = generate_hardware_lut_and_plot(wm=4, k=4)
    
    print("\n生成的硬件 LUT 向量 (直接用于 Verilog/SpinalHDL 初始化):")
    print(lut_constants.tolist())