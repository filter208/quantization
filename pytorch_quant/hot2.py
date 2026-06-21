import numpy as np
import matplotlib.pyplot as plt

def generate_hardware_lut_and_plot(wm=4, k=1, save_path='heatmapk1.png'):
    """
    核心功能：计算 Mitchell 近似误差、生成下采样 LUT 常数并绘制热力图。
    """
    
    # --- 步骤 1：建立高分辨率的原始误差矩阵 ---
    num_steps = 2 ** wm
    exact_error_map = np.zeros((num_steps, num_steps))
    
    for mx in range(num_steps):
        for my in range(num_steps):
            val_x = 1.0 + mx / num_steps
            val_y = 1.0 + my / num_steps
            golden_log = np.log2(val_x * val_y) * num_steps
            approx_log = (mx / num_steps + my / num_steps) * num_steps
            exact_error_map[mx, my] = golden_log - approx_log

    # --- 步骤 2：执行下采样 (Down-sampling) 提取硬件常数 ---
    lut_size = 2 ** k
    window_size = 2 ** (wm - k)
    hardware_lut = np.zeros((lut_size, lut_size), dtype=int)

    for i in range(lut_size):
        for j in range(lut_size):
            window = exact_error_map[i*window_size : (i+1)*window_size, 
                                     j*window_size : (j+1)*window_size]
            hardware_lut[i, j] = int(np.round(np.mean(window)))

    # --- 步骤 3：绘制并保存热力图 ---
    # 【核心修改点】控制图片物理尺寸很小，但字号保持巨大，逼迫字占满图片空间
    plt.rcParams.update({
        'font.size': 14,            # 全局基础字号
        'axes.labelsize': 18,       # X/Y 轴大标签字号（相对于小画布已经极其巨大）
        'xtick.labelsize': 15,      # X 轴刻度数字字号
        'ytick.labelsize': 15,      # Y 轴刻度数字字号
        'figure.figsize': (5.5, 4.5) # 【关键】限制画布物理尺寸，刚好适合 A4 纸横排两张
    })
    
    # 绘制热力图
    im = plt.imshow(hardware_lut, cmap='Reds', interpolation='nearest', origin='lower')
    
    # 颜色条处理
    cbar = plt.colorbar(im)
    cbar.set_label('Value (ulp)', fontsize=16) # 适当精简标签文案，防止字太长溢出
    cbar.ax.tick_params(labelsize=14)
    
    # 设置坐标轴标签（硬件地址含义保持不变）
    plt.xlabel(f'MSB {k}-bits of X')
    plt.ylabel(f'MSB {k}-bits of Y')
    
    # 使用 tight_layout 强制让大字体包裹在小画布内
    plt.tight_layout()
    
    # 保存图片，保持 300 DPI 保证打印的高清晰度
    plt.savefig(save_path, dpi=300)
    print(f"高占比大字号热力图已保存至: {save_path}")
    
    return hardware_lut.flatten()

if __name__ == "__main__":
    lut_constants = generate_hardware_lut_and_plot(wm=4, k=1)