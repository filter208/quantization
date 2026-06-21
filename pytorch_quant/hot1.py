import numpy as np
import matplotlib.pyplot as plt

def generate_e4m3_lut_and_plot(save_path='e4m3_heatmap.png'):
    # 完美修正版的 E4M3 (尾数3位) 硬件底层误差比特矩阵
    # 0 代表无误差，1 代表需要补偿 1 ulp
    exact_matrix = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0], # X=0
        [0, 0, 0, 0, 1, 1, 0, 0], # X=1
        [0, 0, 0, 1, 1, 1, 1, 0], # X=2
        [0, 0, 1, 1, 1, 1, 1, 0], # X=3
        [0, 1, 1, 1, 1, 1, 0, 0], # X=4
        [0, 1, 1, 1, 1, 1, 0, 0], # X=5
        [0, 0, 1, 1, 0, 0, 0, 0], # X=6
        [0, 0, 0, 0, 0, 0, 0, 0]  # X=7
    ])

    # 1. 组装 64 位 LUT 常数
    init_bit0 = 0  
    for x in range(8):
        for y in range(8):
            comp_bit = int(exact_matrix[x, y])
            # 硬件地址映射：X 作为高 3 位，Y 作为低 3 位
            addr = (x << 3) + y  
            init_bit0 |= (comp_bit << addr)

    # 强制截断为 64 位无符号整数
    init_bit0_unsigned = init_bit0 & 0xFFFFFFFFFFFFFFFF
    
    print("==================================================")
    print(" 🚀 E4M3 LUT6 实例化代码")
    print("==================================================")
    print(f"LUT6 #(")
    print(f"    .INIT(64'h{init_bit0_unsigned:016x})")
    print(") lut_comp_b0 (")
    print("    .O(comp_val[0]),")
    print("    .I0(addr[0]), .I1(addr[1]), .I2(addr[2]),")
    print("    .I3(addr[3]), .I4(addr[4]), .I5(addr[5])")
    print(");")
    print("==================================================")

    # 2. 绘制并保存热力图
    plt.figure(figsize=(6, 5))
    # 使用 cmap='Reds'，只有 0 和 1 两种状态，看起来会非常锐利（菱形分布）
    im = plt.imshow(exact_matrix, cmap='Reds', interpolation='nearest', origin='lower')
    
    # 颜色刻度条
    cbar = plt.colorbar(im, ticks=[0, 1])
    cbar.set_label('Compensation Value (ulp)')
    
    plt.xlabel('MSB 3-bits of Y_mant (addr[2:0])')
    plt.ylabel('MSB 3-bits of X_mant (addr[5:3])')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 热力图已保存至当前目录: {save_path} (请在 VS Code 中点击查看)")

if __name__ == "__main__":
    generate_e4m3_lut_and_plot()