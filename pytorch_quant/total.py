import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

# 1. 设置图片路径
base_path = "/home/sxy/code/quantization/pytorch_quant/"
img_paths = [f"{base_path}heatmapk{i}.png" for i in range(1, 5)]

# 2. 创建画布
fig = plt.figure(figsize=(12, 10))

# 3. 使用 ImageGrid 创建 2x2 布局
# cbar_mode="single" 表示整个组图共享一个 colorbar
# cbar_location="right" 表示 colorbar 放在右侧
grid = ImageGrid(fig, 111, 
                 nrows_ncols=(2, 2), 
                 axes_pad=0.15, 
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="5%",
                 cbar_pad=0.15)

# 4. 循环读取并显示图片
for ax, path in zip(grid, img_paths):
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.axis('off')  # 隐藏坐标轴

# 5. 设置 Colorbar
# 注意：由于读取的是 PNG，这里的 colorbar 默认映射的是像素(0-1)
# 如果你需要显示原始数据的刻度，需要手动调整刻度或使用原始数据绘图
cbar = grid.cbar_axes[0].colorbar(plt.cm.ScalarMappable(norm=None, cmap='viridis')) 
# 如果你记得原始数据的范围（例如 0 到 1），可以这样设置：
# from matplotlib.colors import Normalize
# sm = plt.cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=1))
# cbar = grid.cbar_axes[0].colorbar(sm)

plt.savefig(f"{base_path}combined_heatmap.png", bbox_inches='tight', dpi=300)
plt.show()