import os
import shutil
from torchvision.datasets import ImageFolder

print("1. 读取当前官方 ImageNet 文件夹映射...")
val_dir = '/data2/model_zoo/ImageNet/val'

# 利用 PyTorch 自带的 ImageFolder 自动获取官方最标准的 0~999 字母排序字典
dataset = ImageFolder(val_dir)
official_mapping = dataset.class_to_idx 

print("2. 正在将 n0xxxx 文件夹重命名为作者要求的 000~999...")
success_count = 0
for folder_name, class_id in official_mapping.items():
    old_path = os.path.join(val_dir, folder_name)
    # 强制格式化为 3 位数字，例如 0 -> 000, 15 -> 015
    new_folder_name = f"{class_id:03d}"
    new_path = os.path.join(val_dir, new_folder_name)
    
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        success_count += 1

print(f"3. 大功告成！成功重命名了 {success_count} 个文件夹。")
print("现在的验证集既装对了图片，又完全符合作者的纯数字命名要求！")