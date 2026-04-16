import os
import shutil
import re

def fix_with_script_content(data_root, script_file):
    val_dir = os.path.join(data_root, 'val')
    
    # 1. 解析你下载的文件内容
    # 你的文件里每一行长这样：mkdir -p n02268443 或 mv ILSVRC2012_val_00000001.JPEG n02268443
    print(f"正在从 {script_file} 解析分类逻辑...")
    
    with open(script_file, 'r') as f:
        lines = f.readlines()

    # 我们通过正则表达式提取：图片名 和 对应的目录名 (WNID)
    # 匹配模式：mv [图片名] [目录名]
    move_pattern = re.compile(r'mv\s+(ILSVRC2012_val_\d+\.JPEG)\s+(n\d+)')
    
    move_tasks = []
    unique_wnids = set()

    for line in lines:
        match = move_pattern.search(line)
        if match:
            img_name, wnid = match.groups()
            move_tasks.append((img_name, wnid))
            unique_wnids.add(wnid)

    if not move_tasks:
        print("错误：无法在文件中解析出移动命令。请检查文件内容是否包含 'mv' 指令。")
        return

    print(f"解析成功：找到 {len(move_tasks)} 个移动任务，涉及 {len(unique_wnids)} 个类别。")

    # 2. 创建 WNID 文件夹并移动图片
    for wnid in unique_wnids:
        os.makedirs(os.path.join(val_dir, wnid), exist_ok=True)

    print("正在按 WNID 分类图片...")
    for img_name, wnid in move_tasks:
        src = os.path.join(val_dir, img_name)
        dst = os.path.join(val_dir, wnid, img_name)
        if os.path.exists(src):
            shutil.move(src, dst)

    # 3. 将 WNID 重命名为数字 (0-999) 以适配 April 代码
    # 这里的顺序必须严格按照 WNID 的字母顺序排列，这是 ImageNet 的标准
    print("正在将 WNID 目录重命名为数字 (0-999)...")
    sorted_wnids = sorted(list(unique_wnids)) # 字母排序：n01440764, n01443537...
    
    for idx, wnid in enumerate(sorted_wnids):
        old_path = os.path.join(val_dir, wnid)
        new_path = os.path.join(val_dir, str(idx))
        
        # 如果目标数字文件夹已存在，先处理冲突（通常不会发生）
        if os.path.exists(new_path) and old_path != new_path:
            # 这是一个简单的工程处理：如果已经数字命名了，跳过
            continue
        os.rename(old_path, new_path)

    print("✅ 处理完成！数据集现在以 0-999 文件夹结构存放，适配 April 源码。")

if __name__ == "__main__":
    # 路径确保正确
    SCRIPT_PATH = '/home/sxy/code/quantization/val_labels.txt'
    DATA_ROOT = '/data2/model_zoo/ImageNet'
    fix_with_script_content(DATA_ROOT, SCRIPT_PATH)