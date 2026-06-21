import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
# 强制使用作者写好的模型接口，完美兼容 BN-folded 魔改权重
from models.mobilenet_v2_quantized import mobilenetv2_quantized

print("1. 正在加载原作者的 FP32 魔改预训练模型...")
model_path = '/home/sxy/code/quantization/April/accuracy_evaluation/model_dir/mobilenet_v2.pth.tar'
model = mobilenetv2_quantized(pretrained=True, model_dir=model_path)
model.eval().cuda()

print("2. 配置标准数据预处理...")
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("3. 加载本地数据集...")
val_dir = '/data2/model_zoo/ImageNet/val'
dataset = ImageFolder(val_dir, transform)
print(f"当前前 3 个文件夹映射: {list(dataset.class_to_idx.items())[:3]}")

print("4. 抽取第 1 张图片进行推理透视...")
# 取出数据集的第一张图（属于 000 文件夹）
img, target = dataset[0] 
img = img.unsqueeze(0).cuda()

with torch.no_grad():
    output = model(img)
    prob = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_idx = torch.topk(prob, 5)

print("\n" + "="*50)
print(f"你提供给模型的标准答案 (Target): {target} (来自文件夹: {dataset.classes[target]})")
print(f"模型预测的真实结果 (Model Prediction Top-5):")
for i in range(5):
    print(f"  预测排名 {i+1}: 类别 ID = {top5_idx[i].item()}, 置信概率 = {top5_prob[i].item()*100:.2f}%")
print("="*50)