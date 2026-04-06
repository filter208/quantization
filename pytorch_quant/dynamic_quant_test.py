import torch
import torch.nn as nn
import os
import time

# 1. 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # 使用较大的线性层以便明显观察到体积变化
        self.fc1 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 1024)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    print(f"{label} 模型大小: {size:.2f} MB")
    os.remove("temp.p")

def test_inference_speed(model, data, iterations=100):
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(data)
    end_time = time.time()
    print(f"推理 {iterations} 次总耗时: {end_time - start_time:.4f} 秒\n")

if __name__ == "__main__":
    # 实例化并切换到验证模式
    model_fp32 = SimpleModel()
    model_fp32.eval()

    # 进行动态量化
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, 
        {nn.Linear}, 
        dtype=torch.qint8
    )

    print("=== 模型体积对比 ===")
    print_size_of_model(model_fp32, "FP32 (原始)")
    print_size_of_model(model_int8, "INT8 (量化)")

    print("=== 推理速度对比 ===")
    # 构造一个虚拟输入张量 (Batch Size=1)
    dummy_input = torch.randn(1, 4096)
    
    print("FP32 (原始) 速度测试:")
    test_inference_speed(model_fp32, dummy_input)
    
    print("INT8 (量化) 速度测试:")
    test_inference_speed(model_int8, dummy_input)