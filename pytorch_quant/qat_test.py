import torch
import torch.nn as nn
import torch.optim as optim

class QATModel(nn.Module):
    def __init__(self):
        super(QATModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x

def train_dummy(model, optimizer, criterion, epoch, data, target):
    """模拟一步训练，并返回 Loss"""
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_dummy(model, criterion, data, target):
    """在验证集上评估误差"""
    model.eval() # 评估时必须切回 eval 模式
    with torch.no_grad():
        output = model(data)
        loss = criterion(output, target)
    model.train() # 评估完切回 train 模式继续训练
    return loss.item()

if __name__ == "__main__":
    # 固定随机种子，让每次跑出来的结果一样，方便观察
    torch.manual_seed(42)

    model = QATModel()
    model.train()
    
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    
    print("=== 1. QAT 准备完毕，查看模型内部结构 ===")
    print(model)
    print("\n⚠️ 请注意看上面的输出，Linear 层变成了 'Linear()', 并且内部多了 'FakeQuantize' 机制。\n")
    
    # 构建固定的伪造数据集（模拟真实的训练集和验证集）
    train_data = torch.randn(64, 1024) 
    train_target = torch.randn(64, 128)
    val_data = torch.randn(16, 1024)
    val_target = torch.randn(16, 128)

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    print("=== 2. 开始量化感知微调 (QAT) ===")
    for epoch in range(5): 
        # 1. 训练一步
        train_loss = train_dummy(model, optimizer, criterion, epoch, train_data, train_target)
        # 2. 评估一步（查看当前的量化模拟误差）
        val_loss = evaluate_dummy(model, criterion, val_data, val_target)
        
        print(f"Epoch {epoch+1:02d} | 训练 Loss: {train_loss:.4f} | 验证 Loss: {val_loss:.4f}")
        
    print("\n=== 3. 转换为纯 INT8 推理模型 ===")
    model.eval() 
    model_int8 = torch.quantization.convert(model, inplace=False)
    
    print("=== 转换后的 INT8 模型结构 ===")
    print(model_int8)
    print("\n🎉 观察上面的结构，'Linear' 已经变成了底层的 'QuantizedLinear'！")