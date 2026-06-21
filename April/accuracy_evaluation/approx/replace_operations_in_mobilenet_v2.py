import copy

import torch.nn as nn
from approx.approx_calculation import *
from models.mobilenet_v2 import InvertedResidual, MobileNetV2
from models.mobilenet_v2_quantized import QuantizedInvertedResidual, BNQConv
from quantization.base_quantized_classes import QuantizedActivation

from quantization.autoquant_utils import QuantConv, QuantLinear

def replace_conv2d_with_numpy(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, CustomConv2dNumPy(child.in_channels, child.out_channels, 
                                                    child.kernel_size, stride=child.stride,
                                                    padding=child.padding, dilation=child.dilation,
                                                    groups=child.groups, bias=child.bias is not None))
        elif isinstance(child, nn.Sequential) or isinstance(child, InvertedResidual):
            replace_conv2d_with_numpy(child)
        elif isinstance(child, nn.ModuleList):
            for i, submodule in enumerate(child):
                replace_conv2d_with_numpy(submodule)
                
def replace_linear_with_numpy(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, CustomLinearNumPy(child.in_features, child.out_features, bias=child.bias is not None))
        elif isinstance(child, nn.Sequential) or isinstance(child, InvertedResidual):
            replace_linear_with_numpy(child)
        elif isinstance(child, nn.ModuleList):
            for i, submodule in enumerate(child):
                replace_linear_with_numpy(submodule)

def replace_operations_in_mobilenet_v2(model):
    replace_conv2d_with_numpy(model)
    replace_linear_with_numpy(model)


'''
replace the conv2d and linear operations in the mobilenet_v2_quantized model with numpy operations
'''
def replace_conv2d_with_numpy(module):
    for name, child in module.named_children():
        if isinstance(child, QuantConv):
            new_conv = QCustomConv2dNumPy(child.in_channels, child.out_channels, 
                                         child.kernel_size, stride=child.stride,
                                         padding=child.padding, dilation=child.dilation,
                                         groups=child.groups, bias=child.bias is not None)
            # 复制权重和偏置
            new_conv.weight.data = child.weight.data
            if child.bias is not None:
                new_conv.bias.data = child.bias.data
            setattr(module, name, new_conv)
        elif isinstance(child, BNQConv):
            new_conv = QCustomBNConv2dNumPy(child.in_channels, child.out_channels, 
                                         child.kernel_size, stride=child.stride,
                                         padding=child.padding, dilation=child.dilation,
                                         groups=child.groups, bias=child.bias is not None)
            # 复制权重和偏置
            new_conv.weight.data = child.weight.data
            if child.bias is not None:
                new_conv.bias.data = child.bias.data
            setattr(module, name, new_conv)
        elif isinstance(child, (nn.Sequential, InvertedResidual, QuantizedInvertedResidual, QuantizedActivation)):
            replace_conv2d_with_numpy(child)
        elif isinstance(child, nn.ModuleList):
            for i, submodule in enumerate(child):
                replace_conv2d_with_numpy(submodule)

def replace_linear_with_numpy(module):
    for name, child in module.named_children():
        if isinstance(child, QuantLinear):
            new_linear = QCustomLinearNumPy(child.in_features, child.out_features, bias=child.bias is not None)
            # 复制权重和偏置
            new_linear.weight.data = child.weight.data
            if child.bias is not None:
                new_linear.bias.data = child.bias.data
            setattr(module, name, new_linear)
        elif isinstance(child, (nn.Sequential, InvertedResidual, QuantizedInvertedResidual, QuantizedActivation)):
            replace_linear_with_numpy(child)
        elif isinstance(child, nn.ModuleList):
            for i, submodule in enumerate(child):
                replace_linear_with_numpy(submodule)

'''
replace the conv2d and linear operations in the mobilenet_v2_quantized model with cupy operations
'''
def replace_conv2d_with_cupy(module):
    for name, child in module.named_children():
        if isinstance(child, QuantConv):
            new_conv = QCustomConv2dCuPy(child.in_channels, child.out_channels, 
                                         child.kernel_size, stride=child.stride,
                                         padding=child.padding, dilation=child.dilation,
                                         groups=child.groups, bias=child.bias is not None)
            # 复制权重和偏置
            new_conv.weight.data = child.weight.data
            if child.bias is not None:
                new_conv.bias.data = child.bias.data
            new_conv = new_conv.to(child.weight.device)
            setattr(module, name, new_conv)
        elif isinstance(child, BNQConv):
            new_conv = QCustomBNConv2dCuPy(child.in_channels, child.out_channels, 
                                         child.kernel_size, stride=child.stride,
                                         padding=child.padding, dilation=child.dilation,
                                         groups=child.groups, bias=child.bias is not None)
            # 复制权重和偏置
            new_conv.weight.data = child.weight.data
            if child.bias is not None:
                new_conv.bias.data = child.bias.data
            new_conv = new_conv.to(child.weight.device)
            setattr(module, name, new_conv)
        elif isinstance(child, (nn.Sequential, InvertedResidual, QuantizedInvertedResidual, QuantizedActivation)):
            replace_conv2d_with_cupy(child)
        elif isinstance(child, nn.ModuleList):
            for i, submodule in enumerate(child):
                replace_conv2d_with_cupy(submodule)

def replace_linear_with_cupy(module):
    for name, child in module.named_children():
        if isinstance(child, QuantLinear):
            new_linear = QCustomLinearCuPy(child.in_features, child.out_features, bias=child.bias is not None)
            # 复制权重和偏置
            new_linear.weight.data = child.weight.data
            if child.bias is not None:
                new_linear.bias.data = child.bias.data
            new_linear = new_linear.to(child.weight.device)
            setattr(module, name, new_linear)
        elif isinstance(child, (nn.Sequential, InvertedResidual, QuantizedInvertedResidual, QuantizedActivation)):
            replace_linear_with_cupy(child)
        elif isinstance(child, nn.ModuleList):
            for i, submodule in enumerate(child):
                replace_linear_with_cupy(submodule)

'''
replace the conv2d and linear operations in the mobilenet_v2_quantized model with torch operations
'''         
def create_and_copy_conv(child, new_conv_class, **quant_params):
    device = child.weight.device
    dtype = child.weight.dtype
    if isinstance(child, BNQConv):
        new_conv = new_conv_class(
            child.in_channels, child.out_channels, 
            child.kernel_size, stride=child.stride,
            padding=child.padding, dilation=child.dilation,
            groups=child.groups, bias=child.bias is not None, **quant_params
        ).to(device=device, dtype=dtype)
        new_conv.gamma.data = child.gamma.data
        new_conv.beta.data = child.beta.data
        new_conv.running_mean.data = child.running_mean.data
        new_conv.running_var.data = child.running_var.data
        new_conv.epsilon = child.epsilon
    else:
        new_conv = new_conv_class(
            child.in_channels, child.out_channels, 
            child.kernel_size, stride=child.stride,
            padding=child.padding, dilation=child.dilation,
            groups=child.groups, bias=child.bias is not None, **quant_params
        ).to(device=device, dtype=dtype)
    # 复制权重和偏置
    new_conv.weight.data = child.weight.data
    if child.bias is not None:
        new_conv.bias.data = child.bias.data
        # ===== 新增：DC Offset 偏置补偿逻辑 =====
        # 仅针对容易产生误差累积的 1x1 Pointwise 投影层进行补偿
        if child.kernel_size == (1, 1) and child.groups == 1:
            # TODO: 这里填入你之前从 act.csv 统计得出的单次近似乘加的均值误差。
            # 如果不确定，可以先设为 0，随后通过 hook 观察张量偏移方向再填入极小值（如 0.02 或 -0.015）
            mean_error_per_mac = 0.0 
            
            # 总补偿量 = 单次均值误差 * 累加次数 (即输入通道数)
            compensation = mean_error_per_mac * child.in_channels
            
            # 在原本的 Bias 上强行抹平系统误差
            new_conv.bias.data -= compensation
    new_conv.padding_mode = child.padding_mode
    return new_conv

def create_and_copy_linear(child, new_linear_class, **quant_params):
    device = child.weight.device
    dtype = child.weight.dtype
    new_linear = new_linear_class(
        child.in_features, child.out_features, bias=child.bias is not None, **quant_params
    ).to(device=device, dtype=dtype)
    # 复制权重和偏置
    new_linear.weight.data = child.weight.data
    if child.bias is not None:
        new_linear.bias.data = child.bias.data
    return new_linear

'''
def replace_conv2d_with_torch(module, **quant_params):
    for name, child in module.named_children():
        # 【修改1】加上 child.groups == 1，只替换标准卷积，放过 Depthwise 卷积！
        if isinstance(child, QuantConv) and child.groups == 1:
            new_conv = create_and_copy_conv(child, QCustomConv2dTorch, **quant_params)
            setattr(module, name, new_conv)
        # 【修改2】同样加上 child.groups == 1
        elif isinstance(child, BNQConv) and child.groups == 1:
            new_conv = create_and_copy_conv(child, QCustomBNConv2dTorch, **quant_params)
            setattr(module, name, new_conv)
        elif isinstance(child, (nn.Sequential, InvertedResidual, QuantizedInvertedResidual, QuantizedActivation)):
            # new_child = replace_conv2d_with_torch(child)
            # setattr(module, name, new_child)
            replace_conv2d_with_torch(child, **quant_params)
        elif isinstance(child, nn.ModuleList):
            #copychild = copy.deepcopy(child)
            for i, submodule in enumerate(child):
               # new_submodule = replace_conv2d_with_torch(submodule)
               # child[i] = new_submodule
                replace_conv2d_with_torch(submodule, **quant_params)
'''

def replace_linear_with_torch(module, **quant_params):
    for name, child in module.named_children():
        if isinstance(child, QuantLinear):
            new_linear = create_and_copy_linear(child, QCustomLinearTorch, **quant_params)
            setattr(module, name, new_linear)
        elif isinstance(child, (nn.Sequential, InvertedResidual, QuantizedInvertedResidual, QuantizedActivation)):
            # new_child = replace_linear_with_torch(child)
            # setattr(module, name, new_child)
            replace_linear_with_torch(child, **quant_params)
        elif isinstance(child, nn.ModuleList):
            #copychild = copy.deepcopy(child)
            for i, submodule in enumerate(child):
                #new_submodule = replace_linear_with_torch(submodule)
                #child[i] = new_submodule
                replace_linear_with_torch(submodule, **quant_params)
  

def replace_operations_in_mobilenet_v2_quantized(model, **quant_params):
    # replace_conv2d_with_numpy(model)
    # replace_linear_with_numpy(model)
    # replace_conv2d_with_cupy(model)
    # replace_linear_with_cupy(model)
    # 【核心修复】在传给底层之前，安全地弹出原生的 torch 层不认识的参数
    #quant_params.pop('quant_setup', None)
    # 1. 定义网络首尾绝对不能走近似计算的白名单
    whitelist = [
        'features.0.0',       # 第一层 3x3 标准卷积
        'features.18.0',      # 倒数第二层 1x1 升维卷积
        'classifier.1'        # 最后一层全连接层
    ]
    
    # 2. 带路径追踪的递归替换函数
    def _replace_recursively(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # 判断当前层是否在白名单中
            is_in_whitelist = any(full_name.startswith(w) for w in whitelist)
            # 自动识别并保护 Depthwise 卷积 (groups == in_channels)
            is_depthwise = isinstance(child, nn.Conv2d) and child.groups > 1 and child.groups == child.in_channels
            
            # 3. 【新增】深层超宽 1x1 卷积方差保护 (通道数激增时的噪声隔离)
            # 如果是 1x1 卷积，且输入通道数大于等于 320 (MobileNetV2 深层的分水岭)
            is_deep_pointwise = (isinstance(child, nn.Conv2d) and 
                                 child.kernel_size == (1, 1) and 
                                 child.in_channels >= 320)
            
            if is_in_whitelist or is_depthwise or is_deep_pointwise:
                reason = "硬性白名单" if is_in_whitelist else ("Depthwise保护" if is_depthwise else "高方差通道保护")
                print(f"[{full_name}] 受到保护，跳过近似替换 (精确计算)")
                # 依然需要递归，以防白名单只匹配了父模块
                if isinstance(child, (nn.Sequential, InvertedResidual)):
                    from models.mobilenet_v2_quantized_approx import QuantizedInvertedResidual
                    if isinstance(child, QuantizedInvertedResidual) or hasattr(child, 'children'):
                        _replace_recursively(child, full_name)
                continue
                
            # 执行近似算子替换
            from quantization.autoquant_utils import QuantConv
            from models.mobilenet_v2_quantized import BNQConv
            
            if isinstance(child, QuantConv):
                new_conv = create_and_copy_conv(child, QCustomConv2dTorch, **quant_params) # 请确保 QCustomConv2dTorch 是你的近似类
                setattr(module, name, new_conv)
                print(f"[{full_name}] 已替换为近似算子")
            elif isinstance(child, BNQConv):
                new_conv = create_and_copy_conv(child, QCustomBNConv2dTorch, **quant_params)
                setattr(module, name, new_conv)
                print(f"[{full_name}] 已替换为近似算子 (BN融合)")
            elif isinstance(child, (nn.Sequential, InvertedResidual)) or type(child).__name__ == 'QuantizedInvertedResidual':
                _replace_recursively(child, full_name)
            elif isinstance(child, nn.ModuleList):
                for i, submodule in enumerate(child):
                    _replace_recursively(submodule, f"{full_name}.{i}")

    # 开始替换 Conv2d
    print("\n--- 开始执行智能算子替换 ---")
    _replace_recursively(model)
    
    # 线性层替换保留原有逻辑（但注意前面 classifier.1 已经在白名单保护，这里通常不会再被近似化）
    # replace_linear_with_torch(model, **quant_params)
    print("--- 替换完成 ---\n")
    #replace_conv2d_with_torch(model, **quant_params)
    #replace_linear_with_torch(model, **quant_params)

# Example usage
if __name__ == "__main__":
    from models.mobilenet_v2_quantized import mobilenetv2_quantized  # Add this import
    model = mobilenetv2_quantized(pretrained=True, model_dir='/home/zou/codes/FP8-quantization/model_dir/mobilenet_v2.pth.tar')
    replace_operations_in_mobilenet_v2_quantized(model)

    print(model)

