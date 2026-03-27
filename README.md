# 面向端侧大模型低比特量化推理的近似矩阵乘法加速单元设计

## 📌 项目简介
本项目旨在为边缘计算设备上的大语言模型设计高效的硬件加速单元。结合**低比特量化算法**与**近似矩阵乘法**，在保证推理精度的前提下，优化底层硬件的功耗（Power）、面积（Area）与延迟（Performance）。

---

## 🛠️ 环境配置与运行指南（核心速查）

本项目包含“算法验证”与“硬件生成”两大部分，为了防止依赖冲突，部署在两个完全独立的 Conda 虚拟环境中。**每次登录服务器跑代码前，请务必先激活对应的环境！**

### 1. 算法端：低比特量化验证 (Python + PyTorch)
* **环境管家**：Conda
* **环境名称**：`quant`
* **技术栈**：Python 3.10 + PyTorch (适配 CUDA 12.1)
* **激活命令**：
    ```bash
    conda activate quant
    ```
* **⚠️ 运行注意事项 (GPU 资源保护)**：
    服务器 0 号显卡常驻大型训练任务。运行 Python 脚本时**必须**给代码戴上“眼罩”，强制指定使用空闲的 1 号 H100 显卡，严禁直接使用 `python xxx.py`：
    ```bash
    CUDA_VISIBLE_DEVICES=1 python 你的脚本名称.py
    ```

### 2. 硬件端：加速单元底层设计 (Scala + SpinalHDL)
* **环境管家**：Conda (conda-forge 源)
* **环境名称**：`hw_env`
* **技术栈**：Java 17 (OpenJDK) + SBT
* **激活命令**：
    ```bash
    conda activate hw_env
    ```
* **硬件生成命令**：
    进入对应的硬件工程目录后，呼叫 SBT 管家将 Scala 架构代码翻译成 Verilog 底层芯片代码：
    ```bash
    cd spinal_test
    sbt "runMain MyAdderVerilog"
    ```

---

## 📂 核心目录结构
* `test_gpu.py`：GPU 连通性测试脚本（用于验证 1 号卡隔离状态）。
* `spinal_test/`：SpinalHDL 初始验证工程，遵循 `src/main/scala` 行业标准目录树，用于存放 MAC 单元等硬件架构的源文件并生成 `.v` 文件。
