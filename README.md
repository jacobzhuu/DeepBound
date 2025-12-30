根据您的要求，我删除了图片占位符，并用您提供的详细环境配置方案替换了之前的通用安装步骤。这份 README 文档现在更贴合实际代码仓库的部署需求（包含 Conda 环境隔离、Fairseq 编译安装、以及具体的 Demo 运行指南）。

您可以直接复制以下内容作为项目的 `README.md`。

---

# DeepBound: 基于 Transformer 架构的二进制函数边界检测系统

**北京邮电大学 | 网络空间安全学院 | 计算机系统结构课程大作业 (第10组)**

## 📖 项目简介

**DeepBound** 是一个针对剥离符号表（Stripped Binaries）的二进制函数边界检测工具。针对传统工具（如 IDA Pro）在处理高优化级别（-O3）和复杂编译器行为时的局限性，本项目提出并实现了一种基于 **Transformer** 架构的深度学习反汇编方案。

核心优势在于利用自注意力机制（Self-Attention）捕捉二进制字节流中的长距离依赖关系（如栈平衡指令对），从而在不依赖特征签名的情况下，实现高精度的函数起始（Start）与结束（End）预测。

## ✨ 核心特性

* **Transformer 架构驱动**：利用双向 Transformer 编码器并行处理长序列，有效捕捉跨基本块的语义依赖。
* 
**抗编译器优化**：在 `-O3` 激进优化等级下，F1-Score 仍保持在 **96.5%** 。


* 
**鲁棒性**：针对指令重排、函数内联、尾调用优化等场景具有极强的适应性 。


* **可视化交互平台**：提供像素级对齐的序列视图与增强反汇编视图，支持实时推理监控。

## 🛠️ 环境配置与安装

为了确保 `fairseq` 及其 C++/Cython 扩展正确编译，强烈建议使用 **Conda** 创建独立环境（避免与系统现有 torch 版本冲突）。

### 1. 基础依赖准备

确保系统中有可用的 C/C++ 编译器（Linux 上通常是 `gcc`/`g++`）。

* *可选*：如果你设置了 `CUDA_HOME` 且想编译 CUDA 扩展，则需要完整 CUDA Toolkit（含 `nvcc`）；不需要的话不要设置 `CUDA_HOME` 也能正常安装。

### 2. 创建并激活 Conda 环境

```bash
# 创建环境 (指定 Python 3.7)
conda create -n deepbound python=3.7 numpy scipy scikit-learn colorama

# 激活环境
conda activate deepbound

```

### 3. 安装 PyTorch

根据你的硬件选择合适的版本。

* **GPU 版本 (推荐，示例为 CUDA 11.0)**：
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

```


* **CPU 版本**：
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch

```



### 4. 安装 DeepBound (包含 Fairseq 编译)

在项目根目录下运行以下命令，这将编译必要的 Cython/C++ 扩展：

```bash
# 更新构建工具
pip install -U pip setuptools wheel

# 以可编辑模式安装项目
pip install -e .

```

### 5. 快速自检

运行以下命令验证 PyTorch 和 Fairseq 是否加载成功：

```bash
python -c "import torch, fairseq; print(f'Torch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}')"

```

## 📂 模型权重与数据

在运行推理或 Demo 之前，请确保已下载并放置好模型权重文件：

* **预训练权重**：放入 `checkpoints/pretrain_all/`
* **微调权重**：放入 `checkpoints/finetune_msvs_funcbound_64/`

请参考仓库内相关文档获取权重下载链接。

## 🚀 演示系统运行 (Demo)

本项目包含一个基于 Web 的可视化演示系统。

### 1. 启动后端 (Python)

后端负责模型推理与二进制分析服务。

```bash
# 确保在 deepbound 环境下
python demo/server.py

```

### 2. 启动前端 (Node.js + Vite)

前端提供交互式可视化界面。

```bash
cd demo
npm install
npm run dev

```

启动后，浏览器访问终端输出的地址（通常为 `http://localhost:5173`）即可使用。

## 👥 团队分工 (Group 10)

| 姓名 | 学号 | 角色与职责 |
| --- | --- | --- |
| **朱子阳** | 2025141000 | [核心算法] 神经网络结构实现、模型训练、理论推导 

 |
| **吴楷** | 2025140937 | [后端架构] Python 后端服务器、API 封装、LaTeX 报告统筹 

 |
| **何浔航** | 2025141008 | [数据工程] 数据预处理流水线、性能指标测试与对比实验 

 |
| **张昊健** | 2025140933 | [前端逻辑] Web 核心业务开发、检测流程控制、答辩展示 

 |
| **陈万桥** | 2025140916 | [可视化] 模型决策热力图开发、前端性能优化、交互设计 

 |

# DeepBound: 基于 Transformer 架构的二进制函数边界检测系统

**北京邮电大学 | 网络空间安全学院 | 计算机系统结构课程大作业 (第10组)**

## 📖 项目简介

**DeepBound** 是一个针对剥离符号表（Stripped Binaries）的二进制函数边界检测工具。针对传统工具（如 IDA Pro）在处理高优化级别（-O3）和复杂编译器行为时的局限性，本项目提出并实现了一种基于 **Transformer** 架构的深度学习反汇编方案。

核心优势在于利用自注意力机制（Self-Attention）捕捉二进制字节流中的长距离依赖关系（如栈平衡指令对），从而在不依赖特征签名的情况下，实现高精度的函数起始（Start）与结束（End）预测。

## ✨ 核心特性

- **Transformer 架构驱动**：利用双向 Transformer 编码器并行处理长序列，有效捕捉跨基本块的语义依赖。
- **抗编译器优化**：在 `-O3` 激进优化等级下，F1-Score 仍保持在 **96.5%**。
- **鲁棒性**：针对指令重排、函数内联、尾调用优化等场景具有极强的适应性。
- **可视化交互平台**：提供像素级对齐的序列视图与增强反汇编视图，支持实时推理监控。

## 🛠️ 环境配置与安装

为了确保 `fairseq` 及其 C++/Cython 扩展正确编译，强烈建议使用 **Conda** 创建独立环境（避免与系统现有 torch 版本冲突）。

### 1. 基础依赖准备

确保系统中有可用的 C/C++ 编译器（Linux 上通常是 `gcc`/`g++`）。

- *可选*：如果你设置了 `CUDA_HOME` 且想编译 CUDA 扩展，则需要完整 CUDA Toolkit（含 `nvcc`）；不需要的话不要设置 `CUDA_HOME` 也能正常安装。

### 2. 创建并激活 Conda 环境

Bash

```
# 创建环境 (指定 Python 3.7)
conda create -n deepbound python=3.7 numpy scipy scikit-learn colorama

# 激活环境
conda activate deepbound
```

### 3. 安装 PyTorch

根据你的硬件选择合适的版本。

- **GPU 版本 (推荐，示例为 CUDA 11.0)**：

  Bash

  ```
  conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
  ```

- **CPU 版本**：

  Bash

  ```
  conda install pytorch torchvision torchaudio cpuonly -c pytorch
  ```

### 4. 安装 DeepBound (包含 Fairseq 编译)

在项目根目录下运行以下命令，这将编译必要的 Cython/C++ 扩展：

Bash

```
# 更新构建工具
pip install -U pip setuptools wheel

# 以可编辑模式安装项目
pip install -e .
```

### 5. 快速自检

运行以下命令验证 PyTorch 和 Fairseq 是否加载成功：

Bash

```
python -c "import torch, fairseq; print(f'Torch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}')"
```

## 📂 模型权重与数据

在运行推理或 Demo 之前，请确保已下载并放置好模型权重文件：

- **预训练权重**：放入 `checkpoints/pretrain_all/`
- **微调权重**：放入 `checkpoints/finetune_msvs_funcbound_64/`

请参考仓库内相关文档获取权重下载链接。

## 🚀 演示系统运行 (Demo)

本项目包含一个基于 Web 的可视化演示系统。

### 1. 启动后端 (Python)

后端负责模型推理与二进制分析服务。

Bash

```
# 确保在 deepbound 环境下
python demo/server.py
```

### 2. 启动前端 (Node.js + Vite)

前端提供交互式可视化界面。

Bash

```
cd demo
npm install
npm run dev
```

启动后，浏览器访问终端输出的地址（通常为 `http://localhost:5173`）即可使用。

## 👥 团队分工 (Group 10)

| **姓名**   | **学号**   | **角色与职责**                                         |
| ---------- | ---------- | ------------------------------------------------------ |
| **朱子阳** | 2025141000 | [核心算法] 神经网络结构实现、模型训练、理论推导        |
| **吴楷**   | 2025140937 | [后端架构] Python 后端服务器、API 封装、LaTeX 报告统筹 |
| **何浔航** | 2025141008 | [数据工程] 数据预处理流水线、性能指标测试与对比实验    |
| **张昊健** | 2025140933 | [前端逻辑] Web 核心业务开发、检测流程控制、答辩展示    |
| **陈万桥** | 2025140916 | [可视化] 模型决策热力图开发、前端性能优化、交互设计    |