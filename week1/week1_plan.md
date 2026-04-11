# 📅 Week 1：LLM概述 + PyTorch热身

**对应章节**：Chapter 1 + Appendix A  
**预计时间**：8-10 小时  
**难度**：⭐⭐☆☆☆

---

## 🎯 本周目标

- 理解 LLM 的工作原理与训练阶段（预训练、微调、RLHF）
- 了解本书/本项目的整体架构
- 掌握 PyTorch 基础操作（张量、自动微分、神经网络模块）
- 搭建并验证学习环境

---

## 📚 学习内容

### Day 1-2：环境搭建 + Chapter 1（约3小时）

**任务清单**：
- [ ] 克隆仓库：`git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git`
- [ ] 安装依赖（参见 README 的环境配置部分）
- [ ] 验证安装：`python -c "import torch; print(torch.__version__)"`
- [ ] 阅读 `ch01/01_main-chapter-code/ch01.ipynb`

**Chapter 1 核心概念**：

| 概念 | 简要说明 |
|------|----------|
| 预训练（Pretraining） | 在大规模无标注文本上训练 |
| 监督微调（SFT） | 在指令-回答对上微调 |
| RLHF | 通过人类反馈强化学习对齐 |
| Transformer 架构 | 现代 LLM 的基础架构 |
| GPT vs BERT | 解码器-only vs 编码器-only |

**Chapter 1 无练习 notebook，但记录以下思考题**：
1. 为什么 LLM 使用无监督预训练而非直接监督训练？
2. GPT 系列为什么选择解码器-only 架构？
3. RLHF 的核心思想是什么？

---

### Day 3-5：Appendix A — PyTorch 介绍（约4小时）

**Notebooks**：
- `appendix-A/01_main-chapter-code/code-part1.ipynb` — 张量与基础操作
- `appendix-A/01_main-chapter-code/code-part2.ipynb` — 神经网络与训练

**核心知识点**：

```python
# 张量创建与操作
import torch

# 创建张量
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(x.shape)   # torch.Size([2, 2])

# 矩阵乘法
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B  # shape: [3, 5]

# 自动微分
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # 6.0

# GPU操作（如有）
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
```

**练习 notebook**：  
`appendix-A/01_main-chapter-code/exercise-solutions.ipynb`

---

### Day 6-7：复习 + 环境验证（约2小时）

**验证脚本**（运行以下代码，确保无报错）：

```python
import torch
import tiktoken
import matplotlib
import numpy as np

print("✅ PyTorch:", torch.__version__)
print("✅ tiktoken:", tiktoken.__version__)
print("✅ matplotlib:", matplotlib.__version__)
print("✅ numpy:", np.__version__)
print("✅ GPU:", "可用" if torch.cuda.is_available() else "不可用（CPU模式）")

# 简单神经网络测试
model = torch.nn.Linear(10, 5)
x = torch.randn(3, 10)
y = model(x)
print("✅ 简单前向传播成功，输出形状:", y.shape)
```

---

## 📝 Week 1 核心笔记模板

在 `week1/` 文件夹中创建你的笔记文件，记录：

```markdown
## LLM 训练三阶段（我的理解）

1. 预训练：
   - 数据：
   - 目标：
   - 代表模型：

2. 监督微调（SFT）：
   - 数据：
   - 目标：

3. RLHF：
   - 核心思想：

## PyTorch 我还不熟悉的操作
- 
-
```

---

## ⚡ Bonus（选做）

如果你已经熟悉 PyTorch 基础，可以提前预习：
- [ ] 阅读 `appendix-A/01_main-chapter-code/README.md` 了解多GPU训练（DDP）
- [ ] 浏览 Ch02 notebook 第一个 cell，了解下周内容

---

## ✅ Week 1 打卡清单

- [ ] 环境搭建完成
- [ ] Ch01 notebook 阅读完毕
- [ ] Appendix A part1 + part2 运行完毕
- [ ] Appendix A 练习 notebook 完成
- [ ] 思考题记录在笔记中
