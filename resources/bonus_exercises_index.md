# 📌 Bonus 练习完整索引

> 本文档整理了 `rasbt/LLMs-from-scratch` 中所有 Bonus 和练习材料的位置与内容说明

---

## 📁 各章节练习题（Exercise Solutions）

每章都有一个 `exercise-solutions.ipynb`，包含章末练习的参考答案：

| 章节 | 路径 | 练习数量 | 难度 |
|------|------|----------|------|
| Appendix A | `appendix-A/01_main-chapter-code/exercise-solutions.ipynb` | 3题 | ⭐⭐ |
| Chapter 2 | `ch02/01_main-chapter-code/exercise-solutions.ipynb` | 3题 | ⭐⭐ |
| Chapter 3 | `ch03/01_main-chapter-code/exercise-solutions.ipynb` | 3题 | ⭐⭐⭐ |
| Chapter 4 | `ch04/01_main-chapter-code/exercise-solutions.ipynb` | 3题 | ⭐⭐⭐ |
| Chapter 5 | `ch05/01_main-chapter-code/exercise-solutions.ipynb` | 3题 | ⭐⭐⭐ |
| Chapter 6 | `ch06/01_main-chapter-code/exercise-solutions.ipynb` | 3题 | ⭐⭐⭐ |
| Chapter 7 | `ch07/01_main-chapter-code/exercise-solutions.ipynb` | 3题 | ⭐⭐⭐⭐ |
| Appendix C | `appendix-C/` | 汇总所有答案 | — |

---

## 📁 Appendix 材料

### Appendix A — PyTorch 介绍
**路径**：`appendix-A/01_main-chapter-code/`

| 文件 | 内容 |
|------|------|
| `code-part1.ipynb` | 张量、自动微分、基础操作 |
| `code-part2.ipynb` | 神经网络、优化器、训练循环 |
| `exercise-solutions.ipynb` | 练习答案 |
| `DDP-script.py` | 多GPU并行训练示例（需要多GPU） |
| `DDP-script-torchrun.py` | torchrun方式的多GPU训练 |

**重点练习题**：
1. 实现一个不使用 PyTorch 自动微分的简单梯度下降
2. 从头实现一个 DataLoader（不用 torch.utils.data）
3. 实现模型检查点（checkpoint）保存与恢复

---

### Appendix B — 参考文献与延伸阅读
**路径**：`appendix-B/`

包含书中引用的所有重要论文列表，推荐精读：
- *Attention Is All You Need*（Transformer 原论文）
- *Language Models are Unsupervised Multitask Learners*（GPT-2 论文）
- *Training language models to follow instructions with human feedback*（InstructGPT）
- *LoRA: Low-Rank Adaptation of Large Language Models*

---

### Appendix C — 练习答案汇总
**路径**：`appendix-C/`

所有章节练习的汇总参考答案。建议先自己做，再对照检查。

---

### Appendix D — 高级训练技巧
**路径**：`appendix-D/01_main-chapter-code/appendix-D.ipynb`

**内容**：

1. **学习率预热（Linear Warmup）**
   - 训练前 N 步线性增大 LR
   - 防止训练初期大步长不稳定

2. **余弦衰减（Cosine Annealing）**
   - 训练后期平滑降低 LR
   - 比固定 LR 通常有更好的收敛

3. **梯度裁剪（Gradient Clipping）**
   - `clip_grad_norm_(params, max_norm=1.0)`
   - 防止梯度爆炸，对 LLM 训练至关重要

4. **综合训练函数**：将上述三个技巧集成到训练循环

**代码亮点**：
```python
# 综合学习率调度器
def get_lr(step, warmup_steps, total_steps, peak_lr, min_lr):
    if step < warmup_steps:
        # 线性预热
        return peak_lr * step / warmup_steps
    elif step > total_steps:
        return min_lr
    else:
        # 余弦衰减
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

---

### Appendix E — LoRA 参数高效微调
**路径**：`appendix-E/01_main-chapter-code/appendix-E.ipynb`

**内容**：

1. LoRA 数学原理（低秩矩阵分解）
2. `LoRALayer` 实现
3. `LinearWithLoRA` 包装器
4. 将 LoRA 应用到 GPT 的 Attention 层
5. 只训练 LoRA 参数（冻结原始权重）
6. 参数量对比：Full FT vs LoRA (r=8)

**消融实验建议**（自己做）：

| rank | 可训练参数 | 分类准确率 | 备注 |
|------|-----------|-----------|------|
| r=1 | ~200K | ? | 欠拟合风险 |
| r=4 | ~400K | ? | 轻量级 |
| r=8 | ~800K | ? | 推荐起点 |
| r=16 | ~1.6M | ? | 高容量 |
| Full | 124M | ? | 对照组 |

---

## 📁 章节 Bonus 材料

### Ch02 Bonus — DataLoader 对比
**路径**：`ch02/02_bonus_dataloader/`

比较三种 DataLoader 实现的效率：
1. 简单实现（本书方式）
2. `num_workers > 0` 多进程加载
3. 使用 `pin_memory` 加速 GPU 传输

**实验方法**：
```python
import time

def benchmark_dataloader(dataloader, n_epochs=3):
    start = time.time()
    for epoch in range(n_epochs):
        for batch in dataloader:
            pass  # 只测加载时间
    return time.time() - start
```

---

### Ch03 Bonus — Multi-head Attention 多种实现
**路径**：`ch03/02_bonus_multihead-attention/`

四种等价实现的性能对比：

| 实现方式 | 特点 |
|----------|------|
| `MultiHeadAttentionWrapper` | 多个头拼接（概念最清晰） |
| `MultiHeadAttention` | 内部 reshape（本书主要实现） |
| `MHAPyTorchScaledDotProduct` | 使用 PyTorch 内置 SDPA |
| `MHAFlashAttentionBenchmark` | Flash Attention 基准 |

**关键洞察**：不同实现数学等价，但速度差异显著——Flash Attention 可快 2-4x。

---

### Ch05 Bonus — 替代权重加载
**路径**：`ch05/02_alternative_weight_loading/`

两种加载预训练权重的方式：

1. **原始 TensorFlow checkpoint**（本书主要方式）
2. **HuggingFace 格式**（`safetensors`）

```python
# 方式2：从 HuggingFace 加载
from transformers import GPT2Model

hf_model = GPT2Model.from_pretrained("gpt2")
# 将 HuggingFace 权重映射到我们的 GPTModel
```

---

### Ch07 Bonus — 数据集工具
**路径**：`ch07/02_dataset-utilities/`

1. **数据集格式转换**：将各种格式（ShareGPT、Alpaca 等）统一为本书格式
2. **偏好数据集准备**：为 RLHF/DPO 准备 (chosen, rejected) 数据对

---

## 🎯 Bonus 完成建议顺序

对于时间有限的学习者，优先级排序：

1. ✅ **必做**：各章 `exercise-solutions.ipynb`（直接验证理解）
2. 🔥 **强烈推荐**：Appendix D（训练技巧，实用价值极高）
3. 🔥 **强烈推荐**：Appendix E LoRA（当前最主流微调方法）
4. 📚 **推荐**：Ch03 Bonus（深化注意力理解）
5. 📚 **推荐**：Ch05 Bonus（了解 HuggingFace 生态）
6. 💡 **感兴趣再做**：Ch02/Ch07 Bonus（工程细节）
