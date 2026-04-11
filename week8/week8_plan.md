# 📅 Week 8：综合实战 + Bonus 深挖 + 项目总结

**对应内容**：综合复习 + Appendix B/C + 个人项目  
**预计时间**：10-15 小时  
**难度**：⭐⭐⭐⭐⭐（自由发挥）

---

## 🎯 本周目标

- 系统梳理8周知识体系
- 完成所有未完成的 Bonus 练习
- 实现一个端到端的个人项目
- 为下一步学习规划方向

---

## 📚 学习内容

### Day 1-2：综合复习与知识图谱（约3小时）

**从零到一的完整流程回顾**：

```
原始文本
    ↓ Tokenization (Ch02)
Token IDs
    ↓ Embedding (Ch02)
嵌入向量序列
    ↓ Multi-Head Causal Self-Attention (Ch03)
上下文感知表示
    ↓ Feed-Forward + Residual + LayerNorm (Ch04)
Transformer Block × 12
    ↓ Output Head
Logits [vocab_size]
    ↓ Softmax + Sampling
下一个 Token
```

**模块依赖图**：

```
GPTModel
├── tok_emb (nn.Embedding)
├── pos_emb (nn.Embedding)
├── drop_emb (nn.Dropout)
├── trf_blocks × N_LAYERS
│   ├── norm1 (LayerNorm)
│   ├── att (MultiHeadAttention)
│   │   ├── W_q, W_k, W_v (Linear)
│   │   ├── out_proj (Linear)
│   │   └── mask (triu buffer)
│   ├── norm2 (LayerNorm)
│   └── ff (FeedForward)
│       ├── Linear → GELU → Linear
├── final_norm (LayerNorm)
└── out_head (Linear)
```

---

### Day 3-4：完成所有 Bonus 练习（约4小时）

**本周重点攻克未完成的 Bonus 材料**：

#### Bonus 清单

**Ch02 Bonus**：`ch02/02_bonus_dataloader/`
- [ ] 高效 DataLoader 性能对比
- [ ] `num_workers` 并行加载实验

**Ch03 Bonus**：`ch03/02_bonus_multihead-attention/`
- [ ] 实现 `einsum` 版 Multi-head Attention
- [ ] 性能基准测试：不同实现方式的速度对比

**Ch05 Bonus**：`ch05/02_alternative_weight_loading/`
- [ ] 从 HuggingFace 加载 GPT-2 权重
- [ ] 验证两种加载方式的输出一致性

**Ch07 Bonus**：`ch07/02_dataset-utilities/`
- [ ] 偏好数据集格式（为 DPO/RLHF 做准备）

**Appendix D**（如 Week 5 未完成）：
- [ ] LR Warmup + Cosine Decay 完整训练循环

**Appendix E**（如 Week 7 未完成）：
- [ ] LoRA rank 消融实验（r=4/8/16/32 对比）

---

### Day 5-6：个人实战项目（约5小时）

选择以下项目之一完成：

#### 项目选项 A：自定义文本分类器（入门）

**目标**：用 GPT-2 微调一个情感分析模型（正/负面评论）

```python
# 数据集建议：IMDB 影评（HuggingFace Datasets）
from datasets import load_dataset
dataset = load_dataset("imdb")

# 步骤：
# 1. 加载 GPT-2 预训练权重
# 2. 替换 out_head 为 2 分类头
# 3. 微调（参考 Ch06）
# 4. 评估准确率目标 > 92%
```

#### 项目选项 B：自定义指令数据集（中级）

**目标**：制作一个特定领域的问答数据集并微调

```python
# 自建数据集格式（JSON）
custom_data = [
    {
        "instruction": "解释什么是反向传播",
        "input": "",
        "output": "反向传播是..."
    },
    # 至少准备 100 条样本
]

# 步骤：
# 1. 准备 100-500 条领域 QA 数据
# 2. 参考 Ch07 数据集格式
# 3. 微调 GPT-2 124M
# 4. 定性评估回答质量
```

#### 项目选项 C：实现 Flash Attention（高级）

**目标**：用 `torch.nn.functional.scaled_dot_product_attention` 替换手写 Attention

```python
# PyTorch 2.0+ 内置 Flash Attention
import torch.nn.functional as F

# 替换原始 attention 计算
output = F.scaled_dot_product_attention(
    Q, K, V,
    attn_mask=None,
    dropout_p=dropout if self.training else 0.0,
    is_causal=True  # 自动应用因果掩码！
)

# 对比性能：原始 vs Flash Attention
# 测量：速度、显存使用、精度
```

#### 项目选项 D：中文文本生成（高级）

**目标**：用中文语料微调 GPT-2，实现中文续写

```python
# 步骤：
# 1. 准备中文语料（如 wiki_zh 数据集）
# 2. 使用中文 BPE 分词器（如 BERT tokenizer）
# 3. 重新训练 GPT-2 小型版本（调小参数）
# 4. 生成中文文本
```

---

### Day 7：学习总结与下一步规划（约2小时）

**填写你的学习总结**：

```markdown
## 我的8周学习总结

### 最有收获的3个知识点
1. 
2. 
3. 

### 最困难的概念（和我是如何克服的）
- 

### 我实现的代码成果
- Ch02-07 所有主 notebook 已运行
- 完成练习题数量：
- Bonus 材料：

### 下一步学习方向
```

---

## 🚀 下一步学习路线

完成本课程后，推荐以下进阶方向：

### 方向1：更大规模预训练
- 阅读 GPT-3 论文（Scaling Laws）
- 学习 DeepSpeed / FSDP 分布式训练
- 资源：[nanoGPT by Karpathy](https://github.com/karpathy/nanoGPT)

### 方向2：RLHF 与对齐
- 学习 PPO 算法（OpenAI 的 InstructGPT 论文）
- 学习 DPO（Direct Preference Optimization）
- 资源：TRL（HuggingFace）

### 方向3：高效微调前沿
- LoRA → QLoRA（4-bit 量化 + LoRA）
- Prefix Tuning / Prompt Tuning
- 资源：PEFT（HuggingFace）

### 方向4：推理优化
- KV Cache 实现
- 量化（INT8/INT4）
- Speculative Decoding
- 资源：llama.cpp, vLLM

### 方向5：rasbt 续集
- [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)
- 从 DeepSeek R1 视角学习推理模型

---

## ✅ Week 8 打卡清单

- [ ] 完整知识回顾（能用图示描述 GPT 全流程）
- [ ] 至少完成 3 个 Bonus 材料
- [ ] 个人项目选择并完成（A/B/C/D 任选一）
- [ ] 学习总结文档填写完毕
- [ ] 下一步学习方向已确定

---

**🎉 恭喜完成8周 LLMs-from-scratch 学习计划！**
