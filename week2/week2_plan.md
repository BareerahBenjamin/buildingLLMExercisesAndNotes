# 📅 Week 2：文本处理 — Tokenization 与 Embedding

**对应章节**：Chapter 2  
**预计时间**：8-10 小时  
**难度**：⭐⭐⭐☆☆

---

## 🎯 本周目标

- 理解 LLM 如何将文本转化为数字（Tokenization）
- 掌握 BPE（Byte Pair Encoding）算法原理
- 实现 Token Embedding + Positional Embedding
- 构建用于训练的 DataLoader

---

## 📚 学习内容

### Day 1-2：文本预处理与分词（约3小时）

**Notebook**：`ch02/01_main-chapter-code/ch02.ipynb`

**核心流程**：

```
原始文本 → 正则分词 → 词汇表构建 → Token ID → Embedding向量
```

**代码要点**：

```python
# 简单分词器示例
import re

text = "Hello, world! This is a test."
tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
tokens = [t.strip() for t in tokens if t.strip()]

# 构建词汇表
all_words = sorted(set(tokens))
vocab = {word: i for i, word in enumerate(all_words)}

# 编码
def encode(text):
    return [vocab[t] for t in re.split(r'([,.:;?_!"()\']|--|\s)', text) if t.strip() in vocab]

# BPE 分词（使用 tiktoken，GPT-2 的实际分词器）
import tiktoken
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("Hello, world!")
print(ids)  # [15496, 11, 995, 0]
print(enc.decode(ids))  # Hello, world!
```

---

### Day 3-4：Embedding 层实现（约3小时）

**核心概念**：

```python
import torch
import torch.nn as nn

# Token Embedding：将 token ID 映射到向量
vocab_size = 50257  # GPT-2 词汇表大小
embedding_dim = 256

token_embedding = nn.Embedding(vocab_size, embedding_dim)

# Positional Embedding：编码位置信息
context_length = 1024
pos_embedding = nn.Embedding(context_length, embedding_dim)

# 实际使用
token_ids = torch.tensor([[1, 2, 3, 4]])  # batch_size=1, seq_len=4
positions = torch.arange(token_ids.shape[1])

embeddings = token_embedding(token_ids) + pos_embedding(positions)
print(embeddings.shape)  # [1, 4, 256]
```

**为什么需要 Positional Embedding？**
> Transformer 的注意力机制本身没有位置感知能力。输入 "cat sat" 和 "sat cat" 会产生完全相同的注意力分数，因此需要显式注入位置信息。

---

### Day 5-6：DataLoader 构建（约3小时）

**Sliding Window 数据准备**：

```python
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(text)
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# 使用
dataset = GPTDataset(text, enc, max_length=256, stride=128)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

**关键参数**：
- `max_length`：上下文窗口长度
- `stride`：滑动步长（stride < max_length → 数据重叠，有助于训练）

---

### Day 7：练习与 Bonus（约1-2小时）

**必做练习**：  
`ch02/01_main-chapter-code/exercise-solutions.ipynb`

**练习题预览（Chapter 2）**：
1. BPE 分词器的词汇表大小是多少？如何查看？
2. 用不同的 `stride` 值训练，对 loss 有什么影响？
3. 实现一个不使用 `nn.Embedding` 的等价 Embedding 层

**Bonus 材料（选做）**：  
`ch02/02_bonus_dataloader/`
- 对比不同 DataLoader 实现的效率
- 了解 `num_workers` 参数对数据加载速度的影响

---

## 🔑 本周关键概念总结

| 概念 | 作用 | 实现方式 |
|------|------|----------|
| Tokenization | 文本→整数序列 | tiktoken (BPE) |
| Token Embedding | 整数→稠密向量 | `nn.Embedding(vocab_size, d_model)` |
| Positional Embedding | 注入位置信息 | `nn.Embedding(max_len, d_model)` |
| DataLoader | 批量数据供给 | `torch.utils.data.DataLoader` |
| Sliding Window | 训练数据生成 | 步长滑动切片 |

---

## ✅ Week 2 打卡清单

- [ ] Ch02 主 notebook 全部运行完毕
- [ ] 能手写简单分词器（不依赖 tiktoken）
- [ ] 理解 Token Embedding vs Positional Embedding 的区别
- [ ] DataLoader 示例代码运行成功
- [ ] 完成 exercise-solutions.ipynb 练习
- [ ] （选做）阅读 Bonus DataLoader 材料
