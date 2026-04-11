# 📅 Week 3：注意力机制详解

**对应章节**：Chapter 3  
**预计时间**：10-12 小时  
**难度**：⭐⭐⭐⭐☆（本书最核心章节之一）

---

## 🎯 本周目标

- 理解注意力机制的直觉与数学
- 从简单注意力 → 自注意力 → 因果自注意力 → 多头注意力逐步实现
- 掌握 Dropout 在注意力中的应用
- 理解 Causal Mask（因果掩码）的作用

---

## 📚 学习内容

### Day 1-2：简单注意力机制（约3小时）

**Notebook**：`ch03/01_main-chapter-code/ch03.ipynb`（前半部分）

**注意力的核心思想**：

> 当模型处理某个词时，注意力允许它"关注"输入序列中其他位置的词，并动态加权它们的信息贡献。

**数学公式**：

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

**代码实现（简单版）**：

```python
import torch
import torch.nn.functional as F

def simple_attention(x):
    """x shape: [seq_len, d_model]"""
    # 计算注意力分数
    scores = x @ x.T  # [seq_len, seq_len]
    
    # Softmax 归一化
    weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    context = weights @ x  # [seq_len, d_model]
    return context
```

**Self-Attention（引入 Q, K, V 投影）**：

```python
class SelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = torch.nn.Linear(d_in, d_out, bias=False)
        self.W_k = torch.nn.Linear(d_in, d_out, bias=False)
        self.W_v = torch.nn.Linear(d_in, d_out, bias=False)
    
    def forward(self, x):
        Q = self.W_q(x)  # [batch, seq, d_out]
        K = self.W_k(x)
        V = self.W_v(x)
        
        scale = Q.shape[-1] ** 0.5
        scores = Q @ K.transpose(-2, -1) / scale  # [batch, seq, seq]
        weights = F.softmax(scores, dim=-1)
        
        return weights @ V  # [batch, seq, d_out]
```

---

### Day 3-4：因果自注意力（Causal Self-Attention）（约3小时）

**为什么需要 Causal Mask？**

> GPT 是自回归语言模型，预测第 t 个 token 时，只能看到 1..t-1 的 token，不能"作弊"看到未来。

```python
class CausalSelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout):
        super().__init__()
        self.W_q = torch.nn.Linear(d_in, d_out, bias=False)
        self.W_k = torch.nn.Linear(d_in, d_out, bias=False)
        self.W_v = torch.nn.Linear(d_in, d_out, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        
        # 注册因果掩码（下三角矩阵）
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        batch, seq, _ = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        
        scale = Q.shape[-1] ** 0.5
        scores = Q @ K.transpose(-2, -1) / scale
        
        # 应用因果掩码：未来位置填 -inf → softmax 后为 0
        scores = scores.masked_fill(self.mask[:seq, :seq].bool(), float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        return weights @ V
```

**可视化掩码**：

```python
import matplotlib.pyplot as plt

mask = torch.triu(torch.ones(6, 6), diagonal=1)
plt.figure(figsize=(5, 5))
plt.imshow(mask, cmap='gray_r')
plt.title("Causal Mask (黑=可见, 白=遮掩)")
plt.colorbar()
plt.savefig("causal_mask.png")
```

---

### Day 5-6：多头注意力（Multi-Head Attention）（约4小时）

**核心思想**：

> 多个注意力头允许模型同时从不同子空间（不同"视角"）关注信息，然后拼接并投影。

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度
        
        self.W_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, seq, _ = x.shape
        Q = self.W_q(x).view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        # 形状：[batch, num_heads, seq, head_dim]
        
        scale = self.head_dim ** 0.5
        scores = Q @ K.transpose(-2, -1) / scale
        scores = scores.masked_fill(self.mask[:seq, :seq].bool(), float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        context = (weights @ V).transpose(1, 2).contiguous().view(b, seq, self.d_out)
        return self.out_proj(context)
```

---

### Day 7：练习与 Bonus（约2小时）

**必做练习**：  
`ch03/01_main-chapter-code/exercise-solutions.ipynb`

**练习题预览（Chapter 3）**：
1. 修改 `SelfAttention` 使其支持 `bias=True` 的 QKV 投影
2. 实现使用 `nn.Parameter` 替代 `nn.Linear` 的版本
3. 验证 `MultiHeadAttention` 与 `MultiHeadAttentionWrapper`（头拼接版）的输出等价性

**Bonus 材料（强烈推荐）**：  
`ch03/02_bonus_multihead-attention/`
- 多种 Multi-head Attention 的等价实现对比
- 包含性能基准测试
- 理解 `einsum` 实现方式

---

## 🔑 本周关键概念总结

| 概念 | 关键理解 |
|------|----------|
| Q, K, V | Query（我要查什么）, Key（我有什么标签）, Value（我的实际内容） |
| Scaling `sqrt(d_k)` | 防止点积过大导致 softmax 梯度消失 |
| Causal Mask | 保证自回归生成的合法性 |
| Multi-head | 多角度并行关注，捕获不同类型的依赖关系 |
| Dropout in Attn | 正则化，防止过拟合某些注意力模式 |

---

## ✅ Week 3 打卡清单

- [ ] 能从头手写 Simple Attention（不看代码）
- [ ] 理解 Q/K/V 的几何含义
- [ ] Causal Mask 可以自己绘制并解释
- [ ] Multi-head Attention 代码运行成功
- [ ] 完成 exercise-solutions.ipynb 练习
- [ ] （强烈推荐）阅读 Bonus 多头注意力材料
