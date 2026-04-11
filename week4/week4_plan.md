# 📅 Week 4：GPT 模型架构实现

**对应章节**：Chapter 4  
**预计时间**：10-12 小时  
**难度**：⭐⭐⭐⭐☆

---

## 🎯 本周目标

- 将注意力机制组装成完整的 Transformer Block
- 理解 LayerNorm、GELU 激活、Feed-Forward 子层
- 实现完整的 GPT-2 小型模型（124M参数）
- 理解残差连接的作用

---

## 📚 学习内容

### Day 1-2：构建 Transformer Block（约3小时）

**Notebook**：`ch04/01_main-chapter-code/ch04.ipynb`

**GPT Block 组成**：

```
Input
  ↓
LayerNorm 1
  ↓
Multi-Head Causal Self-Attention
  ↓ (+ Residual Connection)
LayerNorm 2
  ↓
Feed-Forward Network (MLP)
  ↓ (+ Residual Connection)
Output
```

**LayerNorm 实现**：

```python
class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift
```

**GELU 激活函数**：

```python
class GELU(torch.nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

> GELU 比 ReLU 更平滑，在负值区域有小梯度，实践中对 LLM 效果更好。

**Feed-Forward 子层**：

```python
class FeedForward(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            torch.nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)
```

---

### Day 3-4：完整 Transformer Block + 残差连接（约3小时）

```python
class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = torch.nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        # 残差连接 1：注意力子层
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut  # 残差
        
        # 残差连接 2：FFN 子层
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # 残差
        
        return x
```

**残差连接的作用**：
- 解决深层网络的梯度消失问题
- 允许网络学习"增量"变换而非从头重建表示
- GPT-2 有 12 个这样的 Block 堆叠

---

### Day 5-6：完整 GPT 模型（约4小时）

**GPT-2 (124M) 配置**：

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # tiktoken GPT-2 词汇表
    "context_length": 1024,  # 最大上下文长度
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 12,           # 注意力头数
    "n_layers": 12,          # Transformer Block 数量
    "drop_rate": 0.1,        # Dropout 率
    "qkv_bias": False        # QKV 是否有偏置
}
```

**完整 GPTModel**：

```python
class GPTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = torch.nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = torch.nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = torch.nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = torch.nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx):
        batch, seq_len = in_idx.shape
        tok_embs = self.tok_emb(in_idx)
        pos_embs = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_embs + pos_embs)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits  # [batch, seq, vocab_size]
```

**参数量计算**：

```python
model = GPTModel(GPT_CONFIG_124M)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")  # 约 163,009,536

# 如果共享 embedding 权重（与书中对齐）
# out_head.weight = tok_emb.weight
```

---

### Day 7：文本生成（贪婪解码）+ 练习

**简单文本生成**：

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # 截取最近 context_size 个 token
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 取最后一个位置的 logits
        logits = logits[:, -1, :]  # [batch, vocab_size]
        
        # 贪婪选择概率最大的 token
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, idx_next], dim=1)
    
    return idx

# 使用示例
model = GPTModel(GPT_CONFIG_124M)
model.eval()

start_text = "Hello, I am"
enc = tiktoken.get_encoding("gpt2")
start_ids = torch.tensor(enc.encode(start_text)).unsqueeze(0)

out_ids = generate_text_simple(model, start_ids, max_new_tokens=20, context_size=1024)
print(enc.decode(out_ids[0].tolist()))
# 注意：此时权重是随机的，输出是乱码，这是正常的！
```

**必做练习**：  
`ch04/01_main-chapter-code/exercise-solutions.ipynb`

**练习题预览（Chapter 4）**：
1. 修改 GPT 配置，计算不同规模模型（345M, 762M, 1542M）的参数量
2. 在 FeedForward 中使用不同的激活函数（ReLU, SiLU）并比较效果
3. 实现 weight tying（共享 token embedding 和 output head 的权重）

---

## 🔑 本周关键概念总结

| 组件 | 功能 | 为什么这样设计 |
|------|------|----------------|
| LayerNorm | 归一化每个样本的特征 | 稳定训练，替代 BatchNorm |
| GELU | 非线性激活 | 比 ReLU 更平滑，LLM 首选 |
| FFN (4x扩展) | 特征变换与存储 | 4倍扩展是实验发现的最优比例 |
| 残差连接 | 梯度高速通道 | 解决深层网络训练问题 |
| Pre-LN vs Post-LN | 归一化位置 | GPT-2 使用 Pre-LN（更稳定） |

---

## ✅ Week 4 打卡清单

- [ ] TransformerBlock 代码手写一遍
- [ ] 理解残差连接的梯度作用（能用图示解释）
- [ ] GPTModel 完整实现并能生成文本（随机权重）
- [ ] 计算过 GPT-2 124M 的参数量
- [ ] 完成 exercise-solutions.ipynb 练习
