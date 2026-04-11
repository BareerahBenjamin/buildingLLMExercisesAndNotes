# 🗒️ LLMs-from-Scratch 快速参考手册

> 把所有关键代码片段整理在一个地方，方便快速查找

---

## 模型配置

```python
# GPT-2 Small (124M) - 书中主要使用
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# GPT-2 Medium (355M)
GPT_CONFIG_355M = {
    "vocab_size": 50257, "context_length": 1024,
    "emb_dim": 1024, "n_heads": 16, "n_layers": 24,
    "drop_rate": 0.1, "qkv_bias": False
}

# GPT-2 Large (774M)
GPT_CONFIG_774M = {
    "vocab_size": 50257, "context_length": 1024,
    "emb_dim": 1280, "n_heads": 20, "n_layers": 36,
    "drop_rate": 0.1, "qkv_bias": False
}

# 小型调试用配置（CPU友好）
GPT_CONFIG_SMALL = {
    "vocab_size": 50257, "context_length": 256,
    "emb_dim": 128, "n_heads": 4, "n_layers": 2,
    "drop_rate": 0.0, "qkv_bias": False
}
```

---

## 文本编码 / 解码

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")

# 编码
ids = enc.encode("Hello, world!")
print(ids)  # [15496, 11, 995, 0]

# 解码
text = enc.decode(ids)
print(text)  # Hello, world!

# 批量处理
batch_ids = [enc.encode(t) for t in texts]
```

---

## 数据集与 DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids, self.target_ids = [], []
        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+max_length]))
            self.target_ids.append(torch.tensor(token_ids[i+1:i+max_length+1]))
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx): return self.input_ids[idx], self.target_ids[idx]

# 使用
enc = tiktoken.get_encoding("gpt2")
dataset = GPTDataset(text, enc, max_length=256, stride=128)
loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
```

---

## 注意力机制

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, n, _ = x.shape
        Q = self.W_q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / self.head_dim**0.5
        scores = scores.masked_fill(self.mask[:n, :n].bool(), float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = (attn @ V).transpose(1, 2).contiguous().view(b, n, self.d_out)
        return self.out_proj(ctx)
```

---

## 完整 GPT 模型

```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            (2/torch.pi)**0.5 * (x + 0.044715 * x**3)))

class LayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.shift = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        mean, var = x.mean(-1, keepdim=True), x.var(-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / (var + 1e-5)**0.5 + self.shift

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]), GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]))
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"],
            cfg["context_length"], cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1, self.norm2 = LayerNorm(cfg["emb_dim"]), LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])
    def forward(self, x):
        x = x + self.drop(self.att(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, x):
        b, t = x.shape
        x = self.drop(self.tok_emb(x) + self.pos_emb(torch.arange(t, device=x.device)))
        return self.out_head(self.norm(self.blocks(x)))
```

---

## 文本生成

```python
def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]
        if top_k:
            top_vals, _ = torch.topk(logits, top_k)
            logits[logits < top_vals[:, -1:]] = float('-inf')
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, 1)
        else:
            idx_next = logits.argmax(-1, keepdim=True)
        idx = torch.cat([idx, idx_next], dim=1)
    return idx

# 使用
enc = tiktoken.get_encoding("gpt2")
start = torch.tensor(enc.encode("Hello"), device=device).unsqueeze(0)
out = generate(model, start, max_new_tokens=50, context_size=1024, temperature=0.7, top_k=50)
print(enc.decode(out[0].tolist()))
```

---

## 训练循环

```python
import torch.nn.functional as F

def calc_loss(input_batch, target_batch, model, device):
    logits = model(input_batch.to(device))
    return F.cross_entropy(logits.flatten(0,1), target_batch.to(device).flatten())

def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = calc_loss(x, y, model, device)
        loss.backward()
        if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
```

---

## LoRA（参数高效微调）

```python
import math

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scale = alpha / rank
    
    def forward(self, x):
        return self.scale * (x @ self.A @ self.B)

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)

# 冻结原始权重，只训练 LoRA
def freeze_except_lora(model):
    for name, param in model.named_parameters():
        param.requires_grad = "lora" in name
```

---

## 常用工具函数

```python
# 参数量统计
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

# 模型保存/加载
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

# 困惑度
def perplexity(loss):
    return torch.exp(torch.tensor(loss)).item()

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```
