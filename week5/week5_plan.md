# 📅 Week 5：预训练 — 训练循环、评估与权重加载

**对应章节**：Chapter 5 + Appendix D  
**预计时间**：10-12 小时  
**难度**：⭐⭐⭐⭐☆

---

## 🎯 本周目标

- 实现完整的 LLM 预训练训练循环
- 理解交叉熵损失与困惑度（Perplexity）评估指标
- 实现温度采样、Top-k 采样等解码策略
- 加载 OpenAI 预训练 GPT-2 权重到我们的模型
- （Appendix D）学习 LR Warmup、Cosine Decay、梯度裁剪

---

## 📚 学习内容

### Day 1-2：训练循环实现（约3小时）

**Notebook**：`ch05/01_main-chapter-code/ch05.ipynb`

**损失函数**：

```python
import torch.nn.functional as F

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch)  # [batch, seq, vocab_size]
    
    # 展平为 [batch*seq, vocab_size] 和 [batch*seq]
    loss = F.cross_entropy(
        logits.flatten(0, 1),   # [B*T, V]
        target_batch.flatten()  # [B*T]
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if num_batches is None:
        num_batches = len(data_loader)
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    
    return total_loss / num_batches
```

**完整训练循环**：

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                        num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    
    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1
            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        # 每 epoch 结束生成样本文本
        generate_and_print_sample(model, tokenizer, device, start_context)
    
    return train_losses, val_losses, track_tokens_seen
```

---

### Day 3-4：评估指标与解码策略（约3小时）

**困惑度（Perplexity）**：

```python
# 困惑度 = exp(平均交叉熵损失)
# 越低越好；随机猜测 = vocab_size = 50257
perplexity = torch.exp(torch.tensor(loss_value))
print(f"困惑度: {perplexity:.2f}")
```

**温度采样（Temperature Sampling）**：

```python
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]  # [batch, vocab]
        
        if top_k is not None:
            # Top-k 过滤：只保留前 k 个概率最高的 token
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        
        if temperature > 0.0:
            # 温度缩放
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # 贪婪解码（temperature=0）
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        idx = torch.cat([idx, idx_next], dim=1)
    
    return idx
```

**温度参数的效果**：

| Temperature | 效果 |
|-------------|------|
| 0.0 | 贪婪（确定性，可能重复） |
| 0.7 | 均衡（推荐） |
| 1.0 | 原始分布 |
| > 1.5 | 更随机，可能不连贯 |

---

### Day 5-6：加载 OpenAI GPT-2 预训练权重（约3小时）

这是本章最激动人心的部分！我们将 OpenAI 的预训练权重加载到我们自己实现的模型中：

```python
# 下载并加载 GPT-2 权重
from gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(
    model_size="124M",  # 可选: "355M", "774M", "1558M"
    models_dir="gpt2"
)

# 将权重映射到我们的 GPTModel
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        # 加载注意力权重、FFN权重、LayerNorm权重...
        # (详见 notebook 中的完整实现)
        pass
    
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params['g'])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params['b'])
    gpt.out_head.weight = assign(gpt.out_head.weight, params['wte'])  # weight tying

# 验证：模型应该能生成有意义的文本！
model.eval()
out = generate(model, start_ids, max_new_tokens=50, 
               context_size=1024, temperature=0.7, top_k=50)
print(enc.decode(out[0].tolist()))
```

**Bonus 材料（选做）**：  
`ch05/02_alternative_weight_loading/`
- 使用 HuggingFace transformers 加载权重的替代方式

---

### Day 7：Appendix D — 高级训练技巧（约2小时）

**Notebook**：`appendix-D/01_main-chapter-code/appendix-D.ipynb`

**三大技巧**：

```python
# 1. 学习率预热（Linear Warmup）
warmup_steps = 20
initial_lr = 0.0001
peak_lr = 0.01
lr_increment = (peak_lr - initial_lr) / warmup_steps

# 2. 余弦衰减（Cosine Annealing）
import math
min_lr = peak_lr * 0.1
lr = min_lr + 0.5 * (peak_lr - min_lr) * (
    1 + math.cos(math.pi * progress)  # progress: 0→1
)

# 3. 梯度裁剪（Gradient Clipping）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 🔑 本周关键概念总结

| 概念 | 作用 |
|------|------|
| 交叉熵损失 | 衡量预测分布与真实 token 的差距 |
| 困惑度 | 损失的指数，更直观的语言模型评估指标 |
| Temperature | 控制生成的随机程度 |
| Top-k | 限制候选词范围，提升生成质量 |
| LR Warmup | 训练初期用小学习率，防止不稳定 |
| Cosine Decay | 训练后期平滑降低学习率 |
| 梯度裁剪 | 防止梯度爆炸 |

---

## ✅ Week 5 打卡清单

- [ ] 训练循环代码手写并运行成功
- [ ] 理解困惑度与损失的关系
- [ ] 实现温度采样和 Top-k 采样
- [ ] 成功加载 GPT-2 124M 预训练权重
- [ ] 验证加载后的模型能生成有意义文本
- [ ] 完成 exercise-solutions.ipynb 练习
- [ ] （推荐）完成 Appendix D notebook
