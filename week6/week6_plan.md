# 📅 Week 6：微调 — 文本分类（Spam Detection）

**对应章节**：Chapter 6  
**预计时间**：8-10 小时  
**难度**：⭐⭐⭐☆☆

---

## 🎯 本周目标

- 理解分类微调（Classification Finetuning）与预训练的区别
- 将预训练 GPT-2 改造为文本分类器
- 实现 spam/not-spam 邮件分类任务
- 掌握选择性参数冻结（Parameter Freezing）技术
- 评估分类准确率

---

## 📚 学习内容

### Day 1-2：数据准备与模型改造（约3小时）

**Notebook**：`ch06/01_main-chapter-code/ch06.ipynb`

**任务**：SMS Spam 数据集（垃圾短信分类）

```python
# 数据格式
# label    text
# ham      Free entry in 2 a wkly comp to win FA Cup...
# spam     WINNER! You have been selected...

# 编码标签
label_map = {"ham": 0, "spam": 1}
```

**模型改造**：

```python
# 原始 GPT 输出: [batch, seq, vocab_size] (50257维)
# 分类任务需要: [batch, num_classes] (2维)

# 修改方法：替换输出头
num_classes = 2

model = GPTModel(BASE_CONFIG)
# 加载预训练权重...

# 替换最后的 Linear 层
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],  # 768
    out_features=num_classes             # 2
)
```

**关键决策**：使用哪个位置的输出做分类？

```python
# 选择：使用最后一个 token 的表示
# （GPT 是从左到右的，最后 token 含有最多上下文信息）

def forward_classification(model, input_ids):
    logits = model(input_ids)
    # 取最后一个 token 的输出
    last_token_logit = logits[:, -1, :]  # [batch, num_classes]
    return last_token_logit
```

---

### Day 3-4：参数冻结策略（约3小时）

**为什么要冻结部分参数？**
- 预训练权重包含了大量语言知识，随意修改会"灾难性遗忘"
- 只微调少量参数可以减少过拟合风险
- 计算效率更高

**三种策略对比**：

```python
# 策略1：只训练输出头（最保守）
for param in model.parameters():
    param.requires_grad = False  # 先冻结所有
model.out_head.weight.requires_grad = True
model.out_head.bias.requires_grad = True

# 策略2：解冻最后几层（平衡）
for param in model.parameters():
    param.requires_grad = False
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
for param in model.out_head.parameters():
    param.requires_grad = True

# 策略3：微调所有参数（最激进，数据少时容易过拟合）
for param in model.parameters():
    param.requires_grad = True
```

**检查可训练参数数量**：

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
```

---

### Day 5-6：训练与评估（约3小时）

**分类训练循环**：

```python
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_preds, total_examples = 0, 0
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            break
        
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]  # 最后 token
        
        predicted_labels = torch.argmax(logits, dim=-1)
        correct_preds += (predicted_labels == target_batch).sum().item()
        total_examples += predicted_labels.shape[0]
    
    return correct_preds / total_examples

# 分类损失
def calc_loss_batch_classification(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
```

**推理示例**：

```python
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    
    # 截断或填充
    if max_length is None:
        max_length = model.pos_emb.weight.shape[0]
    
    input_ids = input_ids[:max_length]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"

# 测试
text = "You are a winner! Claim your prize now!"
print(classify_review(text, model, tokenizer, device))
```

---

### Day 7：练习与总结（约2小时）

**必做练习**：  
`ch06/01_main-chapter-code/exercise-solutions.ipynb`

**练习题预览（Chapter 6）**：
1. 在更大的 GPT-2 模型（355M）上微调，比较准确率差异
2. 使用第一个 token（而非最后一个）做分类，效果如何？
3. 尝试仅微调输出头 vs 最后 2 层 vs 全部，对比准确率

---

## 🔑 本周关键概念总结

| 概念 | 理解要点 |
|------|----------|
| 分类微调 | 替换 LM Head → Classification Head |
| 参数冻结 | `requires_grad = False` 阻断梯度 |
| 最后 token 分类 | GPT 自回归特性：最后位置含最多上下文 |
| 灾难性遗忘 | 微调过度会丢失预训练语言知识 |
| 准确率 vs 损失 | 分类任务用准确率评估，训练用交叉熵损失 |

---

## ✅ Week 6 打卡清单

- [ ] 理解 LM Head 替换为 Classification Head 的原理
- [ ] 三种参数冻结策略都尝试过
- [ ] 分类准确率 > 90%（spam 数据集）
- [ ] 能用 classify_review 函数分类新文本
- [ ] 完成 exercise-solutions.ipynb 练习
