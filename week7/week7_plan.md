# 📅 Week 7：微调 — 指令跟随 + LoRA

**对应章节**：Chapter 7 + Appendix E  
**预计时间**：10-12 小时  
**难度**：⭐⭐⭐⭐⭐（最复杂章节）

---

## 🎯 本周目标

- 理解指令微调（Instruction Finetuning）原理
- 构建 Alpaca 风格的指令数据集
- 实现批量填充（Padding）策略
- 用预训练 GPT-2 微调为"听话"的助手
- 用 Ollama + LLM 自动评估模型输出
- （Appendix E）实现 LoRA 参数高效微调

---

## 📚 学习内容

### Day 1-2：指令数据集准备（约3小时）

**Notebook**：`ch07/01_main-chapter-code/ch07.ipynb`

**Alpaca 数据格式**：

```json
{
    "instruction": "Evaluate the following phrase by transforming it...",
    "input": "incredibly happy",
    "output": "extremely delighted"
}
```

**提示模板（Prompt Template）**：

```python
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    
    return instruction_text + input_text

def format_output(entry):
    return f"\n\n### Response:\n{entry['output']}"

# 完整训练文本 = 指令 + 输出（训练时只计算输出部分的损失）
full_text = format_input(entry) + format_output(entry)
```

---

### Day 3-4：批量填充策略（约3小时）

**挑战**：指令数据集中，不同样本长度不一致

```python
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100,
                       allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    
    inputs_lst, targets_lst = [], []
    
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]  # 结尾 token
        
        # 填充
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])   # 输入：去掉最后一个
        targets = torch.tensor(padded[1:])   # 目标：右移一位
        
        # 关键：将 padding 位置的目标替换为 -100（ignore_index）
        # 这样 cross_entropy 不会计算 padding 位置的损失
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        
        # 将指令部分的 target 也设为 -100（只学习 response 部分）
        # (详见 notebook 的完整实现)
        
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    
    return inputs_tensor, targets_tensor
```

**为什么要 mask 指令部分？**
> 我们想让模型学习"如何回应"，而非学习"如何复述指令"。只对 response 部分的 token 计算损失，让训练信号更干净。

---

### Day 5：模型评估（Ollama 自动评估）（约2小时）

```python
import urllib.request
import json

def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    """使用本地 Ollama 评估生成质量"""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"seed": 123, "temperature": 0}
    }
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    
    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
            if response_json.get("done"):
                break
    return response_data

# 自动评估提示
def format_eval_prompt(entry, model_response):
    return f"""
Given the input `{format_input(entry)}` and correct output `{entry['output']}`, 
score the model response `{model_response}` on a scale from 0 to 100, 
where 100 is the best score.
Respond with the integer number only.
"""
```

> 注意：Ollama 需要本地安装。如果没有，可以跳过此部分，直接定性评估生成质量。

---

### Day 6：Appendix E — LoRA（约3小时）

**Notebook**：`appendix-E/01_main-chapter-code/appendix-E.ipynb`

**LoRA 核心思想**：

> 不修改原始大矩阵 W（d×k），而是学习两个小矩阵 A（d×r）和 B（r×k），其中 r << min(d,k)。
> 
> 实际权重 = W + α/r × (B @ A)

```python
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
    
    def forward(self, x):
        x = self.alpha / x.shape[-1] * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear          # 冻结的原始线性层
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank, alpha
        )
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)

# 将模型中的 Linear 层替换为 LoRA 版本
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)
```

**LoRA 的优势**：

| 指标 | 全量微调 | LoRA (r=8) |
|------|----------|------------|
| 可训练参数 | 124M | ~800K |
| 显存需求 | 高 | 低 |
| 性能 | 最好 | 接近 |
| 推理速度 | 正常 | 正常（可合并权重） |

---

### Day 7：Bonus + 综合练习（约2小时）

**Bonus 材料**：  
`ch07/02_dataset-utilities/`
- 数据集预处理工具
- 偏好数据（用于 RLHF/DPO）格式准备

**必做练习**：  
`ch07/01_main-chapter-code/exercise-solutions.ipynb`

**练习题预览（Chapter 7）**：
1. 修改训练，使 mask 只覆盖指令部分（而非 padding），比较效果
2. 使用不同的评分 prompt 评估同一批模型输出
3. 在 LoRA 中尝试不同的 rank (r=4, 8, 16)，观察准确率/参数量权衡

---

## 🔑 本周关键概念总结

| 概念 | 理解要点 |
|------|----------|
| 指令微调 | 用指令-回答对让 LLM "听话" |
| Alpaca 格式 | 标准化的 instruction/input/output 三元组 |
| ignore_index=-100 | 告诉 CE loss 忽略特定 token（padding/指令） |
| LoRA | 低秩近似，大幅减少训练参数 |
| 自动评估 | 用更强的 LLM 评估较弱 LLM 的输出 |

---

## ✅ Week 7 打卡清单

- [ ] 理解指令微调与预训练的本质区别
- [ ] 自定义 collate_fn 实现并理解 ignore_index 机制
- [ ] 微调模型能对指令做出有意义的回应
- [ ] （选做）Ollama 本地评估运行成功
- [ ] 完成 Appendix E LoRA 实现
- [ ] 完成 exercise-solutions.ipynb 练习
