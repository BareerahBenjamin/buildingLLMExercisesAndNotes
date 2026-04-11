# 🤖 从零开始构建大语言模型 — 8周系统学习计划

> 基于 [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 整理  
> 书籍：*Build a Large Language Model (From Scratch)* by Sebastian Raschka  
> Manning出版社，2024，ISBN: 978-1633437166

---

## 📋 目录

- [项目简介](#项目简介)
- [8周学习总览](#8周学习总览)
- [环境配置](#环境配置)
- [仓库结构说明](#仓库结构说明)
- [Notebook使用指南](#Notebook使用指南)
- [每周详细计划](#每周详细计划)
- [Bonus练习说明](#Bonus练习说明)
- [常见问题](#常见问题)
- [参考资源](#参考资源)

---

## 项目简介

本学习计划将 Sebastian Raschka 的开源教材 **LLMs-from-scratch** 重新组织成8周的系统学习课程。你将通过纯PyTorch（无高级LLM库）从零构建一个类ChatGPT的语言模型，深入理解：

- Token化与嵌入机制
- 注意力机制（Self-attention / Multi-head attention）
- GPT架构实现
- 预训练与微调
- RLHF基础（指令跟随）

**建议学习者**：具备 Python 基础 + 基本神经网络概念。PyTorch 不是硬性前提但有帮助（Week 0 有热身材料）。

---

## 8周学习总览

| 周次 | 主题 | 对应章节 | 预计时间 |
|------|------|----------|----------|
| Week 1 | 环境配置 + 理解LLM概述 + PyTorch热身 | Ch01 + Appendix A | 8-10小时 |
| Week 2 | 文本处理：Tokenization与Embedding | Ch02 | 8-10小时 |
| Week 3 | 注意力机制详解 | Ch03 | 10-12小时 |
| Week 4 | GPT模型架构实现 | Ch04 | 10-12小时 |
| Week 5 | 预训练：训练循环与评估 | Ch05 + Appendix D | 10-12小时 |
| Week 6 | 微调：文本分类 | Ch06 | 8-10小时 |
| Week 7 | 微调：指令跟随（Instruction Finetuning） | Ch07 + Appendix E | 10-12小时 |
| Week 8 | 综合实战 + Bonus练习 + 项目总结 | 综合复习 | 10-15小时 |

**总计：约 74-91 小时**

---

## 环境配置

### 快速开始（推荐）

```bash
# 1. 克隆原始仓库（获取最新notebooks）
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch

# 2. 创建虚拟环境
python -m venv llm_env
source llm_env/bin/activate  # Windows: llm_env\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动 Jupyter
jupyter lab
```

### 核心依赖版本

```
torch>=2.0.1
tiktoken>=0.5.1
matplotlib>=3.7.1
numpy>=1.24.3
jupyterlab>=4.0
transformers>=4.40.0  # 用于加载预训练权重
tqdm>=4.66.1
```

### 云端运行（无GPU时推荐）

- **Google Colab**：每个 notebook 顶部有 Colab 图标，点击即可在浏览器运行
- **GPU建议**：Ch05（预训练）和 Ch06/07（微调）建议使用 GPU，其余章节 CPU 够用

---

## 仓库结构说明

原始仓库 `rasbt/LLMs-from-scratch` 的结构如下：

```
LLMs-from-scratch/
├── setup/                          # 环境配置指南
├── ch01/                           # 第1章
│   └── 01_main-chapter-code/
│       └── ch01.ipynb             # 主章节代码
├── ch02/
│   ├── 01_main-chapter-code/
│   │   ├── ch02.ipynb             # 主代码
│   │   └── exercise-solutions.ipynb  # 练习答案
│   └── 02_bonus_dataloader/       # Bonus材料
├── ch03/
│   ├── 01_main-chapter-code/
│   └── 02_bonus_multihead-attention/
├── ch04/
│   └── 01_main-chapter-code/
├── ch05/
│   ├── 01_main-chapter-code/
│   └── 02_alternative_weight_loading/
├── ch06/
│   └── 01_main-chapter-code/
├── ch07/
│   ├── 01_main-chapter-code/
│   └── 02_dataset-utilities/
├── appendix-A/                     # PyTorch介绍
│   └── 01_main-chapter-code/
├── appendix-B/                     # 参考资料
├── appendix-C/                     # 练习答案汇总
├── appendix-D/                     # 高级训练技巧
│   └── 01_main-chapter-code/
└── appendix-E/                     # LoRA参数高效微调
    └── 01_main-chapter-code/
```

### 每章文件说明

| 文件 | 说明 |
|------|------|
| `chXX.ipynb` | 主章节 notebook（核心内容） |
| `exercise-solutions.ipynb` | 章节练习答案 |
| `02_bonus_*/` | 可选 Bonus 深入材料 |
| `previous_chapters.py` | 来自前章的复用代码 |

---

## Notebook使用指南

### 第一次使用建议

1. **先运行，再理解**：第一遍跑通所有 cell，观察输出
2. **注释阅读**：每个 cell 上方的 markdown 解释了"为什么"
3. **不要跳跃**：notebook 之间有依赖，按顺序学习

### 常见操作

```python
# 检查 PyTorch 版本和 GPU 可用性
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # True = 有GPU

# 如果需要强制使用 CPU
device = torch.device("cpu")

# 如果有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Notebook 执行技巧

- `Shift+Enter`：执行当前 cell 并移到下一个
- `Ctrl+Enter`：执行当前 cell（停留）
- `Kernel → Restart & Run All`：重新执行整个 notebook（解决状态混乱问题）

---

## 每周详细计划

详见各周文件夹内的 `week_plan.md`。

---

## Bonus练习说明

每章在主代码之外还有 Bonus 材料（`02_bonus_*/` 文件夹），这些是**可选深入**内容：

| 章节 | Bonus内容 |
|------|-----------|
| Ch02 | 高效 DataLoader 实现对比 |
| Ch03 | Multi-head Attention 的多种等价实现 |
| Ch05 | 替代权重加载方式（HuggingFace格式） |
| Ch07 | 数据集处理工具，RLHF偏好数据准备 |
| Appendix D | LR Warmup + Cosine Decay + Gradient Clipping |
| Appendix E | LoRA 参数高效微调（推荐Week 7后学习） |

---

## 常见问题

**Q：SSL证书错误怎么办？**  
A：运行 `pip install --upgrade certifi` 或在代码中加：
```python
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
```

**Q：GPU 显存不足（CUDA OOM）？**  
A：减小 `batch_size`，或在 config 中将 `n_layers` 从 12 改为 4

**Q：找不到 `previous_chapters.py`？**  
A：每章的 `01_main-chapter-code/` 文件夹中已包含，也可通过 pip 安装：
```bash
pip install llms-from-scratch
```

**Q：Notebook 输出和书上不一致？**  
A：随机种子、模型权重初始化可能导致数值不完全相同，这是正常的

---

## 参考资源

- 📖 [原书购买链接（Manning）](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- 🎥 [配套视频课程（17小时）](https://www.manning.com/liveprojectseries/build-a-large-language-model-from-scratch)  
- 💬 [官方讨论论坛](https://github.com/rasbt/LLMs-from-scratch/discussions)
- 📊 [免费测验PDF（170页）](https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch)
- 🐍 [PyPI 包（llms-from-scratch）](https://pypi.org/project/llms-from-scratch/)

---

*整理日期：2026年4月 | 基于原仓库最新版本*
