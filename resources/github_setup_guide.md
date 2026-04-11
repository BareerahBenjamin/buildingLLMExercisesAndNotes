# 🚀 新仓库 GitHub 使用指南

> 本指南说明如何将本学习计划上传到你自己的 GitHub，以及如何配合原始仓库使用

---

## 方案一：Fork + 本地整合（推荐）

### Step 1：Fork 原始仓库

1. 访问 https://github.com/rasbt/LLMs-from-scratch
2. 点击右上角 **Fork** 按钮
3. 选择你的账号，完成 Fork

### Step 2：克隆你的 Fork

```bash
git clone https://github.com/你的用户名/LLMs-from-scratch.git
cd LLMs-from-scratch
```

### Step 3：添加本学习计划文件

将本压缩包解压，把所有文件复制到克隆目录的根目录：

```bash
# 解压本压缩包
unzip LLMs-from-scratch-8week-plan.zip -d ./study-plan

# 将学习计划文件整合进去
cp -r study-plan/* ./

# 查看结构
ls -la
```

### Step 4：提交并推送

```bash
git add .
git commit -m "Add 8-week study plan and personal notes"
git push origin main
```

---

## 方案二：创建全新仓库

### Step 1：在 GitHub 创建新仓库

1. 点击 GitHub 右上角 **+** → **New repository**
2. 仓库名：`LLMs-from-scratch-study`（或你喜欢的名字）
3. 设置为 **Public**（方便分享）或 **Private**
4. **不要**勾选 "Initialize with README"（我们已有 README）
5. 点击 **Create repository**

### Step 2：本地初始化

```bash
# 创建并进入项目目录
mkdir LLMs-from-scratch-study
cd LLMs-from-scratch-study

# 初始化 git
git init
git branch -M main

# 将本压缩包解压到此目录
# 然后克隆原始仓库内容到子目录
git submodule add https://github.com/rasbt/LLMs-from-scratch.git original

# 或者直接克隆并整合
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git original
```

### Step 3：整合目录结构

推荐的最终目录结构：

```
LLMs-from-scratch-study/
├── README.md                    # 本项目主 README（已提供）
├── original/                    # 原始仓库（可以是 submodule 或直接克隆）
│   ├── ch01/ ... ch07/
│   ├── appendix-A/ ... appendix-E/
│   └── setup/
├── week1/
│   ├── week1_plan.md           # 本计划提供
│   └── my_notes.md             # 你自己的笔记（待创建）
├── week2/
│   ├── week2_plan.md
│   └── my_notes.md
├── ...（week3-week8）
├── resources/
│   └── bonus_exercises_index.md
├── my_notebooks/               # 你自己修改过的 notebook
│   └── ch02_my_version.ipynb
└── .gitignore
```

### Step 4：添加 .gitignore

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# 模型权重（通常很大）
gpt2/
*.safetensors
*.bin
*.pt
*.pth

# 数据集
*.csv
*.jsonl
*.txt
!requirements.txt

# 系统
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
EOF
```

### Step 5：推送到 GitHub

```bash
git add .
git commit -m "Initial commit: 8-week LLMs-from-scratch study plan"
git remote add origin https://github.com/你的用户名/LLMs-from-scratch-study.git
git push -u origin main
```

---

## 学习过程中的 Git 工作流

### 每日学习后提交进度

```bash
# 添加你的笔记和修改的 notebook
git add week3/my_notes.md
git add my_notebooks/ch03_attention_experiments.ipynb

# 提交（用有意义的信息）
git commit -m "Week 3 Day 2: Implemented CausalSelfAttention, understand causal mask"

# 推送
git push
```

### 建议的分支策略

```bash
# 为每周创建一个分支（可选）
git checkout -b week-3-attention

# 完成后合并回 main
git checkout main
git merge week-3-attention
git push
```

---

## 如何同步原始仓库的更新

原始仓库会持续更新（bug修复、新内容）：

```bash
# 如果使用 submodule
git submodule update --remote original

# 如果是直接克隆的 original 文件夹
cd original
git pull origin main
cd ..
git add original
git commit -m "Sync with upstream rasbt/LLMs-from-scratch"
git push
```

---

## GitHub Pages（可选）：将学习计划发布为网页

1. 进入仓库 **Settings** → **Pages**
2. Source 选择 **Deploy from a branch**
3. Branch 选 `main`，目录选 `/（root）`
4. 点击 **Save**

几分钟后，你的学习计划就会发布到：  
`https://你的用户名.github.io/LLMs-from-scratch-study/`

---

## 常用 Git 命令速查

```bash
# 查看状态
git status

# 查看提交历史
git log --oneline

# 撤销未暂存的修改
git checkout -- 文件名

# 查看差异
git diff

# 暂存所有修改
git add -A

# 拉取最新
git pull

# 查看远程仓库
git remote -v
```

---

## 隐私注意事项

如果你的 notebook 运行结果中包含 API key 或个人信息：

```bash
# 清理 notebook 输出（安装 nbconvert 后）
jupyter nbconvert --clear-output --inplace my_notebooks/*.ipynb

# 或使用 nbstripout（自动在 commit 前清理）
pip install nbstripout
nbstripout --install  # 安装 git hook
```

---

*本指南基于 2026 年 GitHub 功能整理*
