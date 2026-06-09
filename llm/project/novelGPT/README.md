# NovelGPT

基于金庸武侠小说训练的小型 GPT（Decoder-Only）语言模型，支持中文段落续写。

## 项目结构

```
novelGPT/
├── config.py            # 全局配置（模型超参、路径、训练参数）
├── data_preprocess.py   # 数据加载（读取原始小说文本）
├── train_tokenizer.py   # BPE 分词器训练脚本
├── dataset.py           # 数据集 + DataLoader
├── model.py             # GPT 模型定义（Decoder-Only）
├── train.py             # 训练入口脚本
└── README.md
```

## 数据流

```
金庸小说原始文本 (.txt)
    │
    ▼
data_preprocess.load_all_data()
    │  拼接 5 部小说，用 \n 分隔
    ▼
BPE Tokenizer 训练 (train_tokenizer.py)
    │  词表大小: 8000, 基于 HuggingFace tokenizers
    ▼
bpe_tokenizer.json
    │
    ▼
dataset.py: 全量编码 → 滑动窗口切分
    │  block_size=256, stride=1
    │  input:  tokens[i : i+256]
    │  target: tokens[i+1 : i+257]  (输入右移1位)
    ▼
DataLoader → model(x, targets) → loss → backward
```

## 模型架构

| 组件 | 配置 |
|------|------|
| 类型 | GPT Decoder-Only |
| 词嵌入 | `nn.Embedding(8000, 256)` |
| 位置编码 | 可学习 `nn.Embedding(256, 256)` |
| Transformer 层 | 6 × `nn.TransformerEncoderLayer` |
| 注意力头数 | 8 |
| 隐层维度 | 256 |
| FFN 维度 | 1024 (4 × d_model) |
| 激活函数 | GELU |
| 总参数量 | **~8.9M** |

### 因果掩码（Causal Mask）

使用 `torch.triu` 构造上三角 `-inf` 掩码，确保每个位置只能注意到自身及之前的 token：

```
[  0, -inf, -inf, -inf]
[  0,    0, -inf, -inf]
[  0,    0,    0, -inf]
[  0,    0,    0,    0]
```

### 训练目标

标准自回归语言模型：给定前 t 个 token，预测第 t+1 个 token。

```
输入:  [t0, t1, t2, ..., t255]
输出:  [t1, t2, t3, ..., t256]  ← 每个位置的预测目标
```

使用交叉熵损失 `F.cross_entropy(logits, targets)`，logits 展平为 `[B×T, V]`，targets 展平为 `[B×T]`。

## 训练数据

| 小说 | 大小 |
|------|------|
| 天龙八部 | 3.8 MB |
| 笑傲江湖 | 3.0 MB |
| 倚天屠龙记 | 3.0 MB |
| 侠客行 | 1.2 MB |
| 越女剑 | 51 KB |
| **合计** | **~11 MB** |

- 格式：UTF-8 繁体中文
- BPE 编码后约 200~300 万 token
- 每部小说用换行符 `\n` 拼接后统一编码

## 使用方法

### 1. 训练分词器

```bash
cd novelGPT
python train_tokenizer.py
```

会在 `llm/data/jinyong/bpe_tokenizer.json` 生成分词器文件。

### 2. 训练模型

```bash
python train.py
```

训练配置（`config.py` 中可调）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BLOCK_SIZE` | 256 | 上下文窗口长度 |
| `BATCH_SIZE` | 16 | 批次大小 |
| `LR` | 1e-4 | 学习率 |
| `EPOCHS` | 10 | 训练轮数 |
| `D_MODEL` | 256 | 隐层维度 |
| `N_HEAD` | 8 | 注意力头数 |
| `NUM_LAYERS` | 6 | Transformer 层数 |
| `VOCAB_SIZE` | 8000 | 词表大小 |

每轮保存最优模型至 `novelGPT/bin/best_novel_gpt.pth`。

### 3. 续写推理（示例）

```python
import torch
from model import NovelGPT
from config import BLOCK_SIZE
from tokenizers import Tokenizer

# 加载模型和分词器
model = NovelGPT()
model.load_state_dict(torch.load("bin/best_novel_gpt.pth")["model_state_dict"])
model.eval()

tokenizer = Tokenizer.from_file("../data/jinyong/bpe_tokenizer.json")

# 给定开头，自回归生成续写
prompt = "张无忌踏上光明顶，只见"
ids = tokenizer.encode(prompt).ids
ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

for _ in range(100):  # 生成 100 个 token
    x = ids[:, -BLOCK_SIZE:]  # 截断到 block_size
    with torch.no_grad():
        logits, _ = model(x)
    next_token = logits[:, -1, :].argmax(dim=-1)  # 贪心解码
    ids = torch.cat([ids, next_token.unsqueeze(0)], dim=1)

generated = tokenizer.decode(ids[0].tolist())
print(generated)
```

## 已知改进方向

1. **梯度裁剪** — 已添加 `clip_grad_norm_(max_norm=1.0)`
2. **学习率调度** — 可添加 warmup + cosine annealing 提升训练稳定性
3. **embedding 权重** — 可指定 `padding_idx=1`，或做 weight tying（embedding 与输出层共享权重）
4. **数据增强** — 可混入更多中文语料（维基百科、其他小说）
5. **推理策略** — 当前为贪心解码，可改用 temperature sampling、top-k、top-p

## 依赖

- PyTorch
- HuggingFace `tokenizers`
- Python 3.8+
