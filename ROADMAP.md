# LLM 学习路线

## 已完成

- [x] **阶段一：深度学习基础** — Titanic / Otto 分类，掌握训练流程
- [x] **阶段二：分词器** — BPE、WordPiece 从零实现
- [x] **阶段三：嵌入层** — Token Embedding + Sinusoidal Positional Encoding
- [x] **阶段四：注意力机制** — Scaled Dot-Product Attention + Multi-Head Attention
- [x] **阶段五：Transformer 层** — Encoder Layer / Decoder Layer
- [x] **阶段六：微型 NSP 项目** — Decoder-only 滕王阁序上下句生成

---

## 阶段七：GPT 风格自回归预训练

**目标**：从 NSP（上句→下句）升级为真正的 Causal LM（逐 token 预测下一个词）

### 核心内容

| 要点 | 说明 |
|------|------|
| Causal LM 目标 | `input[:, :-1]` → `input[:, 1:]`，一次前传预测所有位置 |
| 数据管道 | 文档拼接 + 滑动窗口切分 + attention mask 处理 |
| 生成策略 | greedy / temperature / top-k / top-p (nucleus) sampling |
| 评估指标 | Perplexity（困惑度） |

### 论文

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019) — 首次展示 zero-shot transfer
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3, 2020) — scaling law + in-context learning

### 项目

- 用中文维基百科或小说语料训练一个字符级/子词级 GPT，能续写中文段落

---

## 阶段八：完整 Encoder-Decoder Transformer

**目标**：补全 Cross-Attention，拼接 Encoder + Decoder 做 seq2seq 任务

### 核心内容

| 要点 | 说明 |
|------|------|
| Cross-Attention | Decoder 用 Encoder 输出作为 K/V，自身 hidden state 作为 Q |
| Encoder-Decoder 拼接 | Encoder 编码源序列 → Decoder 自回归生成目标序列 |
| Teacher Forcing | 训练时用真实目标 token 而非模型自己预测的 token 作为输入 |

### 论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — 一切开始的地方

### 项目

- 中英翻译模型（WMT 或 TED 语料），或中文文本摘要

---

## 阶段九：推理加速 — KV-Cache

**目标**：理解自回归推理的瓶颈和缓存优化

### 核心内容

| 要点 | 说明 |
|------|------|
| 原理 | 已生成的 token 的 K/V 不需要重复计算，缓存后只算新 token |
| 实现 | 在 Decoder 层维护 K/V cache，每步追加新 token 的 K/V |
| 效果 | 推理速度从 O(n²) 降到 O(n) |

### 论文

- [Fast Transformer Decoding: One Write-Head Is All You Need](https://arxiv.org/abs/1911.02150) (Multi-Query Attention, 2019)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (2023)

### 项目

- 给阶段七的 GPT 模型加上 KV-Cache，对比加速前后的推理时间

---

## 阶段十：现代位置编码 — RoPE

**目标**：用旋转位置编码替代 Sinusoidal 编码

### 核心内容

| 要点 | 说明 |
|------|------|
| 原理 | 通过旋转矩阵将位置信息注入 Q 和 K 的内积 |
| 相对位置 | RoPE 天然具备相对位置感知，外推性更好 |
| 实现 | 复数旋转 / 实数分块旋转两种写法 |

### 论文

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (2021)
- [ALiBi: Train Short, Test Long](https://arxiv.org/abs/2108.12409) (2021) — 另一种简洁的外推方案

### 项目

- 用 RoPE 改写阶段七的 GPT，对比 Sinusoidal 在长序列上的 perplexity 差异

---

## 阶段十一：高效微调 — LoRA

**目标**：理解大模型微调范式和低秩适配

### 核心内容

| 要点 | 说明 |
|------|------|
| 原理 | 冻结原权重 W，只训练低秩增量 ΔW = AB |
| 变体 | LoRA / QLoRA (4-bit 量化底座) / DoRA |
| 实践 | PEFT 库使用，合并/卸载 adapter 权重 |

### 论文

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (2023)

### 项目

- 用 LoRA 微调一个开源小模型（Qwen2-0.5B / GPT-2）做特定领域文本生成

---

## 阶段十二：模型量化

**目标**：理解模型压缩和端侧部署

### 核心内容

| 要点 | 说明 |
|------|------|
| 量化基础 | INT8 / INT4 量化，对称 vs 非对称，per-tensor vs per-channel |
| GPTQ | 基于 OBQ 的逐层量化，用 Hessian 信息补偿误差 |
| AWQ | 按权重重要性保护显著通道 |

### 论文

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (2023)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) (2023)

### 项目

- 用 GPTQ/AWQ 量化一个 7B 模型，对比量化前后的显存和推理速度

---

## 阶段十三：从人类反馈中学习 — RLHF / DPO

**目标**：理解对齐训练的核心范式

### 核心内容

| 要点 | 说明 |
|------|------|
| RLHF 流程 | SFT → Reward Model → PPO |
| DPO | 跳过 reward model，直接在偏好数据上优化 policy |
| 偏好数据 | 对比对构建、标注偏差处理 |

### 论文

- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) (InstructGPT, 2022)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO, 2023)

### 项目

- 用 DPO 在一个 SFT 模型上做偏好对齐（如 Stack Exchange 偏好数据集）

---

## 阶段十四：进阶架构理解

### 变体速览

| 架构 | 代表模型 | 特点 |
|------|---------|------|
| Prefix Decoder | GLM / ChatGLM | 前缀双向 + 后缀单向 |
| MoE | Mixtral / DeepSeek-V3 | 稀疏激活，参数多但计算少 |
| Mamba / SSM | Mamba / Mamba-2 | 状态空间模型，线性复杂度替代 attention |
| DeepSeek MLA | DeepSeek-V2/V3 | 多头潜在注意力，KV 极致压缩 |
| Hybrid | Jamba | Mamba + Transformer 混合 |

### 论文

- [Mixture of Experts](https://arxiv.org/abs/1701.06538) (2017) — MoE 原始论文
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (2024)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) (2023)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) (2024)

---

## 学习建议

1. **不要跳阶段**。每个阶段写代码到能跑通再去下一阶段。
2. **论文不要通读**。看 Abstract + Figures + 核心公式就行，代码实现才是真理解。
3. **项目要小而完整**。一个 100 行能跑通的 demo 胜过 1000 行没跑通的半成品。
4. **阶段七到十二是按顺序的**，阶段十三、十四可以按兴趣穿插学习。
5. **Tokenizer 别重复造**。从阶段七开始建议用 `tiktoken` 或 `AutoTokenizer`，节省精力聚焦模型本身。
