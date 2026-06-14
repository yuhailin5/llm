# LLM 详细学习计划 · 阶段十一 ~ 阶段十九

> 承接 [ROADMAP.md](ROADMAP.md) 阶段一~十。
> 每个阶段包含：**产出文件清单** → 核心算法复现 → 项目实战 → 论文阅读清单 → 验收标准。
>
> **原则**：项目驱动，先跑通再优化，每个阶段产出可运行的代码和结构化文件。

---

## 阶段十一：高效微调 — LoRA / QLoRA / DoRA

**前置条件**：阶段七（GPT 预训练）或任意可训练的 GPT 模型

### 11.1 必须复现的算法

| 序号 | 算法 | 说明 |
|------|------|------|
| 11-1 | **LoRA 核心层** | 从零实现 `LoRALinear`，冻结原权重 W，只训练低秩矩阵 A（d×r）和 B（r×d），forward: `y = Wx + (α/r) · BAx` |
| 11-2 | **LoRA 合并/卸载** | `merge()` 将 BA 加回 W 得到无推理开销的权重；`unmerge()` 拆回去继续训练 |
| 11-3 | **多 LoRA 热切换** | 按 task_id 动态切换不同的 A/B 矩阵，实现一个底座服务多个下游任务 |
| 11-4 | **DoRA 分解** | 将 W 分解为 magnitude（标量）和 direction（单位向量），只对 direction 做 LoRA |

### 11.2 产出文件

```
llm/project/lora/
├── lora.py                 # LoRA 核心模块：LoRALinear（低秩分解 A/B、merge/unmerge、多任务热切换）
├── lora_train.py           # 训练脚本：冻结原权重，仅训练 LoRA 参数，对比 full ft 的参数量/显存/速度
├── lora_inference.py       # 推理脚本：加载 LoRA 权重，支持按 task_id 切换不同 adapter
├── config_lora.py          # LoRA 超参（r、alpha、dropout、target_modules）+ 微调数据路径
├── data/
│   ├── base_train.txt      # 底座训练数据（2-3 本金庸小说拼接）
│   └── target_style.txt    # 目标风格数据（用于微调的特定作者小说）
└── bin/
    ├── lora_adapter.pth    # 训练好的 LoRA 权重（A/B 矩阵，通常 < 1MB）
    └── lora_log.json       # 训练日志：loss 曲线、step/s、显存占用
```

| 文件 | 说明 |
|------|------|
| `lora.py` | 核心产出。LoRALinear 从零实现：`y = Wx + (α/r)·B·A·x`，含 merge/unmerge/hot_swap |
| `lora_train.py` | 冻结原权重，只优化 A/B 矩阵，记录参数量/显存/速度对比 |
| `lora_inference.py` | 演示多 LoRA 热切换：同一底座按 task_id 加载不同 adapter |
| `lora_adapter.pth` | 训练产物，通常仅几十 KB~几百 KB |

### 11.3 项目实战

#### 项目 A：LoRA- NovelGPT 风格微调（2 天）

```
目标：用 LoRA 微调你的 NovelGPT，学会一个特定作者的文风

数据集构建：
  - 选取 2-3 本金庸小说作为「底座训练数据」
  - 留 1 本风格迥异的（如《越女剑》短篇）作为「微调目标数据」
  - 构造 prompt-completion 对，或将目标小说切成训练样本

实现步骤：
  1. 实现 LoRALinear 模块（~80 行）
  2. 将 NovelGPT 的 qkv / out_proj / ffn 中的 Linear 替换为 LoRALinear
  3. 冻结原始权重，只训练 LoRA 参数
  4. 对比 full fine-tuning vs LoRA 的：
     - 可训练参数量（LoRA rank=8 时仅 ~0.3% 参数）
     - 训练显存占用
     - 训练速度（step/s）
     - 生成文本的目标风格相似度

关键代码骨架：
  class LoRALinear(nn.Module):
      def __init__(self, linear, r=8, alpha=16, dropout=0.0):
          # linear: 原始 nn.Linear，冻结
          # A: (in_features, r), B: (r, out_features)
          # scaling = alpha / r

验收标准：
  □ LoRA 模型可训练参数量 < 原模型的 1%
  □ 微调后生成文本风格明显偏向目标小说
  □ merge + unmerge 功能正常，merge 后推理结果一致
```

#### 项目 B：QLoRA 量化底座微调（1 天）

```
目标：在 4-bit 量化后的模型上做 LoRA 微调

实现步骤：
  1. 用 bitsandbytes 或自实现 NF4/INT4 量化 NovelGPT 的 Linear 权重
  2. 量化后的 Linear + LoRA adapter 组成 QLoRALinear
  3. 对比 LoRA vs QLoRA 的显存占用

验收标准：
  □ 4-bit 量化后模型大小 ≈ 原始 1/4
  □ QLoRA 可正常训练且 loss 下降
```

### 11.3 论文阅读清单

| 论文 | 重点章节 | 阅读目标 |
|------|---------|---------|
| [LoRA (2021)](https://arxiv.org/abs/2106.09685) | Sec 3 (Method), Sec 4 (Experiments), Sec 7.1 (Which matrices to adapt) | 理解 r=1~4 就够用，理解 α/r 的缩放因 |
| [QLoRA (2023)](https://arxiv.org/abs/2305.14314) | Sec 3 (NF4), Sec 4 (QLoRA), Sec 5 (Experiments) | 理解 NF4 量化 + 双重量化 |
| [DoRA (2024)](https://arxiv.org/abs/2402.09353) | Sec 3 (Method), Fig 1-2 | 理解 magnitude/direction 分解 |

### 11.4 验收标准

- [ ] `LoRALinear` 从零实现，不依赖 PEFT 库
- [ ] 微调后模型生成文本风格明显变化
- [ ] merge/unmerge 功能正常
- [ ] 能解释 rank r 对效果的影响（r=1 vs r=8 vs r=64）

---

## 阶段十二：模型量化 — INT8 / INT4 / GPTQ / AWQ

**前置条件**：阶段七（有可训练的 GPT 模型）

### 12.1 必须复现的算法

| 序号 | 算法 | 说明 |
|------|------|-------------|
| 12-1 | **INT8 对称量化** | `q = round(w / scale)`, `w_hat = q × scale`, 计算 scale = max(abs(w)) / 127 |
| 12-2 | **INT8 非对称量化** | `q = round((w - zero) / scale)`, 支持非对称分布 |
| 12-3 | **Per-channel 量化** | 沿输出维度独立计算 scale，比 per-tensor 精度高得多 |
| 12-4 | **INT4 分组量化** | 每 128 个权重一组，共享 scale + zero point，存储开销约 4.25 bit/weight |
| 12-5 | **GPTQ 逐层量化** | 核心公式：`δ_w = -(w_q - w) × H^(-1)`，用 Hessian 逆补偿量化误差 |
| 12-6 | **AWQ 显著性保护** | 按激活值幅度保护重要通道，对显著通道放大后量化 |

### 12.2 产出文件

```
llm/project/quantization/
├── quantize.py             # 量化核心：对称/非对称量化器、per-tensor/per-channel、INT8/INT4/分组量化
├── quantized_linear.py     # QuantizedLinear 层：存储 int 权重 + scale/zero，forward 时反量化
├── gptq.py                 # GPTQ 核心算法：Hessian 近似计算 + 逐列量化 + 误差补偿 + 秩-1 更新
├── quantize_model.py       # 模型量化脚本：逐层替换 Linear → QuantizedLinear，保存量化权重
├── benchmark_quant.py      # 对比脚本：FP32 vs INT8 vs GPTQ-INT4 的 perplexity / 模型大小 / 推理速度
├── data/
│   └── calibration.txt     # 校准数据（用于 GPTQ 计算 Hessian，几百条就够）
└── bin/
    ├── model_int8.pth      # INT8 量化后的模型权重
    ├── model_gptq_int4.pth # GPTQ INT4 量化后的模型权重
    └── quant_log.json      # 量化精度对比：perplexity、模型大小、推理速度
```

| 文件 | 说明 |
|------|------|
| `quantize.py` | 量化器实现。`quantize(tensor, bits)` → `(q, scale, zero)`；`dequantize(q, scale, zero)` → `tensor` |
| `gptq.py` | GPTQ 算法核心。逐列 round + Hessian 误差重分配，比 naive round 精度高得多 |
| `quantized_linear.py` | 推理用的量化 Linear 层，存 int8/int4 权重，forward 时反量化再计算 |
| `benchmark_quant.py` | 生成对比报告：各种量化策略的 perplexity vs 模型大小 vs 速度 |

### 12.3 项目实战

#### 项目 A：从零实现 INT8 量化推理（1.5 天）

```
目标：将 NovelGPT 量化为 INT8，实现量化推理

实现步骤：
  1. 实现 per-channel 对称量化：
     - quantize(tensor, bits=8) → (q_values, scales)
     - dequantize(q_values, scales) → tensor
  2. 实现 QuantizedLinear：forward 时先 dequantize 再计算
  3. 对 NovelGPT 所有权重矩阵逐层量化
  4. 对比 FP32 vs INT8：
     - 模型文件大小
     - 推理显存占用
     - 生成质量（perplexity 对比）
     - 推理速度（tokens/s）

验收标准：
  □ INT8 模型大小 ≈ FP32 的 1/4
  □ perplexity 下降 < 0.5
  □ 推理速度提升 > 1.3x（受限于 dequantize 开销，真正的加速需要 INT8 kernel）
```

#### 项目 B：GPTQ 逐层量化（2 天）

```
目标：实现简化版 GPTQ 量化你的 NovelGPT

核心算法理解（先手算一个小矩阵）：
  GPTQ 本质是逐行量化 + 用 Hessian 信息修正剩余权重
  
  for col in range(n_columns):
      quantize(W[:, col])
      error = W_quantized[:, col] - W_original[:, col]
      W[:, col+1:] -= H_inv[col, col+1:] / H_inv[col, col] × error
      更新 H_inv（秩-1 更新）

实现步骤：
  1. 用校准数据跑一次 forward，收集每层输入的激活值
  2. 用激活值计算 Hessian 近似：H = X^T X
  3. 实现逐列量化 + Hessian 误差补偿
  4. 对比 round-to-nearest vs GPTQ 的量化精度

验收标准：
  □ GPTQ-INT4 的 perplexity 显著优于 round-to-nearest INT4
  □ 理解为什么逐列量化 + 误差补偿比直接 round 好
  □ 量化后的模型能正常生成通顺文本
```

### 12.3 论文阅读清单

| 论文 | 重点章节 | 阅读目标 |
|------|---------|---------|
| [GPTQ (2023)](https://arxiv.org/abs/2210.17323) | Sec 2 (Background), Sec 3 (GPTQ Algorithm), Algorithm 1 | 逐行量化 + Hessian 补偿的完整流程 |
| [AWQ (2023)](https://arxiv.org/abs/2306.00978) | Sec 3 (Method), Fig 1-3 | 理解「显著通道」的概念，为什么 1% 的通道决定 99% 的质量 |

### 12.4 验收标准

- [ ] 从零实现 INT8/INT4 对称 & 非对称量化
- [ ] GPTQ 核心算法从零实现（不依赖 AutoGPTQ 库）
- [ ] 量化前后的 perplexity 对比有数据支撑
- [ ] 理解 per-channel vs per-tensor 的差异

---

## 阶段十三：偏好对齐 — DPO（优先）/ RLHF

**前置条件**：阶段七（GPT 模型）+ 基本的 SFT 能力

> **建议先做 DPO**，RLHF 的 PPO 训练极不稳定，DPO 直接优化偏好数据，更简单且效果相当。

### 13.1 必须复现的算法

| 序号 | 算法 | 说明 |
|------|------|------|
| 13-1 | **DPO Loss** | `L_DPO = -log σ(β × (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))` |
| 13-2 | **Reward Model** | 用偏好数据训练一个打分的 reward model（RLHF 路线需要） |
| 13-3 | **PPO with KL Penalty** | 用 reward model 引导 policy 优化，同时加 KL 散度约束不要跑太远 |

### 13.2 产出文件

```
llm/project/dpo/
├── dpo_loss.py             # DPO Loss 实现：log-sigmoid 公式，含 reference model 冻结逻辑
├── dpo_dataset.py          # 偏好数据集构建：从原始数据构造 (prompt, chosen, rejected) 三元组
├── dpo_train.py            # DPO 训练脚本：policy + reference 双模型，计算 log-prob 差异
├── dpo_eval.py             # 对齐评估：对抗性 prompt 测试，对比 chosen vs rejected 的概率
├── data/
│   ├── preference_pairs.jsonl  # 偏好数据（手动标注或自动构造，~300-500 对）
│   └── test_prompts.txt        # 对抗性测试 prompt 集
└── bin/
    ├── dpo_policy.pth       # DPO 训练后的 policy 模型
    └── dpo_log.json         # 训练日志：DPO loss、reward margin、chosen/rejected 概率差
```

| 文件 | 说明 |
|------|------|
| `dpo_loss.py` | DPO 核心 Loss。`L = -log σ(β·Δlog_prob)`，需正确处理 ref model 的 log-prob |
| `dpo_dataset.py` | 偏好数据构建与加载，支持多种来源 |
| `dpo_train.py` | 训练循环：保持 ref model 冻结，仅更新 policy，监控 chosen/rejected 的概率差 |
| `dpo_eval.py` | 自动评估：用对抗性 prompt 生成响应，无需人工即可判断对齐效果 |

### 13.3 项目实战

#### 项目 A：DPO 对齐 NovelGPT（2.5 天）

```
目标：用 DPO 让你的 NovelGPT 在特定维度上对齐（如「不写暴力内容」）

数据集构建（小规模，~500 对就够）：
  方案A（人工）：自己写 50 个 prompt，每个 prompt 写 2 个 response（好/差）
  方案B（自动）：用 NovelGPT 生成多个 response，用规则/人工标注好坏
  方案C（开源）：用 Anthropic/hh-rlhf 或 Stanford SHP 数据集
  
  数据格式：
  {
    "prompt": "段誉面对敌人，他决定",
    "chosen": "以理服人，晓之以情动之以理",     # ← 人类偏好的续写
    "rejected": "拔出剑来，一剑刺向对方心口"    # ← 人类不想要的续写
  }

实现步骤：
  1. 准备偏好数据集（至少 300 对）
  2. 复制一份 NovelGPT 作为 reference model（冻结）
  3. 实现 DPO Loss：
     - 对同一 prompt，分别用 policy 和 reference 计算 chosen/rejected 的 log-prob
     - 代入 DPO 公式计算 loss
  4. 训练 policy model
  5. 测试：用对抗性 prompt 检验模型是否真的「学乖了」

验收标准：
  □ DPO Loss 正确实现，不调包
  □ chosen response 的概率明显高于 rejected
  □ 模型保留了语言能力（perplexity 不崩）
  □ 能解释 β 参数对训练的影响
```

#### 项目 B（选做）：RLHF 迷你版（2 天）

```
目标：跑通 SFT → Reward Model → PPO 的完整管线

实现步骤：
  1. SFT：用高质量单轮对话数据微调 NovelGPT
  2. Reward Model：用偏好数据训练一个打分模型（架构 = NovelGPT + value head）
  3. PPO：用 reward model 引导 SFT 模型优化
     - 实现 KL 散度惩罚项
     - 实现 advantage 计算（可用简单 REINFORCE 替代 GAE）

验收标准：
  □ 理解 RLHF 三个阶段的角色
  □ 理解为什么需要 KL 惩罚（防止 reward hacking）
  □ 对比 DPO 和 RLHF 的实现复杂度
```

### 13.3 论文阅读清单

| 论文 | 重点章节 | 阅读目标 |
|------|---------|---------|
| [InstructGPT (2022)](https://arxiv.org/abs/2203.02155) | Sec 3 (Methods), Fig 2 (三步流程) | 理解 RLHF 的完整管线 |
| [DPO (2023)](https://arxiv.org/abs/2305.18290) | Sec 3 (Derivation), Sec 4 (Experiments), Eq 7 | 理解 DPO 如何从 RLHF 目标推导出 |
| [ORPO (2024)](https://arxiv.org/abs/2403.07691) | Sec 2 (Method) | 将 SFT + 对齐合并为一个 loss（简化版 DPO） |

### 13.4 验收标准

- [ ] DPO Loss 从零实现并验证正确性（用一个小数值例子手工验证）
- [ ] 对齐后的模型在目标维度上有可观测的行为变化
- [ ] Reference model 冻结正确（梯度不传播到 ref）
- [ ] 能说清楚 DPO vs RLHF 的取舍

---

## 阶段十四：进阶架构理解 — MoE / Mamba / MLA

**前置条件**：阶段七~十所有内容

> 本阶段为「广度理解」，每个架构做一个最小可运行 demo 即可，不需要完整训练。

### 14.1 必须复现的算法/模型

| 序号 | 算法 | 说明 |
|------|------|-------------|
| 14-1 | **MoE Layer** | 实现 Top-K gating + 稀疏 FFN experts + load balancing loss |
| 14-2 | **Mamba Block** | 实现 SSM 核心：A 矩阵离散化（ZOH）、选择性扫描 |
| 14-3 | **MLA (Multi-head Latent Attention)** | 实现 KV 先压缩到低维潜在空间再展开 |

### 14.2 产出文件

```
llm/project/advanced_arch/
├── moe_layer.py            # MoE FFN：Top-K Gating + N 个 Expert FFN + Load Balancing Loss
├── mamba_block.py          # Mamba Block：S6 选择性 SSM + ZOH 离散化 + 选择性扫描
├── mla_attention.py        # MLA：KV 潜在压缩 + 解压到各 head，对比 MHA 的 KV Cache 大小
├── arch_compare.py         # 对比脚本：Transformer vs MoE vs Mamba 的速度/显存/perplexity
├── train_demo.py           # 每个架构的最小训练 demo
├── data/
│   └── demo_corpus.txt     # 小规模测试语料
└── bin/
    └── arch_log.json       # 架构对比数据
```

| 文件 | 说明 |
|------|------|
| `moe_layer.py` | MoE FFN 从零实现。Router（Top-K softmax gating）+ N 个 Expert + 辅助负载均衡 loss |
| `mamba_block.py` | Mamba Block 核心。Δ 随输入变化（选择性），A 矩阵 ZOH 离散化，parallel scan |
| `mla_attention.py` | MLA 简化实现。KV 先压到 latent_dim 再解压，对比 MHA 的 cache 省多少 |
| `arch_compare.py` | 统一对比框架：同数据/同参数量下三种架构的 perf 对比 |

### 14.3 项目实战

#### 项目 A：MoE- NovelGPT（2 天）

```
目标：将 NovelGPT 的 FFN 替换为 MoE，理解稀疏激活

实现步骤：
  1. 实现 MoE FFN：
     - Gating Network：Linear → Softmax → Top-K（K=2）
     - N 个 Expert（每个就是一个小 FFN）
     - Load Balancing Loss：鼓励 tokens 均匀分配到各 expert
  2. 用 MoE-FFN 替换 NovelGPT 中的普通 FFN
  3. 配置：4 experts，Top-2，对比原模型：
     - 总参数量（增加约 3x）
     - 实际激活参数量（仅增加 ~1.1x）
     - 每 step 训练速度
     - Loss 下降曲线

验收标准：
  □ Gating 网络能学到有意义的专家分工（不同 expert 激活比例不同）
  □ Load balancing loss 有效防止 expert collapse
  □ 推理时 Top-1 vs Top-2 的速度/质量差异有数据
```

#### 项目 B：Mamba 迷你版（2 天）

```
目标：实现一个 Mamba Block，与 Transformer 对比

实现步骤：
  1. 实现 Mamba 核心组件：
     - S6 选择性 SSM：A, B, C 矩阵参数化
     - Δ（离散化步长）随输入变化
     - 选择性扫描（parallel scan 或 sequential）
  2. 堆叠 8 个 Mamba Block 作为语言模型
  3. 在同数据上对比 Mamba vs Transformer：
     - 长序列（T=1024, 2048）上的训练速度
     - 显存占用
     - Perplexity

验收标准：
  □ Mamba 模型能生成通顺文本
  □ 在 T=1024 时显存显著低于 Transformer（O(n) vs O(n²)）
  □ 理解 SSM 的「选择性」机制——为什么比 S4 好
```

#### 项目 C：MLA 理解复现（1 天）

```
目标：实现简化版 MLA，理解 KV 压缩

DeepSeek MLA 核心思想：
  标准 MHA：Q, K, V 各 (B, n_head, T, head_dim) → KV Cache = 2 × n_head × T × head_dim
  MLA：K = W_DK × c, V = W_DV × c，其中 c 是压缩后的潜在向量
      → KV Cache = 2 × T × latent_dim  ← 不再乘 n_head，大幅减小

实现步骤：
  1. 实现 MLA Block：
     - 输入 x → Linear → 压缩到 latent_dim（如 128）
     - 从 latent 解压出 K 和 V（每个 head）
     - Q 正常计算
     - 用 F.scaled_dot_product_attention 做注意力
  2. 对比 MHA vs MLA 的 KV Cache 大小
  3. 在 NovelGPT 上测试生成质量

验收标准：
  □ 理解 MLA 的压缩比：latent_dim / (n_head × head_dim)
  □ 对比 MHA 和 MLA 的 KV Cache 大小（MLA 可省 5-10x）
```

### 14.3 论文阅读清单

| 论文 | 重点章节 | 阅读目标 |
|------|---------|---------|
| [MoE (2017)](https://arxiv.org/abs/1701.06538) | Sec 2 (Routing), Sec 3 (Load Balancing) | 理解专家路由和负载均衡 |
| [Mixtral (2024)](https://arxiv.org/abs/2401.04088) | Sec 2 (Architecture), Sec 3 (Results) | 理解 MoE 在大模型中的实际表现 |
| [Mamba (2023)](https://arxiv.org/abs/2312.00752) | Sec 3 (Selective SSM), Algorithm 1-2 | 理解选择性扫描机制 |
| [DeepSeek-V2 (2024)](https://arxiv.org/abs/2405.04434) | Sec 2.2 (MLA), Fig 3 | 理解 MLA 的 KV 压缩方法 |

### 14.4 验收标准

- [ ] 理解 MoE 的 trade-off：参数量 ↑ vs 计算量 ≈ 不变
- [ ] 理解 Mamba 的线性复杂度从何而来
- [ ] 理解 MLA 如何在不损质量的前提下压缩 KV Cache
- [ ] 每个架构至少有一个可运行的最小 demo

---

## 阶段十五：训练工程化 — 分布式训练 + 混合精度 + 数据管线

**前置条件**：阶段七~十四的核心模型实现

### 15.1 必须掌握的技术

| 序号 | 技术 | 说明 |
|------|------|-------------|
| 15-1 | **Gradient Accumulation** | 小 batch 模拟大 batch，`loss / accum_steps` |
| 15-2 | **Distributed Data Parallel (DDP)** | 多卡数据并行，AllReduce 同步梯度 |
| 15-3 | **FSDP / ZeRO** | 分片优化器状态 + 梯度 + 参数（可训练更大模型） |
| 15-4 | **Flash Attention 2/3** | 理解分块计算 + online softmax |
| 15-5 | **混合精度训练** | FP16/BF16 + Loss Scaling（阶段七优化版已有，深化理解） |
| 15-6 | **数据管线** | WebDataset / Mosaic Streaming，处理 TB 级语料 |

### 15.2 产出文件

```
llm/project/distributed/
├── ddp_train.py            # DDP 训练脚本：torchrun 启动，DDP 包装模型，只在 rank=0 保存 ckpt
├── launch_ddp.sh           # torchrun 启动命令：指定 nproc_per_node、master_port
├── streaming_dataset.py    # 流式数据加载器：读取预编码的 .bin 文件，不全部加载到内存
├── data_pipeline.py        # 数据预处理管线：语料下载 → BPE 编码 → 存为二进制 .bin + .idx 索引
├── gradient_accum.py       # 梯度累积对比实验：accum_steps=1/2/4/8 对 loss/batch_size 的影响
├── checkpoint_resume.py    # 断点续训：保存/加载 optimizer + scheduler + dataloader 状态
├── data/
│   ├── corpus/             # 更大规模语料（中文维基 + 新闻等）
│   └── encoded/            # BPE 编码后的 .bin 文件
└── bin/
    ├── ddp_checkpoint.pth  # DDP 训练 checkpoint
    └── ddp_log.json        # 训练日志：单卡 vs 双卡加速比、显存、吞吐
```

| 文件 | 说明 |
|------|------|
| `ddp_train.py` | DDP 训练入口。`torch.distributed.init_process_group` + `DDP(model)`，处理 rank 逻辑 |
| `launch_ddp.sh` | 一行启动：`torchrun --nproc_per_node=2 ddp_train.py` |
| `streaming_dataset.py` | 流式 IterableDataset，按需从磁盘 memmap 读取 token 序列，不占内存 |
| `data_pipeline.py` | 语料 → tokenize → 存 .bin 的完整管线，处理 GB 级数据 |
| `checkpoint_resume.py` | 训练中断恢复：保存 global_step、optimizer/scheduler state_dict、rng 状态 |

### 15.3 项目实战

#### 项目：DDP 训练 NovelGPT + 数据管线升级（3 天）

```
目标：用 DDP 在 2+ 卡上训练，数据管线支持流式读取

实现步骤：
  1. DDP 改造：
     - torchrun 启动脚本
     - 模型用 DDP 包装
     - 只在 rank=0 保存 checkpoint
  2. 数据管线：
     - 数据预处理：下载更大语料（如中文维基 + 新闻）→ BPE 编码 → 存为 .bin
     - 实现流式数据加载器（不把所有数据读进内存）
  3. 配置对比：
     - 单卡 vs 双卡 DDP 的加速比
     - 梯度累积 steps=1/2/4/8 对 loss 的影响

验收标准：
  □ DDP 双卡加速比 > 1.8x
  □ 数据加载不再一次性吃完内存
  □ 训练过程支持中断恢复（checkpoint 保存/加载）
```

### 15.3 论文阅读清单

| 论文 | 重点章节 | 阅读目标 |
|------|---------|---------|
| [Flash Attention (2022)](https://arxiv.org/abs/2205.14135) | Sec 2 (Background), Sec 3 (Algorithm) | 理解 tiling + online softmax |
| [ZeRO (2020)](https://arxiv.org/abs/1910.02054) | Sec 3-5 (ZeRO-1/2/3) | 理解三种分片层级 |

### 15.4 验收标准

- [ ] DDP 训练能跑通，加速比正常
- [ ] 支持梯度累积和恢复训练
- [ ] 理解 ZeRO-1/2/3 的区别

---

## 阶段十六：推理优化 — KV Cache 进阶 + Speculative Decoding

**前置条件**：阶段九（基础 KV Cache）+ 阶段十四 MLA

### 16.1 必须复现的算法

| 序号 | 算法 | 说明 |
|------|------|------|
| 16-1 | **GQA (Grouped Query Attention)** | 多个 Q 头共享一组 K/V，减少 KV Cache |
| 16-2 | **PagedAttention** | 将 KV Cache 按 page 管理，减少显存碎片（vLLM 的核心） |
| 16-3 | **Speculative Decoding** | 用草稿模型快速生成候选，目标模型并行验证 |

### 16.2 产出文件

```
llm/project/inference/
├── attention_variants.py   # 三种 Attention：MHA (n_kv=n_head) / GQA (n_kv=n_head/4) / MQA (n_kv=1)
├── kv_cache.py             # KV Cache 实现：prefill 阶段缓存 K/V，decode 阶段只算新 token
├── speculative_decode.py   # Speculative Decoding：draft model 生成候选 → target model 并行验证
├── benchmark_inference.py  # 推理性能对比：有/无 cache、MHA/GQA/MQA、speculative 的 tokens/s
├── data/
│   └── test_prompts.txt    # 标准测试 prompt 集
└── bin/
    ├── draft_model.pth     # 草稿模型（小模型 2 层）
    └── inference_log.json  # 推理性能数据
```

| 文件 | 说明 |
|------|------|
| `attention_variants.py` | MHA/GQA/MQA 统一实现，通过 `n_kv_heads` 参数切换，含 `repeat_interleave` 的 K/V 复制 |
| `kv_cache.py` | 完整 KV Cache：prefill（并行，存 cache）+ decode（逐 token，追加 cache），验证与无 cache 结果一致 |
| `speculative_decode.py` | Draft-verify-accept 循环。γ=4 候选 → target 一次验证 → 概率接受/拒绝 → 重新采样 |
| `benchmark_inference.py` | 生成同样 512 tokens，对比不同策略的 wall-clock 延迟和显存 |

### 16.3 项目实战

#### 项目 A：GQA + KV Cache 完整实现（2 天）

```
目标：实现 MHA/GQA/MQA 三种 Attention，对比 KV Cache 和生成速度

实现步骤：
  1. 实现三种 Attention 模式：
     - MHA：n_kv_heads = n_heads
     - GQA：n_kv_heads = n_heads / 4
     - MQA：n_kv_heads = 1
  2. 为每种模式实现完整的 KV Cache（prefill + decode）
  3. 对比三种模式在 512 tokens 生成中的：
     - KV Cache 显存占用
     - 每个 decode step 的延迟
     - 生成质量（perplexity）
  
  关键：GQA 的 K/V 复制逻辑
    k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)

验收标准：
  □ KV Cache 正确实现（验证：有 cache 和无 cache 的生成结果完全一致）
  □ GQA 的 KV Cache 比 MHA 省 4x
  □ 生成速度（tokens/s）比无 cache 提升 > 10x（512 token 生成）
```

#### 项目 B：Speculative Decoding（2 天）

```
目标：用小模型 + 大模型配合加速推理

实现步骤：
  1. 训练一个小模型（如 2 层）和大模型（8 层）
  2. Speculative Decoding 流程：
     a. 小模型自回归生成 γ=4 个候选 token
     b. 大模型一次 forward 验证这 4 个 token 的概率
     c. 接受概率最高的前 k 个，拒绝的重新采样
  3. 对比原速度 vs Speculative Decoding

验收标准：
  □ 理解 speculative decoding 的「无损加速」原理
  □ wall-clock 延迟降低 > 1.5x
```

### 16.3 论文阅读清单

| 论文 | 重点章节 | 阅读目标 |
|------|---------|---------|
| [GQA (2023)](https://arxiv.org/abs/2305.13245) | Sec 2 (Method), Sec 3 (Experiments) | MHA → GQA 的 uptraining 转换方法 |
| [PagedAttention (2023)](https://arxiv.org/abs/2309.06180) | Sec 3 (PagedAttention), Fig 3-4 | 理解 page table 管理 KV Cache |
| [Speculative Decoding (2023)](https://arxiv.org/abs/2211.17192) | Sec 2 (Method), Algorithm 1 | 理解 draft-verify-accept 流程 |

### 16.4 验收标准

- [ ] GQA/MQA/MHA 三种 Attention 全部实现
- [ ] KV Cache 在训练和推理中均可正常工作
- [ ] Speculative Decoding 实现并验证无损加速

---

## 阶段十七：RAG — 检索增强生成

**前置条件**：阶段七（GPT 模型）+ 基础 Embedding 知识

### 17.1 必须复现的算法

| 序号 | 算法 | 说明 |
|------|------|-------------|
| 17-1 | **Dense Retrieval** | 双塔模型：query encoder + doc encoder，用对比学习训练 |
| 17-2 | **Chunking 策略** | 固定大小 / 语义分块 / 递归分块 |
| 17-3 | **Reranking** | 粗召回 → 精排的两阶段检索 |
| 17-4 | **Self-RAG** | 模型自己判断是否需要检索、检索结果是否相关 |

### 17.2 产出文件

```
llm/project/rag/
├── chunker.py              # 文本分块器：固定大小分块 / 段落分块 / 滑动窗口 + overlap
├── embedding_index.py      # 向量检索引擎：sentence-transformers 编码 + FAISS/numpy 余弦检索
├── rag_pipeline.py         # RAG 主流程：Query → Retrieve Top-K → Build Prompt → Generate → 返回
├── eval_rag.py             # RAG 评估：有 RAG vs 无 RAG 的事实准确率、检索命中率
├── build_knowledge_base.py # 知识库构建脚本：小说 → 分块 → embedding → 索引，一键构建
├── data/
│   ├── novels/             # 5 本金庸小说原文
│   └── chunks/             # 分块后的文本片段
├── index/                  # FAISS 向量索引文件
└── bin/
    └── rag_log.json        # 评估日志：检索延迟、Top-K 命中率、回答准确率
```

| 文件 | 说明 |
|------|------|
| `chunker.py` | 多种分块策略实现，支持固定 token 数 / 段落分割 / 递归分块，overlap 可控 |
| `embedding_index.py` | 向量检索引擎，支持 cosine/L2 距离，可切换 FAISS 加速或 numpy 手写 |
| `rag_pipeline.py` | 端到端管线：接收问题 → 检索 → 构建 in-context prompt → 模型生成 |
| `eval_rag.py` | 量化评估：对比有/无 RAG 生成的答案质量，检索 recall@k |

### 17.3 项目实战

#### 项目：金庸小说 RAG 问答系统（3 天）

```
目标：搭建完整的 RAG 系统，能用 NovelGPT 回答金庸小说相关问题

系统架构：
  用户问题 → Embedding 检索 → 取 Top-K 相关段落 → NovelGPT 生成回答

实现步骤：
  1. 构建知识库：
     - 将 5 本金庸小说按段落分块（chunk_size=256 tokens）
     - 用 sentence-transformers 或训练一个简单 encoder 做 embedding
     - 存到向量索引（用 FAISS 或 numpy 手写余弦相似度搜索）
  2. 检索模块：
     - 用户问题 → embedding → 检索 Top-5 相关块
  3. 生成模块：
     - Prompt = "参考以下段落：{retrieved_chunks}\n\n问题：{question}\n\n回答："
     - NovelGPT 生成回答
  4. 对比评估：
     - 有 RAG vs 无 RAG 的回答准确率
     - 不同 chunk_size 的检索效果
     - 不同 Top-K 对生成质量的影响

验收标准：
  □ RAG 系统能回答小说中的人物、情节问题
  □ 有 RAG 比无 RAG 的事实准确率显著提升
  □ 检索延迟 < 100ms（在 5 本书的规模上）
```

### 17.3 论文阅读清单

| 论文 | 重点章节 | 阅读目标 |
|------|---------|---------|
| [RAG (2020)](https://arxiv.org/abs/2005.11401) | Sec 2 (Method), Fig 1 | 理解 RAG 的两种形式：RAG-Sequence vs RAG-Token |
| [Self-RAG (2023)](https://arxiv.org/abs/2310.11511) | Sec 2 (Method), Fig 2 | 理解 reflection tokens 的自检机制 |

### 17.4 验收标准

- [ ] 完整 RAG 管线跑通（检索 + 生成）
- [ ] 理解 embedding 质量对 RAG 效果的巨大影响
- [ ] 理解 chunk_size 和 overlap 的 trade-off

---

## 阶段十八：Agent & Tool Use — 工具调用与自主智能体

**前置条件**：阶段七（GPT）+ 阶段十七（RAG 理解）

### 18.1 必须复现的算法

| 序号 | 算法 | 说明 |
|------|------|-------------|
| 18-1 | **ReAct** | Reason + Act 交替，Thought → Action → Observation 循环 |
| 18-2 | **Function Calling** | 定义 tool schema，模型生成 JSON 格式的函数调用 |
| 18-3 | **Tool-Integrated LLM** | 模型训练时融入 tool use 数据，学会何时调用工具 |

### 18.2 产出文件

```
llm/project/agent/
├── tools.py                # 工具集定义：search_character / search_chapter / compare_characters / calculator
├── tool_schema.py          # Tool Schema 定义：工具名、描述、参数 JSON Schema（OpenAI Function Calling 格式）
├── react_agent.py          # ReAct Agent 主循环：Thought → Action → Observation → ... → Final Answer
├── agent_prompt.py         # System Prompt 模板 + Few-shot 示例（引导模型正确格式化工具调用）
├── eval_agent.py           # Agent 评估：工具调用成功率、多步推理正确率、格式解析率
├── data/
│   ├── character_db.json   # 角色知识库（武功、门派、关系）
│   └── test_questions.json # Agent 测试问题集（含预期工具调用和答案）
└── bin/
    └── agent_log.json      # Agent 运行日志：每步的 thought/action/observation/结果
```

| 文件 | 说明 |
|------|------|
| `tools.py` | Agent 可调用的工具函数实现，每个工具返回结构化 JSON 或文本 |
| `tool_schema.py` | 工具的 JSON Schema 定义（类似 OpenAI Function Calling 格式），模型据此生成调用 |
| `react_agent.py` | ReAct 主循环。解析 Thought/Action/Action Input → 执行工具 → 拼接 Observation → 循环 |
| `agent_prompt.py` | System Prompt 工程：定义输出格式、工具描述、Few-shot 示例 |
| `eval_agent.py` | 自动化评估：检测格式解析错误、工具调用幻觉、死循环 |

### 18.3 项目实战

#### 项目：金庸知识 Agent（3 天）

```
目标：构建一个能调用工具的 Agent，回答需要「查资料 + 推理」的问题

工具集：
  1. search_character(name) → 返回角色信息（武功、门派、关系）
  2. search_chapter(book, ch) → 返回章节摘要
  3. compare_characters(a, b) → 返回两角色的武功/关系对比
  4. calculator(expr) → 执行计算（如「计算段誉一共会多少种武功」）

实现步骤：
  1. 设计 System Prompt + Tool Schema：
     system = """你是金庸武侠专家。你可以使用以下工具：
     - search_character(name): 查询角色信息
     - search_chapter(book, chapter): 查询章节内容
     - compare_characters(a, b): 对比两个角色
     - calculator(expr): 执行数学计算
     
     回答问题时，格式为：
     Thought: <你的推理>
     Action: <tool_name>
     Action Input: <tool_params>
     """
  
  2. 实现 ReAct Loop：
     while not finished:
         response = model.generate(prompt)
         if "Final Answer:" in response:
             break
         parse Action + Action Input → 执行工具 → 拼接 Observation → 继续生成
  
  3. 测试用例：
     - "令狐冲和杨过谁的剑法更高？列出对比依据"
     - "段誉虚竹乔峰三人中，会北冥神功的有谁？"
     - "计算郭靖学过的武功总数"

验收标准：
  □ Agent 能自主选择何时调用工具
  □ 工具调用格式解析正确率 > 90%
  □ 能应对多步推理问题（需要调用 2+ 个工具）
  □ 有最大步数上限防止死循环
```

### 18.3 论文阅读清单

| 论文 | 重点章节 | 阅读目标 |
|------|---------|---------|
| [ReAct (2022)](https://arxiv.org/abs/2210.03629) | Sec 2 (Method), Sec 3 (Experiments) | 理解 Reasoning + Acting 的协同 |
| [Toolformer (2023)](https://arxiv.org/abs/2302.04761) | Sec 2 (Method), Fig 1 | 理解自监督学习工具调用的方法 |
| [Function Calling (OpenAI)](https://platform.openai.com/docs/guides/function-calling) | 文档 | 理解工业界的 Tool Schema 设计 |

### 18.4 验收标准

- [ ] ReAct Agent 能完成多步推理 + 工具调用
- [ ] 理解 Tool Schema 设计原则（description 质量决定调用准确率）
- [ ] 理解 Agent 的 failure mode：幻觉调用、死循环、格式错误

---

## 阶段十九（选做）：从零预训练一个小型 LLM

**前置条件**：阶段七~十八的核心能力

> 这是终极项目——将之前所有阶段串联起来。

### 19.1 产出文件

```
llm/project/mini_llm/
├── model/
│   ├── transformer_block.py # 优化版 Transformer Block：Pre-LN + GQA + SwiGLU FFN + RoPE
│   ├── attention.py         # GQA Attention：Q 独立、K/V 分组共享，支持 `is_causal=True`
│   ├── rope.py              # RoPE 位置编码：复数旋转实现，支持外推
│   ├── config.py            # 模型超参：layers=12, d_model=768, n_heads=12, n_kv_heads=3, block=1024
│   └── minilm.py            # MiniLLM 完整组装：Embedding → Blocks → LayerNorm → LM Head
├── tokenizer/
│   ├── train_tokenizer.py   # BPE 分词器训练：vocab_size=32000，HuggingFace tokenizers
│   └── tokenizer.json       # 训练好的分词器
├── data/
│   └── preprocess.py        # 数据预处理：原始文本 → BPE 编码 → .bin 文件（支持流式处理）
├── train/
│   ├── train.py             # 训练脚本：AMP + 梯度累积 + DDP（可选）+ Wandb/TensorBoard 日志
│   ├── scheduler.py         # Warmup + Cosine 学习率调度（阶段七复用）
│   └── checkpoint.py        # Checkpoint 保存/加载：model + optimizer + scheduler + rng_state
├── inference/
│   ├── generate.py          # 文本生成：temperature/top-k/top-p 采样 + KV Cache
│   └── serve.py             # 简单 HTTP 服务（FastAPI）暴露生成 API
├── eval/
│   ├── perplexity.py        # Perplexity 评估（在 hold-out 集上）
│   └── lm_eval_harness.py   # 接入 lm-evaluation-harness 做标准 benchmark
├── export/
│   └── export_onnx.py       # 模型导出 ONNX / TorchScript，用于部署
├── scripts/
│   ├── download_corpus.sh   # 语料下载脚本（中文维基 dump、新闻、小说）
│   └── run_pretrain.sh      # 一行启动预训练
├── data/
│   ├── corpus/              # 原始语料
│   ├── encoded/             # BPE 编码后的 .bin
│   └── eval/                # 验证集
├── logs/                    # 训练日志 + TensorBoard
├── checkpoints/             # 模型 checkpoint
└── README.md                # 阶段总结：模型设计、训练心得、遇到的问题
```

| 文件 | 说明 |
|------|------|
| `minilm.py` | 150M GPT 模型完整组装，从零实现所有组件 |
| `train_tokenizer.py` | 在大语料上训练 BPE 分词器，vocab=32000 |
| `train.py` | 完整训练脚本，支持 AMP/梯度累积/DDP/日志/断点续训 |
| `generate.py` | 推理生成，含 KV Cache 加速 |
| `export_onnx.py` | 模型导出，便于部署 |

### 19.2 项目：MiniLLM-150M（5 天）

```
目标：在 10-20GB 中文语料上从零训练一个 1.5 亿参数的 GPT

模型规格：
  - Layers: 12
  - d_model: 768
  - n_heads: 12
  - vocab_size: 32000
  - block_size: 1024
  - 总参数: ~150M

技术栈（前面的积累全用上）：
  □ RoPE 位置编码（阶段十）
  □ GQA Attention（阶段十六）
  □ Pre-LN + Flash Attention（阶段十四）
  □ SwiGLU FFN（阶段十四的 MoE 简化版）
  □ AMP 混合精度（阶段十五）
  □ Gradient Accumulation（阶段十五）
  □ 数据管线（阶段十五）
  □ BPE Tokenizer（阶段二 + HuggingFace tokenizers）

训练目标：
  - 语料：中文维基百科 + 新闻语料 + 小说（10-20GB text）
  - 训练 1-3 个 epoch
  - Perplexity < 20（中文）
  - 能生成通顺的中文段落

验收标准：
  □ 所有组件自己实现
  □ 训练稳定不崩（loss 曲线平滑下降）
  □ 生成文本通顺、无明显语法错误
  □ 模型可导出为 ONNX 或 TorchScript 用于部署
```

### 19.2 验收标准

- [ ] 完整理解从 tokenizer 训练到模型部署的全链路
- [ ] 所有组件（tokenizer, embedding, attention, ffn, norm, head）均为自实现
- [ ] 能解释每个设计选择的原因（为什么用 RoPE 而不是 Sinusoidal？为什么 Pre-LN？）

---

## 附录 A：推荐学习节奏

| 周次 | 阶段 | 内容 | 时间 |
|------|------|------|------|
| 第 1 周 | 十一-A | LoRA 核心实现 + NovelGPT 风格微调 | 3 天 |
| 第 2 周 | 十一-B + 十二-A | QLoRA + INT8 量化实现 | 3 天 |
| 第 3 周 | 十二-B | GPTQ 量化算法实现 | 2 天 |
| 第 4 周 | 十三-A | DPO 对齐训练 | 3 天 |
| 第 5 周 | 十四-A | MoE 实现 | 2 天 |
| 第 6 周 | 十四-B | Mamba 实现 | 2 天 |
| 第 7 周 | 十五 | DDP 训练 + 数据管线 | 3 天 |
| 第 8 周 | 十六-A | GQA + KV Cache 完整实现 | 2 天 |
| 第 9 周 | 十六-B | Speculative Decoding | 2 天 |
| 第 10 周 | 十七 | RAG 系统 | 3 天 |
| 第 11 周 | 十八 | Agent & Tool Use | 3 天 |
| 第 12-13 周 | 十九（选做） | MiniLLM-150M | 5 天 |

> **总计约 8-11 周**（可并行推进的部分可以压缩到 6-8 周）。

## 附录 B：每个阶段的文件组织建议

```
llm/project/<stage>/
  ├── model.py          # 模型定义
  ├── train.py          # 训练脚本
  ├── inference.py      # 推理/生成脚本
  ├── data/
  │   └── *.txt         # 数据文件
  ├── bin/              # 模型权重
  └── README.md         # 阶段总结：遇到的问题和解决思路
```

## 附录 C：关键的「不要」

1. **不要一上来就调包**。每个阶段的核心算法先从零实现，理解原理后再用 PEFT/transformers 库
2. **不要在阶段十一~十三跳过阶段九的 KV Cache**。推理加速和微调是正交的
3. **不要在 1700 万参数模型上做 RLHF**。DPO 更简单效果更好
4. **阶段十四的 MoE/Mamba/MLA 做最小 demo 就行**，不要追求完整训练
5. **时刻对照 ROADMAP.md 的阶段七~十**，确保基础扎实
