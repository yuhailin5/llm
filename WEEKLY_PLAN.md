# 📋 LLM 学习周计划

> 生成日期: 2026-06-03 | 当前分支: master | 最近提交: `adda8ad` 复现transformer翻译任务

---

## 一、当前进度总览

```
阶段一   [████████] DL基础 (Titanic/Otto)
阶段二   [████████] 分词器 (BPE + WordPiece)
阶段三   [████████] 嵌入层 (Embedding + PositionalEncoding)
阶段四   [████████] 注意力 (SDPA + MultiHeadAttention)
阶段五   [████████] Transformer层 (Encoder + Decoder)
阶段六   [████████] NSP项目 (滕王阁序 上下句生成)
阶段七   [░░░░░░░░] GPT 自回归预训练           ← 待开始
阶段八   [████░░░░] Encoder-Decoder 翻译        ← 代码OK，数据有问题
阶段九   [░░░░░░░░] KV-Cache 推理加速
阶段十   [░░░░░░░░] RoPE 旋转位置编码
阶段十一  [░░░░░░░░] LoRA 高效微调
阶段十二  [░░░░░░░░] 模型量化 (GPTQ/AWQ)
阶段十三  [░░░░░░░░] RLHF / DPO
阶段十四  [░░░░░░░░] 进阶架构 (MoE/Mamba/MLA)
```

### 已有代码资产

| 模块 | 位置 | 状态 |
|------|------|------|
| BPE/WordPiece 分词器 | `llm/MyTokenizer/` | ✅ 完整可用 |
| Embedding + Sinusoidal PE | `llm/MyEmbedding/` | ✅ 完整可用 |
| MultiHeadAttention | `llm/attention/` | ✅ 完整可用 |
| Encoder / Decoder | `llm/encode/` `llm/decode/` | ✅ 完整可用 |
| Transformer (seq2seq) | `llm/model/transformer.py` | ✅ 结构正确 |
| 翻译训练脚本 | `llm/model/train_translation.py` | ✅ 支持内置/OPUS数据 |
| NSP 滕王阁序 | `llm/project/NSP.py` | ✅ 完整项目 |

---

## 二、本周总体目标（6天）

```
           Mon       Tue       Wed       Thu       Fri       Sat
阶段八修复  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
阶段七     ░░░░████████████████████████████░░░░░░░░░░░░░░░░░░░░
阶段九     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████████
```

**三件事:**
1. **周一**: 快速修复阶段八翻译 — 用字符级分词器 + 内置20句跑通，验证模型能力
2. **周二-周四**: 完成阶段七 GPT 自回归预训练 — 数据管道 + 训练 + 多策略生成
3. **周五-周六**: 完成阶段九 KV-Cache — 给 GPT 加上缓存，实测加速比

---

## 三、每日详细计划

### 周一 6/4 | 修复翻译 + 验证 Encoder-Decoder

**目标**: 证明你的 Transformer 代码能正常翻译，换掉有问题的数据即可

| 时间 | 任务 | 产出 |
|------|------|------|
| 上午 | 阅读 `train_translation.py`，理解 char-level tokenizer 方案 | 理解代码逻辑 |
| 上午 | 运行内置20条数据训练: `python -m llm.model.train_translation` | 得到可翻译的模型 |
| 下午 | 测试翻译效果: "你好→hello"、"猫在桌子上→the cat is on the table" | 验证模型能学会 |
| 下午 | 尝试加载 OPUS-100 数据: `python -m llm.model.train_translation --real --size 5000` | 更大规模验证 |
| 晚上 | **产出**: 记录实验结果到 `translate.ipynb`，标注数据问题的根因 | 笔记 |

**验收标准**: 模型翻译 "你好" → "hello"，"北京是中国的首都" → 通顺英文（不等同reference但语义对）

**关键提醒**:
- 先不跑 `--real`，内置 20 条能过再扩展，避免浪费时间
- char-level tokenizer 无需预训练，零门槛启动
- 如果 OPUS-100 下载慢（镜像问题），直接用内置 20 条验证完就进入下一阶段

---

### 周二 6/5 | 阶段七: GPT 数据管道 + 滑动窗口

**目标**: 为 GPT 自回归训练准备好数据流

| 时间 | 任务 | 产出 |
|------|------|------|
| 上午 | 选语料: 中文维基百科 或 金庸小说 txt（推荐: 找一篇长文先验证） | `data/corpus.txt` |
| 上午 | 实现 `GPTDataset`: 文档拼接 → 滑动窗口 → (input, target) 对 | `llm/data/gpt_dataset.py` |
| 下午 | 可视化验证: 取几个样本，打印 `input[i] → target[i]` 确认因果掩码逻辑 | Jupyter 验证 |
| 下午 | 写单元测试: 检查 input/target 的 offset=1 关系、padding 处理 | test |
| 晚上 | **产出**: `llm/data/gpt_dataset.py` 完成并验证 | 代码 |

**核心理解**:
```
原文: "今天天气很好，我们去公园"
tokenize: [1, 234, 567, 89, 12, 345, 67, 2]
                            ↓
input:  [1, 234, 567, 89, 12]      # 不含最后1个token
target: [234, 567, 89, 12, 345]    # 不含第1个token (bos)
                            ↓
loss = CrossEntropy(pred[0], target[0]) + CrossEntropy(pred[1], target[1]) + ...
# 每个位置都在预测"下一个词"，这就是 Causal LM
```

**滑动窗口示意**:
```
长文档: [tok1, tok2, tok3, ..., tok1000]
max_len=128, stride=64

窗口1: [tok1  ... tok128]  →  input[:127], target[1:128]
窗口2: [tok65 ... tok192]  →  input[:127], target[1:128]
窗口3: [tok129... tok256]  →  ...
```

---

### 周三 6/6 | 阶段七: GPT 模型 + 训练循环

**目标**: 复用已有 Decoder 组件，搭出 GPT 并跑通训练

| 时间 | 任务 | 产出 |
|------|------|------|
| 上午 | 写 `GPTModel`: DecodeBlock(无cross-attn) + Embedding + output_proj | `llm/model/gpt.py` |
| 上午 | 复用已有 `DecodeBlock`（`llm/decode/decode_block.py`），无需改代码 | — |
| 下午 | 写训练循环: Causal LM loss + 梯度裁剪 + 学习率 warmup | `llm/model/train_gpt.py` |
| 下午 | 小规模训练: 1000条语料、embed_dim=128、2层，先确认 loss 下降 | 验证 |
| 晚上 | **产出**: GPT模型训练成功，train_loss 持续下降 | 代码+模型 |

**GPT 模型结构** (对比已有的 Transformer):

```
Transformer (翻译)              GPT (自回归)
─────────────────────────      ────────────────────
src_embedding + src_pe         embedding + pe        ← 复用 MyEmbedding
encoder (EncodeBlock)          [不需要]
tgt_embedding + tgt_pe         [同一个 embedding]
decoder (cross-attn)           decoder (纯 self-attn) ← 复用 DecodeBlock
output_proj                    output_proj
```

**关键**: GPT 只有一条路径 — Embedding → Decoder → LM Head，比翻译模型更简单！

---

### 周四 6/7 | 阶段七: 多策略文本生成

**目标**: 实现 4 种生成策略，理解各自效果差异

| 时间 | 任务 | 产出 |
|------|------|------|
| 上午 | 实现 `generate()`: greedy decoding（基础版） | 模型能续写 |
| 上午 | 实现 temperature sampling: `softmax(logits/T)` | 多样性控制 |
| 下午 | 实现 top-k sampling: 只从概率最高的k个中采样 | 避免低概率词 |
| 下午 | 实现 top-p (nucleus) sampling: 累积概率≥p 的 token 集合中采样 | 动态候选集 |
| 晚上 | **产出**: 4种策略对比，同一 prompt 不同输出，记录分析 | 笔记 |

**对比验证**:
```
prompt: "深度学习是"
temperature=0.0 (greedy):   "深度学习是一种人工智能技术，它通过神经网络..."   (确定的)
temperature=0.8:             "深度学习是一种基于数据驱动的方法，可以自动提取..." (多样)
temperature=1.5:             "深度学习是一个很广泛的话来说这个领域的研究方向..."    (开始乱了)
top-k=50, temp=0.8:          "深度学习是一种机器学习方法，通过多层神经网络..."     (质量好)
top-p=0.9, temp=0.8:         "深度学习是近年来人工智能领域的重要突破..."          (质量好)
```

---

### 周五 6/8 | 阶段九: KV-Cache 原理 + 实现

**目标**: 理解自回归推理的冗余计算，用缓存消除它

| 时间 | 任务 | 产出 |
|------|------|------|
| 上午 | 读论文 Multi-Query Attention / GQA 的 KV-Cache 部分 | 理解原理 |
| 上午 | 在 `GPTModel` 中加 `use_cache` 参数，修改 `forward()` 接收 `past_kv` | 模型改造 |
| 下午 | 实现带缓存的 `generate_with_cache()`: 首 token 完整前传，后续只算增量 | 推理代码 |
| 下午 | 对比实验: 相同 prompt，128 token 生成，测有/无 cache 的推理时间 | 数据 |
| 晚上 | **产出**: 加速比报告（预期 5-20x），记录到笔记 | 代码+数据 |

**KV-Cache 原理**:

```
无 Cache (naive):                     有 Cache:
Step 0: [a]      → 全序列 attention    Step 0: [a]      → 全序列 attention, 存 K₀,V₀
Step 1: [a,b]    → 全序列 attention    Step 1: [b]      → 只算新token, K₁V₁ append
Step 2: [a,b,c]  → 全序列 attention    Step 2: [c]      → 只算新token, K₂V₂ append
...                                    ...
复杂度: O(n²) per step                 复杂度: O(n) per step（只看新token）
总计算: 1+2+3+...+n = n(n+1)/2         总计算: 1+1+1+...+1 = n
```

**实现要点**:
```python
# 伪代码
def forward(self, x, past_kv=None):
    # 1. 计算当前 Q, K, V
    q, k, v = self.qkv_proj(x)
    
    # 2. 如果有缓存，拼接历史 K, V
    if past_kv is not None:
        past_k, past_v = past_kv
        k = torch.cat([past_k, k], dim=1)  # 沿序列维度拼接
        v = torch.cat([past_v, v], dim=1)
    
    # 3. 正常的 attention 计算
    output = self.attention(q, k, v)
    
    # 4. 返回输出 + 当前层的 K, V (给下一步用)
    return output, (k, v)
```

---

### 周六 6/9 | 阶段九收尾 + 周总结

**目标**: 整理代码、写周报、规划下周

| 时间 | 任务 | 产出 |
|------|------|------|
| 上午 | KV-Cache 完善: 处理多层 decoder 的缓存传递、batch 推理 | 完善代码 |
| 上午 | 性能基准测试: 不同生成长度(32/64/128/256)的加速比曲线 | 图表 |
| 下午 | 整理本周所有代码，确保目录结构清晰 | 目录整理 |
| 下午 | 写本周学习笔记: 关键收获、踩坑记录、下周计划 | 笔记 |
| 晚上 | Git commit: 按阶段拆分提交 | commit |

**本周建议的 commit 拆分**:
```
[阶段八] 修复翻译数据问题，验证 Encoder-Decoder 能力
[阶段七] 实现 GPT 数据管道 + 滑动窗口
[阶段七] 实现 GPT Causal LM 模型 + 训练
[阶段七] 实现 4 种文本生成策略
[阶段九] 实现 KV-Cache 推理加速
[文档] 更新周计划和学习笔记
```

---

## 四、目录结构规划

本周结束后的目录结构:

```
d:\workspace\py\
├── ROADMAP.md                          # 总路线
├── WEEKLY_PLAN.md                      # 本周计划（本文件）
├── README.md
├── llm/
│   ├── data/
│   │   ├── gpt_dataset.py              # [新] GPT 滑动窗口数据集
│   │   └── corpus/                     # [新] 训练语料
│   │       └── sample_zh.txt
│   ├── model/
│   │   ├── gpt.py                      # [新] GPT 模型
│   │   ├── train_gpt.py                # [新] GPT 训练脚本
│   │   ├── generate.py                 # [新] 4种生成策略
│   │   ├── kv_cache.py                 # [新] KV-Cache 实现
│   │   ├── transformer.py              # [已有] seq2seq Transformer
│   │   └── train_translation.py        # [已有] 翻译训练
│   ├── project/
│   │   ├── translate.ipynb             # [已有] 翻译实验（补充记录）
│   │   ├── gpt_pretrain.ipynb          # [新] GPT 训练实验
│   │   └── NSP.py                      # [已有] 滕王阁序
│   ├── MyTokenizer/                    # [已有]
│   ├── MyEmbedding/                    # [已有]
│   ├── attention/                      # [已有]
│   ├── encode/                         # [已有]
│   ├── decode/                         # [已有]
│   └── bin/                            # [已有] 模型+分词器
└── dl/                                 # [已有] 阶段一
```

---

## 五、学习建议

1. **先跑通再优化**。一天一个可运行的最小版本，不要陷入调参。
2. **阶段七用 char-level tokenizer 快速启动**，ROADMAP 也建议 "Tokenizer 别重复造，从阶段七开始用简单方案"。
3. **生成策略理解比实现重要**，温度/top-k/top-p 的数学原理弄懂，面试常考。
4. **KV-Cache 是面试高频题**，不仅要实现，要能说清楚为什么 `O(n²)` 变 `O(n)`。
5. **遇到问题先查自己的代码**，Transformer 的 shape 错误占了 80% 的 debug 时间。
6. **每天 commit 一次**，即使代码没跑通，记录 WIP 也有价值。

---

## 六、下周预告

- 阶段九深入: Multi-Query Attention / GQA
- 阶段十: RoPE 替代 Sinusoidal PE，对比长序列 perplexity
- 阶段十一: LoRA 微调开源小模型

---

*Plan generated with Claude Code — 2026-06-03*
