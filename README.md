# 大模型学习记录

## 项目结构

```
py/
├── dl/                          # 深度学习基础阶段
│   ├── stage1/                  # 阶段一：经典分类任务
│   │   └── demo1.ipynb          # Titanic & Otto 分类实战
│   └── data/                    # 数据集（Titanic / Otto / MNIST）
│
└── llm/                         # 大模型核心组件
    ├── MyTokenizer/             # 分词器
    │   ├── BPETokenizer.py      # Byte Pair Encoding 实现
    │   ├── WordPieceTokenizer.py# WordPiece 实现
    │   └── tokenizer.ipynb      # BPE vs WordPiece 对比实验
    ├── MyEmbedding/             # 嵌入层
    │   ├── Embeding.py          # Token Embedding
    │   └── PositionalEncoding.py# 正弦/余弦位置编码
    ├── attention/               # 注意力机制
    │   ├── MySDPAttention.py    # 缩放点积注意力
    │   └── MyMultiHeadAttention.py # 多头注意力
    ├── encode/                  # Transformer 编码器
    │   └── encode.py            # Encoder Layer（MHA + FFN + 残差归一化）
    ├── decode/                  # Transformer 解码器
    │   └── Decode.py            # Decoder Layer（Masked MHA + FFN + 残差归一化）
    ├── project/                 # 实战项目
    │   └── NSP.py               # 滕王阁序上下句生成模型
    ├── bin/                     # 模型权重与分词器文件
    │   ├── bpe_tokenizer.bin    # 训练好的 BPE 分词器
    │   └── poem_nsp_model.pth   # 训练好的诗句生成模型
    └── test/                    # 单元测试
```

## 学习路线

### 阶段一：深度学习基础（`dl/`）

- Titanic 生存预测、Otto 商品分类等经典任务
- 掌握数据预处理、模型训练、提交流程

### 阶段二：分词器（`llm/MyTokenizer/`）

**BPE（Byte Pair Encoding）**

- 从 256 个基础字节出发，反复合并频率最高的字节对
- 构建词表直到达到 `max_vocab_size`
- 编码时按训练顺序重放合并规则，解码时拼接字节还原文本
- 支持特殊 token（`<|im_start|>` / `<|im_end|>` / `<|padding|>` 等）

**WordPiece**

- 与 BPE 的区别：合并时选择使语言模型似然增益最大的字节对，而非单纯频率最高

### 阶段三：嵌入层（`llm/MyEmbedding/`）

**Token Embedding**

- `nn.Embedding(vocab_size, embedding_dim)`
- 输出乘以 $\sqrt{d_{model}}$ 保持数值稳定

**Positional Encoding**
$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

### 阶段四：注意力机制（`llm/attention/`）

**缩放点积注意力（SDPA）**
$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- 支持 causal mask，将未来位置填充为 $-\infty$

**多头注意力（MHA）**

- 将 Q/K/V 分拆成多个头并行计算注意力，再拼接输出

### 阶段五：编码器 & 解码器（`llm/encode/` & `llm/decode/`）

| 组件              | 结构                                       |
| ----------------- | ------------------------------------------ |
| **Encoder Layer** | MHA → Add & Norm → FFN → Add & Norm        |
| **Decoder Layer** | Masked MHA → Add & Norm → FFN → Add & Norm |

两者均含残差连接（Add）与层归一化（LayerNorm），以及 Dropout 正则化。

### 阶段六：实战项目（`llm/project/`）

**滕王阁序上下句生成（NSP）**

以《滕王阁序》为语料，训练一个 Decoder-only 的小型语言模型：

1. 将古文按标点切分为"上句 → 下句"训练对
2. 用 BPE 分词器对文本编码
3. 搭建 2 层 Decoder 结构（Embedding + Masked MHA + FFN）
4. 输入上句，预测下句

```
输入上句：落霞与孤鹜齐飞
模型输出：秋水共长天一色
```

**模型超参数**

| 参数          | 值             |
| ------------- | -------------- |
| vocab_size    | 3000           |
| embedding_dim | 128            |
| num_heads     | 4              |
| ffn_dim       | 256            |
| max_seq_len   | 20             |
| epochs        | 50             |
| optimizer     | Adam (lr=1e-3) |
