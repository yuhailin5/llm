import torch
from attention import MyMultiHeadAttention
# 1. 构造输入
batch_size = 2
seq_len = 4
embedding_dim = 128
num_heads = 4

x = torch.randn(batch_size, seq_len, embedding_dim)

# 2. 【关键】生成 Decoder 专用的因果掩码（下三角为1，上三角为0） # tril: 生成下三角矩阵 unsqueeze(0): 升维
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
# mask shape: (1, 1, seq_len, seq_len) 适配多头维度

# 3. 初始化掩码多头注意力
masked_mha = MyMultiHeadAttention(embedding_dim, num_heads)

# 4. Decoder 前向传播（传入mask）
output = masked_mha(x, mask)

print(x)