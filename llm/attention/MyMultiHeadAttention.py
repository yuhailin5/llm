import torch
from .MySDPAttention import MySDPAttention

class MyMultiHeadAttention(torch.nn.Module):

    def __init__(self, embedding_dim, num_heads):
        super(MyMultiHeadAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # 定义线性层用于生成Q、K、V
        self.q_linear = torch.nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = torch.nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = torch.nn.Linear(embedding_dim, embedding_dim)

        # 输出线性层
        self.out_linear = torch.nn.Linear(embedding_dim, embedding_dim)

        # 缩放点积注意力
        self.sdp_attention = MySDPAttention(self.head_dim)

    def forward(self,x,mask=None):
        batch_size = x.size(0)

        # 线性变换生成Q、K、V
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算多头注意力
        attn_output = self.sdp_attention(q, k, v, mask)

        # 将多头输出拼接并通过输出线性层
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        output = self.out_linear(attn_output)

        return output