import torch
from attention import MyMultiHeadAttention

class MyEncode(torch.nn.Module):

    def __init__(self,embedding_dim,num_heads,ffn_dim,dropout=0.1):
        super(MyEncode,self).__init__()
        self.embedding_dim = embedding_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        # 多头注意力
        self.mha = MyMultiHeadAttention(embedding_dim,num_heads)

        # 归一化层
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)

        # 前馈网络
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, ffn_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ffn_dim, embedding_dim)
        )
        # dropout层
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self,x):
        # 多头注意力
        attn_output = self.mha(x)
        x = self.norm1(x + self.dropout(attn_output)) # 残差连接和归一化

        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output)) # 残差连接和归一化

        return x