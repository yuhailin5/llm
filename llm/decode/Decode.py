import torch
from attention import MyMultiHeadAttention

class MyDecoder(torch.nn.Module):

    def __init__(self,embeding_dim,num_heads,ffn_dim,dropout=0.1):
        super(MyDecoder,self).__init__()
        self.embeding_dim = embeding_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        # 掩码多头注意力
        self.mha = MyMultiHeadAttention(embeding_dim,num_heads)

        # 归一化层
        self.norm1 = torch.nn.LayerNorm(embeding_dim)
        self.norm2 = torch.nn.LayerNorm(embeding_dim)

        # 前馈网络 先线性变换到ffn_dim维度，经过ReLU激活，再线性变换回embeding_dim维度
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embeding_dim, ffn_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ffn_dim, embeding_dim)
        )

        # dropout层
        self.dropout = torch.nn.Dropout(dropout)
    

    def forward(self,x,mask):
        # 多头注意力
        attn_output = self.mha(x, mask)
        x = self.norm1(x + self.dropout(attn_output)) # 残差连接和归一化

        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output)) # 残差连接和归一化

        return x