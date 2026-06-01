# 编码堆叠块
import torch
from .encode import MyEncoder

class EncodeBlock(torch.nn.Module):

    def __init__(self, embedding_dim, num_heads, ff_dim,num_layers):
        super(EncodeBlock, self).__init__()
        
        # 创建多个编码器层
        self.encoder_layers = torch.nn.ModuleList([
            MyEncoder(embedding_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        # 依次通过每一层编码器，传递 mask 填充掩码，以确保模型不会关注填充位置
        for layer in self.encoder_layers:
            x = layer(x, mask=mask)
        return x