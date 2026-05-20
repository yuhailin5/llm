# 缩放点积注意力
import torch

class MySDPAttention(torch.nn.Module):

    def __init__(self, embedding_dim):
        super(MySDPAttention, self).__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self,q,k,v):
        # 计算权重矩阵 QK^T / sqrt(d_k)
        d_k = q.size(-1)  # 每个头的维度
        scores = torch.matmul(q,k.transpose(-2,-1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        torch.softmax(scores, dim=-1, out=scores) # 直接在scores上进行softmax计算，节省内存
        output = torch.matmul(scores, v)
        return output