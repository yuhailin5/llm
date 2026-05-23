# 缩放点积注意力
import torch
import torch.nn.functional as F  # 必须导入，用官方Softmax
class MySDPAttention(torch.nn.Module):

    def __init__(self, embedding_dim):
        super(MySDPAttention, self).__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self,q,k,v,mask=None):
        # 计算权重矩阵 QK^T / sqrt(d_k)
        d_k = q.size(-1)  # 每个头的维度
        scores = torch.matmul(q,k.transpose(-2,-1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)  # 直接在scores上进行softmax计算，节省内存
        output = torch.matmul(scores, v)
        return output