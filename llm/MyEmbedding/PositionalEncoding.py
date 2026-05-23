import torch

class PositionalEncoding(torch.nn.Module):
    
    def __init__(self,max_pos,embedding_dim):
        super(PositionalEncoding,self).__init__()
        self.max_pos = max_pos
        self.embedding_dim = embedding_dim

        # 位置矩阵
        pe = torch.zeros(max_pos,embedding_dim)

        position = torch.arange(0,max_pos).unsqueeze(1) # pos信息

        # 公式 pos / 10000^(2i/embedding_dim) -> 取指数避免数值过大
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) *        # 取偶数维度：0,2,4...
            (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        # 新增 batch维度
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        return x + self.pe[:,:x.size(1),:]