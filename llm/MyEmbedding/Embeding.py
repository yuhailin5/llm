import torch.nn as nn


# embeding层
class MyEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(MyEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids):
        # 将嵌入向量乘以嵌入维度的平方根，以便在训练过程中保持数值稳定性
        return self.embedding(input_ids)*self.embedding.embedding_dim ** 0.5
