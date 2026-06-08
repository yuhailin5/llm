# GPT Decode—Only 模型
import torch
import torch.nn as nn
from config import VOCAB_SIZE, BLOCK_SIZE, D_MODEL, N_HEAD, NUM_LAYERS
from torch.nn import functional as F

class NovelGPT(nn.Module):

    def __init__(self):
        super(NovelGPT,self).__init__()
        self.em = nn.Embedding(VOCAB_SIZE, D_MODEL)  # 词嵌入层 TODO 词嵌入过程
        self.pos_encode = nn.Embedding(BLOCK_SIZE, D_MODEL)  # 位置编码层

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=N_HEAD,
                dim_feedforward=D_MODEL*4,
                batch_first=True,
                activation="gelu"
            ) for _ in range(NUM_LAYERS)
        ])

        self.ln_f = nn.LayerNorm(D_MODEL)  # 最后层归一化
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)  # 输出层


    def forward(self, x, targets=None):
        B, T = x.shape
        # Causal Mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        # 嵌入 + 位置编码
        token_embeddings = self.em(x)  # (B, T, D_MODEL)
        pos_emb = self.pos_encode(torch.arange(T, device=x.device))
        x = token_embeddings + pos_emb  # (B, T, D_MODEL)
        #  transformer 层
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)

        # 计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return logits, loss