# 构建transformer翻译模型from
import torch
from torch import nn
from config import *
from MyEmbedding import PositionalEncoding

print('='*60)
print('using device:', DEVICE)
print('='*60)

class TransformerModel(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 max_len=48,
                 pad_idx=0):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(max_len, d_model)

        # Transformer 模块
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
            dropout=0.1
        )
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # 生成 Transformer 必需的掩码
    def create_masks(self, src, tgt_input):
        # 1. 源语言padding掩码：忽略填充0
        src_pad_mask = (src == self.pad_idx)
        # 2. 目标语言padding掩码
        tgt_pad_mask = (tgt_input == self.pad_idx)
        # 3. 前瞻掩码：禁止解码器看到未来单词
        tgt_subseq_mask = self.transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(DEVICE)
        return src_pad_mask, tgt_pad_mask, tgt_subseq_mask
    
    def forward(self, src, tgt_input):
        # 1. 嵌入 + 位置编码
        src_emb = self.src_embedding(src) * torch.sqrt(torch.tensor(self.d_model, device=src.device))
        tgt_emb = self.tgt_embedding(tgt_input) * torch.sqrt(torch.tensor(self.d_model, device=src.device))
        
        # 位置编码
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)

        # 创建掩码
        src_pad_mask, tgt_pad_mask, tgt_subseq_mask = self.create_masks(src, tgt_input)

        # Transformer 前向传播（完全正确）
        output = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_subseq_mask
        )

        # 输出层
        output = self.fc_out(output)
        return output