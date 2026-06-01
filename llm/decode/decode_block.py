"""
Decoder Block：多层 Decoder 堆叠

两种模式:
  - DecodeBlock（无 cross-attention）：GPT / NSP 等自回归任务
  - DecodeBlockWithCrossAttn：翻译等 seq2seq 任务
"""
import torch
from .Decode import MyDecoder
from ..attention import MyMultiHeadAttention


# ======================================================================
# 原始 DecodeBlock（无 cross-attention，保持向后兼容）
# ======================================================================

class DecodeBlock(torch.nn.Module):
    """堆叠 N 层 MyDecoder（纯 self-attention），用于 GPT / NSP 等"""

    def __init__(self, embedding_dim, num_heads, ff_dim, num_layers):
        super(DecodeBlock, self).__init__()
        self.decoders = torch.nn.ModuleList([
            MyDecoder(embedding_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        for decoder in self.decoders:
            x = decoder(x, mask)
        return x


# ======================================================================
# 带 Cross-Attention 的 Decoder Layer（翻译任务用）
# ======================================================================

class DecoderLayer(torch.nn.Module):
    """
    单层 Decoder（含 Cross-Attention）

    结构: Masked Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm
    """

    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()

        # 1. 掩码自注意力
        self.self_attn = MyMultiHeadAttention(embedding_dim, num_heads)

        # 2. 交叉注意力（Q=decoder, K/V=encoder）
        self.cross_attn = MyMultiHeadAttention(embedding_dim, num_heads)

        # 3. 前馈网络
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, ffn_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ffn_dim, embedding_dim),
        )

        # 3 个残差路径的归一化
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)
        self.norm3 = torch.nn.LayerNorm(embedding_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        """
        x:              [batch, tgt_seq_len, dim]
        encoder_output: [batch, src_seq_len, dim]
        self_mask:      [batch, 1, tgt_seq_len, tgt_seq_len]  causal mask
        cross_mask:     [batch, 1, 1, src_seq_len]  padding mask
        """
        # ---- 1. 掩码自注意力 ----
        out = self.self_attn(x, mask=self_mask)
        x = self.norm1(x + self.dropout(out))

        # ---- 2. 交叉注意力：Q=x, K/V=encoder_output ----
        out = self.cross_attn(x, key_value=encoder_output, mask=cross_mask)
        x = self.norm2(x + self.dropout(out))

        # ---- 3. 前馈网络 ----
        out = self.ffn(x)
        x = self.norm3(x + self.dropout(out))

        return x


class DecodeBlockWithCrossAttn(torch.nn.Module):
    """堆叠 N 层 DecoderLayer（含 cross-attention），用于翻译等 seq2seq"""

    def __init__(self, embedding_dim, num_heads, ffn_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        """
        x:              [batch, tgt_len, dim]
        encoder_output: [batch, src_len, dim]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)
        return x
