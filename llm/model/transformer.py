"""
完整 Transformer（seq2seq 翻译模型）

架构:
  src → Embedding + PE → Encoder Stack ──┐
                                         ├→ Decoder Stack → Linear → tgt
  tgt → Embedding + PE ──────────────────┘

参考论文: Attention Is All You Need (Vaswani et al., 2017)
"""
import torch
from ..encode import EncodeBlock
from ..decode import DecodeBlockWithCrossAttn
from ..MyEmbedding import MyEmbedding, PositionalEncoding


class Transformer(torch.nn.Module):
    """Seq2seq Transformer for translation tasks.

    使用示例:
        model = Transformer(
            src_vocab_size=5000, tgt_vocab_size=5000,
            embedding_dim=256, num_heads=8, ffn_dim=512,
            num_layers=3, max_seq_len=128,
        )
        output = model(src_ids, tgt_ids)  # [batch, tgt_len, tgt_vocab_size]
    """

    def __init__(self,
                 src_vocab_size: int,        # 源语言词表大小
                 tgt_vocab_size: int,        # 目标语言词表大小
                 embedding_dim: int = 256,
                 num_heads: int = 8,
                 ffn_dim: int = 512,
                 num_layers: int = 3,
                 max_seq_len: int = 128,
                 dropout: float = 0.1,
                 pad_idx: int = 0):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx

        # 源语言
        self.src_embedding = MyEmbedding(src_vocab_size, embedding_dim)
        self.src_pe = PositionalEncoding(max_seq_len, embedding_dim)

        # 目标语言
        self.tgt_embedding = MyEmbedding(tgt_vocab_size, embedding_dim)
        self.tgt_pe = PositionalEncoding(max_seq_len, embedding_dim)

        # ---- Encoder / Decoder 堆叠 ----
        self.encoder = EncodeBlock(embedding_dim, num_heads, ffn_dim,
                                   num_layers)
        self.decoder = DecodeBlockWithCrossAttn(embedding_dim, num_heads, ffn_dim,
                                                num_layers, dropout)

        # ---- 输出投影 ----
        self.output_proj = torch.nn.Linear(embedding_dim, tgt_vocab_size)

        self.dropout = torch.nn.Dropout(dropout)

    def _create_pad_mask(self, x):
        """
        填充掩码
        创建 padding mask：标记 pad=0 的位置为 False
        """
        # x: [batch, seq_len] → mask: [batch, 1, 1, seq_len]
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def _create_causal_mask(self, seq_len, device):
        """创建 causal mask（下三角），防止看到未来 token"""
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt):
        """
        src: [batch, src_seq_len]  — 源语言 token ids
        tgt: [batch, tgt_seq_len]  — 目标语言 token ids（训练时右移一位）

        返回: [batch, tgt_seq_len, tgt_vocab_size]
        """
        # ---- Encoder ----
        src_mask = self._create_pad_mask(src)
        src_emb = self.dropout(self.src_pe(self.src_embedding(src)))
        encoder_output = self.encoder(src_emb, src_mask)

        # ---- Decoder ----
        causal_mask = self._create_causal_mask(tgt.size(1), tgt.device)
        tgt_pad_mask = self._create_pad_mask(tgt)
        # 合并 causal + padding mask（都用 bool 避免类型冲突）
        self_mask = causal_mask.bool() & tgt_pad_mask

        tgt_emb = self.dropout(self.tgt_pe(self.tgt_embedding(tgt)))
        decoder_output = self.decoder(tgt_emb, encoder_output, self_mask, src_mask)

        return self.output_proj(decoder_output)

    @torch.no_grad()
    def translate(self, src, tokenizer_src, tokenizer_tgt,
                  max_len=50, bos_id=1, eos_id=2):
        """
        贪心解码：逐 token 生成翻译结果

        src:          [1, src_seq_len] 源语言 ids
        tokenizer_*:  对应的分词器（用于解码输出文本）
        返回:         翻译文本字符串
        """
        self.eval()

        # Encoder 只跑一次
        src_mask = self._create_pad_mask(src)
        src_emb = self.dropout(self.src_pe(self.src_embedding(src)))
        encoder_output = self.encoder(src_emb, src_mask)

        # 从 <bos> 开始自回归生成
        generated = [bos_id]

        for _ in range(max_len):
            tgt = torch.tensor([generated], device=src.device)  # [1, cur_len]

            cur_len = tgt.size(1)
            self_mask = torch.tril(torch.ones(cur_len, cur_len, device=src.device))
            self_mask = self_mask.unsqueeze(0).unsqueeze(0)

            tgt_emb = self.dropout(self.tgt_pe(self.tgt_embedding(tgt)))
            dec_out = self.decoder(tgt_emb, encoder_output, self_mask, src_mask)

            # 取最后一个 token 的 logits
            logits = self.output_proj(dec_out[:, -1, :])  # [1, vocab_size]
            next_token = logits.argmax(dim=-1).item()

            generated.append(next_token)
            if next_token == eos_id:
                break

        return tokenizer_tgt.decode(generated)
