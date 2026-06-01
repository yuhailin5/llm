"""
Transformer 翻译任务训练脚本

数据来源:
  内置: 20 句中英对照（快速验证）
  真实: HuggingFace IWSLT 2017 中英 TED 演讲（~230K）

使用方式:
  python -m llm.model.train_translation                  # 内置数据
  python -m llm.model.train_translation --real           # IWSLT 真实数据
  python -m llm.model.train_translation --real --size 50000  # 取前 5 万条
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import random
import argparse
from torch.utils.data import Dataset, DataLoader
from llm.model.transformer import Transformer

# ============================================================================
# 超参数
# ============================================================================
EMBEDDING_DIM = 256      # 真实数据建议 256-512
NUM_HEADS = 8
FFN_DIM = 512
NUM_LAYERS = 3
MAX_SEQ_LEN = 40         # TED 句子较长，适当增大
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 分词器
# ============================================================================

class CharTokenizer:
    """字符级分词器（无需预训练，直接用字符集构建词表）"""

    def __init__(self):
        self.special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        self.token2id = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.id2token = {i: tok for i, tok in enumerate(self.special_tokens)}
        self._next_id = len(self.special_tokens)

    def fit(self, texts: list[str]):
        for text in texts:
            for ch in text:
                if ch not in self.token2id:
                    self.token2id[ch] = self._next_id
                    self.id2token[self._next_id] = ch
                    self._next_id += 1

    def encode(self, text: str) -> list[int]:
        return [self.token2id.get(ch, self.token2id['<unk>']) for ch in text]

    def decode(self, ids: list[int]) -> str:
        chars = []
        for tid in ids:
            tok = self.id2token.get(tid, '<unk>')
            if tok in self.special_tokens:
                continue
            chars.append(tok)
        return ''.join(chars)

    def vocab_size(self) -> int:
        return len(self.token2id)


# ============================================================================
# 内置测试数据
# ============================================================================

BUILTIN_DATA = [
    ("你好", "hello"), ("谢谢", "thank you"), ("再见", "goodbye"),
    ("猫在桌子上", "the cat is on the table"),
    ("我喜欢学习", "I like learning"),
    ("今天天气很好", "the weather is nice today"),
    ("他是一名学生", "he is a student"),
    ("这本书很有趣", "this book is interesting"),
    ("我们一起去公园", "let us go to the park together"),
    ("她喜欢听音乐", "she likes listening to music"),
    ("北京是中国的首都", "Beijing is the capital of China"),
    ("我昨天看了一部电影", "I watched a movie yesterday"),
    ("请把门打开", "please open the door"),
    ("春天来了花开了", "spring comes and flowers bloom"),
    ("他比我高", "he is taller than me"),
    ("这道菜很好吃", "this dish is very delicious"),
    ("明天会下雨吗", "will it rain tomorrow"),
    ("我住在上海", "I live in Shanghai"),
    ("学习语言很重要", "learning languages is important"),
    ("祝你生日快乐", "happy birthday to you"),
]


# ============================================================================
# Dataset
# ============================================================================

class TranslationDataset(Dataset):
    def __init__(self, pairs, src_tok, tgt_tok, max_len=MAX_SEQ_LEN):
        self.pairs = pairs
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src_ids = [BOS_ID] + self.src_tok.encode(src_text) + [EOS_ID]
        tgt_ids = [BOS_ID] + self.tgt_tok.encode(tgt_text) + [EOS_ID]
        src_ids = self._pad(src_ids)
        tgt_ids = self._pad(tgt_ids)
        return torch.LongTensor(src_ids), torch.LongTensor(tgt_ids)

    def _pad(self, ids):
        if len(ids) < self.max_len:
            return ids + [PAD_ID] * (self.max_len - len(ids))
        return ids[:self.max_len]


# ============================================================================
# 训练 & 评估
# ============================================================================

def run_epoch(model, dataloader, criterion, optimizer=None):
    """一个 epoch 的训练或评估。optimizer=None 时为评估模式"""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_tokens = 0, 0

    for src, tgt in dataloader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        dec_input = tgt[:, :-1]   # Teacher Forcing
        dec_target = tgt[:, 1:]

        with torch.set_grad_enabled(is_train):
            output = model(src, dec_input)
            loss = criterion(output.reshape(-1, output.size(-1)), dec_target.reshape(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * dec_target.numel()
        total_tokens += dec_target.numel()

    return total_loss / total_tokens  # 返回 per-token 平均 loss（≈ perplexity 的 log）


def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR,
          patience=5, save_path='llm/model/transformer_translation.pth'):
    """完整训练流程，含验证集监控和早停"""
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)

        if val_loader:
            val_loss = run_epoch(model, val_loader, criterion, optimizer=None)
            ppl = torch.exp(torch.tensor(val_loss)).item()

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1

            if epoch % 2 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{epochs} | train_loss: {train_loss:.4f}"
                      f" | val_loss: {val_loss:.4f} | val_ppl: {ppl:.1f}")

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        else:
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d}/{epochs} | train_loss: {train_loss:.4f}")

    if not val_loader:
        torch.save(model.state_dict(), save_path)
        print(f"\nFinal loss: {train_loss:.4f}")
    else:
        model.load_state_dict(torch.load(save_path))  # 加载最佳模型
        print(f"\nBest val_loss: {best_val_loss:.4f} | Best val_ppl: {torch.exp(torch.tensor(best_val_loss)).item():.1f}")


# ============================================================================
# 真实数据加载
# ============================================================================

def _setup_hf_mirror():
    """国内用户自动使用 HuggingFace 镜像"""
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def load_opus_data(max_samples=None):
    """
    从 HuggingFace mirror 加载 OPUS-100 中英平行语料

    数据集:
      OPUS-100 en-zh: ~1M 条多领域平行句对
      来源: 字幕/新闻/维基等多源混合
      质量: 中等偏上（自动+人工校验）

    国内镜像:
      set HF_ENDPOINT=https://hf-mirror.com  (脚本会自动设置)
    """
    _setup_hf_mirror()
    try:
        from datasets import load_dataset

        # OPUS-100: en→zh 方向，我们取出来后交换为 zh→en
        ds = load_dataset("opus100", "en-zh", split="train", streaming=True)

        pairs = []
        for item in ds:
            en = item['translation']['en'].strip()
            zh = item['translation']['zh'].strip()
            # 过滤过长/过短/含乱码的句子
            if len(zh) < 2 or len(en) < 2:
                continue
            if len(zh) > 60 or len(en) > 60:
                continue
            pairs.append((zh, en))  # zh → en

            if max_samples and len(pairs) >= max_samples:
                break

        return pairs

    except Exception as e:
        print(f"[WARN] OPUS-100 加载失败: {e}")
        print("  请手动设置镜像: set HF_ENDPOINT=https://hf-mirror.com")
        return None


# ============================================================================
# 数据划分
# ============================================================================

def split_data(pairs, train_ratio=0.90, val_ratio=0.05):
    """
    划分训练/验证/测试集

    规模 < 100:  不划分，全用于训练
    规模 < 1000: 9:1 (train:val)，无测试集
    规模 >= 1000: 按参数比例划分
    """
    random.shuffle(pairs)
    n = len(pairs)

    if n < 100:
        print(f"  数据太少 ({n} 条)，不划分验证集")
        return pairs, [], []

    if n < 1000:
        split = int(n * 0.9)
        return pairs[:split], pairs[split:], []

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    return train_pairs, val_pairs, test_pairs


# ============================================================================
# 翻译测试
# ============================================================================

def evaluate_translations(model, src_tok, tgt_tok, test_pairs, num_samples=5):
    """打印翻译样例 + 计算 token 准确率"""
    model.eval()
    print("\n" + "=" * 60)
    print("翻译样例")
    print("=" * 60)

    samples = random.sample(test_pairs, min(num_samples, len(test_pairs)))

    for zh_text, en_ref in samples:
        src_ids = [BOS_ID] + src_tok.encode(zh_text) + [EOS_ID]
        if len(src_ids) < MAX_SEQ_LEN:
            src_ids += [PAD_ID] * (MAX_SEQ_LEN - len(src_ids))
        src_ids = src_ids[:MAX_SEQ_LEN]
        src_tensor = torch.LongTensor([src_ids]).to(DEVICE)

        translation = model.translate(
            src_tensor, src_tok, tgt_tok,
            max_len=MAX_SEQ_LEN, bos_id=BOS_ID, eos_id=EOS_ID
        )
        print(f"  SRC : {zh_text}")
        print(f"  REF : {en_ref}")
        print(f"  GEN : {translation}")
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Transformer 翻译模型训练")
    parser.add_argument('--real', action='store_true',
                        help='使用 IWSLT 真实数据')
    parser.add_argument('--news', action='store_true',
                        help='使用 news-commentary 数据')
    parser.add_argument('--size', type=int, default=20000,
                        help='真实数据条数（默认 20000）')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--embed_dim', type=int, default=EMBEDDING_DIM)
    args = parser.parse_args()

    # ---- 1. 加载数据 ----
    if args.real:
        print(f"Loading OPUS-100 en-zh (max {args.size} samples)...")
        pairs = load_opus_data(max_samples=args.size)

        if pairs is None:
            print("Falling back to built-in data.")
            pairs = BUILTIN_DATA
    else:
        pairs = BUILTIN_DATA
        print(f"Built-in test data ({len(pairs)} pairs)")

    print(f"  Total pairs: {len(pairs)}")

    # ---- 2. 划分数据 ----
    train_pairs, val_pairs, test_pairs = split_data(pairs)
    print(f"  Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    # ---- 3. 构建分词器 ----
    src_tok = CharTokenizer()
    tgt_tok = CharTokenizer()
    # 只用训练集构建词表（避免验证集泄漏）
    src_tok.fit([p[0] for p in train_pairs])
    tgt_tok.fit([p[1] for p in train_pairs])
    print(f"  Src vocab: {src_tok.vocab_size()} | Tgt vocab: {tgt_tok.vocab_size()}")

    # ---- 4. 模型 ----
    model = Transformer(
        src_vocab_size=src_tok.vocab_size(),
        tgt_vocab_size=tgt_tok.vocab_size(),
        embedding_dim=args.embed_dim,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        pad_idx=PAD_ID,
    ).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 5. DataLoader ----
    train_ds = TranslationDataset(train_pairs, src_tok, tgt_tok, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if val_pairs:
        val_ds = TranslationDataset(val_pairs, src_tok, tgt_tok, MAX_SEQ_LEN)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ---- 6. 训练 ----
    train(model, train_loader, val_loader,
          epochs=args.epochs, lr=LR,
          save_path='llm/model/transformer_translation.pth')

    # ---- 7. 测试 ----
    if test_pairs:
        evaluate_translations(model, src_tok, tgt_tok, test_pairs)
    elif val_pairs:
        evaluate_translations(model, src_tok, tgt_tok, val_pairs)
    else:
        test_sentences = [p[0] for p in train_pairs[:5]]
        for text in test_sentences:
            src_ids = [BOS_ID] + src_tok.encode(text) + [EOS_ID]
            if len(src_ids) < MAX_SEQ_LEN:
                src_ids += [PAD_ID] * (MAX_SEQ_LEN - len(src_ids))
            src_ids = src_ids[:MAX_SEQ_LEN]
            src_tensor = torch.LongTensor([src_ids]).to(DEVICE)
            translation = model.translate(
                src_tensor, src_tok, tgt_tok,
                max_len=MAX_SEQ_LEN, bos_id=BOS_ID, eos_id=EOS_ID
            )
            print(f"  {text}  ->  {translation}")


if __name__ == '__main__':
    main()
