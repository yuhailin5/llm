"""
run.py —— 中译英 Transformer 翻译模型 完整训练管线
====================================================
自包含单文件：包含 config、分词器训练、数据加载、模型定义、训练循环、推理。
直接运行:  python run.py

数据流:
  cn.txt + en.txt
       │
       ▼
  [共享分词器] → 中英混合语料 → 一个 spm_bpe.model
       │
       ▼
  [数据加载] → [(src, tgt), ...] → Dataset → DataLoader
       │                                       │
       ▼                                       ▼
  [模型] TransformerModel(共享Embedding)       (src_batch, tgt_input, tgt_label)
       │
       ▼
  [训练循环] → transformer_translation.pth
       │
       ▼
  [推理测试] translate() 输出示例翻译结果
"""

import os
import math
from typing import List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# 第 1 部分：配置
# ═══════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data', 'transformer_train')
DATA_DIR = os.path.normpath(DATA_DIR)

CN_FILE        = os.path.join(DATA_DIR, 'cn.txt')
EN_FILE        = os.path.join(DATA_DIR, 'en.txt')
CORPUS_FILE    = os.path.join(DATA_DIR, 'corpus.txt')       # 混编语料
SPM_MODEL      = os.path.join(DATA_DIR, 'spm_bpe')          # 共享分词器前缀
SPM_MODEL_FILE = SPM_MODEL + '.model'
SAVE_PATH      = os.path.join(BASE_DIR, 'transformer_translation.pth')

# 超参数
VOCAB_SIZE       = 4000      # 共享词表大小（中英混合）
MAX_LEN          = 64        # 最大序列长度
D_MODEL          = 256       # embedding / 隐层维度
NHEAD            = 4         # d_model // nhead = 64
NUM_ENCODER      = 2         # 编码器层数
NUM_DECODER      = 2         # 解码器层数
DROPOUT          = 0.3       # 强力 dropout 防过拟合
BATCH_SIZE       = 32
LR               = 3e-4
EPOCHS           = 150       # 小数据多跑几轮
WARMUP_STEPS     = 400       # 学习率 warmup

# 特殊 token ID（与 SentencePiece 训练参数一致）
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print('=' * 60)
print(f'DEVICE: {DEVICE}')
print(f'DATA:   {DATA_DIR}')
print(f'MODEL:  d_model={D_MODEL}  layers={NUM_ENCODER}/{NUM_DECODER}  heads={NHEAD}')
print(f'TRAIN:  batch={BATCH_SIZE}  lr={LR}  epochs={EPOCHS}  max_len={MAX_LEN}')
print(f'TOKEN:  shared vocab={VOCAB_SIZE}')
print('=' * 60)


# ═══════════════════════════════════════════════════════════════
# 第 2 部分：位置编码
# ═══════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """正弦位置编码，与 "Attention Is All You Need" 完全一致"""

    def __init__(self, max_pos: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_pos, d_model)
        position = torch.arange(0, max_pos).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_pos, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ═══════════════════════════════════════════════════════════════
# 第 3 部分：共享分词器
# ═══════════════════════════════════════════════════════════════

def make_corpus():
    """将中英文混合写入一个文件，用于训练共享 BPE"""
    with open(CORPUS_FILE, 'w', encoding='utf-8') as out:
        for lang_file in (CN_FILE, EN_FILE):
            with open(lang_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line + '\n')
    print(f'  [OK] 混编语料: {CORPUS_FILE}')


def train_tokenizer(vocab_size: int = VOCAB_SIZE):
    """在混编语料上训练共享 SentencePiece BPE 分词器"""
    make_corpus()

    spm.SentencePieceTrainer.train(
        input=CORPUS_FILE,
        model_prefix=SPM_MODEL,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        max_sentence_length=4192,
        num_threads=4,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
    )
    print(f'  [OK] 共享分词器: {SPM_MODEL_FILE}  (vocab={vocab_size})')


def ensure_tokenizer() -> spm.SentencePieceProcessor:
    """确保共享分词器存在，不存在则训练"""
    # 删除旧的分离式分词器模型，避免混淆
    for old in ('cn_spm_bpe.model', 'cn_spm_bpe.vocab', 'en_spm_bpe.model', 'en_spm_bpe.vocab'):
        old_path = os.path.join(DATA_DIR, old)
        if os.path.exists(old_path):
            os.remove(old_path)
            print(f'  [清理] 已删除旧分词器: {old}')

    if not os.path.exists(SPM_MODEL_FILE):
        print('\n[步骤 1/4] 训练共享分词器...')
        train_tokenizer(VOCAB_SIZE)
    else:
        print('\n[步骤 1/4] 共享分词器已存在，跳过训练')

    sp = spm.SentencePieceProcessor(model_file=SPM_MODEL_FILE)
    print(f'  vocab size: {sp.vocab_size()}')
    return sp


# ═══════════════════════════════════════════════════════════════
# 第 4 部分：数据加载
# ═══════════════════════════════════════════════════════════════

def load_data(src_file: str, tgt_file: str) -> List[Tuple[str, str]]:
    """加载平行语料，返回 [(cn, en), ...]"""
    pairs = []
    with open(src_file, 'r', encoding='utf-8') as f_src, \
         open(tgt_file, 'r', encoding='utf-8') as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            if src_line and tgt_line:
                pairs.append((src_line, tgt_line))
    return pairs


class TranslationDataset(Dataset):
    """翻译数据集：文本 → 共享分词 → padding → tensor"""

    def __init__(self, data: List[Tuple[str, str]], sp, max_len: int = MAX_LEN):
        self.data = data
        self.sp = sp
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]

        # 共享分词器编码（编码器不加 BOS/EOS）
        src_ids = self.sp.encode(src_text, out_type=int)
        tgt_ids = self.sp.encode(tgt_text, out_type=int)

        # 解码器输入加 BOS/EOS
        tgt_ids = [BOS_ID] + tgt_ids + [EOS_ID]

        # 截断（保留 EOS 在末尾）
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]
        if len(tgt_ids) == self.max_len:
            tgt_ids[-1] = EOS_ID

        # 填充
        src_ids = src_ids + [PAD_ID] * (self.max_len - len(src_ids))
        tgt_ids = tgt_ids + [PAD_ID] * (self.max_len - len(tgt_ids))

        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    """组装训练 batch：src, tgt_input, tgt_label"""
    src_list, tgt_list = zip(*batch)
    src_batch = torch.stack(src_list)
    tgt_batch = torch.stack(tgt_list)

    tgt_input = tgt_batch[:, :-1]   # [B, max_len-1]  去掉末尾
    tgt_label = tgt_batch[:, 1:]    # [B, max_len-1]  去掉 BOS

    return src_batch, tgt_input, tgt_label


# ═══════════════════════════════════════════════════════════════
# 第 5 部分：Transformer 翻译模型（共享 Embedding）
# ═══════════════════════════════════════════════════════════════

class TransformerModel(nn.Module):
    """Transformer Encoder-Decoder，共享源/目标 Embedding + 权重绑定"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_encoder_layers: int = NUM_ENCODER,
        num_decoder_layers: int = NUM_DECODER,
        max_len: int = MAX_LEN,
        pad_idx: int = PAD_ID,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.register_buffer('scale', torch.tensor(d_model ** 0.5))

        # 共享 embedding（编码器和解码器用同一套词向量）
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(max_len, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

        # 权重共享：输出投影也绑定到 embedding（weight tying）
        self.fc_out.weight = self.embedding.weight

    def create_masks(self, src, tgt_input):
        src_pad_mask = (src == self.pad_idx)
        tgt_pad_mask = (tgt_input == self.pad_idx)
        tgt_causal_mask = self.transformer.generate_square_subsequent_mask(
            tgt_input.size(1), device=tgt_input.device
        ).bool()
        return src_pad_mask, tgt_pad_mask, tgt_causal_mask

    def forward(self, src, tgt_input):
        # 共享 embedding + 缩放 + 位置编码
        src_emb = self.embedding(src) * self.scale
        tgt_emb = self.embedding(tgt_input) * self.scale
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)

        src_pad_mask, tgt_pad_mask, tgt_causal_mask = self.create_masks(src, tgt_input)

        output = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_causal_mask,
        )
        return self.fc_out(output)


# ═══════════════════════════════════════════════════════════════
# 第 6 部分：训练
# ═══════════════════════════════════════════════════════════════

def train(sp):
    print('\n[步骤 2/4] 加载数据...')
    train_pairs = load_data(CN_FILE, EN_FILE)
    print(f'  训练句对: {len(train_pairs)}')

    print('\n[步骤 3/4] 构建模型...')
    model = TransformerModel(vocab_size=sp.vocab_size()).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  参数量: {total_params:,}  可训练: {trainable_params:,}')

    # 验证共享 embedding
    is_shared = model.fc_out.weight is model.embedding.weight
    print(f'  Embedding 共享: {is_shared}')

    dataset = TranslationDataset(train_pairs, sp, max_len=MAX_LEN)
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

    total_steps = EPOCHS * len(train_loader)

    # Warmup + 余弦退火学习率调度
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step) / max(1.0, float(WARMUP_STEPS))
        progress = float(step - WARMUP_STEPS) / max(1.0, float(total_steps - WARMUP_STEPS))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f'\n[步骤 4/4] 开始训练 ({EPOCHS} epochs, {total_steps} steps)...')
    best_loss = float('inf')
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}/{EPOCHS}')
        for src_batch, tgt_input, tgt_label in pbar:
            src_batch  = src_batch.to(DEVICE)
            tgt_input  = tgt_input.to(DEVICE)
            tgt_label  = tgt_label.to(DEVICE)

            output = model(src_batch, tgt_input)
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt_label.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        avg_loss = total_loss / len(train_loader)
        lr_now = scheduler.get_last_lr()[0]
        print(f'  Epoch {epoch+1:3d}/{EPOCHS}  avg_loss = {avg_loss:.4f}  lr = {lr_now:.2e}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f'  [OK] 已保存最优模型: {SAVE_PATH}')

    print(f'\n训练完成！最优 loss = {best_loss:.4f}')
    return model


# ═══════════════════════════════════════════════════════════════
# 第 7 部分：推理
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def translate(model, sentence: str, sp, max_len: int = MAX_LEN) -> str:
    """自回归贪婪解码：中文 → 英文"""
    model.eval()

    # 编码源语言
    src_ids = sp.encode(sentence, out_type=int)
    src_ids = src_ids[:max_len]
    src_ids = src_ids + [PAD_ID] * (max_len - len(src_ids))
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(DEVICE)

    # 自回归生成
    tgt_ids = [BOS_ID]
    for _ in range(max_len - 1):  # -1 留空间给 BOS，防止超出位置编码范围
        tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(DEVICE)
        output = model(src_tensor, tgt_tensor)
        next_token = output.argmax(-1)[:, -1].item()
        tgt_ids.append(next_token)
        if next_token == EOS_ID:
            break

    result_ids = [i for i in tgt_ids if i not in (BOS_ID, EOS_ID, PAD_ID)]
    return sp.decode(result_ids)


def test_inference(model, sp):
    print('\n' + '=' * 60)
    print('推理测试')
    print('=' * 60)

    tests = [
        ("目前 粮食 出现 阶段性 过剩",
         "the present food surplus can specifically serve the purpose of helping china"),
        ("中国 人民 银行 应当 将 改版 后 的 人民币 的 发行 时间 予以 公告",
         "the people 's bank of china should make a public announcement on the issuing time"),
        ("勤劳 勇敢 聪明 智慧 的 中国人 一定 会 解决 好 自己 的 事情",
         "the hard-working brave clever and wise chinese people can surely solve their own affairs"),
    ]

    for cn_text, expected in tests:
        en_text = translate(model, cn_text, sp)
        print(f'  中文: {cn_text}')
        print(f'  英文: {en_text}')
        print(f'  参考: {expected}')
        print()


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    sp = ensure_tokenizer()
    model = train(sp)
    test_inference(model, sp)
