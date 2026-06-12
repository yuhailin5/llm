#!/usr/bin/env python3
"""
NovelGPT — 金庸武侠小说小型 GPT（Decoder-Only）模型
======================================================
自包含脚本，整合：配置 / 数据加载 / 分词器训练 / 模型定义 / 训练 / 推理

用法:
    python run.py train              # 训练模型
    python run.py generate -p "..."  # 单次续写
    python run.py interactive        # 交互式续写

依赖: torch, tokenizers (HuggingFace)
"""

import os
import argparse
import math
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer

# ──────────────────────────────────────────────
# 全局配置
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "bin"))
TOKENIZER_PATH = os.path.join(DATA_DIR, "bpe_tokenizer.json")

TRAIN_FILES = [
    "金庸-天龙八部.txt",
    "金庸-侠客行.txt",
    "金庸-笑傲江湖.txt",
    "金庸-倚天屠龙记.txt",
    "金庸-越女剑.txt",
]

# 模型
VOCAB_SIZE = 8000
BLOCK_SIZE = 512
BATCH_SIZE = 32
D_MODEL = 384
N_HEAD = 12
NUM_LAYERS = 8
DROPOUT = 0.1

# 训练
LR = 6e-4
EPOCHS = 10
WARMUP_STEPS = 200
GRAD_CLIP = 1.0

# BPE 特殊 token — [PAD] 排在第二个，ID=1
PAD_ID = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────
def load_all_data():
    """加载全部小说，用换行符拼接"""
    texts = []
    for fname in TRAIN_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return "\n".join(texts)


# ──────────────────────────────────────────────
# 分词器
# ──────────────────────────────────────────────
def ensure_tokenizer():
    """分词器不存在则自动训练"""
    if os.path.exists(TOKENIZER_PATH):
        print(f"[分词器] 已存在: {TOKENIZER_PATH}")
        return
    print("[分词器] 开始训练 BPE ...")
    text = load_all_data()
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = BertPreTokenizer()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    tokenizer.save(TOKENIZER_PATH)
    print(f"[分词器] 已保存: {TOKENIZER_PATH}")


# ──────────────────────────────────────────────
# 数据集 & DataLoader
# ──────────────────────────────────────────────
def build_dataloader():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    text = load_all_data()
    ids = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)

    class GPTDataset(Dataset):
        def __len__(self):
            return len(ids) - BLOCK_SIZE

        def __getitem__(self, idx):
            return ids[idx : idx + BLOCK_SIZE], ids[idx + 1 : idx + BLOCK_SIZE + 1]

    ds = GPTDataset()
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )


# ──────────────────────────────────────────────
# GPT 模型（Decoder-Only）— 自定义 TransformerBlock
# ──────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block, 使用 F.scaled_dot_product_attention 触发 Flash Attention"""

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # QKV 合并投影：一次矩阵乘法出 Q/K/V，比三个独立 Linear 快
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=False),
            nn.Dropout(dropout),
        )
        self.drop_attn = nn.Dropout(dropout)
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model

    def forward(self, x):
        # --- Self-Attention (Pre-LN) ---
        residual = x
        x_norm = self.ln1(x)
        B, T, C = x_norm.shape

        qkv = self.qkv(x_norm).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # 各 (B, T, n_head, head_dim)
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # is_causal=True → 自动走 Flash Attention / Memory-Efficient Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        attn_out = self.out_proj(attn_out)

        x = residual + self.drop_attn(attn_out)

        # --- Feed-Forward (Pre-LN) ---
        x = x + self.ffn(self.ln2(x))
        return x


class NovelGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)

        self.layers = nn.ModuleList([
            TransformerBlock(D_MODEL, N_HEAD, DROPOUT)
            for _ in range(NUM_LAYERS)
        ])

        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        # Weight tying: 输出投影与词嵌入共享权重
        self.head.weight = self.tok_emb.weight

        # 预计算位置索引，避免每个 forward 重复分配
        pos_idx = torch.arange(BLOCK_SIZE, dtype=torch.long)
        self.register_buffer("pos_idx", pos_idx, persistent=False)

    def forward(self, x, targets=None):
        _, T = x.shape

        # Embedding + 可学习位置编码
        x = self.tok_emb(x) + self.pos_emb(self.pos_idx[:T])
        x = self.drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=PAD_ID,
            )
        return logits, loss


# ──────────────────────────────────────────────
# Warmup + Cosine 学习率调度
# ──────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_count = 0

        # 预热阶段线性增长速率
        self.warmup_slope = self.base_lrs[0] / max(1, warmup_steps)

    def step(self):
        self.step_count += 1
        lr = self._lr()
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = lr * base / self.base_lrs[0]

    def _lr(self):
        if self.step_count <= self.warmup_steps:
            return self.warmup_slope * self.step_count
        # Cosine annealing
        progress = (self.step_count - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )


# ──────────────────────────────────────────────
# 训练
# ──────────────────────────────────────────────
def train():
    # 加速设置
    torch.backends.cudnn.benchmark = True

    ensure_tokenizer()
    dataloader = build_dataloader()

    model = NovelGPT().to(DEVICE)

    # torch.compile 加速（PyTorch >= 2.0）
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[模型] torch.compile 已启用 (reduce-overhead)")
        except Exception as e:
            print(f"[模型] torch.compile 不可用: {e}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 参数量: {n_params:,}")

    # Fused AdamW（PyTorch >= 2.0），回退普通 Adam
    try:
        optimizer = optim.AdamW(model.parameters(), lr=LR, fused=True)
        print("[优化器] Fused AdamW")
    except (RuntimeError, TypeError):
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        print("[优化器] AdamW (未融合)")

    total_steps = EPOCHS * len(dataloader)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_STEPS, total_steps)

    # AMP 混合精度
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[训练] AMP 混合精度: {'启用' if use_amp else '关闭'}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_path = os.path.join(MODEL_DIR, "best_novel_gpt.pth")
    best_loss = float("inf")

    steps_per_epoch = len(dataloader)
    print(f"[训练] 设备: {DEVICE} | Epoch: {EPOCHS} | Steps/Epoch: {steps_per_epoch}")
    print(f"[训练] LR: {LR} | Warmup: {WARMUP_STEPS} | Total Steps: {total_steps}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                steps_done = batch_idx + 1
                steps_left = steps_per_epoch - steps_done
                eta = elapsed / steps_done * steps_left if steps_done > 0 else 0
                print(
                    f"  Epoch [{epoch+1:2d}/{EPOCHS}] "
                    f"Step [{steps_done:4d}/{steps_per_epoch}] "
                    f"Loss: {loss.item():.4f} | "
                    f"{steps_done / elapsed:.1f} step/s | "
                    f"ETA: {eta:.0f}s"
                )

        avg_loss = epoch_loss / steps_per_epoch
        elapsed = time.time() - t0
        print(f"==> Epoch [{epoch+1}/{EPOCHS}] 平均 Loss: {avg_loss:.4f} | 耗时: {elapsed:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "block_size": BLOCK_SIZE,
                },
                best_path,
            )
            print(f"    已保存最优模型 → {best_path}  (Loss: {best_loss:.4f})\n")

    print("[训练] 全部完成!")


# ──────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────
def load_model(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(MODEL_DIR, "best_novel_gpt.pth")

    model = NovelGPT().to(DEVICE)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"[模型] 已加载: epoch={ckpt.get('epoch', '?')}, "
            f"loss={ckpt.get('loss', 0):.4f}"
        )
    else:
        print(f"[模型] 警告: 未找到 {checkpoint_path}，使用随机权重")
    model.eval()
    return model


# ──────────────────────────────────────────────
# 文本生成
# ──────────────────────────────────────────────
@torch.no_grad()
def generate(model, prompt, max_new_tokens=200, temperature=0.8, top_k=50, top_p=None):
    """
    自回归续写。

    参数:
        model:           NovelGPT
        prompt:          起始文本
        max_new_tokens:  最大生成 token 数
        temperature:     温度系数 (<1 保守, >1 随机)
        top_k:           Top-K 采样
        top_p:           Top-P (nucleus) 采样
    """
    ensure_tokenizer()
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("prompt 编码为空，请更换起始文本")

    ids = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(max_new_tokens):
        # 截断到上下文窗口
        x = ids[:, -BLOCK_SIZE:]
        logits, _ = model(x)
        logits = logits[:, -1, :]  # 只取最后一个位置

        # Greedy（temperature=0）：直接 argmax
        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=-1)
            continue

        # 温度缩放
        if temperature != 1.0:
            logits = logits / temperature

        # Top-K 过滤
        if top_k and top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
            mask = torch.full_like(logits, float("-inf"))
            logits = mask.scatter(-1, topk_idx, topk_vals)

        # Top-P (nucleus) 过滤
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            cutoff = (cum_probs > top_p).float().argmax(dim=-1).item() + 1
            sorted_logits[:, cutoff:] = float("-inf")
            logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_token], dim=-1)

    return tokenizer.decode(ids[0].tolist())


# ──────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="NovelGPT — 金庸武侠小说 GPT 模型"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    sub.add_parser("train", help="训练模型")

    # generate（单次）
    p_gen = sub.add_parser("generate", help="单次续写")
    p_gen.add_argument("--prompt", "-p", type=str, required=True, help="起始文本")
    p_gen.add_argument("--max-tokens", "-n", type=int, default=200, help="最大生成 token 数")
    p_gen.add_argument("--temperature", "-t", type=float, default=0.8, help="温度系数")
    p_gen.add_argument("--top-k", "-k", type=int, default=50, help="Top-K 采样")
    p_gen.add_argument("--top-p", type=float, default=None, help="Top-P 阈值")
    p_gen.add_argument("--checkpoint", "-c", type=str, default=None, help="模型路径")

    # interactive
    sub.add_parser("interactive", aliases=["i"], help="交互式续写")

    args = parser.parse_args()

    if args.cmd == "train":
        train()

    elif args.cmd == "generate":
        model = load_model(args.checkpoint)
        result = generate(
            model,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(result)

    elif args.cmd == "interactive":
        model = load_model()
        print("\n" + "=" * 50)
        print("  NovelGPT — 金庸武侠小说续写")
        print("  输入 quit 退出")
        print("=" * 50 + "\n")
        while True:
            p = input(">>> ").strip()
            if p.lower() == "quit":
                print("再见！")
                break
            if not p:
                continue
            print(f"\n{generate(model, p)}\n")


if __name__ == "__main__":
    main()