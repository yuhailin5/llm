# 测试脚本 — 验证当前模型的生成能力
import torch
import sys
import os

# 确保项目目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import (
    NovelGPT, load_model, generate, ensure_tokenizer,
    DEVICE, BLOCK_SIZE, TOKENIZER_PATH, DATA_DIR, TRAIN_FILES,
)
from tokenizers import Tokenizer


def test_perplexity(model, tokenizer, n_samples=50):
    """在随机样本上计算困惑度"""
    print("\n" + "=" * 60)
    print("  困惑度 (Perplexity) 评估")
    print("=" * 60)

    # 加载一页文本做测试
    from run import load_all_data
    text = load_all_data()
    ids = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            start = torch.randint(0, len(ids) - BLOCK_SIZE - 1, (1,)).item()
            x = ids[start : start + BLOCK_SIZE].unsqueeze(0).to(DEVICE)
            y = ids[start + 1 : start + BLOCK_SIZE + 1].unsqueeze(0).to(DEVICE)

            _, loss = model(x, y)
            total_loss += loss.item() * BLOCK_SIZE
            total_tokens += BLOCK_SIZE

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"  测试样本数: {n_samples} × {BLOCK_SIZE} tokens")
    print(f"  平均 Loss:  {avg_loss:.4f}")
    print(f"  困惑度:     {ppl:.2f}")
    return ppl


def test_generation(model, tokenizer, prompts, strategies):
    """用多种策略测试生成"""
    print("\n" + "=" * 60)
    print("  文本生成测试")
    print("=" * 60)

    for i, prompt in enumerate(prompts):
        print(f"\n{'─' * 50}")
        print(f"  Prompt [{i+1}]: {prompt}")

        for name, kwargs in strategies.items():
            result = generate(
                model, prompt, max_new_tokens=80, **kwargs
            )
            # 截取生成部分（去掉 prompt）
            print(f"\n  [{name}]")
            print(f"  {result}")


def test_token_distribution(model, tokenizer, prompt, n_tokens=20):
    """查看 top token 的概率分布 — 判断模型是否过于均匀 或 过于尖锐"""
    print("\n" + "=" * 60)
    print("  Top-10 Token 概率分布")
    print("=" * 60)

    ids = tokenizer.encode(prompt).ids
    ids = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits, _ = model(ids)
        probs = torch.softmax(logits[:, -1, :], dim=-1)  # 最后一个位置的分布

    top_probs, top_ids = torch.topk(probs[0], k=10)

    print(f"\n  Prompt: {prompt}")
    print(f"  {'Token ID':<12} {'Token':<20} {'Prob':>8}")
    print(f"  {'─'*12} {'─'*20} {'─'*8}")
    for tid, prob in zip(top_ids.tolist(), top_probs.tolist()):
        token_str = tokenizer.decode([tid])
        # 截断过长的 token
        if len(token_str) > 18:
            token_str = token_str[:15] + "..."
        print(f"  {tid:<12} {token_str:<20} {prob:>8.4f}")


def main():
    print("=" * 60)
    print("  NovelGPT 模型生成能力测试")
    print("=" * 60)

    # 1. 加载模型
    model = load_model()
    ensure_tokenizer()
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"  词表大小: {tokenizer.get_vocab_size()}")

    # 2. 困惑度
    test_perplexity(model, tokenizer, n_samples=30)

    # 3. 多种策略生成测试
    prompts = [
        "张无忌踏上光明顶，只见",
        "令狐冲拔出长剑，大喝一声",
        "郭靖左手画了个圆圈，右手",
    ]

    strategies = {
        "greedy (t=0)":       {"temperature": 0},
        "t=0.8 top-k=50":     {"temperature": 0.8, "top_k": 50},
        "t=0.8 top-p=0.9":    {"temperature": 0.8, "top_p": 0.9},
        "t=1.0 top-k=100":    {"temperature": 1.0, "top_k": 100},
    }

    test_generation(model, tokenizer, prompts, strategies)

    # 4. Token 分布
    test_token_distribution(model, tokenizer, "张无忌跟着掀帷而入，圆真却已不知去向。")

    print("\n" + "=" * 60)
    print("  测试结束")
    print("=" * 60)


if __name__ == "__main__":
    main()
