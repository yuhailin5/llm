# 推理接口
import torch
import argparse
import os
from tokenizers import Tokenizer

from config import (
    BLOCK_SIZE, TRAIN_DATA_PATH, MODEL_SAVE_PATH, DEVICE
)
from model import NovelGPT


def load_model(checkpoint_path: str = None) -> NovelGPT:
    """加载训练好的模型"""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(MODEL_SAVE_PATH, "best_novel_gpt.pth")

    model = NovelGPT().to(DEVICE)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"已加载模型: {checkpoint_path}")
        print(f"  Epoch: {ckpt.get('epoch', '?')}, Loss: {ckpt.get('loss', '?'):.4f}")
    else:
        print(f"警告: 未找到模型文件 {checkpoint_path}，使用随机初始化权重")
    model.eval()
    return model


def load_tokenizer(tokenizer_path: str = None) -> Tokenizer:
    """加载 BPE 分词器"""
    if tokenizer_path is None:
        tokenizer_path = os.path.join(TRAIN_DATA_PATH, "bpe_tokenizer.json")
    return Tokenizer.from_file(tokenizer_path)


@torch.no_grad()
def generate(
    model: NovelGPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
) -> str:
    """
    自回归文本生成

    参数:
        model:       NovelGPT 模型
        tokenizer:   BPE 分词器
        prompt:      起始文本
        max_new_tokens: 最多生成的 token 数
        temperature: 温度系数（<1 更保守，>1 更随机）
        top_k:       Top-K 采样，仅从概率最高的 k 个 token 中采样
        top_p:       Top-P（nucleus）采样，累积概率阈值

    返回:
        生成的完整文本（prompt + 续写）
    """
    model.eval()

    # 编码 prompt
    ids = tokenizer.encode(prompt).ids
    if len(ids) == 0:
        raise ValueError("prompt 编码后为空，请更换起始文本")

    ids = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # [1, T_prompt]

    # 自回归生成
    for _ in range(max_new_tokens):
        # 截断到上下文窗口
        x = ids[:, -BLOCK_SIZE:]  # [1, T]

        logits, _ = model(x)
        # 取最后一个位置的 logits
        next_logits = logits[:, -1, :]  # [1, V]

        # --- 采样策略 ---
        # 1. 温度缩放
        if temperature > 0 and temperature != 1.0:
            next_logits = next_logits / temperature

        # 2. Top-K 过滤
        if top_k is not None and top_k > 0:
            topk_vals, topk_idx = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
            mask = torch.full_like(next_logits, float("-inf"))
            mask.scatter_(-1, topk_idx, topk_vals)
            next_logits = mask

        # 3. Top-P (nucleus) 过滤
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # 保留累积概率 <= top_p 的 token，再加一个
            cutoff = (cum_probs > top_p).float().argmax(dim=-1).item() + 1
            sorted_logits[:, cutoff:] = float("-inf")
            # 恢复到原始顺序
            next_logits = torch.zeros_like(next_logits).scatter_(
                -1, sorted_idx, sorted_logits
            )

        # 4. 采样
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

        ids = torch.cat([ids, next_token], dim=-1)

    # 解码
    return tokenizer.decode(ids[0].tolist())


def interactive():
    """交互式续写模式"""
    model = load_model()
    tokenizer = load_tokenizer()

    print("\n" + "=" * 50)
    print("  NovelGPT - 金庸武侠小说续写")
    print("  输入 'quit' 退出, 'reset' 清空上下文")
    print("=" * 50 + "\n")

    while True:
        prompt = input(">>> ").strip()
        if not prompt:
            continue
        if prompt.lower() == "quit":
            print("再见！")
            break
        if prompt.lower() == "reset":
            print("[上下文已清空]\n")
            continue

        result = generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=50)
        print(f"\n{result}\n")


def main():
    parser = argparse.ArgumentParser(description="NovelGPT 推理接口")
    parser.add_argument("--prompt", "-p", type=str, default=None,
                        help="起始文本（不指定则进入交互模式）")
    parser.add_argument("--max-tokens", "-n", type=int, default=200,
                        help="最多生成 token 数（默认 200）")
    parser.add_argument("--temperature", "-t", type=float, default=0.8,
                        help="温度系数（默认 0.8）")
    parser.add_argument("--top-k", "-k", type=int, default=50,
                        help="Top-K 采样（默认 50）")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Top-P/nucleus 采样阈值（与 top-k 互斥时优先 top-k）")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="模型权重路径")
    parser.add_argument("--greedy", "-g", action="store_true",
                        help="贪心解码（相当于 temperature=0）")

    args = parser.parse_args()

    model = load_model(args.checkpoint)
    tokenizer = load_tokenizer()

    if args.greedy:
        args.temperature = 0
        args.top_k = None
        args.top_p = None

    if args.prompt:
        result = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(result)
    else:
        interactive()


if __name__ == "__main__":
    main()
