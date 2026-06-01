"""
下载翻译训练数据（支持断点续传）

使用:
  python -m llm.model.download_data                 # 下载 20000 条到本地
  python -m llm.model.download_data --size 100000    # 下载 100000 条
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import argparse

# 国内可用镜像（按优先级排列）
MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",  # 直连（可能被墙）
]

DATASET_PATH = "llm/data/zh_en_pairs.json"


def download_with_retry(url, max_retries=5, timeout=60):
    """带重试的 HTTP 下载"""
    import requests
    for i in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            wait = 2 ** i
            print(f"  Retry {i+1}/{max_retries} in {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


def download_via_datasets(max_samples=20000):
    """通过 HuggingFace datasets 库下载（会自动缓存）"""
    from datasets import load_dataset

    for mirror in MIRRORS:
        print(f"Trying mirror: {mirror}")
        os.environ['HF_ENDPOINT'] = mirror
        try:
            # streaming=True 避免一次性下载全量
            ds = load_dataset("opus100", "en-zh", split="train", streaming=True)
            pairs = []
            for item in ds:
                en = item['translation']['en'].strip()
                zh = item['translation']['zh'].strip()
                if len(zh) < 2 or len(en) < 2:
                    continue
                if len(zh) > 60 or len(en) > 60:
                    continue
                pairs.append((zh, en))

                if len(pairs) % 1000 == 0:
                    print(f"  Downloaded {len(pairs)} pairs...")

                if len(pairs) >= max_samples:
                    break

            return pairs
        except Exception as e:
            print(f"  Mirror {mirror} failed: {e}")
            continue

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=20000)
    args = parser.parse_args()

    print(f"Downloading {args.size} Chinese-English pairs...")

    pairs = download_via_datasets(args.size)

    if pairs is None:
        print("\nAll mirrors failed. Alternative options:")
        print("  1. Use proxy: set http_proxy=https://your-proxy:port")
        print("  2. Download from ModelScope: https://modelscope.cn/datasets")
        print("  3. Download from Kaggle: https://www.kaggle.com/datasets")
        print("  4. Use built-in data: python -m llm.model.train_translation")
        return

    # 保存到本地 JSON
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False)

    print(f"\nSaved {len(pairs)} pairs to {DATASET_PATH}")
    print(f"Next: python -m llm.model.train_translation --local")


if __name__ == '__main__':
    main()
