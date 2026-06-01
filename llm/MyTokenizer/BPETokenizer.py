"""
BPE (Byte Pair Encoding) Tokenizer
==================================
UTF-8 字节级 BPE 分词器，从单字节开始迭代合并高频字节对。

ID 分配规范:
  - 特殊 token（<pad>, <bos>, <eos>, <unk>）→ ID 0~3，__init__ 时预留
  - 单字节 token（0x00~0xFF）        → ID 4~259，train 时初始化
  - 合并 token（多字节组合）          → ID 260+，BPE 训练时逐步生成

使用:
    tok = BPETokenizer(max_vocab_size=4000)
    tok.fit(texts)                        # 训练
    ids = tok.encode("hello world")       # 编码 → [4, 153, 112, ...]
    text = tok.decode(ids)                # 解码 → "hello world"
    tok.save("tokenizer.pkl")            # 保存
    tok.load("tokenizer.pkl")            # 加载
"""

import pickle
from collections import OrderedDict, defaultdict
from typing import List, Optional, Tuple

from tqdm import tqdm


class BPETokenizer:
    """UTF-8 字节级 BPE 分词器。

    Parameters
    ----------
    max_vocab_size : int
        最大词表大小（含特殊 token 和 256 个单字节 token）。
        例如 max_vocab_size=4000 时，最多产生 4000-4-256=3740 次合并。
    special_tokens : list[str] | None
        特殊 token 列表，默认 ['<pad>', '<bos>', '<eos>', '<unk>']。
        按顺序分配 ID 0, 1, 2, 3...
    verbose : bool
        训练时是否打印合并过程。
    """

    # ── 类常量 ──────────────────────────────────────────────
    DEFAULT_SPECIAL_TOKENS = ['<pad>', '<bos>', '<eos>', '<unk>']
    NUM_BYTE_TOKENS = 256

    def __init__(
        self,
        max_vocab_size: int = 5000,
        special_tokens: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        if max_vocab_size < self.NUM_BYTE_TOKENS + len(self.DEFAULT_SPECIAL_TOKENS):
            raise ValueError(
                f"max_vocab_size 至少为 {self.NUM_BYTE_TOKENS + len(self.DEFAULT_SPECIAL_TOKENS)}"
            )

        self.max_vocab_size = max_vocab_size
        self.verbose = verbose

        # ── 特殊 token（ID: 0 ~ N-1，__init__ 时固定） ──
        special_tokens = special_tokens or self.DEFAULT_SPECIAL_TOKENS
        self.special_tokens = list(special_tokens)
        self.s2i: OrderedDict[str, int] = OrderedDict()
        self.i2s: OrderedDict[int, str] = OrderedDict()
        for i, tok in enumerate(self.special_tokens):
            self.s2i[tok] = i
            self.i2s[i] = tok

        # 快捷访问
        self.pad_id = self.s2i.get('<pad>', 0)
        self.bos_id = self.s2i.get('<bos>', 1)
        self.eos_id = self.s2i.get('<eos>', 2)
        self.unk_id = self.s2i.get('<unk>', 3)

        # ── BPE token（ID: base_id ~ max_vocab_size-1） ──
        self._base_id = len(self.special_tokens)  # BPE token 起始 ID
        self.next_id = self._base_id

        self.b2i: OrderedDict[bytes, int] = OrderedDict()  # bytes → ID
        self.i2b: OrderedDict[int, bytes] = OrderedDict()  # ID → bytes
        self.merges: List[Tuple[bytes, bytes]] = []        # 合并历史（有序）

        self._trained = False

    # ══════════════════════════════════════════════════════════
    # 公开 API
    # ══════════════════════════════════════════════════════════

    def fit(self, texts: List[str]) -> "BPETokenizer":
        """训练 BPE 分词器（兼容 sklearn 风格接口名）。"""
        return self.train(texts)

    def train(self, texts: List[str]) -> "BPETokenizer":
        """在语料上训练 BPE 合并规则。

        Parameters
        ----------
        texts : list[str]
            训练语料文本列表。

        Returns
        -------
        self : BPETokenizer
        """
        # 1. 初始化单字节 token
        for i in range(self.NUM_BYTE_TOKENS):
            bid = self._base_id + i
            self.b2i[bytes([i])] = bid
            self.i2b[bid] = bytes([i])
        self.next_id = self._base_id + self.NUM_BYTE_TOKENS

        # 2. 文本 → 字节序列
        tokens_list = self._texts_to_byte_sequences(texts)

        # 3. 迭代合并（带进度条）
        max_merges = self.max_vocab_size - self.next_id
        pbar = tqdm(total=max_merges, desc="BPE training", unit="merge",
                     ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')

        merge_count = 0
        while self.next_id < self.max_vocab_size:
            pair = self._most_frequent_pair(tokens_list)
            if pair is None:
                pbar.close()
                break

            new_token = self._register_merge(pair)
            self.merges.append(pair)
            tokens_list = self._apply_merge(tokens_list, pair[0], pair[1], new_token)
            merge_count += 1
            pbar.update(1)

            if self.verbose and merge_count % 100 == 0:
                self._log_merge(pair, new_token, merge_count)

        if not pbar.disable:
            pbar.close()

        if self.verbose:
            print(f"[OK] BPE 训练完成: {merge_count} 次合并, "
                  f"词表大小 {self.vocab_size}")

        self._trained = True
        return self

    def encode(self, text: str) -> List[int]:
        """将文本编码为 token ID 列表。

        Parameters
        ----------
        text : str
            输入文本。

        Returns
        -------
        List[int]
            token ID 序列。
        """
        if not self._trained:
            raise RuntimeError("请先调用 fit() / train() 训练分词器")

        # 文本 → 单字节序列
        tokens = [bytes([b]) for b in text.encode('utf-8')]

        # 按训练时的顺序依次应用合并规则
        for a, b in self.merges:
            merged = a + b
            if merged not in self.b2i:
                continue
            tokens = self._apply_merge_single(tokens, a, b, merged)

        # bytes → ID（未登录的字节/合并 token 回退到 unk_id）
        return [self.b2i.get(tok, self.unk_id) for tok in tokens]

    def decode(self, ids: List[int]) -> str:
        """将 token ID 列表解码回文本。

        Parameters
        ----------
        ids : List[int]
            token ID 序列。

        Returns
        -------
        str
            解码后的文本。
        """
        byte_list: List[bytes] = []
        for tid in ids:
            if tid in self.i2s:
                continue  # 跳过特殊 token
            if tid in self.i2b:
                byte_list.append(self.i2b[tid])
            # 未知 ID 静默跳过

        return b''.join(byte_list).decode('utf-8', errors='replace')

    # ══════════════════════════════════════════════════════════
    # 属性
    # ══════════════════════════════════════════════════════════

    @property
    def vocab_size(self) -> int:
        """当前词表大小（含特殊 token）。"""
        return self.next_id

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ══════════════════════════════════════════════════════════
    # 序列化
    # ══════════════════════════════════════════════════════════

    def save(self, file_path: str) -> None:
        """保存分词器到文件（pickle 格式）。"""
        state = {
            'max_vocab_size': self.max_vocab_size,
            'special_tokens': self.special_tokens,
            's2i': self.s2i,
            'i2s': self.i2s,
            'pad_id': self.pad_id,
            'bos_id': self.bos_id,
            'eos_id': self.eos_id,
            'unk_id': self.unk_id,
            '_base_id': self._base_id,
            'next_id': self.next_id,
            'b2i': self.b2i,
            'i2b': self.i2b,
            'merges': self.merges,
            '_trained': self._trained,
        }
        with open(file_path, 'wb') as fp:
            pickle.dump(state, fp)

    def load(self, file_path: str) -> None:
        """从文件加载分词器。"""
        with open(file_path, 'rb') as fp:
            state = pickle.load(fp)
        self.max_vocab_size = state['max_vocab_size']
        self.special_tokens = state['special_tokens']
        self.s2i = state['s2i']
        self.i2s = state['i2s']
        self.pad_id = state['pad_id']
        self.bos_id = state['bos_id']
        self.eos_id = state['eos_id']
        self.unk_id = state.get('unk_id', 3)
        self._base_id = state['_base_id']
        self.next_id = state['next_id']
        self.b2i = state['b2i']
        self.i2b = state['i2b']
        self.merges = state['merges']
        self._trained = state.get('_trained', True)

    # ══════════════════════════════════════════════════════════
    # 内部方法
    # ══════════════════════════════════════════════════════════

    def _texts_to_byte_sequences(self, texts):
        return [[bytes([b]) for b in text.encode('utf-8')] for text in texts]

    def _most_frequent_pair(self, tokens_list):
        pair_freq = defaultdict(int)
        for seq in tokens_list:
            for i in range(len(seq) - 1):
                pair_freq[(seq[i], seq[i + 1])] += 1
        if not pair_freq:
            return None
        return max(pair_freq, key=pair_freq.get)

    def _register_merge(self, pair):
        a, b = pair
        new_token = a + b
        self.b2i[new_token] = self.next_id
        self.i2b[self.next_id] = new_token
        self.next_id += 1
        return new_token

    @staticmethod
    def _apply_merge(tokens_list, a, b, new_token):
        """对整个 tokens_list 执行一次合并（训练时用）。"""
        result = []
        for seq in tokens_list:
            result.append(BPETokenizer._apply_merge_single(seq, a, b, new_token))
        return result

    @staticmethod
    def _apply_merge_single(tokens, a, b, new_token):
        """对单条 token 序列执行一次合并（编码时用）。"""
        new_seq = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_seq.append(new_token)
                i += 2
            else:
                new_seq.append(tokens[i])
                i += 1
        return new_seq

    @staticmethod
    def _log_merge(pair, new_token, merge_count):
        a, b = pair
        try:
            a_str = a.decode('utf-8', errors='replace')
            b_str = b.decode('utf-8', errors='replace')
            new_str = new_token.decode('utf-8', errors='replace')
        except Exception:
            a_str = str(a)
            b_str = str(b)
            new_str = str(new_token)
        print(f"[merge #{merge_count}] {a_str!r} + {b_str!r} → {new_str!r}")


# ===========================================================================
# 便捷函数
# ===========================================================================

def build_vocab(texts, max_vocab_size=4000, special_tokens=None):
    """快速从文本列表构建并训练一个 BPE 分词器。"""
    tok = BPETokenizer(max_vocab_size=max_vocab_size,
                       special_tokens=special_tokens)
    tok.train(texts)
    return tok
