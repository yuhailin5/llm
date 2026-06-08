"""
训练中英翻译的 SentencePiece BPE 分词器

选择 SentencePiece BPE 的理由：
1. 子词粒度：中文已做词级分词，但词表仍会很大（~2万+ 不同词）；
   BPE 将词进一步拆成子字/子词，用 ~8000 个 subword 覆盖几乎所有文本，
   同时通过 subword 组合能处理训练时未见过的词（UNK 问题）。
2. 共享词表：中英数据混合训练一个词表，编码器和解码器可以共享 Embedding，
   减少参数量，且跨语言 subword 共享能帮助迁移学习。
3. 语言无关：SentencePiece 把输入当纯 Unicode 序列处理，
   不需要特定语言的分词规则（中文不用 jieba，英文不用 Moses）。
4. 可逆性：SentencePiece 的 tokenization 是完全可逆的（detokenization），
   空格用元字符 ▁ 编码，不会丢信息。
"""

import sentencepiece as spm
import os

from config import TRAIN_DATA_PATH, VOCAB_SIZE

cn_corpus = os.path.join(TRAIN_DATA_PATH, 'cn.txt')
en_corpus = os.path.join(TRAIN_DATA_PATH, 'en.txt')

def train_tokenizer(corpus_file, vocab_size=8000, model_prefix='spm_bpe'):
    """
    训练 BPE 分词器。

    参数选择理由：
    - vocab_size=8000：对于 6800 句对的规模，8000 足够覆盖常见的
      中文字字和英文 subword。更大的词表（16000/32000）会引入过多
      低频 token，浪费嵌入参数。
    - model_type='bpe'：BPE 是 NMT 最通用的子词算法，
      unigram 也行但 BPE 对中文更友好（中文偏 character-level merge）。
    - character_coverage=0.9995：覆盖 99.95% 的字符，
      剩余极少数字符回退到 UNK。中日韩文本设这个值比较合适。
    - max_sentence_length=4192：远大于最长句，不截断。
    - num_threads=4：加速训练。
    """
    model_path = os.path.join(TRAIN_DATA_PATH, model_prefix)

    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_path,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        max_sentence_length=4192,
        num_threads=4,
        pad_id=0,            # <pad>
        unk_id=1,            # <unk>
        bos_id=2,            # <s>
        eos_id=3,            # </s>
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        # 让 SentencePiece 按空格分割预处理（中文已有空格分词，英文也有）
        # 不加 --split-by-whitespace 让 BPE 自己学空格边界
    )
    print(f'分词器已保存为: {model_path}.model')
    print(f'词表大小: {vocab_size}')
    return model_path


def cn_test_tokenizer(model_path):
    """测试分词器的编解码效果"""
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_path}.model')

    test_sentences = [
        # 中文测试
        "目前粮食出现阶段性过剩",
        "中国人民应当将改版后的人民币的发行时间予以公告",
        "勤劳勇敢聪明的中国人一定会解决好祖国统一的事情",
    ]

    for text in test_sentences:
        ids = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)
        pieces = sp.encode(text, out_type=str)
        print(f'原文: {text}')
        print(f'子词: {" ".join(pieces)}')
        print(f'ID:   {ids}')
        print(f'解码: {decoded}')
        print(f'长度: {len(ids)} tokens')
        print('-' * 60)

def en_test_tokenizer(model_path):
    """测试分词器的编解码效果"""
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_path}.model')
    print('vocab size:', sp.get_piece_size(),sp.vocab_size())
    test_sentences = [
        # 英文测试
        "The quick brown fox jumps over the lazy dog.",
        "I love natural language processing and machine translation.",
        "SentencePiece is a great tool for subword tokenization.",
    ]

    for text in test_sentences:
        ids = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)
        pieces = sp.encode(text, out_type=str)
        print(f'原文: {text}')
        print(f'子词: {" ".join(pieces)}')
        print(f'ID:   {ids}')
        print(f'解码: {decoded}')
        print(f'长度: {len(ids)} tokens')
        print('-' * 60)


if __name__ == '__main__':
    # 2. 训练分词器
    cn_model_path = train_tokenizer(cn_corpus, vocab_size=VOCAB_SIZE,model_prefix='cn_spm_bpe')
    en_model_path = train_tokenizer(en_corpus, vocab_size=VOCAB_SIZE,model_prefix='en_spm_bpe')
    # 3. 测试效果
    cn_test_tokenizer(cn_model_path)
    en_test_tokenizer(en_model_path)

