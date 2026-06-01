import os
print(os.getcwd())

from MyTokenizer import BPETokenizer
cn_bpe_train = []
en_bpe_train = []

# 构建BPE语料
def build_corpus(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf-8') as f_src, open (tgt_file, 'r', encoding='utf-8') as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            if src_line and tgt_line:
                cn_bpe_train.append(src_line)
                en_bpe_train.append(tgt_line)
build_corpus('llm/data/transformer_train/cn.txt', 'llm/data/transformer_train/en.txt')

# 训练BPE分词器
cn_tokenizer = BPETokenizer(max_vocab_size=3000)
en_tokenizer = BPETokenizer(max_vocab_size=3000)

cn_tokenizer.fit(cn_bpe_train)
en_tokenizer.fit(en_bpe_train)
cn_tokenizer.save('llm/bin/cn_transformer_bpe_tokenizer.bin')
en_tokenizer.save('llm/bin/en_transformer_bpe_tokenizer.bin')

print("中文BPE词表大小:", cn_tokenizer.vocab_size)
print("英文BPE词表大小:", en_tokenizer.vocab_size)