# 测试两种Tokenizer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from MyTokenizer import BPETokenizer, WordPieceTokenizer
bpe_tokenizer = BPETokenizer(max_vocab_size=300)
wordpiece_tokenizer = WordPieceTokenizer(max_vocab_size=5000)

def train(texts):
    """
    params:
        texts:对两种算法初始化训练
    """
    bpe_tokenizer.train(texts)
    bpe_tokenizer.add_special_tokens((['<|im_start|>','<|im_end|>','<|endoftext|>','<|padding|>']))
    print('BPE_vocab size:',bpe_tokenizer.vocab_size())
    #bpe_tokenizer.save('bpe_tokenizer.bin')
    #bpe_tokenizer.load('bpe_tokenizer.bin')
    wordpiece_tokenizer.train(texts)
    wordpiece_tokenizer.save('word_piece_tokenizer.bin')
    print('wordPiece_vocab size:',wordpiece_tokenizer.vocab_size())
    


def test_BPE(text):
    ids,tokens = bpe_tokenizer.encode(text)
    print(f"ids = {ids}")
    print(f"tokens = {tokens}")

    print(f"还原文字 = {bpe_tokenizer.decode(ids)}")

def test_wordpiece(text):
    ids,tokens = wordpiece_tokenizer.encode(text)
    print(f"ids = {ids}")
    print(f"tokens = {tokens}")

    print(f"还原文字 = {wordpiece_tokenizer.decode(ids)}")


if __name__ == '__main__':
    cn = open('llm/MyTokenizer/dataset/train-cn.txt','r',encoding='utf-8').read()
    en = open('llm/MyTokenizer/dataset/train-en.txt','r',encoding='utf-8').read()
    print('hh')
    train([cn,en])
    text = 'system\nyou are a helper assistant\n\nuser\n今天的天气怎么样\nassistant\n'
    test_BPE(text)
    test_wordpiece(text)