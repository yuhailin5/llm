# 训练分词器
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer
from config import VOCAB_SIZE, TRAIN_DATA_PATH
from data_preprocess import load_data, load_all_data
import os

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = BertPreTokenizer() # 使用BERT预分词

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

# 获取全部训练文本
train_text = load_all_data()

tokenizer.train_from_iterator([train_text], trainer=trainer)

tokenizer.save(os.path.join(TRAIN_DATA_PATH, 'bpe_tokenizer.json'))

def test_tokenizer():
    tokenizer = Tokenizer.from_file(os.path.join(TRAIN_DATA_PATH, 'bpe_tokenizer.json'))
    text = "张无忌跟着掀帷而入，圆真却已不知去向。"
    encoding = tokenizer.encode(text)
    print("Tokens:", encoding.tokens)
    print("IDs:", encoding.ids)
    print('VOCAB_SIZE:', tokenizer.get_vocab_size())


test_tokenizer()