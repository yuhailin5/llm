import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import os
from config import BLOCK_SIZE, BATCH_SIZE, TRAIN_DATA_PATH
from data_preprocess import load_all_data

tokenizer = Tokenizer.from_file(os.path.join(TRAIN_DATA_PATH, 'bpe_tokenizer.json'))


train_text = load_all_data()
encoded_tokens = tokenizer.encode(train_text).ids

encoded_tokens = torch.tensor(encoded_tokens, dtype=torch.long)

class JinYongDataset(Dataset):
    def __len__(self):
        # 避免越界
        return len(encoded_tokens) - BLOCK_SIZE

    def __getitem__(self, idx):
        # 起始位置idx+BLOCK_SIZE
        input_seq = encoded_tokens[idx : idx + BLOCK_SIZE]
        # 输入右移1位
        target_seq = encoded_tokens[idx + 1 : idx + BLOCK_SIZE + 1]
        return input_seq, target_seq

dataset = JinYongDataset()
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    drop_last=True
)