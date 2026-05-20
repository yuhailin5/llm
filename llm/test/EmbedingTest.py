import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from MyEmbeding import MyEmbedding, PositionalEncoding
from MyTokenizer import BPETokenizer

tokenizer = BPETokenizer(300)

tokenizer.load('bpe_tokenizer.bin')

embedding = MyEmbedding(tokenizer.vocab_size(), 32)

input_ids, tokens = tokenizer.encode('今天的天气怎么样')

print(f"input_ids = {input_ids}")
print(f"tokens = {tokens}")
print(f"还原文字 = {tokenizer.decode(input_ids)}")
embeddings = embedding(torch.tensor(input_ids).unsqueeze(0))
print(f"embeddings shape = {embeddings.shape}")
print('embeddings =', embeddings)
positional_encoding = PositionalEncoding(max_pos=512, embedding_dim=32)

embeddings = positional_encoding(embeddings)

print(f"embeddings = {embeddings.shape}")
print('embeddings + positional encoding =', embeddings)