import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, '..', '..', 'data', 'transformer_train')
TRAIN_DATA_PATH = os.path.normpath(TRAIN_DATA_PATH)

VOCAB_SIZE = 8000

MAX_LEN = 48

PAD_ID = 0
SOS_ID = 2
EOS_ID = 3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'