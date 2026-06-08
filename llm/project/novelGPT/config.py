# 全局配置文件
import os
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 项目根目录

TRAIN_DATA_PATH = os.path.join(BASE_DIR, '..', '..','data','jinyong')  # 训练数据路径
TRAIN_DATA_PATH = os.path.normpath(TRAIN_DATA_PATH)  # 规范化路径

MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'bin')  # 模型保存路径
MODEL_SAVE_PATH = os.path.normpath(MODEL_SAVE_PATH)  # 规范化路径

VOCAB_SIZE = 8000  # 词汇表大小

TRAIN_FILE = [
    "金庸-天龙八部.txt",
    "金庸-侠客行.txt",
    "金庸-笑傲江湖.txt",
    "金庸-倚天屠龙记.txt",
    "金庸-越女剑.txt"
]

BLOCK_SIZE = 256  # 输入文本块大小
BATCH_SIZE = 16

LR = 1e-4
EPOCHS = 10  

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  

D_MODEL = 256
N_HEAD = 8

NUM_LAYERS = 6 # Transformer层数