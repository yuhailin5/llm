# 训练数据预处理
import os
from config import TRAIN_DATA_PATH,TRAIN_FILE

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def load_all_data():
    texts = []
    for filename in TRAIN_FILE:
        file_path = os.path.join(TRAIN_DATA_PATH, filename)
        texts.append(load_data(file_path))
    return "\n".join(texts)


def test_load_data():
    file_path = os.path.join(TRAIN_DATA_PATH, '金庸-越女剑.txt')
    text = load_data(file_path)
    print(text[:500])

def test_load_all_data():
    text = load_all_data()
    print(text[:500])