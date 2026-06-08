from config import *

def load_data(src_file, tgt_file):
    train_data = []
    with open(src_file, 'r', encoding='utf-8') as f_src, open(tgt_file, 'r', encoding='utf-8') as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            if src_line and tgt_line:
                train_data.append((src_line, tgt_line))
    return train_data

def test_load_data():
    src_file = f'{TRAIN_DATA_PATH}/cn.txt'
    tgt_file = f'{TRAIN_DATA_PATH}/en.txt'
    data = load_data(src_file, tgt_file)
    for src, tgt in data:
        print(f'Source: {src} | Target: {tgt}')

def test_load_data_val():
    src_file = f'{TRAIN_DATA_PATH}/cn.test.txt'
    tgt_file = f'{TRAIN_DATA_PATH}/en.test.txt'
    data = load_data(src_file, tgt_file)
    for src, tgt in data:
        print(f'Source: {src}\nTarget: {tgt}')
