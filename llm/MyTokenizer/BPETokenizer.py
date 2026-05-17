# BPE算法实现
import pickle
import re
from collections import OrderedDict,defaultdict
from tqdm import tqdm

class BPETokenizer:
    def __init__(self,max_vocab_size):
        self.next_id = 0 # 下一个Token的ID
        self.b2i = OrderedDict() # 字节对到ID的映射 映射
        self.i2b = OrderedDict() # ID到字节对的映射 反映射
        self.merges = [] # 保留合并规则


        self.s2i = OrderedDict() # special token
        self.i2s = OrderedDict()
        self.max_vocab_size = max_vocab_size

    def get_most_frequent_pair(self, tokens_list):
        # 统计字节对的频率，找到出现频率最高的字节对
        pair_fre = defaultdict(int)
        # 遍历每一条 token 序列
        for seq in tokens_list:
            # 遍历每条序列里相邻的 token 对
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                pair_fre[pair] += 1

        if not pair_fre:
            return None
        return max(pair_fre, key=pair_fre.get)

    def merge_pair(self, pair):
        # 将出现频率最高的字节对合并为一个新的Token
        token_a, token_b = pair
        new_token = token_a + token_b
        self.b2i[new_token] = self.next_id
        self.i2b[self.next_id] = new_token
        self.next_id += 1
        return new_token
    
    def vocab_size(self):
        return self.next_id
    
    def vocab(self):
        v={}
        v.update(self.i2b)
        v.update({id:token.encode('utf-8') for id,token in self.sp_i2s.items()})
        return v


    def add_special_tokens(self,special_tokens):
        for token in special_tokens:
            if token not in self.s2i:
                self.s2i[token]=self.next_id
                self.i2s[self.next_id]=token
                self.next_id+=1

    def train(self,texts_list):
        # 单字节id
        for i in range(256):
            self.b2i[bytes([i])] = i
            self.i2b[i] = bytes([i])
        self.next_id = 256

        # 将语料转为字节码
        tokens_list = []
        for text in texts_list:
            token = [bytes([b]) for b in text.encode('utf-8')]
            tokens_list.append(token)
        
        progress=tqdm(total=self.max_vocab_size-256)
        while True:
            if self.next_id >= self.max_vocab_size:
                break

            # 统计字节对的频率，找到出现频率最高的字节对
            most_frequent_pair = self.get_most_frequent_pair(tokens_list)
            if not most_frequent_pair:
                break
            # 将该相邻token组成新token并编号
            new_token = self.merge_pair(most_frequent_pair)
            self.merges.append(most_frequent_pair)   # 记录合并顺序
            token_a,token_b = most_frequent_pair
            print('='*20)
            print(f"合并{token_a},{token_b} ======> {new_token}")
            # 将原token序列中的合并上述产生的新token
            new_tokens_list = []
            for seq in tokens_list:
                new_seq = []
                i = 0
                while i < len(seq):
                    # 如果当前位置是要合并的对 → 替换成新 token
                    if i < len(seq)-1 and seq[i] == token_a and seq[i+1] == token_b:
                        new_seq.append(new_token)
                        i += 2  # 跳过下一个
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_tokens_list.append(new_seq)

            # 刷新进度条
            progress.update(1)
            # 更新 token 列表，进入下一轮合并
            tokens_list = new_tokens_list
    
    # 保存序列化
    def save(self, file_path):
        with open(file_path, 'wb') as fp:
            pickle.dump((self.b2i, self.s2i, self.next_id, self.merges), fp)

    # ===================== 修复：加载时读取 merges 合并规则 =====================
    def load(self, file_path):
        with open(file_path, 'rb') as fp:
            self.b2i, self.s2i, self.next_id, self.merges = pickle.load(fp)
        # 重建反向映射
        self.i2b = {v: k for k, v in self.b2i.items()}
        self.i2s = {v: k for k, v in self.s2i.items()}

    def encode(self, text):
        """
        重写标准BPE编码：
        1. 拆分特殊token
        2. 普通文本按训练的合并规则合并
        3. 无无效方法调用
        """
        # 1. 正则拆分特殊token（保留你原逻辑）
        if self.s2i:
            pattern = '(' + '|'.join([re.escape(tok) for tok in self.s2i]) + ')'
            splits = re.split(pattern, text)
        else:
            splits = [text]

        enc_ids = []
        enc_tokens = []

        for sub_text in splits:
            # 处理特殊token
            if sub_text in self.s2i:
                enc_ids.append(self.s2i[sub_text])
                enc_tokens.append(sub_text.encode('utf-8'))
                continue

            # 2. 普通文本转字节
            tokens = [bytes([b]) for b in sub_text.encode('utf-8')]
            
            # 3. 【核心】按照训练的合并规则合并（标准BPE逻辑）
            for pair in self.merges:
                a, b = pair
                new_token = a + b
                if new_token not in self.b2i:
                    continue
                
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens)-1 and tokens[i] == a and tokens[i+1] == b:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            # 4. 收集结果
            enc_ids.extend([self.b2i[tok] for tok in tokens])
            enc_tokens.extend(tokens)

        return enc_ids, enc_tokens
    
    def decode(self,ids):
        bytes_list=[]
        for id in ids:
            if id in self.i2s:
                bytes_list.append(self.i2s[id].encode('utf-8'))
            else:
                bytes_list.append(self.i2b[id])
        return b''.join(bytes_list).decode('utf-8',errors='replace')