# wordPiece算法实现（最终无BUG版）
from collections import OrderedDict,defaultdict
import pickle
class WordPieceTokenizer:
    def __init__(self,max_vocab_size):
        self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        self.token2id = OrderedDict()
        self.id2token = OrderedDict()
        self._init_special_tokens()
        self.merges = dict()
        self.word_freq = defaultdict(int)
        self.char_freq = defaultdict(int)
        self.max_vocab_size = max_vocab_size
        self.next_id = len(self.special_tokens)

    def _init_special_tokens(self):
        for idx, token in enumerate(self.special_tokens):
            self.token2id[token] = idx
            self.id2token[idx] = token

    def _seq_pre_process(self,texts):
        words = []
        for text in texts:
            text = text.replace('\n', ' ').strip()
            words.extend(text.split())

        splits = {}
        for word in words:
            self.word_freq[word] += 1
            chars = list(word)
            splits[word] = chars
            for c in chars:
                self.char_freq[c] += 1
                if c not in self.token2id:
                    self.token2id[c] = self.next_id
                    self.id2token[self.next_id] = c
                    self.next_id += 1
        return splits

    def _compute_score(self,splited):
        pair_freq = defaultdict(int)
        for word,chars in splited.items():
            if len(chars) < 2:
                continue
            for i in range(len(chars)-1):
                a, b = chars[i], chars[i+1]
                pair_freq[(a, b)] += self.word_freq[word]
        
        max_score = -1
        best_pair = None
        for (a,b),cnt in pair_freq.items():
            cur_score = cnt/(self.char_freq[a]*self.char_freq[b])
            if cur_score > max_score:
                best_pair = (a,b)
                max_score = cur_score
        return best_pair
    
    def _merge_pair(self,best_pair,splited):
        a,b = best_pair
        new_token = a+b
        if new_token not in self.token2id:
            self.token2id[new_token] = self.next_id
            self.id2token[self.next_id] = new_token
            self.next_id += 1
        self.merges[(a, b)] = new_token

        new_splited = dict()
        for word,chars in splited.items():
            new_chars = []
            i = 0
            while i<len(chars):
                if i<len(chars)-1 and chars[i] == a and chars[i+1] == b:
                    new_chars.append(new_token)
                    i+=2
                else:
                    new_chars.append(chars[i])
                    i+=1
            new_splited[word] = new_chars
            self.char_freq[new_token] += self.word_freq[word]
        return new_splited

    def train(self,texts):
        splited = self._seq_pre_process(texts)
        while len(self.token2id) < self.max_vocab_size:
            best_pair = self._compute_score(splited)
            if not best_pair:
                break
            splited = self._merge_pair(best_pair, splited)
        print("训练完成！词表大小:", len(self.token2id))
    
    # ===================== 修复核心BUG：##仅展示，不参与ID映射 =====================
    def tokenizer(self, text):
        text = text.replace('\n', ' ').strip()
        words = text.split()
        tokens = []
        for word in words:
            chars = list(word)
            # 合并逻辑
            while True:
                merged = False
                for i in range(len(chars)-1):
                    pair = (chars[i], chars[i+1])
                    if pair in self.merges:
                        new_t = self.merges[pair]
                        chars = chars[:i] + [new_t] + chars[i+2:]
                        merged = True
                        break
                if not merged:
                    break
            # 生成展示用token（带##），同时保留原始token用于编码
            if len(chars) > 0:
                tokens.append(chars[0])
                for t in chars[1:]:
                    tokens.append(f"##{t}")
        return tokens

    def encode(self, text):
        # 核心修复：去掉##，用原始token查词表
        display_tokens = self.tokenizer(text)
        real_tokens = [t.replace("##", "") for t in display_tokens]
        ids = [self.token2id.get(t, self.token2id['[UNK]']) for t in real_tokens]
        return ids, display_tokens

    def decode(self, ids):
        tokens = [self.id2token.get(i, '[UNK]') for i in ids]
        res = ""
        for t in tokens:
            res += t
        return res.strip()
    
    # ===================== 保存/加载 =====================
    def save(self, file):
        state = (self.token2id, self.id2token, self.merges, self.special_tokens, self.next_id, self.max_vocab_size)
        with open(file, 'wb') as fp:
            pickle.dump(state, fp)

    def load(self, file):
        with open(file, 'rb') as fp:
            state = pickle.load(fp)
        self.token2id, self.id2token, self.merges, self.special_tokens, self.next_id, self.max_vocab_size = state

    def vocab_size(self):
        return self.next_id