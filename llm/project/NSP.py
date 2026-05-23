# 先对数据预处理
# 将滕王阁序分成句子对，切分为 上句 + 下句 训练对 原数据有三句式的格式，所以删除了嗟乎、呜呼、勃等句子
import re

def process_tengwangge_text(file_path):
    """
    处理滕王阁序：按 ；。！？切分句子 → 句子内按【中文逗号，】拆分为上句+下句
    :return: clean_sentences 清洗后的原句, train_pairs 训练对[[上句,下句], ...]
    """
    # 1. 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    # 2. 清洗全角空格、空白
    data = data.replace('\u3000', '').strip()
    data = re.sub(r'\s+', ' ', data)

    # 3. 按句末标点切分句子
    sentences = re.split(r'[；。！？\n]+', data)

    # 4. 清洗无效句子
    clean_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if sent and len(sent) > 1:
            clean_sentences.append(sent)

    train_pairs = []
    for sent in clean_sentences:
        # 只处理包含中文逗号、且能拆分成2段的句子
        if '，' in sent:
            parts = sent.split('，', 1)  # 只分割第一个逗号，避免多逗号混乱
            if len(parts) == 2:
                shangju = parts[0].strip()  # 上句
                xiaju = parts[1].strip()    # 下句
                train_pairs.append([shangju, xiaju])

    return clean_sentences, train_pairs

def save_train_pairs(train_pairs, output_file):
    """
    将训练对保存到文件，每行格式：上句 \t 下句
    :param train_pairs: 训练对列表，格式 [[上句1,下句1], [上句2,下句2], ...]
    :param output_file: 输出文件路径，如 train_pairs.txt
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # 遍历每一对 上句+下句
        for upper_sent, lower_sent in train_pairs:
            # 写入：上句 + 制表符 + 下句 + 换行
            f.write(f"{upper_sent}\t{lower_sent}\n")


import torch
from torch.utils.data import Dataset, DataLoader
from MyTokenizer import BPETokenizer

# 超参数设计
VOCAB_SIZE = 3000  # 假设词表大小为3000
EMBEDDING_DIM = 128
NUM_HEADS = 4
FFN_DIM = 256
BATCH_SIZE = 2
MAX_SEQ_LEN = 20
EPOCHS = 50
LR = 1e-3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_file = 'llm/data/train_pairs.txt'

def load_train_pairs(file_path):
    train_pairs = []
    corpus = ""  # 用来训练BPE的总语料
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                shang, xia = parts
                train_pairs.append([shang, xia])
                corpus += shang + xia  # 拼接所有文本训BPE
    return train_pairs, corpus

train_pairs, corpus = load_train_pairs(data_file)

bpe = BPETokenizer(max_vocab_size=VOCAB_SIZE)
bpe.train([corpus])  # 用整个语料训练BPE
bpe.add_special_tokens(['<|im_start|>', '<|im_end|>', '<|endoftext|>', '<|padding|>'])  # 添加特殊token
bpe.save('llm/bin/bpe_tokenizer.bin')  # 保存训练好的BPE模型
#bpe.load('llm/bin/bpe_tokenizer.bin')  # 加载预训练的BPE模型
class PoemDataset(Dataset):
    def __init__(self, train_pairs, bpe):
        self.train_pairs = train_pairs
        self.bpe = bpe

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        # 1. 取出一组 上句→下句
        src_sent, tgt_sent = self.train_pairs[idx]
        
        # 2. BPE 编码（你已经验证过的步骤）
        src_ids,_ = self.bpe.encode(src_sent)
        tgt_ids,_ = self.bpe.encode(tgt_sent)

        # 3. 填充到固定长度（[PAD] = 0）
        src_ids = self._pad(src_ids)
        tgt_ids = self._pad(tgt_ids)

        # 返回模型需要的张量
        return torch.LongTensor(src_ids), torch.LongTensor(tgt_ids)

    def _pad(self, ids):
        # 不足补0，超长截断
        if len(ids) < MAX_SEQ_LEN:
            return ids + [0] * (MAX_SEQ_LEN - len(ids))
        return ids[:MAX_SEQ_LEN]

from MyEmbedding import MyEmbedding
from decode import MyDecoder
from attention import MyMultiHeadAttention

dataset = PoemDataset(train_pairs, bpe)




class PoemNSPModel(torch.nn.Module):
    def __init__(self):
        super(PoemNSPModel, self).__init__()
        self.embedding = MyEmbedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.decoder_layers = torch.nn.ModuleList([
            MyDecoder(EMBEDDING_DIM, NUM_HEADS, FFN_DIM) for _ in range(2)
        ])
        self.fc = torch.nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)
    
    def forward(self, x, mask):
        # 数据流：token → 嵌入 → 多层Decoder → 输出
        x = self.embedding(x)
        for layer in self.decoder_layers:
            x = layer(x, mask)
        return self.fc(x)
    
model = PoemNSPModel().to(DEVICE)

dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)
# 2. 获取掩码
causal_mask = torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN,device=DEVICE)).unsqueeze(0).unsqueeze(0)
# 3. 损失函数+优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def train():
    print("==== 开始批量训练 ====")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        # 批量读取数据
        for src, tgt in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            
            # 前向传播
            output = model(src, causal_mask)
            # 计算损失
            loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # 打印训练日志
        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), 'llm/bin/poem_nsp_model.pth')
    print(f"\n✅模型已保存：llm/bin/poem_nsp_model.pth")

def load_model(model, path):
    model = PoemNSPModel().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model





# ===================== 诗句生成函数（输入上句 → 输出下句） =====================
def generate_poem(model, bpe, input_sentence):
    model.eval()
    with torch.no_grad():
        # ============== 核心修复：和训练代码完全一致！取 ids（数字），丢弃 tokens（字节） ==============
        ids, _ = bpe.encode(input_sentence)  # 只取数字ID，第二个返回值不用管
        tokens = list(ids)  # 把数字ID转列表
        
        # 填充到固定长度
        if len(tokens) < MAX_SEQ_LEN:
            tokens += [0] * (MAX_SEQ_LEN - len(tokens))
        tokens = tokens[:MAX_SEQ_LEN]
        
        # 构建张量（纯数字，无字节，无报错）
        input_tensor = torch.LongTensor([tokens]).to(DEVICE)
        
        # 模型预测
        output = model(input_tensor, causal_mask)
        pred_tokens = output.argmax(dim=-1).squeeze().tolist()

        # 过滤无效token：PAD(0) + 清理乱码
        valid_tokens = [t for t in pred_tokens if t != 0]
        valid_tokens = valid_tokens[:10]  # 限制输出长度，适配诗句
        
        # 解码+清理乱码
        result = bpe.decode(valid_tokens)
        result = result.encode("utf-8", errors="ignore").decode("utf-8")
        result = result.replace("[PAD]", "").replace("[UNK]", "").strip()
        return result

if __name__ == '__main__':
    # 加载模型
    train()
    model = load_model(PoemNSPModel(), 'llm/bin/poem_nsp_model.pth')
    # ===================== 开始测试：诗句接龙 =====================
    print("="*50)
    print("滕王阁序 诗句接龙模型（训练完成）")
    print("="*50)

    # 测试列表（你可以随便加）
    test_sentences = [
        "豫章故郡",
        "星分翼轸",
        "襟三江而带五湖",
        "物华天宝",
        "人杰地灵",
        "都督阎公之雅望",
        "十旬休假",
        "落霞与孤鹜齐飞"
    ]

    for input_s in test_sentences:
        output_s = generate_poem(model, bpe, input_s)
        print(f"输入上句：{input_s}")
        print(f"模型输出：{output_s}")
        print("-" * 30)

