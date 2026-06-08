from config import *
import torch
from torch import nn
from torch import optim
from model import TransformerModel
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import TranslationDataset, collate_fn
from load_data import load_data

train_data = load_data(f'{TRAIN_DATA_PATH}/cn.txt', f'{TRAIN_DATA_PATH}/en.txt')

dataset = TranslationDataset(train_data, max_len=MAX_LEN)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


# 初始化模型
model = TransformerModel(
    src_vocab_size=VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE,
    d_model=512,
    nhead=8,
    num_encoder_layers=3,   # 建议用3层，6层太深难训练
    num_decoder_layers=3,
    max_len=MAX_LEN,
    pad_idx=PAD_ID
).to(DEVICE)

# 损失函数（忽略PAD）
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 50 

print("开始训练...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    # 遍历批次数据
    for src_batch, tgt_input, tgt_label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        # 数据移至设备
        src_batch = src_batch.to(DEVICE)
        tgt_input = tgt_input.to(DEVICE)
        tgt_label = tgt_label.to(DEVICE)

        # 1. 前向传播
        output = model(src_batch, tgt_input)

        # 2. 计算损失
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt_label.reshape(-1)
        )

        # 3. 反向传播 & 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 打印本轮平均损失
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}  平均损失: {avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "transformer_translation.pth")

def translate_sentence(model, cn_sentence, cn_sp, en_sp):
    model.eval()
    with torch.no_grad():
        # 中文编码为ID
        src_ids = [SOS_ID] + cn_sp.encode(cn_sentence, out_type=int) + [EOS_ID]
        src_ids = src_ids[:MAX_LEN] + [PAD_ID] * (MAX_LEN - len(src_ids))
        src_tensor = torch.tensor([src_ids]).to(DEVICE)

        # 自回归生成英文
        tgt_ids = [SOS_ID]
        for _ in range(MAX_LEN):
            tgt_tensor = torch.tensor([tgt_ids]).to(DEVICE)
            output = model(src_tensor, tgt_tensor)
            
            # 取预测的下一个词
            next_token = output.argmax(-1)[:, -1].item()
            tgt_ids.append(next_token)
            
            # 遇到结束符停止
            if next_token == EOS_ID:
                break

        # 解码为英文句子
        english = en_sp.decode([idx for idx in tgt_ids if idx not in (SOS_ID, EOS_ID, PAD_ID)])
        return english
