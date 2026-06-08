import torch
from torch import optim
import os
from config import LR, EPOCHS, DEVICE, MODEL_SAVE_PATH, BLOCK_SIZE
from dataset import dataloader
from model import NovelGPT

model = NovelGPT().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
best_model_path = os.path.join(MODEL_SAVE_PATH, "best_novel_gpt.pth")

def train():
    # 最优损失设为无穷大
    best_loss = float('inf')
    total_step = len(dataloader)
    print(f"设备：{DEVICE}，总轮数：{EPOCHS}")

    for epoch in range(EPOCHS):
        model.train()
        total_epoch_loss = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # 数据移至GPU/CPU
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            _, loss = model(inputs, targets)

            loss.backward()
            # NOTE 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 缺少这行
            optimizer.step()

            # 累计损失
            total_epoch_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{total_step}], Loss: {loss.item():.4f}")

        avg_loss = total_epoch_loss / total_step
        print(f"\n Epoch [{epoch+1}/{EPOCHS}] 平均损失: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存模型权重（最优版本）
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'block_size': BLOCK_SIZE
            }, best_model_path)
            print(f"已保存最优模型！最优损失: {best_loss:.4f}\n")

    print("训练全部完成！")

train()