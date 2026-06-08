import torch
from config import DEVICE, MAX_LEN, PAD_ID, SOS_ID, EOS_ID
model = torch.load("transformer_translation.pth", map_location='cpu')

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


def test_translation(model, cn_sp, en_sp):
    test_sentences = [
        "目前粮食出现阶段性过剩",
        "中国人民应当将改版后的人民币的发行时间予以公告",
        "勤劳勇敢聪明的中国人一定会解决好祖国统一的事情",
    ]

    for cn_sentence in test_sentences:
        en_translation = translate_sentence(model, cn_sentence, cn_sp, en_sp)
        print(f'中文: {cn_sentence}')
        print(f'英文翻译: {en_translation}')
        print('-' * 60)

