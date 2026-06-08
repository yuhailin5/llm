import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from load_data import load_data
from config import TRAIN_DATA_PATH
# 加载分词器模型
cn_sam = spm.SentencePieceProcessor(model_file=f'{TRAIN_DATA_PATH}/cn_spm_bpe.model')
en_sam = spm.SentencePieceProcessor(model_file=f'{TRAIN_DATA_PATH}/en_spm_bpe.model')

cn_vocab_size = cn_sam.vocab_size()
en_vocab_size = en_sam.vocab_size()

SOS_ID = en_sam.bos_id()  # 句子开始
EOS_ID = en_sam.eos_id()  # 句子结束
PAD_ID = en_sam.pad_id()  # 填充

class TranslationDataset(Dataset):
    def __init__(self,data,max_len=48):
        """
        data: List of (src_sentence, tgt_sentence) pairs
        max_len: Maximum length of the sequences
        """
        self.data = data
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_sentence, tgt_sentence = self.data[idx]
        # 编码并添加特殊标记
        src_ids = cn_sam.encode(src_sentence, out_type=int)
        tgt_ids = en_sam.encode(tgt_sentence, out_type=int)
        
        # 添加 <s> 和 </s> 标记
        src_ids = [SOS_ID] + src_ids + [EOS_ID]
        tgt_ids = [SOS_ID] + tgt_ids + [EOS_ID]

        # 截断或填充到 max_len
        src_ids = src_ids[:self.max_len] + [PAD_ID] * max(0, self.max_len - len(src_ids))
        tgt_ids = tgt_ids[:self.max_len] + [PAD_ID] * max(0, self.max_len - len(tgt_ids))

        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    # batch 是列表：[(src1, tgt1), (src2, tgt2), ...]
    
    # 1. 分离源语言、目标语言（批量解包）
    src_list, tgt_list = zip(*batch)
    
    # 2. 堆叠成批量张量 因为你的所有句子长度都是48，直接stack即可
    src_batch = torch.stack(src_list)    # shape: [batch_size, 48]
    tgt_batch = torch.stack(tgt_list)    # shape: [batch_size, 48]
    
    # 3. 【翻译必做】批量拆分 Target
    tgt_input = tgt_batch[:, :-1]        # 输入：去掉最后一个词 [batch, 47]
    tgt_label = tgt_batch[:, 1:]         # 标签：去掉第一个词 [batch, 47]
    
    # 返回训练需要的三个张量
    return src_batch, tgt_input, tgt_label
