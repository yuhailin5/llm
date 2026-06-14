# 旋转位置编码
import torch
import torch.nn as nn


class RopeEncoding(nn.Module):

    """
    NOTE 把所有需要计算的全部先计算好，多次使用存入buffer
    """
    def __init__(self,head_dim:int,max_seq_len:int=1024,base:float=10000.0):
        super(RopeEncoding,self).__init__()
        # 定义旋转矩阵
        self.head_dim = head_dim
        assert head_dim % 2 == 0,"RoPE必须单头偶数"
        self.base = base

        # 1 计算所有位置的theta theta_i 表示第i组的旋转角度 = 1/base^(2i/d)
        theta = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))

        # 2 枚举所有token位置
        m = torch.arange(0,max_seq_len,dtype=torch.float32)

        # 3 计算m*theta outer把一维a的每个元素分别乘一维b的全部元素，拼成二维矩阵
        freqs = torch.outer(m,theta)

        # 4 算出所有cos 和 sin表
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        self.register_buffer("cos_table",cos)
        self.register_buffer("sin_table",sin)

    def forward(self,x:torch.Tensor,start_pos:int = 0):
        """
        因为 m 最大是max_seq_len
        start_pos: 开始的词向量位置
        """
        
        assert x.dim() == 4, "输入维度必须为4维 [batch, seq_len, num_heads, head_dim]"
        bsz, seq_len, n_head, dim = x.shape

        # 截取一句话所需的cos，sin
        cos = self.cos[start_pos : start_pos + seq_len]  # [seq_len, dim_half]
        sin = self.sin[start_pos : start_pos + seq_len]

        # 广播为x同类矩阵 [1, seq_len, 1, dim_half]
        cos = cos[None,:,None,:]
        sin = sin[None,:,None,:]

        x_group = x.reshape(bsz, seq_len, n_head, self.head_dim//2, 2)
        x0 = x_group[..., 0]
        x1 = x_group[..., 1]

        x0_new = x0 * cos - x1 * sin
        x1_new = x0 * sin + x1 * cos

        # stack 用法 在指定维度上新增一维，把两个张量拼在一起。
        # x0_new[b,l,d,4] -> [b,l,d,4,2]
        # flatten(n) 把n维后打包为一个维度
        x_rot = torch.stack([x0_new, x1_new], dim=-1).flatten(3)
        return x_rot
