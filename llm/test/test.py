x = [1,2,3,4,5,6,7,8]
import torch

base = torch.tensor(10000,dtype=torch.float32)


# 旋转编码测试
def encode():

    # theta
    theta = 1.0 / (base ** (torch.arange(0,8,2,dtype=torch.float32)/8))
    print(theta)
    # m
    m = torch.arange(0,8,dtype=torch.float32)
    print(m)
    freqs = torch.outer(m,theta)
    print("freqs shape = ",freqs.shape)
    print(freqs)

    # 批量并行计算，不用实时算三角函数
    cos = torch.cos(freqs)

    sin = torch.sin(freqs)

    return cos,sin

cos , sin = encode()
print(cos[0:2])
print(sin[0:2])

