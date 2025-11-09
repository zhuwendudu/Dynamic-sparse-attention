import torch
from dynamic_sparse_attention import DynamicSparseAttention

# 模拟输入 (批次=1, 序列长度=5, 特征维度=16)
q = torch.randn(1, 5, 16)
k = torch.randn(1, 5, 16)
v = torch.randn(1, 5, 16)

attn = DynamicSparseAttention(dim=16)

output, weights, tau = attn(q, k, v)

print("输出向量形状:", output.shape)
print("注意力权重:", weights[0].data)
print("动态τ温度:", tau[0].data)
