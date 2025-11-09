
import torch
from dynamic_sparse_attention import DynamicSparseAttention

q = torch.randn(1, 5, 16)
k = torch.randn(1, 5, 16)
v = torch.randn(1, 5, 16)

attn = DynamicSparseAttention(dim=16)
output, weights, tau = attn(q, k, v)

print("Output shape:", output.shape)
print("Attention weights:", weights)
print("Dynamic tau:", tau)
