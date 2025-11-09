
import torch
import torch.nn.functional as F

class DynamicSparseAttention(torch.nn.Module):
    def __init__(self, dim, base_tau=1.0, min_tau=0.3):
        super().__init__()
        self.scale = dim ** -0.5
        self.base_tau = base_tau
        self.min_tau = min_tau

    def forward(self, q, k, v):
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        complexity = attn_scores.var(dim=-1, keepdim=True)
        tau = self.base_tau / (1 + complexity)
        tau = tau.clamp(min=self.min_tau)
        attn = F.softmax(attn_scores / tau, dim=-1)
        return attn @ v, attn, tau
