"""
Transformer Block with Selective Attention
"""
import math
from typing import Optional
import torch
import torch.nn as nn


class SemanticHeadSelectionAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        h_on: Optional[int] = None,
        dropout: float = 0.1,
        soft_mask_eps: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.h_on = h_on if h_on is not None else max(1, num_heads // 2)
        self.scale = math.sqrt(self.head_dim)
        self.soft_mask_eps = soft_mask_eps  

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, num_visual_tokens: int) -> torch.Tensor:

        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) 
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (torch.matmul(q, k.transpose(-2, -1)) / self.scale).softmax(dim=-1)
        attn = self.attn_dropout(attn)

        head_scores = attn.sum(dim=(0, 2, 3))

        h_on = min(self.h_on, self.num_heads)
        _, top_idx = torch.topk(head_scores, h_on, largest=True)
        eps = self.soft_mask_eps
        head_scale = torch.full(
            (self.num_heads,), eps, device=x.device, dtype=x.dtype
        )
        head_scale[top_idx] = (float(self.num_heads) - float(self.num_heads - h_on) * eps) / float(h_on)
        head_scale = head_scale.view(1, self.num_heads, 1, 1)

        out = torch.matmul(attn, v) 
        out = out * head_scale
        out = out.transpose(1, 2).contiguous().view(B, N, self.dim)
        return out


class HeadSelectionTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        h_on: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        soft_mask_eps: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SemanticHeadSelectionAttention(
            dim=dim, num_heads=num_heads, h_on=h_on, dropout=dropout, soft_mask_eps=soft_mask_eps
        )
        self.out_proj = nn.Linear(dim, dim) 

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, num_visual_tokens: int) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, num_visual_tokens=num_visual_tokens)
        h = self.out_proj(h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x
