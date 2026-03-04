"""
DPP 模块
"""
import torch
import torch.nn as nn


class DPPModule(nn.Module):


    def __init__(self, feature_dim: int, k: int = 16):
        super().__init__()
        self.k = k

    def forward(
        self,
        visual_features: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        visual_features: (B, N_v, D)
        text_tokens: (B, N_t, D)
        return: (B, k, D)
        """
        B, N, D = visual_features.shape
        device = visual_features.device
        if N == 0:
            raise ValueError("visual_features has zero tokens.")

        target_k = self.k
        T = min(target_k, N)

        # [CDPruner] Cosine similarity (visual-visual)
        vis_norm = visual_features / (visual_features.norm(dim=-1, keepdim=True) + 1e-6)
        vis_norm = vis_norm.float()
        similarity = torch.bmm(vis_norm, vis_norm.transpose(1, 2))  # (B, N, N)

        # Query relevance (visual-text)
        txt_norm = text_tokens / (text_tokens.norm(dim=-1, keepdim=True) + 1e-6)
        txt_norm = txt_norm.float()
        relevance = torch.bmm(vis_norm, txt_norm.transpose(1, 2))  # (B, N_v, N_t)
        relevance = relevance.max(dim=-1)[0] 
        r_min = relevance.min(dim=1, keepdim=True)[0]
        r_max = relevance.max(dim=1, keepdim=True)[0]
        relevance = (relevance - r_min + 1e-6) / (r_max - r_min + 1e-6)  # (B, N)

        # kernel matrix
        kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)  # (B, N, N)

        # Fast MAP inference of conditional DPP
        cis = torch.zeros((T, B, N), device=device, dtype=kernel.dtype)
        di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()  # (B, N)
        select_idx = torch.empty((T, B), dtype=torch.long, device=device)

        batch_arange = torch.arange(B, device=device)
        for i in range(T):
            # 数值稳定：避免负值导致 sqrt NaN
            safe_di2s = torch.clamp(di2s, min=1e-12)
            j = torch.argmax(safe_di2s, dim=-1)  # (B,)
            select_idx[i] = j

            # eis = (kernel[b,j,:] - sum_t cis[t,b,j[b]]*cis[t,b,:]) / sqrt(di2s[b,j[b]])
            k_j = kernel[batch_arange, j, :]  # (B, N)
            denom = torch.sqrt(torch.clamp(di2s[batch_arange, j], min=1e-12)).unsqueeze(-1) + 1e-8
            if i == 0:
                eis = k_j / denom
            else:
                cov = torch.einsum("tb,tbn->bn", cis[:i, batch_arange, j], cis[:i])  # (B, N)
                eis = (k_j - cov) / denom
            cis[i] = eis
            di2s = di2s - eis.pow(2)
            di2s[batch_arange, j] = -float("inf")

        # select_idx (T, B) -> (B, T)
        select_idx = torch.sort(select_idx.t(), dim=1).values  # (B, T)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, T)
        selected = visual_features[batch_idx, select_idx]
        if T < target_k:
            pad = torch.zeros((B, target_k - T, D), dtype=selected.dtype, device=device)
            selected = torch.cat([selected, pad], dim=1)
        return selected
