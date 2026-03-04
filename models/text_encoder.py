"""
文本编码器
"""
from typing import Optional
import torch
import torch.nn as nn
from transformers import CLIPTextModel


class TextEncoder(nn.Module):

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        num_learnable_tokens: int = 4,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        kwargs = {} if cache_dir is None else {"cache_dir": cache_dir}
        self.backbone = CLIPTextModel.from_pretrained(model_name, **kwargs)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.num_learnable_tokens = num_learnable_tokens
        hidden_size = self.backbone.config.hidden_size
        self.learnable_tokens = nn.Parameter(torch.randn(num_learnable_tokens, hidden_size) * 0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = input_ids.size(0)
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state
        cls_token = hidden[:, 0:1, :]
        if attention_mask is not None:
            eos_index = attention_mask.sum(dim=1).long().clamp(min=1) - 1
            eos_token = hidden[torch.arange(B, device=hidden.device), eos_index].unsqueeze(1)
        else:
            eos_token = hidden[:, -1:, :]
        learnable = self.learnable_tokens.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([learnable, cls_token, eos_token], dim=1)
