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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B = input_ids.size(0)
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids)
            hidden = out.last_hidden_state 
        cls_token = hidden[:, 0:1, :]
        eos_token = hidden[:, -1:, :] 
        learnable = self.learnable_tokens.unsqueeze(0).expand(B, -1, -1)  #
        return torch.cat([learnable, cls_token, eos_token], dim=1)  
