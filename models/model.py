"""
多模态 X 光分类模型

主图 + 局部图 + 文本 + DPP 选择 + Transformer 选头
"""
from typing import Optional

import torch
import torch.nn as nn

from models.dpp_module import DPPModule
from models.head_selection_transformer import HeadSelectionTransformerBlock
from models.text_encoder import TextEncoder
from models.vision_encoder import VisionEncoder


class XrayMultimodalModel(nn.Module):
    def __init__(
        self,
        num_learnable_tokens: int = 4,
        vision_dim: int = 768,
        text_dim: int = 512,
        hidden_dim: int = 512,
        num_classes: int = 4,
        dpp_k: int = 6,
        num_heads: int = 2,
        h_on: int = 1,
        mlp_ratio: float = 2.0,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        clip_model_cache_dir: Optional[str] = None,
        freeze_vision: bool = False,
        use_vision_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.dpp_k = dpp_k
        self.num_learnable_tokens = num_learnable_tokens

        trainable_vision = not freeze_vision
        use_lora = trainable_vision and use_vision_lora

        common = dict(
            model_name=clip_model_name,
            trainable=trainable_vision,
            cache_dir=clip_model_cache_dir,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        self.global_encoder = VisionEncoder(**common)
        self.local_encoder = VisionEncoder(**common)

        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.Dropout(0.2),
        )

        self.text_encoder = TextEncoder(clip_model_name, num_learnable_tokens, clip_model_cache_dir)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.Dropout(0.2),
        )

        self.dpp = DPPModule(hidden_dim, k=dpp_k)

        n_plus_2 = num_learnable_tokens + 2
        seq_len = dpp_k + n_plus_2
        self.transformer = HeadSelectionTransformerBlock(
            dim=hidden_dim, num_heads=num_heads, h_on=h_on, mlp_ratio=mlp_ratio, dropout=0.1
        )
        self.feat_linear = nn.Linear(seq_len * hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        main_image: torch.Tensor,
        local_image: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        main_image / local_image: (B, 3, H, W)
        text_tokens: (B, 77)
        return: (B, num_classes)
        """
        global_feat = self.global_encoder(main_image)
        local_feat = self.local_encoder(local_image)
        visual_seq = torch.cat([global_feat, local_feat], dim=1)
        visual_seq = self.vision_proj(visual_seq)

        text_seq = self.text_encoder(text_tokens)
        text_seq = self.text_proj(text_seq)

        visual_sel = self.dpp(visual_seq, text_seq)
        mixed = torch.cat([visual_sel, text_seq], dim=1)
        n_vis = visual_sel.size(1)
        mixed = self.transformer(mixed, num_visual_tokens=n_vis)
        flat = mixed.flatten(1)
        fused = self.feat_linear(flat)
        return self.classifier(fused)

    def get_trainable_params(self):
        """获取可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]
