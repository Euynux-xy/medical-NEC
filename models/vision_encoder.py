"""
视觉编码器
"""
from typing import Optional
import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from peft import get_peft_model, LoraConfig, TaskType


class VisionEncoder(nn.Module):

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        trainable: bool = True,
        cache_dir: Optional[str] = None,
        # LoRA 参数
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        kwargs = {} if cache_dir is None else {"cache_dir": cache_dir}
        backbone = CLIPVisionModel.from_pretrained(model_name, **kwargs)

        self._use_lora = False

        if not trainable:
            for p in backbone.parameters():
                p.requires_grad = False
            self.backbone = backbone
        elif use_lora:
            for p in backbone.parameters():
                p.requires_grad = False
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"], 
                task_type=TaskType.FEATURE_EXTRACTION,
                init_lora_weights=True,
            )
            self.backbone = get_peft_model(backbone, lora_config)
            self._use_lora = True
        else:
            self.backbone = backbone

        self.hidden_size = backbone.config.hidden_size
        self.feature_dim = backbone.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 直接走 self.backbone，确保 PEFT/LoRA 前向钩子生效
        out = self.backbone(pixel_values=pixel_values)
        return out.last_hidden_state

    def get_trainable_params(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params
