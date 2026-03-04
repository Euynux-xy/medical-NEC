"""
配置文件
"""
import os
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Config:
    """训练与推理配置"""

    # 数据路径
    data_root: str = "./data"
    text_data_dir: str = "./data/text_generate"
    global_image_root: str = "./data/NEC_global"
    local_image_root: str = "./data/NEC_local"
    local_image_subdir: str = "image"
    class_names: Optional[List[str]] = None

    # 模型
    freeze_vision: bool = False
    use_vision_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # 模型维度
    num_learnable_tokens: int = 8
    vision_dim: int = 768
    text_dim: int = 512
    hidden_dim: int = 64
    num_classes: int = 4
    dpp_k: int = 50
    num_heads: int = 8
    h_on: int = 4
    mlp_ratio: float = 2.0

    # 训练超参数
    batch_size: int = 8
    num_epochs: int = 40
    learning_rate: float = 1e-4
    warmup_epochs: int = 5
    weight_decay: float = 0.0
    num_workers: int = 2
    focal_gamma: float = 2.0
    focal_alpha: Optional[List[float]] = None

    # 输出路径
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # 设备与 CLIP（clip_model_cache_dir 为 None 时使用 HuggingFace 默认缓存）
    device: Optional[str] = None
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_model_cache_dir: Optional[str] = None

    # 其他
    seed: int = 42
    image_size: int = 224

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["normal", "stage1", "stage2", "stage3"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    