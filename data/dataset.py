"""
数据集模块
"""
import json
import os
import re
from typing import Dict, List, Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer

from config import Config

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class XrayDataset(Dataset):
    """单类别 X 光多模态数据集"""

    def __init__(
        self,
        json_path: str,
        global_image_root: str,
        local_image_root: str,
        local_image_subdir: str = "image",
        image_size: int = 224,
        is_train: bool = True,
        class_names: List[str] = None,
    ):
        """
        Args:
            json_path: JSON数据文件路径（如 ./text_generate/train_normal.json）
            global_image_root: 全局图像根目录（如 ./NEC_global）
            local_image_root: 局部图像根目录（如 ./NEC_local）
            local_image_subdir: 局部图子目录名，"image" 或 "mask"，路径为 .../NEC_local/{train|test}/{class}/{local_image_subdir}/xxx
            image_size: 图像尺寸
            is_train: 是否为训练集
            class_names: 类别名称列表，用于映射
        """
        self.global_image_root = global_image_root
        self.local_image_root = local_image_root
        self.local_image_subdir = local_image_subdir
        self.image_size = image_size
        self.is_train = is_train
        
        if class_names is None:
            self.class_names = ["normal", "stage1", "stage2", "stage3"]
        else:
            self.class_names = class_names
        
        # 从JSON文件名提取类别和数据集类型
        json_filename = os.path.basename(json_path)
        match = re.match(r'(train|test)_(.+)\.json', json_filename)
        if match:
            split_type, class_name = match.groups()
            self.split_type = split_type  # train 或 test
            self.class_name = class_name  # normal, stage1, stage2, stage3
        else:
            raise ValueError(f"无法从文件名 {json_filename} 解析类别和数据集类型")
        

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        cfg = Config()
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                cfg.clip_model_name,
                cache_dir=cfg.clip_model_cache_dir,
                local_files_only=True,
            )
        except Exception as e:
            print(f"本地加载 CLIP tokenizer 失败（cache_dir={cfg.clip_model_cache_dir}）：{e}")
            print("回退到 transformers 默认行为（可能会尝试联网下载 tokenizer）")
            self.tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_model_name)
        
        # 图像预处理（与 CLIP 预训练分布对齐）
        if is_train:
            self.transform_main = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
            ])
            self.transform_local = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
            ])
        else:
            self.transform_main = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
            ])
            self.transform_local = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
            ])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        item = self.data[idx]
        image_name = item["item"]
        description = item["description"]
        
        global_image_path = os.path.join(
            self.global_image_root,
            self.split_type,  
            self.class_name, 
            "image",
            image_name
        )
        
 
        local_image_path = os.path.join(
            self.local_image_root,
            self.split_type, 
            self.class_name,  
            self.local_image_subdir,  
            image_name
        )

        try:
            main_image = Image.open(global_image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading global image {global_image_path}: {e}")

            main_image = Image.new('RGB', (self.image_size, self.image_size), color='black')
        

        try:
            local_image = Image.open(local_image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading local image {local_image_path}: {e}")

            local_image = main_image.copy()
        
        main_image = self.transform_main(main_image)
        local_image = self.transform_local(local_image)
        
        text_inputs = self.tokenizer(
            description,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        text_tokens = text_inputs["input_ids"].squeeze(0)
        text_attention_mask = text_inputs["attention_mask"].squeeze(0)
        
        result = {
            "main_image": main_image,
            "local_image": local_image,
            "text": description,
            "text_tokens": text_tokens,
            "text_attention_mask": text_attention_mask,
            "item": image_name
        }
        
        if "label" in item:
            result["label"] = torch.tensor(item["label"], dtype=torch.long)
        else:

            label = self.class_names.index(self.class_name)
            result["label"] = torch.tensor(label, dtype=torch.long)
        
        return result

