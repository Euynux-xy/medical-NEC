"""
组合数据集 - 合并多个类别的数据
"""
import os

import numpy as np
from torch.utils.data import ConcatDataset, Sampler

from data.dataset import XrayDataset


class BalancedBatchSampler(Sampler):
    """按类别均衡采样的批次采样器"""

    def __init__(self, combined_dataset, batch_size):
        """
        Args:
            combined_dataset: CombinedXrayDataset 对象
            batch_size: 批大小
        """
        self.batch_size = batch_size
        self.dataset = combined_dataset.dataset 
        

        self.class_indices = {i: [] for i in range(len(combined_dataset.class_names))}

        cumulative_idx = 0
        for sub_dataset_idx, sub_dataset in enumerate(self.dataset.datasets):
            for sample_idx in range(len(sub_dataset)):
                self.class_indices[sub_dataset_idx].append(cumulative_idx + sample_idx)
            cumulative_idx += len(sub_dataset)

        for class_idx in self.class_indices:
            np.random.shuffle(self.class_indices[class_idx])

    def __iter__(self):
        class_indices = {k: list(v) for k, v in self.class_indices.items()}

        valid_classes = [k for k, v in class_indices.items() if len(v) > 0]
        num_valid_classes = len(valid_classes)

        if num_valid_classes == 0:
            raise ValueError("没有有效的类别数据")

        samples_per_class = self.batch_size // num_valid_classes
        remaining = self.batch_size % num_valid_classes

        all_indices = []

        min_samples = min(len(class_indices[k]) for k in valid_classes)
        num_batches = min_samples // samples_per_class if samples_per_class > 0 else 0
        
        for batch_num in range(num_batches):
            batch = []
            for class_idx_pos, class_idx in enumerate(valid_classes):
                take = samples_per_class + (1 if class_idx_pos < remaining else 0)
                batch.extend(class_indices[class_idx][:take])
                class_indices[class_idx] = class_indices[class_idx][take:]

            np.random.shuffle(batch)
            all_indices.extend(batch)

        for idx in all_indices:
            yield idx

    def __len__(self):
        valid_classes = [k for k, v in self.class_indices.items() if len(v) > 0]
        if not valid_classes:
            return 0
        
        num_valid_classes = len(valid_classes)
        samples_per_class = self.batch_size // num_valid_classes
        min_samples = min(len(self.class_indices[k]) for k in valid_classes)
        num_batches = min_samples // samples_per_class if samples_per_class > 0 else 0
        return num_batches * self.batch_size


class CombinedXrayDataset:
    """组合多类别的 X 光数据集"""

    def __init__(
        self,
        text_data_dir: str,
        global_image_root: str,
        local_image_root: str,
        local_image_subdir: str = "image",
        image_size: int = 224,
        is_train: bool = True,
        class_names: list = None,
    ):
        """
        Args:
            text_data_dir: JSON数据目录（如 ./text_generate）
            global_image_root: 全局图像根目录（如 ./NEC_global）
            local_image_root: 局部图像根目录（如 ./NEC_local）
            local_image_subdir: 局部图子目录，"image" 或 "mask"
            image_size: 图像尺寸
            is_train: 是否为训练集
            class_names: 类别名称列表
        """
        self.text_data_dir = text_data_dir
        self.global_image_root = global_image_root
        self.local_image_root = local_image_root
        self.local_image_subdir = local_image_subdir
        self.image_size = image_size
        self.is_train = is_train
        
        if class_names is None:
            self.class_names = ["normal", "stage1", "stage2", "stage3"]
        else:
            self.class_names = class_names
        
        # 构建数据集
        self.dataset = self._build_dataset()
    
    def _build_dataset(self):
        """构建组合数据集"""
        datasets = []
        split_type = "train" if self.is_train else "test"
        
        for class_name in self.class_names:
            json_path = os.path.join(
                self.text_data_dir,
                f"{split_type}_{class_name}.json"
            )
            
            # 检查文件是否存在
            if os.path.exists(json_path):
                dataset = XrayDataset(
                    json_path=json_path,
                    global_image_root=self.global_image_root,
                    local_image_root=self.local_image_root,
                    local_image_subdir=self.local_image_subdir,
                    image_size=self.image_size,
                    is_train=self.is_train,
                    class_names=self.class_names,
                )
                datasets.append(dataset)
                print(f"加载 {json_path}: {len(dataset)} 个样本")
            else:
                print(f"警告: 文件 {json_path} 不存在，跳过")
        
        if len(datasets) == 0:
            raise ValueError(f"没有找到任何 {split_type} 数据文件")
        
        # 合并所有数据集
        combined_dataset = ConcatDataset(datasets)
        print(f"总共加载 {len(combined_dataset)} 个样本")
        
        return combined_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

