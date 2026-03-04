"""
测试脚本
"""
import glob
import os
from collections import Counter

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import logging as transformers_logging

from config import Config
from data.combined_dataset import CombinedXrayDataset
from models.model import XrayMultimodalModel

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()


def load_model(path, device, cfg):
    ckpt = torch.load(path, map_location=device)

    use_vision_lora = ckpt.get("use_vision_lora", cfg.use_vision_lora)
    lora_r = ckpt.get("lora_r", cfg.lora_r)
    lora_alpha = ckpt.get("lora_alpha", cfg.lora_alpha)
    lora_dropout = ckpt.get("lora_dropout", cfg.lora_dropout)

    model = XrayMultimodalModel(
        num_learnable_tokens=cfg.num_learnable_tokens,
        vision_dim=cfg.vision_dim,
        text_dim=cfg.text_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=cfg.num_classes,
        dpp_k=cfg.dpp_k,
        num_heads=cfg.num_heads,
        h_on=cfg.h_on,
        mlp_ratio=cfg.mlp_ratio,
        clip_model_name=cfg.clip_model_name,
        clip_model_cache_dir=cfg.clip_model_cache_dir,
        freeze_vision=cfg.freeze_vision,
        use_vision_lora=use_vision_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    ).to(device)
    
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    cfg = Config()
    cfg.num_classes = len(cfg.class_names)
    device = torch.device(cfg.device)

    test_ds = CombinedXrayDataset(
        text_data_dir=cfg.text_data_dir,
        global_image_root=cfg.global_image_root,
        local_image_root=cfg.local_image_root,
        local_image_subdir=cfg.local_image_subdir,
        image_size=cfg.image_size,
        is_train=False,
        class_names=cfg.class_names,
    )
    
    loader = DataLoader(
        test_ds, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    checkpoint_dir = cfg.checkpoint_dir
    train_output_path = os.path.join(checkpoint_dir, "train_output.txt")

    ckpt_path = None
    best_val_acc = -1.0

    if os.path.isfile(train_output_path):
        try:
            current_val_acc = None
            with open(train_output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "Val   Acc:" in line:
                        after = line.split("Val   Acc:", 1)[-1].strip()
                        acc_str = after.split("%")[0].strip()
                        current_val_acc = float(acc_str)
                    elif line.startswith("Saved best to "):
                        current_ckpt = line.split("Saved best to ", 1)[-1].strip()
                        if current_val_acc is not None and os.path.isfile(current_ckpt):
                            if current_val_acc > best_val_acc:
                                best_val_acc = current_val_acc
                                ckpt_path = current_ckpt
        except Exception as e:
            print(f"Warning: failed to parse {train_output_path}: {e}")

    if ckpt_path is None:
        pattern = os.path.join(checkpoint_dir, "best_epoch_*.pth")
        ckpts = glob.glob(pattern)
        if not ckpts:
            print(f"No checkpoint found in {checkpoint_dir}. Train first.")
            return
        ckpt_path = max(ckpts, key=os.path.getctime)


    model = load_model(ckpt_path, device, cfg)
    all_pred, all_label = [], []
    
    print(f"Testing on checkpoint: {ckpt_path}")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            main_img = batch["main_image"].to(device)
            local_img = batch["local_image"].to(device)
            text = batch["text_tokens"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            label = batch["label"]
            
            logits = model(
                main_image=main_img,
                local_image=local_img,
                text_tokens=text,
                text_attention_mask=text_attention_mask,
            )
            pred = logits.argmax(dim=1).cpu().numpy()
            
            all_pred.extend(pred)
            all_label.extend(label.numpy())


    acc = accuracy_score(all_label, all_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_label, all_pred, average="macro", zero_division=0
    )

    labels_present = sorted(set(all_label) | set(all_pred))
    cm = confusion_matrix(all_label, all_pred, labels=labels_present)
    class_names = cfg.class_names or [f"C{i}" for i in range(cfg.num_classes)]
    names_present = [class_names[i] for i in labels_present]


    print("\n" + "="*30 + " Test Results " + "="*30)
    true_counts = Counter(all_label)
    print("Class distribution in Test Set:", {class_names[i]: true_counts.get(i, 0) for i in range(cfg.num_classes)})
    
    print("\nConfusion Matrix:")
    n = len(names_present)
    header_row = "True \\ Pred" + "".join(f"{cn:>12}" for cn in names_present)
    print(header_row)
    print("-" * len(header_row))
    for i in range(n):
        row_str = f"{names_present[i]:>11}" + "".join(f"{cm[i, j]:>12}" for j in range(n))
        print(row_str)
    
    print("\nMain Metrics:")
    metrics_header = "Accuracy\tPrecision\tRecall  \tF1-Score"
    metrics_row = f"{acc*100:.2f}%\t{macro_p*100:.2f}%\t{macro_r*100:.2f}%\t{macro_f1*100:.2f}%"
    print(metrics_header)
    print("-" * 60)
    print(metrics_row)
    print("="*74)


if __name__ == "__main__":
    main()