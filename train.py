"""
训练脚本
"""
import gc
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import logging as transformers_logging

from config import Config
from data.combined_dataset import CombinedXrayDataset
from models.model import XrayMultimodalModel
from utils.losses import FocalLoss

# 降低 transformers 的日志级别
transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(loader, desc="Train"):
        main = batch["main_image"].to(device)
        local = batch["local_image"].to(device)
        text = batch["text_tokens"].to(device)
        label = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(main_image=main, local_image=local, text_tokens=text)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            main = batch["main_image"].to(device)
            local = batch["local_image"].to(device)
            text = batch["text_tokens"].to(device)
            label = batch["label"].to(device)
            logits = model(main_image=main, local_image=local, text_tokens=text)
            loss = criterion(logits, label)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    acc = 100.0 * correct / total if total > 0 else 0.0
    return total_loss / len(loader), acc, correct, total


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print("Device:", device, f"({n_gpu} GPUs available)")
    print(f"Warmup: {cfg.warmup_epochs} epochs (lr 0.01x -> 1x)")
    print("LoRA mode:", cfg.use_vision_lora)
    if cfg.use_vision_lora:
        print(f"  LoRA r: {cfg.lora_r}, alpha: {cfg.lora_alpha}, dropout: {cfg.lora_dropout}")

    checkpoint_dir = cfg.checkpoint_dir
    log_dir = cfg.log_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    train_output_path = os.path.join(checkpoint_dir, "train_output.txt")
    open(train_output_path, "w").close()
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Log dir: {log_dir}")
    print(f"Epochs with saved checkpoints will be logged to: {train_output_path}")

    train_ds = CombinedXrayDataset(
        text_data_dir=cfg.text_data_dir,
        global_image_root=cfg.global_image_root,
        local_image_root=cfg.local_image_root,
        local_image_subdir=cfg.local_image_subdir,
        image_size=cfg.image_size,
        is_train=True,
        class_names=cfg.class_names,
    )
    val_ds = CombinedXrayDataset(
        text_data_dir=cfg.text_data_dir,
        global_image_root=cfg.global_image_root,
        local_image_root=cfg.local_image_root,
        local_image_subdir=cfg.local_image_subdir,
        image_size=cfg.image_size,
        is_train=False,
        class_names=cfg.class_names,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

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
        use_vision_lora=cfg.use_vision_lora,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
    ).to(device)

    if n_gpu > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel on {n_gpu} GPUs")

    alpha = torch.tensor(cfg.focal_alpha, dtype=torch.float32, device=device) if cfg.focal_alpha else None
    criterion = FocalLoss(gamma=cfg.focal_gamma, alpha=alpha)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # Warmup + CosineAnnealing
    warmup_epochs = min(cfg.warmup_epochs, cfg.num_epochs)
    if warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_epochs - warmup_epochs
        )
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [warmup_epochs])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    writer = SummaryWriter(log_dir=log_dir)

    # 是否从最新检查点继续训练（按时间最新的 best_epoch_*.pth）
    start_epoch = 1
    best_val_acc = 0.0
    resume_pattern = os.path.join(checkpoint_dir, "best_epoch_*.pth")
    resume_ckpts = glob.glob(resume_pattern)
    if resume_ckpts:
        latest_ckpt = max(resume_ckpts, key=os.path.getctime)
        print("Resuming from latest checkpoint:", latest_ckpt)
        ckpt = torch.load(latest_ckpt, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        # DataParallel 期望 module. 前缀；若 checkpoint 无此前缀则补充
        if n_gpu > 1 and not any(k.startswith("module.") for k in state.keys()):
            state = {f"module.{k}": v for k, v in state.items()}
        elif n_gpu <= 1 and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: failed to load optimizer state, reinit optimizer. {e}")
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                print(f"Warning: failed to load scheduler state, reinit scheduler. {e}")
        best_val_acc = ckpt.get("val_acc", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%")

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{cfg.num_epochs} ---")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_correct, val_total = validate(model, val_loader, criterion, device)
        scheduler.step()

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Acc", train_acc, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Acc", val_acc, epoch)
        
        if cfg.use_vision_lora:
            writer.add_scalar("LoRA/r", cfg.lora_r, epoch)
            writer.add_scalar("LoRA/alpha", cfg.lora_alpha, epoch)

        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.2f}%  (correct={val_correct}/{val_total})")

        if val_acc > best_val_acc or epoch == cfg.num_epochs:
            best_val_acc = val_acc
            path = os.path.join(checkpoint_dir, f"best_epoch_{epoch}.pth")
            # 保存时去掉 module. 前缀，便于单卡 test 加载
            state_to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": state_to_save,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "hidden_dim": cfg.hidden_dim,
                    "use_vision_lora": cfg.use_vision_lora,
                    "lora_r": cfg.lora_r,
                    "lora_alpha": cfg.lora_alpha,
                    "lora_dropout": cfg.lora_dropout,
                },
                path,
            )
            print(f"Saved best to {path}")
            with open(train_output_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- Epoch {epoch}/{cfg.num_epochs} ---\n")
                f.write(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%\n")
                f.write(f"Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.2f}%  (correct={val_correct}/{val_total})\n")
                f.write(f"LoRA mode: {cfg.use_vision_lora}\n")
                if cfg.use_vision_lora:
                    f.write(f"  r: {cfg.lora_r}, alpha: {cfg.lora_alpha}, dropout: {cfg.lora_dropout}\n")
                f.write(f"Saved best to {path}\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    writer.close()
    print("Training done.")


if __name__ == "__main__":
    main()