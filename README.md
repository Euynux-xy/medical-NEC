# 多模态 X 光分类模型

基于 CLIP 的 X 光图像多模态分类：主图 + 局部图 + 文本 + DPP 选择 + Transformer 选头。

## 环境

```bash
pip install torch torchvision transformers peft scikit-learn tqdm tensorboard
```

## 数据格式

- `text_data_dir`: `train_*.json` / `test_*.json`
- `global_image_root`: 主图目录
- `local_image_root`: 局部图目录

## 训练

```bash
python train.py
```

在 `config.py` 中修改数据路径等配置。

## 测试

```bash
python test.py
```
