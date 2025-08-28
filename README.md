# pytorch-metric-learning-template

基于 [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) 开源工具，实现了包括模型训练、模型验证、模型推理的相关代码。

## 目录
- [项目介绍](#项目介绍)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
- [模型训练](#模型训练)
- [模型推理](#模型推理)
- [特征抽取](#特征抽取)
- [可视化](#可视化)
- [致谢](#致谢)

## 项目介绍

本项目是一个基于 PyTorch 和 pytorch-metric-learning 的度量学习模板项目，旨在提供一个完整的图像特征提取和相似度匹配解决方案。项目支持多种深度学习模型和损失函数，可用于图像检索、人脸识别、签名验证等任务。

主要功能：
- 支持多种 CNN 架构（ResNet、DenseNet、MobileNet等）
- 集成多种度量学习损失函数（TripletMarginLoss、SupervisedContrastiveLoss、CircleLoss等）
- 提供完整的训练、验证、推理流程
- 支持特征提取和可视化
- 支持自定义数据集

## 项目结构

```
.
├── config/                 # 配置文件目录
│   └── embedding.yaml      # 训练配置文件
├── models/                 # 模型定义目录
│   ├── backbone.py         # 基础模型
│   ├── backbone_attention.py # 带注意力机制的模型
│   └── ...                 # 其他模型组件
├── utils/                  # 工具函数目录
│   ├── data_loader.py      # 数据加载器
│   ├── trainer.py          # 训练器
│   └── ...                 # 其他工具函数
├── pretrained_models/      # 预训练模型保存目录
├── logs/                   # 训练日志和模型保存目录
├── train.py                # 训练脚本
├── model_inference.py      # 模型推理脚本
├── feature_extraction.py   # 特征抽取脚本
└── visualizer.py           # 可视化脚本
```

## 环境依赖

- Python 3.x
- PyTorch 1.13.0
- torchvision 0.14.0
- pytorch-metric-learning 2.4.1
- faiss-gpu 1.7.2
- numpy 1.24.4
- opencv-python 4.7.0.72
- scikit-learn 1.3.2
- matplotlib 3.7.1
- umap-learn 0.5.5

安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始

1. 准备数据集，按照以下结构组织：
```
dataset/
├── class1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── class2/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── ...
```

2. 修改配置文件 `config/embedding.yaml` 中的数据路径和其他参数

3. 运行训练脚本：
```bash
python train.py --cfg config/embedding.yaml
```

## 模型训练

训练脚本支持以下功能：
- 多种 CNN 架构（ResNet、DenseNet、MobileNet 等）
- 多种度量学习损失函数
- 数据增强（随机裁剪、翻转、颜色抖动等）
- Mixup 数据增强
- XBM（Cross-Batch Memory）训练策略
- 学习率调度

配置训练参数在 `config/embedding.yaml` 文件中，主要参数包括：
- `model_name`: 使用的模型架构
- `train_dataset_dir`: 训练数据路径
- `save_dir`: 模型保存路径
- `out_dimension`: 输出特征维度
- `xbm_enable`: 是否启用XBM
- `mixup_enable`: 是否启用Mixup

运行训练：
```bash
python train.py --cfg config/embedding.yaml
```

## 模型推理

使用训练好的模型进行推理，计算查询图像与数据库图像的相似度。

运行推理脚本：
```bash
python model_inference.py
```

## 特征抽取

从图像中提取特征向量，用于后续的相似度计算或聚类任务。

运行特征抽取脚本：
```bash
python feature_extraction.py
```

## 可视化

对提取的特征进行降维和可视化，帮助理解模型的特征分布。

运行可视化脚本：
```bash
python visualizer.py
```

## 致谢

- [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
- [PyTorch](https://pytorch.org/)
- [torchvision](https://github.com/pytorch/vision)