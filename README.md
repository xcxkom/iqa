# IQA 图像质量评估项目

本项目用于图像质量评估（IQA）模型训练与测试，基于 PyTorch 实现。

## 项目结构

主要目录与文件说明如下：

```
IQA/
├── main.py                 # 程序入口
├── config.yaml             # 运行与训练配置
├── pyproject.toml          # 依赖定义
├── datasets/
│   └── raw/
│       ├── koniq10k/       # Koniq10k 数据集
│       │   ├── 512x384/    # 图像目录
│       │   ├── koniq10k_indicators.csv
│       │   └── koniq10k_scores_and_distributions.csv
│       └── livec/          # LIVEC 数据集
│           ├── images/     # 图像目录
│           └── scores.csv
├── src/
│   ├── data/               # 数据集读取与划分
│   ├── models/             # 模型构建与训练逻辑
│   └── utils/              # 日志与计时工具
├── logs/                   # 日志输出目录
└── models/                 # 模型权重保存目录
```

## 运行配置

配置文件：`config.yaml`

核心参数：

- device: 运行设备，支持 cuda、mps、cpu
- koniq10k_img_dir: Koniq10k 图像目录路径
- koniq10k_indicators_path: Koniq10k 指标 CSV 路径
- koniq10k_scores_path: Koniq10k 评分 CSV 路径
- livec_img_dir: LIVEC 图像目录路径
- livec_scores_path: LIVEC 评分 CSV 路径
- save_dir: 模型保存根目录
- log_dir: 日志目录
- model: 模型名称（df_iqa_cnn）
- mode: 运行模式（train 或 test）
- models_dir: 测试时加载的模型目录
- seed: 随机种子
- num_workers: 数据加载线程数
- lr: 学习率
- batch_size: 批大小
- epochs: 训练轮数

## 运行方式

1. 安装依赖
```shell
uv sync
```

2. 训练模型
```shell
# 修改 config.yaml 中 mode 为 train
uv run python main.py
```

3. 测试模型
```shell
# 修改 config.yaml 中 mode 为 test，并指定 models_dir
uv run python main.py
```