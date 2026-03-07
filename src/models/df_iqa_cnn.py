import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from tqdm import tqdm
from torch.utils.data import DataLoader
from ..data.dataset import IQADataset


def create_regression_head(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 2048),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 1),
    )

def create_iqa_model(model_name):
    if model_name == "resnet50":
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = base_model.fc.in_features
        base_model.fc = create_regression_head(in_features)

    elif model_name == "densenet201":
        base_model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        in_features = base_model.classifier.in_features
        base_model.classifier = create_regression_head(in_features)

    elif model_name == "vgg16":
        base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = base_model.classifier[0].in_features
        base_model.classifier = create_regression_head(in_features)

    return base_model

def train(config, logger, save_path):
    logger.info("加载数据集...")
    img_dir = config["img_dir"]
    indicators_path = config["indicators_path"]
    scores_path = config["scores_path"]
    seed = config["seed"]
    train_dataset = IQADataset(img_dir, indicators_path, scores_path, split="train", random_state=seed)
    val_dataset = IQADataset(img_dir, indicators_path, scores_path, split="val", random_state=seed)
    test_dataset = IQADataset(img_dir, indicators_path, scores_path, split="test", random_state=seed)
    logger.info(f"训练集：{len(train_dataset)} 样本")
    logger.info(f"验证集：{len(val_dataset)} 样本")
    logger.info(f"测试集：{len(test_dataset)} 样本")

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    lr = config["lr"]
    epochs = config["epochs"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logger.info(f"随机种子：{seed} | 批次大小：{batch_size} | 线程数：{num_workers} | 学习率：{lr} | Epoch：{epochs} | \n")

    device = config["device"]

    for model_name in ["resnet50", "densenet201", "vgg16"]:
        logger.info(f"开始训练模型：{model_name}")

        model = create_iqa_model(model_name)
        model = model.to(device)

        loss_func = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in range(1, epochs + 1):
            model.train()

            total_loss = 0.0
            for inputs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train", leave=False):
                inputs = inputs.float() / 255.
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.unsqueeze(-1)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)

            train_loss = total_loss / len(train_loader.dataset)

            model.eval()

            with torch.no_grad():
                total_loss = 0.0
                for inputs, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} Val", leave=False):
                    inputs = inputs.float() / 255.
                    inputs, labels = inputs.to(device), labels.to(device)
                    labels = labels.unsqueeze(-1)

                    outputs = model(inputs)

                    loss = loss_func(outputs, labels)

                    total_loss += loss.item() * inputs.size(0)

                val_loss = total_loss / len(val_loader.dataset)

            logger.info(f"Epoch [{epoch:2d}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        
        save_file = os.path.join(save_path, f"{model_name}.pth")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        torch.save(model.state_dict(), save_file)
        logger.info(f"模型已保存：{save_file} \n")
