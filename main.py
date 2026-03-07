import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data.dataset import IQADataset
from src.models.df_iqa_cnn import create_iqa_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "/home/xoxkom/projects/IQA/models/df_iqa_cnn"

IMG_DIR = "/home/xoxkom/projects/IQA/datasets/raw/512x384"
INDICATORS_PATH = "/home/xoxkom/projects/IQA/datasets/raw/koniq10k_indicators.csv"
SCORES_PATH = "/home/xoxkom/projects/IQA/datasets/raw/koniq10k_scores_and_distributions.csv"

LR = 0.001
BATCH_SIZE = 4
EPOCHS = 10


if __name__ == "__main__":
    print(f"---> 使用设备：{DEVICE}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "df_iqa_cnn")
    print(f"---> 模型将被保存在：{save_path}/")

    train_dataset = IQADataset(IMG_DIR, INDICATORS_PATH, SCORES_PATH, split="train")
    val_dataset = IQADataset(IMG_DIR, INDICATORS_PATH, SCORES_PATH, split="val")
    test_dataset = IQADataset(IMG_DIR, INDICATORS_PATH, SCORES_PATH, split="test")
    print(f"---> 训练集大小：{len(train_dataset)}")
    print(f"---> 验证集大小：{len(val_dataset)}")
    print(f"---> 测试集大小：{len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for model_name in ["vgg16"]:
        model = create_iqa_model(model_name)
        model = model.to(DEVICE)

        loss_func = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

        print(f"------> 开始训练：{model_name}")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0

            for inputs, labels, _ in tqdm(train_loader, desc="Train", leave=False):
                inputs = inputs.float() / 255.
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
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
                for inputs, labels, _ in tqdm(val_loader, desc="Val", leave=False):
                    inputs = inputs.float() / 255.
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    labels = labels.unsqueeze(-1)

                    outputs = model(inputs)

                    loss = loss_func(outputs, labels)

                    total_loss += loss.item() * inputs.size(0)
                    val_loss = total_loss / len(val_loader.dataset)

            print(f"Epoch [{epoch}/{EPOCHS}]: Train Loss {train_loss:4f}, Val Loss {val_loss:4f}")

        
        save_file = os.path.join(SAVE_DIR, f"{model_name}.pth")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        torch.save(model.state_dict(), save_file)
        
