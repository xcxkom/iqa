import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision.io import decode_image


class Koniq10kDataset(Dataset):
    def __init__(self, img_dir, indicators, scores, split="train", train_ratio=0.6, val_ratio=0.2, random_state=42):
        self.img_dir = img_dir

        df_indicators = pd.read_csv(indicators)
        df_scores = pd.read_csv(scores)

        df_indicators["image_name"] = df_indicators["image_id"].astype(str) + ".jpg"
        data = pd.merge(df_indicators, df_scores, on="image_name", how="inner").reset_index(drop=True)

        if split in ["train", "val", "test"]:
            train_data, temp_data = train_test_split(data, test_size=1-train_ratio, random_state=random_state)
            val_data, test_data = train_test_split(temp_data, test_size=1-val_ratio/(1-train_ratio), random_state=random_state)

            if split == "train":
                self.data = train_data
            elif split == "val":
                self.data = val_data
            elif split == "test":
                self.data = test_data

        else:
            self.data = data

        scaler = StandardScaler()
        self.data["MOS"] = scaler.fit_transform(self.data[["MOS"]])

        self.indicators_columns = ["brightness", "contrast", "colorfulness", "sharpness", "quality_factor", "bitrate", "hxw", "deep_feature"]
        self.scores_columns = ["MOS"]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]

        img_path = os.path.join(self.img_dir, row["image_name"])
        image = decode_image(img_path)
        indicators = row[self.indicators_columns]
        scores = row[self.scores_columns]

        indicators = torch.from_numpy(indicators.values.astype(np.float32))
        scores = torch.tensor(scores.values[0], dtype=torch.float32) 

        return image, scores, indicators


class LivecDataset(Dataset):
    def __init__(self, img_dir, scores, target_size=(384, 512)):
        self.img_dir = img_dir

        self.scores = pd.read_csv(scores)

        scaler = StandardScaler()
        self.scores["MOS"] = scaler.fit_transform(self.scores[["MOS"]])

        self.target_size = target_size

    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, index):
        row = self.scores.iloc[index]

        image_path = os.path.join(self.img_dir, row["image_name"])

        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()

        score = row["MOS"]

        return image, score