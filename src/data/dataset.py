import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.io import decode_image


class IQADataset(Dataset):
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

        self.indicators_columns = ["brightness", "contrast", "colorfulness", "sharpness", "quality_factor", "bitrate", "hxw", "deep_feature"]
        # self.scores_columns = ["c1", "c2", "c3", "c4", "c5", "c_total", "MOS", "SD", "MOS_zscore"]
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
        # scores = torch.from_numpy(scores.values.astype(np.float32))
        scores = torch.tensor(scores.values[0], dtype=torch.float32) 

        return image, scores, indicators
