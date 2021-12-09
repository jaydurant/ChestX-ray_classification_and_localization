import os
import torch
from torch.utils.data import Dataset, Dataloader
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFile
from sklearn.preprocessing import MultiLabelBinarizer
from models.selected_labels import selected_labels

class XrayDataset(Dataset):
    """[summary]"""
    def __init__(self, csv_file, root_dir, transform=None):
        """[summary]

        Args:
            csv_file ([type]): [description]
            root_dir ([type]): [description]
            labels_file ([type]): [description]
            transform ([type], optional): [description]. Defaults to None.
        """
        self.dataset = pd.read_csv(csv_file)
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([selected_labels])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
        sample_labels = self.dataset.iloc[idx, 1]

        labels = self.mlb.transform([sample_labels])[0]

        sample = Image.open(img_path)

        if self.transform:
            sample = self.transform(sample)

        return sample, labels



