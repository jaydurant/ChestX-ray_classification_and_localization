import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFile
from sklearn.preprocessing import MultiLabelBinarizer
from models.selected_labels import selected_labels

class XrayDataset(Dataset):
    def __init__(self,data_dir, data, transform=None):
        
        self.root_dir = data_dir
        self.dataset = data
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img_path = os.path.join(self.root_dir, self.dataset.iloc[idx, 0])
        sample_labels = self.dataset.iloc[idx, 1]
        print(sample_labels)
        sample = Image.open(img_path)

        if self.transform:
            sample = self.transform(sample)

        return sample, sample_labels



