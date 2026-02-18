import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

class BeerDataset(Dataset):
    def __init__(self, dataframe, scaler_x=None, scaler_y=None, fit=False):
        self.x = dataframe.iloc[:, 1:].values.astype(np.float32)
        self.y = dataframe.iloc[:, 0].values.astype(np.float32).reshape(-1, 1)

        self.scaler_x = scaler_x if scaler_x is not None else StandardScaler()
        self.scaler_y = scaler_y if scaler_y is not None else StandardScaler()

        if fit:
            self.x = self.scaler_x.fit_transform(self.x)
            self.y = self.scaler_y.fit_transform(self.y)
        else:
            self.x = self.scaler_x.transform(self.x)
            self.y = self.scaler_y.transform(self.y)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]