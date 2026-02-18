import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from genNetTrainer import GenNetTrainer
from cvae_gan_trainer import GAN_trainer

BATCH_SIZE = 4
EPOCHS = 20

torch.manual_seed(42)
np.random.seed(42)

df = pd.read_csv('data/beer.csv')

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

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

train_ds = BeerDataset(train_df, fit=True)
test_ds = BeerDataset(
    test_df,
    scaler_x=train_ds.scaler_x,
    scaler_y=train_ds.scaler_y,
    fit=False
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()

# trainer = GenNetTrainer(device, criterion)

# trainer.train(train_loader, EPOCHS)
# trainer.test(test_loader)

x, y = df.iloc[:, 1:], df.iloc[:, 0:1]
print(x)

gan_trainer = GAN_trainer(
                batch_size=4, 
                data_length=x.shape[1], 
                num_conditions=1)

x_new, y_new = gan_trainer.train(x, y, times=1)