import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import GenNet

BATCH_SIZE = 4
EPOCHS = 20
LR = 4e-4
WEIGHT_DECAY = 1e-4

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

model = GenNet(input_length=train_ds.x.shape[1])
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.NAdam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# Training
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f}  "
    )

model.eval()
test_loss = 0

all_preds = []
all_targets = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss = criterion(pred, y)
        test_loss += loss.item()

        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

test_loss /= len(test_loader)

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

mse = F.mse_loss(torch.tensor(all_preds), torch.tensor(all_targets))
mae = F.l1_loss(torch.tensor(all_preds), torch.tensor(all_targets))
rmse = torch.sqrt(mse)

print(f"Test Loss (criterion): {test_loss:.6f}")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")