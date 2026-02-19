import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from genNet.genNetTrainer import GenNetTrainer
from utils.beerDS import BeerDataset
from cvaegan.cvae_gan_trainer import GAN_trainer
from cvaegan.filter import Filter

BATCH_SIZE = 4
EPOCHS = 20

torch.manual_seed(42)
np.random.seed(42)
df = pd.read_csv('data/beerNir/beer.csv')

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

x_train = train_df.iloc[:, 1:]
y_train = train_df.iloc[:, 0:1]

gan_trainer = GAN_trainer(
    batch_size=4,
    data_length=x_train.shape[1],
    num_conditions=1
)

x_new, y_new = gan_trainer.train(x_train, y_train, times=6, epochs=100)

filter = Filter(split=0.05)
x_filter, y_filter = filter.filter(x_new, y_new, x_train)

fig, ax = plt.subplots()

for sample in x_filter:
    ax.plot(np.arange(len(sample)), sample)

plt.savefig('figs/gan.png')
plt.close(fig)

synthetic_df = pd.concat(
    [
        pd.DataFrame(y_filter, columns=y_train.columns),
        pd.DataFrame(x_filter, columns=x_train.columns)
    ],
    axis=1
)

train_df = pd.concat(
    [train_df, synthetic_df],
    axis=0
).reset_index(drop=True)

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

trainer = GenNetTrainer(device, criterion)

trainer.train(train_loader, EPOCHS)
trainer.test(test_loader)
