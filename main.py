from cvaegan import GAN_trainer
from genericPredictor import GenericNet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import numpy as np

data = pd.read_csv('data/aqueousGlucose/aqueousglucose.csv')
data = data.dropna()

scalerx = StandardScaler()
scalery = StandardScaler()

x_data = pd.read_excel('data/ignore/Matriz_B51R002.xlsx', sheet_name='X')
x_data = x_data.iloc[:, 1:]

y_data = pd.read_excel('data/ignore/Matriz_B51R002.xlsx', sheet_name='Y')
y_data = y_data.iloc[:, 1:]

input_size = data.shape[1] - 1  # Assuming the last column is the target variable

x_data = torch.tensor(x_data.values, dtype=torch.float32).unsqueeze(1)
y_data = torch.tensor(y_data.values, dtype=torch.float32)

x_data = F.interpolate(x_data, size=256, mode='linear', align_corners=False)
x_data = x_data.squeeze(1)

gan_trainer = GAN_trainer(split=0.2)
net = GenericNet(input_size=input_size, output_size=6)

x_aug, y_aug = gan_trainer.train(x_data, y_data, epochs=200)

x_aug = torch.tensor(x_aug, dtype=torch.float32)
y_aug = torch.tensor(y_aug, dtype=torch.float32)

x_train = torch.cat((x_data, x_aug), dim=0)
y_train = torch.cat((y_data, y_aug), dim=0)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

x_train = scalerx.fit_transform(x_train)
y_train = scalery.fit_transform(y_train)

net.fit(x_train, y_train, epochs=5000, learning_rate=0.0001)


