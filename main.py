from cvaegan import GAN_trainer
from genericPredictor import GenericNet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

data = pd.read_csv('data/aqueousGlucose/aqueousglucose.csv')
data = data.dropna()

x_data = data.iloc[:, 6:]
x_data = torch.tensor(x_data.values, dtype=torch.float32).unsqueeze(1)
x_data = F.interpolate(x_data, size=256, mode='bilinear', align_corners=False)
y_data = data.iloc[:, :6]

input_size = data.shape[1] - 1  # Assuming the last column is the target variable
print(input_size)
gan_trainer = GAN_trainer(data_length=input_size)
net = GenericNet(input_size=input_size, output_size=6)

x_aug, y_aug = gan_trainer.train(x_data, y_data, epochs=100)

x_train = torch.cat((x_data, x_aug), dim=0)
y_train = torch.cat((y_data, y_aug), dim=0)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

net.fit(x_train, y_train, epochs=1000, learning_rate=0.0001)
net.predict(x_test)

