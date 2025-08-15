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
from sklearn.cross_decomposition import PLSRegression

scalerx = StandardScaler()
scalery = StandardScaler()

x_data = pd.read_excel('data/ignore/Matriz_B51R002.xlsx', sheet_name='X')
y_data = pd.read_excel('data/ignore/Matriz_B51R002.xlsx', sheet_name='Y')
x_data = x_data.iloc[:, 1:]  # Select only the first column
y_data = y_data.iloc[:, 1:]  # Select only the first column

x_data = np.array(x_data.values)
y_data = np.array(y_data.values)

input_size = x_data.shape[1] - 1  

x_data = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
y_data = torch.tensor(y_data, dtype=torch.float32)

x_data = F.interpolate(x_data, size=256, mode='linear', align_corners=False)
x_data = x_data.squeeze(1)  

gan_trainer = GAN_trainer(split=0.25)

x_aug, y_aug = gan_trainer.train(x_data, y_data, epochs=800)


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

x_train = np.vstack((x_data, x_aug))
y_train = np.vstack((y_data, y_aug))

x_train = scalerx.fit_transform(x_train)
y_train = scalery.fit_transform(y_train)

model = PLSRegression(n_components=6)
model.fit(x_train, y_train)

result = model.predict(x_test)
print("PLS Regression Results:")
print("MAE:", mean_absolute_error(y_test, result))
print("MSE:", mean_squared_error(y_test, result))
print("R2 Score:", r2_score(y_test, result))
print("Explained Variance Score:", explained_variance_score(y_test, result))