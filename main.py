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

x_data = torch.tensor(x_data.values, dtype=torch.float32)
y_data = torch.tensor(y_data.values, dtype=torch.float32)

# x_data = F.interpolate(x_data, size=256, mode='linear', align_corners=False)
# x_data = x_data.squeeze(1)

net = GenericNet(input_size=input_size, output_size=6)
gan_trainer = GAN_trainer(split=0.17)

x_aug, y_aug = gan_trainer.train(x_data, y_data, epochs=200)

x_aug = torch.tensor(x_aug, dtype=torch.float32)
y_aug = torch.tensor(y_aug, dtype=torch.float32)

x_train = torch.cat((x_data, x_aug), dim=0)
y_train = torch.cat((y_data, y_aug), dim=0)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

x_train = scalerx.fit_transform(x_train)
y_train = scalery.fit_transform(y_train)

net.fit(x_train, y_train, epochs=5000, learning_rate=0.0001)

# Avaliação
x_test_scaled = scalerx.transform(x_test)
y_test_scaled = scalery.transform(y_test)

# Previsão
y_pred_scaled = net.predict(x_test_scaled)

# Inverter a escala
y_pred = scalery.inverse_transform(y_pred_scaled)
y_true = scalery.inverse_transform(y_test_scaled)

# Métricas para todos os alvos
mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred, multioutput='raw_values')
evs = explained_variance_score(y_true, y_pred, multioutput='raw_values')

colunas = ['Proteína', 'Óleo', 'Umidade', 'Fibra', 'Cinzas', 'Densidade']

print("\n Métricas por variável:")
for i, col in enumerate(colunas):
    print(f"{col}:")
    print(f"  MAE:  {mae[i]:.4f}")
    print(f"  RMSE: {rmse[i]:.4f}")
    print(f"  R²:   {r2[i]:.4f}")
    print(f"  EVS:  {evs[i]:.4f}")

# Foco especial na Proteína
print("\n Foco na variável: Proteína")
print(f"MAE (Proteína):  {mae[0]:.4f}")
print(f"RMSE (Proteína): {rmse[0]:.4f}")
print(f"R² (Proteína):   {r2[0]:.4f}")
print(f"EVS (Proteína):  {evs[0]:.4f}")
