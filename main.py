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
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

scalerx = StandardScaler()
scalery = StandardScaler()

x_data = pd.read_excel('data/ignore/Matriz_B51R002.xlsx', sheet_name='X')
y_data = pd.read_excel('data/ignore/Matriz_B51R002.xlsx', sheet_name='Y')
x_data = x_data.iloc[:, 1:]  
y_data = y_data.iloc[:, 1:]  

x_data = np.array(x_data.values)
y_data = np.array(y_data.values)

x_data = torch.tensor(x_data, dtype=torch.float32)
y_data = torch.tensor(y_data, dtype=torch.float32)

gan_trainer = GAN_trainer(split=0.25)

x_aug, y_aug = gan_trainer.train(x_data, y_data, epochs=800)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

x_train = np.vstack((x_data, x_aug))
y_train = np.vstack((y_data, y_aug))

x_train = x_train[:, :19]
x_test = x_test[:, :19]

x_train = scalerx.fit_transform(x_train)
y_train = scalery.fit_transform(y_train)
x_test = scalerx.transform(x_test)
y_test = scalery.transform(y_test)

# Define different PLS configurations
pls_models = [
    ('PLS-15', PLSRegression(n_components=15)),
]

# Train and evaluate each PLS model
for model_name, model in pls_models:
    print(f"\n{model_name} Results:")
    model.fit(x_train, y_train)
    
    # Predictions
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    
    # Testing metrics
    print("Testing Metrics:")
    print("MAE:", mean_absolute_error(y_test, test_pred))
    print("MSE:", mean_squared_error(y_test, test_pred))
    print("R2 Score:", r2_score(y_test, test_pred))
    print("Explained Variance Score:", explained_variance_score(y_test, test_pred))