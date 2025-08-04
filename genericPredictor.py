import torch
from torch import nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GenericNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(GenericNet, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.seq1(x)

    def fit(self, X_train, y_train, epochs=200, learning_rate=0.001):
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train)
        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(1).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.to(device)

        best_loss = float('inf')  # Inicializa com infinito
        best_state = None         # Para armazenar o melhor estado do modelo

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            outputs = self(X_train_tensor).squeeze()
            loss = criterion(outputs, y_train_tensor)

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = self.state_dict()  # Salva o melhor estado

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        if best_state is not None:
            self.load_state_dict(best_state)

        print(f"\nMelhor loss durante o treinamento: {best_loss:.4f}")


    def predict(self, X_test):
        scaler = StandardScaler()
        X_test_np = scaler.fit_transform(X_test)
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).unsqueeze(1).to(device)

        self.eval()
        self.to(device)

        with torch.no_grad():
            outputs = self(X_test_tensor).squeeze()
        return outputs.cpu().numpy()  