import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Funções utilitárias
def get_out_shape(model: nn.ModuleList | nn.Sequential, input_shape: tuple[int, ...]) -> tuple:
    input = torch.randn(input_shape, device=next(model.parameters()).device)
    return model(input).shape

def get_out_features(model: nn.ModuleList | nn.Sequential, input_shape: tuple[int, ...]) -> int:
    size = model(torch.rand(*(input_shape))).data.shape
    return int(np.prod(list(size)))

class GenericNet(nn.Module):
    def __init__(self, expected_input: int, model_output: int):
        super(GenericNet, self).__init__()

        # Convoluções
        self.convolutions = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=24, stride=2, padding=4),
            nn.ELU(),
            nn.Conv1d(8, 24, kernel_size=8, stride=2, padding=3),
            nn.ELU(),
            nn.Flatten(),
        )

        # Calcula dinamicamente o número de features de saída
        out_features = get_out_features(self.convolutions, (1, 1, expected_input))

        # Lineares
        self.linears = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(out_features, 672),
            nn.ELU(),
            nn.Linear(672, 336),
            nn.ELU(),
            nn.Linear(336, model_output),
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = self.linears(x)
        return x

    def fit(self, x_train, y_train, epochs=10, batch_size=32, lr=1e-3):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                loss = loss_fn(self(xb), yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.6f}")
