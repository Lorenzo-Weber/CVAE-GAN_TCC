from genNet.model import GenNet
import torch
import torch.nn.functional as F
import numpy as np

class GenNetTrainer():

  def __init__(self, device, criterion):
    self.device = device
    self.criterion = criterion
    self.model = GenNet()
    self.model.to(device)
    
    self.optimizer = torch.optim.NAdam(
      self.model.parameters(),
      lr=4e-4,
      weight_decay=1e-4
    )

  def train(self, dataloader, epochs=25):

    for epoch in range(epochs):
      self.model.train()
      train_loss = 0

      for x, y in dataloader:
          x = x.to(self.device)
          y = y.to(self.device)

          self.optimizer.zero_grad()

          pred = self.model(x)
          loss = self.criterion(pred, y)

          loss.backward()
          self.optimizer.step()

          train_loss += loss.item()

      train_loss /= len(dataloader)

      print(
          f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_loss:.4f}  "
      )
  def test(self, dataloader):
    self.model.eval()
    test_loss = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)

            loss = self.criterion(pred, y)
            test_loss += loss.item()

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    test_loss /= len(dataloader)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mse = F.mse_loss(torch.tensor(all_preds), torch.tensor(all_targets))
    mae = F.l1_loss(torch.tensor(all_preds), torch.tensor(all_targets))
    rmse = torch.sqrt(mse)

    print(f"Test Loss (criterion): {test_loss:.6f}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")