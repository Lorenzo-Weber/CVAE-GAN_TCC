from model import CVAE_GAN
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.neighbors import LocalOutlierFactor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GAN_trainer():
    def __init__(self, alpha=0.1, gamma=20, lr=1e-5, split = 0.17, batch_size=64, data_length=256, pre_training=False):
        super(GAN_trainer, self).__init__()

        self.BATCH_SIZE = batch_size
        self.DATA_LENGTH = data_length
        self.SPLIT = split
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.PRE_TRAINING = pre_training

        self.model = CVAE_GAN(DATA_LENGTH=self.DATA_LENGTH).to(device)
    
    def load_data(self, x_train, y_train):

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=self.SPLIT, random_state=42)

        if type(x_train) is pd.DataFrame:
            x_train = x_train.iloc[:, :self.DATA_LENGTH].values.astype(np.float32)

        x_train_tensor = torch.tensor(x_train).float().unsqueeze(1).to(device)
        y_train_tensor = torch.tensor(y_train).float().to(device)

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        dataset = TensorDataset(x_train_tensor, y_train_tensor)
        data_loader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True, drop_last=True)
        return data_loader, x_test, y_test
    
    def load_wheights(self, path):
        if os.path.exists(path):
            pretrained_dict = torch.load(path)
        filtered_dict = {k: v for k, v in pretrained_dict.items() if not (
            k.startswith('encoder.input') or k.startswith('decoder.input') or k.startswith('discriminator.input') or k.startswith('decoder.reconstruction_head')
        )}

        print(f'Carregando pesos do modelo {path}')
        
        self.model.load_state_dict(filtered_dict, strict=False)
        self.model.encoder.sequential[0:7].requires_grad_(False)
        self.model.decoder.sequential[0:8].requires_grad_(False)

    def diff_loss(self, x):
        dx = x[:, :, 1:] - x[:, :, :-1]
        return torch.mean(dx ** 2)

    def train(self, x_data, y_data, epochs=300, pre_trained_path="data/pre-train/aqueousGlucose/cvae_gan_split:17_chkpt.pth", pre_training=False):

        scalerx = StandardScaler()
        scalery = StandardScaler()

        x_data = scalerx.fit_transform(x_data)
        y_data = scalery.fit_transform(y_data)

        # self.load_wheights(pre_trained_path)

        x_train_full, x_test, y_train_full, y_test = train_test_split(x_data, y_data, test_size=self.SPLIT, random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.3, random_state=42)
        train_loader = DataLoader(TensorDataset(
            torch.tensor(x_train).float().unsqueeze(1).to(device),
            torch.tensor(y_train).float().to(device)
        ), batch_size=self.BATCH_SIZE, shuffle=True, drop_last=True)

        val_loader = DataLoader(TensorDataset(
            torch.tensor(x_val).float().unsqueeze(1).to(device),
            torch.tensor(y_val).float().to(device)
        ), batch_size=self.BATCH_SIZE, shuffle=False, drop_last=False)

        criterion = nn.BCEWithLogitsLoss().to(device)
        optim_E = torch.optim.Adam(self.model.encoder.parameters(), lr=self.lr)
        optim_D = torch.optim.Adam(self.model.decoder.parameters(), lr=self.lr)
        optim_Dis = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.lr * self.alpha)

        best_loss = float('inf')
        best_model = None

        for epoch in range(epochs):
            self.model.train()
            for i, (data, labels) in enumerate(train_loader):

                bs = data.shape[0]
                ones_label = torch.ones(bs, 1, device=device)
                zeros_label = torch.zeros(bs, 1, device=device)

                datav = data.to(device)
                labelsv = labels.to(device)

                optim_Dis.zero_grad()

                output_real, _, _ = self.model.discriminator(datav, labelsv)
                errD_real = criterion(output_real, ones_label)

                mean, logvar, rec_enc = self.model(datav, labelsv)
                z_p = torch.randn(bs, self.DATA_LENGTH, device=device)
                x_p_tilda, _ = self.model.decoder(z_p, labelsv)

                output_rec, _, _ = self.model.discriminator(rec_enc.detach(), labelsv)
                errD_rec_enc = criterion(output_rec, zeros_label)

                output_prior, _, _ = self.model.discriminator(x_p_tilda.detach(), labelsv)
                errD_rec_noise = criterion(output_prior, zeros_label)

                _, _, output_labels = self.model.discriminator(rec_enc.detach(), labelsv)
                errD_labels = criterion(output_labels, labelsv)

                dis_loss = errD_real + errD_rec_enc + errD_rec_noise + errD_labels
                dis_loss.backward()
                optim_Dis.step()

                optim_E.zero_grad()
                optim_D.zero_grad()

                mean, logvar, rec_enc = self.model(datav, labelsv)
                z_p = torch.randn(bs, self.DATA_LENGTH, device=device)
                x_p_tilda, _ = self.model.decoder(z_p, labelsv)

                output_rec_gen, features_rec, _ = self.model.discriminator(rec_enc, labelsv)
                output_prior_gen, _, _ = self.model.discriminator(x_p_tilda, labelsv)
                gan_loss_g = criterion(output_rec_gen, ones_label) + criterion(output_prior_gen, ones_label)

                # _, features_real, _ = self.model.discriminator(datav, labelsv)

                # datav_diff = torch.diff(datav, dim=2)
                # rec_enc_diff = torch.diff(rec_enc, dim=2)

                rec_loss = self.gamma * self.diff_loss(rec_enc)

                prior_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

                _, _, output_labels = self.model.discriminator(rec_enc.detach(), labelsv)
                errD_labels_gen = criterion(output_labels, labelsv)

                gen_loss = prior_loss + gan_loss_g + errD_labels_gen + rec_loss 

                gen_loss.backward()
                optim_E.step()
                optim_D.step()

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for val_data, val_labels in val_loader:
                    val_mean, val_logvar, val_rec = self.model(val_data, val_labels)
                    val_data_diff = torch.diff(val_data, dim=2)
                    val_rec_diff = torch.diff(val_rec, dim=2)

                    val_rec_loss = self.gamma * criterion(val_rec_diff, val_data_diff) + criterion(val_rec, val_data)
                    val_losses.append(val_rec_loss.item())

            avg_val_loss = np.mean(val_losses)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model = copy.deepcopy(self.model.state_dict())

            # if epoch % 10 == 0:
            #     print(f"[{epoch}/{epochs}]  Loss_D: {dis_loss.item():.4f}  Loss_G: {gen_loss.item():.4f}  Train_Rec_loss: {rec_loss.item():.6f}  Val_Rec_loss: {avg_val_loss:.6f}")

        self.model.load_state_dict(best_model)

        x_aug, y_aug = self.gen_data(x_train, y_train)

        x_aug = scalerx.inverse_transform(x_aug)
        y_aug = scalery.inverse_transform(y_aug)
        x_test = scalerx.inverse_transform(x_test)
        x_train = scalerx.inverse_transform(x_train)

        self.plot_data(x_aug, x_train, split=self.SPLIT)

        return x_aug, y_aug
    
    def gen_data(self, x_test, y_test):

        samplesx = x_test

        sigmaY = np.std(y_test, axis=0)
        noise = np.random.normal(loc=0.0, scale=sigmaY, size=y_test.shape)
        samplesy = y_test + 0.1 * noise

        samples_torchx = torch.tensor(samplesx).unsqueeze(1).float().to(device)

        samples_torchy = torch.tensor(samplesy).float().to(device)

        z_mean, z_logvar = self.model.encoder(samples_torchx, samples_torchy)
        std = z_logvar.mul(0.5).exp_()
        epsilon = torch.randn(samples_torchx.size(0), self.DATA_LENGTH).to(device)
        z = z_mean + std * epsilon

        fake_samples, labels = self.model.decoder(z, samples_torchy)
        fake_samples = fake_samples.squeeze(1).cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        return fake_samples, labels
    
    def plot_data(self, x_aug, x_samples, split, num_samples=9):
        fig, axs = plt.subplots(3, 3, figsize=(25,10), constrained_layout=True)

        for idx in range(num_samples):
            row, col = divmod(idx, 3)
            ax = axs[row, col]
            ax.plot(x_aug[idx], label='Sintetic', color='green', linestyle='-')
            ax.plot(x_samples[idx], label='Original', color='blue', linestyle='-')
            ax.set_title(f'Sample {idx+1}')
            ax.legend()
        os.makedirs('data/plots/cvae_gan', exist_ok=True)
        fig.suptitle(f'Sintetic vs Original (split={split})', fontsize=16)
        fig.savefig(f'data/plots/cvae_gan/sintetic_vs_original_split_{split}.png')
        plt.close(fig)