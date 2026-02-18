from cvae_gan import  CVAE_GAN
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
from utils import MSC, SNV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GAN_trainer():
    def __init__(self, alpha=0.1, gamma=20, lr=1e-5, batch_size=128, data_length=256, num_conditions=6):

        """
            Trainer para o CVAE-GAN

            Atributos:
                alpha (float): fator de escala para o learning rate do discriminador
                gamma (float): fator de escala para a loss de suavidade (derivada primeira)
                lr (float): learning rate
                batch_size (int): tamanho do batch
                data_length (int): tamanho do espectro 
                num_conditions (int): numero de condicoes (features, soja e farelo)
            
            O que ja foi testado (evitar repeticao):
                - Gammas entre 5 e 25 (20 foi o melhor)
                - Alphas entre 0.05 e 0.2 (0.1 foi o melhor)
                - Calcular a derivada, comparando com os espectros reais (ficou pior)
                - Antes eram gerados dados sobre espectros nunca vistos pelo modelo, porem, ele gerava poucos e gerava espectros ruins
                - Foi testado no range(6) valores entre 1 e 10, 6 foi o que gerou melhor
                - Testei tambem inserir uma quantidade random de noise entre 10% e 30%, porem, ficou pior, 15% fixo foi o melhor
                - Testei usar o y_test original, porem, ficou pior, adicionar ruido foi melhor de 10% foi o melhor
                - Testei usar o y_test com ruido de 10% e 20%, 10% foi o melhor

                - Tecnicas de pre-processamento testadas:
                    - SNV: Ficou bom e segue sendo usado
                    - MSC: Ficou bom e segue sendo usado
                    - BC (baseline correction): Nao funcionou
                    - DT (de trending): Nao funcionou
        """

        super(GAN_trainer, self).__init__()
        self.BATCH_SIZE = batch_size
        self.DATA_LENGTH = data_length
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.NUM_CONDITIONS = num_conditions

        self.model = CVAE_GAN(DATA_LENGTH=self.DATA_LENGTH, NUM_CONDITIONS=self.NUM_CONDITIONS).to(device)

    def diff_loss(self, x):
        dx = x[:, :, 1:] - x[:, :, :-1]
        return torch.mean(dx ** 2)

    def train(self, x_data, y_data, epochs=300, pre_trained_path="data/models/cvae_gan/aqueousGlucose/cvae_gan_split:17_chkpt.pth", pre_training=False, times=6):

        """
            Treina o CVAE-GAN e gera espectros sintéticos

            Atributos:
                x_data (array): espectros de treino
                y_data (array): labels de treino
                epochs (int): numero de epocas para treinar
                pre_trained_path (str): caminho para o modelo pre treinado

            Usamos o SNV para pre processamento de dados e o MSC para pos processamento
        """
        
        if isinstance(x_data, pd.DataFrame):
            x_data = x_data.to_numpy()
        if isinstance(y_data, pd.DataFrame):
            y_data = y_data.to_numpy()

        scalerx = StandardScaler()
        scalery = StandardScaler()
        snv = SNV()

        x_data = scalerx.fit_transform(x_data)
        y_data = scalery.fit_transform(y_data)
        snv.fit_transform(x_data)

        msc = MSC()

        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

        train_loader = DataLoader(TensorDataset(
            torch.tensor(x_train).float().unsqueeze(1).to(device),
            torch.tensor(y_train).float().to(device)
        ), batch_size=self.BATCH_SIZE, shuffle=True, drop_last=True)

        val_loader = DataLoader(TensorDataset(
            torch.tensor(x_val).float().unsqueeze(1).to(device),
            torch.tensor(y_val).float().to(device)
        ), batch_size=self.BATCH_SIZE, shuffle=False, drop_last=False)

        criterion = nn.BCEWithLogitsLoss().to(device)
        optim_E = torch.optim.Adam(self.model.encoder.parameters(), lr=self.lr, weight_decay=1e-5)
        optim_D = torch.optim.Adam(self.model.decoder.parameters(), lr=self.lr, weight_decay=1e-5)
        optim_Dis = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.lr * self.alpha, weight_decay=1e-6)

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

            if (gen_loss + avg_val_loss * 0.2) < best_loss:
                best_loss = (gen_loss.item() + avg_val_loss * 0.2)
                best_model = copy.deepcopy(self.model.state_dict())

            if epoch % 10 == 0:
                print(f"[{epoch}/{epochs}]  Loss_D: {dis_loss.item():.4f}  Loss_G: {gen_loss.item():.4f}  Train_Rec_loss: {rec_loss.item():.6f}  Val_Rec_loss: {avg_val_loss:.6f}")

        self.model.load_state_dict(best_model)

        x_gen = x_train
        y_gen = y_train

        for i in range(times):
        
            sigmaX = np.std(x_train, axis=0)
            noiseX = np.random.normal(loc=0.0, scale=sigmaX, size=x_train.shape)
            x_train_aug = x_train + 0.15 * noiseX

            sigmaY = np.std(y_train, axis=0)
            noiseY = np.random.normal(loc=0.0, scale=sigmaY, size=y_train.shape)
            y_train_aug = y_train + 0.15 * noiseY

            x_gen = np.concatenate((x_train_aug, x_gen), axis=0)
            y_gen = np.concatenate((y_train_aug, y_gen), axis=0)

        print(f"Gerando {len(x_gen)} amostras sintéticas")
        x_aug, y_aug = self.gen_data(x_gen, y_gen)

        x_train = snv.inverse_transform(x_train)
        x_aug = snv.inverse_transform(x_aug)
        x_train = scalerx.inverse_transform(x_train)
        x_aug = scalerx.inverse_transform(x_aug)
        y_aug = scalery.inverse_transform(y_aug)

        msc.fit(x_train)
        x_aug = msc.transform(x_aug)

        if pre_training:
            run_name = f'data/models/cvae_gan/aqueousGlucose/cvae_gan_split:{self.SPLIT}.pth'
            torch.save(self.model.state_dict(), run_name)

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