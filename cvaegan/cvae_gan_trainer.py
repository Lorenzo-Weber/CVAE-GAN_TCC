import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

from cvaegan.cvae_gan import CVAE_GAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAN_trainer():
    def __init__(
        self,
        alpha=0.1,
        gamma=20,
        lr=1e-5,
        batch_size=32,
        data_length=256,
        num_conditions=1,
        generator_layer_sizes=[128, 128],
        discriminator_layer_sizes=[128, 256, 256]
    ):

        self.BATCH_SIZE = batch_size
        self.DATA_LENGTH = data_length
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.NUM_CONDITIONS = num_conditions

        self.generator_layer_sizes = list(generator_layer_sizes)
        self.discriminator_layer_sizes = list(discriminator_layer_sizes)

        self.model = CVAE_GAN(
            DATA_LENGTH=self.DATA_LENGTH,
            NUM_CONDITIONS=self.NUM_CONDITIONS,
            generator_layer_sizes=self.generator_layer_sizes,
            discriminator_layer_sizes=self.discriminator_layer_sizes
        ).to(device)

        self.adv_loss = nn.BCEWithLogitsLoss()
        self.rec_loss_fn = nn.MSELoss()

    def diff_loss(self, x):
        dx = x[:, :, 1:] - x[:, :, :-1]
        return torch.mean(dx ** 2)

    def train(self, train_loader, val_loader, epochs=100):

        optim_E = torch.optim.Adam(self.model.encoder.parameters(), lr=self.lr)
        optim_D = torch.optim.Adam(self.model.decoder.parameters(), lr=self.lr)
        optim_Dis = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.lr * self.alpha
        )

        best_loss = float('inf')
        best_model = None

        for epoch in range(epochs):

            self.model.train()

            for data, labels in train_loader:

                data = data.to(device)
                labels = labels.to(device)

                bs = data.size(0)
                ones = torch.ones(bs, 1, device=device)
                zeros = torch.zeros(bs, 1, device=device)

                # -------- DISCRIMINATOR --------
                optim_Dis.zero_grad()

                out_real, _, _ = self.model.discriminator(data, labels)
                loss_real = self.adv_loss(out_real, ones)

                mean, logvar, rec = self.model(data, labels)

                z = torch.randn(bs, self.DATA_LENGTH, device=device)
                fake, _ = self.model.decoder(z, labels)

                out_rec, _, _ = self.model.discriminator(rec.detach(), labels)
                loss_rec = self.adv_loss(out_rec, zeros)

                out_fake, _, _ = self.model.discriminator(fake.detach(), labels)
                loss_fake = self.adv_loss(out_fake, zeros)

                dis_loss = loss_real + loss_rec + loss_fake
                dis_loss.backward()
                optim_Dis.step()

                # -------- GENERATOR --------
                optim_E.zero_grad()
                optim_D.zero_grad()

                mean, logvar, rec = self.model(data, labels)

                z = torch.randn(bs, self.DATA_LENGTH, device=device)
                fake, _ = self.model.decoder(z, labels)

                out_rec, _, _ = self.model.discriminator(rec, labels)
                out_fake, _, _ = self.model.discriminator(fake, labels)

                gan_loss = self.adv_loss(out_rec, ones) + self.adv_loss(out_fake, ones)

                rec_loss = self.rec_loss_fn(rec, data) + self.gamma * self.diff_loss(rec)

                kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

                gen_loss = gan_loss + rec_loss + kl_loss
                gen_loss.backward()

                optim_E.step()
                optim_D.step()

            # -------- VALIDATION --------
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for val_data, val_labels in val_loader:

                    val_data = val_data.to(device)
                    val_labels = val_labels.to(device)

                    _, _, val_rec = self.model(val_data, val_labels)
                    loss = self.rec_loss_fn(val_rec, val_data)

                    val_losses.append(loss.item())

            avg_val = sum(val_losses) / len(val_losses)

            score = gen_loss.item() + 0.2 * avg_val

            if score < best_loss:
                best_loss = score
                best_model = copy.deepcopy(self.model.state_dict())

            if epoch % 10 == 0:
                print(f"[{epoch}/{epochs}] D: {dis_loss.item():.4f} G: {gen_loss.item():.4f} Val: {avg_val:.6f}")

        self.model.load_state_dict(best_model)

    # =========================================================

    def generate(self, loader, n_times=6):

        self.model.eval()
        all_samples = []
        all_labels = []

        with torch.no_grad():
            for data, labels in loader:

                data = data.to(device)
                labels = labels.to(device)

                for _ in range(n_times):
                    z = torch.randn(data.size(0), self.DATA_LENGTH, device=device)
                    fake, _ = self.model.decoder(z, labels)

                    all_samples.append(fake.squeeze(1).cpu())
                    all_labels.append(labels.cpu())

        return torch.cat(all_samples), torch.cat(all_labels)