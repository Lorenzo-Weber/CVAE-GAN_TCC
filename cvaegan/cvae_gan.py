import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Encoder(nn.Module):
    def __init__(
        self,
        NUM_CONDITIONS=6,
        BATCH_SIZE=64,
        DATA_LENGTH=256,
        DROPOUT=0.3,
        layer_sizes=[128, 128]
    ):
        super(Encoder, self).__init__()

        layer_sizes = list(layer_sizes)

        self.input = nn.Sequential(
            nn.Conv1d(1 + NUM_CONDITIONS, BATCH_SIZE, 5, padding=2, stride=2),
            nn.BatchNorm1d(BATCH_SIZE, momentum=0.9),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = BATCH_SIZE

        for out_channels in layer_sizes:
            layers += [
                nn.Conv1d(in_channels, out_channels, 5, padding=2, stride=2),
                nn.BatchNorm1d(out_channels, momentum=0.9),
                nn.LeakyReLU(0.2),
                nn.Dropout(DROPOUT)
            ]
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, 2048),
            nn.BatchNorm1d(2048, momentum=0.9),
            nn.LeakyReLU(0.2)
        )

        self.fc_mean = nn.Linear(2048, DATA_LENGTH)
        self.fc_logvar = nn.Linear(2048, DATA_LENGTH)

    def forward(self, x, c):
        if c.dim() == 2:
            c = c.unsqueeze(-1)

        if c.size(-1) != x.size(-1):
            c = c.repeat(1, 1, x.size(-1))

        concat = torch.cat((x, c), dim=1)

        out = self.input(concat)
        out = self.conv_blocks(out)
        out = self.global_pool(out)
        out = out.squeeze(-1)
        out = self.fc(out)

        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)

        return mean, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        NUM_CONDITIONS=6,
        DATA_LENGTH=256,
        DROPOUT=0.3,
        layer_sizes=[128, 128]
    ):
        super(Decoder, self).__init__()

        layer_sizes = list(layer_sizes)

        self.DATA_LENGTH = DATA_LENGTH

        self.initial_length = DATA_LENGTH // (2 ** (len(layer_sizes) + 1))
        self.initial_channels = layer_sizes[0]

        self.input = nn.Sequential(
            nn.Linear(DATA_LENGTH + NUM_CONDITIONS, self.initial_channels * self.initial_length),
            nn.BatchNorm1d(self.initial_channels * self.initial_length, momentum=0.9),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = self.initial_channels

        for out_channels in layer_sizes[1:]:
            layers += [
                nn.ConvTranspose1d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm1d(out_channels, momentum=0.9),
                nn.LeakyReLU(0.2),
                nn.Dropout(DROPOUT)
            ]
            in_channels = out_channels

        layers.append(nn.ConvTranspose1d(in_channels, 1, 4, stride=2, padding=1))

        self.deconv = nn.Sequential(*layers)

        self.reconstruction_head = nn.Sequential(
            nn.Linear(DATA_LENGTH, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CONDITIONS)
        )

    def forward(self, x, c):
        z = x
        concat = torch.cat((x, c), dim=1)

        x = self.input(concat)
        x = x.view(x.size(0), self.initial_channels, self.initial_length)
        x = self.deconv(x)

        if x.size(-1) != self.DATA_LENGTH:
            x = torch.nn.functional.interpolate(
                x,
                size=self.DATA_LENGTH,
                mode="linear",
                align_corners=False
            )

        c_reconstructed = self.reconstruction_head(z)

        return x, c_reconstructed


class Discriminator(nn.Module):
    def __init__(
        self,
        NUM_CONDITIONS=6,
        DATA_LENGTH=256,
        DROPOUT=0.3,
        layer_sizes=[128, 256, 256]
    ):
        super(Discriminator, self).__init__()

        layer_sizes = list(layer_sizes)

        self.DATA_LENGTH = DATA_LENGTH
        self.NUM_CONDITIONS = NUM_CONDITIONS

        self.input = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2, stride=1),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = 32

        for out_channels in layer_sizes:
            layers += [
                nn.Conv1d(in_channels, out_channels, 5, padding=2, stride=2),
                nn.BatchNorm1d(out_channels, momentum=0.9),
                nn.LeakyReLU(0.2),
                nn.Dropout(DROPOUT)
            ]
            in_channels = out_channels

        self.sequential1 = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(32)

        self.flatten_dim = self._get_flatten_dim()

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.BatchNorm1d(512, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.labels_head = nn.Sequential(
            nn.BatchNorm1d(in_channels, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Conv1d(in_channels, 64, 5, padding=2, stride=2),
            nn.BatchNorm1d(64, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Linear(32, NUM_CONDITIONS),
            nn.ReLU()
        )

    def _get_flatten_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.DATA_LENGTH)
            out = self.input(dummy)
            out = self.sequential1(out)
            out = self.pool(out)
            return out.view(1, -1).shape[1]

    def forward(self, x, c):
        x = self.input(x)
        x = self.sequential1(x)

        x1 = x
        features = self.labels_head(x1)

        x = self.pool(x1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x, x1, features


class CVAE_GAN(nn.Module):
    def __init__(
        self,
        BATCH_SIZE=64,
        DATA_LENGTH=256,
        NUM_CONDITIONS=6,
        generator_layer_sizes=[128, 128],
        discriminator_layer_sizes=[128, 256, 256]
    ):
        super(CVAE_GAN, self).__init__()

        self.encoder = Encoder(
            NUM_CONDITIONS=NUM_CONDITIONS,
            BATCH_SIZE=BATCH_SIZE,
            DATA_LENGTH=DATA_LENGTH,
            layer_sizes=generator_layer_sizes
        )

        self.decoder = Decoder(
            NUM_CONDITIONS=NUM_CONDITIONS,
            DATA_LENGTH=DATA_LENGTH,
            layer_sizes=list(reversed(generator_layer_sizes))
        )

        self.discriminator = Discriminator(
            NUM_CONDITIONS=NUM_CONDITIONS,
            DATA_LENGTH=DATA_LENGTH,
            layer_sizes=discriminator_layer_sizes
        )

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.discriminator.apply(weights_init)

    def forward(self, x, c):
        bs = x.shape[0]

        z_mean, z_logvar = self.encoder(x, c)
        std = z_logvar.mul(0.5).exp_()

        epsilon = Variable(torch.randn(bs, self.encoder.fc_mean.out_features)).to(device)
        z = z_mean + std * epsilon

        x_tilda, _ = self.decoder(z, c)

        return z_mean, z_logvar, x_tilda