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
    def __init__(self, NUM_CONDITIONS=6, BATCH_SIZE=64, DATA_LENGTH=256, DROPOUT=0.3):

        """
            Encoder do CVAE-GAN

            Atributos:
                NUM_CONDITIONS (int): numero de labels que serao usadas
                BATCH_SIZE (int): tamanho do batch
                DATA_LENGTH (int): tamanho do espectro
                DROPOUT (float): taxa de dropout

            Retorna:
                mean (tensor): media da distribuicao latente
                logvar (tensor): log da variancia da distribuicao latente
        """

        super(Encoder, self).__init__()

        self.input = nn.Sequential(
            nn.Conv1d(1, BATCH_SIZE, 5, padding=2, stride=2),
            nn.BatchNorm1d(BATCH_SIZE, momentum=0.9),
            nn.LeakyReLU(0.2),
        )

        self.sequential = nn.Sequential(
            nn.Conv1d(BATCH_SIZE, 128, 5, padding=2, stride=2),
            nn.BatchNorm1d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Conv1d(128, DATA_LENGTH, 5, padding=2, stride=2),
            nn.BatchNorm1d(DATA_LENGTH, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Conv1d(DATA_LENGTH, 128, 5, padding=2, stride=2),
            nn.BatchNorm1d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Flatten(),
            nn.Linear(2176, 2048),
            nn.BatchNorm1d(2048, momentum=0.9),
            nn.LeakyReLU(0.2)
        )

        self.fc_mean = nn.Linear(2048, DATA_LENGTH)
        self.fc_logvar = nn.Linear(2048, DATA_LENGTH)

    def forward(self, x, c):
        c_reshape = c.unsqueeze(1)
        concat = torch.cat((x, c_reshape), dim=-1)

        out = self.input(concat)
        out = self.sequential(out)
        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)

        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, NUM_CONDITIONS=6, DATA_LENGTH=256, DROPOUT=0.3):

        """
            Decoder do CVAE-GAN

            Atributos:
                NUM_CONDITIONS (int): numero de labels que serao usadas
                DATA_LENGTH (int): tamanho do espectro 
                DROPOUT (float): taxa de dropout
            
            Retorna:
                x (tensor): espectro reconstruido
                c_reconstructed (tensor): labels reconstruidas 
        """

        super(Decoder, self).__init__()

        self.input = nn.Sequential(
            nn.Linear(DATA_LENGTH + NUM_CONDITIONS, 256 * 8),
            nn.BatchNorm1d(256 * 8, momentum=0.9),
            nn.LeakyReLU(0.2),
        )

        self.sequential = nn.Sequential(
            nn.Unflatten(1, (256, 8)),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   
            nn.BatchNorm1d(32, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),    
            nn.BatchNorm1d(16, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid()
        )

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
        x = self.sequential(x)
        c_reconstructed = self.reconstruction_head(z)
        return x, c_reconstructed


class Discriminator(nn.Module):
    def __init__(self, NUM_CONDITIONS=6, DATA_LENGTH=256, DROPOUT=0.3):

        """
            Discriminador do CVAE-GAN
            
            Atributos:
                NUM_CONDITIONS (int): numero de labels que serao usadas
                DATA_LENGTH (int): tamanho do espectro 
                DROPOUT (float): taxa de dropout

            Retorna:
                x (tensor): probabilidade do espectro ser real
                x1 (tensor): features extraidas do espectro
                features (tensor): labels preditas pelo discriminador
        """

        super(Discriminator, self).__init__()

        self.input = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2, stride=1),
            nn.LeakyReLU(0.2)
        )

        self.sequential1 = nn.Sequential(
            nn.Conv1d(32, 128, 5, padding=2, stride=2),
            nn.BatchNorm1d(128, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Conv1d(128, DATA_LENGTH, 5, padding=2, stride=2),
            nn.BatchNorm1d(DATA_LENGTH, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Conv1d(DATA_LENGTH, DATA_LENGTH, 5, padding=2, stride=2),
        )

        self. sequential2 = nn.Sequential(
            nn.BatchNorm1d(DATA_LENGTH, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(32*DATA_LENGTH, 512),
            nn.BatchNorm1d(512, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.labels_head = nn.Sequential(
            nn.BatchNorm1d(DATA_LENGTH, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Conv1d(DATA_LENGTH, 64, 5, padding=2, stride=2),
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

    def forward(self, x, c):

        c_reshape = c.unsqueeze(1)
        concat = torch.cat((x, c_reshape), dim=-1)

        x = self.input(concat)

        x = self.sequential1(x)
        
        x1 = x

        features = self.labels_head(x1)

        x = self.sequential2(x)

        return x, x1, features

class CVAE_GAN(nn.Module):
    def __init__(self, BATCH_SIZE=64, DATA_LENGTH=256, NUM_CONDITIONS=6):

        """
            Modelo da CVAE-GAN que junta tudo

            Atributos:
                BATCH_SIZE (int): tamanho do batch
                DATA_LENGTH (int): tamanho do espectro
                NUM_CONDITIONS (int): numero de labels que serao usadas
            
            Retorna:
                z_mean (tensor): media da distribuicao latente
                z_logvar (tensor): log da variancia da distribuicao latente
                x_tilda (tensor): espectro reconstruido
        """

        super(CVAE_GAN, self).__init__()

        self.BATCH_SIZE = BATCH_SIZE
        self.DATA_LENGTH = DATA_LENGTH
        self.NUM_CONDITIONS = NUM_CONDITIONS

        self.encoder = Encoder(NUM_CONDITIONS = NUM_CONDITIONS)
        self.decoder = Decoder(NUM_CONDITIONS = NUM_CONDITIONS)
        self.discriminator = Discriminator(NUM_CONDITIONS= NUM_CONDITIONS)

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.discriminator.apply(weights_init)

    def forward(self, x, c):
        bs = x.shape[0]
        z_mean, z_logvar = self.encoder(x, c)
        std = z_logvar.mul(0.5).exp_()

        epsilon = Variable(torch.randn(bs, self.DATA_LENGTH)).to(device)
        z = z_mean+std*epsilon
        x_tilda, _ = self.decoder(z, c)

        return z_mean, z_logvar, x_tilda
