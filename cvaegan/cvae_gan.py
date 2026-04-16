import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


class Encoder(nn.Module):
    def __init__(self,
                 data_length=256,
                 num_conditions=6,
                 latent_dim=64,
                 num_layers=4,
                 base_channels=32,
                 cond_channels=8,
                 dropout=0.3):
        """Modular, dimension-proof 1D encoder.

        - conditions are embedded as additional channels
        - downsamples by factor ~2**num_layers using Conv1d
        - uses AdaptiveAvgPool1d to produce fixed-size embedding
        """
        super().__init__()
        self.data_length = data_length
        self.num_conditions = num_conditions
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # condition projection to extra channels
        self.cond_proj = nn.Sequential(
            nn.Linear(num_conditions, cond_channels),
            nn.LeakyReLU(0.2)
        )

        in_channels = 1 + cond_channels

        layers = []
        channels = []
        cur_ch = base_channels
        for i in range(num_layers):
            out_ch = cur_ch
            layers += [
                nn.Conv1d(in_channels, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_ch, momentum=0.9),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ]
            channels.append(out_ch)
            in_channels = out_ch
            cur_ch = min(cur_ch * 2, 512)

        self.conv = nn.Sequential(*layers)

        # Global pooling to fixed-size feature
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * 1, 256),
            nn.BatchNorm1d(256, momentum=0.9),
            nn.LeakyReLU(0.2)
        )

        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x, c):
        # x: (B,1,L), c: (B,num_conditions)
        bc = self.cond_proj(c)  # (B, cond_channels)
        bc = bc.unsqueeze(-1).expand(-1, -1, x.size(-1))  # (B,cond_channels,L)
        x = torch.cat([x, bc], dim=1)

        out = self.conv(x)
        out = self.pool(out)
        out = self.fc(out)
        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self,
                 data_length=256,
                 num_conditions=6,
                 latent_dim=64,
                 num_layers=4,
                 base_channels=32,
                 cond_channels=8,
                 dropout=0.3):
        """Modular decoder mirroring the encoder."""
        super().__init__()
        self.data_length = data_length
        self.num_conditions = num_conditions
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.cond_proj = nn.Sequential(
            nn.Linear(num_conditions, cond_channels),
            nn.LeakyReLU(0.2)
        )

        # start channels is the last encoder channel
        start_ch = min(base_channels * (2 ** (num_layers - 1)), 512)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_channels, start_ch),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_ch = start_ch
        cur_ch = in_ch
        for i in range(num_layers - 1):
            out_ch = max(cur_ch // 2, 8)
            layers += [
                nn.ConvTranspose1d(cur_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_ch, momentum=0.9),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ]
            cur_ch = out_ch

        # final conv to single channel
        layers += [
            nn.Conv1d(cur_ch, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        ]

        self.deconv = nn.Sequential(*layers)

        # small head to reconstruct/assist conditions
        self.labels_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_conditions)
        )

    def forward(self, z, c):
        # z: (B, latent_dim), c: (B, num_conditions)
        bc = self.cond_proj(c)  # (B, cond_channels)
        x = torch.cat([z, bc], dim=1)
        x = self.fc(x)  # (B, start_ch)
        # reshape to (B, start_ch, 1)
        x = x.unsqueeze(-1)
        x = self.deconv(x)

        # ensure output length matches data_length
        if x.size(-1) != self.data_length:
            x = nn.functional.interpolate(x, size=self.data_length, mode='linear', align_corners=False)

        c_recon = self.labels_head(z)
        return x, c_recon


class Discriminator(nn.Module):
    def __init__(self,
                 data_length=256,
                 num_conditions=6,
                 num_layers=4,
                 base_channels=32,
                 cond_channels=8,
                 dropout=0.3):
        super().__init__()
        self.data_length = data_length
        self.num_conditions = num_conditions

        # condition projection as extra channels
        self.cond_proj = nn.Sequential(
            nn.Linear(num_conditions, cond_channels),
            nn.LeakyReLU(0.2)
        )

        in_ch = 1 + cond_channels
        layers = []
        cur_ch = base_channels
        for i in range(num_layers):
            layers += [
                nn.Conv1d(in_ch, cur_ch, kernel_size=4 if i>0 else 5, stride=2, padding=1),
                nn.BatchNorm1d(cur_ch, momentum=0.9),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ]
            in_ch = cur_ch
            cur_ch = min(cur_ch * 2, 512)

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.adv_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 1, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

        self.label_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 1, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_conditions)
        )

    def forward(self, x, c):
        # x: (B,1,L), c: (B,num_conditions)
        bc = self.cond_proj(c).unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat([x, bc], dim=1)
        out = self.conv(x)
        out = self.pool(out)
        adv = self.adv_head(out)
        labels = self.label_head(out)
        return adv, out, labels


class CVAE_GAN(nn.Module):
    def __init__(self,
                 data_length=256,
                 num_conditions=6,
                 latent_dim=64,
                 num_layers=4,
                 base_channels=32,
                 cond_channels=8,
                 dropout=0.3):
        super().__init__()
        self.data_length = data_length
        self.num_conditions = num_conditions
        self.latent_dim = latent_dim

        self.encoder = Encoder(data_length=data_length,
                               num_conditions=num_conditions,
                               latent_dim=latent_dim,
                               num_layers=num_layers,
                               base_channels=base_channels,
                               cond_channels=cond_channels,
                               dropout=dropout)

        self.decoder = Decoder(data_length=data_length,
                               num_conditions=num_conditions,
                               latent_dim=latent_dim,
                               num_layers=num_layers,
                               base_channels=base_channels,
                               cond_channels=cond_channels,
                               dropout=dropout)

        self.discriminator = Discriminator(data_length=data_length,
                                           num_conditions=num_conditions,
                                           num_layers=num_layers,
                                           base_channels=base_channels,
                                           cond_channels=cond_channels,
                                           dropout=dropout)

        self.apply(weights_init)

    def reparameterize(self, mean, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std, device=std.device)
        return mean + eps * std

    def forward(self, x, c):
        # x: (B,1,L), c: (B,num_conditions)
        mean, logvar = self.encoder(x, c)
        z = self.reparameterize(mean, logvar)
        x_tilde, c_recon = self.decoder(z, c)
        return mean, logvar, x_tilde, c_recon
