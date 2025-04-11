import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, text_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.text_mlp = nn.Linear(text_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU()
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t, text):
        h = self.block1(x)
        t_emb = self.time_mlp(t).view(t.size(0), -1, 1, 1)
        text_emb = self.text_mlp(text).view(text.size(0), -1, 1, 1)
        h = h + t_emb + text_emb
        h = self.block2(h)
        return h + self.residual(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=256, raw_text_dim=768, text_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(raw_text_dim, text_emb_dim),
            nn.ReLU(),
            nn.Linear(text_emb_dim, text_emb_dim)
        )

        # Encoder-Part
        self.encoder1 = ResidualBlock(in_channels, 64, time_emb_dim, text_emb_dim)
        self.down1 = nn.Conv2d(64, 128, 4, 2, 1)
        self.encoder2 = ResidualBlock(128, 128, time_emb_dim, text_emb_dim)
        self.down2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.encoder3 = ResidualBlock(256, 256, time_emb_dim, text_emb_dim)

        # Bottleneck without skip connection
        self.middle = ResidualBlock(256, 256, time_emb_dim, text_emb_dim)

        # Decoder-Part
        self.up2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.decoder2 = ResidualBlock(256, 128, time_emb_dim, text_emb_dim)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.decoder1 = ResidualBlock(128, 64, time_emb_dim, text_emb_dim)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t, text_raw):
        t = self.time_mlp(t)
        text = self.text_proj(text_raw)

        x1 = self.encoder1(x, t, text)
        x2 = self.encoder2(self.down1(x1), t, text)
        x3 = self.encoder3(self.down2(x2), t, text)

        x_mid = self.middle(x3, t, text)

        x = self.decoder2(torch.cat([self.up2(x_mid), x2], dim=1), t, text)
        x = self.decoder1(torch.cat([self.up1(x), x1], dim=1), t, text)

        return self.final(x)
