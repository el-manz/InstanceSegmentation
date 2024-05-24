import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
from time import time
    

class DeepColoring(nn.Module):
    def __init__(self, colors):
        super().__init__()
        
        # ReLU activation
        self.act = nn.ReLU

        # U-net encoder
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            self.act(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            self.act()
        )
        self.pool0 = nn.MaxPool2d(2, 2)  # 256 -> 128
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            self.act(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            self.act()
        )
        self.pool1 = nn.MaxPool2d(2, 2) # 128 -> 64
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            self.act(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            self.act()
        )
        self.pool2 = nn.MaxPool2d(2, 2) # 64 -> 32
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            self.act(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            self.act()
        )
        self.pool3 = nn.MaxPool2d(2, 2) # 32 -> 16

        # U-net bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            self.act(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            self.act()
        )

        # U-net decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear') # 16 -> 32
        self.dec_conv0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024 + 512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            self.act(),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            self.act()
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear') # 32 -> 64
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512 + 256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            self.act(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            self.act()
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 64 -> 128
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256 +128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            self.act(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            self.act()
        )
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 128 -> 256
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128 + 64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            self.act(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            self.act(),
            nn.ConvTranspose2d(in_channels=64, out_channels=colors, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # U-net encoder
        e0 = self.enc_conv0(x)
        e1 = self.enc_conv1(self.pool0(e0))
        e2 = self.enc_conv2(self.pool1(e1))
        e3 = self.enc_conv3(self.pool2(e2))

        # U-net bottleneck
        b = self.bottleneck_conv(self.pool3(e3))

        # U-net decoder
        d0 = self.upsample0(b)
        d0 = torch.cat((e3, d0), dim=1)
        d0 = self.dec_conv0(d0)
        d1 = self.upsample1(d0)
        d1 = torch.cat((e2, d1), dim=1)
        d1 = self.dec_conv1(d1)
        d2 = self.upsample2(d1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.dec_conv2(d2)
        d3 = self.upsample3(d2)
        d3 = torch.cat((e0, d3), dim=1)
        d3 = self.dec_conv3(d3)

        return torch.sigmoid(d3)